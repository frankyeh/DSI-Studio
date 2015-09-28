#ifndef DDI_PROCESS_HPP
#define DDI_PROCESS_HPP
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/math/special_functions/sinc.hpp>
#include "basic_process.hpp"
#include "basic_voxel.hpp"
#include "image_model.hpp"


class QSpace2Odf  : public BaseProcess
{
public:// recorded for scheme balanced
    std::vector<image::vector<3,double> > q_vectors_time;
public:
    std::vector<unsigned int> b0_images;
    std::vector<float> sinc_ql;
    double base_function(double theta)
    {
        if(std::abs(theta) < 0.000001)
            return 1.0/3.0;
        return (2*std::cos(theta)+(theta-2.0/theta)*std::sin(theta))/theta/theta;
    }
public:
    virtual void init(Voxel& voxel)
    {
        b0_images.clear();
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
            if(voxel.bvalues[index] == 0)
                b0_images.push_back(index);
        if(b0_images.size() > 1)
            throw std::runtime_error("Correct B0 failed. Two b0 images found in src file");


        unsigned int odf_size = voxel.ti.half_vertices_count;
        float sigma = voxel.param[0]; //optimal 1.24
        if(!voxel.grad_dev.empty())
        {
            q_vectors_time.resize(voxel.bvalues.size());
            for (unsigned int index = 0; index < voxel.bvalues.size(); ++index)
            {
                q_vectors_time[index] = voxel.bvectors[index];
                q_vectors_time[index] *= std::sqrt(voxel.bvalues[index]*0.01506);// get q in (mm) -1
                q_vectors_time[index] *= sigma;
            }
            return;
        }
        sinc_ql.resize(odf_size*voxel.bvalues.size());
        // calculate reconstruction matrix
        for (unsigned int j = 0,index = 0; j < odf_size; ++j)
            for (unsigned int i = 0; i < voxel.bvalues.size(); ++i,++index)
                sinc_ql[index] = voxel.bvectors[i]*
                             image::vector<3,float>(voxel.ti.vertices[j])*
                               std::sqrt(voxel.bvalues[i]*0.01506);

        for (unsigned int index = 0; index < sinc_ql.size(); ++index)
            sinc_ql[index] = voxel.r2_weighted ?
                         base_function(sinc_ql[index]*sigma):
                         boost::math::sinc_pi(sinc_ql[index]*sigma);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(b0_images.size() == 1 && voxel.half_sphere)
            data.space[b0_images.front()] /= 2.0;
        if(!voxel.grad_dev.empty()) // correction for gradient nonlinearity
        {
            /*
            new_bvecs = (I+grad_dev) * bvecs;
            */
            float grad_dev[9];
            for(unsigned int i = 0; i < 9; ++i)
                grad_dev[i] = voxel.grad_dev[i][data.voxel_index];
            std::vector<float> new_sinc_ql(data.odf.size()*data.space.size());
            for (unsigned int j = 0,index = 0; j < data.odf.size(); ++j)
            {
                image::vector<3,double> dir(voxel.ti.vertices[j]),from;
                image::matrix::vector_product(grad_dev,dir.begin(),from.begin(),image::dim<3,3>());
                from.normalize();
                if(voxel.r2_weighted)
                    for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                        new_sinc_ql[index] = base_function(q_vectors_time[i]*from);
                else
                    for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                        new_sinc_ql[index] = boost::math::sinc_pi(q_vectors_time[i]*from);

            }
            image::matrix::vector_product(&*new_sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                    image::dyndim(data.odf.size(),data.space.size()));
        }
        else
            image::matrix::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                    image::dyndim(data.odf.size(),data.space.size()));
    }
};

class SchemeConverter : public BaseProcess
{
    QSpace2Odf from,to;
    std::vector<int> piv;
    std::vector<image::vector<3,float> > bvectors;
    std::vector<float> bvalues;
    std::vector<std::vector<unsigned short> > dwi;
    std::vector<unsigned short> b0;
    std::vector<float> A,Rt;
    unsigned int total_value;
    unsigned int total_negative_value;
public:
    virtual void init(Voxel& voxel)
    {

        if(!voxel.file_name.empty())
        {
            std::ifstream read(voxel.file_name.c_str());
            std::vector<float> values;
            std::copy(std::istream_iterator<float>(read),std::istream_iterator<float>(),std::back_inserter(values));
            for(unsigned int i = 0;i < values.size()/4;++i)
            {
                if(values[i*4] == 0.0)
                    continue;
                bvalues.push_back(voxel.param[1]);
                bvectors.push_back(image::vector<3,float>(values[i*4+1],values[i*4+2],values[i*4+3]));
            }
        }
        else
        {
            bvalues.resize(voxel.ti.half_vertices_count);
            bvectors.resize(voxel.ti.half_vertices_count);
            for (unsigned int index = 0; index < bvectors.size(); ++index)
                bvectors[index] = voxel.ti.vertices[index];
            std::fill(bvalues.begin(),bvalues.end(),voxel.param[1]); // set the output b-value
        }

        //allocated output image space
        dwi.resize(bvalues.size());
        for(unsigned int index = 0;index < dwi.size();++index)
            dwi[index].resize(voxel.dim.size());


        from.init(voxel);
        voxel.bvalues.swap(bvalues);
        voxel.bvectors.swap(bvectors);
        to.init(voxel);
        voxel.bvalues.swap(bvalues);
        voxel.bvectors.swap(bvectors);

        if(!from.b0_images.empty())
            b0.resize(voxel.dim.size());

        Rt.resize(dwi.size()*dwi.size());
        image::matrix::transpose(&*to.sinc_ql.begin(),&*Rt.begin(),image::dyndim(dwi.size(),dwi.size()));
        A.resize(dwi.size()*dwi.size());
        piv.resize(dwi.size());
        image::matrix::product_transpose(&*Rt.begin(),&*Rt.begin(),&*A.begin(),
                                       image::dyndim(dwi.size(),dwi.size()),image::dyndim(dwi.size(),dwi.size()));
        float max_value = *std::max_element(A.begin(),A.end());
        for (unsigned int i = 0,index = 0; i < dwi.size(); ++i,index += dwi.size() + 1)
            A[index] += max_value*voxel.param[2];
        image::matrix::lu_decomposition(A.begin(),piv.begin(),image::dyndim(dwi.size(),dwi.size()));

        total_negative_value = 0;
        total_value = 0;

        voxel.recon_report
                << " The converted HARDI has a total of " << dwi.size() << " diffusion sampling directions.";
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(!from.b0_images.empty())
        {
            b0[data.voxel_index] = data.space[from.b0_images.front()];
            data.space[from.b0_images.front()] = 0;
        }
        from.run(voxel,data);
        std::vector<float> hardi_data(dwi.size()),tmp(dwi.size());
        image::matrix::vector_product(&*Rt.begin(),&*data.odf.begin(),&*tmp.begin(),image::dyndim(dwi.size(),dwi.size()));
        image::matrix::lu_solve(&*A.begin(),&*piv.begin(),&*tmp.begin(),&*hardi_data.begin(),image::dyndim(dwi.size(),dwi.size()));
        for(unsigned int index = 0;index < dwi.size();++index)
        {
            if(hardi_data[index] < 0.0)
                ++total_negative_value;
            else
                dwi[index][data.voxel_index] = hardi_data[index];
            ++total_value;
        }
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        voxel.recon_report
                << " The percentage of the negative signals was " << (float)100.0*total_negative_value/(float)total_value << "%.";
        std::vector<float> b_table(4); // space for b0
        for (unsigned int index = 0;index < bvectors.size();++index)
        {
            b_table.push_back(bvalues.front());
            std::copy(bvectors[index].begin(),bvectors[index].end(),std::back_inserter(b_table));
        }
        mat_writer.write("b_table",&b_table[0],4,b_table.size()/4);
        unsigned int image_num = 0;
        if(!b0.empty())
        {
            mat_writer.write("image0",&b0[0],1,b0.size());
            ++image_num;
        }
        for (unsigned int index = 0;index < dwi.size();++index)
        {
            std::ostringstream out;
            out << "image" << image_num;
            mat_writer.write(out.str().c_str(),&(dwi[index][0]),1,dwi[index].size());
            ++image_num;
        }
    }
};

class QSpaceSpectral  : public BaseProcess
{
public:
    static const int max_length = 50; // 50 microns
    std::vector<unsigned int> b0_images;
    std::vector<std::vector<float> > cdf,dis,cdfw,disw;
    double sinint(double x)
    {
        bool sgn = x > 0;
        x = std::fabs(x);
        double eps = 1e-15;
        double x2 = x*x;
        double si;
        if(x == 0.0)
            return 0.0;
        if(x <= 16.0)
        {
            double xr = x;
            si = x;
            for(unsigned int k = 1;k <= 40;++k)
            {
                si += (xr *= -0.5*(double)(2*k-1)/(double)k/(double)(4*k*(k+1)+1)*x2);
                if(std::fabs(xr) < std::fabs(si)*eps)
                    break;
            }
            return sgn ? si:-si;
        }

        if(x <= 32.0)
        {
            unsigned int m = std::floor(47.2+0.82*x);
            std::vector<double> bj(m+1);
            double xa1 = 0.0;
            double xa0 = 1.0e-100;
            for(unsigned int k=m;k>=1;--k)
            {
                double xa = 4.0*(double)k*xa0/x-xa1;
                bj[k-1] = xa;
                xa1 = xa0;
                xa0 = xa;
            }
            double xs = bj[0];
            for(unsigned int k=3;k <= m;k += 2)
                xs += 2.0*bj[k-1];
            for(unsigned int k=0;k < m;++k)
                bj[k] /= xs;
            double xr = 1.0;
            double xg1 = bj[0];
            for(int k=2;k <= m;++k)
                xg1 += bj[k-1]*(xr *= 0.25*(2*k-3)*(2*k-3)/((k-1)*(2*k-1)*(2*k-1))*x);
            xr = 1.0;
            double xg2 = bj[0];
            for(int k=2;k <= m;++k)
                xg2 += bj[k-1]*(xr *= 0.25*(2*k-5)*(2*k-5)/((k-1)*(2*k-3)*(2*k-3))*x);
            si = x*std::cos(x/2.0)*xg1+2.0*std::sin(x/2.0)*xg2-std::sin(x);
            return sgn ? si:-si;
        }

        double xr = 1.0;
        double xf = 1.0;
        for(unsigned int k=1;k <= 9;++k)
            xf += (xr *= -2.0*k*(2*k-1)/x2);
        xr = 1.0/x;
        double xg = xr;
        for(unsigned int k=1;k <= 8;++k)
            xg += (xr *= -2.0*(2*k+1)*k/x2);
        si = 1.570796326794897-xf*std::cos(x)/x-xg*std::sin(x)/x;
        return sgn ? si:-si;
    }

    double base_function(double theta)
    {
        if(std::abs(theta) < 0.000001)
            return 1.0/3.0;
        return (2*std::cos(theta)+(theta-2.0/theta)*std::sin(theta))/theta/theta;
    }
public:
    virtual void init(Voxel& voxel)
    {
        b0_images.clear();
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
            if(voxel.bvalues[index] == 0)
                b0_images.push_back(index);
        if(b0_images.size() > 1)
            throw std::runtime_error("Correct B0 failed. Two b0 images found in src file");

        float diffusion_time = voxel.param[1];
        float diffusion_length = std::sqrt(6.0*3.0*diffusion_time); // sqrt(6Dt)
        dis.clear();
        cdf.clear();
        disw.clear();
        cdfw.clear();
        dis.resize(max_length);
        cdf.resize(max_length);
        disw.resize(max_length);
        cdfw.resize(max_length);
        for(unsigned int n = 0;n < max_length;++n) // from 0 micron to 49 microns
        {
            // calculate the diffusion length ratio
            float sigma = ((float)n)/diffusion_length;

            dis[n].resize(voxel.dim.size());
            cdf[n].resize(voxel.dim.size());

            disw[n].resize(voxel.bvalues.size());
            cdfw[n].resize(voxel.bvalues.size());
            for(unsigned int index = 0;index < voxel.bvalues.size();++index)
            {
                // 2pi*L*q = sigma*sqrt(6D*b_value)
                double lq_2pi = sigma*std::sqrt(voxel.bvalues[index]*0.018);
                disw[n][index] = boost::math::sinc_pi(lq_2pi);
                cdfw[n][index] = (voxel.bvalues[index] == 0.0 ?
                                 sigma : sinint(lq_2pi)/std::sqrt(voxel.bvalues[index]*0.018));
            }
        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(b0_images.size() == 1 && voxel.half_sphere)
            data.space[b0_images.front()] /= 2.0;
        for(unsigned int index = 0;index < max_length;++index)
        {
            dis[index][data.voxel_index] =
                    image::vec::dot(data.space.begin(),data.space.end(),disw[index].begin());
            cdf[index][data.voxel_index] =
                    image::vec::dot(data.space.begin(),data.space.end(),cdfw[index].begin());
            // make sure that cdf is increamental
            if(index && cdf[index][data.voxel_index] < cdf[index-1][data.voxel_index])
                cdf[index][data.voxel_index] = cdf[index-1][data.voxel_index];
            // make sure that dis is positive
            if(dis[index][data.voxel_index] < 0.0)
                dis[index][data.voxel_index] = 0.0;
        }
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        for(unsigned int index = 0;index < max_length;++index)
        {
            std::ostringstream out;
            out << "pdf_" << index << "um";
            mat_writer.write(out.str().c_str(),&*dis[index].begin(),1,dis[index].size());

        }
        for(unsigned int index = 0;index < max_length;++index)
        {
            std::ostringstream out;
            out << "cdf_" << index << "um";
            mat_writer.write(out.str().c_str(),&*cdf[index].begin(),1,cdf[index].size());
        }
        mat_writer.write("fa0",&*dis[0].begin(),1,dis[0].size());
        std::vector<short> index0(voxel.dim.size());
        mat_writer.write("index0",&*index0.begin(),1,index0.size());
    }
};


#endif//DDI_PROCESS_HPP
