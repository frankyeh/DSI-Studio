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
        sinc_ql.resize(odf_size*voxel.bvalues.size());
        float sigma = voxel.param[0]; //optimal 1.24

        // calculate reconstruction matrix
        for (unsigned int j = 0,index = 0; j < odf_size; ++j)
            for (unsigned int i = 0; i < voxel.bvalues.size(); ++i,++index)
                sinc_ql[index] = voxel.bvectors[i]*
                             image::vector<3,float>(voxel.ti.vertices[j])*
                               std::sqrt(voxel.bvalues[i]*0.01506); // £^G£_

        for (unsigned int index = 0; index < sinc_ql.size(); ++index)
            sinc_ql[index] = voxel.r2_weighted ?
                         base_function(sinc_ql[index]*sigma):
                         boost::math::sinc_pi(sinc_ql[index]*sigma);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(b0_images.size() == 1 && voxel.half_sphere)
            data.space[b0_images.front()] /= 2.0;
        image::matrix::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                    image::dyndim(data.odf.size(),data.space.size()));
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

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
        std::cout << "percentage of negative value = " << (float)100.0*total_negative_value/(float)total_value << std::endl;
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
    std::vector<unsigned int> b0_images;
    std::vector<std::vector<float> > sinc_ql;
    std::vector<std::vector<float> > sinc_ql_cdf;
    std::vector<std::vector<float> > cdf;
    std::vector<std::vector<float> > dis;
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
        sinc_ql.clear();
        sinc_ql_cdf.clear();
        const int max_length = 50; // 50 microns
        dis.resize(max_length);
        cdf.resize(max_length);
        sinc_ql.resize(max_length);
        sinc_ql_cdf.resize(max_length);
        unsigned int odf_size = voxel.ti.half_vertices_count;
        for(unsigned int n = 0;n < max_length;++n) // from 0 micron to 49 microns
        {
            // calculate the diffusion length ratio
            float sigma = ((float)n)/diffusion_length;
            float delta = 0.001/diffusion_length;

            sinc_ql[n].resize(odf_size*voxel.bvalues.size());
            sinc_ql_cdf[n].resize(odf_size*voxel.bvalues.size());
            dis[n].resize(voxel.dim.size());
            cdf[n].resize(voxel.dim.size());
            // calculate reconstruction matrix
            for (unsigned int j = 0,index = 0; j < odf_size; ++j)
                for (unsigned int i = 0; i < voxel.bvalues.size(); ++i,++index)
                    sinc_ql[n][index] = voxel.bvectors[i]*
                                 image::vector<3,float>(voxel.ti.vertices[j])*
                                   std::sqrt(voxel.bvalues[i]*0.018); // £^G£_

            for (unsigned int index = 0; index < sinc_ql_cdf[n].size(); ++index)
                sinc_ql_cdf[n][index] = voxel.r2_weighted ?
                            base_function(sinc_ql[n][index]*sigma)*sigma:
                            boost::math::sinc_pi(sinc_ql[n][index]*sigma)*sigma;

            for (unsigned int index = 0; index < sinc_ql[n].size(); ++index)
                sinc_ql[n][index] = voxel.r2_weighted ?
                             ((sigma+delta)*base_function(sinc_ql[n][index]*(sigma+delta))-
                             (sigma-delta)*base_function(sinc_ql[n][index]*(sigma-delta)))/delta:
                              ((sigma+delta)*boost::math::sinc_pi(sinc_ql[n][index]*(sigma+delta))-
                               (sigma-delta)*boost::math::sinc_pi(sinc_ql[n][index]*(sigma-delta)))/delta;
        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(b0_images.size() == 1 && voxel.half_sphere)
            data.space[b0_images.front()] /= 2.0;
        for(unsigned int index = 0;index < sinc_ql.size();++index)
        {
            image::matrix::vector_product(&*sinc_ql[index].begin(),&*data.space.begin(),&*data.odf.begin(),
                                        image::dyndim(data.odf.size(),data.space.size()));
            dis[index][data.voxel_index] = std::max<float>(0.0,image::mean(data.odf.begin(),data.odf.end()));

            image::matrix::vector_product(&*sinc_ql_cdf[index].begin(),&*data.space.begin(),&*data.odf.begin(),
                                        image::dyndim(data.odf.size(),data.space.size()));
            cdf[index][data.voxel_index] = image::mean(data.odf.begin(),data.odf.end());
            // make sure that cdf is increamental
            if(index && cdf[index][data.voxel_index] < cdf[index-1][data.voxel_index])
                cdf[index][data.voxel_index] = cdf[index-1][data.voxel_index];

        }
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        for(unsigned int index = 0;index < sinc_ql.size();++index)
        {
            std::ostringstream out;
            out << "pdf_" << index << "um";
            mat_writer.write(out.str().c_str(),&*dis[index].begin(),1,dis[index].size());

        }
        for(unsigned int index = 0;index < sinc_ql.size();++index)
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
