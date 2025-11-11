#ifndef DDI_PROCESS_HPP
#define DDI_PROCESS_HPP
#include <cmath>
#include "basic_process.hpp"
#include "basic_voxel.hpp"
#include "image_model.hpp"
#include "odf_process.hpp"
template<typename value_type>
value_type base_function(value_type theta)
{
    return (std::fabs(theta) < 0.000001f) ?
            1.0f/3.0f:
            (2.0f*std::cos(theta)+(theta-2.0f/theta)*std::sin(theta))/theta/theta;
}

template<typename value_type>
value_type sinc_pi_imp(value_type x)
{
    static const float taylor_0_bound = std::ldexp(1.0f, 1-std::numeric_limits<float>::digits);
    static const float taylor_2_bound = std::sqrt(taylor_0_bound);
    static const float taylor_n_bound = std::sqrt(taylor_2_bound);
    if (std::abs(x) >= taylor_n_bound)
        return(sin(x)/x);
    else
    {
        value_type result(1);
        if(std::abs(x) >= taylor_0_bound)
        {
            value_type x2(x*x);
            result -= x2/value_type(6.0);
            if(std::abs(x) >= taylor_2_bound)
                result += (x2*x2)/value_type(120.0);
        }
        return(result);
    }
}

class GQI_Recon  : public BaseProcess
{
public:
    std::vector<tipl::vector<3,float> > q_vectors_time;
    std::vector<float> sinc_ql;
    bool dsi_half_sphere = false;
private:
    void calculate_sinc_ql(Voxel& voxel);
    void calculate_q_vec_t(Voxel& voxel);
public:
    virtual void init(Voxel& voxel) override;
    virtual void run(Voxel& voxel, VoxelData& data) override;
};

class HGQI_Recon  : public BaseProcess
{
public:
    GQI_Recon gr;
public:
    bool hgqi = false;
    std::vector<float> hraw;
    std::vector<int> offset;
    std::vector<float> scaling;
public:
    virtual void init(Voxel& voxel) override
    {
        if(voxel.bvalues.size() != 1)
        {
            hgqi = false;
            return;
        }
        hgqi = true;
        hraw.resize(voxel.dim.size());
        int range = 2;
        for(int dz = 0;dz <= range;++dz) // half sphere
            for(int dy = -range;dy <= range;++dy)
                for(int dx = -range;dx <= range;++dx)
                {
                    int r2 = dx*dx+dy*dy+dz*dz;
                    voxel.bvalues.push_back(r2*500);
                    tipl::vector<3> dir(dx,dy,dz);
                    dir.normalize();
                    voxel.bvectors.push_back(dir);
                    offset.push_back(dx + dy*int(voxel.dim.width()) + dz*int(voxel.dim.plane_size()));
                    scaling.push_back(float(std::exp(-r2)));
                }
        gr.init(voxel);
    }
    virtual void run(Voxel& voxel, VoxelData& data) override
    {
        if(!hgqi)
            return;
        if(int(data.space[0]) == 0)
        {
            hraw[data.voxel_index] = 0;
            std::fill(data.odf.begin(),data.odf.end(),0.0f);
            return;
        }
        hraw[data.voxel_index] = data.space[0];
        auto I = tipl::make_image(voxel.dwi_data[0],voxel.dim);
        data.space.resize(voxel.bvalues.size());
        for(size_t i = 1;i < voxel.bvalues.size();++i)
        {
            int pos1 = int(data.voxel_index);
            int pos2 = int(data.voxel_index);
            pos1 += int(offset[i-1]);
            pos2 -= int(offset[i-1]);
            if(pos1 < 0 || pos1 >= int(voxel.dim.size()))
                pos1 = pos2;
            if(pos2 < 0 || pos2 >= int(voxel.dim.size()))
                pos2 = pos1;
            data.space[i] = std::fabs(data.space[0]-(I[pos1]+I[pos2])/2.0f)*scaling[i-1];
        }

        gr.run(voxel,data);
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer) override
    {
        mat_writer.write<tipl::io::masked_sloped>("hraw",hraw,voxel.dim.plane_size());
    }
};

class SchemeConverter : public BaseProcess
{
    GQI_Recon from,to;
    std::vector<int> piv;
    std::vector<tipl::vector<3,float> > bvectors;
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
            std::ifstream read(voxel.file_name);
            std::vector<float> values;
            std::copy(std::istream_iterator<float>(read),std::istream_iterator<float>(),std::back_inserter(values));
            for(unsigned int i = 0;i < values.size()/4;++i)
            {
                if(values[i*4] == 0.0f)
                    continue;
                bvalues.push_back(voxel.param[1]);
                bvectors.push_back(tipl::vector<3,float>(values[i*4+1],values[i*4+2],values[i*4+3]));
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

        b0.resize(voxel.dim.size());

        Rt.resize(dwi.size()*dwi.size());
        tipl::mat::transpose(&*to.sinc_ql.begin(),&*Rt.begin(),tipl::shape<2>(dwi.size(),dwi.size()));
        A.resize(dwi.size()*dwi.size());
        piv.resize(dwi.size());
        tipl::mat::product_transpose(&*Rt.begin(),&*Rt.begin(),&*A.begin(),
                                       tipl::shape<2>(dwi.size(),dwi.size()),tipl::shape<2>(dwi.size(),dwi.size()));
        float max_value = tipl::max_value(A);
        for (unsigned int i = 0,index = 0; i < dwi.size(); ++i,index += dwi.size() + 1)
            A[index] += max_value*voxel.param[2];
        tipl::mat::lu_decomposition(A.begin(),piv.begin(),tipl::shape<2>(dwi.size(),dwi.size()));

        total_negative_value = 0;
        total_value = 0;

        voxel.recon_report
                << " The converted HARDI has a total of " << dwi.size() << " diffusion sampling directions.";
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        {
            b0[data.voxel_index] = uint16_t(data.space[0]);
            data.space[0] = 0;
        }
        from.run(voxel,data);
        std::vector<float> hardi_data(dwi.size()),tmp(dwi.size());
        tipl::mat::vector_product(&*Rt.begin(),&*data.odf.begin(),&*tmp.begin(),tipl::shape<2>(dwi.size(),dwi.size()));
        tipl::mat::lu_solve(&*A.begin(),&*piv.begin(),&*tmp.begin(),&*hardi_data.begin(),tipl::shape<2>(dwi.size(),dwi.size()));
        for(unsigned int index = 0;index < dwi.size();++index)
        {
            if(hardi_data[index] < 0.0f)
                ++total_negative_value;
            else
                dwi[index][data.voxel_index] = uint16_t(hardi_data[index]);
            ++total_value;
        }
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        voxel.recon_report
                << " The percentage of the negative signals was " << float(100.0f)*total_negative_value/float(total_value) << "%.";
        std::vector<float> b_table(4); // space for b0
        for (unsigned int index = 0;index < bvectors.size();++index)
        {
            b_table.push_back(bvalues.front());
            std::copy(bvectors[index].begin(),bvectors[index].end(),std::back_inserter(b_table));
        }
        mat_writer.write("b_table",b_table,4);
        unsigned int image_num = 0;
        if(!b0.empty())
        {
            mat_writer.write<tipl::io::masked_sloped>("image0",b0,voxel.dim.plane_size());
            ++image_num;
        }
        for (unsigned int index = 0;index < dwi.size();++index)
        {
            std::ostringstream out;
            out << "image" << image_num;
            mat_writer.write<tipl::io::masked_sloped>(out.str(),dwi[index],voxel.dim.plane_size());
            ++image_num;
        }
    }
};



class RDI_Recon  : public BaseProcess
{
private:
    static float sinint(float x)
    {
        bool sgn = x > 0;
        x = std::fabs(x);
        float eps = 1e-15f;
        float x2 = x*x;
        float si;
        if(x == 0.0f)
            return 0.0f;
        if(x <= 16.0f)
        {
            float xr = x;
            si = x;
            for(unsigned int k = 1;k <= 40;++k)
            {
                si += (xr *= -0.5f*float(2*k-1)/float(k)/float(4*k*(k+1)+1)*x2);
                if(std::fabs(xr) < std::fabs(si)*eps)
                    break;
            }
            return sgn ? si:-si;
        }

        if(x <= 32.0f)
        {
            unsigned int m = uint32_t(std::floor(47.2f+0.82f*x));
            std::vector<float> bj(m+1);
            float xa1 = 0.0f;
            float xa0 = 1.0e-40f;
            for(unsigned int k=m;k>=1;--k)
            {
                float xa = 4.0f*float(k)*xa0/x-xa1;
                bj[k-1] = xa;
                xa1 = xa0;
                xa0 = xa;
            }
            float xs = bj[0];
            for(unsigned int k=3;k <= m;k += 2)
                xs += 2.0f*bj[k-1];
            for(unsigned int k=0;k < m;++k)
                bj[k] /= xs;
            float xr = 1.0f;
            float xg1 = bj[0];
            for(unsigned int k=2;k <= m;++k)
                xg1 += bj[k-1]*(xr *= 0.25f*(2*k-3)*(2*k-3)/((k-1)*(2*k-1)*(2*k-1))*x);
            xr = 1.0;
            float xg2 = bj[0];
            for(unsigned int k=2;k <= m;++k)
                xg2 += bj[k-1]*(xr *= 0.25f*(2*k-5)*(2*k-5)/((k-1)*(2*k-3)*(2*k-3))*x);
            si = x*std::cos(x/2.0f)*xg1+2.0f*std::sin(x/2.0f)*xg2-std::sin(x);
            return sgn ? si:-si;
        }

        float xr = 1.0f;
        float xf = 1.0f;
        for(unsigned int k=1;k <= 9;++k)
            xf += (xr *= -2.0f*k*(2*k-1)/x2);
        xr = 1.0f/x;
        float xg = xr;
        for(unsigned int k=1;k <= 8;++k)
            xg += (xr *= -2.0f*(2*k+1)*k/x2);
        si = 1.570796326794897f-xf*std::cos(x)/x-xg*std::sin(x)/x;
        return sgn ? si:-si;
    }
    std::vector<std::vector<float> > rdi_weightings;
public:
    virtual bool needed(Voxel& voxel)
    {
        return voxel.needs("rdi");
    }
    virtual void init(Voxel& voxel)
    {
        float sigma = voxel.param[0]; //optimal 1.24
        for(float L = 0.2f;L <= sigma;L+= 0.2f)
        {
            rdi_weightings.push_back(std::vector<float>(voxel.bvalues.size()));
            for(unsigned int index = 0;index < voxel.bvalues.size();++index)
            {
                float q = std::sqrt(voxel.bvalues[index]*0.018f);
                rdi_weightings.back()[index] = (q > 0)? sinint(L*q)/q:L;
            }
        }
    }
    virtual void run(Voxel&, VoxelData& data)
    {
        if(data.space.front() == 0.0f)
        {
            data.rdi = std::vector<float>(rdi_weightings.size());
            return;
        }
        float last_value = 0;
        std::vector<float> rdi_values(rdi_weightings.size());
        for(unsigned int index = 0;index < rdi_weightings.size();++index)
        {
            // force incremental
            rdi_values[index] = std::max<float>(last_value,tipl::vec::dot(rdi_weightings[index].begin(),rdi_weightings[index].end(),data.space.begin()));
            last_value = rdi_values[index];
        }
        data.rdi.swap(rdi_values);
    }
};
#endif//DDI_PROCESS_HPP
