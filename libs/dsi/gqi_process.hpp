#ifndef DDI_PROCESS_HPP
#define DDI_PROCESS_HPP
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/math/special_functions/sinc.hpp>
#include "basic_process.hpp"
#include "basic_voxel.hpp"
#include "image_model.hpp"

class CorrectB0  : public BaseProcess
{
public:
    std::vector<unsigned int> b0_images;
public:
    virtual void init(Voxel& voxel)
    {
        b0_images.clear();
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
            if(voxel.bvalues[index] == 0)
                b0_images.push_back(index);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        //average all the b0 iamges
        if((b0_images.size() == 1 && !voxel.half_sphere) || b0_images.empty())
            return;
        float sum_b0 = data.space[b0_images.front()];
        if(b0_images.size() >= 2)
        {
            for(unsigned int index = 1;index < b0_images.size();++index)
            {
                    sum_b0 += data.space[b0_images[index]];
                    data.space[b0_images[index]] = 0;
            }
            sum_b0 /= b0_images.size();
        }
        if(voxel.half_sphere)
                sum_b0 /= 2.0;
        data.space[b0_images.front()] = sum_b0;
    }
};


class QSpace2Odf  : public BaseProcess
{
public:
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
        image::matrix::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                    image::dyndim(data.odf.size(),data.space.size()));
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

    }
};

class QSpaceSpectral  : public BaseProcess
{
public:
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
        const unsigned int num_spec = 40;
        const float spec_bandwidth = 0.05;
        dis.clear();
        cdf.clear();
        sinc_ql.clear();
        sinc_ql_cdf.clear();
        dis.resize(num_spec);
        cdf.resize(num_spec);
        sinc_ql.resize(num_spec);
        sinc_ql_cdf.resize(num_spec);
        unsigned int odf_size = voxel.ti.half_vertices_count;
        float sigma = 0.0;
        float delta = spec_bandwidth/100.0;
        for(unsigned int n = 0;n < num_spec;++n)
        {
            sinc_ql[n].resize(odf_size*voxel.bvalues.size());
            sinc_ql_cdf[n].resize(odf_size*voxel.bvalues.size());
            dis[n].resize(voxel.dim.size());
            cdf[n].resize(voxel.dim.size());
            // calculate reconstruction matrix
            for (unsigned int j = 0,index = 0; j < odf_size; ++j)
                for (unsigned int i = 0; i < voxel.bvalues.size(); ++i,++index)
                    sinc_ql[n][index] = voxel.bvectors[i]*
                                 image::vector<3,float>(voxel.ti.vertices[j])*
                                   std::sqrt(voxel.bvalues[i]*0.01506); // £^G£_

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
            sigma += spec_bandwidth;
        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        for(unsigned int index = 0;index < sinc_ql.size();++index)
        {
            image::matrix::vector_product(&*sinc_ql[index].begin(),&*data.space.begin(),&*data.odf.begin(),
                                        image::dyndim(data.odf.size(),data.space.size()));
            dis[index][data.voxel_index] = std::max<float>(0.0,image::mean(data.odf.begin(),data.odf.end()));

            image::matrix::vector_product(&*sinc_ql_cdf[index].begin(),&*data.space.begin(),&*data.odf.begin(),
                                        image::dyndim(data.odf.size(),data.space.size()));
            cdf[index][data.voxel_index] = image::mean(data.odf.begin(),data.odf.end());

        }
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        for(unsigned int index = 0;index < sinc_ql.size();++index)
        {
            std::ostringstream out;
            if(index < 10)
                out << "dis0" << index;
            else
                out << "dis" << index;
            mat_writer.write(out.str().c_str(),&*dis[index].begin(),1,dis[index].size());

        }
        for(unsigned int index = 0;index < sinc_ql.size();++index)
        {
            image::divide(cdf[index],cdf.back());
            std::ostringstream out;
            if(index < 10)
                out << "cdf0" << index;
            else
                out << "cdf" << index;
            mat_writer.write(out.str().c_str(),&*cdf[index].begin(),1,cdf[index].size());
        }
        mat_writer.write("fa0",&*dis[0].begin(),1,dis[0].size());
        std::vector<short> index0(voxel.dim.size());
        mat_writer.write("index0",&*index0.begin(),1,index0.size());
    }
};


#endif//DDI_PROCESS_HPP
