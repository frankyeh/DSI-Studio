#ifndef DDI_PROCESS_HPP
#define DDI_PROCESS_HPP
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/math/special_functions/sinc.hpp>
#include "math/matrix_op.hpp"
#include "common.hpp"
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
        math::matrix_vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                    math::dyndim(data.odf.size(),voxel.bvalues.size()));
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {

    }
};


template<int k = 20>
class QSpaceAdoptor  : public QSpace2Odf
{
private:
    std::vector<std::vector<short> > images;
    std::vector<std::vector<float> > images_sinc;
    std::vector<float> b_table;
    unsigned int b0_index;
public:
    virtual void init(Voxel& voxel)
    {
        QSpace2Odf::init(voxel);

        float dif_sampling_length_6Dt = voxel.param[0]; //optimal 1.24
        b0_index = std::find(voxel.bvalues.begin(),voxel.bvalues.end(),0)-voxel.bvalues.begin();
        for (unsigned int index = 3; 1; index+=3)
        {
            image::vector<3,float> bvec(voxel.param + index);
            if (bvec[0] == 0.0 &&
                    bvec[1] == 0.0 &&
                    bvec[2] == 0.0)
                break;

            b_table.push_back(1000.0*dif_sampling_length_6Dt*dif_sampling_length_6Dt);
            b_table.push_back(bvec[0]);
            b_table.push_back(bvec[1]);
            b_table.push_back(bvec[2]);


            float r[9]; // 3-by-3 matrix
            rotation_matrix(r,bvec.begin());

            std::vector<float> sin_rq(voxel.bvectors.size());

            float angle[3],rC[3];
            for (unsigned int i = 0; i < k; ++i)
            {
                image::vector<3,float> dir;
                angle[0] = std::cos(2.0*M_PI*((float)i+1)/((float)k));
                angle[1] = std::sin(2.0*M_PI*((float)i+1)/((float)k));
                angle[2] = 0.0;
                math::matrix_vector_product(r,angle,dir.begin(),math::dim<3,3>());
                for (unsigned int j = 0; j < voxel.bvectors.size(); ++j)
                    if(voxel.bvalues[j] == 0)
                        sin_rq[j] += 0.0;
                    else
                        sin_rq[j] += boost::math::sinc_pi(
                                         (voxel.bvectors[j]*dir)*
                                         std::sqrt(voxel.bvalues[j]*0.01506)*dif_sampling_length_6Dt)*dif_sampling_length_6Dt;
            }

            for (unsigned int j = 0; j < voxel.bvectors.size(); ++j)
                sin_rq[j] /= (float)k;

            images_sinc.push_back(sin_rq);
            images.push_back(std::vector<short>(voxel.total_size));

        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(b0_index < voxel.bvalues.size())
            std::for_each(data.space.begin(),data.space.end(),boost::lambda::_1 /= data.space[b0_index]);
        QSpace2Odf::run(voxel,data);
        float min = *std::min_element(data.odf.begin(),data.odf.end());
        float scale = 1.0;
        if(b0_index < voxel.bvalues.size())
            scale = data.space[b0_index];
        for (unsigned int index = 0; index < images.size(); ++index)
        {
            std::vector<float>::const_iterator from = images_sinc[index].begin();
            std::vector<float>::const_iterator to = images_sinc[index].end();
            std::vector<float>::const_iterator iter = data.space.begin();
            float value = 0;
            for (; from != to; ++from,++iter)
                value += (*from)*(*iter);
            images[index][data.voxel_index] = value*scale;
        }
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {
        for (unsigned int index = 0; index < images.size(); ++index)
        {
            std::ostringstream out;
            out << "image" << index;
            mat_writer.add_matrix(out.str().c_str(),&*images[index].begin(),1,images[index].size());
        }
        mat_writer.add_matrix("b_table",&*b_table.begin(),4,b_table.size()/4);
    }
};


// for normalization
class RecordQA  : public BaseProcess
{
public:
    virtual void init(Voxel& voxel)
    {
        voxel.qa_map.resize(image::geometry<3>(voxel.matrix_width,voxel.matrix_height,voxel.slice_number));
        std::fill(voxel.qa_map.begin(),voxel.qa_map.end(),0.0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        voxel.qa_map[data.voxel_index] = data.fa[0];
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {

    }
};

#endif//DDI_PROCESS_HPP
