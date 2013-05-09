#ifndef DSI_PROCESS_HPP
#define DSI_PROCESS_HPP
#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include "basic_process.hpp"
#include "basic_voxel.hpp"
#include "space_mapping.hpp"

class QSpace2Pdf  : public BaseProcess
{
    std::vector<unsigned int> qspace_mapping1;
    std::vector<unsigned int> qspace_mapping2;
    std::vector<float> hanning_filter;
    std::auto_ptr<image::fftn<3> > fft;
public:
    static double get_min_b(const Voxel& voxel)
    {
        float b_min;
        {
            std::vector<float> bvalues(voxel.bvalues.begin(),voxel.bvalues.end());
            do
            {
                b_min = *std::min_element(bvalues.begin(),bvalues.end());
                if (b_min != 0)
                    return b_min;
                bvalues.erase(std::min_element(bvalues.begin(),bvalues.end()));
            }
            while (!bvalues.empty());
        }
        return 0;
    }
    static void get_q_table(const Voxel& voxel,std::vector<image::vector<3,int> >& q_table)
    {
        float b_min = get_min_b(voxel);
        unsigned int n = voxel.bvalues.size();

        for (unsigned int index = 0; index < n; ++index)
        {
            int q2 = std::floor(std::abs(voxel.bvalues[index]/b_min)+0.5);
            image::vector<3,float> bvec = voxel.bvectors[index];
            bvec.normalize();
            bvec *= std::sqrt(std::abs(voxel.bvalues[index]/b_min));
            bvec[0] = std::floor(bvec[0]+0.5);
            bvec[1] = std::floor(bvec[1]+0.5);
            bvec[2] = std::floor(bvec[2]+0.5);
            q_table.push_back(image::vector<3,int>(bvec[0],bvec[1],bvec[2]));
        }

    }
public:
    virtual void init(Voxel& voxel)
    {
        unsigned int n = voxel.bvalues.size();
        qspace_mapping1.resize(n);
        qspace_mapping2.resize(n);
        hanning_filter.resize(n);
        std::vector<image::vector<3,int> > q_table;
        get_q_table(voxel,q_table);


        float filter_width = voxel.param[0];
        for (unsigned int index = 0; index < n; ++index)
        {
            int x = q_table[index][0];
            int y = q_table[index][1];
            int z = q_table[index][2];

            qspace_mapping1[index] = SpaceMapping<dsi_range>::getIndex(x,y,z);
            qspace_mapping2[index] = SpaceMapping<dsi_range>::getIndex(-x,-y,-z);
            float r = (float)std::sqrt((float)(x*x+y*y+z*z));
            hanning_filter[index] = 0.5 * (1.0+std::cos(2.0*r*M_PI/((float)filter_width)));
        }
        fft.reset(new image::fftn<3>(image::geometry<3>(space_length,space_length,space_length)));
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> pdf(qspace_size),buffer(qspace_size);
        for (unsigned int index = 0; index < qspace_mapping1.size(); ++index)
        {
            float value = data.space[index]*hanning_filter[index];
            pdf[qspace_mapping1[index]] += value;
            pdf[qspace_mapping2[index]] += value;
        }
        fft->apply_inverse(pdf,buffer);
        data.space.resize(qspace_size);
        for(unsigned int index = 0; index < qspace_size; ++index)
            data.space[index] = std::abs(pdf[index]);
    }

};

/** Perform integration over r
 */
struct Pdf2Odf : public BaseProcess
{
    std::vector<SamplePoint> sample_group;
    unsigned int b0_index;
public:
    virtual void init(Voxel& voxel)
    {
        // initialize dsi sample points
        unsigned int odf_size = voxel.ti.half_vertices_count;
        for (unsigned int index = 0; index < odf_size; ++index)
            for (float r = odf_min_radius; r <= odf_max_radius; r += odf_sampling_interval)
                sample_group.push_back(
                    SamplePoint(index,voxel.ti.vertices[index][0]*r,
                                voxel.ti.vertices[index][1]*r,
                                voxel.ti.vertices[index][2]*r,r*r));
        b0_index = SpaceMapping<dsi_range>::getIndex(0,0,0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        // calculate sum(p(u*r)*r^2)dr
        using namespace boost::lambda;
        std::fill(data.odf.begin(),data.odf.end(),0.0f);
        std::for_each(sample_group.begin(),sample_group.end(),
                      bind(&SamplePoint::sampleODFValueWeighted,
                           boost::lambda::_1,
                           boost::ref(data.space),
                           boost::ref(data.odf)));

        // normalization
        float sum = image::mean(data.odf.begin(),data.odf.end());
        if (sum != 0.0)
            std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= (data.space[b0_index]/sum));
    }
};

#endif//DSI_PROCESS_HPP
