#ifndef RDSI_PROCESS_HPP
#define RDSI_PROCESS_HPP
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

class RQSpace2Pdf  : public BaseProcess
{
    std::vector<float> hanning_filter;
    std::vector<std::vector<float> > data2odf;

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
    static double get_max_b(const Voxel& voxel)
    {
        float b_max;
        {
            std::vector<float> bvalues(voxel.bvalues.begin(),voxel.bvalues.end());
            do
            {
                b_max = *std::max_element(bvalues.begin(),bvalues.end());
                if (b_max != 0)
                    return b_max;
                bvalues.erase(std::max_element(bvalues.begin(),bvalues.end()));
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
    static void get_b_table(const Voxel& voxel,std::vector<image::vector<3,float> >& b_table)
    {
        float b_max = get_max_b(voxel);
        unsigned int n = voxel.bvalues.size();

        for (unsigned int index = 0; index < n; ++index)
        {
            image::vector<3,float> bvec = voxel.bvectors[index];
            std::for_each(bvec.begin(),bvec.end(),boost::lambda::_1 *= -(sqrt(voxel.bvalues[index])/sqrt(b_max)));
            b_table.push_back(image::vector<3,float>(bvec[0],bvec[1],bvec[2]));
        }

    }
    static double get_b_step(const Voxel& voxel){
        double b_step;

        std::vector<float> bvalues(voxel.bvalues.begin(),voxel.bvalues.end());
        std::sort(bvalues.begin(),bvalues.end());

        std::vector<float>::iterator it;
        it = std::unique (bvalues.begin(), bvalues.end());
        bvalues.resize( std::distance(bvalues.begin(),it) );

        for(int i=0; i < (bvalues.size()-1); i++){
            bvalues[i] = bvalues[i+1]-bvalues[i];
        }
        bvalues.resize(bvalues.size()-1);

        std::sort(bvalues.begin(),bvalues.end());

        it = std::unique (bvalues.begin(), bvalues.end());
        bvalues.resize( std::distance(bvalues.begin(),it) );

        b_step = bvalues[bvalues.size()-1];
        return b_step;
    }
    static float sincpp(float x){
        float y;
        if (std::fabs(x) <= 0.001)
            y = -1.0/3.0 + x*x/1;
        else
            y = 2.0*sin(x)/x/x/x - 2.0*cos(x)/x/x - sin(x)/x;
        return y;
    }
public:
    virtual void init(Voxel& voxel)
    {
        // calculate the reconstruction matrix
        float edge_factor = voxel.param[5];
        float filter_width = voxel.param[0];
        unsigned int n = voxel.bvalues.size();
        hanning_filter.resize(n);
        float b_max = get_max_b(voxel);

        // initialize dsi sample points
        unsigned int odf_size = voxel.ti.half_vertices_count;

        // initialize q_table
        std::vector<image::vector<3,float> > b_table;
        get_b_table(voxel,b_table);

        // initialize q_table
        std::vector<image::vector<3,int> > q_table;
        get_q_table(voxel,q_table);

        //find b_step
        float b_step = get_b_step(voxel);
        std::cout << " b_step " << b_step << std::endl;

        for (unsigned int index = 0; index < n; ++index)
        {
            int x = q_table[index][0];
            int y = q_table[index][1];
            int z = q_table[index][2];

            // hanning-filter and h-scaling
            float r = (float)std::sqrt((float)(x*x+y*y+z*z));
            hanning_filter[index] = 0.5 * (1.0+std::cos(2.0*r*M_PI/((float)filter_width))) * sqrt(voxel.bvalues[index]/b_max);

            // rescale q_table
            for (unsigned int index2 = 0; index2 < 3; ++index2)
                b_table[index][index2]= b_table[index][index2] * sqrt(voxel.bvalues[index]);
        }

        // Rmax
        float Rmax = edge_factor * 1.0/sqrt((float)b_step)/2.0;

        for (unsigned int index = 0; index < odf_size; ++index)
        {
            std::vector<float> row;
            for (unsigned int index2 = 0; index2 < n; ++index2){
                // E
                float E=0;
                for (unsigned int ii = 0; ii < 3; ++ii)
                    E += voxel.ti.vertices[index][ii]*b_table[index2][ii];

                // data2odf-matrix
                row.push_back(-sincpp(2.0*M_PI*E*Rmax)*hanning_filter[index2]);

            }
            data2odf.push_back(row);
        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        using namespace boost::lambda;
        std::fill(data.odf.begin(),data.odf.end(),0.0f);

        for (unsigned int index = 0; index < data.odf.size(); ++index)
        {
            float tmp = 0.0;
            for (unsigned int index2 = 0; index2 < data.space.size(); ++index2){
                tmp += data2odf[index][index2]*data.space[index2];
            }
            data.odf[index] = tmp;
        }

        // minimum value
        //float min_odf = *std::min_element(data.odf.begin(), data.odf.end());
        //float max_odf = *std::max_element(data.odf.begin(), data.odf.end());

        // make the negative values zero
        for (unsigned int index = 0; index < data.odf.size(); ++index)
        {
            if( data.odf[index] < 0.0 )
                data.odf[index] = 0.0;
        }
        // normalization
        float sum;
        for (unsigned int index = 0; index < data.odf.size(); ++index)
            sum += data.odf[index];

        if (sum != 0.0){
            //std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= (data.space[0]/sum));
            std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= (1.0/sum));
        }
    }

};

#endif//RDSI_PROCESS_HPP
