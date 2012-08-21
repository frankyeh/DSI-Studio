#ifndef ODF_DECOMPOSITION_HPP
#define ODF_DECOMPOSITION_HPP
#include <boost/lambda/lambda.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <map>
#include "common.hpp"
#include "math/matrix_op.hpp"
#include "basic_process.hpp"
#include "basic_voxel.hpp"
#include "odf_process.hpp"

struct ODFDecomposition : public BaseProcess
{
    SearchLocalMaximum lm;
    boost::mutex mutex;
protected:
    std::vector<float> fiber_ratio;
    float max_iso;
protected:
    std::vector<float> Rt;
    unsigned int half_odf_size;

    double inner_angle(double cos_value)
    {
        double abs_cos = std::abs(cos_value);
        if (abs_cos > 1.0)
            abs_cos = 1.0;
        if (abs_cos < 0.0)
            abs_cos = 0.0;
        return std::acos(abs_cos)*2.0/M_PI;
    }
    double kernel_regression(const std::vector<float>& fiber_profile,
                             const std::vector<double>& inner_angles,
                             double cur_angle,double sigma)
    {
        sigma *= sigma;
        std::vector<double> weighting(inner_angles.size());
        for (unsigned int index = 0; index < inner_angles.size(); ++index)
        {
            double dx = cur_angle-inner_angles[index];
            weighting[index] = std::exp(-dx*dx/2.0/sigma);
        }
        double result = 0.0;
        for (unsigned int index = 0; index < fiber_profile.size(); ++index)
            result += fiber_profile[index]*weighting[index];
        result /= std::accumulate(weighting.begin(),weighting.end(),0.0);
        return result;
    }

    template<typename iterator_type>
    void normalize_vector(iterator_type from,iterator_type to)
    {
        std::for_each(from,to,boost::lambda::_1 -= std::accumulate(from,to,0.0)/((float)(to-from)));
        float length = math::vector_op_norm2(from,to);
        if(length+1.0 != 1.0)
            std::for_each(from,to,boost::lambda::_1 /= length);
    }

    void estimate_Rt(Voxel& voxel)
    {
        std::vector<double> inner_angles(half_odf_size);
        {
            unsigned int max_index = std::max_element(voxel.response_function.begin(),voxel.response_function.end())-voxel.response_function.begin();
            for (unsigned int index = 0; index < inner_angles.size(); ++index)
                inner_angles[index] = inner_angle(voxel.ti.vertices_cos(index,max_index));
        }


        Rt.resize(half_odf_size*half_odf_size);
        for (unsigned int i = 0,index = 0; i < half_odf_size; ++i)
        {
            for (unsigned int j = 0; j < half_odf_size; ++j,++index)
                Rt[index] = kernel_regression(voxel.response_function,inner_angles,inner_angle(voxel.ti.vertices_cos(i,j)),9.0/180.0*M_PI);
            normalize_vector(Rt.begin()+index-half_odf_size,Rt.begin()+index);
        }
    }
    void lasso(const std::vector<float>& y,const std::vector<float>& x,std::vector<float>& w,unsigned int max_fiber)
    {
        unsigned int y_dim = y.size();
        std::vector<float> residual(y);
        std::vector<float> tmp(y_dim);
        std::vector<char> fib_map(y_dim);
        w.resize(y_dim);

        float step_size = 0.5;
        unsigned int max_iter = ((float)max_fiber/step_size);
        unsigned char total_fiber = 0;
        for(int fib_index = 0;fib_index < max_iter;++fib_index)
        {
            // calculate the correlation with each SFO
            math::matrix_vector_product(&*x.begin(),&*residual.begin(),&*tmp.begin(),
                                            math::dyndim(y_dim,y_dim));
            // get the most correlated orientation
            int dir = std::max_element(tmp.begin(),tmp.end())-tmp.begin();
            if(!fib_map[dir])
            {
                total_fiber++;
                if(total_fiber > max_fiber)
                    break;
                fib_map[dir] = 1;
            }
            float corr = tmp[dir];
            if(corr <= 0.0)
                break;
            std::vector<float>::const_iterator SFO = x.begin()+dir*y_dim;

            /*
            float value = corr/2.0,lower = 0.0,upper = corr;
            if(fib_index == 0)
            for(int index = 0;index < 20;++index)
            {
                std::vector<float> next_re(residual);
                math::vector_op_axpy(next_re.begin(),next_re.end(),-value,SFO);
                math::matrix_vector_product(&*x.begin(),&*next_re.begin(),&*tmp.begin(),
                                                math::dyndim(y_dim,y_dim));
                if(std::max_element(tmp.begin(),tmp.end())-tmp.begin() == dir)
                    lower = value;
                else
                    {
                        upper = value;
                        if(index > 8)
                        {
                            w[dir] += value;
                            ++fib_index;
                            residual = next_re;
                            break;
                        }
                    }
                value = (upper+lower)/2.0;
            }
            else*/
            {
                w[dir] += corr*step_size;
                math::vector_op_axpy(residual.begin(),residual.end(),-corr*step_size,SFO);
            }
        }
    }

public:
    virtual void init(Voxel& voxel)
    {
        if (!voxel.odf_decomposition)
            return;
        lm.init(voxel);
        fiber_ratio.resize(voxel.total_size);
        max_iso = 0.0;
        half_odf_size = voxel.ti.half_vertices_count;
        // scale the free water diffusion to 1
        std::for_each(voxel.free_water_diffusion.begin(),voxel.free_water_diffusion.end(),(boost::lambda::_1 /= voxel.reponse_function_scaling));

        estimate_Rt(voxel);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {

        if (!voxel.odf_decomposition)
            return;
        std::vector<float> old_odf(data.odf);
        normalize_vector(data.odf.begin(),data.odf.end());
        std::vector<float> w;
        lasso(data.odf,Rt,w,voxel.max_fiber_number << 1);

        {
            std::vector<int> dir_list;

            {
                boost::mutex::scoped_lock lock(mutex);
                lm.search(w);
                std::map<float,unsigned short,std::greater<float> >::const_iterator iter = lm.max_table.begin();
                std::map<float,unsigned short,std::greater<float> >::const_iterator end = lm.max_table.end();

                for (unsigned int index = 0;iter != end;++index,++iter)
                    dir_list.push_back(iter->second);
            }

            std::vector<float> results;
            while(1)
            {
                if(dir_list.empty())
                {
                    results.resize(1);
                    results[1] = image::mean(old_odf.begin(),old_odf.end());
                    break;
                }
                std::vector<float> RRt(half_odf_size);
                std::fill(RRt.begin(),RRt.end(),1.0);
                for (unsigned int index = 0;index < dir_list.size();++index)
                {
                    int dir = dir_list[index];
                    std::copy(Rt.begin()+dir*half_odf_size,
                              Rt.begin()+(1+dir)*half_odf_size,
                              std::back_inserter(RRt));
                }
                results.resize(dir_list.size()+1);
                math::matrix_pseudo_inverse_solve(&*RRt.begin(),&*old_odf.begin(),&*results.begin(),math::dyndim(dir_list.size()+1,half_odf_size));

                float threshold = std::max<float>(*std::max_element(results.begin()+1,results.end())*0.25,0.0);
                int min_index = std::min_element(results.begin()+1,results.end())-results.begin();
                if(results[min_index] > threshold)
                    break;

                dir_list.erase(dir_list.begin()+min_index-1);
                results.erase(results.begin()+min_index);
            }
            float fiber_sum = std::accumulate(results.begin()+1,results.end(),0.0f);
            std::fill(data.odf.begin(),data.odf.end(),std::max<float>(results[0],0.0));
            for(int index = 0;index < dir_list.size();++index)
                data.odf[dir_list[index]] += results[index+1];

            if(results[0] > max_iso)
                max_iso = results[0];
            fiber_ratio[data.voxel_index] = fiber_sum;

        }
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {
        if (!voxel.odf_decomposition)
            return;

        if(max_iso + 1.0 != 1.0)
            std::for_each(fiber_ratio.begin(),fiber_ratio.end(),boost::lambda::_1 /= max_iso);
        mat_writer.add_matrix("fiber_ratio",&*fiber_ratio.begin(),1,fiber_ratio.size());

    }


};

#endif//ODF_DECOMPOSITION_HPP
