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
    float decomposition_fraction;
protected:
    std::vector<float> fiber_ratio;
    float max_iso;
protected:
    std::vector<float> Rt,oRt;
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
        }
        oRt = Rt;
        for (unsigned int i = 0; i < half_odf_size; ++i)
        {
            normalize_vector(Rt.begin()+i*half_odf_size,Rt.begin()+(i+1)*half_odf_size);
            std::for_each(oRt.begin()+i*half_odf_size,oRt.begin()+(i+1)*half_odf_size,
                          boost::lambda::_1 /= image::mean(oRt.begin()+i*half_odf_size,oRt.begin()+(i+1)*half_odf_size));

        }
    }
    /*
     * the step is assigned to make the correlation c(y,xi) and c(y,xj) equal
     * xi(y-step*u) = xj(y-step*u)
     * xi*y-xj*y = step(xi*u-xj*u);
     */
    template<typename iterator1,typename iterator2,typename iterator3>
    float lar_get_step(iterator1 u,iterator2 xi,iterator2 xj,iterator3 y,unsigned int y_dim)
    {
        float t1 = 0,t2 = 0;
        for(int i = 0;i < y_dim;++i)
        {
            float dif = (xi[i]-xj[i]);
            t1 += dif*y[i];
            t2 += dif*u[i];
        }
        if(t2 + 1.0 == 1.0)
            return 0.0;
        return t1/t2;
    }

    void lasso(const std::vector<float>& y,const std::vector<float>& x,std::vector<float>& w,unsigned int max_fiber)
    {
        unsigned int y_dim = y.size();
        std::vector<float> residual(y);
        std::vector<float> tmp(y_dim);
        std::vector<char> fib_map(y_dim);
        std::vector<int> fib_list;
        w.resize(y_dim);

        std::vector<float> Xt;

        for(int fib_index = 1;fib_index < max_fiber;++fib_index)
        {
            math::matrix_vector_product(&*x.begin(),&*residual.begin(),&*tmp.begin(),
                                            math::dyndim(y_dim,y_dim));
            std::vector<float> u(y_dim);
            std::vector<float> s(fib_index);
            if(fib_index == 1)
            {
                // get the most correlated orientation
                int dir = std::max_element(tmp.begin(),tmp.end())-tmp.begin();
                if(tmp[dir] < 0.0)
                    return;
                fib_map[dir] = 1;
                fib_list.push_back(dir);
                std::copy(x.begin()+dir*y_dim,x.begin()+(dir+1)*y_dim,u.begin());
                Xt = u;
                s[0] = 1.0;
            }
            else
            {
                std::vector<float> XtX(fib_index*fib_index);
                std::vector<int> piv(y_dim);
                math::matrix_product_transpose(Xt.begin(),Xt.begin(),XtX.begin(),math::dyndim(fib_index,y_dim),math::dyndim(fib_index,y_dim));
                math::matrix_lu_decomposition(XtX.begin(),piv.begin(),math::dyndim(fib_index,fib_index));
                math::matrix_lu_solve(XtX.begin(),piv.begin(),math::one<float>(),s.begin(),math::dyndim(fib_index,fib_index));
                float Aa = 1.0/std::sqrt(std::accumulate(s.begin(),s.end(),0.0f));
                image::multiply_constant(s.begin(),s.end(),Aa);
                math::matrix_product(s.begin(),Xt.begin(),u.begin(),math::dyndim(1,fib_index),math::dyndim(fib_index,y_dim));
            }

            unsigned int dir = 0;
            float min_step_value = std::numeric_limits<float>::max();
            std::vector<float>::const_iterator xi = x.begin()+fib_list.front()*y_dim;
            std::vector<float>::const_iterator xj = x.begin();
            for(unsigned int cur_dir = 0;cur_dir < y_dim;xj += y_dim,++cur_dir)
            {
                if(fib_map[cur_dir])
                    continue;

                float value = lar_get_step(u.begin(),xi,xj,residual.begin(),y_dim);
                if(value < min_step_value)
                    {
                        dir = cur_dir;
                        min_step_value = value;
                    }
            }
            /*
            std::vector<float> new_w(w);
            for(int index = 0;index < s.size();++index)
                if((new_w[fib_list[index]] += s[index]*min_step_value) < 0.0)
                    return;
            new_w.swap(w);
            */
            for(int index = 0;index < s.size();++index)
                w[fib_list[index]] += s[index]*min_step_value;

            // update residual
            math::vector_op_axpy(residual.begin(),residual.end(),-min_step_value,u.begin());

            // update the X matrix
            std::copy(x.begin()+dir*y_dim,x.begin()+(dir+1)*y_dim,std::back_inserter(Xt));
            fib_list.push_back(dir);
            fib_map[dir] = 1;
        }
    }

    void lasso2(const std::vector<float>& y,const std::vector<float>& x,std::vector<float>& w,unsigned int max_fiber)
    {
        unsigned int y_dim = y.size();
        std::vector<float> residual(y);
        std::vector<float> tmp(y_dim);
        std::vector<char> fib_map(y_dim);
        w.resize(y_dim);

        float step_size = decomposition_fraction;
        unsigned int max_iter = ((float)max_fiber/step_size);
        unsigned char total_fiber = 0;
        for(int fib_index = 0;fib_index < max_iter;++fib_index)
        {
            // calculate the correlation with each SFO

            math::matrix_vector_product(&*x.begin(),&*residual.begin(),&*tmp.begin(),
                                            math::dyndim(y_dim,y_dim));
            // get the most correlated orientation
            int dir = std::max_element(tmp.begin(),tmp.end())-tmp.begin();
            float corr = tmp[dir];
            if(corr < 0.0)
                break;
            if(!fib_map[dir])
            {
                total_fiber++;
                if(total_fiber > max_fiber)
                    break;
                fib_map[dir] = 1;
            }
            std::vector<float>::const_iterator xi = x.begin()+dir*y_dim;
            if(fib_index == 0)
            {
                std::vector<float>::const_iterator xj = x.begin();
                float min_step_value = std::numeric_limits<float>::max();
                int min_step_dir = 0;
                for(int index = 0;index < y_dim;++index,xj += y_dim)
                {
                    if(index == dir)
                        continue;
                    float value = lar_get_step(xi,xi,xj,residual.begin(),y_dim);
                    if(value < min_step_value)
                    {
                        min_step_dir = index;
                        min_step_value = value;
                    }
                }
                w[dir] += min_step_value;
                math::vector_op_axpy(residual.begin(),residual.end(),-min_step_value,xi);
            }
            else
            {
                w[dir] += corr*step_size;
                math::vector_op_axpy(residual.begin(),residual.end(),-corr*step_size,xi);
            }
        }
    }

public:
    virtual void init(Voxel& voxel)
    {
        if (!voxel.odf_decomposition)
            return;
        decomposition_fraction = voxel.param[3];
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
        lasso2(data.odf,Rt,w,voxel.max_fiber_number << 1);
        //lasso(data.odf,Rt,w,voxel.max_fiber_number << 1);


        //std::fill(data.odf.begin(),data.odf.end(),1.0f);
        //image::add(data.odf.begin(),data.odf.end(),w.begin());
        //return;
        std::vector<int> dir_list;

        for(unsigned int index = 0;index < half_odf_size;++index)
            if(w[index] > 0.0)
                dir_list.push_back(index);

        std::vector<float> results;
        int has_isotropic = 1;
        while(1)
        {
            if(dir_list.empty())
            {
                results.resize(1);
                results[0] = image::mean(old_odf.begin(),old_odf.end());
                has_isotropic = 1;
                break;
            }
            std::vector<float> RRt;
            if(has_isotropic)
            {
                RRt.resize(half_odf_size);
                std::fill(RRt.begin(),RRt.end(),1.0);
            }
            for (unsigned int index = 0;index < dir_list.size();++index)
            {
                int dir = dir_list[index];
                std::copy(oRt.begin()+dir*half_odf_size,
                          oRt.begin()+(1+dir)*half_odf_size,
                          std::back_inserter(RRt));
            }
            results.resize(dir_list.size()+has_isotropic);

            math::matrix_pseudo_inverse_solve(&*RRt.begin(),&*old_odf.begin(),&*results.begin(),math::dyndim(results.size(),half_odf_size));

            float threshold = 0.0;//std::max<float>(*std::max_element(results.begin()+has_isotropic,results.end())*0.25,0.0);
            int min_index = std::min_element(results.begin()+has_isotropic,results.end())-results.begin();
            if(results[min_index] < threshold)
            {
                dir_list.erase(dir_list.begin()+min_index-has_isotropic);
                results.erase(results.begin()+min_index);
                continue;
            }
            if(has_isotropic && results[0] < 0.0)
            {
                has_isotropic = 0;
                continue;
            }
            break;
        }
        float fiber_sum = std::accumulate(results.begin()+has_isotropic,results.end(),0.0f);

        data.min_odf = has_isotropic ? std::max<float>(results[0],0.0):0.0;
        std::fill(data.odf.begin(),data.odf.end(), data.min_odf);
        for(int index = 0;index < dir_list.size();++index)
            data.odf[dir_list[index]] += results[index+has_isotropic];

        if(has_isotropic && results[0] > max_iso)
            max_iso = results[0];
        fiber_ratio[data.voxel_index] = fiber_sum;
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
