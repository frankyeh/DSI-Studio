#ifndef ODF_DECOMPOSITION_HPP
#define ODF_DECOMPOSITION_HPP
#include <boost/lambda/lambda.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <map>
#include "image/image.hpp"
#include "basic_process.hpp"
#include "basic_voxel.hpp"

struct ODFDecomposition : public BaseProcess
{
    std::vector<std::vector<unsigned char> > is_neighbor;
    float decomposition_fraction;
protected:
    std::vector<float> fiber_ratio;
    float max_iso;
protected:
    std::vector<float> Rt,oRt;
    unsigned int half_odf_size;
    unsigned char m;

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
        float length = image::vec::norm2(from,to);
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
                          boost::lambda::_1 /= *std::max_element(oRt.begin()+i*half_odf_size,oRt.begin()+(i+1)*half_odf_size));

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

            image::mat::vector_product(&*x.begin(),&*residual.begin(),&*tmp.begin(),
                                            image::dyndim(y_dim,y_dim));
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
                image::vec::axpy(residual.begin(),residual.end(),-min_step_value,xi);
            }
            else
            {
                w[dir] += corr*step_size;
                image::vec::axpy(residual.begin(),residual.end(),-corr*step_size,xi);
            }
        }
    }

public:
    virtual void init(Voxel& voxel)
    {
        if (!voxel.odf_decomposition)
            return;
        voxel.recon_report << "Diffusion ODF decomposition (Yeh et al., PLoS ONE 8(10): e75747, 2013) was conducted using a decomposition fraction of " << voxel.param[3];
        decomposition_fraction = voxel.param[3];
        m = std::max<int>(voxel.param[4],voxel.max_fiber_number);
        fiber_ratio.resize(voxel.dim.size());
        max_iso = 0.0;
        half_odf_size = voxel.ti.half_vertices_count;
        is_neighbor.resize(half_odf_size);
        for(unsigned int index = 0;index < half_odf_size;++index)
            is_neighbor[index].resize(half_odf_size);
        for(unsigned int index = 0;index < voxel.ti.faces.size();++index)
        {
            short i1 = voxel.ti.faces[index][0];
            short i2 = voxel.ti.faces[index][1];
            short i3 = voxel.ti.faces[index][2];
            if (i1 >= half_odf_size)
                i1 -= half_odf_size;
            if (i2 >= half_odf_size)
                i2 -= half_odf_size;
            if (i3 >= half_odf_size)
                i3 -= half_odf_size;
            is_neighbor[i1][i2] = 1;
            is_neighbor[i2][i1] = 1;
            is_neighbor[i1][i3] = 1;
            is_neighbor[i3][i1] = 1;
            is_neighbor[i2][i3] = 1;
            is_neighbor[i3][i2] = 1;
        }
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
        lasso2(data.odf,Rt,w,m);

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

            image::mat::pseudo_inverse_solve(&*RRt.begin(),&*old_odf.begin(),&*results.begin(),image::dyndim(results.size(),half_odf_size));

            //  drop negative
            int min_index = std::min_element(results.begin()+has_isotropic,results.end())-results.begin();
            if(results[min_index] < 0.0)
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

            // drop non local maximum
            std::vector<int> neighbors;
            for(unsigned int i = 0;i < dir_list.size();++i)
                for(unsigned int j = i+1;j < dir_list.size();++j)
                    if(is_neighbor[dir_list[i]][dir_list[j]])
                    {
                        neighbors.push_back(i);
                        neighbors.push_back(j);
                    }
            if(neighbors.empty())
                break;
            int smallest_neighbor = neighbors[0];
            float value = results[smallest_neighbor+has_isotropic];
            for(unsigned int i = 1;i < neighbors.size();++i)
                if(results[neighbors[i]+has_isotropic] < value)
                {
                    smallest_neighbor = neighbors[i];
                    value = results[smallest_neighbor+has_isotropic];
                }
            dir_list.erase(dir_list.begin()+smallest_neighbor);
            results.erase(results.begin()+smallest_neighbor+has_isotropic);
        }
        float fiber_sum = std::accumulate(results.begin()+has_isotropic,results.end(),0.0f);

        data.min_odf = has_isotropic ? std::max<float>(results[0],0.0):0.0;
        std::fill(data.odf.begin(),data.odf.end(), data.min_odf);
        for(int index = 0;index < dir_list.size();++index)
            data.odf[dir_list[index]] += results[index+has_isotropic];

        if(data.min_odf > max_iso)
            max_iso = data.min_odf;
        fiber_ratio[data.voxel_index] = fiber_sum;
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if (!voxel.odf_decomposition)
            return;
        if(max_iso + 1.0 != 1.0)
            std::for_each(fiber_ratio.begin(),fiber_ratio.end(),boost::lambda::_1 /= max_iso);
        mat_writer.write("fiber_ratio",&*fiber_ratio.begin(),1,fiber_ratio.size());

    }


};

#endif//ODF_DECOMPOSITION_HPP
