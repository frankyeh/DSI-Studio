#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include "zlib.h"
#include "TIPL/tipl.hpp"
extern bool has_cuda;
void cdm2_cuda(const tipl::image<3>& It,
               const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& d,
               tipl::image<3,tipl::vector<3> >& inv_d,
               bool& terminated,
               tipl::reg::cdm_param param);

inline void cdm_common(const tipl::image<3>& It,
         const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& dis,
               tipl::image<3,tipl::vector<3> >& inv_dis,
               bool& terminated,
               tipl::reg::cdm_param param = tipl::reg::cdm_param(),
               bool use_cuda = true)
{
    if(use_cuda && has_cuda)
    {
        if constexpr (tipl::use_cuda)
        {
            cdm2_cuda(It,It2,Is,Is2,dis,inv_dis,terminated,param);
            return;
        }
    }
    tipl::reg::cdm2(It,It2,Is,Is2,dis,inv_dis,terminated,param);
}



tipl::vector<3> adjust_to_vs(const tipl::image<3,float>& from,
               const tipl::vector<3>& from_vs,
               const tipl::image<3,float>& to,
               const tipl::vector<3>& to_vs);

inline float linear_with_cc(const tipl::image<3,float>& from,
                              const tipl::vector<3>& from_vs,
                              const tipl::image<3,float>& to,
                              const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    auto new_to_vs = to_vs;
    if(reg_type == tipl::reg::affine)
        new_to_vs = adjust_to_vs(from,from_vs,to,to_vs);

    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,to_vs).to_affine_transform(arg,from,from_vs,to,new_to_vs);
    float result = tipl::reg::linear_mr<tipl::reg::correlation>(from,from_vs,to,new_to_vs,arg,tipl::reg::reg_type(reg_type),[&](void){return terminated;},0.01,bound);
    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,new_to_vs).to_affine_transform(arg,from,from_vs,to,to_vs);
    tipl::out() << "R: " << -result << std::endl;
    tipl::out() << arg << std::endl;
    return -result;
}

size_t linear_cuda(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound);
size_t linear_cuda_refine(const tipl::image<3,float>& from,
                          tipl::vector<3> from_vs,
                          const tipl::image<3,float>& to,
                          tipl::vector<3> to_vs,
                          tipl::affine_transform<float>& arg,
                          tipl::reg::reg_type reg_type,
                          bool& terminated,
                          double precision);

inline size_t linear_with_mi_refine(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              double precision = 0.01)
{
    if(has_cuda)
    {
        if constexpr (tipl::use_cuda)
            return linear_cuda_refine(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,precision);
    }
    return tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),[&](void){return terminated;},precision,false,tipl::reg::narrow_bound,10);
}
inline size_t linear_with_mi(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    auto new_to_vs = to_vs;
    if(reg_type == tipl::reg::affine)
        new_to_vs = adjust_to_vs(from,from_vs,to,to_vs);
    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,to_vs).to_affine_transform(arg,from,from_vs,to,new_to_vs);
    float result = 0.0f;
    if(has_cuda)
    {
        if constexpr (tipl::use_cuda)
            result = linear_cuda(from,from_vs,to,new_to_vs,arg,tipl::reg::reg_type(reg_type),terminated,bound);

    }
    if(result == 0.0f)
        result = tipl::reg::linear_mr<tipl::reg::mutual_information>
                (from,from_vs,to,new_to_vs,arg,tipl::reg::reg_type(reg_type),[&](void){return terminated;},
                    0.01,bound != tipl::reg::narrow_bound,bound);
    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,new_to_vs).to_affine_transform(arg,from,from_vs,to,to_vs);

    linear_with_mi_refine(from,from_vs,to,to_vs,arg,reg_type,terminated);
    tipl::out() << arg << std::endl;
    return result;
}


inline size_t linear_with_mi(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::transformation_matrix<float>& T,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    tipl::affine_transform<float> arg;
    size_t result = linear_with_mi(from,from_vs,to,to_vs,arg,reg_type,terminated,bound);
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    tipl::out() << T << std::endl;
    return result;
}

inline size_t linear_with_cc(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::transformation_matrix<float>& T,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    tipl::affine_transform<float> arg;
    size_t result = linear_with_cc(from,from_vs,to,to_vs,arg,reg_type,terminated,bound);
    tipl::out() << arg << std::endl;
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    tipl::out() << T << std::endl;
    return result;
}


#endif//REG_HPP
