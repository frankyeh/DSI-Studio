#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include "TIPL/tipl.hpp"
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
    if(use_cuda)
    {
        if constexpr (tipl::use_cuda)
        {
            cdm2_cuda(It,It2,Is,Is2,dis,inv_dis,terminated,param);
            return;
        }
    }
    tipl::reg::cdm2(It,It2,Is,Is2,dis,inv_dis,terminated,param);
}



void adjust_vs(const tipl::image<3,float>& from,
               const tipl::vector<3>& from_vs,
               const tipl::image<3,float>& to,
               tipl::vector<3>& to_vs);

inline float linear_with_cc(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    if(reg_type == tipl::reg::affine)
        adjust_vs(from,from_vs,to,to_vs);
    float result = tipl::reg::linear<tipl::reg::correlation>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound);
    std::cout << "R:" << -result << std::endl;
    std::cout << "T:" << arg;
    return -result;
}

float linear_cuda(const tipl::image<3,float>& from,
                  tipl::vector<3> from_vs,
                  const tipl::image<3,float>& to,
                  tipl::vector<3> to_vs,
                  tipl::affine_transform<float>& arg,
                  tipl::reg::reg_type reg_type,
                  bool& terminated,
                  const float* bound = tipl::reg::reg_bound);


inline float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    if(reg_type == tipl::reg::affine)
        adjust_vs(from,from_vs,to,to_vs);

    float result = 0.0f;
    if constexpr (tipl::use_cuda)
    {
        result = linear_cuda(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,bound);
    }
    else
    {
        result = tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound);
    }
    std::cout << "MI:" << size_t(result) << std::endl;
    std::cout << arg;
    return result;
}


inline float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::transformation_matrix<float>& T,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    tipl::affine_transform<float> arg;
    auto result = linear_with_mi(from,from_vs,to,to_vs,arg,reg_type,terminated,bound);
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    return result;
}


#endif//REG_HPP
