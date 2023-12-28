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

size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<tipl::image<3,float>,tipl::image<3,float> > > reg,
                     bool& terminated);

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
    auto reg = tipl::reg::linear_reg(from,from_vs,to,new_to_vs,arg);
    reg->type = reg_type;
    reg->set_bound(bound);
    float result = reg->optimize(std::make_shared<tipl::reg::correlation>(),terminated);

    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,new_to_vs).to_affine_transform(arg,from,from_vs,to,to_vs);
    tipl::out() << "R: " << -result << std::endl;
    tipl::out() << arg << std::endl;
    return -result;
}


inline size_t linear_with_mi_refine(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated)
{

    auto reg = tipl::reg::linear_reg(from,from_vs,to,to_vs,arg);
    reg->type = reg_type;
    reg->set_bound(tipl::reg::narrow_bound,false);

    if(has_cuda)
    {
        if constexpr (tipl::use_cuda)
            return optimize_mi_cuda(reg,terminated);
    }
    return reg->optimize(std::make_shared<tipl::reg::mutual_information>(),terminated);
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

    auto reg = tipl::reg::linear_reg(from,from_vs,to,new_to_vs,arg);
    reg->type = reg_type;
    reg->set_bound(bound);
    if(has_cuda)
    {
        if constexpr (tipl::use_cuda)
            result = optimize_mi_cuda(reg,terminated);
    }
    if(result == 0.0f)
        result = reg->optimize_mr(std::make_shared<tipl::reg::mutual_information>(),terminated);

    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from,from_vs,to,new_to_vs).to_affine_transform(arg,from,from_vs,to,to_vs);
    tipl::out() << arg << std::endl;
    return result;
}
inline tipl::transformation_matrix<float> linear_with_mi(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    tipl::affine_transform<float> arg;
    linear_with_mi(from,from_vs,to,to_vs,arg,reg_type,terminated,bound);
    return tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
}


#endif//REG_HPP
