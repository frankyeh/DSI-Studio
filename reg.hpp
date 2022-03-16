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
               tipl::reg::cdm_param param = tipl::reg::cdm_param())
{
    if(It2.shape() == It.shape() && Is.shape() == Is2.shape())
    {
        std::cout << "dual modality normalization" << std::endl;
        if constexpr (tipl::use_cuda)
            cdm2_cuda(It,It2,Is,Is2,dis,inv_dis,terminated,param);
        else
            tipl::reg::cdm2(It,It2,Is,Is2,dis,inv_dis,terminated,param);
    }
    else
    {
        std::cout << "single modality normalization" << std::endl;
        tipl::reg::cdm(It,Is,dis,inv_dis,terminated,param);
    }

}

float linear_with_cc(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::large_bound);

float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::large_bound);


float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::transformation_matrix<float>& T,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::large_bound);


#endif//REG_HPP
