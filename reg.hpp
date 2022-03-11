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
        {
            tipl::reg::cdm2(It,It2,Is,Is2,dis,terminated,param);
            tipl::invert_displacement(dis,inv_dis);
        }
    }
    else
    {
        std::cout << "single modality normalization" << std::endl;
        tipl::reg::cdm(It,Is,dis,terminated,param);
        tipl::invert_displacement(dis,inv_dis);
    }

}

float two_way_linear_cuda(const tipl::image<3,float>& I,
                          const tipl::vector<3>& Ivs,
                         const tipl::image<3,float>& J,
                         const tipl::vector<3>& Jvs,
                         tipl::transformation_matrix<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         tipl::affine_transform<float>* arg_min,
                         const float* bound);

inline float linear_common(const tipl::image<3,float>& I,
                         const tipl::vector<3>& Ivs,
                         const tipl::image<3,float>& J,
                         const tipl::vector<3>& Jvs,
                         tipl::transformation_matrix<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         tipl::affine_transform<float>* arg_min = nullptr,
                         const float* bound = tipl::reg::reg_bound)
{
    auto Jvs2 = Jvs;
    if(reg_type == tipl::reg::affine)
        Jvs2 *= std::sqrt((float(I.plane_size())*Ivs[0]*Ivs[1])/(float(J.plane_size())*Jvs[0]*Jvs[1]));

    if constexpr (tipl::use_cuda)
        return two_way_linear_cuda(I,Ivs,J,Jvs2,T,reg_type,terminated,arg_min,bound);
    else
        return tipl::reg::two_way_linear_mr<tipl::reg::mutual_information>(I,Ivs,J,Jvs2,T,reg_type,terminated,arg_min,bound);
}

#endif//REG_HPP
