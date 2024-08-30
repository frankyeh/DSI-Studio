#include "TIPL/reg/cdm.hpp"

namespace tipl::reg
{
template
void cdm_cuda<void,unsigned char,3>(const std::vector<tipl::const_pointer_image<3,unsigned char> >& It,
                   const std::vector<tipl::const_pointer_image<3,unsigned char> >& Is,
                   tipl::image<3,tipl::vector<3> >& d,
                   bool& terminated,
                   const cdm_param& param); //forces instantiation
template
void cdm_cuda<void,unsigned char,2>(const std::vector<tipl::const_pointer_image<2,unsigned char> >& It,
                   const std::vector<tipl::const_pointer_image<2,unsigned char> >& Is,
                   tipl::image<2,tipl::vector<2> >& d,
                   bool& terminated,
                   const cdm_param& param); //forces instantiation
}
