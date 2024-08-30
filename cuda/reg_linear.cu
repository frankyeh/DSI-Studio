#include "TIPL/reg/linear.hpp"

namespace tipl::reg
{

template
float optimize_mi_cuda<false,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);
template
float optimize_mi_cuda<true,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);

}
