#include "TIPL/prog.hpp"
#include "TIPL/reg/linear.hpp"

namespace tipl::reg
{

template
float optimize_mi_cuda<tipl::out,false,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);
template
float optimize_mi_cuda<tipl::out,true,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);

}
