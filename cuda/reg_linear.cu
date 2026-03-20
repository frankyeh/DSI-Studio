#include "TIPL/prog.hpp"
#include "TIPL/reg/linear.hpp"

namespace tipl::reg
{

template
float linear_reg_imp<tipl::out,false,unsigned char,3>(std::true_type,
            std::shared_ptr<tipl::reg::linear_reg_runner<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);
template
float linear_reg_imp<tipl::out,true,unsigned char,3>(std::true_type,
            std::shared_ptr<tipl::reg::linear_reg_runner<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);

}
