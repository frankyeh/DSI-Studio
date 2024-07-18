#include <vector>
#include "TIPL/reg/linear.hpp"
#include "TIPL/prog.hpp"
bool distribute_gpu(void);
float optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return cost_type == tipl::reg::mutual_info ?
                reg->optimize<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated) :
                reg->optimize<tipl::reg::correlation_cuda<3,tipl::device_vector> >(terminated);
}

float optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return cost_type == tipl::reg::mutual_info ?
                reg->optimize_mr<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated) :
                reg->optimize_mr<tipl::reg::correlation_cuda<3,tipl::device_vector> >(terminated);
}


