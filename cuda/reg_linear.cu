#include <vector>
#include "TIPL/reg/linear.hpp"
#include "TIPL/prog.hpp"
void distribute_gpu(void);
size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated);
}

size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize_mr<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated);
}


