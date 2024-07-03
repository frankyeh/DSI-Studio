#include "TIPL/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void cuda_test(){
    ;
}

extern bool has_cuda;
extern int gpu_count;
void distribute_gpu(void)
{
    static int cur_gpu = 0;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if(gpu_count <= 1)
        return;
    if(cudaSetDevice(cur_gpu) != cudaSuccess)
        std::cout << "│ │ cudaSetDevice error: " << cudaSetDevice(cur_gpu) << std::endl;
    ++cur_gpu;
    if(cur_gpu >= gpu_count)
        cur_gpu = 0;
}

void check_cuda(std::string& error_msg)
{
    int Ver;
    if(cudaGetDeviceCount(&gpu_count) != cudaSuccess ||
       cudaDriverGetVersion(&Ver) != cudaSuccess)
    {
        error_msg = "cannot obtain GPU driver and device information (CUDA ERROR ";
        error_msg += std::to_string(int(cudaGetDeviceCount(&gpu_count)));
        error_msg += "). please update the Nvidia driver and install CUDA Toolkit.";
        return;
    }
    std::cout << "├─CUDA Driver Version: " << Ver << " CUDA Run Time Version: " << CUDART_VERSION << std::endl;
    cuda_test<<<1,1>>>();
    if(cudaPeekAtLastError() != cudaSuccess)
    {
        error_msg = "failed to lauch cuda kernel:";
        error_msg += cudaGetErrorName(cudaGetLastError());
        error_msg += ". please update Nvidia driver.";
        return;
    }

    std::cout << "│ │ device count: " << gpu_count << std::endl;
    for (int i = 0; i < gpu_count; i++)
    {
        std::cout << "│ │ device number: " << std::to_string(i) << std::endl;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            error_msg = "cannot obtain device information. please update Nvidia driver";
            return;
        }
        auto arch = prop.major*10+prop.minor;
        std::cout << "│ │ arch: " << arch << std::endl;
        std::cout << "│ │ device name: " << prop.name << std::endl;
        std::cout << "│ │ memory clock rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "│ │ memory bus width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "│ │ peak memory bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;

    }
    has_cuda = true;
}

template<int dim>
void cdm_cuda_base(const std::vector<tipl::const_pointer_image<dim,unsigned char> >& It,
              const std::vector<tipl::const_pointer_image<dim,unsigned char> >& Is,
              tipl::image<dim,tipl::vector<dim> >& d,
              bool& terminated,
              const tipl::reg::cdm_param& param)
{
    distribute_gpu();
    tipl::device_image<dim,tipl::vector<dim> > dd(It[0].shape()),inv_dd(It[0].shape());
    std::vector<tipl::device_image<dim,unsigned char> > dIt(It.size()),dIs(Is.size());
    std::vector<tipl::const_pointer_device_image<dim,unsigned char> > pIt,pIs;
    std::copy(It.begin(),It.end(),dIt.begin());
    std::copy(Is.begin(),Is.end(),dIs.begin());
    for(auto& each : dIt)
        pIt.push_back(tipl::make_device_shared(each));
    for(auto& each : dIs)
        pIs.push_back(tipl::make_device_shared(each));

    try{
        tipl::reg::cdm(pIt,pIs,dd,terminated,param);
    }

    catch(std::runtime_error& er)
    {
        std::cout << "ERROR: " << er.what() << std::endl;
        std::cout << "switch to CPU" << std::endl;
        tipl::reg::cdm(It,Is,d,terminated,param);
        return;
    }
    d.resize(It[0].shape());
    dd.buf().copy_to(d);
    cudaDeviceSynchronize();

}

void cdm_cuda(const std::vector<tipl::const_pointer_image<3,unsigned char> >& It,
              const std::vector<tipl::const_pointer_image<3,unsigned char> >& Is,
              tipl::image<3,tipl::vector<3> >& d,
              bool& terminated,
              tipl::reg::cdm_param param)
{
    cdm_cuda_base(It,Is,d,terminated,param);
}

void cdm_cuda(const std::vector<tipl::const_pointer_image<2,unsigned char> >& It,
              const std::vector<tipl::const_pointer_image<2,unsigned char> >& Is,
              tipl::image<2,tipl::vector<2> >& d,
              bool& terminated,
              tipl::reg::cdm_param param)
{
    cdm_cuda_base(It,Is,d,terminated,param);
}


size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,float,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return  cost_type == tipl::reg::mutual_info ?
            reg->optimize<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated):
            reg->optimize<tipl::reg::correlation>(terminated);
}
size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return  cost_type == tipl::reg::mutual_info ?
            reg->optimize<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated):
            reg->optimize<tipl::reg::correlation>(terminated);
}

size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,float,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return  cost_type == tipl::reg::mutual_info ?
            reg->optimize_mr<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated):
            reg->optimize_mr<tipl::reg::correlation>(terminated);
}
size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    distribute_gpu();
    return  cost_type == tipl::reg::mutual_info ?
            reg->optimize_mr<tipl::reg::mutual_information<3,tipl::device_vector> >(terminated):
            reg->optimize_mr<tipl::reg::correlation>(terminated);
}


