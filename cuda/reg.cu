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

void cdm2_cuda(const tipl::image<3>& It,
               const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& d,
               tipl::image<3,tipl::vector<3> >& inv_d,
               bool& terminated,
               tipl::reg::cdm_param param)
{
    tipl::device_image<3> dIt(It),dIt2(It2),dIs(Is),dIs2(Is2);
    tipl::device_image<3,tipl::vector<3> > dd(It.shape()),inv_dd(It.shape());
    try{
        tipl::reg::cdm2(dIt,dIt2,dIs,dIs2,dd,inv_dd,terminated,param);
    }
    catch(std::runtime_error& er)
    {
        std::cout << "ERROR: " << er.what() << std::endl;
        std::cout << "switch to CPU" << std::endl;
        tipl::reg::cdm2(It,It2,Is,Is2,d,inv_d,terminated,param);
        return;
    }
    d.resize(It.shape());
    dd.buf().copy_to(d);
    inv_d.resize(It.shape());
    inv_dd.buf().copy_to(inv_d);

    cudaDeviceSynchronize();

}

size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,float> > reg,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize<tipl::reg::mutual_information_cuda>(terminated);
}
size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize_mr<tipl::reg::mutual_information_cuda>(terminated);
}

size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,float> > reg,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize<tipl::reg::mutual_information_cuda>(terminated);
}
size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                        bool& terminated)
{
    distribute_gpu();
    return reg->optimize_mr<tipl::reg::mutual_information_cuda>(terminated);
}


