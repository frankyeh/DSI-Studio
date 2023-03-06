#include "TIPL/tipl.hpp"
#include "TIPL/cuda/mem.hpp"
#include "TIPL/cuda/basic_image.hpp"
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
        std::cout << "│ │ cudaSetDevice error:" << cudaSetDevice(cur_gpu) << std::endl;
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
        error_msg = "cannot obtain GPU driver and device information (CUDA ERROR:";
        error_msg += std::to_string(int(cudaGetDeviceCount(&gpu_count)));
        error_msg += "). Please update the Nvidia driver and install CUDA Toolkit.";
        return;
    }
    std::cout << "├─CUDA Driver Version: " << Ver << " CUDA Run Time Version: " << CUDART_VERSION << std::endl;
    cuda_test<<<1,1>>>();
    if(cudaPeekAtLastError() != cudaSuccess)
    {
        error_msg = "Failed to lauch cuda kernel:";
        error_msg += cudaGetErrorName(cudaGetLastError());
        error_msg += ". Please update Nvidia driver.";
        return;
    }

    std::cout << "│ │ Device Count:" << gpu_count << std::endl;
    for (int i = 0; i < gpu_count; i++)
    {
        std::cout << "│ │ Device Number:" << std::to_string(i) << std::endl;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            error_msg = "Cannot obtain device information. Please update Nvidia driver";
            return;
        }
        auto arch = prop.major*10+prop.minor;
        std::cout << "│ │ Arch: " << arch << std::endl;
        std::cout << "│ │ Device name: " << prop.name << std::endl;
        std::cout << "│ │ Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "│ │ Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "│ │ Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;

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
        tipl::reg::cdm2_cuda(dIt,dIt2,dIs,dIs2,dd,inv_dd,terminated,param);
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

size_t linear_cuda(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound)
{
    distribute_gpu();
    return tipl::reg::linear_mr<tipl::reg::mutual_information_cuda>
            (from,from_vs,to,to_vs,arg,reg_type,[&](void){return terminated;},
                0.01,bound != tipl::reg::narrow_bound,bound);
}

size_t linear_cuda_refine(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              double precision)
{
    distribute_gpu();
    return tipl::reg::linear<tipl::reg::mutual_information_cuda>(
                from,from_vs,to,to_vs,arg,reg_type,[&](void){return terminated;},
                precision,false,tipl::reg::narrow_bound,10);
}




