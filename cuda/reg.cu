#include <iostream>
#include <string>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void cuda_test(){
    ;
}

extern bool has_cuda;
extern int gpu_count;
bool distribute_gpu(void)
{
    static int cur_gpu = 0;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if(gpu_count <= 1)
        return true;
    if(cudaSetDevice(cur_gpu) != cudaSuccess)
        return false;
    ++cur_gpu;
    if(cur_gpu >= gpu_count)
        cur_gpu = 0;
    return true;
}

void check_cuda(std::string& error_msg)
{
    int Ver;
    if(cudaGetDeviceCount(&gpu_count) != cudaSuccess ||
       cudaDriverGetVersion(&Ver) != cudaSuccess)
    {
        error_msg = "Failed to obtain GPU driver and device information (CUDA ERROR ";
        error_msg += std::to_string(int(cudaGetDeviceCount(&gpu_count)));
        error_msg += "). Please install CUDA Toolkit or use the CPU version.";
        return;
    }
    std::cout << "├──CUDA Driver Version: " << Ver << " CUDA Run Time Version: " << CUDART_VERSION << std::endl;
    cuda_test<<<1,1>>>();
    if(cudaPeekAtLastError() != cudaSuccess)
    {
        error_msg = "failed to lauch cuda kernel:";
        error_msg += cudaGetErrorName(cudaGetLastError());
        error_msg += ". please update Nvidia driver.";
        return;
    }

    std::cout << "│  ├──device count: " << gpu_count << std::endl;
    for (int i = 0; i < gpu_count; i++)
    {
        std::cout << "│  ├──device number: " << std::to_string(i) << std::endl;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            error_msg = "cannot obtain device information. please update Nvidia driver";
            return;
        }
        auto arch = prop.major*10+prop.minor;
        std::cout << "│  ├──arch: " << arch << std::endl;
        std::cout << "│  ├──device name: " << prop.name << std::endl;
        std::cout << "│  ├──memory clock rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "│  ├──memory bus width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "│  ├──peak memory bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;

    }
    has_cuda = true;
}
