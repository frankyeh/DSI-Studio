#include "TIPL/tipl.hpp"
#include "TIPL/cuda/mem.hpp"
#include "TIPL/cuda/basic_image.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_test(){
    ;
}

bool check_cuda(std::string& error_msg)
{
    int nDevices,Ver;
    if(cudaGetDeviceCount(&nDevices) != cudaSuccess ||
       cudaDriverGetVersion(&Ver) != cudaSuccess)
    {
        error_msg = "Cannot obtain GPU driver and device information. Please install a Nvidia driver";
        return false;
    }

    std::cout << "Device Count:" << nDevices << std::endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            error_msg = "Cannot obtain device information. Please update Nvidia driver";
            return false;
        }
        auto arch = prop.major*10+prop.minor;
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Arch: " << arch << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;

        if(i == 0 && arch != CUDA_ARCH)
        {
            error_msg = "Incorrect CUDA Architecture. Please use DSI Studio (CUDA SM";
            error_msg += std::to_string(arch);
            error_msg += ").";
            return false;
        }
    }

    std::cout << "Driver Version: " << Ver << " DSI Studio CUDA Version: " << CUDART_VERSION << std::endl;
    cuda_test<<<1,1>>>();
    if(cudaPeekAtLastError() != cudaSuccess)
    {
        error_msg = "Failed to lauch cuda kernel:";
        error_msg += cudaGetErrorName(cudaGetLastError());
        error_msg += ". Please update Nvidia driver.";
        return false;
    }

    return true;
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
    std::cout << "normalization using GPU" << std::endl;
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
    dd.vector().copy_to(d);
    inv_d.resize(It.shape());
    inv_dd.vector().copy_to(inv_d);

    cudaDeviceSynchronize();

}

void adjust_vs(const tipl::image<3,float>& from,
               const tipl::vector<3>& from_vs,
               const tipl::image<3,float>& to,
               tipl::vector<3>& to_vs)
{
    std::cout << "FOV width:" << float(from.width())*from_vs[0] << " to " << float(to.width())*to_vs[0] << std::endl;
    if(float(from.width())*from_vs[0]*0.75f > float(to.width())*to_vs[0])
    {
        std::cout << "adjust voxel size due to match FOV" << std::endl;
        to_vs *= std::sqrt((float(from.plane_size())*from_vs[0]*from_vs[1])/
                           (float(to.plane_size())*to_vs[0]*to_vs[1]));
    }
}

float linear_with_cc(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound)
{
    size_t iterations = 3;
    if(reg_type == tipl::reg::affine)
    {
        std::cout << "affine registration using correlation" << std::endl;
        iterations = 5;
        adjust_vs(from,from_vs,to,to_vs);
    }
    else
        std::cout << "rigid body registration using correlation" << std::endl;


    float result = tipl::reg::linear<tipl::reg::correlation>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound);
    std::cout << "R:" << -result << std::endl;
    std::cout << "T:" << arg;
    return -result;
}

float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound)
{

    size_t iterations = 3;
    if(reg_type == tipl::reg::affine)
    {
        std::cout << "affine registration using mutual information" << std::endl;
        iterations = 5;
        adjust_vs(from,from_vs,to,to_vs);
    }
    else
        std::cout << "rigid body registration using mutual information" << std::endl;


    float result = 0.0f;
    if constexpr (tipl::use_cuda)
    {
        std::cout << "registration using GPU" << std::endl;
        result = tipl::reg::linear<tipl::reg::mutual_information_cuda>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound,iterations);
    }
    else
    {
        result = tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound,iterations);
    }
    std::cout << "MI:" << size_t(result) << std::endl;
    std::cout << "T:" << arg;
    return result;
}

float linear_with_mi(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::transformation_matrix<float>& T,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound)
{
    size_t iterations = 3;
    if(reg_type == tipl::reg::affine)
    {
        std::cout << "affine registration using mutual information" << std::endl;
        iterations = 5;
        adjust_vs(from,from_vs,to,to_vs);
    }
    else
        std::cout << "rigid body registration using mutual information" << std::endl;

    float result = 0.0f;
    tipl::affine_transform<float> arg;
    if constexpr (tipl::use_cuda)
    {
        std::cout << "registration using GPU" << std::endl;
        tipl::reg::linear<tipl::reg::mutual_information_cuda>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound,iterations);
    }
    else
    {
        tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,0.01,true,bound,iterations);
    }
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    std::cout << "MI:" << size_t(result) << std::endl;
    std::cout << "T:" << arg;
    return result;
}

