#include "tipl/tipl.hpp"
#include "tipl/cuda/mem.hpp"
#include "tipl/cuda/basic_image.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


bool check_cuda(std::string& error_msg)
{
    int nDevices,Ver;
    if(cudaGetDeviceCount(&nDevices) != cudaSuccess ||
       cudaDriverGetVersion(&Ver) != cudaSuccess)
    {
        error_msg = "Cannot obtain GPU driver and device information. Please install a Nvidia driver";
        return false;
    }
    std::cout << "Driver Version: " << Ver << " DSI Studio CUDA Version: " << CUDART_VERSION << std::endl;
    if (Ver < CUDART_VERSION)
    {
        error_msg = "Older version of CUDA driver found. Some functions may not be supported. Please consider update your Nvidia driver";
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
            error_msg = "Current DSI Studio (SM";
            error_msg += std::to_string(CUDA_ARCH);
            error_msg += ") does not match your GPU (SM";
            error_msg += std::to_string(arch);
            error_msg += "). Please download the correct SM version ";
            return false;
        }
    }

    return true;
}


float two_way_linear_cuda(const tipl::image<3,float>& I,
                          const tipl::vector<3>& Ivs,
                         const tipl::image<3,float>& J,
                         const tipl::vector<3>& Jvs,
                         tipl::transformation_matrix<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         tipl::affine_transform<float>* arg_min,
                         const float* bound)
{    
    float result(0);
    std::cout << "linear registration using GPU" << std::endl;
    try{
        result = tipl::reg::two_way_linear_mr<tipl::reg::mutual_information_cuda>
                (I,Ivs,J,Jvs,T,reg_type,terminated,arg_min,bound);
    }
    catch(std::runtime_error& er)
    {
        std::cout << "ERROR: " << er.what() << std::endl;
        std::cout << "switch to CPU" << std::endl;
        result = tipl::reg::two_way_linear_mr<tipl::reg::mutual_information>
                    (I,Ivs,J,Jvs,T,reg_type,terminated,arg_min,bound);
    }
    std::cout << "MI:" << size_t(result) << std::endl;
    std::cout << "T:" << T;
    return result;
}


void cdm2_cuda(const tipl::image<3>& It,
               const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& d,
               bool& terminated,
               tipl::reg::cdm_param param)
{
    std::cout << "normalization using GPU" << std::endl;
    tipl::device_image<3> dIt(It),dIt2(It2),dIs(Is),dIs2(Is2);
    tipl::device_image<3,tipl::vector<3> > dd(It.shape());
    tipl::reg::cdm2_cuda(dIt,dIt2,dIs,dIs2,dd,terminated,param);

    try{
        tipl::reg::cdm2_cuda(dIt,dIt2,dIs,dIs2,dd,terminated,param);
    }
    catch(std::runtime_error& er)
    {
        std::cout << "ERROR: " << er.what() << std::endl;
        std::cout << "switch to CPU" << std::endl;
        tipl::reg::cdm2(It,It2,Is,Is2,d,terminated,param);
        return;
    }
    d.resize(It.shape());
    dd.vector().copy_to(d);
}

float linear_mr(tipl::const_pointer_image<3,float> I,
                         const tipl::vector<3>& Ivs,
                         tipl::const_pointer_image<3,float> J,
                         const tipl::vector<3>& Jvs,
                         tipl::affine_transform<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         double precision,
                         const float* bound)
{
    std::cout << "linear registration using GPU" << std::endl;
    return tipl::reg::linear_mr<tipl::reg::mutual_information_cuda>(I,Ivs,J,Jvs,T,reg_type,terminated,precision,bound);
}

float linear_mr_uint8(tipl::const_pointer_image<3,unsigned char> I,
                         const tipl::vector<3>& Ivs,
                         tipl::const_pointer_image<3,unsigned char> J,
                         const tipl::vector<3>& Jvs,
                         tipl::affine_transform<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         double precision,
                         const float* bound)
{
    std::cout << "linear registration using GPU" << std::endl;
    return tipl::reg::linear_mr<tipl::reg::mutual_information_cuda>(I,Ivs,J,Jvs,T,reg_type,terminated,precision,bound);
}
