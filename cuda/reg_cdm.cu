#include <vector>
#include "TIPL/reg/cdm.hpp"
#include "TIPL/cu.hpp"
bool distribute_gpu(void);

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
        std::cout << "❌️" << er.what() << std::endl;
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
