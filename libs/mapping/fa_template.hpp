#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP
#include <boost/thread.hpp>
#include "image/image.hpp"
struct fa_template{
    std::vector<float> tran;
    image::basic_image<float,3> I;
    bool load_from_file(const char* file_name);
    void to_mni(image::vector<3,float>& p);
    void add_transformation(std::vector<float>& t);
    fa_template(void):tran(16){}
};


template<typename bfnorm_type,typename terminated_class>
void multi_thread_reg(bfnorm_type& mapping,
                      const image::basic_image<float,3>& VG,
                      const image::basic_image<float,3>& VFF,unsigned int thread_count,terminated_class& terminated)
{
    image::reg::bfnorm_mrqcof<image::basic_image<float,3>,float> bf_optimize(VG,VFF,mapping,thread_count);
    // image::reg::bfnorm(VG,VFF,*mni.get(),terminated);
    for(int iter = 0; iter < 16 && !terminated; ++iter)
    {

        bf_optimize.start();
        boost::thread_group threads;
        bool terminated_buf = false;
        for (unsigned int index = 1;index < thread_count;++index)
                threads.add_thread(new boost::thread(
                                   &image::reg::bfnorm_mrqcof<image::basic_image<float,3>,float>::run<bool>,
                                   &bf_optimize,index,boost::ref(terminated_buf)));
        bf_optimize.run(0,terminated_buf);
        if(thread_count > 1)
            threads.join_all();
        bf_optimize.end();
        boost::thread_group threads2;
        for (unsigned int index = 1;index < thread_count;++index)
                threads2.add_thread(new boost::thread(
                                    &image::reg::bfnorm_mrqcof<image::basic_image<float,3>,float>::run2,
                                    &bf_optimize,index,40));
        bf_optimize.run2(0,40);
        if(thread_count > 1)
            threads2.join_all();
    }
}

#endif // FA_TEMPLATE_HPP
