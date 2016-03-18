#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP

#include "image/image.hpp"
struct fa_template{
    std::string template_file_name;
    image::vector<3> vs,shift;
    image::basic_image<float,3> I;
    float tran[16];
    bool load_from_file(void);
    void to_mni(image::vector<3,float>& p);
};


template<typename bfnorm_type,typename terminated_class>
void multi_thread_reg(bfnorm_type& mapping,
                      const image::basic_image<float,3>& VG,
                      const image::basic_image<float,3>& VFF,unsigned int thread_count,unsigned int& iteration,terminated_class& terminated)
{
    image::reg::bfnorm_mrqcof<image::basic_image<float,3>,float> bf_optimize(VG,VFF,mapping,thread_count);
    // image::reg::bfnorm(VG,VFF,*mni.get(),terminated);
    for(iteration = 0; iteration < 16 && !terminated; ++iteration)
    {

        bf_optimize.start();
        std::vector<std::shared_ptr<std::future<void> > > threads;
        bool terminated_buf = false;
        for (unsigned int index = 1;index < thread_count;++index)
            threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                    [&bf_optimize,index,&terminated_buf](){bf_optimize.run(index,terminated_buf);})));

        bf_optimize.run(0,terminated_buf);
        for(int i = 0;i < threads.size();++i)
            threads[i]->wait();

        bf_optimize.end();
        std::vector<std::shared_ptr<std::future<void> > > threads2;
        for (unsigned int index = 1;index < thread_count;++index)
                threads2.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                                   [&bf_optimize,index](){bf_optimize.run2(index,40);})));
        bf_optimize.run2(0,40);
        for(int i = 0;i < threads2.size();++i)
            threads2[i]->wait();
    }
}

#endif // FA_TEMPLATE_HPP
