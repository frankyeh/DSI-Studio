#ifndef linear_hpp
#define linear_hpp
#include <boost/lambda/lambda.hpp>
#include <boost/thread/thread.hpp>
#include "image/image.hpp"

template<typename ImageType,typename TransformType>
class LinearMapping
{
    static const int dim = ImageType::dimension;

public:
    ImageType from,to;
    std::auto_ptr<boost::thread> thread;
    TransformType arg_min;
    unsigned char cost_function_id;
    bool terminated;
    bool ended;
public:
    mutable image::transformation_matrix<dim,float> result;
public:

    LinearMapping(void):terminated(false),ended(false) {}
    ~LinearMapping(void)
    {
        terminate();
    }

    void terminate(void)
    {
        terminated = true;
        if(thread.get())
        {
            thread->joinable();
            thread->join();
        }
        ended = true;
    }

    void argmin(int reg_type)
    {
        terminated = false;
        ended = false;
        //opti_method.minimize(*cost_function.get(),arg_min);
        switch (cost_function_id)
        {
        case 0:
        case 1:
            image::reg::linear_seq(from,to,arg_min,reg_type,image::reg::square_error(),terminated);
            break;
        case 2:
            image::reg::linear_seq(from,to,arg_min,reg_type,image::reg::mutual_information(),terminated);
            break;

        }
    }

    const float* get(void) const
    {
        image::transformation_matrix<dim,float> T(arg_min);
        image::reg::linear_get_trans(from.geometry(),to.geometry(),T);
        result = T;
        return result.get();
    }
    void thread_argmin(int reg_type)
    {
        thread.reset(new boost::thread(&LinearMapping::argmin,this,reg_type));
    }
};

// dimenion = 2
typedef  LinearMapping<image::basic_image<float,2,image::const_pointer_memory<float> >,image::affine_transform<2> > lm2_type;
// dimenion = 3
typedef  LinearMapping<image::basic_image<float,3,image::const_pointer_memory<float> >,image::affine_transform<3> > lm3_type;

#endif//linear_hpp
