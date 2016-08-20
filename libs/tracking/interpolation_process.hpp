#ifndef INTERPOLATION_PROCESS_HPP
#define INTERPOLATION_PROCESS_HPP
#include <cstdlib>
#include "image/image.hpp"
class tracking_data;
class basic_interpolation
{
public:
    virtual bool evaluate(const tracking_data& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle) = 0;
};

class trilinear_interpolation_with_gaussian_basis : public basic_interpolation
{
public:
    virtual bool evaluate(const tracking_data& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle);
};


class trilinear_interpolation : public basic_interpolation
{
public:
    virtual bool evaluate(const tracking_data& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle);
};


class nearest_direction : public basic_interpolation
{
public:
    virtual bool evaluate(const tracking_data& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle);
};



#endif//INTERPOLATION_PROCESS_HPP
