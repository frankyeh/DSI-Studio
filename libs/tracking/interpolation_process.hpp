#ifndef INTERPOLATION_PROCESS_HPP
#define INTERPOLATION_PROCESS_HPP
#include <cstdlib>
#include "image/image.hpp"
class fiber_orientations;
class basic_interpolation
{
public:
    virtual bool evaluate(const fiber_orientations& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result) = 0;
};

class trilinear_interpolation_with_gaussian_basis : public basic_interpolation
{
public:
    virtual bool evaluate(const fiber_orientations& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result);
};


class trilinear_interpolation : public basic_interpolation
{
public:
    virtual bool evaluate(const fiber_orientations& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result);
};


class nearest_direction : public basic_interpolation
{
public:
    virtual bool evaluate(const fiber_orientations& fib,
                          const image::vector<3,float>& position,
                          const image::vector<3,float>& ref_dir,
                          image::vector<3,float>& result);
};



#endif//INTERPOLATION_PROCESS_HPP
