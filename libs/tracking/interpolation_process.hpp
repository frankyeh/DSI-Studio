#ifndef INTERPOLATION_PROCESS_HPP
#define INTERPOLATION_PROCESS_HPP
#include <cstdlib>
#include "tipl/tipl.hpp"
class tracking_data;
class basic_interpolation
{
public:
    virtual ~basic_interpolation(){}
    virtual bool evaluate(std::shared_ptr<tracking_data> fib,
                          const tipl::vector<3,float>& position,
                          const tipl::vector<3,float>& ref_dir,
                          tipl::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle,
                          float dt_threshold) = 0;
};

class trilinear_interpolation : public basic_interpolation
{
public:
    virtual bool evaluate(std::shared_ptr<tracking_data> fib,
                          const tipl::vector<3,float>& position,
                          const tipl::vector<3,float>& ref_dir,
                          tipl::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle,
                          float dt_threshold);
};


class nearest_direction : public basic_interpolation
{
public:
    virtual bool evaluate(std::shared_ptr<tracking_data> fib,
                          const tipl::vector<3,float>& position,
                          const tipl::vector<3,float>& ref_dir,
                          tipl::vector<3,float>& result,
                          float threshold,
                          float cull_cos_angle,
                          float dt_threshold);
};



#endif//INTERPOLATION_PROCESS_HPP
