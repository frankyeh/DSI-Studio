#ifndef TRACKING_UTILITY_HPP
#define TRACKING_UTILITY_HPP
#include <cmath>
#include <cstdlib>
#include "math/vtor.hpp"

namespace tracking_utility{

void generate_random_vector(vtor<double,3>& result)
{
	double txy = ((double)std::rand())*M_2_PI/((double)RAND_MAX);
	double tz = ((double)std::rand())*M_PI_2/((double)RAND_MAX);

	double x = std::sin(txy)*std::sin(tz);
	double y = std::cos(txy)*std::sin(tz);
	double z = std::cos(tz);

	result = vtor<double,3>(x,y,z);
	result.normalize();
}
// cur_dir,offset must be normalized
void next_direction(const vtor<double,3>& cur_dir,const vtor<double,3>& offset,vtor<double,3>& result)
{
	result = offset;
	result *= 2.0*(offset*cur_dir);
	result -= cur_dir;
}


}






#endif//TRACKING_UTILITY_HPP