#ifndef TRACKING_INFO_HPP
#define TRACKING_INFO_HPP
#include <ctime>
#include "boost/random.hpp"
#include "interpolation_process.hpp"
#include "fib_data.hpp"


struct TrackingParam
{

    float threshold;
    float allowed_cos_angle;

    float cull_cos_angle;

    float step_size;
    float step_size_in_voxel[3];

    float smooth_fraction;

    unsigned int min_points_count3;
    unsigned int max_points_count3;

    unsigned int method_id;
    unsigned int initial_dir;
    unsigned int interpo_id;

    void scaling_in_voxel(image::vector<3,float>& dir) const
    {
        dir[0] *= step_size_in_voxel[0];
        dir[1] *= step_size_in_voxel[1];
        dir[2] *= step_size_in_voxel[2];
    }
};

struct TrackingInfo{
private:
	std::auto_ptr<basic_interpolation> interpolation;
public:
	const FibData& fib_data;
	const TrackingParam& param;
public:// Parameters
    image::vector<3,float> position;
    image::vector<3,float> dir;
    image::vector<3,float> next_dir;
	bool terminated;
	bool forward;
	bool failed;
public:
	TrackingInfo(const FibData& fib_data_,const TrackingParam& param_,
				   basic_interpolation* interpolation_):
            fib_data(fib_data_),param(param_),interpolation(interpolation_){}
        void init(const image::vector<3,float>& position_)
	{
		position = position_;
		terminated = false;
		forward = true;
	}
        bool evaluate_dir(const image::vector<3,float>& position,const image::vector<3,float>& ref_dir,image::vector<3,float>& result_dir)
	{
		return interpolation->evaluate(*this,position,ref_dir,result_dir);
	}
};

#endif//TRACKING_INFO_HPP
