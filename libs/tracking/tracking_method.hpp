#ifndef STREAM_LINE_HPP
#define STREAM_LINE_HPP
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/utility.hpp>
#include <deque>
#include <vector>
#include "image/image.hpp"
#include "interpolation_process.hpp"
#include "tracking_info.hpp"
#include "stream_line.hpp"


class TrackingMethod{
private:
        const TrackingParam& param;
	const RoiMgr& roi_mgr;
	std::vector<float> track_buffer;
	mutable std::vector<float> reverse_buffer;
    unsigned int buffer_front_pos;
    unsigned int buffer_back_pos;
public:
        unsigned int get_buffer_size(void) const
	{
		return buffer_back_pos-buffer_front_pos;
	}
        unsigned int get_point_count(void) const
	{
		return (buffer_back_pos-buffer_front_pos)/3;
	}
public:
	std::auto_ptr<TrackingInfo> info;
public:
        TrackingMethod(TrackingInfo* info_,
                       const RoiMgr& roi_mgr_,
                       const TrackingParam& param_):
                       info(info_),roi_mgr(roi_mgr_),param(param_)
	{
		// floatd for full backward or full forward
		track_buffer.resize(info->param.max_points_count3 << 1);
		reverse_buffer.resize(info->param.max_points_count3 << 1);
	}
public:
	template<typename Process>
    void operator()(Process)
    {
        Process()(*info.get());
    }
	template<typename Process>
    bool set_seeding()
    {
        boost::mpl::for_each<Process>(boost::ref(*this));
		return !info->terminated;
    }
	template<typename ProcessList>
    void tracking(ProcessList)
    {
        boost::mpl::for_each<ProcessList>(boost::ref(*this));
    }
public:


	std::vector<float>& get_track_buffer(void){return track_buffer;}
	std::vector<float>& get_reverse_buffer(void){return reverse_buffer;}

	template<typename ProcessList>
        bool start_tracking(const image::vector<3,float>& seed_pos)
	{
		buffer_front_pos = info->param.max_points_count3;
		buffer_back_pos = info->param.max_points_count3;
                image::vector<3,float> begin_dir = info->dir;
                image::vector<3,float> end_point1;
		info->failed = false;
		do
		{
			if(get_buffer_size() > info->param.max_points_count3 || buffer_back_pos + 3 >= track_buffer.size())
				return false;
			if(roi_mgr.is_excluded_point(info->position))
				return false;
			std::copy(info->position.begin(),info->position.end(),track_buffer.begin()+buffer_back_pos);
			buffer_back_pos += 3;
			tracking(ProcessList());
			// make sure that the length won't overflow
			
		}
		while(!info->terminated);
                if(info->failed)
			return false;
		
                end_point1 = info->position;
                info->terminated = false;
		info->position = seed_pos;
		info->dir = -begin_dir;
		info->forward = false;
		do
		{
		    tracking(ProcessList());	
			// make sure that the length won't overflow
			if(get_buffer_size() > info->param.max_points_count3 || buffer_front_pos < 3)
				return false;			
			if(info->terminated)
				break;
			buffer_front_pos -= 3;
			if(roi_mgr.is_excluded_point(info->position))
				return false;
			std::copy(info->position.begin(),info->position.end(),track_buffer.begin()+buffer_front_pos);
		}
		while(1);

                return !info->failed &&
                       get_buffer_size() > 0 &&
                       get_buffer_size() >= info->param.min_points_count3 &&
                       roi_mgr.have_include(get_result(),get_buffer_size()) &&
                       roi_mgr.fulfill_end_point(info->position,end_point1);
	}

        const float* tracking(float* position_,unsigned int& point_count)
        {
            point_count = 0;
            image::vector<3,float> position(position_);
            info->init(position);
            switch (param.seed_id)
            {
            case 0:
                if (!set_seeding<main_fiber_direction_seeding>())
                    return 0;
                break;
            case 1:
                if (!set_seeding<random_direction_seeding>())
                    return 0;
            }
            switch (param.method_id)
            {
            case 0:
                if (!start_tracking<streamline_method_process>(position))
                    return 0;
                break;
            case 1:
                if (!start_tracking<streamline_runge_kutta_4_method_process>(position))
                    return 0;
                break;
            case 2:
                if (!start_tracking<streamline_method_process_with_relocation>(position))
                    return 0;
                break;

                /*
                case 2:
                    //method[thread_id].start_tracking<second_order_streamline_method_process>();
                    cur_method.info->dummy1 = (fib_data.voxel_size[0]+fib_data.voxel_size[1]+fib_data.voxel_size[2])/6.0;
                    cur_method.info->dummy2 = cur_method.info->dummy1/10.0;
                    if (!cur_method.start_tracking<streamline_adpative_method_process>(position))
                        return 0;

                    break;
                */
                    default:
                            return 0;
            }
            point_count = get_point_count();
            return get_result();
        }

	const float* get_result(void) const
	{
                image::vector<3,float> head(&*(track_buffer.begin() + buffer_front_pos));
                image::vector<3,float> tail(&*(track_buffer.begin() + buffer_back_pos-3));
		tail -= head;
                image::vector<3,float> abs_dis(std::abs(tail[0]),std::abs(tail[1]),std::abs(tail[2]));
		
		if((abs_dis[0] > abs_dis[1] && abs_dis[0] > abs_dis[2] && tail[0] < 0) ||
		   (abs_dis[1] > abs_dis[0] && abs_dis[1] > abs_dis[2] && tail[1] < 0) ||
		   (abs_dis[2] > abs_dis[1] && abs_dis[2] > abs_dis[0] && tail[2] < 0))
		{
			std::vector<float>::const_iterator src = track_buffer.begin() + buffer_back_pos-3;
			std::vector<float>::iterator iter = reverse_buffer.begin();
			std::vector<float>::iterator end = reverse_buffer.begin()+buffer_back_pos-buffer_front_pos;
			for(;iter < end;iter += 3,src -= 3)
				std::copy(src,src+3,iter);
			return &*reverse_buffer.begin();
		}

		return &*(track_buffer.begin() + buffer_front_pos);
	}
};






#endif//STREAM_LINE_HPP
