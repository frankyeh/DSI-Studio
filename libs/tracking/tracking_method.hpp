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
#include "basic_process.hpp"
#include "roi.hpp"

typedef boost::mpl::vector<
            EstimateNextDirection,
            SmoothDir,
            MoveTrack
> streamline_method_process;

typedef boost::mpl::vector<
            //EstimateNextDirection,
            //SmoothDir,
            //MoveTrack2
            LocateVoxel
> voxel_tracking;


typedef boost::mpl::vector<
            EstimateNextDirectionRungeKutta4,
            MoveTrack
> streamline_runge_kutta_4_method_process;



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
    TrackingInfo info;
public:
    TrackingMethod(const FibData& fib_data_,basic_interpolation* interpolation_,
                   const RoiMgr& roi_mgr_,const TrackingParam& param_):
                       info(fib_data_,param_,interpolation_),roi_mgr(roi_mgr_),param(param_)
	{
		// floatd for full backward or full forward
        track_buffer.resize(info.param.max_points_count3 << 1);
        reverse_buffer.resize(info.param.max_points_count3 << 1);
	}
public:
	template<typename Process>
    void operator()(Process)
    {
        Process()(info);
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
        bool start_tracking(const image::vector<3,float>& seed_pos,bool smoothing)
	{
        buffer_front_pos = info.param.max_points_count3;
        buffer_back_pos = info.param.max_points_count3;
                image::vector<3,float> begin_dir = info.dir;
                image::vector<3,float> end_point1;
        info.failed = false;
		do
		{
            if(get_buffer_size() > info.param.max_points_count3 || buffer_back_pos + 3 >= track_buffer.size())
				return false;
            if(roi_mgr.is_excluded_point(info.position))
				return false;
            std::copy(info.position.begin(),info.position.end(),track_buffer.begin()+buffer_back_pos);
			buffer_back_pos += 3;
			tracking(ProcessList());
			// make sure that the length won't overflow
			
		}
        while(!info.terminated);
                if(info.failed)
			return false;
		
                end_point1 = info.position;
                info.terminated = false;
        info.position = seed_pos;
        info.dir = -begin_dir;
        info.forward = false;
		do
		{
		    tracking(ProcessList());	
			// make sure that the length won't overflow
            if(get_buffer_size() > info.param.max_points_count3 || buffer_front_pos < 3)
				return false;			
            if(info.terminated)
				break;
			buffer_front_pos -= 3;
            if(roi_mgr.is_excluded_point(info.position))
				return false;
            std::copy(info.position.begin(),info.position.end(),track_buffer.begin()+buffer_front_pos);
		}
		while(1);

        if(smoothing)
        {
            std::vector<float> smoothed(track_buffer.size());
            float w[5] = {1.0,2.0,4.0,2.0,1.0};
            int dis[5] = {-6, -3, 0, 3, 6};
            for(int index = buffer_front_pos;index < buffer_back_pos;++index)
            {
                float sum_w = 0.0;
                float sum = 0.0;
                for(char i = 0;i < 5;++i)
                {
                    int cur_index = index + dis[i];
                    if(cur_index < buffer_front_pos || cur_index >= buffer_back_pos)
                        continue;
                    sum += w[i]*track_buffer[cur_index];
                    sum_w += w[i];
                }
                if(sum_w != 0.0)
                    smoothed[index] = sum/sum_w;
            }
            smoothed.swap(track_buffer);
        }


                return !info.failed &&
                       get_buffer_size() >= info.param.min_points_count3 &&
                       roi_mgr.have_include(get_result(),get_buffer_size()) &&
                       roi_mgr.fulfill_end_point(info.position,end_point1);
	}

        const float* tracking(float* position_,unsigned int& point_count)
        {
            point_count = 0;
            image::vector<3,float> position(position_);
            info.init(position);
            switch (param.seed_id)
            {
            case 0:// main direction
                {
                    image::pixel_index<3> index(std::floor(info.position[0]+0.5),
                                            std::floor(info.position[1]+0.5),
                                            std::floor(info.position[2]+0.5),info.fib_data.dim);

                    if (!info.fib_data.dim.is_valid(index) ||
                         info.fib_data.fib.getFA(index.index(),0) < info.param.threshold)
                        info.terminated = true;
                    else
                        info.dir = info.fib_data.fib.getDir(index.index(),0);
                }
                break;
            case 1:// random direction
                info.terminated = true;
                for (unsigned int index = 0;index < 10;++index)
                {
                    float txy = info.gen();
                    float tz = info.gen()/2.0;
                    float x = std::sin(txy)*std::sin(tz);
                    float y = std::cos(txy)*std::sin(tz);
                    float z = std::cos(tz);
                    if (info.evaluate_dir(info.position,image::vector<3,float>(x,y,z),info.dir))
                    {
                        info.terminated = false;
                        break;
                    }
                }
                break;
            case 2:// all direction
                {
                    static unsigned int fib_index = 0;
                    image::pixel_index<3> index(std::floor(info.position[0]+0.5),
                                        std::floor(info.position[1]+0.5),
                                        std::floor(info.position[2]+0.5),info.fib_data.dim);

                    if (!info.fib_data.dim.is_valid(index) ||
                         info.fib_data.fib.getFA(index.index(),fib_index) < info.param.threshold)
                        info.terminated = true;
                    else
                        info.dir = info.fib_data.fib.getDir(index.index(),fib_index);
                    ++fib_index;
                    if(fib_index > info.fib_data.fib.num_fiber)
                        fib_index = 0;
                }
                break;
            }

            if(info.terminated)
                return 0;

            switch (param.method_id)
            {
            case 0:
                if (!start_tracking<streamline_method_process>(position,false))
                    return 0;
                break;
            case 1:
                if (!start_tracking<streamline_runge_kutta_4_method_process>(position,false))
                    return 0;
                break;
            case 2:
                info.position[0] = std::floor(info.position[0]+0.5);
                info.position[1] = std::floor(info.position[1]+0.5);
                info.position[2] = std::floor(info.position[2]+0.5);
                if (!start_tracking<voxel_tracking>(position,true))
                    return 0;
                break;
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
