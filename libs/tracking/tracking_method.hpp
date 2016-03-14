#ifndef STREAM_LINE_HPP
#define STREAM_LINE_HPP
#include <ctime>
#include <random>
#include <boost/random.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/utility.hpp>
#include <deque>
#include <vector>
#include "image/image.hpp"
#include "interpolation_process.hpp"
#include "basic_process.hpp"
#include "roi.hpp"
#include "fib_data.hpp"

typedef boost::mpl::vector<
            EstimateNextDirection,
            SmoothDir,
            MoveTrack
> streamline_method_process;

typedef boost::mpl::vector<
            LocateVoxel
> voxel_tracking;


typedef boost::mpl::vector<
            EstimateNextDirectionRungeKutta4,
            MoveTrack
> streamline_runge_kutta_4_method_process;



struct TrackingParam
{
    float step_size;
    float step_size_in_voxel[3];

    float smooth_fraction;

    unsigned int min_points_count3;
    unsigned int max_points_count3;
    void scaling_in_voxel(image::vector<3,float>& dir) const
    {
        dir[0] *= step_size_in_voxel[0];
        dir[1] *= step_size_in_voxel[1];
        dir[2] *= step_size_in_voxel[2];
    }
};


class TrackingMethod{
private:
    std::auto_ptr<basic_interpolation> interpolation;
public:// Parameters
    image::vector<3,float> position;
    image::vector<3,float> dir;
    image::vector<3,float> next_dir;
    bool terminated;
    bool forward;
    bool failed;
public:
    const fiber_orientations& fib;
    const TrackingParam& param;
private:
    const RoiMgr& roi_mgr;
	std::vector<float> track_buffer;
	mutable std::vector<float> reverse_buffer;
    unsigned int buffer_front_pos;
    unsigned int buffer_back_pos;
private:
    unsigned int init_fib_index;
public:
    unsigned int get_buffer_size(void) const
	{
		return buffer_back_pos-buffer_front_pos;
	}
    unsigned int get_point_count(void) const
	{
		return (buffer_back_pos-buffer_front_pos)/3;
	}
    bool get_dir(const image::vector<3,float>& position,
                      const image::vector<3,float>& ref_dir,
                      image::vector<3,float>& result_dir)
    {
        return interpolation->evaluate(fib,position,ref_dir,result_dir);
    }
public:
    TrackingMethod(const fiber_orientations& fib_,basic_interpolation* interpolation_,
                   const RoiMgr& roi_mgr_,const TrackingParam& param_):
        fib(fib_),interpolation(interpolation_),roi_mgr(roi_mgr_),param(param_),init_fib_index(0)
	{
        // floatd for full backward or full forward
        track_buffer.resize(param.max_points_count3 << 1);
        reverse_buffer.resize(param.max_points_count3 << 1);
	}
public:
	template<typename Process>
    void operator()(Process)
    {
        Process()(*this);
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
    bool start_tracking(bool smoothing)
    {
        image::vector<3,float> seed_pos(position);
        image::vector<3,float> begin_dir(dir);
        buffer_front_pos = param.max_points_count3;
        buffer_back_pos = param.max_points_count3;
        image::vector<3,float> end_point1;
        failed = false;
        terminated = false;
		do
		{
            if(get_buffer_size() > param.max_points_count3 || buffer_back_pos + 3 >= track_buffer.size())
				return false;
            if(roi_mgr.is_excluded_point(position))
				return false;
            track_buffer[buffer_back_pos] = position[0];
            track_buffer[buffer_back_pos+1] = position[1];
            track_buffer[buffer_back_pos+2] = position[2];
            buffer_back_pos += 3;
            if(roi_mgr.is_terminate_point(position))
                break;
            tracking(ProcessList());
			// make sure that the length won't overflow
			
		}
        while(!terminated);

        if(failed)
			return false;
		
        end_point1 = position;
        terminated = false;
        position = seed_pos;
        dir = -begin_dir;
        forward = false;
		do
		{
		    tracking(ProcessList());	
			// make sure that the length won't overflow
            if(get_buffer_size() > param.max_points_count3 || buffer_front_pos < 3)
				return false;			
            if(terminated)
				break;
			buffer_front_pos -= 3;
            if(roi_mgr.is_excluded_point(position))
				return false;
            track_buffer[buffer_front_pos] = position[0];
            track_buffer[buffer_front_pos+1] = position[1];
            track_buffer[buffer_front_pos+2] = position[2];
        }
        while(!roi_mgr.is_terminate_point(position));

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

        return !failed &&
               get_buffer_size() >= param.min_points_count3 &&
               roi_mgr.have_include(get_result(),get_buffer_size()) &&
               roi_mgr.fulfill_end_point(position,end_point1);


	}
        bool init(unsigned char initial_direction,
                  const image::vector<3,float>& position_,
                  std::mt19937& seed)
        {
            std::uniform_real_distribution<float> gen(0,1);
            position = position_;
            terminated = false;
            forward = true;
            image::pixel_index<3> index(std::floor(position[0]+0.5),
                                    std::floor(position[1]+0.5),
                                    std::floor(position[2]+0.5),fib.dim);
            if (!fib.dim.is_valid(index))
                return false;

            switch (initial_direction)
            {
            case 0:// main direction
                {
                    if(fib.fa[0][index.index()] < fib.threshold)
                        return false;
                    dir = fib.get_dir(index.index(),0);
                }
                return true;
            case 1:// random direction
                for (unsigned int index = 0;index < 10;++index)
                {
                    float txy = gen(seed);
                    float tz = gen(seed)/2.0;
                    float x = std::sin(txy)*std::sin(tz);
                    float y = std::cos(txy)*std::sin(tz);
                    float z = std::cos(tz);
                    if (get_dir(position,image::vector<3,float>(x,y,z),dir))
                        return true;
                }
                return false;
            case 2:// all direction
                {
                    if (init_fib_index >= fib.fib_num ||
                        fib.fa[init_fib_index][index.index()] < fib.threshold)
                    {
                        init_fib_index = 0;
                        return false;
                    }
                    else
                        dir = fib.get_dir(index.index(),init_fib_index);
                    ++init_fib_index;
                }
                return true;
            }
            return false;
        }

        const float* tracking(unsigned char tracking_method,unsigned int& point_count)
        {
            point_count = 0;
            switch (tracking_method)
            {
            case 0:
                if (!start_tracking<streamline_method_process>(false))
                    return 0;
                break;
            case 1:
                if (!start_tracking<streamline_runge_kutta_4_method_process>(false))
                    return 0;
                break;
            case 2:
                position[0] = std::floor(position[0]+0.5);
                position[1] = std::floor(position[1]+0.5);
                position[2] = std::floor(position[2]+0.5);
                if (!start_tracking<voxel_tracking>(true))
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
