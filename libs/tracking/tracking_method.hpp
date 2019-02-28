#ifndef STREAM_LINE_HPP
#define STREAM_LINE_HPP
#include <ctime>
#include <random>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <deque>
#include <vector>
#include "tipl/tipl.hpp"
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
    float threshold;
    float default_otsu = 0.6f;
    float cull_cos_angle;
    float step_size;
    float smooth_fraction;
    float min_length;
    float max_length;
    unsigned int termination_count = 10000;
    unsigned int max_seed_count = 0;
    unsigned char stop_by_tract = 1;
    unsigned char center_seed = 0;
    unsigned char check_ending = 0;
    unsigned char interpolation_strategy = 0;

    unsigned char tracking_method = 0;
    unsigned char initial_direction = 0;
    unsigned char random_seed = 0;
    unsigned char tip_iteration = 0;

    float dt_threshold = 0;
    unsigned char reserved1 = 0;
    unsigned char reserved2 = 0;
    unsigned char reserved3 = 0;
    unsigned char reserved4 = 0;

    static char char2index(unsigned char c)
    {
        if(c < 10)
            return c+'0';
        return c+'A'-10;
    }
    static unsigned char index2char(char h,char l)
    {
        if(h >= '0' && h <= '9')
            h -= '0';
        else
            h -= 'A'-10;
        if(l >= '0' && l <= '9')
            l -= '0';
        else
            l -= 'A'-10;
        return (unsigned char)l | ((unsigned char)h << 4);
    }

    std::string get_code(void) const
    {
        const unsigned char* p = (const unsigned char*)this;
        std::string code;
        for(int i = 0;i < sizeof(*this);++i)
        {
            code.push_back(char2index(p[i] >> 4));
            code.push_back(char2index(p[i] & 15));
        }
        char rep[8] = {'0','a','b','c','d','e','f','g'};
        for(int i = 0;i < 7;++i)
        {
            for(int j = 1;j < code.size();++j)
                if(code[j-1] == code[j] && code[j] == rep[i])
                {
                    code[j-1] = rep[i+1];
                    code.erase(code.begin()+j);
                    --j;
                }
        }
        return code;
    }
    void set_code(std::string code)
    {
        char rep[8] = {'0','a','b','c','d','e','f','g'};
        for(int i = 7;i > 0;--i)
        {
            for(int j = 0;j < code.size();++j)
                if(code[j] == rep[i])
                {
                    code[j] = rep[i-1];
                    code.insert(code.begin()+j,rep[i-1]);
                }
        }
        unsigned char* p = (unsigned char*)this;
        for(int i = 0;i < code.size();i += 2,++p)
            *p = index2char(code[i],code[i+1]);
    }

    std::string get_report(void)
    {
        std::ostringstream report;
        if(cull_cos_angle != 1.0)
            report << " The angular threshold was " << (int)std::round(std::acos(cull_cos_angle)*180.0/3.14159265358979323846) << " degrees.";
        else
            report << " The angular threshold was randomly selected from 15 degrees to 90 degrees.";

        if(step_size != 0.0)
            report << " The step size was " << step_size << " mm.";
        else
            report << " The step size was randomly selected from 0.5 voxel to 1.5 voxels.";

        if(smooth_fraction != 0.0)
        {
            if(smooth_fraction != 1.0)
                report << " The fiber trajectories were smoothed by averaging the propagation direction with "
                       << (int)std::round(smooth_fraction * 100.0) << "% of the previous direction.";
            else
                report << " The fiber trajectories were smoothed by averaging the propagation direction with a percentage of the previous direction. The percentage was randomly selected from 0% to 95%.";
        }

        report << " Tracks with length shorter than " << min_length << " or longer than " << max_length  << " mm were discarded.";
        report << " A total of " << termination_count << (stop_by_tract ? " tracts were calculated.":" seeds were placed.");
        if(tip_iteration)
            report << " Topology-informed pruning (Yeh et al. Neurotherapeutics 2018) was applied to the tractography with " << (int)tip_iteration <<
                      " iteration(s) to remove false connections.";
        report << " parameter_id=" << get_code() << " ";
        return report.str();
    }

};


class TrackingMethod{
private:
    std::auto_ptr<basic_interpolation> interpolation;
public:// Parameters
    tipl::vector<3,float> position;
    tipl::vector<3,float> dir;
    tipl::vector<3,float> next_dir;
    bool terminated;
    bool forward;
public:
    const tracking_data& trk;
    float current_fa_threshold;
    float current_dt_threshold = 0;
    float current_tracking_angle;
    float current_tracking_smoothing;
    float current_step_size_in_voxel[3];
    int current_min_steps3;
    int current_max_steps3;
    void scaling_in_voxel(tipl::vector<3,float>& dir) const
    {
        dir[0] *= current_step_size_in_voxel[0];
        dir[1] *= current_step_size_in_voxel[1];
        dir[2] *= current_step_size_in_voxel[2];
    }
private:
    std::shared_ptr<RoiMgr> roi_mgr;
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
    bool get_dir(const tipl::vector<3,float>& position,
                      const tipl::vector<3,float>& ref_dir,
                      tipl::vector<3,float>& result_dir)
    {
        return interpolation->evaluate(trk,position,ref_dir,result_dir,current_fa_threshold,current_tracking_angle,current_dt_threshold);
    }
public:
    TrackingMethod(const tracking_data& trk_,basic_interpolation* interpolation_,
                   std::shared_ptr<RoiMgr> roi_mgr_):
        trk(trk_),interpolation(interpolation_),roi_mgr(roi_mgr_),init_fib_index(0)
	{


	}
public:
	template<class Process>
    void operator()(Process)
    {
        Process()(*this);
    }
	template<class ProcessList>
    void tracking(ProcessList)
    {
        boost::mpl::for_each<ProcessList>(boost::ref(*this));
    }
public:


	std::vector<float>& get_track_buffer(void){return track_buffer;}
	std::vector<float>& get_reverse_buffer(void){return reverse_buffer;}

	template<class ProcessList>
    bool start_tracking(bool smoothing)
    {
        tipl::vector<3,float> seed_pos(position);
        tipl::vector<3,float> begin_dir(dir);
        // floatd for full backward or full forward
        track_buffer.resize(current_max_steps3 << 1);
        reverse_buffer.resize(current_max_steps3 << 1);
        buffer_front_pos = current_max_steps3;
        buffer_back_pos = current_max_steps3;
        tipl::vector<3,float> end_point1;
        terminated = false;
		do
		{
            if(get_buffer_size() > current_max_steps3 || buffer_back_pos + 3 >= track_buffer.size())
				return false;
            if(roi_mgr->is_excluded_point(position))
				return false;
            track_buffer[buffer_back_pos] = position[0];
            track_buffer[buffer_back_pos+1] = position[1];
            track_buffer[buffer_back_pos+2] = position[2];
            buffer_back_pos += 3;
            if(roi_mgr->is_terminate_point(position))
                break;
            tracking(ProcessList());
			// make sure that the length won't overflow
			
		}
        while(!terminated);
		
        end_point1 = position;
        terminated = false;
        position = seed_pos;
        dir = -begin_dir;
        forward = false;
		do
		{
            tracking(ProcessList());
			// make sure that the length won't overflow
            if(get_buffer_size() > current_max_steps3 || buffer_front_pos < 3)
				return false;			
            if(terminated)
				break;
			buffer_front_pos -= 3;
            if(roi_mgr->is_excluded_point(position))
				return false;
            track_buffer[buffer_front_pos] = position[0];
            track_buffer[buffer_front_pos+1] = position[1];
            track_buffer[buffer_front_pos+2] = position[2];
        }
        while(!roi_mgr->is_terminate_point(position));

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

        return get_buffer_size() > current_min_steps3 &&
               roi_mgr->have_include(get_result(),get_buffer_size()) &&
               roi_mgr->fulfill_end_point(position,end_point1);


	}
        bool init(unsigned char initial_direction,
                  const tipl::vector<3,float>& position_,
                  std::mt19937& seed)
        {
            std::uniform_real_distribution<float> gen(0,1);
            position = position_;
            terminated = false;
            forward = true;
            tipl::pixel_index<3> pindex(std::round(position[0]),
                                    std::round(position[1]),
                                    std::round(position[2]),trk.dim);
            if (!trk.dim.is_valid(pindex))
                return false;

            switch (initial_direction)
            {
            case 0:// main direction
                {
                    if(trk.fa[0][pindex.index()] < current_fa_threshold)
                        return false;
                    dir = trk.get_dir(pindex.index(),0);
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
                    if (get_dir(position,tipl::vector<3,float>(x,y,z),dir))
                        return true;
                }
                return false;
            case 2:// all direction
                {
                    if (init_fib_index >= trk.fib_num ||
                        trk.fa[init_fib_index][pindex.index()] < current_fa_threshold)
                    {
                        init_fib_index = 0;
                        return false;
                    }
                    else
                        dir = trk.get_dir(pindex.index(),init_fib_index);
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
                position[0] = std::round(position[0]);
                position[1] = std::round(position[1]);
                position[2] = std::round(position[2]);
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
                tipl::vector<3,float> head(&*(track_buffer.begin() + buffer_front_pos));
                tipl::vector<3,float> tail(&*(track_buffer.begin() + buffer_back_pos-3));
		tail -= head;
                tipl::vector<3,float> abs_dis(std::abs(tail[0]),std::abs(tail[1]),std::abs(tail[2]));
		
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
