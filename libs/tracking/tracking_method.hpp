#ifndef STREAM_LINE_HPP
#define STREAM_LINE_HPP
#include <ctime>
#include <random>
#include <deque>
#include <vector>
#include "basic_process.hpp"
#include "roi.hpp"
#include "fib_data.hpp"


struct TrackingParam
{
    float threshold = 0.0f;
    float default_otsu = 0.6f;
    float cull_cos_angle = 1.0f;
    float step_size = 0.0f;
    float smooth_fraction = 0.0f;
    float min_length = 30.0f;
    float max_length = 300.0f;
    unsigned int termination_count = 100000;
    unsigned int max_seed_count = 0;
    unsigned char stop_by_tract = 1;
    unsigned char reserved0 = 0; // center_seed DEPRECATED
    unsigned char check_ending = 0;
    unsigned char reserved5 = 0; // interpolation_strategy DEPRECATED

    unsigned char tracking_method = 0;
    unsigned char reserved7 = 0; // initial_direction DEPRECATED
    unsigned char reserved6 = 0; // random_seed DEPRECATED
    unsigned char tip_iteration = 0;

    float dt_threshold = 0;
    unsigned short random_seed = 0; // used in connectometry to generate different seed sequence for each permutation
    unsigned char reserved3 = 0;
    unsigned char reserved4 = 0;

    static char char2index(unsigned char c)
    {
        if(c < 10)
            return char(c)+'0';
        return char(c+'A'-10);
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
        return uint8_t(l) | uint8_t(uint8_t(h) << 4);
    }

    std::string get_code(void) const
    {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(this);
        std::string code;
        for(size_t i = 0;i < sizeof(*this);++i)
        {
            code.push_back(char2index(p[i] >> 4));
            code.push_back(char2index(p[i] & 15));
        }
        char rep[8] = {'0','a','b','c','d','e','f','g'};
        for(int i = 0;i < 7;++i)
        {
            for(size_t j = 1;j < code.size();++j)
                if(code[j-1] == code[j] && code[j] == rep[i])
                {
                    code[j-1] = rep[i+1];
                    code.erase(code.begin()+int(j));
                    --j;
                }
        }
        return code;
    }
    bool set_code(std::string code)
    {
        char rep[8] = {'0','a','b','c','d','e','f','g'};
        for(int i = 7;i > 0;--i)
        {
            for(size_t j = 0;j < code.size();++j)
                if(code[j] == rep[i])
                {
                    code[j] = rep[i-1];
                    code.insert(code.begin()+int(j),rep[i-1]);
                }
        }
        if(code.size()/2 > sizeof(*this))
            return false;
        unsigned char* p = reinterpret_cast<unsigned char*>(this);
        for(size_t i = 0;i < code.size();i += 2,++p)
            *p = index2char(code[i],code[i+1]);

        tipl::out() << "threshold:" << threshold << std::endl;
        tipl::out() << "default_otsu:" << default_otsu << std::endl;
        tipl::out() << "cull_cos_angle:" << cull_cos_angle << std::endl;
        tipl::out() << "step_size:" << step_size << std::endl;
        tipl::out() << "smooth_fraction:" << smooth_fraction << std::endl;
        tipl::out() << "min_length:" << min_length << std::endl;
        tipl::out() << "max_length:" << max_length << std::endl;
        tipl::out() << "termination_count:" << termination_count << std::endl;
        tipl::out() << "max_seed_count:" << max_seed_count << std::endl;
        tipl::out() << "stop_by_tract:" << int(stop_by_tract) << std::endl;
        tipl::out() << "reserved0(center_seed DEPRECATED):" << int(reserved0) << std::endl;
        tipl::out() << "check_ending:" << int(check_ending) << std::endl;
        tipl::out() << "reserved5(interpolation_strategy DEPRECATED):" << int(reserved5) << std::endl;
        tipl::out() << "tracking_method:" << int(tracking_method) << std::endl;
        tipl::out() << "reserved7(initial_direction DEPRECATED):" << int(reserved7) << std::endl;
        tipl::out() << "reserved6(random_seed DEPRECATED):" << int(reserved6) << std::endl;
        tipl::out() << "tip_iteration:" << int(tip_iteration) << std::endl;
        tipl::out() << "dt_threshold:" << dt_threshold << std::endl;
        tipl::out() << "random_seed:" << random_seed << std::endl;
        tipl::out() << "reserved3:" << int(reserved3) << std::endl;
        tipl::out() << "reserved4:" << int(reserved4) << std::endl;
        return true;
    }

    std::string get_report(void)
    {
        std::ostringstream report;
        if(threshold == 0.0f)
            report << " The anisotropy threshold was randomly selected.";
        else
            report << " The anisotropy threshold was " << threshold << ".";

        if(cull_cos_angle != 1.0f)
            report << " The angular threshold was " << int(std::round(std::acos(double(cull_cos_angle))*180.0/3.14159265358979323846)) << " degrees.";
        else
            report << " The angular threshold was randomly selected from 15 degrees to 90 degrees.";

        if(step_size != 0.0f)
            report << " The step size was " << step_size << " mm.";
        else
            report << " The step size was randomly selected from 0.5 voxel to 1.5 voxels.";

        if(smooth_fraction != 0.0f)
        {
            if(smooth_fraction != 1.0f)
                report << " The fiber trajectories were smoothed by averaging the propagation direction with "
                       << int(std::round(smooth_fraction * 100.0f)) << "% of the previous direction.";
            else
                report << " The fiber trajectories were smoothed by averaging the propagation direction with a percentage of the previous direction. The percentage was randomly selected from 0% to 95%.";
        }

        report << " Tracks with length shorter than " << min_length << " or longer than " << max_length  << " mm were discarded.";
        if(termination_count)
            report << " A total of " << termination_count << (stop_by_tract ? " tracts were calculated.":" seeds were placed.");
        if(tip_iteration)
            report << " Topology-informed pruning (Yeh et al. Neurotherapeutics, 16(1), 52-58, 2019) was applied to the tractography with " << int(tip_iteration) <<
                      " iteration(s) to remove false connections.";
        report << " parameter_id=" << get_code() << " ";
        return report.str();
    }

};


class TrackingMethod{
public:// Parameters
    tipl::vector<3,float> position;
    tipl::vector<3,float> dir;
    tipl::vector<3,float> next_dir;
public:
    std::shared_ptr<tracking_data> trk;
    float current_fa_threshold;
    float current_dt_threshold = 0;
    float current_tracking_angle;
    float current_tracking_smoothing;
    float current_step_size_in_voxel[3];
    unsigned int current_min_steps3;
    unsigned int current_max_steps3;
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
    unsigned char init_fib_index;
public:
    bool get_dir(const tipl::vector<3,float>& position,
                 const tipl::vector<3,float>& ref_dir,
                 tipl::vector<3,float>& result) const
    {
        return trk->get_dir_under_termination_criteria(position,ref_dir,result,
                    current_fa_threshold,current_tracking_angle,current_dt_threshold);
    }
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
    TrackingMethod(std::shared_ptr<tracking_data> trk_,
                   std::shared_ptr<RoiMgr> roi_mgr_):
                   trk(trk_),roi_mgr(roi_mgr_),init_fib_index(0)
    {}
private:
    inline bool tracking_continue(void) const
    {
        return !roi_mgr->is_terminate_point(position) &&
               get_buffer_size() < current_max_steps3;
    }
public:
    bool initialize_direction(void)
    {
        auto round_pos = position;
        round_pos.round();
        if(!trk->dim.is_valid(round_pos))
            return false;
        return get_dir(position,trk->get_fib(tipl::pixel_index<3>(round_pos[0],round_pos[1],round_pos[2],trk->dim).index(),0),dir);
    }
    template<typename tracking_algo>
    bool start_tracking(tracking_algo track)
    {
        tipl::vector<3,float> seed_pos(position);
        tipl::vector<3,float> begin_dir(dir);
        // floatd for full backward or full forward
        track_buffer.resize(current_max_steps3 << 1);
        buffer_front_pos = uint32_t(current_max_steps3);
        buffer_back_pos = uint32_t(current_max_steps3);
        tipl::vector<3,float> end_point1;

        while(tracking_continue())
        {
            if(roi_mgr->is_excluded_point(position))
				return false;
            track_buffer[buffer_back_pos] = position[0];
            track_buffer[buffer_back_pos+1] = position[1];
            track_buffer[buffer_back_pos+2] = position[2];
            buffer_back_pos += 3;
            if(!track(*this))
                break;
		}

        end_point1 = position;
        position = seed_pos;
        dir = -begin_dir;
        if(tracking_continue() && track(*this))
        {
            while(tracking_continue())
            {
                if(roi_mgr->is_excluded_point(position))
                    return false;
                buffer_front_pos -= 3;
                track_buffer[buffer_front_pos] = position[0];
                track_buffer[buffer_front_pos+1] = position[1];
                track_buffer[buffer_front_pos+2] = position[2];
                if(!track(*this))
                    break;
            }
        }



        return get_buffer_size() >= current_min_steps3 &&
               roi_mgr->have_include(get_result(),get_buffer_size()) &&
               roi_mgr->fulfill_end_point(position,end_point1);


	}
        const float* tracking(unsigned char tracking_method,unsigned int& point_count)
        {
            point_count = 0;
            switch (tracking_method)
            {
            case 0:
                if (!start_tracking(EulerTracking()))
                    return nullptr;
                break;
            case 1:
                if (!start_tracking(RungeKutta4()))
                    return nullptr;
                break;
            case 2:
                position.round();
                if (!start_tracking(VoxelTracking()))
                    return nullptr;

                // smooth trajectories
                {
                    std::vector<float> smoothed(track_buffer.size());
                    float w[5] = {1.0,2.0,4.0,2.0,1.0};
                    int dis[5] = {-6, -3, 0, 3, 6};
                    for(size_t index = buffer_front_pos;index < buffer_back_pos;++index)
                    {
                        float sum_w = 0.0;
                        float sum = 0.0;
                        for(int i = 0;i < 5;++i)
                        {
                            int cur_index = int(index) + dis[i];
                            if(cur_index < int(buffer_front_pos) || cur_index >= int(buffer_back_pos))
                                continue;
                            sum += w[i]*track_buffer[uint32_t(cur_index)];
                            sum_w += w[i];
                        }
                        if(sum_w != 0.0f)
                            smoothed[index] = sum/sum_w;
                    }
                    smoothed.swap(track_buffer);
                }

                break;
            default:
                return nullptr;
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
            reverse_buffer.resize(track_buffer.size());
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
