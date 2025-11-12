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
    unsigned int max_tract_count = 0;
    unsigned int max_seed_count = 0;
    float track_voxel_ratio = 2.0f;

    unsigned char tracking_method = 0;
    unsigned char check_ending = 0; // initial_direction DEPRECATED
    unsigned char reserved6 = 0; // random_seed DEPRECATED
    unsigned char tip_iteration = 0;

    float dt_threshold = 0.0f;
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
        print();
        return true;
    }
    void print(void)
    {
        tipl::out() << "threshold: " << threshold << std::endl;
        tipl::out() << "default_otsu: " << default_otsu << std::endl;
        tipl::out() << "cull_cos_angle: " << cull_cos_angle << std::endl;
        tipl::out() << "step_size: " << step_size << std::endl;
        tipl::out() << "smooth_fraction: " << smooth_fraction << std::endl;
        tipl::out() << "min_length: " << min_length << std::endl;
        tipl::out() << "max_length: " << max_length << std::endl;
        tipl::out() << "max_tract_count: " << max_tract_count << std::endl;
        tipl::out() << "max_seed_count: " << max_seed_count << std::endl;
        tipl::out() << "track_voxel_ratio: " << track_voxel_ratio << std::endl;
        tipl::out() << "tracking_method: " << int(tracking_method) << std::endl;
        tipl::out() << "check_ending: " << int(check_ending) << std::endl;
        tipl::out() << "reserved6(random_seed DEPRECATED): " << int(reserved6) << std::endl;
        tipl::out() << "tip_iteration: " << int(tip_iteration) << std::endl;
        tipl::out() << "dt_threshold: " << dt_threshold << std::endl;
        tipl::out() << "random_seed: " << random_seed << std::endl;
        tipl::out() << "reserved3: " << int(reserved3) << std::endl;
        tipl::out() << "reserved4: " << int(reserved4) << std::endl;
    }

    std::string get_report(void)
    {
        std::ostringstream report;
        if(threshold == 0.0f)
            report << " The anisotropy threshold was randomly selected between " << std::fixed << std::setprecision(1) <<
                        default_otsu-0.1 << " and " << default_otsu+0.1 << " otsu threshold.";
        else
            report << " The anisotropy threshold was " << threshold << ".";

        if(cull_cos_angle != 1.0f)
            report << " The angular threshold was " << int(std::round(std::acos(double(cull_cos_angle))*180.0/3.14159265358979323846)) << " degrees.";
        else
            report << " The angular threshold was randomly selected from 45 degrees to 90 degrees.";

        if(step_size > 0.0f)
            report << " The step size was " << std::fixed << std::setprecision(2) << step_size << " mm.";
        else
        {
            if(step_size < 0.0f) // older versions before june 2023
                report << " The step size was randomly selected from 0.5 voxel to 1.5 voxels.";
            else
                report << " The step size was set to voxel spacing.";
        }

        if(smooth_fraction != 0.0f)
        {
            if(smooth_fraction != 1.0f)
                report << " The fiber trajectories were smoothed by averaging the propagation direction with "
                       << int(std::round(smooth_fraction * 100.0f)) << "% of the previous direction.";
            else
                report << " The fiber trajectories were smoothed by averaging the propagation direction with a percentage of the previous direction. The percentage was randomly selected from 0% to 95%.";
        }

        report << " Tracks with length shorter than " << min_length << " or longer than " << max_length  << " mm were discarded.";

        if(max_tract_count || max_seed_count)
        {
            if(max_seed_count)
                report << " A total of " << max_seed_count << " seeds were placed.";
            if(max_tract_count)
                report << " A total of " << max_tract_count << " tracts were tracked.";
        }
        else
            report << " The tract-to-voxel ratio was set to " << track_voxel_ratio << ".";

        if(tip_iteration)
            report << " Topology-informed pruning (Yeh et al. Neurotherapeutics, 16(1), 52-58, 2019) was applied to the tractography with " << int(tip_iteration) <<
                      " iteration(s) to remove false connections.";
        report << " parameter_id=" << get_code() << " ";
        return report.str();
    }

};


class TrackingMethod{
public:
    tipl::vector<3> position,dir,next_dir;

public:
    std::shared_ptr<tracking_data> trk;
    float current_fa_threshold;
    float current_dt_threshold = 0;
    float current_tracking_angle;
    float current_tracking_smoothing = 0.0f;
    float current_step_size_in_voxel[3] = {1.0f,1.0f,1.0f};
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
    float* buffer_front_pos;
    float* buffer_back_pos;
    unsigned int total_steps3;
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
    TrackingMethod(std::shared_ptr<tracking_data> trk_,
                   std::shared_ptr<RoiMgr> roi_mgr_):
                   trk(trk_),roi_mgr(roi_mgr_),init_fib_index(0)
    {}
private:
    inline bool tracking_continue(void) const
    {
        return !roi_mgr->within_terminative(position) &&
               total_steps3 < current_max_steps3;
    }
public:
    bool initialize_direction(unsigned char fib_order = 0)
    {
        auto round_pos = position;
        round_pos.round();
        if(!trk->dim.is_valid(round_pos))
            return false;
        return get_dir(position,trk->get_fib(tipl::pixel_index<3>(round_pos[0],round_pos[1],round_pos[2],trk->dim).index(),fib_order),dir);
    }
    void init_buffer(void)
    {
        track_buffer.resize(current_max_steps3 << 1);
        reverse_buffer.resize(current_max_steps3);
    }
    template<typename tracking_algo>
    const float* start_tracking(tracking_algo&& track)
    {
        tipl::vector<3> seed_pos(position),begin_dir(dir),end_point1;
        buffer_back_pos = buffer_front_pos = track_buffer.data() + current_max_steps3;
        total_steps3 = 0;
        next_dir = dir;
        while(tracking_continue())
        {
            if(roi_mgr->within_roa(position) ||
              !roi_mgr->within_limiting(position))
                return nullptr;
            *(buffer_back_pos++) = position[0];
            *(buffer_back_pos++) = position[1];
            *(buffer_back_pos++) = position[2];
            total_steps3 += 3;
            if(!track(*this))
                break;
		}

        end_point1 = position;
        position = seed_pos;
        next_dir = dir = -begin_dir;
        if(tracking_continue() && track(*this))
        {
            while(tracking_continue())
            {
                if(roi_mgr->within_roa(position) ||
                  !roi_mgr->within_limiting(position))
                    return nullptr;
                *(--buffer_front_pos) = position[2];
                *(--buffer_front_pos) = position[1];
                *(--buffer_front_pos) = position[0];
                total_steps3 += 3;
                if(!track(*this))
                    break;
            }
        }

        {
            buffer_back_pos-=3;
            tipl::vector<3> tail(buffer_back_pos);
            tail -= tipl::vector<3>(buffer_front_pos);
            float x_ = std::abs(tail[0]),y_ = std::abs(tail[0]), z_ = std::abs(tail[0]);
            size_t max_dim = (y_ > x_) ? ((z_ > y_) ? 2 : 1) : ((z_ > x_) ? 2 : 0);
            if(tail[max_dim] < 0.0f)
                for(auto dst = buffer_front_pos = reverse_buffer.data(),end_dst = dst + total_steps3;
                    dst < end_dst;dst += 3,buffer_back_pos -= 3)
                {
                    dst[0] = buffer_back_pos[0];
                    dst[1] = buffer_back_pos[1];
                    dst[2] = buffer_back_pos[2];
                }
        }
        if(total_steps3 >= current_min_steps3 &&
           roi_mgr->within_roi(buffer_front_pos,total_steps3) &&
           roi_mgr->fulfill_end_point(position,end_point1))
            return buffer_front_pos;
        return nullptr;
	}
    const float* tracking(unsigned char tracking_method,unsigned int& total_steps)
    {
        const float* track_ptr = nullptr;
        switch (tracking_method)
        {
        case 0:
            track_ptr = start_tracking(EulerTracking());
            break;
        case 1:
            track_ptr = start_tracking(RungeKutta4());
            break;
        default:
            return nullptr;
        }
        if(track_ptr)
        {
            total_steps = total_steps3/3;
            return track_ptr;
        }
        total_steps = 0;
        return nullptr;
    }


};






#endif//STREAM_LINE_HPP
