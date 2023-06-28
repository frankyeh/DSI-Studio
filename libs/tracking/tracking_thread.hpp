#ifndef TRACKING_THREAD_HPP
#define TRACKING_THREAD_HPP
#include <vector>
#include <ctime>
#include <random>
#include <memory>

#include "roi.hpp"
#include "tracking_method.hpp"
#include "tract_model.hpp"

#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif
struct ThreadData
{
private:
    std::mt19937 seed;
    std::uniform_real_distribution<float> rand_gen,subvoxel_gen,angle_gen,smoothing_gen,step_gen,step_gen2,threshold_gen;
public:
    std::shared_ptr<tracking_data> trk;
    std::shared_ptr<RoiMgr> roi_mgr;
public:
    std::ostringstream report;
    TrackingParam param;
    float fa_threshold1,fa_threshold2;// use only if fa_threshold=0
    bool ready_to_track = false;
public:
    ThreadData(std::shared_ptr<fib_data> handle):seed(0),
        rand_gen(0,1),subvoxel_gen(-0.5f,0.5f),angle_gen(float(15.0*M_PI/180.0),float(90.0*M_PI/180.0)),
        smoothing_gen(0.0f,0.95f),step_gen(0.5f,1.5f),step_gen2(1.0f,3.0f),threshold_gen(0.0,1.0),
        roi_mgr(new RoiMgr(handle)){}
    ~ThreadData(void)
    {
        end_thread();
    }
public:
    bool joining = false;
    std::vector<std::thread> threads;
    std::vector<unsigned int> seed_count,tract_count;
    std::vector<unsigned char> running;
    std::mutex lock_seed_function;
    std::chrono::high_resolution_clock::time_point begin_time,end_time;
    unsigned int get_total_seed_count(void)const
    {
        return seed_count.empty() ? 0 : std::accumulate(seed_count.begin(),seed_count.end(),uint32_t(0));
    }
    unsigned int get_total_tract_count(void)const
    {
        return tract_count.empty() ? 0 : std::accumulate(tract_count.begin(),tract_count.end(),uint32_t(0));
    }
    bool is_ended(void)
    {
        return running.empty() ? true : std::find(running.begin(),running.end(),1) == running.end();
    }
public:
    bool buffer_switch = true;
    std::vector<std::vector<std::vector<float> > > track_buffer_back,track_buffer_front;
    void end_thread(void);

public:
    void run_thread(unsigned int thread_id,unsigned int thread_count);
    bool fetchTracks(TractModel* handle);
    void apply_tip(TractModel* handle);
    void run(std::shared_ptr<tracking_data> trk,unsigned int thread_count,bool wait);
    void run(unsigned int thread_count,bool wait);



};
#endif // TRACKING_THREAD_HPP
