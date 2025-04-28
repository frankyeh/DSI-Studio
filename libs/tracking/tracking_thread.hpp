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
    std::uniform_real_distribution<float> rand_gen,subvoxel_gen,angle_gen,angle_gen2,smoothing_gen,step_gen,step_gen2,threshold_gen;
    template<typename T>
    T rand(T size)
    {
        return std::min<T>(size-1,T(rand_gen(seed)*float(size)));
    }
public:
    std::shared_ptr<tracking_data> trk;
    std::shared_ptr<RoiMgr> roi_mgr;
public:
    unsigned int max_tract_count = 0;
    unsigned int max_seed_count = 0;
public:
    std::ostringstream report;
    TrackingParam param;
    float fa_threshold1,fa_threshold2;// use only if fa_threshold=0
    bool ready_to_track = false;
public:
    ThreadData(std::shared_ptr<fib_data> handle):seed(0),
        rand_gen(0.0f,1.0f),subvoxel_gen(-0.5f,0.5f),
        angle_gen(float(45.0f*M_PI/180.0f),float(90.0f*M_PI/180.0f)),
        smoothing_gen(0.0f,0.95f),step_gen(0.5f,1.5f),threshold_gen(0.0f,1.0f),
        roi_mgr(new RoiMgr(handle)){}
    ~ThreadData(void)
    {
        end_thread();
    }
public:
    bool joining = false;
    std::vector<std::thread> threads;
    std::vector<size_t> seed_count,tract_count;
    std::vector<unsigned char> running;
    std::mutex lock_seed_function;
    std::chrono::high_resolution_clock::time_point begin_time,end_time;
    size_t get_total_seed_count(void)const
    {
        return seed_count.empty() ? 0 : std::accumulate(seed_count.begin(),seed_count.end(),size_t(0));
    }
    size_t get_total_tract_count(void)const
    {
        return tract_count.empty() ? 0 : std::accumulate(tract_count.begin(),tract_count.end(),size_t(0));
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
    void run(std::shared_ptr<tracking_data> trk,unsigned int thread_count,bool wait);
    void run(unsigned int thread_count,bool wait);



};
#endif // TRACKING_THREAD_HPP
