#ifndef TRACKING_THREAD_HPP
#define TRACKING_THREAD_HPP
#include <vector>
#include <ctime>
#include <random>
#include <memory>

#include "roi.hpp"
#include "tracking_method.hpp"
#include "fib_data.hpp"
#include "tract_model.hpp"

struct ThreadData
{
private:
    std::mt19937 seed;

public:
    std::shared_ptr<tracking_data> trk;
    std::shared_ptr<RoiMgr> roi_mgr;
public:
    std::ostringstream report;
    TrackingParam param;
    float fa_threshold1,fa_threshold2;// use only if fa_threshold=0

public:
    ThreadData(std::shared_ptr<fib_data> handle):seed(0),roi_mgr(new RoiMgr(handle)){}
    ~ThreadData(void)
    {
        end_thread();
    }
public:
    bool joinning = false;
    bool pushing_data = false;
    std::vector<std::shared_ptr<std::future<void> > > threads;
    std::vector<unsigned int> seed_count;
    std::vector<unsigned int> tract_count;
    std::vector<unsigned int> end_count;
    std::vector<unsigned char> running;
    std::mutex  lock_feed_function,lock_seed_function;
    unsigned int get_total_seed_count(void)const
    {
        if(seed_count.empty())
            return 0;
        return std::accumulate(seed_count.begin(),seed_count.end(),0);
    }
    unsigned int get_total_tract_count(void)const
    {
        if(tract_count.empty())
            return 0;
        return std::accumulate(tract_count.begin(),tract_count.end(),0);
    }
    bool is_ended(void)
    {
        if(running.empty())
            return true;
        return std::find(running.begin(),running.end(),1) == running.end();
    }

public:
    std::vector<std::vector<float> > track_buffer;
    void push_tracts(std::vector<std::vector<float> >& local_tract_buffer);
    void end_thread(void);

public:
    void run_thread(unsigned int thread_count,unsigned int thread_id);
    bool fetchTracks(TractModel* handle);
    void apply_tip(TractModel* handle);
    void run(std::shared_ptr<tracking_data> trk,unsigned int thread_count,bool wait);
    void run(unsigned int thread_count,bool wait);



};
#endif // TRACKING_THREAD_HPP
