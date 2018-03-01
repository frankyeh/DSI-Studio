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
    RoiMgr roi_mgr;

public:
    std::ostringstream report;
    TrackingParam param;
    bool stop_by_tract = true;
    bool center_seed = false;
    bool check_ending = false;
    unsigned int termination_count = 1000;
    unsigned char interpolation_strategy = 0;//trilinear_interpolation
    unsigned char tracking_method = 0; // streamline
    unsigned char initial_direction = 0; // primary
    unsigned int max_seed_count = 0;
public:
    ThreadData(bool random_seed):
        joinning(false),
        seed(random_seed ? std::random_device()():0){}
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
    void run_thread(TrackingMethod* method_ptr,unsigned int thread_count,unsigned int thread_id,unsigned int max_count);
    bool fetchTracks(TractModel* handle);
    TrackingMethod* new_method(const tracking_data& trk);
    void run(const tracking_data& trk,
             unsigned int thread_count,
             unsigned int termination_count,
             bool wait = false);



};
#endif // TRACKING_THREAD_HPP
