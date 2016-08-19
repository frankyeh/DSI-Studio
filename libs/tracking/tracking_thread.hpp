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
    std::vector<image::vector<3,short> > seeds;
public:
    std::ostringstream report;
    std::string seed_report;
    TrackingParam param;
    bool stop_by_tract;
    bool center_seed;
    bool check_ending;
    unsigned int termination_count;
    unsigned char interpolation_strategy;
    unsigned char tracking_method;
    unsigned char initial_direction;
    unsigned int max_seed_count;
public:
    ThreadData(bool random_seed):
        joinning(false),
        seed(random_seed ? std::random_device()():0),
        stop_by_tract(true),
        center_seed(false),
        check_ending(true),
        termination_count(1000),
        interpolation_strategy(0),//trilinear_interpolation
        tracking_method(0),//streamline
        initial_direction(0),// main direction
        max_seed_count(0)
    {}
    ~ThreadData(void)
    {
        end_thread();
    }
public:
    std::vector<std::shared_ptr<std::future<void> > > threads;
    std::vector<unsigned int> seed_count;
    std::vector<unsigned int> tract_count;
    std::vector<unsigned char> running;
    bool joinning,pushing_data;
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
    void setRegions(image::geometry<3> dim,
                    const std::vector<image::vector<3,short> >& points,
                    unsigned char type,
                    const char* roi_name);
    TrackingMethod* new_method(const tracking_data& trk);
    void run(const tracking_data& trk,
             unsigned int thread_count,
             unsigned int termination_count,
             bool wait = false);



};
#endif // TRACKING_THREAD_HPP
