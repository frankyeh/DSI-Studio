#ifndef TRACKING_THREAD_HPP
#define TRACKING_THREAD_HPP
#include <vector>
#include <ctime>
#include <memory>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/random.hpp>
#include <boost/thread/thread.hpp>
#include "roi.hpp"
#include "tracking_method.hpp"
#include "tracking_model.hpp"
#include "tract_model.hpp"


struct ThreadData
{

public:
    RoiMgr roi_mgr;
    std::vector<image::vector<3,short> > seeds;
public:
    TrackingParam param;
    bool stop_by_tract;
    bool center_seed;
    unsigned int termination_count;
    unsigned char interpolation_strategy;
    unsigned char tracking_method;
    unsigned char initial_direction;
    unsigned int max_seed_count;
public:
    ThreadData(void):
        joinning(false),generator(0),uniform_rand(0,1.0),rand_gen(generator,uniform_rand),
        stop_by_tract(true),
        center_seed(false),
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
    std::auto_ptr<boost::thread_group> threads;
    std::vector<unsigned int> seed_count;
    std::vector<unsigned int> tract_count;
    std::vector<unsigned char> running;
    bool joinning;
    boost::mutex lock_feed_function,lock_seed_function;
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
    void push_tracts(std::vector<std::vector<float> >& local_tract_buffer)
    {
        boost::mutex::scoped_lock lock(lock_feed_function);
        for(unsigned int index = 0;index < local_tract_buffer.size();++index)
        {
            track_buffer.push_back(std::vector<float>());
            track_buffer.back().swap(local_tract_buffer[index]);
        }
        local_tract_buffer.clear();
    }
    void end_thread(void)
    {
        if (threads.get())
        {
            joinning = true;
            threads->join_all();
            threads.reset(0);
        }
    }

private:
    boost::mt19937 generator;
    boost::uniform_real<float> uniform_rand;
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rand_gen;
public:
    void run_thread(TrackingMethod* method_ptr,unsigned int thread_count,unsigned int thread_id,unsigned int max_count)
    {
        std::auto_ptr<TrackingMethod> method(method_ptr);
        unsigned int iteration = thread_id; // for center seed
        if(!seeds.empty())
        try{
            std::vector<std::vector<float> > local_track_buffer;
            while(!joinning &&
                  (!stop_by_tract || tract_count[thread_id] < max_count) &&
                  (stop_by_tract || seed_count[thread_id] < max_count) &&
                  (max_seed_count == 0 || seed_count[thread_id] < max_seed_count) &&
                  (!center_seed || iteration < seeds.size()))
            {
                ++seed_count[thread_id];
                if(center_seed)
                {
                    if(!method->init(initial_direction,
                                     image::vector<3,float>(seeds[iteration].x(),seeds[iteration].y(),seeds[iteration].z()),
                                     rand_gen))
                    {
                        iteration+=thread_count;
                        continue;
                    }
                    if(initial_direction == 0)// primary direction
                        iteration+=thread_count;
                }
                else
                {
                    // this ensure consistency
                    boost::mutex::scoped_lock lock(lock_seed_function);
                    iteration+=thread_count;
                    unsigned int i = rand_gen()*((float)seeds.size()-1.0);
                    image::vector<3,float> pos;
                    pos[0] = (float)seeds[i].x() + rand_gen()-0.5;
                    pos[1] = (float)seeds[i].y() + rand_gen()-0.5;
                    pos[2] = (float)seeds[i].z() + rand_gen()-0.5;
                    if(!method->init(initial_direction,pos,rand_gen))
                        continue;
                }
                unsigned int point_count;
                const float *result = method->tracking(tracking_method,point_count);
                if (result && point_count)
                {    
                    ++tract_count[thread_id];
                    local_track_buffer.push_back(std::vector<float>(result,result+point_count+point_count+point_count));
                }
                if((iteration & 0x00000FFF) == 0x00000FFF && !local_track_buffer.empty())
                    push_tracts(local_track_buffer);
            }
            push_tracts(local_track_buffer);
        }
        catch(...)
        {

        }
        running[thread_id] = 0;
    }

    bool fetchTracks(TractModel* handle)
    {
        if (track_buffer.empty())
            return false;

        boost::mutex::scoped_lock lock(lock_feed_function);
        handle->add_tracts(track_buffer);
        track_buffer.clear();
        return true;

    }
    void setRegions(image::geometry<3> dim,
                    const std::vector<image::vector<3,short> >& points,unsigned char type)
    {
        switch(type)
        {
        case 0: //ROI
            roi_mgr.add_inclusive_roi(dim,points);
                break;
        case 1: //ROA
            roi_mgr.add_exclusive_roi(dim,points);
                break;
        case 2: //End
            roi_mgr.add_end_roi(dim,points);
                break;
        case 4: //Terminate
            roi_mgr.add_terminate_roi(dim,points);
                break;
        case 3: //seed
            for (unsigned int index = 0;index < points.size();++index)
                seeds.push_back(points[index]);
            break;

        }
    }
    TrackingMethod* new_method(const fiber_orientations& fib)
    {
        std::auto_ptr<basic_interpolation> interpo_method;
        switch (interpolation_strategy)
        {
        case 0:
            interpo_method.reset(new trilinear_interpolation);
            break;
        case 1:
            interpo_method.reset(new trilinear_interpolation_with_gaussian_basis);
            break;
        case 2:
            interpo_method.reset(new nearest_direction);
            break;

        }
        return new TrackingMethod(fib,interpo_method.release(),roi_mgr,param);
    }

    void run(const fiber_orientations& fib,unsigned int thread_count,unsigned int termination_count,bool wait = false)
    {
        // to ensure consistency, seed initialization with all orientation only fits with single thread
        if(initial_direction == 2)
            thread_count = 1;
        param.step_size_in_voxel[0] = param.step_size/fib.vs[0];
        param.step_size_in_voxel[1] = param.step_size/fib.vs[1];
        param.step_size_in_voxel[2] = param.step_size/fib.vs[2];

        if(center_seed)
        {
            std::srand(0);
            std::random_shuffle(seeds.begin(),seeds.end());
        }
        seed_count.clear();
        tract_count.clear();
        seed_count.resize(thread_count);
        tract_count.resize(thread_count);
        running.resize(thread_count);
        std::fill(running.begin(),running.end(),1);
        threads.reset(new boost::thread_group);
        unsigned int run_count = termination_count/thread_count+1;
        unsigned int total_run_count = 0;
        for (unsigned int index = 0;index < thread_count-1;++index,total_run_count += run_count)
            threads->add_thread(new boost::thread(&ThreadData::run_thread,this,new_method(fib),thread_count,index,run_count));

        if(wait)
        {
            run_thread(new_method(fib),thread_count,thread_count-1,termination_count-total_run_count);
            threads->join_all();
        }
        else
            threads->add_thread(new boost::thread(&ThreadData::run_thread,this,new_method(fib),thread_count,thread_count-1,termination_count-total_run_count));

    }
};
#endif // TRACKING_THREAD_HPP
