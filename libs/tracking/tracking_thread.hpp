#ifndef TRACKING_THREAD_HPP
#define TRACKING_THREAD_HPP
#include <vector>
#include <ctime>
#include <memory>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/random.hpp>
#include <boost/thread/thread.hpp>
#include "stream_line.hpp"
#include "roi.hpp"
#include "tracking_method.hpp"
#include "tracking_model.hpp"
#include "tract_model.hpp"

struct ThreadData
{
public:
    static ThreadData* new_thread(ODFModel* handle,float* param,
                                      unsigned char* methods,
                                      unsigned int termination_count)
    {
        std::auto_ptr<ThreadData> new_thread(new ThreadData(handle,termination_count));
        new_thread->param.step_size = param[0];
        new_thread->param.step_size_in_voxel[0] = param[0]/handle->fib_data.vs[0];
        new_thread->param.step_size_in_voxel[1] = param[0]/handle->fib_data.vs[1];
        new_thread->param.step_size_in_voxel[2] = param[0]/handle->fib_data.vs[2];
        new_thread->param.allowed_cos_angle = std::cos(param[1]);
        new_thread->param.cull_cos_angle = std::cos(param[2]);
        new_thread->param.threshold = param[3];
        new_thread->param.smooth_fraction = param[4];

        new_thread->param.min_points_count3 = 3*param[5]/param[0];
        if(new_thread->param.min_points_count3 < 6)
            new_thread->param.min_points_count3 = 6;
        new_thread->param.max_points_count3 = 3*param[6]/param[0];

        new_thread->param.method_id = methods[0];
        new_thread->param.seed_id = methods[1];
        new_thread->param.interpo_id = methods[2];
        new_thread->stop_by_track = methods[3];
        new_thread->center_seed = methods[4];
        return new_thread.release();
    }
public:
    RoiMgr roi_mgr;
    std::vector<image::vector<3,short> > seeds;
public:
    TrackingParam param;
    bool stop_by_track;
    bool center_seed;
    unsigned int total_seedings;
    unsigned int total_tracks;
    unsigned int termination_count;
public:
    ODFModel* handle;
public:
    boost::ptr_vector<TrackingMethod> method;
    std::auto_ptr<boost::thread_group> threads;
    unsigned int thread_count;
    bool joinning;
    boost::mutex lock_feed_function, lock_seed_function, lock_tract_function;
private:
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
public:
    ThreadData(ODFModel* handle_,unsigned int termination_count_):
            handle(handle_),
            termination_count(termination_count_),
            total_seedings(0),total_tracks(0),joinning(false),
            generator(0),uniform_rand(0,1.0),rand_gen(generator,uniform_rand){}
    ~ThreadData(void)
    {
        end_thread();
    }
private:
    boost::mt19937 generator;
    boost::uniform_real<float> uniform_rand;
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rand_gen;
public:
    void run_thread(unsigned int thread_id)
    {
        if(seeds.empty())
            return;
        try{
            std::vector<std::vector<float> > local_track_buffer;
            for (unsigned int iteration = thread_id;!joinning && !is_ended();iteration+=thread_count)
            {
                float pos[3];
                if(center_seed)
                {
                    {
                        boost::mutex::scoped_lock lock(lock_seed_function);
                        if(iteration >= seeds.size() ||
                           !stop_by_track && total_seedings >= termination_count)
                            break;
                        ++total_seedings;
                    }
                    pos[0] = seeds[iteration].x();
                    pos[1] = seeds[iteration].y();
                    pos[2] = seeds[iteration].z();
                }
                else
                {
                    // to ensure the consistency
                    {
                        boost::mutex::scoped_lock lock(lock_seed_function);
                        if(!stop_by_track && total_seedings >= termination_count)
                            break;
                        ++total_seedings;
                    }
                    unsigned int i = rand_gen()*((float)seeds.size()-1.0);
                    pos[0] = (float)seeds[i].x() + rand_gen()-0.5;
                    pos[1] = (float)seeds[i].y() + rand_gen()-0.5;
                    pos[2] = (float)seeds[i].z() + rand_gen()-0.5;
                }

                {
                    unsigned int point_count;
                    const float *result = method[thread_id].tracking(pos,point_count);
                    if (result && point_count)
                    {
                        {
                            boost::mutex::scoped_lock lock(lock_tract_function);
                            if(stop_by_track && total_tracks >= termination_count)
                                break;
                            ++total_tracks;
                        }
                        local_track_buffer.push_back(std::vector<float>(result,result+point_count+point_count+point_count));
                    }
                }
                if((iteration & 0x00000FFF) == 0x00000FFF && !local_track_buffer.empty())
                    push_tracts(local_track_buffer);
            }
            push_tracts(local_track_buffer);

        }
        catch(...)
        {

        }
    }

    bool is_ended(void)
    {
        if(center_seed && total_seedings >= seeds.size())
            return true;
        if (stop_by_track)
        {
            if (total_tracks >= termination_count)
                return true;
        }
        else
        {
            if (total_seedings >= termination_count)
                return true;
        }
        return false;

    }

    bool fetchTracks(TractModel* handle)
    {
        if (track_buffer.empty())
            return false;

        boost::mutex::scoped_lock lock(lock_feed_function);
        // handle overrun
        while(stop_by_track &&
              handle->get_visible_track_count() + track_buffer.size() > termination_count &&
              !track_buffer.empty())
            track_buffer.pop_back();
        handle->add_tracts(track_buffer);
        track_buffer.clear();
        return true;

    }
    void setRegions(const std::vector<image::vector<3,short> >& points,
                       unsigned type)
    {

            switch(type)
            {
            case 0: //ROI
                roi_mgr.add_inclusive_roi(handle->fib_data.dim,points);
                    break;
            case 1: //ROA
                roi_mgr.add_exclusive_roi(handle->fib_data.dim,points);
                    break;
            case 2: //End
                roi_mgr.add_end_roi(handle->fib_data.dim,points);
                    break;
            case 3: //seed
                for (unsigned int index = 0;index < points.size();++index)
                    seeds.push_back(points[index]);
                break;
            }

    }
    void add_new_method(void)
    {
        std::auto_ptr<basic_interpolation> interpo_method;
        switch (param.interpo_id)
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
        method.push_back(new TrackingMethod(
            new TrackingInfo(handle->fib_data,param,interpo_method.release()),roi_mgr,param));
    }

    void run(unsigned int thread_count_)
    {
        thread_count = thread_count_;
        method.clear();
        threads.reset(new boost::thread_group);
        for (unsigned int index = 0;index < thread_count;++index)
        {
            add_new_method();
            threads->add_thread(new boost::thread(&ThreadData::run_thread,this,index));
        }
    }
    void run_until_terminate(unsigned int thread_count_)
    {
        thread_count = thread_count_;
        method.clear();
        if(thread_count > 1)
            run(thread_count-1);
        add_new_method();
        run_thread(method.size()-1);
        if(threads.get())
            threads->join_all();
    }
};
#endif // TRACKING_THREAD_HPP
