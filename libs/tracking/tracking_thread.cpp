#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif
#include "tracking_thread.hpp"
#include "fib_data.hpp"
void ThreadData::push_tracts(std::vector<std::vector<float> >& local_tract_buffer)
{
    std::lock_guard<std::mutex> lock(lock_feed_function);
    pushing_data = true;
    for(unsigned int index = 0;index < local_tract_buffer.size();++index)
    {
        track_buffer.push_back(std::vector<float>());
        track_buffer.back().swap(local_tract_buffer[index]);
    }
    local_tract_buffer.clear();
    pushing_data = false;
}
void ThreadData::end_thread(void)
{
    if (!threads.empty())
    {
        joinning = true;
        for(size_t i = 0;i < threads.size();++i)
            threads[i]->wait();
        threads.clear();
    }
}

void ThreadData::run_thread(TrackingMethod* method_ptr,
                            unsigned int thread_count,
                            unsigned int thread_id)
{
    std::auto_ptr<TrackingMethod> method(method_ptr);
    std::uniform_real_distribution<float> rand_gen(0,1),
            angle_gen(float(15.0*M_PI/180.0),float(90.0*M_PI/180.0)),
            smoothing_gen(0.0f,0.95f),
            step_gen(method->trk.vs[0]*0.5f,method->trk.vs[0]*1.5f),
            threshold_gen(0.0,1.0);
    unsigned int iteration = thread_id; // for center seed
    float white_matter_t = param.threshold*1.2f;
    if(!roi_mgr->seeds.empty())
    try{
        std::vector<std::vector<float> > local_track_buffer;
        while(!joinning &&
              !(param.stop_by_tract == 1 && tract_count[thread_id] >= end_count[thread_id]) &&
              !(param.stop_by_tract == 0 && seed_count[thread_id] >= end_count[thread_id]) &&
              !(param.max_seed_count > 0 && seed_count[thread_id] >= param.max_seed_count))
        {
            if(!local_track_buffer.empty() && !pushing_data)
                push_tracts(local_track_buffer);
            if(param.threshold == 0.0f)
            {
                float w = threshold_gen(seed);
                method->current_fa_threshold = w*fa_threshold1 + (1.0f-w)*fa_threshold2;
                white_matter_t = method->current_fa_threshold*1.2f;
            }
            if(param.cull_cos_angle == 1.0f)
                method->current_tracking_angle = std::cos(angle_gen(seed));
            if(param.smooth_fraction == 1.0f)
                method->current_tracking_smoothing = smoothing_gen(seed);
            if(param.step_size == 0.0f)
            {
                float step_size_in_mm = step_gen(seed);
                method->current_step_size_in_voxel[0] = step_size_in_mm/method->trk.vs[0];
                method->current_step_size_in_voxel[1] = step_size_in_mm/method->trk.vs[1];
                method->current_step_size_in_voxel[2] = step_size_in_mm/method->trk.vs[2];
                method->current_max_steps3 = uint32_t(std::round(3.0f*param.max_length/step_size_in_mm));
                method->current_min_steps3 = uint32_t(std::round(3.0f*param.min_length/step_size_in_mm));
            }
            ++seed_count[thread_id];
            {
                // this ensure consistency
                std::lock_guard<std::mutex> lock(lock_seed_function);
                iteration+=thread_count;
                unsigned int i = uint32_t(rand_gen(seed)*(float(roi_mgr->seeds.size())-1.0f));
                tipl::vector<3,float> pos(roi_mgr->seeds[i]);
                if(!param.center_seed)
                {
                    pos[0] += rand_gen(seed);
                    pos[1] += rand_gen(seed);
                    pos[2] += rand_gen(seed);
                    pos -= 0.5f;
                }
                if(roi_mgr->seeds_r[i] != 1.0f)
                    pos /= roi_mgr->seeds_r[i];
                if(!method->init(param.initial_direction,pos,seed))
                    continue;
            }
            unsigned int point_count;
            const float *result = method->tracking(param.tracking_method,point_count);
            if(!result)
                continue;
            const float* end = result+point_count+point_count+point_count;
            if(param.check_ending)
            {
                if(point_count < 2)
                    continue;
                if(result[2] > 0) // not the bottom slice
                {
                    tipl::vector<3> p0(result),p1(result+3);
                    p1 -= p0;
                    p0 -= p1;
                    if(method->trk.is_white_matter(p0,white_matter_t))
                        continue;
                }
                tipl::vector<3> p2(end-6),p3(end-3);
                if(*(end-1) > 0) // not the bottom slice
                {
                    p2 -= p3;
                    p3 -= p2;
                    if(method->trk.is_white_matter(p3,white_matter_t))
                        continue;
                }
            }

            ++tract_count[thread_id];
            local_track_buffer.push_back(std::vector<float>(result,end));
        }
        push_tracts(local_track_buffer);
    }
    catch(...)
    {

    }
    running[thread_id] = 0;
}

bool ThreadData::fetchTracks(TractModel* handle)
{
    if (track_buffer.empty())
        return false;
    if(handle->parameter_id.empty())
        handle->parameter_id = param.get_code();
    std::lock_guard<std::mutex> lock(lock_feed_function);
    handle->add_tracts(track_buffer);
    track_buffer.clear();
    return true;

}

void ThreadData::apply_tip(TractModel* handle)
{
    if (param.tip_iteration == 0 || handle->get_visible_track_count() == 0)
        return;
    float max_length = 0.0f;
    for(size_t i = 0;i < 20 && i < handle->get_tracts().size();++i)
        max_length = std::max(max_length,float(handle->get_tracts()[i].size()));
    float t_index = float(handle->get_visible_track_count())*max_length/3.0f;
    if(t_index/float(roi_mgr->seeds.size()) > 20.0f || !handle->get_fib().dt_threshold_name.empty())
        for(size_t i = 0;i < param.tip_iteration;++i)
            handle->trim();
}
TrackingMethod* ThreadData::new_method(const tracking_data& trk)
{

    std::auto_ptr<basic_interpolation> interpo_method;
    switch (param.interpolation_strategy)
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
    TrackingMethod* method = new TrackingMethod(trk,interpo_method.release(),roi_mgr);
    method->current_fa_threshold = param.threshold;
    method->current_dt_threshold = param.dt_threshold;
    method->current_tracking_angle = param.cull_cos_angle;
    method->current_tracking_smoothing = param.smooth_fraction;
    method->current_step_size_in_voxel[0] = param.step_size/method->trk.vs[0];
    method->current_step_size_in_voxel[1] = param.step_size/method->trk.vs[1];
    method->current_step_size_in_voxel[2] = param.step_size/method->trk.vs[2];
    if(param.step_size != 0.0)
    {
        method->current_max_steps3 = std::round(3.0*param.max_length/param.step_size);
        method->current_min_steps3 = std::round(3.0*param.min_length/param.step_size);
    }
    return method;
}

void ThreadData::run(const tracking_data& trk,
                     unsigned int thread_count,
                     bool wait)
{
    if(!param.termination_count)
        return;
    if(param.threshold == 0.0f)
    {
        float otsu = tipl::segmentation::otsu_threshold(tipl::make_image(trk.fa[0],trk.dim));
        fa_threshold1 = (param.default_otsu-0.1f)*otsu;
        fa_threshold2 = (param.default_otsu+0.1f)*otsu;
    }
    else
        fa_threshold1 = fa_threshold2 = 0.0;

    report.clear();
    report.str("");
    if(!trk.dt_threshold_name.empty())
    {
        report << " Differential tractography (Yeh et al., 2019) was applied to map pathways with ";
        if(trk.dt_threshold_name.substr(0,4) == "inc_")
            report << "an increase";
        else
            report << "a decrease";
        report << " in " << trk.dt_threshold_name.substr(4,std::string::npos) << ".";
    }
    else {
        report << " A deterministic fiber tracking algorithm (Yeh et al., PLoS ONE 8(11): e80713, 2013) was used";
        if(param.threshold == 0.0f && param.cull_cos_angle == 1.0f && param.step_size == 0.0f) // parameter saturation, pruning
            report << " with augmented tracking strategies (Yeh, Neuroimage, 2020) to improve reproducibility.";
        else
            report << ".";
    }
    report << roi_mgr->report;
    report << param.get_report();
    // to ensure consistency, seed initialization with all orientation only fits with single thread
    if(param.initial_direction == 2)
        thread_count = 1;
    if(param.center_seed)
    {
        std::srand(0);
        std::random_shuffle(roi_mgr->seeds.begin(),roi_mgr->seeds.end());
    }

    end_thread();


    if(thread_count > param.termination_count)
        thread_count = param.termination_count;
    if(thread_count < 1)
        thread_count = 1;

    // initialize multi-thread for tracking
    {
        seed_count.clear();
        tract_count.clear();
        end_count.clear();

        seed_count.resize(thread_count);
        tract_count.resize(thread_count);
        end_count.resize(thread_count);
        running.resize(thread_count);

        std::fill(running.begin(),running.end(),1);

        std::fill(end_count.begin(),end_count.end(),param.termination_count/thread_count);
        end_count.back() = param.termination_count-end_count.front()*(thread_count-1);
    }


    joinning = false;
    pushing_data = false;
    seed = std::mt19937(param.random_seed ? std::random_device()():0);
    for (unsigned int index = 0;index < thread_count-1;++index)
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [&,thread_count,index](){run_thread(new_method(trk),thread_count,index);})));

    if(wait)
    {
        run_thread(new_method(trk),thread_count,thread_count-1);
        for(size_t i = 0;i < threads.size();++i)
            threads[i]->wait();
    }
    else
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [&,thread_count](){run_thread(new_method(trk),thread_count,thread_count-1);})));
}
