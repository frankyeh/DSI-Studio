
#include "tracking_thread.hpp"
#include "fib_data.hpp"

// generated from x^2+y^2+z^2 < 6 . first 40 at x > 0
char fib_dx[80] = {0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,0,0,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-2,0,0,0,0,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2};
char fib_dy[80] = {1,0,0,1,1,1,0,0,-1,1,1,-1,-1,2,0,0,2,2,1,1,2,0,0,-2,1,0,0,-1,2,2,1,1,-1,-1,-2,-2,1,1,-1,-1,-1,0,0,-1,-1,-1,0,0,1,-1,-1,1,1,-2,0,0,-2,-2,-1,-1,-2,0,0,2,-1,0,0,1,-2,-2,-1,-1,1,1,2,2,-1,-1,1,1};
char fib_dz[80] = {0,1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,2,0,1,-1,2,-2,0,2,-2,0,0,1,-1,0,1,-1,2,-2,2,-2,1,-1,1,-1,1,-1,0,-1,0,-1,1,0,-1,1,0,-1,1,-1,1,0,-2,0,-1,1,-2,2,0,-2,2,0,0,-1,1,0,-1,1,-2,2,-2,2,-1,1,-1,1,-1,1};



void ThreadData::push_tracts(std::vector<std::vector<float> >& local_tract_buffer)
{
    std::lock_guard<std::mutex> lock(lock_feed_function);
    pushing_data = true;
    for(unsigned int index = 0;index < local_tract_buffer.size();++index)
        track_buffer.push_back(std::move(local_tract_buffer[index]));
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

void ThreadData::run_thread(unsigned int thread_id)
{
    std::shared_ptr<TrackingMethod> method(new TrackingMethod(trk,roi_mgr));
    method->current_fa_threshold = param.threshold;
    method->current_dt_threshold = param.dt_threshold;
    method->current_tracking_angle = param.cull_cos_angle;
    method->current_tracking_smoothing = param.smooth_fraction;
    method->current_step_size_in_voxel[0] = param.step_size/method->trk->vs[0];
    method->current_step_size_in_voxel[1] = param.step_size/method->trk->vs[1];
    method->current_step_size_in_voxel[2] = param.step_size/method->trk->vs[2];
    if(param.step_size != 0.0f)
    {
        method->current_max_steps3 = 3*uint32_t(std::round(param.max_length/param.step_size));
        method->current_min_steps3 = 3*uint32_t(std::round(param.min_length/param.step_size));
    }
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

            ++seed_count[thread_id];
            tipl::vector<3,float> pos;
            uint32_t seed_id;
            {
                // this ensure consistency
                std::lock_guard<std::mutex> lock(lock_seed_function);
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
                    float step_size_in_voxel = step_gen(seed);
                    float step_size_in_mm = step_size_in_voxel*method->trk->vs[0];
                    method->current_step_size_in_voxel[0] = step_size_in_voxel;
                    method->current_step_size_in_voxel[1] = step_size_in_voxel;
                    method->current_step_size_in_voxel[2] = step_size_in_voxel;
                    method->current_max_steps3 = 3*uint32_t(std::round(param.max_length/step_size_in_mm));
                    method->current_min_steps3 = 3*uint32_t(std::round(param.min_length/step_size_in_mm));
                }

                seed_id = std::min<uint32_t>(uint32_t(roi_mgr->seeds.size()-1),uint32_t(rand_gen(seed)*float(roi_mgr->seeds.size())));
                pos = roi_mgr->seeds[seed_id];
                pos[0] += subvoxel_gen(seed);
                pos[1] += subvoxel_gen(seed);
                pos[2] += subvoxel_gen(seed);
            }

            if(roi_mgr->seeds_r[seed_id] != 1.0f)
                pos /= roi_mgr->seeds_r[seed_id];
            if(!method->init(param.initial_direction,pos,seed))
                continue;

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
                    if(method->trk->is_white_matter(p0,white_matter_t))
                        continue;
                }
                tipl::vector<3> p2(end-6),p3(end-3);
                if(*(end-1) > 0) // not the bottom slice
                {
                    p2 -= p3;
                    p3 -= p2;
                    if(method->trk->is_white_matter(p3,white_matter_t))
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
    if(t_index/float(roi_mgr->seeds.size()) > 20.0f || !trk->dt_threshold_name.empty())
        for(size_t i = 0;i < param.tip_iteration;++i)
            handle->trim();
}

void ThreadData::run(unsigned int thread_count,
                     bool wait)
{
    std::shared_ptr<tracking_data> trk_(new tracking_data);
    trk_->read(roi_mgr->handle);
    run(trk_,thread_count,wait);
}

void ThreadData::run(std::shared_ptr<tracking_data> trk_,unsigned int thread_count,bool wait)
{
    trk = trk_;
    if(!param.termination_count)
        return;
    if(param.threshold == 0.0f)
    {
        float otsu = tipl::segmentation::otsu_threshold(tipl::make_image(trk->fa[0],trk->dim));
        fa_threshold1 = (param.default_otsu-0.1f)*otsu;
        fa_threshold2 = (param.default_otsu+0.1f)*otsu;
    }
    else
        fa_threshold1 = fa_threshold2 = 0.0;

    report.clear();
    report.str("");
    if(!trk->dt_threshold_name.empty())
    {
        report << " Differential tractography (Yeh et al., 2019) was applied to map pathways with ";
        if(trk->dt_threshold_name.substr(0,4) == "inc_" || trk->dt_threshold_name.substr(0,4) == "dec_")
        {
            if(trk->dt_threshold_name.substr(0,4) == "inc_")
                report << "an increase";
            else
                report << "a decrease";
            report << " in " << trk->dt_threshold_name.substr(4,std::string::npos);

        }
        else
        {
            auto pos = trk->dt_threshold_name.find('-');
            if(pos == std::string::npos)
                report << "changes in " << trk->dt_threshold_name;
            else
                report << trk->dt_threshold_name.substr(0,pos) << " larger than " << trk->dt_threshold_name.substr(pos+1);
        }
        report << ", and only the differences greater than " << int(param.dt_threshold * 100) << "% were tracked.";
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
    seed = std::mt19937(0);
    for (unsigned int index = 0;index < thread_count-1;++index)
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [&,thread_count,index](){run_thread(index);})));

    if(wait)
    {
        run_thread(thread_count-1);
        for(size_t i = 0;i < threads.size();++i)
            threads[i]->wait();
    }
    else
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [&,thread_count](){run_thread(thread_count-1);})));
}
