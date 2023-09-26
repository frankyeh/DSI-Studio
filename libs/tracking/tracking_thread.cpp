
#include "tracking_thread.hpp"
#include "fib_data.hpp"

// generated from x^2+y^2+z^2 < 6 . first 40 at x > 0
char fib_dx[80] = {0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,0,0,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-2,0,0,0,0,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2};
char fib_dy[80] = {1,0,0,1,1,1,0,0,-1,1,1,-1,-1,2,0,0,2,2,1,1,2,0,0,-2,1,0,0,-1,2,2,1,1,-1,-1,-2,-2,1,1,-1,-1,-1,0,0,-1,-1,-1,0,0,1,-1,-1,1,1,-2,0,0,-2,-2,-1,-1,-2,0,0,2,-1,0,0,1,-2,-2,-1,-1,1,1,2,2,-1,-1,1,1};
char fib_dz[80] = {0,1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,2,0,1,-1,2,-2,0,2,-2,0,0,1,-1,0,1,-1,2,-2,2,-2,1,-1,1,-1,1,-1,0,-1,0,-1,1,0,-1,1,0,-1,1,-1,1,0,-2,0,-1,1,-2,2,0,-2,2,0,0,-1,1,0,-1,1,-2,2,-2,2,-1,1,-1,1,-1,1};



void ThreadData::end_thread(void)
{
    if (!threads.empty())
    {
        joining = true;
        for(auto& thread : threads)
            if(thread.joinable())
                thread.join();
        threads.clear();
    }
}

void ThreadData::run_thread(unsigned int thread_id,unsigned int thread_count)
{
    while(!ready_to_track)
    {
        if(thread_id)
        {
            std::this_thread::yield();
            continue;
        }
        // configure seedings
        {
            seed = std::mt19937(param.random_seed);  // always 0, except in connectometry for changing seed sequence
            if(roi_mgr->use_auto_track)
            {
                if(!roi_mgr->setAtlas(joining))
                    joining = true;
            }
            if(roi_mgr->seeds.empty())
                roi_mgr->setWholeBrainSeed(fa_threshold1);

            if(param.termination_count == 0)
            {
                param.termination_count = std::max<uint32_t>(1,roi_mgr->track_voxel_ratio*roi_mgr->seeds.size());
                param.max_seed_count = param.termination_count*5000; //yield rate easy:1/100 hard:1/5000
            }
        }
        ready_to_track = true;
    }
    std::shared_ptr<TrackingMethod> method(new TrackingMethod(trk,roi_mgr));
    method->current_fa_threshold = param.threshold;
    method->current_dt_threshold = param.dt_threshold;
    method->current_tracking_angle = param.cull_cos_angle;
    method->current_tracking_smoothing = param.smooth_fraction;
    method->current_step_size_in_voxel[0] = param.step_size/method->trk->vs[0];
    method->current_step_size_in_voxel[1] = param.step_size/method->trk->vs[1];
    method->current_step_size_in_voxel[2] = param.step_size/method->trk->vs[2];
    method->check_ending = param.check_ending;
    if(param.step_size > 0.0f)
    {
        method->current_max_steps3 = 3*uint32_t(std::round(param.max_length/param.step_size));
        method->current_min_steps3 = 3*uint32_t(std::round(param.min_length/param.step_size));
    }
    unsigned int termination_count = (thread_id == 0 ?
        param.termination_count-(param.termination_count/thread_count)*(thread_count-1):
        param.termination_count/thread_count);
    unsigned int max_seed_per_thread = param.max_seed_count/thread_count;
    if(!roi_mgr->seeds.empty())
    try{
        while(!joining &&
              !(param.stop_by_tract == 1 && tract_count[thread_id] >= termination_count) &&
              !(param.stop_by_tract == 0 && seed_count[thread_id] >= termination_count) &&
              !(param.max_seed_count > 0 && seed_count[thread_id] >= max_seed_per_thread))
        {
            ++seed_count[thread_id];
            {
                // this ensure consistency
                std::lock_guard<std::mutex> lock(lock_seed_function);
                if(param.threshold == 0.0f)
                {
                    float w = threshold_gen(seed);
                    method->current_fa_threshold = w*fa_threshold1 + (1.0f-w)*fa_threshold2;
                }
                if(param.cull_cos_angle == 1.0f)
                    method->current_tracking_angle = (method->check_ending? 0.0f : std::cos(angle_gen(seed)));
                if(param.smooth_fraction == 1.0f)
                    method->current_tracking_smoothing = smoothing_gen(seed);
                if(param.step_size <= 0.0f) // 0: same as voxel spacing   -1: previous version voxel_size* [0.5 1.5]
                {
                    float step_size_in_voxel = (param.step_size == 0 ? 1.0f : step_gen(seed));
                    float step_size_in_mm = step_size_in_voxel*method->trk->vs[0];
                    method->current_step_size_in_voxel[0] = step_size_in_voxel;
                    method->current_step_size_in_voxel[1] = step_size_in_voxel;
                    method->current_step_size_in_voxel[2] = step_size_in_voxel;
                    method->current_max_steps3 = 3*uint32_t(std::round(param.max_length/step_size_in_mm));
                    method->current_min_steps3 = 3*uint32_t(std::round(param.min_length/step_size_in_mm));
                }

                uint32_t seed_index = std::min<uint32_t>(uint32_t(roi_mgr->seeds.size()-1),uint32_t(rand_gen(seed)*float(roi_mgr->seeds.size())));
                tipl::vector<3> pos = roi_mgr->seeds[seed_index];
                pos[0] += subvoxel_gen(seed);
                pos[1] += subvoxel_gen(seed);
                pos[2] += subvoxel_gen(seed);
                if(roi_mgr->need_trans[roi_mgr->seed_space[seed_index]])
                    pos.to(roi_mgr->to_diffusion_space[roi_mgr->seed_space[seed_index]]);
                method->position = pos;
            }

            if(!method->initialize_direction())
                continue;

            unsigned int point_count;
            const float *result = method->tracking(param.tracking_method,point_count);
            if(!result)
                continue;
            const float* end = result+point_count+point_count+point_count;

            ++tract_count[thread_id];
            if(buffer_switch)
                track_buffer_front[thread_id].push_back(std::vector<float>(result,end));
            else
                track_buffer_back[thread_id].push_back(std::vector<float>(result,end));
        }
    }
    catch(...)
    {

    }
    running[thread_id] = 0;
    end_time = std::chrono::high_resolution_clock::now();
}

bool ThreadData::fetchTracks(TractModel* handle)
{
    bool has_track = false;
    if(handle->parameter_id.empty())
        handle->parameter_id = param.get_code();
    auto& buffer_at_rest = buffer_switch ? track_buffer_back : track_buffer_front;
    for(auto& tract_per_thread : buffer_at_rest)
        if(!tract_per_thread.empty())
        {
            handle->add_tracts(tract_per_thread);
            tract_per_thread.clear();
            has_track = true;
        }
    buffer_switch = !buffer_switch;
    return has_track;
}

void ThreadData::apply_tip(TractModel* handle)
{
    for(size_t i = 0;i < param.tip_iteration && handle->get_visible_track_count() && handle->trim();++i)
        ;
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
    if(param.threshold == 0.0f)
    {
        fa_threshold1 = (param.default_otsu-0.1f)*trk->fa_otsu;
        fa_threshold2 = (param.default_otsu+0.1f)*trk->fa_otsu;
    }
    else
        fa_threshold1 = fa_threshold2 = param.threshold;

    report.clear();
    report.str("");
    if(!trk->dt_threshold_name.empty())
    {
        report << " Differential tractography (Yeh et al., Neuroimage, 2019 Nov 15;202:116131.) was applied to map pathways with ";
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
    else
        report << " A deterministic fiber tracking algorithm (Yeh et al., PLoS ONE 8(11): e80713, 2013) was used with augmented tracking strategies (Yeh, Neuroimage, 2020 Dec;223:117329) to improve reproducibility.";
    report << roi_mgr->report;
    report << param.get_report();
    report << " Shape analysis (Yeh, Neuroimage, 2020 Dec;223:117329) was conducted to derive shape metrics for tractography.";
    end_thread();
    if(thread_count < 1)
        thread_count = 1;

    joining = false;
    ready_to_track = false;
    begin_time = std::chrono::high_resolution_clock::now();

    //  multi-thread controls
    {
        seed_count  = std::move(std::vector<unsigned int>(thread_count));
        tract_count = std::move(std::vector<unsigned int>(thread_count));
        running     = std::move(std::vector<unsigned char>(thread_count,1));
    }
    // setting up output buffers
    {
        track_buffer_back.resize(thread_count);
        track_buffer_front.resize(thread_count);
    }
    for (unsigned int index = 0;index < thread_count-1;++index)
        threads.push_back(std::thread([=](){run_thread(index,thread_count);}));

    if(wait)
    {
        run_thread(thread_count-1,thread_count);
        for(auto& thread : threads)
            if(thread.joinable())
                thread.join();
        // make sure fetch tract get all data.
        buffer_switch = !buffer_switch;
    }
    else
        threads.push_back(std::thread([=](){run_thread(thread_count-1,thread_count);}));
}
