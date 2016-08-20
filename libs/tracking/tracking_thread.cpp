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
        for(int i = 0;i < threads.size();++i)
            threads[i]->wait();
        threads.clear();
    }
}

void ThreadData::run_thread(TrackingMethod* method_ptr,unsigned int thread_count,unsigned int thread_id,unsigned int max_count)
{
    std::auto_ptr<TrackingMethod> method(method_ptr);
    std::uniform_real_distribution<float> rand_gen(0,1);
    unsigned int iteration = thread_id; // for center seed
    float white_matter_t = method_ptr->param.threshold*1.2;
    if(!seeds.empty())
    try{
        std::vector<std::vector<float> > local_track_buffer;
        while(!joinning &&
              (!stop_by_tract || tract_count[thread_id] < max_count) &&
              (stop_by_tract || seed_count[thread_id] < max_count) &&
              (max_seed_count == 0 || seed_count[thread_id] < max_seed_count) &&
              (!center_seed || iteration < seeds.size()))
        {
            if(!pushing_data && (iteration & 0x00000FFF) == 0x00000FFF && !local_track_buffer.empty())
                push_tracts(local_track_buffer);
            ++seed_count[thread_id];
            if(center_seed)
            {
                if(!method->init(initial_direction,
                                 image::vector<3,float>(seeds[iteration].x(),seeds[iteration].y(),seeds[iteration].z()),
                                 seed))
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
                std::lock_guard<std::mutex> lock(lock_seed_function);
                iteration+=thread_count;
                unsigned int i = rand_gen(seed)*((float)seeds.size()-1.0);
                image::vector<3,float> pos;
                pos[0] = (float)seeds[i].x() + rand_gen(seed)-0.5;
                pos[1] = (float)seeds[i].y() + rand_gen(seed)-0.5;
                pos[2] = (float)seeds[i].z() + rand_gen(seed)-0.5;
                if(!method->init(initial_direction,pos,seed))
                    continue;
            }
            unsigned int point_count;
            const float *result = method->tracking(tracking_method,point_count);
            if(!result)
                continue;
            const float* end = result+point_count+point_count+point_count;
            if(check_ending)
            {
                if(point_count < 2)
                    continue;
                image::vector<3> p0(result),p1(result+3),p2(end-6),p3(end-3);
                p1 -= p0;
                p0 -= p1;
                p2 -= p3;
                p3 -= p2;
                if(method->trk.is_white_matter(p0,white_matter_t) ||
                   method->trk.is_white_matter(p3,white_matter_t))
                    continue;
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

    std::lock_guard<std::mutex> lock(lock_feed_function);
    handle->add_tracts(track_buffer);
    track_buffer.clear();
    return true;

}
void ThreadData::setRegions(image::geometry<3> dim,
                const std::vector<image::vector<3,short> >& points,
                unsigned char type,
                const char* roi_name)
{
    switch(type)
    {
    case 0: //ROI
        roi_mgr.add_inclusive_roi(dim,points);
        seed_report += " An ROI was placed at ";
        break;
    case 1: //ROA
        roi_mgr.add_exclusive_roi(dim,points);
        seed_report += " An ROA was placed at ";
        break;
    case 2: //End
        roi_mgr.add_end_roi(dim,points);
        seed_report += " An ending region was placed at ";
        break;
    case 4: //Terminate
        roi_mgr.add_terminate_roi(dim,points);
        seed_report += " A terminative region was placed at ";
        break;
    case 3: //seed
        for (unsigned int index = 0;index < points.size();++index)
            seeds.push_back(points[index]);
        seed_report += " A seeding region was placed at ";
        break;
    }
    seed_report += roi_name;
    seed_report += ".";
}
TrackingMethod* ThreadData::new_method(const tracking_data& trk)
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
    return new TrackingMethod(trk,interpo_method.release(),roi_mgr,param);
}

void ThreadData::run(const tracking_data& trk,
         unsigned int thread_count,
         unsigned int termination_count,
         bool wait)
{
    report.clear();
    report.str("");
    report << "\nA deterministic fiber tracking algorithm (Yeh et al., PLoS ONE 8(11): e80713) was used."
           << seed_report
           << " The angular threshold was " << (int)std::round(std::acos(param.cull_cos_angle)*180/3.1415926) << " degrees."
           << " The step size was " << param.step_size << " mm.";
    if(int(param.threshold*1000) == int(600*image::segmentation::otsu_threshold(image::make_image(trk.fa[0],trk.dim))))
        report << " The anisotropy threshold was determined automatically by DSI Studio.";
    else
        report << " The anisotropy threshold was " << param.threshold << ".";


    if(param.smooth_fraction != 0.0)
        report << " The fiber trajectories were smoothed by averaging the propagation direction with "
               << (int)std::round(param.smooth_fraction * 100.0) << "% of the previous direction.";

    if(param.min_points_count3 != 6)
        report << " Tracks with length less than "
               << (int)std::round(param.min_points_count3 * param.step_size /3.0) << " mm were discarded.";

    if(!termination_count)
        return;
    // to ensure consistency, seed initialization with all orientation only fits with single thread
    if(initial_direction == 2)
        thread_count = 1;
    param.step_size_in_voxel[0] = param.step_size/trk.vs[0];
    param.step_size_in_voxel[1] = param.step_size/trk.vs[1];
    param.step_size_in_voxel[2] = param.step_size/trk.vs[2];

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
    pushing_data = false;
    std::fill(running.begin(),running.end(),1);

    end_thread();
    if(thread_count > termination_count)
        thread_count = termination_count;
    unsigned int run_count = std::max<int>(1,termination_count/thread_count);
    unsigned int total_run_count = 0;
    for (unsigned int index = 0;index < thread_count-1;++index,total_run_count += run_count)
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [this,&trk,thread_count,index,run_count](){run_thread(new_method(trk),thread_count,index,run_count);})));


    if(wait)
    {
        run_thread(new_method(trk),thread_count,thread_count-1,termination_count-total_run_count);
        for(int i = 0;i < threads.size();++i)
            threads[i]->wait();
    }
    else
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                [this,&trk,thread_count,termination_count,total_run_count](){run_thread(new_method(trk),thread_count,thread_count-1,termination_count-total_run_count);})));

    report << " A total of " << termination_count << (stop_by_tract ? " tracts were calculated.":" seeds were placed.");
}
