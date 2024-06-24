#include "roi.hpp"

bool RoiMgr::setAtlas(bool& terminated,float seed_threshold,float not_end_threshold)
{
    if(!handle->load_track_atlas())
        return false;
    track_ids = handle->get_track_ids(tract_name);
    if(track_ids.empty())
    {
        handle->error_msg = "invalid tract name: ";
        handle->error_msg += tract_name;
        return false;
    }
    if(terminated)
        return false;

    {
        float tolerance_dis_in_icbm_voxels = tolerance_dis_in_icbm152_mm/handle->template_vs[0];
        tolerance_dis_in_subject_voxels = tolerance_dis_in_icbm_voxels/handle->tract_atlas_jacobian;
        tipl::out() << "convert tolerance distance of " << std::fixed << std::setprecision(2) << tolerance_dis_in_icbm152_mm << " from ICBM mm to " <<
                                tolerance_dis_in_subject_voxels << " subject voxels" << std::endl;
    }

    std::vector<tipl::vector<3,short> > tract_coverage;
    {
        for(auto id : track_ids)
        {
            std::vector<tipl::vector<3,short> > region;
            handle->track_atlas->to_voxel(region,tipl::identity_matrix(),int(id));
            if(!tract_coverage.empty())
            {
                std::vector<tipl::vector<3,short> > merged;
                std::merge(tract_coverage.begin(),tract_coverage.end(),region.begin(),region.end(),
                            std::back_inserter(merged));
                merged.swap(tract_coverage);
            }
            else
                region.swap(tract_coverage);
        }
    }



    {
        // add limiting region to speed up tracking
        tipl::out() << "creating limiting region to limit tracking results" << std::endl;

        bool is_left = (tract_name.back() == 'L' || tipl::contains(tract_name,"L_")) ;
        bool is_right = (tract_name.back() == 'R' || tipl::contains(tract_name,"R_"));
        auto mid_x = handle->template_I.width() >> 1;
        auto& s2t = handle->get_sub2temp_mapping();
        if(is_left)
            tipl::out() << "apply left limiting mask for " << tract_name << std::endl;
        if(is_right)
            tipl::out() << "apply right limiting mask for " << tract_name << std::endl;

        tipl::image<3,char> limiting_mask(handle->dim);
        const float *fa0 = handle->dir.fa[0];
        tipl::par_for(tract_coverage.size(),[&](unsigned int i)
        {
            tipl::for_each_neighbors(tipl::pixel_index<3>(tract_coverage[i].begin(),handle->dim),
                                handle->dim,int(std::ceil(tolerance_dis_in_subject_voxels)),
                    [&](const auto& pos)
            {
                if(fa0[pos.index()] <= 0.0f)
                    return;
                if(is_left && s2t[pos.index()][0] < mid_x)
                    return;
                if(is_right && s2t[pos.index()][0] > mid_x)
                    return;
                limiting_mask[pos.index()] = 1;
            });
        });
        if(terminated)
            return false;

        std::vector<std::vector<tipl::vector<3,short> > > limiting_points(tipl::max_thread_count),
                                                          seed_points(tipl::max_thread_count),
                                                          not_end_points(tipl::max_thread_count);
        tipl::par_for<tipl::sequential_with_id>(tipl::begin_index(limiting_mask.shape()),tipl::end_index(limiting_mask.shape()),
                      [&](const auto& pos,unsigned int thread_id)
        {
            if(!limiting_mask[pos.index()])
                return;
            auto point = tipl::vector<3,short>(pos.x(), pos.y(),pos.z());
            limiting_points[thread_id].push_back(point);
            if(fa0[pos.index()] >= seed_threshold)
                seed_points[thread_id].push_back(point);
            if(not_end_threshold != 0.0f && fa0[pos.index()] >= not_end_threshold)
                not_end_points[thread_id].push_back(point);
        });

        tipl::aggregate_results(std::move(limiting_points),atlas_limiting);
        tipl::aggregate_results(std::move(seed_points),atlas_seed);
        tipl::aggregate_results(std::move(not_end_points),atlas_not_end);

        setRegions(atlas_limiting,limiting_id,"track tolerance region");
        if(!atlas_not_end.empty())
            setRegions(atlas_not_end,not_end_id,"white matter region");
        if(seeds.empty())
            setRegions(atlas_seed,seed_id,tract_name.c_str());

    }

    if(handle->tractography_atlas_roi.get())
    {
        tipl::out() << "checking additional ROI for refining tracking";
        const auto& regions = handle->tractography_atlas_roi->get_list();
        for(size_t i = 0;i < regions.size();++i)
            if(tipl::contains_case_insensitive(tract_name,regions[i]))
            {
                if(!handle->get_atlas_roi(handle->tractography_atlas_roi,i,atlas_roi))
                {
                    tipl::out() << "cannot add ROI: " << regions[i] << " " << handle->error_msg;
                    return false;
                }
                if(atlas_roi.empty())
                {
                    tipl::out() << "no region in the ROI. skipping";
                    continue;
                }
                tipl::out() << "additional ROI added: " << regions[i];
                setRegions(atlas_roi,roi_id,regions[i].c_str());
            }
    }
    if(handle->tractography_atlas_roa.get())
    {
        tipl::out() << "checking additional ROA for refining tracking";
        const auto& regions = handle->tractography_atlas_roa->get_list();
        for(size_t i = 0;i < regions.size();++i)
            if(tipl::contains_case_insensitive(tract_name,regions[i]))
            {
                if(!handle->get_atlas_roi(handle->tractography_atlas_roa,i,atlas_roa))
                {
                    tipl::out() << "cannot add ROA: " << regions[i] << " " << handle->error_msg;
                    return false;
                }
                if(atlas_roa.empty())
                {
                    tipl::out() << "no region in the ROA. skipping";
                    continue;
                }
                tipl::out() << "additional ROA added: " << regions[i];
                setRegions(atlas_roa,roa_id,regions[i].c_str());
            }
    }
    {
        const auto& atlas_tract = handle->track_atlas->get_tracts();
        const auto& atlas_cluster = handle->track_atlas->tract_cluster;
        auto tolerance_dis_in_subject_voxels2 = tolerance_dis_in_subject_voxels*2;

        std::vector<bool> is_target(atlas_tract.size());
        tipl::par_for(atlas_tract.size(),[&](unsigned int i)
        {
            is_target[i] = (std::find(track_ids.begin(),track_ids.end(),atlas_cluster[i]) != track_ids.end());
        });

        std::vector<std::vector<std::vector<float> > > selected_atlas_tracts_threads(tipl::max_thread_count);
        std::vector<std::vector<unsigned int> > selected_atlas_cluster_threads(tipl::max_thread_count);
        tipl::par_for<tipl::sequential_with_id>(atlas_tract.size(),[&](unsigned int i,unsigned int id)
        {
            if(!is_target[i])
            {
                bool needed = false;
                for(size_t j = 0;j < atlas_tract.size();++j)
                    if(is_target[j])
                    {
                        if(distance_over_limit(&atlas_tract[i][0],atlas_tract[i].size(),
                                               &atlas_tract[j][0],atlas_tract[j].size(),
                                               tolerance_dis_in_subject_voxels2))
                            continue;
                        needed = true;
                        break;
                    }
                if(!needed)
                    return;
            }
            selected_atlas_tracts_threads[id].push_back(atlas_tract[i]);
            selected_atlas_cluster_threads[id].push_back(atlas_cluster[i]);

        });
        tipl::aggregate_results(std::move(selected_atlas_tracts_threads),selected_atlas_tracts);
        tipl::aggregate_results(std::move(selected_atlas_cluster_threads),selected_atlas_cluster);
    }
    return true;
}
