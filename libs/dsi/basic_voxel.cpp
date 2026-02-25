#include "basic_voxel.hpp"
#include "image_model.hpp"

bool Voxel::init(void)
{
    tipl::progress prog("pre-reconstruction",true);
    if(is_histology)
    {
        hist_data.resize(thread_count);
        for (unsigned int index = 0; index < thread_count; ++index)
            hist_data[index].init();
    }
    else
    {
        voxel_data.resize(thread_count);
        for (unsigned int index = 0; index < thread_count; ++index)
        {
            voxel_data[index].space.resize(bvalues.size());
            voxel_data[index].odf.resize(ti.half_vertices_count);
            voxel_data[index].fa.resize(max_fiber_number);
            voxel_data[index].dir_index.resize(max_fiber_number);
            voxel_data[index].dir.resize(max_fiber_number);
        }
    }
    for (unsigned int index = 0; prog(index,process_list.size()); ++index)
    {
        tipl::out() << process_name[index];
        process_list[index]->init(*this);
    }
    return !prog.aborted();
}

void calculate_shell(std::vector<float> sorted_bvalues,
                     std::vector<unsigned int>& shell)
{
    shell.clear();
    float b_dif_threshold = 0.0f;
    std::sort(sorted_bvalues.begin(),sorted_bvalues.end());
    for(uint32_t i = 0;i < sorted_bvalues.size();++i)
        if(sorted_bvalues[i] > 100.0f)
            {
                shell.push_back(i);
                b_dif_threshold = sorted_bvalues[i];
                for(uint32_t index = i+1;index < sorted_bvalues.size();++index)
                    b_dif_threshold = std::max<float>(b_dif_threshold,std::abs(sorted_bvalues[index]-sorted_bvalues[index-1]));
                b_dif_threshold *= 0.5f;
                break;
            }
    if(shell.empty())
        return;
    for(uint32_t index = shell.back()+1;index < sorted_bvalues.size();++index)
        if(std::abs(sorted_bvalues[index]-sorted_bvalues[index-1]) > b_dif_threshold)
            shell.push_back(index);
}


void Voxel::load_from_src(src_data& image_model)
{
    if(image_model.src_bvalues.empty()) // e.g. template recon
        return;
    std::vector<size_t> sorted_index(image_model.get_sorted_dwi_index());
    bvalues.clear();
    bvectors.clear();
    dwi_data.clear();
    // include only the first b0
    if(image_model.src_bvalues[sorted_index[0]] == 0.0f)
    {
        bvalues.push_back(0);
        bvectors.push_back(tipl::vector<3,float>(0,0,0));
        dwi_data.push_back(image_model.src_dwi_data[sorted_index[0]]);
    }
    for(size_t i = 0;i < sorted_index.size();++i)
        if(image_model.src_bvalues[sorted_index[i]] != 0.0f)
        {
            bvalues.push_back(image_model.src_bvalues[sorted_index[i]]);
            bvectors.push_back(image_model.src_bvectors[sorted_index[i]]);
            dwi_data.push_back(image_model.src_dwi_data[sorted_index[i]]);
        }

    calculate_shell(bvalues,shell);
    auto is_dsi = [this](void){return shell.size() > 4 && shell[1] - shell[0] <= 6;};

    if(method_id == 7 || method_id == 4)
    {
        need_resample_shells = shell.size() < 5;
        if((need_resample_dsi = shell.size() >= 5 && dwi_data.size() < 64))
            tipl::out() << "dsi resampling needed";
    }
    if(method_id == 1)
        max_fiber_number = 1;
    else
        max_fiber_number = 3;
}
bool Voxel::run_hist(void)
{
    tipl::progress prog("reconstructing histology");
    margin = 16;
    auto ceil = hist_tensor_smoothing << hist_downsampling;
    while(margin < ceil)
        margin <<= 1;
    std::cout << "margin=" << margin << std::endl;
    crop_size = margin << 3;
    std::cout << "crop_size=" << crop_size << std::endl;


    std::vector<tipl::vector<2,int> > from_list;
    std::vector<tipl::vector<2,int> > to_list;
    for(int y = 0;y < hist_image.height(); y+= crop_size)
        for(int x = 0;x < hist_image.width(); x+= crop_size)
        {
            tipl::vector<2,int> from(x-int(margin),y-int(margin)),to(x+int(crop_size+margin),y+int(crop_size+margin));
            if(from[0] < 0)
                from[0] = 0;
            if(from[1] < 0)
                from[1] = 0;
            if(to[0] >= hist_image.width())
                to[0] = hist_image.width()-1;
            if(to[1] >= hist_image.height())
                to[1] = hist_image.height()-1;
            from_list.push_back(from);
            to_list.push_back(to);
        }

    size_t p = 0;
    tipl::par_for(thread_count,[&](size_t thread_id)
    {
        hist_data[thread_id].init();
        for(size_t i = thread_id;i < from_list.size() && prog(p++,from_list.size());i += thread_count)
        {
            hist_data[thread_id].from = from_list[i];
            hist_data[thread_id].to = to_list[i];
            for (unsigned int j = 0; j < process_list.size(); ++j)
                process_list[j]->run_hist(*this,hist_data[thread_id]);
        }
    });
    return !prog.aborted();
}
bool Voxel::run(const char* title)
{
    tipl::progress prog(title, true);
    std::atomic<size_t> count = 0;
    size_t total = mask.size();
    tipl::par_for<tipl::dynamic_with_id>(total, [&](size_t voxel_index, size_t thread_id)
    {
        size_t current = ++count;
        if (thread_id == 0 && (current & 63) == 0)
            prog(current, total);

        if (!mask[voxel_index] || prog.aborted())
            return;

        voxel_data[thread_id].init();
        voxel_data[thread_id].voxel_index = voxel_index;

        for (const auto& each : process_list)
            each->run(*this, voxel_data[thread_id]);
    }, thread_count);

    return !prog.aborted();
}


bool Voxel::end(tipl::io::gz_mat_write& writer)
{
    tipl::progress prog("post-reconstruction",true);
    for (size_t index = 0;prog(uint32_t(index),uint32_t(process_list.size())); ++index)
    {
        tipl::out() << process_name[index];
        process_list[index]->end(*this,writer);
    }
    return !prog.aborted();
}

BaseProcess* Voxel::get(unsigned int index)
{
    return process_list[index].get();
}
