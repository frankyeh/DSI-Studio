#include "basic_voxel.hpp"
#include "image_model.hpp"

bool Voxel::init(void)
{
    tipl::progress prog("initializing",true);
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
        process_list[index]->init(*this);
    return !prog.aborted();
}

void calculate_shell(std::vector<float> sorted_bvalues,
                     std::vector<unsigned int>& shell)
{
    shell.clear();
    std::sort(sorted_bvalues.begin(),sorted_bvalues.end());
    for(uint32_t i = 0;i < sorted_bvalues.size();++i)
        if(sorted_bvalues[i] > 100.0f)
            {
                shell.push_back(i);
                break;
            }
    if(shell.empty())
        return;
    for(uint32_t index = shell.back()+1;index < sorted_bvalues.size();++index)
        if(std::abs(sorted_bvalues[index]-sorted_bvalues[index-1]) > 100.0f)
            shell.push_back(index);
}


void Voxel::load_from_src(ImageModel& image_model)
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

    scheme_balance = !is_dsi() && shell.size() < 5 && (method_id == 7 || method_id == 4);
    if(scheme_balance)
        tipl::out() << "scheme balance is applied to avoid directional bias";

    if(method_id == 1)
        max_fiber_number = 1;
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
    tipl::par_for(from_list.size(),[&](size_t i,size_t thread_id)
    {
        prog(p++,from_list.size());
        if(prog.aborted())
            return;
        hist_data[thread_id].init();
        hist_data[thread_id].from = from_list[i];
        hist_data[thread_id].to = to_list[i];
        for (unsigned int j = 0; j < process_list.size(); ++j)
            process_list[j]->run_hist(*this,hist_data[thread_id]);
    });
    return !prog.aborted();
}
bool Voxel::run(const char* title)
{
    tipl::progress prog(title,true);
    size_t total_size = 0;
    tipl::par_for(thread_count,[&](size_t thread_id)
    {
        for(size_t voxel_index = thread_id;voxel_index < mask.size() && prog(++total_size,mask.size());voxel_index += thread_count)
        {
            if(!mask[voxel_index])
                continue;
            voxel_data[thread_id].init();
            voxel_data[thread_id].voxel_index = voxel_index;
            for (size_t index = 0; index < process_list.size(); ++index)
                process_list[index]->run(*this,voxel_data[thread_id]);
        }
    },thread_count);
    return !prog.aborted();
}


void Voxel::end(tipl::io::gz_mat_write& writer)
{
    tipl::progress prog("saving results",true);
    for (size_t index = 0;prog(uint32_t(index),uint32_t(process_list.size())); ++index)
        process_list[index]->end(*this,writer);
}

BaseProcess* Voxel::get(unsigned int index)
{
    return process_list[index].get();
}
