#include "basic_voxel.hpp"



void Voxel::init(void)
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
    for (unsigned int index = 0; index < process_list.size(); ++index)
        process_list[index]->init(*this);
}


void Voxel::sort_b_table(void)
{
    std::vector<int> sorted_index(bvalues.size());
    for(int i = 0;i < sorted_index.size();++i)
        sorted_index[i] = i;

    std::sort(sorted_index.begin(),sorted_index.end(),
              [&](int left,int right)
    {
        return bvalues[left] < bvalues[right];
    }
    );

    std::vector<image::vector<3,float> > new_bvectors;
    std::vector<float> new_bvalues;
    std::vector<const unsigned short*> new_dwi_data;

    // include only the first b0
    if(bvalues[sorted_index[0]] == 0.0f)
    {
        new_bvalues.push_back(0);
        new_bvectors.push_back(image::vector<3,float>(0,0,0));
        new_dwi_data.push_back(dwi_data[sorted_index[0]]);
    }
    for(int i = 0;i < sorted_index.size();++i)
        if(bvalues[sorted_index[i]] != 0.0f)
        {
            new_bvalues.push_back(bvalues[sorted_index[i]]);
            new_bvectors.push_back(bvectors[sorted_index[i]]);
            new_dwi_data.push_back(dwi_data[sorted_index[i]]);
        }
    bvalues.swap(new_bvalues);
    bvectors.swap(new_bvectors);
    dwi_data.swap(new_dwi_data);
}

void Voxel::calculate_dwi_sum(void)
{
    dwi_sum.clear();
    dwi_sum.resize(dim);
    image::par_for(dwi_sum.size(),[&](unsigned int pos)
    {
        for (unsigned int index = 0;index < dwi_data.size();++index)
            dwi_sum[pos] += dwi_data[index][pos];
    });

    float max_value = *std::max_element(dwi_sum.begin(),dwi_sum.end());
    float min_value = max_value;
    for (unsigned int index = 0;index < dwi_sum.size();++index)
        if (dwi_sum[index] < min_value && dwi_sum[index] > 0)
            min_value = dwi_sum[index];


    image::minus_constant(dwi_sum,min_value);
    image::lower_threshold(dwi_sum,0.0f);
    float t = image::segmentation::otsu_threshold(dwi_sum);
    image::upper_threshold(dwi_sum,t*3.0f);
    image::normalize(dwi_sum,1.0);
}
void Voxel::calculate_mask(void)
{
    image::threshold(dwi_sum,mask,0.2f,1,0);
    if(dwi_sum.depth() < 10)
    {
        for(unsigned int i = 0;i < mask.depth();++i)
        {
            image::pointer_image<unsigned char,2> I(&mask[0]+i*mask.plane_size(),
                    image::geometry<2>(mask.width(),mask.height()));
            image::morphology::defragment(I);
            image::morphology::recursive_smoothing(I,10);
            image::morphology::defragment(I);
        }
    }
    else
    {
        image::morphology::recursive_smoothing(mask,10);
        image::morphology::defragment(mask);
        image::morphology::recursive_smoothing(mask,10);
    }
}


void Voxel::run(void)
{
    try{

    size_t total_voxel = 0;
    bool terminated = false;
    begin_prog("reconstructing");
    for(size_t index = 0;index < mask.size();++index)
        if (mask[index])
            ++total_voxel;

    unsigned int total = 0;
    image::par_for2(mask.size(),
                    [&](int voxel_index,int thread_index)
    {
        ++total;
        if(terminated || !mask[voxel_index])
            return;
        if(thread_index == 0)
        {
            if(prog_aborted())
            {
                terminated = true;
                return;
            }
            check_prog(total,mask.size());
        }
        voxel_data[thread_index].init();
        voxel_data[thread_index].voxel_index = voxel_index;
        for (int index = 0; index < process_list.size(); ++index)
            process_list[index]->run(*this,voxel_data[thread_index]);
    },thread_count);
    check_prog(1,1);
    }
    catch(std::exception& error)
    {
        std::cout << error.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "unknown error" << std::endl;
    }

}


void Voxel::end(gz_mat_write& writer)
{
    begin_prog("output data");
    for (unsigned int index = 0; check_prog(index,process_list.size()); ++index)
        process_list[index]->end(*this,writer);
}

BaseProcess* Voxel::get(unsigned int index)
{
    return process_list[index].get();
}
