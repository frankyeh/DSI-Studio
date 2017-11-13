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
