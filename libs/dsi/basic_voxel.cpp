#include <boost/math/special_functions/sinc.hpp>
#include "basic_voxel.hpp"
#include "image_model.hpp"
float base_function(float theta)
{
    if(std::fabs(theta) < 0.000001f)
        return 1.0f/3.0f;
    return (2.0f*std::cos(theta)+(theta-2.0f/theta)*std::sin(theta))/theta/theta;
}

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

void Voxel::calculate_sinc_ql(std::vector<float>& sinc_ql)
{
    unsigned int odf_size = ti.half_vertices_count;
    float sigma = param[0];
    sinc_ql.resize(odf_size*bvalues.size());
    for (unsigned int j = 0,index = 0; j < odf_size; ++j)
        for (unsigned int i = 0; i < bvalues.size(); ++i,++index)
            sinc_ql[index] = bvectors[i]*
                         tipl::vector<3,float>(ti.vertices[j])*
                           std::sqrt(bvalues[i]*0.01506f);

    for (unsigned int index = 0; index < sinc_ql.size(); ++index)
        sinc_ql[index] = r2_weighted ?
                     base_function(sinc_ql[index]*sigma):
                     boost::math::sinc_pi(sinc_ql[index]*sigma);
}
void Voxel::calculate_q_vec_t(std::vector<tipl::vector<3,float> >& q_vectors_time)
{
    float sigma = param[0];
    q_vectors_time.resize(bvalues.size());
    for (unsigned int index = 0; index < bvalues.size(); ++index)
    {
        q_vectors_time[index] = bvectors[index];
        q_vectors_time[index] *= std::sqrt(bvalues[index]*0.01506f);// get q in (mm) -1
        q_vectors_time[index] *= sigma;
    }
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

    if(image_model.has_image_rotation)
        for (unsigned int index = 0;index < bvectors.size();++index)
            {
                tipl::vector<3> tmp;
                tipl::vector_rotation(bvectors[index].begin(),tmp.begin(),image_model.src_bvectors_rotate,tipl::vdim<3>());
                tmp.normalize();
                bvectors[index] = tmp;
            }

}

bool Voxel::run(void)
{
    size_t total_voxel = std::accumulate(mask.begin(),mask.end(),size_t(0),[](size_t sum,unsigned char value){return value ? sum+1:sum;});
    size_t total = 0;
    bool terminated = false;
    tipl::par_for2(mask.size(),[&](size_t voxel_index,size_t thread_id)
    {
        if(terminated || !mask[voxel_index])
            return;
        ++total;
        if(thread_id == 0)
        {
            if(prog_aborted())
            {
                terminated = true;
                return;
            }
            check_prog(uint32_t(total*100/total_voxel),100);
        }
        voxel_data[thread_id].init();
        voxel_data[thread_id].voxel_index = voxel_index;
        for (size_t index = 0; index < process_list.size(); ++index)
            process_list[index]->run(*this,voxel_data[thread_id]);
    },thread_count);

    return !prog_aborted();
}


void Voxel::end(gz_mat_write& writer)
{
    for (size_t index = 0; check_prog(uint32_t(index),uint32_t(process_list.size())); ++index)
        process_list[index]->end(*this,writer);
}

BaseProcess* Voxel::get(unsigned int index)
{
    return process_list[index].get();
}
