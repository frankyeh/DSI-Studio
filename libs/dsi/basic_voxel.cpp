#include <boost/math/special_functions/sinc.hpp>
#include "basic_voxel.hpp"
#include "image_model.hpp"
double base_function(double theta)
{
    if(std::abs(theta) < 0.000001)
        return 1.0/3.0;
    return (2*std::cos(theta)+(theta-2.0/theta)*std::sin(theta))/theta/theta;
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
                           std::sqrt(bvalues[i]*0.01506);

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
    std::vector<int> sorted_index(image_model.src_bvalues.size());
    for(int i = 0;i < sorted_index.size();++i)
        sorted_index[i] = i;

    std::sort(sorted_index.begin(),sorted_index.end(),
              [&image_model](int left,int right)
    {
        return image_model.src_bvalues[left] < image_model.src_bvalues[right];
    }
    );


    bvalues.clear();
    bvectors.clear();
    dwi_data.clear();
    // include only the first b0
    if(image_model.src_bvalues[sorted_index[0]] == 0.0f)
    {
        bvalues.push_back(0);
        bvectors.push_back(tipl::vector<3,float>(0,0,0));
        dwi_data.push_back(image_model.src_dwi_data[sorted_index[0]]);
        b0_index = 0;
    }
    for(int i = 0;i < sorted_index.size();++i)
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


void Voxel::calculate_mask(const tipl::image<float,3>& dwi_sum)
{
    tipl::threshold(dwi_sum,mask,0.2f,1,0);
    if(dwi_sum.depth() < 10)
    {
        for(unsigned int i = 0;i < mask.depth();++i)
        {
            tipl::pointer_image<unsigned char,2> I(&mask[0]+i*mask.plane_size(),
                    tipl::geometry<2>(mask.width(),mask.height()));
            tipl::morphology::defragment(I);
            tipl::morphology::recursive_smoothing(I,10);
            tipl::morphology::defragment(I);
        }
    }
    else
    {
        tipl::morphology::recursive_smoothing(mask,10);
        tipl::morphology::defragment(mask);
        tipl::morphology::recursive_smoothing(mask,10);
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
    tipl::par_for_asyn2(mask.size(),
                    [&](int voxel_index,int thread_id)
    {
        ++total;
        if(terminated || !mask[voxel_index])
            return;
        if(thread_id == 0)
        {
            if(prog_aborted())
            {
                terminated = true;
                return;
            }
            check_prog(total,mask.size());
        }
        voxel_data[thread_id].init();
        voxel_data[thread_id].voxel_index = voxel_index;
        for (int index = 0; index < process_list.size(); ++index)
            process_list[index]->run(*this,voxel_data[thread_id]);
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
