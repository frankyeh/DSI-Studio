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

// the gqi balanced b-table and its transformation of the signals (new count x old count)
// physical: each row sums to one; otherwise the rows carry the gqi scale (1/N for dsi; for shells, b_k/(n_k*B) and no b0)
void Voxel::balance_scheme(bool physical,std::vector<float>& trans,
                           std::vector<float>& new_bvalues,std::vector<tipl::vector<3,float> >& new_bvectors) const
{
    unsigned int b_count = uint32_t(bvalues.size());
    unsigned int total_signals = 0;

    trans.clear();
    new_bvalues.clear();
    new_bvectors.clear();

    // if b0
    if(bvalues.front() == 0.0f)
    {
        trans.resize(b_count);
        trans[0] = (physical || need_resample_dsi) ? 1.0f : 0.0f; // a shell scheme gives b0 no weight: q = 0 has no surface
        new_bvectors.resize(1);
        new_bvalues.resize(1);
        total_signals = 1;
    }

    if(need_resample_dsi)
    {
        float mean_dis = 0.0f;
        // estimate average grid distance
        {
            std::vector<float> grid_distance;
            for(size_t i = 1;i < bvalues.size();++i)
                grid_distance.push_back(bvalues[i]-bvalues[i-1]);
            float dis_threshold = tipl::max_value(grid_distance.begin(),grid_distance.end())*0.5f;
            grid_distance.erase(std::remove_if(grid_distance.begin(),grid_distance.end(),[dis_threshold](auto v){return v < dis_threshold;}),grid_distance.end());
            mean_dis = tipl::mean(grid_distance.begin(),grid_distance.end())*2.0f;
        }

        float max_q2 = 16.0f;
        float b_max = bvalues.back();
        for(int z = -4;z <= 4;++z)
            for(int y = -4;y <= 4;++y)
                for(int x = -4;x <= 4;++x)
                {
                    auto l2 = x*x+y*y+z*z;
                    if(l2 == 0)
                        goto end;
                    if(l2 <= max_q2)
                    {
                        new_bvalues.push_back(b_max*float(l2)/max_q2);
                        new_bvectors.push_back(tipl::v(x,y,z));
                        new_bvectors.back().normalize();
                    }
                }
        end:
        trans.resize(new_bvectors.size()*size_t(b_count));
        for(size_t i = total_signals; i < new_bvectors.size();++i)
        {
            auto nb = new_bvectors[i];
            nb *= new_bvalues[i]/mean_dis;
            std::vector<double> w;
            for(size_t b = total_signals;b < b_count;++b)
            {
                auto cb = bvectors[b];
                cb *= bvalues[b]/mean_dis;
                auto d2 = std::min<float>((nb + cb).length2(),(nb - cb).length2());
                w.push_back(std::exp(-2.0*d2));
            }
            auto sum_w = tipl::sum(w.begin(),w.end());
            tipl::multiply_constant(w.begin(),w.end(),1.0/sum_w/(physical ? 1.0 : double(new_bvectors.size())));
            std::copy(w.begin(),w.end(),trans.begin() + total_signals + b_count*i);
        }
    }
    else
    {
        // each shell is sampled uniformly on its q-space surface: per-dwi weight b_k/(n_k*B), B = sum of the shell b-values
        double sum_b = 0.0;
        for(unsigned int shell_index = 0;shell_index < shell.size();++shell_index)
            sum_b += tipl::mean(bvalues.begin()+shell[shell_index],bvalues.begin()+(shell_index + 1 == shell.size() ? b_count : shell[shell_index+1]));
        for(unsigned int shell_index = 0;shell_index < shell.size();++shell_index)
        {
            unsigned int from = shell[shell_index];
            unsigned int to = (shell_index + 1 == shell.size() ? b_count : shell[shell_index+1]);
            unsigned int num = to-from;


            //calculate averaged angle distance
            double averaged_angle = 0.0;
            for(unsigned int i = from;i < to;++i)
            {
                double max_cos = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    if(i == j)
                        continue;
                    double cur_cos = std::abs(std::cos(double(bvectors[i]*bvectors[j])));
                    if(cur_cos > 0.998)
                        continue;
                    max_cos = std::max<double>(max_cos,cur_cos);
                }
                averaged_angle += std::acos(max_cos);
            }
            averaged_angle /= num;

            //calculate averaged b_value
            double avg_b = tipl::mean(bvalues.begin()+from,bvalues.begin()+to);
            tessellated_icosahedron new_dir(avg_b > 2500.0 ? 6 : avg_b > 1500.0 ? 5 : avg_b > 500.0 ? 4 : 3); // 181, 126, 81, or 46 directions
            size_t trans_old_size = trans.size();
            trans.resize(trans.size() + size_t(new_dir.half_vertices_count)*size_t(b_count));
            for(unsigned int i = 0; i < new_dir.half_vertices_count;++i)
            {
                std::vector<double> t(b_count);
                double effective_b = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    double angle = std::acos(std::min<double>(1.0,std::abs(double(new_dir.vertices[i]*bvectors[j]))));
                    angle/=averaged_angle;
                    t[j] = std::exp(-2.0*angle*angle); // if the angle == 1, then weighting = 0.135
                    effective_b += t[j]*double(bvalues[j]);
                }
                double sum_t = std::accumulate(t.begin(),t.end(),0.0);
                tipl::multiply_constant(t.begin(),t.end(),(physical ? 1.0 : avg_b/sum_b/double(new_dir.half_vertices_count))/sum_t);
                std::copy(t.begin(),t.end(),trans.begin() + int64_t(trans_old_size + size_t(i) * size_t(b_count)));
                new_bvalues.push_back(float(effective_b/sum_t));
                new_bvectors.push_back(new_dir.vertices[i]);
            }
        }
    }
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
    tipl::progress prog("post-reconstruction");
    for (size_t index = 0;prog(index,process_list.size());++index)
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
