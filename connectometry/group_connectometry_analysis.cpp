#include <QFileInfo>
#include <ctime>
#include "connectometry/group_connectometry_analysis.h"
#include "fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "tracking/tracking_window.h"
#include "tracking/region/regiontablewidget.h"
#include <filesystem>


bool group_connectometry_analysis::load_database(const char* database_name)
{
    handle.reset(new fib_data);
    if(!handle->load_from_file(database_name))
    {
        error_msg = "invalid fib file: ";
        error_msg += handle->error_msg;
        return false;
    }
    if(!handle->db.has_db())
    {
        error_msg = "not a database file ";
        return false;
    }
    fiber_threshold = 0.6f*handle->dir.fa_otsu;
    return true;
}


int group_connectometry_analysis::run_track(std::shared_ptr<tracking_data> fib,
                                            std::vector<std::vector<float> >& tracks,
                                            unsigned int seed_count,
                                            unsigned int random_seed,
                                            unsigned int thread_count)
{
    ThreadData tracking_thread(handle);
    tracking_thread.param.random_seed = random_seed;
    tracking_thread.param.threshold = fiber_threshold;
    tracking_thread.param.dt_threshold = t_threshold;
    tracking_thread.param.cull_cos_angle = 1.0f;
    tracking_thread.param.step_size = handle->vs[0];
    tracking_thread.param.min_length = float(length_threshold_voxels)*handle->vs[0];
    tracking_thread.param.max_length = 2.0f*float(std::max<unsigned int>(handle->dim[0],std::max<unsigned int>(handle->dim[1],handle->dim[2])))*handle->vs[0];
    tracking_thread.param.max_seed_count = uint32_t(seed_count);
    tracking_thread.roi_mgr = roi_mgr;
    tracking_thread.run(fib,thread_count,true);
    for(auto& tracts_per_thread : tracking_thread.track_buffer_front)
        for(auto& tract : tracts_per_thread)
            if(!tract.empty())
                tracks.push_back(std::move(tract));
    return int(tracks.size());
}

void cal_hist(const std::vector<std::vector<float> >& track,std::vector<unsigned int>& dist)
{
    for(unsigned int j = 0; j < track.size();++j)
    {
        if(track[j].size() <= 3)
            continue;
        unsigned int length = track[j].size()/3-1;
        if(length < dist.size())
            ++dist[length];
        else
            if(!dist.empty())
                ++dist.back();
    }
}

void group_connectometry_analysis::exclude_cerebellum(void)
{
    if(handle->is_human_data)
    {
        std::vector<tipl::vector<3,short> > points;
        if(!handle->get_atlas_roi("BrainSeg","Cerebellum",points))
            return;
        roi_mgr->setRegions(points,4/*terminative*/,"Cerebellum");
        roi_mgr->report = " Cerebellum was excluded.";
    }
}

void group_connectometry_analysis::run_permutation_multithread(unsigned int id,unsigned int thread_count,unsigned int permutation_count)
{
    connectometry_result data;
    std::shared_ptr<tracking_data> fib(new tracking_data);
    fib->read(handle);
    bool null = true;
    for(unsigned int i = id;i < permutation_count && !terminated;)
    {
        std::vector<std::vector<float> > pos_tracks,neg_tracks;

        stat_model info;

        info.resample(*model.get(),null,true,i);
        calculate_spm(data,info);
        fib->dt_fa = data.dec_ptr;

        run_track(fib,neg_tracks,seed_count,i);
        cal_hist(neg_tracks,(null) ? tract_count_dec_null : tract_count_dec);


        info.resample(*model.get(),null,true,i);
        calculate_spm(data,info);
        fib->dt_fa = data.inc_ptr;

        run_track(fib,pos_tracks,seed_count,i);
        cal_hist(pos_tracks,(null) ? tract_count_inc_null : tract_count_inc);

        {
            std::lock_guard<std::mutex> lock(lock_add_tracks);
            if(null)
            {
                neg_null_corr_track->add_tracts(neg_tracks,length_threshold_voxels,tipl::rgb(0x004040F0));
                pos_null_corr_track->add_tracts(pos_tracks,length_threshold_voxels,tipl::rgb(0x00F04040));
            }
            else
            {
                dec_track->add_tracts(neg_tracks,length_threshold_voxels,tipl::rgb(0x004040F0));
                inc_track->add_tracts(pos_tracks,length_threshold_voxels,tipl::rgb(0x00F04040));
            }
        }

        if(!null)
        {
            ++preprocess;
            i += thread_count;
            if(id == 0)    
                prog = uint32_t(i*95/permutation_count);
        }
        null = !null;
    }
    if(id == 0 && !terminated)
    {
        wait(1); // current thread occupies 0, wait from 1
        save_result();
        prog = 100;
    }
}
extern int fib_ver;
bool save_fz(tipl::io::gz_mat_read& mat_reader,
              tipl::io::gz_mat_write& matfile,
              const std::vector<std::string>& skip_list,
              const std::vector<std::string>& skip_head_list);
void group_connectometry_analysis::save_result(void)
{
    tipl::progress p("save correlational tractography results");
    for(size_t index = 0;index < tip_iteration;++index)
    {
        neg_null_corr_track->trim();
        pos_null_corr_track->trim();
        dec_track->trim();
        inc_track->trim();
    }
    if(region_pruning)
    {
        neg_null_corr_track->region_pruning(0.2f);
        pos_null_corr_track->region_pruning(0.2f);
        dec_track->region_pruning(0.2f);
        inc_track->region_pruning(0.2f);
    }
    // update fdr table
    std::fill(tract_count_dec_null.begin(),tract_count_dec_null.end(),0);
    std::fill(tract_count_inc_null.begin(),tract_count_inc_null.end(),0);
    std::fill(tract_count_dec.begin(),tract_count_dec.end(),0);
    std::fill(tract_count_inc.begin(),tract_count_inc.end(),0);
    cal_hist(dec_track->get_tracts(),tract_count_dec);
    cal_hist(neg_null_corr_track->get_tracts(),tract_count_dec_null);
    cal_hist(inc_track->get_tracts(),tract_count_inc);
    cal_hist(pos_null_corr_track->get_tracts(),tract_count_inc_null);
    calculate_FDR();

    // output distribution values
    {
        std::ofstream out((output_file_name+".fdr_dist.values.txt").c_str());
        out << "voxel_dis\tfdr_inc\tfdr_dec\t#track_inc_null\t#track_dec_null\t#track_inc\t#track_dec" << std::endl;
        for(size_t index = length_threshold_voxels;index < fdr_inc.size()-1;++index)
        {
            out << index
                << "\t" << fdr_inc[index]
                << "\t" << fdr_dec[index]
                << "\t" << tract_count_inc_null[index]
                << "\t" << tract_count_dec_null[index]
                << "\t" << tract_count_inc[index]
                << "\t" << tract_count_dec[index] << std::endl;
        }
    }

    if(fdr_threshold != 0.0f)
    {
        bool has_result = false;
        for(size_t length = length_threshold_voxels;length < fdr_inc.size();++length)
            if(fdr_inc[length] <= fdr_threshold)
            {
                inc_track->delete_by_length(length);
                inc_track->clear_deleted();
                has_result = true;
                break;
            }
        if(!has_result)
            inc_track->clear();
        has_result = false;
        for(size_t length = length_threshold_voxels;length < fdr_dec.size();++length)
            if(fdr_dec[length] <= fdr_threshold)
            {
                dec_track->delete_by_length(length);
                dec_track->clear_deleted();
                has_result = true;
                break;
            }
        if(!has_result)
            dec_track->clear();
    }

    {
        tipl::progress prog("deleting repeated tracts");
        inc_track->delete_repeated(1.0);
        dec_track->delete_repeated(1.0);
    }
    {
        tipl::progress prog("saving correlational tractography");
        if(inc_track->get_visible_track_count())
            inc_track->save_tracts_to_file((output_file_name+".inc.tt.gz").c_str());
        else
            std::ofstream((output_file_name+".inc.no_tract.txt").c_str());

        if(dec_track->get_visible_track_count())
            dec_track->save_tracts_to_file((output_file_name+".dec.tt.gz").c_str());
        else
            std::ofstream((output_file_name+".dec.no_tract.txt").c_str());
    }

    {
        tipl::progress p2("save statistics.fz");
        tipl::io::gz_mat_write mat_write(output_file_name+".t_statistics.fz");
        if(!mat_write)
            return;
        save_fz(handle->mat_reader,mat_write,{"odf_faces","odf_vertices","z0","mapping","report","steps"},{"subject"});
        for(unsigned int i = 0;i < spm_map->inc_ptr.size();++i)
        {
            mat_write.write<tipl::io::masked_sloped>(std::string("inc_t")+std::to_string(i),spm_map->inc_ptr[i],handle->dim.plane_size(),handle->dim.depth());
            mat_write.write<tipl::io::masked_sloped>(std::string("dec_t")+std::to_string(i),spm_map->dec_ptr[i],handle->dim.plane_size(),handle->dim.depth());
        }
    }
}
void group_connectometry_analysis::wait(size_t index)
{
    for(;index < threads.size();++index)
        if(threads[index].joinable())
            threads[index].join();
}

void group_connectometry_analysis::clear(void)
{
    if(!threads.empty())
    {
        while(terminated)
            std::this_thread::yield();
        terminated = true;
        wait();
        threads.clear();
        terminated = false;
    }
}

std::string iterate_items(const std::vector<std::string>& item)
{
    std::string result;
    for(unsigned int index = 0;index < item.size();++index)
    {
        if(index)
        {
            if(item.size() > 2)
                result += ",";
            result += " ";
        }
        if(item.size() >= 2 && index+1 == item.size())
            result += "and ";
        result += item[index];
    }
    return result;
}
std::string group_connectometry_analysis::get_file_post_fix(void)
{
    std::string postfix = handle->db.index_name + ".";
    {
        if(foi_str.find("increased") == 0 || foi_str.find("decreased") == 0)
            postfix += foi_str.substr(10);
        else
            postfix += foi_str;
        postfix += ".r";
        postfix += std::to_string((int)(rho_threshold*10));
    }

    if(fdr_threshold == 0.0f)
    {
        postfix += ".l";
        postfix += std::to_string(length_threshold_voxels);
    }
    else
    {
        postfix += ".fdr";
        std::string value = std::to_string(fdr_threshold);
        value.resize(4);
        postfix += value;
    }
    return postfix;
}

bool can_be_normalized_by_iso(const std::string& name)
{
    return name == "qa" || name == "rdi" || tipl::begins_with(name,"nrdi");
}
void normalize_data_by_iso(const float* iso_ptr,float* out_data_ptr,size_t n)
{
    std::vector<float> iso(iso_ptr,iso_ptr+n),inv_iso;
    for(auto& each : iso)
        if(each != 0.0f)
            inv_iso.push_back(each = 1.0f/each);
    tipl::upper_threshold(iso.begin(),iso.end(),tipl::outlier_range(inv_iso.begin(),inv_iso.end()).second);
    tipl::multiply(out_data_ptr,out_data_ptr+n,iso.begin());
}

void group_connectometry_analysis::calculate_adjusted_qa(stat_model& info)
{
    if(!info.X.empty())
    {
        bool has_partial_correlation = false;
        std::ostringstream out;
        for(size_t i = 1;i < info.variables.size();++i) // skip intercept at i = 0
            if(i != info.study_feature)
            {
                has_partial_correlation = true;
                out << info.variables[i] << " ";
            }
        if(has_partial_correlation)
            tipl::out() << "adjusting " << handle->db.index_name << " using partial correlation of " << out.str();
    }

    subject_data.resize(info.selected_subject.size());
    for(size_t index = 0;index < info.selected_subject.size();++index)
        subject_data[index] = handle->db.subject_indices[info.selected_subject[index]];

    if(!handle->db.is_longitudinal && normalize_iso &&
        tipl::contains(handle->db.index_list,"iso"))
    {
        tipl::out() << "normalize "+handle->db.index_name + " by iso";
        size_t n = handle->db.mask_size;
        auto old_index = handle->db.index_name;
        handle->db.set_current_index("iso");
        subject_data_buffer.resize(subject_data.size()*n);
        tipl::par_for(subject_data.size(),[&](size_t index)
        {
            auto data_ptr = subject_data_buffer.data() + index*n;
            std::copy_n(subject_data[index],n,data_ptr);

            normalize_data_by_iso(handle->db.subject_indices[info.selected_subject[index]],data_ptr,n);

            subject_data[index] = data_ptr;
        });
        handle->db.set_current_index(old_index);
    }
    // population_value_adjusted is a transpose of handle->db.subject_indices
    population_value_adjusted.clear();
    population_value_adjusted.resize(handle->db.mask_size);
    const auto& si2vi = handle->mat_reader.si2vi;
    const auto& dir = handle->dir;
    tipl::par_for(si2vi.size(),[&](size_t si)
    {
        if(dir.fa[0][si2vi[si]] < fiber_threshold)
            return;
        std::vector<float> population(subject_data.size());
        for(unsigned int index = 0;index < subject_data.size();++index)
            population[index] = subject_data[index][si];
        info.partial_correlation(population);
        population_value_adjusted[si] = std::move(population);
    });
}

void group_connectometry_analysis::calculate_spm(connectometry_result& data,stat_model& info)
{
    data.clear_result(handle->dir.num_fiber,handle->dim.size());
    const auto& si2vi = handle->mat_reader.si2vi;
    for(size_t s_index = 0;s_index < si2vi.size() && !terminated;++s_index)
    {
        size_t pos = si2vi[s_index];
        double T_stat(0.0); // declare here so that the T-stat of the 1st fiber can be applied to others if there is only one metric per voxel
        for(size_t fib = 0,cur_s_index = s_index;
            fib < handle->dir.num_fiber && handle->dir.fa[fib][pos] > fiber_threshold;
            ++fib,cur_s_index += si2vi.size())
        {
            // some connectometry database only have 1 metrics per voxel
            // and thus the computed statistics will be applied to all fibers
            if(cur_s_index < population_value_adjusted.size())
            {
                if(population_value_adjusted[cur_s_index][0] == 0.0f)
                    continue;
                T_stat = info(population_value_adjusted[cur_s_index]);
            }

            if(T_stat > 0.0)
                data.inc[fib][pos] = T_stat;
            if(T_stat < 0.0)
                data.dec[fib][pos] = -T_stat;
        }
    }
}

void group_connectometry_analysis::run_permutation(unsigned int thread_count,unsigned int permutation_count)
{
    tipl::out() << "run permutation test";
    clear();

    {
        auto index_name = QString(handle->db.index_name.c_str()).toUpper().toStdString();
        // if study intercept (changes)
        if(model->study_feature == 0)
        {
            // if db has filtered longitudinal changes to study decreased values
            if(handle->db.longitudinal_filter_type == 2)
            {
                hypothesis_inc = std::string("decreased ") + index_name;
                hypothesis_dec = std::string("increased ") + index_name;
            }
            else
            // default
            {
                hypothesis_inc = std::string("increased ") + index_name;
                hypothesis_dec = std::string("decreased ") + index_name;
            }
        }
        else
        {
            hypothesis_inc = index_name;
            hypothesis_dec = index_name;
            if(model->variables_is_categorical[model->study_feature])
            {
                hypothesis_inc += std::string(" in ")+foi_str+" "+std::to_string(model->variables_max[model->study_feature]);
                hypothesis_dec += std::string(" in ")+foi_str+" "+std::to_string(model->variables_min[model->study_feature]);
            }
            else
            {
                if(foi_str.find("increase") == 0)
                {
                    hypothesis_inc += std::string(" associated with ")+ foi_str;
                    hypothesis_dec += std::string(" associated with de")+ foi_str.substr(2);
                }
                else
                if(foi_str.find("decrease") == 0)
                {
                    hypothesis_inc += std::string(" associated with in")+ foi_str.substr(2);
                    hypothesis_dec += std::string(" associated with ")+ foi_str;
                }
                else
                {
                    hypothesis_inc += std::string(" associated with higher ")+ foi_str;
                    hypothesis_dec += std::string(" associated with lower ")+ foi_str;
                }
            }

            std::string prefix = "higher ";
            if(handle->db.longitudinal_filter_type == 1) // db filter to only positive values
                prefix = "increased ";
            if(handle->db.longitudinal_filter_type == 2) // db filter to only negative values
                prefix = "decreased ";

            hypothesis_inc = prefix + hypothesis_inc;
            hypothesis_dec = prefix + hypothesis_dec;
        }
    }
    // output report
    {
        std::ostringstream out;

        out << "\nCorrelational tractography (Yeh, et al. Neuroimage 245 (2021): 118651) was derived to visualize pathways that have ";
        if(handle->db.is_longitudinal)
            out << "a longitudinal change of ";
        out << handle->db.index_name;
        if(model->study_feature)
            out << " correlated with " << foi_str;
        out << ".";

        if(normalize_iso && tipl::contains(handle->db.index_list,"iso") &&
           (can_be_normalized_by_iso(handle->db.index_name)))
            out << " The " << handle->db.index_name << " values were normalized by the isotropic diffusion.";
        else
            normalize_iso = false;

        {
            auto items = model->variables;
            items.erase(items.begin()); // remove longitudinal change
            {
                if(model->study_feature)
                    items.erase(items.begin() + model->study_feature-1);
                out << " A nonparametric Spearman" << (items.empty() ? " ":" partial ") << "correlation was used to derive the correlation";
                if(items.empty())
                    out << ".";
                else
                    out << ", and the effect of " << iterate_items(items) << " was removed using a multiple regression model.";
            }
        }
        out << " The statistical significance of the correlation was examined using a permutation test (Yeh et al. NeuroImage 125 (2016): 162-171).";

        // report subject cohort
        out << model->cohort_report;
        out << " A total of " << model->selected_subject.size() << " subjects were included in the analysis.";

        // report other parameters
        if(rho_threshold == 0.1f || rho_threshold == 0.3f || rho_threshold == 0.5f)
            out << " An effect size of " << rho_threshold << " (T statistics = " << t_threshold;
        else
            out << " A T-score threshold of " << t_threshold << " (effect size = " << rho_threshold;
        out << ") was used in the fiber tracking algorithm (Yeh et al. PLoS ONE 8(11): e80713, 2013).";

        if(!roi_mgr->report.empty())
            out << roi_mgr->report << std::endl;
        if(tip_iteration)
            out << " The tracks were filtered by topology-informed pruning (Yeh et al. Neurotherapeutics, 16(1), 52-58, 2019) with "
                << tip_iteration << " iteration(s).";
        if(fdr_threshold == 0.0f)
            out << " A length threshold of " << length_threshold_voxels << " voxel distance was used to select tracks.";
        else
            out << " An FDR threshold of " << fdr_threshold << " was used to select tracks.";

        out << " To estimate the false discovery rate, a total of "
            << permutation_count
            << " randomized permutations were applied to the group label to obtain the null distribution of the track length.";
        report = out.str().c_str();
    }

    // setup output file name
    if(output_file_name.empty())
        output_file_name = get_file_post_fix();
    tipl::out() << "output file prefix: " << output_file_name << std::endl;

    size_t max_dimension = tipl::max_value(handle->dim.begin(),handle->dim.end())*2;

    tract_count_inc_null.clear();
    tract_count_inc_null.resize(max_dimension);
    tract_count_dec_null.clear();
    tract_count_dec_null.resize(max_dimension);
    tract_count_inc.clear();
    tract_count_inc.resize(max_dimension);
    tract_count_dec.clear();
    tract_count_dec.resize(max_dimension);
    fdr_inc.clear();
    fdr_inc.resize(max_dimension);
    fdr_dec.clear();
    fdr_dec.resize(max_dimension);

    inc_track = std::make_shared<TractModel>(handle);
    dec_track = std::make_shared<TractModel>(handle);
    pos_null_corr_track = std::make_shared<TractModel>(handle);
    neg_null_corr_track = std::make_shared<TractModel>(handle);
    spm_map = std::make_shared<connectometry_result>();

    terminated = false;
    prog = 0;
    // preliminary run
    {
        std::shared_ptr<tracking_data> fib(new tracking_data);
        fib->read(handle);

        calculate_adjusted_qa(*model.get());

        stat_model info;
        info.resample(*model.get(),false,false,0);
        tipl::out() << "preliminary run to determine seed count" << std::endl;
        calculate_spm(*spm_map.get(),info);
        preprocess = 0;
        seed_count = 1000;

        const size_t expected_tract_count = 50000;
        auto expected_tract_per_permutation = expected_tract_count/permutation_count;
        while(seed_count < 128000)
        {
            std::vector<std::vector<float> > tracks;
            fib->dt_fa = spm_map->dec_ptr;
            run_track(fib,tracks,seed_count,0,tipl::max_thread_count);
            fib->dt_fa = spm_map->inc_ptr;
            run_track(fib,tracks,seed_count,0,tipl::max_thread_count);
            if(tracks.size() > expected_tract_per_permutation)
                break;
            seed_count *= 2;
        }
        tipl::out() << "seed count: " << seed_count << std::endl;
    }

    for(unsigned int index = 0;index < thread_count;++index)
        threads.push_back(std::thread([=](){run_permutation_multithread(index,thread_count,permutation_count);}));
}

void group_connectometry_analysis::calculate_FDR(void)
{
    double sum_inc_null = 0.0;
    double sum_dec_null = 0.0;
    double sum_inc = 0.0;
    double sum_dec = 0.0;
    for(int index = int(tract_count_inc_null.size())-1;index >= 0;--index)
    {
        if(sum_inc_null == 0.0 && (tract_count_inc_null[size_t(index)] != 0 || index == 0) && sum_inc > 0.0)
        {
            float tail_fdr = 1.0f/float(sum_inc);
            for(int j = int(tract_count_inc.size())-1;j >= index;--j)
                if(tract_count_inc[size_t(j)])
                {
                    for(int k = index;k <= j;++k)
                        fdr_inc[size_t(k)] = tail_fdr;
                    break;
                }
        }
        if(sum_dec_null == 0.0 && (tract_count_dec_null[size_t(index)] != 0 || index == 0) && sum_dec > 0.0)
        {
            float tail_fdr = 1.0f/float(sum_dec);
            for(int j = int(tract_count_dec.size())-1;j >= index;--j)
                if(tract_count_dec[size_t(j)])
                {
                    for(int k = index;k <= j;++k)
                        fdr_dec[size_t(k)] = tail_fdr;
                    break;
                }
        }
        sum_inc_null += tract_count_inc_null[size_t(index)];
        sum_dec_null += tract_count_dec_null[size_t(index)];
        sum_inc += tract_count_inc[size_t(index)];
        sum_dec += tract_count_dec[size_t(index)];
        fdr_inc[size_t(index)] = float((sum_inc > 0.0) ? std::min<double>(1.0,sum_inc_null/sum_inc): 1.0);
        fdr_dec[size_t(index)] = float((sum_dec > 0.0) ? std::min<double>(1.0,sum_dec_null/sum_dec): 1.0);
    }
    if(tipl::min_value(fdr_inc) < 0.05f)
        std::replace(fdr_inc.begin(),fdr_inc.end(),1.0f,0.0f);
    if(tipl::min_value(fdr_dec) < 0.05f)
        std::replace(fdr_dec.begin(),fdr_dec.end(),1.0f,0.0f);
}

void group_connectometry_analysis::generate_report(std::string& output)
{
    std::ostringstream html_report((output_file_name+".report.html").c_str());
    html_report << "<!DOCTYPE html>" << std::endl;
    html_report << "<html><head><title>Report</title></head>" << std::endl;
    html_report << "<body>" << std::endl;
    if(!handle->report.empty())
    {
        html_report << "<h2>MRI Acquisition</h2>" << std::endl;
        html_report << "<p>" << handle->report << "</p>" << std::endl;
    }
    if(!report.empty())
    {
        html_report << "<h2>Correlational Tractography Analysis</h2>" << std::endl;
        html_report << "<p>" << report.c_str() << "</p>" << std::endl;
    }

    std::string fdr_result_pos,fdr_result_neg;
    auto output_fdr = [](float fdr)
    {
        std::string str("(FDR = ");
        str += std::to_string(fdr);
        str += ")";
        return str;
    };
    if(fdr_threshold != 0.0f) // fdr control
    {
        fdr_result_pos = "(FDR ≤ ";
        fdr_result_pos += std::to_string(fdr_threshold);
        fdr_result_pos += ")";
        fdr_result_neg = fdr_result_pos;
    }
    else
    {
        fdr_result_pos = output_fdr(fdr_inc[length_threshold_voxels]);
        fdr_result_neg = output_fdr(fdr_dec[length_threshold_voxels]);
    }


    auto output_track_image = [&](std::string name,std::string hypo,std::string fdr)
    {
        if(fdr.empty())
            return;
        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+"."+name+"_map.jpg").filename().u8string() << "\" width=\"600\"/>" << std::endl;
        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+"."+name+"_map2.jpg").filename().u8string() << "\" width=\"600\"/>" << std::endl;
        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+"."+name+".jpg").filename().u8string() << "\" width=\"1200\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Tracks with " << hypo << " " << fdr << "</p>" << std::endl;
    };
    auto report_fdr = [&](bool sig,std::string hypo,std::string result){
        html_report << "<p>";
        if(sig)
            html_report << " The connectometry analysis found tracts showing ";
        else
            html_report << " The connectometry analysis did not find tracts showing significant ";
        html_report << hypo << " " << result << ".</p>" << std::endl;
    };
    auto has_inc_finding = [&](void){
        if(!inc_track->get_visible_track_count())
            return false;
        if(fdr_threshold == 0.0f)
            return fdr_inc[length_threshold_voxels]<= 0.2f;
        for(size_t i = length_threshold_voxels;i < fdr_inc.size();++i)
                if(fdr_inc[i] < fdr_threshold)
                    return true;
        return false;
    };
    auto has_dec_finding = [&](void){
        if(!dec_track->get_visible_track_count())
            return false;
        if(fdr_threshold == 0.0f)
            return fdr_dec[length_threshold_voxels]<=0.2f;
        for(size_t i = length_threshold_voxels;i < fdr_dec.size();++i)
                if(fdr_dec[i] < fdr_threshold)
                    return true;
        return false;
    };

    html_report << "<h2>Results</h2>" << std::endl;
    html_report << "<h3>Tracks with " << hypothesis_inc << "</h3>" << std::endl;


    if(prog == 100)
        output_track_image("inc",hypothesis_inc,fdr_result_pos);
    report_fdr(has_inc_finding(),hypothesis_inc,fdr_result_pos);

    html_report << "<h3>Tracks with " << hypothesis_dec << "</h3>" << std::endl;

    if(prog == 100)
        output_track_image("dec",hypothesis_dec,fdr_result_neg);
    report_fdr(has_dec_finding(),hypothesis_dec,fdr_result_neg);

    if(prog == 100)
    {
        if(fdr_inc[length_threshold_voxels] < 0.2f || fdr_dec[length_threshold_voxels] < 0.2f)
        {
            html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+".pos_neg.jpg").filename().u8string() << "\" width=\"1200\"/>" << std::endl;
            html_report << "<p><b>Fig.</b> Correlational tractography showing " << hypothesis_inc << " (red)" << fdr_result_pos
                    << " and " << hypothesis_dec << " (blue)" << fdr_result_neg << ".</p>" << std::endl;
        }

        std::string permutation_explained =
    " The permutation was applied to subject labels to test results against permuted condition.\
     The histogram under permuted condition represents the result under the null hypothesis.\
     This null result is then used to test the histogram under nonpermuted condition to compute the FDR.\
     A smaller difference between histograms suggests that the study finding is similar to null finding and having a lower significance,\
     whereas a larger difference suggests greater significance of the study finding.";

        html_report << "<h3>False discovery rate analysis</h3>" << std::endl;

        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+".inc.dist.jpg").filename().u8string() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Permutation test showing the histograms of track counts with "<< hypothesis_inc << ".</p>";


        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+".dec.dist.jpg").filename().u8string() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Permutation test showing the histograms of track counts with "<< hypothesis_dec << ".</p>";

        html_report << permutation_explained << std::endl;
        html_report << "<p></p><img src = \""<< std::filesystem::path(output_file_name+".fdr.jpg").filename().u8string() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> The False discovery rate (FDR) at different track length </p>";
    }

    html_report << "</body></html>" << std::endl;
    output = html_report.str();


    if(prog == 100 && !no_tractogram)
    {
        tipl::progress p3("create tract figures");
        std::shared_ptr<fib_data> new_data(new fib_data);
        *(new_data.get()) = *(handle);
        std::shared_ptr<tracking_window> new_mdi(new tracking_window(nullptr,new_data));
        new_mdi->set_memorize_parameters(false);
        new_mdi->setAttribute(Qt::WA_DeleteOnClose);
        new_mdi->setWindowTitle(output_file_name.c_str());
        new_mdi->show();
        new_mdi->resize(2000,1000);
        new_mdi->update();
        for(auto const& [key, value] : {
            std::pair{"show_surface", "1"},
            std::pair{"show_slice", "0"},
            std::pair{"show_region", "0"},
            std::pair{"bkg_color", "16777215"},
            std::pair{"surface_alpha", "0.2"}
        })
            new_mdi->set_data(key, value);

        new_mdi->command({"add_surface","0","25"});

        for(auto const& [cmd1, cmd2] : {
            std::pair{"set_slice_by_name","t1w_template"},
            std::pair{"set_slice_contrast","0.0 400.0"},
            std::pair{"set_zoom","1.0"}
        })
            new_mdi->command({cmd1,cmd2});



        auto show_track_result = [&](std::shared_ptr<TractModel> track,std::string name,unsigned int color){
            if(track->get_visible_track_count())
            {
                new_mdi->tractWidget->addNewTracts(name.c_str());
                new_mdi->tractWidget->tract_models[0]->add(*track.get());
                new_mdi->set_data("tract_color_style","0");
                new_mdi->command({"update_tract"});

                std::vector<tipl::vector<3,short> > points;
                new_mdi->tractWidget->tract_models[0]->to_voxel(points,new_mdi->current_slice->to_slice);
                new_mdi->regionWidget->add_region(name.c_str(),0,color);
                new_mdi->regionWidget->regions.back()->add_points(std::move(points));

            }
            new_mdi->command({"save_h3view_screen",output_file_name+"." + name + ".jpg"});
            // do it twice to eliminate 3D artifact
            new_mdi->command({"save_h3view_screen",output_file_name+"." + name + ".jpg"});

            for(auto const& [key, value] : {
                std::pair{"roi_zoom", "8"},
                std::pair{"roi_layout", "5"},
                std::pair{"roi_track", "0"},
                std::pair{"roi_draw_edge", "0"}
            })
                new_mdi->set_data(key, value);

            for(auto const& [cmd, arg] : std::initializer_list<std::pair<std::string, std::string>>{
                std::pair{"save_roi_screen", output_file_name + "." + name + "_map.jpg"},
                std::pair{"set_roi_view", "1"},
                std::pair{"save_roi_screen", output_file_name + "." + name + "_map2.jpg"},
                std::pair{"set_roi_view", "2"},
                std::pair{"delete_all_tracts", ""},
                std::pair{"delete_all_regions", ""}
            })
                new_mdi->command({cmd, arg});

        };
        show_track_result(inc_track,"inc",0x00F01010);
        show_track_result(dec_track,"dec",0x001010F0);

        if(has_inc_finding() || has_dec_finding())
        {
            if(has_inc_finding())
            {
                new_mdi->tractWidget->addNewTracts("inc_track");
                new_mdi->tractWidget->tract_models.back()->add(*inc_track.get());
            }
            if(has_dec_finding())
            {
                new_mdi->tractWidget->addNewTracts("dec_track");
                new_mdi->tractWidget->tract_models.back()->add(*dec_track.get());
            }
            new_mdi->set_data("tract_color_style","1");
            new_mdi->command({"update_tract"});
            new_mdi->command({"save_h3view_screen",output_file_name+".pos_neg.jpg"});
            // do it twice to eliminate 3D artifact
            new_mdi->command({"save_h3view_screen",output_file_name+".pos_neg.jpg"});
        }
        new_mdi->deleteLater();
    }
}

