#include <QFileInfo>
#include <ctime>
#include "connectometry/group_connectometry_analysis.h"
#include "fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "tracking/tracking_window.h"

group_connectometry_analysis::group_connectometry_analysis():handle(0),normalize_qa(true)
{

}

bool group_connectometry_analysis::create_database(const char* template_name)
{
    handle.reset(new fib_data);
    if(!handle->load_from_file(template_name))
    {
        error_msg = handle->error_msg;
        return false;
    }
    fiber_threshold = 0.6*tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    handle->db.calculate_si2vi();
    return true;
}
bool group_connectometry_analysis::load_database(const char* database_name)
{
    handle.reset(new fib_data);
    if(!handle->load_from_file(database_name))
    {
        error_msg = "Invalid fib file:";
        error_msg += handle->error_msg;
        return false;
    }
    fiber_threshold = 0.6*tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    return handle->db.has_db();
}


int group_connectometry_analysis::run_track(const tracking_data& fib,std::vector<std::vector<float> >& tracks,int seed_count, unsigned int thread_count)
{
    ThreadData tracking_thread(handle.get());
    tracking_thread.param.threshold = tracking_threshold;
    tracking_thread.param.cull_cos_angle = 1.0f;
    tracking_thread.param.step_size = handle->vs[0];
    tracking_thread.param.smooth_fraction = 0;
    tracking_thread.param.min_length = 0;
    tracking_thread.param.max_length = 2.0*std::max<int>(fib.dim[0],std::max<int>(fib.dim[1],fib.dim[2]))*handle->vs[0];
    tracking_thread.param.tracking_method = 0;// streamline fiber tracking
    tracking_thread.param.initial_direction = 0;// main directions
    tracking_thread.param.interpolation_strategy = 0; // trilinear interpolation
    tracking_thread.param.stop_by_tract = 0;// stop by seed
    tracking_thread.param.center_seed = 0;// subvoxel seeding
    tracking_thread.param.random_seed = 0;
    tracking_thread.param.termination_count = seed_count;
    tracking_thread.roi_mgr = roi_mgr;
    tracking_thread.run(fib,thread_count,true);
    tracking_thread.track_buffer.swap(tracks);

    if(track_trimming)
    {
        TractModel t(handle.get());
        t.add_tracts(tracks);
        for(int i = 0;i < track_trimming && t.get_visible_track_count();++i)
            t.trim();
        tracks.swap(t.get_tracts());
    }
    return tracks.size();
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

void group_connectometry_analysis::run_permutation_multithread(unsigned int id,unsigned int thread_count,unsigned int permutation_count)
{
    connectometry_result data;
    tracking_data fib;
    fib.read(*handle);
    std::vector<std::vector<float> > tracks;
    const int max_visible_track = 1000000;
    {
        bool null = true;
        for(int i = id;i < permutation_count && !terminated;)
        {

            stat_model info;
            info.resample(*model.get(),null,true);
            calculate_spm(data,info,normalize_qa);

            fib.fa = data.neg_corr_ptr;
            unsigned int s = run_track(fib,tracks,seed_count);
            if(null)
                seed_neg_corr_null[i] = s;
            else
                seed_neg_corr[i] = s;

            cal_hist(tracks,(null) ? subject_neg_corr_null : subject_neg_corr);

            if(output_resampling && !null)
            {
                std::lock_guard<std::mutex> lock(lock_neg_corr_tracks);
                if(tracks.size() > max_visible_track/permutation_count)
                    tracks.resize(max_visible_track/permutation_count);
                neg_corr_track->add_tracts(tracks,length_threshold);
                if(id == 1)
                {
                    neg_corr_track->delete_repeated(1.0f);
                    neg_corr_track->clear_deleted();
                }
                tracks.clear();
            }

            info.resample(*model.get(),null,true);
            calculate_spm(data,info,normalize_qa);
            fib.fa = data.pos_corr_ptr;
            s = run_track(fib,tracks,seed_count);
            if(null)
                seed_pos_corr_null[i] = s;
            else
                seed_pos_corr[i] = s;
            cal_hist(tracks,(null) ? subject_pos_corr_null : subject_pos_corr);

            if(output_resampling && !null)
            {
                std::lock_guard<std::mutex> lock(lock_pos_corr_tracks);
                if(tracks.size() > max_visible_track/permutation_count)
                    tracks.resize(max_visible_track/permutation_count);
                pos_corr_track->add_tracts(tracks,length_threshold);
                if(id == 1)
                {
                    pos_corr_track->delete_repeated(1.0f);
                    pos_corr_track->clear_deleted();
                }
                tracks.clear();
            }

            if(!null)
            {
                i += thread_count;
                if(id == 0)
                    progress = i*100/permutation_count;
            }
            null = !null;
        }
        if(id == 0)
        {
            stat_model info;
            info.resample(*model.get(),false,false);
            calculate_spm(*spm_map.get(),info,normalize_qa);

            if(terminated)
                return;
            if(!output_resampling)
            {
                fib.fa = spm_map->neg_corr_ptr;
                run_track(fib,tracks,seed_count*permutation_count,threads.size());
                if(tracks.size() > max_visible_track)
                    tracks.resize(max_visible_track);
                neg_corr_track->add_tracts(tracks,length_threshold);
                fib.fa = spm_map->pos_corr_ptr;
                run_track(fib,tracks,seed_count*permutation_count,threads.size());
                if(tracks.size() > max_visible_track)
                    tracks.resize(max_visible_track);
                pos_corr_track->add_tracts(tracks,length_threshold);
            }
        }
    }
    if(id == 0 && !terminated)
        progress = 100;
}
void group_connectometry_analysis::clear(void)
{
    if(!threads.empty())
    {
        terminated = true;
        wait();
        threads.clear();
        terminated = false;
    }
}
void group_connectometry_analysis::wait(void)
{
    for(int i = 0;i < threads.size();++i)
        threads[i]->wait();
}


void group_connectometry_analysis::save_tracks_files(void)
{
    for(int i = 0;i < threads.size();++i)
        threads[i]->wait();
    has_pos_corr_result = false;
    has_neg_corr_result = false;
    {
        if(pos_corr_track->get_visible_track_count())
        {
            if(fdr_threshold != 0.0)
            {
                fdr_pos_corr.back() = 0.0f;
                for(int length = 10;length < fdr_pos_corr.size();++length)
                    if(fdr_pos_corr[length] < fdr_threshold)
                    {
                        pos_corr_track->delete_by_length(length);
                        break;
                    }
            }
            pos_corr_track->delete_repeated(1.0);
            std::ostringstream out1;
            out1 << output_file_name << ".pos_corr.trk.gz";
            pos_corr_track->save_tracts_to_file(out1.str().c_str());
            pos_corr_tracks_result = "";
            handle->recognize_report(pos_corr_track,pos_corr_tracks_result);
            if(pos_corr_tracks_result.empty())
                pos_corr_tracks_result = "tracks";
            has_pos_corr_result = true;
        }
        else
        {
            std::ostringstream out1;
            out1 << output_file_name << ".pos_corr.no_trk.txt";
            std::ofstream(out1.str().c_str());
        }


        if(neg_corr_track->get_visible_track_count())
        {
            if(fdr_threshold != 0.0)
            {
                fdr_neg_corr.back() = 0.0f;
                for(int length = 10;length < fdr_neg_corr.size();++length)
                    if(fdr_neg_corr[length] < fdr_threshold)
                    {
                        neg_corr_track->delete_by_length(length);
                        break;
                    }
            }
            neg_corr_track->delete_repeated(1.0);
            std::ostringstream out1;
            out1 << output_file_name << ".neg_corr.trk.gz";
            neg_corr_track->save_tracts_to_file(out1.str().c_str());
            neg_corr_tracks_result = "";
            handle->recognize_report(neg_corr_track,neg_corr_tracks_result);
            if(neg_corr_tracks_result.empty())
                neg_corr_tracks_result = "tracks";
            has_neg_corr_result = true;
        }
        else
        {
            std::ostringstream out1;
            out1 << output_file_name << ".neg_corr.no_trk.txt";
            std::ofstream(out1.str().c_str());
        }


        {
            std::ostringstream out1;
            out1 << output_file_name << ".t_statistics.fib.gz";
            gz_mat_write mat_write(out1.str().c_str());
            for(unsigned int i = 0;i < handle->mat_reader.size();++i)
            {
                std::string name = handle->mat_reader.name(i);
                if(name == "dimension" || name == "voxel_size" ||
                        name == "odf_vertices" || name == "odf_faces" || name == "trans")
                    mat_write.write(handle->mat_reader[i]);
                if(name == "fa0")
                    mat_write.write("qa_map",handle->dir.fa[0],handle->dim.plane_size(),handle->dim.depth());
            }
            for(unsigned int i = 0;i < spm_map->pos_corr_ptr.size();++i)
            {
                std::ostringstream out1,out2,out3,out4;
                out1 << "fa" << i;
                out2 << "index" << i;
                out3 << "inc_t" << i;
                out4 << "dec_t" << i;
                mat_write.write(out1.str().c_str(),handle->dir.fa[i],1,handle->dim.size());
                mat_write.write(out2.str().c_str(),handle->dir.findex[i],1,handle->dim.size());
                mat_write.write(out3.str().c_str(),spm_map->pos_corr_ptr[i],1,handle->dim.size());
                mat_write.write(out4.str().c_str(),spm_map->neg_corr_ptr[i],1,handle->dim.size());
            }
        }

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
void group_connectometry_analysis::run_permutation(unsigned int thread_count,unsigned int permutation_count)
{
    clear();
    // output report
    {
        std::ostringstream out;

        out << "\nDiffusion MRI connectometry (Yeh et al. NeuroImage 125 (2016): 162-171) was used to derive the correlation tractography that has ";
        if(handle->db.is_longitudinal)
            out << "a longitudinal change of ";
        out << handle->db.index_name;

        if(foi_str == "Intercept")
            out << ".";
        else
            out << " correlated with " << foi_str << ".";

        if(model->type == 1) // regression model
        {
            auto items = model->variables;
            items.erase(items.begin()); // remove intercept
            if(model->nonparametric)
            {
                items.erase(items.begin() + model->study_feature-1);
                out << " A nonparametric Spearman" << (items.empty() ? " ":" partial ") << "correlation was used to derive the correlation";
                if(items.empty())
                    out << ".";
                else
                    out << ", and the effect of " << iterate_items(items) << " was removed using a multiple regression model.";
            }
            else
                out << " A multiple regression model was used to consider the effect of " << iterate_items(items) << ".";
        }

        // report subject cohort
        out << model->cohort_text;
        out << " A total of " << model->subject_index.size() << " subjects were included in the analysis.";

        // report other parameters
        out << " A T-score threshold of " << tracking_threshold;
        out << " was assigned and tracked using a deterministic fiber tracking algorithm (Yeh et al. PLoS ONE 8(11): e80713, 2013) to obtain correlation tractography.";

        if(normalize_qa)
            out << " The QA values were normalized.";
        if(track_trimming)
            out << " Topology-informed pruning (Yeh et al. Neurotherapeutics, 16(1), 52-58, 2019) was conducted with " << track_trimming << " iterations to remove false connections.";

        if(output_resampling)
            out << " All tracks generated from bootstrap resampling were included.";

        if(fdr_threshold == 0.0f)
            out << " A length threshold of " << length_threshold << " voxel distance was used to select tracks.";
        else
            out << " An FDR threshold of " << fdr_threshold << " was used to select tracks.";
        out << " The seeding number for each permutation was " << seed_count << ".";

        out << " To estimate the false discovery rate, a total of "
            << permutation_count
            << " randomized permutations were applied to the group label to obtain the null distribution of the track length.";
        if(!roi_mgr_text.empty())
            out << roi_mgr_text << std::endl;
        report = out.str().c_str();
    }

    // setup output file name
    {
        if(model->type == 1) // run regression model
        {
            output_file_name += ".t";
            output_file_name += std::to_string((int)tracking_threshold);
            output_file_name += ".";
            output_file_name += foi_str;
        }
        if(model->type == 3) // longitudinal change
        {
            output_file_name += ".sd";
            output_file_name += std::to_string((int)tracking_threshold);
        }

        if(normalize_qa)
            output_file_name += ".nqa";
        if(fdr_threshold == 0)
        {
            output_file_name += ".length";
            output_file_name += std::to_string((int)length_threshold);
        }
        else
        {
            output_file_name += ".fdr";
            output_file_name += std::to_string(fdr_threshold);
        }
        output_file_name += output_roi_suffix;
    }

    size_t max_dimension = *std::max_element(handle->dim.begin(),handle->dim.end());

    terminated = false;
    subject_pos_corr_null.clear();
    subject_pos_corr_null.resize(max_dimension);
    subject_neg_corr_null.clear();
    subject_neg_corr_null.resize(max_dimension);
    subject_pos_corr.clear();
    subject_pos_corr.resize(max_dimension);
    subject_neg_corr.clear();
    subject_neg_corr.resize(max_dimension);
    fdr_pos_corr.clear();
    fdr_pos_corr.resize(max_dimension);
    fdr_neg_corr.clear();
    fdr_neg_corr.resize(max_dimension);

    seed_pos_corr_null.clear();
    seed_pos_corr_null.resize(permutation_count);
    seed_neg_corr_null.clear();
    seed_neg_corr_null.resize(permutation_count);
    seed_pos_corr.clear();
    seed_pos_corr.resize(permutation_count);
    seed_neg_corr.clear();
    seed_neg_corr.resize(permutation_count);

    model->rand_gen.reset();
    std::srand(0);

    has_pos_corr_result = true;
    has_neg_corr_result = true;
    pos_corr_tracks_result = "tracks";
    neg_corr_tracks_result = "tracks";

    pos_corr_track = std::make_shared<TractModel>(handle.get());
    neg_corr_track = std::make_shared<TractModel>(handle.get());
    spm_map = std::make_shared<connectometry_result>();

    progress = 0;
    for(unsigned int index = 0;index < thread_count;++index)
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
            [this,index,thread_count,permutation_count](){run_permutation_multithread(index,thread_count,permutation_count);})));
}
void group_connectometry_analysis::calculate_FDR(void)
{
    double sum_pos_corr_null = 0;
    double sum_neg_corr_null = 0;
    double sum_pos_corr = 0;
    double sum_neg_corr = 0;
    for(int index = subject_pos_corr_null.size()-1;index >= 0;--index)
    {
        sum_pos_corr_null += subject_pos_corr_null[index];
        sum_neg_corr_null += subject_neg_corr_null[index];
        sum_pos_corr += subject_pos_corr[index];
        sum_neg_corr += subject_neg_corr[index];
        fdr_pos_corr[index] = (sum_pos_corr > 0.0 && sum_pos_corr_null > 0.0) ? std::min(1.0,sum_pos_corr_null/sum_pos_corr) : 1.0;
        fdr_neg_corr[index] = (sum_neg_corr > 0.0 && sum_neg_corr_null > 0.0) ? std::min(1.0,sum_neg_corr_null/sum_neg_corr): 1.0;

    }
    if(*std::min_element(fdr_pos_corr.begin(),fdr_pos_corr.end()) < 0.05)
        std::replace(fdr_pos_corr.begin(),fdr_pos_corr.end(),1.0,0.0);
    if(*std::min_element(fdr_neg_corr.begin(),fdr_neg_corr.end()) < 0.05)
        std::replace(fdr_neg_corr.begin(),fdr_neg_corr.end(),1.0,0.0);
}

void group_connectometry_analysis::generate_report(std::string& output)
{
    std::ostringstream html_report((output_file_name+".report.html").c_str());
    html_report << "<!DOCTYPE html>" << std::endl;
    html_report << "<html><head><title>Connectometry Report</title></head>" << std::endl;
    html_report << "<body>" << std::endl;
    if(!handle->report.empty())
    {
        html_report << "<h2>MRI Acquisition</h2>" << std::endl;
        html_report << "<p>" << handle->db.report << "</p>" << std::endl;
    }
    if(!report.empty())
    {
        html_report << "<h2>Connectometry analysis</h2>" << std::endl;
        html_report << "<p>" << report.c_str() << "</p>" << std::endl;
    }


    std::string index_name = QString(handle->db.index_name.c_str()).toUpper().toStdString();
    std::string track_hypothesis1 =
        (model->type == 1 ? index_name+" positively correlated with "+foi_str : std::string("increased ")+index_name);
    std::string track_hypothesis2 =
        (model->type == 1 ? index_name+" negatively correlated with "+foi_str : std::string("decreased ")+index_name);
    std::string fdr_result1,fdr_result2;
    if(fdr_threshold == 0.0f) // fdr control
    {
        fdr_result1 = "(FDR &lt ";
        fdr_result1 += std::to_string(fdr_threshold);
        fdr_result1 += ")";
        fdr_result2 = fdr_result1;
    }
    else
    {
        fdr_result1 = "(FDR = ";
        fdr_result1 += std::to_string(fdr_pos_corr[uint32_t(length_threshold)]);
        fdr_result1 += ")";
        fdr_result2 = "(FDR = ";
        fdr_result2 += std::to_string(fdr_neg_corr[uint32_t(length_threshold)]);
        fdr_result2 += ")";
    }


    html_report << "<h2>Results</h2>" << std::endl;

    // Positive correlation results
    html_report << "<h3>Tracks with " << track_hypothesis1 << "</h3>" << std::endl;

    if(progress == 100)
    {
        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".pos_corr_map.jpg").fileName().toStdString() << "\" width=\"600\"/>" << std::endl;
        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".pos_corr.jpg").fileName().toStdString() << "\" width=\"1200\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Tracks with " << track_hypothesis1 << " " << fdr_result1 << "</p>" << std::endl;
    }

    {
        html_report << "<p>";
        if(fdr_pos_corr[uint32_t(length_threshold)]>0.5f || !has_pos_corr_result)
            html_report << " The connectometry analysis found no significant result in tracks with ";
        else
            html_report << " The connectometry analysis found " << pos_corr_tracks_result << " showing ";
        html_report << track_hypothesis1 << " " << fdr_result1 <<  ".</p>" << std::endl;
    }


    // Negative correlation results
    html_report << "<h3>Tracks with " << track_hypothesis2 << "</h3>" << std::endl;

    if(progress == 100)
    {
        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".neg_corr_map.jpg").fileName().toStdString() << "\" width=\"600\"/>" << std::endl;
        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".neg_corr.jpg").fileName().toStdString() << "\" width=\"1200\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Tracks with " << track_hypothesis2 << " " << fdr_result2 << "</p>" << std::endl;
    }

    {
        html_report << "<p>";
        if(fdr_neg_corr[uint32_t(length_threshold)]>0.5f || !has_neg_corr_result)
            html_report << " The connectometry analysis found no significant result in tracks with ";
        else
            html_report << " The connectometry analysis found " << neg_corr_tracks_result << " showing ";
        html_report << track_hypothesis2 << " " << fdr_result2 <<  ".</p>" << std::endl;
    }

    if(progress == 100)
    {
        std::string permutation_explained =
    " The permutation was applied to subject labels to test results against permuted condition.\
     The histogram under permutated condition represents the result under the null hypothesis.\
     This null result is then used to test the histogram under nonpermutated condition to compute the FDR.\
     A smaller difference between histograms suggests that the study finding is similar to null finding and having a lower significance,\
     whereas a larger difference suggests greater significance of the study finding.";

        html_report << "<h3>False discovery rate analysis</h3>" << std::endl;

        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".pos_corr.dist.jpg").fileName().toStdString() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Permutation test showing the histograms of track counts with "<< track_hypothesis1 << ".</p>";


        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".neg_corr.dist.jpg").fileName().toStdString() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> Permutation test showing the histograms of track counts with "<< track_hypothesis2 << ".</p>";

        html_report << permutation_explained << std::endl;
        html_report << "<p></p><img src = \""<< QFileInfo(QString(output_file_name.c_str())+".fdr.jpg").fileName().toStdString() << "\" width=\"320\"/>" << std::endl;
        html_report << "<p><b>Fig.</b> The False discovery rate (FDR) at different track length </p>";
    }

    html_report << "</body></html>" << std::endl;
    output = html_report.str();


    // output track images
    if(progress == 100)
    {
        std::shared_ptr<fib_data> new_data(new fib_data);
        *(new_data.get()) = *(handle);
        tracking_window* new_mdi = new tracking_window(0,new_data);
        new_mdi->setWindowTitle(output_file_name.c_str());
        new_mdi->show();
        new_mdi->resize(2000,1000);
        new_mdi->update();
        new_mdi->command("set_zoom","1.0");
        new_mdi->command("set_param","show_surface","1");
        new_mdi->command("set_param","show_slice","0");
        new_mdi->command("set_param","show_region","0");
        new_mdi->command("set_param","bkg_color","16777215");
        new_mdi->command("set_param","surface_alpha","0.1");
        new_mdi->command("set_roi_view_index","icbm_wm");
        new_mdi->command("add_surface");
        new_mdi->tractWidget->addNewTracts("greater");
        new_mdi->tractWidget->tract_models[0]->add(*pos_corr_track.get());
        new_mdi->command("update_track");
        new_mdi->command("save_h3view_image",(output_file_name+".pos_corr.jpg").c_str());
        // do it twice to eliminate 3D artifact
        new_mdi->command("save_h3view_image",(output_file_name+".pos_corr.jpg").c_str());

        new_mdi->command("tract_to_roi");
        new_mdi->command("set_roi_view_index","icbm_t1w");
        new_mdi->command("set_roi_view_contrast","0.0","190.0");

        new_mdi->command("set_param","roi_zoom","16");
        new_mdi->command("set_param","roi_layout","4");
        new_mdi->command("save_roi_image",(output_file_name+".pos_corr_map.jpg").c_str(),"260");
        new_mdi->command("detele_all_region");


        new_mdi->command("delete_all_tract");
        new_mdi->tractWidget->addNewTracts("lesser");
        new_mdi->tractWidget->tract_models[0]->add(*neg_corr_track.get());
        new_mdi->command("update_track");
        new_mdi->command("save_h3view_image",(output_file_name+".neg_corr.jpg").c_str());

        new_mdi->command("tract_to_roi");
        new_mdi->command("save_roi_image",(output_file_name+".neg_corr_map.jpg").c_str(),"260");
        new_mdi->command("detele_all_region");
        // restore roi layout
        new_mdi->command("set_param","roi_layout","0");

        new_mdi->close();
    }
}

