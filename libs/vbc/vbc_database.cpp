#include <QFileInfo>
#include <ctime>
#include "vbc_database.h"
#include "fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "tracking/tracking_window.h"



vbc_database::vbc_database():handle(0),roi_type(0),normalize_qa(true)
{
}

bool vbc_database::create_database(const char* template_name)
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
bool vbc_database::load_database(const char* database_name)
{
    handle.reset(new fib_data);
    if(!handle->load_from_file(database_name))
    {
        error_msg = "Invalid fib file:";
        error_msg += handle->error_msg;
        return false;
    }
    fiber_threshold = 0.6*tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    voxels_in_threshold = 0;
    for(int i = 0;i < handle->dim.size();++i)
        if(handle->dir.fa[0][i] > fiber_threshold)
            ++voxels_in_threshold;
    return handle->db.has_db();
}


int vbc_database::run_track(const tracking_data& fib,std::vector<std::vector<float> >& tracks,float seed_ratio, unsigned int thread_count)
{
    std::vector<tipl::vector<3,short> > seed;
    for(tipl::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
        if(fib.fa[0][index.index()] > tracking_threshold)
            seed.push_back(tipl::vector<3,short>(index.x(),index.y(),index.z()));
    unsigned int count = seed.size()*seed_ratio*10000.0f/(float)voxels_in_threshold;
    if(!count)
    {
        tracks.clear();
        return 0;
    }
    ThreadData tracking_thread;
    tracking_thread.param.threshold = tracking_threshold;
    tracking_thread.param.cull_cos_angle = 1.0f;
    tracking_thread.param.step_size = handle->vs[0];
    tracking_thread.param.smooth_fraction = 0;
    tracking_thread.param.min_length = 0;
    tracking_thread.param.max_length = 200;
    tracking_thread.param.tracking_method = 0;// streamline fiber tracking
    tracking_thread.param.initial_direction = 0;// main directions
    tracking_thread.param.interpolation_strategy = 0; // trilinear interpolation
    tracking_thread.param.stop_by_tract = 0;// stop by seed
    tracking_thread.param.center_seed = 0;// subvoxel seeding
    tracking_thread.param.random_seed = 0;
    tracking_thread.param.termination_count = count;
    // if no seed assigned, assign whole brain
    if(roi_list.empty() || std::find(roi_type.begin(),roi_type.end(),3) == roi_type.end())
        tracking_thread.roi_mgr->setRegions(fib.dim,seed,1.0,3,"whole brain",tipl::vector<3>());
    if(!roi_list.empty())
    {
        for(unsigned int index = 0;index < roi_list.size();++index)
            tracking_thread.roi_mgr->setRegions(fib.dim,roi_list[index],roi_r_list[index],roi_type[index],
                                               "user assigned region",fib.vs);
    }
    tracking_thread.run(fib,thread_count,true);
    tracking_thread.track_buffer.swap(tracks);

    if(track_trimming)
    {
        TractModel t(handle);
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

void vbc_database::run_permutation_multithread(unsigned int id,unsigned int thread_count,unsigned int permutation_count)
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

            fib.fa = data.lesser_ptr;
            unsigned int s = run_track(fib,tracks,seed_ratio);
            if(null)
                seed_lesser_null[i] = s;
            else
                seed_lesser[i] = s;

            cal_hist(tracks,(null) ? subject_lesser_null : subject_lesser);

            if(output_resampling && !null)
            {
                std::lock_guard<std::mutex> lock(lock_lesser_tracks);
                if(tracks.size() > max_visible_track/permutation_count)
                    tracks.resize(max_visible_track/permutation_count);
                lesser_track->add_tracts(tracks,length_threshold);
                if(id == 1)
                {
                    lesser_track->delete_repeated(1.0f);
                    lesser_track->clear_deleted();
                }
                tracks.clear();
            }

            info.resample(*model.get(),null,true);
            calculate_spm(data,info,normalize_qa);
            fib.fa = data.greater_ptr;
            s = run_track(fib,tracks,seed_ratio);
            if(null)
                seed_greater_null[i] = s;
            else
                seed_greater[i] = s;
            cal_hist(tracks,(null) ? subject_greater_null : subject_greater);

            if(output_resampling && !null)
            {
                std::lock_guard<std::mutex> lock(lock_greater_tracks);
                if(tracks.size() > max_visible_track/permutation_count)
                    tracks.resize(max_visible_track/permutation_count);
                greater_track->add_tracts(tracks,length_threshold);
                if(id == 1)
                {
                    greater_track->delete_repeated(1.0f);
                    greater_track->clear_deleted();
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
                fib.fa = spm_map->lesser_ptr;
                run_track(fib,tracks,seed_ratio*permutation_count,threads.size());
                if(tracks.size() > max_visible_track)
                    tracks.resize(max_visible_track);
                lesser_track->add_tracts(tracks,length_threshold);
                fib.fa = spm_map->greater_ptr;
                run_track(fib,tracks,seed_ratio*permutation_count,threads.size());
                if(tracks.size() > max_visible_track)
                    tracks.resize(max_visible_track);
                greater_track->add_tracts(tracks,length_threshold);
            }
        }
    }
    if(id == 0 && !terminated)
        progress = 100;
}
void vbc_database::clear(void)
{
    if(!threads.empty())
    {
        terminated = true;
        wait();
        threads.clear();
        terminated = false;
    }
}
void vbc_database::wait(void)
{
    for(int i = 0;i < threads.size();++i)
        threads[i]->wait();
}


void vbc_database::save_tracks_files(void)
{
    for(int i = 0;i < threads.size();++i)
        threads[i]->wait();
    has_greater_result = false;
    has_lesser_result = false;
    {
        if(greater_track->get_visible_track_count())
        {
            if(fdr_threshold != 0.0)
            {
                fdr_greater.back() = 0.0f;
                for(int length = 10;length < fdr_greater.size();++length)
                    if(fdr_greater[length] < fdr_threshold)
                    {
                        greater_track->delete_by_length(length);
                        break;
                    }
            }

            std::ostringstream out1;
            out1 << output_file_name << ".greater.trk.gz";
            greater_track->save_tracts_to_file(out1.str().c_str());
            greater_tracks_result = "";
            greater_track->recognize_report(greater_tracks_result);
            if(greater_tracks_result.empty())
                greater_tracks_result = "tracks";
            has_greater_result = true;
        }
        else
        {
            std::ostringstream out1;
            out1 << output_file_name << ".greater.no_trk.txt";
            std::ofstream(out1.str().c_str());
        }
        {
            std::ostringstream out1;
            out1 << output_file_name << ".greater.fib.gz";
            gz_mat_write mat_write(out1.str().c_str());
            for(unsigned int i = 0;i < handle->mat_reader.size();++i)
            {
                std::string name = handle->mat_reader.name(i);
                if(name == "dimension" || name == "voxel_size" ||
                        name == "odf_vertices" || name == "odf_faces" || name == "trans")
                    mat_write.write(handle->mat_reader[i]);
                if(name == "fa0")
                    mat_write.write("qa_map",handle->dir.fa[0],1,handle->dim.size());
            }
            for(unsigned int i = 0;i < spm_map->greater_ptr.size();++i)
            {
                std::ostringstream out1,out2;
                out1 << "fa" << i;
                out2 << "index" << i;
                mat_write.write(out1.str().c_str(),spm_map->greater_ptr[i],1,handle->dim.size());
                mat_write.write(out2.str().c_str(),handle->dir.findex[i],1,handle->dim.size());
            }
        }

        if(lesser_track->get_visible_track_count())
        {
            if(fdr_threshold != 0.0)
            {
                fdr_lesser.back() = 0.0f;
                for(int length = 10;length < fdr_lesser.size();++length)
                    if(fdr_lesser[length] < fdr_threshold)
                    {
                        lesser_track->delete_by_length(length);
                        break;
                    }
            }

            std::ostringstream out1;
            out1 << output_file_name << ".lesser.trk.gz";
            lesser_track->save_tracts_to_file(out1.str().c_str());
            lesser_tracks_result = "";
            lesser_track->recognize_report(lesser_tracks_result);
            if(lesser_tracks_result.empty())
                lesser_tracks_result = "tracks";
            has_lesser_result = true;
        }
        else
        {
            std::ostringstream out1;
            out1 << output_file_name << ".lesser.no_trk.txt";
            std::ofstream(out1.str().c_str());
        }

        {
            std::ostringstream out1;
            out1 << output_file_name << ".lesser.fib.gz";
            gz_mat_write mat_write(out1.str().c_str());
            for(unsigned int i = 0;i < handle->mat_reader.size();++i)
            {
                std::string name = handle->mat_reader.name(i);
                if(name == "dimension" || name == "voxel_size" ||
                        name == "odf_vertices" || name == "odf_faces" || name == "trans")
                    mat_write.write(handle->mat_reader[i]);
                if(name == "fa0")
                    mat_write.write("qa_map",handle->dir.fa[0],1,handle->dim.size());
            }
            for(unsigned int i = 0;i < spm_map->greater_ptr.size();++i)
            {
                std::ostringstream out1,out2;
                out1 << "fa" << i;
                out2 << "index" << i;
                mat_write.write(out1.str().c_str(),spm_map->lesser_ptr[i],1,handle->dim.size());
                mat_write.write(out2.str().c_str(),handle->dir.findex[i],1,handle->dim.size());
            }
        }

    }
}

void vbc_database::run_permutation(unsigned int thread_count,unsigned int permutation_count)
{
    clear();
    // output report
    {
        std::ostringstream out;

        if(model->type == 1) // run regression model
        {
            out << "\nDiffusion MRI connectometry (Yeh et al. NeuroImage 125 (2016): 162-171) was used to study the effect of "
                << foi_str
                << ". A multiple regression model was used to consider ";
            for(unsigned int index = 1;index < model->variables.size();++index)
            {
                if(index && model->variables.size() > 3)
                    out << ",";
                out << " ";
                if(model->variables.size() >= 3 && index+1 == model->variables.size())
                    out << "and ";
                out << model->variables[index];
            }
            out << " in a total of " << model->subject_index.size() << " subjects. ";
            out << " A T-score threshold of " << tracking_threshold;
            out << " was assigned to select local connectomes, and the local connectomes were tracked using a deterministic fiber tracking algorithm (Yeh et al. PLoS ONE 8(11): e80713, 2013).";
        }

        if(normalize_qa)
            out << " The SDF was normalized.";
        if(track_trimming)
            out << " Track trimming was conducted with " << track_trimming << " iterations.";

        if(output_resampling)
            out << " All tracks generated from bootstrap resampling were included.";

        if(fdr_threshold == 0.0f)
            out << " A length threshold of " << length_threshold << " voxel distance was used to select tracks.";
        else
            out << " An FDR threshold of " << fdr_threshold << " was used to select tracks.";
        out << " The track density was " << seed_ratio << " per voxel.";

        out << " To estimate the false discovery rate, a total of "
            << permutation_count
            << " randomized permutations were applied to the group label to obtain the null distribution of the track length.";

        if(!roi_list.empty())
        {
            out << " The tracking algorithm used ";
            const char roi_type_name[5][20] = {"region of interst","region of avoidance","ending region","seeding region","terminating region"};
            for(unsigned int index = 0;index < roi_list.size();++index)
            {
                if(index && roi_list.size() > 2)
                    out << ",";
                out << " ";
                if(roi_list.size() >= 2 && index+1 == roi_list.size())
                    out << "and ";
                out << roi_name[index] << " as the " << roi_type_name[roi_type[index]];
            }
            out << ".";
        }
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
            output_file_name += std::to_string((int)fdr_threshold*100);
        }
        if(!roi_list.empty())
        {
            const char roi_type_name2[5][5] = {"roi","roa","end","seed"};
            for(unsigned int index = 0;index < roi_list.size();++index)
            {
                output_file_name += ".";
                output_file_name += roi_type_name2[roi_type[index]];
                output_file_name += ".";
                output_file_name += roi_name[index];
            }
        }
    }

    terminated = false;
    subject_greater_null.clear();
    subject_greater_null.resize(200);
    subject_lesser_null.clear();
    subject_lesser_null.resize(200);
    subject_greater.clear();
    subject_greater.resize(200);
    subject_lesser.clear();
    subject_lesser.resize(200);
    fdr_greater.clear();
    fdr_greater.resize(200);
    fdr_lesser.clear();
    fdr_lesser.resize(200);

    seed_greater_null.clear();
    seed_greater_null.resize(permutation_count);
    seed_lesser_null.clear();
    seed_lesser_null.resize(permutation_count);
    seed_greater.clear();
    seed_greater.resize(permutation_count);
    seed_lesser.clear();
    seed_lesser.resize(permutation_count);

    model->rand_gen.reset();
    std::srand(0);

    has_greater_result = true;
    has_lesser_result = true;
    greater_tracks_result = "tracks";
    lesser_tracks_result = "tracks";

    greater_track = std::make_shared<TractModel>(handle);
    lesser_track = std::make_shared<TractModel>(handle);
    spm_map = std::make_shared<connectometry_result>();

    progress = 0;
    for(unsigned int index = 0;index < thread_count;++index)
        threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
            [this,index,thread_count,permutation_count](){run_permutation_multithread(index,thread_count,permutation_count);})));
}
void vbc_database::calculate_FDR(void)
{
    double sum_greater_null = 0;
    double sum_lesser_null = 0;
    double sum_greater = 0;
    double sum_lesser = 0;
    for(int index = subject_greater_null.size()-1;index >= 0;--index)
    {
        sum_greater_null += subject_greater_null[index];
        sum_lesser_null += subject_lesser_null[index];
        sum_greater += subject_greater[index];
        sum_lesser += subject_lesser[index];
        fdr_greater[index] = (sum_greater > 0.0 && sum_greater_null > 0.0) ? std::min(1.0,sum_greater_null/sum_greater) : 1.0;
        fdr_lesser[index] = (sum_lesser > 0.0 && sum_lesser_null > 0.0) ? std::min(1.0,sum_lesser_null/sum_lesser): 1.0;

    }
    if(*std::min_element(fdr_greater.begin(),fdr_greater.end()) < 0.05)
        std::replace(fdr_greater.begin(),fdr_greater.end(),1.0,0.0);
    if(*std::min_element(fdr_lesser.begin(),fdr_lesser.end()) < 0.05)
        std::replace(fdr_lesser.begin(),fdr_lesser.end(),1.0,0.0);
}

void vbc_database::generate_report(std::string& output)
{
    std::ostringstream html_report((output_file_name+".report.html").c_str());
    html_report << "<!DOCTYPE html>" << std::endl;
    html_report << "<html><head><title>Connectometry Report</title></head>" << std::endl;
    html_report << "<body>" << std::endl;
    if(!handle->report.empty())
    {
        html_report << "<h2>MRI Acquisition</h2>" << std::endl;
        html_report << "<p>" << handle->report << "</p>" << std::endl;
    }
    if(!report.empty())
    {
        html_report << "<h2>Connectometry analysis</h2>" << std::endl;
        html_report << "<p>" << report.c_str() << "</p>" << std::endl;
    }


    std::ostringstream out_greater,out_lesser;
    if(fdr_threshold == 0.0)
    {
        out_greater << " The connectometry analysis identified "
        << (fdr_greater[length_threshold]>0.5 || !has_greater_result ? "no track": greater_tracks_result.c_str())
        << " with increased connectivity";

        if(model->type == 1) // regression
            out_greater << " related to " << foi_str;
        out_greater << " (FDR=" << fdr_greater[length_threshold] << ").";

        out_lesser << " The connectometry analysis identified "
        << (fdr_lesser[length_threshold]>0.5 || !has_lesser_result ? "no track": lesser_tracks_result.c_str())
        << " with decreased connectivity";
        if(model->type == 1) // regression
            out_lesser << "related to " << foi_str;
        out_lesser << " (FDR=" << fdr_lesser[length_threshold] << ").";
    }
    else
    {
        out_greater << " The connectometry analysis identified "
        << (!has_greater_result ? "no track": greater_tracks_result.c_str())
        << " with increased connectivity";
        if(model->type == 1) // regression
            out_greater << " related to " << foi_str;
        out_greater << ".";


        out_lesser << " The connectometry analysis identified "
        << (!has_lesser_result ? "no track": lesser_tracks_result.c_str())
        << " with decreased connectivity";
        if(model->type == 1) // regression
            out_lesser << "related to " << foi_str;
        out_lesser << ".";
    }


    html_report << "<h2>Results</h2>" << std::endl;
    if(model->type == 1) // regression
        html_report << "<h3>Positive Correlation with " << foi_str << "</h3>" << std::endl;
    if(model->type == 3)
        html_report << "<h3>Increased connectivity</h3>" << std::endl;

    if(progress == 100)
    {
        html_report << "<img src = \""<< QFileInfo(QString(output_file_name.c_str())+".positive.jpg").fileName().toStdString() << "\" width=\"800\"/>" << std::endl;
        if(model->type == 1) // regression
            html_report << "<p><b>Fig.</b> Tracks positively correlated with "<< foi_str << "</p>";
        if(model->type ==3)
            html_report << "<p><b>Fig.</b> Tracks with increased connectivity</p>";
    }
    html_report << "<p>" << out_greater.str().c_str() << "</p>" << std::endl;

    if(model->type == 1) // regression
        html_report << "<h3>Negatively Correlation with " << foi_str << "</h3>" << std::endl;
    if(model->type == 3) // regression
        html_report << "<h3>Decreased connectivity</h3>" << std::endl;

    if(progress == 100)
    {
        html_report << "<img src = \""<< QFileInfo(QString(output_file_name.c_str())+".negative.jpg").fileName().toStdString() << "\" width=\"800\"/>" << std::endl;
        if(model->type == 1) // regression
            html_report << "<p><b>Fig.</b> Tracks negatively correlated with "<< foi_str << "</p>";
        if(model->type ==3)
            html_report << "<p><b>Fig.</b> Tracks with decreased connectivity.</p>";
    }
    html_report << "<p>" << out_lesser.str().c_str() << "</p>" << std::endl;
    html_report << "</body></html>" << std::endl;
    output = html_report.str();


    // output track images
    if(progress == 100)
    {
        std::shared_ptr<fib_data> new_data(new fib_data);
        *(new_data.get()) = *(handle);
        tracking_window* new_mdi = new tracking_window(0,new_data);
        new_mdi->setWindowTitle(output_file_name.c_str());
        new_mdi->showMaximized();
        new_mdi->update();
        new_mdi->command("set_zoom","0.8");
        new_mdi->command("set_param","show_surface","1");
        new_mdi->command("set_param","show_slice","0");
        new_mdi->command("set_param","show_region","0");
        new_mdi->command("set_param","bkg_color","16777215");
        new_mdi->command("set_param","surface_alpha","0.1");
        new_mdi->command("set_roi_view_index","icbm_wm");
        new_mdi->command("add_surface");
        new_mdi->tractWidget->addNewTracts("greater");
        new_mdi->tractWidget->tract_models[0]->add(*greater_track.get());
        new_mdi->command("update_track");
        new_mdi->command("save_h3view_image",(output_file_name+".positive.jpg").c_str());
        // do it twice to eliminate 3D artifact
        new_mdi->command("save_h3view_image",(output_file_name+".positive.jpg").c_str());
        new_mdi->command("delete_all_tract");
        new_mdi->tractWidget->addNewTracts("lesser");
        new_mdi->tractWidget->tract_models[0]->add(*lesser_track.get());
        new_mdi->command("update_track");
        new_mdi->command("save_h3view_image",(output_file_name+".negative.jpg").c_str());
        new_mdi->close();
    }
}

