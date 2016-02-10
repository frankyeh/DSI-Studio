#include <cstdlib>     /* srand, rand */
#include <ctime>
#include <boost/thread.hpp>
#include <boost/math/distributions/students_t.hpp>
#include "vbc_database.h"
#include "fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"



vbc_database::vbc_database():handle(0),roi_type(0),normalize_qa(true)
{
}

bool vbc_database::create_database(const char* template_name)
{
    handle.reset(new FibData);
    if(!handle->load_from_file(template_name))
    {
        error_msg = handle->error_msg;
        return false;
    }
    fiber_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(handle->dim,handle->fib.fa[0]));
    handle->calculate_si2vi();
    return true;
}
bool vbc_database::load_database(const char* database_name)
{
    handle.reset(new FibData);
    if(!handle->load_from_file(database_name))
    {
        error_msg = "Invalid fib file:";
        error_msg += handle->error_msg;
        return false;
    }
    fiber_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(handle->dim,handle->fib.fa[0]));
    return !handle->subject_qa.empty();
}

void fib_data::initialize(FibData* handle)
{
    unsigned char num_fiber = handle->fib.num_fiber;
    greater.resize(num_fiber);
    lesser.resize(num_fiber);
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        greater[fib].resize(handle->dim.size());
        lesser[fib].resize(handle->dim.size());
    }
    greater_ptr.resize(num_fiber);
    lesser_ptr.resize(num_fiber);
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        greater_ptr[fib] = &greater[fib][0];
        lesser_ptr[fib] = &lesser[fib][0];
    }
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        std::fill(greater[fib].begin(),greater[fib].end(),0.0);
        std::fill(lesser[fib].begin(),lesser[fib].end(),0.0);
    }
}

void fib_data::add_greater_lesser_mapping_for_tracking(FibData* handle)
{
    for(unsigned int index = 0;index < handle->fib.index_name.size();++index)
        if(handle->fib.index_name[index].find('%') != std::string::npos)
        {
            handle->fib.index_name.erase(handle->fib.index_name.begin()+index);
            handle->fib.index_data.erase(handle->fib.index_data.begin()+index);
            index = 0;
        }
    handle->fib.index_name.push_back(">%");
    handle->fib.index_data.push_back(std::vector<const float*>());
    handle->fib.index_data.back() = greater_ptr;
    handle->fib.index_name.push_back("<%");
    handle->fib.index_data.push_back(std::vector<const float*>());
    handle->fib.index_data.back() = lesser_ptr;
}
bool fib_data::add_dif_mapping_for_tracking(FibData* handle,std::vector<std::vector<float> >& fa_data)
{
    // normalization
    float max_qa1 = 0.0,max_qa2 = 0.0;
    for(unsigned char fib = 0;fib < handle->fib.num_fiber;++fib)
    {
        max_qa1 = std::max<float>(max_qa1,*std::max_element(handle->fib.fa[fib],handle->fib.fa[fib] + handle->dim.size()));
        max_qa2 = std::max<float>(max_qa2,*std::max_element(fa_data[fib].begin(),fa_data[fib].end()));
    }
    if(max_qa1 == 0.0 || max_qa2 == 0.0)
        return false;
    //calculating dif
    for(unsigned char fib = 0;fib < handle->fib.num_fiber;++fib)
    {
        for(unsigned int index = 0;index < handle->dim.size();++index)
            if(handle->fib.fa[fib][index] > 0.0 && fa_data[fib][index] > 0.0)
            {
                float f1 = handle->fib.fa[fib][index];
                float f2 = fa_data[fib][index]*max_qa1/max_qa2;
                if(f1 > f2)
                    lesser[fib][index] = f1-f2;  // subject decreased connectivity
                else
                    greater[fib][index] = f2-f1; // subject increased connectivity
            }
    }
    for(unsigned int index = 0;index < handle->fib.index_name.size();++index)
        if(handle->fib.index_name[index] == "inc" ||
                handle->fib.index_name[index] == "dec")
        {
            handle->fib.index_name.erase(handle->fib.index_name.begin()+index);
            handle->fib.index_data.erase(handle->fib.index_data.begin()+index);
            index = 0;
        }
    handle->fib.index_name.push_back("inc");
    handle->fib.index_data.push_back(std::vector<const float*>());
    handle->fib.index_data.back() = greater_ptr;
    handle->fib.index_name.push_back("dec");
    handle->fib.index_data.push_back(std::vector<const float*>());
    handle->fib.index_data.back() = lesser_ptr;
    return true;
}

void vbc_database::run_track(const fiber_orientations& fib,std::vector<std::vector<float> >& tracks,float seed_ratio, unsigned int thread_count)
{
    std::vector<image::vector<3,short> > seed;
    for(image::pixel_index<3> index;index.is_valid(handle->dim);index.next(handle->dim))
        if(fib.fa[0][index.index()] > fib.threshold)
            seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
    if(seed.empty() || seed.size()*seed_ratio < 1.0)
    {
        tracks.clear();
        return;
    }
    ThreadData tracking_thread(false);
    tracking_thread.param.step_size = 1.0; // fixed 1 mm
    tracking_thread.param.smooth_fraction = 0;
    tracking_thread.param.min_points_count3 = 6;
    tracking_thread.param.max_points_count3 = std::max<unsigned int>(6,3.0*500/tracking_thread.param.step_size);
    tracking_thread.tracking_method = 0;// streamline fiber tracking
    tracking_thread.initial_direction = 0;// main directions
    tracking_thread.interpolation_strategy = 0; // trilinear interpolation
    tracking_thread.stop_by_tract = 0;// stop by seed
    tracking_thread.center_seed = 0;// subvoxel seeding
    // if no seed assigned, assign whole brain
    if(roi_list.empty() || std::find(roi_type.begin(),roi_type.end(),3) == roi_type.end())
        tracking_thread.setRegions(fib.dim,seed,3,"whole brain");
    if(!roi_list.empty())
    {
        for(unsigned int index = 0;index < roi_list.size();++index)
            tracking_thread.setRegions(fib.dim,roi_list[index],roi_type[index],"user assigned region");
    }
    tracking_thread.run(fib,thread_count,seed.size()*seed_ratio,true);
    tracking_thread.track_buffer.swap(tracks);
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
/*
bool vbc_database::calculate_individual_affected_tracks(const char* file_name,
                                                        std::vector<std::vector<std::vector<float> > >& greater,
                                                        std::vector<std::vector<std::vector<float> > >& lesser)
{
    fib_data data;
    std::vector<float> cur_subject_data;
    if(!get_odf_profile(file_name,cur_subject_data))
    {
        error_msg = "Cannot read subject file ";
        error_msg += file_name;
        return false;
    }
    std::vector<unsigned int> resample;
    calculate_percentile(&cur_subject_data[0],resample,data);

    std::vector<std::vector<float> > greater_tracks;
    std::vector<std::vector<float> > lesser_tracks;
    fiber_orientations fib;
    fib.read(*handle);
    fib.threshold = tracking_threshold;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    fib.fa = data.greater_ptr;
    fib.findex = data.greater_dir_ptr;
    run_track(fib,greater_tracks);
    fib.fa = data.lesser_ptr;
    fib.findex = data.lesser_dir_ptr;
    run_track(fib,lesser_tracks);

    greater.clear();
    lesser.clear();
    for(unsigned int index = 0;index < greater_tracks.size();++index)
    {
        float length = (float)greater_tracks[index].size()/3.0-1.0;
        if(length <= 10.0)
            continue;
        length -= 10.0;
        unsigned int pos = std::floor(length/10.0);
        if(greater.size() <= pos)
            greater.resize(pos+1);
        greater[pos].push_back(greater_tracks[index]);
    }
    for(unsigned int index = 0;index < lesser_tracks.size();++index)
    {
        float length = (float)lesser_tracks[index].size()/3.0-1.0;
        if(length <= 10.0)
            continue;
        length -= 10.0;
        unsigned int pos = std::floor(length/10.0);
        if(lesser.size() <= pos)
            lesser.resize(pos+1);
        lesser[pos].push_back(lesser_tracks[index]);
    }
    return true;
}
*/

void stat_model::init(unsigned int subject_count)
{
    subject_index.resize(subject_count);
    for(unsigned int index = 0;index < subject_count;++index)
        subject_index[index] = index;
}

bool stat_model::pre_process(void)
{
    switch(type)
    {
    case 0: // group
        group2_count = 0;
        for(unsigned int index = 0;index < label.size();++index)
            if(label[index])
                ++group2_count;
        group1_count = label.size()-group2_count;
        return group2_count > 3 && group1_count > 3;
    case 1: // multiple regression
        {
            X_min = X_max = std::vector<double>(X.begin(),X.begin()+feature_count);
            unsigned int subject_count = X.size()/feature_count;
            for(unsigned int i = 1,index = feature_count;i < subject_count;++i)
                for(unsigned int j = 0;j < feature_count;++j,++index)
                {
                    if(X[index] < X_min[j])
                        X_min[j] = X[index];
                    if(X[index] > X_max[j])
                        X_max[j] = X[index];
                }
            X_range.resize(feature_count);
            for(unsigned int j = 0;j < feature_count;++j)
                X_range[j] = X_max[j]-X_min[j];
        }
        return mr.set_variables(&*X.begin(),feature_count,X.size()/feature_count);
    case 2:
        return true;
    case 3: // paired
        return subject_index.size() == paired.size() && !subject_index.empty();
    }
    return false;
}
void stat_model::remove_subject(unsigned int index)
{
    if(index >= subject_index.size())
        throw std::runtime_error("remove subject out of bound");
    if(!label.empty())
        label.erase(label.begin()+index);
    if(!X.empty())
        X.erase(X.begin()+index*feature_count,X.begin()+(index+1)*feature_count);
    subject_index.erase(subject_index.begin()+index);
    pre_process();
}

void stat_model::remove_missing_data(double missing_value)
{
    std::vector<unsigned int> remove_list;
    switch(type)
    {
        case 0:  // group
            for(unsigned int index = 0;index < label.size();++index)
            {
                if(label[index] == missing_value)
                    remove_list.push_back(index);
            }
            break;

        case 1: // multiple regression
            for(unsigned int index = 0;index < subject_index.size();++index)
            {
                for(unsigned int j = 1;j < feature_count;++j)
                {
                    if(X[index*feature_count + j] == missing_value)
                    {
                        remove_list.push_back(index);
                        break;
                    }
                }
            }
            break;
        case 3:
            break;
    }

    if(remove_list.empty())
        return;
    while(!remove_list.empty())
    {
        unsigned int index = remove_list.back();
        if(!label.empty())
            label.erase(label.begin()+index);
        if(!X.empty())
            X.erase(X.begin()+index*feature_count,X.begin()+(index+1)*feature_count);
        subject_index.erase(subject_index.begin()+index);
        remove_list.pop_back();
    }
    pre_process();
}

bool stat_model::resample(stat_model& rhs,bool null,bool bootstrap)
{
    boost::mutex::scoped_lock lock(rhs.lock_random);
    type = rhs.type;
    feature_count = rhs.feature_count;
    study_feature = rhs.study_feature;
    subject_index.resize(rhs.subject_index.size());

    unsigned int trial = 0;
    do
    {
        if(trial > 100)
            throw std::runtime_error("Invalid subject demographics for multiple regression");
        ++trial;


        switch(type)
        {
        case 0: // group
        {
            std::vector<unsigned int> group0,group1;
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
                if(rhs.label[index])
                    group1.push_back(index);
                else
                    group0.push_back(index);
            label.resize(rhs.subject_index.size());
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = index;
                if(bootstrap)
                    new_index = rhs.label[index] ? group1[rhs.rand_gen(group1.size())]:group0[rhs.rand_gen(group0.size())];
                subject_index[index] = rhs.subject_index[new_index];
                label[index] = rhs.label[new_index];
            }
        }
            break;
        case 1: // multiple regression
            X.resize(rhs.X.size());
            for(unsigned int index = 0,pos = 0;index < rhs.subject_index.size();++index,pos += feature_count)
            {
                unsigned int new_index = bootstrap ? rhs.rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
                std::copy(rhs.X.begin()+new_index*feature_count,
                          rhs.X.begin()+new_index*feature_count+feature_count,X.begin()+pos);
            }

            break;
        case 2: // individual
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = bootstrap ? rhs.rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
            }
            break;
        case 3: // paired
            paired.resize(rhs.subject_index.size());
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = bootstrap ? rhs.rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
                paired[index] = rhs.paired[new_index];
                if(null && rhs.rand_gen(2) == 1)
                    std::swap(subject_index[index],paired[index]);
            }
            break;
        }
        if(null)
            std::random_shuffle(subject_index.begin(),subject_index.end(),rhs.rand_gen);
    }while(!pre_process());

    return true;
}
void stat_model::select(const std::vector<double>& population,std::vector<double>& selected_population) const
{
    for(unsigned int index = 0;index < subject_index.size();++index)
        selected_population[index] = population[subject_index[index]];
    selected_population.resize(subject_index.size());
}

double stat_model::operator()(const std::vector<double>& original_population,unsigned int pos) const
{
    std::vector<double> population(subject_index.size());
    select(original_population,population);
    switch(type)
    {
    case 0: // group
        {
        float sum1 = 0.0;
        float sum2 = 0.0;
        for(unsigned int index = 0;index < label.size();++index)
            if(label[index]) // group 1
                sum2 += population[index];
            else
                // group 0
                sum1 += population[index];
        float mean1 = sum1/((double)group1_count);
        float mean2 = sum2/((double)group2_count);
        float result = (mean1 + mean2)/2.0;
        if(result != 0.0)
            result = (mean1 - mean2) / result;
        return result;
        }
        break;
    case 1: // multiple regression
        {
            std::vector<double> b(feature_count);
            mr.regress(&*population.begin(),&*b.begin());
            double mean = image::mean(population.begin(),population.end());
            return mean == 0 ? 0:b[study_feature]*X_range[study_feature]/mean;
        }
        break;
    case 2: // individual
        {
            float value = (individual_data_sd == 1.0) ? individual_data[pos]:individual_data[pos]/individual_data_sd;
            if(value == 0.0)
                return 0.0;
            int rank = 0;
            for(unsigned int index = 0;index < population.size();++index)
                if(value > population[index])
                    ++rank;
            return (rank > (population.size() >> 1)) ?
                                (double)rank/(double)population.size():
                                (double)(rank-(int)population.size())/(double)population.size();
        }
        break;
    case 3: // paired
        {
            unsigned int half_size = population.size() >> 1;
            float g1 = std::accumulate(population.begin(),population.begin()+half_size,0.0);
            float g2 = std::accumulate(population.begin()+half_size,population.end(),0.0);
            return 2.0*(g1-g2)/(g1+g2);
        }
        break;
    }

    return 0.0;
}
void calculate_spm(FibData* handle,fib_data& data,stat_model& info,
                   float fiber_threshold,bool normalize_qa,bool& terminated)
{
    data.initialize(handle);
    std::vector<double> population(handle->subject_qa.size());
    for(unsigned int s_index = 0;s_index < handle->si2vi.size() && !terminated;++s_index)
    {
        unsigned int cur_index = handle->si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < handle->fib.num_fiber && handle->fib.fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=handle->si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            if(normalize_qa)
                for(unsigned int index = 0;index < population.size();++index)
                    population[index] = handle->subject_qa[index][pos]/handle->subject_qa_sd[index];
            else
                for(unsigned int index = 0;index < population.size();++index)
                    population[index] = handle->subject_qa[index][pos];

            if(std::find(population.begin(),population.end(),0.0) != population.end())
                continue;
            double result = info(population,pos);

            if(result > 0.0) // group 0 > group 1
                data.greater[fib][cur_index] = result;
            if(result < 0.0) // group 0 < group 1
                data.lesser[fib][cur_index] = -result;

        }
    }
}


bool vbc_database::read_subject_data(const std::vector<std::string>& files,std::vector<std::vector<float> >& data)
{
    begin_prog("reading",true);
    data.resize(files.size());
    for(unsigned int index = 0;check_prog(index,files.size());++index)
        if(!handle->get_odf_profile(files[index].c_str(),data[index]))
        {
            error_msg = "Cannot read file ";
            error_msg += files[index];
            check_prog(0,0);
            return false;
        }
    begin_prog("reading",false);
    check_prog(0,0);
    return true;
}

void vbc_database::run_permutation_multithread(unsigned int id)
{
    fib_data data;
    fiber_orientations fib;
    fib.read(*handle);
    fib.threshold = tracking_threshold;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    float total_track_count = seeding_density*fib.vs[0]*fib.vs[1]*fib.vs[2];
    std::vector<std::vector<float> > tracks;

    if(model->type == 2) // individual
    {
        if(id == 0)
        for(unsigned int subject_id = 0;subject_id < individual_data.size() && !terminated;++subject_id)
        {
            stat_model info;
            info.resample(*model.get(),false,false);
            info.individual_data = &(individual_data[subject_id][0]);
            info.individual_data_sd = normalize_qa ? individual_data_sd[subject_id]:1.0;
            calculate_spm(spm_maps[subject_id],info);
            if(terminated)
                return;
            fib.fa = spm_maps[subject_id].lesser_ptr;
            run_track(fib,tracks,total_track_count,threads->size());
            lesser_tracks[subject_id].add_tracts(tracks);
            fib.fa = spm_maps[subject_id].greater_ptr;
            if(terminated)
                return;
            run_track(fib,tracks,total_track_count,threads->size());
            greater_tracks[subject_id].add_tracts(tracks);
        }

        bool null = true;
        while(total_count < permutation_count && !terminated)
        {
            if(null)
                ++total_count_null;
            else
                ++total_count;

            for(unsigned int subject_id = 0;subject_id < individual_data.size() && !terminated;++subject_id)
            {
                stat_model info;
                info.resample(*model.get(),null,true);
                if(null)
                {
                    unsigned int random_subject_id = model->rand_gen(model->subject_index.size());
                    info.individual_data = handle->subject_qa[random_subject_id];
                    info.individual_data_sd = normalize_qa ? handle->subject_qa_sd[random_subject_id]:1.0;
                }
                else
                {
                    info.individual_data = &(individual_data[subject_id][0]);
                    info.individual_data_sd = normalize_qa ? individual_data_sd[subject_id]:1.0;
                }
                calculate_spm(data,info);
                fib.fa = data.lesser_ptr;
                run_track(fib,tracks,total_track_count/permutation_count);
                cal_hist(tracks,(null) ? subject_lesser_null : subject_lesser);
                /*
                if(!null)
                {
                    boost::mutex::scoped_lock lock(lock_lesser_tracks);
                    lesser_tracks[subject_id].add_tracts(tracks,30); // at least 30 mm
                    tracks.clear();
                }
                */

                fib.fa = data.greater_ptr;
                run_track(fib,tracks,total_track_count/permutation_count);
                cal_hist(tracks,(null) ? subject_greater_null : subject_greater);
                /*
                if(!null)
                {
                    boost::mutex::scoped_lock lock(lock_greater_tracks);
                    greater_tracks[subject_id].add_tracts(tracks,30);  // at least 30 mm
                    tracks.clear();
                }
                */
            }
            null = !null;
        }
    }
    else
    {
        bool null = true;
        while(total_count < permutation_count && !terminated)
        {
            if(null)
                ++total_count_null;
            else
                ++total_count;
            stat_model info;
            info.resample(*model.get(),null,true);
            calculate_spm(data,info);

            fib.fa = data.lesser_ptr;
            run_track(fib,tracks,total_track_count/permutation_count);
            cal_hist(tracks,(null) ? subject_lesser_null : subject_lesser);

            if(output_resampling && !null)
            {
                boost::mutex::scoped_lock lock(lock_lesser_tracks);
                lesser_tracks[0].add_tracts(tracks);
                tracks.clear();
            }

            fib.fa = data.greater_ptr;
            run_track(fib,tracks,total_track_count/permutation_count);
            cal_hist(tracks,(null) ? subject_greater_null : subject_greater);

            if(output_resampling && !null)
            {
                boost::mutex::scoped_lock lock(lock_greater_tracks);
                greater_tracks[0].add_tracts(tracks);
                tracks.clear();
            }

            null = !null;
        }
        if(id == 0)
        {
            stat_model info;
            info.resample(*model.get(),false,false);
            calculate_spm(spm_maps[0],info);
            if(terminated)
                return;
            fib.fa = spm_maps[0].lesser_ptr;
            run_track(fib,tracks,total_track_count,threads->size());
            lesser_tracks[0].add_tracts(tracks);
            fib.fa = spm_maps[0].greater_ptr;
            if(terminated)
                return;
            run_track(fib,tracks,total_track_count,threads->size());
            greater_tracks[0].add_tracts(tracks);
        }
    }
}
void vbc_database::clear_thread(void)
{
    if(threads.get())
    {
        terminated = true;
        threads->join_all();
        threads.reset(0);
        terminated = false;
    }
}
void vbc_database::save_tracks_files(std::vector<std::string>& saved_file_name)
{
    threads->join_all();
    if(trk_file_names.size() != greater_tracks.size())
        throw std::runtime_error("Please assign file name for saving trk files.");
    saved_file_name.clear();
    has_greater_result = false;
    has_lesser_result = false;
    for(unsigned int index = 0;index < greater_tracks.size();++index)
    {
        if(length_threshold_greater)
        {
            TractModel tracks(handle.get());
            tracks.add_tracts(greater_tracks[index].get_tracts(),length_threshold_greater);
            if(tracks.get_visible_track_count())
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".greater.trk.gz";
                tracks.save_tracts_to_file(out1.str().c_str());
                saved_file_name.push_back(out1.str().c_str());
                has_greater_result = true;
            }
            else
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".greater.no_trk.txt";
                std::ofstream(out1.str().c_str());
            }
            greater_tracks[index] = tracks;
        }
        {
            std::ostringstream out1;
            out1 << trk_file_names[index] << ".greater.fib.gz";
            gz_mat_write mat_write(out1.str().c_str());
            for(unsigned int i = 0;i < handle->mat_reader.size();++i)
            {
                std::string name = handle->mat_reader.name(i);
                if(name == "dimension" || name == "voxel_size" ||
                        name == "odf_vertices" || name == "odf_faces" || name == "trans")
                    mat_write.write(handle->mat_reader[i]);
                if(name == "fa0")
                    mat_write.write("qa_map",handle->fib.fa[0],1,handle->dim.size());
            }
            for(unsigned int i = 0;i < spm_maps[index].greater_ptr.size();++i)
            {
                std::ostringstream out1,out2;
                out1 << "fa" << i;
                out2 << "index" << i;
                mat_write.write(out1.str().c_str(),spm_maps[index].greater_ptr[i],1,handle->dim.size());
                mat_write.write(out2.str().c_str(),handle->fib.findex[i],1,handle->dim.size());
            }
        }

        if(length_threshold_lesser)
        {
            TractModel tracks(handle.get());
            tracks.add_tracts(lesser_tracks[index].get_tracts(),length_threshold_lesser);
            if(tracks.get_visible_track_count())
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".lesser.trk.gz";
                tracks.save_tracts_to_file(out1.str().c_str());
                saved_file_name.push_back(out1.str().c_str());
                has_lesser_result = true;
            }
            else
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".lesser.no_trk.txt";
                std::ofstream(out1.str().c_str());
            }

            lesser_tracks[index] = tracks;
        }

        {
            std::ostringstream out1;
            out1 << trk_file_names[index] << ".lesser.fib.gz";
            gz_mat_write mat_write(out1.str().c_str());
            for(unsigned int i = 0;i < handle->mat_reader.size();++i)
            {
                std::string name = handle->mat_reader.name(i);
                if(name == "dimension" || name == "voxel_size" ||
                        name == "odf_vertices" || name == "odf_faces" || name == "trans")
                    mat_write.write(handle->mat_reader[i]);
                if(name == "fa0")
                    mat_write.write("qa_map",handle->fib.fa[0],1,handle->dim.size());
            }
            for(unsigned int i = 0;i < spm_maps[index].greater_ptr.size();++i)
            {
                std::ostringstream out1,out2;
                out1 << "fa" << i;
                out2 << "index" << i;
                mat_write.write(out1.str().c_str(),spm_maps[index].lesser_ptr[i],1,handle->dim.size());
                mat_write.write(out2.str().c_str(),handle->fib.findex[i],1,handle->dim.size());
            }
        }

    }
}

void vbc_database::run_permutation(unsigned int thread_count)
{
    clear_thread();
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

    model->generator.seed(0);
    std::srand(0);
    total_count_null = 0;
    total_count = 0;
    greater_tracks.clear();
    lesser_tracks.clear();
    spm_maps.clear();
    has_greater_result = true;
    has_lesser_result = true;
    unsigned int num_subjects = (model->type == 2 ? individual_data.size():1);
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        greater_tracks.push_back(new TractModel(handle.get()));
        lesser_tracks.push_back(new TractModel(handle.get()));
        spm_maps.push_back(new fib_data);
    }
    threads.reset(new boost::thread_group);
    for(unsigned int index = 0;index < thread_count;++index)
        threads->add_thread(new boost::thread(&vbc_database::run_permutation_multithread,this,index));
}
void vbc_database::calculate_FDR(void)
{
    double sum_greater_null = 0;
    double sum_lesser_null = 0;
    double sum_greater = 0;
    double sum_lesser = 0;
    length_threshold_greater = 0;
    length_threshold_lesser = 0;
    for(int index = subject_greater_null.size()-1;index >= 0;--index)
    {
        sum_greater_null += subject_greater_null[index];
        sum_lesser_null += subject_lesser_null[index];
        sum_greater += subject_greater[index];
        sum_lesser += subject_lesser[index];
        fdr_greater[index] = (sum_greater > 0.0 && sum_greater_null > 0.0) ? std::min(1.0,sum_greater_null/sum_greater) : 1.0;
        fdr_lesser[index] = (sum_lesser > 0.0 && sum_lesser_null > 0.0) ? std::min(1.0,sum_lesser_null/sum_lesser): 1.0;
        if(fdr_greater[index] < fdr_threshold)
            length_threshold_greater = index;
        if(fdr_lesser[index] < fdr_threshold)
            length_threshold_lesser = index;
    }
    if(use_track_length)
    {
        length_threshold_greater = length_threshold;
        length_threshold_lesser = length_threshold;
    }
}
