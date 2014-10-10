#include <cstdlib>     /* srand, rand */
#include <ctime>
#include <boost/thread.hpp>
#include <boost/math/distributions/students_t.hpp>
#include "vbc_database.h"
#include "fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"



vbc_database::vbc_database():handle(0),num_subjects(0)
{
}

void vbc_database::read_template(void)
{
    dim = (handle->dim);
    num_fiber = handle->fib.fa.size();
    findex.resize(num_fiber);
    fa.resize(num_fiber);
    for(unsigned int index = 0;index < num_fiber;++index)
    {
        findex[index] = handle->fib.findex[index];
        fa[index] = handle->fib.fa[index];
    }
    fiber_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(dim,fa[0]));
    vi2si.resize(dim.size());
    for(unsigned int index = 0;index < dim.size();++index)
    {
        if(fa[0][index] != 0.0)
        {
            vi2si[index] = si2vi.size();
            si2vi.push_back(index);
        }
    }
    vertices = handle->fib.odf_table;
    half_odf_size = vertices.size()/2;
}
bool vbc_database::create_database(const char* template_name)
{
    handle.reset(new FibData);
    if(!handle->load_from_file(template_name))
        return false;
    read_template();
    return true;
}
bool vbc_database::load_database(const char* database_name)
{
    handle.reset(new FibData);
    if(!handle->load_from_file(database_name))
    {
        error_msg = "Invalid fib file";
        return false;
    }
    read_template();
    // read databse data

    // does it contain subject info?
    gz_mat_read& matfile = handle->mat_reader;
    subject_qa.clear();
    subject_qa_buffer.clear();
    unsigned int row,col;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        const float* buf = 0;
        matfile.read(out.str().c_str(),row,col,buf);
        if (!buf)
            break;
        subject_qa.push_back(buf);
    }
    num_subjects = subject_qa.size();
    subject_names.resize(num_subjects);
    R2.resize(num_subjects);
    if(num_subjects)
    {
        const char* str = 0;
        matfile.read("subject_names",row,col,str);
        if(str)
        {
            std::istringstream in(str);
            for(unsigned int index = 0;in && index < num_subjects;++index)
                std::getline(in,subject_names[index]);
        }
        const float* r2_values = 0;
        matfile.read("R2",row,col,r2_values);
        if(r2_values == 0)
        {
            error_msg = "Memory insufficiency. Use 64-bit program instead";
            return false;
        }
        std::copy(r2_values,r2_values+num_subjects,R2.begin());
    }
    return !subject_qa.empty();
}


void vbc_database::get_subject_slice(unsigned int subject_index,
                                     unsigned int z_pos,
                                     image::basic_image<float,2>& slice) const
{
    slice.clear();
    slice.resize(image::geometry<2>(dim.width(),dim.height()));
    unsigned int slice_offset = z_pos*dim.plane_size();
    for(unsigned int index = 0;index < slice.size();++index)
    {
        unsigned int cur_index = index + slice_offset;
        if(fa[0][cur_index] == 0.0)
            continue;
        slice[index] = subject_qa[subject_index][vi2si[cur_index]];
    }
}

void vbc_database::get_data_at(unsigned int index,unsigned int fib,std::vector<float>& data) const
{
    data.clear();
    if(index >= dim.size() || fa[0][index] == 0.0)
        return;
    unsigned int s_index = vi2si[index];
    unsigned int fib_offset = fib*si2vi.size();
    data.resize(num_subjects);
    for(unsigned int index = 0;index < num_subjects;++index)
        data[index] = subject_qa[index][s_index+fib_offset];
}
bool vbc_database::is_consistent(gz_mat_read& mat_reader) const
{
    unsigned int row,col;
    const float* odf_buffer;
    mat_reader.read("odf_vertices",row,col,odf_buffer);
    if (!odf_buffer)
    {
        error_msg = "Invalid subject data format in ";
        return false;
    }
    if(col != vertices.size())
    {
        error_msg = "Inconsistent ODF dimension in ";
        return false;
    }
    for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
    {
        if(vertices[index][0] != odf_buffer[0] ||
           vertices[index][1] != odf_buffer[1] ||
           vertices[index][2] != odf_buffer[2])
        {
            error_msg = "Inconsistent ODF dimension in ";
            return false;
        }
    }
    return true;
}
bool vbc_database::sample_odf(gz_mat_read& mat_reader,std::vector<float>& data)
{
    ODFData subject_odf;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "odf" << index;
        const float* odf = 0;
        unsigned int row,col;
        mat_reader.read(out.str().c_str(),row,col,odf);
        if (!odf)
            break;
        subject_odf.setODF(index,odf,row*col);
    }

    const float* fa0 = 0;
    unsigned int row,col;
    mat_reader.read("fa0",row,col,fa0);
    if (!fa0)
    {
        error_msg = "Invalid file format. Cannot find fa0 matrix in ";
        return false;
    }

    if(!subject_odf.has_odfs())
    {
        error_msg = "No ODF data in the subject file:";
        return false;
    }
    subject_odf.initializeODF(dim,fa0,half_odf_size);

    set_title("load data");
    for(unsigned int index = 0;index < si2vi.size();++index)
    {
        unsigned int cur_index = si2vi[index];
        const float* odf = subject_odf.get_odf_data(cur_index);
        if(odf == 0)
            continue;
        float min_value = *std::min_element(odf, odf + half_odf_size);
        unsigned int pos = index;
        for(unsigned char fib = 0;fib < num_fiber;++fib,pos += si2vi.size())
        {
            if(fa[fib][cur_index] == 0.0)
                break;
            // 0: subject index 1:findex by s_index (fa > 0)
            data[pos] = odf[findex[fib][cur_index]]-min_value;
        }
    }
    return true;
}

bool vbc_database::load_subject_files(const std::vector<std::string>& file_names,
                                      const std::vector<std::string>& subject_names_)
{
    if(!handle.get())
        return false;
    num_subjects = file_names.size();
    subject_qa.clear();
    subject_qa_buffer.clear();
    subject_qa.resize(num_subjects);
    subject_qa_buffer.resize(num_subjects);
    R2.resize(num_subjects);
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        subject_qa_buffer[index].resize(num_fiber*si2vi.size());
        subject_qa[index] = &*(subject_qa_buffer[index].begin());
    }
    // load subject data
    for(unsigned int subject_index = 0;check_prog(subject_index,num_subjects);++subject_index)
    {
        gz_mat_read mat_reader;
        if(!mat_reader.load_from_file(file_names[subject_index].c_str()))
        {
            error_msg = "failed to load subject data ";
            error_msg += file_names[subject_index];
            return false;
        }
        // check if the odf table is consistent or not
        if(!is_consistent(mat_reader) ||
           !sample_odf(mat_reader,subject_qa_buffer[subject_index]))
        {
            error_msg += file_names[subject_index];
            return false;
        }
        // load R2
        const float* value= 0;
        unsigned int row,col;
        mat_reader.read("R2",row,col,value);
        if(value)
            R2[subject_index] = *value;
        if(subject_index == 0)
        {
            const char* report_buf = 0;
            if(mat_reader.read("report",row,col,report_buf))
                subject_report = std::string(report_buf,report_buf+row*col);
        }
    }
    subject_names = subject_names_;
    return true;
}
void vbc_database::save_subject_data(const char* output_name) const
{
    if(!handle.get())
        return;
    // store results
    gz_mat_write matfile(output_name);
    if(!matfile)
    {
        error_msg = "Cannot output file";
        return;
    }
    gz_mat_read& mat_source = handle->mat_reader;
    for(unsigned int index = 0;index < mat_source.size();++index)
        if(mat_source[index].get_name() != "report" &&
           mat_source[index].get_name().find("subject") != 0)
            matfile.write(mat_source[index]);
    for(unsigned int index = 0;check_prog(index,subject_qa.size());++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        matfile.write(out.str().c_str(),subject_qa[index],num_fiber,si2vi.size());
    }
    std::string name_string;
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        name_string += subject_names[index];
        name_string += "\n";
    }
    matfile.write("subject_names",name_string.c_str(),1,name_string.size());
    matfile.write("R2",&*R2.begin(),1,R2.size());

    {
        std::ostringstream out;
        out << "A total of " << num_subjects << " subjects were included in the connectometry database." << subject_report.c_str();
        std::string report = out.str();
        matfile.write("report",&*report.c_str(),1,report.length());
    }
}

bool vbc_database::get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data)
{
    gz_mat_read single_subject;
    if(!single_subject.load_from_file(file_name))
    {
        error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_consistent(single_subject))
    {
        error_msg = "Inconsistent ODF dimension";
        return false;
    }
    cur_subject_data.clear();
    cur_subject_data.resize(num_fiber*si2vi.size());
    if(!sample_odf(single_subject,cur_subject_data))
    {
        error_msg += file_name;
        return false;
    }
    return true;
}


void fib_data::initialize(FibData* handle)
{
    unsigned char num_fiber = handle->fib.num_fiber;
    image::geometry<3> dim(handle->dim);
    greater.resize(num_fiber);
    lesser.resize(num_fiber);
    greater_dir.resize(num_fiber);
    lesser_dir.resize(num_fiber);
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        greater[fib].resize(dim.size());
        lesser[fib].resize(dim.size());
        greater_dir[fib].resize(dim.size());
        lesser_dir[fib].resize(dim.size());
    }
    greater_ptr.resize(num_fiber);
    lesser_ptr.resize(num_fiber);
    greater_dir_ptr.resize(num_fiber);
    lesser_dir_ptr.resize(num_fiber);
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        greater_ptr[fib] = &greater[fib][0];
        lesser_ptr[fib] = &lesser[fib][0];
        greater_dir_ptr[fib] = &greater_dir[fib][0];
        lesser_dir_ptr[fib] = &lesser_dir[fib][0];
    }
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        std::fill(greater[fib].begin(),greater[fib].end(),0.0);
        std::fill(lesser[fib].begin(),lesser[fib].end(),0.0);
        std::fill(greater_dir[fib].begin(),greater_dir[fib].end(),0);
        std::fill(lesser_dir[fib].begin(),lesser_dir[fib].end(),0);
    }
}

void fib_data::add_greater_lesser_mapping_for_tracking(FibData* handle)
{

    unsigned int greater_index_id =
            std::find(handle->fib.index_name.begin(),
                      handle->fib.index_name.end(),
                      "greater")-handle->fib.index_name.begin();
    if(greater_index_id == handle->fib.index_name.size())
    {
        handle->fib.index_name.push_back("greater");
        handle->fib.index_data.push_back(std::vector<const float*>());
        handle->fib.index_data_dir.push_back(std::vector<const short*>());
    }
    handle->fib.index_data[greater_index_id] = greater_ptr;
    handle->fib.index_data_dir[greater_index_id] = greater_dir_ptr;

    unsigned int lesser_index_id =
            std::find(handle->fib.index_name.begin(),
                      handle->fib.index_name.end(),
                      "lesser")-handle->fib.index_name.begin();
    if(lesser_index_id == handle->fib.index_name.size())
    {
        handle->fib.index_name.push_back("lesser");
        handle->fib.index_data.push_back(std::vector<const float*>());
        handle->fib.index_data_dir.push_back(std::vector<const short*>());
    }
    handle->fib.index_data[lesser_index_id] = lesser_ptr;
    handle->fib.index_data_dir[lesser_index_id] = lesser_dir_ptr;
}



void vbc_database::run_track(const fiber_orientations& fib,std::vector<std::vector<float> >& tracks,float seed_ratio)
{
    std::vector<image::vector<3,short> > seed;
    for(image::pixel_index<3> index;index.is_valid(dim);index.next(dim))
        if(fib.fa[0][index.index()] > fib.threshold)
            seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
    if(seed.empty())
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
    tracking_thread.setRegions(fib.dim,seed,3);
    tracking_thread.run(fib,1,seed.size()*seed_ratio,true);
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
        return mr.set_variables(&*X.begin(),feature_count,X.size()/feature_count);
    case 2:
        return true;
    case 3: // paired
        return pre.size() == post.size() && !pre.empty();
    }
    return false;
}
bool stat_model::resample(const stat_model& rhs,std::vector<unsigned int>& permu,bool null)
{
    type = rhs.type;
    feature_count = rhs.feature_count;
    study_feature = rhs.study_feature;
    unsigned int trial = 0;
    do
    {
        if(trial > 100)
            throw std::runtime_error("Invalid subject demographics for multiple regression");
        ++trial;

        switch(type)
        {
        case 0: // group
            permu.resize(rhs.label.size());
            label.resize(rhs.label.size());
            for(unsigned int index = 0;index < rhs.label.size();++index)
                label[index] = rhs.label[permu[index] = rhs.rand_gen(rhs.label.size())];
            if(null)
                std::random_shuffle(permu.begin(),permu.end(),rhs.rand_gen);
            break;
        case 1: // multiple regression
            permu.resize(rhs.X.size()/rhs.feature_count);
            X.resize(rhs.X.size());
            for(unsigned int index = 0,pos = 0;index < permu.size();++index,pos += feature_count)
            {
                permu[index] = rhs.rand_gen(permu.size());
                std::copy(rhs.X.begin()+permu[index]*feature_count,
                          rhs.X.begin()+permu[index]*feature_count+feature_count,X.begin()+pos);
            }
            if(null)
                std::random_shuffle(permu.begin(),permu.end(),rhs.rand_gen);
            break;
        case 3: // paired
            pre.resize(rhs.pre.size());
            post.resize(rhs.post.size());
            for(unsigned int index = 0;index < rhs.pre.size();++index)
            {
                unsigned int sel = rhs.rand_gen(rhs.pre.size());
                pre[index] = rhs.pre[sel];
                post[index] = rhs.post[sel];
                if(null && rhs.rand_gen(2) == 1)
                    std::swap(pre[index],post[index]);
            }
            permu.clear();
            std::copy(pre.begin(),pre.end(),std::back_inserter(permu));
            std::copy(post.begin(),post.end(),std::back_inserter(permu));
            break;
        }
    }while(!pre_process());

    return true;
}

double stat_model::operator()(const std::vector<double>& population,unsigned int pos) const
{

    switch(type)
    {
    case 0: // group
        {
        float sum1 = 0.0;
        float sum2 = 0.0;
        for(unsigned int index = 0;index < population.size();++index)
            if(label[index]) // group 1
                sum2 += population[index];
            else
                // group 0
                sum1 += population[index];
        float mean1 = sum1/((double)group1_count);
        float mean2 = sum2/((double)group2_count);
        float result = (mean1 + mean2);
        if(result != 0.0)
            result = (mean1 - mean2) / result;
        return result;
        }
        break;
    case 1: // multiple regression
        {
            std::vector<double> b(feature_count),t(feature_count);
            mr.regress(&*population.begin(),&*b.begin(),&*t.begin());
            return t[study_feature];
        }
        break;
    case 2: // individual
        {
            float value = individual_data[pos];
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
            return (g1-g2)/(g1+g2);
        }
        break;
    }

    return 0.0;
}

void vbc_database::calculate_spm(const stat_model& info,fib_data& data,const std::vector<unsigned int>& permu)
{
    data.initialize(handle.get());
    std::vector<double> selected_population(permu.size());
    std::vector<unsigned char> greater_fib_count(dim.size()),lesser_fib_count(dim.size());
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < num_fiber && fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            for(unsigned int index = 0;index < permu.size();++index)
                selected_population[index] = subject_qa[permu[index]][pos];

            if(std::find(selected_population.begin(),selected_population.end(),0.0) != selected_population.end())
                continue;
            double result = info(selected_population,pos);

            if(result > 0.0) // group 0 > group 1
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                data.greater[fib_count][cur_index] = result;
                data.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
                for(char j = fib_count;j;--j)
                    if(data.greater[j][cur_index] > data.greater[j-1][cur_index])
                    {
                        std::swap(data.greater[j][cur_index],data.greater[j-1][cur_index]);
                        std::swap(data.greater_dir[j][cur_index],data.greater_dir[j-1][cur_index]);
                    }
            }
            if(result < 0.0) // group 0 < group 1
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                data.lesser[fib_count][cur_index] = -result;
                data.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
                for(char j = fib_count;j;--j)
                    if(data.lesser[j][cur_index] > data.lesser[j-1][cur_index])
                    {
                        std::swap(data.lesser[j][cur_index],data.lesser[j-1][cur_index]);
                        std::swap(data.lesser_dir[j][cur_index],data.lesser_dir[j-1][cur_index]);
                    }
            }
        }
    }
}


bool vbc_database::read_subject_data(const std::vector<std::string>& files,std::vector<std::vector<float> >& data)
{
    begin_prog("reading");
    data.resize(files.size());
    for(unsigned int index = 0;check_prog(index,files.size());++index)
        if(!get_odf_profile(files[index].c_str(),data[index]))
        {
            error_msg = "Cannot read file ";
            error_msg += files[index];
            check_prog(0,0);
            return false;
        }
    return true;
}

void vbc_database::run_permutation_multithread(unsigned int id)
{
    fib_data data;
    fiber_orientations fib;
    fib.read(*handle);
    fib.threshold = tracking_threshold;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    std::vector<std::vector<float> > tracks;
    bool null = true;
    unsigned int num_subject = (model.type == 2 ? individual_data.size():1);
    while(total_count < permutation_count && !terminated)
    {
        if(null)
            ++total_count_null;
        else
            ++total_count;

        for(unsigned int subject_id = 0;subject_id < num_subject && !terminated;++subject_id)
        {
            switch(model.type)
            {
            case 0: // grouop
            case 1: // mr
            case 3: // paired
                {
                    std::vector<unsigned int> permu;
                    stat_model resampled_info;
                    {
                        boost::mutex::scoped_lock lock(lock_resampling);
                        resampled_info.resample(model,permu,null);
                    }
                    calculate_spm(resampled_info,data,permu);
                }
                break;
            case 2: // individual
                {
                    std::vector<unsigned int> permu(200);
                    unsigned int random_subject_id = 0;
                    {
                        boost::mutex::scoped_lock lock(lock_resampling);
                        random_subject_id = model.rand_gen(subject_qa.size());
                        for(unsigned int index = 0;index < permu.size();++index)
                            permu[index] = model.rand_gen(subject_qa.size());
                    }
                    stat_model resampled_info = model;
                    if(null)
                        resampled_info.individual_data = subject_qa[random_subject_id];
                    else
                        resampled_info.individual_data = &(individual_data[subject_id][0]);

                    calculate_spm(resampled_info,data,permu);
                }
                break;
            }

            fib.fa = data.lesser_ptr;
            fib.findex = data.lesser_dir_ptr;
            run_track(fib,tracks,10.0/permutation_count);
            if(null)
                cal_hist(tracks,subject_lesser_null);
            else
            {
                cal_hist(tracks,subject_lesser);
                {
                    boost::mutex::scoped_lock lock(lock_lesser_tracks);
                    lesser_tracks[subject_id].add_tracts(tracks,40);
                    tracks.clear();
                }
            }

            fib.fa = data.greater_ptr;
            fib.findex = data.greater_dir_ptr;
            run_track(fib,tracks,10.0/permutation_count);
            if(null)
                cal_hist(tracks,subject_greater_null);
            else
            {
                cal_hist(tracks,subject_greater);
                {
                    boost::mutex::scoped_lock lock(lock_greater_tracks);
                    greater_tracks[subject_id].add_tracts(tracks,40);
                    tracks.clear();
                }
            }
            }

        null = !null;
    }
}
void vbc_database::clear_thread(void)
{
    if(threads.get())
    {
        terminated = true;
        threads->join_all();
        threads.reset(0);
    }
}
void vbc_database::save_tracks_files(std::vector<std::string>& saved_file_name)
{
    threads->join_all();
    if(trk_file_names.size() != greater_tracks.size())
        throw std::runtime_error("Please assign file name for saving trk files.");
    saved_file_name.clear();
    for(unsigned int index = 0;index < greater_tracks.size();++index)
    {
        if(fdr_greater[length_threshold] < 0.5)
        {
            TractModel tracks(handle.get());
            tracks.add_tracts(greater_tracks[index].get_tracts(),length_threshold);
            for(unsigned int j = 0;j < pruning;++j)
            {
                unsigned int track_count = tracks.get_visible_track_count();
                tracks.trim();
                if(track_count == tracks.get_visible_track_count())
                    break;
            }
            if(tracks.get_visible_track_count())
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".greater" << length_threshold << ".trk.gz";
                tracks.save_tracts_to_file(out1.str().c_str());
                saved_file_name.push_back(out1.str().c_str());
            }
            greater_tracks[index] = tracks;
        }

        if(fdr_lesser[length_threshold] < 0.5)
        {
            TractModel tracks(handle.get());
            tracks.add_tracts(lesser_tracks[index].get_tracts(),length_threshold);
            for(unsigned int j = 0;j < pruning;++j)
            {
                unsigned int track_count = tracks.get_visible_track_count();
                tracks.trim();
                if(track_count == tracks.get_visible_track_count())
                    break;
            }
            if(tracks.get_visible_track_count())
            {
                std::ostringstream out1;
                out1 << trk_file_names[index] << ".lesser" << length_threshold << ".trk.gz";
                tracks.save_tracts_to_file(out1.str().c_str());
                saved_file_name.push_back(out1.str().c_str());
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

    model.generator.seed(0);
    total_count_null = 0;
    total_count = 0;
    greater_tracks.clear();
    lesser_tracks.clear();
    unsigned int num_subjects = (model.type == 2 ? individual_data.size():1);
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        greater_tracks.push_back(new TractModel(handle.get()));
        lesser_tracks.push_back(new TractModel(handle.get()));
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
    for(int index = subject_greater_null.size()-1;index >= 0;--index)
    {
        sum_greater_null += subject_greater_null[index];
        sum_lesser_null += subject_lesser_null[index];
        sum_greater += subject_greater[index];
        sum_lesser += subject_lesser[index];
        fdr_greater[index] = (sum_greater > 0.0) ? std::min(1.0,sum_greater_null/sum_greater) : 1.0;
        fdr_lesser[index] = (sum_lesser > 0.0) ? std::min(1.0,sum_lesser_null/sum_lesser): 1.0;
    }

}
