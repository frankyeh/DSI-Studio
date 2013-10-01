#include <cstdlib>     /* srand, rand */
#include <ctime>
#include <boost/thread.hpp>
#include <boost/math/distributions/students_t.hpp>
#include "vbc_database.h"
#include "libs/tracking/tracking_model.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"

vbc_database::vbc_database():handle(0),num_subjects(0)
{
}

void vbc_database::read_template(void)
{
    dim = (handle->fib_data.dim);
    num_fiber = handle->fib_data.fib.fa.size();
    findex.resize(num_fiber);
    fa.resize(num_fiber);
    for(unsigned int index = 0;index < num_fiber;++index)
    {
        findex[index] = handle->fib_data.fib.findex[index];
        fa[index] = handle->fib_data.fib.fa[index];
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
    vertices = handle->fib_data.fib.odf_table;
    half_odf_size = vertices.size()/2;
}
bool vbc_database::create_database(const char* template_name)
{
    handle.reset(new ODFModel);
    if(!handle->fib_data.load_from_file(template_name))
    {
        error_msg = "Invalid fib file";
        return false;
    }
    read_template();
    return true;
}
bool vbc_database::load_database(const char* database_name)
{
    handle.reset(new ODFModel);
    if(!handle->fib_data.load_from_file(database_name))
    {
        error_msg = "Invalid fib file";
        return false;
    }
    read_template();
    // read databse data

    // does it contain subject info?
    gz_mat_read& matfile = handle->fib_data.mat_reader;
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
                in >> subject_names[index];
        }
        const float* r2_values = 0;
        matfile.read("R2",row,col,r2_values);
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
    std::vector<const float*> cur_fa(num_fiber);
    for(unsigned int fib = 0;fib < num_fiber;++fib)
    {
        std::ostringstream out;
        out << "fa" << fib;
        unsigned int row,col;
        mat_reader.read(out.str().c_str(),row,col,cur_fa[fib]);
        if (!cur_fa[fib])
        {
            error_msg = "Inconsistent fa number in subject fib file";
            return false;
        }

    }

    if(!subject_odf.has_odfs())
    {
        error_msg = "No ODF data in the subject file:";
        return false;
    }
    subject_odf.initializeODF(dim,cur_fa,half_odf_size);

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
    gz_mat_read& mat_source = handle->fib_data.mat_reader;
    for(unsigned int index = 0;index < mat_source.size();++index)
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
    begin_prog("output data");
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


void fib_data::initialize(ODFModel* handle)
{
    unsigned char num_fiber = handle->fib_data.fib.num_fiber;
    image::geometry<3> dim(handle->fib_data.dim);
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

void fib_data::add_greater_lesser_mapping_for_tracking(ODFModel* handle)
{

    unsigned int greater_index_id =
            std::find(handle->fib_data.fib.index_name.begin(),
                      handle->fib_data.fib.index_name.end(),
                      "greater")-handle->fib_data.fib.index_name.begin();
    if(greater_index_id == handle->fib_data.fib.index_name.size())
    {
        handle->fib_data.fib.index_name.push_back("greater");
        handle->fib_data.fib.index_data.push_back(std::vector<const float*>());
        handle->fib_data.fib.index_data_dir.push_back(std::vector<const short*>());
    }
    handle->fib_data.fib.index_data[greater_index_id] = greater_ptr;
    handle->fib_data.fib.index_data_dir[greater_index_id] = greater_dir_ptr;

    unsigned int lesser_index_id =
            std::find(handle->fib_data.fib.index_name.begin(),
                      handle->fib_data.fib.index_name.end(),
                      "lesser")-handle->fib_data.fib.index_name.begin();
    if(lesser_index_id == handle->fib_data.fib.index_name.size())
    {
        handle->fib_data.fib.index_name.push_back("lesser");
        handle->fib_data.fib.index_data.push_back(std::vector<const float*>());
        handle->fib_data.fib.index_data_dir.push_back(std::vector<const short*>());
    }
    handle->fib_data.fib.index_data[lesser_index_id] = lesser_ptr;
    handle->fib_data.fib.index_data_dir[lesser_index_id] = lesser_dir_ptr;
}


void vbc_database::single_subject_analysis(const float* cur_subject_data,
                                           float percentile,fib_data& result)
{
    result.initialize(handle.get());
    std::vector<unsigned char> greater_fib_count(dim.size()),lesser_fib_count(dim.size());
    std::vector<float> population;

    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;
            fib < num_fiber && fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            float cur_value = cur_subject_data[pos];
            if(cur_value == 0.0)
                continue;

            population.clear();
            for(unsigned int subject_id = 0;subject_id < subject_qa.size();++subject_id)
            {
                float value = subject_qa[subject_id][pos];
                if(value != 0.0)
                    population.push_back(value);
            }
            unsigned int greater_rank = 0;
            unsigned int lesser_rank = 0;
            for(unsigned int subject_id = 0;subject_id < population.size();++subject_id)
            {
                if(cur_value > population[subject_id])
                    ++greater_rank;
                if(cur_value < population[subject_id])
                    ++lesser_rank;
            }
            if(population.empty())
                continue;
            if(greater_rank > (population.size() >> 1)) // greater
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                result.greater[fib_count][cur_index] = (double)greater_rank/population.size();
                result.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
                if(result.greater[fib_count][cur_index] > percentile)
                    ++total_greater;
            }
            if(lesser_rank > (population.size() >> 1)) // lesser
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                result.lesser[fib_count][cur_index] = (double)lesser_rank/population.size();
                result.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
                if(result.lesser[fib_count][cur_index] > percentile)
                    ++total_lesser;
            }
            ++total;
        }
    }
}
bool vbc_database::single_subject_analysis(const char* filename,float percentile,fib_data& result)
{
    std::vector<float> cur_subject_data;
    if(!get_odf_profile(filename,cur_subject_data))
        return false;
    single_subject_analysis(&cur_subject_data[0],percentile,result);
    return true;
}

void vbc_database::run_track(const fiber_orientations& fib,std::vector<std::vector<float> >& tracks)
{
    std::vector<image::vector<3,short> > seed;
    for(image::pixel_index<3> index;index.valid(dim);index.next(dim))
        if(fib.fa[0][index.index()] > fib.threshold)
            seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
    if(seed.empty())
    {
        tracks.clear();
        return;
    }
    ThreadData tracking_thread;
    tracking_thread.param.step_size = fib.vs[0];
    tracking_thread.param.smooth_fraction = 0;
    tracking_thread.param.min_points_count3 = 6;
    tracking_thread.param.max_points_count3 = std::max<unsigned int>(6,3.0*500/tracking_thread.param.step_size);
    tracking_thread.tracking_method = 0;// streamline fiber tracking
    tracking_thread.initial_direction = 2;// all directions
    tracking_thread.interpolation_strategy = 0; // trilinear interpolation
    tracking_thread.stop_by_tract = 1;// stop by tract
    tracking_thread.center_seed = 1;// center seeded
    tracking_thread.setRegions(fib.dim,seed,3);
    tracking_thread.run(fib,1,seed.size()*10,true);
    tracks.swap(tracking_thread.track_buffer);
}
bool vbc_database::save_track_as(const char* file_name,std::vector<std::vector<float> >& track,unsigned int length_threshold)
{
    std::vector<std::vector<float> > new_track;
    for(unsigned int j = 0; j < track.size();++j)
    {
        if(track[j].size() > 3 && track[j].size()/3-1 > length_threshold)
        {
            new_track.push_back(std::vector<float>());
            new_track.back().swap(track[j]);
        }
    }
    if(new_track.empty())
    {
        std::string null_name(file_name);
        null_name += ".no_tracks.txt";
        std::ofstream out(null_name.c_str());
        out << std::endl;
        return true;
    }
    TractModel tract_model(handle.get());
    tract_model.add_tracts(new_track);
    return tract_model.save_tracts_to_file(file_name);
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

void hist_to_dist(const std::vector<unsigned int>& count,
                  std::vector<float>& dist)
{
    dist.resize(count.size());
    std::copy(count.begin(),count.end(),dist.begin());
    std::for_each(dist.begin(),dist.end(),boost::lambda::_1 /=
            std::accumulate(dist.begin(),dist.end(),0.0f));
}

void dist_to_cdf(std::vector<float>& dist)
{
    float sum = 1.0;
    for(unsigned int index = 0;index < dist.size();++index)
    {
        float value = dist[index];
        dist[index] = sum;
        sum -= value;
        if(sum < 0.0)
            sum = 0.0;
    }
}



void vbc_database::calculate_subject_distribution(float percentile,const fib_data& data,
                                                  std::vector<float>& subject_greater,
                                                  std::vector<float>& subject_lesser)
{
    // calculate subject fiber distribution
    std::vector<std::vector<float> > tracks;
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fiber_orientations fib;
    fib.read(handle->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    fib.fa = data.greater_ptr;
    fib.findex = data.greater_dir_ptr;

    run_track(fib,tracks);
    cal_hist(tracks,dist_greater);


    fib.fa = data.lesser_ptr;
    fib.findex = data.lesser_dir_ptr;

    run_track(fib,tracks);
    cal_hist(tracks,dist_lesser);

    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
}

bool vbc_database::save_subject_distribution(float percentile,
                               unsigned int length_threshold,
                               const char* file_name,
                               const fib_data& data)
{
    std::vector<std::vector<float> > tracks;
    fiber_orientations fib;
    fib.read(handle->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    fib.fa = data.greater_ptr;
    fib.findex = data.greater_dir_ptr;

    run_track(fib,tracks);
    {
        std::ostringstream out;
        out << file_name << ".g" << length_threshold << ".trk";
        if(!save_track_as(out.str().c_str(),tracks,length_threshold))
        {
            error_msg = "Cannot save trk file ";
            error_msg += out.str();
            return false;
        }
    }

    fib.fa = data.lesser_ptr;
    fib.findex = data.lesser_dir_ptr;

    run_track(fib,tracks);
    {
        std::ostringstream out;
        out << file_name << ".l" << length_threshold << ".trk";
        if(!save_track_as(out.str().c_str(),tracks,length_threshold))
        {
            error_msg = "Cannot save trk file ";
            error_msg += out.str();
            return false;
        }
    }
    return true;
}

bool vbc_database::calculate_individual_distribution(float percentile,
                                                unsigned int length_threshold,
                                                const std::vector<std::string>& files,
                                                std::vector<float>& subject_greater,
                                                std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<std::vector<float> > tracks;
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fib_data data;
    fiber_orientations fib;
    fib.read(handle->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    total_greater = 0;
    total_lesser = 0;
    total = 0;
    bool is_null = files.empty();
    unsigned int total_subject = is_null? subject_qa.size() : files.size();
    for(unsigned int main_index = 0;check_prog(main_index,total_subject);++main_index)
    {
        if(!is_null)
        {
            std::vector<float> cur_subject_data;
            if(!get_odf_profile(files[main_index].c_str(),cur_subject_data))
                return false;
            single_subject_analysis(&cur_subject_data[0],percentile,data);
            //single_subject_analysis(subject_qa[rand()%subject_qa.size()],percentile,data);
        }
        else
            single_subject_analysis(subject_qa[main_index],percentile,data);

        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        run_track(fib,tracks);
        if(!is_null && length_threshold)
        {
            std::ostringstream out;
            out << files[main_index] << ".g" << length_threshold << ".trk";
            if(!save_track_as(out.str().c_str(),tracks,length_threshold))
            {
                error_msg = "Cannot save trk file ";
                error_msg += out.str();
                return false;
            }
        }
        cal_hist(tracks,dist_greater);

        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        run_track(fib,tracks);
        if(!is_null && length_threshold)
        {
            std::ostringstream out;
            out << files[main_index] << ".l" << length_threshold << ".trk";
            if(!save_track_as(out.str().c_str(),tracks,length_threshold))
            {
                error_msg = "Cannot save trk file ";
                error_msg += out.str();
                return false;
            }
        }
        cal_hist(tracks,dist_lesser);
    }
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
    return true;
}


double vbc_database::get_trend_std(const std::vector<float>& data)
{
    std::map<float,unsigned int> accounting;
    for(unsigned int index = 0;index < data.size();++index)
        ++accounting[data[index]];

    std::vector<std::pair<float,unsigned int> > group_size(accounting.begin(),accounting.end());
    double n = subject_qa.size();
    double permutations = n*(n-1.0)*(2.0*n+5.0);

    for(unsigned int index = 0;index < group_size.size();++index)
        if(group_size[index].second > 1)
        {
            double t = group_size[index].second;
            permutations -= t*(t-1.0)*(2.0*t+5.0);
        }
    return std::sqrt(permutations/18.0);
}

void vbc_database::trend_analysis(float sqrt_var_S,const std::vector<unsigned int>& permu,fib_data& data)
{
    data.initialize(handle.get());

    boost::math::normal gaussian;
    std::vector<float> population(subject_qa.size());
    unsigned int n = subject_qa.size();
    std::vector<unsigned char> greater_fib_count(dim.size()),lesser_fib_count(dim.size());
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < num_fiber && fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            for(unsigned int subject_id = 0;subject_id < subject_qa.size();++subject_id)
                population[subject_id] = subject_qa[subject_id][pos];
            if(std::find(population.begin(),population.end(),0.0) != population.end())
                continue;

            int S = 0;
            for(unsigned int i = 0;i < n;++i)
                for(unsigned int j = i+1;j < n;++j)
                {
                    float v1 = population[permu[j]];
                    float v2 = population[permu[i]];
                    if(v1 > v2)
                        ++S;
                    else
                        if(v1 < v2)
                            --S;
                }
            float Z = ((S > 0) ? (float)(S-1):(float)(S+1))/sqrt_var_S;
            float p_value = Z < 0.0 ? boost::math::cdf(gaussian,Z):
                                boost::math::cdf(boost::math::complement(gaussian, Z));
            if(S > 0) // incremental
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                data.greater[fib_count][cur_index] = 1.0-p_value;
                data.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
            }
            if(S < 0) // decremental
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                data.lesser[fib_count][cur_index] = 1.0-p_value;
                data.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
            }
        }
    }

}


void vbc_database::group_analysis(const std::vector<int>& label,fib_data& data)
{
    data.initialize(handle.get());
    std::vector<unsigned int> group1,group2;
    std::vector<float> g1,g2;
    for(unsigned int index = 0;index < label.size();++index)
        if(label[index])
            group2.push_back(index);
        else
            group1.push_back(index);
    std::vector<unsigned char> greater_fib_count(dim.size()),lesser_fib_count(dim.size());
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < num_fiber && fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            g1.clear();g2.clear();
            for(unsigned int subject_id = 0;subject_id < group1.size();++subject_id)
            {
                float value = subject_qa[group1[subject_id]][pos];
                if(value != 0.0)
                    g1.push_back(value);
            }
            for(unsigned int subject_id = 0;subject_id < group2.size();++subject_id)
            {
                float value = subject_qa[group2[subject_id]][pos];
                if(value != 0.0)
                    g2.push_back(value);
            }
            if(g1.size() < 6 || g2.size() < 6)
                continue;
            float mean1 = image::mean(g1.begin(),g1.end());
            float mean2 = image::mean(g2.begin(),g2.end());
            float v1 = image::variance(g1.begin(),g1.end(),mean1)/g1.size();
            float v2 = image::variance(g2.begin(),g2.end(),mean2)/g2.size();
            float v = v1 + v2;
            v *= v;
            v /= (v1*v1/(g1.size()-1) + v2*v2/(g2.size()-1));
            float t_stat = (mean1 - mean2) / std::sqrt(v1+v2);
            boost::math::students_t dist(v);
            if(t_stat > 0.0) // group 1 > group 2
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                data.greater[fib_count][cur_index] = 1.0-boost::math::cdf(boost::math::complement(dist, t_stat));
                data.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
            }
            if(t_stat < 0.0) // group 1 < group 2
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                data.lesser[fib_count][cur_index] = 1.0-boost::math::cdf(dist, t_stat);
                data.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
            }
        }
    }
}
void vbc_database::trend_analysis(const std::vector<float>& data,fib_data& result)
{
    std::multimap<float,unsigned int> rank;
    for(unsigned int index = 0;index < data.size();++index)
        rank.insert(std::make_pair(data[index],index));
    std::vector<std::pair<float,unsigned int> > sorted(rank.begin(),rank.end());
    std::vector<unsigned int> permu(sorted.size());
    for(unsigned int index = 0;index < sorted.size();++index)
        permu[index] = sorted[index].second;
    trend_analysis(get_trend_std(data),permu,result);
}
void vbc_database::calculate_null_trend_multithread(unsigned int id,float sqrt_var_S,float percentile,
                                               std::vector<unsigned int>& dist_greater,
                                               std::vector<unsigned int>& dist_lesser,
                                                    bool progress,
                                                    unsigned int* total_count)
{
    fib_data data;
    fiber_orientations fib;
    fib.read(handle->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    std::srand(1000+id);
    std::vector<unsigned int> permu(subject_qa.size());
    for(unsigned int index = 0;index < permu.size();++index)
        permu[index] = index;

    std::vector<std::vector<float> > tracks;
    for(;*total_count < 1000;++(*total_count))
    {
        if(progress)
            check_prog(*total_count,1000);
        if(prog_aborted())
            break;
        std::random_shuffle(permu.begin(),permu.end());

        trend_analysis(sqrt_var_S,permu,data);

        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        run_track(fib,tracks);
        cal_hist(tracks,dist_greater);

        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        run_track(fib,tracks);
        cal_hist(tracks,dist_lesser);

    }
}

void vbc_database::calculate_null_group_multithread(unsigned int id,
                                      const std::vector<int>& label,float dif,
                                      std::vector<unsigned int>& dist_greater,
                                      std::vector<unsigned int>& dist_lesser,
                                      bool progress,
                                      unsigned int* total_count)
{
    fib_data data;
    fiber_orientations fib;
    fib.read(handle->fib_data);
    fib.threshold = dif;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    std::srand(1000+id);
    std::vector<int> new_label(label);
    std::vector<std::vector<float> > tracks;
    for(;*total_count < 1000;++(*total_count))
    {
        if(progress)
            check_prog(*total_count,1000);
        if(prog_aborted())
            break;
        std::random_shuffle(new_label.begin(),new_label.end());
        group_analysis(new_label,data);

        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        run_track(fib,tracks);
        cal_hist(tracks,dist_greater);

        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        run_track(fib,tracks);
        cal_hist(tracks,dist_lesser);

    }
}

void vbc_database::calculate_null_trend_distribution(float sqrt_var_S,float percentile,
                                               std::vector<float>& subject_greater,
                                               std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    boost::thread_group threads;
    unsigned int total_count = 0;
    for(unsigned int index = 0;index < 6;++index)
        threads.add_thread(new boost::thread(&vbc_database::calculate_null_trend_multithread,
                                             this,index,sqrt_var_S,percentile,dist_greater,dist_lesser,false,&total_count));
    calculate_null_trend_multithread(6,sqrt_var_S,percentile,dist_greater,dist_lesser,true,&total_count);
    threads.join_all();
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
    check_prog(0,0);
}

void vbc_database::calculate_null_group_distribution(const std::vector<int>& label,float dif,
                                               std::vector<float>& subject_greater,
                                               std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    boost::thread_group threads;
    unsigned int total_count = 0;
    for(unsigned int index = 0;index < 6;++index)
        threads.add_thread(new boost::thread(&vbc_database::calculate_null_group_multithread,
                                             this,index,label,dif,dist_greater,dist_lesser,false,&total_count));
    calculate_null_group_multithread(6,label,dif,dist_greater,dist_lesser,true,&total_count);
    threads.join_all();
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
    check_prog(0,0);
}



/*
bool vbc_database::single_subject_paired_analysis(const char* file_name1,const char* file_name2)
{
    std::vector<float> cur_subject_data1,cur_subject_data2;
    if(!get_odf_profile(file_name1,cur_subject_data1) ||
       !get_odf_profile(file_name2,cur_subject_data2))
        return false;
    initialize_greater_lesser();

    float max_value = *std::max_element(cur_subject_data1.begin(),cur_subject_data1.end());
    image::minus(cur_subject_data1.begin(),cur_subject_data1.end(),cur_subject_data2.begin());
    image::divide_constant(cur_subject_data1,max_value);

    std::vector<unsigned char> greater_fib_count(dim.size()),lesser_fib_count(dim.size());

    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;
            fib < num_fiber && fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            float cur_value = cur_subject_data1[pos];
            if(cur_value == 0.0)
                continue;

            if(cur_value > 0.0) // greater
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                greater[fib_count][cur_index] = cur_value;
                greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
            }
            if(cur_value < 0.0) // lesser
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                lesser[fib_count][cur_index] = -cur_value;
                lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
            }
        }
    }
    add_greater_lesser_mapping_for_tracking();
    return true;
}
*/
