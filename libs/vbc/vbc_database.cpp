#include "vbc_database.h"
#include "libs/tracking/tracking_model.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"

vbc_database::vbc_database():fib_file(0),num_subjects(0)
{
}
bool vbc_database::load_template(const char* template_name)
{
    fib_file_buffer.reset(new ODFModel);
    if(!fib_file_buffer->fib_data.load_from_file(template_name))
    {
        error_msg = "Invalid fib file";
        return false;
    }
    read_template(fib_file_buffer.get());
    return true;
}

void vbc_database::read_template(ODFModel* fib_file_)
{
    fib_file = fib_file_;
    dim = (fib_file->fib_data.dim);
    num_fiber = fib_file->fib_data.fib.fa.size();
    findex.resize(num_fiber);
    fa.resize(num_fiber);
    for(unsigned int index = 0;index < num_fiber;++index)
    {
        findex[index] = fib_file->fib_data.fib.findex[index];
        fa[index] = fib_file->fib_data.fib.fa[index];
    }
    fiber_threshold = 0.6*image::segmentation::otsu_threshold(
        image::basic_image<float, 3,image::const_pointer_memory<float> >(fa[0],dim));
    vi2si.resize(dim.size());
    for(unsigned int index = 0;index < dim.size();++index)
    {
        if(fa[0][index] != 0.0)
        {
            vi2si[index] = si2vi.size();
            si2vi.push_back(index);
        }
    }
    vertices = fib_file->fib_data.fib.odf_table;
    half_odf_size = vertices.size()/2;
}
bool vbc_database::read_database(ODFModel* fib_file_)
{
    read_template(fib_file_);
    // does it contain subject info?
    MatFile& matfile = fib_file->fib_data.mat_reader;
    subject_qa.clear();
    subject_qa_buffer.clear();
    unsigned int row,col;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        const float* buf = 0;
        matfile.get_matrix(out.str().c_str(),row,col,buf);
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
        matfile.get_matrix("subject_names",row,col,str);
        if(str)
        {
            std::istringstream in(str);
            for(unsigned int index = 0;in && index < num_subjects;++index)
                in >> subject_names[index];
        }
        const float* r2_values = 0;
        matfile.get_matrix("R2",row,col,r2_values);
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
bool vbc_database::is_consistent(MatFile& mat_reader) const
{
    unsigned int row,col;
    const float* odf_buffer;
    mat_reader.get_matrix("odf_vertices",row,col,odf_buffer);
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
bool vbc_database::sample_odf(MatFile& mat_reader,std::vector<float>& data)
{
    ODFData subject_odf;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "odf" << index;
        const float* odf = 0;
        unsigned int row,col;
        mat_reader.get_matrix(out.str().c_str(),row,col,odf);
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
        mat_reader.get_matrix(out.str().c_str(),row,col,cur_fa[fib]);
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
    if(!fib_file)
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
        MatFile mat_reader;
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
        mat_reader.get_matrix("R2",row,col,value);
        if(value)
            R2[subject_index] = *value;
    }
    subject_names = subject_names_;
    return true;
}
void vbc_database::save_subject_data(const char* output_name) const
{
    if(!fib_file)
        return;
    // store results
    MatFile& matfile = fib_file->fib_data.mat_reader;
    matfile.write_to_file(output_name);
    for(unsigned int index = 0;check_prog(index,subject_qa.size());++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        matfile.add_matrix(out.str().c_str(),subject_qa[index],num_fiber,si2vi.size());
    }
    std::string name_string;
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        name_string += subject_names[index];
        name_string += "\n";
    }
    matfile.add_matrix("subject_names",name_string.c_str(),1,name_string.size());
    matfile.add_matrix("R2",&*R2.begin(),1,R2.size());
    begin_prog("output data");
    matfile.close_file();
}

bool vbc_database::get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data)
{
    std::auto_ptr<MatFile> single_subject(new MatFile);
    if(!single_subject->load_from_file(file_name))
    {
        error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_consistent(*single_subject.get()))
    {
        error_msg = "Inconsistent ODF dimension";
        return false;
    }
    cur_subject_data.clear();
    cur_subject_data.resize(num_fiber*si2vi.size());
    if(!sample_odf(*single_subject.get(),cur_subject_data))
    {
        error_msg += file_name;
        return false;
    }
    return true;
}


void fib_data::initialize(ODFModel* fib_file)
{
    unsigned char num_fiber = fib_file->fib_data.fib.num_fiber;
    image::geometry<3> dim(fib_file->fib_data.dim);
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

void fib_data::add_greater_lesser_mapping_for_tracking(ODFModel* fib_file)
{

    unsigned int greater_index_id =
            std::find(fib_file->fib_data.fib.index_name.begin(),
                      fib_file->fib_data.fib.index_name.end(),
                      "greater mapping")-fib_file->fib_data.fib.index_name.begin();
    if(greater_index_id == fib_file->fib_data.fib.index_name.size())
    {
        fib_file->fib_data.fib.index_name.push_back("greater mapping");
        fib_file->fib_data.fib.index_data.push_back(std::vector<const float*>());
        fib_file->fib_data.fib.index_data_dir.push_back(std::vector<const short*>());
    }
    fib_file->fib_data.fib.index_data[greater_index_id] = greater_ptr;
    fib_file->fib_data.fib.index_data_dir[greater_index_id] = greater_dir_ptr;

    unsigned int lesser_index_id =
            std::find(fib_file->fib_data.fib.index_name.begin(),
                      fib_file->fib_data.fib.index_name.end(),
                      "lesser mapping")-fib_file->fib_data.fib.index_name.begin();
    if(lesser_index_id == fib_file->fib_data.fib.index_name.size())
    {
        fib_file->fib_data.fib.index_name.push_back("lesser mapping");
        fib_file->fib_data.fib.index_data.push_back(std::vector<const float*>());
        fib_file->fib_data.fib.index_data_dir.push_back(std::vector<const short*>());
    }
    fib_file->fib_data.fib.index_data[lesser_index_id] = lesser_ptr;
    fib_file->fib_data.fib.index_data_dir[lesser_index_id] = lesser_dir_ptr;
}


bool vbc_database::single_subject_analysis(const char* file_name,fib_data& result)
{
    std::vector<float> cur_subject_data;
    if(!get_odf_profile(file_name,cur_subject_data))
        return false;
    result.initialize(fib_file);
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
            if(greater_rank > (population.size() >> 1)) // greater
            {
                unsigned char fib_count = greater_fib_count[cur_index];
                result.greater[fib_count][cur_index] = (double)greater_rank/(population.size()+1);
                result.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++greater_fib_count[cur_index];
            }
            if(lesser_rank > (population.size() >> 1)) // lesser
            {
                unsigned char fib_count = lesser_fib_count[cur_index];
                result.lesser[fib_count][cur_index] = (double)lesser_rank/(population.size()+1);
                result.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                ++lesser_fib_count[cur_index];
            }
        }
    }
    return true;
}
void vbc_database::run_span(const fiber_orientations& fib,std::vector<std::vector<float> >& span)
{
    float param[8]={0};
    unsigned char methods[5];
    param[0] = fib.vs[0];
    param[4] = 0;//ui->smoothing->value();
    param[5] = 0;//ui->min_length->value();
    param[6] = 500;//ui->max_length->value();
    methods[0] = 0;//voxel tracking ui->tracking_method->currentIndex();
    methods[1] = 2;//all directions ui->initial_direction->currentIndex();
    methods[2] = 0;//trilinear interpolation ui->interpolation->currentIndex();
    methods[3] = 1;//stop by seed ui->tracking_plan->currentIndex();
    methods[4] = 1;//voxelwise tracking ui->seed_plan->currentIndex();
    std::vector<image::vector<3,short> > seed;

    for(image::pixel_index<3> index;index.valid(dim);index.next(dim))
        if(fib.fa[0][index.index()] > fib.threshold)
            seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));

    ThreadData tracking_thread;
    tracking_thread.setRegions(fib.dim,seed,3);
    tracking_thread.run(fib,param,methods,1,seed.size()*10,true);
    span.swap(tracking_thread.track_buffer);
}

void cal_hist(const std::vector<std::vector<float> >& span,std::vector<unsigned int>& dist)
{
    for(unsigned int j = 0; j < span.size();++j)
    {
        unsigned int length = span[j].size()/3-1;
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


void vbc_database::calculate_span_distribution(const fiber_orientations& fib,std::vector<unsigned int>& dist)
{
    std::vector<std::vector<float> > span;
    run_span(fib,span);
    cal_hist(span,dist);
}



void vbc_database::calculate_subject_distribution(float percentile,const fib_data& data,
                                                  std::vector<float>& subject_greater,
                                                  std::vector<float>& subject_lesser)
{
    // calculate subject fiber distribution
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fiber_orientations fib;
    fib.read(fib_file->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    fib.fa = data.greater_ptr;
    fib.findex = data.greater_dir_ptr;
    calculate_span_distribution(fib,dist_greater);

    fib.fa = data.lesser_ptr;
    fib.findex = data.lesser_dir_ptr;
    calculate_span_distribution(fib,dist_lesser);

    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
}

bool vbc_database::calculate_group_distribution(float percentile,const std::vector<std::string>& files,
                                                std::vector<float>& subject_greater,
                                                std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fib_data data;
    fiber_orientations fib;
    fib.read(fib_file->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    for(unsigned int main_index = 0;check_prog(main_index,files.size());++main_index)
    {
        if(!single_subject_analysis(files[main_index].c_str(),data))
            return false;
        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        calculate_span_distribution(fib,dist_greater);
        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        calculate_span_distribution(fib,dist_lesser);
    }
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
}

void vbc_database::tend_analysis(const std::vector<unsigned int>& permu,fib_data& data)
{
    data.initialize(fib_file);

    boost::math::normal gaussian;
    std::vector<float> population(subject_qa.size());
    unsigned int n = subject_qa.size();
    float sqrt_var_S = std::sqrt(n*(n-1)*(2.0*n+5.0)/18.0);
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

void vbc_database::calculate_null_trend_distribution(float percentile,
                                               std::vector<float>& subject_greater,
                                               std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fib_data data;
    fiber_orientations fib;
    fib.read(fib_file->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);

    std::vector<unsigned int> permu(subject_qa.size());
    for(unsigned int index = 0;index < permu.size();++index)
        permu[index] = index;

    unsigned int permu_count = 100;

    for(unsigned int main_index = 0;check_prog(main_index,permu_count);++main_index)
    {
        std::random_shuffle(permu.begin(),permu.end());

        tend_analysis(permu,data);

        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        calculate_span_distribution(fib,dist_greater);
        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        calculate_span_distribution(fib,dist_lesser);
    }
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
}

void vbc_database::calculate_null_distribution(float percentile,
                                               std::vector<float>& subject_greater,
                                               std::vector<float>& subject_lesser)
{
    begin_prog("processing");
    std::vector<unsigned int> dist_greater(200);
    std::vector<unsigned int> dist_lesser(200);
    fib_data data;
    fiber_orientations fib;
    fib.read(fib_file->fib_data);
    fib.threshold = percentile;
    fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
    for(unsigned int main_index = 0;check_prog(main_index,subject_qa.size());++main_index)
    {
        data.initialize(fib_file);
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
                float cur_value = subject_qa[main_index][pos];
                if(cur_value == 0.0)
                    continue;
                population.clear();
                for(unsigned int subject_id = 0;subject_id < subject_qa.size();++subject_id)
                {
                    float value = subject_qa[subject_id][pos];
                    if(subject_id != main_index && value != 0.0)
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
                if(greater_rank > (population.size() >> 1)) // greater
                {
                    unsigned char fib_count = greater_fib_count[cur_index];
                    data.greater[fib_count][cur_index] = (double)greater_rank/(population.size()+1);
                    data.greater_dir[fib_count][cur_index] = findex[fib][cur_index];
                    ++greater_fib_count[cur_index];
                }
                if(lesser_rank > (population.size() >> 1)) // lesser
                {
                    unsigned char fib_count = lesser_fib_count[cur_index];
                    data.lesser[fib_count][cur_index] = (double)lesser_rank/(population.size()+1);
                    data.lesser_dir[fib_count][cur_index] = findex[fib][cur_index];
                    ++lesser_fib_count[cur_index];
                }
            }
        }
        fib.fa = data.greater_ptr;
        fib.findex = data.greater_dir_ptr;
        calculate_span_distribution(fib,dist_greater);
        fib.fa = data.lesser_ptr;
        fib.findex = data.lesser_dir_ptr;
        calculate_span_distribution(fib,dist_lesser);
    }
    hist_to_dist(dist_greater,subject_greater);
    hist_to_dist(dist_lesser,subject_lesser);
}

void vbc_database::calculate_subject_fdr(float percentile,const fib_data& result,
                                           std::vector<std::vector<float> >& lesser_tracts,
                                           std::vector<float>& fdr)
{
    std::vector<std::vector<float> > spans;
    {
        fiber_orientations fib;
        fib.read(fib_file->fib_data);
        fib.fa = result.lesser_ptr;
        fib.findex = result.lesser_dir_ptr;
        fib.threshold = percentile;
        fib.cull_cos_angle = std::cos(60 * 3.1415926 / 180.0);
        run_span(fib,spans);
    }

    std::vector<float> greater_dist,lesser_dist;
    calculate_null_distribution(percentile,greater_dist,lesser_dist);
    dist_to_cdf(lesser_dist);

    std::map<unsigned int,unsigned int,std::greater<unsigned int> > length_map;
    {
        std::vector<unsigned int> length_list(spans.size());
        for(unsigned int i = 0; i < spans.size();++i)
            length_map[length_list[i] = spans[i].size()/3-1] = 0;
        for(unsigned int i = 0; i < spans.size();++i)
            ++length_map[length_list[i]];
    }

    std::map<unsigned int,unsigned int,std::greater<unsigned int> >::iterator beg = length_map.begin();
    std::map<unsigned int,unsigned int,std::greater<unsigned int> >::iterator end = length_map.end();
    std::vector<float> fdr_of_length(beg->first+1);
    std::fill(fdr_of_length.begin(),fdr_of_length.end(),1.0);
    for(unsigned int accumulated_count = 0;beg != end;++beg)
    {
        unsigned int cur_length = beg->first;
        accumulated_count += beg->second;
        float expected_n = accumulated_count;
        if(cur_length < lesser_dist.size())
            expected_n = lesser_dist[cur_length]*(float)spans.size();
        fdr_of_length[cur_length] = expected_n/(float)accumulated_count;
    }

    for(unsigned int i = 0; i < spans.size();++i)
    {
        unsigned int length = spans[i].size()/3-1;
        if(fdr_of_length[length] < 0.5)
        {
            lesser_tracts.push_back(spans[i]);
            fdr.push_back(fdr_of_length[length]);
        }
    }
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
