#include "connectometry_db.hpp"
#include "fib_data.hpp"

void connectometry_db::read_db(fib_data* handle_)
{
    handle = handle_;
    subject_qa.clear();
    subject_qa_sd.clear();
    unsigned int row,col;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        const float* buf = 0;
        handle->mat_reader.read(out.str().c_str(),row,col,buf);
        if (!buf)
            break;
        if(!index)
            subject_qa_length = row*col;
        subject_qa.push_back(buf);
        subject_qa_sd.push_back(image::standard_deviation(buf,buf+col*row));
        if(subject_qa_sd.back() == 0.0)
            subject_qa_sd.back() = 1.0;
        else
            subject_qa_sd.back() = 1.0/subject_qa_sd.back();
    }
    num_subjects = (unsigned int)subject_qa.size();
    subject_names.resize(num_subjects);
    R2.resize(num_subjects);
    if(!num_subjects)
        return;
    {
        const char* str = 0;
        handle->mat_reader.read("subject_names",row,col,str);
        if(str)
        {
            std::istringstream in(str);
            for(unsigned int index = 0;in && index < num_subjects;++index)
                std::getline(in,subject_names[index]);
        }
        const float* r2_values = 0;
        handle->mat_reader.read("R2",row,col,r2_values);
        if(r2_values == 0)
        {
            handle->error_msg = "Memory insufficiency. Use 64-bit program instead";
            num_subjects = 0;
            subject_qa.clear();
            return;
        }
        std::copy(r2_values,r2_values+num_subjects,R2.begin());
    }

    calculate_si2vi();
}

void connectometry_db::remove_subject(unsigned int index)
{
    if(index >= subject_qa.size())
        return;
    subject_qa.erase(subject_qa.begin()+index);
    subject_qa_sd.erase(subject_qa_sd.begin()+index);
    subject_names.erase(subject_names.begin()+index);
    R2.erase(R2.begin()+index);
    --num_subjects;
}
void connectometry_db::calculate_si2vi(void)
{
    vi2si.resize(handle->dim);
    for(unsigned int index = 0;index < (unsigned int)handle->dim.size();++index)
    {
        if(handle->dir.fa[0][index] != 0.0)
        {
            vi2si[index] = (unsigned int)si2vi.size();
            si2vi.push_back(index);
        }
    }
}
bool connectometry_db::sample_odf(gz_mat_read& m,std::vector<float>& data)
{
    odf_data subject_odf;
    if(!subject_odf.read(m))
        return false;
    set_title("load data");
    for(unsigned int index = 0;index < si2vi.size();++index)
    {
        unsigned int cur_index = si2vi[index];
        const float* odf = subject_odf.get_odf_data(cur_index);
        if(odf == 0)
            continue;
        float min_value = *std::min_element(odf, odf + handle->dir.half_odf_size);
        unsigned int pos = index;
        for(unsigned char i = 0;i < handle->dir.num_fiber;++i,pos += (unsigned int)si2vi.size())
        {
            if(handle->dir.fa[i][cur_index] == 0.0)
                break;
            // 0: subject index 1:findex by s_index (fa > 0)
            data[pos] = odf[handle->dir.findex[i][cur_index]]-min_value;
        }
    }
    return true;
}
bool connectometry_db::sample_index(gz_mat_read& m,std::vector<float>& data,const char* index_name)
{
    const float* index_of_interest = 0;
    unsigned int row,col;
    if(!m.read(index_name,row,col,index_of_interest))
        return false;
    for(unsigned int index = 0;index < si2vi.size();++index)
    {
        unsigned int cur_index = si2vi[index];
        unsigned int pos = index;
        for(unsigned char i = 0;i < handle->dir.num_fiber;++i,pos += (unsigned int)si2vi.size())
        {
            if(handle->dir.fa[i][cur_index] == 0.0)
                break;
            data[pos] = index_of_interest[cur_index];
        }
    }
    return true;
}
bool connectometry_db::is_consistent(gz_mat_read& m)
{
    unsigned int row,col;
    const float* odf_buffer = 0;
    m.read("odf_vertices",row,col,odf_buffer);
    if (!odf_buffer)
    {
        handle->error_msg = "No odf_vertices matrix in ";
        return false;
    }
    if(col != handle->dir.odf_table.size())
    {
        handle->error_msg = "Inconsistent ODF dimension in ";
        return false;
    }
    for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
    {
        if(handle->dir.odf_table[index][0] != odf_buffer[0] ||
           handle->dir.odf_table[index][1] != odf_buffer[1] ||
           handle->dir.odf_table[index][2] != odf_buffer[2])
        {
            handle->error_msg = "Inconsistent ODF in ";
            return false;
        }
    }
    const float* voxel_size = 0;
    m.read("voxel_size",row,col,voxel_size);
    if(!voxel_size)
    {
        handle->error_msg = "No voxel_size matrix in ";
        return false;
    }
    if(voxel_size[0] != handle->vs[0])
    {
        std::ostringstream out;
        out << "Inconsistency in image resolution. Please use a correct atlas. The atlas resolution (" << handle->vs[0] << " mm) is different from that in ";
        handle->error_msg = out.str();
        return false;
    }
    return true;
}

bool connectometry_db::load_subject_files(const std::vector<std::string>& file_names,
                        const std::vector<std::string>& subject_names_,
                        const char* index_name)
{
    num_subjects = (unsigned int)file_names.size();
    subject_qa.clear();
    subject_qa.resize(num_subjects);
    subject_qa_buf.resize(num_subjects);
    R2.resize(num_subjects);
    for(unsigned int index = 0;index < num_subjects;++index)
        subject_qa_buf[index].resize(handle->dir.num_fiber*si2vi.size());
    for(unsigned int index = 0;index < num_subjects;++index)
        subject_qa[index] = &(subject_qa_buf[index][0]);
    for(unsigned int subject_index = 0;check_prog(subject_index,num_subjects);++subject_index)
    {
        if(prog_aborted())
        {
            check_prog(1,1);
            return false;
        }
        gz_mat_read m;
        if(!m.load_from_file(file_names[subject_index].c_str()))
        {
            handle->error_msg = "failed to load subject data ";
            handle->error_msg += file_names[subject_index];
            return false;
        }
        // check if the odf table is consistent or not
        if(std::string(index_name) == "sdf")
        {
            if(!is_consistent(m))
            {
                handle->error_msg += file_names[subject_index];
                return false;
            }
            if(!sample_odf(m,subject_qa_buf[subject_index]))
            {
                handle->error_msg = "Failed to read odf ";
                handle->error_msg += file_names[subject_index];
                return false;
            }
        }
        else
        {
            if(!sample_index(m,subject_qa_buf[subject_index],index_name))
            {
                handle->error_msg = "failed to sample ";
                handle->error_msg += index_name;
                handle->error_msg += " in ";
                handle->error_msg += file_names[subject_index];
                return false;
            }
        }
        // load R2
        const float* value= 0;
        unsigned int row,col;
        m.read("R2",row,col,value);
        if(!value || *value != *value)
        {
            handle->error_msg = "Invalid R2 value in ";
            handle->error_msg += file_names[subject_index];
            return false;
        }
        R2[subject_index] = *value;
        if(subject_index == 0)
        {
            const char* report_buf = 0;
            if(m.read("report",row,col,report_buf))
                subject_report = std::string(report_buf,report_buf+row*col);
        }
    }
    subject_names = subject_names_;
    return true;
}
void connectometry_db::get_subject_vector(std::vector<std::vector<float> >& subject_vector,
                        const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp) const
{
    subject_vector.clear();
    subject_vector.resize(num_subjects);
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!cerebrum_mask[cur_index])
            continue;
        for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;
                ++j,fib_offset+=si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            for(unsigned int index = 0;index < num_subjects;++index)
                subject_vector[index].push_back(subject_qa[index][pos]);
        }
    }
    if(normalize_fp)
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        float sd = image::standard_deviation(subject_vector[index].begin(),subject_vector[index].end(),image::mean(subject_vector[index].begin(),subject_vector[index].end()));
        if(sd > 0.0)
            image::multiply_constant(subject_vector[index].begin(),subject_vector[index].end(),1.0/sd);
    }
}

void connectometry_db::get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                        const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp) const
{
    subject_vector.clear();
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!cerebrum_mask[cur_index])
            continue;
        for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;
                ++j,fib_offset+=si2vi.size())
            subject_vector.push_back(subject_qa[subject_index][s_index + fib_offset]);
    }
    if(normalize_fp)
    {
        float sd = image::standard_deviation(subject_vector.begin(),subject_vector.end(),image::mean(subject_vector.begin(),subject_vector.end()));
        if(sd > 0.0)
            image::multiply_constant(subject_vector.begin(),subject_vector.end(),1.0/sd);
    }
}
void connectometry_db::get_dif_matrix(std::vector<float>& matrix,const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp)
{
    matrix.clear();
    matrix.resize(num_subjects*num_subjects);
    std::vector<std::vector<float> > subject_vector;
    get_subject_vector(subject_vector,cerebrum_mask,fiber_threshold,normalize_fp);
    begin_prog("calculating");
    for(unsigned int i = 0; check_prog(i,num_subjects);++i)
        for(unsigned int j = i+1; j < num_subjects;++j)
        {
            double result = image::root_mean_suqare_error(
                        subject_vector[i].begin(),subject_vector[i].end(),
                        subject_vector[j].begin());
            matrix[i*num_subjects+j] = result;
            matrix[j*num_subjects+i] = result;
        }
}

void connectometry_db::save_subject_vector(const char* output_name,
                         const image::basic_image<int,3>& cerebrum_mask,
                         float fiber_threshold,
                         bool normalize_fp) const
{
    gz_mat_write matfile(output_name);
    if(!matfile)
    {
        handle->error_msg = "Cannot output file";
        return;
    }
    std::vector<std::vector<float> > subject_vector;
    get_subject_vector(subject_vector,cerebrum_mask,fiber_threshold,normalize_fp);
    std::string name_string;
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        name_string += subject_names[index];
        name_string += "\n";
    }
    matfile.write("subject_names",name_string.c_str(),1,(unsigned int)name_string.size());
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        matfile.write(out.str().c_str(),&subject_vector[index][0],1,(unsigned int)subject_vector[index].size());
    }
    matfile.write("dimension",&*handle->dim.begin(),1,3);
    std::vector<int> voxel_location;
    std::vector<float> mni_location;
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!cerebrum_mask[cur_index])
            continue;
        for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j,fib_offset+=si2vi.size())
        {
            voxel_location.push_back(cur_index);
            image::pixel_index<3> p(cur_index,handle->dim);
            image::vector<3> p2(p);
            handle->subject2mni(p2);
            mni_location.push_back(p2[0]);
            mni_location.push_back(p2[1]);
            mni_location.push_back(p2[2]);
        }
    }
    matfile.write("voxel_location",&voxel_location[0],1,voxel_location.size());
    matfile.write("mni_location",&mni_location[0],3,voxel_location.size());
}
void connectometry_db::save_subject_data(const char* output_name)
{
    // store results
    gz_mat_write matfile(output_name);
    if(!matfile)
    {
        handle->error_msg = "Cannot output file";
        return;
    }
    for(unsigned int index = 0;index < handle->mat_reader.size();++index)
        if(handle->mat_reader[index].get_name() != "report" &&
           handle->mat_reader[index].get_name().find("subject") != 0)
            matfile.write(handle->mat_reader[index]);
    for(unsigned int index = 0;check_prog(index,(unsigned int)subject_qa.size());++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        matfile.write(out.str().c_str(),subject_qa[index],handle->dir.num_fiber,(unsigned int)si2vi.size());
    }
    std::string name_string;
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        name_string += subject_names[index];
        name_string += "\n";
    }
    matfile.write("subject_names",name_string.c_str(),1,(unsigned int)name_string.size());
    matfile.write("R2",&*R2.begin(),1,(unsigned int)R2.size());

    {
        std::ostringstream out;
        out << "A total of " << num_subjects << " subjects were included in the connectometry database." << subject_report.c_str();
        std::string report = out.str();
        matfile.write("report",&*report.c_str(),1,(unsigned int)report.length());
    }
}

void connectometry_db::get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                        image::basic_image<float,2>& slice) const
{
    image::basic_image<unsigned int,2> tmp;
    image::reslicing(vi2si, tmp, dim, pos);
    slice.clear();
    slice.resize(tmp.geometry());
    for(unsigned int index = 0;index < slice.size();++index)
        if(tmp[index])
            slice[index] = subject_qa[subject_index][tmp[index]];
}
void connectometry_db::get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data) const
{
    fa_data.resize(handle->dir.num_fiber);
    for(unsigned int index = 0;index < handle->dir.num_fiber;++index)
        fa_data[index].resize(handle->dim.size());
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int i = 0,fib_offset = 0;i < handle->dir.num_fiber && handle->dir.fa[i][cur_index] > 0;++i,fib_offset+=(unsigned int)si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            fa_data[i][cur_index] = subject_qa[subject_index][pos];
        }
    }
}
void connectometry_db::get_data_at(unsigned int index,unsigned int fib_index,std::vector<double>& data,bool normalize_qa) const
{
    data.clear();
    if((int)index >= handle->dim.size() || handle->dir.fa[0][index] == 0.0)
        return;
    unsigned int s_index = vi2si[index];
    unsigned int fib_offset = fib_index*(unsigned int)si2vi.size();
    data.resize(num_subjects);
    if(normalize_qa)
        for(unsigned int index = 0;index < num_subjects;++index)
            data[index] = subject_qa[index][s_index+fib_offset]*subject_qa_sd[index];
    else
    for(unsigned int index = 0;index < num_subjects;++index)
        data[index] = subject_qa[index][s_index+fib_offset];
}
bool connectometry_db::get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data)
{
    gz_mat_read single_subject;
    if(!single_subject.load_from_file(file_name))
    {
        handle->error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_consistent(single_subject))
        return false;
    cur_subject_data.clear();
    cur_subject_data.resize(handle->dir.num_fiber*si2vi.size());
    if(!sample_odf(single_subject,cur_subject_data))
    {
        handle->error_msg += file_name;
        return false;
    }
    return true;
}
bool connectometry_db::get_qa_profile(const char* file_name,std::vector<std::vector<float> >& data)
{
    gz_mat_read single_subject;
    if(!single_subject.load_from_file(file_name))
    {
        handle->error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_consistent(single_subject))
        return false;
    odf_data subject_odf;
    if(!subject_odf.read(single_subject))
    {
        handle->error_msg = "The fib file contains no ODF information. Please reconstruct the SRC file again with ODF output.";
        return false;
    }
    data.clear();
    data.resize(handle->dir.num_fiber);
    for(unsigned int index = 0;index < data.size();++index)
        data[index].resize(handle->dim.size());

    for(unsigned int index = 0;index < handle->dim.size();++index)
        if(handle->dir.fa[0][index] != 0.0)
        {
            const float* odf = subject_odf.get_odf_data(index);
            if(odf == 0)
                continue;
            float min_value = *std::min_element(odf, odf + handle->dir.half_odf_size);
            for(unsigned char i = 0;i < handle->dir.num_fiber;++i)
            {
                if(handle->dir.fa[i][index] == 0.0)
                    break;
                data[i][index] = odf[handle->dir.findex[i][index]]-min_value;
            }
        }
    return true;
}
bool connectometry_db::is_db_compatible(const connectometry_db& rhs)
{
    if(rhs.handle->dim != handle->dim || subject_qa_length != rhs.subject_qa_length)
    {
        handle->error_msg = "Image dimension does not match";
        return false;
    }
    for(unsigned int index = 0;index < handle->dim.size();++index)
        if(handle->dir.fa[0][index] != rhs.handle->dir.fa[0][index])
        {
            handle->error_msg = "The connectometry db was created using a different template.";
            return false;
        }
    return true;
}
void connectometry_db::read_subject_qa(std::vector<std::vector<float> >&data) const
{
    data.resize(num_subjects);
    for(unsigned int i = 0;i < num_subjects;++i)
    {
        std::vector<float> buf(subject_qa[i],subject_qa[i]+subject_qa_length);
        data[i].swap(buf);
    }
}

bool connectometry_db::add_db(const connectometry_db& rhs)
{
    if(!is_db_compatible(rhs))
        return false;
    num_subjects += rhs.num_subjects;
    R2.insert(R2.end(),rhs.R2.begin(),rhs.R2.end());
    subject_qa_sd.insert(subject_qa_sd.end(),rhs.subject_qa_sd.begin(),rhs.subject_qa_sd.end());
    subject_names.insert(subject_names.end(),rhs.subject_names.begin(),rhs.subject_names.end());
    // copy the qa memeory
    for(unsigned int index = 0;index < rhs.num_subjects;++index)
    {
        subject_qa_buf.push_back(std::vector<float>());
        subject_qa_buf.back().resize(subject_qa_length);
        std::copy(rhs.subject_qa[index],
                  rhs.subject_qa[index]+subject_qa_length,subject_qa_buf.back().begin());
    }

    // everytime subject_qa_buf has a change, its memory may have been reallocated. Thus we need to assign all pointers.
    subject_qa.resize(num_subjects);
    for(unsigned int index = 0;index < subject_qa_buf.size();++index)
        subject_qa[num_subjects+index-subject_qa_buf.size()] = &(subject_qa_buf[index][0]);
}



void calculate_spm(std::shared_ptr<fib_data> handle,connectometry_result& data,stat_model& info,
                   float fiber_threshold,bool normalize_qa,bool& terminated)
{
    data.initialize(handle);
    std::vector<double> population(handle->db.subject_qa.size());
    for(unsigned int s_index = 0;s_index < handle->db.si2vi.size() && !terminated;++s_index)
    {
        unsigned int cur_index = handle->db.si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < handle->dir.num_fiber && handle->dir.fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=handle->db.si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            if(normalize_qa)
                for(unsigned int index = 0;index < population.size();++index)
                    population[index] = handle->db.subject_qa[index][pos]*handle->db.subject_qa_sd[index];
            else
                for(unsigned int index = 0;index < population.size();++index)
                    population[index] = handle->db.subject_qa[index][pos];

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


void connectometry_result::initialize(std::shared_ptr<fib_data> handle)
{
    unsigned char num_fiber = handle->dir.num_fiber;
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
void connectometry_result::remove_old_index(std::shared_ptr<fib_data> handle)
{
    for(unsigned int index = 0;index < handle->dir.index_name.size();++index)
        if(handle->dir.index_name[index] == ">%" ||
           handle->dir.index_name[index] == "<%" ||
           handle->dir.index_name[index] == "inc" ||
           handle->dir.index_name[index] == "dec")
        {
            handle->dir.index_name.erase(handle->dir.index_name.begin()+index);
            handle->dir.index_data.erase(handle->dir.index_data.begin()+index);
            index = 0;
        }
}

void connectometry_result::add_mapping_for_tracking(std::shared_ptr<fib_data> handle,const char* t1,const char* t2)
{
    remove_old_index(handle);
    handle->dir.index_name.push_back(t1);
    handle->dir.index_data.push_back(std::vector<const float*>());
    handle->dir.index_data.back() = greater_ptr;
    handle->dir.index_name.push_back(t2);
    handle->dir.index_data.push_back(std::vector<const float*>());
    handle->dir.index_data.back() = lesser_ptr;
}

bool connectometry_result::individual_vs_db(std::shared_ptr<fib_data> handle,const char* file_name)
{
    if(!handle->db.has_db())
    {
        error_msg = "Please open a connectometry database first.";
        return false;
    }
    handle->dir.set_tracking_index(0);
    std::vector<float> data;
    if(!handle->db.get_odf_profile(file_name,data))
    {
        error_msg = handle->error_msg;
        return false;
    }
    bool normalized_qa = false;
    bool terminated = false;
    stat_model info;
    info.init(handle->db.has_db());
    info.type = 2;
    info.individual_data = &(data[0]);
    //info.individual_data_sd = normalize_qa ? individual_data_sd[subject_id]:1.0;
    info.individual_data_sd = 1.0;
    float fa_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(handle->dir.fa[0],handle->dim));
    calculate_spm(handle,*this,info,fa_threshold,normalized_qa,terminated);
    add_mapping_for_tracking(handle,">%","<%");
    return true;
}
bool connectometry_result::compare(std::shared_ptr<fib_data> handle,const std::vector<const float*>& fa1,
                                        const std::vector<const float*>& fa2,unsigned char normalization)
{
    if(normalization == 0) // no normalization
    {
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                {
                    float f1 = fa1[fib][index];
                    float f2 = fa2[fib][index];
                    if(f1 > f2)
                        lesser[fib][index] = f1-f2;  // subject decreased connectivity
                    else
                        greater[fib][index] = f2-f1; // subject increased connectivity
                }
        }
    }
    if(normalization == 1) // max to one
    {
        float max1 = *std::max_element(fa1[0],fa1[0] + handle->dim.size());
        float max2 = *std::max_element(fa2[0],fa2[0] + handle->dim.size());

        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                {
                    float f1 = fa1[fib][index]/max1;
                    float f2 = fa2[fib][index]/max2;
                    if(f1 > f2)
                        lesser[fib][index] = f1-f2;  // subject decreased connectivity
                    else
                        greater[fib][index] = f2-f1; // subject increased connectivity
                }
        }
    }
    if(normalization == 2) // linear regression
    {
        std::pair<double,double> r = image::linear_regression(fa2[0],fa2[0] + handle->dim.size(),fa1[0]);
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                {
                    float f1 = fa1[fib][index];
                    float f2 = fa2[fib][index]*r.first+r.second;
                    if(f1 > f2)
                        lesser[fib][index] = f1-f2;  // subject decreased connectivity
                    else
                        greater[fib][index] = f2-f1; // subject increased connectivity
                }
        }
    }
    if(normalization == 3) // variance to one
    {
        float sd1 = image::standard_deviation(fa1[0],fa1[0] + handle->dim.size());
        float sd2 = image::standard_deviation(fa2[0],fa2[0] + handle->dim.size());
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                {
                    float f1 = fa1[fib][index]/sd1;
                    float f2 = fa2[fib][index]/sd2;
                    if(f1 > f2)
                        lesser[fib][index] = f1-f2;  // subject decreased connectivity
                    else
                        greater[fib][index] = f2-f1; // subject increased connectivity
                }
        }
    }
    return true;
}

bool connectometry_result::individual_vs_atlas(std::shared_ptr<fib_data> handle,
                                               const char* file_name,unsigned char normalization)
{
    // restore fa0 to QA
    handle->dir.set_tracking_index(0);
    std::vector<std::vector<float> > fa_data;
    if(!handle->db.get_qa_profile(file_name,fa_data))
    {
        error_msg = handle->error_msg;
        return false;
    }
    std::vector<const float*> ptr(fa_data.size());
    for(unsigned int i = 0;i < ptr.size();++i)
        ptr[i] = &(fa_data[i][0]);
    initialize(handle);
    if(!compare(handle,handle->dir.fa,ptr,normalization))
        return false;
    add_mapping_for_tracking(handle,"inc","dec");
    return true;
}

bool connectometry_result::individual_vs_individual(std::shared_ptr<fib_data> handle,const char* file_name1,const char* file_name2,unsigned char normalization)
{
    // restore fa0 to QA
    handle->dir.set_tracking_index(0);
    std::vector<std::vector<float> > data1,data2;
    if(!handle->db.get_qa_profile(file_name1,data1))
    {
        error_msg = handle->error_msg;
        return false;
    }
    if(!handle->db.get_qa_profile(file_name2,data2))
    {
        error_msg = handle->error_msg;
        return false;
    }
    initialize(handle);
    std::vector<const float*> ptr1(data1.size()),ptr2(data2.size());
    for(unsigned int i = 0;i < ptr1.size();++i)
    {
        ptr1[i] = &(data1[i][0]);
        ptr2[i] = &(data2[i][0]);
    }
    if(!compare(handle,ptr1,ptr2,normalization))
        return false;
    add_mapping_for_tracking(handle,"inc","dec");
    return true;
}



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
    std::lock_guard<std::mutex> lock(rhs.lock_random);
    *this = rhs;
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
        if(threshold_type == t)
        {
            std::vector<double> g0(group1_count),g1(group2_count);
            for(unsigned int index = 0,i0 = 0,i1 = 0;index < label.size();++index)
                if(label[index])
                {
                    g1[i1] = population[index];
                    ++i1;
                }
            else
                {
                    g0[i0] = population[index];
                    ++i0;
                }
            return image::t_statistics(g0.begin(),g0.end(),g1.begin(),g1.end());
        }
        else
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

            if(threshold_type == percentage)
            {
                float m = (mean1 + mean2)/2.0;
                if(m == 0.0)
                    return 0.0;
                return (mean1 - mean2)/m;
            }
            else
                if(threshold_type == mean_dif)
                    return mean1-mean2;

        }
        break;
    case 1: // multiple regression
        if(threshold_type == percentage)
        {
            std::vector<double> b(feature_count);
            mr.regress(&*population.begin(),&*b.begin());
            double mean = image::mean(population.begin(),population.end());
            return mean == 0 ? 0:b[study_feature]*X_range[study_feature]/mean;
        }
        else
            if(threshold_type == beta)
            {
                std::vector<double> b(feature_count);
                mr.regress(&*population.begin(),&*b.begin());
                return b[study_feature];
            }
            else
                if(threshold_type == t)
                {
                    std::vector<double> b(feature_count),t(feature_count);
                    mr.regress(&*population.begin(),&*b.begin(),&*t.begin());
                    return t[study_feature];
                }
        break;
    case 2: // individual
    {
        float value = (individual_data_sd == 1.0) ? individual_data[pos]:individual_data[pos]*individual_data_sd;
        if(value == 0.0)
            return 0.0;
        if(threshold_type == mean_dif)
            return value-image::mean(population.begin(),population.end());
        else
            if(threshold_type == percentage)
            {
                float mean = image::mean(population.begin(),population.end());
                if(mean == 0.0)
                    return 0.0;
                return value/mean-1.0;
            }
            else
            if(threshold_type == percentile)
            {
                int rank = 0;
                for(unsigned int index = 0;index < population.size();++index)
                    if(value > population[index])
                        ++rank;
                return (rank > (population.size() >> 1)) ?
                                    (double)rank/(double)population.size():
                                    (double)(rank-(int)population.size())/(double)population.size();
            }
    }
        break;
    case 3: // paired
        if(threshold_type == t)
        {
            unsigned int half_size = population.size() >> 1;
            std::vector<double> dif(population.begin(),population.begin()+half_size);
            image::minus(dif.begin(),dif.end(),population.begin()+half_size);
            return image::t_statistics(dif.begin(),dif.end());
        }
        else
        {
            unsigned int half_size = population.size() >> 1;
            float g1 = std::accumulate(population.begin(),population.begin()+half_size,0.0);
            float g2 = std::accumulate(population.begin()+half_size,population.end(),0.0);
            if(threshold_type == percentage)
                return 2.0*(g1-g2)/(g1+g2);
            else
                if(threshold_type == mean_dif)
                    return (g1-g2)/half_size;
        }
        break;
    }

    return 0.0;
}
