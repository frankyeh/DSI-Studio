#include "connectometry_db.hpp"
#include "fib_data.hpp"

void connectometry_db::read_db(FibData* handle_)
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
            handle->error_msg = "Inconsistent ODF dimension in ";
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
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!cerebrum_mask[cur_index])
            continue;
        for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j,fib_offset+=si2vi.size())
            voxel_location.push_back(cur_index);
    }
    matfile.write("voxel_location",&voxel_location[0],1,voxel_location.size());
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
            data[index] = subject_qa[index][s_index+fib_offset]/subject_qa_sd[index];
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
