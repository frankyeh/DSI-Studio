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
        subject_qa_sd.push_back(0);
    }

    tipl::par_for(subject_qa.size(),[&](int i){
        subject_qa_sd[i] = tipl::standard_deviation(subject_qa[i],subject_qa[i]+subject_qa_length);
        if(subject_qa_sd[i] == 0.0)
            subject_qa_sd[i] = 1.0;
        else
            subject_qa_sd[i] = 1.0/subject_qa_sd[i];

    });

    num_subjects = (unsigned int)subject_qa.size();
    subject_names.resize(num_subjects);
    R2.resize(num_subjects);
    if(!num_subjects)
        return;
    {
        const char* report_buf = 0;
        if(handle->mat_reader.read("report",row,col,report_buf))
        {
            report = std::string(report_buf,report_buf+row*col);
            if(report.find(" sdf ") != std::string::npos)
            {
                report.resize(report.find(" sdf "));
                report += " local connectome fingerprint (LCF, Yeh et al. PLoS Comput Biol 12(11): e1005203) values were extracted from the data and used in the connectometry analysis.";
            }
        }
        if(handle->mat_reader.read("subject_report",row,col,report_buf))
            subject_report = std::string(report_buf,report_buf+row*col);

        const char* str = 0;
        handle->mat_reader.read("subject_names",row,col,str);
        if(str)
        {
            std::istringstream in(str);
            for(unsigned int index = 0;in && index < num_subjects;++index)
                std::getline(in,subject_names[index]);
        }
        handle->mat_reader.read("index_name",row,col,str);
        if(str)
            index_name = std::string(str,str+row*col);
        if(index_name.empty() || index_name.find("sdf") != std::string::npos)
            index_name = "qa";
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


bool connectometry_db::parse_demo(const std::string& filename,float missing_value)
{
    titles.clear();
    items.clear();
    int col_count = 0;
    {
        int row_count = 0,last_item_size = 0;
        bool is_csv = (filename.substr(filename.length()-4,4) == std::string(".csv"));
        std::ifstream in(filename.c_str());
        if(!in)
        {
            error_msg = "Cannot open the demographic file";
            return false;
        }
        std::string line;
        while(std::getline(in,line))
        {
            if(is_csv)
            {
                std::regex re(",");
                std::sregex_token_iterator first{line.begin(), line.end(),re, -1},last;
                std::copy(first,last,std::back_inserter(items));
            }
            else
            {
                std::istringstream in2(line);
                std::copy(std::istream_iterator<std::string>(in2),
                          std::istream_iterator<std::string>(),std::back_inserter(items));
            }
            if(items.size() == last_item_size)
                break;
            ++row_count;
            if(col_count == 0)
                col_count = items.size();
            else
                if(items.size()-last_item_size != col_count)
                {
                    std::ostringstream out;
                    out << subject_names[row_count-1] << " at row=" << row_count << " has " << items.size()-last_item_size <<
                            " fields, which is different from the column size " << col_count << ".";
                    error_msg = out.str();
                    return false;
                }
            last_item_size = items.size();
        }
        if(row_count == 1)
            col_count = items.size()/(num_subjects+1);

        if(items.size()/col_count < 2)
        {
            error_msg = "Invalid demographic file format";
            return false;
        }
        // check subject count for command line
        if(items.size()/col_count != num_subjects+1) // +1 for title
        {
            std::ostringstream out;
            out << "Subject number mismatch. The demographic file has " << row_count-1 << " subjects, but the database has " << num_subjects << " subjects.";
            error_msg = out.str();
            return false;
        }
    }
    // first line moved to title vector
    titles.insert(titles.end(),items.begin(),items.begin()+col_count);
    items.erase(items.begin(),items.begin()+col_count);

    // convert special characters
    for(int i = 0;i < titles.size();++i)
    {
        std::replace(titles[i].begin(),titles[i].end(),'/','_');
        std::replace(titles[i].begin(),titles[i].end(),'\\','_');
    }

    // find which column can be used as features
    feature_location.clear();
    feature_titles.clear();
    for(int i = 0;i < titles.size();++i)
    {
        try{
            std::stof(items[i]);
            feature_location.push_back(i);
            feature_titles.push_back(titles[i]);
        }
        catch (...)
        {;}
    }
    //  get feature matrix
    X.clear();
    for(unsigned int i = 0;i < num_subjects;++i)
    {
        X.push_back(1); // for the intercep
        for(unsigned int j = 0;j < feature_location.size();++j)
        {
            int item_pos = i*titles.size() + feature_location[j];
            if(item_pos >= items.size())
            {
                X.push_back(missing_value);
                continue;
            }
            try{

                X.push_back(std::stof(items[item_pos]));
            }
            catch(...)
            {
                std::ostringstream out;
                out << "Cannot parse '" << items[item_pos] << "' at " << subject_names[i] << "'s " << titles[feature_location[j]] << ".";
                error_msg = out.str();
                X.clear();
                return false;
            }
        }
    }
    return true;
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
    modified = true;
}
void connectometry_db::calculate_si2vi(void)
{
    vi2si.resize(handle->dim);
    for(unsigned int index = 0;index < (unsigned int)handle->dim.size();++index)
    {
        if(handle->dir.fa[0][index] != 0.0f)
        {
            vi2si[index] = (unsigned int)si2vi.size();
            si2vi.push_back(index);
        }
    }
    subject_qa_length = handle->dir.num_fiber*si2vi.size();
}

size_t convert_index(size_t old_index,
                     const tipl::geometry<3>& from_geo,
                     const tipl::geometry<3>& to_geo,
                     float ratio,
                     const tipl::vector<3>& shift)
{
    tipl::pixel_index<3> pos(old_index,from_geo);
    tipl::vector<3> v(pos);
    if(ratio != 1.0f)
        v *= ratio;
    v += shift;
    v.round();
    tipl::pixel_index<3> new_pos(v[0],v[1],v[2],to_geo);
    return new_pos.index();
}

bool connectometry_db::sample_subject_profile(gz_mat_read& m,std::vector<float>& data)
{
    bool trans_consistent = true;
    float ratio = 1.0f;
    tipl::vector<3> shift;
    tipl::geometry<3> subject_dim;
    {
        tipl::matrix<4,4,float> subject_trans;
        const float* trans = nullptr;
        unsigned int row,col;
        if(!m.read("trans",row,col,trans))
        {
            handle->error_msg = "Not a QSDR reconstructed file: ";
            return false;
        }
        std::copy(trans,trans+16,subject_trans.begin());
        const unsigned short* dim = nullptr;
        if(!m.read("dimension",row,col,dim))
        {
            handle->error_msg = "Invalid FIB format: ";
            return false;
        }
        std::copy(dim,dim+3,subject_dim.begin());
        if(subject_dim != handle->dim)
            trans_consistent = false;
        for(int i = 0; i < 16;++i)
            if(subject_trans[i] != handle->trans_to_mni[i])
                trans_consistent = false;
        if(!trans_consistent)
        {
            ratio = std::fabs(handle->trans_to_mni[0])/std::fabs(subject_trans[0]);
            shift = tipl::vector<3>(handle->trans_to_mni[3]-subject_trans[3],
                                    handle->trans_to_mni[7]-subject_trans[7],
                                    handle->trans_to_mni[11]-subject_trans[11]);
            shift /= std::fabs(subject_trans[0]);
        }
    }


    if(index_name == "qa" || index_name.empty())
    {
        if(!is_odf_consistent(m))
            return false;
        odf_data subject_odf;
        if(!subject_odf.read(m))
        {
            handle->error_msg = "Failed to read odf at ";
            return false;
        }
        set_title("Loading Data");
        tipl::par_for(si2vi.size(),[&](unsigned int index)
        {
            unsigned int cur_index = si2vi[index];
            unsigned int subject_index = trans_consistent ? cur_index:
                                                            convert_index(cur_index,handle->dim,subject_dim,ratio,shift);

            if(subject_index >= subject_dim.size())
                return;
            const float* odf = subject_odf.get_odf_data(subject_index);
            if(odf == nullptr)
                return;
            float min_value = *std::min_element(odf, odf + handle->dir.half_odf_size);
            unsigned int pos = index;
            for(unsigned char i = 0;i < handle->dir.num_fiber;++i,pos += uint32_t(si2vi.size()))
            {
                if(handle->dir.fa[i][cur_index] == 0.0f)
                    break;
                // 0: subject index 1:findex by s_index (fa > 0)
                data[pos] = odf[handle->dir.findex[i][cur_index]]-min_value;
            }
        });
        return true;
    }
    else
    {
        const float* index_of_interest = nullptr;
        unsigned int row,col;
        if(!m.read(index_name.c_str(),row,col,index_of_interest))
        {
            handle->error_msg = "Failed to sample ";
            handle->error_msg += index_name;
            handle->error_msg += " at ";
            return false;
        }
        tipl::par_for(si2vi.size(),[&](unsigned int index)
        {
            unsigned int cur_index = si2vi[index];
            if(handle->dir.fa[0][cur_index] == 0.0f)
                return;
            unsigned int subject_index = trans_consistent ? cur_index:
                                                            convert_index(cur_index,handle->dim,subject_dim,ratio,shift);
            if(subject_index >= subject_dim.size())
                return;
            unsigned int pos = index;
            for(unsigned char i = 0;i < handle->dir.num_fiber;++i,pos += uint32_t(si2vi.size()))
            {
                if(handle->dir.fa[i][cur_index] == 0.0f)
                    break;
                data[pos] = index_of_interest[subject_index];
            }
        });
        return true;
    }
}
bool connectometry_db::is_odf_consistent(gz_mat_read& m)
{
    unsigned int row,col;
    const float* odf_buffer = nullptr;
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
    /*
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
    }*/
    return true;
}
bool connectometry_db::add_subject_file(const std::string& file_name,
                                         const std::string& subject_name)
{
    gz_mat_read m;
    if(!m.load_from_file(file_name.c_str()))
    {
        handle->error_msg = "failed to load subject data ";
        handle->error_msg += file_name;
        return false;
    }
    std::vector<float> new_subject_qa(subject_qa_length);
    if(!sample_subject_profile(m,new_subject_qa))
    {
        handle->error_msg += file_name;
        return false;
    }
    // load R2
    const float* value= nullptr;
    unsigned int row,col;
    m.read("R2",row,col,value);
    if(!value || *value != *value)
    {
        handle->error_msg = "Invalid R2 value in ";
        handle->error_msg += file_name;
        return false;
    }
    R2.push_back(*value);
    const char* report_buf = 0;
    if(subject_report.empty() && m.read("report",row,col,report_buf))
        subject_report = std::string(report_buf,report_buf+row*col);
    subject_qa_buf.push_back(std::move(new_subject_qa));
    subject_qa.push_back(&(subject_qa_buf.back()[0]));
    subject_names.push_back(subject_name);
    subject_qa_sd.push_back(tipl::standard_deviation(subject_qa.back(),
                                                      subject_qa.back()+subject_qa_length));
    if(subject_qa_sd.back() == 0.0)
        subject_qa_sd.back() = 1.0;
    else
        subject_qa_sd.back() = 1.0/subject_qa_sd.back();
    num_subjects++;
    modified = true;
    return true;
}

void connectometry_db::get_subject_vector(unsigned int from,unsigned int to,
                                          std::vector<std::vector<float> >& subject_vector,
                        const tipl::image<int,3>& fp_mask,float fiber_threshold,bool normalize_fp) const
{
    unsigned int total_count = to-from;
    subject_vector.clear();
    subject_vector.resize(total_count);
    tipl::par_for(total_count,[&](unsigned int index)
    {
        unsigned int subject_index = index + from;
        for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
        {
            unsigned int cur_index = si2vi[s_index];
            if(!fp_mask[cur_index])
                continue;
            for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;
                    ++j,fib_offset+=si2vi.size())
                subject_vector[index].push_back(subject_qa[subject_index][s_index + fib_offset]);
        }
    });
    if(normalize_fp)
    tipl::par_for(num_subjects,[&](unsigned int index)
    {
        float sd = tipl::standard_deviation(subject_vector[index].begin(),subject_vector[index].end(),tipl::mean(subject_vector[index].begin(),subject_vector[index].end()));
        if(sd > 0.0)
            tipl::multiply_constant(subject_vector[index].begin(),subject_vector[index].end(),1.0/sd);
    });
}
void connectometry_db::get_subject_vector_pos(std::vector<int>& subject_vector_pos,
                            const tipl::image<int,3>& fp_mask,float fiber_threshold) const
{
    subject_vector_pos.clear();
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!fp_mask[cur_index])
            continue;
        for(unsigned int j = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j)
            subject_vector_pos.push_back(cur_index);
    }
}

void connectometry_db::get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                        const tipl::image<int,3>& fp_mask,float fiber_threshold,bool normalize_fp) const
{
    subject_vector.clear();
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        if(!fp_mask[cur_index])
            continue;
        for(unsigned int j = 0,fib_offset = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;
                ++j,fib_offset+=si2vi.size())
            subject_vector.push_back(subject_qa[subject_index][s_index + fib_offset]);
    }
    if(normalize_fp)
    {
        float sd = tipl::standard_deviation(subject_vector.begin(),subject_vector.end(),tipl::mean(subject_vector.begin(),subject_vector.end()));
        if(sd > 0.0)
            tipl::multiply_constant(subject_vector.begin(),subject_vector.end(),1.0/sd);
    }
}
void connectometry_db::get_dif_matrix(std::vector<float>& matrix,const tipl::image<int,3>& fp_mask,float fiber_threshold,bool normalize_fp)
{
    matrix.clear();
    matrix.resize(num_subjects*num_subjects);
    std::vector<std::vector<float> > subject_vector;
    get_subject_vector(0,num_subjects,subject_vector,fp_mask,fiber_threshold,normalize_fp);
    begin_prog("calculating");
    tipl::par_for2(num_subjects,[&](int i,int id){
        if(id == 0)
            check_prog(i,num_subjects);
        for(unsigned int j = i+1; j < num_subjects;++j)
        {
            double result = tipl::root_mean_suqare_error(
                        subject_vector[i].begin(),subject_vector[i].end(),
                        subject_vector[j].begin());
            matrix[i*num_subjects+j] = result;
            matrix[j*num_subjects+i] = result;
        }
    });
    check_prog(0,0);
}

void connectometry_db::save_subject_vector(const char* output_name,
                         const tipl::image<int,3>& fp_mask,
                         float fiber_threshold,
                         bool normalize_fp) const
{
    const unsigned int block_size = 400;
    std::string file_name = output_name;
    file_name = file_name.substr(0,file_name.length()-4); // remove .mat
    begin_prog("saving");
    for(unsigned int from = 0,iter = 0;from < num_subjects;from += block_size,++iter)
    {
        unsigned int to = std::min<unsigned int>(from+block_size,num_subjects);
        std::ostringstream out;
        out << file_name << iter << ".mat";
        std::string out_name = out.str();
        gz_mat_write matfile(out_name.c_str());
        if(!matfile)
        {
            handle->error_msg = "Cannot output file";
            return;
        }
        std::string name_string;
        for(unsigned int index = from;index < to;++index)
        {
            name_string += subject_names[index];
            name_string += "\n";
        }
        matfile.write("subject_names",name_string);
        for(unsigned int index = from,i = 0;index < to;++index,++i)
        {
            check_prog(from,num_subjects);
            std::vector<float> subject_vector;
            get_subject_vector(index,subject_vector,fp_mask,fiber_threshold,normalize_fp);

            std::ostringstream out;
            out << "subject" << index;
            matfile.write(out.str().c_str(),subject_vector);
        }

        if(iter == 0)
        {
            matfile.write("dimension",&*handle->dim.begin(),1,3);


            std::vector<int> voxel_location;
            std::vector<float> mni_location;
            std::vector<float> fiber_direction;

            for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
            {
                unsigned int cur_index = si2vi[s_index];
                if(!fp_mask[cur_index])
                    continue;
                for(unsigned int j = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j)
                {
                    voxel_location.push_back(cur_index);
                    tipl::pixel_index<3> p(cur_index,handle->dim);
                    tipl::vector<3> p2;
                    handle->subject2mni(p,p2);
                    mni_location.push_back(p2[0]);
                    mni_location.push_back(p2[1]);
                    mni_location.push_back(p2[2]);

                    const float* dir = handle->dir.get_dir(cur_index,j);
                    fiber_direction.push_back(dir[0]);
                    fiber_direction.push_back(dir[1]);
                    fiber_direction.push_back(dir[2]);
                }
            }
            matfile.write("voxel_location",voxel_location);
            matfile.write("mni_location",mni_location,3);
            matfile.write("fiber_direction",fiber_direction,3);
        }
    }
    check_prog(0,0);
}
bool connectometry_db::save_subject_data(const char* output_name)
{
    // store results
    gz_mat_write matfile(output_name);
    if(!matfile)
    {
        handle->error_msg = "Cannot output file";
        return false;
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
    matfile.write("subject_names",name_string);
    matfile.write("index_name",index_name);
    matfile.write("R2",R2);

    {
        std::ostringstream out;
        out << "A total of " << num_subjects << " diffusion MRI scans were included in the connectometry database." << subject_report.c_str();
        if(index_name.find("sdf") != std::string::npos || index_name.find("qa") != std::string::npos)
            out << " The quantitative anisotropy was extracted as the local connectome fingerprint (LCF, Yeh et al. PLoS Comput Biol 12(11): e1005203) and used in the connectometry analysis.";
        else
            out << " The " << index_name << " values were used in the connectometry analysis.";
        std::string report = out.str();
        matfile.write("subject_report",subject_report);
        matfile.write("report",report);
    }
    modified = false;
    return true;
}

void connectometry_db::get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                        tipl::image<float,2>& slice) const
{
    tipl::image<unsigned int,2> tmp;
    tipl::volume2slice(vi2si, tmp, dim, pos);
    slice.clear();
    slice.resize(tmp.geometry());
    for(unsigned int index = 0;index < slice.size();++index)
        if(tmp[index])
            slice[index] = subject_qa[subject_index][tmp[index]];
}
void connectometry_db::get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data,bool normalize_qa) const
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
            if(normalize_qa)
                fa_data[i][cur_index] *= subject_qa_sd[subject_index];
        }
    }
}
bool connectometry_db::get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data)
{
    gz_mat_read single_subject;
    if(!single_subject.load_from_file(file_name))
    {
        handle->error_msg = "fail to load the fib file";
        return false;
    }
    cur_subject_data.clear();
    cur_subject_data.resize(handle->dir.num_fiber*si2vi.size());
    if(!sample_subject_profile(single_subject,cur_subject_data))
    {
        handle->error_msg += file_name;
        return false;
    }
    const char* report_buf = nullptr;
    unsigned int row,col;
    if(single_subject.read("report",row,col,report_buf))
        subject_report = std::string(report_buf,report_buf+row*col);
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
    if(!is_odf_consistent(single_subject))
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
    const char* report_buf = 0;
    unsigned int row,col;
    if(single_subject.read("report",row,col,report_buf))
        subject_report = std::string(report_buf,report_buf+row*col);
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
    R2.insert(R2.end(),rhs.R2.begin(),rhs.R2.end());
    subject_qa_sd.insert(subject_qa_sd.end(),rhs.subject_qa_sd.begin(),rhs.subject_qa_sd.end());
    subject_names.insert(subject_names.end(),rhs.subject_names.begin(),rhs.subject_names.end());
    // copy the qa memeory
    for(unsigned int index = 0;index < rhs.num_subjects;++index)
    {
        subject_qa_buf.push_back(std::vector<float>(subject_qa_length));
        std::copy(rhs.subject_qa[index],
                  rhs.subject_qa[index]+subject_qa_length,subject_qa_buf.back().begin());
        subject_qa.push_back(&(subject_qa_buf.back()[0]));
    }
    num_subjects += rhs.num_subjects;
    modified = true;
    return true;
}
void connectometry_db::move_up(int id)
{
    if(id == 0)
        return;
    std::swap(subject_names[id],subject_names[id-1]);
    std::swap(R2[id],R2[id-1]);
    std::swap(subject_qa[id],subject_qa[id-1]);
    std::swap(subject_qa_sd[id],subject_qa_sd[id-1]);
}

void connectometry_db::move_down(int id)
{
    if(id >= num_subjects-1)
        return;
    std::swap(subject_names[id],subject_names[id+1]);
    std::swap(R2[id],R2[id+1]);
    std::swap(subject_qa[id],subject_qa[id+1]);
    std::swap(subject_qa_sd[id],subject_qa_sd[id+1]);
}

void connectometry_db::auto_match(const tipl::image<int,3>& fp_mask,float fiber_threshold,bool normalize_fp)
{
    std::vector<float> dif;
    get_dif_matrix(dif,fp_mask,fiber_threshold,normalize_fp);

    std::vector<float> half_dif;
    for(int i = 0;i < handle->db.num_subjects;++i)
        for(int j = i+1;j < handle->db.num_subjects;++j)
            half_dif.push_back(dif[i*handle->db.num_subjects+j]);

    // find the largest gap
    std::vector<float> v(half_dif);
    std::sort(v.begin(),v.end());
    float max_dif = 0,t = 0;
    for(int i = 1;i < v.size()/2;++i)
    {
        float dif = v[i]-v[i-1];
        if(dif > max_dif)
        {
            max_dif = dif;
            t = v[i]+v[i-1];
            t *= 0.5;
        }
    }
    std::cout << std::endl;

    match.clear();
    for(int i = 0,index = 0;i < handle->db.num_subjects;++i)
        for(int j = i+1;j < handle->db.num_subjects;++j,++index)
            if(half_dif[index] < t)
                match.push_back(std::make_pair(i,j));
}
void connectometry_db::calculate_change(unsigned char dif_type,bool norm)
{
    std::ostringstream out;


    std::vector<std::string> new_subject_names(match.size());
    std::vector<float> new_R2(match.size());
    std::vector<float> new_subject_qa_sd(match.size());


    std::list<std::vector<float> > new_subject_qa_buf;
    std::vector<const float*> new_subject_qa;
    begin_prog("calculating");
    for(unsigned int index = 0;check_prog(index,match.size());++index)
    {
        const float* baseline = subject_qa[match[index].first];
        const float* study = subject_qa[match[index].second];
        new_R2[index] = std::min<float>(R2[match[index].first],R2[match[index].second]);
        new_subject_names[index] = subject_names[match[index].second] + " - " + subject_names[match[index].first];
        std::vector<float> change(subject_qa_length);
        if(norm)
        {
            float ratio = subject_qa_sd[match[index].first] == 0 ?
                            0:subject_qa_sd[match[index].second]/subject_qa_sd[match[index].first];
            if(dif_type == 0)
            {
                if(!index)
                    out << " The difference between longitudinal scans were calculated";
                new_subject_qa_sd[index] = subject_qa_sd[match[index].first];
                for(int i = 0;i < subject_qa_length;++i)
                    change[i] = study[i]*ratio-baseline[i];
            }
            else
            {
                if(!index)
                    out << " The percentage difference between longitudinal scans were calculated";
                new_subject_qa_sd[index] = 1.0;
                for(int i = 0;i < subject_qa_length;++i)
                {
                    float new_s = study[i]*ratio;
                    float s = new_s+baseline[i];
                    change[i] = (s == 0 ? 0 : (new_s-baseline[i])/s);
                }
            }
            if(!index)
                out << " after the data variance in each individual was normalized to one";
        }
        else
        {
            if(dif_type == 0)
            {
                if(!index)
                    out << " The difference between longitudinal scans were calculated";
                new_subject_qa_sd[index] = subject_qa_sd[match[index].first];
                for(int i = 0;i < subject_qa_length;++i)
                    change[i] = study[i]-baseline[i];
            }
            else
            {
                if(!index)
                    out << " The percentage difference between longitudinal scans were calculated";
                new_subject_qa_sd[index] = 1.0;
                for(int i = 0;i < subject_qa_length;++i)
                {
                    float s = study[i]+baseline[i];
                    change[i] = (s == 0? 0 : (study[i]-baseline[i])/s);
                }
            }
        }
        new_subject_qa_buf.push_back(change);
        new_subject_qa.push_back(&(new_subject_qa_buf.back()[0]));
    }
    out << " (n=" << match.size() << ").";
    R2.swap(new_R2);
    subject_names.swap(new_subject_names);
    subject_qa_sd.swap(new_subject_qa_sd);
    subject_qa_buf.swap(new_subject_qa_buf);
    subject_qa.swap(new_subject_qa);
    num_subjects = match.size();
    match.clear();
    report += out.str();
    modified = true;

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
    pos_corr_ptr.resize(num_fiber);
    neg_corr_ptr.resize(num_fiber);
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        pos_corr_ptr[fib] = &greater[fib][0];
        neg_corr_ptr[fib] = &lesser[fib][0];
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
        if(handle->dir.index_name[index] == "inc" ||
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
    handle->dir.dt_index_name.push_back(t1);
    handle->dir.dt_index_data.push_back(std::vector<const float*>());
    handle->dir.dt_index_data.back() = pos_corr_ptr;
    handle->dir.dt_index_name.push_back(t2);
    handle->dir.dt_index_data.push_back(std::vector<const float*>());
    handle->dir.dt_index_data.back() = neg_corr_ptr;
}

bool connectometry_result::individual_vs_db(std::shared_ptr<fib_data> handle,const char* file_name)
{
    report = " Individual connectometry was conducted by comparing individuals to a group of subjects.";
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
    info.read_demo(handle->db);
    info.type = 2;
    info.individual_data = &(data[0]);
    //info.individual_data_sd = normalize_qa ? individual_data_sd[subject_id]:1.0;
    info.individual_data_sd = 1.0;
    float fa_threshold = 0.6*tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    calculate_spm(handle,*this,info,fa_threshold,normalized_qa,terminated);
    add_mapping_for_tracking(handle,"inc_db","dec_db");
    return true;
}

inline void calculate_dif(float& greater,
                          float& lesser,
                          float f1,float f2)
{
    float mean = 0.5f*(f1+f2);
    if(mean == 0.0f)
        return;
    if(f1 > f2)
        lesser = (f1-f2)/mean;  // subject decreased study index
    else
        greater = (f2-f1)/mean; // subject increased study index
}
bool connectometry_result::compare(std::shared_ptr<fib_data> handle,const std::vector<const float*>& fa1,
                                        const std::vector<const float*>& fa2,unsigned char normalization)
{
    std::ostringstream out;
    if(normalization == 0) // no normalization
    {
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                    calculate_dif(greater[fib][index],lesser[fib][index],fa1[fib][index],fa2[fib][index]);
        }
    }
    if(normalization == 1) // max to one
    {
        out << " Normalization was conducted to make the highest anisotropy to one.";
        float max1 = *std::max_element(fa1[0],fa1[0] + handle->dim.size());
        float max2 = *std::max_element(fa2[0],fa2[0] + handle->dim.size());

        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                    calculate_dif(greater[fib][index],lesser[fib][index],fa1[fib][index]/max1,fa2[fib][index]/max2);
        }
    }
    if(normalization == 2) // linear regression
    {
        out << " Normalization was conducted by a linear regression between the comparison scans.";
        std::pair<double,double> r = tipl::linear_regression(fa2[0],fa2[0] + handle->dim.size(),fa1[0]);
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                    calculate_dif(greater[fib][index],lesser[fib][index],fa1[fib][index],fa2[fib][index]*r.first+r.second);
        }
    }
    if(normalization == 3) // variance to one
    {
        out << " Normalization was conducted by scaling the variance to one.";
        float sd1 = tipl::standard_deviation(fa1[0],fa1[0] + handle->dim.size());
        float sd2 = tipl::standard_deviation(fa2[0],fa2[0] + handle->dim.size());
        for(unsigned char fib = 0;fib < handle->dir.num_fiber;++fib)
        {
            for(unsigned int index = 0;index < handle->dim.size();++index)
                if(fa1[fib][index] > 0.0 && fa2[fib][index] > 0.0)
                    calculate_dif(greater[fib][index],lesser[fib][index],fa1[fib][index]/sd1,fa2[fib][index]/sd2);
        }
    }
    report += out.str();
    return true;
}

bool connectometry_result::individual_vs_atlas(std::shared_ptr<fib_data> handle,
                                               const char* file_name,unsigned char normalization)
{
    report = " Individual connectometry was conducted by comparing individuals to a group-averaged template.";
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
    add_mapping_for_tracking(handle,"inc_qa","dec_qa");
    return true;
}

bool connectometry_result::individual_vs_individual(std::shared_ptr<fib_data> handle,
                                                    const char* file_name1,const char* file_name2,
                                                    unsigned char normalization)
{
    report = " Individual connectometry was conducted by comparing individual scans (Yeh, NeuroImage: Clinical 2,912-921,2013).";
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
    add_mapping_for_tracking(handle,"inc_qa","dec_qa");
    return true;
}



void stat_model::read_demo(const connectometry_db& db)
{
    subject_index.resize(db.num_subjects);
    std::iota(subject_index.begin(),subject_index.end(),0);

    X = db.X;
    feature_count = db.feature_location.size()+1; // additional one for intercept

}

void stat_model::select_variables(const std::vector<char>& sel)
{
    unsigned int subject_count = X.size()/feature_count;
    unsigned int new_feature_count = 0;
    std::vector<int> feature_map;
    for(int i = 0;i < sel.size();++i)
        if(sel[i])
        {
            ++new_feature_count;
            feature_map.push_back(i);
        }
    std::vector<double> new_X(subject_count*new_feature_count);
    for(int i = 0,index = 0;i < subject_count;++i)
        for(int j = 0;j < new_feature_count;++j,++index)
            new_X[index] = X[i*feature_count+feature_map[j]];
    feature_count = new_feature_count;
    X.swap(new_X);
}

bool stat_model::pre_process(void)
{
    if(X.empty())
        return false;
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
            if(subject_count <= feature_count)
                return false;
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
    case 3: //longitudinal change
        return true;
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
}

void stat_model::remove_data(const std::vector<char>& remove_list)
{
    for(int index = int(remove_list.size())-1;index >= 0;--index)
        if(remove_list[uint32_t(index)])
        {
            if(!label.empty())
                label.erase(label.begin()+index);
            if(!X.empty())
                X.erase(X.begin()+index*feature_count,X.begin()+(index+1)*feature_count);
            subject_index.erase(subject_index.begin()+index);
        }
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
        case 3: // longitudinal
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = bootstrap ? rhs.rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
            }
            if(null)
            {
                label.resize(subject_index.size());
                for(int i = 0;i < label.size();++i)
                    label[i] = rhs.rand_gen(2);
            }
            else
            {
                label.resize(subject_index.size());
                for(int i = 0;i < label.size();++i)
                    label[i] = 1;
            }
            break;
        case 2: // individual
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = bootstrap ? rhs.rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
            }
            break;
        }
        if(null)
            std::random_shuffle(subject_index.begin(),subject_index.end(),rhs.rand_gen);
    }while(!pre_process());

    return true;
}

double stat_model::operator()(const std::vector<double>& original_population,unsigned int pos) const
{
    std::vector<double> population(subject_index.size());
    for(unsigned int index = 0;index < subject_index.size();++index)
        population[index] = original_population[subject_index[index]];

    switch(type)
    {
    case 0: // group
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
        return tipl::t_statistics(g0.begin(),g0.end(),g1.begin(),g1.end());
    }
    case 1: // multiple regression
        {
            std::vector<double> b(feature_count),t(feature_count);
            mr.regress(&*population.begin(),&*b.begin(),&*t.begin());
            return t[study_feature];
        }
    case 2: // individual
        {
        float value = (individual_data_sd == 1.0) ? individual_data[pos]:individual_data[pos]*individual_data_sd;
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
    case 3: // longitudinal change
        {
            for(int i = 0;i < population.size();++i)
                if(label[i] == 0.0)
                    population[i] = -population[i];
            double mean = tipl::mean(population);
            double sd = tipl::standard_deviation(population.begin(),population.end(),mean);
            return sd == 0.0? 0.0 : mean/sd;
        }
    }

    return 0.0;
}
