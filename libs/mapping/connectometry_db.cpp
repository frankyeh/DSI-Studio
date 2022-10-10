#include <filesystem>
#include "prog_interface_static_link.h"
#include "connectometry_db.hpp"
#include "fib_data.hpp"

bool parse_age_sex(const std::string& file_name,std::string& age,std::string& sex)
{
    // look for _M020Y_
    for(size_t j = 0;j+6 < file_name.size();++j)
    {
        if(file_name[j] == '_' &&
           (file_name[j+1] == 'M' || file_name[j+1] == 'F') &&
           file_name[j+2] >= '0' && file_name[j+2] <= '9' &&
           file_name[j+3] >= '0' && file_name[j+3] <= '9' &&
           file_name[j+4] >= '0' && file_name[j+4] <= '9' &&
           file_name[j+5] == 'Y' &&
           file_name[j+6] == '_')// find two underscore
        {
            age = std::string(file_name.begin()+int(j)+2,file_name.begin()+int(j)+5); // age
            sex = (file_name[j+1] == 'M' ? "1":"0");
            return true;
        }
    }
    return false;
}

bool connectometry_db::read_db(fib_data* handle_)
{
    handle = handle_;
    subject_qa.clear();
    unsigned int row,col;
    for(unsigned int index = 0;1;++index)
    {
        const float* buf = nullptr;
        if (!handle->mat_reader.read((std::string("subjects")+std::to_string(index)).c_str(),row,col,buf) &&
            !handle->mat_reader.read((std::string("subject")+std::to_string(index)).c_str(),row,col,buf))
            break;
        if(!index)
        {
            subject_qa_length = row*col;
            is_longitudinal = false;
            for(size_t i = 0;i < subject_qa_length;++i)
                if(buf[i] < 0.0f)
                {
                    is_longitudinal = true;
                    break;
                }
        }
        subject_qa.push_back(buf);
    }
    num_subjects = uint32_t(subject_qa.size());
    subject_names.resize(num_subjects);
    R2.resize(num_subjects);
    if(!num_subjects)
        return true;

    progress prog("loading databse");
    std::string subject_names_str;
    if(!handle->mat_reader.read("subject_names",subject_names_str) ||
       !handle->mat_reader.read("R2",R2))
    {
        error_msg = "Invalid connectometry DB format.";
        num_subjects = 0;
        subject_qa.clear();
        return false;
    }
    // optional
    handle->mat_reader.read("report",report);
    handle->mat_reader.read("subject_report",subject_report);
    handle->mat_reader.read("index_name",index_name);
    // update index name
    if(index_name == "sdf")
        index_name = "qa";

    // make sure qa is normalized
    if(!is_longitudinal && (index_name == "qa" || index_name.empty()))
    {
        auto max_qa = tipl::max_value(subject_qa[0],subject_qa[0]+subject_qa_length);
        if(max_qa != 1.0f)
        {
            show_progress() << "converting raw QA to normalized QA" << std::endl;
            tipl::par_for(subject_qa.size(),[&](size_t i)
            {
                auto max_qa = tipl::max_value(subject_qa[i],subject_qa[i]+subject_qa_length);
                if(max_qa != 0.0f)
                    tipl::multiply_constant(const_cast<float*>(subject_qa[i]),
                                            const_cast<float*>(subject_qa[i])+subject_qa_length,1.0f/max_qa);
            });
        }
    }


    // update report
    if(report.find(" sdf ") != std::string::npos)
    {
        report.resize(report.find(" sdf "));
        report += " local connectome fingerprint (LCF, Yeh et al. PLoS Comput Biol 12(11): e1005203) values were extracted from the data and used in the connectometry analysis.";
    }
    // process subject names
    {
        std::istringstream in(subject_names_str);
        for(unsigned int index = 0;in && index < num_subjects;++index)
            std::getline(in,subject_names[index]);
    }


    if(handle->mat_reader.read("demo",demo))
    {
        if(!parse_demo())
            return false;
    }
    else
    {
        // try to get demo from subject bame
        demo += "age,sex\n";
        for(size_t i = 0;i < subject_names.size();++i)
        {
            std::string age,sex;
            if(!parse_age_sex(subject_names[i],age,sex))
            {
                demo.clear();
                break;
            }
            demo += age;
            demo += ",";
            demo += sex;
            demo += "\n";
        }
        if(!demo.empty() && !parse_demo())
            demo.clear();
    }
    calculate_si2vi();
    return true;
}

bool connectometry_db::parse_demo(const std::string& filename)
{
    std::ifstream in(filename.c_str());
    if(!in)
    {
        error_msg = "Cannot open the demographic file";
        return false;
    }
    // CSV BOM at the front
    if(std::filesystem::path(filename).extension() == ".csv" && in.peek() == 0xEF)
    {
        char dummy[3];
        in.read(dummy,3);
    }

    demo = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse_demo();
}

bool connectometry_db::parse_demo(void)
{
    show_progress() << "parsing demographics" << std::endl;
    std::string saved_demo(std::move(demo));
    titles.clear();
    items.clear();
    size_t col_count = 0;
    {
        size_t row_count = 0,last_item_size = 0;
        std::string line;
        bool is_csv = true;
        std::istringstream in(saved_demo);
        while(std::getline(in,line))
        {
            if(row_count == 0)
                is_csv = line.find(',') != std::string::npos;
            if(is_csv)
            {
                std::string col;
                std::istringstream in2(line);
                while(std::getline(in2,col,','))
                    items.push_back(col);
                if(line.back() == ',')
                    items.push_back(std::string());
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
        if(items.size() < 2*col_count)
        {
            error_msg = "Invalid demographic format";
            return false;
        }
        // select demographics by matching subject name
        if(items.size() > (num_subjects+1)*col_count)
        {
            std::vector<std::string> new_items;
            // copy titles
            std::copy(items.begin(),items.begin()+int(col_count),std::back_inserter(new_items));

            for(size_t i = 0;i < num_subjects;++i)
            {
                bool find = false;
                // find first column for subject name
                for(size_t j = col_count;j+col_count <= items.size();j += col_count)
                    if(subject_names[i].find(items[j]) != std::string::npos ||
                       items[j].find(subject_names[i]) != std::string::npos)
                    {
                        find = true;
                        std::copy(items.begin()+int(j),items.begin()+int(j+col_count),std::back_inserter(new_items));
                        break;
                    }
                if(!find)
                    break;
            }
            if(new_items.size() == (num_subjects+1)*col_count)
                items.swap(new_items);
        }

        if(items.size() != (num_subjects+1)*col_count)
        {
            std::ostringstream out;
            out << "Subject number mismatch. The demographic file has " << row_count-1 << " subject rows, but the database has " << num_subjects << " subjects.";
            error_msg = out.str();
            return false;
        }
    }
    // first line moved to title vector
    titles.insert(titles.end(),items.begin(),items.begin()+int(col_count));
    items.erase(items.begin(),items.begin()+int(col_count));

    // convert special characters
    show_progress pout;
    pout << "demographic columns:";
    for(size_t i = 0;i < titles.size();++i)
    {
        std::replace(titles[i].begin(),titles[i].end(),' ','_');
        std::replace(titles[i].begin(),titles[i].end(),'/','_');
        std::replace(titles[i].begin(),titles[i].end(),'\\','_');
        pout << "\t" << titles[i];
    }
    pout << std::endl;

    // find which column can be used as features
    feature_location.clear();
    feature_titles.clear();

    {
        std::vector<char> not_number(titles.size());
        for(size_t i = 0;i < items.size();++i)
        {
            if(not_number[i%titles.size()])
                continue;
            if(items[i] == " " || items[i] == "\r")
                items[i].clear();
            if(items[i].empty())
                continue;
            try{
                std::stof(items[i]);
            }
            catch (...)
            {
                not_number[i%titles.size()] = 1;
            }
        }
        for(size_t i = 0;i < titles.size();++i)
        if(!not_number[i])
        {
            feature_location.push_back(i);
            feature_titles.push_back(titles[i]);
            feature_selected.push_back(true);
        }
    }

    //  get feature matrix
    X.clear();
    for(unsigned int i = 0;i < num_subjects;++i)
    {
        X.push_back(1); // for the intercep
        for(unsigned int j = 0;j < feature_location.size();++j)
        {
            size_t item_pos = i*titles.size() + feature_location[j];
            if(item_pos >= items.size())
            {
                X.push_back(NAN);
                continue;
            }
            try{
                if(items[item_pos].empty())
                {
                    X.push_back(NAN);
                }
                else
                    X.push_back(double(std::stof(items[item_pos])));
            }
            catch(...)
            {
                std::ostringstream out;
                out << "cannot parse '" << items[item_pos] << "' at " << subject_names[i] << "'s " << titles[feature_location[j]] << ".";
                error_msg = out.str();
                X.clear();
                return false;
            }
        }
    }
    demo.swap(saved_demo);
    return true;
}

void connectometry_db::clear(void)
{
    subject_names.clear();
    R2.clear();
    subject_qa.clear();
    subject_qa_buf.clear();
    num_subjects = 0;
    modified = true;
}

void connectometry_db::remove_subject(unsigned int index)
{
    if(index >= subject_qa.size())
        return;
    subject_qa.erase(subject_qa.begin()+index);
    subject_names.erase(subject_names.begin()+index);
    R2.erase(R2.begin()+index);
    --num_subjects;
    modified = true;
}
void connectometry_db::calculate_si2vi(void)
{
    vi2si.resize(handle->dim);
    si2vi.clear();
    for(size_t index = 0;index < handle->dim.size();++index)
        if(handle->dir.fa[0][index] != 0.0f)
        {
            vi2si[index] = si2vi.size();
            si2vi.push_back(index);
        }
}

size_t convert_index(size_t old_index,
                     const tipl::shape<3>& from_geo,
                     const tipl::shape<3>& to_geo,
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

bool connectometry_db::is_odf_consistent(gz_mat_read& m)
{
    unsigned int row,col;
    const float* odf_buffer = nullptr;
    m.read("odf_vertices",row,col,odf_buffer);
    if (!odf_buffer)
    {
        error_msg = "No odf_vertices matrix in ";
        return false;
    }
    if(col != handle->dir.odf_table.size())
    {
        error_msg = "Inconsistent ODF dimension in ";
        return false;
    }
    for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
    {
        if(handle->dir.odf_table[index][0] != odf_buffer[0] ||
           handle->dir.odf_table[index][1] != odf_buffer[1] ||
           handle->dir.odf_table[index][2] != odf_buffer[2])
        {
            error_msg = "Inconsistent ODF in ";
            return false;
        }
    }
    /*
    const float* voxel_size = 0;
    m.read("voxel_size",row,col,voxel_size);
    if(!voxel_size)
    {
        error_msg = "No voxel_size matrix in ";
        return false;
    }
    if(voxel_size[0] != handle->vs[0])
    {
        std::ostringstream out;
        out << "Inconsistency in image resolution. Please use a correct atlas. The atlas resolution (" << handle->vs[0] << " mm) is different from that in ";
        error_msg = out.str();
        return false;
    }*/
    return true;
}
void connectometry_db::sample_from_image(tipl::const_pointer_image<3,float> I,
                       const tipl::matrix<4,4>& trans,std::vector<float>& data)
{
    tipl::image<3> J(handle->dim);
    tipl::resample_mt<tipl::interpolation::cubic>(I,J,
            tipl::transformation_matrix<float>(tipl::from_space(handle->trans_to_mni).to(trans)));

    data.clear();
    data.resize(si2vi.size());
    tipl::par_for(si2vi.size(),[&](size_t si)
    {
        data[si] = J[si2vi[si]];
    });
}
bool connectometry_db::add_subject_file(const std::string& file_name,
                                         const std::string& subject_name)
{
    std::vector<float> data;
    float subject_R2 = 1.0f;
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if(ext == ".nii" || ext == "i.gz")
    {
        tipl::vector<3> vs;
        tipl::image<3> I;
        tipl::matrix<4,4> trans;
        if(!gz_nifti::load_from_file(file_name.c_str(),I,vs,trans))
        {
            error_msg = "Cannot read file ";
            error_msg += file_name;
            return false;
        }
        sample_from_image(tipl::make_image(&I[0],I.shape()),trans,data);
    }
    else
    {
        fib_data fib;
        if(!fib.load_from_file(file_name.c_str()))
        {
            error_msg = fib.error_msg;
            return false;
        }
        if(subject_report.empty())
            subject_report = fib.report;
        fib.mat_reader.read("R2",subject_R2);

        if(fib.is_mni && fib.has_odfs() &&
           (index_name == "qa" || index_name == "nqa" || index_name.empty()))
        {
            odf_data subject_odf;
            if(!is_odf_consistent(fib.mat_reader))
                return false;
            if(!subject_odf.read(fib.mat_reader))
            {
                error_msg = subject_odf.error_msg;
                return false;
            }
            tipl::transformation_matrix<float> template2subject(tipl::from_space(handle->trans_to_mni).to(fib.trans_to_mni));

            progress::show("loading");
            data.clear();
            data.resize(si2vi.size()*size_t(handle->dir.num_fiber));
            tipl::par_for(si2vi.size(),[&](size_t si)
            {
                size_t vi = si2vi[si];
                if(handle->dir.fa[0][vi] == 0.0f)
                    return;

                tipl::vector<3> pos(tipl::pixel_index<3>(vi,handle->dim));
                template2subject(pos);
                pos.round();
                if(!fib.dim.is_valid(pos))
                    return;
                tipl::pixel_index<3> subject_pos(pos[0],pos[1],pos[2],fib.dim);
                const float* odf = subject_odf.get_odf_data(uint32_t(subject_pos.index()));
                if(odf == nullptr)
                    return;
                float min_value = tipl::min_value(odf, odf + handle->dir.half_odf_size);
                for(char i = 0;i < handle->dir.num_fiber;++i,si += si2vi.size())
                {
                    if(handle->dir.fa[i][vi] == 0.0f)
                        break;
                    // 0: subject index 1:findex by s_index (fa > 0)
                    data[si] = odf[handle->dir.findex[i][vi]]-min_value;
                }
            });
        }
        else
        {
            auto index = fib.get_name_index(index_name);
            if(index == fib.view_item.size())
            {
                error_msg = "cannot export ";
                error_msg += index_name;
                error_msg += " from ";
                error_msg += file_name;
                return false;
            }

            {
                if(fib.is_mni)
                    sample_from_image(fib.view_item[index].get_image(),fib.trans_to_mni,data);
                else
                {
                    fib.set_template_id(handle->template_id);
                    fib.map_to_mni();
                    while(!progress::aborted() && fib.prog != 6)
                        std::this_thread::yield();
                    if(progress::aborted())
                    {
                        error_msg = "aborted";
                        return false;
                    }
                    tipl::image<3> Iss(fib.t2s.shape());
                    tipl::compose_mapping(fib.view_item[index].get_image(),fib.t2s,Iss);
                    sample_from_image(tipl::make_image(&Iss[0],Iss.shape()),fib.template_to_mni,data);
                }
            }
        }
    }

    // normalize QA
    if(index_name == "qa" || index_name == "nqa" || index_name.empty())
    {
        float m = tipl::max_value(data);
        if(m != 1.0f && m != 0.0f)
            tipl::multiply_constant(data,1.0f/m);
    }

    if(data.empty())
    {
        error_msg = "failed to sample ";
        error_msg += index_name;
        error_msg += " in ";
        error_msg += file_name;
        return false;
    }

    R2.push_back(subject_R2);
    subject_qa_length = std::min<size_t>(subject_qa_length,data.size());
    subject_qa_buf.push_back(std::move(data));
    subject_qa.push_back(&(subject_qa_buf.back()[0]));
    subject_names.push_back(subject_name);
    num_subjects++;
    modified = true;
    return true;
}

void connectometry_db::get_subject_vector(unsigned int from,unsigned int to,
                                          std::vector<std::vector<float> >& subject_vector,
                        const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp) const
{
    unsigned int total_count = to-from;
    subject_vector.clear();
    subject_vector.resize(total_count);
    tipl::par_for(total_count,[&](unsigned int index)
    {
        size_t subject_index = index + from;
        for(size_t s_index = 0;s_index < si2vi.size();++s_index)
        {
            size_t cur_index = si2vi[s_index];
            if(!fp_mask[cur_index])
                continue;
            size_t fib_offset = 0;
            for(char j = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;
                    ++j,fib_offset+=si2vi.size())
            {
                size_t pos = s_index + fib_offset;
                if(pos >= subject_qa_length)
                    break;
                subject_vector[index].push_back(subject_qa[subject_index][pos]);
            }
        }
    });
    if(normalize_fp)
    tipl::par_for(num_subjects,[&](unsigned int index)
    {
        float sd = float(tipl::standard_deviation(subject_vector[index].begin(),subject_vector[index].end(),tipl::mean(subject_vector[index].begin(),subject_vector[index].end())));
        if(sd > 0.0f)
            tipl::multiply_constant(subject_vector[index].begin(),subject_vector[index].end(),1.0f/sd);
    });
}

void connectometry_db::get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                        const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp) const
{
    subject_vector.clear();
    for(size_t s_index = 0;s_index < si2vi.size();++s_index)
    {
        size_t cur_index = si2vi[s_index];
        if(!fp_mask[cur_index])
            continue;
        size_t fib_offset = 0;
        for(char j = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j,fib_offset+=si2vi.size())
        {
            size_t pos = s_index + fib_offset;
            if(pos >= subject_qa_length)
                break;
            subject_vector.push_back(subject_qa[subject_index][pos]);
        }
    }
    if(normalize_fp)
    {
        float sd = float(tipl::standard_deviation(subject_vector.begin(),subject_vector.end(),tipl::mean(subject_vector.begin(),subject_vector.end())));
        if(sd > 0.0f)
            tipl::multiply_constant(subject_vector.begin(),subject_vector.end(),1.0f/sd);
    }
}
void connectometry_db::get_dif_matrix(std::vector<float>& matrix,const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp)
{
    matrix.clear();
    matrix.resize(size_t(num_subjects)*size_t(num_subjects));
    std::vector<std::vector<float> > subject_vector;
    get_subject_vector(0,num_subjects,subject_vector,fp_mask,fiber_threshold,normalize_fp);
    progress prog_("calculating");
    size_t prog = 0;
    tipl::par_for(num_subjects,[&](unsigned int i){
        progress::at(prog++,num_subjects);
        for(unsigned int j = i+1; j < num_subjects;++j)
        {
            float result = float(tipl::root_mean_suqare_error(
                        subject_vector[i].begin(),subject_vector[i].end(),
                        subject_vector[j].begin()));
            matrix[i*num_subjects+j] = result;
            matrix[j*num_subjects+i] = result;
        }
    });
}

bool connectometry_db::save_subject_vector(const char* output_name,
                         const tipl::image<3,int>& fp_mask,
                         float fiber_threshold,
                         bool normalize_fp) const
{
    const unsigned int block_size = 400;
    std::string file_name = output_name;
    file_name = file_name.substr(0,file_name.length()-4); // remove .mat
    progress prog_("saving ","output_name");
    for(unsigned int from = 0,iter = 0;from < num_subjects;from += block_size,++iter)
    {
        unsigned int to = std::min<unsigned int>(from+block_size,num_subjects);
        std::ostringstream out;
        out << file_name << iter << ".mat";
        std::string out_name = out.str();
        gz_mat_write matfile(out_name.c_str());
        if(!matfile)
        {
            error_msg = "Cannot save file ";
            error_msg += out_name;
            return false;
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
            progress::at(from,num_subjects);
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
                for(char j = 0;j < handle->dir.num_fiber && handle->dir.fa[j][cur_index] > fiber_threshold;++j)
                {
                    voxel_location.push_back(int(cur_index));
                    tipl::pixel_index<3> p(cur_index,handle->dim);
                    tipl::vector<3> p2(p);
                    handle->sub2mni(p2);
                    mni_location.push_back(p2[0]);
                    mni_location.push_back(p2[1]);
                    mni_location.push_back(p2[2]);

                    auto dir = handle->dir.get_fib(cur_index,j);
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
    return true;
}
bool connectometry_db::save_db(const char* output_name)
{
    // store results
    gz_mat_write matfile(output_name);
    if(!matfile)
    {
        error_msg = "Cannot save file ";
        error_msg += output_name;
        return false;
    }
    for(unsigned int index = 0;index < handle->mat_reader.size();++index)
        if(handle->mat_reader[index].get_name() != "report" &&
           handle->mat_reader[index].get_name() != "steps" &&
           handle->mat_reader[index].get_name().find("subject") != 0)
            matfile.write(handle->mat_reader[index]);
    for(unsigned int index = 0;progress::at(index,subject_qa.size());++index)
        matfile.write((std::string("subjects")+std::to_string(index)).c_str(),subject_qa[index],1,subject_qa_length);
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
    if(!demo.empty())
        matfile.write("demo",demo);
    modified = false;
    return true;
}

void connectometry_db::get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                        tipl::image<2,float>& slice) const
{
    tipl::image<2,unsigned int> tmp;
    tipl::volume2slice(vi2si, tmp, dim, pos);
    slice.clear();
    slice.resize(tmp.shape());
    for(unsigned int index = 0;index < slice.size();++index)
        if(tmp[index])
            slice[index] = subject_qa[subject_index][tmp[index]];
}

bool connectometry_db::get_demo_matched_volume(const std::string& matched_demo,tipl::image<3>& volume) const
{
    if(demo.empty())
    {
        error_msg = "no demographic data found in the database";
        return false;
    }
    if(matched_demo.empty())
    {
        error_msg = "no demographics provided for the study subject";
        return false;
    }

    std::vector<double> v;
    {
        std::string s(matched_demo);
        std::replace(s.begin(),s.end(),',',' ');
        std::istringstream in(s);
        std::copy(std::istream_iterator<double>(in),
                  std::istream_iterator<double>(),
                  std::back_inserter(v));

        if(v.size() != feature_location.size())
        {
            error_msg = "invalid demographic input: ";
            error_msg += matched_demo;
            return false;
        }
        show_progress out;
        out << "matching ";
        for(size_t i = 0;i < feature_titles.size();++i)
            out << feature_titles[i] << ":" << v[i] << " ";
        out << std::endl;
    }
    size_t feature_size = 1+feature_location.size(); // +1 for intercept
    tipl::multiple_regression<double> mr;
    mr.set_variables(X.begin(),uint32_t(feature_size),uint32_t(subject_qa.size()));

    tipl::image<3> I(handle->dim);
    tipl::par_for(I.size(),[&](size_t index)
    {
        if(vi2si[index])
        {
            //I[index] = subject_qa[subject_index][vi2si[index]];
            std::vector<double> y(subject_qa.size());
            for(size_t s = 0;s < subject_qa.size();++s)
                y[s] = double(subject_qa[s][vi2si[index]]);
            std::vector<double> b(feature_size);
            mr.regress(y.begin(),b.begin());
            double predict = b[0];
            for(size_t i = 1;i < b.size();++i)
                predict += b[i]*v[i-1];
            I[index] = std::max<float>(0.0f,float(predict));
        }
    });
    volume.swap(I);
    return true;
}
bool connectometry_db::save_demo_matched_image(const std::string& matched_demo,const std::string& filename) const
{
    tipl::image<3> I;
    if(!get_demo_matched_volume(matched_demo,I))
        return false;
    if(!gz_nifti::save_to_file(filename.c_str(),I,handle->vs,handle->trans_to_mni,true,matched_demo.c_str()))
    {
        error_msg = "Cannot save file to ";
        error_msg += filename;
        return false;
    }
    return true;
}
void connectometry_db::get_subject_volume(unsigned int subject_index,tipl::image<3>& volume) const
{
    tipl::image<3> I(handle->dim);
    for(unsigned int index = 0;index < I.size();++index)
        if(vi2si[index])
            I[index] = subject_qa[subject_index][vi2si[index]];
    volume.swap(I);
}
void connectometry_db::get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data) const
{
    fa_data.resize(handle->dir.num_fiber);
    for(char index = 0;index < handle->dir.num_fiber;++index)
        fa_data[index].resize(handle->dim.size());
    tipl::par_for(si2vi.size(),[&](unsigned int s_index)
    {
        size_t cur_index = si2vi[s_index];
        size_t fib_offset = 0;
        for(char i = 0;i < handle->dir.num_fiber && handle->dir.fa[i][cur_index] > 0;++i,fib_offset+=si2vi.size())
        {
            size_t pos = s_index + fib_offset;
            fa_data[i][cur_index] = (pos < subject_qa_length ? subject_qa[subject_index][pos] : fa_data[0][cur_index]);
        }
    });
}
bool connectometry_db::get_qa_profile(const char* file_name,std::vector<std::vector<float> >& data)
{
    gz_mat_read single_subject;
    if(!single_subject.load_from_file(file_name))
    {
        error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_odf_consistent(single_subject))
        return false;
    odf_data subject_odf;
    if(!subject_odf.read(single_subject))
    {
        error_msg = subject_odf.error_msg;
        return false;
    }
    data.clear();
    data.resize(handle->dir.num_fiber);
    for(unsigned int index = 0;index < data.size();++index)
        data[index].resize(handle->dim.size());

    for(size_t index = 0;index < handle->dim.size();++index)
        if(handle->dir.fa[0][index] != 0.0f)
        {
            const float* odf = subject_odf.get_odf_data(index);
            if(!odf)
                continue;
            float min_value = tipl::min_value(odf, odf + handle->dir.half_odf_size);
            for(char i = 0;i < handle->dir.num_fiber;++i)
            {
                if(handle->dir.fa[i][index] == 0.0f)
                    break;
                data[i][index] = odf[handle->dir.findex[i][index]]-min_value;
            }
        }
    single_subject.read("report",subject_report);
    return true;
}
bool connectometry_db::is_db_compatible(const connectometry_db& rhs)
{
    if(rhs.handle->dim != handle->dim || subject_qa_length != rhs.subject_qa_length)
    {
        error_msg = "Image dimension does not match";
        return false;
    }
    for(size_t index = 0;index < handle->dim.size();++index)
        if(handle->dir.fa[0][index] != rhs.handle->dir.fa[0][index])
        {
            error_msg = "The connectometry db was created using a different template.";
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
    std::swap(subject_names[uint32_t(id)],subject_names[uint32_t(id-1)]);
    std::swap(R2[uint32_t(id)],R2[uint32_t(id-1)]);
    std::swap(subject_qa[uint32_t(id)],subject_qa[uint32_t(id-1)]);
}

void connectometry_db::move_down(int id)
{
    if(uint32_t(id) >= num_subjects-1)
        return;
    std::swap(subject_names[uint32_t(id)],subject_names[uint32_t(id+1)]);
    std::swap(R2[uint32_t(id)],R2[uint32_t(id+1)]);
    std::swap(subject_qa[uint32_t(id)],subject_qa[uint32_t(id+1)]);
}

void connectometry_db::auto_match(const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp)
{
    std::vector<float> dif;
    get_dif_matrix(dif,fp_mask,fiber_threshold,normalize_fp);

    std::vector<float> half_dif;
    for(unsigned int i = 0;i < handle->db.num_subjects;++i)
        for(unsigned int j = i+1;j < handle->db.num_subjects;++j)
            half_dif.push_back(dif[i*handle->db.num_subjects+j]);

    // find the largest gap
    std::vector<float> v(half_dif);
    std::sort(v.begin(),v.end());
    float max_dif = 0,t = 0;
    for(unsigned int i = 1;i < v.size()/2;++i)
    {
        float dif = v[i]-v[i-1];
        if(dif > max_dif)
        {
            max_dif = dif;
            t = v[i]+v[i-1];
            t *= 0.5f;
        }
    }
    match.clear();
    for(unsigned int i = 0,index = 0;i < handle->db.num_subjects;++i)
        for(unsigned int j = i+1;j < handle->db.num_subjects;++j,++index)
            if(half_dif[index] < t)
                match.push_back(std::make_pair(i,j));
}
void connectometry_db::calculate_change(unsigned char dif_type,bool norm)
{
    std::ostringstream out;


    std::vector<std::string> new_subject_names(match.size());
    std::vector<float> new_R2(match.size());


    std::list<std::vector<float> > new_subject_qa_buf;
    std::vector<const float*> new_subject_qa;
    progress prog_("calculating");
    for(unsigned int index = 0;progress::at(index,match.size());++index)
    {
        auto first = uint32_t(match[index].first);
        auto second = uint32_t(match[index].second);
        const float* baseline = subject_qa[first];
        const float* study = subject_qa[second];
        new_R2[index] = std::min<float>(R2[first],R2[second]);
        new_subject_names[index] = subject_names[second] + " - " + subject_names[first];
        std::vector<float> change(subject_qa_length);
        if(norm)
        {
            if(dif_type == 0)
            {
                if(!index)
                    out << " The difference between longitudinal scans were calculated";
                for(unsigned int i = 0;i < subject_qa_length;++i)
                    change[i] = study[i]-baseline[i];
            }
            else
            {
                if(!index)
                    out << " The percentage difference between longitudinal scans were calculated";
                for(unsigned int i = 0;i < subject_qa_length;++i)
                {
                    float new_s = study[i];
                    float s = new_s+baseline[i];
                    change[i] = (s == 0.0f ? 0 : (new_s-baseline[i])/s);
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
                for(unsigned int i = 0;i < subject_qa_length;++i)
                    change[i] = study[i]-baseline[i];
            }
            else
            {
                if(!index)
                    out << " The percentage difference between longitudinal scans were calculated";
                for(unsigned int i = 0;i < subject_qa_length;++i)
                {
                    float s = study[i]+baseline[i];
                    change[i] = (s == 0.0f? 0 : (study[i]-baseline[i])/s);
                }
            }
        }
        new_subject_qa_buf.push_back(change);
        new_subject_qa.push_back(&(new_subject_qa_buf.back()[0]));
    }
    out << " (n=" << match.size() << ").";
    R2.swap(new_R2);
    subject_names.swap(new_subject_names);
    subject_qa_buf.swap(new_subject_qa_buf);
    subject_qa.swap(new_subject_qa);
    index_name += "_dif";
    num_subjects = uint32_t(match.size());
    match.clear();
    report += out.str();
    modified = true;

}

void calculate_spm(std::shared_ptr<fib_data> handle,connectometry_result& data,stat_model& info,
                   float fiber_threshold,bool& terminated)
{
    data.clear_result(handle->dir.num_fiber,handle->dim.size());
    std::vector<double> population(handle->db.subject_qa.size());
    for(unsigned int s_index = 0;s_index < handle->db.si2vi.size() && !terminated;++s_index)
    {
        unsigned int cur_index = handle->db.si2vi[s_index];
        double result(0.0);
        size_t fib_offset = 0;
        for(char fib = 0;fib < handle->dir.num_fiber && handle->dir.fa[fib][cur_index] > fiber_threshold;
                ++fib,fib_offset+=handle->db.si2vi.size())
        {
            unsigned int pos = s_index + fib_offset;
            if(pos < handle->db.subject_qa_length)
            {
                for(unsigned int index = 0;index < population.size();++index)
                    population[index] = double(handle->db.subject_qa[index][pos]);

                if(std::find(population.begin(),population.end(),0.0) != population.end())
                    continue;
                result = info(population,pos);
            }
            if(result > 0.0) // group 0 > group 1
                data.pos_corr[fib][cur_index] = result;
            if(result < 0.0) // group 0 < group 1
                data.neg_corr[fib][cur_index] = -result;

        }
    }
}


void connectometry_result::clear_result(char num_fiber,size_t image_size)
{
    pos_corr.resize(num_fiber);
    neg_corr.resize(num_fiber);
    pos_corr_ptr.resize(num_fiber);
    neg_corr_ptr.resize(num_fiber);
    for(char fib = 0;fib < num_fiber;++fib)
    {
        pos_corr[fib].resize(image_size);
        neg_corr[fib].resize(image_size);
        std::fill(pos_corr[fib].begin(),pos_corr[fib].end(),0.0);
        std::fill(neg_corr[fib].begin(),neg_corr[fib].end(),0.0);
        pos_corr_ptr[fib] = &pos_corr[fib][0];
        neg_corr_ptr[fib] = &neg_corr[fib][0];
    }
}

inline void calculate_dif(float& pos_corr,
                          float& neg_corr,
                          float f1,float f2)
{
    float mean = 0.5f*(f1+f2);
    if(mean == 0.0f)
        return;
    if(f1 > f2)
        neg_corr = (f1-f2)/mean;  // subject decreased study index
    else
        pos_corr = (f2-f1)/mean; // subject increased study index
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
    auto subject_count = X.size()/feature_count;
    unsigned int new_feature_count = 0;
    std::vector<size_t> feature_map;
    for(size_t i = 0;i < sel.size();++i)
        if(sel[i])
        {
            ++new_feature_count;
            feature_map.push_back(i);
        }
    std::vector<double> new_X(size_t(subject_count)*size_t(new_feature_count));
    for(size_t i = 0,index = 0;i < subject_count;++i)
        for(size_t j = 0;j < new_feature_count;++j,++index)
            new_X[index] = X[i*feature_count+feature_map[j]];
    feature_count = new_feature_count;
    X.swap(new_X);
}

bool stat_model::pre_process(void)
{

    switch(type)
    {
    case 0: // group
        if(X.empty())
            return false;
        group2_count = 0;
        for(unsigned int index = 0;index < label.size();++index)
            if(label[index])
                ++group2_count;
        group1_count = label.size()-group2_count;
        return group2_count > 3 && group1_count > 3;
    case 1: // multiple regression
        {
            if(X.empty())
                return false;
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
        return mr.set_variables(&*X.begin(),feature_count,uint32_t(X.size()/feature_count));
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
        X.erase(X.begin()+int64_t(index)*feature_count,X.begin()+int64_t(index+1)*feature_count);
    subject_index.erase(subject_index.begin()+index);
}


bool stat_model::select_cohort(connectometry_db& db,
                               std::string select_text)
{
    error_msg.clear();
    cohort_report.clear();
    remove_list.clear();
    remove_list.resize(X.size()/feature_count);
    // handle missing value
    for(size_t k = 0;k < db.feature_titles.size();++k)
        if(db.feature_selected[k])
        {
            for(size_t i = 0,pos = 0;pos < X.size();++i,pos += feature_count)
                if(std::isnan(X[pos+size_t(k)+1]))
                    remove_list[i] = 1;
        }
    // select cohort
    if(!select_text.empty())
    {
        std::istringstream ss(select_text);
        std::string text;
        while(std::getline(ss,text,','))
        {
            bool parsed = false;
            auto select = [](char sel,float value,float threshold)
            {
                if(sel == '=')
                    return int(value*1000.0f) == int(threshold*1000.0f);
                if(sel == '>')
                    return value > threshold;
                if(sel == '<')
                    return value < threshold;
                return int(value*1000.0f) != int(threshold*1000.0f);
            };
            if(text.length() < 2)
                continue;
            for(size_t j = text.length()-2;j > 1;--j)
                if(text[j] == '=' || text[j] == '<' || text[j] == '>' || text[j] == '/')
                {
                    auto fov_name = text.substr(0,j);
                    auto value_text = text.substr(j+1);
                    bool okay;
                    std::istringstream si(value_text);
                    float threshold = 0.0f;
                    if(!(si >> threshold))
                    {
                        error_msg = "invalid selection text: ";
                        error_msg += text;
                        goto error;
                    }
                    if(fov_name == "value")
                    {
                        for(size_t k = 0;k < db.feature_titles.size();++k)
                            if(db.feature_selected[k])
                            {
                                for(size_t i = 0,pos = 0;pos < X.size();++i,pos += feature_count)
                                    if(!select(text[j],float(X[pos+size_t(k)+1]),threshold))
                                        remove_list[i] = 1;
                            }
                        parsed = true;
                        break;
                    }
                    size_t fov_index = 0;
                    okay = false;
                    for(size_t k = 0;k < db.feature_titles.size();++k)
                        if(db.feature_titles[k] == fov_name)
                        {
                            fov_index = k;
                            okay = true;
                            break;
                        }
                    if(!okay)
                        break;

                    for(size_t i = 0,pos = 0;pos < X.size();++i,pos += feature_count)
                        if(!select(text[j],float(X[pos+fov_index+1]),threshold))
                            remove_list[i] = 1;

                    std::ostringstream out;
                    if(text[j] == '/')
                        out << " Subjects with " << fov_name << "≠" << value_text << " were selected.";
                    else
                        out << " Subjects with " << text << " were selected.";
                    cohort_report += out.str();
                    parsed = true;
                    break;
                }
            if(!parsed)
            {
                error_msg = "cannot parse selection text:";
                error_msg += text;
                goto error;
            }
        }
    }
    return true;
    error:
    remove_list.clear();
    cohort_report.clear();
    return false;
}
bool stat_model::select_feature(connectometry_db& db,std::string foi_text)
{
    error_msg.clear();

    std::vector<char> sel(uint32_t(db.feature_titles.size()+1));
    sel[0] = 1; // intercept
    type = 1;

    variables.clear();
    variables.push_back("Intercept");
    bool has_variable = false;
    for(size_t i = 1;i < sel.size();++i)
        if(db.feature_selected[i-1])
        {
            std::set<double> unique_values;
            for(size_t j = 0,pos = 0;pos < X.size();++j,pos += feature_count)
                if(!remove_list[j])
                {
                    unique_values.insert(X[pos+i]);
                    if(unique_values.size() > 1)
                    {
                        sel[i] = 1;
                        variables.push_back(db.feature_titles[i-1]);
                        has_variable = true;
                        break;
                    }
                }
        }
    if(!has_variable)
    {
        // look at longitudinal change without considering any demographics
        if(db.is_longitudinal)
        {
            type = 3;
            read_demo(db);
            X.clear();
            return true;
        }
        else
        {
            error_msg = "No variables selected for regression. Please check selected variables.";
            return false;
        }
    }

    // select feature of interest
    {
        bool find_study_feature = false;
        // variables[0] = "intercept"
        for(unsigned int i = 0;i < variables.size();++i)
            if(variables[i] == foi_text)
            {
                study_feature = i;
                find_study_feature = true;
                break;
            }
        if(!find_study_feature)
        {
            error_msg = "Please select the targeted study feature.";
            return false;
        }
    }
    select_variables(sel);


    // remove subjects according to the cohort selection
    for(int index = int(remove_list.size())-1;index >= 0;--index)
        if(remove_list[uint32_t(index)])
        {
            if(!label.empty())
                label.erase(label.begin()+index);
            if(!X.empty())
                X.erase(X.begin()+int64_t(index)*feature_count,X.begin()+(int64_t(index)+1)*feature_count);
            subject_index.erase(subject_index.begin()+index);
        }

    if(!pre_process())
    {
        error_msg = "Some demographic data are duplicated. Please check the demographics.";
        return false;
    }
    return true;
}

bool stat_model::resample(stat_model& rhs,bool null,bool bootstrap,unsigned int seed)
{
    tipl::uniform_dist<int> rand_gen(0,1,seed);
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
                    new_index = rhs.label[index] ? group1[rand_gen(group1.size())]:group0[rand_gen(group0.size())];
                subject_index[index] = rhs.subject_index[new_index];
                label[index] = rhs.label[new_index];
            }
        }
            break;
        case 1: // multiple regression
            X.resize(rhs.X.size());
            for(unsigned int index = 0,pos = 0;index < rhs.subject_index.size();++index,pos += feature_count)
            {
                unsigned int new_index = bootstrap ?
                            uint32_t(rand_gen(uint32_t(rhs.subject_index.size()))) : index;
                subject_index[index] = rhs.subject_index[new_index];
                std::copy(rhs.X.begin()+int64_t(new_index)*feature_count,
                          rhs.X.begin()+int64_t(new_index)*feature_count+feature_count,X.begin()+pos);
            }
            if(nonparametric)
            {
                std::vector<double> x_study_feature;
                for(size_t index = study_feature;index< X.size();index += feature_count)
                    x_study_feature.push_back(X[index]);
                x_study_feature_rank = tipl::rank(x_study_feature,std::less<double>());

                unsigned int n = uint32_t(X.size()/feature_count);
                rank_c = 6.0/double(n)/double(n*n-1);
            }
            break;
        case 3: // longitudinal
            for(unsigned int index = 0;index < rhs.subject_index.size();++index)
            {
                unsigned int new_index = bootstrap ? rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
            }
            if(null)
            {
                label.resize(subject_index.size());
                for(int i = 0;i < label.size();++i)
                    label[i] = rand_gen(2);
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
                unsigned int new_index = bootstrap ? rand_gen(rhs.subject_index.size()) : index;
                subject_index[index] = rhs.subject_index[new_index];
            }
            break;
        }
        if(null)
            std::shuffle(subject_index.begin(),subject_index.end(),rand_gen.gen);
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
            if(nonparametric)
            {
                // partial correlation
                std::vector<double> b(feature_count);
                mr.regress(&*population.begin(),&*b.begin());
                for(size_t i = 1;i < feature_count;++i) // skip intercept at i = 0
                    if(i != study_feature)
                    {
                        auto cur_b = b[i];
                        for(size_t j = 0,p = i;j < population.size();++j,p += feature_count)
                            population[j] -= mr.X[p]*cur_b;
                    }

                auto rank = tipl::rank(population,std::less<double>());
                int sum_d2 = 0;
                for(size_t i = 0;i < rank.size();++i)
                {
                    int d = int(rank[i])-int(x_study_feature_rank[i]);
                    sum_d2 += d*d;
                }
                double r = 1.0-double(sum_d2)*rank_c;
                double result = r*std::sqrt(double(population.size()-2.0)/(1.0-r*r));
                return std::isnormal(result) ? result : 0.0;
            }
            else
            {
                std::vector<double> b(feature_count),t(feature_count);
                mr.regress(&*population.begin(),&*b.begin(),&*t.begin());
                return t[study_feature];
            }

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
