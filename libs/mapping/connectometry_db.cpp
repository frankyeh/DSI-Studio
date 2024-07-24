#include <filesystem>
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
            // check if the db is longitudinal, for older db, the only way to check is by the negative values.
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

    tipl::progress prog("loading database");
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

    // new db can be all positive, the checking the report text can confirm longitudinal setting
    if(report.find("longitudinal scans were calculated") != std::string::npos)
    {
        is_longitudinal = true;
        if(report.find("Only increased longitudinal changes") != std::string::npos)
            longitudinal_filter_type = 1;
        if(report.find("Only decreased longitudinal changes") != std::string::npos)
            longitudinal_filter_type = 2;
    }

    // make sure qa is normalized
    if(!is_longitudinal && (index_name == "qa" || index_name.empty()))
    {
        auto max_qa = tipl::max_value(subject_qa[0],subject_qa[0]+subject_qa_length);
        if(max_qa != 1.0f)
        {
            tipl::out() << "converting raw QA to normalized QA" << std::endl;
            tipl::adaptive_par_for(subject_qa.size(),[&](size_t i)
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
    tipl::out() << "parsing demographics" << std::endl;
    std::string saved_demo(std::move(demo));
    titles.clear();
    items.clear();
    size_t col_count = 0;
    {
        size_t row_count = 0,last_item_size = 0;
        std::string line;
        bool is_csv = true;
        bool is_tsv = true;
        std::istringstream in(saved_demo);
        while(std::getline(in,line))
        {
            if(row_count == 0)
            {
                is_csv = line.find(',') != std::string::npos;
                if(!is_csv)
                    is_tsv = line.find('\t') != std::string::npos;
            }

            if(is_csv || is_tsv)
            {
                std::string col;
                std::istringstream in2(line);
                while(std::getline(in2,col,is_csv ? ',' : '\t'))
                    items.push_back(col);
                if((line.back() == ',' && is_csv) || (line.back() == '\t' && is_csv))
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
        // select demographics by matching subject name
        if(items.size() != (num_subjects+1)*col_count)
        {
            tipl::out() << "demographic rows different from database subject numbers. trying to match subject name with the first column...";
            bool found = false;
            std::vector<std::string> new_items((num_subjects+1)*col_count);
            // copy titles
            std::copy(items.begin(),items.begin()+int(col_count),new_items.begin());

            for(size_t i = 0;i < num_subjects;++i)
                for(size_t j = col_count;j+col_count <= items.size();j += col_count)
                {
                    if(subject_names[i].find(items[j]) != std::string::npos ||
                       items[j].find(subject_names[i]) != std::string::npos)
                    {
                        found = true;
                        std::copy(items.begin()+int(j),items.begin()+int(j+col_count),new_items.begin() + (i+1)*col_count);
                        break;
                    }
                }
            if(!found)
            {
                std::ostringstream out;
                out << "Subject number mismatch. The demographic file has " << row_count-1 << " subject rows, but the database has " << num_subjects << " subjects.";
                error_msg = out.str();
                return false;
            }
            items.swap(new_items);
        }
    }
    // first line moved to title vector
    titles.insert(titles.end(),items.begin(),items.begin()+int(col_count));
    items.erase(items.begin(),items.begin()+int(col_count));

    // convert special characters
    tipl::out pout;
    pout << "demographic columns: ";
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
    feature_is_float.clear();

    {
        std::vector<char> not_number(titles.size()),not_categorical(titles.size());
        for(size_t i = 0;i < items.size();++i)
        {
            if(not_number[i%titles.size()])
                continue;
            if(items[i] == " " || items[i] == "\r")
                items[i].clear();
            if(items[i].empty())
                continue;
            try{
                float value = std::stof(items[i]);
                if(std::floor(value) != value)
                    not_categorical[i%titles.size()] = 1;
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
            feature_is_float.push_back(not_categorical[i]); // 0: categorical 1: floating

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

bool connectometry_db::is_odf_consistent(tipl::io::gz_mat_read& m)
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
    tipl::resample<tipl::interpolation::cubic>(I,J,
            tipl::transformation_matrix<float>(tipl::from_space(handle->trans_to_mni).to(trans)));

    data.clear();
    data.resize(si2vi.size());
    tipl::adaptive_par_for(si2vi.size(),[&](size_t si)
    {
        data[si] = J[si2vi[si]];
    });
}
void connectometry_db::add(float subject_R2,std::vector<float>& data,
                           const std::string& subject_name)
{
    // remove negative values due to interpolation
    tipl::lower_threshold(data,0.0f);
    // normalize QA
    if(index_name == "qa" || index_name == "nqa" || index_name.empty())
    {
        float m = tipl::max_value(data);
        if(m != 1.0f && m != 0.0f)
            tipl::multiply_constant(data,1.0f/m);
    }
    R2.push_back(subject_R2);
    subject_qa_length = std::min<size_t>(subject_qa_length,data.size());
    subject_qa_buf.push_back(std::move(data));
    subject_qa.push_back(&(subject_qa_buf.back()[0]));
    subject_names.push_back(subject_name);
    num_subjects++;
    modified = true;
}
bool connectometry_db::add(const std::string& file_name,
                                         const std::string& subject_name)
{
    tipl::progress prog(file_name.c_str());
    std::vector<float> data;
    float subject_R2 = 1.0f;
    if(tipl::ends_with(file_name,".nii") || tipl::ends_with(file_name,".nii.gz"))
    {
        tipl::image<3> I;
        tipl::matrix<4,4> trans;
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(file_name))
        {
            error_msg = "Cannot read file ";
            error_msg += file_name;
            return false;
        }
        nii.get_image_transformation(trans);
        if(nii.dim(4) > 1)
        {
            for(size_t i = 0;prog(i,nii.dim(4));++i)
            {
                nii >> I;
                sample_from_image(I.alias(),trans,data);
                add(subject_R2,data,subject_name+std::to_string(i));
            }
            if(prog.aborted())
            {
                error_msg = "aborted";
                return false;
            }
            return true;
        }
        else
        {
            nii >> I;
            sample_from_image(I.alias(),trans,data);
        }
    }
    else
    {
        fib_data fib;
        if(!fib.load_from_file(file_name.c_str()))
        {
            error_msg = "Cannot read file ";
            error_msg += file_name;
            error_msg += " : ";
            error_msg += fib.error_msg;
            return false;
        }

        if(fib.db.has_db())
            return add_db(fib.db);

        if(subject_report.empty())
            subject_report = fib.report;
        fib.mat_reader.read("R2",subject_R2);

        if(fib.is_mni && fib.has_odfs() &&
           (index_name == "qa" || index_name == "nqa" || index_name.empty()))
        {
            odf_data subject_odf;
            if(!is_odf_consistent(fib.mat_reader))
            {
                error_msg = "Inconsistent ODF at ";
                error_msg += file_name;
                return false;
            }
            if(!subject_odf.read(fib.mat_reader))
            {
                error_msg = "Failed to read ODF at ";
                error_msg += file_name;
                error_msg += " : ";
                error_msg += subject_odf.error_msg;
                return false;
            }
            tipl::transformation_matrix<float> template2subject(tipl::from_space(handle->trans_to_mni).to(fib.trans_to_mni));

            tipl::out() << "loading";
            data.clear();
            data.resize(si2vi.size()*size_t(handle->dir.num_fiber));
            tipl::adaptive_par_for(si2vi.size(),[&](size_t si)
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
                    fib.map_to_mni(tipl::show_prog);
                    while(!prog.aborted() && fib.prog != 6)
                        std::this_thread::yield();
                    if(prog.aborted())
                    {
                        error_msg = "aborted";
                        return false;
                    }
                    tipl::image<3> Iss(fib.t2s.shape());
                    tipl::compose_mapping(fib.view_item[index].get_image(),fib.t2s,Iss);
                    sample_from_image(Iss.alias(),fib.template_to_mni,data);
                }
            }
        }
    }
    if(data.empty())
    {
        error_msg = "failed to sample ";
        error_msg += index_name;
        error_msg += " in ";
        error_msg += file_name;
        return false;
    }
    if(prog.aborted())
    {
        error_msg = "aborted";
        return false;
    }
    add(subject_R2,data,subject_name);
    return true;
}

bool connectometry_db::save_db(const char* output_name)
{
    // store results
    tipl::io::gz_mat_write matfile(output_name);
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
    tipl::progress prog("save db");
    for(unsigned int index = 0;prog(index,subject_qa.size());++index)
        matfile.write((std::string("subjects")+std::to_string(index)).c_str(),subject_qa[index],1,subject_qa_length);
    if(prog.aborted())
    {
        error_msg = "aborted";
        return false;
    }
    std::string name_string;
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        name_string += subject_names[index];
        name_string += "\n";
    }
    matfile.write("subject_names",name_string);
    matfile.write("subject_report",subject_report);
    matfile.write("index_name",index_name);
    matfile.write("R2",R2);

    if(is_longitudinal)
        matfile.write("report",report);
    else
    {
        std::ostringstream out;
        out << "A total of " << num_subjects << " diffusion MRI scans were included in the connectometry database." << subject_report.c_str();
        if(index_name.find("sdf") != std::string::npos || index_name.find("qa") != std::string::npos)
            out << " The quantitative anisotropy was extracted as the local connectome fingerprint (LCF, Yeh et al. PLoS Comput Biol 12(11): e1005203) and used in the connectometry analysis.";
        else
            out << " The " << index_name << " values were used in the connectometry analysis.";
        matfile.write("report",out.str());
    }
    if(!demo.empty())
        matfile.write("demo",demo);
    modified = false;
    return true;
}

void connectometry_db::get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                        tipl::image<2,float>& slice) const
{
    tipl::image<2,size_t> tmp;
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
        tipl::out out;
        out << "creating subject-matching image by regressing against ";
        for(size_t i = 0;i < feature_titles.size();++i)
            out << feature_titles[i] << ": " << v[i] << " ";
        out << std::endl;
    }
    size_t feature_size = 1+feature_location.size(); // +1 for intercept
    tipl::multiple_regression<double> mr;
    mr.set_variables(X.begin(),uint32_t(feature_size),uint32_t(subject_qa.size()));

    tipl::image<3> I(handle->dim);
    tipl::adaptive_par_for(I.size(),[&](size_t index)
    {
        if(vi2si[index])
        {
            //I[index] = subject_qa[selected_subject][vi2si[index]];
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
    if(index_name == "qa")
    {
        tipl::out() << "normalizing qa map" << std::endl;
        tipl::normalize(I);
    }
    volume.swap(I);
    return true;
}
bool connectometry_db::save_demo_matched_image(const std::string& matched_demo,const std::string& filename) const
{
    tipl::image<3> I;
    if(!get_demo_matched_volume(matched_demo,I))
        return false;
    if(!tipl::io::gz_nifti::save_to_file(filename.c_str(),I,handle->vs,handle->trans_to_mni,true,matched_demo.c_str()))
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
    tipl::adaptive_par_for(si2vi.size(),[&](unsigned int s_index)
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
    tipl::io::gz_mat_read single_subject;
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

bool connectometry_db::add_db(const connectometry_db& rhs)
{
    if(!is_db_compatible(rhs))
        return false;
    R2.insert(R2.end(),rhs.R2.begin(),rhs.R2.end());
    subject_names.insert(subject_names.end(),rhs.subject_names.begin(),rhs.subject_names.end());
    // copy the qa memory
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

void connectometry_db::calculate_change(unsigned char dif_type,unsigned char filter_type)
{
    std::ostringstream out;


    std::vector<std::string> new_subject_names(match.size());
    std::vector<float> new_R2(match.size());


    std::list<std::vector<float> > new_subject_qa_buf;
    std::vector<const float*> new_subject_qa;
    tipl::progress prog("calculating");
    for(unsigned int index = 0;prog(index,match.size());++index)
    {
        auto first = uint32_t(match[index].first);
        auto second = uint32_t(match[index].second);
        const float* scan1 = subject_qa[first];
        const float* scan2 = subject_qa[second];
        new_R2[index] = std::min<float>(R2[first],R2[second]);
        std::vector<float> change(subject_qa_length);
        switch(dif_type)
        {
        case 0:
            {
                if(!index)
                    out << " The difference between longitudinal scans were calculated by scan2-scan1.";
                for(unsigned int i = 0;i < subject_qa_length;++i)
                    change[i] = scan2[i]-scan1[i];
                new_subject_names[index] = subject_names[second] + "-" + subject_names[first];
            }
            break;
        case 1:
            {
                if(!index)
                    out << " The percentage difference between longitudinal scans were calculated by (scan2-scan1)/scan1.";
                for(unsigned int i = 0;i < subject_qa_length;++i)
                {
                    float s = scan1[i];
                    change[i] = (s == 0.0f? 0 : (scan2[i]-scan1[i])/s);
                }
                new_subject_names[index] = subject_names[second] + "-" + subject_names[first];
            }
            break;
        }

        switch(filter_type)
        {
        case 1: // increase only
            if(!index)
                out << " Only increased longitudinal changes were used in the analysis.";
            for(auto& v : change)
                if(v <= 0.0f)
                    v = 0.0;
            break;
        case 2: // decrease only
            if(!index)
                out << " Only decreased longitudinal changes were used in the analysis.";
            for(auto& v : change)
            {
                if(v >= 0.0f)
                    v = 0.0;
                v = -v;
            }
            break;
        }
        new_subject_qa_buf.push_back(change);
        new_subject_qa.push_back(&(new_subject_qa_buf.back()[0]));
    }
    out << " The total number of longitudinal subjects was " << match.size() << ".";
    R2.swap(new_R2);
    subject_names.swap(new_subject_names);
    subject_qa_buf.swap(new_subject_qa_buf);
    subject_qa.swap(new_subject_qa);
    num_subjects = uint32_t(match.size());
    match.clear();
    report += out.str();
    modified = true;
    is_longitudinal = true;
    longitudinal_filter_type = filter_type;

}





void connectometry_result::clear_result(char num_fiber,size_t image_size)
{
    inc.resize(num_fiber);
    dec.resize(num_fiber);
    inc_ptr.resize(num_fiber);
    dec_ptr.resize(num_fiber);
    for(char fib = 0;fib < num_fiber;++fib)
    {
        inc[fib].resize(image_size);
        dec[fib].resize(image_size);
        std::fill(inc[fib].begin(),inc[fib].end(),0.0);
        std::fill(dec[fib].begin(),dec[fib].end(),0.0);
        inc_ptr[fib] = &inc[fib][0];
        dec_ptr[fib] = &dec[fib][0];
    }
}

void stat_model::read_demo(const connectometry_db& db)
{
    selected_subject.resize(db.num_subjects);
    std::iota(selected_subject.begin(),selected_subject.end(),0);

    X = db.X;
    x_col_count = db.feature_location.size()+1; // additional one for longitudinal change

}

void stat_model::select_variables(const std::vector<char>& sel)
{
    auto row_count = X.size()/x_col_count;
    unsigned int new_col_count = 0;
    std::vector<size_t> feature_map;
    for(size_t i = 0;i < sel.size();++i)
        if(sel[i])
        {
            ++new_col_count;
            feature_map.push_back(i);
        }
    if(new_col_count == 1) // only testing longitudinal change
    {
        X.clear();
        return;
    }
    std::vector<double> new_X(size_t(row_count)*size_t(new_col_count));
    for(size_t i = 0,index = 0;i < row_count;++i)
        for(size_t j = 0;j < new_col_count;++j,++index)
            new_X[index] = X[i*x_col_count+feature_map[j]];
    x_col_count = new_col_count;
    X.swap(new_X);

}

bool stat_model::pre_process(void)
{
    if(X.empty())
        return true;
    X_min = X_max = std::vector<double>(X.begin(),X.begin()+x_col_count);
    auto subject_count = X.size()/x_col_count;
    if(subject_count <= x_col_count)
    {
        error_msg = "not enough subject samples for regression analysis";
        return false;
    }
    for(unsigned int i = 1,index = x_col_count;i < subject_count;++i)
        for(unsigned int j = 0;j < x_col_count;++j,++index)
        {
            if(X[index] < X_min[j])
                X_min[j] = X[index];
            if(X[index] > X_max[j])
                X_max[j] = X[index];
        }

    X_range = X_max;
    tipl::minus(X_range,X_min);

    {
        std::vector<double> sum(x_col_count);
        for(unsigned int pos = 0;pos < X.size();pos += x_col_count)
            tipl::add(sum.begin(),sum.end(),X.begin()+pos);
        tipl::divide_constant(sum,subject_count);
        X_mean.swap(sum);
    }

    return mr.set_variables(&*X.begin(),x_col_count,subject_count);
}

bool stat_model::select_cohort(connectometry_db& db,
                               std::string select_text)
{
    error_msg.clear();
    cohort_report.clear();
    remove_list.clear();
    remove_list.resize(X.size()/x_col_count);
    // handle missing value
    std::ostringstream out;
    for(size_t i = 0,pos = 0;pos < X.size();++i,pos += x_col_count)
    {
        for(size_t k = 0;k < db.feature_titles.size();++k)
            if(db.feature_selected[k] && std::isnan(X[pos+size_t(k)+1]))
            {
                out << i << " ";
                remove_list[i] = 1;
                break;
            }
    }
    {
        auto excluded_list = out.str();
        if(!excluded_list.empty())
           tipl::out() << "excluding subjects with missing values: " << excluded_list << std::endl;
    }

    if(!select_text.empty())
    {
        std::istringstream ss(select_text);
        std::string text;
        while(std::getline(ss,text,','))
        {
            tipl::out() << "selecting subjects with " << text << std::endl;
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
                                for(size_t i = 0,pos = 0;pos < X.size();++i,pos += x_col_count)
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

                    std::ostringstream out1;
                    for(size_t i = 0,pos = 0;pos < X.size();++i,pos += x_col_count)
                        if(!select(text[j],float(X[pos+fov_index+1]),threshold))
                        {
                            out1 << i << " ";
                            remove_list[i] = 1;
                        }
                    tipl::out() << "subjects excluded: " << out1.str() << std::endl;

                    {
                        std::ostringstream out;
                        std::string op_text;
                        if(text[j] == '=')
                            op_text = " is ";
                        if(text[j] == '<')
                            op_text = " less than ";
                        if(text[j] == '>')
                            op_text = " greater than ";
                        if(text[j] == '/')
                            op_text = " is not ";
                        out << " Subjects with " << text.substr(0,j) << op_text << text.substr(j+1,std::string::npos) << " were selected.";
                        cohort_report += out.str();
                    }
                    parsed = true;
                    break;
                }
            if(!parsed)
            {
                error_msg = "cannot parse selection text: ";
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
    sel[0] = 1; // intercept is always selected

    variables.clear();
    variables.push_back("longitudinal change");
    variables_is_categorical.clear();
    variables_is_categorical.push_back(0);
    variables_max.clear();
    variables_max.push_back(0);
    variables_min.clear();
    variables_min.push_back(0);


    bool has_variable = false;
    std::ostringstream out;
    for(size_t i = 1;i < sel.size();++i)
        if(db.feature_selected[i-1])
        {
            std::set<double> unique_values;
            for(size_t j = 0,pos = 0;pos < X.size();++j,pos += x_col_count)
                if(!remove_list[j] && X[pos+i] != NAN)
                    unique_values.insert(X[pos+i]);

            // if unique_values.size() = 1, then there is no variation to regress

            if(unique_values.size() > 1)
            {
                sel[i] = 1;
                variables.push_back(db.feature_titles[i-1]);
                out << variables.back();
                has_variable = true;

                bool is_categorical = (unique_values.size() <= 2);
                if(is_categorical)
                {
                    for(auto v : unique_values)
                        if(std::floor(v) != v)
                        {
                            is_categorical = false;
                            break;
                        }
                }

                variables_min.push_back(int(*(unique_values.begin())));
                variables_max.push_back(int(*(++unique_values.begin())));
                variables_is_categorical.push_back(is_categorical);
                if(is_categorical)
                    out << "(categorical)";
                out << " ";
            }
        }
    tipl::out() << "variables to be considered: "<< out.str() << std::endl;

    if(!has_variable && !db.is_longitudinal)
    {
        error_msg = "No variables selected for regression. Please check selected variables.";
        return false;
    }

    // select feature of interest
    {
        bool find_study_feature = false;
        // variables[0] = "longitudinal change"
        for(unsigned int i = 0;i < variables.size();++i)
            if(variables[i] == foi_text)
            {
                study_feature = i;
                find_study_feature = true;
                break;
            }
        if(!find_study_feature)
        {
            error_msg = "Invalid study variable: ";
            error_msg += foi_text;
            error_msg += " does not have variations";
            return false;
        }
        tipl::out() << "study variable: " << foi_text << std::endl;
    }
    select_variables(sel);

    // remove subjects according to the cohort selection
    for(int index = int(remove_list.size())-1;index >= 0;--index)
        if(remove_list[uint32_t(index)])
        {
            if(!X.empty())
                X.erase(X.begin()+int64_t(index)*x_col_count,X.begin()+(int64_t(index)+1)*x_col_count);
            selected_subject.erase(selected_subject.begin()+index);
        }

    if(!pre_process())
    {
        error_msg = "The subject number is not enough, or some demographic are duplicated. Please check the demographics.";
        return false;
    }
    return true;
}

bool stat_model::resample(stat_model& rhs,bool null,bool bootstrap,unsigned int seed)
{
    tipl::uniform_dist<int> rand_gen(0,1,seed);
    *this = rhs;
    {
        // resampling
        resample_order.clear();
        if(bootstrap)
        {
            resample_order.resize(rhs.selected_subject.size());
            for(unsigned int index = 0,pos = 0;index < rhs.selected_subject.size();++index,pos += x_col_count)
            {
                unsigned int new_index = resample_order[index] = uint32_t(rand_gen(uint32_t(rhs.selected_subject.size())));
                if(!X.empty())
                    std::copy(rhs.X.begin()+int64_t(new_index)*x_col_count,
                              rhs.X.begin()+int64_t(new_index)*x_col_count+x_col_count,X.begin()+pos);
            }
        }

        if(!X.empty())
        {
            std::vector<double> x_study_feature;
            for(size_t index = study_feature;index< X.size();index += x_col_count)
                x_study_feature.push_back(X[index]);
            x_study_feature_rank = tipl::rank(x_study_feature,std::less<double>());
            unsigned int n = uint32_t(X.size()/x_col_count);
            rank_c = 6.0/double(n)/double(n*n-1);
        }

        // compute permutation
        permutation_order.clear();
        if(study_feature)
        {
            if(null)
            {
                permutation_order.resize(selected_subject.size());
                for(int i = 0;i < permutation_order.size();++i)
                    permutation_order[i] = i;
                std::shuffle(permutation_order.begin(),permutation_order.end(),rand_gen.gen);
            }
        }
        else
        //if study longitudinal change, then permute the intercept sign
        {
            if(null)
            {
                permutation_order.resize(selected_subject.size());
                for(int i = 0;i < permutation_order.size();++i)
                    permutation_order[i] = rand_gen(2);
            }
        }
    }

    return true;
}
void stat_model::partial_correlation(std::vector<float>& population) const
{
    if(!X.empty())
    {
        std::vector<double> b(x_col_count);
        mr.regress(&*population.begin(),&*b.begin());
        for(size_t i = 1;i < x_col_count;++i) // skip intercept at i = 0
            if(i != study_feature)
            {
                auto mean = X_mean[i];
                auto cur_b = b[i];
                for(size_t j = 0,p = i;j < population.size();++j,p += x_col_count)
                    population[j] -= (mr.X[p]-mean)*cur_b;
            }
    }
}
double stat_model::operator()(const std::vector<float>& original_population) const
{
    std::vector<float> population(selected_subject.size());
    // apply resampling
    if(!resample_order.empty())
    {
        for(unsigned int index = 0;index < resample_order.size();++index)
            population[index] = original_population[resample_order[index]];
    }
    else
        population = original_population;

    // apply permutation
    if(!permutation_order.empty())
    {
        std::vector<float> permuted_population(population.size());
        if(study_feature)
        {
            for(unsigned int index = 0;index < permutation_order.size();++index)
                permuted_population[index] = population[permutation_order[index]];
        }
        else
        {
            for(unsigned int index = 0;index < permutation_order.size();++index)
                if(permutation_order[index])
                    permuted_population[index] = -population[index];
        }
        permuted_population.swap(population);
    }

    // calculate t-statistics
    if(study_feature)
    {
        auto rank = tipl::rank(population,std::less<float>());
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
    // if study longitudinal change
    {
        double mean = tipl::mean(population);
        double se = tipl::standard_deviation(population.begin(),population.end(),mean)/std::sqrt(population.size());
        return se == 0.0 ? 0.0 : mean/se;
    }
    return 0.0;
}
