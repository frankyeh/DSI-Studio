#include <filesystem>
#include <cstring>
#include <unordered_set>
#include "connectometry_db.hpp"
#include "fib_data.hpp"
#include "reg.hpp"

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
bool connectometry_db::load_db_from_fib(fib_data* handle_)
{
    handle = handle_;
    {
        std::vector<std::string> cur_subject_names;
        std::string subject_names_str,name;
        if(!handle->mat_reader.read("subject_names",subject_names_str))
            return true;
        std::istringstream in(subject_names_str);
        while(std::getline(in,name))
            cur_subject_names.push_back(name);
        std::vector<float> cur_R2(cur_subject_names.size());
        if(!handle->mat_reader.read("R2",cur_R2))
            return true;
        subject_names = std::move(cur_subject_names);
        R2 = std::move(cur_R2);
    }
    init_db();

    tipl::progress prog("loading database");
    size_t matrix_index = 0;
    // the old version database store subject matrices
    if(handle->mat_reader.has("subjects0") || handle->mat_reader.has("subject0"))
    {
        tipl::out() << "parsing old single-metric database";
        if(!handle->mat_reader.read("index_name",index_name))
        {
            handle->error_msg = "incompatible database format";
            return false;
        }
        // allocate memory in the mat reader
        auto mat = std::make_shared<tipl::io::mat_matrix>(index_name,float(0),subject_names.size(),mask_size);
        handle->mat_reader.push_back(mat);
        auto index_ptr = mat->get_data<float>();

        for(size_t index = 0;index < subject_names.size();++index,index_ptr += mask_size)
        {
            auto matrix_index = std::min<size_t>(handle->mat_reader.index_of("subjects"+std::to_string(index)),
                                                 handle->mat_reader.index_of("subject"+std::to_string(index)));
            if(matrix_index == handle->mat_reader.size() || handle->mat_reader[matrix_index].cols % mask_size)
            {
                handle->error_msg = "incompatible database format";
                return false;
            }
            std::copy_n(handle->mat_reader[matrix_index].get_data<float>(),mask_size,index_ptr);
            handle->mat_reader.remove(matrix_index); // save memory
        }
        index_list = {index_name};
    }
    else
    {
        for (unsigned int matrix_index = 0;matrix_index < handle->mat_reader.size();++matrix_index)
            if(handle->mat_reader[matrix_index].cols == mask_size &&
               handle->mat_reader[matrix_index].rows == subject_names.size())
            {
                // make sure the scale and inter are applied to uint8
                handle->mat_reader[matrix_index].convert_to<float>();
                index_list.push_back(handle->mat_reader[matrix_index].name);
            }
        if(index_list.empty())
        {
            handle->error_msg = "incompatible database format";
            return false;
        }
        tipl::out() << "available database: " << tipl::merge(index_list,',');
    }

    set_current_index(0);

    handle->mat_reader.read("subject_report",subject_report);


    // new db can be all positive, the checking the report text can confirm longitudinal setting
    if(handle->report.find("longitudinal scans were calculated") != std::string::npos)
    {
        is_longitudinal = true;
        if(handle->report.find("Only increased longitudinal changes") != std::string::npos)
            longitudinal_filter_type = 1;
        if(handle->report.find("Only decreased longitudinal changes") != std::string::npos)
            longitudinal_filter_type = 2;
    }


    if(handle->mat_reader.read("demo",demo))
        parse_demo();
    else
    {
        // try to get demo from subject name
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
    return true;
}

bool connectometry_db::parse_demo(const std::string& filename)
{
    std::ifstream in(filename);
    if(!in)
    {
        handle->error_msg = "cannot open the demographic file " + filename;
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
    std::string saved_demo(demo);
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
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            if(row_count == 0 && std::count(line.begin(),line.end(),',') < std::count(line.begin(),line.end(),'\t'))
                is_csv = false;

            {
                std::string col;
                std::istringstream in2(line);
                while(std::getline(in2,col,is_csv ? ',' : '\t'))
                    items.push_back(col);
            }


            if(items.size() == last_item_size)
                break;
            ++row_count;
            if(col_count == 0)
                col_count = items.size();
            else
            {
                while(items.size()-last_item_size < col_count)
                    items.push_back(std::string());
                while(items.size()-last_item_size > col_count)
                    items.pop_back();
            }
            last_item_size = items.size();
        }
        if(items.empty())
        {
            handle->error_msg = "no demographics";
            return false;
        }
        size_t found = 0;
        std::vector<std::string> new_items((subject_names.size()+1)*col_count);
        std::string first_column_title = QString::fromStdString(items.front()).toLower().toStdString();
        bool first_column_has_text = false;
        if(!tipl::contains(first_column_title,"age") && !tipl::contains(first_column_title,"sex"))
        {
            // copy titles
            std::copy_n(items.begin(),col_count,new_items.begin());
            for(size_t i = 0;i < subject_names.size();++i)
            {
                size_t matched_pos = 0;
                size_t matched_length = 0;
                for(size_t pos = col_count;pos+col_count <= items.size();pos += col_count)
                {
                    if(!items[pos].empty() && items[pos][0] >= 'A')
                        first_column_has_text = true;
                    if(subject_names[i].find(items[pos]) != std::string::npos ||
                       items[pos].find(subject_names[i]) != std::string::npos)
                    {
                        if(items[pos].size() > matched_length)
                        {
                            matched_pos = pos;
                            matched_length = items[pos].size();
                        }
                    }
                }
                if(matched_length > 1)
                {
                    ++found;
                    std::copy_n(items.begin()+matched_pos,col_count,new_items.begin() + size_t(i+1)*col_count);
                }
            }
        }
        if(items.size() != (subject_names.size()+1)*col_count)
        {
            if(found > subject_names.size()*0.75f)
            {
                tipl::out() << "rematch subject name with the first column.";
                items.swap(new_items);
            }
            else
            {
                std::ostringstream out;
                out << "Subject number mismatch. The demographic file has " << row_count-1 << " subject rows, but the database has " << subject_names.size() << " subjects.";
                handle->error_msg = out.str();
                return false;
            }
        }
        else
        {
            if(found == subject_names.size() && first_column_has_text)
            {
                tipl::out() << "rematch subject name with the first column.";
                items.swap(new_items);
            }
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
        for(char each : std::string("/://|*?<>\"\\^~[]"))
            std::replace(titles[i].begin(),titles[i].end(),each,'_');
        if(i)
            pout << ",";
        pout << titles[i];
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
            if(items[i] == " " || items[i] == "\r" || items[i] == "n/a" || items[i] == "na")
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
            if(not_number[i])
            {
                std::string group1(items[i]),group2;
                for(size_t j = i;j < items.size();j += titles.size())
                {
                    if(items[j] == group1)
                        continue;
                    if(group2.empty())
                        group2 = items[j];
                    if(items[j] != group2)
                    {
                        group1.clear();
                        break;
                    }
                }
                if(group1.empty())
                    continue;
                tipl::out() << "'" << titles[i] << "' treated as a group label, assign numbers 0:" << group1 << " 1:" << group2;
                titles[i] += "(0=" + group1 + " 1=" + group2 + ")";
                for(size_t j = i;j < items.size();j += titles.size())
                    items[j] = (items[j] == group1 ? "0":"1");
                not_number[i] = 0;
                not_categorical[i] = 0;
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
    for(unsigned int i = 0;i < subject_names.size();++i)
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
                handle->error_msg = out.str();
                X.clear();
                return false;
            }
        }
    }
    return true;
}

void connectometry_db::init_db(void)
{
    mask_size = handle->mat_reader.si2vi.size();
    vi2si.resize(handle->dim);
    size_t ms = 0;
    for(size_t index = 0;index < handle->dim.size();++index)
        if(handle->mask[index])
            vi2si[index] = (ms++);
}

bool connectometry_db::extract_indices(const std::string& file_name,const std::vector<std::string>& index_list_to_extract,
                                       float& R2,const std::vector<float*>& data)
{
    fib_data fib;
    if(!fib.load_from_file(file_name) ||
       fib.db.has_db() ||
       (fib.is_mni && !fib.mat_reader.read("R2",R2)))
    {
        handle->error_msg = "cannot use " + file_name + " as a subject file..." + fib.error_msg;
        return false;
    }
    if(!fib.is_mni)
    {
        fib.set_template_id(handle->template_id);
        if(!fib.map_to_mni(tipl::show_prog))
        {
            handle->error_msg = fib.error_msg + " at " + file_name;
            return false;
        }
        R2 = fib.R2;
    }

    auto sample = [this](tipl::const_pointer_image<3,float> I,const tipl::matrix<4,4>& trans,float* data)
    {
        tipl::image<3> J(handle->dim);
        if(I.shape() != J.shape() && trans != handle->trans_to_mni)
        {
            tipl::resample<tipl::interpolation::cubic>(I,J,
                tipl::transformation_matrix<float>(tipl::from_space(handle->trans_to_mni).to(trans)));
        }
        else
            J = I;
        tipl::lower_threshold(J,0.0f);
        const auto& si2vi = handle->mat_reader.si2vi;
        for(size_t i = 0;i < si2vi.size();++i)
            data[i] = J[si2vi[i]];
    };

    for(size_t i = 0;i < index_list_to_extract.size();++i)
    {
        auto index = fib.get_name_index(index_list_to_extract[i]);
        if(index == fib.slices.size())
        {
            handle->error_msg = "cannot find " + index_name + " in " + fib.fib_file_name;
            return false;
        }
        if(fib.is_mni)
            sample(fib.slices[index]->get_image(),fib.trans_to_mni,data[i]);
        else
            sample(tipl::compose_mapping(fib.slices[index]->get_image(),fib.t2s).alias(),fib.template_to_mni,data[i]);
        tipl::lower_threshold(data[i],data[i] + mask_size,0.0f);
    }
    return true;
}
void connectometry_db::set_current_index(size_t m)
{
    if(m < index_list.size())
    {
        auto index_ptr = handle->mat_reader[index_name = index_list[m]].get_data<float>();
        subject_indices.resize(subject_names.size());
        for(size_t i = 0;i < subject_indices.size();++i,index_ptr += mask_size)
            subject_indices[i] = index_ptr;
    }
}
bool connectometry_db::set_current_index(const std::string& name)
{
    for(size_t m = 0;m < index_list.size();++m)
        if(index_list[m] == name)
        {
            set_current_index(m);
            return true;
        }
    return false;
}
bool connectometry_db::create_db(const std::vector<std::string>& file_names,
                                 const std::vector<std::string>& included_index)
{        
    if(file_names.empty())
        return false;
    tipl::progress prog("create database",true);
    if(has_db())
    {
        handle->error_msg = "cannot create database from a database file " + handle->fib_file_name;
        return false;
    }
    if(!handle->is_mni)
    {
        handle->error_msg = "invalid template. not a QSDR FIB file: " + handle->fib_file_name;
        return false;
    }
    if(handle->mat_reader.si2vi.empty())
    {
        handle->error_msg = "invalid mask";
        return false;
    }
    init_db();
    fib_data fib;
    if(!fib.load_from_file(file_names[0]))
    {
        handle->error_msg = fib.error_msg + " at " + file_names[0];
        return false;
    }
    index_list = (!included_index.empty() ? included_index:fib.get_index_list());
    if(!add_subjects(file_names))
    {
        index_list.clear();
        return false;
    }
    handle->report = subject_report = fib.report;
    handle->intro = fib.intro;
    return true;
}

bool connectometry_db::add_subjects(const std::vector<std::string>& file_names)
{
    tipl::progress prog("extract data",true);
    bool failed = false;
    std::vector<std::shared_ptr<tipl::io::mat_matrix> > extracted_matrix;
    std::vector<std::string> extracted_subject_name(file_names.size());
    std::vector<float> extracted_R2(file_names.size());
    for(const auto& each : index_list)
    {
        extracted_matrix.push_back(std::make_shared<tipl::io::mat_matrix>(each,float(0),file_names.size() + subject_names.size(),mask_size));
        if(!subject_names.empty())
            std::copy_n(handle->mat_reader[each].get_data<float>(),subject_names.size()*mask_size,extracted_matrix.back()->get_data<float>());
    }
    size_t p = 0;
    tipl::par_for(file_names.size(),[&](size_t subject_index)
    {
        if(!prog(p++,file_names.size()) || failed)
            return;
        std::vector<float*> data(index_list.size());
        for(size_t i = 0;i < extracted_matrix.size();++i)
            data[i] = extracted_matrix[i]->get_data<float>()+mask_size*(subject_index + subject_names.size());
        if(!extract_indices(file_names[subject_index],index_list,extracted_R2[subject_index],data))
        {
            failed = true;
            return;
        }
        extracted_subject_name[subject_index] = tipl::split(std::filesystem::path(file_names[subject_index]).filename().string(),'.')[0];
    });
    if(prog.aborted() || failed)
        return false;
    R2.insert(R2.end(),extracted_R2.begin(),extracted_R2.end());
    subject_names.insert(subject_names.end(),extracted_subject_name.begin(),extracted_subject_name.end());
    for(auto each : extracted_matrix)
        handle->mat_reader.push_back(each);
    set_current_index(index_name);
    modified = true;
    return true;

}

bool connectometry_db::add_db(const std::string& file_name)
{
    fib_data fib;
    if(!fib.load_from_file(file_name))
        return false;
    if(!fib.db.has_db())
    {
        handle->error_msg = "not a database file: " + file_name;
        return false;
    }
    if(fib.dim != handle->dim || fib.db.mask_size != mask_size)
    {
        handle->error_msg = "mask does not match";
        return false;
    }
    for(const auto& each : index_list)
        if(!fib.mat_reader.has(each))
        {
            handle->error_msg = "cannot find " + each + " in database " + file_name;
            return false;
        }
    if(!std::equal(handle->mask.begin(),handle->mask.end(),fib.mask.begin()))
        {
            handle->error_msg = "The connectometry db was created using a different template.";
            return false;
        }

    tipl::progress prog("copying data",true);
    // load all delayed load data
    for(const auto& each : index_list)
    {
        handle->mat_reader[each].convert_to<float>();
        handle->mat_reader[each].set_row_col(subject_names.size()+fib.db.subject_names.size(),mask_size);
    }
    size_t p = 0;
    for(const auto& each : index_list)
    {
        prog(p++,index_list.size());
        auto src_ptr = fib.mat_reader[each].get_data<float>();
        auto dst_ptr = handle->mat_reader[each].get_data<float>();
        std::copy_n(src_ptr,fib.db.subject_names.size()*mask_size,dst_ptr + subject_names.size()*mask_size);
    }
    R2.insert(R2.end(),fib.db.R2.begin(),fib.db.R2.end());
    subject_names.insert(subject_names.end(),fib.db.subject_names.begin(),fib.db.subject_names.end());
    set_current_index(index_name);
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
            slice[index] = subject_indices[subject_index][tmp[index]];
}

bool connectometry_db::get_demo_matched_volume(const std::string& matched_demo,tipl::image<3>& volume) const
{
    if(demo.empty())
    {
        handle->error_msg = "no demographic data found in the database";
        return false;
    }
    if(matched_demo.empty())
    {
        handle->error_msg = "no demographics provided for the study subject";
        return false;
    }

    std::vector<double> v;
    {
        std::string s(matched_demo);
        std::replace(s.begin(),s.end(),',',' ');
        std::istringstream in(s);
        while(in)
        {
            std::string str;
            in >> str;
            try{
                float value = std::stof(str);
                v.push_back(value);
            }
            catch (...){}
        }
        if(v.size() != feature_location.size())
        {
            handle->error_msg = "invalid demographic input: " + matched_demo;
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
    mr.set_variables(X.begin(),uint32_t(feature_size),uint32_t(subject_indices.size()));


    tipl::image<3> I(handle->dim);
    const auto& si2vi = handle->mat_reader.si2vi;
    tipl::adaptive_par_for(si2vi.size(),[&](size_t index)
    {
        std::vector<double> y(subject_indices.size());
        for(size_t s = 0;s < subject_indices.size();++s)
            y[s] = double(subject_indices[s][index]);
        std::vector<double> b(feature_size);
        mr.regress(y.begin(),b.begin());
        double predict = b[0];
        for(size_t i = 1;i < b.size();++i)
            predict += b[i]*v[i-1];
        I[si2vi[index]] = std::max<float>(0.0f,float(predict));
    });
    volume.swap(I);
    return true;
}
void connectometry_db::get_avg_volume(tipl::image<3>& volume) const
{
    tipl::image<3> I(handle->dim);
    const auto& si2vi = handle->mat_reader.si2vi;
    tipl::adaptive_par_for(si2vi.size(),[&](size_t index)
    {
        std::vector<float> data(subject_indices.size());
        for(size_t s = 0;s < subject_indices.size();++s)
            data[s] = double(subject_indices[s][index]);
        I[si2vi[index]] = tipl::mean(data.begin(),data.end());
    });
    volume.swap(I);
}
bool connectometry_db::save_demo_matched_image(const std::string& matched_demo,const std::string& filename) const
{
    tipl::image<3> I;
    if(!get_demo_matched_volume(matched_demo,I))
        return false;
    return tipl::io::gz_nifti(filename,std::ios::out)
            << handle->bind(I)
            << [&](const std::string& e){tipl::error() << (handle->error_msg = e);};
}
tipl::image<3> connectometry_db::get_index_image(unsigned int subject_index) const
{
    tipl::image<3> result(handle->dim);
    const auto& si2vi = handle->mat_reader.si2vi;
    for(size_t si = 0;si < si2vi.size();++si)
        result[si2vi[si]] = subject_indices[subject_index][si];
    return result;
}


void connectometry_db::remove_subject(unsigned int index)
{
    if(index >= subject_names.size())
        return;
    subject_names.erase(subject_names.begin()+index);
    R2.erase(R2.begin()+index);
    size_t new_n = subject_names.size();
    size_t offset = index * mask_size;
    size_t copy_count = (new_n - index)*mask_size;
    if(copy_count)
        for(const auto& each : index_list)
        {
            auto ptr = handle->mat_reader[each].get_data<float>() + offset;
            std::memmove(ptr, ptr + mask_size, copy_count * sizeof *ptr);
            handle->mat_reader[each].set_row_col(new_n,mask_size);
        }
    subject_indices.resize(new_n);
    modified = true;
}

void connectometry_db::move_up(int id)
{
    if(id == 0)
        return;
    std::swap(subject_names[uint32_t(id)],subject_names[uint32_t(id-1)]);
    std::swap(R2[uint32_t(id)],R2[uint32_t(id-1)]);
    for(const auto& each : index_list)
    {
        auto ptr = handle->mat_reader[each].get_data<float>() + id*mask_size;
        std::swap_ranges(ptr, ptr + mask_size, ptr - mask_size);
    }
}

void connectometry_db::move_down(int id)
{
    if(uint32_t(id) >= subject_names.size()-1)
        return;
    std::swap(subject_names[uint32_t(id)],subject_names[uint32_t(id+1)]);
    std::swap(R2[uint32_t(id)],R2[uint32_t(id+1)]);
    for(const auto& each : index_list)
    {
        auto ptr = handle->mat_reader[each].get_data<float>() + id*mask_size;
        std::swap_ranges(ptr, ptr + mask_size, ptr + mask_size);
    }
}
bool can_be_normalized_by_iso(const std::string& name);
void normalize_data_by_iso(const float* iso_ptr,float* out_data_ptr,size_t n);
void connectometry_db::calculate_change(unsigned char dif_type,unsigned char filter_type,bool normalize_iso)
{

    std::vector<std::shared_ptr<tipl::io::mat_matrix> > dif_matrices;

    // for used when normalized by iso
    std::vector<float> scan1_buf(mask_size),scan2_buf(mask_size);
    std::vector<std::string> index_normalized_by_iso;


    tipl::progress prog("calculating");

    for(size_t m = 0;prog(m,index_list.size());++m)
    {
        const float* iso_ptr = nullptr;
        if(normalize_iso && can_be_normalized_by_iso(index_list[m]))
        {
            iso_ptr = handle->mat_reader["iso"].get_data<float>();
            index_normalized_by_iso.push_back(index_list[m]);
        }

        auto ptr = handle->mat_reader[index_list[m]].get_data<float>();
        auto dif_mat = std::make_shared<tipl::io::mat_matrix>(index_list[m],float(0),match.size(),mask_size);
        auto change = dif_mat->get_data<float>();

        for(const auto& each : match)
        {
            auto first_base = each.first*mask_size;
            auto second_base = each.second*mask_size;

            auto scan1 = ptr + first_base;
            auto scan2 = ptr + second_base;
            if(iso_ptr)
            {
                std::copy_n(scan1,mask_size,scan1_buf.data());
                std::copy_n(scan2,mask_size,scan2_buf.data());
                normalize_data_by_iso(iso_ptr + first_base,scan1_buf.data(),mask_size);
                normalize_data_by_iso(iso_ptr + second_base,scan2_buf.data(),mask_size);
                scan1 = scan1_buf.data();
                scan2 = scan2_buf.data();
            }

            for(unsigned int i = 0;i < mask_size;++i)
                change[i] = scan2[i]-scan1[i];
            if(dif_type == 1)
                for(unsigned int i = 0;i < mask_size;++i)
                    change[i] = (scan1[i] == 0.0f ? 0.0f : change[i] / scan1[i]);

            if(filter_type == 2)
                tipl::neg(change,change+mask_size);
            if(filter_type)  // 1 or 2
                tipl::lower_threshold(change,change+mask_size,0.0f);

            change += mask_size;
        }
        dif_matrices.push_back(dif_mat);
    }
    if(prog.aborted())
        return;


    {
        std::vector<std::string> new_subject_names(match.size());
        std::vector<float> new_R2(match.size());
        for(size_t index = 0;index < match.size();++index)
        {
            new_subject_names[index] = subject_names[match[index].second] + "-" + subject_names[match[index].first];
            new_R2[index] = std::min<float>(R2[match[index].first],R2[match[index].second]);
        }
        R2.swap(new_R2);
        subject_names.swap(new_subject_names);
    }

    if(dif_type)
        handle->report += " The percentage difference between longitudinal scans were calculated by (scan2-scan1)/scan1.";
    else
        handle->report += " The difference between longitudinal scans were calculated by scan2-scan1.";

    if(filter_type)
    {
        if(filter_type == 1)
            handle->report += " Only increased longitudinal changes were used in the analysis.";
        if(filter_type == 2)
            handle->report += " Only decreased longitudinal changes were used in the analysis.";
    }
    if(!index_normalized_by_iso.empty())
        handle->report += " " + tipl::merge(index_normalized_by_iso,',') + " were normalized by iso.";
    handle->report += " The total number of longitudinal subjects was " + std::to_string(match.size()) + ".";
    match.clear();

    for(const auto& each : dif_matrices)
        handle->mat_reader.push_back(each);
    set_current_index(0);

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
    selected_subject.resize(db.subject_names.size());
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
    tipl::minus(X_range.begin(),X_range.end(),X_min.begin());

    {
        std::vector<double> sum(x_col_count);
        for(unsigned int pos = 0;pos < X.size();pos += x_col_count)
            tipl::add(sum.begin(),sum.end(),X.begin()+pos);
        tipl::divide_constant(sum.begin(),sum.end(),subject_count);
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

                    if(fov_name == "subject")
                    {
                        for(size_t i = 0;i < db.subject_names.size();++i)
                        {
                            if(text[j] == '=')
                                remove_list[i] = (db.subject_names[i] == value_text ? 1:0);
                            if(text[j] == '/')
                                remove_list[i] = (db.subject_names[i] != value_text ? 1:0);
                            if(text[j] == '>')
                                remove_list[i] = (tipl::contains(db.subject_names[i],value_text) ? 1:0);
                            if(text[j] == '<')
                                remove_list[i] = (tipl::contains(db.subject_names[i],value_text) ? 0:1);
                        }
                        parsed = true;
                        break;
                    }

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
                    std::copy_n(rhs.X.begin()+int64_t(new_index)*x_col_count,x_col_count,X.begin()+pos);
            }
        }

        if(!X.empty())
        {
            std::vector<double> x_study_feature;
            for(size_t index = study_feature;index< X.size();index += x_col_count)
                x_study_feature.push_back(X[index]);
            x_study_feature_rank = tipl::rank_avg_tie(x_study_feature,std::less<double>());
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
        auto ry = tipl::rank_avg_tie(population, std::less<float>());
        if(ry.size() < 3)
            return 0.0;
        double r = tipl::correlation(ry.begin(),ry.end(),x_study_feature_rank.begin()); // Spearman rho (tie-safe)
        double t = r * std::sqrt((ry.size() - 2.0) / (1.0 - r*r));
        return std::isnormal(t) ? t : 0.0;
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
