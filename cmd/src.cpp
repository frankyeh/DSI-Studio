#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <iostream>
#include <filesystem>
#include <iterator>
#include <string>
#include "dicom/dwi_header.hpp"
extern std::string src_error_msg;
QStringList search_files(QString dir,QString filter);
bool load_bval(const char* file_name,std::vector<double>& bval);
bool load_bvec(const char* file_name,std::vector<double>& b_table,bool flip_by = true);
bool parse_dwi(const std::vector<std::string>& file_list,
                    std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
void dicom2src_and_nii(std::string dir_);
bool load_4d_nii(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,bool need_bvalbvec);

bool get_bval_bvec(const std::string& bval_file,const std::string& bvec_file,size_t dwi_count,
                   std::vector<double>& bvals_out,std::vector<double>& bvecs_out,
                   std::string& error_msg)
{
    std::vector<double> bvals,bvecs;
    if(!load_bval(bval_file.c_str(),bvals))
    {
        error_msg = "cannot load bval at ";
        error_msg += bval_file;
        return false;
    }
    if(!load_bvec(bvec_file.c_str(),bvecs))
    {
        error_msg = "cannot load bvec at ";
        error_msg += bvec_file;
        return false;
    }
    if(!bvals.empty() && dwi_count != bvals.size())
    {
        std::ostringstream out;
        out << "bval number does not match NIFTI file: " << dwi_count << " DWI in nifti file and " << bvals.size() << " in bvals " << std::endl;
        error_msg = out.str();
        return false;
    }
    if(bvals.size()*3 != bvecs.size())
    {
        error_msg = "bval and bvec does not match";
        return false;
    }
    bvals_out.swap(bvals);
    bvecs_out.swap(bvecs);
    return true;
}
bool create_src(const std::vector<std::string>& nii_names,std::string src_name)
{
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    for(auto& nii_name : nii_names)
    {
        tipl::out() << "opening " << nii_name;
        if(!load_4d_nii(nii_name.c_str(),dwi_files,true))
            tipl::warning() << "skipping " << nii_name << ": " << src_error_msg;
    }
    if(!DwiHeader::output_src(src_name.c_str(),dwi_files,0,false))
    {
        tipl::error() << src_error_msg;
        return false;
    }
    return true;
}
bool create_src(std::string nii_name,std::string src_name)
{
    std::vector<std::string> nii_names;
    nii_names.push_back(nii_name);
    return create_src(nii_names,src_name);
}

bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec);
bool is_dwi_nii(const std::string& nii_name)
{
    QString bval_name,bvec_name;
    return find_bval_bvec(nii_name.c_str(),bval_name,bvec_name);
}
void search_dwi_nii(const std::string& dir,std::vector<std::string>& dwi_nii_files)
{
    std::vector<std::string> nii_files;
    tipl::search_files(dir,"*.nii.gz",nii_files);
    tipl::search_files(dir,"*.nii",nii_files);
    for(auto& each : nii_files)
        if(is_dwi_nii(each))
            dwi_nii_files.push_back(each);
}

std::vector<std::string> search_dwi_nii_bids(const std::string& dir)
{
    std::vector<std::string> dwi_nii_files;
    tipl::progress prog("searching BIDS at ",dir.c_str());
    std::vector<std::string> sub_dir;
    tipl::search_dirs(dir,"sub-*",sub_dir);
    auto subject_num = sub_dir.size();
    for(int j = 0;prog(j,sub_dir.size());++j)
    {
        // look for sessions
        if(j < subject_num)
            tipl::search_dirs(sub_dir[j],"ses-*",sub_dir);
        tipl::out() << "searching " << sub_dir[j];
        search_dwi_nii(sub_dir[j] + "/dwi",dwi_nii_files);
    }
    return dwi_nii_files;
}

bool handle_bids_folder(const std::vector<std::string>& dwi_nii_files,
                        const std::string& output_dir,
                        bool overwrite,
                        std::string& error_msg)
{
    std::vector<std::string> dwi_file;
    std::vector<tipl::shape<3> > image_size;
    std::vector<size_t> dwi_count;
    std::vector<std::string> phase_dir;
    // extract all information
    for(const auto& each : dwi_nii_files)
    {
        if (each.size() < 7 || each.compare(each.size() - 7, 7, ".nii.gz") != 0)
        {
            error_msg = "invalid file name: ";
            error_msg += each;
            return false;
        }
        std::string json(each),phase_str;
        json.replace(json.size() - 7, 7, ".json");
        QFile input_file(json.c_str());
        if (input_file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            tipl::out () << "read jaon file : " << json;
            QJsonDocument json_doc = QJsonDocument::fromJson(input_file.readAll());
            if (json_doc.isObject())
            {
                QJsonObject json_obj = json_doc.object();
                if (json_obj.contains("PhaseEncodingDirection"))
                    phase_str = json_obj["PhaseEncodingDirection"].toString().toStdString();
            }
        }
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(each))
        {
            error_msg = nii.error_msg;
            return false;
        }
        dwi_count.push_back(nii.dim(4));
        tipl::shape<3> s;
        nii.get_image_dimension(s);
        image_size.push_back(s);
        phase_dir.push_back(phase_str);

        dwi_file.push_back(each);
        tipl::out() << std::filesystem::path(dwi_file.back()).filename().u8string();
        tipl::out() << "image size: " << image_size.back();
        tipl::out() << "dwi count: " << dwi_count.back();
        tipl::out() << "phase encoding: " << phase_dir.back();
    }
    auto arg = tipl::arg_sort(dwi_count,std::greater<float>());
    tipl::reorder(dwi_file,arg);
    tipl::reorder(image_size,arg);
    tipl::reorder(dwi_count,arg);
    tipl::reorder(phase_dir,arg);
    // for each image size, generate an SRC
    for(size_t i = 0;i < arg.size();++i)
    {
        if(!image_size[i].size())
            continue;
        std::vector<std::string> main_dwi_list,rev_pe_list;
        main_dwi_list.push_back(dwi_file[i]);
        for(size_t j = i + 1;j < arg.size();++j)
            if(image_size[i] == image_size[j])
            {
                if(phase_dir[i] == phase_dir[j])
                    main_dwi_list.push_back(dwi_file[j]);
                else
                    rev_pe_list.push_back(dwi_file[j]);
                image_size[j] = tipl::shape<3>();
            }
        image_size[i] = tipl::shape<3>();
        dwi_file[i].erase(dwi_file[i].length() - 7, 7); // remove .nii.gz
        auto src_name = output_dir + "/" + std::filesystem::path(dwi_file[i]).filename().u8string() + ".src.gz";
        auto rsrc_name = output_dir + "/" + std::filesystem::path(dwi_file[i]).filename().u8string() + ".rsrc.gz";
        if(!overwrite && std::filesystem::exists(src_name))
            tipl::out() << "skipping " << src_name << " already exists";
        else
            if(!create_src(main_dwi_list,src_name))
                return false;
        if(!rev_pe_list.empty())
        {
            if(!overwrite && std::filesystem::exists(rsrc_name))
                tipl::out() << "skipping " << rsrc_name << " already exists";
            else
                if(!create_src(rev_pe_list,rsrc_name))
                    return false;
        }
    }
    return true;
}

bool nii2src(const std::vector<std::string>& dwi_nii_files,
             const std::string& output_dir,
             bool is_bids,
             bool overwrite)
{
    for(size_t i = 0;i < dwi_nii_files.size();++i)
    {
        if(is_bids)
        {
            std::vector<std::string> dwi_list;
            dwi_list.push_back(dwi_nii_files[i]);
            for(size_t j = i + 1;j < dwi_nii_files.size();++j)
                if(std::filesystem::path(dwi_nii_files[j]).parent_path() == std::filesystem::path(dwi_nii_files[i]).parent_path())
                {
                    ++i;
                    dwi_list.push_back(dwi_nii_files[i]);
                }
                else
                    break;
            std::string error_msg;
            if(!handle_bids_folder(dwi_list,output_dir,overwrite,error_msg))
            {
                tipl::error() << error_msg;
                return false;
            }
        }
        else
        {
            auto src_name = output_dir + "/" + std::filesystem::path(dwi_nii_files[i]).filename().u8string() + ".src.gz";
            if(!overwrite && std::filesystem::exists(src_name))
                tipl::out() << "skipping " << src_name << " already exists";
            else
                if(!create_src(dwi_nii_files[i],src_name))
                    return false;
        }
    }
    return true;
}

int src(tipl::program_option<tipl::out>& po)
{      
    std::string source = po.get("source");
    std::vector<std::string> file_list;
    if(std::filesystem::is_directory(source))
    {

        // handling NIFTI files
        {
            auto dwi_nii_files = search_dwi_nii_bids(source);
            bool is_bids = po.get("bids",dwi_nii_files.empty() ? 0 : 1);

            if(dwi_nii_files.empty())
            {
                tipl::out() << "searching NIFTI files in the folder";
                search_dwi_nii(source,dwi_nii_files);
            }

            if(!dwi_nii_files.empty())
            {
                auto output_dir = po.get("output",source);
                if(!std::filesystem::exists(output_dir) && !std::filesystem::create_directory(output_dir))
                {
                    tipl::error() << "cannot create the output folder. please check write privileges";
                    return 1;
                }
                if(!std::filesystem::is_directory(output_dir))
                {
                    tipl::error() << output_dir << " is not a valid output directory.";
                    return 1;
                }
                std::sort(dwi_nii_files.begin(),dwi_nii_files.end());
                if(nii2src(dwi_nii_files,output_dir,is_bids,po.get("overwrite",0)))
                    return 0;
                return 1;
            }
        }

        tipl::out() << "searching DICOM files in directory " << source.c_str() << std::endl;
        tipl::search_files(source,"*.dcm",file_list);
        if(file_list.empty())
            tipl::search_files(source,"*.fdf",file_list);
        if(file_list.empty())
        {
            dicom2src_and_nii(source);
            return 0;
        }
        tipl::out() << "a total of " << file_list.size() << " files found in the directory" << std::endl;
    }
    else
        file_list.push_back(source);

    if(po.has("other_source"))
        po.get_files("other_source",file_list);

    if(file_list.empty())
    {
        tipl::error() << "no file found for creating src" << std::endl;
        return 1;
    }

    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    if(!parse_dwi(file_list,dwi_files))
    {
        tipl::error() << "cannot read dwi file: " << src_error_msg << std::endl;
        return 1;
    }
    if(po.has("b_table"))
    {
        std::string table_file_name = po.get("b_table");
        std::ifstream in(table_file_name.c_str());
        if(!in)
        {
            tipl::error() << "failed to open b-table" <<std::endl;
            return 1;
        }
        std::string line;
        std::vector<double> b_table;
        while(std::getline(in,line))
        {
            std::istringstream read_line(line);
            std::copy(std::istream_iterator<double>(read_line),
                      std::istream_iterator<double>(),
                      std::back_inserter(b_table));
        }
        if(b_table.size() != dwi_files.size()*4)
        {
            tipl::error() << "mismatch between b-table and the loaded images" << std::endl;
            return 1;
        }
        for(unsigned int index = 0,b_index = 0;index < dwi_files.size();++index,b_index += 4)
        {
            dwi_files[index]->bvalue = float(b_table[b_index]);
            dwi_files[index]->bvec = tipl::vector<3>(b_table[b_index+1],b_table[b_index+2],b_table[b_index+3]);
        }
        tipl::out() << "b-table " << table_file_name << " loaded" << std::endl;
    }
    if(po.has("bval") && po.has("bvec"))
    {
        std::vector<double> bval,bvec;
        std::string error_msg;
        if(!get_bval_bvec(po.get("bval"),po.get("bvec"),dwi_files.size(),bval,bvec,error_msg))
        {
            tipl::error() << error_msg;
            return 1;
        }
        for(unsigned int index = 0;index < dwi_files.size();++index)
        {
            dwi_files[index]->bvalue = float(bval[index]);
            dwi_files[index]->bvec = tipl::vector<3>(bvec[index*3],bvec[index*3+1],bvec[index*3+2]);
        }
    }
    if(dwi_files.empty())
    {
        tipl::error() << "no file readed. Abort." << std::endl;
        return 1;
    }

    double max_b = 0;
    for(unsigned int index = 0;index < dwi_files.size();++index)
    {
        if(dwi_files[index]->bvalue < 100.0f)
            dwi_files[index]->bvalue = 0.0f;
        max_b = std::max(max_b,double(dwi_files[index]->bvalue));
    }
    if(max_b == 0.0)
    {
        tipl::error() << "cannot create SRC file: " << src_error_msg;
        return 1;
    }

    auto output = po.get("output",file_list.front() + ".src.gz");

    if(std::filesystem::is_directory(output))
        output += std::string("/") + std::filesystem::path(file_list[0]).filename().u8string() + ".src.gz";
    if(!tipl::ends_with(output,".src.gz"))
        output += ".src.gz";

    if(!po.get("overwrite",0) && std::filesystem::exists(output))
    {
        tipl::out() << "skipping " << output << " already exists";
        return 0;
    }
    if(!DwiHeader::output_src(output.c_str(),dwi_files,po.get<int>("up_sampling",0),po.get<int>("sort_b_table",0)))
    {
        tipl::error() << src_error_msg << std::endl;
        return 1;
    }
    return 0;
}
