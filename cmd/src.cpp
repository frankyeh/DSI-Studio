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
#include "image_model.hpp"
QStringList search_files(QString dir,QString filter);
bool load_bval(const std::string& file_name,std::vector<double>& bval);
bool load_bvec(const std::string& file_name,std::vector<double>& b_table,bool flip_by = true);
bool parse_dwi(const std::vector<std::string>& file_list,
                    std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg);
void dicom2src_and_nii(std::string dir_,bool overwrite);

bool get_bval_bvec(const std::string& bval_file,const std::string& bvec_file,size_t dwi_count,
                   std::vector<double>& bvals_out,std::vector<double>& bvecs_out,
                   std::string& error_msg)
{
    std::vector<double> bvals,bvecs;
    if(!load_bval(bval_file,bvals))
    {
        error_msg = "cannot load bval at ";
        error_msg += bval_file;
        return false;
    }
    if(!load_bvec(bvec_file,bvecs))
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

bool find_readme(const std::string& file,std::string& intro_file_name)
{
    auto path = std::filesystem::path(file).parent_path();
    for (int i = 0; i < 3; ++i)
    {
        auto readme_path = path / "README";
        if (std::filesystem::exists(readme_path))
        {
            tipl::out() << "README file found at " << (intro_file_name = readme_path.string());
            return true;
        }
        path = path.parent_path();
    }
    return false;
}

bool find_bval_bvec(const std::string& file_name, std::string& bval, std::string& bvec);
bool is_dwi_nii(const std::string& nii_name)
{
    std::string bval_name,bvec_name;
    return find_bval_bvec(nii_name,bval_name,bvec_name);
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
    tipl::progress prog("searching BIDS in " + dir);
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
                        bool topup_eddy,
                        std::string& error_msg)
{
    if(dwi_nii_files.empty())
        return false;

    std::vector<std::tuple<std::filesystem::path/*file path*/,
                           std::string /*phase dir*/,
                           tipl::shape<3> /*image dimension*/,
                           size_t/*dwi count*/> > dwi_info;
    for(const auto& each : dwi_nii_files)
    {
        if (!tipl::ends_with(each,".nii.gz") &&
            !tipl::ends_with(each,".nii"))
        {
            tipl::out() << "ignore " << each << " : not a nifti file";
            continue;
        }
        if (tipl::contains(each,".sz."))
        {
            tipl::out() << "ignore file with '.sz.':" << each;
            continue;
        }
        auto file_name = std::filesystem::path(each).filename().u8string();
        tipl::out() << "opening " << file_name;

        std::string json(each),phase_str;
        json.erase(json.size()-7);
        json += ".json";

        QFile input_file(json.c_str());
        if (input_file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            tipl::out () << "read json : " << json;
            QJsonDocument json_doc = QJsonDocument::fromJson(input_file.readAll());
            if (json_doc.isObject() && json_doc.object().contains("PhaseEncodingDirection"))
                phase_str = json_doc.object()["PhaseEncodingDirection"].toString().toStdString();
            else
                tipl::out() << "json file does not include PhaseEncodingDirection information";
        }

        if(phase_str.empty())
        {
            for(auto dir : {"_ap","ap_","_pa","pa_","_lr","lr_","_rl","rl_"})
                if(tipl::contains_case_insensitive(file_name,dir))
                {
                    tipl::out() << file_name << " filename suggests phase direction is " << (phase_str = dir);
                    break;
                }
        }

        tipl::io::gz_nifti nii;
        if(!nii.open(each,std::ios::in))
        {
            error_msg = nii.error_msg;
            return false;
        }
        tipl::shape<3> s;
        nii >> s;
        dwi_info.emplace_back(std::filesystem::path(each),phase_str,s,nii.dim(4));
        tipl::out() << "image size: " << s << " dwi count: " << nii.dim(4) << " phase encoding: " << phase_str;
    }

    // sort based on dwi count
    std::sort(dwi_info.begin(),dwi_info.end(),
              [&](const auto& lhs,const auto& rhs){return std::get<size_t>(lhs) > std::get<size_t>(rhs);});

    // for each image size, generate an SRC
    for(size_t i = 0;i < dwi_info.size();++i)
    {
        if(!std::get<tipl::shape<3> >(dwi_info[i]).size())
            continue;
        std::vector<std::string> main_dwi_list,rev_pe_list;
        main_dwi_list.push_back(std::get<std::filesystem::path>(dwi_info[i]).string());
        tipl::out() << "creating src for " << main_dwi_list.back();
        // match phase encoding
        for(size_t j = i + 1;j < dwi_info.size();++j)
            if(std::get<tipl::shape<3> >(dwi_info[i]) == std::get<tipl::shape<3> >(dwi_info[j])) // image dimension the same
            {
                if(std::get<std::string>(dwi_info[i]) == std::get<std::string>(dwi_info[j])) //phase direction the same
                {
                    main_dwi_list.push_back(std::get<std::filesystem::path>(dwi_info[j]).string());
                    tipl::out() << "adding " << main_dwi_list.back();
                }
                else
                {
                    rev_pe_list.push_back(std::get<std::filesystem::path>(dwi_info[j]).string());
                    tipl::out() << "reverse encoding adding " << rev_pe_list.back();

                }
                std::get<tipl::shape<3> >(dwi_info[j]) = tipl::shape<3>();
            }
        std::get<tipl::shape<3> >(dwi_info[i]) = tipl::shape<3>();


        auto dwi_file_name = main_dwi_list.back();
        if(!output_dir.empty())
            dwi_file_name = output_dir + "/" + tipl::split(std::filesystem::path(dwi_file_name).filename().string(),'.').front();
        else
            dwi_file_name.erase(dwi_file_name.length() - 7, 7); // remove .nii.gz
        auto src_name = dwi_file_name + ".sz";
        auto rsrc_name = dwi_file_name + ".rz";
        if(!overwrite && std::filesystem::exists(src_name))
        {
            tipl::out() << "skipping " << src_name << ": already exists";
            continue;
        }

        src_data src;
        if(!src.load_from_file(main_dwi_list,true))
        {
            error_msg = src.error_msg;
            return false;
        }

        {
            std::string intro_file_name("README");
            if(!std::filesystem::exists(intro_file_name))
                find_readme(main_dwi_list[0],intro_file_name);
            if(std::filesystem::exists(intro_file_name))
                src.load_intro(intro_file_name);
        }

        if(!rev_pe_list.empty())
        {
            src.rev_pe_src = std::make_shared<src_data>();
            if(!src.rev_pe_src->load_from_file(rev_pe_list,rev_pe_list.size() > 1 /*if more than one file, need bval bvec*/))
            {
                error_msg = src.error_msg;
                return false;
            }
        }

        if(topup_eddy)
        {
            if(!src.rev_pe_src.get())
                src.get_rev_pe(std::string());
            if(src.rev_pe_src.get())
            {
                if(!src.run_topup())
                    tipl::warning() << src.error_msg;
            }
            else
                tipl::out() << "no reverse pe data. skip topup";
            if(!src.run_eddy())
                tipl::warning() << src.error_msg;
        }
        else
        {
            if(src.rev_pe_src.get())
            {
                if(!src.rev_pe_src->save_to_file(rsrc_name))
                {
                    error_msg = src.rev_pe_src->error_msg;
                    return false;
                }
            }
        }
        if(!src.save_to_file(src_name))
        {
            error_msg = src.error_msg;
            return false;
        }
    }
    return true;
}

bool nii2src(const std::vector<std::string>& dwi_nii_files,
             const std::string& output_dir,
             bool is_bids,
             bool overwrite,
             bool topup_eddy)
{
    tipl::progress prog("convert nifti to src files");
    for(size_t i = 0;prog(i,dwi_nii_files.size());++i)
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
            if(!handle_bids_folder(dwi_list,output_dir,overwrite,topup_eddy,error_msg))
            {
                tipl::error() << error_msg;
                return false;
            }
        }
        else
        {
            auto src_name = dwi_nii_files[i] + ".sz";
            if(!output_dir.empty())
                src_name = output_dir + "/" + std::filesystem::path(dwi_nii_files[i]).stem().stem().u8string() + ".sz";
            if(!overwrite && std::filesystem::exists(src_name))
                tipl::out() << "skipping " << src_name << " already exists";
            else
            {
                src_data src;
                if(!src.load_from_file(std::vector<std::string>({dwi_nii_files[i]}),true) ||
                   !src.save_to_file(src_name))
                {
                    tipl::error() << src.error_msg;
                    return false;
                }
            }
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
                tipl::out() << "could not find bids format files, try searching NIFTI files in " << source;
                if(!tipl::search_filesystem<tipl::out>((std::filesystem::path(source)/"*.nii.gz").string(),dwi_nii_files) &&
                   !tipl::search_filesystem<tipl::out>((std::filesystem::path(source)/"*.nii").string(),dwi_nii_files))
                    tipl::out() << "cannot find NIFTI files in " << source;
            }

            if(!dwi_nii_files.empty())
            {
                std::string output_dir;
                if(po.has("output"))
                {
                    output_dir = po.get("output");
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
                }
                else
                    tipl::out() << "no --output specified. write src files to the same directory of the nifti images";
                std::sort(dwi_nii_files.begin(),dwi_nii_files.end());
                if(nii2src(dwi_nii_files,output_dir,is_bids,po.get("overwrite",0),po.get("topup_eddy",0)))
                    return 0;
                return 1;
            }
        }

        tipl::out() << "cannot find NIFTI files...try looking for DICOM files in directory " << source.c_str() << std::endl;
        dicom2src_and_nii(source,po.get("overwrite",0));
        return 0;
    }
    else
        po.get_files("source",file_list);

    if(po.has("other_source"))
    {
        if(std::filesystem::is_directory(source))
        {
            tipl::out() << "try searching NIFTI files in " << source;
            if(!tipl::search_filesystem<tipl::out>((std::filesystem::path(source)/"*.nii.gz").string(),file_list) &&
               !tipl::search_filesystem<tipl::out>((std::filesystem::path(source)/"*.nii").string(),file_list))
                tipl::warning() << "cannot find NIFTI files in " << source;
        }
        else
            po.get_files("other_source",file_list);
    }

    if(file_list.empty())
    {
        tipl::error() << "no file found for creating src" << std::endl;
        return 1;
    }


    auto output = po.get("output",std::filesystem::path(file_list[0]).stem().stem().u8string() + ".sz");
    if(std::filesystem::is_directory(output))
        output += std::string("/") + std::filesystem::path(file_list[0]).stem().stem().u8string() + ".sz";
    if(!tipl::ends_with(output,".sz") && !tipl::ends_with(output,".rz"))
        output += ".sz";
    if(!po.get("overwrite",0) && std::filesystem::exists(output))
    {
        tipl::out() << "skipping " << output << " already exists";
        return 0;
    }


    src_data src;
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    std::sort(file_list.begin(),file_list.end());
    if(!parse_dwi(file_list,dwi_files,src.error_msg))
    {
        tipl::error() << src.error_msg;
        return 1;
    }

    if(po.has("b_table"))
    {
        std::string table_file_name = po.get("b_table");
        std::ifstream in(table_file_name);
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
        if(!get_bval_bvec(po.get("bval"),po.get("bvec"),dwi_files.size(),bval,bvec,src.error_msg))
        {
            tipl::error() << src.error_msg;
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
        tipl::error() << "no DWI data. abort." << std::endl;
        return 1;
    }

    {
        for(unsigned int index = 0;index < dwi_files.size();++index)
            if(dwi_files[index]->bvalue < 100.0f)
                dwi_files[index]->bvalue = 0.0f;
    }

    if(!src.load_from_file(dwi_files,po.get<int>("sort_b_table",0)) ||
       (po.has("intro") && !src.load_intro(po.get("intro"))) ||
       !src.save_to_file(output))
    {
        tipl::error() << src.error_msg;
        return 1;
    }

    return 0;


}
