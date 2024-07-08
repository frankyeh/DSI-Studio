#include <QDir>
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
    if(tipl::max_value(bvals) == 0.0)
    {
        error_msg = "only have b0 image(s)";
        return false;
    }
    bvals_out.swap(bvals);
    bvecs_out.swap(bvecs);
    return true;
}
void create_src(const std::vector<std::string>& nii_names,std::string src_name)
{
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    for(auto& nii_name : nii_names)
    {
        if(!load_4d_nii(nii_name.c_str(),dwi_files,true))
        {
            tipl::out() << "skipping " << nii_name << ": " << src_error_msg;
            return;
        }
    }
    if(!DwiHeader::output_src(src_name.c_str(),dwi_files,0,false))
        tipl::error() << src_name << " : " << src_error_msg;
}
void create_src(std::string nii_name,std::string src_name)
{
    std::vector<std::string> nii_names;
    nii_names.push_back(nii_name);
    create_src(nii_names,src_name);
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

void search_dwi_nii_bids(const std::string& dir,std::vector<std::string>& dwi_nii_files)
{
    tipl::progress prog((std::string("parsing BIDS directory: ") + dir).c_str());
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
}
bool nii2src(tipl::program_option<tipl::out>& po)
{
    auto source = po.get("source");
    auto output_dir = po.get("output",source);
    int overwrite = po.get("overwrite",0);
    tipl::max_thread_count = po.get("thread_count",tipl::max_thread_count);

    std::vector<std::string> dwi_nii_files;
    tipl::out() << "checking BIDS format";
    search_dwi_nii_bids(source,dwi_nii_files);
    if(dwi_nii_files.empty())
    {
        tipl::out() << "no BIDS found. searching for NIFTI files in the folder";
        search_dwi_nii(source,dwi_nii_files);
        if(dwi_nii_files.empty())
        {
            tipl::out() << "no nifti files found";
            return false;
        }
    }
    if(!std::filesystem::exists(output_dir) && !std::filesystem::create_directory(output_dir))
    {
        tipl::error() << "cannot create the output folder. please check write privileges";
        return false;
    }
    if(!std::filesystem::is_directory(output_dir))
    {
        tipl::error() << output_dir << " is not a valid output directory.";
        return false;
    }
    tipl::par_for(dwi_nii_files.size(),[&](unsigned int index)
    {
        auto src_name = output_dir + "/" + std::filesystem::path(dwi_nii_files[index]).filename().string() + ".src.gz";
        if(!overwrite && std::filesystem::exists(src_name))
        {
            tipl::out() << "skipping " << src_name << ": already exists";
            return;
        }
        create_src(dwi_nii_files[index],src_name);
    });
    return true;
}

int src(tipl::program_option<tipl::out>& po)
{      
    std::string source = po.get("source");
    std::vector<std::string> file_list;
    if(std::filesystem::is_directory(source))
    {
        if(nii2src(po))
            return 0;

        tipl::out() << "open files in directory " << source.c_str() << std::endl;
        if(po.get("recursive",1))
        {
            tipl::out() << "search recursively in the subdir" << std::endl;
            dicom2src_and_nii(source);
            return 0;
        }
        else
        {
            tipl::search_files(source,"*.dcm",file_list);
            if(file_list.empty())
                tipl::search_files(source,"*.fdf",file_list);
            tipl::out() << "a total of " << file_list.size() << " files found in the directory" << std::endl;
        }
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

    std::string output = file_list.front() + ".src.gz";
    if(po.has("output"))
    {
        output = po.get("output");
        if(std::filesystem::is_directory(output))
            output += std::string("/") + std::filesystem::path(file_list[0]).filename().string() + ".src.gz";
        else
            if(output.find(".src.gz") == std::string::npos)
                output += ".src.gz";
    }
    if(!po.get("overwrite",0) && std::filesystem::exists(output))
    {
        tipl::out() << "skipping " << output << ": already exists";
        return 0;
    }
    tipl::out() << "output src to " << output << std::endl;
    if(!DwiHeader::output_src(output.c_str(),dwi_files,
                          po.get<int>("up_sampling",0),
                          po.get<int>("sort_b_table",0)))
    {
        tipl::error() << src_error_msg << std::endl;
        return 1;
    }
    return 0;
}
