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
bool parse_dwi(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
void dicom2src_and_nii(std::string dir_);
bool nii2src_bids(QString dir,QString output_dir,std::string& error_msg);
int src(tipl::program_option<tipl::out>& po)
{
    std::string source = po.get("source");
    QStringList file_list;
    if(std::filesystem::is_directory(source))
    {
        std::string error_msg;
        if(nii2src_bids(source.c_str(),po.get("output",source).c_str(),error_msg))
            return 0;
        tipl::out() << "load files in directory " << source.c_str() << std::endl;
        if(po.get("recursive",0))
        {
            tipl::out() << "search recursively in the subdir" << std::endl;
            dicom2src_and_nii(source);
            return 0;
        }
        else
        {
            QDir directory = QString(source.c_str());
            file_list = directory.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
            if(file_list.empty())
                file_list = directory.entryList(QStringList("*.fdf"),QDir::Files|QDir::NoSymLinks);
            for (int index = 0;index < file_list.size();++index)
                file_list[index] = QString(source.c_str()) + "/" + file_list[index];
        }
        tipl::out() << "a total of " << file_list.size() << " files found in the directory" << std::endl;
    }
    else
        file_list << source.c_str();

    if(po.has("other_source"))
        file_list << QString(po.get("other_source").c_str()).split(',');

    if(file_list.empty())
    {
        tipl::out() << "no file found for creating src" << std::endl;
        return 1;
    }

    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    if(!parse_dwi(file_list,dwi_files))
    {
        tipl::out() << "ERROR: cannot read dwi file: " << src_error_msg << std::endl;
        return 1;
    }
    if(po.has("b_table"))
    {
        std::string table_file_name = po.get("b_table");
        std::ifstream in(table_file_name.c_str());
        if(!in)
        {
            tipl::out() << "failed to open b-table" <<std::endl;
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
            tipl::out() << "mismatch between b-table and the loaded images" << std::endl;
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
        QStringList bval_files = QString(po.get("bval").c_str()).split(',');
        QStringList bvec_files = QString(po.get("bvec").c_str()).split(',');

        for(auto path : bval_files)
            if(!load_bval(path.toStdString().c_str(),bval))
            {
                tipl::out() << "cannot find bval at " << path.toStdString() << std::endl;
                return 1;
            }
        for(auto path : bvec_files)
            if(!load_bvec(path.toStdString().c_str(),bvec,po.get("flip_by",1)))
            {
                tipl::out() << "cannot find bvec at " << path.toStdString() << std::endl;
                return 1;
            }

        if(bval.size() != dwi_files.size())
        {
            tipl::out() << "mismatch between bval file and the loaded images" << std::endl;
            tipl::out() << "dwi number: " << dwi_files.size() << std::endl;
            tipl::out() << "bval number: " << bval.size() << std::endl;
            return 1;
        }
        if(bvec.size() != dwi_files.size()*3)
        {
            tipl::out() << "mismatch between bvec file and the loaded images" << std::endl;
            tipl::out() << "dwi number: " << dwi_files.size() << std::endl;
            tipl::out() << "bvec number: " << bvec.size() << std::endl;
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
        tipl::out() << "no file readed. Abort." << std::endl;
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
        tipl::out() << "cannot find b-table from the header. You may need to load an external b-table using--b_table or --bval and --bvec." << std::endl;
        return 1;
    }

    std::string output;
    if(po.has("output"))
    {
        output = po.get("output");
        if(QFileInfo(output.c_str()).isDir())
            output += std::string("/") + QFileInfo(source.c_str()).baseName().toStdString() + ".src.gz";
        else
        if(output.find(".src.gz") == std::string::npos)
            output += ".src.gz";
    }
    else
    {
        if(QFileInfo(source.c_str()).isDir())
            output = source + ".src.gz";
        else
            output = file_list.front().toStdString() + ".src.gz";
    }
    tipl::out() << "output src to " << output << std::endl;
    if(!DwiHeader::output_src(output.c_str(),dwi_files,
                          po.get<int>("up_sampling",0),
                          po.get<int>("sort_b_table",0)))
    {
        tipl::out() << "ERROR: " << src_error_msg << std::endl;
        return 1;
    }
    return 0;
}
