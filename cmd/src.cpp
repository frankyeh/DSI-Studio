#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include "tipl/tipl.hpp"
#include "dicom/dwi_header.hpp"
#include "program_option.hpp"
extern std::string src_error_msg;
QStringList search_files(QString dir,QString filter);
bool load_bval(const char* file_name,std::vector<double>& bval);
bool load_bvec(const char* file_name,std::vector<double>& b_table,bool flip_by = true);
bool parse_dwi(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
int src(program_option& po)
{
    std::string source = po.get("source");
    std::string ext;
    if(source.size() > 4)
        ext = std::string(source.end()-4,source.end());

    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    QStringList file_list;
    if(ext ==".nii" || ext == ".dcm" || ext == "dseq" || ext == "i.gz")
        file_list << source.c_str();
    else
    {

        std::cout << "load files in directory " << source.c_str() << std::endl;
        QDir directory = QString(source.c_str());
        if(po.has("recursive"))
        {
            std::cout << "search recursively in the subdir" << std::endl;
            file_list = search_files(source.c_str(),"*.dcm");
        }
        else
        {
            file_list = directory.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
            if(file_list.empty())
                file_list = directory.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
            if(file_list.empty())
                file_list = directory.entryList(QStringList("*.fdf"),QDir::Files|QDir::NoSymLinks);
            for (int index = 0;index < file_list.size();++index)
                file_list[index] = QString(source.c_str()) + "/" + file_list[index];
        }
        std::cout << "a total of " << file_list.size() <<" files found in the directory" << std::endl;
    }

    if(file_list.empty())
    {
        std::cout << "no file found for creating src" << std::endl;
        return 1;
    }

    if(!parse_dwi(file_list,dwi_files))
    {
        std::cout << "ERROR loading dwi file:" << src_error_msg << std::endl;
        return 1;
    }
    if(po.has("b_table"))
    {
        std::string table_file_name = po.get("b_table");
        std::ifstream in(table_file_name.c_str());
        if(!in)
        {
            std::cout << "failed to open b-table" <<std::endl;
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
            std::cout << "mismatch between b-table and the loaded images" << std::endl;
            return 1;
        }
        for(unsigned int index = 0,b_index = 0;index < dwi_files.size();++index,b_index += 4)
        {
            dwi_files[index]->bvalue = float(b_table[b_index]);
            dwi_files[index]->bvec = tipl::vector<3>(b_table[b_index+1],b_table[b_index+2],b_table[b_index+3]);
        }
        std::cout << "b-table " << table_file_name << " loaded" << std::endl;
    }
    if(po.has("bval") && po.has("bvec"))
    {
        std::vector<double> bval,bvec;
        if(!load_bval(po.get("bval").c_str(),bval))
        {
            std::cout << "cannot find bval at " << po.get("bval") << std::endl;
            return 1;
        }
        if(!load_bvec(po.get("bvec").c_str(),bvec,po.get("flip_by",1)))
        {
            std::cout << "cannot find bvec at " << po.get("bvec") << std::endl;
            return 1;
        }
        if(bval.size() != dwi_files.size())
        {
            std::cout << "mismatch between bval file and the loaded images" << std::endl;
            return 1;
        }
        if(bvec.size() != dwi_files.size()*3)
        {
            std::cout << "mismatch between bvec file and the loaded images" << std::endl;
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
        std::cout << "no file readed. Abort." << std::endl;
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
        std::cout << "cannot find b-table from the header. You may need to load an external b-table using--b_table or --bval and --bvec." << std::endl;
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
    std::cout << "output src to " << output << std::endl;
    if(!DwiHeader::output_src(output.c_str(),dwi_files,
                          po.get<int>("up_sampling",0),
                          po.get<int>("sort_b_table",0)))
    {
        std::cout << "ERROR: " << src_error_msg << std::endl;
        return 1;
    }
    return 0;
}
