#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "dicom/dwi_header.hpp"
#include "program_option.hpp"

QStringList search_files(QString dir,QString filter);
void load_bval(const char* file_name,std::vector<double>& bval);
void load_bvec(const char* file_name,std::vector<double>& b_table);
bool load_all_files(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
int src(void)
{
    std::string source = po.get("source");
    std::string ext;
    if(source.size() > 4)
        ext = std::string(source.end()-4,source.end());

    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    QStringList file_list;
    if(ext ==".nii" || ext == ".dcm" || ext == "dseq" || ext == "i.gz")
    {
        std::cout << "image=" << source.c_str() << std::endl;
        file_list << source.c_str();
    }
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
            for (unsigned int index = 0;index < file_list.size();++index)
                file_list[index] = QString(source.c_str()) + "/" + file_list[index];
        }
        std::cout << "A total of " << file_list.size() <<" files found in the directory" << std::endl;
    }

    if(file_list.empty())
    {
        std::cout << "No file found for creating src" << std::endl;
        return -1;
    }

    if(!load_all_files(file_list,dwi_files))
    {
        std::cout << "Invalid file format" << std::endl;
        return -1;
    }
    if(po.has("b_table"))
    {
        std::string table_file_name = po.get("b_table");
        std::ifstream in(table_file_name.c_str());
        if(!in)
        {
            std::cout << "Failed to open b-table" <<std::endl;
            return -1;
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
            std::cout << "Mismatch between b-table and the loaded images" << std::endl;
            return -1;
        }
        for(unsigned int index = 0,b_index = 0;index < dwi_files.size();++index,b_index += 4)
        {
            dwi_files[index]->set_bvalue(b_table[b_index]);
            dwi_files[index]->set_bvec(b_table[b_index+1],b_table[b_index+2],b_table[b_index+3]);
        }
        std::cout << "B-table " << table_file_name << " loaded" << std::endl;
    }
    if(po.has("bval") && po.has("bvec"))
    {
        std::vector<double> bval,bvec;
        std::cout << "load bval=" << po.get("bval") << std::endl;
        std::cout << "load bvec=" << po.get("bvec") << std::endl;
        load_bval(po.get("bval").c_str(),bval);
        load_bvec(po.get("bvec").c_str(),bvec);
        if(bval.size() != dwi_files.size())
        {
            std::cout << "Mismatch between bval file and the loaded images" << std::endl;
            return -1;
        }
        if(bvec.size() != dwi_files.size()*3)
        {
            std::cout << "Mismatch between bvec file and the loaded images" << std::endl;
            return -1;
        }
        for(unsigned int index = 0;index < dwi_files.size();++index)
        {
            dwi_files[index]->set_bvalue(bval[index]);
            dwi_files[index]->set_bvec(bvec[index*3],bvec[index*3+1],bvec[index*3+2]);
        }
    }
    if(dwi_files.empty())
    {
        std::cout << "No file readed. Abort." << std::endl;
        return 1;
    }

    double max_b = 0;
    for(unsigned int index = 0;index < dwi_files.size();++index)
    {
        if(dwi_files[index]->get_bvalue() < 100)
            dwi_files[index]->set_bvalue(0);
        max_b = std::max(max_b,(double)dwi_files[index]->get_bvalue());
    }
    if(max_b == 0.0)
    {
        std::cout << "Cannot find b-table from the header. You may need to load an external b-table using--b_table or --bval and --bvec." << std::endl;
        return 1;
    }
    std::cout << "Output src " << po.get("output").c_str() << std::endl;
    DwiHeader::output_src(po.get("output").c_str(),dwi_files,0);
    return 0;
}
