#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "dicom/dwi_header.hpp"

namespace po = boost::program_options;

bool load_4d_nii(const char* file_name,boost::ptr_vector<DwiHeader>& dwi_files);
bool load_dicom_multi_frame(const char* file_name,boost::ptr_vector<DwiHeader>& dwi_files);
bool load_4d_2dseq(const char* file_name,boost::ptr_vector<DwiHeader>& dwi_files);
bool load_multiple_slice_dicom(QStringList file_list,boost::ptr_vector<DwiHeader>& dwi_files);
int src(int ac, char *av[])
{
    po::options_description rec_desc("dicom parsing options");
    rec_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "src:dicom parsing")
    ("source", po::value<std::string>(), "assign the directory for the dicom files")
    ("b_table", po::value<std::string>(), "assign the b-table")
    ("output", po::value<std::string>(), "assign the output filename")
    ;
    std::ofstream out("log.txt");
    if(!ac)
    {
        std::cout << rec_desc << std::endl;
        return 1;
    }
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(rec_desc).allow_unregistered().run(), vm);
    po::notify(vm);


    std::string source = vm["source"].as<std::string>();
    std::string ext;
    if(source.size() > 4)
        ext = std::string(source.end()-4,source.end());

    boost::ptr_vector<DwiHeader> dwi_files;
    if(ext ==".nii")
        // load nii file
    {
        if(!load_4d_nii(source.c_str(),dwi_files))
        {
            out << "Invalid file format" << std::endl;
            return -1;
        }
    }
    else
        if(ext == ".dcm")
        {
            if(!load_dicom_multi_frame(source.c_str(),dwi_files))
            {
                out << "Invalid file format" << std::endl;
                return -1;
            }
        }
    else
        if(ext == "dseq")
        {
            if(!load_4d_2dseq(source.c_str(),dwi_files))
            {
                out << "Invalid file format" << std::endl;
                return -1;
            }
        }
    else
        // load directory
    {
        QDir directory = QString(source.c_str());
        QStringList file_list = directory.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        out << "A total of " << file_list.size() <<" files found" << std::endl;
        if(!load_multiple_slice_dicom(file_list,dwi_files))
        {
            for (unsigned int index = 0;index < file_list.size();++index)
            {
                std::string file_name = source;
                file_name += "/";
                file_name += file_list[index].toLocal8Bit().begin();
                out << "Reading " << file_list[index].toLocal8Bit().begin() << std::endl;
                std::auto_ptr<DwiHeader> new_file(new DwiHeader);
                if (!new_file->open(file_name.c_str()))
                {
                    out << "Failed" << std::endl;
                    continue;
                }
                new_file->file_name = file_list[index].toLocal8Bit().begin();
                dwi_files.push_back(new_file.release());
            }
        }
    }

    if(vm.count("b_table"))
    {
        std::string table_file_name = vm["b_table"].as<std::string>();
        std::ifstream in(table_file_name.c_str());
        if(!in)
        {
            out << "Failed to open b-table" <<std::endl;
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
            out << "Mismatch between b-table and the loaded image" << std::endl;
            return -1;
        }
        for(unsigned int index = 0,b_index = 0;index < dwi_files.size();++index,b_index += 4)
        {
            dwi_files[index].set_bvalue(b_table[b_index]);
            dwi_files[index].set_bvec(b_table[b_index+1],b_table[b_index+2],b_table[b_index+3]);
        }
        out << "B-table " << table_file_name << " loaded" << std::endl;
    }

    if(dwi_files.empty())
    {
        out << "No file readed. Abort." << std::endl;
        return 1;
    }
    out << "Output src" << std::endl;
    DwiHeader::output_src(vm["output"].as<std::string>().c_str(),dwi_files,false,false);
    return 0;
}
