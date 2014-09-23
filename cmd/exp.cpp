#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/dsi/image_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"

namespace po = boost::program_options;

// test example

int exp(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "exp: export information")
    ("source", po::value<std::string>(), "assign the .fib.gz or .src.gz file name")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=fa0,fa1,gfa)")
    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).run(), vm);
    po::notify(vm);


    if(vm["export"].as<std::string>() == "4dnii")
    {
        std::string file_name = vm["source"].as<std::string>();
        ImageModel handle;
        if(!handle.load_from_file(file_name.c_str()))
        {
            std::cout << handle.error_msg << std::endl;
            return 1;
        }
        std::cout << "exporting " << vm["export"].as<std::string>()+".nii.gz" << std::endl;
        handle.save_to_nii((file_name+".nii.gz").c_str());
        std::cout << "exporting " << vm["export"].as<std::string>()+".b_table.txt" << std::endl;
        handle.save_b_table((file_name+".b_table.txt").c_str());
        std::cout << "exporting " << vm["export"].as<std::string>()+".bvec" << std::endl;
        handle.save_bvec((file_name+".bvec").c_str());
        std::cout << "exporting " << vm["export"].as<std::string>()+".bval" << std::endl;
        handle.save_bval((file_name+".bval").c_str());
        return 1;
    }

    gz_mat_read mat_reader;
    std::string file_name = vm["source"].as<std::string>();
    std::cout << "loading " << file_name << "..." <<std::endl;
    if(!QFileInfo(file_name.c_str()).exists())
    {
        std::cout << file_name << " does not exist. terminating..." << std::endl;
        return 0;
    }
    if (!mat_reader.load_from_file(file_name.c_str()))
    {
        std::cout << "Invalid file format" << std::endl;
        return 0;
    }

    unsigned int col,row;
    const unsigned short* dim_buf = 0;
    if(!mat_reader.read("dimension",row,col,dim_buf))
    {
        std::cout << "Cannot find dimension matrix in the file" << file_name.c_str() <<std::endl;
        return 0;
    }
    const float* vs = 0;
    if(!mat_reader.read("voxel_size",row,col,vs))
    {
        std::cout << "Cannot find voxel_size matrix in the file" << file_name.c_str() <<std::endl;
        return 0;
    }
    const float* trans = 0;
    if(mat_reader.read("trans",row,col,trans))
        std::cout << "Transformation matrix found." << std::endl;

    image::geometry<3> geo(dim_buf[0],dim_buf[1],dim_buf[2]);
    std::string export_option = vm["export"].as<std::string>();
    std::replace(export_option.begin(),export_option.end(),',',' ');
    std::istringstream in(export_option);
    std::string cmd;
    while(in >> cmd)
    {
        std::string file_name_stat(file_name);
        file_name_stat += ".";
        file_name_stat += cmd;
        file_name_stat += ".nii.gz";

        // export fiber orientations
        if(cmd.length() > 3 && cmd.substr(0,3) == std::string("dir"))
        {
            FiberDirection fib;
            if(!fib.add_data(mat_reader))
            {
                std::cout << "Invalid file for exporting fiber orientations." << std::endl;
                continue;
            }

            image::basic_image<float,4> fibers;
            if(cmd[3] == 's') // all directions
            {
                fibers.resize(image::geometry<4>(geo[0],geo[1],geo[2],fib.num_fiber*3));
                for(unsigned int i = 0,ptr = 0;i < fib.num_fiber;++i)
                for(unsigned int j = 0;j < 3;++j)
                for(unsigned int index = 0;index < geo.size();++index,++ptr)
                    if(fib.getFA(index,i))
                        fibers[ptr] = fib.getDir(index,i)[j];
            }
            else
            {
                unsigned char dir_index = cmd[3] - '0';
                if(dir_index < 0 || dir_index >= fib.num_fiber)
                {
                    std::cout << "Invalid fiber index. The maximum fiber per voxel is " << (int) fib.num_fiber << std::endl;
                    continue;
                }
                fibers.resize(image::geometry<4>(geo[0],geo[1],geo[2],3));
                for(unsigned int j = 0,ptr = 0;j < 3;++j)
                for(unsigned int index = 0;index < geo.size();++index,++ptr)
                    if(fib.getFA(index,dir_index))
                        fibers[ptr] = fib.getDir(index,dir_index)[j];
            }
            gz_nifti nifti_header;
            if(trans) //QSDR condition
            {
                nifti_header.set_image_transformation(trans);
                std::cout << "Output transformation matrix" << std::endl;
            }
            else
                image::flip_xy(fibers);
            nifti_header << fibers;
            nifti_header.set_voxel_size(vs);
            nifti_header.save_to_file(file_name_stat.c_str());
            std::cout << "write to file " << file_name_stat << std::endl;
            continue;
        }
        const float* volume = 0;
        if(mat_reader.read(cmd.c_str(),row,col,volume))
        {
            std::cout << "retriving matrix " << cmd << std::endl;
            if(row*col != geo.size())
            {
                std::cout << "The matrix "<< cmd.c_str() <<" is not an image volume" <<std::endl;
                std::cout << "matrix size: " << row << " by " << col << std::endl;
                std::cout << "expected dimension: " << geo[0] << " by " << geo[1] << " by " << geo[2] << std::endl;
                continue;
            }
            image::basic_image<float,3> data(geo);
            std::copy(volume,volume+geo.size(),data.begin());
            gz_nifti nifti_header;

            if(trans) //QSDR condition
            {
                nifti_header.set_image_transformation(trans);
                std::cout << "Output transformation matrix" << std::endl;
            }
            else
                image::flip_xy(data);
            nifti_header << data;
            nifti_header.set_voxel_size(vs);
            nifti_header.save_to_file(file_name_stat.c_str());
            std::cout << "write to file " << file_name_stat << std::endl;
            continue;
        }
        std::cout << "Cannot find matrix "<< cmd.c_str() <<" in the file" << file_name.c_str() <<std::endl;
        continue;

    }
}
