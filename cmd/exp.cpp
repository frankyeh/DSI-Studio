#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "tracking_static_link.h"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "libs/gzip_interface.hpp"

namespace po = boost::program_options;

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

int exp(int ac, char *av[],std::ostream& out)
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
        out << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).allow_unregistered().run(), vm);
    po::notify(vm);

    MatFile mat_reader;
    std::string file_name = vm["source"].as<std::string>();
    out << "loading " << file_name.c_str() << "..." <<std::endl;
    if(!QFileInfo(file_name.c_str()).exists())
    {
        out << file_name.c_str() << " does not exist. terminating..." << std::endl;
        return 0;
    }
    if (!mat_reader.load_from_file(file_name.c_str()))
    {
        out << "Invalid file format" << std::endl;
        return 0;
    }

    unsigned int col,row;
    const unsigned short* dim_buf = 0;
    if(!mat_reader.get_matrix("dimension",row,col,dim_buf))
    {
        out << "Cannot find dimension matrix in the file" << file_name.c_str() <<std::endl;
        return 0;
    }
    const float* vs = 0;
    if(!mat_reader.get_matrix("voxel_size",row,col,vs))
    {
        out << "Cannot find voxel_size matrix in the file" << file_name.c_str() <<std::endl;
        return 0;
    }

    image::geometry<3> geometry(dim_buf[0],dim_buf[1],dim_buf[2]);
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
        gz_nifti nifti_header;
        const float* volume = 0;
        out << "retriving matrix " << cmd.c_str() << std::endl;
        if(!mat_reader.get_matrix(cmd.c_str(),row,col,volume))
        {
            out << "Cannot find matrix "<< cmd.c_str() <<" in the file" << file_name.c_str() <<std::endl;
            continue;
        }
        if(row*col != geometry.size())
        {
            out << "The matrix "<< cmd.c_str() <<" is not an image volume" <<std::endl;
            out << "matrix size: " << row << " by " << col << std::endl;
            out << "expected dimension: " << geometry[0] << " by " << geometry[1] << " by " << geometry[2] << std::endl;
            continue;
        }
        image::basic_image<float,3> data(geometry);
        std::copy(volume,volume+geometry.size(),data.begin());
        image::flip_xy(data);
        nifti_header << data;
        nifti_header.set_voxel_size(vs);
        nifti_header.save_to_file(file_name_stat.c_str());
        out << "write to file " << file_name_stat.c_str() << std::endl;
    }
}
