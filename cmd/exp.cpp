#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "tipl/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/dsi/image_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "program_option.hpp"

std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);
bool trk2tt(const char* trk_file,const char* tt_file);
bool tt2trk(const char* tt_file,const char* trk_file);
int exp(void)
{
    std::string file_name = po.get("source");
    if(QString(file_name.c_str()).endsWith(".trk.gz"))
    {
        std::string output_name = po.get("output");
        if(QString(output_name.c_str()).endsWith(".tt.gz"))
        {
            if(trk2tt(file_name.c_str(),output_name.c_str()))
            {
                std::cout << "file converted." << std::endl;
                return 0;
            }
            else
            {
                std::cout << "Cannot write to file:" << output_name << std::endl;
                return 1;
            }
        }
        std::cout << "unsupported file format" << std::endl;
        return 1;
    }
    if(QString(file_name.c_str()).endsWith(".tt.gz"))
    {
        std::string output_name = po.get("output");
        if(QString(output_name.c_str()).endsWith(".trk.gz"))
        {
            if(tt2trk(file_name.c_str(),output_name.c_str()))
            {
                std::cout << "file converted." << std::endl;
                return 0;
            }
            else
            {
                std::cout << "Cannot write to file:" << output_name << std::endl;
                return 1;
            }
        }
        std::cout << "unsupported file format" << std::endl;
        return 1;
    }
    std::string export_name = po.get("export");
    if(export_name == "4dnii")
    {
        ImageModel handle;
        if(!handle.load_from_file(file_name.c_str()))
        {
            std::cout << handle.error_msg << std::endl;
            return 1;
        }
        std::cout << "exporting " << file_name << ".nii.gz" << std::endl;
        handle.save_to_nii((file_name+".nii.gz").c_str());
        std::cout << "exporting " << file_name << ".b_table.txt" << std::endl;
        handle.save_b_table((file_name+".b_table.txt").c_str());
        std::cout << "exporting " << file_name << ".bvec" << std::endl;
        handle.save_bvec((file_name+".bvec").c_str());
        std::cout << "exporting " << file_name << ".bval" << std::endl;
        handle.save_bval((file_name+".bval").c_str());
        return 1;
    }

    gz_mat_read mat_reader;
    std::cout << "loading " << file_name << "..." <<std::endl;
    if(!QFileInfo(file_name.c_str()).exists())
    {
        std::cout << file_name << " does not exist. terminating..." << std::endl;
        return 1;
    }
    if (!mat_reader.load_from_file(file_name.c_str()))
    {
        std::cout << "invalid file format" << std::endl;
        return 1;
    }

    unsigned int col,row;
    tipl::geometry<3> geo;
    if(!mat_reader.read("dimension",geo))
    {
        std::cout << "cannot find dimension matrix in the file" << file_name.c_str() <<std::endl;
        return 1;
    }
    tipl::vector<3> vs;
    if(!mat_reader.read("voxel_size",vs))
    {
        std::cout << "cannot find voxel_size matrix in the file" << file_name.c_str() <<std::endl;
        return 1;
    }
    tipl::matrix<4,4,float> trans;
    {
        if(mat_reader.read("trans",trans))
            std::cout << "transformation matrix found." << std::endl;
        else
        {
            std::fill(trans.begin(),trans.end(),0.0f);
            trans[0] = vs[0];
            trans[5] = vs[1];
            trans[10] = vs[2];
            trans[15] = 1.0f;
        }
    }
    std::shared_ptr<fib_data> handle;
    std::string export_option = po.get("export");
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
            fiber_directions dir;
            if(!dir.add_data(mat_reader))
            {
                std::cout << "invalid file for exporting fiber orientations." << std::endl;
                continue;
            }

            tipl::image<float,4> fibers;
            if(cmd[3] == 's') // all directions
            {
                fibers.resize(tipl::geometry<4>(geo[0],geo[1],geo[2],dir.num_fiber*3));
                for(unsigned int i = 0,ptr = 0;i < dir.num_fiber;++i)
                for(unsigned int j = 0;j < 3;++j)
                for(unsigned int index = 0;index < geo.size();++index,++ptr)
                    if(dir.get_fa(index,i))
                        fibers[ptr] = dir.get_dir(index,i)[j];
            }
            else
            {
                unsigned char dir_index = cmd[3] - '0';
                if(dir_index < 0 || dir_index >= dir.num_fiber)
                {
                    std::cout << "invalid fiber index. The maximum fiber per voxel is " << (int) dir.num_fiber << std::endl;
                    continue;
                }
                fibers.resize(tipl::geometry<4>(geo[0],geo[1],geo[2],3));
                for(unsigned int j = 0,ptr = 0;j < 3;++j)
                for(unsigned int index = 0;index < geo.size();++index,++ptr)
                    if(dir.get_fa(index,dir_index))
                        fibers[ptr] = dir.get_dir(index,dir_index)[j];
            }
            std::cout << "write to file " << file_name_stat << std::endl;
            if(!gz_nifti::save_to_file(file_name_stat.c_str(),fibers,tipl::vector<3>(vs),trans))
                std::cout << "cannot write output to file:" << file_name_stat << std::endl;
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
            tipl::image<float,3> data(geo);
            std::copy(volume,volume+geo.size(),data.begin());
            std::cout << "write to file " << file_name_stat << std::endl;
            if(!gz_nifti::save_to_file(file_name_stat.c_str(),data,tipl::vector<3>(vs),trans))
                std::cout << "cannot write output to file:" << file_name_stat << std::endl;
            continue;
        }

        if(!handle.get())
            handle = cmd_load_fib(po.get("source"));
        if(handle.get())
        {
            tipl::value_to_color<float> v2c;// used if "color" is wanted
            v2c.set_range(handle->view_item[0].contrast_min,handle->view_item[0].contrast_max);
            v2c.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
            if(handle->save_mapping(cmd,file_name_stat,v2c))
            {
                std::cout << cmd << " saved to " << file_name_stat << std::endl;
                continue;
            }
        }
        std::cout << "cannot find matrix "<< cmd.c_str() <<" in the file" << file_name.c_str() <<std::endl;
        continue;

    }
    return 0;
}
