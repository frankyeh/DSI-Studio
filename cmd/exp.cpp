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
            std::cout << "ERROR: " << handle.error_msg << std::endl;
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

    if(QString(file_name.c_str()).endsWith(".fib.gz"))
    {
        std::shared_ptr<fib_data> handle;
        handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
        {
            std::cout << "ERROR: " << handle->error_msg << std::endl;
            return 0;
        }

        std::istringstream in(po.get("export"));
        std::string cmd;
        while(std::getline(in,cmd,','))
        {
            std::string file_name_stat(file_name);
            file_name_stat += ".";
            file_name_stat += cmd;
            file_name_stat += ".nii.gz";
            tipl::value_to_color<float> v2c;// used if "color" is wanted
            v2c.set_range(handle->view_item[0].contrast_min,handle->view_item[0].contrast_max);
            v2c.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
            if(!handle->save_mapping(cmd,file_name_stat,v2c))
            {
                std::cout << "ERROR: cannot find "<< cmd.c_str() <<" in " << file_name.c_str() <<std::endl;
                return 1;
            }
            std::cout << cmd << " saved to " << file_name_stat << std::endl;
        }
        return 0;
    }

    std::cout << "unsupported file format" << std::endl;
    return 1;
}
