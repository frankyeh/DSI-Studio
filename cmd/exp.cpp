#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/dsi/image_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"

std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);
bool trk2tt(const char* trk_file,const char* tt_file);
bool tt2trk(const char* tt_file,const char* trk_file);
int exp(tipl::program_option<tipl::out>& po)
{
    std::string file_name = po.get("source");
    if(QString(file_name.c_str()).endsWith(".trk.gz"))
    {
        std::string output_name = po.get("output");
        if(QString(output_name.c_str()).endsWith(".tt.gz"))
        {
            if(trk2tt(file_name.c_str(),output_name.c_str()))
            {
                tipl::out() << "file converted." << std::endl;
                return 0;
            }
            else
            {
                tipl::error() << "cannot write to " << output_name << std::endl;
                return 1;
            }
        }
        tipl::error() << "unsupported file format" << std::endl;
        return 1;
    }
    if(QString(file_name.c_str()).endsWith(".tt.gz"))
    {
        std::string output_name = po.get("output");
        if(QString(output_name.c_str()).endsWith(".trk.gz"))
        {
            if(tt2trk(file_name.c_str(),output_name.c_str()))
            {
                tipl::out() << "file converted." << std::endl;
                return 0;
            }
            else
            {
                tipl::error() << "cannot write to " << output_name << std::endl;
                return 1;
            }
        }
        tipl::error() << "unsupported file format" << std::endl;
        return 1;
    }
    if(QString(file_name.c_str()).endsWith(".fib.gz"))
    {
        std::shared_ptr<fib_data> handle;
        handle = cmd_load_fib(po);
        if(!handle.get())
        {
            tipl::error() << handle->error_msg << std::endl;
            return 1;
        }
        if(po.has("match"))
        {
            if(!handle->db.has_db())
            {
                tipl::error() << "the FIB file is not a connectometry database" << std::endl;
                return 1;
            }
            if(handle->db.demo.empty())
            {
                tipl::error() << "the connectometry database does not include demographics for matching." << std::endl;
                return 1;
            }
            if(!handle->db.save_demo_matched_image(po.get("match"),po.get("output",po.get("source")+".matched.nii.gz")))
            {
                tipl::error() << handle->db.error_msg << std::endl;
                return 1;
            }
            return 0;
        }

        std::istringstream in(po.get("export"));
        std::string cmd;
        while(std::getline(in,cmd,','))
        {
            if(!handle->save_mapping(cmd,file_name + "." + cmd + ".nii.gz"))
            {
                tipl::error() << "cannot find "<< cmd <<" in " << file_name.c_str() <<std::endl;
                return 1;
            }
            tipl::out() << cmd << ".nii.gz saved " << std::endl;
        }
        return 0;
    }

    tipl::error() << "unsupported file format" << std::endl;
    return 1;
}
