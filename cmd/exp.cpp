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
bool trk2tt(const std::string& trk_file,const char* tt_file);
bool tt2trk(const std::string& tt_file,const char* trk_file);
int exp(tipl::program_option<tipl::out>& po)
{
    std::string file_name = po.get("source");
    if(tipl::ends_with(file_name,{".trk.gz",".tt.gz"}))
    {
        auto output_name = po.get("output");
        bool from_trk = tipl::ends_with(file_name,".trk.gz");
        if(!tipl::ends_with(output_name,from_trk ? ".tt.gz" : ".trk.gz"))
            return tipl::error() << "unsupported file format",1;
        if(!(from_trk ? trk2tt(file_name,output_name.c_str()) :
                  tt2trk(file_name,output_name.c_str())))
            return tipl::error() << "cannot write to " << output_name,1;
        return tipl::out() << "file converted.",0;
    }
    if(tipl::ends_with(file_name,{".fib.gz",".fz"}))
    {
        std::shared_ptr<fib_data> handle;
        handle = cmd_load_fib(po);
        if(!handle.get())
            return 1;
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
                tipl::error() << handle->error_msg << std::endl;
                return 1;
            }
            return 0;
        }

        for(const auto& each : tipl::split(po.get("export"),','))
            if(!handle->save_slice(each,file_name + "." + each + ".nii.gz",po.has("export_to_mni")))
            {
                tipl::error() << handle->error_msg;
                return 1;
            }
        return 0;
    }

    tipl::error() << "unsupported file format" << std::endl;
    return 1;
}
