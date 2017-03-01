#include <QFileInfo>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "program_option.hpp"
#include "atlas.hpp"

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

extern std::vector<atlas> atlas_list;
bool atl_load_atlas(std::string atlas_name);
bool load_roi(std::shared_ptr<fib_data> handle,image::basic_image<image::vector<3>,3>& mapping,RoiMgr& roi_mgr);
void get_connectivity_matrix(std::shared_ptr<fib_data> handle,
                             TractModel& tract_model,
                             image::basic_image<image::vector<3>,3>& mapping);

void get_regions_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result);
void export_indices(std::shared_ptr<fib_data> handle,ROIRegion& region,const std::string& file_name)
{
    if(po.has("export") && po.get("export") == std::string("stat"))
    {
        std::string file_name_stat(file_name);
        std::replace(file_name_stat.begin(),file_name_stat.end(),':','_');
        std::replace(file_name_stat.begin(),file_name_stat.end(),',','_');
        file_name_stat += ".statistics.txt";
        std::cout << "export ROI statistics to file:" << file_name_stat << std::endl;
        std::ofstream out(file_name_stat.c_str());
        std::vector<std::string> titles;
        std::vector<float> data;
        region.get_quantitative_data(handle,titles,data);
        for(unsigned int i = 0;i < titles.size() && i < data.size();++i)
            out << titles[i] << "\t" << data[i] << std::endl;
    }
    else
        std::cout << "Please specify the export parameters" << std::endl;
}
void export_track_info(const std::string& file_name,
                       std::string export_option,
                       std::shared_ptr<fib_data> handle,
                       TractModel& tract_model)
{
    std::replace(export_option.begin(),export_option.end(),',',' ');
    std::istringstream in(export_option);
    std::string cmd;
    while(in >> cmd)
    {
        // track analysis report
        if(cmd.find("report") == 0)
        {
            std::cout << "export track analysis report..." << std::endl;
            std::replace(cmd.begin(),cmd.end(),':',' ');
            std::istringstream in(cmd);
            std::string report_tag,index_name;
            int profile_dir = 0,bandwidth = 0;
            in >> report_tag >> index_name >> profile_dir >> bandwidth;
            std::vector<float> values,data_profile;
            // check index
            if(index_name != "qa" && index_name != "fa" &&  handle->get_name_index(index_name) == handle->view_item.size())
            {
                std::cout << "cannot find index name:" << index_name << std::endl;
                continue;
            }
            if(bandwidth == 0)
            {
                std::cout << "please specify bandwidth value" << std::endl;
                continue;
            }
            if(profile_dir > 4)
            {
                std::cout << "please specify a valid profile type" << std::endl;
                continue;
            }
            std::cout << "calculating report" << std::endl;
            tract_model.get_report(
                                profile_dir,
                                bandwidth,
                                index_name,
                                values,data_profile);

            std::replace(cmd.begin(),cmd.end(),' ','.');
            std::string file_name_stat(file_name);
            file_name_stat += ".";
            file_name_stat += cmd;
            file_name_stat += ".txt";
            std::cout << "output report:" << file_name_stat << std::endl;
            std::ofstream report(file_name_stat.c_str());
            report << "position\t";
            std::copy(values.begin(),values.end(),std::ostream_iterator<float>(report,"\t"));
            report << std::endl;
            report << "value";
            std::copy(data_profile.begin(),data_profile.end(),std::ostream_iterator<float>(report,"\t"));
            report << std::endl;
            continue;
        }

        std::string file_name_stat(file_name);
        file_name_stat += ".";
        file_name_stat += cmd;
        // export statistics
        if(cmd == "tdi" || cmd == "tdi_end")
        {
            file_name_stat += ".nii.gz";
            std::cout << "export TDI to " << file_name_stat << std::endl;
            tract_model.save_tdi(file_name_stat.c_str(),false,cmd == "tdi_end",handle->trans_to_mni);
            continue;
        }
        if(cmd == "tdi2" || cmd == "tdi2_end")
        {
            file_name_stat += ".nii.gz";
            std::cout << "export subvoxel TDI to " << file_name_stat << std::endl;
            tract_model.save_tdi(file_name_stat.c_str(),true,cmd == "tdi2_end",handle->trans_to_mni);
            continue;
        }

        file_name_stat += ".txt";

        if(cmd == "stat")
        {
            std::cout << "export statistics..." << std::endl;
            std::ofstream out_stat(file_name_stat.c_str());
            std::string result;
            tract_model.get_quantitative_info(result);
            out_stat << result;
            continue;
        }

        if(handle->get_name_index(cmd) != handle->view_item.size())
            tract_model.save_data_to_file(file_name_stat.c_str(),cmd);
        else
        {
            std::cout << "invalid export option:" << cmd << std::endl;
            continue;
        }
    }
}
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text,
                 image::basic_image<image::vector<3>,3>& mapping);

int trk_post(std::shared_ptr<fib_data> handle,
             TractModel& tract_model,
             image::basic_image<image::vector<3>,3>& mapping,
             const std::string& file_name);
std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);
int ana(void)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    if(!handle.get())
        return 0;

    image::basic_image<image::vector<3>,3> mapping;

    if(po.has("atlas"))
    {
        std::cout << "export information from " << po.get("atlas") << std::endl;
        if(!atl_load_atlas(po.get("atlas")))
            return 0;
        for(unsigned int i = 0;i < atlas_list.size();++i)
        {
            std::vector<std::shared_ptr<ROIRegion> > regions;
            std::vector<std::string> region_list;
            for(unsigned int j = 0;j < atlas_list[i].get_list().size();++j)
            {
                std::shared_ptr<ROIRegion> region(std::make_shared<ROIRegion>(handle->dim,handle->vs));
                std::string region_name = atlas_list[i].name;
                region_name += ":";
                region_name += atlas_list[i].get_list()[j];
                if(!load_region(handle,*region.get(),region_name,mapping))
                {
                    std::cout << "Fail to load the ROI file:" << region_name << std::endl;
                    return 0;
                }
                region_list.push_back(atlas_list[i].get_list()[j]);
                regions.push_back(region);
            }
            std::string result;
            get_regions_statistics(handle,regions,region_list,result);

            std::string file_name(po.get("source"));
            file_name += ".";
            file_name += atlas_list[i].name;
            file_name += ".statistics.txt";
            std::cout << "export ROI statistics to file:" << file_name << std::endl;
            std::ofstream out(file_name.c_str());
            out << result <<std::endl;
        }
        return 0;
    }

    if(!po.has("tract"))
    {
        if(!po.has("roi"))
        {
            std::cout << "No tract file or ROI file assigned." << std::endl;
            return 0;
        }
        ROIRegion region(handle->dim,handle->vs);
        if(!load_region(handle,region,po.get("roi"),mapping))
        {
            std::cout << "Fail to load the ROI file." << std::endl;
            return 0;
        }
        export_indices(handle,region,po.get("roi"));
        return 0;
    }

    TractModel tract_model(handle);
    std::string file_name = po.get("tract");
    {
        std::cout << "loading " << file_name << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!tract_model.load_from_file(file_name.c_str()))
        {
            std::cout << "Cannot open file " << file_name << std::endl;
            return 0;
        }
        std::cout << file_name << " loaded" << std::endl;

    }
    RoiMgr roi_mgr;
    if(!load_roi(handle,mapping,roi_mgr))
        return -1;
    tract_model.filter_by_roi(roi_mgr);
    return trk_post(handle,tract_model,mapping,po.get("output"));
}
