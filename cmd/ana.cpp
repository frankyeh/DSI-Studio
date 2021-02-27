#include <regex>
#include <QFileInfo>
#include <QStringList>
#include <QImage>
#include <iostream>
#include <iterator>
#include <string>
#include "tipl/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "program_option.hpp"
#include "atlas.hpp"

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000
bool atl_load_atlas(std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list);
bool load_roi(std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);
void get_connectivity_matrix(std::shared_ptr<fib_data> handle,
                             TractModel& tract_model);

void get_regions_statistics(const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result);
void export_track_info(std::shared_ptr<fib_data> handle,
                       TractModel& tract_model)
{
    std::string export_option = po.get("export");
    std::string file_name = po.get("output",po.get("tract",po.get("source")+"tt.gz"));
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
            uint32_t profile_dir = 0,bandwidth = 0;
            in >> report_tag >> index_name >> profile_dir >> bandwidth;
            std::vector<float> values,data_profile,data_ci1,data_ci2;
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
            std::cout << "profile_dir:" << profile_dir << std::endl;
            std::cout << "bandwidth:" << bandwidth << std::endl;
            std::cout << "index_name:" << index_name << std::endl;
            tract_model.get_report(
                                profile_dir,
                                bandwidth,
                                index_name,
                                values,data_profile,data_ci1,data_ci2);

            std::replace(cmd.begin(),cmd.end(),' ','.');
            std::string file_name_stat = file_name + "." + cmd + ".txt";
            std::cout << "output report:" << file_name_stat << std::endl;
            std::ofstream report(file_name_stat.c_str());
            report << "position\t";
            std::copy(values.begin(),values.end(),std::ostream_iterator<float>(report,"\t"));
            report << std::endl;
            report << "value";
            std::copy(data_profile.begin(),data_profile.end(),std::ostream_iterator<float>(report,"\t"));
            if(!data_ci1.empty())
            {
                report << "CI";
                std::copy(data_ci1.begin(),data_ci1.end(),std::ostream_iterator<float>(report,"\t"));
            }
            if(!data_ci2.empty())
            {
                report << "CI";
                std::copy(data_ci2.begin(),data_ci2.end(),std::ostream_iterator<float>(report,"\t"));
            }
            report << std::endl;
            continue;
        }

        std::string file_name_stat = file_name + "." + cmd;
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
        if(cmd == "tdi_color" || cmd == "tdi2_color")
        {
            file_name_stat += ".bmp";
            std::cout << "export subvoxel TDI to " << file_name_stat << std::endl;
            tipl::image<tipl::rgb,3> tdi;
            tipl::matrix<4,4,float> tr;
            tr.identity();
            if(cmd == "tdi_color")
                tdi.resize(handle->dim);
            else
            {
                tdi.resize(tipl::geometry<3>(handle->dim[0]*4,handle->dim[1]*4,handle->dim[2]*4));
                tr[0] = tr[5] = tr[10] = 4.0f;
            }
            tract_model.get_density_map(tdi,tr,false);
            tipl::image<tipl::rgb,2> mosaic;
            tipl::mosaic(tdi,mosaic,uint32_t(std::sqrt(tdi.depth())));
            QImage qimage(reinterpret_cast<unsigned char*>(&*mosaic.begin()),
                          mosaic.width(),mosaic.height(),QImage::Format_RGB32);
            qimage.save(file_name_stat.c_str());
            continue;
        }


        if(cmd == "stat")
        {
            file_name_stat += ".txt";
            std::cout << "export statistics to " << file_name_stat << std::endl;
            std::ofstream out_stat(file_name_stat.c_str());
            if(!out_stat)
            {
                std::cout << "Output statistics to file_name_stat failed. Please check write permission" << std::endl;
                return;
            }
            std::string result;
            tract_model.get_quantitative_info(result);
            out_stat << result;
            continue;
        }

        {
            if(cmd.find('.') != std::string::npos)
                cmd = cmd.substr(0,cmd.find('.'));
            else
                file_name_stat += ".txt";
            if(handle->get_name_index(cmd) != handle->view_item.size())
            {
                tract_model.save_data_to_file(file_name_stat.c_str(),cmd);
                continue;
            }
        }
        std::cout << "invalid export option:" << cmd << std::endl;
        continue;
    }
}
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
bool get_t1t2_nifti(std::shared_ptr<fib_data> handle,
                    tipl::geometry<3>& nifti_geo,
                    tipl::matrix<4,4,float>& convert);
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<std::pair<tipl::geometry<3>,tipl::matrix<4,4,float> > >& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names,bool verbose);
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names)
{
    std::vector<std::pair<tipl::geometry<3>,tipl::matrix<4,4,float> > > transform_lookup;
    // --t1t2 provide registration
    {
        tipl::geometry<3> t1t2_geo;
        tipl::matrix<4,4,float> convert;
        if(get_t1t2_nifti(handle,t1t2_geo,convert))
            transform_lookup.push_back(std::make_pair(t1t2_geo,convert));
    }
    if(!load_nii(handle,file_name,transform_lookup,regions,names,true))
    {
        std::cout << "ERROR: fail to load multi-region NIFTI file." << std::endl;
        return false;
    }
    return true;
}
void trk_post(std::shared_ptr<fib_data> handle,
             TractModel& tract_model);
std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);
int ana(void)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    if(!handle.get())
        return 1;
    if(po.has("info"))
    {
        float otsu = tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim))*0.6f;
        auto result = evaluate_fib(handle->dim,otsu,handle->dir.fa,[handle](size_t pos,unsigned int fib)
                                        {return tipl::vector<3>(handle->dir.get_dir(pos,fib));});
        std::ofstream out(po.get("info"));
        out << "fiber coherence index\t" << result.first << std::endl;
    }

    if(po.has("atlas") || po.has("region") || po.has("regions"))
    {
        std::vector<std::string> region_list;
        std::vector<std::shared_ptr<ROIRegion> > regions;
        if(po.has("atlas"))
        {
            std::vector<std::shared_ptr<atlas> > atlas_list;
            if(!atl_load_atlas(po.get("atlas"),atlas_list))
                return 1;
            for(unsigned int i = 0;i < atlas_list.size();++i)
            {
                for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
                {
                    std::shared_ptr<ROIRegion> region(std::make_shared<ROIRegion>(handle.get()));
                    std::string region_name = atlas_list[i]->name;
                    region_name += ":";
                    region_name += atlas_list[i]->get_list()[j];
                    if(!load_region(handle,*region.get(),region_name))
                    {
                        std::cout << "fail to load the ROI file:" << region_name << std::endl;
                        return 1;
                    }
                    region_list.push_back(atlas_list[i]->get_list()[j]);
                    regions.push_back(region);
                }

            }
        }
        if(po.has("region"))
        {
            std::string text = po.get("region");
            std::regex reg("[,]");
            std::sregex_token_iterator first{text.begin(), text.end(),reg, -1},last;
            std::vector<std::string> roi_list = {first, last};
            for(size_t i = 0;i < roi_list.size();++i)
            {
                std::shared_ptr<ROIRegion> region(new ROIRegion(handle.get()));
                if(!load_region(handle,*region.get(),roi_list[i]))
                {
                    std::cout << "fail to load the ROI file." << std::endl;
                    return 1;
                }
                region_list.push_back(roi_list[i]);
                regions.push_back(region);
            }
        }
        if(po.has("regions") && !load_nii(handle,po.get("regions"),regions,region_list))
            return 1;
        if(regions.empty())
        {
            std::cout << "ERROR: no region assigned" << std::endl;
            return 1;
        }
        std::string result;
        std::cout << "calculating region statistics at a total of " << regions.size() << " regions" << std::endl;
        get_regions_statistics(regions,region_list,result);
        std::string file_name(po.get("source"));
        file_name += ".statistics.txt";
        if(po.has("output"))
            file_name = po.get("output");
        std::cout << "export ROI statistics to file:" << file_name << std::endl;
        std::ofstream out(file_name.c_str());
        out << result <<std::endl;
        return 0;
    }

    if(!po.has("tract"))
    {
        std::cout << "no tract file or ROI file assigned." << std::endl;
        return 0;
    }

    TractModel tract_model(handle.get());
    {
        std::string file_name = po.get("tract");
        {
            std::cout << "loading " << file_name << "..." <<std::endl;
            if(!QFileInfo(file_name.c_str()).exists())
            {
                std::cout << file_name << " does not exist. terminating..." << std::endl;
                return 1;
            }
            if (!tract_model.load_from_file(file_name.c_str()))
            {
                std::cout << "cannot open file " << file_name << std::endl;
                return 1;
            }
            std::cout << file_name << " loaded" << std::endl;
        }
        std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle.get()));
        if(!load_roi(handle,roi_mgr))
            return 1;
        tract_model.filter_by_roi(roi_mgr);
        if(tract_model.get_visible_track_count() == 0)
        {
            std::cout << "no tracks remained after ROI selection." << std::endl;
            return 1;
        }
        trk_post(handle,tract_model);
    }
    return 0;
}
