#include <regex>
#include <QFileInfo>
#include <QDir>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "atlas.hpp"

#include <filesystem>

#include "SliceModel.h"

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000
bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list);
bool load_roi(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);

void get_regions_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result);
bool load_region(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<SliceModel*>& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names,
              std::string& error_msg,
              bool is_mni);

extern std::vector<std::shared_ptr<CustomSliceModel> > other_slices;
bool check_other_slices(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle);
bool load_nii(tipl::program_option<tipl::out>& po,
              std::shared_ptr<fib_data> handle,
              const std::string& region_text,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names)
{
    std::vector<SliceModel*> transform_lookup;
    if(!check_other_slices(po,handle))
        return false;
    for(const auto& each : other_slices)
        transform_lookup.push_back(each.get());

    QStringList str_list = QString(region_text.c_str()).split(",");// splitting actions
    QString file_name = str_list[0];
    std::string error_msg;
    if(!load_nii(handle,file_name.toStdString(),transform_lookup,regions,names,error_msg,QFileInfo(file_name).baseName().toLower().contains("mni")))
    {
        tipl::error() << error_msg << std::endl;
        return false;
    }

    // now perform actions
    for(int i = 1;i < str_list.size();++i)
    {
        tipl::out() << str_list[i].toStdString() << " applied." << std::endl;
        for(size_t j = 0;j < regions.size();++j)
            regions[j]->perform(str_list[i].toStdString());
    }

    return true;
}


int trk_post(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model,std::string tract_file_name,bool output_track);
std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);

bool load_tracts(const char* file_name,std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model,std::shared_ptr<RoiMgr> roi_mgr)
{
    if(!std::filesystem::exists(file_name))
    {
        tipl::error() << file_name << " does not exist. terminating..." << std::endl;
        return false;
    }
    if(!tract_model->load_tracts_from_file(file_name,handle.get(),std::string(file_name).find("mni") != std::string::npos))
    {
        tipl::error() << "cannot read or parse " << file_name << std::endl;
        return false;
    }
    tipl::out() << "A total of " << tract_model->get_visible_track_count() << " tracks loaded" << std::endl;
    if(!roi_mgr->report.empty())
    {
        tipl::out() << "filtering tracts using roi/roa/end regions." << std::endl;
        tract_model->filter_by_roi(roi_mgr);
        tipl::out() << "remaining tract count: " << tract_model->get_visible_track_count() << std::endl;
    }
    return true;
}
int ana_region(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    std::vector<std::string> region_list;
    std::vector<std::shared_ptr<ROIRegion> > regions;
    if(po.has("atlas"))
    {
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
        {
            tipl::out() << "fail to load atlas" << std::endl;
            return 1;
        }
        for(unsigned int i = 0;i < atlas_list.size();++i)
        {
            for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
            {
                std::shared_ptr<ROIRegion> region(std::make_shared<ROIRegion>(handle));
                std::string region_name = atlas_list[i]->name;
                region_name += ":";
                region_name += atlas_list[i]->get_list()[j];
                if(!load_region(po,handle,*region.get(),region_name))
                {
                    tipl::out() << "fail to load the ROI file " << region_name << std::endl;
                    return 1;
                }
                region_list.push_back(atlas_list[i]->get_list()[j]);
                regions.push_back(region);
            }

        }
    }
    if(po.has("region"))
    {
        QStringList roi_list = QString(po.get("region").c_str()).split("+");
        for(int i = 0;i < roi_list.size();++i)
        {
            std::shared_ptr<ROIRegion> region(new ROIRegion(handle));
            if(!load_region(po,handle,*region.get(),roi_list[i].toStdString()))
            {
                tipl::error() << "fail to load the ROI file." << std::endl;
                return 1;
            }
            region_list.push_back(roi_list[i].toStdString());
            regions.push_back(region);
        }
    }
    if(po.has("regions"))
    {
        QStringList roi_list = QString(po.get("regions").c_str()).split("+");
        for(int i = 0;i < roi_list.size();++i)
        {
            if(!load_nii(po,handle,po.get("regions"),regions,region_list))
                return 1;
        }
    }
    if(regions.empty())
    {
        tipl::error() << "no region assigned" << std::endl;
        return 1;
    }

    std::string result;
    tipl::out() << "calculating region statistics at a total of " << regions.size() << " regions" << std::endl;
    get_regions_statistics(handle,regions,region_list,result);

    std::string file_name(po.get("source"));
    file_name += ".statistics.txt";
    if(po.has("output"))
    {
        std::string output = po.get("output");
        if(QFileInfo(output.c_str()).isDir())
            file_name = output + std::string("/") + std::filesystem::path(file_name).filename().string();
        else
            file_name = output;
        if(file_name.find(".txt") == std::string::npos)
            file_name += ".txt";
    }
    tipl::out() << "export ROI statistics to " << file_name << std::endl;
    std::ofstream out(file_name.c_str());
    out << result <<std::endl;
    return 0;
}
void get_track_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          const std::vector<std::string>& track_name,
                          std::string& result);
int ana_tract(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle));
    if(!load_roi(po,handle,roi_mgr))
        return 1;

    std::string output = po.get("output");
    std::vector<std::string> tract_files;
    if(!po.get_files("tract",tract_files))
    {
        tipl::error() << po.error_msg << std::endl;
        return 1;
    }

    if(tract_files.size() == 0)
    {
        tipl::error() << "No tract file assign to --tract" << std::endl;
        return 1;
    }

    // accumulate multiple tracts into one probabilistic nifti volume
    if(tract_files.size() > 1 && QString(output.c_str()).endsWith(".nii.gz"))
    {
        tipl::out() << "computing tract probability to " << output << std::endl;
        if(std::filesystem::exists(output))
        {
            tipl::out() << output << " exists. terminating..." << std::endl;
            return 0;
        }
        auto dim = handle->dim;
        tipl::image<3,uint32_t> accumulate_map(dim);
        for(size_t i = 0;i < tract_files.size();++i)
        {
            tipl::out() << "accumulating " << tract_files[i] << "..." <<std::endl;
            std::shared_ptr<TractModel> tract(new TractModel(handle));
            if(!load_tracts(tract_files[i].c_str(),handle,tract,roi_mgr))
                return 1;
            std::vector<tipl::vector<3,short> > points;
            tract->to_voxel(points);
            tipl::image<3,char> tract_mask(dim);
            tipl::par_for(points.size(),[&](size_t j)
            {
                auto p = points[j];
                if(dim.is_valid(p))
                    tract_mask[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]=1;
            });
            accumulate_map += tract_mask;
        }
        tipl::image<3> pdi(accumulate_map);
        pdi *= 1.0f/float(tract_files.size());
        if(!tipl::io::gz_nifti::save_to_file(output.c_str(),pdi,handle->vs,handle->trans_to_mni,handle->is_mni))
        {
            tipl::error() << "cannot write to " << output << std::endl;
            return 1;
        }
        tipl::out() << "file saved at " << output << std::endl;
        return 0;
    }


    std::vector<std::shared_ptr<TractModel> > tracts;
    std::vector<std::string> tract_name;
    for(size_t i = 0;i < tract_files.size();++i)
    {
        tracts.push_back(std::make_shared<TractModel>(handle));
        if(!load_tracts(tract_files[i].c_str(),handle,tracts.back(),roi_mgr))
            return 1;
        tract_name.push_back(std::filesystem::path(tract_files[i]).filename().string());
    }

    tipl::out() << "a total of " << tract_files.size() << " tract file(s) loaded" << std::endl;
    // load multiple track files and save as one multi-cluster tract file
    if(tracts.size() > 1)
    {
        if(QString(output.c_str()).endsWith(".trk.gz") ||
           QString(output.c_str()).endsWith(".tt.gz"))
        {
            tipl::out() << "save all tracts to " << output << std::endl;
            if(!TractModel::save_all(output.c_str(),tracts,tract_files))
            {
                tipl::error() << "cannot write to " << output << std::endl;
                return 1;
            }
            tipl::out() << "file saved at " << output << std::endl;
            return 0;
        }
        if(po.get("export") == "stat")
        {
            std::string result,file_name_stat("stat.txt");
            get_track_statistics(handle,tracts,tract_name,result);
            std::ofstream out_stat(file_name_stat.c_str());
            if(!out_stat)
            {
                tipl::out() << "cannot save statistics. please check write permission" << std::endl;
                return false;
            }
            out_stat << result;
            return 0;
        }
    }

    tipl::out() << "merging all tracts into one for post-tract analysis";
    std::shared_ptr<TractModel> tract_model = tracts[0];
    for(unsigned int i = 1;i < tracts.size();++i)
        tract_model->add(*tracts[i].get());

    if(po.has("output") && QFileInfo(output.c_str()).isDir())
        return trk_post(po,handle,tract_model,output + "/" + QFileInfo(tract_files[0].c_str()).baseName().toStdString(),false);
    if(po.has("output"))
        return trk_post(po,handle,tract_model,output,true);
    return trk_post(po,handle,tract_model,tract_files[0],false);

}
int exp(tipl::program_option<tipl::out>& po);
int ana(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po);
    if(!handle.get())
        return 1;
    if(po.has("atlas") || po.has("region") || po.has("regions"))
        return ana_region(po,handle);
    if(po.has("tract"))
        return ana_tract(po,handle);
    if(po.has("info"))
    {
        auto result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,[handle](size_t pos,unsigned int fib)
                                        {return handle->dir.get_fib(pos,fib);});
        std::ofstream out(po.get("info"));
        out << "fiber coherence index\t" << result.first << std::endl;
        return 0;
    }
    if(po.has("export"))
        return exp(po);
    tipl::error() << "no tract file or ROI file assigned." << std::endl;
    return 1;
}
