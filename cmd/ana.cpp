#include <regex>
#include <QFileInfo>
#include <QDir>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>

#include "TIPL/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "atlas.hpp"

#include <filesystem>

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000
bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list);
bool load_roi(tipl::io::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);

void get_regions_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result);
bool load_region(tipl::io::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
bool get_t1t2_nifti(const std::string& t1t2,
                    std::shared_ptr<fib_data> handle,
                    tipl::shape<3>& nifti_geo,
                    tipl::vector<3>& nifti_vs,
                    tipl::matrix<4,4>& convert);
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<std::pair<tipl::shape<3>,tipl::matrix<4,4> > >& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names,
              std::string& error_msg,
              bool is_mni);


bool load_nii(tipl::io::program_option<tipl::out>& po,
              std::shared_ptr<fib_data> handle,
              const std::string& region_text,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names)
{
    std::vector<std::pair<tipl::shape<3>,tipl::matrix<4,4> > > transform_lookup;
    // --t1t2 provide registration
    if(po.has("t1t2"))
    {
        tipl::shape<3> t1t2_geo;
        tipl::vector<3> vs;
        tipl::matrix<4,4> T;
        if(!get_t1t2_nifti(po.get("t1t2"),handle,t1t2_geo,vs,T))
            return false;
        T.inv();
        transform_lookup.push_back(std::make_pair(t1t2_geo,T));
    }

    QStringList str_list = QString(region_text.c_str()).split(",");// splitting actions
    QString file_name = str_list[0];
    std::string error_msg;
    if(QFileInfo(file_name).baseName().toLower().contains("mni"))
    {
        tipl::out() << QFileInfo(file_name).baseName().toStdString() <<
                     " has mni in the file name. It will be loaded as an MNI space image" << std::endl;
    }
    if(!load_nii(handle,file_name.toStdString(),transform_lookup,regions,names,error_msg,QFileInfo(file_name).baseName().toLower().contains("mni")))
    {
        tipl::out() << "ERROR:" << error_msg << std::endl;
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

void get_filenames_from(const std::string name,std::vector<std::string>& filenames)
{
    std::istringstream in(name);
    std::string line;
    std::vector<std::string> file_list;
    while(std::getline(in,line,','))
        file_list.push_back(line);
    for(size_t index = 0;index < file_list.size();++index)
    {
        std::string cur_file = file_list[index];
        if(cur_file.find('*') != std::string::npos)
        {
            QStringList new_list;
            std::string search_path;
            if(cur_file.find('/') != std::string::npos)
            {
                search_path = cur_file.substr(0,cur_file.find_last_of('/'));
                std::string filter = cur_file.substr(cur_file.find_last_of('/')+1);
                tipl::out() << "searching " << filter << " in directory " << search_path << std::endl;
                new_list = QDir(search_path.c_str()).entryList(QStringList(filter.c_str()),QDir::Files);
                search_path += "/";
            }
            else
            {
                tipl::out() << "searching " << cur_file << std::endl;
                new_list = QDir().entryList(QStringList(cur_file.c_str()),QDir::Files);
            }
            tipl::out() << "found " << new_list.size() << " files." << std::endl;
            for(int i = 0;i < new_list.size();++i)
                file_list.push_back(search_path + new_list[i].toStdString());
        }
        else
            filenames.push_back(file_list[index]);
    }
    if(filenames.size() > file_list.size())
        tipl::out() << "a total of " << filenames.size() << "files matching the search" << std::endl;
}

int trk_post(tipl::io::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model,std::string tract_file_name,bool output_track);
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);

bool load_tracts(const char* file_name,std::shared_ptr<TractModel> tract_model,std::shared_ptr<RoiMgr> roi_mgr)
{
    if(!std::filesystem::exists(file_name))
    {
        tipl::out() << "ERROR:" << file_name << " does not exist. terminating..." << std::endl;
        return 1;
    }
    if(!tract_model->load_from_file(file_name))
    {
        tipl::out() << "ERROR: cannot read or parse the tractography file :" << file_name << std::endl;
        return false;
    }
    tipl::out() << "A total of " << tract_model->get_visible_track_count() << " tracks loaded" << std::endl;
    if(!roi_mgr->report.empty())
    {
        tipl::out() << "filtering tracts using roi/roa/end regions." << std::endl;
        tract_model->filter_by_roi(roi_mgr);
        tipl::out() << "remaining tract count:" << tract_model->get_visible_track_count() << std::endl;
    }
    return true;
}
bool check_other_slices(const std::string& other_slices,std::shared_ptr<fib_data> handle);
int ana_region(tipl::io::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    if(!handle.get())
        return 1;
    std::vector<std::string> region_list;
    std::vector<std::shared_ptr<ROIRegion> > regions;
    if(po.has("atlas"))
    {
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
            return 1;
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
                    tipl::out() << "fail to load the ROI file:" << region_name << std::endl;
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
                tipl::out() << "fail to load the ROI file." << std::endl;
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
        tipl::out() << "ERROR: no region assigned" << std::endl;
        return 1;
    }

    // allow adding other slices for connectivity and statistics
    if(po.has("other_slices") && !check_other_slices(po.get("other_slices"),handle))
        return 1;

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
    tipl::out() << "export ROI statistics to file:" << file_name << std::endl;
    std::ofstream out(file_name.c_str());
    out << result <<std::endl;
    return 0;
}
int ana_tract(tipl::io::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle));
    std::string output = po.get("output");
    std::vector<std::string> tract_files;
    if(!handle.get() || !load_roi(po,handle,roi_mgr))
        return 1;

    get_filenames_from(po.get("tract"),tract_files);
    if(tract_files.size() == 0)
    {
        tipl::out() << "No tract file assign to --tract" << std::endl;
        return 1;
    }

    // accumulate multiple tracts into one probabilistic nifti volume
    if(tract_files.size() > 1 && QString(output.c_str()).endsWith(".nii.gz"))
    {
        tipl::out() << "computing tract probability to " << output << std::endl;
        if(std::filesystem::exists(output))
        {
            tipl::out() << "output file:" << output << " exists. terminating..." << std::endl;
            return 0;
        }
        auto dim = handle->dim;
        tipl::image<3,uint32_t> accumulate_map(dim);
        for(size_t i = 0;i < tract_files.size();++i)
        {
            tipl::out() << "accumulating " << tract_files[i] << "..." <<std::endl;
            std::shared_ptr<TractModel> tract(new TractModel(handle));
            if(!load_tracts(tract_files[i].c_str(),tract,roi_mgr))
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
        if(!gz_nifti::save_to_file(output.c_str(),pdi,handle->vs,handle->trans_to_mni,handle->is_mni))
        {
            tipl::out() << "ERROR: cannot write to " << output << std::endl;
            return 1;
        }
        tipl::out() << "file saved at " << output << std::endl;
        return 0;
    }


    std::vector<std::shared_ptr<TractModel> > tracts;
    for(size_t i = 0;i < tract_files.size();++i)
    {
        tracts.push_back(std::make_shared<TractModel>(handle));
        if(!load_tracts(tract_files[i].c_str(),tracts.back(),roi_mgr))
            return 1;
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
                tipl::out() << "ERROR: cannot write to " << output << std::endl;
                return 1;
            }
            tipl::out() << "file saved at " << output << std::endl;
            return 0;
        }
    }

    // accumulate all tracts into one
    std::shared_ptr<TractModel> tract_model = tracts[0];
    for(unsigned int i = 1;i < tracts.size();++i)
        tract_model->add(*tracts[i].get());

    if(po.has("output") && QFileInfo(output.c_str()).isDir())
        return trk_post(po,handle,tract_model,output + "/" + QFileInfo(tract_files[0].c_str()).baseName().toStdString(),false);
    if(po.has("output"))
        return trk_post(po,handle,tract_model,output,true);
    return trk_post(po,handle,tract_model,tract_files[0],false);

}
int ana(tipl::io::program_option<tipl::out>& po)
{
    if(po.has("atlas") || po.has("region") || po.has("regions"))
        return ana_region(po);
    if(po.has("tract"))
        return ana_tract(po);
    if(po.has("info"))
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
            return 1;
        auto result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,[handle](size_t pos,unsigned int fib)
                                        {return handle->dir.get_fib(pos,fib);});
        std::ofstream out(po.get("info"));
        out << "fiber coherence index\t" << result.first << std::endl;
        return 0;
    }
    tipl::out() << "no tract file or ROI file assigned." << std::endl;
    return 1;
}
