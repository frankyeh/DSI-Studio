#include <regex>
#include <QFileInfo>
#include <QDir>
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

void get_regions_statistics(const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result);
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
void check_other_slices(std::shared_ptr<fib_data> handle);
bool get_t1t2_nifti(std::shared_ptr<fib_data> handle,
                    tipl::geometry<3>& nifti_geo,
                    tipl::vector<3>& nifti_vs,
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
        tipl::vector<3> vs;
        tipl::matrix<4,4,float> convert;
        if(get_t1t2_nifti(handle,t1t2_geo,vs,convert))
            transform_lookup.push_back(std::make_pair(t1t2_geo,convert));
    }
    if(!load_nii(handle,file_name,transform_lookup,regions,names,true))
    {
        std::cout << "ERROR: fail to load multi-region NIFTI file." << std::endl;
        return false;
    }
    return true;
}
void trk_post(std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model);
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

    std::string file_name = po.get("tract");
    if(file_name.find('*') != std::string::npos && po.has("output"))
    {
        std::string file_list = po.get("output");
        QDir dir = QDir::currentPath();
        QStringList name_list = dir.entryList(QStringList(file_name.c_str()),QDir::Files|QDir::NoSymLinks);

        if(QString(file_list.c_str()).endsWith("nii.gz"))
        {
            auto dim = handle->dim;
            tipl::image<uint32_t,3> accumulate_map(dim);
            for(int i = 0;i < name_list.size();++i)
            {
                std::cout << "loading " << name_list[i].toStdString() << "..." <<std::endl;
                TractModel tract_model(handle.get());
                if(!tract_model.load_from_file(name_list[i].toStdString().c_str()))
                {
                    std::cout << "open file error. terminating..." << std::endl;
                    return 1;
                }
                std::cout << "accumulating " << name_list[i].toStdString() << "..." <<std::endl;
                std::vector<tipl::vector<3,short> > points;
                tract_model.to_voxel(points,1.0f);
                tipl::par_for(points.size(),[&](size_t j)
                {
                    tipl::vector<3,short> p = points[j];
                    if(dim.is_valid(p))
                        accumulate_map[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]++;
                });
            }
            tipl::image<float,3> pdi(accumulate_map);
            tipl::multiply_constant(pdi,1.0f/float(name_list.size()));
            if(gz_nifti::save_to_file(file_name.c_str(),pdi,handle->vs,handle->trans_to_mni))
            {
                std::cout << "file saved at " << file_name << std::endl;
                return 0;
            }
            return 1;
        }
        std::vector<std::shared_ptr<TractModel> > tracts;
        std::vector<std::string> name_list_str;
        for(int i = 0;i < name_list.size();++i)
        {
            std::cout << "loading " << name_list[i].toStdString() << "..." <<std::endl;
            tracts.push_back(std::make_shared<TractModel>(handle.get()));
            if(!tracts.back()->load_from_file(name_list[i].toStdString().c_str()))
            {
                std::cout << "open file error. terminating..." << std::endl;
                return 1;
            }
            name_list_str.push_back(name_list[i].toStdString());
        }
        TractModel::save_all(file_name.c_str(),tracts,name_list_str);
    }
    else
    {
        std::shared_ptr<TractModel> tract_model(new TractModel(handle.get()));
        {
            std::cout << "loading " << file_name << "..." <<std::endl;
            if(!QFileInfo(file_name.c_str()).exists())
            {
                std::cout << file_name << " does not exist. terminating..." << std::endl;
                return 1;
            }
            if (!tract_model->load_from_file(file_name.c_str()))
            {
                std::cout << "cannot open file " << file_name << std::endl;
                return 1;
            }
            std::cout << file_name << " loaded" << std::endl;
        }
        std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle.get()));
        if(!load_roi(handle,roi_mgr))
            return 1;
        tract_model->filter_by_roi(roi_mgr);
        if(tract_model->get_visible_track_count() == 0)
        {
            std::cout << "no tracks remained after ROI selection." << std::endl;
            return 1;
        }
        trk_post(handle,tract_model);
    }
    return 0;
}
