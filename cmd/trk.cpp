#include <QFileInfo>
#include <QStringList>
#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include "tipl/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "SliceModel.h"
#include "connectometry/group_connectometry_analysis.h"
#include "program_option.hpp"
bool atl_load_atlas(const std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list);
void export_track_info(const std::string& file_name,
                       std::string export_option,
                       std::shared_ptr<fib_data> handle,
                       TractModel& tract_model);

void save_connectivity_matrix(TractModel& tract_model,
                              ConnectivityMatrix& data,
                              const std::string& source,
                              const std::string& connectivity_roi,
                              const std::string& connectivity_value,
                              double t,
                              bool use_end_only)
{
    std::cout << "count tracks by " << (use_end_only ? "ending":"passing") << std::endl;
    std::cout << "calculate matrix using " << connectivity_value << std::endl;
    if(!data.calculate(tract_model,connectivity_value,use_end_only,t))
    {
        std::cout << "connectivity calculation error:" << data.error_msg << std::endl;
        return;
    }
    if(data.overlap_ratio > 0.5)
    {
        std::cout << "the ROIs have a large overlapping area (ratio="
                  << data.overlap_ratio << "). The network measure calculated may not be reliable" << std::endl;
    }
    if(connectivity_value == "trk")
        return;
    std::string file_name_stat(source);
    file_name_stat += ".";
    file_name_stat += (QFileInfo(connectivity_roi.c_str()).exists()) ? QFileInfo(connectivity_roi.c_str()).baseName().toStdString():connectivity_roi;
    file_name_stat += ".";
    file_name_stat += connectivity_value;
    file_name_stat += use_end_only ? ".end":".pass";
    std::string network_measures(file_name_stat),connectogram(file_name_stat);
    file_name_stat += ".connectivity.mat";
    std::cout << "export connectivity matrix to " << file_name_stat << std::endl;
    data.save_to_file(file_name_stat.c_str());
    connectogram += ".connectogram.txt";
    std::cout << "export connectogram to " << connectogram << std::endl;
    data.save_to_connectogram(connectogram.c_str());

    network_measures += ".network_measures.txt";
    std::cout << "export network measures to " << network_measures << std::endl;
    std::string report;
    data.network_property(report);
    std::ofstream out(network_measures.c_str());
    out << report;
}
void get_roi_label(QString file_name,std::map<int,std::string>& label_map,
                          std::map<int,tipl::rgb>& label_color,bool is_freesurfer,bool mute_cmd);
bool qsdr_convert_native(std::shared_ptr<fib_data> handle,
                         tipl::image<unsigned int, 3>& from);
void get_connectivity_matrix(std::shared_ptr<fib_data> handle,
                             TractModel& tract_model)
{
    std::string source;
    QStringList connectivity_list = QString(po.get("connectivity").c_str()).split(",");
    QStringList connectivity_type_list = QString(po.get("connectivity_type","end").c_str()).split(",");
    QStringList connectivity_value_list = QString(po.get("connectivity_value","count").c_str()).split(",");
    if(po.has("output"))
        source = po.get("output");
    if(source == "no_file" || source.empty())
        source = po.get("source");
    for(unsigned int i = 0;i < connectivity_list.size();++i)
    {
        std::string roi_file_name = connectivity_list[i].toStdString();
        std::cout << "loading " << roi_file_name << std::endl;
        ConnectivityMatrix data;

        if(QFileInfo(roi_file_name.c_str()).suffix() == "txt") // a roi list
        {
            std::string dir = QFileInfo(roi_file_name.c_str()).absolutePath().toStdString();
            dir += "/";
            std::ifstream in(roi_file_name.c_str());
            std::string line;
            std::vector<std::vector<tipl::vector<3,short> > > regions;
            while(std::getline(in,line))
            {
                ROIRegion region(handle);
                std::string fn;
                if(QFileInfo(line.c_str()).exists())
                    fn = line;
                else
                    fn = dir + line;
                if(!region.LoadFromFile(fn.c_str()))
                {
                    std::cout << "failed to open file as a region:" << fn << std::endl;
                    return;
                }
                regions.push_back(std::vector<tipl::vector<3,short> >());
                region.get_region_voxels(regions.back());
                data.region_name.push_back(QFileInfo(line.c_str()).baseName().toStdString());
            }
            data.set_regions(handle->dim,regions);
            std::cout << "a total of " << data.region_count << " regions are loaded." << std::endl;
        }
        else
        {
            gz_nifti header;
            tipl::image<unsigned int, 3> from;
            // if an ROI file is assigned, load it
            if (header.load_from_file(roi_file_name))
                header.toLPS(from);
            if(!from.empty() && qsdr_convert_native(handle,from))
                std::cout << roi_file_name << " converted from native space to QSDR space." << std::endl;
            // if atlas or MNI space ROI is used
            if(from.geometry() != handle->dim &&
               (from.empty() || QFileInfo(roi_file_name.c_str()).baseName() != "aparc+aseg"))
            {
                std::cout << roi_file_name << " is used as an MNI space ROI." << std::endl;
                if(!handle->can_map_to_mni())
                {
                    std::cout << "cannot output connectivity: no mni mapping" << std::endl;
                    continue;
                }
                std::vector<std::shared_ptr<atlas> > atlas_list;
                if(atl_load_atlas(roi_file_name,atlas_list))
                    data.set_atlas(atlas_list[0],handle->get_mni_mapping());
                else
                {
                    std::cout << "file or atlas does not exist:" << roi_file_name << std::endl;
                    continue;
                }
            }
            else
            {
                if(handle->is_qsdr)
                    std::cout << roi_file_name << " is used as a QSDR space ROI." << std::endl;
                else
                    std::cout << roi_file_name << " is used as a native space ROI." << std::endl;
                std::vector<unsigned char> value_map(std::numeric_limits<unsigned short>::max());
                unsigned int max_value = 0;
                for (tipl::pixel_index<3>index(from.geometry()); index < from.size();++index)
                {
                    value_map[(unsigned short)from[index.index()]] = 1;
                    max_value = std::max<unsigned short>(from[index.index()],max_value);
                }
                value_map.resize(max_value+1);
                unsigned short region_count = std::accumulate(value_map.begin(),value_map.end(),(unsigned short)0);
                if(region_count < 2)
                {
                    std::cout << "the ROI file should contain at least two regions to calculate the connectivity matrix." << std::endl;
                    continue;
                }
                std::cout << "total number of regions=" << region_count << std::endl;

                // get label file
                std::map<int,std::string> label_map;
                std::map<int,tipl::rgb> label_color;
                std::string des(header.get_descrip());
                get_roi_label(roi_file_name.c_str(),label_map,label_color,des.find("FreeSurfer") == 0,false);
                std::vector<std::vector<tipl::vector<3,short> > > regions;
                for(unsigned int value = 1;value < value_map.size();++value)
                    if(value_map[value])
                    {
                        tipl::image<unsigned char,3> mask(from.geometry());
                        for(unsigned int i = 0;i < mask.size();++i)
                            if(from[i] == value)
                                mask[i] = 1;
                        ROIRegion region(handle);
                        region.LoadFromBuffer(mask);
                        regions.push_back(std::vector<tipl::vector<3,short> >());
                        region.get_region_voxels(regions.back());
                        if(label_map.find(value) != label_map.end())
                            data.region_name.push_back(label_map[value]);
                        else
                        {
                            std::ostringstream out;
                            out << "region" << value;
                            data.region_name.push_back(out.str());
                        }
                    }
                data.set_regions(handle->dim,regions);
            }
        }
        for(unsigned int j = 0;j < connectivity_type_list.size();++j)
        for(unsigned int k = 0;k < connectivity_value_list.size();++k)
            save_connectivity_matrix(tract_model,data,source,roi_file_name,connectivity_value_list[k].toStdString(),
                                     po.get("connectivity_threshold",0.001),
                                     connectivity_type_list[j].toLower() == QString("end"));
    }
}

// test example
// --action=trk --source=./test/20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name)
{
    std::shared_ptr<fib_data> handle(new fib_data);
    std::cout << "loading " << file_name << "..." <<std::endl;
    if(!QFileInfo(file_name.c_str()).exists())
    {
        std::cout << file_name << " does not exist. terminating..." << std::endl;
        return std::shared_ptr<fib_data>();
    }
    if (!handle->load_from_file(file_name.c_str()))
    {
        std::cout << "open file " << file_name << " failed" << std::endl;
        std::cout << "msg:" << handle->error_msg << std::endl;
        return std::shared_ptr<fib_data>();
    }
    return handle;
}

bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text)
{
    std::cout << "read region from " << region_text << std::endl;
    QStringList str_list = QString(region_text.c_str()).split(",");// splitting actions
    std::string file_name = str_list[0].toStdString();
    std::string region_name;

    // --roi=file_name:value
    if(file_name.find_last_of(':') != std::string::npos &&
       file_name.at(file_name.find_last_of(':')+1) != '\\')
    {
        region_name = file_name.substr(file_name.find_last_of(':')+1);
        file_name = file_name.substr(0,file_name.find_last_of(':'));
    }

    if(!QFileInfo(file_name.c_str()).exists())
    {
        LOAD_MNI:
        std::cout << "searching " << file_name << " from the atlas pool..." << std::endl;
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(file_name,atlas_list))
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return false;
        }

        if(!handle->can_map_to_mni())
        {
            std::cout << "cannot output connectivity: no mni mapping." << std::endl;
            return false;
        }
        if(region_name.empty())
        {
            std::cout << "please assign region name of an atlas." << std::endl;
            return false;
        }
        const tipl::image<tipl::vector<3,float>,3 >& mapping = handle->get_mni_mapping();
        std::cout << "loading " << region_name << " from " << file_name << " atlas" << std::endl;
        tipl::vector<3> null;
        std::vector<tipl::vector<3,short> > cur_region;
        for(unsigned int i = 0;i < atlas_list.size();++i)
            if(atlas_list[i]->name == file_name)
                for (unsigned int label_index = 0; label_index < atlas_list[i]->get_list().size(); ++label_index)
                    if(atlas_list[i]->get_list()[label_index] == region_name)
                {
                    for (tipl::pixel_index<3>index(mapping.geometry());index < mapping.size();++index)
                        if(mapping[index.index()] != null && atlas_list[i]->is_labeled_as(mapping[index.index()],label_index))
                            cur_region.push_back(tipl::vector<3,short>(index.begin()));
                }
        roi.add_points(cur_region,false);
    }
    else
    {
        tipl::geometry<3> t1t2_geo;
        tipl::matrix<4,4,float> convert;

        if(po.has("t1t2"))
        {
            std::shared_ptr<CustomSliceModel> other_slice(std::make_shared<CustomSliceModel>(handle));
            std::vector<std::string> files;
            files.push_back(po.get("t1t2"));
            if(!other_slice->initialize(files,true))
            {
                std::cout << "fail to insert T1T2" << std::endl;
                return false;
            }
            handle->view_item.pop_back(); // remove the new item added by initialize
            if(other_slice->thread.get())
                other_slice->thread->wait();
            t1t2_geo = other_slice->source_images.geometry();
            convert = other_slice->invT;
            std::cout << "registeration complete" << std::endl;
            std::cout << convert[0] << " " << convert[1] << " " << convert[2] << " " << convert[3] << std::endl;
            std::cout << convert[4] << " " << convert[5] << " " << convert[6] << " " << convert[7] << std::endl;
            std::cout << convert[8] << " " << convert[9] << " " << convert[10] << " " << convert[11] << std::endl;
        }
        if(!roi.LoadFromFile(file_name.c_str()))
        {
            gz_nifti header;
            if (!header.load_from_file(file_name.c_str()))
            {
                std::cout << "not a valid nifti file:" << file_name << std::endl;
                return false;
            }
            tipl::image<int, 3> from;
            header.toLPS(from);
            if(!region_name.empty())
            {
                int region_value = std::stoi(region_name);
                std::cout << "select region with value=" << region_value << std::endl;
                for(int i = 0 ;i < from.size();++i)
                    from[i] = (from[i] == region_value ? 1:0);
            }

            if(t1t2_geo == from.geometry())
            {
                std::cout << "using t1t2 as the reference to load " << file_name << std::endl;
                roi.LoadFromBuffer(from,convert);
            }
            else
            if(from.geometry() == handle->dim)
            {
                std::cout << "loading " << file_name << "as a native space region" << std::endl;
                roi.LoadFromBuffer(from);
            }
            else
            {
                std::cout << "loading " << file_name << "as an MNI space region" << std::endl;
                goto LOAD_MNI;
            }
        }
    }
    // now perform actions
    for(int i = 1;i < str_list.size();++i)
    {
        std::cout << str_list[i].toStdString() << " applied." << std::endl;
        roi.perform(str_list[i].toStdString());
    }
    if(roi.empty())
        std::cout << "warning: " << file_name << " is an empty region file" << std::endl;
    return true;
}

int trk_post(std::shared_ptr<fib_data> handle,
             TractModel& tract_model,
             const std::string& file_name,
             bool save_track)
{
    if (po.has("delete_repeat"))
    {
        std::cout << "deleting repeat tracks..." << std::endl;
        float distance = po.get("delete_repeat",float(1));
        tract_model.delete_repeated(distance);
        std::cout << "repeat tracks with distance smaller than " << distance <<" voxel distance are deleted" << std::endl;
    }
    if(po.has("trim"))
    {
        std::cout << "trimming tracks..." << std::endl;
        int trim = po.get("trim",int(1));
        for(int i = 0;i < trim;++i)
            tract_model.trim();
    }
    if(save_track)
    {
        std::string file_list = file_name;
        std::replace(file_list.begin(),file_list.end(),',',' ');
        std::istringstream in(file_list);
        std::string f;
        while(in >> f)
        {
            if(po.has("ref")) // save track in T1W/T2W space
            {
                std::vector<std::string> files;
                files.push_back(po.get("ref"));
                CustomSliceModel new_slice(handle);
                if(!new_slice.initialize(files,false))
                {
                    std::cout << "error reading ref image file" << std::endl;
                    return 1;
                }
                new_slice.thread->wait();
                new_slice.update();
                std::cout << "applying linear registration." << std::endl;
                std::cout << new_slice.T[0] << " " << new_slice.T[1] << " " << new_slice.T[2] << " " << new_slice.T[3] << std::endl;
                std::cout << new_slice.T[4] << " " << new_slice.T[5] << " " << new_slice.T[6] << " " << new_slice.T[7] << std::endl;
                std::cout << new_slice.T[8] << " " << new_slice.T[9] << " " << new_slice.T[10] << " " << new_slice.T[11] << std::endl;
                tract_model.save_transformed_tracts_to_file(f.c_str(),&*new_slice.invT.begin(),false);
            }
            else
            if(f != "no_file")
            {
                std::cout << "output file:" << f << std::endl;
                if (!tract_model.save_tracts_to_file(f.c_str()))
                {
                    std::cout << "cannot save tracks as " << f << ". Please check write permission, directory, and disk space." << std::endl;
                }
                if(QFileInfo(f.c_str()).exists())
                    std::cout << "file saved to " << f << std::endl;
            }
        }
    }
    if(po.has("cluster"))
    {
        std::string cmd = po.get("cluster");
        std::replace(cmd.begin(),cmd.end(),',',' ');
        std::istringstream in(cmd);
        int method = 0,count = 0,detail = 0;
        std::string name;
        in >> method >> count >> detail >> name;
        std::cout << "cluster method=" << method << std::endl;
        std::cout << "cluster count=" << count << std::endl;
        std::cout << "cluster resolution (if method is 0) = " << detail << " mm" << std::endl;
        std::cout << "run clustering." << std::endl;
        tract_model.run_clustering(method,count,detail);
        std::ofstream out(name);
        std::cout << "cluster label saved to " << name << std::endl;
        std::copy(tract_model.get_cluster_info().begin(),tract_model.get_cluster_info().end(),std::ostream_iterator<int>(out," "));
    }

    if(po.has(("end_point")))
        tract_model.save_end_points(po.get("end_point").c_str());

    if(po.has("connectivity"))
        get_connectivity_matrix(handle,tract_model);

    if(po.has("export"))
        export_track_info(file_name,po.get("export"),handle,tract_model);
    return 0;

}

extern std::string tractography_atlas_file_name;
extern std::vector<std::string> tractography_name_list;
bool load_roi(std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr)
{
    const int total_count = 18;
    char roi_names[total_count][5] = {"roi","roi2","roi3","roi4","roi5","roa","roa2","roa3","roa4","roa5","end","end2","seed","ter","ter2","ter3","ter4","ter5"};
    unsigned char type[total_count] = {0,0,0,0,0,1,1,1,1,1,2,2,3,4,4,4,4,4};
    for(int index = 0;index < total_count;++index)
    if (po.has(roi_names[index]))
    {
        ROIRegion roi(handle);
        if(!load_region(handle,roi,po.get(roi_names[index])))
            return false;
        roi_mgr->setRegions(handle->dim,roi.get_region_voxels_raw(),roi.resolution_ratio,type[index],po.get(roi_names[index]).c_str(),handle->vs);
    }
    if(po.has("track_id"))
    {
        std::shared_ptr<TractModel> tractography_atlas(new TractModel(handle));
        if(handle->can_map_to_mni() && tractography_atlas->load_from_file(tractography_atlas_file_name.c_str()))
        {
            if(po.get("track_id",0) >= tractography_name_list.size())
            {
                std::cout << "invalid track_id value" << std::endl;
                return false;
            }
            std::cout << "setting target track=" << tractography_name_list[po.get("track_id",0)] << std::endl;
            roi_mgr->setAtlas(tractography_atlas,po.get("track_id",0));
        }
    }
    return true;
}

int trk(std::shared_ptr<fib_data> handle);
int trk(void)
{
    try{
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
            return 1;
        return trk(handle);
    }
    catch(std::exception const&  ex)
    {
        std::cout << "program terminated due to exception:" << ex.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "program terminated due to unkown exception" << std::endl;
    }
    return 0;
}
int trk(std::shared_ptr<fib_data> handle)
{
    if (po.has("threshold_index"))
    {
        if(!handle->dir.set_tracking_index(po.get("threshold_index")))
        {
            std::cout << "failed...cannot find the index" << std::endl;
            return 1;
        }
    }
    if (po.has("dt_threshold_index"))
    {
        if(!handle->dir.set_dt_index(po.get("dt_threshold_index")))
        {
            std::cout << "failed...cannot find the dt index" << std::endl;
            return 1;
        }
    }

    tipl::geometry<3> geometry = handle->dim;
    const float *fa0 = handle->dir.fa[0];
    float otsu = tipl::segmentation::otsu_threshold(tipl::make_image(fa0,geometry));


    ThreadData tracking_thread;
    tracking_thread.param.default_otsu = po.get("otsu_threshold",0.6f);
    tracking_thread.param.threshold = po.get("fa_threshold",tracking_thread.param.default_otsu*otsu);
    tracking_thread.param.dt_threshold = po.get("dt_threshold",0.2f);
    tracking_thread.param.cull_cos_angle = std::cos(po.get("turning_angle",0.0)*3.14159265358979323846/180.0);
    tracking_thread.param.step_size = po.get("step_size",0.0f);
    tracking_thread.param.smooth_fraction = po.get("smoothing",0.0f);
    tracking_thread.param.min_length = po.get("min_length",30.0f);
    tracking_thread.param.max_length = std::max<float>(tracking_thread.param.min_length,po.get("max_length",300.0f));

    tracking_thread.param.tracking_method = po.get("method",int(0));
    tracking_thread.param.initial_direction  = po.get("initial_dir",int(0));
    tracking_thread.param.interpolation_strategy = po.get("interpolation",int(0));
    tracking_thread.param.center_seed = po.get("seed_plan",int(0));
    tracking_thread.param.random_seed = po.get("random_seed",int(0));
    tracking_thread.param.check_ending = po.get("check_ending",int(0));
    tracking_thread.param.tip_iteration = po.get("tip_iteration",int(0));

    if (po.has("fiber_count"))
    {
        tracking_thread.param.termination_count = po.get("fiber_count",int(tracking_thread.param.termination_count));
        tracking_thread.param.stop_by_tract = 1;

        if (po.has("seed_count"))
            tracking_thread.param.max_seed_count = po.get("seed_count",int(tracking_thread.param.termination_count));
    }
    else
    {
        if (po.has("seed_count"))
            tracking_thread.param.termination_count = po.get("seed_count",int(tracking_thread.param.termination_count));
        tracking_thread.param.stop_by_tract = 0;
    }



    QStringList cnt_file_name;
    QString cnt_type;

    if(po.has("connectometry_source"))
    {
        std::string names = po.get("connectometry_source").c_str();
        cnt_file_name = QString(names.c_str()).split(",");
        if(!po.has("connectometry_type"))
        {
            std::cout << "please assign the connectometry analysis type." << std::endl;
            return 1;
        }
        cnt_type = po.get("connectometry_type").c_str();
    }
    TractModel tract_model(handle);

    if(po.get("thread_count",int(std::thread::hardware_concurrency())) < 1)
    {
        std::cout << "invalid thread_count number" << std::endl;
        return 1;
    }
    if(po.has("parameter_id"))
    {
        tracking_thread.param.set_code(po.get("parameter_id"));
        std::cout << "parameter_code=" << tracking_thread.param.get_code() << std::endl;
    }

    if(!load_roi(handle,tracking_thread.roi_mgr))
        return 1;

    if (!po.has("seed"))
    {
        float seed_threshold = tracking_thread.param.threshold;
        if(seed_threshold == 0.0f)
            seed_threshold = otsu*tracking_thread.param.default_otsu;
        tracking_thread.roi_mgr->setWholeBrainSeed(handle,seed_threshold);
    }

    if(!cnt_file_name.empty())
    {
        QStringList connectometry_threshold;
        if(!po.has("connectometry_threshold"))
        {
            std::cout << "please assign the connectometry threshold." << std::endl;
            return 1;
        }
        connectometry_threshold = QString(po.get("connectometry_threshold").c_str()).split(",");
        for(unsigned int i = 0;i < cnt_file_name.size();++i)
        {
            connectometry_result cnt;
            std::cout << "loading individual file:" << cnt_file_name[i].toStdString() << std::endl;
            if(cnt_type == "iva" && !cnt.individual_vs_atlas(handle,cnt_file_name[i].toLocal8Bit().begin(),0))
            {
                std::cout << "error loading connectometry file:" << cnt.error_msg <<std::endl;
                return 1;
            }
            if(cnt_type == "ivp" && !cnt.individual_vs_db(handle,cnt_file_name[i].toLocal8Bit().begin()))
            {
                std::cout << "error loading connectometry file:" << cnt.error_msg <<std::endl;
                return 1;
            }
            if(cnt_type == "ivi")
            {
                std::cout << "loading individual file:" << cnt_file_name[i+1].toStdString() << std::endl;
                if(!cnt.individual_vs_individual(handle,cnt_file_name[i].toLocal8Bit().begin(),
                                                              cnt_file_name[i+1].toLocal8Bit().begin(),0))
                {
                    std::cout << "error loading connectometry file:" << cnt.error_msg <<std::endl;
                    return 1;
                }
                ++i;
            }
            for(unsigned int j = 0;j < connectometry_threshold.size();++j)
            {
                double t = connectometry_threshold[j].toDouble();
                handle->dir.set_tracking_index(handle->dir.index_data.size()-((t > 0) ? 2:1));
                std::cout << "mapping track with " << ((t > 0) ? "increased":"decreased") << " connectivity at " << std::fabs(t) << std::endl;
                std::cout << "start tracking." << std::endl;
                tracking_thread.param.threshold = std::fabs(t);
                tracking_thread.run(tract_model.get_fib(),po.get("thread_count",int(std::thread::hardware_concurrency())),true);
                tracking_thread.fetchTracks(&tract_model);
                std::ostringstream out;
                out << cnt_file_name[i].toStdString() << "." << cnt_type.toStdString()
                        << ((t > 0) ? "inc":"dec") << std::fabs(t) << ".trk.gz" << std::endl;
                if(!tract_model.save_tracts_to_file(out.str().c_str()))
                {
                    std::cout << "cannot save file to " << out.str()
                              << ". Please check write permission, directory, and disk space." << std::endl;
                    return 1;
                }
                std::vector<std::vector<float> > tmp;
                tract_model.release_tracts(tmp);
            }
        }
        return 0;
    }


    std::cout << "start tracking." << std::endl;

    tracking_thread.run(tract_model.get_fib(),po.get("thread_count",int(std::thread::hardware_concurrency())),true);
    tract_model.report += tracking_thread.report.str();
    std::cout << tract_model.report << std::endl;

    tracking_thread.fetchTracks(&tract_model);
    std::cout << "finished tracking." << std::endl;

    for(int i = 0;i < tracking_thread.param.tip_iteration;++i)
        tract_model.trim();

    if(tract_model.get_visible_track_count() == 0)
    {
        std::cout << "no tract generated. Terminating..." << std::endl;
        return 0;
    }
    std::cout << "a total of " << tract_model.get_visible_track_count() << " tracts are generated" << std::endl;

    std::string file_name;
    if (po.has("output"))
        file_name = po.get("output");
    else
    {
        std::ostringstream fout;
        fout << po.get("source") << ".trk.gz";
        file_name = fout.str();
    }
    return trk_post(handle,tract_model,file_name,true/*save track*/);
}
