#include <QFileInfo>
#include <QStringList>
#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include "SliceModel.h"
#include "vbc/vbc_database.h"
#include "program_option.hpp"
bool atl_load_atlas(const std::string atlas_name);
void export_track_info(const std::string& file_name,
                       std::string export_option,
                       std::shared_ptr<fib_data> handle,
                       TractModel& tract_model);
extern std::vector<atlas> atlas_list;

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
        std::cout << "Connectivity calculation error:" << data.error_msg << std::endl;
        return;
    }
    if(data.overlap_ratio > 0.5)
    {
        std::cout << "The ROIs have a large overlapping area (ratio="
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
                          std::map<int,image::rgb_color>& label_color,bool mute_cmd);
void get_connectivity_matrix(std::shared_ptr<fib_data> handle,
                             TractModel& tract_model)
{
    std::string source;
    QStringList connectivity_list = QString(po.get("connectivity").c_str()).split(",");
    QStringList connectivity_type_list = QString( po.get("connectivity_type","end").c_str()).split(",");
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
                    std::cout << "Failed to open file as a region:" << fn << std::endl;
                    return;
                }
                data.regions.push_back(std::vector<image::vector<3,short> >());
                region.get_region_voxels(data.regions.back());
                data.region_name.push_back(QFileInfo(line.c_str()).baseName().toStdString());
            }
            std::cout << "A total of " << data.regions.size() << " regions are loaded." << std::endl;
        }
        else
        {
            gz_nifti header;
            image::basic_image<unsigned int, 3> from;
            // if an ROI file is assigned, load it
            if (header.load_from_file(roi_file_name))
                header.toLPS(from);
            // if atlas or MNI space ROI is used
            if(from.geometry() != handle->dim &&
               (from.empty() || QFileInfo(roi_file_name.c_str()).baseName() != "aparc+aseg"))
            {
                std::cout << roi_file_name << " is used as an MNI space ROI." << std::endl;
                if(handle->get_mni_mapping().empty())
                {
                    std::cout << "Cannot output connectivity: no mni mapping" << std::endl;
                    continue;
                }
                atlas_list.clear(); // some atlas may be loaded in ROI
                if(atl_load_atlas(roi_file_name))
                    data.set_atlas(atlas_list[0],handle->get_mni_mapping());
                else
                {
                    std::cout << "File or atlas does not exist:" << roi_file_name << std::endl;
                    continue;
                }
            }
            else
            {
                std::cout << roi_file_name << " is used as a native space ROI." << std::endl;
                std::vector<unsigned char> value_map(std::numeric_limits<unsigned short>::max());
                unsigned int max_value = 0;
                for (image::pixel_index<3>index(from.geometry()); index < from.size();++index)
                {
                    value_map[(unsigned short)from[index.index()]] = 1;
                    max_value = std::max<unsigned short>(from[index.index()],max_value);
                }
                value_map.resize(max_value+1);
                unsigned short region_count = std::accumulate(value_map.begin(),value_map.end(),(unsigned short)0);
                if(region_count < 2)
                {
                    std::cout << "The ROI file should contain at least two regions to calculate the connectivity matrix." << std::endl;
                    continue;
                }
                std::cout << "total number of regions=" << region_count << std::endl;

                // get label file
                std::map<int,std::string> label_map;
                std::map<int,image::rgb_color> label_color;
                get_roi_label(roi_file_name.c_str(),label_map,label_color,false);
                for(unsigned int value = 1;value < value_map.size();++value)
                    if(value_map[value])
                    {
                        image::basic_image<unsigned char,3> mask(from.geometry());
                        for(unsigned int i = 0;i < mask.size();++i)
                            if(from[i] == value)
                                mask[i] = 1;
                        ROIRegion region(handle);
                        region.LoadFromBuffer(mask);
                        data.regions.push_back(std::vector<image::vector<3,short> >());
                        region.get_region_voxels(data.regions.back());
                        if(label_map.find(value) != label_map.end())
                            data.region_name.push_back(label_map[value]);
                        else
                        {
                            std::ostringstream out;
                            out << "region" << value;
                            data.region_name.push_back(out.str());
                        }
                    }
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
        std::cout << "Open file " << file_name << " failed" << std::endl;
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

    if(file_name.find(':') != std::string::npos &&
       file_name.find(':') != 1)
    {
        if(handle->get_mni_mapping().empty())
        {
            std::cout << "Cannot output connectivity: no mni mapping." << std::endl;
            return false;
        }
        const image::basic_image<image::vector<3,float>,3 >& mapping = handle->get_mni_mapping();
        std::string atlas_name = file_name.substr(0,file_name.find(':'));
        std::string region_name = file_name.substr(file_name.find(':')+1);
        std::cout << "Loading " << region_name << " from " << atlas_name << " atlas" << std::endl;
        if(!atl_load_atlas(atlas_name))
            return false;

        image::vector<3> null;
        std::vector<image::vector<3,short> > cur_region;
        for(unsigned int i = 0;i < atlas_list.size();++i)
            if(atlas_list[i].name == atlas_name)
                for (unsigned int label_index = 0; label_index < atlas_list[i].get_list().size(); ++label_index)
                    if(atlas_list[i].get_list()[label_index] == region_name)
                {
                    for (image::pixel_index<3>index(mapping.geometry());index < mapping.size();++index)
                        if(mapping[index.index()] != null && atlas_list[i].is_labeled_as(mapping[index.index()],label_index))
                            cur_region.push_back(image::vector<3,short>(index.begin()));
                }
        roi.add_points(cur_region,false);
    }
    else
    {
        image::geometry<3> t1t2_geo;
        image::matrix<4,4,float> convert;

        if(po.has("t1t2"))
        {
            std::shared_ptr<CustomSliceModel> other_slice(std::make_shared<CustomSliceModel>());
            std::vector<std::string> files;
            files.push_back(po.get("t1t2"));
            if(!other_slice->initialize(handle,handle->is_qsdr,files,true))
            {
                std::cout << "Fail to insert T1T2" << std::endl;
                return false;
            }
            other_slice->thread->wait();
            t1t2_geo = other_slice->source_images.geometry();
            convert = other_slice->invT;
            std::cout << "Registeration complete" << std::endl;
            std::cout << convert[0] << " " << convert[1] << " " << convert[2] << " " << convert[3] << std::endl;
            std::cout << convert[4] << " " << convert[5] << " " << convert[6] << " " << convert[7] << std::endl;
            std::cout << convert[8] << " " << convert[9] << " " << convert[10] << " " << convert[11] << std::endl;
        }

        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return false;
        }
        if(!roi.LoadFromFile(file_name.c_str()))
        {
            gz_nifti header;
            if (!header.load_from_file(file_name.c_str()))
            {
                std::cout << "Not a valid nifti file:" << file_name << std::endl;
                return false;
            }
            image::basic_image<unsigned int, 3> from;
            {
                image::basic_image<float, 3> tmp;
                header.toLPS(tmp);
                image::add_constant(tmp,0.5);
                from = tmp;
            }
            if(t1t2_geo != from.geometry())
            {
                std::cout << "Invalid region dimension:" << file_name << std::endl;
                return false;
            }
            std::cout << "Region loaded using T1T2 ref image" << std::endl;
            roi.LoadFromBuffer(from,convert);
        }
    }
    // now perform actions
    for(int i = 1;i < str_list.size();++i)
    {
        std::cout << str_list[i].toStdString() << " applied." << std::endl;
        roi.perform(str_list[i].toStdString());
    }
    if(roi.empty())
        std::cout << "Warning: " << file_name << " is an empty region file" << std::endl;
    return true;
}

int trk_post(std::shared_ptr<fib_data> handle,
             TractModel& tract_model,
             const std::string& file_name)
{
    if (po.has("delete_repeat"))
    {
        std::cout << "Deleting repeat tracks..." << std::endl;
        float distance = po.get("delete_repeat",float(1));
        tract_model.delete_repeated(distance);
        std::cout << "Repeat tracks with distance smaller than " << distance <<" voxel distance are deleted" << std::endl;
    }
    if(po.has("trim"))
    {
        std::cout << "Trimming tracks..." << std::endl;
        int trim = po.get("trim",int(1));
        for(int i = 0;i < trim;++i)
            tract_model.trim();
    }
    if(!file_name.empty())
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
                CustomSliceModel new_slice;
                std::cout << "Loading reference image:" << po.get("ref") << std::endl;
                if(!new_slice.initialize(handle,handle->is_qsdr,files,false))
                {
                    std::cout << "Error reading ref image file:" << po.get("ref") << std::endl;
                    return 0;
                }
                new_slice.thread->wait();
                new_slice.update();
                std::cout << "Applying linear registration." << std::endl;
                std::cout << new_slice.transform[0] << " " << new_slice.transform[1] << " " << new_slice.transform[2] << " " << new_slice.transform[3] << std::endl;
                std::cout << new_slice.transform[4] << " " << new_slice.transform[5] << " " << new_slice.transform[6] << " " << new_slice.transform[7] << std::endl;
                std::cout << new_slice.transform[8] << " " << new_slice.transform[9] << " " << new_slice.transform[10] << " " << new_slice.transform[11] << std::endl;
                tract_model.save_transformed_tracts_to_file(f.c_str(),&*new_slice.invT.begin(),false);
            }
            else
            if(f != "no_file")
            {
                std::cout << "output file:" << f << std::endl;
                if (!tract_model.save_tracts_to_file(f.c_str()))
                {
                    std::cout << "Cannot save tracks as " << f << ". Please check write permission, directory, and disk space." << std::endl;
                }
                if(QFileInfo(f.c_str()).exists())
                    std::cout << "File saved to " << f << std::endl;
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
        std::cout << "Cluster method=" << method << std::endl;
        std::cout << "Cluster count=" << count << std::endl;
        std::cout << "Cluster resolution (if method is 0) = " << detail << " mm" << std::endl;
        std::cout << "Run clustering." << std::endl;
        tract_model.run_clustering(method,count,detail);
        std::ofstream out(name);
        std::cout << "Cluster label saved to " << name << std::endl;
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

bool load_roi(std::shared_ptr<fib_data> handle,RoiMgr& roi_mgr)
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
        roi_mgr.setRegions(handle->dim,roi.get_region_voxels_raw(),roi.resolution_ratio,type[index],po.get(roi_names[index]).c_str(),handle->vs);
        std::cout << roi_names[index] << "=" << po.get(roi_names[index]) << std::endl;
    }
    return true;
}


int trk(void)
{
    try{

    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    if(!handle.get())
        return 0;
    if (po.has("threshold_index"))
    {
        std::cout << "setting index to " << po.get("threshold_index") << std::endl;
        if(!handle->dir.set_tracking_index(po.get("threshold_index")))
        {
            std::cout << "failed...cannot find the index" << std::endl;
            return 0;
        }
    }

    image::geometry<3> geometry = handle->dim;
    const float *fa0 = handle->dir.fa[0];
    float otsu = image::segmentation::otsu_threshold(image::make_image(fa0,geometry));


    ThreadData tracking_thread;
    tracking_thread.param.default_otsu = po.get("otsu_threshold",0.6f);
    tracking_thread.param.threshold = po.get("fa_threshold",tracking_thread.param.default_otsu*otsu);
    tracking_thread.param.cull_cos_angle = std::cos(po.get("turning_angle",0.0)*3.14159265358979323846/180.0);
    tracking_thread.param.step_size = po.get("step_size",0.0f);
    tracking_thread.param.smooth_fraction = po.get("smoothing",1.0f);
    tracking_thread.param.min_length = po.get("min_length",0.0f);
    tracking_thread.param.max_length = std::max<float>(tracking_thread.param.min_length,po.get("max_length",400.0f));

    tracking_thread.param.tracking_method = po.get("method",int(0));
    tracking_thread.param.initial_direction  = po.get("initial_dir",int(0));
    tracking_thread.param.interpolation_strategy = po.get("interpolation",int(0));
    tracking_thread.param.center_seed = po.get("seed_plan",int(0));
    tracking_thread.param.random_seed = po.get("random_seed",int(0));
    tracking_thread.param.check_ending = po.get("check_ending",int(0));
    if(po.has("otsu_threshold"))
    {
        if(po.has("fa_threshold"))
            std::cout << "Default Otsu is not used because fa_threshold is assigned" << std::endl;
        else
            std::cout << "A ratio of Otsu threshold of " << po.get("otsu_threshold") << " is used" << std::endl;
    }

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


    if(!load_roi(handle,tracking_thread.roi_mgr))
        return -1;

    QStringList cnt_file_name;
    QString cnt_type;

    if(po.has("connectometry_source"))
    {
        std::string names = po.get("connectometry_source").c_str();
        cnt_file_name = QString(names.c_str()).split(",");
        if(!po.has("connectometry_type"))
        {
            std::cout << "Please assign the connectometry analysis type." << std::endl;
            return -1;
        }
        cnt_type = po.get("connectometry_type").c_str();
    }
    TractModel tract_model(handle);

    if(po.get("thread_count",int(std::thread::hardware_concurrency())) < 1)
    {
        std::cout << "Invalid thread_count number" << std::endl;
        return -1;
    }
    if(po.has("parameter_id"))
    {
        tracking_thread.param.set_code(po.get("parameter_id"));
        std::cout << "parameter_code=" << tracking_thread.param.get_code() << std::endl;
    }
    {
        std::cout << "fa_threshold=" << tracking_thread.param.threshold << std::endl;
        std::cout << "turning_angle=" << std::acos(tracking_thread.param.cull_cos_angle)*180/3.14159265358979323846 << std::endl;
        std::cout << "step_size=" << tracking_thread.param.step_size << std::endl;
        std::cout << "smoothing=" << tracking_thread.param.smooth_fraction << std::endl;
        std::cout << "min_length=" << tracking_thread.param.min_length << std::endl;
        std::cout << "max_length=" << tracking_thread.param.max_length << std::endl;
        std::cout << "tracking_method=" << (int)tracking_thread.param.tracking_method << std::endl;
        std::cout << "initial direction=" << (int)tracking_thread.param.initial_direction << std::endl;
        std::cout << "interpolation=" << (int)tracking_thread.param.interpolation_strategy << std::endl;
        std::cout << "voxelwise=" << (int)tracking_thread.param.center_seed << std::endl;
        std::cout << "default_otsu=" << tracking_thread.param.default_otsu << std::endl;
        std::cout << (tracking_thread.param.stop_by_tract ? "fiber_count=" : "seed_count=") <<
                tracking_thread.param.termination_count << std::endl;
        std::cout << "thread_count=" << po.get("thread_count",int(std::thread::hardware_concurrency())) << std::endl;
    }

    if (!po.has("seed"))
    {
        float seed_threshold = tracking_thread.param.threshold;
        if(seed_threshold == 0)
            seed_threshold = otsu*tracking_thread.param.default_otsu;
        std::vector<image::vector<3,short> > seed;
        std::cout << "no seeding area assigned. use whole brain seeding" << std::endl;
        for(image::pixel_index<3> index(geometry);index < geometry.size();++index)
            if(fa0[index.index()] > seed_threshold)
                seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
        tracking_thread.roi_mgr.setRegions(geometry,seed,1.0,3,"whole brain",image::vector<3>());
    }

    if(!cnt_file_name.empty())
    {
        QStringList connectometry_threshold;
        if(!po.has("connectometry_threshold"))
        {
            std::cout << "Please assign the connectometry threshold." << std::endl;
            return -1;
        }
        connectometry_threshold = QString(po.get("connectometry_threshold").c_str()).split(",");
        for(unsigned int i = 0;i < cnt_file_name.size();++i)
        {
            connectometry_result cnt;
            std::cout << "loading individual file:" << cnt_file_name[i].toStdString() << std::endl;
            if(cnt_type == "iva" && !cnt.individual_vs_atlas(handle,cnt_file_name[i].toLocal8Bit().begin(),0))
            {
                std::cout << "Error loading connectomnetry file:" << cnt.error_msg <<std::endl;
                return -1;
            }
            if(cnt_type == "ivp" && !cnt.individual_vs_db(handle,cnt_file_name[i].toLocal8Bit().begin()))
            {
                std::cout << "Error loading connectomnetry file:" << cnt.error_msg <<std::endl;
                return -1;
            }
            if(cnt_type == "ivi")
            {
                std::cout << "loading individual file:" << cnt_file_name[i+1].toStdString() << std::endl;
                if(!cnt.individual_vs_individual(handle,cnt_file_name[i].toLocal8Bit().begin(),
                                                              cnt_file_name[i+1].toLocal8Bit().begin(),0))
                {
                    std::cout << "Error loading connectomnetry file:" << cnt.error_msg <<std::endl;
                    return -1;
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
                    std::cout << "Cannot save file to " << out.str()
                              << ". Please check write permission, directory, and disk space." << std::endl;
                    return 0;
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
    if(tract_model.get_visible_track_count() == 0)
    {
        std::cout << "No tract generated. Terminating..." << std::endl;
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
    return trk_post(handle,tract_model,file_name);

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
