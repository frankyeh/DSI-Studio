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
bool atl_get_mapping(std::shared_ptr<fib_data> handle,
                     unsigned int factor,
                     unsigned int thread_count,
                     image::basic_image<image::vector<3>,3>& mapping);
void export_track_info(const std::string& file_name,
                       std::string export_option,
                       std::shared_ptr<fib_data> handle,
                       TractModel& tract_model);
extern fa_template fa_template_imp;
extern std::vector<atlas> atlas_list;

void save_connectivity_matrix(TractModel& tract_model,
                              ConnectivityMatrix& data,
                              const std::string& source,
                              const std::string& connectivity_roi,
                              const std::string& connectivity_value,
                              bool use_end_only)
{
    std::cout << "count tracks by " << (use_end_only ? "ending":"passing") << std::endl;
    std::cout << "calculate matrix using " << connectivity_value << std::endl;
    if(!data.calculate(tract_model,connectivity_value,use_end_only))
    {
        std::cout << "failed...invalid connectivity_value:" << connectivity_value;
        return;
    }
    std::string file_name_stat(source);
    file_name_stat += ".";
    file_name_stat += (QFileInfo(connectivity_roi.c_str()).exists()) ? QFileInfo(connectivity_roi.c_str()).baseName().toStdString():connectivity_roi;
    file_name_stat += ".";
    file_name_stat += connectivity_value;
    file_name_stat += use_end_only ? ".end":".pass";
    std::string network_measures(file_name_stat);
    file_name_stat += ".connectivity.mat";
    std::cout << "export connectivity matrix to " << file_name_stat << std::endl;
    data.save_to_file(file_name_stat.c_str());

    network_measures += ".network_measures.txt";
    std::cout << "export network measures to " << network_measures << std::endl;
    std::string report;
    data.network_property(report);
    std::ofstream out(network_measures.c_str());
    out << report;
}
void load_nii_label(const char* filename,std::map<short,std::string>& label_map);
void get_connectivity_matrix(std::shared_ptr<fib_data> handle,
                             TractModel& tract_model,
                             image::basic_image<image::vector<3>,3>& mapping)
{
    std::string source;
    QStringList connectivity_list = QString(po.get("connectivity").c_str()).split(",");
    QStringList connectivity_type_list = QString( po.get("connectivity_type","end").c_str()).split(",");
    QStringList connectivity_value_list = QString(po.get("connectivity_value","count").c_str()).split(",");
    if(po.has("output"))
        source = po.get("output");
    if(source.empty() || source == "no_file")
        source = po.get("source");
    for(unsigned int i = 0;i < connectivity_list.size();++i)
    {
        std::string roi_file_name = connectivity_list[i].toStdString();
        ConnectivityMatrix data;
        gz_nifti header;
        image::basic_image<unsigned int, 3> from;
        std::cout << "loading " << roi_file_name << std::endl;
        if (QFileInfo(roi_file_name.c_str()).exists() && header.load_from_file(roi_file_name))
            header.toLPS(from);
        if(from.geometry() != handle->dim)
        {
            std::cout << roi_file_name << " is used as an MNI space ROI." << std::endl;
            if(mapping.empty() && !atl_get_mapping(handle,1/*7-9-7*/,po.get("thread_count",int(std::thread::hardware_concurrency())),mapping))
                continue;
            atlas_list.clear(); // some atlas may be loaded in ROI
            if(atl_load_atlas(roi_file_name))
                data.set_atlas(atlas_list[0],mapping);
            else
                continue;
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

            std::map<short,std::string> label_map;
            QString label_file = QFileInfo(roi_file_name.c_str()).absolutePath()+"/"+QFileInfo(roi_file_name.c_str()).completeBaseName()+".txt";
            std::cout << "searching for roi label file:" << label_file.toStdString() << std::endl;
            if(QFileInfo(label_file).exists())
            {
                load_nii_label(label_file.toLocal8Bit().begin(),label_map);
                std::cout << "label file loaded." <<std::endl;
            }
            for(unsigned int value = 1;value < value_map.size();++value)
                if(value_map[value])
                {
                    image::basic_image<unsigned char,3> mask(from.geometry());
                    for(unsigned int i = 0;i < mask.size();++i)
                        if(from[i] == value)
                            mask[i] = 1;
                    ROIRegion region(handle->dim,handle->vs);
                    region.LoadFromBuffer(mask);
                    const std::vector<image::vector<3,short> >& cur_region = region.get();
                    image::vector<3,float> pos = std::accumulate(cur_region.begin(),cur_region.end(),image::vector<3,float>(0,0,0));
                    pos /= cur_region.size();
                    data.regions.push_back(cur_region);
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

        for(unsigned int j = 0;j < connectivity_type_list.size();++j)
        for(unsigned int k = 0;k < connectivity_value_list.size();++k)
            save_connectivity_matrix(tract_model,data,source,roi_file_name,connectivity_value_list[k].toStdString(),
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
    image::vector<3> voxel_size = handle->vs;
    image::basic_image<image::vector<3>,3> mapping;
    const float *fa0 = handle->dir.fa[0];


    ThreadData tracking_thread(po.get("random_seed",int(0)));
    tracking_thread.param.step_size = po.get("step_size",float(voxel_size[0]/2.0));
    tracking_thread.param.smooth_fraction = po.get("smoothing",float(0));
    tracking_thread.param.min_points_count3 = 3.0* po.get("min_length",float(10))/tracking_thread.param.step_size;
    if(tracking_thread.param.min_points_count3 < 6)
        tracking_thread.param.min_points_count3 = 6;
    tracking_thread.param.max_points_count3 = std::max<unsigned int>(6,3.0*po.get("max_length",float(500))/tracking_thread.param.step_size);

    tracking_thread.tracking_method = po.get("method",int(0));
    tracking_thread.initial_direction  = po.get("initial_dir",int(0));
    tracking_thread.interpolation_strategy = po.get("interpolation",int(0));
    tracking_thread.center_seed = po.get("seed_plan",int(0));


    unsigned int termination_count = 10000;
    if (po.has("fiber_count"))
    {
        termination_count = po.get("fiber_count",int(termination_count));
        tracking_thread.stop_by_tract = true;

        if (po.has("seed_count"))
            tracking_thread.max_seed_count = po.get("seed_count",int(termination_count));
    }
    else
    {
        if (po.has("seed_count"))
            termination_count = po.get("seed_count",int(termination_count));
        else
            termination_count = 1000000;
        tracking_thread.stop_by_tract = false;
    }
    std::cout << (tracking_thread.stop_by_tract ? "fiber_count=" : "seed_count=") <<
            termination_count << std::endl;



    const int total_count = 14;
    char roi_names[total_count][5] = {"roi","roi2","roi3","roi4","roi5","roa","roa2","roa3","roa4","roa5","end","end2","seed","ter"};
    unsigned char type[total_count] = {0,0,0,0,0,1,1,1,1,1,2,2,3,4};
    for(int index = 0;index < total_count;++index)
    if (po.has(roi_names[index]))
    {
        ROIRegion roi(geometry, voxel_size);
        std::string file_name = po.get(roi_names[index]);
        if(file_name.find(':') != std::string::npos &&
           file_name.find(':') != 1)
        {
            std::string atlas_name = file_name.substr(0,file_name.find(':'));
            std::string region_name = file_name.substr(file_name.find(':')+1);
            std::cout << "Loading " << region_name << " from " << atlas_name << " atlas" << std::endl;
            if(!atl_load_atlas(atlas_name))
                return 0;
            if(mapping.empty() && !atl_get_mapping(handle,1/*7-9-7*/,std::thread::hardware_concurrency(),mapping))
                return 0;
            image::vector<3> null;
            std::vector<image::vector<3,short> > cur_region;
            for(unsigned int i = 0;i < atlas_list.size();++i)
                if(atlas_list[i].name == atlas_name)
                    for (unsigned int label_index = 0; label_index < atlas_list[i].get_list().size(); ++label_index)
                        if(atlas_list[i].get_list()[label_index] == region_name)
                    {
                        for (image::pixel_index<3>index(mapping.geometry());index < mapping.size();++index)
                            if(mapping[index.index()] != null &&
                                atlas_list[i].label_matched(atlas_list[i].get_label_at(mapping[index.index()]),label_index))
                                cur_region.push_back(image::vector<3,short>(index.begin()));
                    }
            roi.add_points(cur_region,false);
        }
        else
        {
            if(!QFileInfo(file_name.c_str()).exists())
            {
                std::cout << file_name << " does not exist. terminating..." << std::endl;
                return 0;
            }
            if(!roi.LoadFromFile(file_name.c_str(),handle->trans_to_mni))
            {
                std::cout << "Invalid file format:" << file_name << std::endl;
                return 0;
            }
        }
        if(roi.get().empty())
        {
            std::cout << "No region found in " << file_name << std::endl;
            continue;
        }
        tracking_thread.setRegions(geometry,roi.get(),type[index],file_name.c_str());
        std::cout << roi_names[index] << "=" << file_name << std::endl;
    }

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



    tract_model.get_fib().threshold = po.get("fa_threshold",float(0.6*image::segmentation::otsu_threshold(image::make_image(geometry,fa0))));
    tract_model.get_fib().cull_cos_angle = std::cos(po.get("turning_angle",float(60))*3.1415926/180.0);

    if (!po.has("seed"))
    {

        std::vector<image::vector<3,short> > seed;
        std::cout << "no seeding area assigned. use whole brain seeding" << std::endl;
        for(image::pixel_index<3> index(geometry);index < geometry.size();++index)
            if(fa0[index.index()] > tract_model.get_fib().threshold)
                seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
        tracking_thread.setRegions(geometry,seed,3,"whole brain");
    }

    {
        std::cout << "turning_angle=" << po.get("turning_angle",float(60)) << std::endl;
        std::cout << "fa_threshold=" << tract_model.get_fib().threshold << std::endl;
        std::cout << "step_size=" << tracking_thread.param.step_size << std::endl;
        std::cout << "smoothing=" << tracking_thread.param.smooth_fraction << std::endl;
        std::cout << "min_length=" << po.get("min_length",float(10)) << std::endl;
        std::cout << "max_length=" << po.get("max_length",float(500)) << std::endl;
        std::cout << "tracking_method=" << (int)tracking_thread.tracking_method << std::endl;
        std::cout << "initial direction=" << (int)tracking_thread.initial_direction << std::endl;
        std::cout << "interpolation=" << (int)tracking_thread.interpolation_strategy << std::endl;
        std::cout << "voxelwise=" << (int)tracking_thread.center_seed << std::endl;
        std::cout << "thread_count=" << po.get("thread_count",int(std::thread::hardware_concurrency())) << std::endl;
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
            if(cnt_type == "iva" && !cnt.individual_vs_atlas(handle,cnt_file_name[i].toLocal8Bit().begin()))
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
                                                              cnt_file_name[i+1].toLocal8Bit().begin()))
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
                tract_model.get_fib().threshold = std::fabs(t);
                std::cout << "start tracking." << std::endl;

                tracking_thread.run(tract_model.get_fib(),po.get("thread_count",int(std::thread::hardware_concurrency())),termination_count,true);
                tracking_thread.fetchTracks(&tract_model);
                std::ostringstream out;
                out << cnt_file_name[i].toStdString() << "." << cnt_type.toStdString()
                        << ((t > 0) ? "inc":"dec") << std::fabs(t) << ".trk.gz" << std::endl;
                tract_model.save_tracts_to_file(out.str().c_str());
                std::vector<std::vector<float> > tmp;
                tract_model.release_tracts(tmp);
            }
        }
        return 0;
    }


    std::cout << "start tracking." << std::endl;

    tracking_thread.run(tract_model.get_fib(),po.get("thread_count",int(std::thread::hardware_concurrency())),termination_count,true);
    tract_model.report += tracking_thread.report.str();
    std::cout << tract_model.report << std::endl;

    tracking_thread.fetchTracks(&tract_model);
    std::cout << "finished tracking." << std::endl;

    if(tract_model.get_visible_track_count() == 0)
    {
        std::cout << "No tract generated. Terminating..." << std::endl;
        return 0;
    }

    std::string file_name;
    if (po.has("output"))
        file_name = po.get("output");
    else
    {
        std::ostringstream fout;
        fout << po.get("source") <<
            ".st" << (int)std::floor(tracking_thread.param.step_size*10.0+0.5) <<
            ".tu" << (int)std::floor(po.get("turning_angle",float(60))+0.5) <<
            ".fa" << (int)std::floor(tract_model.get_fib().threshold*100.0+0.5) <<
            ".sm" << (int)std::floor(tracking_thread.param.smooth_fraction*10.0+0.5) <<
            ".me" << (int)tracking_thread.tracking_method <<
            ".sd" << (int)tracking_thread.initial_direction <<
            ".pd" << (int)tracking_thread.interpolation_strategy <<
            ".trk";
        file_name = fout.str();
    }

    if(po.has("ref")) // save track in T1W/T2W space
    {
        std::vector<std::string> files;
        files.push_back(po.get("ref"));
        FibSliceModel slice(handle);
        CustomSliceModel new_slice;
        std::cout << "Loading reference image:" << po.get("ref") << std::endl;
        if(!new_slice.initialize(slice,!(handle->trans_to_mni.empty())/*is_qsdr*/,files,false))
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
        tract_model.save_transformed_tracts_to_file(file_name.c_str(),&*new_slice.invT.begin(),false);
    }
    else
        tract_model.save_tracts_to_file(file_name.c_str());
    if(po.has(("end_point")))
        tract_model.save_end_points(po.get("end_point").c_str());

    std::cout << "a total of " << tract_model.get_visible_track_count() << " tracts are generated" << std::endl;
    std::cout << "output file:" << file_name << std::endl;

    if(po.has("connectivity"))
        get_connectivity_matrix(handle,tract_model,mapping);

    if(po.has("export"))
        export_track_info(file_name,po.get("export"),handle,tract_model);


    if (po.has("endpoint"))
    {
        std::cout << "output endpoint." << std::endl;
        file_name += ".end.txt";
        tract_model.save_end_points(file_name.c_str());
    }

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
