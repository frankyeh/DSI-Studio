#include <QFileInfo>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include <boost/exception/diagnostic_information.hpp>
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include "SliceModel.h"
bool atl_load_atlas(const std::string atlas_name);
bool atl_get_mapping(gz_mat_read& mat_reader,
                     unsigned int factor,
                     unsigned int thread_count,
                     image::basic_image<image::vector<3>,3>& mapping);
extern fa_template fa_template_imp;
extern std::vector<atlas> atlas_list;
namespace po = boost::program_options;

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
    file_name_stat += connectivity_roi;
    file_name_stat += ".";
    file_name_stat += connectivity_value;
    file_name_stat += use_end_only ? ".end":".pass";
    file_name_stat += ".connectivity.mat";
    std::cout << "export connectivity matrix to " << file_name_stat << std::endl;
    data.save_to_file(file_name_stat.c_str());
}

void get_connectivity_matrix(FibData* handle,
                             TractModel& tract_model,
                             image::basic_image<image::vector<3>,3>& mapping,
                             po::variables_map& vm)
{
    std::string source = vm["source"].as<std::string>();
    QStringList connectivity_list = QString(vm["connectivity"].as<std::string>().c_str()).split(",");
    QStringList connectivity_type_list = QString( vm["connectivity_type"].as<std::string>().c_str()).split(",");
    QStringList connectivity_value_list = QString(vm["connectivity_value"].as<std::string>().c_str()).split(",");
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
            if(mapping.empty() && !atl_get_mapping(handle->mat_reader,1/*7-9-7*/,vm["thread_count"].as<int>(),mapping))
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
            for (image::pixel_index<3>index; index.is_valid(from.geometry());index.next(from.geometry()))
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
                    std::ostringstream out;
                    out << "region" << value;
                    data.regions.push_back(cur_region);
                    data.region_name.push_back(out.str());
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

int trk(int ac, char *av[])
{
    try{
    // options for fiber tracking
    po::options_description trk_desc("fiber tracking options");
    trk_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "rec:diffusion reconstruction trk:fiber tracking")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("method", po::value<int>()->default_value(0), "tracking methods (0:streamline, 1:rk4)")
    ("initial_dir", po::value<int>()->default_value(0), "initial direction (0:primary, 1:random 2:all directions)")
    ("interpolation", po::value<int>()->default_value(0), "interpolation methods (0:trilinear, 1:gaussian radial)")
    ("seed_plan", po::value<int>()->default_value(0), "seeding methods (0:subvoxel, 1:voxelwise)")
    ("thread_count", po::value<int>()->default_value(1), "number of thread (default:1)")
    ("output", po::value<std::string>(), "output file name")
    ("end_point", po::value<std::string>(), "output end point file")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=stat,tdi)")
    ("connectivity", po::value<std::string>(), "export connectivity")
    ("connectivity_type", po::value<std::string>()->default_value("end"), "specify connectivity parameter")
    ("connectivity_value", po::value<std::string>()->default_value("count"), "specify connectivity parameter")
    ("roi", po::value<std::string>(), "file for ROI regions")
    ("roi2", po::value<std::string>(), "file for the second ROI regions")
    ("roi3", po::value<std::string>(), "file for the third ROI regions")
    ("roi4", po::value<std::string>(), "file for the forth ROI regions")
    ("roi5", po::value<std::string>(), "file for the fifth ROI regions")
    ("roa", po::value<std::string>(), "file for ROA regions")
    ("roa2", po::value<std::string>(), "file for ROA regions")
    ("roa3", po::value<std::string>(), "file for ROA regions")
    ("roa4", po::value<std::string>(), "file for ROA regions")
    ("roa5", po::value<std::string>(), "file for ROA regions")
    ("end", po::value<std::string>(), "file for ending regions")
    ("end2", po::value<std::string>(), "file for ending regions")
    ("ter", po::value<std::string>(), "file for terminative regions")
    ("seed", po::value<std::string>(), "file for seed regions")
    ("ref", po::value<std::string>(), "T1W or T2W file for exporting coordinate")
    ("threshold_index", po::value<std::string>(), "index for thresholding")
    ("step_size", po::value<float>(), "the step size in minimeter")
    ("turning_angle", po::value<float>()->default_value(60), "the turning angle in degrees (default:60)")
    ("fa_threshold", po::value<float>(), "the fa threshold (default:0.03)")
    ("smoothing", po::value<float>()->default_value(0), "smoothing fiber tracts, from 0 to 1. (default:0)")
    ("min_length", po::value<float>()->default_value(10), "minimum fiber length in minimeter (default:10)")
    ("max_length", po::value<float>()->default_value(500), "maximum fiber length in minimeter (default:500)")
    ("random_seed", po::value<int>()->default_value(0), "use timer as the random seed")
    ("fiber_count", po::value<int>(), "terminate tracking if fiber count is reached (default:10000)")
    ("seed_count", po::value<int>(), "terminate tracking if seeding count is reached  (default:10000)")
    ;

    if(!ac)
    {
        std::cout << trk_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(trk_desc).run(), vm);
    po::notify(vm);

    std::auto_ptr<FibData> handle(new FibData);
    {
        std::string file_name = vm["source"].as<std::string>();
        std::cout << "loading " << file_name << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!handle->load_from_file(file_name.c_str()))
        {
            std::cout << "Open file " << file_name << " failed" << std::endl;
            std::cout << "msg:" << handle->error_msg << std::endl;
            return 0;
        }
    }
    if (vm.count("threshold_index"))
    {
        std::cout << "setting index to " << vm["threshold_index"].as<std::string>() << std::endl;
        if(!handle->fib.set_tracking_index(vm["threshold_index"].as<std::string>()))
        {
            std::cout << "failed...cannot find the index" << std::endl;
            return 0;
        }
    }




    image::geometry<3> geometry = handle->dim;
    image::vector<3> voxel_size = handle->vs;
    image::basic_image<image::vector<3>,3> mapping;
    const float *fa0 = handle->fib.fa[0];


    ThreadData tracking_thread(vm["random_seed"].as<int>());
    tracking_thread.param.step_size = (vm.count("step_size") ? vm["step_size"].as<float>(): voxel_size[0]/2.0);
    tracking_thread.param.smooth_fraction = vm["smoothing"].as<float>();
    tracking_thread.param.min_points_count3 = 3.0* vm["min_length"].as<float>()/tracking_thread.param.step_size;
    if(tracking_thread.param.min_points_count3 < 6)
        tracking_thread.param.min_points_count3 = 6;
    tracking_thread.param.max_points_count3 = std::max<unsigned int>(6,3.0*vm["max_length"].as<float>()/tracking_thread.param.step_size);

    tracking_thread.tracking_method = vm["method"].as<int>();
    tracking_thread.initial_direction  = vm["initial_dir"].as<int>();
    tracking_thread.interpolation_strategy = vm["interpolation"].as<int>();
    tracking_thread.center_seed = vm["seed_plan"].as<int>();


    unsigned int termination_count = 10000;
    if (vm.count("fiber_count"))
    {
        termination_count = vm["fiber_count"].as<int>();
        tracking_thread.stop_by_tract = true;

        if (vm.count("seed_count"))
            tracking_thread.max_seed_count = vm["seed_count"].as<int>();
    }
    else
    {
        if (vm.count("seed_count"))
            termination_count = vm["seed_count"].as<int>();
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
    if (vm.count(roi_names[index]))
    {
        ROIRegion roi(geometry, voxel_size);
        std::string file_name = vm[roi_names[index]].as<std::string>();
        if(file_name.find(':') != std::string::npos &&
           file_name.find(':') != 1)
        {
            std::string atlas_name = file_name.substr(0,file_name.find(':'));
            std::string region_name = file_name.substr(file_name.find(':')+1);
            std::cout << "Loading " << region_name << " from " << atlas_name << " atlas" << std::endl;
            if(!atl_load_atlas(atlas_name))
                return 0;
            if(mapping.empty() && !atl_get_mapping(handle->mat_reader,1/*7-9-7*/,1/*thread_count*/,mapping))
                return 0;
            image::vector<3> null;
            std::vector<image::vector<3,short> > cur_region;
            for(unsigned int i = 0;i < atlas_list.size();++i)
                if(atlas_list[i].name == atlas_name)
                    for (unsigned int label_index = 0; label_index < atlas_list[i].get_list().size(); ++label_index)
                        if(atlas_list[i].get_list()[label_index] == region_name)
                    {
                        for (image::pixel_index<3>index; index.is_valid(mapping.geometry());index.next(mapping.geometry()))
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
        tracking_thread.setRegions(geometry,roi.get(),type[index]);
        std::cout << roi_names[index] << "=" << file_name << std::endl;
    }

    TractModel tract_model(handle.get());

    if (vm.count("fa_threshold") )
        tract_model.get_fib().threshold = vm["fa_threshold"].as<float>();
    else
        tract_model.get_fib().threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(geometry,fa0));
    tract_model.get_fib().cull_cos_angle = std::cos(vm["turning_angle"].as<float>()*3.1415926/180.0);

    if (!vm.count("seed"))
    {

        std::vector<image::vector<3,short> > seed;
        std::cout << "no seeding area assigned. use whole brain seeding" << std::endl;
        for(image::pixel_index<3> index;index.is_valid(geometry);index.next(geometry))
            if(fa0[index.index()] > tract_model.get_fib().threshold)
                seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
        tracking_thread.setRegions(geometry,seed,3);
    }

    {
        std::cout << "turning_angle=" << vm["turning_angle"].as<float>() << std::endl;
        std::cout << "fa_threshold=" << tract_model.get_fib().threshold << std::endl;
        std::cout << "step_size=" << tracking_thread.param.step_size << std::endl;
        std::cout << "smoothing=" << tracking_thread.param.smooth_fraction << std::endl;
        std::cout << "min_length=" << vm["min_length"].as<float>() << std::endl;
        std::cout << "max_length=" << vm["max_length"].as<float>() << std::endl;
        std::cout << "tracking_method=" << (int)tracking_thread.tracking_method << std::endl;
        std::cout << "initial direction=" << (int)tracking_thread.initial_direction << std::endl;
        std::cout << "interpolation=" << (int)tracking_thread.interpolation_strategy << std::endl;
        std::cout << "voxelwise=" << (int)tracking_thread.center_seed << std::endl;
        std::cout << "thread_count=" << vm["thread_count"].as<int>() << std::endl;
    }

    std::cout << "start tracking." << std::endl;

    tracking_thread.run(tract_model.get_fib(),vm["thread_count"].as<int>(),termination_count,true);

    tracking_thread.fetchTracks(&tract_model);

    std::cout << "finished tracking." << std::endl;

    if(tract_model.get_visible_track_count() == 0)
    {
        std::cout << "No tract generated. Terminating..." << std::endl;
        return 0;
    }

    std::string file_name;
    if (vm.count("output"))
        file_name = vm["output"].as<std::string>();
    else
    {
        std::ostringstream fout;
        fout << vm["source"].as<std::string>() <<
            ".st" << (int)std::floor(tracking_thread.param.step_size*10.0+0.5) <<
            ".tu" << (int)std::floor(vm["turning_angle"].as<float>()+0.5) <<
            ".fa" << (int)std::floor(tract_model.get_fib().threshold*100.0+0.5) <<
            ".sm" << (int)std::floor(tracking_thread.param.smooth_fraction*10.0+0.5) <<
            ".me" << (int)tracking_thread.tracking_method <<
            ".sd" << (int)tracking_thread.initial_direction <<
            ".pd" << (int)tracking_thread.interpolation_strategy <<
            ".trk";
        file_name = fout.str();
    }

    if(vm.count("ref")) // save track in T1W/T2W space
    {
        std::vector<std::string> files;
        files.push_back(vm["ref"].as<std::string>());
        FibSliceModel slice(handle.get());
        CustomSliceModel new_slice;
        std::cout << "Loading reference image:" << vm["ref"].as<std::string>() << std::endl;
        if(!new_slice.initialize(slice,!(handle->trans_to_mni.empty())/*is_qsdr*/,files,false))
        {
            std::cout << "Error reading ref image file:" << vm["ref"].as<std::string>() << std::endl;
            return 0;
        }
        new_slice.thread->join();
        new_slice.update();
        std::cout << "Applying linear registration." << std::endl;
        std::cout << new_slice.transform[0] << " " << new_slice.transform[1] << " " << new_slice.transform[2] << " " << new_slice.transform[3] << std::endl;
        std::cout << new_slice.transform[4] << " " << new_slice.transform[5] << " " << new_slice.transform[6] << " " << new_slice.transform[7] << std::endl;
        std::cout << new_slice.transform[8] << " " << new_slice.transform[9] << " " << new_slice.transform[10] << " " << new_slice.transform[11] << std::endl;
        tract_model.save_transformed_tracts_to_file(file_name.c_str(),&*new_slice.invT.begin(),false);
    }
    else
        tract_model.save_tracts_to_file(file_name.c_str());
    if(vm.count(("end_point")))
        tract_model.save_end_points(vm["end_point"].as<std::string>().c_str());

    std::cout << "a total of " << tract_model.get_visible_track_count() << " tracts are generated" << std::endl;
    std::cout << "output file:" << file_name << std::endl;

    if(vm.count("connectivity"))
        get_connectivity_matrix(handle.get(),tract_model,mapping,vm);

    if(vm.count("export"))
    {
        std::string export_option = vm["export"].as<std::string>();
        std::replace(export_option.begin(),export_option.end(),',',' ');
        std::istringstream in(export_option);
        std::string cmd;
        while(in >> cmd)
        {
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

            if(cmd == "fa" || cmd == "qa")
            {
                std::cout << "export..." << cmd << std::endl;
                tract_model.save_fa_to_file(file_name_stat.c_str());
                continue;
            }

            if(handle->get_name_index(cmd) != handle->view_item.size())
                tract_model.save_data_to_file(file_name_stat.c_str(),cmd);
            else
            {
                std::cout << "cannot find index name:" << cmd << std::endl;
                continue;
            }
        }
    }


    if (vm.count("endpoint"))
    {
        std::cout << "output endpoint." << std::endl;
        file_name += ".end.txt";
        tract_model.save_end_points(file_name.c_str());
    }

    }
    catch(boost::exception const&  ex)
    {
        std::cout << "program terminated due to exception:" <<
               boost::diagnostic_information(ex) << std::endl;
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
