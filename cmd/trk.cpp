#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include <boost/exception/diagnostic_information.hpp>
#include "tracking_static_link.h"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "libs/gzip_interface.hpp"

namespace po = boost::program_options;

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
    ("export", po::value<std::string>(), "export additional information (e.g. --export=stat,tdi)")
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
    ("seed", po::value<std::string>(), "file for seed regions")
    ("threshold_index", po::value<std::string>(), "index for thresholding")
    ("step_size", po::value<float>()->default_value(1), "the step size in minimeter (default:1)")
    ("turning_angle", po::value<float>()->default_value(60), "the turning angle in degrees (default:60)")
    ("fa_threshold", po::value<float>(), "the fa threshold (default:0.03)")
    ("smoothing", po::value<float>()->default_value(0), "smoothing fiber tracts, from 0 to 1. (default:0)")
    ("min_length", po::value<float>()->default_value(10), "minimum fiber length in minimeter (default:10)")
    ("max_length", po::value<float>()->default_value(500), "maximum fiber length in minimeter (default:500)")
    ("fiber_count", po::value<int>(), "terminate tracking if fiber count is reached (default:10000)")
    ("seed_count", po::value<int>(), "terminate tracking if seeding count is reached  (default:10000)")
    ;

    if(!ac)
    {
        std::cout << trk_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(trk_desc).allow_unregistered().run(), vm);
    po::notify(vm);

    std::auto_ptr<ODFModel> handle(new ODFModel);
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
            std::cout << "msg:" << handle->fib_data.error_msg << std::endl;
            return 0;
        }
    }
    if (vm.count("threshold_index"))
    {
        std::cout << "setting index to " << vm["threshold_index"].as<std::string>() << std::endl;
        if(!handle->fib_data.fib.set_tracking_index(vm["threshold_index"].as<std::string>()))
        {
            std::cout << "failed...cannot find the index" << std::endl;
            return 0;
        }
    }




    image::geometry<3> geometry = handle->fib_data.dim;
    image::vector<3> voxel_size = handle->fib_data.vs;
    const float *fa0 = handle->fib_data.fib.fa[0];


    ThreadData tracking_thread;
    tracking_thread.param.step_size = vm["step_size"].as<float>();
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
    }        
    if (vm.count("seed_count"))
    {
        termination_count = vm["seed_count"].as<int>();
        tracking_thread.stop_by_tract = false;
    }
    std::cout << (tracking_thread.stop_by_tract ? "fiber_count=" : "seed_count=") <<
            termination_count << std::endl;




    char roi_names[13][5] = {"roi","roi2","roi3","roi4","roi5","roa","roa2","roa3","roa4","roa5","end","end2","seed"};
    unsigned char type[13] = {0,0,0,0,0,1,1,1,1,1,2,2,3};
    for(int index = 0;index < 13;++index)
    if (vm.count(roi_names[index]))
    {
        ROIRegion roi(geometry, voxel_size);
        std::string file_name = vm[roi_names[index]].as<std::string>();
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if(!roi.LoadFromFile(file_name.c_str(),handle->fib_data.trans_to_mni))
        {
            std::cout << "Invalid file format:" << file_name << std::endl;
            return 0;    
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
        std::cout << "initial direction=" << (int)tracking_thread.tracking_method << std::endl;
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

    std::cout << "a total of " << tract_model.get_visible_track_count() << " tracts are generated" << std::endl;
    std::cout << "output file:" << file_name << std::endl;
    tract_model.save_tracts_to_file(file_name.c_str());

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
            if(cmd == "tdi")
            {
                file_name_stat += ".nii.gz";
                std::cout << "export TDI to " << file_name_stat << std::endl;
                tract_model.save_tdi(file_name_stat.c_str(),false,false);
                continue;
            }
            if(cmd == "tdi2")
            {
                file_name_stat += ".nii.gz";
                std::cout << "export subvoxel TDI to " << file_name_stat << std::endl;
                tract_model.save_tdi(file_name_stat.c_str(),true,false);
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
            if(handle->get_name_index(cmd) != handle->fib_data.view_item.size())
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

}
