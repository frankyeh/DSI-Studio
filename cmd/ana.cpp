#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "tracking_static_link.h"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "libs/gzip_interface.hpp"

namespace po = boost::program_options;

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

int ana(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "ana: analysis")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("tract", po::value<std::string>(), "assign the .trk file name")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=tdi)")
    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).allow_unregistered().run(), vm);
    po::notify(vm);

    std::auto_ptr<ODFModel> handle(new ODFModel);
    {
        std::string file_name = vm["source"].as<std::string>();
        std::cout << "loading " << file_name.c_str() << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name.c_str() << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!handle->load_from_file(file_name.c_str()))
        {
            std::cout << "Cannot open file " << file_name.c_str() <<std::endl;
            return 0;
        }
    }
    image::geometry<3> geometry = handle->fib_data.dim;
    image::vector<3> voxel_size = handle->fib_data.vs;

    TractModel tract_model(handle.get(),geometry,voxel_size);
    float threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(geometry,handle->fib_data.fib.fa[0]));
    tract_model.get_fib().threshold = threshold;
    tract_model.get_fib().cull_cos_angle = std::cos(60.0*3.1415926/180.0);

    std::string file_name = vm["tract"].as<std::string>();
    {
        std::cout << "loading " << file_name.c_str() << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name.c_str() << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!tract_model.load_from_file(file_name.c_str()))
        {
            std::cout << "Cannot open file " << file_name.c_str() << std::endl;
            return 0;
        }
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("tdi"))
    {
        std::cout << "export tract density images..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi.nii.gz";
        tract_model.save_tdi(file_name_stat.c_str(),false,false);
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("tdi2"))
    {
        std::cout << "export tract density images in subvoxel resolution..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi2.nii.gz";
        tract_model.save_tdi(file_name_stat.c_str(),true,false);
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("stat"))
    {
        std::string file_name_stat(file_name);
        file_name_stat += ".statistics.txt";
        std::cout << "export statistics..." << std::endl;
        std::ofstream out_stat(file_name_stat.c_str());
        std::string result;
        tract_model.get_quantitative_info(result);
        out_stat << result;
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>().find("report") == 0)
    {
        std::string report_cmd = vm["export"].as<std::string>();
        std::replace(report_cmd.begin(),report_cmd.end(),',',' ');
        std::istringstream in(report_cmd);
        std::string report_tag,index_name;
        int profile_dir = 0,bandwidth = 0;
        in >> report_tag >> index_name >> profile_dir >> bandwidth;
        std::vector<float> values,data_profile;
        // check index
        if(index_name != "qa" && index_name != "fa" &&  handle->get_name_index(index_name) == handle->fib_data.view_item.size())
        {
            std::cout << "cannot find index name:" << index_name << std::endl;
            return 0;
        }
        if(bandwidth == 0)
        {
            std::cout << "please specify bandwidth value" << std::endl;
            return 0;
        }
        if(profile_dir > 4)
        {
            std::cout << "please specify a valid profile type" << std::endl;
            return 0;
        }
        std::cout << "calculating report" << std::endl;
        tract_model.get_report(
                            profile_dir,
                            bandwidth,
                            index_name,
                            values,data_profile);

        std::replace(report_cmd.begin(),report_cmd.end(),' ','.');
        std::string file_name_stat(file_name);
        file_name_stat += ".";
        file_name_stat += report_cmd;
        file_name_stat += ".txt";
        std::cout << "output report:" << file_name_stat << std::endl;
        std::ofstream report(file_name_stat.c_str());
        report << "position\t";
        std::copy(values.begin(),values.end(),std::ostream_iterator<float>(report,"\t"));
        report << std::endl;
        report << "value";
        std::copy(data_profile.begin(),data_profile.end(),std::ostream_iterator<float>(report,"\t"));
        report << std::endl;
        return 0;
    }
    std::cout << "unknown export specification" << std::endl;
    return 0;
}
