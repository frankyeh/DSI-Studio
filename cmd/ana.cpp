#include <QFileInfo>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"

namespace po = boost::program_options;

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000
void get_connectivity_matrix(FibData* handle,
                             TractModel& tract_model,
                             image::basic_image<image::vector<3>,3>& mapping,
                             po::variables_map& vm);
int ana(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "ana: analysis")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("tract", po::value<std::string>(), "assign the .trk file name")
    ("roi", po::value<std::string>(), "file for region-based analysis")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=tdi)")
    ("connectivity", po::value<std::string>(), "export connectivity")
    ("connectivity_type", po::value<std::string>()->default_value("end"), "specify connectivity parameter")
    ("connectivity_value", po::value<std::string>()->default_value("count"), "specify connectivity parameter")
    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).run(), vm);
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
            std::cout << "Cannot open file " << file_name.c_str() <<std::endl;
            return 0;
        }
    }
    bool is_qsdr = !handle->trans_to_mni.empty();
    image::geometry<3> geometry = handle->dim;

    if(!vm.count("tract"))
    {
        if(!vm.count("roi"))
        {
            std::cout << "No tract file or ROI file assigned." << std::endl;
            return 0;
        }
        std::string roi_file_name = vm["roi"].as<std::string>();
        ROIRegion region(handle->fib.dim,handle->vs);
        if(!region.LoadFromFile(roi_file_name.c_str(),handle->trans_to_mni))
        {
            std::cout << "Fail to load the ROI file." << std::endl;
            return 0;
        }
        if(vm.count("export") && vm["export"].as<std::string>() == std::string("stat"))
        {
            std::string file_name_stat(roi_file_name);
            file_name_stat += ".statistics.txt";
            std::cout << "export ROI statistics..." << std::endl;
            std::ofstream out(file_name_stat.c_str());
            std::vector<std::string> titles;
            region.get_quantitative_data_title(handle.get(),titles);
            std::vector<float> data;
            region.get_quantitative_data(handle.get(),data);
            for(unsigned int i = 0;i < titles.size() && i < data.size();++i)
                out << titles[i] << "\t" << data[i] << std::endl;
            return 0;
        }
        std::cout << "unknown export specification" << std::endl;
        return 0;
    }

    TractModel tract_model(handle.get());
    float threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(geometry,handle->fib.fa[0]));
    tract_model.get_fib().threshold = threshold;
    tract_model.get_fib().cull_cos_angle = std::cos(60.0*3.1415926/180.0);
    std::string file_name = vm["tract"].as<std::string>();
    {
        std::cout << "loading " << file_name << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            std::cout << file_name << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!tract_model.load_from_file(file_name.c_str()))
        {
            std::cout << "Cannot open file " << file_name << std::endl;
            return 0;
        }
        std::cout << file_name << " loaded" << std::endl;

    }
    if(vm.count("connectivity"))
    {
        image::basic_image<image::vector<3>,3> mapping;
        get_connectivity_matrix(handle.get(),tract_model,mapping,vm);
    }

    if(vm.count("export") && vm["export"].as<std::string>() == std::string("tdi"))
    {
        std::cout << "export tract density images..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi.nii.gz";
        tract_model.save_tdi(file_name_stat.c_str(),false,false,handle->trans_to_mni);
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("tdi2"))
    {
        std::cout << "export tract density images in subvoxel resolution..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi2.nii.gz";
        tract_model.save_tdi(file_name_stat.c_str(),true,false,handle->trans_to_mni);
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
        if(index_name != "qa" && index_name != "fa" &&  handle->get_name_index(index_name) == handle->view_item.size())
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
