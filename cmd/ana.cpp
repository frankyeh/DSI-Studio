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
    std::ofstream out("log.txt");
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
        out << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).allow_unregistered().run(), vm);
    po::notify(vm);

    std::auto_ptr<ODFModel> handle(new ODFModel);
    {
        std::string file_name = vm["source"].as<std::string>();
        out << "loading " << file_name.c_str() << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            out << file_name.c_str() << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!handle->load_from_file(file_name.c_str()))
        {
            out << "Cannot open file " << file_name.c_str() <<std::endl;
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
        out << "loading " << file_name.c_str() << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            out << file_name.c_str() << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!tract_model.load_from_file(file_name.c_str()))
        {
            out << "Cannot open file " << file_name.c_str() << std::endl;
            return 0;
        }
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("tdi"))
    {
        out << "export tract density images..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi.nii.gz";
        image::basic_image<unsigned int,3> tdi(geometry);
        std::vector<float> tr(16);
        tr[0] = tr[5] = tr[10] = tr[15] = 1.0;
        tract_model.get_density_map(tdi,tr,false);

        gz_nifti nii_header;
        image::flip_xy(tdi);
        nii_header << tdi;
        nii_header.set_voxel_size(voxel_size.begin());
        nii_header.save_to_file(file_name_stat.c_str());
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("statistics"))
    {
        out << "export statistics..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".statistics.txt";
        std::ofstream out(file_name_stat.c_str());
        std::vector<std::string> titles;
        titles.push_back("number of tracts");
        titles.push_back("tract length mean(mm)");
        titles.push_back("tract length sd(mm)");
        titles.push_back("tracts volume (mm^3)");
        handle->get_index_titles(titles);
        std::vector<float> data;
        tract_model.get_quantitative_data(data);
        for(unsigned int i = 0;i < titles.size();++i)
        {
            out << titles[i] << "\t";
            if(i < data.size())
                out << data[i];
            out << "\t";
            out << std::endl;
        }
        return 0;
    }
    if(vm.count("export") && vm["export"].as<std::string>().find("report,") == 0)
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
            out << "cannot find index name:" << index_name << std::endl;
            return 0;
        }
        if(bandwidth == 0)
        {
            out << "please specify bandwidth value" << std::endl;
            return 0;
        }
        if(profile_dir > 4)
        {
            out << "please specify a valid profile type" << std::endl;
            return 0;
        }
        out << "calculating report" << std::endl;
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
        out << "output report:" << file_name_stat << std::endl;
        std::ofstream report(file_name_stat.c_str());
        report << "position\t";
        std::copy(values.begin(),values.end(),std::ostream_iterator<float>(report,"\t"));
        report << std::endl;
        report << "value";
        std::copy(data_profile.begin(),data_profile.end(),std::ostream_iterator<float>(report,"\t"));
        report << std::endl;
        return 0;
    }
    out << "unknown export specification" << std::endl;
    return 0;
}
