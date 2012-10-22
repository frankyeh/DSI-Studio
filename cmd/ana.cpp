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
    ("export", po::value<std::string>(), "export additional information (e.g. --export=stat,tdi)")
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

    std::auto_ptr<TractModel> tract_model(new TractModel(handle.get(),geometry,voxel_size));
    std::string file_name = vm["tract"].as<std::string>();
    {
        out << "loading " << file_name.c_str() << "..." <<std::endl;
        if(!QFileInfo(file_name.c_str()).exists())
        {
            out << file_name.c_str() << " does not exist. terminating..." << std::endl;
            return 0;
        }
        if (!tract_model->load_from_file(file_name.c_str()))
        {
            out << "Cannot open file " << file_name.c_str() << std::endl;
            return 0;
        }
    }
    if(vm.count("export") && vm["export"].as<std::string>().find("tdi")!=std::string::npos)
    {
        out << "export tract density images..." << std::endl;
        std::string file_name_stat(file_name);
        file_name_stat += ".tdi.nii";
        image::basic_image<unsigned int,3> tdi(geometry);
        std::vector<float> tr(16);
        tr[0] = tr[5] = tr[10] = tr[15] = 1.0;
        tract_model->get_density_map(tdi,tr,false);

        image::io::nifti nii_header;
        image::flip_xy(tdi);
        nii_header << tdi;
        nii_header.set_voxel_size(voxel_size.begin());
        nii_header.save_to_file(file_name_stat.c_str());
    }
}
