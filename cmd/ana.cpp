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

int ana(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "ana: analysis")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("tract", po::value<std::string>(), "assign the .trk file name")
    ("roi", po::value<std::string>(), "file for ROI regions")
    ("end", po::value<std::string>(), "file for END regions")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=tdi)")
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
    if(vm.count("export") && vm["export"].as<std::string>() == std::string("connectivity"))
    {
        bool use_end_only = false;
        std::string roi_file_name;
        if (vm.count("roi"))
        {
            roi_file_name = vm["roi"].as<std::string>();
            use_end_only = false;
            std::cout << "roi=" << roi_file_name << std::endl;
        }
        if (vm.count("end"))
        {
            roi_file_name = vm["end"].as<std::string>();
            use_end_only = true;
            std::cout << "end=" << roi_file_name << std::endl;
        }

        if (roi_file_name.empty())
        {
            std::cout << "No ROI or END region defined for connectivity matrix." << std::endl;
            return 0;
        }
        gz_nifti header;
        std::cout << "loading " << roi_file_name << std::endl;
        if (!header.load_from_file(roi_file_name))
        {
            std::cout << "Cannot open nifti file " << roi_file_name << std::endl;
            return 0;
        }
        std::cout << "region loaded" << std::endl;
        image::basic_image<unsigned int, 3> from;
        header.toLPS(from);
        image::matrix<4,4,float> convert;
        bool has_convert = false;
        if(from.geometry() != geometry)
        {
            if(is_qsdr)
            {
                convert.identity();
                header.get_image_transformation(convert.begin());
                convert.inv();
                convert *= handle->trans_to_mni;
                has_convert = true;

            }
            else
            {
                std::cout << "The ROI needs to be in the diffusion space." << std::endl;
                return 0;
            }
        }
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
            return 0;
        }
        ConnectivityMatrix data;
        data.regions.clear();
        data.region_name.clear();
        for(unsigned int value = 1;value < value_map.size();++value)
            if(value_map[value])
            {
                image::basic_image<unsigned char,3> mask(from.geometry());
                for(unsigned int i = 0;i < mask.size();++i)
                    if(from[i] == value)
                        mask[i] = 1;
                ROIRegion region(geometry,handle->vs);
                if(has_convert)
                    region.LoadFromBuffer(mask,convert);
                else
                    region.LoadFromBuffer(mask);
                const std::vector<image::vector<3,short> >& cur_region = region.get();
                image::vector<3,float> pos = std::accumulate(cur_region.begin(),cur_region.end(),image::vector<3,float>(0,0,0));
                pos /= cur_region.size();
                std::ostringstream out;
                out << "region" << value;
                data.regions.push_back(cur_region);
                data.region_name.push_back(out.str());
            }
        std::cout << "total number of regions=" << data.regions.size() << std::endl;
        std::cout << "total number of tracts=" << tract_model.get_tracts().size() << std::endl;
        std::cout << "calculating connectivity matrix..." << std::endl;
        std::cout << "count tracks by " << (use_end_only ? "ending":"passing") << std::endl;
        QStringList value_list = QString(vm["connectivity_value"].as<std::string>().c_str()).split(",");
        for(unsigned int j = 0;j < value_list.size();++j)
        {
            std::cout << "calculate matrix using " << value_list[j].toStdString() << std::endl;
            if(!data.calculate(tract_model,value_list[j].toStdString(),use_end_only))
            {
                std::cout << "failed...invalid connectivity_value:" << value_list[j].toStdString();
                continue;
            }
            std::string file_name_stat(file_name);
            file_name_stat += ".";
            file_name_stat += QFileInfo(roi_file_name.c_str()).baseName().toStdString();
            file_name_stat += ".";
            file_name_stat += vm["connectivity_value"].as<std::string>();
            file_name_stat += use_end_only ? ".end":".pass";
            file_name_stat += ".connectivity.mat";
            std::cout << "export connectivity matrix to " << file_name_stat << std::endl;
            data.save_to_file(file_name_stat.c_str());
        }
        return 0;
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
