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
    ("roi", po::value<std::string>(), "file for ROI regions")
    ("end", po::value<std::string>(), "file for END regions")
    ("export", po::value<std::string>(), "export additional information (e.g. --export=tdi)")
    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).run(), vm);
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
            std::cout << "Cannot open file " << file_name.c_str() <<std::endl;
            return 0;
        }
    }
    bool is_qsdr = !handle->fib_data.trans_to_mni.empty();
    image::geometry<3> geometry = handle->fib_data.dim;
    TractModel tract_model(handle.get());
    float threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(geometry,handle->fib_data.fib.fa[0]));
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
        std::vector<float> convert;
        if(from.geometry() != geometry)
        {
            if(is_qsdr)
            {
                std::vector<float> t(header.get_transformation(),
                                     header.get_transformation()+12),inv_trans(16);
                convert.resize(16);
                t.resize(16);
                t[15] = 1.0;
                image::matrix::inverse(t.begin(),inv_trans.begin(),image::dim<4,4>());
                image::matrix::product(inv_trans.begin(),
                                       handle->fib_data.trans_to_mni.begin(),
                                       convert.begin(),image::dim<4,4>(),image::dim<4,4>());
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
        ConnectivityMatrix::region_table_type region_table;
        for(unsigned int value = 1;value < value_map.size();++value)
            if(value_map[value])
            {
                image::basic_image<unsigned char,3> mask(from.geometry());
                for(unsigned int i = 0;i < mask.size();++i)
                    if(from[i] == value)
                        mask[i] = 1;
                ROIRegion region(geometry,handle->fib_data.vs);
                if(convert.empty())
                    region.LoadFromBuffer(mask);
                else
                    region.LoadFromBuffer(mask,convert);
                const std::vector<image::vector<3,short> >& cur_region = region.get();
                image::vector<3,float> pos = std::accumulate(cur_region.begin(),cur_region.end(),image::vector<3,float>(0,0,0));
                pos /= cur_region.size();
                std::ostringstream out;
                out << "region" << value;
                region_table[pos[0] > (geometry[0] >> 1) ? pos[1]-geometry[1]:geometry[1]-pos[1]] =
                        std::make_pair(cur_region,out.str());
            }
        std::cout << "total number of regions=" << region_table.size() << std::endl;
        std::cout << "total number of tracts=" << tract_model.get_tracts().size() << std::endl;
        data.set_regions(region_table);
        std::cout << "calculating connectivity matrix..." << std::endl;
        std::cout << "Count tracks by " << (use_end_only ? "ending":"passing") << std::endl;
        data.calculate(tract_model,use_end_only);
        std::string file_name_stat(file_name);
        file_name_stat += ".connectivity.mat";
        std::cout << "export connectivity matrix to " << file_name_stat << std::endl;
        data.save_to_file(file_name_stat.c_str());
        return 0;
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
