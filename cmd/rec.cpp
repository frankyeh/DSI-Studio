#include <QString>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "libs/dsi/image_model.hpp"
#include "boost/program_options.hpp"
#include "dsi_interface_static_link.h"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "reconstruction/reconstruction_window.h"
#include "manual_alignment.h"
extern fa_template fa_template_imp;
namespace po = boost::program_options;
void rec_motion_correction(ImageModel* handle,unsigned int total_thread,
                           std::vector<image::affine_transform<float> >& args,
                           unsigned int& progress,
                           bool& terminated);

/**
 perform reconstruction
 */
int rec(int ac, char *av[])
{
    po::options_description rec_desc("reconstruction options");
    rec_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "rec:diffusion reconstruction trk:fiber tracking")
    ("source", po::value<std::string>(), "assign the .src file name")
    ("mask", po::value<std::string>(), "assign the mask file")
    ("template", po::value<std::string>(), "assign the template file")
    ("affine", po::value<std::string>(), "assign the rotation matrix")
    ("other_image", po::value<std::string>(), "assign other image volume to be warpped with DWI")
    ("method", po::value<int>(), "reconstruction methods (0:dsi, 1:dti, 2:qbi_frt, 3:qbi_sh, 4:gqi)")
    ("odf_order", po::value<int>()->default_value(8), "set odf dimensions (4:162 direcitons, 5:252 directions, 6:362 directions, 8:642 directions)")
    ("record_odf", po::value<int>()->default_value(0), "output odf information")
    ("output_jac", po::value<int>()->default_value(0), "output jacobian determinant")
    ("output_map", po::value<int>()->default_value(0), "output mapping")
    ("output_dif", po::value<int>()->default_value(1), "output diffusivity")
    ("output_tensor", po::value<int>()->default_value(0), "output tensor")
    ("output_rdi", po::value<int>()->default_value(1), "output rdi")
    ("thread_count", po::value<int>()->default_value(boost::thread::hardware_concurrency()), "set the multi-thread count --thread_count=2")
    ("num_fiber", po::value<int>()->default_value(5), "maximum fibers resolved per voxel, default=3")
    ("half_sphere", po::value<int>()->default_value(0), "specific whether half sphere is used")
    ("deconvolution", po::value<int>()->default_value(0), "apply deconvolution")
    ("decomposition", po::value<int>()->default_value(0), "apply decomposition")
    ("r2_weighted", po::value<int>()->default_value(0), "set the r2 weighted in GQI")
    ("reg_method", po::value<int>()->default_value(0), "set the registration method for QSDR")
    ("interpo_method", po::value<int>()->default_value(2), "set the interpolation method for QSDR")
    ("scheme_balance", po::value<int>()->default_value(0), "balance the diffusion sampling scheme")
    ("check_btable", po::value<int>()->default_value(1), "check b-table")
    ("motion_correction", po::value<int>()->default_value(0), "correct for motion")
    ("param0", po::value<float>(), "set parameters")
    ("param1", po::value<float>(), "set parameters")
    ("param2", po::value<float>(), "set parameters")
    ("param3", po::value<float>(), "set parameters")
    ("flip", po::value<std::string>(), "flip image volume")
    ;

    if(!ac)
    {
        std::cout << rec_desc << std::endl;
        return 1;
    }
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(rec_desc).run(), vm);
    po::notify(vm);

    std::string file_name = vm["source"].as<std::string>();
    std::cout << "loading source..." <<std::endl;
    std::auto_ptr<ImageModel> handle(new ImageModel);
    if (!handle->load_from_file(file_name.c_str()))
    {
        std::cout << "Load src file failed:" << handle->error_msg << std::endl;
        return 1;
    }
    std::cout << "src loaded" <<std::endl;
    if (vm.count("flip"))
    {
        std::string flip_seq = vm["flip"].as<std::string>();
        for(unsigned int index = 0;index < flip_seq.length();++index)
            if(flip_seq[index] >= '0' && flip_seq[index] <= '5')
            {
                handle->flip(flip_seq[index]-'0');
                std::cout << "Flip image volume:" << (int)flip_seq[index]-'0' << std::endl;
            }
    }
    // apply affine transformation
    if (vm.count("affine"))
    {
        std::cout << "reading transformation matrix" <<std::endl;
        std::ifstream in(vm["affine"].as<std::string>().c_str());
        std::vector<float> T((std::istream_iterator<float>(in)),
                             (std::istream_iterator<float>()));
        if(T.size() != 12)
        {
            std::cout << "Invalid transfformation matrix." <<std::endl;
            return 1;
        }
        image::transformation_matrix<float> affine;
        affine.load_from_transform(T.begin());
        std::cout << "rotating images" << std::endl;
        handle->rotate(handle->voxel.dim,affine);
    }

    float param[4] = {0,0,0,0};
    int method_index = 0;


    method_index = vm["method"].as<int>();
    std::cout << "method=" << method_index << std::endl;

    if(method_index == 0) // DSI
        param[0] = 17.0;
    if(method_index == 2)
    {
        param[0] = 5;
        param[1] = 15;
    }
    if(method_index == 3) // QBI-SH
    {
        param[0] = 0.006;
        param[1] = 8;
    }
    if(method_index == 4)
        param[0] = 1.2;
    if(method_index == 6) // Convert to HARDI
    {
        param[0] = 1.25;
        param[1] = 3000;
        param[2] = 0.05;
    }
    if(method_index == 7)
    {
        if (vm.count("template"))
            fa_template_imp.template_file_name = vm["template"].as<std::string>();

        if(!fa_template_imp.load_from_file())
            return -1;
        param[0] = 1.2;
        param[1] = 2.0;
        std::fill(handle->mask.begin(),handle->mask.end(),1.0);
    }
    param[3] = 0.0002;

    if(vm["deconvolution"].as<int>())
    {
        param[2] = 7;
    }
    if(vm["decomposition"].as<int>())
    {
        param[3] = 0.05;
        param[4] = 10;
    }
    if (vm.count("param0"))
    {
        param[0] = vm["param0"].as<float>();
        std::cout << "param0=" << param[0] << std::endl;
    }
    if (vm.count("param1"))
    {
        param[1] = vm["param1"].as<float>();
        std::cout << "param1=" << param[1] << std::endl;
    }
    if (vm.count("param2"))
    {
        param[2] = vm["param2"].as<float>();
        std::cout << "param2=" << param[2] << std::endl;
    }
    if (vm.count("param3"))
    {
        param[3] = vm["param3"].as<float>();
        std::cout << "param3=" << param[3] << std::endl;
    }
    if (vm.count("param4"))
    {
        param[4] = vm["param4"].as<float>();
        std::cout << "param4=" << param[4] << std::endl;
    }

    handle->voxel.ti.init(vm["odf_order"].as<int>());
    handle->voxel.need_odf = vm["record_odf"].as<int>();
    handle->voxel.output_jacobian = vm["output_jac"].as<int>();
    handle->voxel.output_mapping = vm["output_map"].as<int>();
    handle->voxel.output_diffusivity = vm["output_dif"].as<int>();
    handle->voxel.output_tensor = vm["output_tensor"].as<int>();
    handle->voxel.output_rdi = vm["output_rdi"].as<int>();
    handle->voxel.odf_deconvolusion = vm["deconvolution"].as<int>();
    handle->voxel.odf_decomposition = vm["decomposition"].as<int>();
    handle->voxel.max_fiber_number = vm["num_fiber"].as<int>();
    handle->voxel.r2_weighted = vm["r2_weighted"].as<int>();
    handle->voxel.reg_method = vm["reg_method"].as<int>();
    handle->voxel.interpo_method = vm["interpo_method"].as<int>();
    handle->voxel.scheme_balance = vm["scheme_balance"].as<int>();
    handle->voxel.half_sphere = vm["half_sphere"].as<int>();

    {
        std::cout << "odf_order=" << vm["odf_order"].as<int>() << std::endl;
        std::cout << "num_fiber=" << vm["num_fiber"].as<int>() << std::endl;
        if(handle->voxel.need_odf)
            std::cout << "record ODF in the fib file" << std::endl;
        if(handle->voxel.odf_deconvolusion)
            std::cout << "apply deconvolution" << std::endl;
        if(handle->voxel.odf_decomposition)
            std::cout << "apply decomposition" << std::endl;
        if(handle->voxel.r2_weighted && method_index == 4)
            std::cout << "r2 weighted is used for GQI" << std::endl;
    }

    if(vm.count("other_image"))
    {
        QStringList file_list = QString(vm["other_image"].as<std::string>().c_str()).split(";");
        for(unsigned int i = 0;i < file_list.size();++i)
        {
            QStringList name_value = file_list[i].split(",");
            if(name_value.size() != 2)
            {
                std::cout << "Invalid command: " << file_list[i].toStdString() << std::endl;
                return 0;
            }
            if(!add_other_image(handle.get(),name_value[0],name_value[1],true))
                return 0;
        }
    }
    if(vm.count("mask"))
    {
        std::string mask_file = vm["mask"].as<std::string>();
        std::cout << "reading mask..." << mask_file << std::endl;
        gz_nifti header;
        if(header.load_from_file(mask_file.c_str()))
        {
            image::basic_image<unsigned char,3> external_mask;
            header.toLPS(external_mask);
            if(external_mask.geometry() != handle->voxel.dim)
                std::cout << "In consistent the mask dimension...using default mask" << std::endl;
            else
                handle->mask = external_mask;
        }
        else
            std::cout << "fail reading the mask...using default mask" << std::endl;
    }

    if(vm["motion_correction"].as<int>())
    {
        std::vector<image::affine_transform<float> > arg;
        unsigned int progress = 0;
        bool terminated = false;
        std::cout << "correct for motion and eddy current..." << std::endl;
        rec_motion_correction(handle.get(),vm["thread_count"].as<int>(),
                arg,progress,terminated);
        std::cout << "Done." <<std::endl;
    }
    std::cout << "start reconstruction..." <<std::endl;
    const char* msg = reconstruction(handle.get(),method_index,
                                     param,vm["check_btable"].as<int>(),
                                     vm["thread_count"].as<int>());
    if (!msg)
        std::cout << "Reconstruction finished:" << msg << std::endl;
    return 0;
}
