#include <QString>
#include <iostream>
#include <iterator>
#include <string>
#include "image/image.hpp"
#include "libs/dsi/image_model.hpp"
#include "dsi_interface_static_link.h"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "reconstruction/reconstruction_window.h"
#include "program_option.hpp"

extern fa_template fa_template_imp;
void rec_motion_correction(ImageModel* handle);
void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_multishell(const std::vector<unsigned int>& shell);

/**
 perform reconstruction
 */
int rec(void)
{
    std::string file_name = po.get("source");
    std::cout << "loading source..." <<std::endl;
    std::auto_ptr<ImageModel> handle(new ImageModel);
    if (!handle->load_from_file(file_name.c_str()))
    {
        std::cout << "Load src file failed:" << handle->error_msg << std::endl;
        return 1;
    }
    std::cout << "src loaded" <<std::endl;
    if (po.has("flip"))
    {
        std::string flip_seq = po.get("flip");
        for(unsigned int index = 0;index < flip_seq.length();++index)
            if(flip_seq[index] >= '0' && flip_seq[index] <= '5')
            {
                handle->flip(flip_seq[index]-'0');
                std::cout << "Flip image volume:" << (int)flip_seq[index]-'0' << std::endl;
            }
    }
    // apply affine transformation
    if (po.has("affine"))
    {
        std::cout << "reading transformation matrix" <<std::endl;
        std::ifstream in(po.get("affine").c_str());
        std::vector<double> T((std::istream_iterator<float>(in)),
                             (std::istream_iterator<float>()));
        if(T.size() != 12)
        {
            std::cout << "Invalid transfformation matrix." <<std::endl;
            return 1;
        }
        image::transformation_matrix<double> affine;
        affine.load_from_transform(T.begin());
        std::cout << "rotating images" << std::endl;
        handle->rotate(handle->voxel.dim,affine);
    }

    float param[4] = {0,0,0,0};
    int method_index = 0;


    method_index = po.get("method",int(0));
    std::cout << "method=" << method_index << std::endl;

    if(method_index == 0) // DSI
        param[0] = 17.0f;
    if(method_index == 2)
    {
        param[0] = 5.0f;
        param[1] = 15.0f;
    }
    if(method_index == 3) // QBI-SH
    {
        param[0] = 0.006f;
        param[1] = 8.0f;
    }
    if(method_index == 4)
        param[0] = 1.2f;
    if(method_index == 6) // Convert to HARDI
    {
        param[0] = 1.25f;
        param[1] = 3000.0f;
        param[2] = 0.05f;
    }
    if(method_index == 7)
    {
        param[0] = 1.2f;
        param[1] = 2.0f;
    }

    if (po.has("template"))
    {
        std::cout << "loading external template:" << po.get("template") << std::endl;
        fa_template_imp.template_file_name = po.get("template");
    }
    if(!fa_template_imp.load_from_file())
    {
        std::cout << fa_template_imp.error_msg << std::endl;
        return -1;
    }

    param[3] = 0.0002f;

    if(po.get("deconvolution",int(0)))
    {
        param[2] = 7.0f;
    }
    if(po.get("decomposition",int(0)))
    {
        param[3] = 0.05f;
        param[4] = 10.0f;
    }
    if (po.has("param0"))
    {
        param[0] = po.get("param0",float(0));
        std::cout << "param0=" << param[0] << std::endl;
    }
    if (po.has("param1"))
    {
        param[1] = po.get("param1",float(0));
        std::cout << "param1=" << param[1] << std::endl;
    }
    if (po.has("param2"))
    {
        param[2] = po.get("param2",float(0));
        std::cout << "param2=" << param[2] << std::endl;
    }
    if (po.has("param3"))
    {
        param[3] = po.get("param3",float(0));
        std::cout << "param3=" << param[3] << std::endl;
    }
    if (po.has("param4"))
    {
        param[4] = po.get("param4",float(0));
        std::cout << "param4=" << param[4] << std::endl;
    }

    handle->voxel.ti.init(po.get("odf_order",int(8)));
    handle->voxel.need_odf = po.get("record_odf",int(0));
    handle->voxel.output_jacobian = po.get("output_jac",int(0));
    handle->voxel.output_mapping = po.get("output_map",int(0));
    handle->voxel.output_diffusivity = po.get("output_dif",int(1));
    handle->voxel.output_tensor = po.get("output_tensor",int(0));
    handle->voxel.output_rdi = po.get("output_rdi",int(1));
    handle->voxel.odf_deconvolusion = po.get("deconvolution",int(0));
    handle->voxel.odf_decomposition = po.get("decomposition",int(0));
    handle->voxel.max_fiber_number = po.get("num_fiber",int(5));
    handle->voxel.r2_weighted = po.get("r2_weighted",int(0));
    handle->voxel.reg_method = po.get("reg_method",int(0));
    handle->voxel.csf_calibration = po.get("csf_calibration",int(0)) && method_index == 4;
    handle->voxel.thread_count = po.get("thread_count",int(std::thread::hardware_concurrency()));

    if(handle->voxel.csf_calibration && !handle->is_human_data())
    {
        std::cout << "Not a human brain data set. Disable CSF calibratoin" << std::endl;
        handle->voxel.csf_calibration = 0;
    }

    std::vector<unsigned int> shell;
    calculate_shell(handle->voxel.bvalues,shell);
    handle->voxel.half_sphere = po.get("half_sphere",is_dsi_half_sphere(shell) ? 1:0);
    handle->voxel.scheme_balance = po.get("scheme_balance",
                    (is_multishell(shell) && handle->voxel.bvalues.size()-shell.back() < 100) ? 1:0);


    {
        if(handle->voxel.need_odf)
            std::cout << "record ODF in the fib file" << std::endl;
        if(handle->voxel.odf_deconvolusion)
            std::cout << "apply deconvolution" << std::endl;
        if(handle->voxel.odf_decomposition)
            std::cout << "apply decomposition" << std::endl;
        if(handle->voxel.r2_weighted && method_index == 4)
            std::cout << "r2 weighted is used for GQI" << std::endl;
    }

    if(po.has("other_image"))
    {
        QStringList file_list = QString(po.get("other_image").c_str()).split(";");
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
    if(po.has("mask"))
    {
        std::string mask_file = po.get("mask");
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

    if(method_index == 7 && handle->voxel.reg_method == 4)
    {
        if(!po.has("t1w"))
        {
            std::cout << "Please assign --t1w with T1W file for CDM normalization" << std::endl;
            return 0;
        }
        handle->voxel.t1w_file_name = po.get("t1w");
    }

    if(po.get("motion_correction",int(0)))
    {
        std::cout << "correct for motion and eddy current..." << std::endl;
        rec_motion_correction(handle.get());
        std::cout << "Done." <<std::endl;
    }
    std::cout << "start reconstruction..." <<std::endl;
    const char* msg = reconstruction(handle.get(),method_index,
                                     param,po.get("check_btable",int(1)));
    if (!msg)
        std::cout << "Reconstruction finished." << std::endl;
    else
        std::cout << msg << std::endl;
    return 0;
}
