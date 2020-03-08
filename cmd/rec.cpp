#include <QString>
#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "fib_data.hpp"
#include "tracking/region/Regions.h"
#include "tipl/tipl.hpp"
#include "libs/dsi/image_model.hpp"
#include "libs/gzip_interface.hpp"
#include "reconstruction/reconstruction_window.h"
#include "program_option.hpp"

extern std::vector<std::string> fa_template_list,iso_template_list;
void rec_motion_correction(ImageModel* handle);
void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_dsi(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
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
        std::cout << "load src file failed:" << handle->error_msg << std::endl;
        return 1;
    }
    std::cout << "src loaded" <<std::endl;
    if(po.has("other_src"))
    {
        std::string file_name2 = po.get("other_src");
        std::auto_ptr<ImageModel> handle2(new ImageModel);
        if (!handle2->load_from_file(file_name2.c_str()))
        {
            std::cout << "load other src file failed:" << handle2->error_msg << std::endl;
            return 1;
        }
        if(handle->voxel.dim != handle2->voxel.dim)
        {
            std::cout << "the image dimension is different." << std::endl;
            return 1;
        }
        handle->distortion_correction(*handle2.get());
        std::cout << "phase correction done with " << file_name2 << std::endl;
    }
    if (po.has("cmd"))
    {
        QStringList cmd_list = QString(po.get("cmd").c_str()).split("+");
        for(int i = 0;i < cmd_list.size();++i)
        {
            QStringList run_list = QString(cmd_list[i]).split("=");
            std::cout << "run " << cmd_list[i].toStdString() << std::endl;
            if(!handle->command(run_list[0].toStdString(),
                                run_list.count() > 1 ? run_list[1].toStdString():std::string()))
                return 1;
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
            std::cout << "invalid transfformation matrix." <<std::endl;
            return 1;
        }
        tipl::transformation_matrix<double> affine;
        affine.load_from_transform(T.begin());
        std::cout << "rotating images" << std::endl;
        handle->rotate(handle->voxel.dim,affine);
    }

    unsigned char method_index = uint8_t(po.get("method",4));
    std::cout << "method=" << int(method_index) << std::endl;

    if(method_index == 4)
        handle->voxel.param[0] = 1.25f;
    if(method_index == 6) // Convert to HARDI
    {
        handle->voxel.param[0] = 1.25f;
        handle->voxel.param[1] = 3000.0f;
        handle->voxel.param[2] = 0.05f;
    }
    if(method_index == 7) // QSDR
    {
        handle->voxel.param[0] = 1.25f;
        handle->voxel.primary_template = fa_template_list[size_t(po.get("template",0))];
        handle->voxel.secondary_template = iso_template_list[size_t(po.get("template",0))];
        std::cout << "template = " << handle->voxel.primary_template << std::endl;
        std::cout << "template2 = " << handle->voxel.secondary_template << std::endl;
    }
    if(po.has("study_src")) // DDI
    {
        handle->voxel.study_src_file_path = po.get("study_src");
        std::cout << "comparison src=" << handle->voxel.study_src_file_path << std::endl;
    }
    if (po.has("param0"))
    {
        handle->voxel.param[0] = po.get("param0",float(0));
        std::cout << "param0=" << handle->voxel.param[0] << std::endl;
    }
    if (po.has("param1"))
    {
        handle->voxel.param[1] = po.get("param1",float(0));
        std::cout << "param1=" << handle->voxel.param[1] << std::endl;
    }
    if (po.has("param2"))
    {
        handle->voxel.param[2] = po.get("param2",float(0));
        std::cout << "param2=" << handle->voxel.param[2] << std::endl;
    }
    if (po.has("param3"))
    {
        handle->voxel.param[3] = po.get("param3",float(0));
        std::cout << "param3=" << handle->voxel.param[3] << std::endl;
    }
    if (po.has("param4"))
    {
        handle->voxel.param[4] = po.get("param4",float(0));
        std::cout << "param4=" << handle->voxel.param[4] << std::endl;
    }

    handle->voxel.method_id = method_index;
    handle->voxel.ti.init(uint16_t(po.get("odf_order",int(8))));
    handle->voxel.odf_resolving = po.get("odf_resolving",int(0));
    handle->voxel.output_odf = po.get("record_odf",int(0));
    handle->voxel.check_btable = po.get("check_btable",int(handle->voxel.dim[2] < handle->voxel.dim[0]*2.0 ? 1:0));
    handle->voxel.output_tensor = po.get("output_tensor",int(0));
    handle->voxel.output_rdi = po.get("output_rdi",int(1)) && (method_index == 4 || method_index == 7);
    handle->voxel.max_fiber_number = uint32_t(po.get("num_fiber",int(5)));
    handle->voxel.r2_weighted = po.get("r2_weighted",int(0));
    handle->voxel.csf_calibration = po.get("csf_calibration",int(0)) && method_index == 4;
    handle->voxel.thread_count = po.get("thread_count",uint32_t(std::thread::hardware_concurrency()));



    if(handle->voxel.csf_calibration && !handle->is_human_data())
    {
        std::cout << "not a human brain data set. Disable CSF calibratoin" << std::endl;
        handle->voxel.csf_calibration = 0;
    }

    handle->voxel.half_sphere = po.get("half_sphere",handle->is_dsi_half_sphere() ? 1:0);
    handle->voxel.scheme_balance = po.get("scheme_balance",handle->need_scheme_balance() ? 1:0);


    {
        if(handle->voxel.output_odf)
            std::cout << "record ODF in the fib file" << std::endl;
        if(handle->voxel.r2_weighted && method_index == 4)
            std::cout << "r2 weighted is used for GQI" << std::endl;
    }

    if(po.has("other_image"))
    {
        QStringList file_list = QString(po.get("other_image").c_str()).split(":");
        for(int i = 0;i < file_list.size();++i)
        {
            QStringList name_value = file_list[i].split(",");
            if(name_value.size() != 2)
            {
                std::cout << "invalid command: " << file_list[i].toStdString() << std::endl;
                return 1;
            }
            if(!add_other_image(handle.get(),name_value[0],name_value[1]))
                return 1;
        }
    }
    if(po.has("mask"))
    {
        std::shared_ptr<fib_data> fib_handle(new fib_data);
        fib_handle->dim = handle->voxel.dim;
        fib_handle->vs = handle->voxel.vs;
        std::string mask_file = po.get("mask");

        if(mask_file == "1")
            std::fill(handle->voxel.mask.begin(),handle->voxel.mask.end(),1);
        else
        {
            gz_nifti nii;
            if(!nii.load_from_file(mask_file) || !nii.toLPS(handle->voxel.mask))
            {
                std::cout << "invalid or nonexisting nifti file:" << mask_file << std::endl;
                return 1;
            }
            if(handle->voxel.mask.geometry() != handle->voxel.dim)
            {
                std::cout << "The mask dimension is different. terminating..." << std::endl;
                return 1;
            }
        }
    }

    if(po.has("rotate_to"))
    {
        std::string file_name = po.get("rotate_to");
        gz_nifti in;
        if(!in.load_from_file(file_name.c_str()))
        {
            std::cout << "failed to read " << file_name << std::endl;
            return 0;
        }
        tipl::image<float,3> I;
        tipl::vector<3> vs;
        in.get_voxel_size(vs);
        in.toLPS(I);
        std::cout << "running rigid body transformation" << std::endl;
        tipl::transformation_matrix<double> T;
        bool terminated = false;
        tipl::reg::two_way_linear_mr(I,vs,handle->dwi_sum,handle->voxel.vs,
                       T,tipl::reg::rigid_body,tipl::reg::mutual_information(),
                        terminated,handle->voxel.thread_count);
        std::cout << "DWI rotated." << std::endl;
        handle->rotate(I.geometry(),T);
    }

    if(po.get("motion_correction",int(0)))
    {
        std::cout << "correct for motion and eddy current..." << std::endl;
        rec_motion_correction(handle.get());
        std::cout << "done." <<std::endl;
    }
    std::cout << "start reconstruction..." <<std::endl;
    const char* msg = handle->reconstruction();
    if (!msg)
        std::cout << "reconstruction finished." << std::endl;
    else
        std::cout << msg << std::endl;
    return 0;
}
