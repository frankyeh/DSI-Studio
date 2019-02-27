#include <QString>
#include <QFileInfo>
#include <iostream>
#include <iterator>
#include <string>
#include "fib_data.hpp"
#include "tracking/region/Regions.h"
#include "tipl/tipl.hpp"
#include "libs/dsi/image_model.hpp"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "reconstruction/reconstruction_window.h"
#include "program_option.hpp"

extern fa_template fa_template_imp;
extern std::vector<std::string> fa_template_list;
void rec_motion_correction(ImageModel* handle);
void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_dsi(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
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
    if(po.has("other_src"))
    {
        std::string file_name2 = po.get("other_src");
        std::auto_ptr<ImageModel> handle2(new ImageModel);
        if (!handle2->load_from_file(file_name2.c_str()))
        {
            std::cout << "Load other src file failed:" << handle2->error_msg << std::endl;
            return 1;
        }

        if(handle->voxel.dim != handle2->voxel.dim)
        {
            std::cout << "The image dimension is different." << std::endl;
            return 1;
        }

        if(handle->src_dwi_data.size() != handle2->src_dwi_data.size())
        {
            std::cout << "The DWI number is different in other src." << std::endl;
            return 1;
        }

        handle->distortion_correction(*handle2.get());
        std::cout << "Phase correction done with " << file_name2 << std::endl;
    }


    if (po.has("flip"))
    {
        std::string flip_seq = po.get("flip");
        for(unsigned int index = 0;index < flip_seq.length();++index)
            if(flip_seq[index] >= '0' && flip_seq[index] <= '5')
            {
                handle->flip_dwi(flip_seq[index]-'0');
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
        tipl::transformation_matrix<double> affine;
        affine.load_from_transform(T.begin());
        std::cout << "rotating images" << std::endl;
        handle->rotate(handle->voxel.dim,affine);
    }

    int method_index = po.get("method",4);
    std::cout << "method=" << method_index << std::endl;

    if(method_index == 0) // DSI
        handle->voxel.param[0] = 17.0f;// Hamming filter
    if(method_index == 2)
    {
        handle->voxel.param[0] = 5.0f;
        handle->voxel.param[1] = 15.0f;
    }
    if(method_index == 3) // QBI-SH
    {
        handle->voxel.param[0] = 0.006f; // Regularization
        handle->voxel.param[1] = 8.0f; // SH order
    }
    if(method_index == 4)
        handle->voxel.param[0] = 1.25f;
    if(method_index == 6) // Convert to HARDI
    {
        handle->voxel.param[0] = 1.25f;
        handle->voxel.param[1] = 3000.0f;
        handle->voxel.param[2] = 0.05f;
    }
    if(method_index == 7) // QSDR
        handle->voxel.param[0] = 1.25f;
    if(po.has("study_src")) // DDI
    {
        handle->voxel.study_src_file_path = po.get("study_src");
        std::cout << "Comparison src=" << handle->voxel.study_src_file_path << std::endl;
    }
    if(po.get("deconvolution",int(0)))
    {
        handle->voxel.param[2] = 7.0f;
    }
    if(po.get("decomposition",int(0)))
    {
        handle->voxel.param[3] = 0.05f;
        handle->voxel.param[4] = 10.0f;
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
    handle->voxel.ti.init(po.get("odf_order",int(8)));
    handle->voxel.need_odf = po.get("record_odf",int(0));
    handle->voxel.check_btable = po.get("check_btable",int(1));
    handle->voxel.output_jacobian = po.get("output_jac",int(0));
    handle->voxel.output_mapping = po.get("output_map",int(0));
    handle->voxel.output_diffusivity = po.get("output_dif",int(1));
    handle->voxel.output_tensor = po.get("output_tensor",int(0));
    handle->voxel.output_rdi = po.get("output_rdi",int(1)) && (method_index == 4 || method_index == 7);
    handle->voxel.odf_deconvolusion = po.get("deconvolution",int(0));
    handle->voxel.odf_decomposition = po.get("decomposition",int(0));
    handle->voxel.max_fiber_number = po.get("num_fiber",int(5));
    handle->voxel.r2_weighted = po.get("r2_weighted",int(0));
    handle->voxel.reg_method = po.get("reg_method",int(0));
    handle->voxel.csf_calibration = po.get("csf_calibration",int(0)) && method_index == 4;
    handle->voxel.thread_count = po.get("thread_count",int(std::thread::hardware_concurrency()));


    if (po.has("template"))
    {
        std::cout << "external template = " << po.get("template") << std::endl;
        std::string template_file_name = po.get("template");
        for(int i = 0;i < fa_template_list.size();++i)
            if(QFileInfo(fa_template_list[i].c_str()).baseName() ==
                    QFileInfo(template_file_name.c_str()).baseName())
                template_file_name = fa_template_list[i];
        if(!QFileInfo(template_file_name.c_str()).exists())
        {
            std::cout << "template does not exist." << std::endl;
            return 0;
        }
        handle->voxel.external_template = template_file_name;
    }
    if(!fa_template_imp.load_from_file())
    {
        std::cout << fa_template_imp.error_msg << std::endl;
        return -1;
    }

    if(handle->voxel.csf_calibration && !handle->is_human_data())
    {
        std::cout << "Not a human brain data set. Disable CSF calibratoin" << std::endl;
        handle->voxel.csf_calibration = 0;
    }

    handle->voxel.half_sphere = po.get("half_sphere",handle->is_dsi_half_sphere() ? 1:0);
    handle->voxel.scheme_balance = po.get("scheme_balance",handle->need_scheme_balance() ? 1:0);


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
        QStringList file_list = QString(po.get("other_image").c_str()).split(":");
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
        std::shared_ptr<fib_data> fib_handle(new fib_data);
        fib_handle->dim = handle->voxel.dim;
        fib_handle->vs = handle->voxel.vs;
        std::string mask_file = po.get("mask");

        if(mask_file == "1")
            std::fill(handle->voxel.mask.begin(),handle->voxel.mask.end(),1);
        else
        {
            ROIRegion roi(fib_handle);
            std::cout << "reading mask..." << mask_file << std::endl;
            if(!load_region(fib_handle,roi,mask_file))
                return 0;
            tipl::image<unsigned char,3> external_mask;
            roi.SaveToBuffer(external_mask);
            if(external_mask.geometry() != handle->voxel.dim)
                std::cout << "In consistent the mask dimension...using default mask" << std::endl;
            else
                handle->voxel.mask = external_mask;
        }
    }

    if(method_index == 7 && handle->voxel.reg_method == 4)
    {
        if(!po.has("t1w"))
        {
            std::cout << "Please assign --t1w with T1W file for CDM normalization or assign --reg_method=3 to use CDM without T1W" << std::endl;
            return 0;
        }
        handle->voxel.t1w_file_name = po.get("t1w");
    }

    if(po.has("rotate_to"))
    {
        std::string file_name = po.get("rotate_to");
        gz_nifti in;
        if(!in.load_from_file(file_name.c_str()))
        {
            std::cout << "Failed to read " << file_name << std::endl;
            return 0;
        }
        tipl::image<float,3> I;
        tipl::vector<3> vs;
        in.get_voxel_size(vs);
        in.toLPS(I);
        std::cout << "Running rigid body transformation" << std::endl;
        tipl::transformation_matrix<double> T;
        bool terminated = false;
        tipl::reg::two_way_linear_mr(I,vs,handle->dwi_sum,handle->voxel.vs,
                       T,tipl::reg::rigid_body,tipl::reg::mutual_information(),
                        terminated,handle->voxel.thread_count);
        std::cout << "DWI rotated." << std::endl;
        handle->rotate(I,T);
    }

    if(po.get("motion_correction",int(0)))
    {
        std::cout << "correct for motion and eddy current..." << std::endl;
        rec_motion_correction(handle.get());
        std::cout << "Done." <<std::endl;
    }
    std::cout << "start reconstruction..." <<std::endl;
    const char* msg = handle->reconstruction();
    if (!msg)
        std::cout << "Reconstruction finished." << std::endl;
    else
        std::cout << msg << std::endl;
    return 0;
}
