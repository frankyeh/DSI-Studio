#include <QString>
#include <QFileInfo>
#include <QDir>
#include <iostream>
#include "mac_filesystem.hpp"
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
size_t match_template(float volume);
void rec_motion_correction(ImageModel* handle);
void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_dsi(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
bool get_src(std::string filename,ImageModel& src2,std::string& error_msg);
bool correct_phase_distortion(ImageModel& src)
{
    if(po.has("other_src"))
    {
        std::cout << "phase correction with " << po.get("other_src") << std::endl;
        if(!src.distortion_correction(po.get("other_src").c_str()))
        {
            std::cout << "ERROR:" << src.error_msg << std::endl;
            return false;
        }
    }
    return true;
}
/**
 perform reconstruction
 */
int rec(void)
{
    std::string file_name = po.get("source");
    std::cout << "loading source..." <<std::endl;
    ImageModel src;
    if (!src.load_from_file(file_name.c_str()))
    {
        std::cout << "load src file failed:" << src.error_msg << std::endl;
        return 1;
    }
    std::cout << "src loaded" <<std::endl;
    if(!correct_phase_distortion(src))
        return 1;
    if (po.has("cmd"))
    {
        QStringList cmd_list = QString(po.get("cmd").c_str()).split("+");
        for(int i = 0;i < cmd_list.size();++i)
        {
            QStringList run_list = QString(cmd_list[i]).split("=");
            std::cout << "run " << cmd_list[i].toStdString() << std::endl;
            if(!src.command(run_list[0].toStdString(),
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
        std::copy(T.begin(),T.end(),affine.begin());
        std::cout << "rotating images" << std::endl;
        src.rotate(src.voxel.dim,affine);
    }

    unsigned char method_index = uint8_t(po.get("method",4));
    if(method_index == 4)
        src.voxel.param[0] = 1.25f;
    if(method_index == 6) // Convert to HARDI
    {
        src.voxel.param[0] = 1.25f;
        src.voxel.param[1] = 3000.0f;
        src.voxel.param[2] = 0.05f;
    }
    if(method_index == 7) // QSDR
    {
        src.voxel.param[0] = 1.25f;
        std::cout << "selecting template..." << std::endl;
        for(size_t index = 0;index < fa_template_list.size();++index)
            std::cout << index << ":" << fa_template_list[index] << std::endl;
        if(po.has("template"))
        {
            src.voxel.primary_template = fa_template_list[size_t(po.get("template",0))];
            src.voxel.secondary_template = iso_template_list[size_t(po.get("template",0))];
        }
        else
        {
            size_t index = match_template(src.voxel.vs[0]*src.voxel.vs[1]*src.voxel.vs[2]*src.voxel.dim.size());
            src.voxel.primary_template = fa_template_list[index];
            src.voxel.secondary_template = iso_template_list[index];
        }
        std::cout << "template: " << src.voxel.primary_template << std::endl;
        std::cout << "template2: " << src.voxel.secondary_template << std::endl;
    }
    if(po.has("study_src")) // DDI
        src.voxel.study_src_file_path = po.get("study_src");
    if (po.has("param0"))
        src.voxel.param[0] = po.get("param0",float(0));
    if (po.has("param1"))
        src.voxel.param[1] = po.get("param1",float(0));
    if (po.has("param2"))
        src.voxel.param[2] = po.get("param2",float(0));
    if (po.has("param3"))
        src.voxel.param[3] = po.get("param3",float(0));
    if (po.has("param4"))
        src.voxel.param[4] = po.get("param4",float(0));

    src.voxel.method_id = method_index;
    src.voxel.ti.init(uint16_t(po.get("odf_order",int(8))));
    src.voxel.odf_resolving = po.get("odf_resolving",int(0));
    src.voxel.output_odf = po.get("record_odf",int(0));
    src.voxel.dti_no_high_b = po.get("dti_no_high_b",src.is_human_data());
    src.voxel.check_btable = po.get("check_btable",int(src.voxel.dim[2] < src.voxel.dim[0]*2.0 ? 1:0));
    src.voxel.other_output = po.get("other_output","fa,ad,rd,md,nqa,iso,rdi,nrdi");
    src.voxel.max_fiber_number = uint32_t(po.get("num_fiber",int(5)));
    src.voxel.r2_weighted = po.get("r2_weighted",int(0));
    src.voxel.thread_count = po.get("thread_count",uint32_t(std::thread::hardware_concurrency()));
    src.voxel.half_sphere = po.get("half_sphere",src.is_dsi_half_sphere() ? 1:0);
    src.voxel.scheme_balance = po.get("scheme_balance",src.need_scheme_balance() ? 1:0);


    {
        if(src.voxel.output_odf)
            std::cout << "record ODF in the fib file" << std::endl;
        if(src.voxel.r2_weighted && method_index == 4)
            std::cout << "r2 weighted is used for GQI" << std::endl;
    }

    if(po.has("other_image"))
    {
        QStringList file_list = QString(po.get("other_image").c_str()).split(",");
        for(int i = 0;i < file_list.size();++i)
        {
            QStringList name_value = file_list[i].split(":");
            if(name_value.size() == 1)
            {
                std::cout << "invalid parameter: " << file_list[i].toStdString() << std::endl;
                return 1;
            }
            if(name_value.size() == 3) // handle windows directory with drive letter
            {
                name_value[1] += ":";
                name_value[1] += name_value[2];
            }
            std::cout << name_value[0].toStdString() << ":" << name_value[1].toStdString() << std::endl;
            if(!add_other_image(&src,name_value[0],name_value[1]))
                return 1;
        }
    }
    if(po.has("mask"))
    {
        std::shared_ptr<fib_data> fib_handle(new fib_data);
        fib_handle->dim = src.voxel.dim;
        fib_handle->vs = src.voxel.vs;
        std::string mask_file = po.get("mask");

        if(mask_file == "1")
            std::fill(src.voxel.mask.begin(),src.voxel.mask.end(),1);
        else
        {
            gz_nifti nii;
            if(!nii.load_from_file(mask_file) || !nii.toLPS(src.voxel.mask))
            {
                std::cout << "invalid or nonexisting nifti file:" << mask_file << std::endl;
                return 1;
            }
            if(src.voxel.mask.geometry() != src.voxel.dim)
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
        tipl::reg::two_way_linear_mr(I,vs,src.dwi_sum,src.voxel.vs,
                       T,tipl::reg::rigid_body,tipl::reg::mutual_information(),
                        terminated,src.voxel.thread_count);
        std::cout << "DWI rotated." << std::endl;
        src.rotate(I.geometry(),T);
        src.voxel.vs = vs;
    }

    if(po.get("motion_correction",int(0)))
    {
        std::cout << "correct for motion and eddy current..." << std::endl;
        rec_motion_correction(&src);
        std::cout << "done." <<std::endl;
    }
    std::cout << "start reconstruction..." <<std::endl;
    if(po.has("output"))
    {
        std::string output = po.get("output");
        if(QFileInfo(output.c_str()).isDir())
            src.file_name = output + "/" + std::filesystem::path(src.file_name).filename().string();
        else
            src.file_name = output;
    }
    if(po.has("save_src"))
    {
        std::string new_src_file = po.get("save_src");
        std::cout << "saving to " << new_src_file << std::endl;
        src.save_to_file(new_src_file.c_str());
        return 0;
    }
    if (src.reconstruction())
        std::cout << "reconstruction finished." << std::endl;
    else
    {
        std::cout << "ERROR:" << src.error_msg << std::endl;
        return 1;
    }
    return 0;
}
