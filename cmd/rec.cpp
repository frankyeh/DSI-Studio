#include <QString>
#include <QFileInfo>
#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include "fib_data.hpp"
#include "tracking/region/Regions.h"
#include "libs/dsi/image_model.hpp"
#include "reconstruction/reconstruction_window.h"
#include "reg.hpp"

extern std::vector<std::string> fa_template_list;
bool get_src(std::string filename,src_data& src2,std::string& error_msg);
/**
 perform reconstruction
 */
int rec(tipl::program_option<tipl::out>& po)
{
    std::string file_name = po.get("source");
    src_data src;
    if (!src.load_from_file(file_name.c_str()))
    {
        tipl::error() << src.error_msg;
        return 1;
    }
    {
        tipl::progress prog("reconstruction parameters");
        src.voxel.method_id = uint8_t(po.get("method",4));
        src.voxel.odf_resolving = po.get("odf_resolving",int(0));
        src.voxel.dti_ignore_high_b = po.get("dti_ignore_high_b",src.is_human_data());
        src.voxel.other_output = po.get("other_output","fa,rd,rdi");
        src.voxel.r2_weighted = po.get("r2_weighted",int(0));
        src.voxel.thread_count = tipl::max_thread_count = po.get("thread_count",tipl::max_thread_count);
        src.voxel.export_r = po.get("export_r",0);
        src.voxel.param[0] = po.get("param0",src.voxel.param[0]);
        src.voxel.param[1] = po.get("param1",src.voxel.param[1]);
        src.voxel.param[2] = po.get("param2",src.voxel.param[2]);
        for(size_t id = 0;id < fa_template_list.size();++id)
            tipl::out() << "template " << id << ": " << std::filesystem::path(fa_template_list[id]).stem().stem().stem() << std::endl;
        src.voxel.template_id = size_t(po.get("template",src.voxel.template_id));

        if(src.voxel.method_id == 7) // is qsdr
            src.voxel.qsdr_reso = po.get("qsdr_reso",src.is_human_data() ?
                    std::min<float>(2.0f,std::max<float>(src.voxel.vs[0],src.voxel.vs[2])) : src.voxel.vs[2]);

    }
    if(po.has("intro") && !src.load_intro(po.get("intro")))
    {
        tipl::error() << src.error_msg;
        return 1;
    }

    if(po.has("mask"))
    {
        std::string mask_file = po.get("mask");
        if(mask_file == "1")
            src.voxel.mask = 1;
        if(mask_file == "unet")
        {
            if(!src.mask_from_unet())
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }
        if(mask_file == "template")
        {
            if(!src.mask_from_template())
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }

        {
            tipl::out() << "opening mask file: " << mask_file << std::endl;
            tipl::io::gz_nifti nii;
            if(!nii.load_from_file(mask_file) || !nii.toLPS(src.voxel.mask))
            {
                tipl::error() << nii.error_msg << std::endl;
                return 1;
            }
            if(src.voxel.mask.shape() != src.voxel.dim)
            {
                tipl::error() << "The mask dimension is different. terminating..." << std::endl;
                return 1;
            }
        }
    }
    else
        src.calculate_dwi_sum(true);

    if(po.has("apply_mask"))
        src.apply_mask = true;

    if(po.has("cmd"))
    {
        QStringList cmd_list = QString(po.get("cmd").c_str()).split("+");
        for(int i = 0;i < cmd_list.size();++i)
        {
            QStringList run_list = QString(cmd_list[i]).split("=");
            if(!src.command(run_list[0].toStdString(),
                                run_list.count() > 1 ? run_list[1].toStdString():std::string()))
            {
                tipl::error() << src.error_msg << std::endl;
                return 1;
            }
        }
    }

    {
        tipl::progress prog("pre-processing steps");
        if(po.has("remove"))
        {
            std::vector<int> remove_index;
            QStringList remove_list = QString(po.get("remove").c_str()).split(",");
            for(auto str : remove_list)
            {
                if(str.contains(":"))
                {
                    QStringList range = str.split(":");
                    if(range.size() != 2)
                    {
                        tipl::error() << "invalid index specified at --remove: " << str.toStdString() << std::endl;
                        return 1;
                    }
                    int from = range[0].toInt();
                    int to = src.src_bvalues.size()-1;
                    if(range[1] != "end")
                        to = range[1].toInt();
                    for(int i = from;i <= to;++i)
                        remove_index.push_back(i);
                }
                else
                    remove_index.push_back(str.toInt());
            }

            if(remove_index.empty())
            {
                tipl::error() << "invalid index specified at --remove " << std::endl;
                return 1;
            }
            std::sort(remove_index.begin(),remove_index.end(),std::greater<int>());
            std::string removed_index("DWI removed at ");
            for(auto i : remove_index)
            {
                if(i < src.src_bvalues.size())
                    src.remove(i);
                removed_index += std::to_string(i);
                removed_index += " ";
            }
            tipl::out() << removed_index << std::endl;
            tipl::out() << "current DWI count: " << src.src_bvalues.size() << std::endl;
            std::ostringstream bvalue_list;
            std::copy(src.src_bvalues.begin(),src.src_bvalues.end(),std::ostream_iterator<float>(bvalue_list," "));
            tipl::out() << "current DWI b values: " << bvalue_list.str() << std::endl;
        }
        if(po.has("rev_pe"))
        {
            if(!src.command("[Step T2][Corrections][TOPUP EDDY]",po.get("rev_pe")))
            {
                tipl::error() << src.error_msg << std::endl;
                return 1;
            }
        }

        if(po.get("volume_correction",0))
        {
            if(!src.command("[Step T2][Corrections][Volume Orientation Correction]"))
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }

        if(po.get("check_btable",0))
        {
            if(!src.command("[Step T2][B-table][Check B-table]"))
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }

        if(po.get("motion_correction",0))
        {
            if(!src.command("[Step T2][Corrections][Motion Correction]"))
            {
                tipl::error() << src.error_msg << std::endl;
                return 1;
            }
        }



        if(po.has("make_isotropic") ||
           (src.voxel.method_id != 7 &&
            src.voxel.vs[2] > src.voxel.vs[0]*1.1f &&
            src.is_human_data()))
            src.command("[Step T2][Edit][Resample]",po.get("make_isotropic",std::to_string(src.is_human_data() ? 2.0f : src.voxel.vs[2])));

        if(po.has("align_acpc"))
            src.command("[Step T2][Edit][Align ACPC]",
                        std::to_string(po.get("align_acpc",std::min<float>(2.0f,std::max<float>(src.voxel.vs[0],src.voxel.vs[2])))));
        else
        {
            if(po.has("rotate_to") || po.has("align_to"))
            {
                std::string file_name = po.has("rotate_to") ? po.get("rotate_to"):po.get("align_to");
                tipl::io::gz_nifti in;
                if(!in.load_from_file(file_name.c_str()))
                {
                    tipl::out() << "failed to read " << file_name << std::endl;
                    return 1;
                }
                tipl::image<3,unsigned char> I;
                tipl::vector<3> vs;
                in.get_voxel_size(vs);
                in >> I;
                if(po.has("rotate_to"))
                    tipl::out() << "running rigid body transformation" << std::endl;
                else
                    tipl::out() << "running affine transformation" << std::endl;

                tipl::filter::gaussian(I);
                tipl::filter::gaussian(src.dwi);

                src.rotate(I.shape(),vs,
                           tipl::reg::linear<tipl::out>(tipl::reg::make_list(I),vs,tipl::reg::make_list(src.dwi),src.voxel.vs,
                                  po.has("rotate_to") ? tipl::reg::rigid_body : tipl::reg::affine,tipl::prog_aborted));
                tipl::out() << "DWI rotated." << std::endl;
            }
        }
    }


    if(po.has("save_src") || po.has("save_nii"))
    {
        std::string new_src_file = po.has("save_src") ? po.get("save_src") : po.get("save_nii");
        if(!src.save_to_file(new_src_file.c_str()))
        {
            tipl::error() << src.error_msg << std::endl;
            return -1;
        }
        return 0;
    }

    if(po.has("reg"))
    {
        auto file_list = tipl::split(po.get("reg"),',');
        if(file_list.size() != 2)
        {
            tipl::error() << "invalid reg setting ";
            return 1;
        }
        if(!src.add_other_image("reg",file_list[0]))
        {
            tipl::error() << src.error_msg;
            return 1;
        }
        tipl::out() << "other modality template: " << (src.voxel.other_modality_template = file_list[1]);
    }
    if(po.has("other_image"))
    {
        std::vector<std::string> others;
        po.get_files("other_image",others);
        for(const auto& each : others)
        {
            tipl::out() << "add image: " << each;
            auto seps = tipl::split(each,':');
            std::string name,path;
            if(tipl::begins_with(each,"http") || seps.size() == 1)
            {
                name = std::filesystem::path(each).stem().stem().string();
                path = each;
            }
            else
            {
                name = seps[0];
                path = seps[1];
            }
            if(!src.add_other_image(name,path))
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }
    }
    if(po.has("output"))
    {
        std::string output = po.get("output");
        if(QFileInfo(output.c_str()).isDir())
            src.output_file_name = output + "/" + std::filesystem::path(src.file_name).filename().u8string();
        else
            src.output_file_name = output;
    }
    if (!src.reconstruction())
    {
        tipl::error() << src.error_msg << std::endl;
        return 1;
    }
    return 0;
}
