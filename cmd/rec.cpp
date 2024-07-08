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
        tipl::error() << src.error_msg << std::endl;
        return 1;
    }
    {
        tipl::progress prog("reconstruction parameters");
        src.voxel.method_id = uint8_t(po.get("method",4));
        src.voxel.odf_resolving = po.get("odf_resolving",int(0));
        src.voxel.output_odf = po.get("record_odf",int(0));
        src.voxel.dti_no_high_b = po.get("dti_no_high_b",src.is_human_data());
        src.voxel.other_output = po.get("other_output","fa,ad,rd,md,iso,rdi");
        src.voxel.r2_weighted = po.get("r2_weighted",int(0));
        src.voxel.thread_count = tipl::max_thread_count = po.get("thread_count",tipl::max_thread_count);
        src.voxel.param[0] = po.get("param0",src.voxel.param[0]);
        src.voxel.param[1] = po.get("param1",src.voxel.param[1]);
        src.voxel.param[2] = po.get("param2",src.voxel.param[2]);
        for(size_t id = 0;id < fa_template_list.size();++id)
            tipl::out() << "template " << id << ": " << std::filesystem::path(fa_template_list[id]).stem() << std::endl;
        src.voxel.template_id = size_t(po.get("template",src.voxel.template_id));

        if(src.voxel.method_id == 7) // is qsdr
            src.voxel.qsdr_reso = po.get("qsdr_reso",src.voxel.vs[2]);

    }
    if(po.has("mask"))
    {
        tipl::progress prog("specify mask");
        std::string mask_file = po.get("mask");
        if(mask_file == "1")
            src.voxel.mask = 1;
        else
        {
            if(mask_file == "unet")
            {
                if(!src.mask_from_unet())
                {
                    tipl::error() << src.error_msg;
                    return 1;
                }
            }
            else
            {
                tipl::out() << "reading mask file: " << mask_file << std::endl;
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
    }
    {
        if(po.has("rev_pe") && !src.run_topup_eddy(po.get("rev_pe")))
        {
            tipl::error() << src.error_msg << std::endl;
            return 1;
        }
        if(po.get("motion_correction",0))
            src.correct_motion();

    }

    {
        tipl::progress prog("additional processing steps");
        if (po.has("remove"))
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
        if(po.get("check_btable",0))
        {
            if(!src.check_b_table())
            {
                tipl::error() << src.error_msg;
                return 1;
            }
        }
        if (po.has("cmd"))
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
                           linear(make_list(I),vs,make_list(src.dwi),src.voxel.vs,
                                  po.has("rotate_to") ? tipl::reg::rigid_body : tipl::reg::affine));
                tipl::out() << "DWI rotated." << std::endl;
            }

            if(po.has("align_acpc"))
                src.align_acpc(po.get("align_acpc",src.voxel.vs[0]));
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
        QStringList file_list = QString(po.get("other_image").c_str()).split(",");
        for(int i = 0;i < file_list.size();++i)
        {
            QStringList name_value = file_list[i].split(":");
            if(name_value.size() == 1)
            {
                tipl::error() << "invalid parameter: " << file_list[i].toStdString() << std::endl;
                return 1;
            }
            if(name_value.size() == 3 && name_value[1].size() == 1) // handle windows directory with drive letter --other_image=t1w:c:/t1w.nii.gz
            {
                name_value[1] += ":";
                name_value[1] += name_value[2];
            }
            tipl::out() << name_value[0].toStdString() << ":" << name_value[1].toStdString() << std::endl;
            if(!src.add_other_image(name_value[0].toStdString(),name_value[1].toStdString()))
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
            src.file_name = output + "/" + std::filesystem::path(src.file_name).filename().string();
        else
            src.file_name = output;
    }
    if (!src.reconstruction())
    {
        tipl::error() << src.error_msg << std::endl;
        return 1;
    }
    return 0;
}
