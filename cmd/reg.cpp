#include <QString>
#include <QImage>
#include "reg.hpp"
#include "tract_model.hpp"



int after_warp(const std::vector<std::string>& apply_warp_filename,dual_reg<3>& r)
{
    for(const auto& each_file: apply_warp_filename)
    {
        if(!r.apply_warping(each_file.c_str(),(each_file+".wp.nii.gz").c_str()))
            tipl::error() << r.error_msg;
        return 1;
    }
    return 0;
}


bool load_nifti_file(std::string file_name_cmd,
                     tipl::image<3>& data,
                     tipl::vector<3>& vs,
                     tipl::matrix<4,4>& trans,
                     bool& is_mni)
{
    std::istringstream in(file_name_cmd);
    std::string file_name,cmd;
    std::getline(in,file_name,'+');
    if(!tipl::io::gz_nifti::load_from_file(file_name.c_str(),data,vs,trans,is_mni))
    {
        tipl::error() << "cannot load file " << file_name << std::endl;
        return false;
    }
    while(std::getline(in,cmd,'+'))
    {
        tipl::out() << "apply " << cmd << std::endl;
        if(cmd == "gaussian")
            tipl::filter::gaussian(data);
        else
        if(cmd == "sobel")
            tipl::filter::sobel(data);
        else
        if(cmd == "mean")
            tipl::filter::mean(data);
        else
        {
            tipl::error() << "unknown command " << cmd << std::endl;
            return false;
        }
    }
    return true;
}
bool load_nifti_file(std::string file_name_cmd,tipl::image<3>& data,tipl::vector<3>& vs)
{
    tipl::matrix<4,4> trans;
    bool is_mni;
    return load_nifti_file(file_name_cmd,data,vs,trans,is_mni);
}


int reg(tipl::program_option<tipl::out>& po)
{
    dual_reg<3> r;
    if(!po.has("source") || !po.has("to"))
    {
        tipl::error() << "please specify the images to normalize using --source and --to";
        return 1;
    }


    std::vector<std::string> from_filename,to_filename;
    if(!po.get_files("source",from_filename))
    {
        tipl::error() << "cannot find file " << po.get("source") <<std::endl;
        return 1;
    }
    if(!po.get_files("to",to_filename))
    {
        tipl::error() << "cannot find file " << po.get("to") <<std::endl;
        return 1;
    }

    if(!po.get("overwrite",0))
    {
        bool skip = true;
        for(const auto& each_file: from_filename)
        {
            if((tipl::ends_with(each_file,".tt.gz") && !std::filesystem::exists(each_file+".wp.tt.gz")) ||
               (tipl::ends_with(each_file,".nii.gz") && !std::filesystem::exists(each_file+".wp.nii.gz")))
            {
                skip = false;
                break;
            }
        }
        if(skip)
        {
            tipl::out() << "output file exists, skipping";
            return 0;
        }
    }

    if(po.has("mapping"))
    {
        tipl::out() << "loading mapping field";
        if(!r.load_warping(po.get("mapping")))
        {
            tipl::error() << r.error_msg;
            return 1;
        }
        return after_warp(from_filename,r);
    }


    for(size_t i = 0;i < from_filename.size() && i < to_filename.size();++i)
    {
        if(!r.load_subject(i,from_filename[i]))
        {
            tipl::error() << r.error_msg;
            return 1;
        }

        if(!r.load_template(i,to_filename[i]))
        {
            tipl::error() << r.error_msg;
            return 1;
        }

        r.modality_names[i] = std::filesystem::path(from_filename[i]).stem().stem().string() + "->" +
                              std::filesystem::path(to_filename[i]).stem().stem().string();
    }

    tipl::out() << "source dim: " << r.Is;
    tipl::out() << "to dim: " << r.Its;

    tipl::out() << "running linear registration." << std::endl;

    if(po.get("large_deform",0))
        r.bound = tipl::reg::large_bound;
    r.reg_type = po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine;
    r.cost_type = po.get("cost_function",r.reg_type==tipl::reg::rigid_body ? "mi" : "corr") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr;
    r.skip_linear = po.get("skip_linear",r.skip_linear);
    r.skip_nonlinear = po.get("skip_nonlinear",r.skip_nonlinear);

    r.linear_reg(tipl::prog_aborted);

    if(r.reg_type != tipl::reg::rigid_body)
    {
        r.param.resolution = po.get("resolution",r.param.resolution);
        r.param.speed = po.get("speed",r.param.speed);
        r.param.smoothing = po.get("smoothing",r.param.smoothing);
        r.param.min_dimension = po.get("min_dimension",r.param.min_dimension);
        r.nonlinear_reg(tipl::prog_aborted);
    }

    if(po.has("output_mapping") && !r.save_warping(po.get("output_mapping").c_str()))
    {
        tipl::error() << r.error_msg;
        return 1;
    }
    return after_warp(from_filename,r);
}
