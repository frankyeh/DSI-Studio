#include <QString>
#include <QImage>
#include "reg.hpp"
#include "tract_model.hpp"



int after_warp(tipl::program_option<tipl::out>& po,dual_reg<3>& r)
{
    if(!po.has("apply_warp"))
        return 0;

    std::vector<std::string> filename_cmds;
    if(!po.get_files("apply_warp",filename_cmds))
    {
        tipl::error() << "cannot find file " << po.get("apply_warp") <<std::endl;
        return 1;
    }

    for(const auto& each_file: filename_cmds)
    {
        if(tipl::ends_with(each_file,".tt.gz"))
        {
            if(!r.apply_warping_tt(each_file.c_str(),(each_file+".wp.tt.gz").c_str()))
                tipl::error() << r.error_msg;
        }
        else
        {
            if(!r.apply_warping(each_file.c_str(),(each_file+".wp.nii.gz").c_str()))
                tipl::error() << r.error_msg;
        }
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

    if(po.has("warp") || po.has("inv_warp"))
    {
        if(!po.has("apply_warp"))
        {
            tipl::error() << "please specify the images or tracts to be warped using --apply_warp";
            return 1;
        }
        tipl::out() << "loading warping field";
        if(!r.load_warping(po.has("warp") ? po.get("warp") : po.get("inv_warp")))
            goto error;
        if(!po.has("warp"))
            r.inv_warping();
        return after_warp(po,r);
    }

    if(!po.has("from") || !po.has("to"))
    {
        tipl::error() << "please specify the images to normalize using --from and --to";
        return 1;
    }

    if(!r.load_subject(0,po.get("from")) ||
       !r.load_template(0,po.get("to")))
        goto error;
    if(po.has("from2") && po.has("to2"))
    {
        if(!r.load_subject(1,po.get("from2")) ||
           !r.load_template(1,po.get("to2")))
            goto error;
    }

    tipl::out() << "from dim: " << r.I[0].shape();
    tipl::out() << "to dim: " << r.It[0].shape();

    tipl::out() << "running linear registration." << std::endl;

    if(po.get("large_deform",0))
        r.bound = tipl::reg::large_bound;
    r.reg_type = po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine;
    r.cost_type = po.get("cost_function",r.reg_type==tipl::reg::rigid_body ? "mi" : "corr") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr;
    r.linear_reg(tipl::prog_aborted);


    if(r.reg_type != tipl::reg::rigid_body)
    {
        r.param.resolution = po.get("resolution",r.param.resolution);
        r.param.speed = po.get("speed",r.param.speed);
        r.param.smoothing = po.get("smoothing",r.param.smoothing);
        r.param.min_dimension = po.get("min_dimension",r.param.min_dimension);
        r.nonlinear_reg(tipl::prog_aborted);
    }

    if(po.has("output_warp") && !r.save_warping(po.get("output_warp").c_str()))
        goto error;
    return after_warp(po,r);

    error:
    tipl::error() << r.error_msg;
    return 1;
}
