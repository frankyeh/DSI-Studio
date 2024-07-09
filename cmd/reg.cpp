#include <QString>
#include <QImage>
#include "reg.hpp"
#include "tract_model.hpp"

template<>
bool dual_reg<3>::apply_warping(const char* from,const char* to) const
{
    tipl::out() << "apply warping to " << from;
    if(tipl::ends_with(from,".tt.gz"))
        return apply_warping_tt(from,to);

    tipl::out() << "opening " << from;
    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(from))
    {
        error_msg = nii.error_msg;
        return false;
    }
    if(nii.dim(4) > 1)
    {
        // check data range
        std::vector<tipl::image<3> > I_list(nii.dim(4));
        for(unsigned int index = 0;index < nii.dim(4);++index)
        {
            if(!nii.toLPS(I_list[index]))
            {
                error_msg = "failed to parse 4D NIFTI file";
                return false;
            }
            std::replace_if(I_list[index].begin(),I_list[index].end(),[](float v){return std::isnan(v) || std::isinf(v) || v < 0.0f;},0.0f);
        }
        if(I[0].shape() != I_list[0].shape())
        {
            error_msg = std::filesystem::path(from).filename().string();
            error_msg += " has an image size or srow matrix from that of the original --from image.";
            return false;
        }
        bool is_label = tipl::is_label_image(I_list[0]);
        tipl::out() << (is_label ? "processed as labels using nearest assignment" : "processed as values using interpolation") << std::endl;
        tipl::image<4> J4(It[0].shape().expand(nii.dim(4)));
        tipl::par_for(nii.dim(4),[&](size_t z)
        {
            tipl::image<3> out(apply_warping(I_list[z],is_label));
            std::copy(out.begin(),out.end(),J4.slice_at(z).begin());
        });
        if(!tipl::io::gz_nifti::save_to_file(to,J4,Itvs,ItR,It_is_mni))
        {
            error_msg = "cannot write to file ";
            error_msg += to;
            return false;
        }
        return true;
    }

    tipl::image<3> I3;
    if(!nii.toLPS(I3))
    {
        error_msg = nii.error_msg;
        return false;
    }
    bool is_label = tipl::is_label_image(I3);
    tipl::out() << (is_label ? "processed as labels using nearest assignment" : "processed as values using interpolation") << std::endl;

    tipl::out() << "dim: " << I3.shape();
    if(I[0].shape() != I3.shape())
    {
        tipl::out() << "--from dim: " << I[0].shape();
        error_msg = std::filesystem::path(from).filename().string();
        error_msg += " has an image size or srow matrix from that of the original --from image.";
        return false;
    }

    tipl::out() << "saving " << to;
    if(!tipl::io::gz_nifti::save_to_file(to,apply_warping(I3,is_label),Itvs,ItR,It_is_mni))
    {
        error_msg = "cannot write to file ";
        error_msg += to;
        return false;
    }
    return true;
}

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
        if(!r.apply_warping(each_file.c_str(),
                        (each_file+(tipl::ends_with(each_file,".tt.gz") ? ".wp.tt.gz" : ".wp.nii.gz")).c_str()))
            tipl::error() << r.error_msg;

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
        if(!r.load_warping(po.has("warp") ? po.get("warp").c_str() : po.get("inv_warp").c_str()))
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

    if(!r.load_subject(0,po.get("from").c_str()) ||
       !r.load_template(0,po.get("to").c_str()))
        goto error;
    if(po.has("from2") && po.has("to2"))
    {
        if(!r.load_subject(1,po.get("from2").c_str()) ||
           !r.load_template(1,po.get("to2").c_str()))
            goto error;
    }

    tipl::out() << "from dim: " << r.I[0].shape();
    tipl::out() << "to dim: " << r.It[0].shape();

    tipl::out() << "running linear registration." << std::endl;

    if(po.get("large_deform",0) == 1)
        r.bound = tipl::reg::large_bound;

    r.linear_reg(po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,
                 po.get("cost_function","mi") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr);


    if(po.get("reg_type",1) != 0)
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
