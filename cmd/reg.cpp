#include <QString>
#include "reg.hpp"
#include "tract_model.hpp"

template<>
bool dual_reg<3>::apply_warping(const char* from,const char* to) const
{
    tipl::out() << "apply warping to " << from;
    if(tipl::ends_with(from,".tt.gz"))
        return apply_warping_tt(from,to);
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
        if(I.shape() != I_list[0].shape())
        {
            error_msg = std::filesystem::path(from).filename().string();
            error_msg += " has an image size or srow matrix from that of the original --from image.";
            return false;
        }
        bool is_label = tipl::is_label_image(I_list[0]);
        tipl::out() << (is_label ? "processed as labels using nearest assignment" : "processed as values using interpolation") << std::endl;
        tipl::image<4> J(to2from.shape().expand(nii.dim(4)));
        tipl::par_for(nii.dim(4),[&](size_t z)
        {
            tipl::image<3> out;
            apply_warping(I_list[z],out,is_label);
            std::copy(out.begin(),out.end(),J.slice_at(z).begin());
        });
        if(!tipl::io::gz_nifti::save_to_file(to,J,Itvs,ItR,It_is_mni))
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

    if(I.shape() != I3.shape())
    {
        error_msg = std::filesystem::path(from).filename().string();
        error_msg += " has an image size or srow matrix from that of the original --from image.";
        return false;
    }

    tipl::image<3> J3(It.shape());
    apply_warping(I3,J3,is_label);
    tipl::out() << "save as to " << to;
    if(!tipl::io::gz_nifti::save_to_file(to,J3,Itvs,ItR,It_is_mni))
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
    if(!tipl::search_filesystem(po.get("apply_warp"),filename_cmds))
    {
        tipl::out() << "ERROR: cannot find file " << po.get("apply_warp") <<std::endl;
        return 1;
    }

    for(const auto& each_file: filename_cmds)
        if(!r.apply_warping(each_file.c_str(),
                        (each_file+(tipl::ends_with(each_file,".tt.gz") ? ".wp.tt.gz" : ".wp.nii.gz")).c_str()))
            tipl::out() << "ERROR: " <<r.error_msg;

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
        tipl::out() << "ERROR: cannot load file " << file_name << std::endl;
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
            tipl::out() << "ERROR: unknown command " << cmd << std::endl;
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

template<>
bool dual_reg<3>::load_subject(const char* file_name)
{
    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(file_name))
    {
        error_msg = "invalid nifti format";
        return false;
    }
    nifti.toLPS(I);
    nifti.get_image_transformation(IR);
    tipl::filter::gaussian(I);
    tipl::segmentation::otsu_median_regulzried(I);
    nifti.get_voxel_size(Ivs);
    I2.clear();
    return true;
}

QImage read_qimage(QString filename,std::string& error);
template<>
bool dual_reg<2>::load_template(const char* file_name)
{
    QImage in = read_qimage(file_name,error_msg);
    if(in.isNull())
        return false;
    tipl::color_image Ic;
    Ic << in;
    It = Ic;
    tipl::segmentation::otsu_median_regulzried(It);
    Itvs = {1.0f,1.0f};
    return true;
}
template<>
bool dual_reg<2>::load_subject(const char* file_name)
{
    QImage in = read_qimage(file_name,error_msg);
    if(in.isNull())
        return false;
    tipl::color_image Ic;
    Ic << in;
    I = Ic;
    tipl::segmentation::otsu_median_regulzried(I);
    Ivs = {1.0f,1.0f};
    return true;
}



template<>
bool dual_reg<3>::load_subject2(const char* file_name)
{
    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(file_name))
    {
        error_msg = "invalid nifti format";
        return false;
    }
    nifti.toLPS(I2);
    if(I2.shape() != I.shape())
    {
        error_msg = "inconsistent subject image size";
        return false;
    }
    tipl::filter::gaussian(I2);
    tipl::segmentation::otsu_median_regulzried(I2);
    return true;
}
template<>
bool dual_reg<3>::load_template(const char* file_name)
{
    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(file_name))
    {
        error_msg = "invalid nifti format";
        return false;
    }
    nifti.toLPS(It);
    nifti.get_image_transformation(ItR);
    nifti.get_voxel_size(Itvs);
    It_is_mni = nifti.is_mni();
    It2.clear();
    return true;
}
template<>
bool dual_reg<3>::load_template2(const char* file_name)
{
    It2.resize(It.shape());
    if(!tipl::io::gz_nifti::load_to_space(file_name,It2,ItR))
    {
        error_msg = "invalid nifti format";
        It2.clear();
        return false;
    }
    return true;
}
template<>
void dual_reg<3>::match_resolution(bool rigid_body)
{
    float ratio = (rigid_body ? Ivs[0]/Itvs[0] : float(I.width())/float(It.width()));
    while(ratio < 0.5f)   // if subject resolution is substantially lower, downsample template
    {
        tipl::downsampling(It);
        if(!It2.empty())
            tipl::downsampling(It2);
        Itvs *= 2.0f;
        for(auto each : {0,1,2,
                         4,5,6,
                         8,9,10})
            ItR[each] *= 2.0f;
        ratio *= 2.0f;
        tipl::out() << "downsampling template to " << Itvs[0] << " mm resolution" << std::endl;
    }
    while(ratio > 2.5f)  // if subject resolution is higher, downsample it for registration
    {
        tipl::downsampling(I);
        if(!I2.empty())
            tipl::downsampling(I2);
        Ivs *= 2.0f;
        for(auto each : {0,1,2,
                         4,5,6,
                         8,9,10})
            IR[each] *= 2.0f;
        ratio /= 2.0f;
        tipl::out() << "downsample subject to " << Ivs[0] << " mm resolution" << std::endl;
    }
}




int reg(tipl::program_option<tipl::out>& po)
{
    dual_reg<3> r;

    if(po.has("warp") || po.has("inv_warp"))
    {
        if(!po.has("apply_warp"))
        {
            tipl::out() << "ERROR: please specify the images or tracts to be warped using --apply_warp";
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
        tipl::out() << "ERROR: please specify the images to normalize using --from and --to";
        return 1;
    }

    if(!r.load_subject(po.get("from").c_str()) ||
       !r.load_template(po.get("to").c_str()))
        goto error;
    if(po.has("from2") && po.has("to2"))
    {
        if(!r.load_subject2(po.get("from2").c_str()) ||
           !r.load_template2(po.get("to2").c_str()))
            goto error;
    }

    if(!r.I2.empty() && r.I2.shape() != r.I.shape())
    {
        tipl::out() << "ERROR: --from2 and --from images have different dimension" << std::endl;
        return 1;
    }
    if(!r.It2.empty() && r.It2.shape() != r.It.shape())
    {
        tipl::out() << "ERROR: --to2 and --to images have different dimension" << std::endl;
        return 1;
    }

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

    if(po.has("output") && !r.save_transformed_image(po.get("output").c_str()))
        goto error;
    if(po.has("output_warp") && !r.save_transformed_image(po.get("output_warp").c_str()))
        goto error;
    return after_warp(po,r);

    error:
    tipl::out() << "ERROR: " << r.error_msg;
    return 1;
}
