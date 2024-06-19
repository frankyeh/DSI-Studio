#include <QString>
#include "reg.hpp"
#include "tract_model.hpp"

int after_warp(tipl::program_option<tipl::out>& po,dual_reg& r)
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

bool dual_reg::load_subject(const char* file_name)
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

bool dual_reg::load_subject2(const char* file_name)
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

bool dual_reg::load_template(const char* file_name)
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
bool dual_reg::load_template2(const char* file_name)
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
void dual_reg::skip_linear(void)
{
    tipl::image<3> J_,J2_;
    if(I.shape() == It.shape())
        J_ = I;
    else
    {
        J_.resize(It.shape());
        tipl::draw(I,J_,tipl::vector<3,int>(0,0,0));
    }

    if(I2.shape() == I.shape())
    {
        if(I.shape() == It.shape())
            J2_ = I2;
        else
        {
            J2_.resize(It.shape());
            tipl::draw(I2,J2_,tipl::vector<3,int>(0,0,0));
        }
    }
    arg.clear();
    J2.swap(J2_);
    J.swap(J_);
}
void dual_reg::match_resolution(bool rigid_body)
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
float dual_reg::linear_reg(tipl::reg::reg_type reg_type,tipl::reg::cost_type cost_type,bool& terminated)
{

    if(export_intermediate)
    {
        tipl::io::gz_nifti::save_to_file("Template_QA.nii.gz",It,Itvs,ItR);
        if(!It2.empty())
            tipl::io::gz_nifti::save_to_file("Template_ISO.nii.gz",It2,Itvs,ItR);
        tipl::matrix<4,4> trans = {-Ivs[0],0.0f,0.0f,0.0f,
                                   0.0f,-Ivs[1],0.0f,0.0f,
                                   0.0f,0.0f,Ivs[2],0.0f,
                                   0.0f,0.0f,0.0f,1.0f};
        tipl::io::gz_nifti::save_to_file("Subject_QA.nii.gz",I,Ivs,IR);
        if(!I2.empty())
            tipl::io::gz_nifti::save_to_file("Subject_ISO.nii.gz",I2,Ivs,IR);
    }

    linear(make_list(It,It2),Itvs,make_list(I,I2),Ivs,
           arg,reg_type,terminated,bound,cost_type);

    tipl::out() << "linear registration completed" << std::endl;
    auto trans = T();

    tipl::image<3> J_,J2_;
    J_.resize(It.shape());
    tipl::resample(I,J_,trans);

    if(I2.shape() == I.shape())
    {
        J2_.resize(It.shape());
        tipl::resample(I2,J2_,trans);
    }

    auto r = tipl::correlation(J_,It);
    tipl::out() << "linear: " << r << std::endl;

    if(export_intermediate)
    {
        tipl::io::gz_nifti::save_to_file("Subject_QA_linear_reg.nii.gz",J_,Itvs,ItR);
        if(!J2_.empty())
            tipl::io::gz_nifti::save_to_file("Subject_ISO_linear_reg.nii.gz",J2_,Itvs,ItR);
    }

    J2.swap(J2_);
    J.swap(J_);

    return r;
}
float dual_reg::nonlinear_reg(bool& terminated,bool use_cuda)
{
    tipl::out() << "begin nonlinear registration" << std::endl;

    cdm_common(make_list(It,It2),make_list(J,J2),t2f_dis,f2t_dis,terminated,param,use_cuda);

    if(export_intermediate)
    {
        tipl::image<4> buffer(tipl::shape<4>(It.width(),It.height(),It.depth(),6));
        tipl::par_for(6,[&](unsigned int d)
        {
            if(d < 3)
            {
                size_t shift = d*It.size();
                for(size_t i = 0;i < It.size();++i)
                    buffer[i+shift] = f2t_dis[i][d];
            }
            else
            {
                size_t shift = d*It.size();
                d -= 3;
                for(size_t i = 0;i < It.size();++i)
                    buffer[i+shift] = t2f_dis[i][d];
            }
        });
        tipl::io::gz_nifti::save_to_file("Subject_displacement.nii.gz",buffer,Itvs,ItR);
    }

    tipl::out() << "nonlinear registration completed.";

    auto trans = T();
    from2to.resize(I.shape());
    tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
    tipl::displacement_to_mapping(t2f_dis,to2from,trans);
    tipl::compose_mapping(I,to2from,JJ);

    if(export_intermediate)
        JJ.save_to_file<tipl::io::gz_nifti>("Subject_QA_nonlinear_reg.nii.gz");

    auto r = tipl::correlation(JJ,It);
    tipl::out() << "nonlinear: " << r;
    return r;
}
void dual_reg::matching_contrast(void)
{
    std::vector<float> X(It.size()*3);
    tipl::par_for(It.size(),[&](size_t pos)
    {
        if(J[pos] == 0.0f)
            return;
        size_t i = pos;
        pos = pos+pos+pos;
        X[pos] = 1;
        X[pos+1] = It[i];
        X[pos+2] = It2[i];
    });
    tipl::multiple_regression<float> m;
    if(m.set_variables(X.begin(),3,It.size()))
    {
        float b[3] = {0.0f,0.0f,0.0f};
        m.regress(J.begin(),b);
        tipl::out() << "image=" << b[0] << " + " << b[1] << " × t1w + " << b[2] << " × t2w ";
        tipl::par_for(It.size(),[&](size_t pos)
        {
            if(J[pos] == 0.0f)
                return;
            It[pos] = b[0] + b[1]*It[pos] + b[2]*It2[pos];
            if(It[pos] < 0.0f)
                It[pos] = 0.0f;
        });
    }
    It2.clear();
}
void dual_reg::apply_warping(const tipl::image<3>& from,tipl::image<3>& to,bool is_label) const
{
    to.resize(It.shape());
    if(is_label)
    {
        if(to2from.empty())
            tipl::resample<tipl::interpolation::nearest>(from,to,T());
        else
            tipl::compose_mapping<tipl::interpolation::nearest>(from,to2from,to);
    }
    else
    {
        if(to2from.empty())
            tipl::resample<tipl::interpolation::cubic>(from,to,T());
        else
            tipl::compose_mapping<tipl::interpolation::cubic>(from,to2from,to);
    }
}
bool dual_reg::apply_warping(const char* from,const char* to) const
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

bool dual_reg::save_warping(const char* filename) const
{
    if(from2to.empty() || to2from.empty())
    {
        error_msg = "no mapping matrix to save";
        return false;
    }
    tipl::io::gz_mat_write out(tipl::ends_with(filename,".map.gz") ?
                               filename : (std::string(filename)+".map.gz").c_str());
    if(!out)
    {
        error_msg = "cannot write to file ";
        error_msg += filename;
        return false;
    }
    out.write("to2from",&to2from[0][0],3,to2from.size());
    out.write("to_dim",to2from.shape());
    out.write("to_vs",Itvs);
    out.write("to_trans",ItR);

    out.write("from2to",&from2to[0][0],3,from2to.size());
    out.write("from_dim",from2to.shape());
    out.write("from_vs",Ivs);
    out.write("from_trans",IR);

    constexpr int method_ver = 202406;
    out.write("method_ver",std::to_string(method_ver));

    return out;
}
bool dual_reg::load_warping(const char* filename)
{
    tipl::io::gz_mat_read in;
    if(!in.load_from_file(filename))
    {
        error_msg = "cannot read file ";
        error_msg += filename;
        return false;
    }
    tipl::shape<3> to_dim,from_dim;
    const float* to2from_ptr = nullptr;
    const float* from2to_ptr = nullptr;
    unsigned int row,col;
    if (!in.read("to_dim",to_dim) ||
        !in.read("to_vs",Itvs) ||
        !in.read("from_dim",from_dim) ||
        !in.read("from_vs",Ivs) ||
        !in.read("from_trans",IR) ||
        !in.read("to_trans",ItR) ||
        !in.read("to2from",row,col,to2from_ptr) ||
        !in.read("from2to",row,col,from2to_ptr))
    {
        error_msg = "invalid warp file format";
        return false;
    }
    to2from.resize(to_dim);
    std::copy(to2from_ptr,to2from_ptr+to2from.size()*3,&to2from[0][0]);
    from2to.resize(from_dim);
    std::copy(from2to_ptr,from2to_ptr+from2to.size()*3,&from2to[0][0]);
    return true;
}

bool dual_reg::save_transformed_image(const char* filename) const
{
    if(!tipl::io::gz_nifti::save_to_file(filename,JJ.empty() ? J : JJ,Itvs,ItR,It_is_mni))
    {
        error_msg = "cannot write to file ";
        error_msg += filename;
        return false;
    }
    return true;
}


int reg(tipl::program_option<tipl::out>& po)
{
    bool terminated = false;
    dual_reg r;

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
                 po.get("cost_function","mi") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr,terminated);


    if(po.get("reg_type",1) != 0)
    {
        r.param.resolution = po.get("resolution",r.param.resolution);
        r.param.speed = po.get("speed",r.param.speed);
        r.param.smoothing = po.get("smoothing",r.param.smoothing);
        r.param.iterations = po.get("iteration",r.param.iterations);
        r.param.min_dimension = po.get("min_dimension",r.param.min_dimension);
        r.nonlinear_reg(terminated);
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
