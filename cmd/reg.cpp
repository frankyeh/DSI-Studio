#include <QString>
#include "reg.hpp"
#include "tract_model.hpp"

bool apply_unwarping_tt(const char* from,
                        const char* to,
                        const tipl::image<3,tipl::vector<3> >& from2to,
                        tipl::shape<3> new_geo,
                        tipl::vector<3> new_vs,
                        const tipl::matrix<4,4>& from_trans_to_mni,
                        const tipl::matrix<4,4>& to_trans_to_mni,
                        std::string& error);

int after_warp(tipl::program_option<tipl::out>& po,
               tipl::image<3,tipl::vector<3> >& to2from,
               tipl::image<3,tipl::vector<3> >& from2to,
               tipl::vector<3> to_vs,
               const tipl::matrix<4,4>& from_trans,
               const tipl::matrix<4,4>& to_trans,
               bool to_is_mni)
{
    if(!po.has("apply_warp"))
        return 0;

    std::string error;
    std::vector<std::string> filename_cmds;
    if(!tipl::search_filesystem(po.get("apply_warp"),filename_cmds))
    {
        tipl::out() << "ERROR: cannot find file " << po.get("apply_warp") <<std::endl;
        return 1;
    }
    for(const auto& each_file: filename_cmds)
    {
        if(tipl::ends_with(each_file,".tt.gz"))
            apply_unwarping_tt(each_file.c_str(),(each_file+".wp.tt.gz").c_str(),from2to,to2from.shape(),to_vs,from_trans,to_trans,error);
        //else
        //    apply_warping(each_file.c_str(),(each_file+".wp.nii.gz").c_str(),from2to.shape(),from_trans,to2from,to_vs,to_trans,to_is_mni,error);
    }
    if(!error.empty())
    {
        tipl::out() << "ERROR: " << error <<std::endl;
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
void dual_reg::linear_reg(tipl::reg::reg_type reg_type,tipl::reg::cost_type cost_type,bool& terminated)
{
    tipl::image<3> J_,J2_;
    linear(make_list(It,It2),Itvs,make_list(I,I2),Ivs,
           arg,reg_type,terminated,bound,cost_type);

    tipl::out() << "linear registration completed" << std::endl;
    auto trans = T();
    J_.resize(It.shape());
    tipl::resample(I,J_,trans);

    if(I2.shape() == I.shape())
    {
        J2.resize(It.shape());
        tipl::resample(I2,J2,trans);
    }

    auto r = tipl::correlation(J_,It);
    tipl::out() << "linear: " << r << std::endl;
    J2.swap(J2_);
    J.swap(J_);
}
bool dual_reg::nonlinear_reg(bool& terminated,bool use_cuda)
{
    tipl::out() << "begin nonlinear registration" << std::endl;

    cdm_common(It,It2,J,J2,t2f_dis,f2t_dis,terminated,param,use_cuda);

    tipl::out() << "nonlinear registration completed.";

    auto trans = T();
    from2to.resize(I.shape());
    tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
    tipl::displacement_to_mapping(t2f_dis,to2from,trans);
    tipl::compose_mapping(I,to2from,JJ);
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

bool dual_reg::apply_warping(const char* from,const char* to) const
{
    tipl::out() << "apply warping to " << from << std::endl;

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
        for(size_t i = 0;i < nii.dim(4);++i)
        {
            auto J_slice = J.slice_at(i);
            if(is_label)
                tipl::compose_mapping<tipl::interpolation::nearest>(I_list[i],to2from,J_slice);
            else
                tipl::compose_mapping<tipl::interpolation::cubic>(I_list[i],to2from,J_slice);
        }
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

    tipl::image<3> J3;
    if(is_label)
        tipl::compose_mapping<tipl::interpolation::nearest>(I3,to2from,J3);
    else
        tipl::compose_mapping<tipl::interpolation::cubic>(I3,to2from,J3);
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
    tipl::io::gz_mat_write out(filename);
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

bool dual_reg::save_transformed_image(const char* filename) const
{
    if(!tipl::io::gz_nifti::save_to_file(filename,JJ,Itvs,ItR,It_is_mni))
    {
        error_msg = "cannot write to file ";
        error_msg += filename;
        return false;
    }
    return true;
}


int reg(tipl::program_option<tipl::out>& po)
{
    tipl::image<3> from,to,from2,to2;
    tipl::vector<3> from_vs,to_vs;
    tipl::matrix<4,4> from_trans,to_trans;
    bool from_is_mni = false,to_is_mni = false;
    tipl::image<3,tipl::vector<3> > t2f_dis,f2t_dis,to2from,from2to;

    if(po.has("warp") || po.has("inv_warp"))
    {
        if(!po.has("apply_warp"))
        {
            tipl::out() << "ERROR: please specify the images or tracts to be warped using --apply_warp";
            return 1;
        }
        tipl::out() << "loading warping field";
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(po.has("warp") ? po.get("warp").c_str() : po.get("inv_warp").c_str()))
        {
            tipl::out() << "ERROR: cannot open or parse the warp file " << std::endl;
            return 1;
        }
        tipl::shape<3> to_dim,from_dim;
        const float* to2from_ptr = nullptr;
        const float* from2to_ptr = nullptr;
        unsigned int row,col;
        if (!in.read("to_dim",to_dim) ||
            !in.read("to_vs",to_vs) ||
            !in.read("from_dim",from_dim) ||
            !in.read("from_vs",from_vs) ||
            !in.read("from_trans",from_trans) ||
            !in.read("to_trans",to_trans) ||
            !in.read("to2from",row,col,to2from_ptr) ||
            !in.read("from2to",row,col,from2to_ptr))
        {
            tipl::out() << "ERROR: invalid warp file " << po.get("warp") << std::endl;
            return 1;
        }
        to2from.resize(to_dim);
        std::copy(to2from_ptr,to2from_ptr+to2from.size()*3,&to2from[0][0]);
        from2to.resize(from_dim);
        std::copy(from2to_ptr,from2to_ptr+from2to.size()*3,&from2to[0][0]);

        if(!po.has("warp"))
        {
            to_dim.swap(from_dim);
            std::swap(to_vs,from_vs);
            to_trans.swap(from_trans);
            to2from.swap(from2to);
        }
        return after_warp(po,to2from,from2to,to_vs,from_trans,to_trans,to_is_mni);
    }

    if(!po.has("from") || !po.has("to"))
    {
        tipl::out() << "ERROR: please specify the images to normalize using --from and --to";
        return 1;
    }

    if(!load_nifti_file(po.get("from").c_str(),from,from_vs,from_trans,from_is_mni) ||
       !load_nifti_file(po.get("to").c_str(),to,to_vs,to_trans,to_is_mni))
        return 1;

    if(po.has("from2") && po.has("to2"))
    {
        if(!load_nifti_file(po.get("from2").c_str(),from2,from_vs) ||
           !load_nifti_file(po.get("to2").c_str(),to2,to_vs))
            return 1;
    }

    if(!from2.empty() && from.shape() != from2.shape())
    {
        tipl::out() << "ERROR: --from2 and --from images have different dimension" << std::endl;
        return 1;
    }
    if(!to2.empty() && to.shape() != to2.shape())
    {
        tipl::out() << "ERROR: --to2 and --to images have different dimension" << std::endl;
        return 1;
    }

    if(po.get("normalize_signal",1))
    {
        tipl::out() << "normalizing signals" << std::endl;
        tipl::segmentation::otsu_median_regulzried(from);
        tipl::segmentation::otsu_median_regulzried(from2);
        tipl::segmentation::otsu_median_regulzried(to);
        tipl::segmentation::otsu_median_regulzried(to2);
    }


    bool terminated = false;
    tipl::out() << "running linear registration." << std::endl;

    tipl::affine_transform<float> arg;
    linear(make_list(to,to2),to_vs,make_list(from,from2),from_vs,arg,
                  po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,terminated,
                  po.get("large_deform",0) == 1 ? tipl::reg::large_bound : tipl::reg::reg_bound,
                  po.get("cost_function","mi") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr);
    auto T = tipl::transformation_matrix<float>(arg,to.shape(),to_vs,from.shape(),from_vs);


    tipl::image<3> from_(to.shape()),from2_;


    if(tipl::is_label_image(from))
        tipl::resample<tipl::interpolation::nearest>(from,from_,T);
    else
        tipl::resample<tipl::interpolation::cubic>(from,from_,T);


    if(!from2.empty())
    {
        from2_.resize(to.shape());
        if(tipl::is_label_image(from2))
            tipl::resample<tipl::interpolation::nearest>(from2,from2_,T);
        else
            tipl::resample<tipl::interpolation::cubic>(from2,from2_,T);
    }
    auto r2 = tipl::correlation(from_.begin(),from_.end(),to.begin());
    tipl::out() << "linear: " << r2 << std::endl;
    if(po.get("reg_type",1) == 0) // just rigidbody
    {
        std::string output_wp_image = po.get("output",po.get("from")+".wp.nii.gz");
        tipl::out() << "output warped image to " << output_wp_image << std::endl;
        tipl::io::gz_nifti::save_to_file(output_wp_image.c_str(),from_,to_vs,to_trans);
        if(po.has("apply_warp"))
        {
            if(!load_nifti_file(po.get("apply_warp").c_str(),from2,from_vs))
                return 1;
            if(from2.shape() != from.shape())
            {
                tipl::out() << "ERROR: --from and --apply_warp image has different dimension" << std::endl;
                return 1;
            }
            from2_.resize(to.shape());;
            if(tipl::is_label_image(from2))
                tipl::resample<tipl::interpolation::nearest>(from2,from2_,T);
            else
                tipl::resample<tipl::interpolation::cubic>(from2,from2_,T);
            tipl::io::gz_nifti::save_to_file((po.get("apply_warp")+".wp.nii.gz").c_str(),from2_,to_vs,to_trans,to_is_mni);
        }
        return 0;
    }

    tipl::reg::cdm_param param;
    param.resolution = po.get("resolution",param.resolution);
    param.speed = po.get("speed",param.speed);
    param.smoothing = po.get("smoothing",param.smoothing);
    param.iterations = po.get("iteration",param.iterations);
    param.min_dimension = po.get("min_dimension",param.min_dimension);

    cdm_common(to,to2,from_,from2_,t2f_dis,f2t_dis,terminated,param);

    tipl::displacement_to_mapping(t2f_dis,to2from,T);
    from2to.resize(from.shape());
    tipl::inv_displacement_to_mapping(f2t_dis,from2to,T);

    {
        tipl::out() << "compose output images" << std::endl;
        tipl::image<3> output;
        if(tipl::is_label_image(from))
            tipl::compose_mapping<tipl::interpolation::nearest>(from,to2from,output);
        else
            tipl::compose_mapping<tipl::interpolation::cubic>(from,to2from,output);

        float r = float(tipl::correlation(to.begin(),to.end(),output.begin()));
        tipl::out() << "nonlinear: " << r << std::endl;
        if(po.has("output"))
        {
            if(!tipl::io::gz_nifti::save_to_file(po.get("output").c_str(),output,to_vs,to_trans))
            {
                tipl::out() << "ERROR: cannot write to " << po.get("output") << std::endl;
                return 1;
            }
        }
    }

    if(po.has("output_warp"))
    {
        std::string filename = po.get("output_warp");
        if(!QString(filename.c_str()).endsWith(".map.gz"))
            filename += ".map.gz";
        tipl::io::gz_mat_write out(filename.c_str());
        if(!out)
        {
            tipl::out() << "ERROR: cannot write to " << filename << std::endl;
            return 1;
        }
        out.write("to2from",&to2from[0][0],3,to2from.size());
        out.write("to_dim",to2from.shape());
        out.write("to_vs",to_vs);
        out.write("to_trans",to_trans);

        out.write("from2to",&from2to[0][0],3,from2to.size());
        out.write("from_dim",from.shape());
        out.write("from_vs",from_vs);
        out.write("from_trans",from_trans);
        tipl::out() << "save mapping to " << filename << std::endl;
    }

    return after_warp(po,to2from,from2to,to_vs,from_trans,to_trans,to_is_mni);
}
