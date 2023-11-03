#include <QString>
#include "reg.hpp"
#include "tract_model.hpp"
bool apply_warping(const char* from,
                   const char* to,
                   const tipl::shape<3>& I_shape,
                   const tipl::matrix<4,4>& IR,
                   tipl::image<3,tipl::vector<3> >& to2from,
                   tipl::vector<3> Itvs,
                   const tipl::matrix<4,4>& ItR,
                   bool It_is_mni,
                   std::string& error);
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
        tipl::out() << "ERROR: cannot find " << po.get("apply_warp") <<std::endl;
        return 1;
    }
    for(const auto& each_file: filename_cmds)
    {
        if(tipl::ends_with(each_file,".tt.gz"))
            apply_unwarping_tt(each_file.c_str(),(each_file+".wp.tt.gz").c_str(),from2to,to2from.shape(),to_vs,from_trans,to_trans,error);
        else
            apply_warping(each_file.c_str(),(each_file+".wp.nii.gz").c_str(),from2to.shape(),from_trans,to2from,to_vs,to_trans,to_is_mni,error);
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
void edge_for_cdm(tipl::image<3>& sIt,
                  tipl::image<3>& sJ,
                  tipl::image<3>& sIt2,
                  tipl::image<3>& sJ2);
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
            tipl::out() << "ERROR: please specify the images to be warped using --apply_warp";
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


    bool terminated = false;
    tipl::out() << "running linear registration." << std::endl;

    tipl::transformation_matrix<float> T;
    auto cost_function = po.get("cost_function","mi");
    if(cost_function == std::string("mi"))
        linear_with_mi(to,to_vs,from,from_vs,T,
                  po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,terminated,
                  po.get("large_deform",0) == 1 ? tipl::reg::large_bound : tipl::reg::reg_bound);
    else
    if(cost_function == std::string("cc"))
        linear_with_cc(to,to_vs,from,from_vs,T,
                  po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,terminated,
                  po.get("large_deform",0) == 1 ? tipl::reg::large_bound : tipl::reg::reg_bound);
    else
    {
        tipl::out() << "ERROR: unknown cost_function " << cost_function << std::endl;
        return 1;
    }

    tipl::image<3> from_(to.shape()),from2_;


    if(tipl::is_label_image(from))
        tipl::resample_mt<tipl::interpolation::nearest>(from,from_,T);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(from,from_,T);


    if(!from2.empty())
    {
        from2_.resize(to.shape());
        if(tipl::is_label_image(from2))
            tipl::resample_mt<tipl::interpolation::nearest>(from2,from2_,T);
        else
            tipl::resample_mt<tipl::interpolation::cubic>(from2,from2_,T);
    }
    auto r2 = tipl::correlation(from_.begin(),from_.end(),to.begin());
    tipl::out() << "correlation cofficient: " << r2 << std::endl;
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
                tipl::resample_mt<tipl::interpolation::nearest>(from2,from2_,T);
            else
                tipl::resample_mt<tipl::interpolation::cubic>(from2,from2_,T);
            tipl::io::gz_nifti::save_to_file((po.get("apply_warp")+".wp.nii.gz").c_str(),from2_,to_vs,to_trans,to_is_mni);
        }
        return 0;
    }

    if(po.get("normalize_signal",1))
    {
        tipl::out() << "normalizing signals" << std::endl;
        tipl::reg::cdm_pre(from_,from2_,to,to2);
    }

    tipl::reg::cdm_param param;
    param.resolution = po.get("resolution",param.resolution);
    param.speed = po.get("speed",param.speed);
    param.smoothing = po.get("smoothing",param.smoothing);
    param.iterations = po.get("iteration",param.iterations);
    param.min_dimension = po.get("min_dimension",param.min_dimension);

    if(po.get("use_edge",0))
    {
        tipl::image<3> to_edge(to),from_edge(from_),to2_edge(to2),from2_edge(from2_);
        edge_for_cdm(to_edge,from_edge,to2_edge,from2_edge);
        cdm_common(to_edge,to2_edge,from_edge,from2_edge,t2f_dis,f2t_dis,terminated,param);
    }
    else
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
        tipl::out() << "R2: " << r*r << std::endl;
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
