#include <QString>
#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "tract_model.hpp"
#include "program_option.hpp"
bool apply_warping(const char* from,
                   const char* to,
                   const tipl::shape<3>& I_shape,
                   const tipl::matrix<4,4>& IR,
                   tipl::image<3,tipl::vector<3> >& to2from,
                   tipl::vector<3> Itvs,
                   const tipl::matrix<4,4>& ItR,
                   std::string& error);
bool apply_unwarping_tt(const char* from,
                        const char* to,
                        const tipl::image<3,tipl::vector<3> >& from2to,
                        tipl::shape<3> new_geo,
                        tipl::vector<3> new_vs,
                        const tipl::matrix<4,4>& new_trans_to_mni,
                        std::string& error);
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
bool is_label_image(const tipl::image<3>& I);


int after_warp(const std::string& warp_name,
               tipl::image<3,tipl::vector<3> >& to2from,
               tipl::image<3,tipl::vector<3> >& from2to,
               tipl::vector<3> to_vs,
               const tipl::matrix<4,4>& from_trans,
               const tipl::matrix<4,4>& to_trans)
{
    std::string error;
    std::vector<std::string> filenames;
    get_filenames_from(warp_name,filenames);
    for(auto filename: filenames)
    {
        if(QString(filename.c_str()).toLower().endsWith(".nii.gz"))
        {
            std::string filename_warp = filename+".wp.nii.gz";
            std::cout << "apply warping to " << filename << std::endl;
            if(!apply_warping(filename.c_str(),filename_warp.c_str(),from2to.shape(),from_trans,
                              to2from,to_vs,to_trans,error))
            {
                std::cout << "ERROR: " << error <<std::endl;
                return 1;
            }
        }
        else
        if(QString(filename.c_str()).toLower().endsWith(".tt.gz"))
        {
            std::string filename_warp = filename+".wp.tt.gz";
            std::cout << "apply warping to " << filename << std::endl;
            if(!apply_unwarping_tt(filename.c_str(),filename_warp.c_str(),from2to,
                                   to2from.shape(),to_vs,to_trans,error))
            {
                std::cout << "ERROR: " << error <<std::endl;
                return 1;
            }
        }
        else
        {
            std::cout << "ERROR: unsupported format " << std::endl;
            return 1;
        }
    }
    return 0;
}
class warping{

};

int reg(program_option& po)
{
    tipl::image<3> from,to,from2,to2;
    tipl::vector<3> from_vs,to_vs;
    tipl::matrix<4,4> from_trans,to_trans;
    tipl::image<3,tipl::vector<3> > t2f_dis,f2t_dis,to2from,from2to;


    if(po.has("warp"))
    {
        gz_mat_read in;
        if(!in.load_from_file(po.get("warp").c_str()))
        {
            std::cout << "ERROR: cannot open or parse warp file " << po.get("warp") << std::endl;
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
            std::cout << "ERROR: invalid warp file " << po.get("warp") << std::endl;
            return 1;
        }
        to2from.resize(to_dim);
        std::copy(to2from_ptr,to2from_ptr+to2from.size()*3,&to2from[0][0]);
        from2to.resize(from_dim);
        std::copy(from2to_ptr,from2to_ptr+from2to.size()*3,&from2to[0][0]);
        if(po.has("apply_warp"))
            return after_warp(po.get("apply_warp"),to2from,from2to,to_vs,from_trans,to_trans);
        return 0;
    }
    if(!gz_nifti::load_from_file(po.get("from").c_str(),from,from_vs,from_trans))
    {
        std::cout << "cannot load from file" << std::endl;
        return 1;
    }
    if(!gz_nifti::load_from_file(po.get("to").c_str(),to,to_vs,to_trans))
    {
        std::cout << "cannot load template file" << std::endl;
        return 1;
    }
    if(po.has("from2") && po.has("to2"))
    {
        if(!gz_nifti::load_from_file(po.get("from2").c_str(),from2,from_vs))
        {
            std::cout << "cannot load from2 file" << std::endl;
            return 1;
        }
        if(!gz_nifti::load_from_file(po.get("to2").c_str(),to2,to_vs))
        {
            std::cout << "cannot load template file" << std::endl;
            return 1;
        }
    }

    if(!from2.empty() && from.shape() != from2.shape())
    {
        std::cout << "from2 image has a dimension different from from image" << std::endl;
        return 1;
    }
    if(!to2.empty() && to.shape() != to2.shape())
    {
        std::cout << "to2 image has a dimension different from tolate image" << std::endl;
        return 1;
    }

    std::string output_wp_image = po.get("output",po.get("from")+".wp.nii.gz");
    bool terminated = false;
    tipl::transformation_matrix<double> T;
    std::cout << "running linear registration." << std::endl;

    tipl::reg::two_way_linear_mr<tipl::reg::mutual_information>(to,to_vs,from,from_vs,T,
                                 po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,
                                 // 0: rigid body rotation 1:nonlinear
                                 terminated);

    std::cout << T;
    tipl::image<3> from_(to.shape()),from2_;


    if(is_label_image(from))
        tipl::resample_mt<tipl::interpolation::nearest>(from,from_,T);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(from,from_,T);


    if(!from2.empty())
    {
        from2_.resize(to.shape());
        if(is_label_image(from2))
            tipl::resample_mt<tipl::interpolation::nearest>(from2,from2_,T);
        else
            tipl::resample_mt<tipl::interpolation::cubic>(from2,from2_,T);
    }
    auto r2 = tipl::correlation(from_.begin(),from_.end(),to.begin());
    std::cout << "correlation cofficient: " << r2 << std::endl;
    if(po.get("reg_type",1) == 0) // just rigidbody
    {
        std::cout << "output warpped image:" << output_wp_image << std::endl;
        gz_nifti::save_to_file(output_wp_image.c_str(),from_,to_vs,to_trans);
        return 0;
    }

    if(po.get("normalize_signal",1))
    {
        std::cout << "normalizing signals" << std::endl;
        tipl::reg::cdm_pre(from_,from2_,to,to2);
    }

    tipl::reg::cdm_param param;
    param.resolution = po.get("resolution",param.resolution);
    param.constraint = po.get("constraint",param.constraint);
    param.iterations = po.get("iteration",param.iterations);
    param.min_dimension = po.get("min_dimension",param.min_dimension);

    if(!from2_.empty())
    {
        std::cout << "nonlinear registration using dual image modalities" << std::endl;
        tipl::reg::cdm2(to,to2,from_,from2_,t2f_dis,terminated,param);
    }
    else
    {
        std::cout << "nonlinear registration using single image modality" << std::endl;
        tipl::reg::cdm(to,from_,t2f_dis,terminated);
    }


    tipl::displacement_to_mapping(t2f_dis,to2from,T);

    // calculate inverted to2from
    {
        from2to.resize(from.shape());
        tipl::invert_displacement(t2f_dis,f2t_dis);
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,T);
    }

    {
        std::cout << "compose output images" << std::endl;
        tipl::image<3> from_wp;
        if(is_label_image(from))
            tipl::compose_mapping<tipl::interpolation::nearest>(from,to2from,from_wp);
        else
            tipl::compose_mapping<tipl::interpolation::cubic>(from,to2from,from_wp);

        float r = float(tipl::correlation(to.begin(),to.end(),from_wp.begin()));
        std::cout << "R2: " << r*r << std::endl;
        if(!gz_nifti::save_to_file(output_wp_image.c_str(),from_wp,to_vs,to_trans))
        {
            std::cout << "ERROR: cannot write warpped image to " << output_wp_image << std::endl;
            return 0;
        }
        else
            std::cout << "output warpped image to " << output_wp_image << std::endl;
    }

    if(po.has("output_warp"))
    {
        std::string filename = po.get("output_warp");
        if(!QString(filename.c_str()).endsWith(".wp.gz"))
            filename += ".map.gz";
        gz_mat_write out(filename.c_str());
        if(!out)
        {
            std::cout << "ERROR: cannot write to " << filename << std::endl;
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
        std::cout << "save mapping to " << filename << std::endl;
    }

    if(po.has("apply_warp"))
        return after_warp(po.get("apply_warp"),to2from,from2to,to_vs,from_trans,to_trans);
    return 0;
}
