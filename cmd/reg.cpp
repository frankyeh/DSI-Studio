#include <QString>
#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "tract_model.hpp"
#include "program_option.hpp"
bool apply_warping(const char* from,
                   const char* to,
                   tipl::image<tipl::vector<3>,3>& to2from,
                   tipl::vector<3> Itvs,
                   const tipl::matrix<4,4,float>& ItR,
                   std::string& error,
                   tipl::interpolation_type interpo);
bool apply_unwarping_tt(const char* from,
                        const char* to,
                        const tipl::image<tipl::vector<3>,3>& from2to,
                        tipl::shape<3> new_geo,
                        tipl::vector<3> new_vs,
                        const tipl::matrix<4,4,float>& new_trans_to_mni,
                        std::string& error);
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
bool is_label_image(const tipl::image<float,3>& I);


int after_warp(tipl::image<tipl::vector<3>,3>& to2from,
               tipl::image<tipl::vector<3>,3>& from2to,
               tipl::vector<3> to_vs,
               const tipl::matrix<4,4,float>& to_trans)
{
    std::string error;
    if(po.has("apply_warp"))
    {
        std::vector<std::string> filenames;
        get_filenames_from("apply_warp",filenames);
        for(auto filename: filenames)
        {
            if(QString(filename.c_str()).toLower().endsWith(".nii.gz"))
            {
                std::string filename_warp = filename+".wp.nii.gz";
                std::cout << "apply warping to " << filename << std::endl;
                if(!apply_warping(filename.c_str(),filename_warp.c_str(),to2from,to_vs,to_trans,error,(po.get("interpolation",1) ? tipl::cubic : tipl::linear)))
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
    }
    return 0;
}
class warping{

};

int reg(void)
{
    tipl::image<float,3> from,to,from2,to2;
    tipl::vector<3> from_vs,to_vs;
    tipl::matrix<4,4,float> from_trans,to_trans;
    tipl::image<tipl::vector<3>,3> t2f_dis,f2t_dis,to2from,from2to;


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
        return after_warp(to2from,from2to,to_vs,to_trans);
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
    auto interpo_method = (po.get("interpolation",1) ? tipl::cubic : tipl::linear);
    tipl::transformation_matrix<double> T;
    std::cout << "running linear registration." << std::endl;

    tipl::reg::two_way_linear_mr(to,to_vs,from,from_vs,T,
                                 po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,
                                 // 0: rigid body rotation 1:nonlinear
                                 tipl::reg::mutual_information(),
                                 terminated);

    std::cout << T;
    tipl::image<float,3> from_(to.shape()),from2_;


    tipl::resample_mt(from,from_,T,is_label_image(from) ? tipl::nearest : interpo_method);


    if(!from2.empty())
    {
        from2_.resize(to.shape());
        tipl::resample_mt(from2,from2_,T,is_label_image(from2) ? tipl::nearest : interpo_method);
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
    param.cdm_smoothness = po.get("smoothness",param.cdm_smoothness);
    param.contraint = po.get("constraint",param.contraint);
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
        tipl::image<float,3> from_wp;
        tipl::compose_mapping(from,to2from,from_wp,is_label_image(from) ? tipl::nearest : interpo_method);
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


    return after_warp(to2from,from2to,to_vs,to_trans);

}
