#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "program_option.hpp"
bool apply_warping(const char* from,
                   const char* to,
                   tipl::geometry<3> from_geo,
                   tipl::geometry<3> to_geo,
                   tipl::image<tipl::vector<3>,3>& dis,
                   tipl::vector<3> Itvs,
                   tipl::matrix<4,4,float>& ItR,
                   tipl::transformation_matrix<double>& T,
                   std::string& error);
bool is_label_image(const tipl::image<float,3>& I);
int reg(void)
{
    tipl::image<float,3> from,to;
    tipl::image<float,3> from2,to2;
    tipl::vector<3> from_vs,to_vs;
    tipl::matrix<4,4,float> to_trans;

    if(!gz_nifti::load_from_file(po.get("from").c_str(),from,from_vs))
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

    if(!from2.empty() && from.geometry() != from2.geometry())
    {
        std::cout << "from2 image has a dimension different from from image" << std::endl;
        return 1;
    }
    if(!to2.empty() && to.geometry() != to2.geometry())
    {
        std::cout << "to2 image has a dimension different from tolate image" << std::endl;
        return 1;
    }

    std::string output_wp_image = po.get("output",po.get("from")+".wp.nii.gz");
    bool terminated = false;
    tipl::transformation_matrix<double> T;
    std::cout << "running linear registration." << std::endl;

    tipl::reg::two_way_linear_mr(to,to_vs,from,from_vs,T,
                                 po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,
                                 // 0: rigid body rotation 1:nonlinear
                                 tipl::reg::mutual_information(),
                                 terminated);

    std::cout << T;
    tipl::image<float,3> from_(to.geometry()),from2_;


    tipl::resample_mt(from,from_,T,is_label_image(from) ? tipl::nearest : tipl::cubic);


    if(!from2.empty())
    {
        from2_.resize(to.geometry());
        tipl::resample_mt(from2,from2_,T,is_label_image(from2) ? tipl::nearest : tipl::cubic);
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

    tipl::image<tipl::vector<3>,3> cdm_dis;
    if(!from2_.empty())
    {
        tipl::reg::cdm_param param;
        param.resolution = po.get("resolution",param.resolution);
        param.cdm_smoothness = po.get("smoothness",param.cdm_smoothness);
        param.contraint = po.get("constraint",param.contraint);
        param.iterations = po.get("iteration",param.iterations);
        param.min_dimension = po.get("min_dimension",param.min_dimension);
        std::cout << "nonlinear registration using dual image modalities" << std::endl;
        tipl::reg::cdm2(to,to2,from_,from2_,cdm_dis,terminated,param);
    }
    else
    {
        std::cout << "nonlinear registration using single image modality" << std::endl;
        tipl::reg::cdm(to,from_,cdm_dis,terminated);
    }

    {
        std::cout << "compose output images" << std::endl;
        tipl::image<float,3> from_wp;
        tipl::compose_displacement_with_affine(from,from_wp,T,cdm_dis,is_label_image(from) ? tipl::nearest : tipl::cubic);
        float r = float(tipl::correlation(to.begin(),to.end(),from_wp.begin()));
        std::cout << "R2: " << r*r << std::endl;
        std::cout << "output warpped image: " << output_wp_image << std::endl;
        gz_nifti::save_to_file(output_wp_image.c_str(),from_wp,to_vs,to_trans);
    }

    if(po.has("apply_warp"))
    {
        std::istringstream in(po.get("apply_warp"));
        std::string from3;
        while(std::getline(in,from3,','))
        {
            std::string to3 = from3+".wp.nii.gz";
            std::string error;
            std::cout << "apply warping to " << from3 << std::endl;
            if(!apply_warping(from3.c_str(),to3.c_str(),
                          from.geometry(),
                          to.geometry(),
                          cdm_dis,
                          to_vs,
                          to_trans,
                          T,error))
            {
                std::cout << "[ERROR]" << error <<std::endl;
                return 1;
            }
        }
    }
    return 0;
}
