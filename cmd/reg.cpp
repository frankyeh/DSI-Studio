#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "program_option.hpp"

int reg(void)
{
    tipl::image<float,3> subject,temp;
    tipl::image<float,3> subject2,temp2;
    tipl::vector<3> subject_vs,temp_vs;
    tipl::matrix<4,4,float> temp_trans;

    if(!gz_nifti::load_from_file(po.get("subject").c_str(),subject,subject_vs))
    {
        std::cout << "Cannot load subject file" << std::endl;
        return 1;
    }
    if(!gz_nifti::load_from_file(po.get("template").c_str(),temp,temp_vs,temp_trans))
    {
        std::cout << "Cannot load template file" << std::endl;
        return 1;
    }
    if(po.has("subject2") && po.has("template2"))
    {
        if(!gz_nifti::load_from_file(po.get("subject2").c_str(),subject2,subject_vs))
        {
            std::cout << "Cannot load subject2 file" << std::endl;
            return 1;
        }
        if(!gz_nifti::load_from_file(po.get("template2").c_str(),temp2,temp_vs))
        {
            std::cout << "Cannot load template2 file" << std::endl;
            return 1;
        }
    }

    if(!subject2.empty() && subject.geometry() != subject2.geometry())
    {
        std::cout << "subject2 image has a dimension different from subject image" << std::endl;
        return 1;
    }
    if(!temp2.empty() && temp.geometry() != temp2.geometry())
    {
        std::cout << "template2 image has a dimension different from template image" << std::endl;
        return 1;
    }

    std::string output_wp_image = po.get("warpped_image",po.get("subject")+".wp.nii.gz");
    bool terminated = false;
    tipl::transformation_matrix<double> T;
    std::cout << "running linear registration." << std::endl;

    tipl::reg::two_way_linear_mr(subject,subject_vs,temp,temp_vs,T,
                                 po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine,
                                 // 0: rigid body rotation 1:nonlinear
                                 tipl::reg::mutual_information(),
                                 terminated);

    std::cout << T << std::endl;
    tipl::image<float,3> subject_(temp.geometry()),subject2_;
    tipl::resample_mt(subject,subject_,T,tipl::cubic);
    if(!subject2.empty())
    {
        subject2_.resize(temp.geometry());
        tipl::resample_mt(subject2,subject2_,T,tipl::cubic);
    }
    auto r2 = tipl::correlation(subject_.begin(),subject_.end(),temp.begin());
    std::cout << "correlation cofficient=" << r2 << std::endl;
    std::cout << "running nonlinear registration." << std::endl;

    if(po.get("reg_type",1) == 0) // just rigidbody
    {
        std::cout << "output warpped image:" << output_wp_image << std::endl;
        gz_nifti::save_to_file(output_wp_image.c_str(),subject_,temp_vs,temp_trans);
        return 0;
    }

    if(po.get("normalize_signal",1))
    {
        std::cout << "normalizing signals for nonlinear registration" << std::endl;
        tipl::reg::cdm_pre(subject_,subject2_,temp,temp2);
    }

    tipl::image<tipl::vector<3>,3> cdm_dis;
    if(!subject2_.empty())
    {
        std::cout << "Normalization using dual image modalities" << std::endl;
        tipl::reg::cdm2(subject_,subject2_,temp,temp2,cdm_dis,terminated,
                        po.get("resolution",2.0f),po.get("smoothness",0.3f),po.get("steps",64));
    }
    else
    {
        std::cout << "Normalization using single image modality" << std::endl;
        tipl::reg::cdm(subject_,temp,cdm_dis,terminated);
    }

    {
        tipl::image<float,3> subject_wp;
        tipl::compose_displacement(subject_,cdm_dis,subject_wp);
        float r = float(tipl::correlation(temp.begin(),temp.end(),subject_wp.begin()));
        std::cout << "R2=" << r*r << std::endl;
        std::cout << "output warpped image:" << output_wp_image << std::endl;
        gz_nifti::save_to_file(output_wp_image.c_str(),subject_,temp_vs,temp_trans);
    }
    return 0;
}
