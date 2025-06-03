#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include "zlib.h"
#include "TIPL/tipl.hpp"
extern bool has_cuda;

template<int dim>
inline auto subject_image_pre(tipl::image<dim>&& I)
{
    tipl::filter::gaussian(I);
    tipl::segmentation::normalize_otsu_median(I,255.99f);
    return tipl::image<dim,unsigned char>(I);
}
template<int dim>
inline auto subject_image_pre(const tipl::image<dim>& I)
{
    return subject_image_pre(tipl::image<dim>(I));
}
template<int dim>
inline auto template_image_pre(tipl::image<dim>&& I)
{
    tipl::image<dim,unsigned char> out;
    tipl::normalize_upper_lower2(I,out,255.999f);
    return out;
}
template<int dim>
inline auto template_image_pre(const tipl::image<dim>& I)
{
    return template_image_pre(tipl::image<dim>(I));
}

extern int map_ver;
struct dual_reg{
    static constexpr int dimension = 3;
    static constexpr int max_modality = 16;
    using image_type = tipl::image<dimension,unsigned char>;
    using mapping_type = tipl::image<dimension,tipl::vector<dimension> >;
    tipl::affine_transform<float,dimension> arg;
    const float (*bound)[8] = tipl::reg::reg_bound;
    tipl::reg::cost_type cost_type = tipl::reg::corr;
    tipl::reg::reg_type reg_type = tipl::reg::affine;
    tipl::reg::cdm_param param;
public:
    bool It_is_mni = true;
    bool Is_is_mni = false;
    bool export_intermediate = false;
    bool use_cuda = true;
    bool skip_linear = false;
    bool skip_nonlinear = false;
    bool match_fov = true;
public:
    dual_reg(void):modality_names(max_modality),I(max_modality),J(max_modality),It(max_modality),r(max_modality)
    {
    }
    std::vector<std::string> modality_names;
    std::vector<image_type> I,J,It;
    std::vector<float> r;
    mapping_type t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<dimension> Itvs,Ivs;
    tipl::matrix<dimension+1,dimension+1> ItR,IR;
    tipl::shape<dimension> Its,Is;
public:
    std::vector<tipl::vector<3> > anchor[2];
public:
    mapping_type previous_t2f,previous_f2t;
    std::vector<image_type> previous_It;
public:
    mutable std::string error_msg;
public:
    void clear(void);
    void clear_reg(void);
    bool save_subject(const std::string& file_name);
    bool save_template(const std::string& file_name);
    bool load_subject(size_t id,const std::string& file_name);
    bool load_template(size_t id,const std::string& file_name);
    void match_resolution(bool use_vs);
public:
    auto T(void) const  {return tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);}
    auto invT(void) const{auto t = tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);t.inverse();return t;}
    bool data_ready(void) const
    {
        return !I[0].empty() && !It[0].empty();
    }
public:
    void show_r(const std::string& prompt);
    void compute_mapping_from_displacement(void);
    void calculate_linear_r(void);
    void It_match_fov(void);
    void calculate_nonlinear_r(void);
public:
    float linear_reg(bool& terminated);
    void nonlinear_reg(bool& terminated);
public:

    template<bool direction,tipl::interpolation itype,typename Itype>
    auto apply_warping(const Itype& input) const
    {
        const auto& mapping = direction ? to2from : from2to;
        return mapping.empty() ?
            tipl::resample<itype>(input, direction ? Its : Is, direction ? T() : invT()) :
            tipl::compose_mapping<itype>(input, mapping);
    }
    template<bool direction>
    bool apply_warping(const char* input,const char* output) const
    {
        if(tipl::ends_with(input,".tt.gz"))
            return apply_warping_tt<direction>(input,output);
        if(tipl::ends_with(input,".nii.gz") || tipl::ends_with(input,".nii"))
            return apply_warping_nii<direction>(input,output);
        if(tipl::ends_with(input,".sz") || tipl::ends_with(input,".src.gz") ||
           tipl::ends_with(input,".fz") || tipl::ends_with(input,".fib.gz"))
            return apply_warping_fzsz<direction>(input,output);
        error_msg = "unsupported file format";
        return false;
    }
    template<bool direction>
    bool apply_warping_tt(const char* input,const char* output) const;
    template<bool direction>
    bool apply_warping_nii(const char* input,const char* output) const;
    template<bool direction>
    bool apply_warping_fzsz(const char* input,const char* output) const;
    bool load_warping(const std::string& filename);
    bool load_alternative_warping(const std::string& filename);
    bool save_warping(const char* filename) const;
public:
    void dis_to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R);
    void to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R);
    void to_I_space(const tipl::shape<3>& new_Is,const tipl::matrix<4,4>& new_IR);
    void to_It_space(const tipl::shape<3>& new_Its,const tipl::matrix<4,4>& new_ItR);
    void to_It_space(const tipl::shape<3>& new_Its);


};



#endif//REG_HPP
