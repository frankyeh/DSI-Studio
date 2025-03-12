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
    void calculate_nonlinear_r(void);
public:
    float linear_reg(bool& terminated);
    void nonlinear_reg(bool& terminated);
public:

    template<bool direction>
    bool apply_warping(const std::vector<std::string>& apply_warp_filename,const std::string& post_fix,const std::string& output_dir)
    {
        bool result = true;
        for(const auto& each_file: apply_warp_filename)
        {
            std::string output_file;
            if (!output_dir.empty())
            {
                std::filesystem::create_directories(output_dir);
                output_file = (std::filesystem::path(output_dir) / (std::filesystem::path(each_file).filename().string() + post_fix)).string();
            }
            else
                output_file = each_file + post_fix;

            tipl::out() << (direction ? "warping " : "unwarping") << each_file << " into " << (each_file+post_fix);
            if(!apply_warping<direction>(each_file.c_str(),output_file.c_str()))
            {
                tipl::error() << error_msg;
                result = false;
            }
        }
        return result;
    }

    template<bool direction,tipl::interpolation itype,typename Itype>
    auto apply_warping(const Itype& input) const
    {
        const auto& mapping = direction ? to2from : from2to;
        return mapping.empty() ?
            tipl::resample<itype>(input, direction ? Its : Is, direction ? T() : invT()) :
            tipl::compose_mapping<itype>(input, mapping);
    }
    template<bool direction>
    auto apply_warping(const char* input) const
    {
        tipl::out() << "opening " << input;
        auto input_size = (direction ? Is : Its);
        // check dimension
        {
            tipl::io::gz_nifti nii;
            if(!nii.load_from_file(input))
            {
                error_msg = nii.error_msg;
                return tipl::image<dimension>();
            }
            tipl::shape<3> d;
            nii.get_image_dimension(d);
            if(d != input_size)
            {
                tipl::warning() << std::filesystem::path(input).filename().string() << " has a size of " << d << " different from the expected size " << input_size;
                tipl::warning() << "transformation will be applied. But please make sure you apply the mapping in the correct direction.";
            }
        }
        tipl::image<dimension> I3(input_size);
        if(!tipl::io::gz_nifti::load_to_space(input,I3,direction ? IR : ItR))
        {
            error_msg = "cannot open " + std::string(input);
            return tipl::image<dimension>();
        }
        bool is_label = tipl::is_label_image(I3);
        tipl::out() << (is_label ? "label image interpolated using majority assignment " : "scalar image interpolated using spline") << std::endl;
        if(is_label)
            return apply_warping<direction,tipl::interpolation::majority>(I3);
        else
            return apply_warping<direction,tipl::interpolation::cubic>(I3);
    }

    template<bool direction>
    bool apply_warping(const char* input,const char* output) const
    {
        if(tipl::ends_with(input,"tt.gz"))
            return apply_warping_tt<direction>(input,output);
        auto I = apply_warping<direction>(input);
        if(I.empty())
            return false;
        tipl::out() << "saving " << output;
        if(!tipl::io::gz_nifti::save_to_file(output,I,
                                             direction ? Itvs : Ivs,
                                             direction ? ItR : IR,
                                             direction ? It_is_mni : Is_is_mni))
        {
            error_msg = "cannot write to file " + std::string(output);
            return false;
        }
        return true;
    }
    template<bool direction>
    bool apply_warping_tt(const char* input,const char* output) const;
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
