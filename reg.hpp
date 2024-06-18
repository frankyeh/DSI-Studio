#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include "zlib.h"
#include "TIPL/tipl.hpp"
extern bool has_cuda;
void cdm2_cuda(const tipl::image<3>& It,
               const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& d,
               tipl::image<3,tipl::vector<3> >& inv_d,
               bool& terminated,
               tipl::reg::cdm_param param);

inline void cdm_common(const tipl::image<3>& It,
         const tipl::image<3>& It2,
               const tipl::image<3>& Is,
               const tipl::image<3>& Is2,
               tipl::image<3,tipl::vector<3> >& dis,
               tipl::image<3,tipl::vector<3> >& inv_dis,
               bool& terminated,
               tipl::reg::cdm_param param = tipl::reg::cdm_param(),
               bool use_cuda = true)
{
    if(use_cuda && has_cuda)
    {
        if constexpr (tipl::use_cuda)
        {
            cdm2_cuda(It,It2,Is,Is2,dis,inv_dis,terminated,param);
            return;
        }
    }
    tipl::reg::cdm2(It,It2,Is,Is2,dis,inv_dis,terminated,param);
}


template<typename image_type>
tipl::vector<3> adjust_to_vs(const image_type& from,
               const tipl::vector<3>& from_vs,
               const image_type& to,
               const tipl::vector<3>& to_vs)
{
    auto from_otsu = tipl::segmentation::otsu_threshold(from)*0.6f;
    auto to_otsu = tipl::segmentation::otsu_threshold(to)*0.6f;
    tipl::vector<3> from_min,from_max,to_min,to_max;
    tipl::bounding_box(from,from_min,from_max,from_otsu);
    tipl::bounding_box(to,to_min,to_max,to_otsu);
    from_max -= from_min;
    to_max -= to_min;
    tipl::vector<3> new_vs(to_vs);
    float rx = (to_max[0] > 0.0f) ? from_max[0]*from_vs[0]/(to_max[0]*to_vs[0]) : 1.0f;
    float ry = (to_max[1] > 0.0f) ? from_max[1]*from_vs[1]/(to_max[1]*to_vs[1]) : 1.0f;

    new_vs[0] *= rx;
    new_vs[1] *= ry;
    new_vs[2] *= (rx+ry)*0.5f; // z direction bounding box is largely affected by slice number, thus use rx and ry
    return new_vs;
}

size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,float> > reg,
                     bool& terminated);
size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                     bool& terminated);
size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,float> > reg,
                     bool& terminated);
size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char> > reg,
                     bool& terminated);
template<typename value_type>
inline size_t linear_refine(std::vector<tipl::const_pointer_image<3,value_type> > from,
                                    tipl::vector<3> from_vs,
                                    std::vector<tipl::const_pointer_image<3,value_type> > to,
                                    tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,tipl::reg::cost_type cost_type = tipl::reg::mutual_info)
{
    auto reg = tipl::reg::linear_reg(from,from_vs,to,to_vs,arg);
    reg->line_search = false;
    reg->type = reg_type;
    reg->set_bound(tipl::reg::narrow_bound,false);
    size_t result = 0;
    if constexpr (tipl::use_cuda)
    {
        if(has_cuda && cost_type == tipl::reg::mutual_info)
            result = optimize_mi_cuda(reg,terminated);
    }
    if(!result)
        result = (cost_type == tipl::reg::mutual_info ? reg->optimize<tipl::reg::mutual_information>(terminated):
                                                        reg->optimize<tipl::reg::correlation>(terminated));
    tipl::out() << "refine registration" << std::endl;
    tipl::out() << arg;
    return result;
}
template<typename image_type>
inline auto make_list(const image_type& I,const image_type& I2)
{
    auto pI = tipl::make_shared(I);
    if(I2.empty())
        return std::vector<decltype(pI)>({pI});
    auto pI2 = tipl::make_shared(I2);
    return std::vector<decltype(pI)>({pI,pI2});
}
template<typename image_type>
inline auto make_list(const image_type& I)
{
    auto pI = tipl::make_shared(I);
    return std::vector<decltype(pI)>({pI});
}
template<typename value_type>
size_t linear(std::vector<tipl::const_pointer_image<3,value_type> > from,
                             tipl::vector<3> from_vs,
                             std::vector<tipl::const_pointer_image<3,value_type> > to,
                             tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound,
                              tipl::reg::cost_type cost_type = tipl::reg::mutual_info)
{
    auto new_to_vs = to_vs;
    if(reg_type == tipl::reg::affine)
        new_to_vs = adjust_to_vs(from[0],from_vs,to[0],to_vs);
    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from[0],from_vs,to[0],to_vs).to_affine_transform(arg,from[0],from_vs,to[0],new_to_vs);
    float result = 0.0f;

    auto reg = tipl::reg::linear_reg(from,from_vs,to,new_to_vs,arg);
    reg->type = reg_type;
    reg->set_bound(bound);

    if constexpr (tipl::use_cuda)
    {
        if(has_cuda && cost_type == tipl::reg::mutual_info)
            result = optimize_mi_cuda_mr(reg,terminated);
    }
    if(result == 0.0f)
    {
        result = (cost_type == tipl::reg::mutual_info ? reg->optimize_mr<tipl::reg::mutual_information>(terminated):
                                                        reg->optimize_mr<tipl::reg::correlation>(terminated));
        tipl::out() << arg;

    }

    if(new_to_vs != to_vs)
        tipl::transformation_matrix<float>(arg,from[0],from_vs,to[0],new_to_vs).to_affine_transform(arg,from[0],from_vs,to[0],to_vs);
    tipl::out() << "initial registration" << std::endl;
    tipl::out() << arg << std::endl;
    return linear_refine(from,from_vs,to,to_vs,arg,reg_type,terminated,cost_type);
}
template<typename value_type>
tipl::transformation_matrix<float> linear(std::vector<tipl::const_pointer_image<3,value_type> > from,
                                          tipl::vector<3> from_vs,
                                          std::vector<tipl::const_pointer_image<3,value_type> > to,
                                          tipl::vector<3> to_vs,
                                          tipl::reg::reg_type reg_type,
                                          bool& terminated,
                                          const float* bound = tipl::reg::reg_bound,
                                          tipl::reg::cost_type cost_type = tipl::reg::mutual_info)
{
    tipl::affine_transform<float> arg;
    linear(from,from_vs,to,to_vs,arg,reg_type,terminated,bound,cost_type);
    return tipl::transformation_matrix<float>(arg,from[0],from_vs,to[0],to_vs);
}


struct dual_reg{

    tipl::affine_transform<float> arg;
    const float* bound = tipl::reg::reg_bound;
    tipl::reg::cdm_param param;

    tipl::image<3> It,I,J,JJ,I2,It2,J2;
    tipl::image<3,tipl::vector<3> > t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<3> Itvs,Ivs;
    tipl::matrix<4,4> ItR,IR;
    bool It_is_mni;
    bool export_intermediate = false;
    mutable std::string error_msg;
    void clear(void)
    {
        J.clear();
        JJ.clear();
        J2.clear();
        t2f_dis.clear();
        to2from.clear();
        f2t_dis.clear();
        from2to.clear();
        arg.clear();
    }
    void inv_warping(void)
    {
        ItR.swap(IR);
        std::swap(Itvs,Ivs);
        to2from.swap(from2to);
    }
    bool load_subject(const char* file_name);
    bool load_subject2(const char* file_name);
    bool load_template(const char* file_name);
    bool load_template2(const char* file_name);
    auto T(void) const
    {
        return tipl::transformation_matrix<float>(arg,It.shape(),Itvs,I.shape(),Ivs);
    }
    const auto& show_subject(bool second) const
    {
        return second && I2.shape() == I.shape() ? I2 : I;
    }
    const auto& show_template(bool second) const
    {
        return second && It2.shape() == It.shape() ? It2 : It;
    }
    const auto& show_subject_warped(bool second) const
    {
        return (second && I2.shape() == I.shape() ? (J2.empty() ? I2:J2) : (J.empty() ? I:J));
    }
    bool data_ready(void) const
    {
        return !I.empty() && !It.empty();
    }
    void skip_linear(void);
    void match_resolution(bool rigid_body);
    float linear_reg(tipl::reg::reg_type reg_type,tipl::reg::cost_type cost_type,bool& terminated);
    float nonlinear_reg(bool& terminated,bool use_cuda = true);
    void matching_contrast(void);
public:
    void apply_warping(const tipl::image<3>& from,tipl::image<3>& to,bool is_label) const;
    bool apply_warping(const char* from,const char* to) const;
    bool apply_warping_tt(const char* from,const char* to) const;
    bool load_warping(const char* filename);
    bool save_warping(const char* filename) const;
    bool save_transformed_image(const char* filename) const;
};



#endif//REG_HPP
