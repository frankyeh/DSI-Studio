#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include "zlib.h"
#include "TIPL/tipl.hpp"
extern bool has_cuda;
void cdm_cuda(const std::vector<tipl::const_pointer_image<3,unsigned char> >& It,
               const std::vector<tipl::const_pointer_image<3,unsigned char> >& Is,
               tipl::image<3,tipl::vector<3> >& d,
               bool& terminated,
               tipl::reg::cdm_param param);
void cdm_cuda(const std::vector<tipl::const_pointer_image<2,unsigned char> >& It,
               const std::vector<tipl::const_pointer_image<2,unsigned char> >& Is,
               tipl::image<2,tipl::vector<2> >& d,
               bool& terminated,
               tipl::reg::cdm_param param);

template<int dim>
inline void cdm_common(std::vector<tipl::const_pointer_image<dim,unsigned char> > It,
                       std::vector<tipl::const_pointer_image<dim,unsigned char> > Is,
                       tipl::image<dim,tipl::vector<dim> >& dis,
                       bool& terminated,
                       tipl::reg::cdm_param param = tipl::reg::cdm_param(),
                       bool use_cuda = true)
{
    if(It.size() < Is.size())
        Is.resize(It.size());
    if(Is.size() < It.size())
        It.resize(Is.size());
    if(use_cuda && has_cuda)
    {
        if constexpr (tipl::use_cuda)
        {
            cdm_cuda(It,Is,dis,terminated,param);
            return;
        }
    }
    tipl::reg::cdm(It,Is,dis,terminated,param);
}


template<typename image_type,int dim>
tipl::vector<dim> adjust_to_vs(const image_type& from,
               const tipl::vector<dim>& from_vs,
               const image_type& to,
               const tipl::vector<dim>& to_vs)
{
    auto from_otsu = tipl::segmentation::otsu_threshold(from)*0.6f;
    auto to_otsu = tipl::segmentation::otsu_threshold(to)*0.6f;
    tipl::vector<dim> from_min,from_max,to_min,to_max;
    tipl::bounding_box(from,from_min,from_max,from_otsu);
    tipl::bounding_box(to,to_min,to_max,to_otsu);
    from_max -= from_min;
    to_max -= to_min;
    tipl::vector<dim> new_vs(to_vs);
    float rx = (to_max[0] > 0.0f) ? from_max[0]*from_vs[0]/(to_max[0]*to_vs[0]) : 1.0f;
    float ry = (to_max[1] > 0.0f) ? from_max[1]*from_vs[1]/(to_max[1]*to_vs[1]) : 1.0f;

    new_vs[0] *= rx;
    new_vs[1] *= ry;
    if constexpr(dim == 3)
        new_vs[2] *= (rx+ry)*0.5f; // z direction bounding box is largely affected by slice number, thus use rx and ry
    return new_vs;
}

size_t optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                     tipl::reg::cost_type cost_type,bool& terminated);
size_t optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                     tipl::reg::cost_type cost_type,bool& terminated);
template<int dim>
inline size_t linear_refine(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                            tipl::vector<dim> from_vs,
                            std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                            tipl::vector<dim> to_vs,
                            tipl::affine_transform<float,dim>& arg,
                            tipl::reg::reg_type reg_type,
                            bool& terminated = tipl::prog_aborted,
                            tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                            bool use_cuda = true)
{
    auto reg = tipl::reg::linear_reg<tipl::progress>(from,from_vs,to,to_vs,arg);
    reg->set_bound(reg_type,tipl::reg::narrow_bound,false);
    size_t result = 0;
    if constexpr (tipl::use_cuda && dim == 3)
    {
        if(has_cuda && use_cuda && cost_type == tipl::reg::mutual_info)
            result = optimize_mi_cuda(reg,cost_type,terminated);
    }
    if(!result)
        result = (cost_type == tipl::reg::mutual_info ? reg->template optimize<tipl::reg::mutual_information<dim> >(terminated):
                                                        reg->template optimize<tipl::reg::correlation>(terminated));
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
template<int dim>
size_t linear(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                             tipl::vector<dim> from_vs,
                             std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                             tipl::vector<dim> to_vs,
                              tipl::affine_transform<float,dim>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated = tipl::prog_aborted,
                              const float* bound = tipl::reg::reg_bound,
                              tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                              bool use_cuda = true)
{
    auto new_to_vs = to_vs;
    if constexpr(dim == 3)
    {
        if(reg_type == tipl::reg::affine)
            new_to_vs = adjust_to_vs(from[0],from_vs,to[0],to_vs);
        if(new_to_vs != to_vs)
            tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],to_vs).
                    to_affine_transform(arg,from[0],from_vs,to[0],new_to_vs);
    }
    float result = 0.0f;

    auto reg = tipl::reg::linear_reg<tipl::progress>(from,from_vs,to,new_to_vs,arg);
    reg->set_bound(reg_type,bound);

    if constexpr (tipl::use_cuda && dim == 3)
    {
        if(has_cuda && use_cuda && cost_type == tipl::reg::mutual_info)
            result = optimize_mi_cuda_mr(reg,cost_type,terminated);
    }
    if(result == 0.0f)
    {
        result = (cost_type == tipl::reg::mutual_info ? reg->template optimize_mr<tipl::reg::mutual_information<dim> >(terminated):
                                                        reg->template optimize_mr<tipl::reg::correlation>(terminated));
    }

    if constexpr(dim == 3)
    {
        if(new_to_vs != to_vs)
            tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],new_to_vs).to_affine_transform(arg,from[0],from_vs,to[0],to_vs);
    }
    tipl::out() << "initial registration" << std::endl;
    tipl::out() << arg << std::endl;
    return linear_refine(from,from_vs,to,to_vs,arg,reg_type,terminated,cost_type,use_cuda);
}
template<int dim>
tipl::transformation_matrix<float> linear(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                                          tipl::vector<dim> from_vs,
                                          std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                                          tipl::vector<dim> to_vs,
                                          tipl::reg::reg_type reg_type,
                                          bool& terminated = tipl::prog_aborted,
                                          const float* bound = tipl::reg::reg_bound,
                                          tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                                          bool use_cuda = true)
{
    tipl::affine_transform<float,dim> arg;
    linear(from,from_vs,to,to_vs,arg,reg_type,terminated,bound,cost_type,use_cuda);
    return tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],to_vs);
}

template<int dim>
inline auto subject_image_pre(tipl::image<dim>&& I)
{
    tipl::image<dim,unsigned char> out;
    tipl::filter::gaussian(I);
    tipl::segmentation::otsu_median_regulzried(I);
    tipl::normalize_upper_lower2(I,out,255.999f);
    return out;
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

template<int dim>
struct dual_reg{
    static constexpr int dimension = dim;
    using image_type = tipl::image<dimension,unsigned char>;
    tipl::affine_transform<float,dimension> arg;
    const float* bound = tipl::reg::reg_bound;
    tipl::reg::cdm_param param;

    image_type It,I,J,JJ,I2,It2,J2;
    tipl::image<dimension,tipl::vector<dimension> > t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<dimension> Itvs,Ivs;
    tipl::matrix<dimension+1,dimension+1> ItR,IR;
    bool It_is_mni;
    bool export_intermediate = false;
    bool use_cuda = true;
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
    void load_subject(tipl::image<dimension>&& in)
    {
        I = subject_image_pre(in);
    }
    void load_subject2(tipl::image<dimension>&& in)
    {
        I2 = subject_image_pre(in);
    }

    bool load_subject(const char* file_name);
    bool load_subject2(const char* file_name);
    bool load_template(const char* file_name);
    bool load_template2(const char* file_name);
    auto T(void) const
    {
        return tipl::transformation_matrix<float,dimension>(arg,It.shape(),Itvs,I.shape(),Ivs);
    }
    const auto& show_subject(bool second) const
    {
        return second ? I2 : I;
    }
    const auto& show_template(bool second) const
    {
        return second ? It2 : It;
    }
    const auto& show_subject_warped(bool second) const
    {
        return second ? J2 : J;
    }
    bool data_ready(void) const
    {
        return !I.empty() && !It.empty();
    }
    void skip_linear(void)
    {
        image_type J_,J2_;
        if(I.shape() == It.shape())
            J_ = I;
        else
        {
            J_.resize(It.shape());
            tipl::draw(I,J_,tipl::vector<dimension,int>());
        }

        if(I2.shape() == I.shape())
        {
            if(I.shape() == It.shape())
                J2_ = I2;
            else
            {
                J2_.resize(It.shape());
                tipl::draw(I2,J2_,tipl::vector<dimension,int>());
            }
        }
        arg.clear();
        J2.swap(J2_);
        J.swap(J_);
    }
    void match_resolution(bool rigid_body);
    inline float linear_reg(tipl::reg::reg_type reg_type,
                     tipl::reg::cost_type cost_type)
    {
        return linear_reg(reg_type,cost_type,tipl::prog_aborted);
    }
    float linear_reg(tipl::reg::reg_type reg_type,
                     tipl::reg::cost_type cost_type,bool& terminated)
    {
        tipl::progress prog("linear registration");
        if(export_intermediate)
        {
            tipl::io::gz_nifti::save_to_file("Template_QA.nii.gz",It,Itvs,ItR);
            if(!It2.empty())
                tipl::io::gz_nifti::save_to_file("Template_ISO.nii.gz",It2,Itvs,ItR);
            tipl::io::gz_nifti::save_to_file("Subject_QA.nii.gz",I,Ivs,IR);
            if(!I2.empty())
                tipl::io::gz_nifti::save_to_file("Subject_ISO.nii.gz",I2,Ivs,IR);
        }

        linear(make_list(It,It2),Itvs,make_list(I,I2),Ivs,
               arg,reg_type,terminated,bound,cost_type,use_cuda);

        auto trans = T();

        image_type J_,J2_;
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
    inline float nonlinear_reg(void)
    {
        return nonlinear_reg(tipl::prog_aborted);
    }
    float nonlinear_reg(bool& terminated)
    {
        tipl::progress prog("nonlinear registration");
        tipl::par_for(2,[&](int id)
        {
            if(id)
                cdm_common(make_list(It,It2),make_list(J,J2),t2f_dis,terminated,param,use_cuda);
            else
                cdm_common(make_list(J,J2),make_list(It,It2),f2t_dis,terminated,param,use_cuda);
        },2);
        if(export_intermediate)
        {
            tipl::image<dimension+1> buffer(It.shape().expand(2*dimension));
            tipl::par_for(2*dimension,[&](unsigned int d)
            {
                if(d < dimension)
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
        auto trans = T();
        from2to.resize(I.shape());
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
        tipl::displacement_to_mapping(t2f_dis,to2from,trans);
        tipl::compose_mapping(I,to2from,JJ);

        if(export_intermediate)
            JJ.template save_to_file<tipl::io::gz_nifti>("Subject_QA_nonlinear_reg.nii.gz");

        auto r = tipl::correlation(JJ,It);
        tipl::out() << "nonlinear: " << r;
        return r;
    }
    void matching_contrast(void)
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
public:
    void apply_warping(const tipl::image<dimension>& from,tipl::image<dimension>& to,bool is_label) const
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
    bool apply_warping(const char* from,const char* to) const;
    bool apply_warping_tt(const char* from,const char* to) const;
    bool load_warping(const char* filename)
    {
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(filename))
        {
            error_msg = "cannot read file ";
            error_msg += filename;
            return false;
        }
        tipl::shape<dimension> to_dim,from_dim;
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
        std::copy(to2from_ptr,to2from_ptr+to2from.size()*dimension,&to2from[0][0]);
        from2to.resize(from_dim);
        std::copy(from2to_ptr,from2to_ptr+from2to.size()*dimension,&from2to[0][0]);
        return true;
    }

    bool save_warping(const char* filename) const
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
        out.write("to2from",&to2from[0][0],dimension,to2from.size());
        out.write("to_dim",to2from.shape());
        out.write("to_vs",Itvs);
        out.write("to_trans",ItR);

        out.write("from2to",&from2to[0][0],dimension,from2to.size());
        out.write("from_dim",from2to.shape());
        out.write("from_vs",Ivs);
        out.write("from_trans",IR);

        constexpr int method_ver = 202406;
        out.write("method_ver",std::to_string(method_ver));

        return out;
    }
    bool save_transformed_image(const char* filename) const
    {
        if(!tipl::io::gz_nifti::save_to_file(filename,JJ.empty() ? J : JJ,Itvs,ItR,It_is_mni))
        {
            error_msg = "cannot write to file ";
            error_msg += filename;
            return false;
        }
        return true;
    }

};



#endif//REG_HPP
