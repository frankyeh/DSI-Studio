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
            tipl::out() << "nonlinear registration using gpu";
            cdm_cuda(It,Is,dis,terminated,param);
            return;
        }
    }
    tipl::out() << "nonlinear registration using cpu";
    tipl::reg::cdm(It,Is,dis,terminated,param);
}


template<typename image_type,int dim>
tipl::vector<dim> adjust_to_vs(const image_type& from,
               const tipl::vector<dim>& from_vs,
               const image_type& to,
               const tipl::vector<dim>& to_vs)
{
    tipl::vector<dim> from_min,from_max,to_min,to_max;
    tipl::bounding_box(from,from_min,from_max,0);
    tipl::bounding_box(to,to_min,to_max,0);
    from_max -= from_min;
    to_max -= to_min;
    tipl::vector<dim> new_vs(to_vs);
    float r = (to_max[0] > 0.0f) ? from_max[0]*from_vs[0]/(to_max[0]*to_vs[0]) : 1.0f;
    tipl::out() << "fov ratio: " << r;
    if(r > 1.5f || r < 1.0f/1.5f)
    {
        new_vs *= r;
        tipl::out() << "large differences in fov found. adjust voxel size to perform linear registration";
        tipl::out() << "old vs: " << to_vs << " new vs:" << new_vs;
    }
    return new_vs;
}

float optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
                     tipl::reg::cost_type cost_type,bool& terminated);
float optimize_mi_cuda_mr(std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::progress> > reg,
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
        if(has_cuda && use_cuda)
            result = optimize_mi_cuda(reg,cost_type,terminated);
    }
    if(!result)
        result = (cost_type == tipl::reg::mutual_info ? reg->template optimize<tipl::reg::mutual_information<dim> >(terminated):
                                                        reg->template optimize<tipl::reg::correlation>(terminated));
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
                              const float bound[3][8] = tipl::reg::reg_bound,
                              tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                              bool use_cuda = true)
{
    tipl::out() << (reg_type == tipl::reg::affine? "affine" : "rigid body")
                << " registration using "
                << (cost_type == tipl::reg::mutual_info? "mutual info" : "correlation")
                << " on "
                << (has_cuda && use_cuda ? "gpu":"cpu");
    auto new_to_vs = to_vs;
    if constexpr(dim == 3)
    {
        if(reg_type == tipl::reg::affine)
            new_to_vs = adjust_to_vs(from[0],from_vs,to[0],to_vs);
        if(new_to_vs != to_vs)
            tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],to_vs).
                    to_affine_transform(arg,from[0],from_vs,to[0],new_to_vs);
    }
    float result = std::numeric_limits<float>::max();

    auto reg = tipl::reg::linear_reg<tipl::progress>(from,from_vs,to,new_to_vs,arg);
    reg->set_bound(reg_type,bound);

    if constexpr (tipl::use_cuda && dim == 3)
    {
        if(has_cuda && use_cuda)
        {
            do{
                auto cost = optimize_mi_cuda_mr(reg,cost_type,terminated);
                tipl::out() << "cost: " << cost;
                if(cost >= result)
                    break;
                result = cost;
            }while(1);
        }
    }
    if(result == std::numeric_limits<float>::max())
    {
        do{
            auto cost = (cost_type == tipl::reg::mutual_info ? reg->template optimize_mr<tipl::reg::mutual_information<dim> >(terminated):
                                                        reg->template optimize_mr<tipl::reg::correlation>(terminated));

            tipl::out() << "cost: " << cost;
            if(cost >= result)
                break;
            result = cost;
        }while(1);
    }

    if constexpr(dim == 3)
    {
        if(new_to_vs != to_vs)
            tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],new_to_vs).to_affine_transform(arg,from[0],from_vs,to[0],to_vs);
    }
    return linear_refine(from,from_vs,to,to_vs,arg,reg_type,terminated,cost_type,use_cuda);
}
template<int dim>
tipl::transformation_matrix<float> linear(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                                          tipl::vector<dim> from_vs,
                                          std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                                          tipl::vector<dim> to_vs,
                                          tipl::reg::reg_type reg_type,
                                          bool& terminated = tipl::prog_aborted,
                                          const float bound[3][8] = tipl::reg::reg_bound,
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
tipl::color_image read_color_image(const std::string& filename,std::string& error);
extern int map_ver;
template<int dim>
struct dual_reg{
    static constexpr int dimension = dim;
    static constexpr int max_modality = 5;
    using image_type = tipl::image<dimension,unsigned char>;
    tipl::affine_transform<float,dimension> arg;
    const float (*bound)[8] = tipl::reg::reg_bound;
    tipl::reg::cost_type cost_type = tipl::reg::corr;
    tipl::reg::reg_type reg_type = tipl::reg::affine;
    tipl::reg::cdm_param param;
public:
    bool It_is_mni;
    bool export_intermediate = false;
    bool use_cuda = true;
    bool skip_linear = false;
    bool skip_nonlinear = false;
public:
    dual_reg(void):I(max_modality),J(max_modality),It(max_modality),r(max_modality)
    {
    }
    std::vector<image_type> I,J,It;
    std::vector<float> r;
    size_t modality_count = 0;
    int version = 0;
    tipl::image<dimension,tipl::vector<dimension> > t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<dimension> Itvs,Ivs;
    tipl::matrix<dimension+1,dimension+1> ItR,IR;

    mutable std::string error_msg;
    void clear(void)
    {
        I.clear();
        I.resize(max_modality);
        It.clear();
        It.resize(max_modality);
        clear_reg();
    }
    void clear_reg(void)
    {
        J.clear();
        J.resize(max_modality);
        r.clear();
        r.resize(max_modality);
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
    void load_subject(size_t id,tipl::image<dimension>&& in)
    {
        if(id == 0)
        {
            I.clear();
            I.resize(max_modality);
        }
        I[id] = subject_image_pre(in);
    }
    bool load_subject(size_t id,const char* file_name)
    {
        if constexpr(dim == 2)
        {
            tipl::color_image Ic = read_color_image(file_name,error_msg);
            if(Ic.empty())
                return false;
            if(id == 0)
            {
                I.clear();
                I.resize(max_modality);
            }
            I[id] = Ic;
            tipl::segmentation::otsu_median_regulzried(I[id]);
            Ivs = {1.0f,1.0f};
            clear_reg();
            return true;
        }
        else
        {
            tipl::io::gz_nifti nifti;
            if(!nifti.load_from_file(file_name))
            {
                error_msg = "invalid nifti format";
                return false;
            }
            if(id == 0)
            {
                I.clear();
                I.resize(max_modality);
            }
            if(nifti.is_int8())
                nifti >> I[id];
            else
                I[id] = subject_image_pre(nifti.toImage<tipl::image<3> >());
            if(id == 0)
            {
                nifti.get_image_transformation(IR);
                nifti.get_voxel_size(Ivs);
                for(size_t i = id+1;i < I.size();++i)
                    I[i].clear();
            }
            else
            {
                if(I[id].shape() != I[0].shape())
                {
                    error_msg = "inconsistent image size:";
                    I[id].clear();
                    return false;
                }
            }
            clear_reg();
            return true;
        }
    }
    bool load_template(size_t id,const char* file_name)
    {
        if constexpr(dim == 2)
        {
            tipl::color_image Ic = read_color_image(file_name,error_msg);
            if(Ic.empty())
                return false;
            if(id == 0)
            {
                It.clear();
                It.resize(max_modality);
            }
            It[id] = Ic;
            tipl::normalize(It[id]);
            Itvs = {1.0f,1.0f};
            clear_reg();
            return true;
        }
        else
        {
            tipl::io::gz_nifti nifti;
            if(!nifti.load_from_file(file_name))
            {
                error_msg = "invalid nifti format";
                return false;
            }
            if(id == 0)
            {
                It.clear();
                It.resize(max_modality);
            }
            if(nifti.is_int8())
                nifti >> It[id];
            else
                It[id] = template_image_pre(nifti.toImage<tipl::image<dim> >());
            if(id == 0)
            {
                nifti.get_image_transformation(ItR);
                nifti.get_voxel_size(Itvs);
                for(size_t i = id+1;i < It.size();++i)
                    It[i].clear();
                It_is_mni = nifti.is_mni();
            }
            else
            {
                tipl::matrix<dimension+1,dimension+1> It2R;
                nifti.get_image_transformation(It2R);
                if(It[id].shape() != It[0].shape() || It2R != ItR)
                    It[id] = tipl::resample(It[id],It[0].shape(),tipl::from_space(ItR).to(It2R));
            }
            clear_reg();
            return true;
        }
    }
    auto T(void) const
    {
        return tipl::transformation_matrix<float,dimension>(arg,It[0].shape(),Itvs,I[0].shape(),Ivs);
    }
    bool data_ready(void) const
    {
        return !I[0].empty() && !It[0].empty();
    }
    void match_resolution(bool rigid_body)
    {
        if(!data_ready())
            return;
        float ratio = (rigid_body ? Ivs[0]/Itvs[0] : float(I[0].width())/float(It[0].width()));

        auto downsample = [](auto& I,auto& vs,auto& trans)
        {
            for(auto& each : I)
            {
                if(each.empty())
                    break;
                tipl::downsampling(each);
            }
            vs *= 2.0f;
            for(auto each : {0,1,2,
                             4,5,6,
                             8,9,10})
                trans[each] *= 2.0f;
        };
        while(ratio < 0.5f)   // if subject resolution is substantially lower, downsample template
        {
            downsample(It,Itvs,ItR);
            ratio *= 2.0f;
            tipl::out() << "downsampling template to " << Itvs[0] << " mm resolution" << std::endl;
        }
        while(ratio > 2.5f)  // if subject resolution is higher, downsample it for registration
        {
            downsample(I,Ivs,IR);
            ratio /= 2.0f;
            tipl::out() << "downsample subject to " << Ivs[0] << " mm resolution" << std::endl;
        }
    }

    inline float linear_reg(void)
    {
        return linear_reg(tipl::prog_aborted);
    }
    auto make_list(const std::vector<image_type>& data)
    {
        std::vector<tipl::const_pointer_image<dim,unsigned char> > ptr;
        for(const auto& each : data)
        {
            if(each.empty())
                break;
            ptr.push_back(tipl::make_shared(each));
        }
        modality_count = ptr.size();
        return ptr;
    }
    float linear_reg(bool& terminated)
    {
        if(!data_ready())
            return 0.0f;
        tipl::progress prog("linear registration");

        if(!skip_linear)
            linear(make_list(It),Itvs,make_list(I),Ivs,
                   arg,reg_type,terminated,bound,cost_type,use_cuda);
        auto trans = T();

        J.clear();
        J.resize(max_modality);
        for(size_t i = 0;i < J.size();++i)
        {
            if(I[i].empty() || It[i].empty())
                break;
            J[i] = It[i];
            tipl::resample(I[i],J[i],trans);
        }

        auto r = tipl::correlation(J[0],It[0]);
        tipl::out() << "linear: " << r << std::endl;

        if(export_intermediate)
        {
            for(size_t i = 0;i < modality_count;++i)
            {
                tipl::io::gz_nifti::save_to_file((std::string("I") + std::to_string(i) + ".nii.gz").c_str(),I[i],Itvs,ItR);
                tipl::io::gz_nifti::save_to_file((std::string("It") + std::to_string(i) + ".nii.gz").c_str(),It[i],Itvs,ItR);
                tipl::io::gz_nifti::save_to_file((std::string("J") + std::to_string(i) + ".nii.gz").c_str(),J[i],Itvs,ItR);
            }
        }
        return r;
    }
    inline void nonlinear_reg(void)
    {
        nonlinear_reg(tipl::prog_aborted);
    }
    void nonlinear_reg(bool& terminated)
    {
        tipl::progress prog("nonlinear registration");
        if(!skip_nonlinear)
        tipl::par_for(2,[&](int id)
        {
            if(id)
                cdm_common(make_list(It),make_list(J),t2f_dis,terminated,param,use_cuda);
            else
                cdm_common(make_list(J),make_list(It),f2t_dis,terminated,param,use_cuda);
        },2);
        else
        {
            f2t_dis.clear();
            t2f_dis.clear();
            f2t_dis.resize(J[0].shape());
            t2f_dis.resize(J[0].shape());
        }
        auto trans = T();
        from2to.resize(I[0].shape());
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
        tipl::displacement_to_mapping(t2f_dis,to2from,trans);


        std::fill(r.begin(),r.end(),0.0f);
        tipl::par_for(modality_count,[&](size_t i)
        {
            image_type JJ0;
            tipl::compose_mapping(I[i],to2from,JJ0);
            if(export_intermediate)
                tipl::io::gz_nifti::save_to_file((std::string("JJ") + std::to_string(i) + ".nii.gz").c_str(),JJ0,Itvs,ItR);
            r[i] = tipl::correlation(JJ0,It[i]);
        },modality_count);

        for(size_t i = 0;i < modality_count;++i)
            tipl::out() << "nonlinear: " << r[i];

        if(export_intermediate)
        {
            tipl::image<dimension+1> buffer(f2t_dis.shape().expand(2*dimension));
            tipl::par_for(2*dimension,[&](unsigned int d)
            {
                if(d < dimension)
                {
                    size_t shift = d*f2t_dis.size();
                    for(size_t i = 0;i < f2t_dis.size();++i)
                        buffer[i+shift] = f2t_dis[i][d];
                }
                else
                {
                    size_t shift = d*t2f_dis.size();
                    d -= 3;
                    for(size_t i = 0;i < t2f_dis.size();++i)
                        buffer[i+shift] = t2f_dis[i][d];
                }
            },2*dimension);
            tipl::io::gz_nifti::save_to_file("dis.nii.gz",buffer,Itvs,ItR);
        }
    }
    void matching_contrast(void)
    {
        auto& J0 = J[0];
        auto& It0 = It[0];
        auto& It1 = It[1];
        std::vector<float> X(It0.size()*3);
        for(size_t pos = 0;pos < It0.size();++pos)
        {
            if(J0[pos] == 0.0f)
                continue;
            size_t i = pos;
            pos = pos+pos+pos;
            X[pos] = 1;
            X[pos+1] = It0[i];
            X[pos+2] = It1[i];
        }
        tipl::multiple_regression<float> m;
        if(m.set_variables(X.begin(),3,It0.size()))
        {
            float b[3] = {0.0f,0.0f,0.0f};
            m.regress(J0.begin(),b);
            tipl::out() << "image=" << b[0] << " + " << b[1] << " × t1w + " << b[2] << " × t2w ";
            tipl::adaptive_par_for(It0.size(),[&](size_t pos)
            {
                if(J0[pos] == 0.0f)
                    return;
                It0[pos] = b[0] + b[1]*It0[pos] + b[2]*It1[pos];
                if(It0[pos] < 0.0f)
                    It0[pos] = 0.0f;
            });
        }
        It[1].clear();
    }
public:
    auto apply_warping(const tipl::image<dimension>& from,bool is_label) const
    {
        tipl::image<dimension> to(It[0].shape());
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
        return to;
    }
    bool apply_warping(const char* from,const char* to) const;
    bool apply_warping_tt(const char* from,const char* to) const;
    bool load_warping(const char* filename)
    {
        tipl::out() << "load warping " << filename;
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(filename))
        {
            error_msg = "cannot read file ";
            error_msg += filename;
            return false;
        }
        tipl::shape<dimension> dim_to,dim_from;
        const float* to2from_ptr = nullptr;
        const float* from2to_ptr = nullptr;
        unsigned int row,col;
        if (!in.read("dim_to",dim_to) ||
            !in.read("dim_from",dim_from) ||
            !in.read("vs_to",Itvs) ||
            !in.read("vs_from",Ivs) ||
            !in.read("trans_from",IR) ||
            !in.read("trans_to",ItR) ||
            !in.read("to2from",row,col,to2from_ptr) ||
            !in.read("from2to",row,col,from2to_ptr))
        {
            error_msg = "invalid warp file format";
            return false;
        }
        to2from.resize(dim_to);
        std::copy(to2from_ptr,to2from_ptr+to2from.size()*dimension,&to2from[0][0]);
        from2to.resize(dim_from);
        std::copy(from2to_ptr,from2to_ptr+from2to.size()*dimension,&from2to[0][0]);
        version = in.read_as_value<int>("version");
        return true;
    }

    bool save_warping(const char* filename) const
    {
        tipl::progress prog("saving ",filename);
        if(from2to.empty() || to2from.empty())
        {
            error_msg = "no mapping matrix to save";
            return false;
        }
        std::string output_name(filename);
        if(!tipl::ends_with(output_name,".map.gz"))
            output_name += ".map.gz";
        tipl::io::gz_mat_write out((output_name + ".tmp.gz").c_str());
        if(!out)
        {
            error_msg = "cannot write to file ";
            error_msg += filename;
            return false;
        }
        out.write("to2from",&to2from[0][0],dimension,to2from.size());
        out.write("dim_to",to2from.shape());
        out.write("vs_to",Itvs);
        out.write("trans_to",ItR);

        out.write("from2to",&from2to[0][0],dimension,from2to.size());
        out.write("dim_from",from2to.shape());
        out.write("vs_from",Ivs);
        out.write("trans_from",IR);

        out.write("version",map_ver);
        out.close();
        std::filesystem::rename((output_name + ".tmp.gz").c_str(),output_name);
        return true;
    }
};



#endif//REG_HPP
