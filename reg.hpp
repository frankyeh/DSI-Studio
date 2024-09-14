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



tipl::color_image read_color_image(const std::string& filename,std::string& error);
extern int map_ver;
template<int dim>
struct dual_reg{
    static constexpr int dimension = dim;
    static constexpr int max_modality = 5;
    using image_type = tipl::image<dimension,unsigned char>;
    using mapping_type = tipl::image<dimension,tipl::vector<dimension> >;
    tipl::affine_transform<float,dimension> arg;
    const float (*bound)[8] = tipl::reg::reg_bound;
    tipl::reg::cost_type cost_type = tipl::reg::corr;
    tipl::reg::reg_type reg_type = tipl::reg::affine;
    tipl::reg::cdm_param param;
public:
    bool It_is_mni = true;
    bool export_intermediate = false;
    bool use_cuda = true;
    bool skip_linear = false;
    bool skip_nonlinear = false;
public:
    dual_reg(void):I(max_modality),J(max_modality),JJ(max_modality),It(max_modality),r(max_modality)
    {
    }
    std::vector<image_type> I,J,It,JJ;
    std::vector<float> r;
    size_t modality_count = 0;
    mapping_type t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<dimension> Itvs,Ivs;
    tipl::matrix<dimension+1,dimension+1> ItR,IR;
    tipl::shape<dimension> Its,Is;


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
        JJ.clear();
        JJ.resize(max_modality);
        r.clear();
        r.resize(max_modality);
        t2f_dis.clear();
        to2from.clear();
        f2t_dis.clear();
        from2to.clear();
        arg.clear();
    }
    bool load_subject(size_t id,const std::string& file_name)
    {
        if(id == 0)
        {
            I.clear();
            I.resize(max_modality);
            clear_reg();
        }
        if constexpr(dim == 2)
        {
            tipl::color_image Ic = read_color_image(file_name,error_msg);
            if(Ic.empty())
                return false;
            I[id] = Ic;
            tipl::segmentation::normalize_otsu_median(I[id]);
            Ivs = {1.0f,1.0f};
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
            if(nifti.is_int8())
                nifti >> I[id];
            else
                I[id] = subject_image_pre(nifti.toImage<tipl::image<3> >());
            if(id == 0)
            {
                nifti.get_image_transformation(IR);
                nifti.get_voxel_size(Ivs);
                nifti.get_image_dimension(Is);
            }
            else
            {
                tipl::matrix<dimension+1,dimension+1> I2R;
                nifti.get_image_transformation(I2R);
                if(I[id].shape() != Is || I2R != IR)
                    I[id] = tipl::resample(I[id],Is,tipl::from_space(IR).to(I2R));
            }    
            return true;
        }
    }
    bool load_template(size_t id,const std::string& file_name)
    {
        if(id == 0)
        {
            It.clear();
            It.resize(max_modality);
            clear_reg();
        }

        if constexpr(dim == 2)
        {
            tipl::color_image Ic = read_color_image(file_name,error_msg);
            if(Ic.empty())
                return false;
            It[id] = Ic;
            tipl::normalize(It[id]);
            Itvs = {1.0f,1.0f};
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
            if(nifti.is_int8())
                nifti >> It[id];
            else
                It[id] = template_image_pre(nifti.toImage<tipl::image<dim> >());
            if(id == 0)
            {
                nifti.get_image_transformation(ItR);
                nifti.get_voxel_size(Itvs);
                nifti.get_image_dimension(Its);
                It_is_mni = nifti.is_mni();
            }
            else
            {
                tipl::matrix<dimension+1,dimension+1> It2R;
                nifti.get_image_transformation(It2R);
                if(It[id].shape() != Its || It2R != ItR)
                    It[id] = tipl::resample(It[id],Its,tipl::from_space(ItR).to(It2R));
            }
            return true;
        }
    }
    auto T(void) const
    {
        return tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);
    }
    auto invT(void) const
    {
        auto t = tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);
        t.inverse();
        return t;
    }
    bool data_ready(void) const
    {
        return !I[0].empty() && !It[0].empty();
    }
    void match_resolution(bool rigid_body)
    {
        if(!data_ready())
            return;
        float ratio = (rigid_body ? Ivs[0]/Itvs[0] : float(Is.width())/float(Its.width()));

        auto downsample = [](auto& I,auto& Is,auto& vs,auto& trans)
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
            Is = I[0].shape();
        };
        while(ratio < 0.5f)   // if subject resolution is substantially lower, downsample template
        {
            downsample(It,Its,Itvs,ItR);
            ratio *= 2.0f;
            tipl::out() << "downsampling template to " << Itvs[0] << " mm resolution" << std::endl;
        }
        while(ratio > 2.5f)  // if subject resolution is higher, downsample it for registration
        {
            downsample(I,Is,Ivs,IR);
            ratio /= 2.0f;
            tipl::out() << "downsample subject to " << Ivs[0] << " mm resolution" << std::endl;
        }
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
    void calculate_linear_r(void)
    {
        for(size_t i = 0;i < It.size();++i)
            if(!It[i].empty() || !J[i].empty())
                tipl::out() << "linear r: "
                            << (r[i] = tipl::correlation(
                            J[i].empty() ? J[0]:J[i],
                            It[i].empty() ? It[0]:It[i]));
    }
    float linear_reg(bool& terminated)
    {
        if(!data_ready())
            return 0.0f;
        tipl::progress prog("linear registration");

        if(!skip_linear)
            tipl::reg::linear<tipl::out>(make_list(It),Itvs,make_list(I),Ivs,
                   arg,reg_type,terminated,bound,cost_type,use_cuda && has_cuda);
        auto trans = T();

        J.clear();
        J.resize(max_modality);
        for(size_t i = 0;i < J.size() && !I[i].empty();++i)
            J[i] = tipl::resample(I[i],Its,trans);

        calculate_linear_r();

        for(size_t i = 0;i < !I[i].empty() && export_intermediate;++i)
        {
            tipl::io::gz_nifti::save_to_file(("I" + std::to_string(i) + ".nii.gz").c_str(),I[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("It" + std::to_string(i) + ".nii.gz").c_str(),It[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("J" + std::to_string(i) + ".nii.gz").c_str(),J[i],Itvs,ItR);
        }
        return r[0];
    }
    void compute_mapping_from_displacement(void)
    {
        if(f2t_dis.empty() || t2f_dis.empty())
            return;
        auto trans = T();
        from2to.resize(Is);
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
        tipl::displacement_to_mapping(t2f_dis,to2from,trans);
    }
    void calculate_nonlinear_r(void)
    {
        tipl::par_for(modality_count,[&](size_t i)
        {
            JJ[i] = tipl::compose_mapping(I[i],to2from);
            r[i] = tipl::correlation(JJ[i],It[i]);
        },modality_count);
        for(size_t i = 0;i < modality_count;++i)
            tipl::out() << "nonlinear r:" << r[i];
    }

    void nonlinear_reg(bool& terminated)
    {
        tipl::progress prog("nonlinear registration");
        f2t_dis.clear();
        t2f_dis.clear();
        if(skip_nonlinear)
        {
            f2t_dis.resize(Its);
            t2f_dis.resize(Its);
            return;
        }
        tipl::par_for(2,[&](int id)
        {
            if(id)
                tipl::reg::cdm_common<tipl::out>(make_list(It),make_list(J),t2f_dis,terminated,param,use_cuda && has_cuda);
            else
                tipl::reg::cdm_common<tipl::out>(make_list(J),make_list(It),f2t_dis,terminated,param,use_cuda && has_cuda);
        },2);

        compute_mapping_from_displacement();
        calculate_nonlinear_r();

        if(export_intermediate)
        {
            for(size_t i = 0;i < JJ.size();++i)
                tipl::io::gz_nifti::save_to_file("JJ" + std::to_string(i) + ".nii.gz",JJ[i],Itvs,ItR);
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
public:
    template<typename image_type>
    auto apply_warping(const image_type& from,bool is_label) const
    {
        image_type to(Its);
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
    template<typename image_type>
    auto apply_inv_warping(const image_type& to,bool is_label) const
    {
        image_type from(Is);
        if(is_label)
        {
            if(from2to.empty())
                tipl::resample<tipl::interpolation::nearest>(to,from,invT());
            else
                tipl::compose_mapping<tipl::interpolation::nearest>(to,from2to,from);
        }
        else
        {
            if(from2to.empty())
                tipl::resample<tipl::interpolation::cubic>(to,from,invT());
            else
                tipl::compose_mapping<tipl::interpolation::cubic>(to,from2to,from);
        }
        return from;
    }
    auto apply_warping(const char* from) const
    {
        tipl::out() << "opening " << from;
        tipl::image<dimension> I3(Is);
        if(!tipl::io::gz_nifti::load_to_space(from,I3,IR))
        {
            error_msg = "cannot open ";
            error_msg += from;
            return tipl::image<dimension>();
        }
        bool is_label = tipl::is_label_image(I3);
        tipl::out() << (is_label ? "label image interpolated using nearest assignment " : "scalar image interpolated using spline") << std::endl;
        tipl::out() << "apply warping to " << from;
        return apply_warping(I3,is_label);
    }
    bool apply_warping(const char* from,const char* to) const
    {
        auto I = apply_warping(from);
        if(I.empty())
            return false;
        tipl::out() << "saving " << to;
        if(!tipl::io::gz_nifti::save_to_file(to,I,Itvs,ItR,It_is_mni))
        {
            error_msg = "cannot write to file ";
            error_msg += to;
            return false;
        }
        return true;
    }
    auto apply_inv_warping(const char* to) const
    {
        tipl::out() << "opening " << to;
        tipl::image<dimension> I3(Its);
        if(!tipl::io::gz_nifti::load_to_space(to,I3,ItR))
        {
            error_msg = "cannot open ";
            error_msg += to;
            return tipl::image<dimension>();
        }
        bool is_label = tipl::is_label_image(I3);
        tipl::out() << (is_label ? "label image interpolated using nearest assignment " : "scalar image interpolated using spline") << std::endl;
        tipl::out() << "apply inverse warping to " << to;
        return apply_inv_warping(I3,is_label);
    }

    bool apply_inv_warping(const char* to,const char* from) const
    {
        auto I = apply_inv_warping(to);
        if(I.empty())
            return false;
        tipl::out() << "saving " << from;
        if(!tipl::io::gz_nifti::save_to_file(from,I,Ivs,IR,false))
        {
            error_msg = "cannot write file ";
            error_msg += from;
            return false;
        }
        return true;
    }
    bool apply_warping_tt(const char* from,const char* to) const;
    bool load_warping(const std::string& filename)
    {
        tipl::out() << "load warping " << filename;
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(filename))
        {
            error_msg = "cannot read file ";
            error_msg += filename;
            return false;
        }

        if(in.read_as_value<int>("version") > map_ver)
        {
            error_msg = "incompatible map file format: the version ";
            error_msg += in.read_as_value<int>("version");
            error_msg += " is not supported within current rage ";
            error_msg += std::to_string(map_ver);
            return false;
        }
        const float* f2t_dis_ptr = nullptr;
        const float* t2f_dis_ptr = nullptr;
        unsigned int row,col;
        if (!in.read("dimension",Its) ||
            !in.read("voxel_size",Itvs) ||
            !in.read("trans",ItR) ||
            !in.read("dimension_from",Is) ||
            !in.read("voxel_size_from",Ivs) ||
            !in.read("trans_from",IR) ||
            !in.read("f2t_dis",row,col,f2t_dis_ptr) ||
            !in.read("t2f_dis",row,col,t2f_dis_ptr) ||
            !in.read("arg",arg))
        {
            error_msg = "invalid warp file format";
            return false;
        }

        tipl::shape<dimension> sub_shape;
        if constexpr(dimension == 3)
            sub_shape = tipl::shape<3>((Its[0]+1)/2,(Its[1]+1)/2,(Its[2]+1)/2);
        else
            sub_shape = tipl::shape<2>((Its[0]+1)/2,(Its[1]+1)/2);

        t2f_dis.resize(sub_shape);
        f2t_dis.resize(sub_shape);
        if(row*col != sub_shape.size()*3)
        {
            error_msg = "invalid displacement field";
            return false;
        }
        std::copy(f2t_dis_ptr,f2t_dis_ptr+f2t_dis.size()*dimension,&f2t_dis[0][0]);
        std::copy(t2f_dis_ptr,t2f_dis_ptr+t2f_dis.size()*dimension,&t2f_dis[0][0]);
        tipl::upsample_with_padding(t2f_dis,Its);
        tipl::upsample_with_padding(f2t_dis,Its);
        compute_mapping_from_displacement();
        return true;
    }

    bool save_warping(const char* filename) const
    {
        tipl::progress prog("saving ",filename);
        if(f2t_dis.empty() || t2f_dis.empty())
        {
            error_msg = "no mapping matrix to save";
            return false;
        }
        std::string output_name(filename);
        if(!tipl::ends_with(output_name,".mz"))
            output_name += ".mz";
        tipl::io::gz_mat_write out((output_name + ".tmp.gz").c_str());
        if(!out)
        {
            error_msg = "cannot write to file ";
            error_msg += filename;
            return false;
        }
        out.apply_slope = true;
        tipl::image<dimension,tipl::vector<dimension> > f2t_dis_sub2,t2f_dis_sub2;
        tipl::downsample_with_padding(f2t_dis,f2t_dis_sub2);
        tipl::downsample_with_padding(t2f_dis,t2f_dis_sub2);
        out.write<tipl::io::sloped>("f2t_dis",&f2t_dis_sub2[0][0],dimension,f2t_dis_sub2.size());
        out.write("dimension",Its);
        out.write("voxel_size",Itvs);
        out.write("trans",ItR);

        out.write<tipl::io::sloped>("t2f_dis",&t2f_dis_sub2[0][0],dimension,t2f_dis_sub2.size());
        out.write("dimension_from",Is);
        out.write("voxel_size_from",Ivs);
        out.write("trans_from",IR);

        out.write("arg",arg);
        out.write("version",map_ver);
        out.close();
        std::filesystem::rename((output_name + ".tmp.gz").c_str(),output_name);
        return true;
    }
    void to_I_space(const tipl::shape<3>& new_Is,const tipl::matrix<4,4>& new_IR)
    {
        if(new_Is == Is && new_IR == IR)
            return;
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_IR).to(IR));
        auto TR = T();
        TR *= trans;
        for(auto& each : I)
            if(!each.empty())
                each = tipl::resample(each,new_Is,trans);
        Is = new_Is;
        IR = new_IR;
        for(int i = 0;i < 3;++i)
            Ivs[i] = std::sqrt(IR[i]*IR[i]+IR[i+4]*IR[i+4]+IR[i+8]*IR[i+8]);
        TR.to_affine_transform(arg,Its,Itvs,Is,Ivs);
        compute_mapping_from_displacement();
    }
    void to_It_space(const tipl::shape<3>& new_Its,const tipl::matrix<4,4>& new_ItR)
    {
        if(new_Its == Its && new_ItR == ItR)
            return;
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_ItR).to(ItR));
        auto TR = trans;
        TR *= T();
        for(auto& each : It)
            if(!each.empty())
                each = tipl::resample(each,new_Its,trans);
        if(!f2t_dis.empty())
            f2t_dis = tipl::resample(f2t_dis,new_Its,trans);
        if(!t2f_dis.empty())
            t2f_dis = tipl::resample(t2f_dis,new_Its,trans);
        Its = new_Its;
        ItR = new_ItR;
        for(int i = 0;i < 3;++i)
            Itvs[i] = std::sqrt(ItR[i]*ItR[i]+ItR[i+4]*ItR[i+4]+ItR[i+8]*ItR[i+8]);
        TR.to_affine_transform(arg,Its,Itvs,Is,Ivs);
        compute_mapping_from_displacement();
    }
};



#endif//REG_HPP
