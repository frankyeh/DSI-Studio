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
    bool It_is_mni;
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
    bool load_subject(size_t id,const std::string& file_name)
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
            tipl::segmentation::normalize_otsu_median(I[id]);
            Ivs = {1.0f,1.0f};
            if(id == 0)
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
                tipl::matrix<dimension+1,dimension+1> I2R;
                nifti.get_image_transformation(I2R);
                if(I[id].shape() != I[0].shape() || I2R != IR)
                    I[id] = tipl::resample(I[id],I[0].shape(),tipl::from_space(IR).to(I2R));
            }
            if(id == 0)
                clear_reg();
            return true;
        }
    }
    bool load_template(size_t id,const std::string& file_name)
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
            if(id == 0)
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
            if(id == 0)
                clear_reg();
            return true;
        }
    }
    auto T(void) const
    {
        return tipl::transformation_matrix<float,dimension>(arg,It[0].shape(),Itvs,I[0].shape(),Ivs);
    }
    auto invT(void) const
    {
        auto t = tipl::transformation_matrix<float,dimension>(arg,It[0].shape(),Itvs,I[0].shape(),Ivs);
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
            tipl::reg::linear<tipl::out>(make_list(It),Itvs,make_list(I),Ivs,
                   arg,reg_type,terminated,bound,cost_type,use_cuda && has_cuda);
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

        for(size_t i = 0;i < modality_count && export_intermediate;++i)
        {
            tipl::io::gz_nifti::save_to_file(("I" + std::to_string(i) + ".nii.gz").c_str(),I[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("It" + std::to_string(i) + ".nii.gz").c_str(),It[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("J" + std::to_string(i) + ".nii.gz").c_str(),J[i],Itvs,ItR);
        }
        return r;
    }
    inline void nonlinear_reg(void)
    {
        nonlinear_reg(tipl::prog_aborted);
    }
    void compute_mapping_from_displacement(void)
    {
        auto trans = T();
        from2to.resize(I[0].shape());
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
        tipl::displacement_to_mapping(t2f_dis,to2from,trans);
    }
    void report_cdm_correlation(void)
    {
        tipl::par_for(modality_count,[&](size_t i)
        {
            JJ[i] = tipl::compose_mapping(I[i],to2from);
            r[i] = tipl::correlation(JJ[i],It[i]);
        },modality_count);
        for(size_t i = 0;i < modality_count;++i)
            tipl::out() << "nonlinear: " << r[i];
    }

    void nonlinear_reg(bool& terminated)
    {
        tipl::progress prog("nonlinear registration");
        if(skip_nonlinear)
        {
            f2t_dis.clear();
            t2f_dis.clear();
            f2t_dis.resize(J[0].shape());
            t2f_dis.resize(J[0].shape());
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
        report_cdm_correlation();

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
    std::pair<size_t,float> matching_contrast(const std::vector<std::string>& contrast_names)
    {
        tipl::out() << "matching contrast";
        std::vector<float> linear_r(contrast_names.size());
        for(size_t i = 0;i < contrast_names.size();++i)
            tipl::out() << contrast_names[i] << " r:" << (linear_r[i] = tipl::correlation(J[0],It[i]));
        size_t max_index = std::max_element(linear_r.begin(),linear_r.end())-linear_r.begin();
        tipl::out() << "best matching contrast: " << contrast_names[max_index];
        return std::make_pair(max_index,linear_r[max_index]);
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
    auto apply_inv_warping(const tipl::image<dimension>& to,bool is_label) const
    {
        tipl::image<dimension> from(I[0].shape());
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
    bool apply_warping(const char* from,const char* to) const
    {
        if(tipl::ends_with(from,".tt.gz"))
            return apply_warping_tt(from,to);
        /*
        tipl::out() << "opening " << from;
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(from))
        {
            error_msg = "cannot open ";
            error_msg += from;
            return false;
        }
        if(nii.dim(dimension+1) > 1)
        {
            // check data range
            std::vector<tipl::image<dimension> > I_list(nii.dim(dimension+1));
            for(unsigned int index = 0;index < nii.dim(dimension+1);++index)
            {
                if(!nii.toLPS(I_list[index]))
                {
                    error_msg = "failed to parse 4D NIFTI file";
                    return false;
                }
                std::replace_if(I_list[index].begin(),I_list[index].end(),[](float v){return std::isnan(v) || std::isinf(v) || v < 0.0f;},0.0f);
            }
            if(I[0].shape() != I_list[0].shape())
            {
                error_msg = std::filesystem::path(from).filename().u8string();
                error_msg += " has an image size or srow matrix from that of the original --from image.";
                return false;
            }
            bool is_label = tipl::is_label_image(I_list[0]);
            tipl::out() << (is_label ? "label image interpolated using nearest assignment " : "scalar image interpolated using spline") << std::endl;
            tipl::image<dimension+1> J4(It[0].shape().expand(nii.dim(dimension+1)));
            tipl::adaptive_par_for(nii.dim(4),[&](size_t z)
            {
                tipl::image<dimension> out(apply_warping(I_list[z],is_label));
                std::copy(out.begin(),out.end(),J4.slice_at(z).begin());
            });
            if(!tipl::io::gz_nifti::save_to_file(to,J4,Itvs,ItR,It_is_mni))
            {
                error_msg = "cannot write to file ";
                error_msg += to;
                return false;
            }
            return true;
        }
        */
        tipl::out() << "apply warping to " << from;
        tipl::out() << "opening " << from;
        tipl::image<dimension> I3(I[0].shape());
        if(!tipl::io::gz_nifti::load_to_space(from,I3,IR))
        {
            error_msg = "cannot open ";
            error_msg += from;
            return false;
        }
        bool is_label = tipl::is_label_image(I3);
        tipl::out() << (is_label ? "label image interpolated using nearest assignment " : "scalar image interpolated using spline") << std::endl;
        tipl::out() << "saving " << to;
        if(!tipl::io::gz_nifti::save_to_file(to,apply_warping(I3,is_label),Itvs,ItR,It_is_mni))
        {
            error_msg = "cannot write to file ";
            error_msg += to;
            return false;
        }
        return true;
    }
    bool apply_inv_warping(const char* to,const char* from) const
    {
        tipl::out() << "apply inverse warping to " << to;
        tipl::out() << "opening " << to;
        tipl::image<dimension> I3(It[0].shape());
        if(!tipl::io::gz_nifti::load_to_space(to,I3,ItR))
        {
            error_msg = "cannot open ";
            error_msg += to;
            return false;
        }
        bool is_label = tipl::is_label_image(I3);
        tipl::out() << (is_label ? "label image interpolated using nearest assignment " : "scalar image interpolated using spline") << std::endl;
        tipl::out() << "saving " << from;
        if(!tipl::io::gz_nifti::save_to_file(from,apply_inv_warping(I3,is_label),Ivs,IR,false))
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
        tipl::shape<dimension> dim_to,dim_from;
        const float* f2t_dis_ptr = nullptr;
        const float* t2f_dis_ptr = nullptr;
        unsigned int row,col;
        if (!in.read("dimension",dim_to) ||
            !in.read("voxel_size",Itvs) ||
            !in.read("trans",ItR) ||
            !in.read("dimension_from",dim_from) ||
            !in.read("voxel_size_from",Ivs) ||
            !in.read("trans_from",IR) ||
            !in.read("f2t_dis",row,col,f2t_dis_ptr) ||
            !in.read("t2f_dis",row,col,t2f_dis_ptr) ||
            !in.read("arg",arg))
        {
            error_msg = "invalid warp file format";
            return false;
        }
        It[0].resize(dim_to);
        I[0].resize(dim_from);

        tipl::shape<dimension> sub_shape;
        if constexpr(dimension == 3)
            sub_shape = tipl::shape<3>(dim_to[0]/2,dim_to[1]/2,dim_to[2]/2);
        else
            sub_shape = tipl::shape<2>(dim_to[0]/2,dim_to[1]/2);

        t2f_dis.resize(sub_shape);
        f2t_dis.resize(sub_shape);
        if(row*col != sub_shape.size()*3)
        {
            error_msg = "invalid displacement field";
            return false;
        }
        std::copy(f2t_dis_ptr,f2t_dis_ptr+f2t_dis.size()*dimension,&f2t_dis[0][0]);
        std::copy(t2f_dis_ptr,t2f_dis_ptr+t2f_dis.size()*dimension,&t2f_dis[0][0]);
        tipl::upsample_with_padding(t2f_dis,dim_to);
        tipl::upsample_with_padding(f2t_dis,dim_to);
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
        out.write("dimension",It[0].shape());
        out.write("voxel_size",Itvs);
        out.write("trans",ItR);

        out.write<tipl::io::sloped>("t2f_dis",&t2f_dis_sub2[0][0],dimension,t2f_dis_sub2.size());
        out.write("dimension_from",I[0].shape());
        out.write("voxel_size_from",Ivs);
        out.write("trans_from",IR);

        out.write("arg",arg);
        out.write("version",map_ver);
        out.close();
        std::filesystem::rename((output_name + ".tmp.gz").c_str(),output_name);
        return true;
    }
    void to_space(const tipl::shape<3>& new_It_shape,
                  const tipl::matrix<4,4>& new_ItR)
    {
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_ItR).to(ItR));
        auto TR = trans;
        TR *= T();
        for(auto& each : It)
            each = tipl::resample(each,new_It_shape,trans);
        f2t_dis = tipl::resample(f2t_dis,new_It_shape,trans);
        t2f_dis = tipl::resample(t2f_dis,new_It_shape,trans);
        ItR = new_ItR;
        for(int i = 0;i < 3;++i)
            Itvs[i] = std::sqrt(ItR[i]*ItR[i]+ItR[i+4]*ItR[i+4]+ItR[i+8]*ItR[i+8]);
        TR.to_affine_transform(arg,It[0].shape(),Itvs,I[0].shape(),Ivs);
        compute_mapping_from_displacement();
    }
};



#endif//REG_HPP
