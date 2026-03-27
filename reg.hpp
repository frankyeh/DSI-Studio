#ifndef REG_HPP
#define REG_HPP
#include <iostream>
#include <thread>
#include "TIPL/tipl.hpp"

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

template<typename io_loader,typename T>
bool load_image(size_t id,const std::string& file_name,
                std::vector<T>& images,bool preprocess,
                tipl::matrix<T::dimension+1,T::dimension+1>& ref_transform,
                tipl::vector<T::dimension,float>& voxel_size,
                tipl::shape<T::dimension>& image_shape,
                bool& is_mni,
                std::string& error_msg)
{
    if(id == 0)
        std::fill(images.begin(),images.end(),T());

    io_loader in;
    if(!in.open(file_name,std::ios::in))
    {
        error_msg = in.error_msg;
        return false;
    }

    for(size_t i = 0;i < in.dim(4);++i,++id)
    {
        if(id >= images.size())
            images.resize(id+1);

        if(in.is_int8())
            in >> images[id];
        else if(preprocess)
            images[id] = subject_image_pre(in.template toImage<tipl::image<T::dimension>>());
        else
            images[id] = template_image_pre(in.template toImage<tipl::image<T::dimension>>());

        if(id == 0)
        {
            in >> voxel_size >> image_shape >> ref_transform;
            is_mni = in.is_mni();
        }
        else
        {
            tipl::matrix<T::dimension+1,T::dimension+1> curr_transform;
            in.get_image_transformation(curr_transform);
            if(images[id].shape() != image_shape || curr_transform != ref_transform)
                images[id] = tipl::resample(images[id],image_shape,tipl::from_space(ref_transform).to(curr_transform));
        }

        if(tipl::max_value(images[id]) == 1)
            images[id] *= 255;
    }
    return true;
}

extern int map_ver;

template<typename out_type>
struct dual_reg{
    static constexpr int dimension = 3;
    static constexpr int max_modality = 16;
    using image_type = tipl::image<dimension,unsigned char>;
    using mapping_type = tipl::image<dimension,tipl::vector<dimension>>;

    tipl::affine_param<float,dimension> arg;
    tipl::reg::linear_reg_param linear_param = {tipl::reg::affine,tipl::reg::corr};
    tipl::reg::cdm_param param;

    bool It_is_mni = true;
    bool Is_is_mni = false;
    bool export_intermediate = false;
    bool use_cuda = true;
    bool skip_linear = false;
    bool skip_nonlinear = false;
    bool match_fov = true;
    bool masked_r = true;

    std::vector<std::string> modality_names;
    std::vector<image_type> I,J,It;
    std::vector<float> r;
    mapping_type t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<dimension> Itvs,Ivs;
    tipl::matrix<dimension+1,dimension+1> ItR,IR;
    tipl::shape<dimension> Its,Is;

    std::vector<tipl::vector<dimension>> anchor[2];
    mapping_type previous_t2f,previous_f2t;
    std::vector<image_type> previous_It;
    mutable std::string error_msg;

    dual_reg():modality_names(max_modality),I(max_modality),J(max_modality),It(max_modality),r(max_modality) {}

    void clear()
    {
        I.assign(max_modality,image_type());
        It.assign(max_modality,image_type());
        clear_reg();
    }

    void clear_reg()
    {
        J.assign(max_modality,image_type());
        r.assign(max_modality,0.0f);
        t2f_dis.clear();
        to2from.clear();
        f2t_dis.clear();
        from2to.clear();
        arg.clear();
    }

    template<typename io_loader>
    bool load_subject(size_t id,const std::string& file_name)
    {
        if(!id)
            clear_reg();
        return load_image<io_loader>(id,file_name,I,true,IR,Ivs,Is,Is_is_mni,error_msg);
    }

    template<typename io_loader>
    bool load_template(size_t id,const std::string& file_name)
    {
        if(!id)
            clear_reg();
        return load_image<io_loader>(id,file_name,It,false,ItR,Itvs,Its,It_is_mni,error_msg);
    }

    void match_resolution(bool use_vs,float lr = 0.5f,float hr = 2.0f)
    {
        if(!data_ready())
            return;

        tipl::vector<dimension,int> range_min,range_max;
        tipl::bounding_box(I[0],range_min,range_max);
        bool need_cropping = false;

        for(int i = 0;i < dimension;++i)
            if(range_max[i]-range_min[i] < Is[i]*0.7f)
                need_cropping = true;

        if(need_cropping)
        {
            for(int d = 0;d < dimension;++d)
            {
                range_min[d] = std::max<int>(0,range_min[d]-float(Is[d])*0.05f);
                range_max[d] = std::min<int>(Is[d],range_max[d]+float(Is[d])*0.05f);
            }
            out_type() << "cropping from " << Is << " to " << (range_max-range_min) << " and shifting " << range_min;

            for(auto& each : I)
            {
                if(each.empty())
                    break;
                image_type newI;
                tipl::crop(each,newI,range_min,range_max);
                each.swap(newI);
            }
            out_type() << "IR:" << IR;
            Is = I[0].shape();
            IR[3] += range_min[0]*IR[0];
            IR[7] += range_min[1]*IR[5];
            IR[11] += range_min[2]*IR[10];
            out_type() << "new IR:" << IR;
        }

        float ratio = (use_vs ? Itvs[0]/Ivs[0]:float(Is.width())/float(Its.width()));
        auto downsample = [](auto& I,auto& Is,auto& vs,auto& trans)
        {
            for(auto& each : I)
            {
                if(each.empty())
                    break;
                tipl::downsampling(each);
            }
            vs *= 2.0f;
            for(auto each : {0,1,2,4,5,6,8,9,10})
                trans[each] *= 2.0f;
            Is = I[0].shape();
        };

        while(ratio <= lr)
        {
            downsample(It,Its,Itvs,ItR);
            ratio *= 2.0f;
            out_type() << "downsampling template to " << Itvs[0] << " mm resolution";
        }

        while(ratio >= hr)
        {
            downsample(I,Is,Ivs,IR);
            ratio /= 2.0f;
            out_type() << "downsample subject to " << Ivs[0] << " mm resolution";
        }
    }

    auto T() const
    {
        return tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);
    }

    auto invT() const
    {
        auto t = tipl::transformation_matrix<float,dimension>(arg,Its,Itvs,Is,Ivs);
        t.inverse();
        return t;
    }

    bool data_ready() const
    {
        return !I[0].empty() && !It[0].empty();
    }

    void show_r(const std::string& prompt)
    {
        std::string result(prompt);
        if(!modality_names.empty())
        {
            for(size_t i = 0;i < modality_names.size();++i)
                result += (i ? "," : " ")+modality_names[i]+":"+std::to_string(r[i]);
        }
        else
        {
            for(size_t i = 0;i < r.size() && r[i] != 0.0;++i)
                result += (i ? "," : " ")+std::to_string(r[i]);
        }
        out_type() << result;
    }

    void compute_mapping_from_displacement()
    {
        if(f2t_dis.empty() || t2f_dis.empty())
            return;
        auto trans = T();
        from2to.resize(Is);
        to2from.resize(Its);
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
        tipl::displacement_to_mapping(t2f_dis,to2from,trans);
    }

    void calculate_linear_r()
    {
        auto trans = T();
        std::fill(r.begin(),r.end(),0.0f);
        J.resize(I.size());

        tipl::par_for(I.size(),[&](size_t i)
        {
            if(!I[i].empty())
                J[i] = tipl::resample(I[i],Its,trans);
        },I.size());

        tipl::par_for(I.size(),[&](size_t i)
        {
            if(J[i].empty() && It[i].empty())
                return;
            const auto& x = J[i].empty() ? J[0]:J[i];
            const auto& y = It[i].empty() ? It[0]:It[i];
            r[i] = masked_r ? tipl::correlation_ygz(x.begin(),x.end(),y.begin())
                            : tipl::correlation(x.begin(),x.end(),y.begin());
        },I.size());

        show_r("linear r: ");
        if(match_fov)
            It_match_fov();
    }

    void It_match_fov()
    {
        auto trans = T();
        size_t max_It = 0;
        while(max_It < It.size() && !It[max_It].empty())
            ++max_It;

        for(tipl::pixel_index<dimension> index(Its);index < Its.size();++index)
        {
            tipl::vector<dimension> pos;
            trans(index,pos);
            if(Is.is_valid(pos))
                continue;
            for(size_t i = 0;i < max_It;++i)
                It[i][index.index()] = 0;
        }
    }

    void calculate_nonlinear_r()
    {
        std::fill(r.begin(),r.end(),0.0f);
        tipl::par_for(I.size(),[&](size_t i)
        {
            if(I[i].empty() || It[i].empty())
                return;
            auto& x = (J[i] = tipl::compose_mapping(I[i],to2from));
            auto& y = previous_It.empty() ? It[i]:previous_It[i];
            r[i] = masked_r ? tipl::correlation_ygz(x.begin(),x.end(),y.begin())
                            : tipl::correlation(x.begin(),x.end(),y.begin());
        },I.size());
        show_r("nonlinear r: ");
    }

    float linear_reg(bool& terminated)
    {
        if(!data_ready())
            return 0.0f;

        float cost = 0.0f;
        linear_param.cuda = use_cuda;
        if(!skip_linear)
            cost = tipl::reg::linear<out_type>(tipl::reg::make_list(It),Itvs,tipl::reg::make_list(I),Ivs,arg,linear_param,terminated);

        calculate_linear_r();
        return cost;
    }

    template<typename io_writer>
    void export_linear()
    {
        for(size_t i = 0;i < I.size() && !I[i].empty();++i)
        {
            io_writer("I"+std::to_string(i)+".nii.gz",std::ios::out) << Ivs << IR << I[i];
            io_writer("It"+std::to_string(i)+".nii.gz",std::ios::out) << Itvs << ItR << It[i];
            io_writer("J"+std::to_string(i)+".nii.gz",std::ios::out) << Itvs << ItR << J[i];
        }
    }

    void nonlinear_reg(bool& terminated)
    {
        if(!data_ready())
            return;

        f2t_dis.clear();
        t2f_dis.clear();

        if(skip_nonlinear)
        {
            f2t_dis.resize(Its);
            t2f_dis.resize(Its);
            return;
        }

        auto param0 = param;
        auto param1 = param;

        std::thread t([&]()
        {
            tipl::reg::cdm_common<out_type>(tipl::reg::make_list(It),tipl::reg::make_list(J),t2f_dis,terminated,param0,use_cuda);
        });
        tipl::reg::cdm_common<out_type>(tipl::reg::make_list(J),tipl::reg::make_list(It),f2t_dis,terminated,param1,use_cuda);
        t.join();

        if(!previous_f2t.empty() && !previous_t2f.empty())
        {
            tipl::accumulate_displacement(previous_f2t,f2t_dis);
            tipl::accumulate_displacement(previous_t2f,t2f_dis);
        }

        compute_mapping_from_displacement();
        calculate_nonlinear_r();
    }

    template<typename io_writer>
    void export_nonlinear()
    {
        for(size_t i = 0;i < J.size();++i)
            io_writer("JJ"+std::to_string(i)+".nii.gz",std::ios::out) << Itvs << ItR << J[i];

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
                for(size_t i = 0;i < t2f_dis.size();++i)
                    buffer[i+shift] = t2f_dis[i][d-dimension];
            }
        },2*dimension);
        io_writer("dis.nii.gz",std::ios::out) << Itvs << ItR << buffer;
    }

    template<bool direction,tipl::interpolation itype,typename Itype>
    auto apply_warping(const Itype& input) const
    {
        const auto& mapping = direction ? to2from:from2to;
        return mapping.empty() ?
            tipl::resample<itype>(input,direction ? Its:Is,direction ? T():invT()) :
            tipl::compose_mapping<itype>(input,mapping);
    }

    void dis_to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R)
    {
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_R).to(ItR));
        tipl::vector<3> new_vs(tipl::to_vs(new_R));
        float jacobian = Itvs[0]/new_vs[0];

        if(!f2t_dis.empty())
        {
            f2t_dis = tipl::resample(f2t_dis,new_s,trans);
            if(jacobian != 1.0f)
                tipl::multiply_constant(f2t_dis,jacobian);
        }
        if(!t2f_dis.empty())
        {
            t2f_dis = tipl::resample(t2f_dis,new_s,trans);
            if(jacobian != 1.0f)
                tipl::multiply_constant(t2f_dis,jacobian);
        }
    }

    void to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R)
    {
        if(new_s == Is && new_R == IR && new_s == Its && new_R == ItR)
            return;

        tipl::vector<3> new_vs(tipl::to_vs(new_R));
        if(ItR != IR || arg != tipl::affine_param<float,3>())
        {
            arg = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_R).to(ItR))
                .accumulate(T())
                .accumulate(tipl::transformation_matrix<float,dimension>(tipl::from_space(IR).to(new_R)))
                .to_affine_param(new_s,new_vs,new_s,new_vs);
        }

        if(new_s != Is || new_R != IR)
        {
            auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_R).to(IR));
            for(auto& each : I)
                if(!each.empty())
                    each = tipl::resample(each,new_s,trans);
        }
        if(new_s != Its || new_R != ItR)
        {
            dis_to_space(new_s,new_R);
            auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_R).to(ItR));
            for(auto& each : It)
                if(!each.empty())
                    each = tipl::resample(each,new_s,trans);
        }
        Its = Is = new_s;
        ItR = IR = new_R;
        Itvs = Ivs = new_vs;
        compute_mapping_from_displacement();
    }

    void to_I_space(const tipl::shape<3>& new_Is,const tipl::matrix<4,4>& new_IR)
    {
        if(new_Is == Is && new_IR == IR)
            return;

        tipl::progress prog("transform subject space");
        out_type() << "Is: " << Is << " new Is:" << new_Is;
        out_type() << "IR: " << IR;
        out_type() << "new IR:" << new_IR;

        tipl::vector<3> new_Ivs(tipl::to_vs(new_IR));
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_IR).to(IR));
        out_type() << "trans: " << trans;

        for(auto& each : I)
            if(!each.empty())
                each = tipl::resample(each,new_Is,trans);

        out_type() << "arg: " << arg;
        arg = T().accumulate(tipl::from_space(IR).to(new_IR)).to_affine_param(Its,Itvs,new_Is,new_Ivs);
        out_type() << "new arg: " << arg;

        Is = new_Is;
        IR = new_IR;
        Ivs = new_Ivs;
        compute_mapping_from_displacement();
    }

    void to_It_space(const tipl::shape<3>& new_Its,const tipl::matrix<4,4>& new_ItR)
    {
        if(new_Its == Its && new_ItR == ItR)
            return;

        tipl::progress prog("transform template space");
        out_type() << "Its: " << Its << " new Its:" << new_Its;
        out_type() << "ItR: " << ItR;
        out_type() << "new ItR:" << new_ItR;

        tipl::vector<3> new_Itvs(tipl::to_vs(new_ItR));
        auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_ItR).to(ItR));
        out_type() << "trans: " << trans;

        for(auto& each : It)
            if(!each.empty())
                each = tipl::resample(each,new_Its,trans);

        dis_to_space(new_Its,new_ItR);
        out_type() << "arg: " << arg;
        arg = trans.accumulate(T()).to_affine_param(new_Its,new_Itvs,Is,Ivs);
        out_type() << "new arg: " << arg;

        Its = new_Its;
        ItR = new_ItR;
        Itvs = new_Itvs;
        compute_mapping_from_displacement();
    }

    void to_It_space(const tipl::shape<3>& new_Its)
    {
        auto new_ItR = ItR;
        for(int i = 0;i < dimension;++i)
            new_ItR[3+i*4] -= new_ItR[i*5]*(float(new_Its[i])-float(Its[i]))*0.5f;
        to_It_space(new_Its,new_ItR);
    }

    void to_I_space(const tipl::shape<3>& new_Is)
    {
        auto new_IR = IR;
        for(int i = 0;i < dimension;++i)
            new_IR[3+i*4] -= new_IR[i*5]*(float(new_Is[i])-float(Is[i]))*0.5f;
        to_I_space(new_Is,new_IR);
    }
};

#endif // REG_HPP
