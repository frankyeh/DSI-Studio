#include <QString>
#include <QImage>
#include "img.hpp"
#include "reg.hpp"
#include "tract_model.hpp"

void dual_reg::clear(void)
{
    I.clear();
    I.resize(max_modality);
    It.clear();
    It.resize(max_modality);
    clear_reg();
}
void dual_reg::clear_reg(void)
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
template<typename image_type>
auto read_buffer(const std::vector<image_type>& data)
{
    size_t image_count = std::distance(data.begin(),
                                           std::find_if(data.begin(), data.end(),
                                           [](const auto& img) { return img.empty(); }));
    if(!image_count)
        return tipl::image<4,unsigned char>();
    tipl::image<4,unsigned char> buffer(data[0].shape().expand(image_count));
    tipl::par_for(image_count,[&](size_t i)
    {
        std::copy(data[i].begin(),data[i].end(),buffer.begin() + i*data[0].size());
    },image_count);
    return buffer;
}
bool dual_reg::save_subject(const std::string& file_name)
{
    return tipl::io::gz_nifti::save_to_file(file_name,read_buffer(I),Ivs,IR,Is_is_mni);
}
bool dual_reg::save_template(const std::string& file_name)
{
    return tipl::io::gz_nifti::save_to_file(file_name,read_buffer(It),Itvs,ItR,It_is_mni);
}

template<typename T>
bool load_image(size_t id, const std::string& file_name,
                std::vector<T>& images, bool preprocess,
                tipl::matrix<T::dimension+1, T::dimension+1>& ref_transform,
                tipl::vector<T::dimension,float>& voxel_size,
                tipl::shape<T::dimension>& image_shape,
                bool& is_mni,
                std::string& error_msg)
{
    if(id == 0)
    {
        auto size = images.size();
        images.clear();
        images.resize(size);
    }

    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(file_name))
    {
        error_msg = nifti.error_msg;
        return false;
    }

    for(size_t i = 0; i < nifti.dim(4); ++i, ++id)
    {
        if(nifti.is_int8())
            nifti >> images[id];
        else
        {
            if(preprocess)
                images[id] = subject_image_pre(nifti.toImage<tipl::image<T::dimension> >());
            else
                images[id] = template_image_pre(nifti.toImage<tipl::image<T::dimension> >());
        }

        if(id == 0)
        {
            nifti.get_image_transformation(ref_transform);
            nifti.get_voxel_size(voxel_size);
            nifti.get_image_dimension(image_shape);
            is_mni = nifti.is_mni();
        }
        else
        {
            tipl::matrix<T::dimension+1,T::dimension+1> curr_transform;
            nifti.get_image_transformation(curr_transform);
            if(images[id].shape() != image_shape || curr_transform != ref_transform)
                images[id] = tipl::resample(images[id], image_shape, tipl::from_space(ref_transform).to(curr_transform));
        }
    }
    return true;
}

bool dual_reg::load_subject(size_t id, const std::string& file_name)
{
    if(!id)
        clear_reg();
    return load_image(id, file_name, I, true, IR, Ivs, Is,Is_is_mni,error_msg);
}
bool dual_reg::load_template(size_t id, const std::string& file_name)
{
    if(!id)
        clear_reg();
    return load_image(id, file_name, It, false, ItR, Itvs, Its,It_is_mni,error_msg);
}


void dual_reg::match_resolution(bool use_vs)
{
    if(!data_ready())
        return;
    float ratio = (use_vs ? Itvs[0]/Ivs[0] : float(Is.width())/float(Its.width()));
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
    while(ratio <= 0.5f)
    {
        downsample(It,Its,Itvs,ItR);
        ratio *= 2.0f;
        tipl::out() << "downsampling template to " << Itvs[0] << " mm resolution" << std::endl;
    }
    while(ratio >= 2.0f)
    {
        downsample(I,Is,Ivs,IR);
        ratio /= 2.0f;
        tipl::out() << "downsample subject to " << Ivs[0] << " mm resolution" << std::endl;
    }
}

void dual_reg::show_r(const std::string& prompt)
{
    std::string result(prompt);
    for(size_t i = 0;i < r.size() && r[i] != 0.0;++i)
    {
        result += (i ? "," : " ");
        result += modality_names[i];
        result += " ";
        result += std::to_string(r[i]);
    }
    tipl::out() << result;
}
void dual_reg::compute_mapping_from_displacement(void)
{
    if(f2t_dis.empty() || t2f_dis.empty())
        return;
    auto trans = T();
    from2to.resize(Is);
    tipl::inv_displacement_to_mapping(f2t_dis,from2to,trans);
    tipl::displacement_to_mapping(t2f_dis,to2from,trans);
}
void dual_reg::calculate_linear_r(void)
{
    auto trans = T();
    std::fill(r.begin(),r.end(),0.0f);
    J.resize(max_modality);
    tipl::par_for(max_modality,[&](size_t i)
    {
        if(I[i].empty())
            return;
        r[i] = tipl::correlation(J[i] = tipl::resample(I[i],Its,trans),
                                 It[i].empty() ? It[0]:It[i]);
    },max_modality);
    show_r("linear r: ");
    if(match_fov)
        It_match_fov();
}
void dual_reg::It_match_fov(void)
{
    auto trans = T();
    size_t max_It = 1;
    while(max_It < max_modality && !It[max_It].empty())
        ++max_It;
    for(tipl::pixel_index<3> index(Its);index < Its.size();++index)
    {
        tipl::vector<3> pos;
        trans(index,pos);
        if(Is.is_valid(pos))
            continue;
        for(size_t i = 0;i < max_It;++i)
            It[i][index.index()] = 0;
    }
}

void dual_reg::calculate_nonlinear_r(void)
{
    std::fill(r.begin(),r.end(),0.0f);
    tipl::par_for(max_modality,[&](size_t i)
    {
        if(I[i].empty() || It[i].empty())
            return;
        r[i] = tipl::correlation(J[i] = tipl::compose_mapping(I[i],to2from),
                                 previous_It.empty() ? It[i] : previous_It[i]);
    },max_modality);
    show_r("nonlinear r: ");
}

float dual_reg::linear_reg(bool& terminated)
{
    if(!data_ready())
        return 0.0f;
    tipl::progress prog("linear registration");
    float cost = 0.0f;
    if(!skip_linear)
        cost = tipl::reg::linear<tipl::out>(tipl::reg::make_list(It),Itvs,tipl::reg::make_list(I),Ivs,
               arg,reg_type,terminated,bound,cost_type,use_cuda && has_cuda);

    calculate_linear_r();

    if(export_intermediate)
        for(size_t i = 0;i < I.size() && !I[i].empty();++i)
        {
            tipl::io::gz_nifti::save_to_file(("I" + std::to_string(i) + ".nii.gz").c_str(),I[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("It" + std::to_string(i) + ".nii.gz").c_str(),It[i],Itvs,ItR);
            tipl::io::gz_nifti::save_to_file(("J" + std::to_string(i) + ".nii.gz").c_str(),J[i],Itvs,ItR);
        }
    return cost;
}



void dual_reg::nonlinear_reg(bool& terminated)
{
    if(!data_ready())
        return;
    tipl::progress prog("nonlinear registration");
    f2t_dis.clear();
    t2f_dis.clear();
    if(skip_nonlinear)
    {
        f2t_dis.resize(Its);
        t2f_dis.resize(Its);
        return;
    }
    else
    {
        auto param0 = param;
        auto param1 = param;
        /*
        auto s2t = invT();
        for(size_t i = 0;i < anchor[0].size() && i < anchor[1].size();++i)
        {
            auto spos = anchor[0][i];
            auto tpos = anchor[1][i];
            s2t(spos);
            param0.anchor.push_back(std::make_pair(tpos,spos));
            param1.anchor.push_back(std::make_pair(spos,tpos));
            tipl::out() << "anchor: " << spos << "->" << tpos;
        }
        tipl::out() << "a total of " << param0.anchor.size() << " anchor points";
        */
        std::thread t([&](void)
        {
            tipl::reg::cdm_common<tipl::out>(tipl::reg::make_list(It),tipl::reg::make_list(J),t2f_dis,terminated,param0,use_cuda && has_cuda);
        });
        tipl::reg::cdm_common<tipl::out>(tipl::reg::make_list(J),tipl::reg::make_list(It),f2t_dis,terminated,param1,use_cuda && has_cuda);
        t.join();
        if(!previous_f2t.empty() && !previous_t2f.empty())
        {
            tipl::accumulate_displacement(previous_f2t,f2t_dis);
            tipl::accumulate_displacement(previous_t2f,t2f_dis);
        }
    }
    compute_mapping_from_displacement();
    calculate_nonlinear_r();

    if(export_intermediate)
    {
        for(size_t i = 0;i < J.size();++i)
            tipl::io::gz_nifti::save_to_file("J" + std::to_string(i) + ".nii.gz",J[i],Itvs,ItR);
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






template<bool direction>
bool dual_reg::apply_warping_tt(const char* input, const char* output) const
{
    auto fib = std::make_shared<fib_data>(direction ? Is : Its,direction ? Ivs : Itvs,direction ? IR : ItR);
    fib->is_mni = direction ? Is_is_mni : It_is_mni;
    TractModel tract_model(fib);
    if (!tract_model.load_tracts_from_file(input,fib.get(), false))
    {
        error_msg = "cannot read tract file";
        return false;
    }

    std::vector<std::vector<float>>& tracts = tract_model.get_tracts();
    const auto& mapping = direction ? from2to : to2from;
    const auto transform = direction ? invT() : T();
    tipl::adaptive_par_for(tracts.size(), [&](size_t i)
    {
        for (size_t j = 0; j < tracts[i].size(); j += 3)
        {
            tipl::vector<3> pos(&tracts[i][j]);
            if (!tipl::estimate(mapping, pos, pos))
                transform(pos);
            std::copy(pos.begin(), pos.end(), &tracts[i][j]);
        }
    });

    tract_model.geo = direction ? Its : Is;
    tract_model.vs = direction ? Itvs : Ivs;
    tract_model.trans_to_mni = direction ? ItR : IR;

    tipl::out() << "saving " << output;
    if (!tract_model.save_tracts_to_file(output))
    {
        error_msg = "failed to save file";
        return false;
    }
    return true;


}
template bool dual_reg::apply_warping_tt<false>(const char* input, const char* output) const;
template bool dual_reg::apply_warping_tt<true>(const char* input, const char* output) const;


template<bool direction>
bool dual_reg::apply_warping_nii(const char* input, const char* output) const
{
    tipl::out() << "opening " << input;
    auto input_size = (direction ? Is : Its);
    // check dimension
    {
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(input))
        {
            error_msg = nii.error_msg;
            return false;
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
        return false;
    }
    bool is_label = tipl::is_label_image(I3);
    tipl::out() << (is_label ? "label image interpolated using majority assignment " : "scalar image interpolated using spline") << std::endl;
    tipl::out() << "saving " << output;
    if(!tipl::io::gz_nifti::save_to_file(output,
                                         is_label ? apply_warping<direction,tipl::interpolation::majority>(I3) : apply_warping<direction,tipl::interpolation::cubic>(I3),
                                         direction ? Itvs : Ivs,
                                         direction ? ItR : IR,
                                         direction ? It_is_mni : Is_is_mni))
    {
        error_msg = "cannot write to file " + std::string(output);
        return false;
    }
    return true;
}

template bool dual_reg::apply_warping_nii<false>(const char* input, const char* output) const;
template bool dual_reg::apply_warping_nii<true>(const char* input, const char* output) const;


bool check_fib_dim_vs(tipl::io::gz_mat_read& mat_reader,
                      tipl::shape<3>& dim,tipl::vector<3>& vs,tipl::matrix<4,4>& trans,bool& is_mni);
tipl::const_pointer_image<3,unsigned char> handle_mask(tipl::io::gz_mat_read& mat_reader);
bool save_fz(tipl::io::gz_mat_read& mat_reader,
              tipl::io::gz_mat_write& matfile,
              const std::vector<std::string>& skip_list,
              const std::vector<std::string>& skip_head_list);
template<bool direction>
bool dual_reg::apply_warping_fzsz(const char* input,const char* output) const
{
    tipl::progress prog("apply warp");
    tipl::io::gz_mat_read mat_reader;
    mat_reader.delay_read = false;
    if(!mat_reader.load_from_file(input))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    tipl::io::gz_mat_write mat_writer(output);
    if(!mat_writer)
    {
        error_msg = std::string("cannot save file to ") + output;
        return false;
    }
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> trans((tipl::identity_matrix()));
    bool is_mni;
    if(!check_fib_dim_vs(mat_reader,dim,vs,trans,is_mni))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    handle_mask(mat_reader);
    if(dim != (direction ? Is : Its))
    {
        error_msg = "dimension does not match";
        return false;
    }
    if(trans != (direction ? IR : ItR))
    {
        error_msg = "transformation matrix does not match";
        return false;
    }

    size_t p = 0;
    bool failed = false;
    tipl::par_for(mat_reader.size(),[&](unsigned int i)
    {
        if(!prog(p++,mat_reader.size()) || failed)
            return;
        auto& mat = mat_reader[i];
        size_t mat_size = mat_reader.cols(i)*mat_reader.rows(i);
        if(mat_size == dim.size()) // image volumes, including fa, and fiber index
        {
            variant_image new_image;
            new_image.shape = dim;
            if(!new_image.read_mat_image(i,mat_reader))
                return;
            if(tipl::begins_with(mat.name,"index") || tipl::begins_with(mat.name,"mask"))
                new_image.interpolation = false;
            new_image.apply([&](auto& I)
            {
                if(new_image.interpolation)
                {
                    auto new_I = apply_warping<direction,tipl::interpolation::cubic>(tipl::image<3>(I));
                    tipl::lower_threshold(new_I,0.0f);
                    I = new_I;
                }
                else
                    I = apply_warping<direction,tipl::interpolation::majority>(I);
            });
            new_image.shape = (direction ? Its : Is);
            new_image.write_mat_image(i,mat_reader);
        }

    },tipl::max_thread_count);

    {
        if((direction ? Its : Is) != dim)
            mat_reader.write("dimension",(direction ? Its : Is).begin(),1,3);
        if((direction ? Itvs : Ivs) != vs)
            mat_reader.write("voxel_size",(direction ? Itvs : Ivs).begin(),1,3);
        if(mat_reader.has("trans") && (direction ? ItR : IR) != trans)
            mat_reader.write("trans",(direction ? ItR : IR).begin(),4,4);
    }
    return save_fz(mat_reader,mat_writer,{"odf_faces","odf_vertices","z0","mapping"},{"subject"});
}
template bool dual_reg::apply_warping_fzsz<false>(const char* input, const char* output) const;
template bool dual_reg::apply_warping_fzsz<true>(const char* input, const char* output) const;

bool dual_reg::load_warping(const std::string& filename)
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

    tipl::shape<3> sub_shape;
    sub_shape = tipl::shape<3>((Its[0]+1)/2,(Its[1]+1)/2,(Its[2]+1)/2);

    t2f_dis.resize(sub_shape);
    f2t_dis.resize(sub_shape);
    if(row*col != sub_shape.size()*3)
    {
        error_msg = "invalid displacement field";
        return false;
    }
    std::copy_n(f2t_dis_ptr,f2t_dis.size()*dimension,&f2t_dis[0][0]);
    std::copy_n(t2f_dis_ptr,t2f_dis.size()*dimension,&t2f_dis[0][0]);
    tipl::upsample_with_padding(t2f_dis,Its);
    tipl::upsample_with_padding(f2t_dis,Its);
    compute_mapping_from_displacement();
    if(It[0].shape() != Its)
    {
        It.clear();
        It.resize(max_modality);
    }
    if(I[0].shape() != Is)
    {
        I.clear();
        I.resize(max_modality);
    }
    return true;
}
bool dual_reg::load_alternative_warping(const std::string& filename)
{
    dual_reg alt_reg;
    tipl::out() << "opening alternative warping " << filename;
    if(!alt_reg.load_warping(filename) ||
        alt_reg.Is != alt_reg.Its ||
        alt_reg.IR != alt_reg.ItR ||
        alt_reg.arg != tipl::affine_transform<float,3>())
    {
        error_msg = "invalid alternative mapping";
        return false;
    }
    alt_reg.to_space(Its,ItR);
    for(auto& each : modality_names)
        each = "alt-"+each;
    previous_f2t.swap(alt_reg.t2f_dis);
    previous_t2f.swap(alt_reg.f2t_dis);
    previous_It.resize(It.size());
    for(size_t i = 0;i < It.size() && !It[i].empty();++i)
    {
        previous_It[i] = alt_reg.apply_warping<true,tipl::interpolation::cubic>(It[i]);
        previous_It[i].swap(It[i]);
    }
    return true;
}

bool dual_reg::save_warping(const char* filename) const
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

void dual_reg::dis_to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R)
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
void dual_reg::to_space(const tipl::shape<3>& new_s,const tipl::matrix<4,4>& new_R)
{
    if(new_s == Is && new_R == IR && new_s == Its && new_R == ItR)
        return;

    tipl::vector<3> new_vs(tipl::to_vs(new_R));
    if(ItR != IR || arg != tipl::affine_transform<float,3>())
        arg = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_R).to(ItR))
            .accumulate(T())
            .accumulate(tipl::transformation_matrix<float,dimension>(tipl::from_space(IR).to(new_R)))
            .to_affine_transform(new_s,new_vs,new_s,new_vs);

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
void dual_reg::to_I_space(const tipl::shape<3>& new_Is,const tipl::matrix<4,4>& new_IR)
{
    if(new_Is == Is && new_IR == IR)
        return;
    auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_IR).to(IR));
    for(auto& each : I)
        if(!each.empty())
            each = tipl::resample(each,new_Is,trans);
    arg = T().accumulate(tipl::from_space(IR).to(new_IR)).
            to_affine_transform(Its,Itvs,new_Is,Ivs);
    Is = new_Is;
    IR = new_IR;
    Ivs = tipl::to_vs(IR);
    compute_mapping_from_displacement();
}
void dual_reg::to_It_space(const tipl::shape<3>& new_Its,const tipl::matrix<4,4>& new_ItR)
{
    if(new_Its == Its && new_ItR == ItR)
        return;
    tipl::vector<3> new_Itvs(tipl::to_vs(new_ItR));
    auto trans = tipl::transformation_matrix<float,dimension>(tipl::from_space(new_ItR).to(ItR));
    for(auto& each : It)
        if(!each.empty())
            each = tipl::resample(each,new_Its,trans);
    dis_to_space(new_Its,new_ItR);
    arg = trans.accumulate(T()).to_affine_transform(new_Its,new_Itvs,Is,Ivs);
    Its = new_Its;
    ItR = new_ItR;
    Itvs = new_Itvs;
    compute_mapping_from_displacement();
}
void dual_reg::to_It_space(const tipl::shape<3>& new_Its)
{
    auto new_ItR = ItR;
    for(int i = 0;i < 3;++i)
        new_ItR[3+i*4] -= new_ItR[i*5]*(float(new_Its[i])-float(Its[i]))*0.5f;
    to_It_space(new_Its,new_ItR);
}


template<bool direction>
bool save_warping(dual_reg& r,
                  const std::vector<std::string>& apply_warp_filename,
                  const std::string& output_dir)
{
    for(const auto& each_file: apply_warp_filename)
    {
        std::string post_fix;
        if(tipl::ends_with(each_file,".nii.gz"))
            post_fix = ".nii.gz";
        if(tipl::ends_with(each_file,".tt.gz"))
            post_fix = ".tt.gz";
        if(tipl::ends_with(each_file,".sz"))
            post_fix = ".sz";
        if(tipl::ends_with(each_file,".fz"))
            post_fix = ".fz";
        if(post_fix.empty())
        {
            tipl::error() << "unsupported file format: " << each_file;
            return false;
        }
        if constexpr(direction)
            post_fix = ".wp" + post_fix;
        else
            post_fix = ".uwp" + post_fix;
        std::string output_file;
        if (!output_dir.empty())
        {
            std::filesystem::create_directories(output_dir);
            output_file = (std::filesystem::path(output_dir) / (std::filesystem::path(each_file).filename().string() + post_fix)).string();
        }
        else
            output_file = each_file + post_fix;

        tipl::out() << (direction ? "warping " : "unwarping") << each_file << " into " << output_file;
        if(!r.apply_warping<direction>(each_file.c_str(),output_file.c_str()))
        {
            tipl::error() << r.error_msg;
            return false;
        }
    }
    return true;
}

int reg(tipl::program_option<tipl::out>& po)
{
    dual_reg r;
    std::vector<std::string> from_filename(tipl::split(po.get("source"),',')),to_filename(tipl::split(po.get("to"),','));
    tipl::out() << from_filename.size() << " file(s) specified at --source";
    tipl::out() << to_filename.size() << " file(s) specified at --to";
    if(po.has("mapping"))
    {
        if(from_filename.empty() && to_filename.empty())
        {
            tipl::error() << "please specify images to warp/unwwarp at --source/--to, respectively.";
            return 1;
        }
        tipl::out() << "loading mapping field";
        if(!r.load_warping(po.get("mapping")))
        {
            tipl::error() << r.error_msg;
            return 1;
        }
        tipl::out() << "dim: " << r.Is << " to " << r.Its;
        tipl::out() << "vs: " << r.Ivs << " to " << r.Itvs;
        bool good = true;
        if(!save_warping<true>(r,from_filename,po.get("output")) ||
           !save_warping<false>(r,to_filename,po.get("output")))
            return 1;
        return 0;
    }


    if(from_filename.empty() || to_filename.empty())
    {
        tipl::error() << "please specify the images to warp using --source and --to";
        return 1;
    }

    if(!po.get("overwrite",0))
    {
        bool skip = true;
        for(const auto& each_file: from_filename)
        {
            if((tipl::ends_with(each_file,".tt.gz") && !std::filesystem::exists(each_file+".wp.tt.gz")) ||
               (tipl::ends_with(each_file,".nii.gz") && !std::filesystem::exists(each_file+".wp.nii.gz")))
            {
                skip = false;
                break;
            }
        }
        if(skip)
        {
            tipl::out() << "output file exists, skipping";
            return 0;
        }
    }


    for(size_t i = 0;i < from_filename.size() && i < to_filename.size();++i)
    {
        if(!r.load_subject(i,from_filename[i]))
        {
            tipl::error() << r.error_msg;
            return 1;
        }

        if(!r.load_template(i,to_filename[i]))
        {
            tipl::error() << r.error_msg;
            return 1;
        }

        r.modality_names[i] = std::filesystem::path(from_filename[i]).stem().stem().string() + "->" +
                              std::filesystem::path(to_filename[i]).stem().stem().string();
    }

    tipl::out() << "source dim: " << r.Is;
    tipl::out() << "to dim: " << r.Its;
    r.match_resolution(po.get("match_vs",1));
    tipl::out() << "running linear registration." << std::endl;

    if(po.get("large_deform",0))
        r.bound = tipl::reg::large_bound;
    r.reg_type = po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine;
    r.cost_type = po.get("cost_function",r.reg_type==tipl::reg::rigid_body ? "mi" : "corr") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr;
    r.skip_linear = po.get("skip_linear",r.skip_linear);
    r.skip_nonlinear = po.get("skip_nonlinear",r.skip_nonlinear);

    r.linear_reg(tipl::prog_aborted);

    if(r.reg_type != tipl::reg::rigid_body)
    {
        r.param.resolution = po.get("resolution",r.param.resolution);
        r.param.speed = po.get("speed",r.param.speed);
        r.param.smoothing = po.get("smoothing",r.param.smoothing);
        r.param.min_dimension = po.get("min_dimension",r.param.min_dimension);
        r.nonlinear_reg(tipl::prog_aborted);
    }
    if(po.has("output_mapping") && !r.save_warping(po.get("output_mapping").c_str()))
    {
        tipl::error() << r.error_msg;
        return 1;
    }
    if(!save_warping<true>(r,tipl::split(po.get("s2t",po.get("source")),','),po.get("output")) ||
       !save_warping<false>(r,tipl::split(po.get("t2s"),','),po.get("output")))
        return 1;
    if(po.get("export_r",0))
        std::ofstream(from_filename.front() + ".r" + std::to_string(int(r.r[0]*100))) << std::endl;
    return 0;
}
