#include <QString>
#include <QImage>
#include "img.hpp"

#include "tract_model.hpp"

template<bool direction,typename reg_type>
bool apply_warping_tt(const reg_type& reg,const std::string& input, const std::string& output)
{
    auto fib = std::make_shared<fib_data>(direction ? reg.Is : reg.Its,direction ? reg.Ivs : reg.Itvs,direction ? reg.IR : reg.ItR);
    fib->is_mni = direction ? reg.Is_is_mni : reg.It_is_mni;
    TractModel tract_model(fib);
    if (!tract_model.load_tracts_from_file(input,fib.get(), false))
        return reg.error_msg = "cannot read tract file",false;

    std::vector<std::vector<float>>& tracts = tract_model.get_tracts();
    const auto& mapping = direction ? reg.from2to : reg.to2from;
    const auto transform = direction ? reg.invT() : reg.T();
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

    tract_model.geo = direction ? reg.Its : reg.Is;
    tract_model.vs = direction ? reg.Itvs : reg.Ivs;
    tract_model.trans_to_mni = direction ? reg.ItR : reg.IR;

    if (!tract_model.save_tracts_to_file(output))
        return reg.error_msg = "failed to save file",false;
    return true;
}

template<bool direction,typename reg_type>
bool apply_warping_nii(const reg_type& reg,const std::string& input, const std::string& output)
{
    auto input_size = (direction ? reg.Is : reg.Its);
    // check dimension
    {
        tipl::shape<3> d;
        if(!(tipl::io::gz_nifti(input,std::ios::in) >> d >>
                [&](const std::string& e){tipl::error() << (reg.error_msg = e);}))
            return false;
        if(d != input_size)
        {
            tipl::warning() << std::filesystem::path(input).filename().string() << " has a size of " << d << " different from the expected size " << input_size;
            tipl::warning() << "transformation will be applied. But please make sure you apply the mapping in the correct direction.";
        }
    }
    tipl::image<3> I3(input_size);
    if(!tipl::io::gz_nifti(input,std::ios::in).to_space<tipl::interpolation::check>(I3,direction ? reg.IR : reg.ItR))
        return reg.error_msg = "cannot open " + std::string(input),false;

    bool is_label = tipl::is_label_image(I3);
    tipl::out() << (is_label ? "label image interpolated using majority assignment " : "scalar image interpolated using spline") << std::endl;
    auto I = is_label ? reg.template apply_warping<direction,tipl::interpolation::majority>(I3) : reg.template apply_warping<direction,tipl::interpolation::cubic>(I3);
    return tipl::io::gz_nifti(output,std::ios::out)
            << (direction ? reg.Itvs : reg.Ivs) << (direction ? reg.ItR : reg.IR) << (direction ? reg.It_is_mni : reg.Is_is_mni) << I
            << [&](const std::string& e){tipl::error() << (reg.error_msg = e);};
}

bool check_fib_dim_vs(tipl::io::gz_mat_read& mat_reader,
                      tipl::shape<3>& dim,tipl::vector<3>& vs,tipl::matrix<4,4>& trans,bool& is_mni);
tipl::const_pointer_image<3,unsigned char> handle_mask(tipl::io::gz_mat_read& mat_reader);
bool save_fz(tipl::io::gz_mat_read& mat_reader,
              tipl::io::gz_mat_write& matfile,
              const std::vector<std::string>& skip_list,
              const std::vector<std::string>& skip_head_list);
template<bool direction,typename reg_type>
bool apply_warping_fzsz(const reg_type& reg,const std::string& input,const std::string& output)
{
    tipl::progress prog("apply warp");
    tipl::io::gz_mat_read mat_reader;
    mat_reader.delay_read = false;
    if(!mat_reader.load_from_file(input))
        return reg.error_msg = mat_reader.error_msg,false;
    tipl::io::gz_mat_write mat_writer(output);
    if(!mat_writer)
        return reg.error_msg = std::string("cannot save file to ") + output,false;
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> trans((tipl::identity_matrix()));
    bool is_mni;
    if(!check_fib_dim_vs(mat_reader,dim,vs,trans,is_mni))
        return reg.error_msg = mat_reader.error_msg,false;
    handle_mask(mat_reader);
    if(dim != (direction ? reg.Is : reg.Its))
        return reg.error_msg = "dimension does not match",false;
    if(trans != (direction ? reg.IR : reg.ItR))
        return reg.error_msg = "transformation matrix does not match",false;

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
            if(tipl::begins_with(mat.name,{"index","mask"}))
                new_image.interpolation = false;
            new_image.apply([&](auto& I)
            {
                if(new_image.interpolation)
                {
                    auto new_I = reg.template apply_warping<direction,tipl::interpolation::cubic>(tipl::image<3>(I));
                    tipl::lower_threshold(new_I,0.0f);
                    I = new_I;
                }
                else
                    I = reg.template apply_warping<direction,tipl::interpolation::majority>(I);
            });
            new_image.shape = (direction ? reg.Its : reg.Is);
            new_image.write_mat_image(i,mat_reader);
        }

    },tipl::max_thread_count);

    {
        if((direction ? reg.Its : reg.Is) != dim)
            mat_reader.write("dimension",(direction ? reg.Its : reg.Is).begin(),1,3);
        if((direction ? reg.Itvs : reg.Ivs) != vs)
            mat_reader.write("voxel_size",(direction ? reg.Itvs : reg.Ivs).begin(),1,3);
        if(mat_reader.has("trans") && (direction ? reg.ItR : reg.IR) != trans)
            mat_reader.write("trans",(direction ? reg.ItR : reg.IR).begin(),4,4);
    }
    return save_fz(mat_reader,mat_writer,{"odf_faces","odf_vertices","z0","mapping"},{"subject"});
}

template<bool direction,typename reg_type>
bool apply_warping(const reg_type& reg,const std::string& input,const std::string& output)
{
    if(tipl::ends_with(input,".tt.gz"))
        return apply_warping_tt<direction>(reg,input,output);
    if(tipl::ends_with(input,{".nii.gz",".nii"}))
        return apply_warping_nii<direction>(reg,input,output);
    if(tipl::ends_with(input,{".sz",".src.gz",".fz",".fib.gz"}))
        return apply_warping_fzsz<direction>(reg,input,output);
    reg.error_msg = "unsupported file format";
    return false;
}

extern int map_ver;
bool load_warping(tipl::reg::mm_reg<tipl::out>& reg,const std::string& filename)
{
    tipl::io::gz_mat_read in;
    if(!in.load_from_file(filename))
        return reg.error_msg = "cannot read file " + filename,false;

    if(in.read_as_value<int>("version") > map_ver)
        return reg.error_msg = "incompatible map file format: the version "
                + std::to_string(in.read_as_value<int>("version"))
                + " is not supported within current rage "
                + std::to_string(map_ver),false;

    const float* f2t_dis_ptr = nullptr;
    const float* t2f_dis_ptr = nullptr;
    unsigned int row,col;
    if (!in.read("dimension",reg.Its) ||
        !in.read("voxel_size",reg.Itvs) ||
        !in.read("trans",reg.ItR) ||
        !in.read("dimension_from",reg.Is) ||
        !in.read("voxel_size_from",reg.Ivs) ||
        !in.read("trans_from",reg.IR) ||
        !in.read("f2t_dis",row,col,f2t_dis_ptr) ||
        !in.read("t2f_dis",row,col,t2f_dis_ptr) ||
        !in.read("arg",reg.arg))
        return reg.error_msg = "invalid warp file format",false;

    tipl::shape<3> sub_shape;
    sub_shape = tipl::shape<3>((reg.Its[0]+1)/2,(reg.Its[1]+1)/2,(reg.Its[2]+1)/2);

    reg.t2f_dis.resize(sub_shape);
    reg.f2t_dis.resize(sub_shape);
    if(row*col != sub_shape.size()*3)
        return reg.error_msg = "invalid displacement field",false;

    std::copy_n(f2t_dis_ptr,reg.f2t_dis.size()*3,&reg.f2t_dis[0][0]);
    std::copy_n(t2f_dis_ptr,reg.t2f_dis.size()*3,&reg.t2f_dis[0][0]);
    tipl::upsample_with_padding(reg.t2f_dis,reg.Its);
    tipl::upsample_with_padding(reg.f2t_dis,reg.Its);
    reg.compute_mapping_from_displacement();
    if(reg.It[0].shape() != reg.Its)
    {
        reg.It.clear();
        reg.It.resize(reg.max_modality);
    }
    if(reg.I[0].shape() != reg.Is)
    {
        reg.I.clear();
        reg.I.resize(reg.max_modality);
    }
    return true;
}


bool save_warping(const tipl::reg::mm_reg<tipl::out>& reg,const std::string& filename)
{
    tipl::progress prog("saving ",filename);
    if(reg.f2t_dis.empty() || reg.t2f_dis.empty())
        return reg.error_msg = "no mapping matrix to save",false;
    std::string output_name(filename);
    if(tipl::ends_with(output_name,".nii.gz"))
    {
        tipl::image<4> buffer(reg.from2to.shape().expand(3)),
                                 buffer2(reg.to2from.shape().expand(3));
        tipl::par_for(6,[&](unsigned int d)
        {
            if(d < 3)
            {
                size_t shift = d*reg.from2to.size();
                for(size_t i = 0;i < reg.from2to.size();++i)
                    buffer[i+shift] = reg.from2to[i][d];
            }
            else
            {
                d -= 3;
                size_t shift = d*reg.to2from.size();
                for(size_t i = 0;i < reg.to2from.size();++i)
                    buffer2[i+shift] = reg.to2from[i][d];
            }
        },6);
        tipl::io::gz_nifti(output_name,std::ios::out) << reg.Ivs << reg.IR << buffer;
        tipl::io::gz_nifti(output_name.substr(0,output_name.length()-7) + ".inv.nii.gz",std::ios::out) << reg.Itvs << reg.ItR << buffer2;
        return true;
    }

    if(!tipl::ends_with(output_name,".mz"))
        output_name += ".mz";
    std::string output_name_tmp = output_name + ".tmp";
    tipl::io::gz_mat_write out(output_name_tmp);
    if(!out)
        return reg.error_msg = "cannot write to a temporary file " + output_name_tmp,false;
    out.apply_slope = true;
    tipl::image<3,tipl::vector<3> > f2t_dis_sub2,t2f_dis_sub2;
    tipl::downsample_with_padding(reg.f2t_dis,f2t_dis_sub2);
    tipl::downsample_with_padding(reg.t2f_dis,t2f_dis_sub2);
    out.write<tipl::io::sloped>("f2t_dis",&f2t_dis_sub2[0][0],3,f2t_dis_sub2.size());
    out.write("dimension",reg.Its);
    out.write("voxel_size",reg.Itvs);
    out.write("trans",reg.ItR);

    out.write<tipl::io::sloped>("t2f_dis",&t2f_dis_sub2[0][0],3,t2f_dis_sub2.size());
    out.write("dimension_from",reg.Is);
    out.write("voxel_size_from",reg.Ivs);
    out.write("trans_from",reg.IR);

    out.write("arg",reg.arg);

    tipl::matrix<4,4> mat;
    reg.T().to(mat);
    out.write("T",mat.data(),4,4);

    out.write("version",map_ver);
    out.close();
    std::error_code error;
    std::filesystem::rename(output_name_tmp,output_name);
    if(error)
    {
        reg.error_msg = error.message();
        std::filesystem::remove(output_name_tmp);
        return false;
    }
    return true;
}

template<bool direction,typename reg_type>
bool save_warping(reg_type& r,
                  const std::vector<std::string>& files,
                  const std::string& out_dir)
{
    for(const auto& file : files)
    {
        std::string ext;
        for(const auto* e : {".nii.gz", ".tt.gz", ".sz", ".fz"})
            if (tipl::ends_with(file, e))
                ext = e;

        if(ext.empty())
            return tipl::error() << "unsupported format: " << file, false;

        ext = (direction ? ".wp" : ".uwp") + ext;
        std::string out_file = file + ext;

        if (!out_dir.empty())
        {
            std::filesystem::create_directories(out_dir);
            auto fname = std::filesystem::path(file).filename().string();
            out_file = (std::filesystem::path(out_dir) / (fname + ext)).string();
        }

        tipl::out() << (direction ? "warping " : "unwarping") << file << " into " << out_file;

        if (!apply_warping<direction>(r,file,out_file))
            return tipl::error() << r.error_msg, false;
    }
    return true;
}

int reg(tipl::program_option<tipl::out>& po)
{
    tipl::reg::mm_reg<tipl::out> r;
    std::vector<std::string> from_filename(tipl::split(po.get("source"),',')),to_filename(tipl::split(po.get("to"),','));
    tipl::out() << from_filename.size() << " file(s) specified at --source";
    tipl::out() << to_filename.size() << " file(s) specified at --to";
    if(po.has("mapping"))
    {
        tipl::out() << "loading mapping field";
        if(!load_warping(r,po.get("mapping")))
            return tipl::error() << r.error_msg,1;
        if(po.has("output_mapping") && !save_warping(r,po.get("output_mapping")))
            return tipl::error() << r.error_msg,1;
        if(from_filename.empty() && to_filename.empty())
            return tipl::error() << "please specify images to warp/unwwarp at --source/--to, respectively.",1;
        tipl::out() << "dim: " << r.Is << " to " << r.Its;
        tipl::out() << "vs: " << r.Ivs << " to " << r.Itvs;
        bool good = true;
        if(!save_warping<true>(r,from_filename,po.get("output")) ||
           !save_warping<false>(r,to_filename,po.get("output")))
            return 1;
        return 0;
    }


    if(from_filename.empty() || to_filename.empty())
        return tipl::error() << "please specify the images to warp using --source and --to",1;
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
            return tipl::out() << "output file exists, skipping",0;
    }


    for(size_t i = 0;i < from_filename.size() && i < to_filename.size();++i)
    {
        if(!r.load_subject<tipl::io::gz_nifti>(i,from_filename[i]) ||
           !r.load_template<tipl::io::gz_nifti>(i,to_filename[i]))
            return tipl::error() << r.error_msg,1;
        r.modality_names[i] = std::filesystem::path(from_filename[i]).stem().stem().string() + "->" +
                              std::filesystem::path(to_filename[i]).stem().stem().string();
    }

    tipl::out() << "source dim: " << r.Is << " to dim: " << r.Its;
    r.match_resolution(po.get("match_vs",1));
    tipl::out() << "running linear registration." << std::endl;

    if(po.get("large_deform",0))
        r.linear_param.bound = tipl::reg::large_bound;
    r.linear_param.reg_type = po.get("reg_type",1) == 0 ? tipl::reg::rigid_body : tipl::reg::affine;
    r.linear_param.cost_type = po.get("cost_function",r.linear_param.reg_type==tipl::reg::rigid_body ? "mi" : "corr") == std::string("mi") ? tipl::reg::mutual_info : tipl::reg::corr;
    r.skip_linear = po.get("skip_linear",r.skip_linear);
    r.skip_nonlinear = po.get("skip_nonlinear",r.skip_nonlinear);

    r.linear_reg(tipl::prog_aborted);

    if(r.linear_param.reg_type != tipl::reg::rigid_body)
    {
        r.param.resolution = po.get("resolution",r.param.resolution);
        r.param.speed = po.get("speed",r.param.speed);
        r.param.smoothing = po.get("smoothing",r.param.smoothing);
        r.param.min_dimension = po.get("min_dimension",r.param.min_dimension);
        r.nonlinear_reg(tipl::prog_aborted);
    }
    if(po.has("output_mapping") && !save_warping(r,po.get("output_mapping")))
        return tipl::error() << r.error_msg,1;
    if(!save_warping<true>(r,tipl::split(po.get("s2t",po.get("source")),','),po.get("output")) ||
       !save_warping<false>(r,tipl::split(po.get("t2s"),','),po.get("output")))
        return 1;
    if(po.get("export_r",0))
        std::ofstream(from_filename.front() + ".r" + std::to_string(int(r.r[0]*100))) << std::endl;
    return 0;
}
