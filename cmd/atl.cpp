#include <filesystem>
#include <fstream>
#include "fib_data.hpp"
#include "mapping/atlas.hpp"
#include "connectometry/group_connectometry_analysis.h"

bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list)
{
    for(const auto& each : tipl::split(atlas_name,','))
    {
        if(auto at = handle->get_atlas(each))
            atlas_list.push_back(at);
        else
            return tipl::error() << handle->error_msg,false;
    }
    return true;
}

extern std::vector<std::string> template_name_list;
extern std::vector<std::filesystem::path> t2w_template_list;
// native 4D BOLD + BIDS events -> fmri.dz on the template fib mask
int fmri_db(tipl::program_option<tipl::out>& po)
{
    const std::filesystem::path source = po.get("source"),events = po.get("events");
    tipl::io::gz_nifti nii;
    if(!nii.open(source,std::ios::in))
        return tipl::error() << nii.error_msg,1;
    const size_t T = nii.dim(4);

    // TR from --tr, otherwise NIfTI pixdim[4] with xyzt time units (8: s, 16: ms, 24: us)
    const int units = nii.nif_header.xyzt_units & 0x38;
    const float tr = po.has("tr") ? po.get("tr",0.0f) :
                     nii.nif_header.pixdim[4]*(units == 8 ? 1.0f : units == 16 ? 1.0e-3f : units == 24 ? 1.0e-6f : 0.0f);
    if(tr <= 0.0f)
        return tipl::error() << "cannot resolve TR from the NIfTI header. please specify --tr",1;

    std::vector<tipl::image<3> > frames(T);
    tipl::vector<3> vs;
    tipl::matrix<4,4> trans;
    for(auto& each : frames)
        nii >> each;
    nii >> vs >> trans;
    if(!nii.error_msg.empty())
        return tipl::error() << nii.error_msg,1;

    // events file stored as text (like demo), parsed with the HRF after loading the .dz
    std::ifstream in(events);
    if(!in)
        return tipl::error() << "cannot read " << events,1;
    const std::string event_text((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());

    fib_data fib;
    if(!fib.load_template_fib(po.get("template",template_name_list,size_t(0)),po.get("template_reso",2.0f)))
        return tipl::error() << fib.error_msg,1;

    // register mean BOLD to the t2w template, then map onto the template fib grid
    tipl::image<3> mean(frames[0].shape());
    for(const auto& each : frames)
        for(size_t i = 0;i < mean.size();++i)
            mean[i] += each[i]/float(T);
    tipl::reg::mm_reg<tipl::out> r;
    if(!r.load_template<tipl::io::gz_nifti>(0,t2w_template_list[fib.template_id]))
        return tipl::error() << "cannot load t2w template: " << r.error_msg,1;
    r.I[0] = tipl::reg::subject_image_pre(std::move(mean));
    r.Is = frames[0].shape();
    r.Ivs = vs;
    r.IR = trans;
    r.match_resolution(true);
    r.linear_reg(tipl::prog_aborted);
    r.nonlinear_reg(tipl::prog_aborted);
    if(tipl::prog_aborted)
        return tipl::error() << "aborted",1;
    const float R2 = tipl::max_value(r.r);
    if(R2 < 0.3f)
        return tipl::error() << "registration failed. R2=" << R2,1;
    r.to_I_space(frames[0].shape(),trans);
    r.to_It_space(fib.dim,fib.trans_to_mni);

    // warp each frame with cubic interpolation and sample at the template mask
    const auto& si2vi = fib.mat_reader.si2vi;
    const size_t M = si2vi.size();
    auto bold_mat = std::make_shared<tipl::io::mat_matrix>("bold",float(0),uint32_t(T),uint32_t(M));
    fib.mat_reader.push_back(bold_mat);
    {
        auto bold = bold_mat->get_data<float>();
        tipl::progress prog("warp BOLD frames");
        for(size_t t = 0;prog(t,T);++t)
        {
            auto J = r.apply_warping<true,tipl::interpolation::cubic>(frames[t]);
            for(size_t s = 0;s < M;++s)
                bold[t*M+s] = J[si2vi[s]];
        }
        if(prog.aborted())
            return tipl::error() << "aborted",1;
    }

    // fib_data::save_to_file keeps the template backbone and stores masked bold[T,M] sloped
    fib.mat_reader.write("events",event_text.c_str(),1,uint32_t(event_text.size()+1));
    fib.mat_reader.write("TR",&tr,1,1);
    fib.report += " The task fMRI BOLD image (" + source.filename().u8string() + ", TR=" + std::to_string(tr) +
                  " s) was registered to the T2w template using linear and nonlinear registration (R2=" + std::to_string(R2) +
                  ") and sampled at the template mask using cubic interpolation. Task events were taken from " +
                  events.filename().u8string() + ".";
    if(!fib.save_to_file(po.get("output","fmri.dz")))
        return tipl::error() << "cannot save fmri db file: " << fib.error_msg,1;
    return 0;
}
int db(tipl::program_option<tipl::out>& po)
{
    if(po.has("events"))
        return fmri_db(po);
    if(po.has("match")) // turn an existing database into a longitudinal-change database
    {
        fib_data fib;
        if(!fib.load_from_file(po.get("source")) || !fib.db.has_db())
            return tipl::error() << "not a database file: " << fib.error_msg,1;

        std::string match_arg = po.get("match");
        if(match_arg == "consecutive")
            fib.db.match_consecutive_subjects();
        else if(!fib.db.match_subjects_from_file(match_arg))
            return tipl::error() << fib.error_msg,1;

        if(!fib.db.calculate_change(uint8_t(po.get("dif_type",0)),
                                     uint8_t(po.get("filter_type",0)),
                                     po.get("normalize_iso",1)))
            return tipl::error() << fib.error_msg,1;

        if(!fib.save_to_file(po.get("output","longitudinal"+fib.db.suggest_output_suffix())))
            return tipl::error() << "cannot save db file:" << fib.error_msg,1;
        return 0;
    }

    auto name_list = po.get_files("source","*.qsdr.fz");
    if(name_list.empty())
        return tipl::error() << po.error_msg,1;

    std::string default_demo;
    if(po.has("demo"))
        default_demo = po.get("demo");
    else
    {
        if(std::filesystem::exists("../participants.tsv"))
            default_demo = "../participants.tsv";
        if(std::filesystem::exists("participants.tsv"))
            default_demo = "participants.tsv";
    }
    if(!default_demo.empty() && !std::filesystem::exists(default_demo))
        return tipl::error() << "cannot find file specified at --demo",1;


    std::vector<std::string> index_list;
    float template_reso = 1.0f;
    {
        fib_data fib;
        if(!fib.load_from_file(name_list[0]))
            return tipl::error() << "cannot load subject fib " << name_list[0],1;
        if(po.has("index_name"))
            index_list = tipl::split(po.get("index_name"),',');
        if(index_list.empty())
            index_list = fib.get_index_list();
        template_reso = fib.vs[0];
    }
    {
        fib_data fib;
        if(!fib.load_template_fib(
               po.get("template",template_name_list,size_t(0)),template_reso) ||
           !fib.db.create_db(name_list,index_list) ||
           (!default_demo.empty() && !fib.db.parse_demo(po.get("demo",default_demo))))
            return tipl::error() << fib.error_msg,1;

        if(po.has("intro"))
        {
            std::ifstream file(po.get("intro"));
            fib.intro = std::string(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
        }

        if(!fib.save_to_file(po.get("output","sub-all.dz")))
            return tipl::error() << "cannot save db file:" << fib.error_msg,1;

        float outlier_threshold = tipl::outlier_range(fib.db.R2.begin(),fib.db.R2.end()).first;
        for(size_t i = 0;i < fib.db.subject_names.size();++i)
            if(fib.db.R2[i] < outlier_threshold)
                tipl::warning() << "low R2: " << fib.db.subject_names[i];
    }
    return 0;
}
bool odf_average(const std::filesystem::path& out_name,std::vector<std::filesystem::path>& file_names,std::string& error_msg);
bool dwi_average(const std::filesystem::path& out_name,std::vector<std::filesystem::path>& file_names,std::string& error_msg);
int tmp(tipl::program_option<tipl::out>& po)
{
    auto name_list = po.get_files("source","*.fz");
    if(name_list.empty())
        return tipl::error() << po.error_msg,1;
    std::string error_msg,output;
    auto source = name_list.front().u8string();
    tipl::out() << "constructing a group average template";
    if(tipl::ends_with(source,{".fib.gz",".fz"}))
        output = (name_list.front().parent_path()/"template").u8string();
    else if(tipl::ends_with(source,".nii.gz"))
        output = source+".avg.nii.gz";
    else
        return tipl::error() << "unsupported format",1;
    output = po.get("output",output);
    return (tipl::ends_with(source,".fz") && tipl::ends_with(output,".sz") ?
                dwi_average(output,name_list,error_msg) : odf_average(output,name_list,error_msg)) ?
               0 : (tipl::error() << error_msg,1);
}
std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);
int atl(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po);
    if(!handle.get())
        return 1;
    std::vector<std::shared_ptr<atlas> > atlas_list;
    if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
        return 1;
    if(handle->get_sub2temp_mapping().empty())
        return tipl::error() << "cannot output connectivity: no mni mapping",1;
    std::string file_name = po.get("source");
    const auto& mapping = handle->get_sub2temp_mapping();
    bool multiple = (po.get("output","multiple") == "multiple");

    for(unsigned int i = 0;i < atlas_list.size();++i)
    {
        std::string base_name = file_name;
        base_name += ".";
        base_name += atlas_list[i]->name;
        for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
        {
            std::string output = base_name;
            output += ".";
            output += atlas_list[i]->get_list()[j];
            output += ".nii.gz";
            tipl::image<3,unsigned char> roi(handle->dim);
            for(unsigned int k = 0;k < mapping.size();++k)
                if (atlas_list[i]->is_labeled_as(mapping[k], j))
                    roi[k] = 1;
            if(multiple)
                tipl::io::gz_nifti(output,std::ios::out) << handle->bind(roi);
        }
        {
            std::ofstream txt_out(base_name + ".txt");
            for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
                txt_out << atlas_list[i]->get_num()[j] << " " << atlas_list[i]->get_list()[j] << std::endl;
        }
    }
    return 0;
}
