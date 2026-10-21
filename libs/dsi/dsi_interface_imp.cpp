#define _MATH_DEFINES_DEFINED
#include "tessellated_icosahedron.hpp"
#include "basic_voxel.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"
#include "gqi_process.hpp"
#include "gqi_mni_reconstruction.hpp"
#include "image_model.hpp"
#include "fib_data.hpp"
#include "hist_process.hpp"

extern std::vector<std::string> template_name_list;
void src_data::check_output_file_name(void)
{
    if(tipl::ends_with(output_file_name.u8string(),{".fz",".fib.gz"}))
        return;
    if(output_file_name.empty()) // the default output never reuses the source name, e.g. a qsdr fz source
        output_file_name = file_name;
    output_file_name = tipl::remove_all_suffix(output_file_name);
    if(voxel.is_histology)
    {
        output_file_name += ".hist.fz";
        return;
    }
    if(voxel.method_id != 1 && voxel.needs("odf"))
        output_file_name += ".odf";

    switch (voxel.method_id)
    {
    case 1://DTI
        output_file_name += ".dti.fz";
        return;
    case 4://GQI
        output_file_name += (voxel.r2_weighted ? ".gqi2.fz":".gqi.fz");
        return;
    case 7:
        output_file_name += (voxel.r2_weighted ? ".qsdr2.fz":".qsdr.fz");
        return;
    default:
        output_file_name += ".fz";
    }
}


extern int fib_ver;
bool src_data::reconstruction_hist(void)
{
    if(!voxel.init_process<
            ReadImages,
            CalculateGradient,
            CalculateStructuralTensor,
            EigenAnalysis>() ||
       !voxel.run_hist())
    {
        error_msg = "reconstruction canceled";
        return false;
    }
    voxel.recon_report << "The parallel processing of histology image were done by tessellation whole slide image into smaller image block with overlapping margin to eliminate boundary effects (Yeh, et al. J Pathol Inform 2014,  5:1).";
    voxel.recon_report << " A total of " << voxel.hist_raw_smoothing << " smoothing iterations were applied to raw image.";
    voxel.recon_report << " Structural tensors were calculated to derive structural orientations and anisotropy (Zhang, IEEEE TMI 35, 294-306 2016, Schurr, Science, 2021) using a Gaussian kernel of " << voxel.hist_tensor_smoothing << " pixel spacing.";
    if(voxel.hist_downsampling)
        voxel.recon_report << " The results were exported at 2^" << voxel.hist_downsampling << " of the original pixel spacing.";
    save_fib();
    return true;
}
bool src_data::reconstruction(void)
{
    voxel.recon_report.clear();
    voxel.recon_report.str("");
    voxel.step_report.clear();
    voxel.step_report.str("");


    try
    {
        if(voxel.is_histology)
            return reconstruction_hist();

        switch (voxel.method_id)
        {
        case 1://DTI
            voxel.step_report << "[Step T2b]=DTI" << std::endl;
            if (!reconstruct2<ReadDWIData,
                    Dwi2Tensor>("DTI reconstruction"))
                return false;
            break;
        case 4://GQI
            voxel.step_report << "[Step T2b]=GQI" << std::endl;
            voxel.step_report << "[Step T2b][Diffusion sampling length ratio]=" << float(voxel.param[0]) << std::endl;

            voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";
            voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of "
                << float(voxel.param[0]) << ".";

            if(voxel.r2_weighted)
                voxel.recon_report << " The ODF calculation was weighted by the square of the diffusion displacement.";

            if(src_dwi_data.size() == 1)
            {
                if (!reconstruct2<
                        ReadDWIData,
                        HGQI_Recon,
                        SaveMetrics,
                        OutputODF>("GQI reconstruction"))
                    return false;
                break;
            }

            if (!reconstruct2<
                    ReadDWIData,
                    Dwi2Tensor,
                    BalanceScheme,
                    GQI_Recon,
                    RDI_Recon,
                    SaveMetrics,
                    OutputODF>("GQI"))
                return false;
            break;
        case 6:
            voxel.recon_report
                    << " The diffusion data were converted to HARDI using generalized q-sampling method with a regularization parameter of " << voxel.param[2] << ".";
            if (!reconstruct2<ReadDWIData,
                    SchemeConverter>("HARDI reconstruction"))
                return false;
            break;
        case 7:
            voxel.step_report << "[Step T2b]=QSDR" << std::endl;
            voxel.step_report << "[Step T2b][QSDR resolution]=" << voxel.qsdr_reso << std::endl;
            voxel.step_report << "[Step T2b][Template]=" << template_name_list[voxel.template_id] << std::endl;
            voxel.step_report << "[Step T2b][Diffusion sampling length ratio]=" << voxel.param[0] << std::endl;
            voxel.recon_report
            << " The diffusion data were reconstructed in the MNI space using q-space diffeomorphic reconstruction (Yeh et al., Neuroimage, 58(1):91-9, 2011) to obtain the spin distribution function (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010). "
            << " A diffusion sampling length ratio of "
            << float(voxel.param[0]) << " was used.";
            // run gqi to get the spin quantity


            // obtain QA map for normalization
            {
                // clear mask to create whole volume QA map
                if (!reconstruct2<
                        ReadDWIData,
                        BalanceScheme,
                        GQI_Recon,
                        RecordQA>("Preparing QA/ISO maps for normalization"))
                    return false;
                if (!reconstruct2<DWINormalization,
                        Dwi2Tensor,
                        BalanceScheme,
                        GQI_Recon,
                        RDI_Recon,
                        SaveMetrics,
                        OutputODF>("QSDR reconstruction"))
                    return false;
            }

            voxel.recon_report
            << " The output resolution in diffeomorphic reconstruction was " << voxel.qsdr_reso << " mm isotropic.";
            if(voxel.needs("rdi"))
                voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";

            break;
        default:
            error_msg = "unknown method";
            return false;
        }

        voxel.step_report << "[Step T2c][Output metrics]=" << voxel.other_output << std::endl;
        if(voxel.dti_ignore_high_b)
            voxel.recon_report << " The tensor metrics were calculated using DWI with b-value lower than 1750 s/mm².";
        return save_fib();
    }
    catch (std::exception& e)
    {
        error_msg = e.what();
        return false;
    }
    catch (...)
    {
        error_msg = "unknown exception";
        return false;
    }
}


bool output_odfs(const tipl::image<3,unsigned char>& mni_mask,
                 const std::filesystem::path& out_name,
                 const char* ext,
                 std::vector<std::vector<float> >& odfs,
                 std::vector<tipl::image<3> >& template_metrics,
                 std::vector<std::string>& template_metrics_name,
                 const tessellated_icosahedron& ti,
                 const float* vs,
                 const float* mni,
                 const std::string& report,
                 std::string& error_msg,
                 bool record_odf = true)
{
    tipl::progress prog_("generating template");
    src_data image_model;
    auto swap_data = [&](void)
    {
        image_model.voxel.template_odfs.swap(odfs);
        image_model.voxel.template_metrics.swap(template_metrics);
        image_model.voxel.template_metrics_name.swap(template_metrics_name);
    };

    if(report.length())
        image_model.voxel.report = report.c_str();
    image_model.voxel.dim = mni_mask.shape();
    image_model.voxel.ti = ti;
    image_model.file_name = out_name;
    image_model.voxel.mask = mni_mask;
    image_model.voxel.trans_to_mni = mni;
    image_model.voxel.vs = vs;
    image_model.voxel.R2 = 1.0f;
    image_model.voxel.qsdr = true;
    image_model.voxel.other_output = record_odf ? "odf" : "";
    swap_data();
    if (!image_model.reconstruct2<ODFLoader,
            SaveMetrics>("template"))
    {
        error_msg = image_model.error_msg;
        swap_data();
        return false;
    }
    image_model.output_file_name = std::filesystem::path(out_name)+=ext;
    image_model.save_fib();
    swap_data();
    return true;
}


bool odf_average(const std::filesystem::path& out_name,std::vector<std::filesystem::path>& file_names,std::string& error_msg)
{
    if(tipl::ends_with(file_names[0].u8string(),".nii.gz"))
    {
        tipl::image<3,double> sum;
        tipl::image<3> each;
        tipl::vector<3> vs;
        tipl::matrix<4,4> T;
        bool is_mni;
        tipl::progress p0("adding images",true);
        for(size_t i = 0;p0(i,file_names.size());++i)
        {
            if(!(tipl::io::gz_nifti(file_names[i],std::ios::in) >> vs >> T >> is_mni >> each >>
                [&](const std::string& e){tipl::error() << (error_msg = e);}))
                return false;
            if(sum.empty())
                sum.resize(each.shape());
            sum += each;
        }
        if(p0.aborted())
            return false;
        sum /= file_names.size();
        return tipl::io::gz_nifti(out_name,std::ios::out)
                    << vs << T << is_mni << sum
                    << [&](const std::string& e){tipl::error() << (error_msg = e);};
    }


    std::string report;
    tessellated_icosahedron ti;
    tipl::vector<3> vs;
    tipl::shape<3> dim;
    std::vector<std::vector<double> > odfs;
    tipl::image<3,unsigned int> odf_count;
    tipl::matrix<4,4> mni;
    std::filesystem::path file_name;

    std::vector<std::string> other_metrics_name;
    std::vector<tipl::image<3> > other_metrics_images;
    std::vector<size_t> other_metrics_count;
    try {
        tipl::progress p0("loading data");
        for (unsigned int index = 0;p0(index,file_names.size());++index)
        {
            file_name = file_names[index];
            fib_data fib;
            tipl::out() << "reading file";
            if(!fib.load_from_file(file_name))
                throw std::runtime_error(fib.error_msg);
            if(!fib.is_mni)
                throw std::runtime_error("not QSDR fib file");
            if(!fib.has_odfs())
                throw std::runtime_error("ODF data required for population template");
            if(index == 0)
            {
                report = fib.report;
                dim = fib.dim;
                vs = fib.vs;
                ti.vertices = fib.dir.odf_table;
                ti.faces = fib.dir.odf_faces;
                ti.vertices_count = uint16_t(ti.vertices.size());
                ti.half_vertices_count = ti.vertices_count/2;
                ti.fold = uint16_t(std::floor(std::sqrt((ti.vertices_count-2)/10.0)+0.5));
                mni = fib.trans_to_mni;
                other_metrics_name = fib.get_index_list();
                other_metrics_images.resize(other_metrics_name.size());
                other_metrics_count.resize(other_metrics_name.size());

                odfs.resize(dim.size());
                odf_count.resize(dim);
            }
            else
            // check odf consistency
            {
                if(ti.vertices.size() != fib.dir.odf_table.size())
                    throw std::runtime_error("inconsistent ODF dimension");
                if(dim != fib.dim)
                    throw std::runtime_error("inconsistent image dimension");
                for (unsigned int index = 0;index < ti.vertices.size();++index)
                {
                    if(std::fabs(ti.vertices[index][0]-fib.dir.odf_table[index][0]) > 0.0f ||
                       std::fabs(ti.vertices[index][1]-fib.dir.odf_table[index][1]) > 0.0f)
                    throw std::runtime_error("inconsistent ODF orientations");
                }
            }

            tipl::out() << "accumulating ODFs";

            odf_data odf;
            if(!odf.read(fib))
                throw std::runtime_error(odf.error_msg);
            tipl::par_for(dim.size(),[&](size_t i)
            {
                if(fib.dir.fa[0][i] == 0.0f)
                    return;
                const float* odf_data = odf.get_odf_data(i);
                if(odf_data == nullptr)
                    return;
                if(odfs[i].empty())
                    odfs[i] = std::vector<double>(odf_data,odf_data+ti.half_vertices_count);
                else
                    tipl::add(odfs[i].begin(),odfs[i].end(),odf_data);
                odf_count[i]++;
            });

            tipl::progress p2("accumulating other metrics");
            for(size_t i = 0;p2(i,other_metrics_name.size());++i)
            {
                auto metric_index = fib.get_name_index(other_metrics_name[i]);
                if(metric_index < fib.slices.size())
                {
                    auto I = fib.slices[metric_index]->get_image();
                    if(other_metrics_images[i].empty())
                        other_metrics_images[i] = I;
                    else
                        other_metrics_images[i] += I;
                    other_metrics_count[i]++;
                }
            }
        }
        if (p0.aborted())
            return false;
    } catch (const std::exception& e) {
        error_msg = e.what();
        error_msg += " at ";
        error_msg += file_name.u8string();
        return false;
    }
    {
        tipl::progress prog("averaging other metrics");
        size_t total = 0;
        tipl::par_for(other_metrics_name.size(),[&](unsigned int i)
        {
            if(other_metrics_count[i])
                other_metrics_images[i] *= 1.0f/other_metrics_count[i];
            prog(total++,other_metrics_name.size());
            other_metrics_name[i] = "avg_" + other_metrics_name[i];
        },other_metrics_name.size());
        if (prog.aborted())
            return false;
    }
    {
        tipl::progress prog("averaging odf");
        size_t p = 0;
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            prog(p++,dim.size());
            if(odf_count[pos] > 1)
                tipl::divide_constant(odfs[pos].begin(),odfs[pos].end(),float(odf_count[pos]));
        },tipl::max_thread_count);
        if (prog.aborted())
            return false;
    }
    // eliminate ODF if missing more than half of the population
    tipl::image<3,unsigned char> mask(dim);
    {
        tipl::progress prog("creating mask");
        size_t odf_size = 0;
        for(size_t i = 0;prog(i,mask.size());++i)
        {
            if(odf_count[i] > (file_names.size() >> 1) || !odfs[i].empty())
            {
                mask[i] = 1;
                odfs[odf_size].swap(odfs[i]);
                ++odf_size;
            }
        }
        if(prog.aborted())
            return false;
        odfs.resize(odf_size);
    }
    std::ostringstream out;
    out << "A group-average template was constructed from a total of " << file_names.size() << " scans." << report;
    report = out.str();


    tipl::progress prog("output odf");
    std::vector<std::vector<float> > odfs_float(odfs.size());
    tipl::par_for(odfs.size(),[&](size_t i)
    {
        odfs_float[i].resize(odfs[i].size());
        std::transform(odfs[i].begin(), odfs[i].end(), odfs_float[i].begin(),
                           [](double d) { return static_cast<float>(d); });

    });

    if(!output_odfs(mask,out_name,".mean.fz",odfs_float,other_metrics_images,other_metrics_name,ti,vs.begin(),mni.begin(),report,error_msg,false) ||
       !output_odfs(mask,out_name,".mean.odf.fz",odfs_float,other_metrics_images,other_metrics_name,ti,vs.begin(),mni.begin(),report,error_msg))
        return false;
    return true;
}
bool check_fib_dim_vs(tipl::io::gz_mat_read& mat_reader,tipl::shape<3>& dim,tipl::vector<3>& vs,tipl::matrix<4,4>& trans,bool& is_mni);
void calculate_shell(std::vector<float> sorted_bvalues,std::vector<unsigned int>& shell);
extern int src_ver;
// average the balanced dwi exported by qsdr (--other_output=dwi) into a template-space sz
bool dwi_average(const std::filesystem::path& out_name,std::vector<std::filesystem::path>& file_names,std::string& error_msg)
{
    if(std::filesystem::exists(out_name))
        return error_msg = out_name.u8string() + " already exists",false;
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4> trans;
    std::vector<float> b_table;
    std::vector<unsigned int> shell;
    std::string report,steps;
    std::vector<size_t> si2vi;      // template voxel of each stored value, identical in all files
    std::vector<std::vector<float> > sum;
    std::vector<uint32_t> count;
    std::filesystem::path file_name;
    try
    {
        tipl::progress p0("averaging dwi",true);
        for(size_t s = 0;p0(s,file_names.size());++s)
        {
            file_name = file_names[s];
            tipl::io::gz_mat_read in;
            in.delay_read = true;
            if(!in.load_from_file(file_name))
                throw std::runtime_error(in.error_msg);
            tipl::shape<3> d;
            tipl::vector<3> v;
            tipl::matrix<4,4> t;
            bool is_mni = false;
            std::string this_steps;
            unsigned int row,col;
            const float* table = nullptr;
            const unsigned char* mask = nullptr;
            if(!check_fib_dim_vs(in,d,v,t,is_mni))
                throw std::runtime_error(in.error_msg);
            if(!is_mni)
                throw std::runtime_error("not a QSDR fz");
            if(!in.read("steps",this_steps) || !tipl::contains(this_steps,"scheme_balance") ||
               !in.read("b_table",row,col,table) || row != 4 || table[0] != 0.0f)
                throw std::runtime_error("no balanced dwi: run qsdr with --other_output=dwi");
            unsigned int r,c;
            if(!in.read("mask",r,c,mask) || size_t(r)*size_t(c) != d.size())
                throw std::runtime_error("invalid mask");
            std::vector<float> bvalues(col);
            for(size_t i = 0;i < col;++i)
                bvalues[i] = table[i*4];
            std::vector<unsigned int> this_shell;
            calculate_shell(bvalues,this_shell);
            if(!s)
            {
                dim = d;
                vs = v;
                trans = t;
                b_table.assign(table,table+4*col);
                in.read("report",report);
                steps = this_steps;
                shell = this_shell;
                tipl::out() << "shell count: " << shell.size();
                for(size_t i = 0;i < shell.size();++i)
                    tipl::out() << "shell " << i+1 << ": " << (i+1 < shell.size() ? shell[i+1] : col)-shell[i] << " directions";
                si2vi = tipl::get_sparse_index(tipl::make_image(mask,dim));
                sum.resize(col,std::vector<float>(si2vi.size()));
                count.resize(si2vi.size());
            }
            else
            {
                if(d != dim || v != vs || t != trans || tipl::get_sparse_index(tipl::make_image(mask,dim)) != si2vi)
                    throw std::runtime_error("inconsistent image dimension, voxel size, transformation, or mask");
                if(4*col != b_table.size() || this_shell != shell)
                    throw std::runtime_error("inconsistent dwi count or shell structure");
            }
            std::vector<unsigned char> covered(si2vi.size()); // native fov coverage: b0 > 0
            for(size_t k = 0;k < col;++k)
            {
                std::string name = "image"+std::to_string(k);
                const float* I = nullptr;
                if(!in.read(name,r,c,I) || size_t(r)*size_t(c) != si2vi.size())
                    throw std::runtime_error("invalid " + name);
                if(!k)
                    for(size_t j = 0;j < si2vi.size();++j)
                        if(I[j] > 0.0f)
                        {
                            covered[j] = 1;
                            ++count[j];
                        }
                tipl::par_for(si2vi.size(),[&](size_t j)
                {
                    if(covered[j])
                        sum[k][j] += I[j];
                });
                in[name].data_buf = std::vector<unsigned char>();
                in[name].converted_data_buf = std::vector<unsigned char>();
            }
        }
        if(p0.aborted())
            return false;
    }
    catch(const std::exception& e)
    {
        error_msg = e.what();
        error_msg += " at ";
        error_msg += file_name.u8string();
        return false;
    }

    tipl::image<3,unsigned char> mask(dim);
    for(size_t j = 0;j < count.size();++j)
        if(count[j])
            mask[si2vi[j]] = 1;
    auto temp_file = std::filesystem::path(out_name)+=".tmp";
    tipl::progress prog("saving ",out_name.u8string());
    {
        tipl::io::gz_mat_write out(temp_file);
        if(!out)
            return error_msg = "cannot write to " + temp_file.u8string(),false;
        out.write("dimension",dim);
        out.write("voxel_size",vs);
        out.write("trans",trans);
        out.write("version",src_ver);
        out.write("b_table",b_table,4);
        out.apply_slope = true;
        out.apply_mask = true;
        out.mask_rows = dim.plane_size();
        out.mask_cols = dim.depth();
        out.si2vi = tipl::get_sparse_index(mask);
        out.write("mask",mask,dim.plane_size());
        for(size_t k = 0;prog(k,sum.size());++k)
        {
            tipl::image<3> I(dim);
            for(size_t j = 0;j < count.size();++j)
                if(count[j])
                    I[si2vi[j]] = sum[k][j]/float(count[j]);
            sum[k] = std::vector<float>();
            out.write<tipl::io::masked_sloped>("image"+std::to_string(k),I.data(),dim.plane_size(),dim.depth());
        }
        out.write("report","A group-average DWI was constructed from a total of " + std::to_string(file_names.size()) + " scans." + report);
        out.write("steps",steps);
    }
    if(prog.aborted())
        return std::filesystem::remove(temp_file),false;
    std::error_code error;
    std::filesystem::rename(temp_file,out_name,error);
    if(error)
        return error_msg = error.message(),std::filesystem::remove(temp_file),false;
    return true;
}









