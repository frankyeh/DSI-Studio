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

extern std::vector<std::string> fa_template_list;
void src_data::check_output_file_name(void)
{
    if(output_file_name.empty())
        output_file_name = file_name;
    if(tipl::ends_with(output_file_name,".fz") || tipl::ends_with(output_file_name,".fib.gz"))
        return;
    tipl::remove_suffix(output_file_name,".sz");
    tipl::remove_suffix(output_file_name,".src.gz");
    tipl::remove_suffix(output_file_name,".nii.gz");
    std::ostringstream out;
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





bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs);
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
    voxel.recon_report << " The parallel processing of histology image were done by tessellation whole slide image into smaller image block with overlapping margin to eliminate boundary effects (Yeh, et al. J Pathol Inform 2014,  5:1).";
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

        if(!has_bias_field_correction())
            correct_bias_field();

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
            voxel.step_report << "[Step T2b][Template]=" << QFileInfo(fa_template_list[voxel.template_id].c_str()).baseName().toLower().toStdString() << std::endl;
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
                 const char* out_name,
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
    image_model.output_file_name = std::string(out_name)+ext;
    image_model.save_fib();
    swap_data();
    return true;
}


bool odf_average(const char* out_name,std::vector<std::string>& file_names,std::string& error_msg)
{
    if(tipl::ends_with(file_names[0],".nii.gz"))
    {
        tipl::image<3,double> sum;
        tipl::image<3> each;
        tipl::vector<3> vs;
        tipl::matrix<4,4> T;
        bool is_mni;
        tipl::progress p0("adding images",true);
        for(size_t i = 0;p0(i,file_names.size());++i)
        {
            tipl::out() << "adding " << file_names[i];
            if(!tipl::io::gz_nifti::load_from_file(file_names[i].c_str(),each,vs,T,is_mni))
            {
                error_msg = "cannot load file" + file_names[i];
                return false;
            }
            if(sum.empty())
                sum.resize(each.shape());
            sum += each;
        }
        if(p0.aborted())
            return false;
        sum /= file_names.size();
        if(!tipl::io::gz_nifti::save_to_file<tipl::progress>(out_name,std::tie(sum,vs,T,is_mni)))
        {
            error_msg = std::string("cannot save file") + out_name;
            return false;
        }
        return true;
    }


    std::string report;
    tessellated_icosahedron ti;
    tipl::vector<3> vs;
    tipl::shape<3> dim;
    std::vector<std::vector<double> > odfs;
    tipl::image<3,unsigned int> odf_count;
    tipl::matrix<4,4> mni;
    std::string file_name;

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
            if(!fib.load_from_file(file_name.c_str()))
                throw std::runtime_error(fib.error_msg);
            if(!fib.is_mni)
                throw std::runtime_error("not QSDR fib file");
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
                if(fib.has_odfs())
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

            if(!odf_count.empty())
            {
                odf_data odf;
                if(!odf.read(fib))
                    throw std::runtime_error(odf.error_msg);
                tipl::adaptive_par_for(dim.size(),[&](size_t i)
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
            }
            else
            {
                if(!fib.dir.findex.empty())
                {
                    for(size_t pos = 0;pos < dim.size();++pos)
                        for(size_t f_index = 0;f_index < fib.dir.num_fiber;++f_index)
                            if(fib.dir.fa[f_index][pos] > 0.0f)
                            {
                                if(odfs[pos].empty())
                                    odfs[pos].resize(ti.half_vertices_count);
                                odfs[pos][fib.dir.findex[f_index][pos]] += fib.dir.fa[f_index][pos];
                            }
                }
                else
                if(!fib.dir.findex.empty())
                {
                    tipl::par_for(dim.size(),[&](size_t pos)
                    {
                        if(fib.dir.fa[0][pos] > 0.0f)
                        {
                            auto fdir = fib.dir.get_fib(pos,0);
                            size_t best_dir = 0;
                            float best_cos = 0.0f;
                            for(size_t i = 0;i < fib.dir.odf_table.size();++i)
                            {
                                auto cur_cos = std::fabs(fdir*fib.dir.odf_table[i]);
                                if(cur_cos > best_cos)
                                {
                                    best_cos = cur_cos;
                                    best_dir = i;
                                }
                            }
                            if(odfs[pos].empty())
                                odfs[pos].resize(ti.half_vertices_count);
                            odfs[pos][best_dir] += fib.dir.fa[0][pos];
                        }
                    },tipl::max_thread_count);

                }
                else
                    throw std::runtime_error("no fiber orientation information");
            }

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
                        tipl::add(other_metrics_images[i].begin(),other_metrics_images[i].end(),I.begin());
                    other_metrics_count[i]++;
                }
            }
        }
        if (p0.aborted())
            return false;
    } catch (const std::exception& e) {
        error_msg = e.what();
        error_msg += " at ";
        error_msg += file_name;
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
            if(odfs[pos].empty())
                return;
            if(!odf_count.empty())
            {
                if(odf_count[pos] > 1)
                    tipl::divide_constant(odfs[pos].begin(),odfs[pos].end(),float(odf_count[pos]));
            }
            else
                tipl::divide_constant(odfs[pos].begin(),odfs[pos].end(),float(file_names.size()));
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
            if((!odf_count.empty() && odf_count[i] > (file_names.size() >> 1)) || !odfs[i].empty())
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
    tipl::adaptive_par_for(odfs.size(),[&](size_t i)
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









