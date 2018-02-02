#include <boost/mpl/vector.hpp>
#include <boost/mpl/insert_range.hpp>
#include <boost/mpl/begin_end.hpp>
#include "tessellated_icosahedron.hpp"
#include "prog_interface_static_link.h"
#include "basic_voxel.hpp"
#include "image_model.hpp"
#include "odf_decomposition.hpp"
#include "odf_process.hpp"

#include "sample_model.hpp"
#include "space_mapping.hpp"

#include "dti_process.hpp"
#include "dsi_process.hpp"
#include "qbi_process.hpp"
#include "sh_process.hpp"
#include "gqi_process.hpp"
#include "gqi_mni_reconstruction.hpp"

#include "odf_deconvolusion.hpp"
#include "odf_decomposition.hpp"
#include "image_model.hpp"


extern std::string t1w_template_file_name;

typedef boost::mpl::vector<
    ReadDWIData,
    Dwi2Tensor
> dti_process;

typedef boost::mpl::vector<
    ReadDWIData,
    HGQI_Recon,
    DetermineFiberDirections,
    ScaleZ0ToMinODF,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> hgqi_process;


template<class reco_type>
struct odf_reco_type{
    typedef boost::mpl::vector<
        ODFDeconvolusion,
        ODFDecomposition,
        DetermineFiberDirections,
        ScaleZ0ToMinODF,
        SaveMetrics,
        SaveDirIndex,
        OutputODF
    > common_odf_process;
    typedef typename boost::mpl::insert_range<common_odf_process,boost::mpl::begin<common_odf_process>::type,reco_type>::type type0;
    typedef typename boost::mpl::push_front<type0,ReadDWIData>::type type; // add ReadDWIData to the front
};

template<class reco_type>
struct estimation_type{
    typedef boost::mpl::vector<
        DetermineFiberDirections,
        EstimateResponseFunction
    > common_estimation_process;

    typedef typename boost::mpl::insert_range<common_estimation_process,boost::mpl::begin<common_estimation_process>::type,reco_type>::type type0;
    typedef typename boost::mpl::push_front<type0,ReadDWIData>::type type; // add ReadDWIData to the front
};


typedef odf_reco_type<boost::mpl::vector<
    QSpace2Pdf,
    Pdf2Odf
> >::type dsi_process;

const unsigned int equator_sample_count = 40;
typedef odf_reco_type<boost::mpl::vector<
    QBIReconstruction<equator_sample_count>
> >::type qbi_process;

typedef odf_reco_type<boost::mpl::vector<
    SHDecomposition
> >::type qbi_sh_process;

typedef boost::mpl::vector<
    DWINormalization,
    BalanceScheme,
    GQI_Recon,
    RDI_Recon,
    ODFDeconvolusion,
    ODFDecomposition,
    EstimateZ0_MNI,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> qsdr_process;

typedef odf_reco_type<boost::mpl::vector<
    BalanceScheme,
    GQI_Recon,
    RDI_Recon
> >::type gqi_process;

typedef boost::mpl::vector<
    ReadDWIData,
    BalanceScheme,
    GQI_Recon,
    DetermineFiberDirections,
    ScaleZ0ToMinODF,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> ddi_process;



typedef boost::mpl::vector<
    ReadDWIData,
    QSpaceSpectral
> gqi_spectral_process;

typedef boost::mpl::vector<
    ReadDWIData,
    SchemeConverter
> hardi_convert_process;

// for ODF deconvolution
typedef estimation_type<boost::mpl::vector<
    QSpace2Pdf,
    Pdf2Odf
> >::type dsi_estimate_response_function;

// for ODF deconvolution
typedef estimation_type<boost::mpl::vector<
    QBIReconstruction<equator_sample_count>
> >::type qbi_estimate_response_function;

// for ODF deconvolution
typedef estimation_type<boost::mpl::vector<
    SHDecomposition
> >::type qbi_sh_estimate_response_function;


// for ODF deconvolution
typedef boost::mpl::vector<
    ReadDWIData,
    BalanceScheme,
    GQI_Recon,
    DetermineFiberDirections,
    RecordQA,
    EstimateResponseFunction

> gqi_estimate_response_function;



typedef boost::mpl::vector<
    ODFLoader,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex
> reprocess_odf;


const char* reconstruction(ImageModel* image_model,
                           unsigned int method_id,
                           const float* param_values,
                           bool check_b_table)
{
    static std::string output_name;
    try
    {
        if(!image_model->is_human_data())
            image_model->voxel.csf_calibration = false;
        image_model->voxel.recon_report.clear();
        image_model->voxel.recon_report.str("");
        image_model->voxel.param = param_values;
        std::ostringstream out;
        if(method_id != 4 && method_id != 7)
            image_model->voxel.output_rdi = 0;
        if(method_id == 1) // DTI
        {
            image_model->voxel.need_odf = 0;
            image_model->voxel.output_jacobian = 0;
            image_model->voxel.output_mapping = 0;
            image_model->voxel.scheme_balance = 0;
            image_model->voxel.half_sphere = 0;
            image_model->voxel.odf_deconvolusion = 0;
            image_model->voxel.odf_decomposition = 0;
        }
        else
        {
            if(method_id == 8) // DDI
            {
                image_model->voxel.odf_deconvolusion = 0;
                image_model->voxel.odf_decomposition = 0;
            }
            out << ".odf" << image_model->voxel.ti.fold;// odf_order
            out << ".f" << image_model->voxel.max_fiber_number;
            if (image_model->voxel.need_odf)
                out << "rec";
            if (image_model->voxel.scheme_balance)
                out << ".bal";
            if (image_model->voxel.half_sphere)
                out << ".hs";
            if (image_model->voxel.csf_calibration &&(method_id == 4 || method_id == 7)) // GQI or QSDR
                out << ".csfc";
            else
                image_model->voxel.csf_calibration = false;
            if (image_model->voxel.odf_deconvolusion)
            {
                out << ".de" << param_values[2];
                if(image_model->voxel.odf_xyz[0] != 0 ||
                   image_model->voxel.odf_xyz[1] != 0 ||
                   image_model->voxel.odf_xyz[2] != 0)
                    out << ".at_" << image_model->voxel.odf_xyz[0]
                        << "_" << image_model->voxel.odf_xyz[1]
                        << "_" << image_model->voxel.odf_xyz[2];
            }
            if (image_model->voxel.odf_decomposition)
            {
                out << ".dec" << param_values[3] << "m" << (int)param_values[4];
                if(image_model->voxel.odf_xyz[0] != 0 ||
                   image_model->voxel.odf_xyz[1] != 0 ||
                   image_model->voxel.odf_xyz[2] != 0)
                    out << ".at_" << image_model->voxel.odf_xyz[0]
                        << "_" << image_model->voxel.odf_xyz[1]
                        << "_" << image_model->voxel.odf_xyz[2];
            }
        }

        // Copy SRC b-table to voxel b-table and sort it
        image_model->voxel.load_from_src(*image_model);

        // correct for b-table orientation
        if(check_b_table)
            out << image_model->check_b_table();

        switch (method_id)
        {
        case 0: //DSI local max
            image_model->voxel.recon_report <<
            " The diffusion data were reconstructed using diffusion spectrum imaging (Wedeen et al. MRM, 2005) with a Hanning filter of " << (int)param_values[0] << ".";
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<dsi_estimate_response_function>())
                    return "reconstruction canceled";
            }
            out << ".dsi."<< (int)param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<dsi_process>())
                return "reconstruction canceled";
            break;
        case 1://DTI
            image_model->voxel.recon_report << " The diffusion tensor was calculated.";
            out << ".dti.fib.gz";
            image_model->voxel.max_fiber_number = 1;
            if (!image_model->reconstruct<dti_process>())
                return "reconstruction canceled";
            break;

        case 2://QBI
            image_model->voxel.recon_report << " The diffusion data was reconstructed using q-ball imaging (Tuch, MRM 2004).";
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_estimate_response_function>())
                    return "reconstruction canceled";
            }
            out << ".qbi."<< param_values[0] << "_" << param_values[1] << ".fib.gz";
            if (!image_model->reconstruct<qbi_process>())
                return "reconstruction canceled";
            break;
        case 3://QBI
            image_model->voxel.recon_report << " The diffusion data was reconstructed using spherical-harmonic-based q-ball imaging (Descoteaux et al., MRM 2007).";
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_sh_estimate_response_function>())
                    return "reconstruction canceled";
            }
            out << ".qbi.sh"<< (int) param_values[1] << "." << param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<qbi_sh_process>())
                return "reconstruction canceled";
            break;

        case 4://GQI
            if(param_values[0] == 0.0) // spectral analysis
            {
                image_model->voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010).";
                out << (image_model->voxel.r2_weighted ? ".gqi2.spec.fib.gz":".gqi.spec.fib.gz");
                if (!image_model->reconstruct<gqi_spectral_process>())
                    return "reconstruction canceled";
                break;
            }
            image_model->voxel.recon_report <<
            " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of " << (float)param_values[0] << ".";
            if(image_model->voxel.output_rdi)
                image_model->voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";

            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<gqi_estimate_response_function>())
                    return "reconstruction canceled";
            }
            if(image_model->voxel.r2_weighted)
                image_model->voxel.recon_report << " The ODF calculation was weighted by the square of the diffuion displacement.";
            if (image_model->voxel.output_rdi)
                out << ".rdi";
            out << (image_model->voxel.r2_weighted ? ".gqi2.":".gqi.") << param_values[0] << ".fib.gz";

            if(image_model->src_dwi_data.size() == 1)
            {
                if (!image_model->reconstruct<hgqi_process>())
                    return "reconstruction canceled";
                break;
            }

            if (!image_model->reconstruct<gqi_process>())
                return "reconstruction canceled";
            break;
        case 6:
            image_model->voxel.recon_report
                    << " The diffusion data were converted to HARDI using generalized q-sampling method with a regularization parameter of " << param_values[2] << ".";
            out << ".hardi."<< param_values[0]
                << ".b" << param_values[1]
                << ".reg" << param_values[2] << ".src.gz";
            if (!image_model->reconstruct<hardi_convert_process>())
                return "reconstruction canceled";
            break;
        case 7:
            if(image_model->voxel.reg_method == 4) // DMDM
            {
                {
                    gz_nifti in;
                    if(!in.load_from_file(t1w_template_file_name.c_str()) || !in.toLPS(image_model->voxel.t1wt))
                        return "Cannot load T1W template";
                    in.get_voxel_size(image_model->voxel.t1wt_vs.begin());
                    in.get_image_transformation(image_model->voxel.t1wt_tran);
                }
                {
                    gz_nifti in;
                    if(!in.load_from_file(image_model->voxel.t1w_file_name.c_str()) || !in.toLPS(image_model->voxel.t1w))
                        return "Cannot load T1W for DMDM normaliztion";
                    in.get_voxel_size(image_model->voxel.t1w_vs.begin());
                }
            }


            image_model->voxel.recon_report
            << " The diffusion data were reconstructed in the MNI space using q-space diffeomorphic reconstruction (Yeh et al., Neuroimage, 58(1):91-9, 2011) to obtain the spin distribution function (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010). "
            << " A diffusion sampling length ratio of "
            << (float)param_values[0] << " was used, and the output resolution was " << param_values[1] << " mm.";
            // run gqi to get the spin quantity

            if(image_model->voxel.output_rdi)
                image_model->voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";

            {
                std::vector<image::pointer_image<float,3> > tmp;
                tmp.swap(image_model->voxel.grad_dev);
                // clear mask to create whole volume QA map
                std::fill(image_model->voxel.mask.begin(),image_model->voxel.mask.end(),1.0);
                if (!image_model->reconstruct<gqi_estimate_response_function>())
                    return "reconstruction canceled";
                tmp.swap(image_model->voxel.grad_dev);
            }

            if(image_model->voxel.reg_method < 4) // 0,1,2 SPM norm, 3 CDM, 4 CDM-T1W
            {
                if(image_model->voxel.reg_method == 3)
                    out << ".cdm";
                else
                    out << ".reg" << (int)image_model->voxel.reg_method;
            }
            else
                out << ".cdmt1w";


            out << (image_model->voxel.r2_weighted ? ".qsdr2.":".qsdr.");
            out << param_values[0] << "." << param_values[1] << "mm";
            if(image_model->voxel.output_jacobian)
                out << ".jac";
            if(image_model->voxel.output_mapping)
                out << ".map";
            if (!image_model->reconstruct<qsdr_process>())
                return "reconstruction canceled";
            out << ".R" << (int)std::floor(image_model->voxel.R2*100.0) << ".fib.gz";
            break;
        case 8:

            image_model->voxel.recon_report <<
            " The diffusion data were compared with baseline scan using diffusion difference imaging with a diffusion sampling length ratio of " << (float)param_values[0] << ".";

            if(image_model->voxel.r2_weighted)
                image_model->voxel.recon_report << " The ODF calculation was weighted by the square of the diffuion displacement.";

            out << (image_model->voxel.r2_weighted ? ".ddi2.":".ddi.") << param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<ddi_process>())
                return "reconstruction canceled";
            break;
        }
        image_model->save_fib(out.str());
        output_name = image_model->file_name + out.str();
    }
    catch (std::exception& e)
    {
        output_name = e.what();
        return output_name.c_str();
    }
    catch (...)
    {
        return "unknown exception";
    }
    return output_name.c_str();
}


bool output_odfs(const image::basic_image<unsigned char,3>& mni_mask,
                 const char* out_name,
                 const char* ext,
                 std::vector<std::vector<float> >& odfs,
                 const tessellated_icosahedron& ti,
                 const float* vs,
                 const float* mni,
                 const std::string& report,
                 bool record_odf = true)
{
    begin_prog("output");
    ImageModel image_model;
    if(report.length())
        image_model.voxel.report = report.c_str();
    image_model.voxel.dim = mni_mask.geometry();
    image_model.voxel.ti = ti;
    image_model.voxel.max_fiber_number = 5;
    image_model.voxel.need_odf = record_odf;
    image_model.voxel.template_odfs.swap(odfs);
    image_model.voxel.param = mni;
    image_model.file_name = out_name;
    image_model.voxel.mask = mni_mask;
    std::copy(vs,vs+3,image_model.voxel.vs.begin());
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>())
        return false;
    image_model.save_fib(ext);
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


const char* odf_average(const char* out_name,std::vector<std::string>& file_names)
{
    static std::string error_msg,report;
    tessellated_icosahedron ti;
    float vs[3];
    image::basic_image<unsigned char,3> mask;
    std::vector<std::vector<float> > odfs;
    unsigned int half_vertex_count = 0;
    unsigned int row,col;
    float mni[16]={0};
    begin_prog("averaging");
    for (unsigned int index = 0;check_prog(index,file_names.size());++index)
    {
        const char* file_name = file_names[index].c_str();
        gz_mat_read reader;
        set_title(file_names[index].c_str());
        if(!reader.load_from_file(file_name))
        {
            error_msg = "Cannot open file ";
            error_msg += file_name;
            check_prog(0,0);
            return error_msg.c_str();
        }
        if(index == 0)
        {
            {
                const char* report_buf = 0;
                if(reader.read("report",row,col,report_buf))
                    report = std::string(report_buf,report_buf+row*col);
            }
            const float* odf_buffer;
            const short* face_buffer;
            const unsigned short* dimension;
            const float* vs_ptr;
            const float* fa0;
            const float* mni_ptr;
            unsigned int face_num,odf_num;
            error_msg = "";
            if(!reader.read("dimension",row,col,dimension))
                error_msg = "dimension";
            if(!reader.read("fa0",row,col,fa0))
                error_msg = "fa0";
            if(!reader.read("voxel_size",row,col,vs_ptr))
                error_msg = "voxel_size";
            if(!reader.read("odf_faces",row,face_num,face_buffer))
                error_msg = "odf_faces";
            if(!reader.read("odf_vertices",row,odf_num,odf_buffer))
                error_msg = "odf_vertices";
            if(!reader.read("trans",row,col,mni_ptr))
                error_msg = "trans";
            if(error_msg.length())
            {
                error_msg += " missing in ";
                error_msg += file_name;
                check_prog(0,0);
                return error_msg.c_str();
            }
            mask.resize(image::geometry<3>(dimension));
            for(unsigned int index = 0;index < mask.size();++index)
                if(fa0[index] != 0.0)
                    mask[index] = 1;
            std::copy(vs_ptr,vs_ptr+3,vs);
            ti.init(odf_num,odf_buffer,face_num,face_buffer);
            half_vertex_count = odf_num >> 1;
            std::copy(mni_ptr,mni_ptr+16,mni);
        }
        else
        // check odf consistency
        {
            const float* odf_buffer;
            const unsigned short* dimension;
            unsigned int odf_num;
            error_msg = "";
            if(!reader.read("dimension",row,col,dimension))
                error_msg = "dimension";
            if(!reader.read("odf_vertices",row,odf_num,odf_buffer))
                error_msg = "odf_vertices";
            if(error_msg.length())
            {
                error_msg += " missing in ";
                error_msg += file_name;
                check_prog(0,0);
                return error_msg.c_str();
            }

            if(odf_num != ti.vertices_count || dimension[0] != mask.width() ||
                    dimension[1] != mask.height() || dimension[2] != mask.depth())
            {
                error_msg = "Inconsistent dimension in ";
                error_msg += file_name;
                check_prog(0,0);
                return error_msg.c_str();
            }
            for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
            {
                if(ti.vertices[index][0] != odf_buffer[0] ||
                   ti.vertices[index][1] != odf_buffer[1] ||
                   ti.vertices[index][2] != odf_buffer[2])
                {
                    error_msg = "Inconsistent ODF orientations in ";
                    error_msg += file_name;
                    return error_msg.c_str();
                }
            }
        }

        {
            const float* fa0;
            if(!reader.read("fa0",row,col,fa0))
            {
                error_msg = "Cannot find image information in ";
                error_msg += file_name;
                check_prog(0,0);
                return error_msg.c_str();
            }
            for(unsigned int index = 0;index < mask.size();++index)
                if(fa0[index] != 0.0)
                    mask[index] = 1;
        }

        std::vector<const float*> odf_bufs;
        std::vector<unsigned int> odf_bufs_size;
        //get_odf_bufs(reader,odf_bufs,odf_bufs_size);
        {
            odf_bufs.clear();
            odf_bufs_size.clear();
            for (unsigned int odf_index = 0;1;++odf_index)
            {
                std::ostringstream out;
                out << "odf" << odf_index;
                const float* odf_buf = 0;
                if (!reader.read(out.str().c_str(),row,col,odf_buf))
                    break;
                odf_bufs.push_back(odf_buf);
                odf_bufs_size.push_back(row*col);
            }
        }
        if(odf_bufs.empty())
        {
            error_msg += "No ODF data found in ";
            error_msg += file_name;
            check_prog(0,0);
            return error_msg.c_str();
        }
        if(index == 0)
        {
            odfs.resize(odf_bufs.size());
            for(unsigned int i = 0;i < odf_bufs.size();++i)
                odfs[i].resize(odf_bufs_size[i]);
        }
        else
        {
            bool inconsistence = false;
            if(odfs.size() != odf_bufs.size())
                inconsistence = true;
            for(unsigned int i = 0;i < odf_bufs.size();++i)
                if(odfs[i].size() != odf_bufs_size[i])
                    inconsistence = true;
            if(inconsistence)
            {
                error_msg = "Inconsistent mask coverage in ";
                error_msg += file_name;
                check_prog(0,0);
                return error_msg.c_str();
            }
        }
        for(unsigned int i = 0;i < odf_bufs.size();++i)
            image::add(odfs[i].begin(),odfs[i].end(),odf_bufs[i]);
    }
    if (prog_aborted())
        return 0;

    set_title("averaging odfs");
    for (unsigned int odf_index = 0;odf_index < odfs.size();++odf_index)
        for (unsigned int j = 0;j < odfs[odf_index].size();++j)
            odfs[odf_index][j] /= (double)file_names.size();

    std::ostringstream out;
    out << "A group average template was constructed from a total of " << file_names.size() << " subjects." << report.c_str();
    report = out.str();
    set_title("output files");
    output_odfs(mask,out_name,".mean.odf.fib.gz",odfs,ti,vs,mni,report);
    output_odfs(mask,out_name,".mean.fib.gz",odfs,ti,vs,mni,report,false);
    return 0;
}









