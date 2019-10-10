#include <boost/mpl/vector.hpp>
#include <boost/mpl/insert_range.hpp>
#include <boost/mpl/begin_end.hpp>
#include "tessellated_icosahedron.hpp"
#include "prog_interface_static_link.h"
#include "basic_voxel.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"
#include "gqi_process.hpp"
#include "gqi_mni_reconstruction.hpp"
#include "image_model.hpp"


typedef boost::mpl::vector<
    ReadDWIData,
    Dwi2Tensor
> dti_process;

typedef boost::mpl::vector<
    ReadDWIData,
    HGQI_Recon,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> hgqi_process;

typedef boost::mpl::vector<
    DWINormalization,
    Dwi2Tensor,
    BalanceScheme,
    GQI_Recon,
    RDI_Recon,
    EstimateZ0_MNI,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> qsdr_process;

typedef boost::mpl::vector<
    ReadDWIData,
    Dwi2Tensor,
    BalanceScheme,
    GQI_Recon,
    RDI_Recon,
    dGQI_Recon,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex,
    OutputODF
> gqi_process;

typedef boost::mpl::vector<
    ReadDWIData,
    QSpaceSpectral
> gqi_spectral_process;

typedef boost::mpl::vector<
    ReadDWIData,
    SchemeConverter
> hardi_convert_process;


typedef boost::mpl::vector<
    ReadDWIData,
    BalanceScheme,
    GQI_Recon,
    DetermineFiberDirections,
    RecordQA
> qa_map;



typedef boost::mpl::vector<
    ODFLoader,
    DetermineFiberDirections,
    SaveMetrics,
    SaveDirIndex
> reprocess_odf;


const char* ImageModel::reconstruction(void)
{
    static std::string output_name;
    try
    {
        if(!is_human_data())
            voxel.csf_calibration = false;
        voxel.recon_report.clear();
        voxel.recon_report.str("");
        voxel.step_report.clear();
        voxel.step_report.str("");
        std::ostringstream out;
        if(voxel.method_id != 4 && voxel.method_id != 7)
            voxel.output_rdi = 0;
        if(voxel.method_id == 1) // DTI
        {
            voxel.output_odf = 0;
            voxel.scheme_balance = 0;
            voxel.half_sphere = 0;
            voxel.odf_resolving = 0;
        }
        else
        {
            if(!voxel.study_src_file_path.empty()) // DDI
                voxel.csf_calibration = false;
            out << ".odf" << voxel.ti.fold;// odf_order
            if (voxel.odf_resolving)
                out << "r";
            out << ".f" << voxel.max_fiber_number;
            if (voxel.output_odf)
            {
                voxel.step_report << "[Step T2b(2)][ODFs]=checked" << std::endl;
                out << "rec";
            }
            if (voxel.scheme_balance)
                out << ".bal";
            if (voxel.half_sphere)
                out << ".hs";
            if (voxel.csf_calibration && voxel.method_id == 4) // GQI
                out << ".csfc";
            else
                voxel.csf_calibration = false;
        }



        // correct for b-table orientation
        if(voxel.check_btable)
        {
            voxel.recon_report <<
            " The b-table was checked by an automatic quality control routine to ensure its accuracy (Schilling et al. MRI, 2019) .";
            out << check_b_table();
        }
        else
            voxel.step_report << "[Step T2b][Check b-table]=unchecked" << std::endl;

        switch (voxel.method_id)
        {
        case 1://DTI
            voxel.step_report << "[Step T2b(1)]=DTI" << std::endl;
            voxel.recon_report << " The diffusion tensor was calculated.";
            out << ".dti.fib.gz";
            voxel.max_fiber_number = 1;
            if (!reconstruct<dti_process>("DTI"))
                return "reconstruction canceled";
            break;

        case 4://GQI
            voxel.step_report << "[Step T2b(1)]=GQI" << std::endl;
            voxel.step_report << "[Step T2b(1)][Diffusion sampling length ratio]=" << (float)voxel.param[0] << std::endl;
            if(voxel.param[0] == 0.0) // spectral analysis
            {
                voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010).";
                out << (voxel.r2_weighted ? ".gqi2.spec.fib.gz":".gqi.spec.fib.gz");
                if (!reconstruct<gqi_spectral_process>("Spectral GQI"))
                    return "reconstruction canceled";
                break;
            }

            if(voxel.output_rdi)
                voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";


            if(!voxel.study_src_file_path.empty())
            {
                rotate_to_mni();
                if(!compare_src(voxel.study_src_file_path.c_str()))
                    return "Failed to load DDI study SRC file.";
                voxel.step_report << "[Step T2b(1)][Compare SRC]=" << QFileInfo(voxel.study_src_file_path.c_str()).baseName().toStdString() << std::endl;
                voxel.recon_report <<
                " The diffusion data were compared with baseline scan using differential tractography with a diffusion sampling length ratio of "
                << (float)voxel.param[0] << " to study neuronal change.";

                out << (voxel.dt_deform ? ".ddf." : ".df.") << voxel.study_name << ".R" << (int)(voxel.R2*100.0f);
            }
            else
                voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of " << (float)voxel.param[0] << ".";

            if(voxel.r2_weighted)
                voxel.recon_report << " The ODF calculation was weighted by the square of the diffuion displacement.";
            if (voxel.output_rdi)
                out << ".rdi";
            out << (voxel.r2_weighted ? ".gqi2.":".gqi.") << voxel.param[0] << ".fib.gz";

            if(src_dwi_data.size() == 1)
            {
                if (!reconstruct<hgqi_process>("Reconstruction"))
                    return "reconstruction canceled";
                break;
            }

            if (!reconstruct<gqi_process>("GQI"))
                return "reconstruction canceled";
            break;
        case 6:
            voxel.recon_report
                    << " The diffusion data were converted to HARDI using generalized q-sampling method with a regularization parameter of " << voxel.param[2] << ".";
            out << ".hardi."<< voxel.param[0]
                << ".b" << voxel.param[1]
                << ".reg" << voxel.param[2] << ".src.gz";
            if (!reconstruct<hardi_convert_process>("HARDI reconstruction"))
                return "reconstruction canceled";
            break;
        case 7:
            voxel.step_report << "[Step T2b(1)]=QSDR" << std::endl;
            voxel.step_report << "[Step T2b(1)][Diffusion sampling length ratio]=" << (float)voxel.param[0] << std::endl;
            voxel.recon_report
            << " The diffusion data were reconstructed in the MNI space using q-space diffeomorphic reconstruction (Yeh et al., Neuroimage, 58(1):91-9, 2011) to obtain the spin distribution function (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010). "
            << " A diffusion sampling length ratio of "
            << (float)voxel.param[0] << " was used";
            // run gqi to get the spin quantity




            out << "." << QFileInfo(voxel.primary_template.c_str()).baseName().toLower().toStdString();
            out << (voxel.r2_weighted ? ".qsdr2.":".qsdr.");
            out << voxel.param[0];
            // obtain QA map for normalization
            {
                std::vector<tipl::pointer_image<float,3> > tmp;
                tmp.swap(voxel.grad_dev);
                auto mask = voxel.mask;
                // clear mask to create whole volume QA map
                std::fill(voxel.mask.begin(),voxel.mask.end(),1.0);
                if (!reconstruct<qa_map>("GQI for QSDR"))
                    return "reconstruction canceled";
                tmp.swap(voxel.grad_dev);
                if (!reconstruct<qsdr_process>("QSDR"))
                    return "reconstruction canceled";
                voxel.mask = mask;
            }

            voxel.recon_report
            << " The output resolution of is " << voxel.vs[0] << " mm isotorpic.";
            if(voxel.output_rdi)
                voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";

            out << ".R" << (int)std::floor(voxel.R2*100.0) << ".fib.gz";
            break;
        default:
            return "Unknown method";
        }
        file_ext = out.str();
        save_fib();
        output_name = file_name + out.str();
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


bool output_odfs(const tipl::image<unsigned char,3>& mni_mask,
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
    image_model.voxel.odf_resolving = true;
    image_model.voxel.output_odf = record_odf;
    image_model.voxel.template_odfs.swap(odfs);
    image_model.file_name = out_name;
    image_model.file_ext = ext;
    image_model.voxel.mask = mni_mask;
    std::copy(mni,mni+16,image_model.voxel.trans_to_mni);
    std::copy(vs,vs+3,image_model.voxel.vs.begin());
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>("Template reconstruction"))
        return false;
    image_model.save_fib();
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


const char* odf_average(const char* out_name,std::vector<std::string>& file_names)
{
    static std::string error_msg,report;
    tessellated_icosahedron ti;
    float vs[3];
    tipl::image<unsigned char,3> mask;
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
            mask.resize(tipl::geometry<3>(dimension));
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
            tipl::add(odfs[i].begin(),odfs[i].end(),odf_bufs[i]);
    }
    if (prog_aborted())
        return 0;

    set_title("Averaging ODFs");
    for (unsigned int odf_index = 0;odf_index < odfs.size();++odf_index)
        for (unsigned int j = 0;j < odfs[odf_index].size();++j)
            odfs[odf_index][j] /= (double)file_names.size();

    std::ostringstream out;
    out << "A group average template was constructed from a total of " << file_names.size() << " subjects." << report.c_str();
    report = out.str();
    set_title("Output Files");
    output_odfs(mask,out_name,".mean.odf.fib.gz",odfs,ti,vs,mni,report);
    output_odfs(mask,out_name,".mean.fib.gz",odfs,ti,vs,mni,report,false);
    return 0;
}









