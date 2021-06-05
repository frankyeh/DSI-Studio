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
#include "fib_data.hpp"

typedef boost::mpl::vector<
    ReadDWIData,
    CorrectGradientNonlinearity,
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
    CorrectGradientNonlinearity,
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
    CorrectGradientNonlinearity,
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

std::string ImageModel::get_file_ext(void)
{
    std::ostringstream out;
    if(voxel.method_id != 1) // DTI
    {
        if (voxel.output_odf)
            out << ".odf";
    }
    switch (voxel.method_id)
    {
    case 1://DTI
        out << ".dti.fib.gz";
        break;
    case 4://GQI
        if(voxel.param[0] == 0.0f) // spectral analysis
        {
            out << (voxel.r2_weighted ? ".gqi2.spec.fib.gz":".gqi.spec.fib.gz");
            break;
        }
        if(!voxel.study_src_file_path.empty())
            out << ".df." << voxel.study_name << ".R" << int(voxel.R2*100.0f);

        out << (voxel.r2_weighted ? ".gqi2.":".gqi.") << voxel.param[0] << ".fib.gz";
        break;
    case 6:
        out << ".hardi."<< voxel.param[0]
            << ".b" << voxel.param[1]
            << ".reg" << voxel.param[2] << ".src.gz";
        break;
    case 7:
        out << "." << QFileInfo(voxel.primary_template.c_str()).baseName().toLower().toStdString();
        out << (voxel.r2_weighted ? ".qsdr2.":".qsdr.");
        out << voxel.param[0];
        out << ".R" << int(std::floor(voxel.R2*100.0f)) << ".fib.gz";
        break;
    }
    return out.str();
}
bool is_human_size(tipl::geometry<3> dim,tipl::vector<3> vs);
bool ImageModel::reconstruction(void)
{
    try
    {
        voxel.recon_report.clear();
        voxel.recon_report.str("");
        voxel.step_report.clear();
        voxel.step_report.str("");
        if(voxel.method_id != 4 && voxel.method_id != 7)
            voxel.output_rdi = 0;
        if(voxel.method_id == 1) // DTI
        {
            voxel.output_odf = 0;
            voxel.scheme_balance = 0;
            voxel.half_sphere = 0;
            voxel.odf_resolving = 0;
            voxel.max_fiber_number = 1;
        }
        else
        {
            voxel.max_fiber_number = 5;
            if (voxel.output_odf)
                voxel.step_report << "[Step T2b(2)][ODFs]=1" << std::endl;
        }

        // correct for b-table orientation
        if(voxel.check_btable)
        {
            voxel.recon_report <<
            " The b-table was checked by an automatic quality control routine to ensure its accuracy (Schilling et al. MRI, 2019).";
            std::string result = check_b_table();
            if(!result.empty())
                voxel.recon_report << " The b-table was flipped by " << result << ".";
        }
        else
            voxel.step_report << "[Step T2b][Check b-table]=unchecked" << std::endl;

        switch (voxel.method_id)
        {
        case 1://DTI
            voxel.step_report << "[Step T2b(1)]=DTI" << std::endl;
            voxel.recon_report << " The diffusion tensor was calculated.";
            if (!reconstruct<dti_process>("DTI reconstruction"))
                return false;
            break;
        case 4://GQI
            voxel.step_report << "[Step T2b(1)]=GQI" << std::endl;
            voxel.step_report << "[Step T2b(1)][Diffusion sampling length ratio]=" << float(voxel.param[0]) << std::endl;
            if(voxel.param[0] == 0.0f) // spectral analysis
            {
                voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010).";
                if (!reconstruct<gqi_spectral_process>("spectral GQI reconstruction"))
                    return false;
                break;
            }

            if(voxel.output_rdi)
                voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";


            if(!voxel.study_src_file_path.empty())
            {
                std::cout << "SRC compared with=" << voxel.study_src_file_path << std::endl;
                if(is_human_size(voxel.dim,voxel.vs))
                    rotate_to_mni(2.0f);
                if(!compare_src(voxel.study_src_file_path.c_str()))
                {
                    error_msg = "Failed to load SRC file.";
                    return false;
                }
                voxel.recon_report <<
                " The diffusion data were compared with baseline scan using differential tractography with a diffusion sampling length ratio of "
                << float(voxel.param[0]) << " to study neuronal change.";
                if (!study_src->reconstruct<dti_process>("GQI reconstruction"))
                    return false;
            }
            else
                voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of "
                << float(voxel.param[0]) << ".";

            if(voxel.r2_weighted)
                voxel.recon_report << " The ODF calculation was weighted by the square of the diffuion displacement.";

            if(src_dwi_data.size() == 1)
            {
                if (!reconstruct<hgqi_process>("Reconstruction"))
                    return false;
                break;
            }

            if (!reconstruct<gqi_process>("GQI reconstruction"))
                return false;
            break;
        case 6:
            voxel.recon_report
                    << " The diffusion data were converted to HARDI using generalized q-sampling method with a regularization parameter of " << voxel.param[2] << ".";
            if (!reconstruct<hardi_convert_process>("HARDI reconstruction"))
                return false;
            break;
        case 7:
            voxel.step_report << "[Step T2b(1)]=QSDR" << std::endl;
            voxel.step_report << "[Step T2b(1)][Diffusion sampling length ratio]=" << (float)voxel.param[0] << std::endl;
            voxel.recon_report
            << " The diffusion data were reconstructed in the MNI space using q-space diffeomorphic reconstruction (Yeh et al., Neuroimage, 58(1):91-9, 2011) to obtain the spin distribution function (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010). "
            << " A diffusion sampling length ratio of "
            << float(voxel.param[0]) << " was used.";
            // run gqi to get the spin quantity

            // obtain QA map for normalization
            {
                std::vector<tipl::pointer_image<float,3> > tmp;
                auto mask = voxel.mask;
                // clear mask to create whole volume QA map
                std::fill(voxel.mask.begin(),voxel.mask.end(),1.0);
                if (!reconstruct<qa_map>("GQI for QSDR"))
                    return false;
                if (!reconstruct<qsdr_process>("QSDR reconstruction"))
                    return false;
                voxel.mask = mask;
            }

            voxel.recon_report
            << " The output resolution in diffeomorphic reconstruction was " << voxel.vs[0] << " mm isotorpic.";
            if(voxel.output_rdi)
                voxel.recon_report <<
                    " The restricted diffusion was quantified using restricted diffusion imaging (Yeh et al., MRM, 77:603–612 (2017)).";

            break;
        default:
            error_msg = "unknown method";
            return false;
        }
        return save_fib(file_name.find(".fib.gz") == std::string::npos ? file_name + get_file_ext():file_name);
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
    image_model.voxel.mask = mni_mask;
    std::copy(mni,mni+16,image_model.voxel.trans_to_mni.begin());
    std::copy(vs,vs+3,image_model.voxel.vs.begin());
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>("Template reconstruction"))
        return false;
    image_model.save_fib(std::string(out_name)+ext);
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


const char* odf_average(const char* out_name,std::vector<std::string>& file_names)
{
    static std::string report,error_msg;
    tessellated_icosahedron ti;
    float vs[3];
    tipl::geometry<3> dim;
    std::vector<std::vector<float> > odfs;
    std::vector<size_t> odf_count;

    unsigned int half_vertex_count = 0;
    unsigned int row,col;
    float mni[16]={0};
    std::string file_name;
    try {
        begin_prog("averaging");
        for (unsigned int index = 0;check_prog(index,file_names.size());++index)
        {
            file_name = file_names[index];
            gz_mat_read reader;
            set_title(file_name.c_str());
            if(!reader.load_from_file(file_name.c_str()))
                throw std::runtime_error("cannot open file");

            const float* odf_buffer;
            const short* face_buffer;
            const unsigned short* dimension;
            const float* vs_ptr;
            const float* fa0;
            const float* mni_ptr;
            unsigned int face_num,odf_num;

            if(!reader.read("dimension",row,col,dimension)||
               !reader.read("fa0",row,col,fa0) ||
               !reader.read("voxel_size",row,col,vs_ptr) ||
               !reader.read("odf_faces",row,face_num,face_buffer) ||
               !reader.read("odf_vertices",row,odf_num,odf_buffer) ||
               !reader.read("trans",row,col,mni_ptr))
                throw std::runtime_error("invalid FIB file format");

            if(index == 0)
            {
                reader.read("report",report);
                dim = tipl::geometry<3>(dimension);
                std::copy(vs_ptr,vs_ptr+3,vs);
                ti.init(uint16_t(odf_num),odf_buffer,uint16_t(face_num),face_buffer);
                half_vertex_count = odf_num >> 1;
                std::copy(mni_ptr,mni_ptr+16,mni);
            }
            else
            // check odf consistency
            {
                if(odf_num != ti.vertices_count)
                    throw std::runtime_error("inconsistent ODF dimension");
                if(dim != tipl::geometry<3>(dimension))
                    throw std::runtime_error("inconsistent image dimension");
                for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
                {
                    if(std::fabs(ti.vertices[index][0]-odf_buffer[0]) > 0.0f ||
                       std::fabs(ti.vertices[index][1]-odf_buffer[1]) > 0.0f)
                    throw std::runtime_error("inconsistent ODF orientations");
                }
            }

            odf_data buf;
            if(!buf.read(reader))
                throw std::runtime_error("cannot find ODF data");

            if(index == 0)
            {
                odfs.resize(dim.size());
                odf_count.resize(dim.size());
            }

            tipl::par_for(dim.size(),[&](unsigned int i){
                const float* odf_data = buf.get_odf_data(i);
                if(odf_data == nullptr)
                    return;
                if(odfs[i].empty())
                    odfs[i] = std::vector<float>(odf_data,odf_data+half_vertex_count);
                else
                    tipl::add(odfs[i].begin(),odfs[i].end(),odf_data);
                odf_count[i]++;
            });

        }
        if (prog_aborted())
            return nullptr;
    } catch (const std::exception& e) {
        error_msg = e.what();
        error_msg += " at ";
        error_msg += file_name;
        check_prog(0,0);
        return error_msg.c_str();
    }

    // averaging ODFs
    tipl::par_for(dim.size(),[&](unsigned int i){
        if(odf_count[i] > 1)
            tipl::divide_constant(odfs[i].begin(),odfs[i].end(),float(odf_count[i]));
    });

    // eliminate ODF if missing more than half of the population
    tipl::image<unsigned char,3> mask(dim);
    unsigned int odf_size = 0;
    for(unsigned int i = 0;i < mask.size();++i)
    {
        if(odf_count[i] > (file_names.size() >> 1))
        {
            mask[i] = 1;
            std::copy(odfs[i].begin(),odfs[i].end(),odfs[odf_size].begin());
            ++odf_size;
        }
    }
    odfs.resize(odf_size);

    std::ostringstream out;
    out << "A group average template was constructed from a total of " << file_names.size() << " subjects." << report.c_str();
    report = out.str();
    output_odfs(mask,out_name,".mean.odf.fib.gz",odfs,ti,vs,mni,report);
    output_odfs(mask,out_name,".mean.fib.gz",odfs,ti,vs,mni,report,false);
    return nullptr;
}









