#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/math/distributions/students_t.hpp>
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
typedef boost::mpl::vector<
    ReadDWIData,
    Dwi2Tensor
> dti_process;


template<typename reco_type>
struct odf_reco_type{
    typedef boost::mpl::vector<
        ODFDeconvolusion,
        ODFDecomposition,
        DetermineFiberDirections,
        ScaleZ0ToMinODF,
        SaveFA,
        SaveDirIndex,
        OutputODF
    > common_odf_process;
    typedef typename boost::mpl::insert_range<common_odf_process,boost::mpl::begin<common_odf_process>::type,reco_type>::type type0;
    typedef typename boost::mpl::push_front<type0,ReadDWIData>::type type; // add ReadDWIData to the front
};

template<typename reco_type>
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


typedef odf_reco_type<boost::mpl::vector<
    BalanceScheme,
    QSpace2Odf
> >::type gqi_process;

typedef boost::mpl::vector<
    ReadDWIData,
    QSpaceSpectral
> gqi_spectral_process;

typedef boost::mpl::vector<
    ReadDWIData,
    BalanceScheme,
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
    QSpace2Odf,
    DetermineFiberDirections,
    RecordQA,
    EstimateResponseFunction

> gqi_estimate_response_function;

typedef boost::mpl::vector<
    DWINormalization,
    BalanceScheme,
    QSDR,
    ODFDeconvolusion,
    ODFDecomposition,
    EstimateZ0_MNI,
    DetermineFiberDirections,
    SaveFA,
    SaveDirIndex,
    OutputODF
> gqi_mni_process;

typedef boost::mpl::vector<
    ODFLoader,
    DetermineFiberDirections,
    SaveFA,
    SaveDirIndex
> reprocess_odf;



extern "C"
    const char* reconstruction(ImageModel* image_model,unsigned int method_id,const float* param_values)
{
    static std::string output_name;
    try
    {
        image_model->voxel.param = param_values;
        std::ostringstream out;
        if(method_id != 1) // not DTI
        {
            out << ".odf" << image_model->voxel.ti.fold;// odf_order
            out << ".f" << image_model->voxel.max_fiber_number;
            if (image_model->voxel.need_odf)
                out << "rec";
            if (image_model->voxel.scheme_balance)
                out << ".bal";
            if (image_model->voxel.half_sphere)
                out << ".hs";
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
        switch (method_id)
        {
        case 0: //DSI local max
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<dsi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            out << ".dsi."<< (int)param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<dsi_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 1://DTI
            out << ".dti.fib.gz";
            image_model->voxel.max_fiber_number = 1;
            if (!image_model->reconstruct<dti_process>(out.str()))
                return "reconstruction canceled";
            break;

        case 2://QBI
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            out << ".qbi."<< param_values[0] << "_" << param_values[1] << ".fib.gz";
            if (!image_model->reconstruct<qbi_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 3://QBI
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_sh_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            out << ".qbi.sh"<< (int) param_values[1] << "." << param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<qbi_sh_process>(out.str()))
                return "reconstruction canceled";
            break;

        case 4://GQI
            if(param_values[0] == 0.0) // spectral analysis
            {
                out << (image_model->voxel.r2_weighted ? ".gqi2.spec.fib.gz":".gqi.spec.fib.gz");
                if (!image_model->reconstruct<gqi_spectral_process>(out.str()))
                    return "reconstruction canceled";
                break;
            }
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<gqi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            out << (image_model->voxel.r2_weighted ? ".gqi2.":".gqi.") << param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<gqi_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 6:
            out << ".hardi."<< param_values[0]
                << ".b" << param_values[1]
                << ".reg" << param_values[2] << ".src.gz";
            if (!image_model->reconstruct<hardi_convert_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 7:
            // run gqi to get the spin quantity
            if (!image_model->reconstruct<gqi_estimate_response_function>())
                return "reconstruction calceled";
            out << ".reg" << (int)image_model->voxel.reg_method;
            out << (image_model->voxel.r2_weighted ? ".qsdr2.":".qsdr.");
            out << param_values[0] << "." << param_values[1] << "mm";
            if(image_model->voxel.output_jacobian)
                out << ".jac";
            if(image_model->voxel.output_mapping)
                out << ".map";
            out << ".fib.gz";
            begin_prog("deforming");
            if (!image_model->reconstruct<gqi_mni_process>(out.str()))
                return "reconstruction canceled";
            break;
        }
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
                 bool record_odf = true)
{
    begin_prog("output");
    ImageModel image_model;
    image_model.voxel.dim = mni_mask.geometry();
    image_model.voxel.ti = ti;
    image_model.voxel.q_count = 0;
    image_model.voxel.odf_decomposition = false;
    image_model.voxel.odf_deconvolusion = false;
    image_model.voxel.half_sphere = false;
    image_model.voxel.max_fiber_number = 5;
    image_model.voxel.z0 = 0.0;
    image_model.voxel.need_odf = record_odf;
    image_model.voxel.template_odfs.swap(odfs);
    image_model.voxel.param = mni;
    image_model.thread_count = 1;
    image_model.file_name = out_name;
    image_model.mask = mni_mask;
    std::copy(vs,vs+3,image_model.voxel.vs.begin());
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>(ext))
        return false;
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


extern "C"
    const char* odf_average(const char* out_name,
                     const char* const * file_names,
                     unsigned int num_files)
{
    static std::string error_msg;
    tessellated_icosahedron ti;
    float vs[3];
    image::basic_image<unsigned char,3> mask;
    std::vector<std::vector<float> > odfs;
    begin_prog("averaging");
    can_cancel(true);
    unsigned int half_vertex_count = 0;
    unsigned int row,col;
    float mni[16]={0};
    for (unsigned int index = 0;check_prog(index,num_files);++index)
    {
        const char* file_name = file_names[index];
        gz_mat_read reader;
        if(!reader.load_from_file(file_name))
        {
            error_msg = "Cannot open file ";
            error_msg += file_name;
            check_prog(0,0);
            return error_msg.c_str();
        }
        if(index == 0)
        {
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

    for (unsigned int odf_index = 0;odf_index < odfs.size();++odf_index)
        for (unsigned int j = 0;j < odfs[odf_index].size();++j)
            odfs[odf_index][j] /= (double)num_files;

    output_odfs(mask,out_name,".mean.odf.fib.gz",odfs,ti,vs,mni);
    output_odfs(mask,out_name,".mean.fib.gz",odfs,ti,vs,mni,false);
    return 0;
}









