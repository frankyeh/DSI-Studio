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


std::pair<unsigned int,unsigned int> evaluate_fib(
        const image::geometry<3>& dim,
        const std::vector<std::vector<float> >& fib_fa,
        const std::vector<std::vector<float> >& fib_dir)
{
    unsigned char num_fib = fib_fa.size();
    char dx[13] = {1,0,0,1,1,0, 1, 1, 0, 1,-1, 1, 1};
    char dy[13] = {0,1,0,1,0,1,-1, 0, 1, 1, 1,-1, 1};
    char dz[13] = {0,0,1,0,1,1, 0,-1,-1, 1, 1, 1,-1};
    std::vector<image::vector<3> > dis(13);
    for(unsigned int i = 0;i < 13;++i)
    {
        dis[i] = image::vector<3>(dx[i],dy[i],dz[i]);
        dis[i].normalize();
    }
    float otsu = image::segmentation::otsu_threshold(fib_fa[0])*0.6;
    std::vector<std::vector<unsigned char> > connected(fib_fa.size());
    for(unsigned int index = 0;index < connected.size();++index)
        connected[index].resize(dim.size());
    unsigned int connection_count = 0;
    for(image::pixel_index<3> index;index.is_valid(dim);index.next(dim))
    {
        if(fib_fa[0][index.index()] <= otsu)
            continue;
        unsigned int index3 = index.index()+index.index()+index.index();
        for(unsigned char fib1 = 0;fib1 < num_fib;++fib1)
        {
            if(fib_fa[fib1][index.index()] <= otsu)
                break;
            for(unsigned int j = 0;j < 2;++j)
            for(unsigned int i = 0;i < 13;++i)
            {
                image::vector<3,int> pos;
                pos = j ? image::vector<3,int>(index[0] + dx[i],index[1] + dy[i],index[2] + dz[i])
                          :image::vector<3,int>(index[0] - dx[i],index[1] - dy[i],index[2] - dz[i]);
                if(!dim.is_valid(pos))
                    continue;
                image::pixel_index<3> other_index(pos[0],pos[1],pos[2],dim);
                unsigned int other_index3 = other_index.index()+other_index.index()+other_index.index();
                if(std::abs(image::vector<3>(&fib_dir[fib1][index3])*dis[i]) <= 0.8665)
                    continue;
                for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                    if(fib_fa[fib2][other_index.index()] > otsu &&
                            std::abs(image::vector<3>(&fib_dir[fib2][other_index3])*dis[i]) > 0.8665)
                    {
                        connected[fib1][index.index()] = 1;
                        connected[fib2][other_index.index()] = 1;
                        ++connection_count;
                    }
            }
        }
    }
    unsigned int no_connection_count = 0;
    for(image::pixel_index<3> index;index.is_valid(dim);index.next(dim))
    {
        for(unsigned int i = 0;i < num_fib;++i)
            if(fib_fa[i][index.index()] > otsu && !connected[i][index.index()])
                ++no_connection_count;
    }

    return std::make_pair(connection_count,no_connection_count);
}

void flip_fib_dir(std::vector<float>& fib_dir,bool x,bool y,bool z)
{
    for(unsigned int j = 0;j+2 < fib_dir.size();j += 3)
    {
        if(x)
            fib_dir[j] = -fib_dir[j];
        if(y)
            fib_dir[j+1] = -fib_dir[j+1];
        if(z)
            fib_dir[j+2] = -fib_dir[j+2];
    }
}

void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);

extern "C"
    const char* reconstruction(ImageModel* image_model,unsigned int method_id,const float* param_values)
{
    static std::string output_name;
    try
    {
        {
            std::vector<unsigned int> shell;
            calculate_shell(image_model->voxel.bvalues,shell);
            image_model->voxel.half_sphere =
                    (method_id == 7 || method_id == 4) &&
                    (shell.size() > 5) && (shell[1] - shell[0] <= 3);
            if(!image_model->voxel.half_sphere)
            {
                image_model->voxel.scheme_balance =
                    (method_id == 7 || method_id == 4) &&
                    (shell.size() <= 5) && !shell.empty() &&
                    image_model->voxel.bvalues.size()-shell.back() < 100;
            }
        }


        image_model->voxel.recon_report.clear();
        image_model->voxel.recon_report.str("");
        image_model->voxel.param = param_values;
        std::ostringstream out;
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
        // correct for b-table orientation

        {
            set_title("checking b-table");
            image_model->reconstruct<dti_process>();
            std::vector<std::vector<float> > fib_fa(1);
            std::vector<std::vector<float> > fib_dir(1);
            fib_fa[0].swap(image_model->voxel.fib_fa);
            fib_dir[0].swap(image_model->voxel.fib_dir);
            unsigned int cur_score = evaluate_fib(image_model->voxel.dim,fib_fa,fib_dir).first;
            flip_fib_dir(fib_dir[0],true,false,false);
            unsigned int flip_x_score = evaluate_fib(image_model->voxel.dim,fib_fa,fib_dir).first;
            flip_fib_dir(fib_dir[0],true,true,false);
            unsigned int flip_y_score = evaluate_fib(image_model->voxel.dim,fib_fa,fib_dir).first;
            flip_fib_dir(fib_dir[0],false,true,true);
            unsigned int flip_z_score = evaluate_fib(image_model->voxel.dim,fib_fa,fib_dir).first;
            if(flip_x_score > cur_score &&
               flip_x_score > flip_y_score && flip_x_score > flip_z_score)
            {
                std::cout << "b-table flipped x" << std::endl;
                for(unsigned int index = 0;index < image_model->voxel.bvectors.size();++index)
                    image_model->voxel.bvectors[index][0] = -image_model->voxel.bvectors[index][0];
                out << ".fx";
            }
            if(flip_y_score > cur_score &&
               flip_y_score > flip_x_score && flip_y_score > flip_z_score)
            {
                std::cout << "b-table flipped y" << std::endl;
                for(unsigned int index = 0;index < image_model->voxel.bvectors.size();++index)
                    image_model->voxel.bvectors[index][1] = -image_model->voxel.bvectors[index][1];
                out << ".fy";
            }
            if(flip_z_score > cur_score &&
               flip_z_score > flip_y_score && flip_z_score > flip_x_score)
            {
                std::cout << "b-table flipped z" << std::endl;
                for(unsigned int index = 0;index < image_model->voxel.bvectors.size();++index)
                    image_model->voxel.bvectors[index][2] = -image_model->voxel.bvectors[index][2];
                out << ".fz";
            }

        }




        switch (method_id)
        {
        case 0: //DSI local max
            image_model->voxel.recon_report <<
            " The diffusion data were reconstructed using diffusion spectrum imaging (Wedeen et al. MRM, 2005) with a Hanning filter of " << (int)param_values[0] << ".";
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<dsi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
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
                    return "reconstruction calceled";
                begin_prog("calculating");
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
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            out << ".qbi.sh"<< (int) param_values[1] << "." << param_values[0] << ".fib.gz";
            if (!image_model->reconstruct<qbi_sh_process>())
                return "reconstruction canceled";
            break;

        case 4://GQI
            if(param_values[0] == 0.0) // spectral analysis
            {
                image_model->voxel.recon_report <<
                " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, 2010).";
                out << (image_model->voxel.r2_weighted ? ".gqi2.spec.fib.gz":".gqi.spec.fib.gz");
                if (!image_model->reconstruct<gqi_spectral_process>())
                    return "reconstruction canceled";
                break;
            }
            image_model->voxel.recon_report <<
            " The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, 2010) with a diffusion sampling length ratio of " << (float)param_values[0] << ".";
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<gqi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("calculating");
            }
            if(image_model->voxel.r2_weighted)
                image_model->voxel.recon_report << " The ODF calculation was weighted by the square of the diffuion displacement.";
            out << (image_model->voxel.r2_weighted ? ".gqi2.":".gqi.") << param_values[0] << ".fib.gz";
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
            image_model->voxel.recon_report
            << " The diffusion data were reconstructed using q-space diffeomorphic reconstruction (Yeh et al. Neuroimage, 2011) with a diffusion sampling length ratio of "
            << (float)param_values[0] << ". The output resolution was " << param_values[1] << " mm.";
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
            begin_prog("deforming");
            if (!image_model->reconstruct<gqi_mni_process>())
                return "reconstruction canceled";
            out << ".R" << (int)std::floor(image_model->voxel.R2*100.0) << ".fib.gz";
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
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>())
        return false;
    image_model.save_fib(ext);
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


extern "C"
    const char* odf_average(const char* out_name,
                     const char* const * file_names,
                     unsigned int num_files)
{
    static std::string error_msg,report;
    tessellated_icosahedron ti;
    float vs[3];
    image::basic_image<unsigned char,3> mask;
    std::vector<std::vector<float> > odfs;
    begin_prog("averaging");
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

    for (unsigned int odf_index = 0;odf_index < odfs.size();++odf_index)
        for (unsigned int j = 0;j < odfs[odf_index].size();++j)
            odfs[odf_index][j] /= (double)num_files;

    std::ostringstream out;
    out << "A group average template was constructed from a total of " << num_files << " subjects." << report.c_str();
    report = out.str();
    output_odfs(mask,out_name,".mean.odf.fib.gz",odfs,ti,vs,mni,report);
    output_odfs(mask,out_name,".mean.fib.gz",odfs,ti,vs,mni,report,false);
    return 0;
}









