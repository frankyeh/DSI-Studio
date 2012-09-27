#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/insert_range.hpp>
#include <boost/mpl/begin_end.hpp>

#include "stdafx.h"
#include "tessellated_icosahedron.hpp"
#include "prog_interface_static_link.h"
#include "mat_file.hpp"
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
    ADCProfile,
    Dwi2Tensor,
    TensorEigenAnalysis
//OutputODF
> dti_process;


template<typename reco_type>
struct odf_reco_type{
    typedef boost::mpl::vector<
        ODFDeconvolusion,
        ODFDecomposition,
        DetermineFiberDirections,
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
    SHDecomposition<8>
> >::type qbi_sh_process;


typedef odf_reco_type<boost::mpl::vector<
    CorrectB0,
    QSpace2Odf
> >::type gqi_process;


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
    SHDecomposition<8>
> >::type qbi_sh_estimate_response_function;


// for ODF deconvolution
typedef boost::mpl::vector<
    ReadDWIData,
    CorrectB0,
    QSpace2Odf,
    DetermineFiberDirections,
    RecordQA,
    EstimateResponseFunction
> gqi_estimate_response_function;

typedef boost::mpl::vector<
    GQI_MNI,
    //GQI_phantom,
    ODFDecomposition,
    DetermineFiberDirections,
    SaveFA,
    SaveDirIndex,
    OutputODF
> gqi_mni_process;

typedef boost::mpl::vector<
    GQI_MNI,
    AccumulateODF
> gqi_mni_template_process;

typedef boost::mpl::vector<
    ODFLoader,
    DetermineFiberDirections,
    SaveFA,
    SaveDirIndex
> reprocess_odf;


#include "mix_gaussian_model.hpp"
#include "racian_noise.hpp"

#include "layout.hpp"


boost::mt19937 RacianNoise::generator(static_cast<unsigned> (std::time(0)));
boost::normal_distribution<float> RacianNoise::normal;
boost::uniform_real<float> RacianNoise::uniform(0.0,1.0);
boost::variate_generator<boost::mt19937&,
boost::normal_distribution<float> > RacianNoise::gen_normal(RacianNoise::generator,RacianNoise::normal);
boost::variate_generator<boost::mt19937&,
boost::uniform_real<float> > RacianNoise::gen_uniform(RacianNoise::generator,RacianNoise::uniform);
std::string error_msg;


extern "C"
    void* init_reconstruction(const char* file_name)
{
    std::auto_ptr<ImageModel> image(new ImageModel);
    if (!image->load_from_file(file_name))
        return 0;
    return image.release();
}
extern "C"
    void free_reconstruction(ImageModel* image_model)
{
    delete image_model;
}


extern "C"
    const float* get_b_table(ImageModel* image_model,unsigned int& b_number)
{
    unsigned int row;
    const float* table;
    image_model->mat_reader->get_matrix("b_table",row,b_number,table);
    return table;
}


extern "C"
    const unsigned short* get_dimension(ImageModel* image_model)
{
    static unsigned short dim[3];
    dim[0] = image_model->voxel.matrix_width;
    dim[1] = image_model->voxel.matrix_height;
    dim[2] = image_model->voxel.slice_number;
    return dim;
}

extern "C"
    const float* get_voxel_size(ImageModel* image_model)
{
    return image_model->voxel.voxel_size;
}

extern "C"
    unsigned char* get_mask_image(ImageModel* image_model)
{
    return &*image_model->mask.begin();
}

extern "C"
    char* check_reconstruction(ImageModel* image_model)
{
    static char ava[13];
    ava[0] = image_model->avaliable<CheckDSI>();
    ava[1] = image_model->avaliable<CheckDTI>();
    ava[2] = ava[3] = image_model->avaliable<CheckHARDI>();
    ava[4] = 1;
    ava[5] = 1;
    ava[6] = 1;
    ava[7] = 1;
    return ava;
}



extern "C"
    const char* reconstruction(ImageModel* image_model,unsigned int method_id,const float* param_values)
{
    static std::string output_name;
    try
    {
        image_model->voxel.param = param_values;
        std::ostringstream out;
        out << ".odf" << image_model->voxel.ti.fold;// odf_order
        out << ".f" << image_model->voxel.max_fiber_number;
        if (image_model->voxel.need_odf && method_id != 1)
            out << "rec";
        if (image_model->voxel.half_sphere && method_id != 1)
            out << ".hs";
        if (image_model->voxel.odf_deconvolusion && method_id != 1)
        {
            out << ".de" << param_values[2];
            if(image_model->voxel.odf_xyz[0] != 0 ||
               image_model->voxel.odf_xyz[1] != 0 ||
               image_model->voxel.odf_xyz[2] != 0)
                out << ".at_" << image_model->voxel.odf_xyz[0]
                    << "_" << image_model->voxel.odf_xyz[1]
                    << "_" << image_model->voxel.odf_xyz[2];
        }
        if (image_model->voxel.odf_decomposition && method_id != 1)
        {
            out << ".dec" << param_values[3];
            if(image_model->voxel.odf_xyz[0] != 0 ||
               image_model->voxel.odf_xyz[1] != 0 ||
               image_model->voxel.odf_xyz[2] != 0)
                out << ".at_" << image_model->voxel.odf_xyz[0]
                    << "_" << image_model->voxel.odf_xyz[1]
                    << "_" << image_model->voxel.odf_xyz[2];
        }
        switch (method_id)
        {
        case 0: //DSI local max
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<dsi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("deconvolving");
            }
            out << ".dsi."<< (int)param_values[0] << ".fib";
            if (!image_model->reconstruct<dsi_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 1://DTI
            out << ".dti.fib";
            image_model->voxel.max_fiber_number = 1;
            if (!image_model->reconstruct<dti_process>(out.str()))
                return "reconstruction canceled";
            break;

        case 2://QBI
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("deconvolving");
            }
            out << ".qbi."<< param_values[0] << "_" << param_values[1] << ".fib";
            if (!image_model->reconstruct<qbi_process>(out.str()))
                return "reconstruction canceled";
            break;
        case 3://QBI
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<qbi_sh_estimate_response_function>())
                    return "reconstruction calceled";

                begin_prog("deconvolving");
            }
            out << ".qbi.sh."<< param_values[0] << ".fib";
            if (!image_model->reconstruct<qbi_sh_process>(out.str()))
                return "reconstruction canceled";
            break;

        case 4://GQI
            if (image_model->voxel.odf_deconvolusion || image_model->voxel.odf_decomposition)
            {
                if (!image_model->reconstruct<gqi_estimate_response_function>())
                    return "reconstruction calceled";
                begin_prog("deconvolving");
            }
            out << (image_model->voxel.r2_weighted ? ".gqir2.":".gqi.") << param_values[0] << ".fib";
            if (!image_model->reconstruct<gqi_process>(out.str()))
                return "reconstruction canceled";
            break;
            /*
        case 6:
            out << ".gqi.hardi"<< param_values[0] << ".src";
            if (!image_model->reconstruct<gqi_adaptor_process>(out.str()))
                return "reconstruction canceled";
            break;
            */
        case 7:
            // run gqi to get the spin quantity
            std::fill(image_model->mask.begin(),image_model->mask.end(),1.0);
            if (!image_model->reconstruct<gqi_estimate_response_function>())
                return "reconstruction calceled";
            out << (image_model->voxel.r2_weighted ? ".qsdr2.":".qsdr.");
            out << param_values[0] << "." << param_values[1] << "mm.fib";
            begin_prog("deforming");
            if (!image_model->reconstruct<gqi_mni_process>(out.str()))
                return "reconstruction canceled";
            break;

        case 8:
            {
                for(int index = 0;index < image_model->voxel.file_list.size();++index)
                {
                    {
                        std::ostringstream msg;
                        msg << "loading (" << index << "/" << image_model->voxel.file_list.size();
                        begin_prog(msg.str().c_str());
                    }
                    if(index)
                        if(!image_model->load_from_file(image_model->voxel.file_list[index].c_str()))
                        {
                            output_name = "Cannot open file ";
                            output_name += image_model->voxel.file_list[index];
                            return output_name.c_str();
                        }
                    {
                        std::ostringstream msg;
                        msg << "running (" << index << "/" << image_model->voxel.file_list.size();
                        begin_prog(msg.str().c_str());
                    }
                    std::fill(image_model->mask.begin(),image_model->mask.end(),1.0);
                    if (!image_model->reconstruct<gqi_estimate_response_function>()||
                        !image_model->reconstruct<gqi_mni_template_process>())
                        return "reconstruction calceled";
                }

                //averaging
                for (unsigned int index = 0;index < image_model->voxel.template_odfs.size();++index)
                {
                    std::for_each(image_model->voxel.template_odfs[index].begin(),
                                  image_model->voxel.template_odfs[index].end(),
                                  boost::lambda::_1 /= image_model->voxel.file_list.size());
                }

                out << (image_model->voxel.r2_weighted ? ".qsdr2.":".qsdr.");
                out << param_values[0] << "." << param_values[1] << "mm.fib";

                begin_prog("output result without odf");
                image_model->voxel.need_odf = false;
                image_model->file_name = image_model->voxel.template_file_name;
                if (!image_model->reconstruct<reprocess_odf>(out.str()))
                    return "reconstruction calceled";

                begin_prog("output result without odf");
                image_model->voxel.need_odf = true;
                image_model->file_name = image_model->voxel.template_file_name;
                image_model->file_name += ".rec";
                if (!image_model->reconstruct<reprocess_odf>(out.str()))
                    return "reconstruction calceled";

            }
        }
        output_name = image_model->file_name + out.str() + ".gz";
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
                 bool record_odf = true)
{
    begin_prog("output");
    ImageModel image_model;
    image_model.set_dimension(mni_mask.width(),mni_mask.height(),mni_mask.depth());
    image_model.voxel.ti = ti;
    image_model.voxel.q_count = 0;
    image_model.voxel.odf_decomposition = false;
    image_model.voxel.odf_deconvolusion = false;
    image_model.voxel.half_sphere = false;
    image_model.voxel.max_fiber_number = 3;
    image_model.voxel.qa_scaling = 1;
    image_model.voxel.need_odf = record_odf;
    image_model.voxel.template_odfs.swap(odfs);
    image_model.thread_count = 1;
    image_model.file_name = out_name;
    std::copy(mni_mask.begin(),mni_mask.end(),image_model.mask.begin());
    std::copy(vs,vs+3,image_model.voxel.voxel_size);
    if (prog_aborted() || !image_model.reconstruct<reprocess_odf>(ext))
        return false;
    image_model.voxel.template_odfs.swap(odfs);
    return true;
}


extern "C"
    bool odf_average(const char* out_name,
                     const char* const * file_names,
                     unsigned int num_files)
{
    tessellated_icosahedron ti;
    float vs[3];
    image::basic_image<unsigned char,3> mask;
    std::vector<std::vector<float> > odfs;
    begin_prog("averaging");
    can_cancel(true);
    unsigned int half_vertex_count = 0;
    unsigned int row,col;
    for (unsigned int index = 0;check_prog(index,num_files);++index)
    {
        const char* file_name = file_names[index];
        MatFile reader;
        if(!reader.load_from_file(file_name))
        {
            std::cout << "Cannot open file " << file_name << std::endl;
            return false;
        }
        if(index == 0)
        {
            const float* odf_buffer;
            const short* face_buffer;
            const unsigned short* dimension;
            const float* vs_ptr;
            const float* fa0;
            unsigned int face_num,odf_num;
            if(!reader.get_matrix("dimension",row,col,dimension) ||
               !reader.get_matrix("fa0",row,col,fa0) ||
               !reader.get_matrix("voxel_size",row,col,vs_ptr) ||
               !reader.get_matrix("odf_faces",row,face_num,face_buffer) ||
               !reader.get_matrix("odf_vertices",row,odf_num,odf_buffer))
            {
                std::cout << "Cannot find image information in " << file_name << std::endl;
                return false;
            }
            mask.resize(image::geometry<3>(dimension));
            for(unsigned int index = 0;index < mask.size();++index)
                if(fa0[index] != 0.0)
                    mask[index] = 1;
            std::copy(vs_ptr,vs_ptr+3,vs);
            ti.init(odf_num,odf_buffer,face_num,face_buffer);
            half_vertex_count = odf_num >> 1;
        }
        else
        // check odf consistency
        {
            const float* odf_buffer;
            const unsigned short* dimension;
            unsigned int odf_num;
            if(!reader.get_matrix("dimension",row,col,dimension) ||
               !reader.get_matrix("odf_vertices",row,odf_num,odf_buffer))
            {
                std::cout << "Cannot find image information in " << file_name << std::endl;
                return false;
            }

            if(odf_num != ti.vertices_count || dimension[0] != mask.width() ||
                    dimension[1] != mask.height() || dimension[2] != mask.depth())
            {
                std::cout << "Inconsistent ODF orientations in " << file_name << std::endl;
                return false;
            }
            for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
            {
                if(ti.vertices[index][0] != odf_buffer[0] ||
                   ti.vertices[index][1] != odf_buffer[1] ||
                   ti.vertices[index][2] != odf_buffer[2])
                {
                    std::cout << "Inconsistent ODF orientations in " << file_name << std::endl;
                    return false;
                }
            }
        }

        {
            const float* fa0;
            if(!reader.get_matrix("fa0",row,col,fa0))
            {
                std::cout << "Cannot find image information in " << file_name << std::endl;
                return false;
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
                if (!reader.get_matrix(out.str().c_str(),row,col,odf_buf))
                    break;
                odf_bufs.push_back(odf_buf);
                odf_bufs_size.push_back(row*col);
            }
        }
        if(odfs.empty())
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
            else
                for(unsigned int i = 0;i < odf_bufs.size();++i)
                    if(odfs[i].size() != odf_bufs_size[i])
                        inconsistence = true;
            if(inconsistence)
            {
                std::cout << "Inconsistence mask coverage in" << file_name << std::endl;
                return false;
            }
        }
        for(unsigned int i = 0;i < odf_bufs.size();++i)
        {
            image::add(odfs[i].begin(),odfs[i].end(),odf_bufs[i]);
            for(unsigned int j = 0;j < odf_bufs_size[i];)
            {
                unsigned int next_j = j + half_vertex_count;
                image::minus_constant(odfs[i].begin() + j,
                                      odfs[i].begin() + next_j,
                    *std::min_element(odf_bufs[i]+j,odf_bufs[i]+next_j));
                j = next_j;
            }
        }
    }
    if (prog_aborted())
        return false;

    for (unsigned int odf_index = 0;odf_index < odfs.size();++odf_index)
        for (unsigned int j = 0;j < odfs[odf_index].size();++j)
            odfs[odf_index][j] /= (double)num_files;

    output_odfs(mask,out_name,".mean.odf.fib",odfs,ti,vs);
    output_odfs(mask,out_name,".mean.fib",odfs,ti,vs,false);
}

extern "C"
    bool generate_simulation(
        const char* bvec_file_name,unsigned char s0_snr,float mean_dif,unsigned char odf_fold,
        const char* fa_iteration,
        const char* crossing_angle_iteration,
        unsigned char repeat_num)
{
    tessellated_icosahedron ti;
    ti.init(odf_fold);
    Layout layout(s0_snr,mean_dif);
    if (!layout.load_b_table(bvec_file_name))
        return false;
    std::vector<float> fa;
    std::vector<float> angle;
    {
        std::string fa_iteration_str(fa_iteration);
        std::istringstream tmp(fa_iteration_str);
        std::copy(std::istream_iterator<float>(tmp),
                  std::istream_iterator<float>(),std::back_inserter(fa));
    }
    {
        std::string crossing_angle_iteration_str(crossing_angle_iteration);
        std::istringstream tmp(crossing_angle_iteration_str);
        std::copy(std::istream_iterator<float>(tmp),
                  std::istream_iterator<float>(),std::back_inserter(angle));
    }

    layout.createLayout(fa,angle,repeat_num);
    std::ostringstream out;
    out << bvec_file_name << "_snr" << (int)s0_snr << "_dif" << mean_dif << "_odf" << (int)odf_fold << "_n" << (int)repeat_num << ".src";
    layout.generate(out.str().c_str());
    return true;
}








