#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <QFileInfo>
#include <chrono>
#include <cmath>
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "gqi_process.hpp"
#include "reg.hpp"
extern std::vector<std::string> fa_template_list,iso_template_list,t1w_template_list;
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
class DWINormalization  : public BaseProcess
{
protected:
    tipl::shape<3> native_geo;
    tipl::vector<3> native_vs;
protected:
    tipl::image<3,tipl::vector<3> > cdm_dis,mapping;
protected:
    tipl::transformation_matrix<float> affine;
protected:
    std::vector<float> jdet;
protected:
    typedef tipl::const_pointer_image<3,unsigned short> point_image_type;
    std::vector<point_image_type> ptr_images;

public:
    virtual void init(Voxel& voxel)
    {
        tipl::progress prog("QA/ISO normalization");
        dual_reg reg;
        reg.param = voxel.reg_param;
        reg.modality_names = {"qa","iso"};
        reg.export_intermediate = voxel.needs("debug");

        if(!reg.load_template(0,fa_template_list[voxel.template_id]) ||
           !reg.load_template(1,iso_template_list[voxel.template_id]))
            throw std::runtime_error("cannot load anisotropy/isotropy template");

        reg.I[0] = subject_image_pre(std::move(voxel.qa_map));
        reg.I[1] = subject_image_pre(std::move(voxel.iso_map));
        reg.Is = native_geo = voxel.dim;
        reg.Ivs = native_vs = voxel.vs;
        initial_LPS_nifti_srow(reg.IR,native_geo,native_vs);
        auto native_trans = reg.IR;


        bool t1w_reg = false;
        if(!voxel.other_modality_template.empty())
        {
            tipl::out() << "adding " << voxel.other_modality_template << " as template for registration";
            if(!reg.load_template(2,voxel.other_modality_template))
                throw std::runtime_error(std::string("cannot load template: ") + voxel.other_modality_template);
            reg.modality_names[2] = "other";
            tipl::out() << "moving QA/ISO to the registration modality space";
            reg.I[2] = subject_image_pre(tipl::image<3>(voxel.other_modality_subject));
            reg.Is = voxel.other_modality_subject.shape();
            reg.Ivs = voxel.vs = voxel.other_modality_vs;
            for(size_t i = 0;i < 2;++i)
                reg.I[i] = tipl::resample(reg.I[i],voxel.other_modality_subject.shape(),voxel.other_modality_trans);
            t1w_reg = true;
        }

        reg.match_resolution(false);


        // linear registration
        {
            if(voxel.manual_alignment)
            {
                tipl::out() << "manual alignment:" << (reg.arg = voxel.qsdr_arg);
                reg.calculate_linear_r();
            }
            else
            {
                auto nonzero_voxels = [](const auto& image)
                {
                    size_t count = 0;
                    for(size_t i = 0;i < image.size();++i)
                        if(image[i])
                            ++count;
                    return count;
                };
                if(!reg.I[0].empty() && !reg.It[0].empty())
                {
                    auto subject_voxels = nonzero_voxels(reg.I[0]);
                    auto template_voxels = nonzero_voxels(reg.It[0]);
                    double subject_volume = double(subject_voxels) * reg.Ivs[0] * reg.Ivs[1] * reg.Ivs[2];
                    double template_volume = double(template_voxels) * reg.Itvs[0] * reg.Itvs[1] * reg.Itvs[2];
                    if(template_volume > 0.0)
                        tipl::out() << "brain volume ratio (subject/template): "
                                    << (subject_volume / template_volume);
                }
                reg.linear_restarts = 6;
                // Use a wider affine search (especially scaling) for lifespan differences.
                static const float lifespan_affine_bound[3][8] = {
                    {1.0f,-1.0f,  0.5f,-0.5f,  3.0f,0.3f,  0.25f,-0.25f},
                    {1.0f,-1.0f,  0.4f,-0.4f,  3.0f,0.3f,  0.25f,-0.25f},
                    {1.0f,-1.0f,  0.4f,-0.4f,  3.0f,0.3f,  0.25f,-0.25f}
                };
                reg.bound = lifespan_affine_bound;
                tipl::run("linear registration",[&](void)
                {
                    reg.linear_reg(tipl::prog_aborted);
                });
                if(tipl::prog_aborted)
                    throw std::runtime_error("reconstruction canceled");

            }
            if((voxel.R2 = tipl::max_value(reg.r)) < 0.4f)
                tipl::warning() << "poor registration found in linear registration. Please check image quality or orientation. consider using manual alignment.";

        }

        {
            tipl::run("nonlinear registration",[&](void)
            {
                reg.nonlinear_reg(tipl::prog_aborted);
            });
            if(tipl::prog_aborted)
                throw std::runtime_error("reconstruction canceled");


            voxel.R2 = reg.r[1]*reg.r[1];
            tipl::out() << "nonlinear R2: " << voxel.R2 << std::endl;
            if(voxel.R2 < 0.3f)
                tipl::warning() << "poor registration found in nonlinear registration. Please check image quality or image orientation";
            if(!reg.t2f_dis.empty())
            {
                float max_displacement_mm = 0.0f;
                for(size_t i = 0;i < reg.t2f_dis.size();++i)
                {
                    const auto& v = reg.t2f_dis[i];
                    float dx = v[0]*reg.Itvs[0];
                    float dy = v[1]*reg.Itvs[1];
                    float dz = v[2]*reg.Itvs[2];
                    float dis = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if(dis > max_displacement_mm)
                        max_displacement_mm = dis;
                }
                if(max_displacement_mm > 10.0f)
                    tipl::warning() << "large nonlinear displacement detected (max "
                                    << max_displacement_mm
                                    << " mm). Please verify template-to-native alignment.";
            }

            auto new_ItR = reg.ItR;
            auto new_Its = reg.Its;
            new_ItR[0] = new_ItR[5] = -voxel.qsdr_reso;
            new_ItR[10] = voxel.qsdr_reso;
            if(reg.Itvs[0] != voxel.qsdr_reso)
                new_Its = tipl::shape<3>(uint32_t(float(reg.Its.width())*reg.Itvs[0]/voxel.qsdr_reso),
                                         uint32_t(float(reg.Its.height())*reg.Itvs[1]/voxel.qsdr_reso),
                                         uint32_t(float(reg.Its.depth())*reg.Itvs[2]/voxel.qsdr_reso));

            // if subject data is only a fragment of FOV, crop images
            if(voxel.partial_min != voxel.partial_max)
            {
                tipl::out() << "partial reconstruction" << std::endl;
                tipl::vector<3> shift(reg.ItR[3],reg.ItR[7],reg.ItR[11]);
                auto bmin = voxel.partial_min - shift;
                auto bmax = voxel.partial_max - shift;
                tipl::out() << "partial min/max: " << voxel.partial_min << " " << voxel.partial_max;
                tipl::out() << "voxel min/max: " << bmin << " " << bmax;
                for(int i = 0;i < 3;++i)
                {
                    if(bmin[i] > bmax[i])
                        std::swap(bmin[i],bmax[i]);
                    if(bmin[i] < 0.0f || bmax[i] > reg.Its[i]*reg.Itvs[i])
                        throw std::runtime_error("out of bounding box in partial reconstruction.");
                }

                // update transformation and dimension
                new_ItR[3] -= bmin[0];
                new_ItR[7] -= bmin[1];
                new_ItR[11] += bmin[2];
                bmax -= bmin;
                bmax /= voxel.qsdr_reso;
                new_Its = tipl::shape<3>(bmax.begin());
            }

            reg.to_I_space(native_geo,native_trans);
            reg.to_It_space(new_Its,new_ItR);
            affine = reg.T();
            if(t1w_reg)
                affine.accumulate(voxel.other_modality_trans);
            cdm_dis.swap(reg.t2f_dis);
            mapping.swap(reg.to2from);

        }

        // assign mask
        tipl::threshold(reg.It[0],voxel.mask,0.0f);


        // setup voxel data for QSDR
        {
            voxel.qsdr = true;
            voxel.dim = reg.Its;
            voxel.vs = reg.Itvs;
            voxel.trans_to_mni = reg.ItR;
            tipl::out() << "output resolution: " << reg.Itvs[0] << std::endl;
            tipl::out() << "output dimension: " << reg.Its << std::endl;

            // other image
            if(!voxel.other_image.empty())
            {
                for(unsigned int i = 0;i < voxel.other_image.size();++i)
                {
                    if(voxel.other_image[i].empty())
                        continue;
                    tipl::image<3> new_other_image(voxel.dim);
                    tipl::adaptive_par_for(voxel.dim.size(),[&](size_t index)
                    {
                        tipl::vector<3,float> Jpos(mapping[index]);
                        if(voxel.other_image[i].shape() != native_geo)
                            voxel.other_image_trans[i](Jpos);
                        tipl::estimate<tipl::interpolation::cubic>(voxel.other_image[i],Jpos,new_other_image[index]);
                    });
                    tipl::lower_threshold(new_other_image,0);
                    voxel.other_image[i].swap(new_other_image);
                    voxel.other_image_voxel_size[i] = voxel.vs;
                    voxel.other_image_trans[i].identity();
                }
            }
            //if(voxel.needs("vol"))
            jdet.resize(voxel.dim.size());
            // setup raw DWI
            ptr_images.clear();
            for (unsigned int index = 0; index < voxel.dwi_data.size(); ++index)
                ptr_images.push_back(tipl::make_image(voxel.dwi_data[index],native_geo));
        }
    }

    tipl::vector<3,int> mni_to_voxel_index(Voxel& voxel,int x,int y,int z) const
    {
        x = int(voxel.trans_to_mni[3])-x;
        y = int(voxel.trans_to_mni[7])-y;
        z -= int(voxel.trans_to_mni[11]);
        x /= voxel.vs[0];
        y /= voxel.vs[1];
        z /= voxel.vs[2];
        return tipl::vector<3,int>(x,y,z);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        // calculate jacobian
        {
            std::copy_n(affine.data(),9,data.jacobian.begin());
            tipl::pixel_index<3> pos_index(data.voxel_index,voxel.dim);
            if(!cdm_dis.shape().is_edge(pos_index))
            {
                tipl::matrix<3,3,float> M;
                tipl::jacobian_dis_at(cdm_dis,pos_index,M.begin());
                data.jacobian *= M;
            }
        }

        tipl::interpolator::cubic<3> interpolation;
        if(!interpolation.get_location(native_geo,mapping[data.voxel_index]))
        {
            std::fill(data.space.begin(),data.space.end(),0);
            std::fill(data.jacobian.begin(),data.jacobian.end(),0.0);
            return;
        }
        data.space.resize(ptr_images.size());
        for (unsigned int i = 0; i < ptr_images.size(); ++i)
            interpolation.estimate(ptr_images[i],data.space[i]);

        tipl::lower_threshold(data.space.begin(),data.space.end(),0.0f);

        if(!jdet.empty())
            jdet[data.voxel_index] = std::abs(data.jacobian.det());
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        voxel.qsdr = false;
        double sum = 0.0;
        size_t sum_count = 0;
        for(size_t i = 0;i < jdet.size();++i)
            if(jdet[i] != 0.0f)
            {
                sum += jdet[i];
                ++sum_count;
            }
        if(sum != 0.0)
            tipl::multiply_constant(jdet,double(sum_count)/sum);
        for(size_t i = 0;i < jdet.size();++i)
            if(jdet[i] == 0.0f)
                jdet[i] = 1.0f;

        mat_writer.write<tipl::io::masked_sloped>("vol",jdet,voxel.dim.plane_size());
        mat_writer.write("native_dimension",native_geo);
        mat_writer.write("native_voxel_size",native_vs);
    }

};

#endif//MNI_RECONSTRUCTION_HPP
