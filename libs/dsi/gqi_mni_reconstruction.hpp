#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <QFileInfo>
#include <chrono>
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
protected: // for warping other image modality
    std::vector<tipl::image<3> > other_image;
protected:
    std::vector<float> jdet;
protected:
    typedef tipl::const_pointer_image<3,unsigned short> point_image_type;
    std::vector<point_image_type> ptr_images;

public:
    virtual void init(Voxel& voxel)
    {
        tipl::progress prog("QA/ISO normalization");
        if(voxel.vs[0] == 0.0f ||
           voxel.vs[1] == 0.0f ||
           voxel.vs[2] == 0.0f)
            throw std::runtime_error("No spatial information found in src file. Recreate src file or contact developer for assistance");

        native_geo = voxel.dim;
        native_vs = voxel.vs;


        dual_reg<3> reg;
        reg.export_intermediate = voxel.needs("debug");

        if(!reg.load_template(0,fa_template_list[voxel.template_id].c_str()) ||
           !reg.load_template(1,iso_template_list[voxel.template_id].c_str()))
            throw std::runtime_error("cannot load anisotropy/isotropy template");
        voxel.trans_to_mni = reg.ItR;

        reg.load_subject(0,std::move(voxel.qa_map));
        reg.load_subject(1,std::move(voxel.iso_map));
        reg.Ivs = voxel.vs;


        bool t1w_reg = false;
        if(!voxel.other_modality_template.empty())
        {
            tipl::out() << "adding " << voxel.other_modality_template << " as template for registration";
            if(!reg.load_template(2,voxel.other_modality_template.c_str()))
                throw std::runtime_error(std::string("cannot load template: ") + voxel.other_modality_template);

            tipl::out() << "moving QA/ISO to the registration modality space";
            reg.load_subject(2,tipl::image<3>(voxel.other_modality_subject));
            reg.Ivs = voxel.vs = voxel.other_modality_vs;
            for(size_t i = 0;i < 2;++i)
                reg.I[i] = tipl::resample(reg.I[i],voxel.other_modality_subject.shape(),voxel.other_modality_trans);
            t1w_reg = true;
        }

        reg.match_resolution(false);


        // linear registration
        {
            if(voxel.manual_alignment)
                tipl::out() << "manual alignment:" << (reg.arg = voxel.qsdr_arg);
            else
            {
                if(tipl::prog_aborted)
                    throw std::runtime_error("reconstruction canceled");
                if((voxel.R2 = reg.linear_reg()) < 0.4f)
                    tipl::warning() << "poor registration found in linear registration. Please check image quality or orientation. consider using manual alignment.";
            }

            {
                affine = reg.T();
                float VFratio = reg.Ivs[0]/voxel.vs[0]; // if subject data are downsampled, then VFratio=2, 4, 8, ...etc
                if(VFratio != 1.0f)
                    tipl::multiply_constant(affine.data(),affine.data()+12,VFratio);
                if(t1w_reg)
                    affine *= voxel.other_modality_trans;
            }
        }
        // nonlinear registration
        {
            reg.nonlinear_reg(tipl::prog_aborted);
            voxel.R2 = reg.r[1];
            if(tipl::prog_aborted)
                throw std::runtime_error("reconstruction canceled");

            cdm_dis.swap(reg.t2f_dis);

            voxel.R2 = voxel.R2*voxel.R2;
            tipl::out() << "nonlinear R2: " << voxel.R2 << std::endl;
            if(voxel.R2 < 0.3f)
                tipl::warning() << "poor registration found in nonlinear registration. Please check image quality or image orientation";
        }


        // VG: FA TEMPLATE
        // VF: SUBJECT QA
        // VF2: SUBJECT ISO
        auto& VG = reg.It[0];
        auto& VG2 = reg.It[1];
        auto& VGvs = reg.Itvs;
        auto& VF = reg.I[0];
        auto& VF2 = reg.I[1];
        auto& VFvs = reg.Ivs;



        // if subject data is only a fragment of FOV, crop images
        if(voxel.partial_min != voxel.partial_max)
        {
            for(int d = 0;d < 3;++d)
                if(voxel.partial_min[d] > voxel.partial_max[d])
                    std::swap(voxel.partial_min[d],voxel.partial_max[d]);
            tipl::out() << "partial reconstruction" << std::endl;
            tipl::out() << "partial_min: " << voxel.partial_min << std::endl;
            tipl::out() << "partial_max: " << voxel.partial_max << std::endl;
            tipl::vector<3,int> bmin((voxel.partial_min[0]-voxel.trans_to_mni[3])/voxel.trans_to_mni[0],
                                     (voxel.partial_min[1]-voxel.trans_to_mni[7])/voxel.trans_to_mni[5],
                                     (voxel.partial_min[2]-voxel.trans_to_mni[11])/voxel.trans_to_mni[10]);
            tipl::vector<3,int> bmax((voxel.partial_max[0]-voxel.trans_to_mni[3])/voxel.trans_to_mni[0],
                                     (voxel.partial_max[1]-voxel.trans_to_mni[7])/voxel.trans_to_mni[5],
                                     (voxel.partial_max[2]-voxel.trans_to_mni[11])/voxel.trans_to_mni[10]);
            tipl::out() << "bmin: " << bmin << std::endl;
            tipl::out() << "bmax: " << bmax << std::endl;
            for(int i = 0;i < 3;++i)
            {
                if(bmin[i] > bmax[i])
                    std::swap(bmin[i],bmax[i]);
                if(bmin[i] < 0.0f || bmax[i] > VG.shape()[i])
                    throw std::runtime_error("out of bounding box in partial reconstruction.");
            }

            // update cdm_dis
            tipl::crop(cdm_dis,bmin,bmax);

            // add the coordinate shift to the displacement matrix
            tipl::add_constant(cdm_dis,bmin);

            tipl::crop(VG,bmin,bmax);

            // update transformation and dimension
            voxel.trans_to_mni[3] -= bmin[0]*VGvs[0];
            voxel.trans_to_mni[7] -= bmin[1]*VGvs[1];
            voxel.trans_to_mni[11] += bmin[2]*VGvs[2];

        }

        // output resolution = acquisition resolution
        float VG_ratio = voxel.qsdr_reso/VGvs[0];

        // update registration results;
        if(voxel.qsdr_reso != VGvs[0])
        {
            tipl::shape<3> new_geo(uint32_t(float(VG.width())*VGvs[0]/voxel.qsdr_reso),
                                   uint32_t(float(VG.height())*VGvs[0]/voxel.qsdr_reso),
                                   uint32_t(float(VG.depth())*VGvs[0]/voxel.qsdr_reso));
            // update VG,VFFF (for mask) and cdm_dis (for mapping)
            tipl::image<3,unsigned char> new_VG(new_geo);
            tipl::image<3,tipl::vector<3> > new_cdm_dis(new_geo);
            tipl::adaptive_par_for(tipl::begin_index(new_geo),tipl::end_index(new_geo),
                          [&](const tipl::pixel_index<3>& pos)
            {
                tipl::vector<3> p(pos);
                p *= VG_ratio;
                tipl::interpolator::linear<3> interp;
                if(!interp.get_location(VG.shape(),p))
                    return;
                interp.estimate(cdm_dis,new_cdm_dis[pos.index()]);
                // here the displacement values are still in the VGvs resolution
                interp.estimate(VG,new_VG[pos.index()]);
            });
            new_cdm_dis.swap(cdm_dis);
            new_VG.swap(VG);
            VGvs[0] = VGvs[1] = VGvs[2] = voxel.qsdr_reso;
        }

        // assign mask
        {
            voxel.mask.resize(VG.shape());
            for(size_t index = 0;index < VG.size();++index)
                voxel.mask[index] = VG[index] > 0.0f ? 1:0;
        }

        // compute mappings
        {
            mapping.resize(cdm_dis.shape());
            tipl::adaptive_par_for(tipl::begin_index(cdm_dis.shape()),tipl::end_index(cdm_dis.shape()),
            [&](const tipl::pixel_index<3>& pos)
            {
                tipl::vector<3> Jpos(pos);
                if(VG_ratio != 1.0f) // if upsampled due to subject high resolution
                    Jpos *= VG_ratio;
                Jpos += cdm_dis[pos.index()]; // VFF space
                affine(Jpos);// VFF to VF space
                mapping[pos.index()] = Jpos;
            });
        }


        // setup voxel data for QSDR
        {
            voxel.qsdr = true;
            voxel.dim = VG.shape();
            voxel.vs = VGvs;
            voxel.trans_to_mni[0] = -VGvs[0];
            voxel.trans_to_mni[5] = -VGvs[1];
            voxel.trans_to_mni[10] = VGvs[2];
            tipl::out() << "output resolution: " << VGvs[0] << std::endl;
            tipl::out() << "output dimension: " << VG.shape() << std::endl;

            // other image
            if(!voxel.other_image.empty())
            {
                other_image.resize(voxel.other_image.size());
                for(unsigned int i = 0;i < voxel.other_image.size();++i)
                {
                    other_image[i].resize(VG.shape());
                    tipl::adaptive_par_for(voxel.mask.size(),[&](size_t index)
                    {
                        tipl::vector<3,float> Jpos(mapping[index]);
                        if(voxel.other_image[i].shape() != native_geo)
                            voxel.other_image_trans[i](Jpos);
                        tipl::estimate<tipl::interpolation::cubic>(voxel.other_image[i],Jpos,other_image[i][index]);
                    });
                    tipl::lower_threshold(other_image[i],0);
                }
            }
            if(voxel.needs("jdet"))
                jdet.resize(VG.size());
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
            std::copy(affine.data(),affine.data()+9,data.jacobian.begin());
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
        mat_writer.write<tipl::io::masked_sloped>("jdet",jdet,voxel.dim.plane_size());
        mat_writer.write("native_dimension",native_geo);
        mat_writer.write("native_voxel_size",native_vs);
        for(unsigned int index = 0;index < other_image.size();++index)
            mat_writer.write<tipl::io::masked_sloped>(voxel.other_image_name[index],other_image[index].data(),other_image[index].plane_size(),other_image[index].depth());
        mat_writer.write("trans",voxel.trans_to_mni);
        mat_writer.write("template",std::filesystem::path(fa_template_list[voxel.template_id]).stem().stem().stem().string());
        mat_writer.write("R2",std::vector<float>({voxel.R2}));
    }

};

#endif//MNI_RECONSTRUCTION_HPP
