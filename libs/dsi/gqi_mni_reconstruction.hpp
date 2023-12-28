#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <QFileInfo>
#include <chrono>
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "gqi_process.hpp"
#include "reg.hpp"
extern std::vector<std::string> fa_template_list,iso_template_list;
void match_template_resolution(tipl::image<3>& VG,
                               tipl::image<3>& VG2,
                               tipl::vector<3>& VGvs,
                               tipl::image<3>& VF,
                               tipl::image<3>& VF2,
                               tipl::vector<3>& VFvs);
class DWINormalization  : public BaseProcess
{
protected:
    tipl::shape<3> src_geo;
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

        // bookkeeping for restoration
        src_geo = voxel.dim;


        // VG: FA TEMPLATE
        // VF: SUBJECT QA
        // VF2: SUBJECT ISO
        tipl::image<3> VG,VF(voxel.qa_map),VG2,VF2;
        tipl::vector<3> VGvs, VFvs(voxel.vs);


        bool is_human_template = QFileInfo(fa_template_list[voxel.template_id].c_str()).baseName().contains("ICBM");
        bool export_intermediate = voxel.needs("debug");
        bool dual_modality = false;

        if(fa_template_list[voxel.template_id].empty())
            throw std::runtime_error("Invalid external template");
        {
            tipl::io::gz_nifti read;
            if(!read.load_from_file(fa_template_list[voxel.template_id].c_str()))
                throw std::runtime_error("Cannot load template");

            read.toLPS(VG);
            read.get_voxel_size(VGvs);
            read.get_image_transformation(voxel.trans_to_mni);
        }
        if(!iso_template_list[voxel.template_id].empty())
        {
            tipl::io::gz_nifti read2;
            if(read2.load_from_file(iso_template_list[voxel.template_id].c_str()))
            {
                read2.toLPS(VG2);
                VF2.swap(voxel.iso_map);
                dual_modality = true;
            }
        }



        {
            match_template_resolution(VG,VG2,VGvs,VF,VF2,VFvs);
            tipl::normalize(VG);
            tipl::normalize(VF);
            if(dual_modality)
                tipl::normalize(VF2);

            tipl::filter::gaussian(VF);
            if(dual_modality)
                tipl::filter::gaussian(VF2);

            if(export_intermediate)
            {
                tipl::io::gz_nifti::save_to_file("Template_QA.nii.gz",VG,VGvs,voxel.trans_to_mni);
                if(!VG2.empty())
                    tipl::io::gz_nifti::save_to_file("Template_ISO.nii.gz",VG2,VGvs,voxel.trans_to_mni);
                tipl::matrix<4,4> trans = {-VFvs[0],0.0f,0.0f,0.0f,
                                           0.0f,-VFvs[1],0.0f,0.0f,
                                           0.0f,0.0f,VFvs[2],0.0f,
                                           0.0f,0.0f,0.0f,1.0f};
                tipl::io::gz_nifti::save_to_file("Subject_QA.nii.gz",VF,VFvs,trans);
                if(!VF2.empty())
                    tipl::io::gz_nifti::save_to_file("Subject_ISO.nii.gz",VF2,VFvs,trans);
            }

            if(voxel.manual_alignment)
            {
                tipl::out() << "manual alignment";
                tipl::out() << (affine = voxel.qsdr_trans);
            }
            else
            {
                bool terminated = false;
                if(!tipl::run("linear registration",[&]()
                {
                    linear_with_mi(VG,VGvs,VF,VFvs,affine,tipl::reg::affine,terminated);
                },terminated))
                    throw std::runtime_error("reconstruction canceled");
            }

            tipl::image<3> VFF(VG.shape()),VFF2;
            tipl::resample_mt<tipl::interpolation::cubic>(VF,VFF,affine);
            float linear_r = tipl::correlation(VFF.begin(),VFF.end(),VG.begin());
            linear_r = linear_r*linear_r;
            tipl::out() << "linear R2: " << linear_r << std::endl;
            if(linear_r < 0.3f)
                throw std::runtime_error("ERROR: Poor R2 found in linear registration. Please check image orientation or use manual alignment.");

            if(dual_modality)
            {
                VFF2.resize(VG.shape());
                tipl::resample_mt<tipl::interpolation::cubic>(VF2,VFF2,affine);
            }

            if(export_intermediate)
            {
                tipl::io::gz_nifti::save_to_file("Subject_QA_linear_reg.nii.gz",VFF,VGvs,voxel.trans_to_mni);
                if(dual_modality)
                    tipl::io::gz_nifti::save_to_file("Subject_ISO_linear_reg.nii.gz",VFF2,VGvs,voxel.trans_to_mni);
            }

            tipl::reg::cdm_pre(VG,VG2,VFF,VFF2);

            bool terminated = false;

            if(!tipl::run("normalization",[&]()
                {
                    tipl::reg::cdm_param param;
                    tipl::image<3,tipl::vector<3> > cdm_dis_inv;
                    cdm_common(VG,VG2,VFF,VFF2,cdm_dis,cdm_dis_inv,terminated,param);
                    if(export_intermediate)
                    {
                        tipl::image<4> buffer(tipl::shape<4>(VG.width(),VG.height(),VG.depth(),6));
                        tipl::par_for(6,[&](unsigned int d)
                        {
                            if(d < 3)
                            {
                                size_t shift = d*VG.size();
                                for(size_t i = 0;i < VG.size();++i)
                                    buffer[i+shift] = cdm_dis_inv[i][d];
                            }
                            else
                            {
                                size_t shift = d*VG.size();
                                d -= 3;
                                for(size_t i = 0;i < VG.size();++i)
                                    buffer[i+shift] = cdm_dis[i][d];
                            }
                        });
                        tipl::io::gz_nifti::save_to_file("Subject_displacement.nii.gz",buffer,voxel.vs,voxel.trans_to_mni);
                    }
                },terminated))
                throw std::runtime_error("reconstruction canceled");

            tipl::image<3> VFFF;
            tipl::compose_displacement(VFF,cdm_dis,VFFF);

            float r = float(tipl::correlation(VG.begin(),VG.end(),VFFF.begin()));
            voxel.R2 = r*r;
            tipl::out() << "linear+nonlinear R2: " << voxel.R2 << std::endl;
            if(voxel.R2 < 0.3f)
                throw std::runtime_error("ERROR: Poor R2 found. Please check image orientation or use manual alignment.");

            if(export_intermediate)
                VFFF.save_to_file<tipl::io::gz_nifti>("Subject_QA_nonlinear_reg.nii.gz");

            // check if partial reconstruction
            size_t total_voxel_count = 0;
            size_t subject_voxel_count = 0;
            for(size_t index = 0;index < VG.size();++index)
            {
                if(VG[index] > 0.0f)
                {
                    ++total_voxel_count;
                    if(VFFF[index] > 0.0f)
                        ++subject_voxel_count;
                }
            }

            float VFratio = VFvs[0]/voxel.vs[0]; // if subject data are downsampled, then VFratio=2, 4, 8, ...etc
            if(VFratio != 1.0f)
                tipl::multiply_constant(affine.data,affine.data+12,VFratio);

        }       
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
            tipl::image<3> new_VG(new_geo);
            tipl::image<3,tipl::vector<3> > new_cdm_dis(new_geo);
            tipl::par_for(tipl::begin_index(new_geo),tipl::end_index(new_geo),
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
            tipl::par_for(tipl::begin_index(cdm_dis.shape()),tipl::end_index(cdm_dis.shape()),
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


            if(is_human_template && voxel.partial_min == voxel.partial_max) // if default template is used
            {
                voxel.csf_pos1 = mni_to_voxel_index(voxel,6,0,18);
                voxel.csf_pos2 = mni_to_voxel_index(voxel,-6,0,18);
                voxel.csf_pos3 = mni_to_voxel_index(voxel,4,18,10);
                voxel.csf_pos4 = mni_to_voxel_index(voxel,-4,18,10);
            }
            // other image
            if(!voxel.other_image.empty())
            {
                other_image.resize(voxel.other_image.size());
                for(unsigned int i = 0;i < voxel.other_image.size();++i)
                {
                    other_image[i].resize(VG.shape());
                    tipl::par_for(voxel.mask.size(),[&](size_t index)
                    {
                        tipl::vector<3,float> Jpos(mapping[index]);
                        if(voxel.other_image[i].shape() != src_geo)
                            voxel.other_image_trans[i](Jpos);
                        tipl::estimate<tipl::interpolation::cubic>(voxel.other_image[i],Jpos,other_image[i][index]);
                    });
                }
            }
            if(voxel.needs("jdet"))
                jdet.resize(VG.size());
            // setup raw DWI
            ptr_images.clear();
            for (unsigned int index = 0; index < voxel.dwi_data.size(); ++index)
                ptr_images.push_back(tipl::make_image(voxel.dwi_data[index],src_geo));
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
            std::copy(affine.data,affine.data+9,data.jacobian.begin());
            tipl::pixel_index<3> pos_index(data.voxel_index,voxel.dim);
            if(!cdm_dis.shape().is_edge(pos_index))
            {
                tipl::matrix<3,3,float> M;
                tipl::jacobian_dis_at(cdm_dis,pos_index,M.begin());
                data.jacobian *= M;
            }
        }

        tipl::interpolator::cubic<3> interpolation;
        if(!interpolation.get_location(src_geo,mapping[data.voxel_index]))
        {
            std::fill(data.space.begin(),data.space.end(),0);
            std::fill(data.jacobian.begin(),data.jacobian.end(),0.0);
            return;
        }
        data.space.resize(ptr_images.size());
        for (unsigned int i = 0; i < ptr_images.size(); ++i)
            interpolation.estimate(ptr_images[i],data.space[i]);

        tipl::lower_threshold(data.space,0.0f);

        if(!jdet.empty())
            jdet[data.voxel_index] = std::abs(data.jacobian.det());
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        voxel.qsdr = false;
        mat_writer.write("jdet",jdet,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("native_dimension",src_geo);
        mat_writer.write("native_voxel_size",voxel.vs);
        mat_writer.write("mapping",&mapping[0][0],3,mapping.size());

        // allow loading native space t1w-based ROI
        for(unsigned int index = 0;index < other_image.size();++index)
        {
            mat_writer.write(voxel.other_image_name[index].c_str(),other_image[index]);
            mat_writer.write((voxel.other_image_name[index]+"_dimension").c_str(),voxel.other_image[index].shape());
            mat_writer.write((voxel.other_image_name[index]+"_trans").c_str(),voxel.other_image_trans[index]);
        }
        mat_writer.write("trans",voxel.trans_to_mni.begin(),4,4);
        mat_writer.write("R2",&voxel.R2,1,1);
    }

};

#endif//MNI_RECONSTRUCTION_HPP
