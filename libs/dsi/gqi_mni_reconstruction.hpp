#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <QFileInfo>
#include <chrono>
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "gqi_process.hpp"
void animal_reg(const tipl::image<float,3>& from,tipl::vector<3> from_vs,
          const tipl::image<float,3>& to,tipl::vector<3> to_vs,
          tipl::transformation_matrix<double>& T,bool& terminated);
class DWINormalization  : public BaseProcess
{
protected:
    tipl::geometry<3> src_geo;
    tipl::geometry<3> des_geo;
protected:
    tipl::image<tipl::vector<3>,3> cdm_dis,mapping;
protected:
    tipl::transformation_matrix<double> affine;
    float affine_volume_scale;
protected: // for warping other image modality
    std::vector<tipl::image<float,3> > other_image;
protected:
    std::vector<float> jdet;
protected:
    typedef tipl::const_pointer_image<unsigned short,3> point_image_type;
    std::vector<point_image_type> ptr_images;

public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.vs[0] == 0.0f ||
           voxel.vs[1] == 0.0f ||
           voxel.vs[2] == 0.0f)
            throw std::runtime_error("No spatial information found in src file. Recreate src file or contact developer for assistance");

        tipl::image<float,3> VG,VF(voxel.qa_map),VG2,VF2;
        tipl::vector<3> VGvs;
        tipl::vector<3> VGshift;
        if(voxel.primary_template.empty())
            throw std::runtime_error("Invalid external template");
        {
            voxel.step_report << "[Step T2b(1)][Template]=" <<
                                 QFileInfo(voxel.primary_template.c_str()).baseName().toStdString() << std::endl;
            gz_nifti read;
            if(read.load_from_file(voxel.primary_template.c_str()))
            {
                read.toLPS(VG);
                read.get_voxel_size(VGvs);
                tipl::matrix<4,4,float> tran;
                read.get_image_transformation(tran);
                VGshift[0] = tran[3];
                VGshift[1] = tran[7];
                VGshift[2] = tran[11];
            }
        }
        if(!voxel.secondary_template.empty())
        {
            gz_nifti read2;
            if(read2.load_from_file(voxel.secondary_template.c_str()))
            {
                read2.toLPS(VG2);
                if(voxel.secondary_template.find("ISO.nii.gz") != std::string::npos)
                    VF2.swap(voxel.iso_map);
            }
        }


        {
            float best_reso = *std::min_element(voxel.vs.begin(),voxel.vs.end());
            // upsample template
            while(best_reso < VGvs[0]*0.5f)
            {
                tipl::upsampling(VG);
                if(!VG2.empty())
                    tipl::upsampling(VG2);
                VGvs *= 0.5f;
            }
            // downsample template
            while(best_reso > VGvs[0]*1.5f)
            {
                tipl::downsampling(VG);
                if(!VG2.empty())
                    tipl::downsampling(VG2);
                VGvs *= 2.0f;
            }
            // setup output bounding box
            {
                des_geo = VG.geometry();
                // setup transformation matrix
                std::fill(voxel.trans_to_mni,voxel.trans_to_mni+16,0.0);
                voxel.trans_to_mni[15] = 1.0;
                voxel.trans_to_mni[0] = -VGvs[0];
                voxel.trans_to_mni[5] = -VGvs[1];
                voxel.trans_to_mni[10] = VGvs[2];
                voxel.trans_to_mni[3] = VGshift[0];
                voxel.trans_to_mni[7] = VGshift[1];
                voxel.trans_to_mni[11] = VGshift[2];
            }
        }



        bool export_intermediate = false;
        // bookkepping for restoration
        src_geo = voxel.dim;

        affine_volume_scale = (voxel.vs[0]*voxel.vs[1]*voxel.vs[2]/VGvs[0]/VGvs[1]/VGvs[2]);

        {

            tipl::normalize(VG,1.0);
            tipl::normalize(VF,1.0);
            if(!VF2.empty())
                tipl::normalize(VF2,1.0);

            tipl::image<float,3> VFF,VFF2;
            {
                // VG: FA TEMPLATE
                // VF: SUBJECT QA
                // VF2: SUBJECT ISO
                if(export_intermediate)
                {
                    VG.save_to_file<gz_nifti>("Template_QA.nii.gz");
                    VF.save_to_file<gz_nifti>("Subject_QA.nii.gz");
                    if(!VF2.empty())
                        VF2.save_to_file<gz_nifti>("Subject_ISO.nii.gz");
                }

                if(voxel.qsdr_trans.data[0] != 0.0) // has manual reg data
                    affine = voxel.qsdr_trans;
                else
                {
                    bool terminated = false;
                    if(!run_prog("Linear Registration",[&](){
                        if(VGvs[0] < 1.0f) // animal recon
                            animal_reg(VG,VGvs,VF,voxel.vs,affine,terminated);
                        else
                            tipl::reg::two_way_linear_mr(VG,VGvs,VF,voxel.vs,affine,
                                    tipl::reg::affine,tipl::reg::correlation(),terminated,voxel.thread_count);
                    },terminated))
                        throw std::runtime_error("Reconstruction canceled");
                }
                VFF.resize(VG.geometry());
                tipl::resample(VF,VFF,affine,tipl::cubic);
                if(!VF2.empty())
                {
                    VFF2.resize(VG.geometry());
                    tipl::resample(VF2,VFF2,affine,tipl::cubic);
                }
            }

            if(export_intermediate)
                VFF.save_to_file<gz_nifti>("Subject_QA_linear_reg.nii.gz");

            tipl::reg::cdm_pre(VG,VG2,VFF,VFF2);

            bool terminated = false;

            if(!run_prog("Normalization",[&]()
                {
                    if(!VFF2.empty())
                    {
                        std::cout << "normalization using dual QA/ISO templates" << std::endl;
                        tipl::reg::cdm2(VG,VG2,VFF,VFF2,cdm_dis,terminated);
                    }
                    else
                        tipl::reg::cdm(VG,VFF,cdm_dis,terminated);
                },terminated))
                throw std::runtime_error("Reconstruction canceled");

            {
                tipl::image<float,3> VFFF;
                tipl::compose_displacement(VFF,cdm_dis,VFFF);
                float r = float(tipl::correlation(VG.begin(),VG.end(),VFFF.begin()));
                voxel.R2 = r*r;
                std::cout << "R2=" << voxel.R2 << std::endl;
            }

        }
        voxel.dim = des_geo;
        voxel.mask.resize(des_geo);
        std::fill(voxel.mask.begin(),voxel.mask.end(),0);
        for(size_t index = 0;index < des_geo.size();++index)
            if(VG[index] > 0.0f)
                voxel.mask[index] = 1;
        for(int i = 0;i < 5;++i)
            tipl::morphology::smoothing_fill(voxel.mask);

        ptr_images.clear();
        for (unsigned int index = 0; index < voxel.dwi_data.size(); ++index)
            ptr_images.push_back(tipl::make_image(voxel.dwi_data[index],src_geo));

        voxel.vs = VGvs;
        if(QFileInfo(voxel.primary_template.c_str()).baseName().contains("HCP")) // if default template is used
        {
            voxel.csf_pos1 = mni_to_voxel_index(voxel,6,0,18);
            voxel.csf_pos2 = mni_to_voxel_index(voxel,-6,0,18);
            voxel.csf_pos3 = mni_to_voxel_index(voxel,4,18,10);
            voxel.csf_pos4 = mni_to_voxel_index(voxel,-4,18,10);
        }
        else
        {
            voxel.csf_pos1 = voxel.csf_pos2 = voxel.csf_pos3 = voxel.csf_pos4 = tipl::vector<3,int>(0,0,0);
        }
        // output jacobian
        jdet.resize(voxel.dim.size());

        // compute mappings
        mapping.resize(voxel.dim);
        mapping.for_each_mt([&](tipl::vector<3>& Jpos,tipl::pixel_index<3> pos)
        {
            Jpos = pos;
            Jpos += cdm_dis[pos.index()];
            affine(Jpos);
        });

        // other image
        if(!voxel.other_image.empty())
        {
            other_image.resize(voxel.other_image.size());
            for(unsigned int i = 0;i < voxel.other_image.size();++i)
            {
                other_image[i].resize(des_geo);
                tipl::par_for(voxel.mask.size(),[&](size_t index)
                {
                    tipl::vector<3,float> Jpos(mapping[index]);
                    if(voxel.other_image[i].geometry() != src_geo)
                        voxel.other_image_trans[i](Jpos);
                    tipl::estimate(voxel.other_image[i],Jpos,other_image[i][index]);
                });
            }
        }
        voxel.qsdr = true;
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
    template<class interpolation_type>
    void interpolate_dwi(Voxel& voxel, VoxelData& data,const tipl::vector<3,float>& Jpos,interpolation_type)
    {
        interpolation_type interpolation;

        if(!interpolation.get_location(src_geo,Jpos))
        {
            std::fill(data.space.begin(),data.space.end(),0);
            std::fill(data.jacobian.begin(),data.jacobian.end(),0.0);
            return;
        }
        data.space.resize(ptr_images.size());
        for (unsigned int i = 0; i < ptr_images.size(); ++i)
            interpolation.estimate(ptr_images[i],data.space[i]);

        if(!voxel.grad_dev.empty())
        {
            tipl::matrix<3,3,float> grad_dev,new_j;
            for(unsigned int i = 0; i < 9; ++i)
                interpolation.estimate(voxel.grad_dev[i],grad_dev[i]);
            tipl::mat::transpose(grad_dev.begin(),tipl::dim<3,3>());
            new_j = grad_dev*data.jacobian;
            data.jacobian = new_j;
        }
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        // calculate jacobian
        {
            std::copy(affine.get(),affine.get()+9,data.jacobian.begin());
            tipl::pixel_index<3> pos_index(data.voxel_index,voxel.dim);
            if(!cdm_dis.geometry().is_edge(pos_index))
            {
                tipl::matrix<3,3,float> M;
                tipl::jacobian_dis_at(cdm_dis,pos_index,M.begin());
                data.jacobian *= M;
            }
        }

        interpolate_dwi(voxel,data,mapping[data.voxel_index],tipl::cubic_interpolation<3>());
        jdet[data.voxel_index] = std::abs(data.jacobian.det()*affine_volume_scale);
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        voxel.qsdr = false;
        mat_writer.write("jdet",jdet,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("native_dimension",&src_geo[0],1,3);
        mat_writer.write("mapping",&mapping[0][0],3,mapping.size());
        for(unsigned int index = 0;index < other_image.size();++index)
        {
            mat_writer.write(voxel.other_image_name[index].c_str(),other_image[index]);
            mat_writer.write((voxel.other_image_name[index]+"_dimension").c_str(),&voxel.other_image[index].geometry()[0],1,3);
            mat_writer.write((voxel.other_image_name[index]+"_trans").c_str(),voxel.other_image_trans[index].get(),1,12);
        }
        mat_writer.write("trans",voxel.trans_to_mni,4,4);
        mat_writer.write("R2",&voxel.R2,1,1);
    }

};

class EstimateZ0_MNI : public BaseProcess
{
    std::vector<float> samples;
    std::mutex mutex;
public:
    void init(Voxel& voxel)
    {
        voxel.z0 = 0.0;
        samples.reserve(20);
    }
    void run(Voxel& voxel, VoxelData& data)
    {
        // perform csf cross-subject normalization
        if(voxel.csf_pos1 != tipl::vector<3,int>(0,0,0))
        {
            tipl::vector<3,int> cur_pos(tipl::pixel_index<3>(data.voxel_index,voxel.dim));
            if((cur_pos-voxel.csf_pos1).length() <= 1.0 || (cur_pos-voxel.csf_pos2).length() <= 1.0 ||
               (cur_pos-voxel.csf_pos3).length() <= 1.0 || (cur_pos-voxel.csf_pos4).length() <= 1.0)
            {
                std::lock_guard<std::mutex> lock(mutex);
                if(voxel.r2_weighted) // multishell GQI2 gives negative ODF, use b0 as the scaling reference
                    samples.push_back(data.space[0]);
                else
                    samples.push_back(*std::min_element(data.odf.begin(),data.odf.end()));
                //std::fill(data.odf.begin(),data.odf.end(),0.0f);
            }
        }
        else
        // if other template is used
        {
            voxel.z0 = std::max<float>(voxel.z0,*std::min_element(data.odf.begin(),data.odf.end()));
        }
    }
    void end(Voxel& voxel,gz_mat_write&)
    {
        if(!samples.empty())
            voxel.z0 = tipl::median(samples.begin(),samples.end());
        if(voxel.z0 == 0.0f)
            voxel.z0 = 1.0f;
    }

};

#endif//MNI_RECONSTRUCTION_HPP
