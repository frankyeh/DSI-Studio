#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <chrono>
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "odf_decomposition.hpp"
#include "odf_deconvolusion.hpp"
#include "gqi_process.hpp"

class DWINormalization  : public BaseProcess
{
protected:
    tipl::geometry<3> src_geo;
    tipl::geometry<3> des_geo;
protected:
    std::auto_ptr<tipl::reg::bfnorm_mapping<double,3> > mni;
protected:
    tipl::image<tipl::vector<3>,3> cdm_dis;
protected:
    tipl::transformation_matrix<double> affine;
    float affine_volume_scale;
    float resolution_ratio; // output resolution
protected: // for warping other image modality
    std::vector<tipl::image<float,3> > other_image,other_image_x,other_image_y,other_image_z;
protected:
    std::vector<float> jdet;
    std::vector<float> mx,my,mz;
protected:
    typedef tipl::const_pointer_image<unsigned short,3> point_image_type;
    std::vector<point_image_type> ptr_images;

public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.vs[0] == 0.0 ||
                voxel.vs[1] == 0.0 ||
                voxel.vs[2] == 0.0)
            throw std::runtime_error("No spatial information found in src file. Recreate src file or contact developer for assistance");

        tipl::image<float,3> VG,VF(voxel.qa_map);
        tipl::vector<3> VGvs;
        tipl::vector<3> VGshift;
        if(voxel.external_template.empty())
            throw std::runtime_error("Invalid external template");
        {
            std::cout << "loading template:" << voxel.external_template << std::endl;
            gz_nifti read;
            if(read.load_from_file(voxel.external_template.c_str()))
            {
                read.toLPS(VG);
                read.get_voxel_size(VGvs);
                float tran[16];
                read.get_image_transformation(tran);
                VGshift[0] = tran[3];
                VGshift[1] = tran[7];
                VGshift[2] = tran[11];
            }
        }
        // check if the template has similar size as the data
        {
            float vG = VGvs[0]*VGvs[0]*VGvs[0]*tipl::segmentation::otsu_count(VG);
            float vF = voxel.vs[0]*voxel.vs[1]*voxel.vs[2]*tipl::segmentation::otsu_count(VF);
            float ratio = vF/(vG+1.0f);
            if(ratio < 0.4 || ratio > 2.0)
                throw std::runtime_error("Template/Subject FOV mismatch: Please check if the template is valid.");
        }

        resolution_ratio = std::round((voxel.vs[0]+voxel.vs[1]+voxel.vs[2])/3.0f/VGvs[0]);
        if(resolution_ratio < 1.0f)
            resolution_ratio = 1.0f;
        if(resolution_ratio > 2.0f)
            resolution_ratio = 2.0f;

        // setup output bounding box
        {
            des_geo[0] = std::ceil((VG.width()-1)/resolution_ratio+1);
            des_geo[1] = std::ceil((VG.height()-1)/resolution_ratio+1);
            des_geo[2] = std::ceil((VG.depth()-1)/resolution_ratio+1);
            // setup transformation matrix
            std::fill(voxel.trans_to_mni,voxel.trans_to_mni+16,0.0);
            voxel.trans_to_mni[15] = 1.0;
            voxel.trans_to_mni[0] = -VGvs[0]*resolution_ratio;
            voxel.trans_to_mni[5] = -VGvs[1]*resolution_ratio;
            voxel.trans_to_mni[10] = VGvs[2]*resolution_ratio;
            voxel.trans_to_mni[3] = VGshift[0];
            voxel.trans_to_mni[7] = VGshift[1];
            voxel.trans_to_mni[11] = VGshift[2];
        }

        bool export_intermediate = false;
        begin_prog("normalization");
        src_geo = voxel.dim;

        affine_volume_scale = (voxel.vs[0]*voxel.vs[1]*voxel.vs[2]/VGvs[0]/VGvs[1]/VGvs[2]);

        if(voxel.reg_method == 4 && !voxel.t1w.empty()) //CDM
        {
            int prog = 0;
            // calculate the space shift between DWI and T1W
            tipl::vector<3> from(VGshift),to;
            from[0] -= (int)VG.width()+voxel.t1wt_tran[3]-(int)voxel.t1wt.width();
            from[1] -= (int)VG.height()+voxel.t1wt_tran[7]-(int)voxel.t1wt.height();
            from[2] -= voxel.t1wt_tran[11];
            to = from;
            to += tipl::vector<3>(VG.geometry());

            tipl::normalize(voxel.t1w,1.0);
            tipl::normalize(voxel.t1wt,1.0);

            tipl::thread thread1,thread2;
            tipl::transformation_matrix<double> reg1T,reg2T;
            thread1.run([&](){
                if(export_intermediate)
                {
                    tipl::flip_xy(voxel.qa_map);
                    gz_nifti nii;
                    nii.set_voxel_size(voxel.vs);
                    nii << voxel.qa_map;
                    nii.save_to_file("b0.nii.gz");
                    tipl::flip_xy(voxel.qa_map);
                }
                tipl::reg::two_way_linear_mr(voxel.t1w,voxel.t1w_vs,voxel.qa_map,voxel.vs,
                               reg1T,tipl::reg::rigid_body,tipl::reg::mutual_information(),
                                thread1.terminated,voxel.thread_count);
            });
            thread2.run([&](){
                prog = 1;
                tipl::reg::two_way_linear_mr(voxel.t1wt,voxel.t1wt_vs,voxel.t1w,voxel.t1w_vs,
                               reg2T,tipl::reg::affine,tipl::reg::mutual_information(),
                               thread2.terminated,voxel.thread_count);
                tipl::image<float,3> J(voxel.t1wt.geometry());
                tipl::resample_mt(voxel.t1w,J,reg2T,tipl::cubic);

                prog = 3;

                {
                    tipl::image<float,3> Is(J),It(voxel.t1wt);
                    tipl::filter::gaussian(Is);
                    if(export_intermediate)
                    {
                        It.save_to_file<gz_nifti>("It.nii.gz");
                        Is.save_to_file<gz_nifti>("Is.nii.gz");
                    }
                    float resolution = 2.0f;
                    float smoothness = 0.5f;
                    voxel.R2 = tipl::reg::cdm(It,Is,cdm_dis,thread2.terminated,resolution,smoothness);
                    voxel.R2 *= voxel.R2;
                }
                tipl::compose_displacement(J,cdm_dis,voxel.t1w);
                // From T1W template space to FA template space
                tipl::crop(voxel.t1w,from,to);
                tipl::crop(cdm_dis,from,to);

                if(export_intermediate)
                {
                    J.save_to_file<gz_nifti>("J.nii.gz");
                    voxel.t1w.save_to_file<gz_nifti>("cdm_JJ.nii.gz");
                }
                prog = 4;
            });
            begin_prog("Normalization");
            for(;check_prog(prog,4);)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            thread1.wait();
            thread2.wait();

            affine.sr[0] = 1.0;
            affine.sr[4] = 1.0;
            affine.sr[8] = 1.0;
            affine.shift[0] = from[0];
            affine.shift[1] = from[1];
            affine.shift[2] = from[2];
            // affine = FA template -> T1W template
            affine += reg2T;
            // affine = FA template -> T1W template -> Subject T1W
            affine += reg1T;
            // affine = FA template -> T1W template -> Subject T1W -> DWI

            if(export_intermediate)
            {
                tipl::image<float,3> b0J(VG.geometry());
                tipl::resample_mt(voxel.qa_map,b0J,affine,tipl::cubic);
                b0J.save_to_file<gz_nifti>("b0j.nii.gz");
            }
            goto end_normalization;
        }
        {

        tipl::filter::gaussian(VF);
        VF -= tipl::segmentation::otsu_threshold(VF);
        tipl::lower_threshold(VF,0.0);
        tipl::normalize(VG,1.0);
        tipl::normalize(VF,1.0);

        tipl::image<float,3> VFF;
        {
            begin_prog("linear registration");

            // VG: FA TEMPLATE
            // VF: SUBJECT QA
            if(export_intermediate)
            {
                VG.save_to_file<gz_nifti>("Template_QA.nii.gz");
                VF.save_to_file<gz_nifti>("Subject_QA.nii.gz");
            }

            if(voxel.qsdr_trans.data[0] != 0.0) // has manual reg data
                affine = voxel.qsdr_trans;
            else
            {
                bool terminated = false;
                tipl::reg::two_way_linear_mr(VG,VGvs,VF,voxel.vs,affine,
                    tipl::reg::affine,tipl::reg::correlation(),terminated,voxel.thread_count);
            }
            VFF.resize(VG.geometry());
            tipl::resample(VF,VFF,affine,tipl::cubic);
            if(prog_aborted())
                throw std::runtime_error("Reconstruction canceled");

        }
        //linear regression
        tipl::match_signal(VG,VFF);


        if(export_intermediate)
            VFF.save_to_file<gz_nifti>("Subject_QA_linear_reg.nii.gz");


        try
        {
            begin_prog("normalization");
            terminated_class ter(64);
            int factor = voxel.reg_method + 1;

            {
                if(factor <= 3)
                {
                    mni.reset(new tipl::reg::bfnorm_mapping<double,3>(VG.geometry(),tipl::geometry<3>(factor*7,factor*9,factor*7)));
                    tipl::reg::bfnorm(*mni.get(),VG,VFF,ter,voxel.thread_count);
                    voxel.R2 = -tipl::reg::correlation()(VG,VFF,(*mni.get()));
                    if(export_intermediate)
                    {
                        tipl::image<float,3> VFFF(VG.geometry());
                        tipl::resample(VFF,VFFF,*mni.get(),tipl::cubic);
                        VFFF.save_to_file<gz_nifti>("Subject_QA_nonlinear_reg.nii.gz");
                    }
                }
                else
                {
                    bool terminated = false;
                    tipl::reg::cdm(VG,VFF,cdm_dis,terminated,2.0*resolution_ratio,0.5);
                    tipl::image<float,3> VFFF;
                    tipl::compose_displacement(VFF,cdm_dis,VFFF);
                    float r = tipl::correlation(VG.begin(),VG.end(),VFFF.begin());
                    voxel.R2 = r*r;
                }
                std::cout << "R2=" << voxel.R2 << std::endl;

            }
        }
        catch(...)
        {
            throw std::runtime_error("Registration failed due to memory insufficiency. Please try norm 7-9-7 or 14-18-14.");
        }

        }
        end_normalization:



        voxel.dim = des_geo;
        voxel.mask.resize(des_geo);
        std::fill(voxel.mask.begin(),voxel.mask.end(),0);
        for(tipl::pixel_index<3> index(des_geo);index < des_geo.size();++index)
        {
            tipl::vector<3,float> mni_pos(index);
            mni_pos *= resolution_ratio;
            mni_pos.round();
            if(VG.geometry().is_valid(mni_pos) &&
                    VG.at(mni_pos[0],mni_pos[1],mni_pos[2]) > 0.0)
                voxel.mask[index.index()] = 1;
        }

        // other image
        if(!voxel.other_image.empty())
        {
            other_image.resize(voxel.other_image.size());
            if(voxel.output_mapping)
            {
                other_image_x.resize(voxel.other_image.size());
                other_image_y.resize(voxel.other_image.size());
                other_image_z.resize(voxel.other_image.size());
            }
            for(unsigned int index = 0;index < voxel.other_image.size();++index)
            {
                other_image[index].resize(des_geo);
                if(voxel.output_mapping)
                {
                    other_image_x[index].resize(des_geo);
                    other_image_y[index].resize(des_geo);
                    other_image_z[index].resize(des_geo);
                }
            }
        }

        ptr_images.clear();
        for (unsigned int index = 0; index < voxel.dwi_data.size(); ++index)
            ptr_images.push_back(tipl::make_image(voxel.dwi_data[index],src_geo));


        std::fill(voxel.vs.begin(),voxel.vs.end(),VGvs[0]*resolution_ratio);

        if(VG.geometry() == tipl::geometry<3>(157,189,136)) // if default template is used
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
        // output mapping
        if(voxel.output_jacobian)
            jdet.resize(voxel.dim.size());

        if(voxel.output_mapping)
        {
            mx.resize(voxel.dim.size());
            my.resize(voxel.dim.size());
            mz.resize(voxel.dim.size());
        }
        voxel.qsdr = true;
    }

    tipl::vector<3,int> mni_to_voxel_index(Voxel& voxel,int x,int y,int z) const
    {               
        x = voxel.trans_to_mni[3]-x;
        y = voxel.trans_to_mni[7]-y;
        z -= voxel.trans_to_mni[11];
        x /= resolution_ratio;
        y /= resolution_ratio;
        z /= resolution_ratio;
        return tipl::vector<3,int>(x,y,z);
    }
    template<class interpolation_type>
    void interpolate_dwi(Voxel& voxel, VoxelData& data,const tipl::vector<3,double>& Jpos,interpolation_type)
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

        // output mapping position
        if(voxel.output_mapping)
        {
            mx[data.voxel_index] = Jpos[0];
            my[data.voxel_index] = Jpos[1];
            mz[data.voxel_index] = Jpos[2];
        }

        if(!voxel.grad_dev.empty())
        {
            tipl::matrix<3,3,float> grad_dev,new_j;
            for(unsigned int i = 0; i < 9; ++i)
                interpolation.estimate(voxel.grad_dev[i],grad_dev[i]);
            tipl::mat::transpose(grad_dev.begin(),tipl::dim<3,3>());
            new_j = grad_dev*data.jacobian;
            data.jacobian = new_j;
        }


        for(unsigned int index = 0;index < voxel.other_image.size();++index)
        {
            if(voxel.other_image[index].geometry() != src_geo)
            {
                tipl::vector<3,double> Opos;
                voxel.other_image_affine[index](Jpos,Opos);
                if(voxel.output_mapping)
                {
                    other_image_x[index][data.voxel_index] = Opos[0];
                    other_image_y[index][data.voxel_index] = Opos[1];
                    other_image_z[index][data.voxel_index] = Opos[2];
                }
                tipl::estimate(voxel.other_image[index],Opos,other_image[index][data.voxel_index]);
            }
            else
                interpolation.estimate(voxel.other_image[index],other_image[index][data.voxel_index]);
        }
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        tipl::vector<3,double> pos(tipl::pixel_index<3>(data.voxel_index,voxel.dim)),Jpos;
        pos[0] *= resolution_ratio;
        pos[1] *= resolution_ratio;
        pos[2] *= resolution_ratio;
        pos.round();
        tipl::vector<3,int> ipos(pos[0],pos[1],pos[2]);
        std::copy(affine.get(),affine.get()+9,data.jacobian.begin());

        if(cdm_dis.empty())
        {
            (*mni.get())(ipos,Jpos);
            affine(Jpos);
            tipl::matrix<3,3,float> M;
            tipl::reg::bfnorm_get_jacobian(*mni.get(),ipos,M.begin());
            data.jacobian *= M;
        }
        else
        {
            tipl::pixel_index<3> pos_index(ipos[0],ipos[1],ipos[2],cdm_dis.geometry());
            if(!cdm_dis.geometry().is_valid(pos_index))
                return;
            Jpos = pos;
            Jpos += cdm_dis[pos_index.index()];
            affine(Jpos);
            if(!cdm_dis.geometry().is_edge(pos_index))
            {
                tipl::matrix<3,3,float> M;
                tipl::jacobian_dis_at(cdm_dis,pos_index,M.begin());
                data.jacobian *= M;
            }
        }
        interpolate_dwi(voxel,data,Jpos,tipl::cubic_interpolation<3>());

        if(voxel.output_jacobian)
            jdet[data.voxel_index] = std::abs(data.jacobian.det()*affine_volume_scale);
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        voxel.mask.resize(src_geo);
        voxel.dim = src_geo;
        if(voxel.output_jacobian)
            mat_writer.write("jdet",&*jdet.begin(),1,jdet.size());
        if(voxel.output_mapping)
        {
            short dimension[3];
            dimension[0] = src_geo.width();
            dimension[1] = src_geo.height();
            dimension[2] = src_geo.depth();
            mat_writer.write("native_d",dimension,1,3);

            if(!mx.empty())
            {
                mat_writer.write("native_x",&*mx.begin(),1,mx.size());
                mat_writer.write("native_y",&*my.begin(),1,my.size());
                mat_writer.write("native_z",&*mz.begin(),1,mz.size());
            }
            if(!voxel.qa_map.empty())
                mat_writer.write("native_fa0",&*voxel.qa_map.begin(),1,voxel.qa_map.size());
        }
        for(unsigned int index = 0;index < other_image.size();++index)
        {
            mat_writer.write(voxel.other_image_name[index].c_str(),&*other_image[index].begin(),1,other_image[index].size());
            if(voxel.other_image[index].geometry() == src_geo)
                continue;
            short dimension[3];
            dimension[0] = voxel.other_image[index].width();
            dimension[1] = voxel.other_image[index].height();
            dimension[2] = voxel.other_image[index].depth();
            mat_writer.write((voxel.other_image_name[index]+"_d").c_str(),dimension,1,3);
            if(voxel.output_mapping)
            {
                mat_writer.write((voxel.other_image_name[index]+"_x").c_str(),&*other_image_x[index].begin(),1,other_image_x[index].size());
                mat_writer.write((voxel.other_image_name[index]+"_y").c_str(),&*other_image_y[index].begin(),1,other_image_y[index].size());
                mat_writer.write((voxel.other_image_name[index]+"_z").c_str(),&*other_image_z[index].begin(),1,other_image_z[index].size());
            }
        }

        if(!cdm_dis.empty() && !voxel.t1w.empty())
        {
            //convert T1W from FA template space to QSDR space
            tipl::image<float,3> output_t1w(des_geo);
            output_t1w.for_each_mt([&](float& v,const tipl::pixel_index<3>& index){
               tipl::vector<3,float> p(index);
               p *= resolution_ratio;
               if(voxel.t1w.geometry().is_valid(p))
                v = voxel.t1w.at(p[0],p[1],p[2]);
            });
            mat_writer.write("t1w",&output_t1w[0],1,output_t1w.size());
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
        if(voxel.z0 == 0.0)
            voxel.z0 = 1.0;
    }

};

#endif//MNI_RECONSTRUCTION_HPP
