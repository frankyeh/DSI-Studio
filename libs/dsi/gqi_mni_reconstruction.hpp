#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include "gqi_process.hpp"
#include "mapping/fa_template.hpp"
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "odf_decomposition.hpp"
#include "odf_deconvolusion.hpp"
#include "gqi_process.hpp"

extern fa_template fa_template_imp;





class DWINormalization  : public BaseProcess
{
protected:
    std::auto_ptr<image::reg::bfnorm_mapping<double,3> > mni;
    image::geometry<3> src_geo;
    image::geometry<3> des_geo;
    int b0_index;
protected:
    image::transformation_matrix<double> affine;
protected:
    image::basic_image<float,3> VG,VF;
    image::vector<3> VGvs;
    float voxel_size; // output resolution
protected: // for warping other image modality
    std::vector<image::basic_image<float,3> > other_image,other_image_x,other_image_y,other_image_z;
protected:
    image::vector<3,int> bounding_box_lower;
    image::vector<3,int> bounding_box_upper;
    image::vector<3,int> des_offset;// = {6,7,11};	// the offset due to bounding box
    image::vector<3,float> scale;
    double trans_to_mni[16];
protected:
    float voxel_volume_scale;
    std::vector<float> jdet;
    std::vector<float> mx,my,mz;
protected:
    typedef image::const_pointer_image<unsigned short,3> point_image_type;
    std::vector<point_image_type> ptr_images;
    std::vector<image::vector<3,float> > q_vectors_time;

public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.vs[0] == 0.0 ||
                voxel.vs[1] == 0.0 ||
                voxel.vs[2] == 0.0)
            throw std::runtime_error("No spatial information found in src file. Recreate src file or contact developer for assistance");
        bool export_intermediate = false;
        begin_prog("normalization");
        voxel_size = voxel.param[1];
        VG = fa_template_imp.I;
        VF = voxel.qa_map;

        image::filter::gaussian(VF);
        VF -= image::segmentation::otsu_threshold(VF);
        image::lower_threshold(VF,0.0);

        src_geo = VF.geometry();

        image::normalize(VG,1.0);
        image::normalize(VF,1.0);

        VGvs = fa_template_imp.vs;

        voxel_volume_scale = (voxel.vs[0]*voxel.vs[1]*voxel.vs[2])/(VGvs[0]*VGvs[1]*VGvs[2]);

        image::basic_image<float,3> VFF;
        {
            begin_prog("linear registration");

            image::affine_transform<double> arg_min;
            // VG: FA TEMPLATE
            // VF: SUBJECT QA
            if(export_intermediate)
            {
                VG.save_to_file<image::io::nifti>("Template_QA.nii.gz");
                VF.save_to_file<image::io::nifti>("Subject_QA.nii.gz");
            }

            if(voxel.qsdr_trans.data[0] != 0.0) // has manual reg data
                affine = voxel.qsdr_trans;
            else
            {
                bool terminated = false;
                image::reg::linear(VF,voxel.vs,VG,VGvs,arg_min,image::reg::affine,image::reg::mt_correlation<image::basic_image<float,3>,
                                   image::transformation_matrix<double> >(0),terminated);
                image::reg::linear(VF,voxel.vs,VG,VGvs,arg_min,image::reg::affine,image::reg::mt_correlation<image::basic_image<float,3>,
                                   image::transformation_matrix<double> >(0),terminated);
                affine = image::transformation_matrix<double>(arg_min,VF.geometry(),voxel.vs,VG.geometry(),VGvs);
            }
            affine.inverse();
            VFF.resize(VG.geometry());
            image::resample(VF,VFF,affine,image::cubic);
            if(prog_aborted())
                throw std::runtime_error("Reconstruction canceled");

        }
        //linear regression
        match_signal(VG,VFF);


        if(export_intermediate)
            VFF.save_to_file<image::io::nifti>("Subject_QA_linear_reg.nii.gz");


        try
        {
            begin_prog("normalization");
            terminated_class ter(17);
            int factor = voxel.reg_method + 1;
            int iteration = 0;

            if(voxel_size < 0.99)
            {
                image::geometry<3> geo2(VG.width()/voxel_size,VG.height()/voxel_size,VG.depth()/voxel_size);
                image::basic_image<float,3> VG2(geo2),VFF2(geo2);
                image::transformation_matrix<float> m;
                m.sr[0] = voxel_size;
                m.sr[4] = voxel_size;
                m.sr[8] = voxel_size;
                image::resample(VG,VG2,m,image::cubic);
                image::resample(VFF,VFF2,m,image::cubic);
                mni.reset(new image::reg::bfnorm_mapping<double,3>(geo2,image::geometry<3>(factor*7,factor*9,factor*7)));
                image::reg::bfnorm(*mni.get(),VG2,VFF2,voxel.voxel_data.size(),ter,iteration);
                voxel.R2 = -image::reg::correlation()(VG2,VFF2,(*mni.get()));
                if(export_intermediate)
                {
                    image::basic_image<float,3> VFFF2(VG2.geometry());
                    image::resample(VFF2,VFFF2,*mni.get(),image::cubic);
                    VFFF2.save_to_file<image::io::nifti>("Subject_QA_nonlinear_reg.nii.gz");
                }
            }
            else
            {
                mni.reset(new image::reg::bfnorm_mapping<double,3>(VG.geometry(),image::geometry<3>(factor*7,factor*9,factor*7)));
                image::reg::bfnorm(*mni.get(),VG,VFF,voxel.voxel_data.size(),ter,iteration);
                voxel.R2 = -image::reg::correlation()(VG,VFF,(*mni.get()));
                if(export_intermediate)
                {
                    image::basic_image<float,3> VFFF(VG.geometry());
                    image::resample(VFF,VFFF,*mni.get(),image::cubic);
                    VFFF.save_to_file<image::io::nifti>("Subject_QA_nonlinear_reg.nii.gz");
                }
            }
        }
        catch(...)
        {
            throw std::runtime_error("Registration failed due to memory insufficiency.");
        }

        // setup output bounding box
        {
            //setBoundingBox(-78,-112,-50,78,76,85,1.0);
            bounding_box_lower[0] = std::floor(-78.0/voxel_size+0.5)*voxel_size;
            bounding_box_lower[1] = std::floor(-112.0/voxel_size+0.5)*voxel_size;
            bounding_box_lower[2] = std::floor(-50.0/voxel_size+0.5)*voxel_size;
            bounding_box_upper[0] = std::floor(78.0/voxel_size+0.5)*voxel_size;
            bounding_box_upper[1] = std::floor(76.0/voxel_size+0.5)*voxel_size;
            bounding_box_upper[2] = std::floor(85.0/voxel_size+0.5)*voxel_size;
            des_geo[0] = (bounding_box_upper[0]-bounding_box_lower[0])/voxel_size+1;//79
            des_geo[1] = (bounding_box_upper[1]-bounding_box_lower[1])/voxel_size+1;//95
            des_geo[2] = (bounding_box_upper[2]-bounding_box_lower[2])/voxel_size+1;//69

            // DSI Studio use radiology convention, the MNI coordinate of the x and y are flipped
            // fa_template is now LPS,but bounding box is RAS
            des_offset[0] = fa_template_imp.shift[0]-bounding_box_upper[0];
            des_offset[1] = fa_template_imp.shift[1]-bounding_box_upper[1];
            des_offset[2] = bounding_box_lower[2]-fa_template_imp.shift[2];
            // units in template space
            des_offset[0] /= VGvs[0];
            des_offset[1] /= VGvs[1];
            des_offset[2] /= VGvs[2];

            scale[0] = voxel_size/VGvs[0];
            scale[1] = voxel_size/VGvs[1];
            scale[2] = voxel_size/VGvs[2];

            // setup transformation matrix
            std::fill(trans_to_mni,trans_to_mni+16,0.0);
            trans_to_mni[15] = 1.0;
            trans_to_mni[0] = -voxel_size;
            trans_to_mni[5] = -voxel_size;
            trans_to_mni[10] = voxel_size;
            trans_to_mni[3] = bounding_box_upper[0];
            trans_to_mni[7] = bounding_box_upper[1];
            trans_to_mni[11] = bounding_box_lower[2];
        }

        // setup mask
        {
            // set the current mask to template space
            voxel.dim = des_geo;
            voxel.image_model->mask.resize(des_geo);
            std::fill(voxel.image_model->mask.begin(),voxel.image_model->mask.end(),0);
            for(image::pixel_index<3> index(des_geo);index < des_geo.size();++index)
            {
                image::vector<3,float> mni_pos(index);
                mni_pos *= voxel.param[1];
                mni_pos[0] /= VGvs[0];
                mni_pos[1] /= VGvs[1];
                mni_pos[2] /= VGvs[2];
                mni_pos += des_offset;
                mni_pos += 0.5;
                mni_pos.floor();
                if(fa_template_imp.I.geometry().is_valid(mni_pos) &&
                        fa_template_imp.I.at(mni_pos[0],mni_pos[1],mni_pos[2]) > 0.0)
                    voxel.image_model->mask[index.index()] = 1;
            }
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


        b0_index = -1;
        if(voxel.half_sphere)
            for(unsigned int index = 0; index < voxel.bvalues.size(); ++index)
                if(voxel.bvalues[index] == 0)
                    b0_index = index;

        ptr_images.clear();
        for (unsigned int index = 0; index < voxel.image_model->dwi_data.size(); ++index)
            ptr_images.push_back(image::make_image(voxel.image_model->dwi_data[index],src_geo));


        std::fill(voxel.vs.begin(),voxel.vs.end(),voxel.param[1]);

        voxel.csf_pos1 = mni_to_voxel_index(6,0,18);
        voxel.csf_pos2 = mni_to_voxel_index(-6,0,18);
        voxel.csf_pos3 = mni_to_voxel_index(4,18,10);
        voxel.csf_pos4 = mni_to_voxel_index(-4,18,10);
        voxel.z0 = 0.0;

        // output mapping
        if(voxel.output_jacobian)
            jdet.resize(voxel.dim.size());

        if(voxel.output_mapping)
        {
            mx.resize(voxel.dim.size());
            my.resize(voxel.dim.size());
            mz.resize(voxel.dim.size());
        }
    }

    image::vector<3,int> mni_to_voxel_index(int x,int y,int z) const
    {
        x = bounding_box_upper[0]-x;
        y = bounding_box_upper[1]-y;
        z -= bounding_box_lower[2];
        x /= scale[0];
        y /= scale[1];
        z /= scale[2];
        return image::vector<3,int>(x,y,z);
    }
    template<class interpolation_type>
    void interpolate_dwi(Voxel& voxel, VoxelData& data,const image::vector<3,double>& Jpos,interpolation_type)
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
        if(voxel.half_sphere && b0_index != -1)
            data.space[b0_index] /= 2.0;
        // output mapping position
        if(voxel.output_mapping)
        {
            mx[data.voxel_index] = Jpos[0];
            my[data.voxel_index] = Jpos[1];
            mz[data.voxel_index] = Jpos[2];
        }

        if(!voxel.grad_dev.empty())
        {
            image::matrix<3,3,float> grad_dev,new_j;
            for(unsigned int i = 0; i < 9; ++i)
                interpolation.estimate(voxel.grad_dev[i],grad_dev[i]);
            image::mat::transpose(grad_dev.begin(),image::dim<3,3>());
            new_j = grad_dev*data.jacobian;
            data.jacobian = new_j;
        }

        for(unsigned int index = 0;index < voxel.other_image.size();++index)
        {
            if(voxel.other_image[index].geometry() != voxel.qa_map.geometry())
            {
                interpolation_type interpo;
                image::vector<3,double> Opos;
                voxel.other_image_affine[index](Jpos,Opos);
                if(voxel.output_mapping)
                {
                    other_image_x[index][data.voxel_index] = Opos[0];
                    other_image_y[index][data.voxel_index] = Opos[1];
                    other_image_z[index][data.voxel_index] = Opos[2];
                }
                interpo.get_location(voxel.other_image[index].geometry(),Opos);
                interpo.estimate(voxel.other_image[index],other_image[index][data.voxel_index]);
            }
            else
                interpolation.estimate(voxel.other_image[index],other_image[index][data.voxel_index]);
        }
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        image::vector<3,double> pos(image::pixel_index<3>(data.voxel_index,voxel.dim)),Jpos;
        pos[0] *= scale[0];
        pos[1] *= scale[1];
        pos[2] *= scale[2];
        pos += des_offset;
        if(voxel_size < 0.99)
            pos /= voxel_size;
        pos += 0.5;
        (*mni.get())(image::vector<3,int>(pos[0],pos[1],pos[2]),Jpos);
        if(voxel_size < 0.99)
            Jpos *= voxel_size;
        affine(Jpos);

        image::matrix<3,3,float> M;
        image::reg::bfnorm_get_jacobian(*mni.get(),pos,M.begin());
        std::copy(affine.get(),affine.get()+9,data.jacobian.begin());
        data.jacobian *= M;


        switch(voxel.interpo_method)
        {
        case 0:
            interpolate_dwi(voxel,data,Jpos,image::interpolation<image::linear_weighting,3>());
            break;
        case 1:
            interpolate_dwi(voxel,data,Jpos,image::interpolation<image::gaussian_radial_basis_weighting,3>());
            break;
        case 2:
            interpolate_dwi(voxel,data,Jpos,image::cubic_interpolation<3>());
            break;
        }
        data.jdet = std::abs(data.jacobian.det()*voxel_volume_scale);
        if(voxel.output_jacobian)
            jdet[data.voxel_index] = data.jdet;
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        voxel.image_model->mask.resize(src_geo);
        voxel.dim = src_geo;
        if(voxel.output_jacobian)
            mat_writer.write("jdet",&*jdet.begin(),1,jdet.size());
        if(voxel.output_mapping)
        {
            mat_writer.write("native_x",&*mx.begin(),1,mx.size());
            mat_writer.write("native_y",&*my.begin(),1,my.size());
            mat_writer.write("native_z",&*mz.begin(),1,mz.size());
            short dimension[3];
            dimension[0] = voxel.qa_map.width();
            dimension[1] = voxel.qa_map.height();
            dimension[2] = voxel.qa_map.depth();
            mat_writer.write("native_d",dimension,1,3);
            mat_writer.write("native_fa0",&*voxel.qa_map.begin(),1,voxel.qa_map.size());
        }
        for(unsigned int index = 0;index < other_image.size();++index)
        {
            mat_writer.write(voxel.other_image_name[index].c_str(),&*other_image[index].begin(),1,other_image[index].size());
            if(voxel.other_image[index].geometry() == voxel.qa_map.geometry())
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

        mat_writer.write("trans",&*trans_to_mni,4,4);
        mat_writer.write("R2",&voxel.R2,1,1);
    }

};

class EstimateZ0_MNI : public BaseProcess
{
public:
    void init(Voxel& voxel)
    {
        voxel.z0 = 0.0;
    }
    void run(Voxel& voxel, VoxelData& data)
    {
        // perform csf cross-subject normalization
        {
            image::vector<3,int> cur_pos(image::pixel_index<3>(data.voxel_index,voxel.dim));
            if((cur_pos-voxel.csf_pos1).length() <= 1.0 || (cur_pos-voxel.csf_pos2).length() <= 1.0 ||
               (cur_pos-voxel.csf_pos3).length() <= 1.0 || (cur_pos-voxel.csf_pos4).length() <= 1.0)
            {
                float odf_dif = *std::min_element(data.odf.begin(),data.odf.end());
                odf_dif /= data.jdet;
                if(odf_dif > voxel.z0)
                    voxel.z0 = odf_dif;
            }
        }
    }
    void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if(voxel.z0 == 0.0)
            voxel.z0 = 1.0;
        mat_writer.write("z0",&voxel.z0,1,1);
    }

};

class QSDR  : public BaseProcess
{
public:
    double r2_base_function(double theta)
    {
        if(std::abs(theta) < 0.000001)
            return 1.0/3.0;
        return (2*std::cos(theta)+(theta-2.0/theta)*std::sin(theta))/theta/theta;
    }
protected:
    std::vector<image::vector<3,double> > q_vectors_time;
public:
    virtual void init(Voxel& voxel)
    {
        float sigma = voxel.param[0];
        q_vectors_time.resize(voxel.bvalues.size());
        for (unsigned int index = 0; index < voxel.bvalues.size(); ++index)
        {
            q_vectors_time[index] = voxel.bvectors[index];
            q_vectors_time[index] *= std::sqrt(voxel.bvalues[index]*0.01506);// get q in (mm) -1
            q_vectors_time[index] *= sigma;
        }
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> sinc_ql(data.odf.size()*data.space.size());
        for (unsigned int j = 0,index = 0; j < data.odf.size(); ++j)
        {
            image::vector<3,double> from(voxel.ti.vertices[j]);
            from.rotate(data.jacobian);
            from.normalize();
            if(voxel.r2_weighted)
                for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                    sinc_ql[index] = r2_base_function(q_vectors_time[i]*from);
            else
                for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                    sinc_ql[index] = boost::math::sinc_pi(q_vectors_time[i]*from);

        }
        image::mat::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                      image::dyndim(data.odf.size(),data.space.size()));
        image::multiply_constant(data.odf,data.jdet);

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

    }

};

#endif//MNI_RECONSTRUCTION_HPP
