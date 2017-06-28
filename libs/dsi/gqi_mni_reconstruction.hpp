#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include <chrono>
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
    image::geometry<3> src_geo;
    image::geometry<3> des_geo;
    int b0_index;
protected:
    std::auto_ptr<image::reg::bfnorm_mapping<double,3> > mni;
protected:
    image::basic_image<image::vector<3>,3> cdm_dis;
protected:
    image::transformation_matrix<double> affine;
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
        src_geo = voxel.dim;
        voxel_volume_scale = (voxel.vs[0]*voxel.vs[1]*voxel.vs[2]);

        if(voxel.reg_method == 4 && !voxel.t1w.empty()) //CDM
        {
            int prog = 0;
            // calculate the space shift between DWI and T1W
            image::vector<3> from(fa_template_imp.shift),to;
            from[0] -= (int)fa_template_imp.I.width()+voxel.t1wt_tran[3]-(int)voxel.t1wt.width();
            from[1] -= (int)fa_template_imp.I.height()+voxel.t1wt_tran[7]-(int)voxel.t1wt.height();
            from[2] -= voxel.t1wt_tran[11];
            to = from;
            to += image::vector<3>(fa_template_imp.I.geometry());

            image::normalize(voxel.t1w,1.0);
            image::normalize(voxel.t1wt,1.0);

            image::thread thread1,thread2;
            image::transformation_matrix<double> reg1T,reg2T;
            thread1.run([&](){
                if(export_intermediate)
                {
                    image::flip_xy(voxel.dwi_sum);
                    gz_nifti nii;
                    nii.set_voxel_size(voxel.vs);
                    nii << voxel.dwi_sum;
                    nii.save_to_file("b0.nii.gz");
                    image::flip_xy(voxel.dwi_sum);
                }
                image::reg::two_way_linear_mr(voxel.t1w,voxel.t1w_vs,voxel.dwi_sum,voxel.vs,
                               reg1T,image::reg::rigid_body,image::reg::mutual_information(),thread1.terminated);
            });
            thread2.run([&](){
                prog = 1;
                image::reg::two_way_linear_mr(voxel.t1wt,voxel.t1wt_vs,voxel.t1w,voxel.t1w_vs,
                               reg2T,image::reg::affine,image::reg::mutual_information(),thread2.terminated);
                image::basic_image<float,3> J(voxel.t1wt.geometry());
                image::resample_mt(voxel.t1w,J,reg2T,image::cubic);

                prog = 3;

                {
                    image::basic_image<float,3> Is(J),It(voxel.t1wt);
                    image::filter::gaussian(Is);
                    if(export_intermediate)
                    {
                        It.save_to_file<gz_nifti>("It.nii.gz");
                        Is.save_to_file<gz_nifti>("Is.nii.gz");
                    }
                    float resolution = 2.0f;
                    float smoothness = 0.5f;
                    voxel.R2 = image::reg::cdm(It,Is,cdm_dis,thread2.terminated,resolution,smoothness);
                    voxel.R2 *= voxel.R2;
                }
                image::compose_displacement(J,cdm_dis,voxel.t1w);
                // From T1W template space to FA template space
                image::crop(voxel.t1w,from,to);
                image::crop(cdm_dis,from,to);

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
                image::basic_image<float,3> b0J(fa_template_imp.I.geometry());
                image::resample_mt(voxel.dwi_sum,b0J,affine,image::cubic);
                b0J.save_to_file<gz_nifti>("b0j.nii.gz");
            }
            goto end_normalization;
        }
        {
        image::basic_image<float,3> VG,VF;
        VG = fa_template_imp.I;
        VF = voxel.qa_map;

        image::filter::gaussian(VF);
        VF -= image::segmentation::otsu_threshold(VF);
        image::lower_threshold(VF,0.0);
        image::normalize(VG,1.0);
        image::normalize(VF,1.0);

        image::basic_image<float,3> VFF;
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
                image::reg::two_way_linear_mr(VG,fa_template_imp.vs,VF,voxel.vs,affine,image::reg::affine,image::reg::mt_correlation<image::basic_image<float,3>,
                                              image::transformation_matrix<double> >(0),terminated);
            }
            VFF.resize(VG.geometry());
            image::resample(VF,VFF,affine,image::cubic);
            if(prog_aborted())
                throw std::runtime_error("Reconstruction canceled");

        }
        //linear regression
        image::match_signal(VG,VFF);


        if(export_intermediate)
            VFF.save_to_file<gz_nifti>("Subject_QA_linear_reg.nii.gz");


        try
        {
            begin_prog("normalization");
            terminated_class ter(64);
            int factor = voxel.reg_method + 1;

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
                image::reg::bfnorm(*mni.get(),VG2,VFF2,ter,voxel.thread_count);
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
                if(factor <= 3)
                {
                    mni.reset(new image::reg::bfnorm_mapping<double,3>(VG.geometry(),image::geometry<3>(factor*7,factor*9,factor*7)));
                    image::reg::bfnorm(*mni.get(),VG,VFF,ter,voxel.thread_count);
                    voxel.R2 = -image::reg::correlation()(VG,VFF,(*mni.get()));
                    if(export_intermediate)
                    {
                        image::basic_image<float,3> VFFF(VG.geometry());
                        image::resample(VFF,VFFF,*mni.get(),image::cubic);
                        VFFF.save_to_file<gz_nifti>("Subject_QA_nonlinear_reg.nii.gz");
                    }
                }
                else
                {
                    bool terminated = false;
                    image::reg::cdm(VG,VFF,cdm_dis,terminated,2.0,0.5);
                    image::basic_image<float,3> VFFF;
                    image::compose_displacement(VFF,cdm_dis,VFFF);
                    float r = image::correlation(VG.begin(),VG.end(),VFFF.begin());
                    voxel.R2 = r*r;
                }
                std::cout << "R2=" << voxel.R2 << std::endl;

            }
        }
        catch(...)
        {
            throw std::runtime_error("Registration failed due to memory insufficiency.");
        }

        }
        end_normalization:

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

            scale[0] = voxel_size;
            scale[1] = voxel_size;
            scale[2] = voxel_size;

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

        voxel.dim = des_geo;
        voxel.image_model->mask.resize(des_geo);
        std::fill(voxel.image_model->mask.begin(),voxel.image_model->mask.end(),0);
        for(image::pixel_index<3> index(des_geo);index < des_geo.size();++index)
        {
            image::vector<3,float> mni_pos(index);
            mni_pos *= voxel_size;
            mni_pos += des_offset;
            mni_pos.round();
            if(fa_template_imp.mask.geometry().is_valid(mni_pos) &&
                    fa_template_imp.mask.at(mni_pos[0],mni_pos[1],mni_pos[2]) > 0.0)
                voxel.image_model->mask[index.index()] = 1;
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
            if(voxel.other_image[index].geometry() != src_geo)
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
        pos.round();
        image::vector<3,int> ipos(pos[0],pos[1],pos[2]);
        std::copy(affine.get(),affine.get()+9,data.jacobian.begin());

        if(cdm_dis.empty())
        {
            (*mni.get())(ipos,Jpos);
            if(voxel_size < 0.99)
                Jpos *= voxel_size;
            affine(Jpos);
            image::matrix<3,3,float> M;
            image::reg::bfnorm_get_jacobian(*mni.get(),ipos,M.begin());
            data.jacobian *= M;
            data.jdet = std::abs(data.jacobian.det()*voxel_volume_scale);
        }
        else
        {
            image::pixel_index<3> pos_index(ipos[0],ipos[1],ipos[2],cdm_dis.geometry());
            if(!cdm_dis.geometry().is_valid(pos_index))
                return;
            Jpos = pos;
            Jpos += cdm_dis[pos_index.index()];
            affine(Jpos);
            data.jdet = std::abs(data.jacobian.det()*voxel_volume_scale);
            if(!cdm_dis.geometry().is_edge(pos_index))
            {
                image::matrix<3,3,float> M;
                image::jacobian_dis_at(cdm_dis,pos_index,M.begin());
                data.jacobian *= M;
            }
        }

        interpolate_dwi(voxel,data,Jpos,image::cubic_interpolation<3>());

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
            mat_writer.write("t1w",&voxel.t1w[0],1,voxel.t1w.size());
        mat_writer.write("trans",&*trans_to_mni,4,4);
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
        {
            image::vector<3,int> cur_pos(image::pixel_index<3>(data.voxel_index,voxel.dim));
            if((cur_pos-voxel.csf_pos1).length() <= 1.0 || (cur_pos-voxel.csf_pos2).length() <= 1.0 ||
               (cur_pos-voxel.csf_pos3).length() <= 1.0 || (cur_pos-voxel.csf_pos4).length() <= 1.0)
            {
                std::lock_guard<std::mutex> lock(mutex);
                samples.push_back(*std::min_element(data.odf.begin(),data.odf.end())/data.jdet);
            }
        }
    }
    void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        voxel.z0 = image::median(samples.begin(),samples.end());
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
    virtual void end(Voxel&,gz_mat_write&)
    {

    }

};

#endif//MNI_RECONSTRUCTION_HPP
