#ifndef MNI_RECONSTRUCTION_HPP
#define MNI_RECONSTRUCTION_HPP
#include "gqi_process.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/normalization.hpp"
#include "basic_voxel.hpp"
#include "basic_process.hpp"
#include "odf_decomposition.hpp"
#include "gqi_process.hpp"

class GQI_phantom  : public BaseProcess
{
private:
    image::geometry<3> src_geo;
    image::geometry<3> des_geo;
    float dir_length;
	float max_accumulated_qa;
private:
    typedef image::basic_image<unsigned short,3,image::const_pointer_memory<unsigned short> > point_image_type;
    std::vector<point_image_type> ptr_images;

    std::vector<image::vector<3,float> > q_vectors_time;
public:
    virtual void init(Voxel& voxel)
    {
        float sigma = voxel.param[0]; //diffusion sampling length ratio, optimal 1.24

        //float dif_sampling_length = voxel.param[2]*0.001;// from um to mm
        dir_length = 0.020;// 20 um, previously =dif_sampling_length/2.0;

        src_geo = image::geometry<3>(voxel.matrix_width,voxel.matrix_height,voxel.slice_number);


        des_geo=src_geo;


        q_vectors_time.resize(voxel.bvalues.size());
        for (unsigned int index = 0; index < voxel.bvalues.size(); ++index)
        {
            q_vectors_time[index] = voxel.bvectors[index];
            q_vectors_time[index] *= std::sqrt(voxel.bvalues[index]*0.01506);// get q in (mm) -1
            q_vectors_time[index] *= sigma;
        }

        ptr_images.clear();
        for (unsigned int index = 0; index < voxel.image_model->dwi_data.size(); ++index)
            ptr_images.push_back(point_image_type((const unsigned short*)voxel.image_model->dwi_data[index],src_geo));

        // has to handle qa scaling here
		voxel.qa_scaling = 0;
		max_accumulated_qa = 0;
    }
    //* used for phantom warpping
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        unsigned int p_index = data.voxel_index;
        unsigned int x = p_index % des_geo[0];
        p_index -= x;
        p_index /= des_geo[0];
        unsigned int y = p_index % des_geo[1];
        p_index -= y;
        p_index /= des_geo[1];
        unsigned int z = p_index;

        image::vector<3,float> pos(x,y,z);
        // resample the images
        {
            image::vector<3,float> b(pos);
            double fw = 2.0*3.1415926535897932384626433832795/128.0*3.0;
            double amp = 2.0;
            double cos_y = std::cos(pos[1]*fw);
            double sin_y = std::sin(pos[1]*fw);
            double cos_x = std::cos(pos[0]*fw);
            double sin_x = std::sin(pos[0]*fw);
            double xx_dx = 1.0 + cos_y*cos_x*fw*amp;
            double xx_dy = -sin_y*sin_x*fw*amp;

            b[0] +=  amp*cos_y*sin_x;
            b[1] +=  amp*sin_y*cos_x;

            image::interpolation<image::linear_weighting,2> trilinear_interpolation;
            image::vector<2,float> b_2d(b[0],b[1]);
            if (!trilinear_interpolation.get_location(image::geometry<2>(src_geo[0],src_geo[1]),b_2d))
                std::fill(data.odf.begin(),data.odf.end(),0);
            else
            {
                std::vector<float> sinc_ql(data.odf.size()*data.space.size());
                unsigned int offset = z*src_geo.plane_size();
                for (unsigned int i = 0; i < data.space.size(); ++i)
                {
                    image::basic_image<unsigned short,2,image::const_pointer_memory<unsigned short> >
                    buf(ptr_images[i].begin()+offset,image::geometry<2>(src_geo[0],src_geo[1]));
                    trilinear_interpolation.estimate(buf,b_2d,data.space[i]);
                }

                double Jdet = std::abs(xx_dx*xx_dx-xx_dy*xx_dy);
                for (unsigned int j = 0,index = 0; j < data.odf.size(); ++j)
                {

                    image::vector<3,float> dir(voxel.ti.vertices[j]);
                    image::vector<3,float> dir_to(dir);
                    dir_to[0] = dir[0]*xx_dx+dir[1]*xx_dy;
                    dir_to[1] = dir[0]*xx_dy+dir[1]*xx_dx;
                    dir_to.normalize();
                    for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                        sinc_ql[index] = boost::math::sinc_pi(q_vectors_time[i]*dir_to);

                }
                math::matrix_vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                            math::dyndim(data.odf.size(),data.space.size()));
                float accumulated_qa = std::accumulate(data.odf.begin(),data.odf.end(),0.0);
                if (max_accumulated_qa < accumulated_qa)
				{
					max_accumulated_qa = accumulated_qa; 
					voxel.qa_scaling = accumulated_qa/data.odf.size();
				}
                std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= Jdet);
            }
        }
    }


};

extern fa_template fa_template_imp;
class GQI_MNI  : public BaseProcess
{
public:
protected:
    normalization<image::basic_image<float,3> > mni;
    image::geometry<3> src_geo;
    image::geometry<3> des_geo;
    double dir_length;
    double max_accumulated_qa;
    double voxel_volume_scale;
    std::vector<float> max_odf;
    std::vector<double> jdet;
    int b0_index;
protected:
    double r2_base_function(double theta)
    {
            if(std::abs(theta) < 0.000001)
                    return 1.0/3.0;
            return (2*std::cos(theta)+(theta-2.0/theta)*std::sin(theta))/theta/theta;
    }
protected:
    typedef image::basic_image<unsigned short,3,image::const_pointer_memory<unsigned short> > point_image_type;
    std::vector<point_image_type> ptr_images;

    std::vector<image::vector<3,double> > q_vectors_time;
public:
    virtual void init(Voxel& voxel)
    {
        begin_prog("normalization");

        mni.VF.swap(voxel.qa_map);
        image::filter::gaussian(mni.VF);
        image::normalize(mni.VF,1.0);
        /*
        image::minus_constant(mni.VF.begin(),mni.VF.end(),0.2);
        image::lower_threshold(mni.VF.begin(),mni.VF.end(),0.0);
        image::normalize(mni.VF,1.0);
        */

        mni.VFvs[0] = voxel.voxel_size[0];
        mni.VFvs[1] = voxel.voxel_size[1];
        mni.VFvs[2] = voxel.voxel_size[2];

        mni.VG = fa_template_imp.I;
        image::normalize(mni.VG,1.0);

        /*
        image::io::nifti nii;
        nii << mni.VF;
        nii.save_to_file("VF.nii");
        nii << mni.VG;
        nii.save_to_file("VG.nii");
        */

        mni.VG_trans.resize(fa_template_imp.tran.size());
        std::copy(fa_template_imp.tran.begin(),fa_template_imp.tran.end(),mni.VG_trans.begin());
        mni.VGvs[0] = fa_template_imp.tran[0];
        mni.VGvs[1] = fa_template_imp.tran[5];
        mni.VGvs[2] = fa_template_imp.tran[10];
        mni.normalize();

        voxel_volume_scale =  mni.VFvs[0]*mni.VFvs[1]*mni.VFvs[2]/mni.VGvs[0]/mni.VGvs[1]/mni.VGvs[2];

        mni.set_voxel_size(voxel.param[1]);

        begin_prog("q-space diffeomorphic reconstruction");
        src_geo = mni.VF.geometry();
        des_geo = mni.BDim;


        float sigma = voxel.param[0]; //diffusion sampling length ratio, optimal 1.24
        //float dif_sampling_length = voxel.param[2]*0.001;// from um to mm
        dir_length = 0.020;// 20 um, previously =dif_sampling_length/2.0;

        // setup mask
        {
            // set the current mask to template space
            voxel.image_model->set_dimension(des_geo[0],des_geo[1],des_geo[2]);
            for(image::pixel_index<3> index;des_geo.is_valid(index);index.next(des_geo))
            {
                image::vector<3,int> mni_pos(index);
                mni_pos *= voxel.param[1];
                mni_pos[0] /= mni.VGvs[0];
                mni_pos[1] /= mni.VGvs[1];
                mni_pos[2] /= mni.VGvs[2];
                mni_pos += mni.BOffset;
                voxel.image_model->mask[index.index()] =
                        fa_template_imp.I.at(mni_pos[0],mni_pos[1],mni_pos[2]) > 0.0? 1: 0;
            }
        }


        q_vectors_time.resize(voxel.bvalues.size());
        for (unsigned int index = 0; index < voxel.bvalues.size(); ++index)
        {
            q_vectors_time[index] = voxel.bvectors[index];
            q_vectors_time[index] *= std::sqrt(voxel.bvalues[index]*0.01506);// get q in (mm) -1
            q_vectors_time[index] *= sigma;
        }

        b0_index = -1;
        if(voxel.half_sphere)
            for(unsigned int index = 0;index < voxel.bvalues.size();++index)
                if(voxel.bvalues[index] == 0)
                    b0_index = index;

        ptr_images.clear();
        for (unsigned int index = 0; index < voxel.image_model->dwi_data.size(); ++index)
            ptr_images.push_back(point_image_type((const unsigned short*)voxel.image_model->dwi_data[index],src_geo));


        if(voxel.voxel_size[0] == 0.0 ||
           voxel.voxel_size[1] == 0.0 ||
           voxel.voxel_size[2] == 0.0)
            throw std::runtime_error("No spatial information found in src file. Recreate src file or contact developer for assistance");
        voxel.qa_scaling = voxel.reponse_function_scaling/voxel.voxel_size[0]/voxel.voxel_size[1]/voxel.voxel_size[2];
        max_accumulated_qa = 0;
		
        std::fill(voxel.voxel_size,voxel.voxel_size+3,voxel.param[1]);
        jdet.resize(des_geo.size());
        std::fill(jdet.begin(),jdet.end(),mni.affine_det);
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        image::pixel_index<3> pos(data.voxel_index,des_geo);
        image::vector<3,double> b;
        float jacobian[9];
        mni.warp_coordinate(pos,b);
        mni.get_jacobian(pos,jacobian);
        // resample the images
        {

            image::interpolation<image::linear_weighting,3> trilinear_interpolation;
            if (!trilinear_interpolation.get_location(src_geo,b))
                std::fill(data.odf.begin(),data.odf.end(),0);
            else
            {
                for (unsigned int i = 0; i < data.space.size(); ++i)
                    trilinear_interpolation.estimate(ptr_images[i],b,data.space[i]);

                if(b0_index >= 0)
                    data.space[b0_index] /= 2.0;

                std::vector<float> sinc_ql(data.odf.size()*voxel.q_count);
                for (unsigned int j = 0,index = 0; j < data.odf.size(); ++j)
                {
                    image::vector<3,double> dir(voxel.ti.vertices[j]),from;
                    math::matrix_product(jacobian,dir.begin(),from.begin(),math::dim<3,3>(),math::dim<3,1>());
                    from.normalize();
                    if(voxel.r2_weighted)
                        for (unsigned int i = 0; i < voxel.q_count; ++i,++index)
                            sinc_ql[index] = r2_base_function(q_vectors_time[i]*from);
                    else
                        for (unsigned int i = 0; i < voxel.q_count; ++i,++index)
                            sinc_ql[index] = boost::math::sinc_pi(q_vectors_time[i]*from);

                }
                math::matrix_vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                            math::dyndim(data.odf.size(),data.space.size()));

                // computing Jacobian determinant
                // but only need the jacobian due to normalization
                float Jdet = std::abs(math::matrix_determinant(jacobian,math::dim<3,3>())*voxel_volume_scale);
                std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= Jdet);

                float accumulated_qa = std::accumulate(data.odf.begin(),data.odf.end(),0.0);
                if (max_accumulated_qa < accumulated_qa)
                {
                    max_accumulated_qa = accumulated_qa;
                    max_odf = data.odf;
                }
                jdet[data.voxel_index] = Jdet;
            }
        }
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {
        mat_writer.add_matrix("free_water_quantity",&voxel.qa_scaling,1,1);
        mat_writer.add_matrix("jdet",&*jdet.begin(),1,jdet.size());
        mat_writer.add_matrix("mni",&*mni.trans_to_mni,4,3);
    }

};

class QSDR_Decomposition : public GQI_MNI{
private:
    ODFDecomposition decomposition;
    QSpace2Odf gqi;
public:
    virtual void init(Voxel& voxel)
    {
        GQI_MNI::init(voxel);
        gqi.init(voxel);
        decomposition.init(voxel);
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        image::pixel_index<3> pos(data.voxel_index,des_geo);
        image::vector<3,double> b;
        float jacobian[9];
        mni.warp_coordinate(pos,b);
        mni.get_jacobian(pos,jacobian);
        // resample the images
        {

            image::interpolation<image::linear_weighting,3> trilinear_interpolation;
            if (!trilinear_interpolation.get_location(src_geo,b))
                std::fill(data.odf.begin(),data.odf.end(),0);
            else
            {
                for (unsigned int i = 0; i < data.space.size(); ++i)
                    trilinear_interpolation.estimate(ptr_images[i],b,data.space[i]);

                if(b0_index >= 0)
                    data.space[b0_index] /= 2.0;

                gqi.run(voxel,data);
                decomposition.run(voxel,data);
                std::vector<float> new_odf(data.odf.size());
                std::fill(new_odf.begin(),new_odf.end(),data.min_odf);
                for(unsigned int i = 0;i < data.odf.size();++i)
                    if(data.odf[i] > data.min_odf)
                    {
                        image::vector<3,double> dir(voxel.ti.vertices[i]),from;
                        math::matrix_product(jacobian,dir.begin(),from.begin(),math::dim<3,3>(),math::dim<3,1>());
                        from.normalize();
                        float max_cos = 0.0;
                        unsigned int max_j = 0;
                        for(unsigned int j = 0;j < data.odf.size();++j)
                        {
                            float cos = std::abs(voxel.ti.vertices[j]*from);
                            if(cos > max_cos)
                            {
                                max_cos = cos;
                                max_j = j;
                            }
                        }
                        new_odf[max_j] += data.odf[i]-data.min_odf;
                    }
                data.odf.swap(new_odf);
                float Jdet = std::abs(math::matrix_determinant(jacobian,math::dim<3,3>())*voxel_volume_scale);
                std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= Jdet);
                jdet[data.voxel_index] = Jdet;
            }
        }
    }

    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {
        mat_writer.add_matrix("free_water_quantity",&voxel.qa_scaling,1,1);
        mat_writer.add_matrix("jdet",&*jdet.begin(),1,jdet.size());
        mat_writer.add_matrix("mni",&*mni.trans_to_mni,4,3);
    }
};

#endif//MNI_RECONSTRUCTION_HPP
