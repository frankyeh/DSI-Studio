#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include <boost/thread.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include "gqi_process.hpp"
#include "image/image.hpp"


struct ImageModel
{
public:
    Voxel voxel;
    unsigned int thread_count;
    std::string file_name,error_msg;
    gz_mat_read mat_reader;
    std::vector<const unsigned short*> dwi_data;
    image::basic_image<float,3> dwi_sum;
    image::basic_image<unsigned char,3> mask;
public:
    // 0: x  1: y  2: z
    // 3: xy 4: yz 5: xz
    void flip(unsigned char type)
    {
        if(type < 3)
            for (unsigned int index = 0;index < voxel.bvectors.size();++index)
                voxel.bvectors[index][type] = -voxel.bvectors[index][type];
        else
            for (unsigned int index = 0;index < voxel.bvectors.size();++index)
                std::swap(voxel.bvectors[index][type%3],voxel.bvectors[index][(type+1)%3]);
        image::flip(dwi_sum,type);
        image::flip(mask,type);
        for (unsigned int index = 0;check_prog(index,dwi_data.size());++index)
        {
            image::pointer_image<unsigned short,3> I = image::make_image(voxel.dim,(unsigned short*)dwi_data[index]);
            image::flip(I,type);
        }
        voxel.dim = dwi_sum.geometry();
    }
    void rotate(image::geometry<3> new_geo,image::transformation_matrix<3,float>& affine)
    {
        if(new_geo.size() > mask.size())
            new_geo[2] = mask.size()/new_geo.plane_size();
        image::basic_image<float,3> tmp(new_geo);
        image::resample(dwi_sum,tmp,affine);
        dwi_sum = tmp;
        image::resample(mask,tmp,affine);
        mask = tmp;
        for (unsigned int index = 0;check_prog(index,dwi_data.size());++index)
        {
            image::pointer_image<unsigned short,3> I = image::make_image(voxel.dim,(unsigned short*)dwi_data[index]);
            image::resample(I,tmp,affine);
            I.resize(new_geo);
            std::copy(tmp.begin(),tmp.end(),I.begin());
        }
        // rotate b-table
        float iT[9];
        image::matrix::inverse(affine.scaling_rotation,iT,image::dim<3,3>());
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            image::vector<3> tmp;
            image::vector_rotation(voxel.bvectors[index].begin(),tmp.begin(),iT,image::vdim<3>());
            tmp.normalize();
            voxel.bvectors[index] = tmp;
        }
        image::morphology::smoothing(mask);
        voxel.dim = new_geo;
    }
    void trim(void)
    {
        image::geometry<3> range_min,range_max;
        image::bounding_box(mask,range_min,range_max,0);
        for (unsigned int index = 0;check_prog(index,dwi_data.size());++index)
        {
            image::pointer_image<unsigned short,3> I = image::make_image(voxel.dim,(unsigned short*)dwi_data[index]);
            image::basic_image<unsigned short,3> I0 = I;
            image::crop(I0,range_min,range_max);
            std::fill(I.begin(),I.end(),0);
            std::copy(I0.begin(),I0.end(),I.begin());
        }
        image::crop(dwi_sum,range_min,range_max);
        image::crop(mask,range_min,range_max);
        voxel.dim = mask.geometry();

    }

public:
    ImageModel(void):thread_count(1) {}
    bool load_from_file(const char* dwi_file_name)
    {
        file_name = dwi_file_name;
        if (!mat_reader.load_from_file(dwi_file_name))
        {
            error_msg = "Cannot open file";
            return false;
        }
        unsigned int row,col;

        const unsigned short* dim_ptr = 0;
        if (!mat_reader.read("dimension",row,col,dim_ptr))
        {
            error_msg = "Cannot find dimension matrix";
            return false;
        }
        const float* voxel_size = 0;
        if (!mat_reader.read("voxel_size",row,col,voxel_size))
        {
            //error_msg = "Cannot find voxel size matrix";
            //return false;
            std::fill(voxel.vs.begin(),voxel.vs.end(),3.0);
        }
        else
            std::copy(voxel_size,voxel_size+3,voxel.vs.begin());

        if (dim_ptr[0]*dim_ptr[1]*dim_ptr[2] <= 0)
        {
            error_msg = "Invalid dimension setting";
            return false;
        }
        voxel.dim[0] = dim_ptr[0];
        voxel.dim[1] = dim_ptr[1];
        voxel.dim[2] = dim_ptr[2];

        const float* table;
        if (!mat_reader.read("b_table",row,col,table))
        {
            error_msg = "Cannot find b_table matrix";
            return false;
        }
        voxel.bvalues.resize(col);
        voxel.bvectors.resize(col);
        for (unsigned int index = 0;index < col;++index)
        {
            voxel.bvalues[index] = table[0];
            voxel.bvectors[index][0] = table[1];
            voxel.bvectors[index][1] = table[2];
            voxel.bvectors[index][2] = table[3];
            voxel.bvectors[index].normalize();
            table += 4;
        }

        dwi_data.resize(voxel.bvalues.size());
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            std::ostringstream out;
            out << "image" << index;
            mat_reader.read(out.str().c_str(),row,col,dwi_data[index]);
            if (!dwi_data[index])
            {
                error_msg = "Cannot find image matrix";
                return false;
            }
        }


        {
            // this grad_dev matrix is rotated
            const float* grad_dev = 0;
            if(mat_reader.read("grad_dev",row,col,grad_dev) && row*col == voxel.dim.size()*9)
            {
                for(unsigned int index = 0;index < 9;index++)
                    voxel.grad_dev.push_back(image::make_image(voxel.dim,grad_dev+index*voxel.dim.size()));
            }
        }

        // create mask;
        dwi_sum.clear();
        dwi_sum.resize(voxel.dim);
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
            image::add(dwi_sum.begin(),dwi_sum.end(),dwi_data[index]);

        float max_value = *std::max_element(dwi_sum.begin(),dwi_sum.end());
        float min_value = max_value;
        for (unsigned int index = 0;index < dwi_sum.size();++index)
            if (dwi_sum[index] < min_value && dwi_sum[index] > 0)
                min_value = dwi_sum[index];


        ::set_title("creating mask");
        check_prog(0,3);
        image::minus_constant(dwi_sum,min_value);
        image::lower_threshold(dwi_sum,0.0f);
        image::normalize(dwi_sum,1.0);
        image::add_constant(dwi_sum,1.0);
        image::log(dwi_sum);
        image::divide_constant(dwi_sum,0.301);
        image::upper_threshold(dwi_sum,1.0f);


        const unsigned char* mask_ptr = 0;
        if(mat_reader.read("mask",row,col,mask_ptr))
        {
            mask.resize(voxel.dim);
            if(row*col == voxel.dim.size())
                std::copy(mask_ptr,mask_ptr+row*col,mask.begin());
        }
        else
        {
            image::threshold(dwi_sum,mask,image::segmentation::otsu_threshold(dwi_sum)*0.8,1,0);
            check_prog(1,3);
            image::morphology::recursive_smoothing(mask,10);
            check_prog(2,3);
            image::morphology::defragment(mask);
            image::morphology::recursive_smoothing(mask,10);
        }
        check_prog(3,3);
        return true;
    }
    void save_to_file(gz_mat_write& mat_writer)
    {

        set_title("saving");

        // dimension
        {
            short dim[3];
            dim[0] = voxel.dim[0];
            dim[1] = voxel.dim[1];
            dim[2] = voxel.dim[2];
            mat_writer.write("dimension",dim,1,3);
        }

        // voxel size
        mat_writer.write("voxel_size",&*voxel.vs.begin(),1,3);

        std::vector<float> float_data;
        std::vector<short> short_data;
        voxel.ti.save_to_buffer(float_data,short_data);
        mat_writer.write("odf_vertices",&*float_data.begin(),3,voxel.ti.vertices_count);
        mat_writer.write("odf_faces",&*short_data.begin(),3,voxel.ti.faces.size());

    }
public:
    template<typename CheckType>
    bool avaliable(void) const
    {
        return CheckType::check(voxel);
    }

    template<typename ProcessType>
    bool reconstruct(void)
    {
        begin_prog("reconstruction");
        voxel.image_model = this;
        voxel.CreateProcesses<ProcessType>();
        voxel.init(thread_count);
        boost::thread_group threads;
        for (unsigned int index = 1;index < thread_count;++index)
            threads.add_thread(new boost::thread(&Voxel::thread_run,&voxel,
                                                 index,thread_count,mask));
        voxel.thread_run(0,thread_count,mask);
        threads.join_all();
        return !prog_aborted();
    }


    template<typename ProcessType>
    bool reconstruct(const std::string& ext)
    {
        if (!reconstruct<ProcessType>())
            return false;
        std::string output_name = file_name;
        output_name += ext;
        begin_prog("saving data");
        gz_mat_write mat_writer(output_name.c_str());
        save_to_file(mat_writer);
        voxel.end(mat_writer);
        return true;
    }

};


#endif//IMAGE_MODEL_HPP
