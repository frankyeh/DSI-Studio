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
    std::auto_ptr<MatFile> mat_reader;
    std::vector<const unsigned short*> dwi_data;
    image::basic_image<float,3> dwi_sum;
    image::basic_image<unsigned char,3> mask;
public:
    ImageModel(void):thread_count(1) {}
    bool load_from_file(const char* dwi_file_name)
    {
        file_name = dwi_file_name;
        mat_reader.reset(new MatFile);
        if (!mat_reader->load_from_file(dwi_file_name))
        {
            error_msg = "Cannot open file";
            return false;
        }
        unsigned int row,col;

        const unsigned short* dim_ptr = 0;
        mat_reader->get_matrix("dimension",row,col,dim_ptr);
        if (!dim_ptr)
        {
            error_msg = "Cannot find dimension matrix";
            return false;
        }
        const float* voxel_size = 0;
        mat_reader->get_matrix("voxel_size",row,col,voxel_size);
        if (!voxel_size)
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
        mat_reader->get_matrix("b_table",row,col,table);
        if (!table)
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
            table += 4;
        }

        {
            // this grad_dev matrix is rotated
            const float* grad_dev = 0;
            if(mat_reader->get_matrix("grad_dev",row,col,grad_dev))
            {
                voxel.grad_dev.resize(9);
                for(unsigned int index = 0;index < 9;index++)
                    voxel.grad_dev[index] = image::make_image(voxel.dim,grad_dev+index*voxel.dim.size());
            }
        }
        voxel.q_count = col;
        dwi_data.resize(voxel.q_count);
        for (unsigned int index = 0;index < voxel.q_count;++index)
        {
            std::ostringstream out;
            out << "image" << index;
            mat_reader->get_matrix(out.str().c_str(),row,col,dwi_data[index]);
            if (!dwi_data[index])
            {
                error_msg = "Cannot find image matrix";
                return false;
            }
        }


        // create mask;
        dwi_sum.clear();
        dwi_sum.resize(voxel.dim);
        for (unsigned int index = 0;index < voxel.q_count;++index)
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
        if(mat_reader->get_matrix("mask",row,col,mask_ptr))
        {
            mask.resize(voxel.dim);
            if(row*col == voxel.dim.size())
                std::copy(mask_ptr,mask_ptr+row*col,mask.begin());
        }
        else
        {
            image::threshold(dwi_sum,mask,image::segmentation::otsu_threshold(dwi_sum)*0.8,1,0);
            check_prog(1,3);
            image::morphology::recursive_smoothing(mask);
            check_prog(2,3);
            image::morphology::defragment(mask);
            image::morphology::recursive_smoothing(mask);
        }
        check_prog(3,3);
        return true;
    }
    void save_to_file(MatFile& mat_writer)
    {

        set_title("saving");

        // dimension
        {
            short dim[3];
            dim[0] = voxel.dim[0];
            dim[1] = voxel.dim[1];
            dim[2] = voxel.dim[2];
            mat_writer.add_matrix("dimension",dim,1,3);
        }

        // voxel size
        mat_writer.add_matrix("voxel_size",&*voxel.vs.begin(),1,3);

        std::vector<float> float_data;
        std::vector<short> short_data;
        voxel.ti.save_to_buffer(float_data,short_data);
        mat_writer.add_matrix("odf_vertices",&*float_data.begin(),3,voxel.ti.vertices_count);
        mat_writer.add_matrix("odf_faces",&*short_data.begin(),3,voxel.ti.faces.size());

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
        MatFile mat_writer(output_name.c_str());
        save_to_file(mat_writer);
        voxel.end(mat_writer);
        check_prog(0,0);
        return true;
    }

};


#endif//IMAGE_MODEL_HPP
