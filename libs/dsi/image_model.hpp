#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include <boost/thread.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include "math/matrix_op.hpp"
#include "gqi_process.hpp"
#include "image/image.hpp"


struct ImageModel
{
public:
    Voxel voxel;
    unsigned int thread_count;
    std::string file_name;
    std::auto_ptr<MatFile> mat_reader;
    std::vector<const unsigned short*> dwi_data;
    image::basic_image<unsigned char,3> mask;
public:
    ImageModel(void):thread_count(1) {}
    bool set_dimension(unsigned int w,unsigned int h,unsigned int d)
    {
        voxel.matrix_width = w;
        voxel.matrix_height = h;
        voxel.slice_number = d;
        voxel.total_size = w*h*d;
        mask.clear();
        mask.resize(image::geometry<3>(w,h,d));
        if (voxel.total_size <= 0)
            return false;
        return true;
    }
    bool load_from_file(const char* dwi_file_name)
    {
        file_name = dwi_file_name;
        mat_reader.reset(new MatFile);
        if (!mat_reader->load_from_file(dwi_file_name))
            return false;
        unsigned int row,col;

        const unsigned short* dim = 0;
        mat_reader->get_matrix("dimension",row,col,dim);
        if (!dim)
            return false;

        const float* voxel_size = 0;
        mat_reader->get_matrix("voxel_size",row,col,voxel_size);
        if (voxel_size)
            std::copy(voxel_size,voxel_size+3,voxel.voxel_size);
        else
            std::fill(voxel.voxel_size,voxel.voxel_size+3,3.0);

        if (!set_dimension(dim[0],dim[1],dim[2]))
            return false;
        const float* table;
        mat_reader->get_matrix("b_table",row,col,table);
        if (!table)
            return false;
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

        voxel.q_count = col;
        dwi_data.resize(voxel.q_count);
        for (unsigned int index = 0;index < voxel.q_count;++index)
        {
            std::ostringstream out;
            out << "image" << index;
            mat_reader->get_matrix(out.str().c_str(),row,col,dwi_data[index]);
            if (!dwi_data[index])
                return false;
        }

        // create mask;
        std::vector<unsigned int> sum_image(voxel.total_size);


        set_title("generating mask");
        for (unsigned int index = 0;check_prog(index,voxel.q_count);++index)
            image::add(sum_image.begin(),sum_image.end(),dwi_data[index]);

        unsigned int range = *std::max_element(sum_image.begin(),sum_image.end());
        {
            unsigned int min_value = range;
            for (unsigned int index = 0;index < sum_image.size();++index)
                if (sum_image[index] < min_value && sum_image[index] > 0)
                    min_value = sum_image[index];
            range -= min_value;
            for (unsigned int index = 0;index < sum_image.size();++index)
                if (sum_image[index] > 0)
                    sum_image[index] -= min_value;
        }

        if (range)
            for (unsigned int index = 0;index < sum_image.size();++index)
            {
                unsigned int value = std::floor(std::log((float)sum_image[index]/(float)range+1.0)/0.301*255.0);
                if (value >= 256)
                    value = 255;
                mask[index] = (unsigned char)value;
            }
        return true;
    }
    void save_to_file(MatFile& mat_writer)
    {

        set_title("saving");

        // dimension
        {
            short dim[3];
            dim[0] = voxel.matrix_width;
            dim[1] = voxel.matrix_height;
            dim[2] = voxel.slice_number;
            mat_writer.add_matrix("dimension",dim,1,3);
        }

        // voxel size
        mat_writer.add_matrix("voxel_size",voxel.voxel_size,1,3);

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
        bool terminated = false;
        for (unsigned int index = 1;index < thread_count;++index)
            threads.add_thread(new boost::thread(&Voxel::thread_run,&voxel,
                                                 index,thread_count,mask,terminated));
        voxel.thread_run(0,thread_count,mask,terminated);
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
