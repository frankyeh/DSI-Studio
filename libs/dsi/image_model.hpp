#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include <boost/thread.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include "gqi_process.hpp"
#include "image/image.hpp"

void get_report(const std::vector<float>& bvalues,image::vector<3> vs,std::string& report);
struct ImageModel
{
private:
    std::vector<image::basic_image<unsigned short,3> > new_dwi;//used in rotated volume
public:
    Voxel voxel;
    std::string file_name,error_msg;
    gz_mat_read mat_reader;
    std::vector<const unsigned short*> dwi_data;
    image::basic_image<unsigned char,3> mask;
public:
    void flip_b_table(unsigned char dim)
    {
        for(unsigned int index = 0;index < voxel.bvectors.size();++index)
            voxel.bvectors[index][dim] = -voxel.bvectors[index][dim];
        if(!voxel.grad_dev.empty())
        {
            // <Flip*Gra_dev*b_table,ODF>
            // = <(Flip*Gra_dev*inv(Flip))*Flip*b_table,ODF>
            unsigned char nindex[3][4] = {{1,2,3,6},{1,3,5,7},{2,5,6,7}};
            for(unsigned int index = 0;index < 4;++index)
            {
                // 1  0  0         1  0  0
                //[0 -1  0] *Grad*[0 -1  0]
                // 0  0  1         0  0  1
                unsigned char pos = nindex[dim][index];
                for(unsigned int i = 0;i < voxel.dim.size();++i)
                    voxel.grad_dev[pos][i] = -voxel.grad_dev[pos][i];
            }
        }
    }
    // 0:xy 1:yz 2: xz
    void rotate_b_table(unsigned char dim)
    {
        for (unsigned int index = 0;index < voxel.bvectors.size();++index)
            std::swap(voxel.bvectors[index][dim],voxel.bvectors[index][(dim+1)%3]);
        if(!voxel.grad_dev.empty())
        {
            unsigned char swap1[3][6] = {{0,3,6,0,1,2},{1,4,7,3,4,5},{0,3,6,0,1,2}};
            unsigned char swap2[3][6] = {{1,4,7,3,4,5},{2,5,8,6,7,8},{2,5,8,6,7,8}};
            for(unsigned int index = 0;index < 6;++index)
            {
                unsigned char s1 = swap1[dim][index];
                unsigned char s2 = swap2[dim][index];
                for(unsigned int i = 0;i < voxel.dim.size();++i)
                    std::swap(voxel.grad_dev[s1][i],voxel.grad_dev[s2][i]);
            }
        }
    }

    // 0: x  1: y  2: z
    // 3: xy 4: yz 5: xz
    void flip(unsigned char type)
    {
        if(type < 3)
            flip_b_table(type);
        else
            rotate_b_table(type-3);
        image::flip(voxel.dwi_sum,type);
        image::flip(mask,type);
        for(unsigned int i = 0;i < voxel.grad_dev.size();++i)
            image::flip(image::make_image(voxel.dim,(float*)&*(voxel.grad_dev[i].begin())),type);
        for (unsigned int index = 0;check_prog(index,dwi_data.size());++index)
            image::flip(image::make_image(voxel.dim,(unsigned short*)dwi_data[index]),type);
        voxel.dim = voxel.dwi_sum.geometry();
    }
    // used in eddy correction for each dwi
    void rotate_dwi(unsigned int dwi_index,const image::transformation_matrix<3,float>& affine)
    {
        image::basic_image<float,3> tmp(voxel.dim);
        image::pointer_image<unsigned short,3> I = image::make_image(voxel.dim,(unsigned short*)dwi_data[dwi_index]);
        image::resample(I,tmp,affine,image::cubic);
        std::copy(tmp.begin(),tmp.end(),I.begin());


        // rotate b-table
        float iT[9];
        image::matrix::inverse(affine.scaling_rotation,iT,image::dim<3,3>());
        image::vector<3> v;
        image::vector_rotation(voxel.bvectors[dwi_index].begin(),v.begin(),iT,image::vdim<3>());
        v.normalize();
        voxel.bvectors[dwi_index] = v;
    }

    void rotate(image::geometry<3> new_geo,image::transformation_matrix<3,float>& affine)
    {
        std::vector<image::basic_image<unsigned short,3> > dwi(dwi_data.size());
        for (unsigned int index = 0;check_prog(index,dwi_data.size());++index)
        {
            dwi[index].resize(new_geo);
            image::resample(image::make_image(voxel.dim,(unsigned short*)dwi_data[index]),dwi[index],affine,image::cubic);
            dwi_data[index] = &(dwi[index][0]);
        }
        dwi.swap(new_dwi);

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

        if(!voxel.grad_dev.empty())
        {
            // <R*Gra_dev*b_table,ODF>
            // = <(R*Gra_dev*inv(R))*R*b_table,ODF>
            float det = std::abs(image::matrix::determinant(iT,image::dim<3,3>()));
            begin_prog("rotating grad_dev");
            for(unsigned int index = 0;check_prog(index,voxel.dim.size());++index)
            {
                float grad_dev[9],G_invR[9];
                for(unsigned int i = 0; i < 9; ++i)
                    grad_dev[i] = voxel.grad_dev[i][index];
                image::matrix::product(grad_dev,affine.scaling_rotation,G_invR,image::dim<3,3>(),image::dim<3,3>());
                image::matrix::product(iT,G_invR,grad_dev,image::dim<3,3>(),image::dim<3,3>());
                for(unsigned int i = 0; i < 9; ++i)
                    voxel.grad_dev[i][index] = grad_dev[i]/det;
            }
            std::vector<image::basic_image<float,3> > new_gra_dev(voxel.grad_dev.size());
            begin_prog("rotating grad_dev volume");
            for (unsigned int index = 0;check_prog(index,new_gra_dev.size());++index)
            {
                new_gra_dev[index].resize(new_geo);
                image::resample(voxel.grad_dev[index],new_gra_dev[index],affine,image::cubic);
                voxel.grad_dev[index] = image::make_image(voxel.dim,(float*)&(new_gra_dev[index][0]));
            }
            new_gra_dev.swap(voxel.new_grad_dev);
        }
        voxel.dim = new_geo;
        calculate_dwi_sum();
        calculate_mask();

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
        calculate_dwi_sum();
        image::crop(mask,range_min,range_max);
        voxel.dim = mask.geometry();

    }
    bool save_to_nii(const char* nifti_file_name) const
    {
        gz_nifti header;
        float vs[4];
        std::copy(voxel.vs.begin(),voxel.vs.end(),vs);
        vs[3] = 1.0;
        header.set_voxel_size(vs);
        image::geometry<4> nifti_dim;
        std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
        nifti_dim[3] = voxel.bvalues.size();
        image::basic_image<unsigned short,4> buffer(nifti_dim);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            std::copy(dwi_data[index],
                      dwi_data[index]+voxel.dim.size(),
                      buffer.begin() + (size_t)index*voxel.dim.size());
        }
        image::flip_xy(buffer);
        header << buffer;
        return header.save_to_file(nifti_file_name);
    }
    bool save_b_table(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            out << voxel.bvalues[index] << " "
                << voxel.bvectors[index][0] << " "
                << voxel.bvectors[index][1] << " "
                << voxel.bvectors[index][2] << std::endl;
        }
        return out.good();
    }
    bool save_bval(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            if(index)
                out << " ";
            out << voxel.bvalues[index];
        }
        return out.good();
    }
    bool save_bvec(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            out << voxel.bvectors[index][0] << " "
                << voxel.bvectors[index][1] << " "
                << voxel.bvectors[index][2] << std::endl;
        }
        return out.good();
    }


public:
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

        const char* report_buf = 0;
        if(mat_reader.read("report",row,col,report_buf))
            voxel.report = std::string(report_buf,report_buf+row*col);
        else
            get_report(voxel.bvalues,voxel.vs,voxel.report);

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
                    voxel.grad_dev.push_back(image::make_image(voxel.dim,(float*)grad_dev+index*voxel.dim.size()));
            }
        }

        // create mask;
        calculate_dwi_sum();

        const unsigned char* mask_ptr = 0;
        if(mat_reader.read("mask",row,col,mask_ptr))
        {
            mask.resize(voxel.dim);
            if(row*col == voxel.dim.size())
                std::copy(mask_ptr,mask_ptr+row*col,mask.begin());
        }
        else
            calculate_mask();

        return true;
    }
    void calculate_dwi_sum(void)
    {
        voxel.dwi_sum.clear();
        voxel.dwi_sum.resize(voxel.dim);
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
            image::add(voxel.dwi_sum.begin(),voxel.dwi_sum.end(),dwi_data[index]);

        float max_value = *std::max_element(voxel.dwi_sum.begin(),voxel.dwi_sum.end());
        float min_value = max_value;
        for (unsigned int index = 0;index < voxel.dwi_sum.size();++index)
            if (voxel.dwi_sum[index] < min_value && voxel.dwi_sum[index] > 0)
                min_value = voxel.dwi_sum[index];


        image::minus_constant(voxel.dwi_sum,min_value);
        image::lower_threshold(voxel.dwi_sum,0.0f);
        image::normalize(voxel.dwi_sum,1.0);
        image::add_constant(voxel.dwi_sum,1.0);
        image::log(voxel.dwi_sum);
        image::divide_constant(voxel.dwi_sum,0.301);
        image::upper_threshold(voxel.dwi_sum,1.0f);
    }
    void calculate_mask(void)
    {
        image::threshold(voxel.dwi_sum,mask,image::segmentation::otsu_threshold(voxel.dwi_sum)*0.8,1,0);
        image::morphology::recursive_smoothing(mask,10);
        image::morphology::defragment(mask);
        image::morphology::recursive_smoothing(mask,10);
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
    bool reconstruct(unsigned int thread_count)
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


    void save_fib(const std::string& ext)
    {
        std::string output_name = file_name;
        output_name += ext;
        begin_prog("saving data");
        gz_mat_write mat_writer(output_name.c_str());
        save_to_file(mat_writer);
        voxel.end(mat_writer);
        std::string final_report = voxel.report.c_str();
        final_report += voxel.recon_report.str();
        mat_writer.write("report",final_report.c_str(),1,final_report.length());
    }

};


#endif//IMAGE_MODEL_HPP
