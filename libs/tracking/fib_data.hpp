#ifndef FIB_DATA_HPP
#define FIB_DATA_HPP
#include <fstream>
#include <sstream>
#include <string>
#include "prog_interface_static_link.h"
#include "image/image.hpp"
#include "gzip_interface.hpp"

struct ODFData{
private:
    const float* odfs;
    unsigned int odfs_size;
private:
    image::basic_image<unsigned int,3> voxel_index_map;
    std::vector<const float*> odf_blocks;
    std::vector<unsigned int> odf_block_size;
    image::basic_image<unsigned int,3> odf_block_map1;
    image::basic_image<unsigned int,3> odf_block_map2;
    unsigned int half_odf_size;
public:
    ODFData(void):odfs(0){}
    void read(gz_mat_read& mat_reader)
    {
        unsigned int row,col;
        {
            const float* odfs = 0;
            if(mat_reader.read("odfs",row,col,odfs))
            {
                setODFs(odfs,row*col);
                return;
            }
        }
        for(unsigned int index = 0;;++index)
        {
            const float* odf = 0;
            std::ostringstream out;
            out << "odf" << index;
            std::string name = out.str();
            if(!mat_reader.read(name.c_str(),row,col,odf))
                return;
            setODF(index,odf,row*col);
        }
    }

    void setODFs(const float* odfs_,unsigned int odfs_size_)
    {
        odfs = odfs_;
        odfs_size = odfs_size_;
    }
    void setODF(unsigned int store_index,const float* odf,unsigned int size)
    {
        if(odf_blocks.size() <= store_index)
        {
            odf_blocks.resize(store_index+1);
            odf_block_size.resize(store_index+1);
        }
        odf_blocks[store_index] = odf;
        odf_block_size[store_index] = size;
    }
    void initializeODF(const image::geometry<3>& dim,const float* fa0,unsigned int half_odf_size_)
    {
        half_odf_size = half_odf_size_;
        // handle the odf mappings
        if (odfs)
        {
            voxel_index_map.resize(dim);
            for (unsigned int index = 0,j = 0;index < voxel_index_map.size();++index)
            {
                if (fa0[index] == 0.0)
                {
                    unsigned int from = j*(half_odf_size);
                    unsigned int to = from + half_odf_size;
                    if (to > odfs_size)
                        break;
                    bool odf_is_zero = true;
                    for (;from < to;++from)
                        if (odfs[from] != 0.0)
                        {
                            odf_is_zero = false;
                            break;
                        }
                    if (!odf_is_zero)
                        continue;
                }
                ++j;
                voxel_index_map[index] = j;
            }
        }

        if (!odf_blocks.empty())
        {
            odf_block_map1.resize(dim);
            odf_block_map2.resize(dim);

            int voxel_index = 0;
            for(unsigned int i = 0;i < odf_block_size.size();++i)
                for(unsigned int j = 0;j < odf_block_size[i];j += half_odf_size)
                {
                    unsigned int k_end = j + half_odf_size;
                    bool is_odf_zero = true;
                    for(unsigned int k = j;k < k_end;++k)
                        if(odf_blocks[i][k] != 0.0)
                        {
                            is_odf_zero = false;
                            break;
                        }
                    if(!is_odf_zero)
                        for(;voxel_index < odf_block_map1.size();++voxel_index)
                            if(fa0[voxel_index] != 0.0)
                                break;
                    if(voxel_index >= odf_block_map1.size())
                        break;
                    odf_block_map1[voxel_index] = i;
                    odf_block_map2[voxel_index] = j;
                    ++voxel_index;
                }
        }
    }
    // odf functions
    bool has_odfs(void) const
    {
        return odfs != 0 || !odf_blocks.empty();
    }
    const float* get_odf_data(unsigned int index) const
    {
        if (odfs != 0)
        {
            if (index >= voxel_index_map.size() || voxel_index_map[index] == 0)
                return 0;
            return odfs+(voxel_index_map[index]-1)*half_odf_size;
        }

        if (!odf_blocks.empty())
        {
            if (index >= odf_block_map2.size())
                return 0;
            return odf_blocks[odf_block_map1[index]] + odf_block_map2[index];
        }
        return 0;
    }
};

class FiberDirection
{
public:
    image::geometry<3> dim;
    std::vector<const float*> dir;
    std::vector<const short*> findex;
    std::vector<std::vector<short> > findex_buf;
public:
    std::vector<std::string> index_name;
    std::vector<std::vector<const float*> > index_data;
    std::vector<std::vector<const short*> > index_data_dir;

public:
    std::vector<const float*> fa;

    std::vector<image::vector<3,float> > odf_table;
    std::vector<image::vector<3,unsigned short> > odf_faces;

    unsigned int num_fiber;
    unsigned int half_odf_size;

    std::string error_msg;
private:
    void check_index(unsigned int index)
    {
        if (fa.size() <= index)
        {
            ++index;
            fa.resize(index);
            findex.resize(index);
            num_fiber = index;
        }
    }
public:

    bool add_data(gz_mat_read& mat_reader)
    {
        unsigned int row,col;
        // dimension
        {
            const unsigned short* dim_buf = 0;
            if (!mat_reader.read("dimension",row,col,dim_buf))
            {
                error_msg = "cannot find dimension matrix";
                return false;
            }
            std::copy(dim_buf,dim_buf+3,dim.begin());
        }
        // odf_vertices
        {
            const float* odf_buffer;
            if (mat_reader.read("odf_vertices",row,col,odf_buffer))
            {
                    odf_table.resize(col);
                for (unsigned int index = 0;index < odf_table.size();++index,odf_buffer += 3)
                {
                    odf_table[index][0] = odf_buffer[0];
                    odf_table[index][1] = odf_buffer[1];
                    odf_table[index][2] = odf_buffer[2];
                }
                half_odf_size = col / 2;
            }
            else
            {
                odf_table.resize(2);
                half_odf_size = 1;
                odf_faces.clear();
            }
        }
        // odf_faces
        {
            const unsigned short* odf_buffer;
            if(mat_reader.read("odf_faces",row,col,odf_buffer))
            {
                odf_faces.resize(col);
                for (unsigned int index = 0;index < odf_faces.size();++index,odf_buffer += 3)
                {
                    odf_faces[index][0] = odf_buffer[0];
                    odf_faces[index][1] = odf_buffer[1];
                    odf_faces[index][2] = odf_buffer[2];
                }
            }
        }

        for (unsigned int index = 0;check_prog(index,mat_reader.size());++index)
        {
            std::string matrix_name = mat_reader.name(index);
            if (matrix_name == "image")
            {
                check_index(0);
                mat_reader.read(index,row,col,fa[0]);
                findex_buf.resize(1);
                findex_buf[0].resize(row*col);
                findex[0] = &*(findex_buf[0].begin());
                odf_table.resize(2);
                half_odf_size = 1;
                odf_faces.clear();
                continue;
            }

            // prefix started here
            std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
            int store_index  = matrix_name[matrix_name.length()-1]-'0';
            if (store_index < 0 || store_index > 9)
                continue;

            if (prefix_name == "index")
            {
                check_index(store_index);
                mat_reader.read(index,row,col,findex[store_index]);
                continue;
            }
            if (prefix_name == "fa")
            {
                check_index(store_index);
                mat_reader.read(index,row,col,fa[store_index]);
                continue;
            }
            if (prefix_name == "dir")
            {
                const float* dir_ptr;
                mat_reader.read(index,row,col,dir_ptr);
                check_index(store_index);
                dir.resize(findex.size());
                dir[store_index] = dir_ptr;
                continue;
            }

            int prefix_name_index = 0;
            for(;prefix_name_index < index_name.size();++prefix_name_index)
                if(index_name[prefix_name_index] == prefix_name)
                    break;
            if(prefix_name_index == index_name.size())
            {
                index_name.push_back(prefix_name);
                index_data.push_back(std::vector<const float*>());
                index_data_dir.push_back(std::vector<const short*>());

            }

            if(index_data[prefix_name_index].size() <= store_index)
                index_data[prefix_name_index].resize(store_index+1);
            mat_reader.read(index,row,col,index_data[prefix_name_index][store_index]);

        }



        // adding the primary fiber index
        index_name.insert(index_name.begin(),fa.size() == 1 ? "fa":"qa");
        index_data.insert(index_data.begin(),fa);
        index_data_dir.insert(index_data_dir.begin(),findex);

        // check index_data integrity
        for(int index = 1;index < index_data.size();++index)
        {
            index_data_dir[index] = findex;
            for(int j = 0;j < index_data[index].size();++j)
                if(!index_data[index][j] || index_data[index].size() != num_fiber)
                {
                    index_data.erase(index_data.begin()+index);
                    index_data_dir.erase(index_data_dir.begin()+index);
                    index_name.erase(index_name.begin()+index);
                    --index;
                    break;
                }
        }


        if(num_fiber == 0)
            error_msg = "No image data found";
        return num_fiber;
    }

    bool set_tracking_index(int new_index)
    {
        if(new_index >= index_data.size())
            return false;
        fa = index_data[new_index];
        findex = index_data_dir[new_index];
        return true;
    }

    bool set_tracking_index(const std::string& name)
    {
        return set_tracking_index(std::find(index_name.begin(),index_name.end(),name)-index_name.begin());
    }
    float getFA(unsigned int index,unsigned char order) const
    {
        if(order >= fa.size())
            return 0.0;
        return fa[order][index];
    }
    const float* getDir(unsigned int index,unsigned int order) const
    {
        if(!dir.empty())
            return dir[order] + index + (index << 1);
        if(order >= findex.size())
            return &*(odf_table[0].begin());
        return &*(odf_table[findex[order][index]].begin());
    }

};



struct ViewItem
{
    std::string name;
    image::const_pointer_image<float,3> image_data;
    float max_value;
    float min_value;
    image::basic_image<image::rgb_color,3> data_buf;
    // used in QSDR
    image::const_pointer_image<float,3> mx,my,mz;
    image::geometry<3> native_geo;
    template<typename input_iterator>
    void set_scale(input_iterator from,input_iterator to)
    {
        max_value = *std::max_element(from,to);
        min_value = *std::min_element(from,to);
        if(max_value == min_value)
        {
            min_value = 0;
            max_value = 1;
        }
    }
};

class FibData
{
public:
    mutable std::string error_msg;
    std::string report;
    gz_mat_read mat_reader;
    FiberDirection fib;
    ODFData odf;
public:
    std::string subject_report;
    std::vector<std::string> subject_names;
    unsigned int num_subjects;
    std::vector<const float*> subject_qa;
    unsigned int subject_qa_length;
    std::vector<float> subject_qa_max;
    std::vector<float> R2;
    std::vector<unsigned int> vi2si;
    std::vector<unsigned int> si2vi;
    std::vector<std::vector<float> > subject_qa_buf;// merged from other db
    void read_db(void)
    {
        subject_qa.clear();
        subject_qa_max.clear();
        unsigned int row,col;
        for(unsigned int index = 0;1;++index)
        {
            std::ostringstream out;
            out << "subject" << index;
            const float* buf = 0;
            mat_reader.read(out.str().c_str(),row,col,buf);
            if (!buf)
                break;
            if(!index)
                subject_qa_length = row*col;
            subject_qa.push_back(buf);
            subject_qa_max.push_back(*std::max_element(buf,buf+col*row));
            if(subject_qa_max.back() == 0.0)
                subject_qa_max.back() = 1.0;
        }
        num_subjects = (unsigned int)subject_qa.size();
        subject_names.resize(num_subjects);
        R2.resize(num_subjects);
        if(!num_subjects)
            return;
        {
            const char* str = 0;
            mat_reader.read("subject_names",row,col,str);
            if(str)
            {
                std::istringstream in(str);
                for(unsigned int index = 0;in && index < num_subjects;++index)
                    std::getline(in,subject_names[index]);
            }
            const float* r2_values = 0;
            mat_reader.read("R2",row,col,r2_values);
            if(r2_values == 0)
            {
                error_msg = "Memory insufficiency. Use 64-bit program instead";
                num_subjects = 0;
                subject_qa.clear();
                return;
            }
            std::copy(r2_values,r2_values+num_subjects,R2.begin());
        }

        calculate_si2vi();
    }
    void remove_subject(unsigned int index)
    {
        if(index >= subject_qa.size())
            return;
        subject_qa.erase(subject_qa.begin()+index);
        subject_qa_max.erase(subject_qa_max.begin()+index);
        subject_names.erase(subject_names.begin()+index);
        R2.erase(R2.begin()+index);
        --num_subjects;
    }

    void calculate_si2vi(void)
    {
        vi2si.resize(dim.size());
        for(unsigned int index = 0;index < (unsigned int)dim.size();++index)
        {
            if(fib.fa[0][index] != 0.0)
            {
                vi2si[index] = (unsigned int)si2vi.size();
                si2vi.push_back(index);
            }
        }
    }

    bool sample_odf(gz_mat_read& m,std::vector<float>& data)
    {
        ODFData subject_odf;
        for(unsigned int index = 0;1;++index)
        {
            std::ostringstream out;
            out << "odf" << index;
            const float* odf = 0;
            unsigned int row,col;
            m.read(out.str().c_str(),row,col,odf);
            if (!odf)
                break;
            subject_odf.setODF(index,odf,row*col);
        }

        const float* fa0 = 0;
        unsigned int row,col;
        m.read("fa0",row,col,fa0);
        if (!fa0)
        {
            error_msg = "Invalid file format. Cannot find fa0 matrix in ";
            return false;
        }
        if(*std::max_element(fa0,fa0+row*col) == 0)
        {
            error_msg = "Invalid file format. No image data in ";
            return false;
        }
        if(!subject_odf.has_odfs())
        {
            error_msg = "No ODF data in the subject file:";
            return false;
        }
        subject_odf.initializeODF(dim,fa0,fib.half_odf_size);

        set_title("load data");
        for(unsigned int index = 0;index < si2vi.size();++index)
        {
            unsigned int cur_index = si2vi[index];
            const float* odf = subject_odf.get_odf_data(cur_index);
            if(odf == 0)
                continue;
            float min_value = *std::min_element(odf, odf + fib.half_odf_size);
            unsigned int pos = index;
            for(unsigned char i = 0;i < fib.num_fiber;++i,pos += (unsigned int)si2vi.size())
            {
                if(fib.fa[i][cur_index] == 0.0)
                    break;
                // 0: subject index 1:findex by s_index (fa > 0)
                data[pos] = odf[fib.findex[i][cur_index]]-min_value;
            }
        }
        return true;
    }
    bool is_consistent(gz_mat_read& m)
    {
        unsigned int row,col;
        const float* odf_buffer;
        m.read("odf_vertices",row,col,odf_buffer);
        if (!odf_buffer)
        {
            error_msg = "No odf_vertices matrix in ";
            return false;
        }
        if(col != fib.odf_table.size())
        {
            error_msg = "Inconsistent ODF dimension in ";
            return false;
        }
        for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
        {
            if(fib.odf_table[index][0] != odf_buffer[0] ||
               fib.odf_table[index][1] != odf_buffer[1] ||
               fib.odf_table[index][2] != odf_buffer[2])
            {
                error_msg = "Inconsistent ODF dimension in ";
                return false;
            }
        }
        return true;
    }

    bool load_subject_files(const std::vector<std::string>& file_names,
                                          const std::vector<std::string>& subject_names_)
    {
        num_subjects = (unsigned int)file_names.size();
        subject_qa.clear();
        subject_qa.resize(num_subjects);
        subject_qa_buf.resize(num_subjects);
        R2.resize(num_subjects);
        for(unsigned int index = 0;index < num_subjects;++index)
            subject_qa_buf[index].resize(fib.num_fiber*si2vi.size());
        for(unsigned int index = 0;index < num_subjects;++index)
            subject_qa[index] = &(subject_qa_buf[index][0]);
        for(unsigned int subject_index = 0;check_prog(subject_index,num_subjects);++subject_index)
        {
            if(prog_aborted())
            {
                check_prog(1,1);
                return false;
            }
            gz_mat_read m;
            if(!m.load_from_file(file_names[subject_index].c_str()))
            {
                error_msg = "failed to load subject data ";
                error_msg += file_names[subject_index];
                return false;
            }
            // check if the odf table is consistent or not
            if(!is_consistent(m) ||
               !sample_odf(m,subject_qa_buf[subject_index]))
            {
                error_msg += file_names[subject_index];
                return false;
            }
            // load R2
            const float* value= 0;
            unsigned int row,col;
            m.read("R2",row,col,value);
            if(!value || *value != *value)
            {
                error_msg = "Invalid R2 value in ";
                error_msg += file_names[subject_index];
                return false;
            }
            R2[subject_index] = *value;
            if(subject_index == 0)
            {
                const char* report_buf = 0;
                if(m.read("report",row,col,report_buf))
                    subject_report = std::string(report_buf,report_buf+row*col);
            }
        }
        subject_names = subject_names_;
        return true;
    }
    void get_subject_vector(std::vector<std::vector<float> >& subject_vector,
                            const image::basic_image<int,3>& cerebrum_mask) const
    {
        float fiber_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(dim,fib.fa[0]));
        subject_vector.clear();
        subject_vector.resize(num_subjects);
        for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
        {
            unsigned int cur_index = si2vi[s_index];
            if(!cerebrum_mask[cur_index])
                continue;
            for(unsigned int j = 0,fib_offset = 0;j < fib.num_fiber && fib.fa[j][cur_index] > fiber_threshold;
                    ++j,fib_offset+=si2vi.size())
            {
                unsigned int pos = s_index + fib_offset;
                for(unsigned int index = 0;index < num_subjects;++index)
                    subject_vector[index].push_back(subject_qa[index][pos]);
            }
        }
        for(unsigned int index = 0;index < num_subjects;++index)
        {
            image::minus_constant(subject_vector[index].begin(),subject_vector[index].end(),image::mean(subject_vector[index].begin(),subject_vector[index].end()));
            float sd = image::standard_deviation(subject_vector[index].begin(),subject_vector[index].end());
            if(sd > 0.0)
                image::divide_constant(subject_vector[index].begin(),subject_vector[index].end(),sd);
        }
    }
    void get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                            const image::basic_image<int,3>& cerebrum_mask) const
    {
        float fiber_threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(dim,fib.fa[0]));
        subject_vector.clear();
        for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
        {
            unsigned int cur_index = si2vi[s_index];
            if(!cerebrum_mask[cur_index])
                continue;
            for(unsigned int j = 0,fib_offset = 0;j < fib.num_fiber && fib.fa[j][cur_index] > fiber_threshold;
                    ++j,fib_offset+=si2vi.size())
                subject_vector.push_back(subject_qa[subject_index][s_index + fib_offset]);
        }
        image::minus_constant(subject_vector.begin(),subject_vector.end(),image::mean(subject_vector.begin(),subject_vector.end()));
        float sd = image::standard_deviation(subject_vector.begin(),subject_vector.end());
        if(sd > 0.0)
            image::divide_constant(subject_vector.begin(),subject_vector.end(),sd);
    }
    void get_dif_matrix(std::vector<float>& matrix,const image::basic_image<int,3>& cerebrum_mask)
    {
        matrix.clear();
        matrix.resize(num_subjects*num_subjects);
        std::vector<std::vector<float> > subject_vector;
        get_subject_vector(subject_vector,cerebrum_mask);
        begin_prog("calculating");
        for(unsigned int i = 0; check_prog(i,num_subjects);++i)
            for(unsigned int j = i+1; j < num_subjects;++j)
            {
                double result = image::root_mean_suqare_error(
                            subject_vector[i].begin(),subject_vector[i].end(),
                            subject_vector[j].begin());
                matrix[i*num_subjects+j] = result;
                matrix[j*num_subjects+i] = result;
            }
    }

    void save_subject_vector(const char* output_name,
                             const image::basic_image<int,3>& cerebrum_mask) const
    {
        gz_mat_write matfile(output_name);
        if(!matfile)
        {
            error_msg = "Cannot output file";
            return;
        }
        std::vector<std::vector<float> > subject_vector;
        get_subject_vector(subject_vector,cerebrum_mask);
        std::string name_string;
        for(unsigned int index = 0;index < num_subjects;++index)
        {
            name_string += subject_names[index];
            name_string += "\n";
        }
        matfile.write("subject_names",name_string.c_str(),1,(unsigned int)name_string.size());
        for(unsigned int index = 0;index < num_subjects;++index)
        {
            std::ostringstream out;
            out << "subject" << index;
            matfile.write(out.str().c_str(),&subject_vector[index][0],1,(unsigned int)subject_vector[index].size());
        }
    }
    void save_subject_data(const char* output_name)
    {
        // store results
        gz_mat_write matfile(output_name);
        if(!matfile)
        {
            error_msg = "Cannot output file";
            return;
        }
        for(unsigned int index = 0;index < mat_reader.size();++index)
            if(mat_reader[index].get_name() != "report" &&
               mat_reader[index].get_name().find("subject") != 0)
                matfile.write(mat_reader[index]);
        for(unsigned int index = 0;check_prog(index,(unsigned int)subject_qa.size());++index)
        {
            std::ostringstream out;
            out << "subject" << index;
            matfile.write(out.str().c_str(),subject_qa[index],fib.num_fiber,(unsigned int)si2vi.size());
        }
        std::string name_string;
        for(unsigned int index = 0;index < num_subjects;++index)
        {
            name_string += subject_names[index];
            name_string += "\n";
        }
        matfile.write("subject_names",name_string.c_str(),1,(unsigned int)name_string.size());
        matfile.write("R2",&*R2.begin(),1,(unsigned int)R2.size());

        {
            std::ostringstream out;
            out << "A total of " << num_subjects << " subjects were included in the connectometry database." << subject_report.c_str();
            std::string report = out.str();
            matfile.write("report",&*report.c_str(),1,(unsigned int)report.length());
        }
    }
    void get_subject_slice(unsigned int subject_index,unsigned int z_pos,
                            image::basic_image<float,2>& slice) const
    {
        slice.clear();
        slice.resize(image::geometry<2>(dim.width(),dim.height()));
        unsigned int slice_offset = z_pos*dim.plane_size();
        for(unsigned int index = 0;index < slice.size();++index)
        {
            unsigned int cur_index = index + slice_offset;
            if(fib.fa[0][cur_index] == 0.0)
                continue;
            slice[index] = subject_qa[subject_index][vi2si[cur_index]];
        }
    }
    void get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data) const
    {
        fa_data.resize(fib.num_fiber);
        for(unsigned int index = 0;index < fib.num_fiber;++index)
            fa_data[index].resize(dim.size());
        for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
        {
            unsigned int cur_index = si2vi[s_index];
            for(unsigned int i = 0,fib_offset = 0;i < fib.num_fiber && fib.fa[i][cur_index] > 0;++i,fib_offset+=(unsigned int)si2vi.size())
            {
                unsigned int pos = s_index + fib_offset;
                fa_data[i][cur_index] = subject_qa[subject_index][pos];
            }
        }
    }
    void get_data_at(unsigned int index,unsigned int fib_index,std::vector<double>& data,bool normalize_qa) const
    {
        data.clear();
        if((int)index >= dim.size() || fib.fa[0][index] == 0.0)
            return;
        unsigned int s_index = vi2si[index];
        unsigned int fib_offset = fib_index*(unsigned int)si2vi.size();
        data.resize(num_subjects);
        if(normalize_qa)
            for(unsigned int index = 0;index < num_subjects;++index)
                data[index] = subject_qa[index][s_index+fib_offset]/subject_qa_max[index];
        else
        for(unsigned int index = 0;index < num_subjects;++index)
            data[index] = subject_qa[index][s_index+fib_offset];
    }
    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data)
    {
        gz_mat_read single_subject;
        if(!single_subject.load_from_file(file_name))
        {
            error_msg = "fail to load the fib file";
            return false;
        }
        if(!is_consistent(single_subject))
        {
            error_msg = "Inconsistent ODF dimension";
            return false;
        }
        cur_subject_data.clear();
        cur_subject_data.resize(fib.num_fiber*si2vi.size());
        if(!sample_odf(single_subject,cur_subject_data))
        {
            error_msg += file_name;
            return false;
        }
        return true;
    }
    bool is_db_compatible(const FibData* rhs)
    {
        if(rhs->dim != dim || subject_qa_length != rhs->subject_qa_length)
        {
            error_msg = "Image dimension does not match";
            return false;
        }
        for(unsigned int index = 0;index < dim.size();++index)
            if(fib.fa[0][index] != rhs->fib.fa[0][index])
            {
                error_msg = "The connectometry db was created using a different template.";
                return false;
            }
        return true;
    }
    void read_subject_qa(std::vector<std::vector<float> >&data) const
    {
        data.resize(num_subjects);
        for(unsigned int i = 0;i < num_subjects;++i)
            data[i].swap(std::vector<float>(subject_qa[i],subject_qa[i]+subject_qa_length));
    }

    bool add_db(const FibData* rhs)
    {
        if(!is_db_compatible(rhs))
            return false;
        num_subjects += rhs->num_subjects;
        R2.insert(R2.end(),rhs->R2.begin(),rhs->R2.end());
        subject_qa_max.insert(subject_qa_max.end(),rhs->subject_qa_max.begin(),rhs->subject_qa_max.end());
        subject_names.insert(subject_names.end(),rhs->subject_names.begin(),rhs->subject_names.end());
        // copy the qa memeory
        for(unsigned int index = 0;index < rhs->num_subjects;++index)
        {
            subject_qa_buf.push_back(std::vector<float>());
            subject_qa_buf.back().resize(subject_qa_length);
            std::copy(rhs->subject_qa[index],
                      rhs->subject_qa[index]+subject_qa_length,subject_qa_buf.back().begin());
        }

        // everytime subject_qa_buf has a change, its memory may have been reallocated. Thus we need to assign all pointers.
        subject_qa.resize(num_subjects);
        for(unsigned int index = 0;index < subject_qa_buf.size();++index)
            subject_qa[num_subjects+index-subject_qa_buf.size()] = &(subject_qa_buf[index][0]);
    }

public:
    image::geometry<3> dim;
    image::vector<3> vs;
    std::vector<float> trans_to_mni;
public:
    std::vector<ViewItem> view_item;
    unsigned int other_mapping_index;
public:
    FibData(void)
    {
        vs[0] = vs[1] = vs[2] = 1.0;
    }
public:
    bool load_from_file(const char* file_name)
    {
        if (!mat_reader.load_from_file(file_name) || prog_aborted())
        {
            if(prog_aborted())
                error_msg = "loading process aborted";
            else
                error_msg = "cannot open file";
            return false;
        }
        {
            unsigned int row,col;
            const char* report_buf = 0;
            if(mat_reader.read("report",row,col,report_buf))
                report = std::string(report_buf,report_buf+row*col);
        }
        if(!fib.add_data(mat_reader))
        {
            error_msg = fib.error_msg;
            return false;
        }

        if(fib.fa.empty())
        {
            error_msg = "invalid fib format:";
            error_msg += file_name;
            return false;
        }
        dim = fib.dim;
        odf.read(mat_reader);
        odf.initializeODF(dim,fib.fa[0],fib.half_odf_size);

        for(int index = 0;index < fib.fa.size();++index)
        {
            view_item.push_back(ViewItem());
            view_item.back().name =  fib.fa.size() == 1 ? "fa0":"qa0";
            view_item.back().name[2] += index;
            view_item.back().image_data = image::make_image(fib.dim,fib.fa[index]);
            view_item.back().set_scale(fib.fa[index],fib.fa[index]+fib.dim.size());
        }

        view_item.push_back(ViewItem());
        view_item.back().name = "color";
        other_mapping_index = view_item.size();

        unsigned int row,col;
        for (unsigned int index = 0;check_prog(index,mat_reader.size());++index)
        {
            std::string matrix_name = mat_reader.name(index);
            ::set_title(matrix_name.c_str());
            if (matrix_name == "voxel_size")
            {
                const float* size_buf = 0;
                mat_reader.read(index,row,col,size_buf);
                if (!size_buf || row*col != 3)
                    return false;
                vs = size_buf;
                if(vs[0]*vs[1]*vs[2] == 0.0)
                    vs[0] = vs[1] = vs[2] = 2.0;
                continue;
            }
            if (matrix_name == "trans")
            {
                const float* trans = 0;
                mat_reader.read(index,row,col,trans);
                trans_to_mni.resize(16);
                std::copy(trans,trans+16,trans_to_mni.begin());
                continue;
            }
            if (matrix_name == "image")
                continue;
            std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
            if (prefix_name == "index" || prefix_name == "fa" || prefix_name == "dir")
                continue;
            const float* buf = 0;
            mat_reader.read(index,row,col,buf);
            if (row*col != dim.size() || !buf)
                continue;
            if(matrix_name.length() >= 2 && matrix_name[matrix_name.length()-2] == '_' &&
               (matrix_name[matrix_name.length()-1] == 'x' ||
                matrix_name[matrix_name.length()-1] == 'y' ||
                matrix_name[matrix_name.length()-1] == 'z' ||
                matrix_name[matrix_name.length()-1] == 'd'))
                continue;
            view_item.push_back(ViewItem());
            view_item.back().name = matrix_name;
            view_item.back().image_data = image::make_image(fib.dim,buf);
            view_item.back().set_scale(buf,buf+dim.size());

        }
        if (!dim[2])
        {
            error_msg = "invalid dimension";
            return false;
        }

        if(!trans_to_mni.empty() && !view_item.empty())
        {
            unsigned int row,col;
            const float* mx = 0;
            const float* my = 0;
            const float* mz = 0;
            const short* native_geo = 0;
            for(unsigned int i = 0; i < view_item.size();++i)
            {
                std::string name;
                if(i)
                    name = view_item[i].name;
                if(mat_reader.read((name+"_x").c_str(),row,col,mx) &&
                   mat_reader.read((name+"_y").c_str(),row,col,my) &&
                   mat_reader.read((name+"_z").c_str(),row,col,mz) &&
                     mat_reader.read((name+"_d").c_str(),row,col,native_geo))
                {
                    view_item[i].mx = image::make_image(fib.dim,mx);
                    view_item[i].my = image::make_image(fib.dim,my);
                    view_item[i].mz = image::make_image(fib.dim,mz);
                    view_item[i].native_geo = image::geometry<3>(native_geo[0],native_geo[1],native_geo[2]);
                }
            }
        }

        read_db();
        return true;
    }
public:
    bool has_odfs(void) const{return odf.has_odfs();}
    const float* get_odf_data(unsigned int index) const{return odf.get_odf_data(index);}
public:
    size_t get_name_index(const std::string& index_name) const
    {
        for(unsigned int index_num = 0;index_num < view_item.size();++index_num)
            if(view_item[index_num].name == index_name)
                return index_num;
        return view_item.size();
    }
    void get_index_list(std::vector<std::string>& index_list) const
    {
        bool is_dti = (view_item[0].name[0] == 'f');
        if(is_dti)
            index_list.push_back("fa");
        else
            index_list.push_back("qa");
        for (int index = other_mapping_index; index < view_item.size(); ++index)
            index_list.push_back(view_item[index].name);
    }
    std::pair<float,float> get_value_range(const std::string& view_name) const
    {
        unsigned int view_index = get_name_index(view_name);
        if(view_index == view_item.size())
            return std::make_pair((float)0.0,(float)0.0);
        if(view_item[view_index].name == "color")
            return std::make_pair((float)0.0,(float)0.0);
        return std::make_pair(view_item[view_index].min_value,view_item[view_index].max_value);
    }

    void get_slice(const std::string& view_name,
                   unsigned char d_index,unsigned int pos,
                   image::color_image& show_image,const image::value_to_color<float>& v2c)
    {
        unsigned int view_index = get_name_index(view_name);
        if(view_index == view_item.size())
            return;

        if(view_item[view_index].name == "color")
        {

            if(view_item[view_index].data_buf.empty())
            {
                image::basic_image<image::rgb_color,3> color_buf(dim);
                float max_value = view_item[0].max_value;
                if(max_value + 1.0 == 1.0)
                    max_value = 1.0;
                float r = 255.9/max_value;
                for (unsigned int index = 0;index < dim.size();++index)
                {
                    image::vector<3,float> dir(fib.getDir(index,0));
                    dir *= std::floor(fib.getFA(index,0)*r);
                    unsigned int color = (unsigned char)std::abs(dir[0]);
                    color <<= 8;
                    color |= (unsigned char)std::abs(dir[1]);
                    color <<= 8;
                    color |= (unsigned char)std::abs(dir[2]);
                    color_buf[index] = color;
                }
                color_buf.swap(view_item[view_index].data_buf);
            }
            image::reslicing(view_item[view_index].data_buf, show_image, d_index,pos);
        }
        else
        {
            image::basic_image<float,2> buf;
            image::reslicing(view_item[view_index].image_data, buf, d_index, pos);
            v2c.convert(buf,show_image);
        }

    }

    void get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
    {
        unsigned int index = (z*dim[1]+y)*dim[0] + x;
        if (index >= dim.size())
            return;
        for(unsigned int i = 0;i < fib.num_fiber;++i)
        {
            image::vector<3,float> dir(fib.getDir(index,i));
            buf.push_back(dir[0]);
            buf.push_back(dir[1]);
            buf.push_back(dir[2]);
        }
    }
    void get_voxel_information(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
    {
        unsigned int index = (z*dim[1]+y)*dim[0] + x;
        if (index >= dim.size())
            return;
        for(unsigned int i = 0;i < view_item.size();++i)
            if(view_item[i].name != "color")
                buf.push_back(view_item[i].image_data.empty() ? 0.0 : view_item[i].image_data[index]);
    }
public:

    void get_index_titles(std::vector<std::string>& titles)
    {
        std::vector<std::string> index_list;
        get_index_list(index_list);
        for(unsigned int index = 0;index < index_list.size();++index)
        {
            titles.push_back(index_list[index]+" mean");
            titles.push_back(index_list[index]+" sd");
        }
    }
    void getSlicesDirColor(unsigned short order,unsigned int* pixels) const
    {
        for (unsigned int index = 0;index < dim.size();++index,++pixels)
        {
            if (fib.getFA(index,order) == 0.0)
            {
                *pixels = 0;
                continue;
            }

            float fa = fib.getFA(index,order)*255.0;
            image::vector<3,float> dir(fib.getDir(index,order));
            unsigned int color = (unsigned char)std::abs(dir[0]*fa);
            color <<= 8;
            color |= (unsigned char)std::abs(dir[1]*fa);
            color <<= 8;
            color |= (unsigned char)std::abs(dir[2]*fa);
            *pixels = color;
        }
    }
};



class fiber_orientations{
public:
    image::geometry<3> dim;
    image::vector<3> vs;
    unsigned char fib_num;
    std::vector<const float*> dir;
    std::vector<const float*> fa;
    std::vector<const short*> findex;
    std::vector<image::vector<3,float> > odf_table;
    float threshold;
    float cull_cos_angle;
private:
    bool get_nearest_dir_fib(unsigned int space_index,
                         const image::vector<3,float>& ref_dir, // reference direction, should be unit vector
                         unsigned char& fib_order_,
                         unsigned char& reverse_) const
    {
        if(space_index >= dim.size() || fa[0][space_index] <= threshold)
            return false;
        float max_value = cull_cos_angle;
        unsigned char fib_order;
        unsigned char reverse;
        for (unsigned char index = 0;index < fib_num;++index)
        {
            if (fa[index][space_index] <= threshold)
                break;
            float value = cos_angle(ref_dir,space_index,index);
            if (-value > max_value)
            {
                max_value = -value;
                fib_order = index;
                reverse = 1;
            }
            else
                if (value > max_value)
                {
                    max_value = value;
                    fib_order = index;
                    reverse = 0;
                }
        }
        if (max_value == cull_cos_angle)
            return false;
        fib_order_ = fib_order;
        reverse_ = reverse;
        return true;
    }
public:
    void read(const FibData& fib_data)
    {
        dim = fib_data.dim;
        vs = fib_data.vs;
        odf_table = fib_data.fib.odf_table;
        fib_num = fib_data.fib.num_fiber;
        fa = fib_data.fib.fa;
        findex = fib_data.fib.findex;
        dir = fib_data.fib.dir;
    }
    bool get_dir(unsigned int space_index,
                         const image::vector<3,float>& dir, // reference direction, should be unit vector
                         image::vector<3,float>& main_dir) const
    {
        unsigned char fib_order;
        unsigned char reverse;
        if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse))
            return false;
        main_dir = get_dir(space_index,fib_order);
        if(reverse)
        {
            main_dir[0] = -main_dir[0];
            main_dir[1] = -main_dir[1];
            main_dir[2] = -main_dir[2];
        }
        return true;
    }

    const float* get_dir(unsigned int space_index,unsigned char fib_order) const
    {
        if(!dir.empty())
            return dir[fib_order] + space_index + (space_index << 1);
        return &*(odf_table[findex[fib_order][space_index]].begin());
    }

    float cos_angle(const image::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const
    {
        if(!dir.empty())
        {
            const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
            return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
        }
        return cur_dir*odf_table[findex[fib_order][space_index]];
    }

    float get_fa(unsigned int space_index,
                             const image::vector<3,float>& dir) const
    {
        unsigned char fib_order;
        unsigned char reverse;
        if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse))
            return 0.0;
        return fa[fib_order][space_index];
    }
};


#endif//FIB_DATA_HPP
