#ifndef FIB_DATA_HPP
#define FIB_DATA_HPP
#include <fstream>
#include <sstream>
#include <string>
#include "prog_interface_static_link.h"
#include "mat_file.hpp"
#include "image/image.hpp"
struct AngStatistics
{
    float ang_dev;
    float ang_dev_sq;
    unsigned int ang_dev_num;
    AngStatistics(void):ang_dev(0),ang_dev_sq(0),ang_dev_num(0) {}
    void add(float cosine_value)
    {
        cosine_value = std::abs(cosine_value);
        if (cosine_value >= 1)
            cosine_value = 1.0;
        cosine_value = std::acos(cosine_value);
        ang_dev += cosine_value;
        ang_dev_sq += cosine_value*cosine_value;
        ++ang_dev_num;
    }
    void calculation(void)
    {
        ang_dev /= ang_dev_num;
        ang_dev_sq /= ang_dev_num;
        ang_dev_sq -= ang_dev*ang_dev;
        ang_dev_sq = std::sqrt(ang_dev_sq);
        ang_dev *= 90.0/3.14159265358979323846;
        ang_dev_sq *= 90.0/3.14159265358979323846;
    }
};


struct ODFData{
private:
    const float* odfs;
    unsigned int odfs_size;
private:
    image::basic_image<unsigned int,3> voxel_index_map;
    std::vector<const float*> odf_blocks;
    std::vector<unsigned int> odf_block_size;
    image::basic_image<unsigned char,3> odf_block_map1;
    image::basic_image<unsigned int,3> odf_block_map2;
    unsigned int half_odf_size;
public:
    ODFData(void):odfs(0){}
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
    void initializeODF(const image::geometry<3>& dim,const std::vector<const float*> fa,unsigned int half_odf_size_)
    {
        half_odf_size = half_odf_size_;
        // handle the odf mappings
        if (odfs)
        {
            voxel_index_map.resize(dim);
            for (unsigned int index = 0,j = 0;index < voxel_index_map.size();++index)
            {
                if (fa[0][index] == 0.0)
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
            for(int i = 0;i < odf_block_size.size();++i)
                for(int j = 0;j < odf_block_size[i];j += half_odf_size)
                {
                    int k_end = j + half_odf_size;
                    bool is_odf_zero = true;
                    for(int k = j;k < k_end;++k)
                        if(odf_blocks[i][k] != 0.0)
                        {
                            is_odf_zero = false;
                            break;
                        }
                    if(!is_odf_zero)
                        for(;voxel_index < odf_block_map1.size();++voxel_index)
                            if(fa[0][voxel_index] != 0.0)
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
    std::vector<std::vector<short> > findex_buf;
public:
    std::vector<const short*> findex;
public:
private:
    ODFData odf;
public:
    bool has_odfs(void) const{return odf.has_odfs();}
    const float* get_odf_data(unsigned int index) const{return odf.get_odf_data(index);}
public:
    std::vector<std::string> index_name;
    std::vector<std::vector<const float*> > index_data;
    std::vector<std::vector<const short*> > index_data_dir;

public:
    std::vector<const float*> fa;
    image::geometry<3> dim;

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

    bool add_data(MatFile& mat_reader)
    {
        unsigned int row,col;
        // dimension
        {
            const unsigned short* dim_buf = 0;
            if (!mat_reader.get_matrix("dimension",row,col,dim_buf))
            {
                error_msg = "cannot find dimension matrix";
                return false;
            }
            std::copy(dim_buf,dim_buf+3,dim.begin());
        }
        // odf_vertices
        {
            const float* odf_buffer;
            if (mat_reader.get_matrix("odf_vertices",row,col,odf_buffer))
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
            if(mat_reader.get_matrix("odf_faces",row,col,odf_buffer))
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

        for (unsigned int index = 0;check_prog(index,mat_reader.get_matrix_count());++index)
        {
            std::string matrix_name = mat_reader.get_matrix_name(index);

            if (matrix_name == "odfs")
            {
                const float* odfs;
                mat_reader.get_matrix(index,row,col,odfs);
                odf.setODFs(odfs,row*col);
                continue;
            }

            if (matrix_name == "image")
            {
                check_index(0);
                mat_reader.get_matrix(index,row,col,fa[0]);
                findex_buf.resize(1);
                findex_buf[0].resize(row*col);
                findex[0] = &(findex_buf[0][0]);
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
                mat_reader.get_matrix(index,row,col,findex[store_index]);
                continue;
            }
            if (prefix_name == "fa")
            {
                check_index(store_index);
                mat_reader.get_matrix(index,row,col,fa[store_index]);
                continue;
            }
            if (prefix_name == "dir")
            {
                const float* dir;
                mat_reader.get_matrix(index,row,col,dir);
                check_index(store_index);
                if(findex_buf.size() <= store_index)
                    findex_buf.resize(store_index+1);
                findex_buf[store_index].resize(row*col/3);
                for(unsigned int index = 0;index < findex_buf[store_index].size();++index)
                {
                    image::vector<3> d(dir+index+index+index);
                    findex_buf[store_index][index] = 0;
                    double cos = std::fabs(d*odf_table[0]);
                    for(unsigned int i = 0;i < half_odf_size;++i)
                        if(d*odf_table[i] > cos)
                        {
                            findex_buf[store_index][index] = i;
                            cos = d*odf_table[i];
                        }
                }
                findex[store_index] = &(findex_buf[store_index][0]);
                continue;
            }
            if (prefix_name == "odf1" || prefix_name == "odf2" || prefix_name == "odf3" ||
                    prefix_name == "odf4" || prefix_name == "odf5" || prefix_name == "odf6" ||
                    prefix_name == "odf7" || prefix_name == "odf8" || prefix_name == "odf9")
            {
                store_index += 10*(prefix_name[prefix_name.length()-1]-'0');
                prefix_name = "odf";
            }
            if (prefix_name == "odf")
            {
                const float* buf;
                mat_reader.get_matrix(index,row,col,buf);
                odf.setODF(store_index,buf,row*col);
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
            mat_reader.get_matrix(index,row,col,index_data[prefix_name_index][store_index]);

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

        odf.initializeODF(dim,fa,half_odf_size);
        if(num_fiber == 0)
            error_msg = "No image data found";
        return num_fiber;
    }

    void set_tracking_index(int new_index)
    {
        if(new_index >= index_data.size())
            return;
        fa = index_data[new_index];
        findex = index_data_dir[new_index];
    }

    void set_tracking_index(const std::string name)
    {
        set_tracking_index(std::find(index_name.begin(),index_name.end(),name)-index_name.begin());
    }

    float getFA(unsigned int index,unsigned char order) const
    {
        if(order >= num_fiber)
            return 0.0;
        return fa[order][index];
    }

    float estimateFA(const image::vector<3,float>& pos,unsigned char order) const
    {
        if(order >= num_fiber)
            return 0.0;
        return
            image::linear_estimate(image::basic_image<float,3,image::pointer_memory<float> >((float*)(fa[order]),dim),pos);
    }
    image::vector<3,float> getDir(unsigned int index,unsigned int order) const
    {
        if(order >= num_fiber)
            return odf_table[0];
        return odf_table[findex[order][index]];
    }
    image::vector<3,float> getReverseDir(unsigned int index,unsigned int order) const
    {
        if(order >= num_fiber)
            return odf_table[0];
        unsigned int odf_index = findex[order][index];
        return odf_index < half_odf_size ? odf_table[odf_index + half_odf_size] : odf_table[odf_index-half_odf_size];
    }

};
struct ViewItem
{
    std::string name;
    image::basic_image<float, 3,image::const_pointer_memory<float> > image_data;
    bool is_overlay;
    float max_value;
    float min_value;
    image::basic_image<image::rgb_color,3> data_buf;
    template<typename input_iterator>
    void set_scale(input_iterator from,input_iterator to)
    {
        max_value = *std::max_element(from,to);
        min_value = *std::min_element(from,to);
    }
};
class FibData
{
public:
    std::string error_msg;
    MatFile mat_reader;
    FiberDirection fib;
public:
    image::geometry<3> dim;
    image::vector<3> vs;
    std::vector<float> trans_to_mni;
    unsigned int total_size;
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
        if (!mat_reader.load_from_file(file_name))
        {
            error_msg = mat_reader.error_msg;
            return false;
        }
        if(!fib.add_data(mat_reader))
        {
            error_msg = fib.error_msg;
            return false;
        }
        for(int index = 0;index < fib.fa.size();++index)
        {
            view_item.push_back(ViewItem());
            view_item.back().name =  fib.fa.size() == 1 ? "fa0":"qa0";
            view_item.back().name[2] += index;
            view_item.back().image_data = fib.fa[index];
            view_item.back().image_data.resize(fib.dim);
            view_item.back().is_overlay = false;
            view_item.back().set_scale(fib.fa[index],fib.fa[index]+fib.dim.size());
        }

        view_item.push_back(ViewItem());
        view_item.back().name = "color";
        view_item.back().is_overlay = false;
        other_mapping_index = view_item.size();

        unsigned int row,col;
        for (unsigned int index = 0;check_prog(index,mat_reader.get_matrix_count());++index)
        {
            std::string matrix_name = mat_reader.get_matrix_name(index);
            ::set_title(matrix_name.c_str());
            if (matrix_name == "dimension")
            {
                const unsigned short* dim_buf = 0;
                mat_reader.get_matrix(index,row,col,dim_buf);
                if (!dim_buf|| row*col != 3)
                    return false;
                std::copy(dim_buf,dim_buf+3,dim.begin());
                total_size = dim.size();
                continue;
            }
            if (matrix_name == "voxel_size")
            {
                const float* size_buf = 0;
                mat_reader.get_matrix(index,row,col,size_buf);
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
                mat_reader.get_matrix(index,row,col,trans);
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
            mat_reader.get_matrix(index,row,col,buf);
            if (row*col != total_size || !buf)
                continue;
            view_item.push_back(ViewItem());
            view_item.back().name = matrix_name;
            view_item.back().is_overlay = false;
            for(unsigned int i = 0;i < total_size;++i)
                if(buf[i] == 0.0 && fib.fa[0][i] != 0.0)
                {
                    view_item.back().is_overlay = true;
                    break;
                }
            view_item.back().image_data = buf;
            view_item.back().image_data.resize(fib.dim);
            view_item.back().set_scale(buf,buf+total_size);

        }
        if (!dim[2])
        {
            error_msg = "invalid dimension";
            return false;
        }
        return true;
    }
    void compare_fiber_directions(const FibData& rhs,const short *points,unsigned int number,std::string& result,std::ostream& report) const
    {
        AngStatistics st_ma;
        unsigned int crossing_count = 0;
        unsigned int match_number = 0;
        unsigned int false_fiber = 0;
        float match_angle = std::cos(9.0*3.14159265358979323846/180.0);
        std::ostringstream out;
        report << "x,y,rfa0,rfa1,rfa2";
        for (unsigned int index = other_mapping_index;index < view_item.size();++index)
            report << "," << view_item[index].name;
        report << ",major dev,minor suc\r\n";
        begin_prog("analyzing");
        for (unsigned int index = 0;check_prog(index,number);++index,points+=3)
        {
            unsigned int pixel_index = points[0] + dim[0]*(points[1] + dim[1]*points[2]);
            unsigned int fiber_number1 = 1;
            unsigned int fiber_number2 = 1;
            for (;fiber_number1 < 3;++fiber_number1)
                if (fib.getFA(pixel_index,fiber_number1) == 0.0)
                    break;
            for (;fiber_number2 < 3;++fiber_number2)
                if (rhs.fib.getFA(pixel_index,fiber_number2) == 0.0)
                    break;
            if (fiber_number2 != fiber_number1)
                ++false_fiber;

            bool mis_match = false;
            {
                float ma_dev = std::abs(fib.getDir(pixel_index,0)*rhs.fib.getDir(pixel_index,0));
                // match the nearest one
                if (rhs.fib.getFA(pixel_index,1) != 0.0)
                {
                    float ma2_dev = std::abs(fib.getDir(pixel_index,0)*rhs.fib.getDir(pixel_index,1));
                    if (ma2_dev > ma_dev)
                    {
                        mis_match = true;
                        ma_dev = ma2_dev;
                    }
                }
                st_ma.add(ma_dev);
            }

            if (fiber_number1 == 1)
            {
                continue;
            }
            ++crossing_count;
            if (fiber_number2 == 1)
            {
                continue;
            }

            if (std::abs(fib.getDir(pixel_index,1)*rhs.fib.getDir(pixel_index,(mis_match) ? 0 : 1)) >= match_angle)
            {
                ++match_number;
                report << points[0] << "," << points[1] << "," <<
                rhs.fib.getFA(pixel_index,0) << "," << rhs.fib.getFA(pixel_index,1) << "," << rhs.fib.getFA(pixel_index,2);
                for (unsigned int index = other_mapping_index;index < view_item.size();++index)
                    report << "," << view_item[index].image_data[pixel_index];
                report << std::endl;
            }
        }
        st_ma.calculation();
        out << "major dev avg_angle:" << st_ma.ang_dev << " std_angle:" << st_ma.ang_dev_sq << "\r\n";
        if (crossing_count)
            out << "sucessful minor fiber resolving rate:" << 100.0*(float)match_number/(float)crossing_count << "%\r\n";
        out << "false minor fiber rate:" << 100.0*(float)false_fiber/(float)number << "%\r\n";
        result = out.str();
        report << result;
    }
public:



    bool get_nearest_dir(unsigned int space_index,
                         const image::vector<3,float>& dir, // reference direction, should be unit vector
                         unsigned char& fib_order_,
                         unsigned char& reverse_,
                         float threshold,float cull_cos_angle) const
    {

        if (fib.getFA(space_index,0) <= threshold)
            return false;
        float max_value = cull_cos_angle;
        unsigned char fib_order;
        unsigned char reverse;
        for (unsigned char index = 0;index < fib.num_fiber;++index)
        {
            if (fib.getFA(space_index,index) <= threshold)
                break;
            float value = dir*fib.getDir(space_index,index);
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

    bool get_nearest_dir(unsigned int space_index,
                         const image::vector<3,float>& dir, // reference direction, should be unit vector
                         image::vector<3,float>& main_dir,
                         float threshold,float cull_cos_angle) const
    {
        unsigned char fib_order;
        unsigned char reverse;
        if (!get_nearest_dir(space_index,dir,fib_order,reverse,threshold,cull_cos_angle))
            return false;
        main_dir = (reverse) ? fib.getReverseDir(space_index,fib_order) : fib.getDir(space_index,fib_order);
        return true;
    }

    float get_directional_fa(unsigned int space_index,
                             const image::vector<3,float>& dir, // reference direction, should be unit vector
                             float threshold,float cull_cos_angle) const
    {
        unsigned char fib_order;
        unsigned char reverse;
        if (!get_nearest_dir(space_index,dir,fib_order,reverse,threshold,cull_cos_angle))
            return 0.0;
        return fib.getFA(space_index,fib_order);
    }



};
#endif//FIB_DATA_HPP
