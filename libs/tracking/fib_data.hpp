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
    std::vector<const float*> dir;
    std::vector<const short*> findex;
    std::vector<std::vector<short> > findex_buf;
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

            if (matrix_name == "odfs")
            {
                const float* odfs;
                mat_reader.read(index,row,col,odfs);
                odf.setODFs(odfs,row*col);
                continue;
            }

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
                mat_reader.read(index,row,col,buf);
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

        odf.initializeODF(dim,fa[0],half_odf_size);
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
    std::string error_msg,report;
    gz_mat_read mat_reader;
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
            error_msg = "Cannot open file";
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
        for(int index = 0;index < fib.fa.size();++index)
        {
            view_item.push_back(ViewItem());
            view_item.back().name =  fib.fa.size() == 1 ? "fa0":"qa0";
            view_item.back().name[2] += index;
            view_item.back().image_data = image::make_image(fib.dim,fib.fa[index]);
            view_item.back().is_overlay = false;
            view_item.back().set_scale(fib.fa[index],fib.fa[index]+fib.dim.size());
        }

        view_item.push_back(ViewItem());
        view_item.back().name = "color";
        view_item.back().is_overlay = false;
        other_mapping_index = view_item.size();

        unsigned int row,col;
        for (unsigned int index = 0;check_prog(index,mat_reader.size());++index)
        {
            std::string matrix_name = mat_reader.name(index);
            ::set_title(matrix_name.c_str());
            if (matrix_name == "dimension")
            {
                const unsigned short* dim_buf = 0;
                mat_reader.read(index,row,col,dim_buf);
                if (!dim_buf|| row*col != 3)
                    return false;
                std::copy(dim_buf,dim_buf+3,dim.begin());
                total_size = dim.size();
                continue;
            }
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
            view_item.back().image_data = image::make_image(fib.dim,buf);
            view_item.back().set_scale(buf,buf+total_size);

        }
        if (!dim[2])
        {
            error_msg = "invalid dimension";
            return false;
        }
        return true;
    }

public:
    unsigned int get_name_index(const std::string& index_name) const
    {
        for(unsigned int index_num = 0;index_num < view_item.size();++index_num)
            if(view_item[index_num].name == index_name)
                return index_num;
        return view_item.size();
    }

    float get_value_range(const std::string& view_name) const
    {
        unsigned int view_index = get_name_index(view_name);
        if(view_index == view_item.size())
            return 0.0;
        if(view_item[view_index].name == "color")
            return 0.0;
        return view_item[view_index].max_value-
               view_item[view_index].min_value;
    }

    void get_slice(const std::string& view_name,const std::string& overlay_name,
                   unsigned char d_index,unsigned int pos,
                   image::color_image& show_image,float contrast,float offset)
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
                for (unsigned int index = 0;index < total_size;++index)
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
            show_image.resize(buf.geometry());
            buf += offset-view_item[view_index].min_value;
            if(contrast != 0.0)
                buf *= 255.99/contrast;
            image::upper_lower_threshold(buf,(float)0.0,(float)255.0);
            std::copy(buf.begin(),buf.end(),show_image.begin());
        }


        if(overlay_name.length()) // has overlay
        {
            unsigned int overlay_index = get_name_index(overlay_name);
            if(overlay_index == view_item.size())
                return;
            if(view_item[overlay_index].data_buf.empty())
            {
                using namespace std;
                const float* data = view_item[overlay_index].image_data.begin();
                float max_value = view_item[overlay_index].max_value;
                float min_value = view_item[overlay_index].min_value;
                float r = max(max_value,-min_value);
                if(r + 1.0 == 1.0)
                    r = 1.0;
                r = 255.9/r;
                image::basic_image<image::rgb_color,3> color_buf(dim);
                for (unsigned int index = 0;index < total_size;++index)
                    if(data[index] != 0.0)
                    {
                        image::rgb_color color;
                        color.g = std::floor(std::abs(data[index]*r));
                        if(data[index] > 0.0)
                            color.r = 255;
                        else
                            color.b = 255;
                        color_buf[index] = color;
                    }
                color_buf.swap(view_item[overlay_index].data_buf);
            }
            image::color_image buf;
            image::reslicing(view_item[overlay_index].data_buf, buf, d_index, pos);
            for(unsigned int index = 0;index < buf.size();++index)
                if((unsigned int)(buf[index]) != 0)
                    show_image[index] = buf[index];
        }
    }

    void get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
    {
        unsigned int index = (z*dim[1]+y)*dim[0] + x;
        if (index >= total_size)
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
        if (index >= total_size)
            return;
        for(unsigned int i = 0;i < view_item.size();++i)
            if(view_item[i].name != "color")
                buf.push_back(view_item[i].image_data.empty() ? 0.0 : view_item[i].image_data[index]);
    }
public:

    void get_index_titles(std::vector<std::string>& titles)
    {
        if (view_item[0].name[0] == 'f')// is dti
        {
            titles.push_back("FA mean");
            titles.push_back("FA sd");
        }
        else
        {
            titles.push_back("QA mean");
            titles.push_back("QA sd");
        }
        for(int data_index = other_mapping_index;
            data_index < view_item.size();++data_index)
        {
            titles.push_back(view_item[data_index].name+" mean");
            titles.push_back(view_item[data_index].name+" sd");
        }

    }
    void getSlicesDirColor(unsigned short order,unsigned int* pixels) const
    {
        for (unsigned int index = 0;index < total_size;++index,++pixels)
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
