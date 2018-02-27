#ifndef FIB_DATA_HPP
#define FIB_DATA_HPP
#include <fstream>
#include <sstream>
#include <string>
#include "prog_interface_static_link.h"
#include "image/image.hpp"
#include "gzip_interface.hpp"
#include "connectometry_db.hpp"

struct odf_data{
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
    odf_data(void):odfs(0){}
    bool read(gz_mat_read& mat_reader);
    bool has_odfs(void) const
    {
        return odfs != 0 || !odf_blocks.empty();
    }
    const float* get_odf_data(unsigned int index) const;
};

class fiber_directions
{
public:
    std::vector<const float*> dir;
    std::vector<const short*> findex;
    std::vector<std::vector<short> > findex_buf;
public:
    std::vector<std::string> index_name;
    std::vector<std::vector<const float*> > index_data;
public:
    std::vector<const float*> fa;
    std::vector<image::vector<3,float> > odf_table;
    std::vector<image::vector<3,unsigned short> > odf_faces;
    unsigned int num_fiber;
    unsigned int half_odf_size;
    std::string error_msg;
private:
    void check_index(unsigned int index);
public:
    bool add_data(gz_mat_read& mat_reader);
    bool set_tracking_index(int new_index);
    bool set_tracking_index(const std::string& name);
    float get_fa(unsigned int index,unsigned char order) const;
    const float* get_dir(unsigned int index,unsigned int order) const;

};

class fib_data;
class tracking_data{
public:
    image::geometry<3> dim;
    image::vector<3> vs;
    unsigned char fib_num;
    std::vector<const float*> dir;
    std::vector<const float*> fa;
    std::vector<const short*> findex;
    std::vector<std::vector<const float*> > other_index;
    std::vector<image::vector<3,float> > odf_table;
public:
    bool get_nearest_dir_fib(unsigned int space_index,
                         const image::vector<3,float>& ref_dir, // reference direction, should be unit vector
                         unsigned char& fib_order_,
                         unsigned char& reverse_,
                             float threshold,
                             float cull_cos_angle) const;
    void read(const fib_data& fib);
    bool get_dir(unsigned int space_index,
                         const image::vector<3,float>& dir, // reference direction, should be unit vector
                         image::vector<3,float>& main_dir,
                 float threshold,
                 float cull_cos_angle) const;
    const float* get_dir(unsigned int space_index,unsigned char fib_order) const;
    float cos_angle(const image::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const;
    float get_track_specific_index(unsigned int space_index,unsigned int index_num,
                             const image::vector<3,float>& dir) const;
    bool is_white_matter(const image::vector<3,float>& pos,float t) const;

};



struct item
{
    std::string name;
    image::const_pointer_image<float,3> image_data;
    float max_value;
    float min_value;
    // used in QSDR
    image::basic_image<unsigned int,3> color_map_buf;
    image::const_pointer_image<float,3> mx,my,mz;
    image::geometry<3> native_geo;
    template<class input_iterator>
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

class fib_data
{
public:
    mutable std::string error_msg;
    std::string report;
    gz_mat_read mat_reader;
public:
    image::geometry<3> dim;
    image::vector<3> vs;
    bool is_human_data;
    bool is_qsdr;
    fiber_directions dir;
    odf_data odf;
    connectometry_db db;
    std::vector<item> view_item;
public:
    image::thread thread;
    int prog;
    std::vector<float> trans_to_mni;
    image::basic_image<image::vector<3,float>,3 > mni_position;
public:
    void run_normalization(bool background);
    void subject2mni(image::vector<3>& pos);
    void subject2mni(image::pixel_index<3>& index,image::vector<3>& pos);
    void get_atlas_roi(int atlas_index,int roi_index,std::vector<image::vector<3,short> >& points,float& r);
    const image::basic_image<image::vector<3,float>,3 >& get_mni_mapping(void);
    bool has_reg(void)const{return thread.has_started();}
    bool get_profile(const std::vector<float>& tract_data,
                     std::vector<float>& profile);

public:
    fib_data(void):is_qsdr(false)
    {
        vs[0] = vs[1] = vs[2] = 1.0;
    }
public:
    bool load_from_file(const char* file_name);
    bool load_from_mat(void);
public:
    bool has_odfs(void) const{return odf.has_odfs();}
    const float* get_odf_data(unsigned int index) const{return odf.get_odf_data(index);}
public:
    size_t get_name_index(const std::string& index_name) const;
    void get_index_list(std::vector<std::string>& index_list) const;
    image::const_pointer_image<float,3> get_view_volume(const std::string& view_name) const;
    std::pair<float,float> get_value_range(const std::string& view_name) const;
    void get_slice(unsigned int view_index,
                   unsigned char d_index,unsigned int pos,
                   image::color_image& show_image,const image::value_to_color<float>& v2c);
    void get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const;
    void get_voxel_information(int x,int y,int z,std::vector<float>& buf) const;
    void get_index_titles(std::vector<std::string>& titles);
};



class track_recognition{
    image::uniform_dist<int> dist;
public:
    image::thread thread;
public:
    image::ml::network cnn;
    image::ml::network_data<float,unsigned char> cnn_data;

    std::vector<std::string> cnn_name;
    std::string err_msg,msg;
    bool is_running = false;
public:
    std::vector<std::string> track_list;
    std::vector<std::string> track_name;
    bool can_recognize(void);
public:
    void clear(void);
    void add_label(const std::string& name){cnn_name.push_back(name);}
    void add_sample(fib_data* handle,unsigned char index,const std::vector<float>& tracks);
};

#endif//FIB_DATA_HPP
