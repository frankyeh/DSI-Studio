#ifndef FIB_DATA_HPP
#define FIB_DATA_HPP
#include <fstream>
#include <sstream>
#include <string>
#include "prog_interface_static_link.h"
#include "tipl/tipl.hpp"
#include "gzip_interface.hpp"
#include "connectometry_db.hpp"
#include "atlas.hpp"

struct odf_data{
private:
    const float* odfs;
    unsigned int odfs_size;
private:
    tipl::image<unsigned int,3> voxel_index_map;
    std::vector<const float*> odf_blocks;
    std::vector<unsigned int> odf_block_size;
    tipl::image<unsigned int,3> odf_block_map1;
    tipl::image<unsigned int,3> odf_block_map2;
    unsigned int half_odf_size;
public:
    odf_data(void):odfs(nullptr){}
    bool read(gz_mat_read& mat_reader);
    bool has_odfs(void) const
    {
        return odfs != nullptr || !odf_blocks.empty();
    }
    const float* get_odf_data(unsigned int index) const;
};

class fiber_directions
{
public:
    std::vector<std::vector<short> > findex_buf;
    std::vector<std::vector<float> > fa_buf;
public:
    std::vector<const float*> dir;
    std::vector<const short*> findex;
public:
    std::vector<std::string> index_name;
    std::vector<std::vector<const float*> > index_data;
    int cur_index = 0;
public:
    std::vector<std::string> dt_index_name;
    std::vector<std::vector<const float*> > dt_index_data;
    int dt_cur_index = 0;
public:
    std::vector<const float*> fa;
    std::vector<const float*> dt_fa;
    std::vector<tipl::vector<3,float> > odf_table;
    std::vector<tipl::vector<3,unsigned short> > odf_faces;
    unsigned int num_fiber;
    unsigned int half_odf_size;
    std::string error_msg;
public:
    void check_index(unsigned int index);
    bool add_data(gz_mat_read& mat_reader);
    bool set_tracking_index(int new_index);
    bool set_tracking_index(const std::string& name);
    bool set_dt_index(int new_index);
    bool set_dt_index(const std::string& name);

    float get_fa(unsigned int index,unsigned char order) const;
    float get_dt_fa(unsigned int index,unsigned char order) const;
    const float* get_dir(unsigned int index,unsigned int order) const;

};


class fib_data;
class tracking_data{
public:
    tipl::geometry<3> dim;
    tipl::vector<3> vs;
    unsigned char fib_num;
    std::string threshold_name,dt_threshold_name;
    std::vector<const float*> dir;
    std::vector<const float*> fa;
    std::vector<const float*> dt_fa;
    std::vector<const short*> findex;
    std::vector<std::vector<const float*> > other_index;
    std::vector<tipl::vector<3,float> > odf_table;
public:
    bool get_nearest_dir_fib(unsigned int space_index,
                         const tipl::vector<3,float>& ref_dir, // reference direction, should be unit vector
                         unsigned char& fib_order_,
                         unsigned char& reverse_,
                             float threshold,
                             float cull_cos_angle,
                             float dt_threshold) const;
    void read(const fib_data& fib);
    bool get_dir(unsigned int space_index,
                         const tipl::vector<3,float>& dir, // reference direction, should be unit vector
                         tipl::vector<3,float>& main_dir,
                 float threshold,
                 float cull_cos_angle,
                 float dt_threshold) const;
    const float* get_dir(unsigned int space_index,unsigned char fib_order) const;
    float cos_angle(const tipl::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const;
    float get_track_specific_index(unsigned int space_index,unsigned int index_num,
                             const tipl::vector<3,float>& dir) const;
    bool is_white_matter(const tipl::vector<3,float>& pos,float t) const;

};



struct item
{
    std::string name;
    tipl::const_pointer_image<float,3> image_data;
    tipl::matrix<4,4,float> T,iT;// T: image->diffusion iT: diffusion->image
    float max_value;
    float min_value;
    float contrast_max;
    float contrast_min;
    unsigned int max_color = 0x00FFFFFF;
    unsigned int min_color = 0;
    // used in QSDR
    tipl::image<unsigned int,3> color_map_buf;
    tipl::geometry<3> native_geo;
    tipl::transformation_matrix<double> native_trans;


    template<class input_iterator>
    void set_scale(input_iterator from,input_iterator to)
    {
        contrast_max = max_value = *std::max_element(from,to);
        contrast_min = min_value = *std::min_element(from,to);
        if(max_value <= min_value)
        {
            min_value = 0.0f;
            max_value = 1.0f;
        }

    }
};

class fib_data
{
public:
    mutable std::string error_msg;
    std::string report,steps,fib_file_name;
    gz_mat_read mat_reader;
public:
    tipl::geometry<3> dim;
    tipl::vector<3> vs;
    bool is_human_data = true;
    bool is_qsdr = false;
    bool need_normalization = true;
    bool trackable = true;
public:
    fiber_directions dir;
    odf_data odf;
    connectometry_db db;
    std::vector<item> view_item;
public:
    int prog;
    tipl::matrix<4,4,float> trans_to_mni;
    tipl::image<tipl::vector<3,float>,3 > mni_position,inv_mni_position;
public:
    tipl::geometry<3> native_geo;
    tipl::image<tipl::vector<3,float>,3 > native_position;
public:
    int template_id = 0;
    tipl::vector<3> template_vs,template_shift;
    tipl::image<float,3> template_I,template_I2;
    std::vector<std::shared_ptr<atlas> > atlas_list;
public:
    void set_template_id(int new_id);
    bool load_template(void);
    bool load_atlas(void);
    void template_to_mni(tipl::vector<3>& p);
    void template_from_mni(tipl::vector<3>& p);

public:
    void run_normalization(bool background,bool inv);
    bool can_map_to_mni(void);
    void mni2subject(tipl::vector<3>& pos);
    void subject2mni(tipl::vector<3>& pos);
    void subject2mni(tipl::pixel_index<3>& index,tipl::vector<3>& pos);
    void get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,std::vector<tipl::vector<3,short> >& points,float& r);
    const tipl::image<tipl::vector<3,float>,3 >& get_mni_mapping(void);
public:
    fib_data(void)
    {
        vs[0] = vs[1] = vs[2] = 1.0;
        trans_to_mni.identity();
    }
    fib_data(tipl::geometry<3> dim_,tipl::vector<3> vs_):dim(dim_),vs(vs_){}
public:
    bool load_from_file(const char* file_name);
    bool load_from_mat(void);
    bool save_mapping(const std::string& index_name,const std::string& file_name,const tipl::value_to_color<float>& v2c);
public:
    bool has_odfs(void) const{return odf.has_odfs();}
    const float* get_odf_data(unsigned int index) const{return odf.get_odf_data(index);}
public:
    size_t get_name_index(const std::string& index_name) const;
    void get_index_list(std::vector<std::string>& index_list) const;
    std::pair<float,float> get_value_range(const std::string& view_name) const;
    void get_slice(unsigned int view_index,
                   unsigned char d_index,unsigned int pos,
                   tipl::color_image& show_image,const tipl::value_to_color<float>& v2c);
    void get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const;
    void get_voxel_information(int x,int y,int z,std::vector<float>& buf) const;
    void get_index_titles(std::vector<std::string>& titles);
};



template<typename fib_fa_type,typename fun1,typename fun2>
void evaluate_connection(
        const tipl::geometry<3>& dim,
        float otsu,
        const fib_fa_type& fib_fa,
        fun1 dir,
        fun2 f,
        bool check_trajectory = true)
{
    unsigned char num_fib = fib_fa.size();
    char dx[13] = {1,0,0,1,1,0, 1, 1, 0, 1,-1, 1, 1};
    char dy[13] = {0,1,0,1,0,1,-1, 0, 1, 1, 1,-1, 1};
    char dz[13] = {0,0,1,0,1,1, 0,-1,-1, 1, 1, 1,-1};
    std::vector<tipl::vector<3> > dis(13);
    for(unsigned int i = 0;i < 13;++i)
    {
        dis[i] = tipl::vector<3>(dx[i],dy[i],dz[i]);
        dis[i].normalize();
    }
    auto I = tipl::make_image(&fib_fa[0][0],dim);
    I.for_each_mt([&](float fa_value,tipl::pixel_index<3> index)
    {
        if(fa_value <= otsu)
            return;
        for(unsigned char fib1 = 0;fib1 < num_fib;++fib1)
        {
            if(fib_fa[fib1][index.index()] <= otsu)
                break;
            for(unsigned int j = 0;j < 2;++j)
            for(unsigned int i = 0;i < 13;++i)
            {
                tipl::vector<3,int> pos;
                pos = j ? tipl::vector<3,int>(index[0] + dx[i],index[1] + dy[i],index[2] + dz[i])
                          :tipl::vector<3,int>(index[0] - dx[i],index[1] - dy[i],index[2] - dz[i]);
                if(!dim.is_valid(pos))
                    continue;
                tipl::pixel_index<3> other_index(pos[0],pos[1],pos[2],dim);
                if(check_trajectory)
                {
                    if(std::abs(dir(index.index(),fib1)*dis[i]) <= 0.8665)
                        continue;
                    for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                        if(fib_fa[fib2][other_index.index()] > otsu &&
                                std::abs(dir(other_index.index(),fib2)*dis[i]) > 0.8665)
                            f(index.index(),fib1,other_index.index(),fib2);
                }
                else
                {
                    for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                        if(fib_fa[fib2][other_index.index()] > otsu &&
                                std::abs(dir(other_index.index(),fib2)*dir(index.index(),fib1)) > 0.8665)
                            f(index.index(),fib1,other_index.index(),fib2);
                }

            }
        }
    });
}


template<typename fib_fa_type,typename fun>
std::pair<float,float> evaluate_fib(
        const tipl::geometry<3>& dim,
        float otsu,
        const fib_fa_type& fib_fa,
        fun dir,
        bool check_trajectory = true)
{
    float connection_count = 0;
    std::vector<std::vector<unsigned char> > connected(fib_fa.size());
        for(unsigned int index = 0;index < connected.size();++index)
            connected[index].resize(dim.size());

    evaluate_connection(dim,otsu,fib_fa,dir,[&](unsigned int pos1,unsigned char fib1,unsigned int pos2,unsigned char fib2)
    {
        connected[fib1][pos1] = 1;
        connected[fib2][pos2] = 1;
        connection_count += fib_fa[fib2][pos2];
        // no need to add fib1 because it will be counted if fib2 becomes fib1
    },check_trajectory);

    unsigned char num_fib = fib_fa.size();
    float no_connection_count = 0;
    for(tipl::pixel_index<3> index(dim);index < dim.size();++index)
    {
        for(unsigned int i = 0;i < num_fib;++i)
            if(fib_fa[i][index.index()] > otsu && !connected[i][index.index()])
                no_connection_count += fib_fa[i][index.index()];
    }

    return std::make_pair(connection_count,no_connection_count);
}

#endif//FIB_DATA_HPP
