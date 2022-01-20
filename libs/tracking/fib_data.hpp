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
    const float* odfs = nullptr;
    unsigned int odfs_size;
private:
    tipl::image<3,unsigned int> voxel_index_map;
    std::vector<const float*> odf_blocks;
    std::vector<unsigned int> odf_block_size;
    tipl::image<3,unsigned int> odf_block_map1;
    tipl::image<3,unsigned int> odf_block_map2;
    unsigned int half_odf_size;
public:
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
    std::vector<const float*> fa;
    std::vector<tipl::vector<3,float> > odf_table;
    std::vector<tipl::vector<3,unsigned short> > odf_faces;
    unsigned int num_fiber;
    unsigned int half_odf_size;
    std::string error_msg;
public: // for differential tractography
    std::vector<std::shared_ptr<tipl::image<3> > > new_dT;
    std::vector<const float*> dt_fa;
    std::vector<std::string> dt_index_name;
    std::vector<std::vector<const float*> > dt_index_data;
    int dt_cur_index = 0;

    bool is_dt(void)const{return !dt_fa.empty();}
    bool set_dt_index(int new_index);
    bool set_dt_index(const std::string& name);
    std::string get_dt_threshold_name(void) const{return dt_fa.empty() ? std::string() : dt_index_name[uint32_t(dt_cur_index)];}
    void add_dt_index(const std::string& name,tipl::image<3>&& I);
public:
    void check_index(unsigned int index);
    bool add_data(gz_mat_read& mat_reader);
    bool set_tracking_index(int new_index);
    bool set_tracking_index(const std::string& name);
    std::string get_threshold_name(void) const{return index_name[uint32_t(cur_index)];}
    const float* get_fib(size_t space_index,unsigned int order) const;
    float cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const;
    float get_track_specific_metrics(size_t space_index,const std::vector<const float*>& index,
                             const tipl::vector<3,float>& dir) const;
};


class fib_data;
class tracking_data{
public:
    tipl::shape<3> dim;
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
    bool has_high_reso = false;
    float high_reso_ratio = 1.0f;
    std::shared_ptr<tracking_data> high_reso;
private:
    const tracking_data& operator=(const tracking_data& rhs);
public:
    bool get_dir(size_t space_index,
                 const tipl::vector<3,float>& ref_dir, // reference direction, should be unit vector
                 unsigned char& fib_order_,
                 unsigned char& reverse_,
                 float threshold,
                 float cull_cos_angle,
                 float dt_threshold) const;
    void read(std::shared_ptr<fib_data> fib);
    bool get_dir(size_t space_index,
                 const tipl::vector<3,float>& dir, // reference direction, should be unit vector
                 tipl::vector<3,float>& main_dir,
                 float threshold,
                 float cull_cos_angle,
                 float dt_threshold) const;
    const float* get_fib(size_t space_index,unsigned char fib_order) const;
    float cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const;
    bool is_white_matter(const tipl::vector<3,float>& pos,float t) const;

};



struct item
{
private:
    tipl::const_pointer_image<3> image_data;
    tipl::image<3> dummy;
    gz_mat_read* mat_reader = nullptr;
    unsigned int image_index = 0;
public:
    template<typename value_type>
    item(const std::string& name_,const value_type* pointer,const tipl::shape<3>& dim_):
        image_data(tipl::make_image(pointer,dim_)),name(name_)
    {
        set_scale(image_data.begin(),image_data.end());
    }
    item(const std::string& name_,const tipl::shape<3>& dim_,gz_mat_read* mat_reader_,unsigned int index_):
        image_data(tipl::make_image((const float*)nullptr,dim_)),mat_reader(mat_reader_),image_index(index_),name(name_)
    {
        image_ready = false;
    }
    tipl::const_pointer_image<3> get_image(void);
    void set_image(tipl::const_pointer_image<3> new_image){image_data = new_image;}
public:
    std::string name;
    bool image_ready = true;
    tipl::matrix<4,4> T,iT;// T: image->diffusion iT: diffusion->image

public:
    tipl::value_to_color<float> v2c;
    float max_value;
    float min_value;
    float contrast_max;
    float contrast_min;
    unsigned int max_color = 0x00FFFFFF;
    unsigned int min_color = 0;
    tipl::image<3,unsigned int> color_map_buf;

    // for other slice in QSDR, allow for loading t1w-based ROIs in QSDR fib
    tipl::shape<3> native_geo;
    tipl::transformation_matrix<float> native_trans;

    template<class input_iterator>
    void set_scale(input_iterator from,input_iterator to)
    {
        auto result = tipl::min_max_value_mt(from,to);
        contrast_min = min_value = result.first;
        contrast_max = max_value = result.second;
        if(max_value <= min_value)
        {
            min_value = 0.0f;
            max_value = 1.0f;
        }
        v2c.set_range(contrast_min,contrast_max);
        v2c.two_color(min_color,max_color);
    }

};

class TractModel;
class fib_data
{
public:
    mutable std::string error_msg;
    std::string report,steps,fib_file_name;
    gz_mat_read mat_reader;
public:
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4> trans_to_mni;
    bool is_human_data = true;
    bool is_histology = true;
    bool is_qsdr = false;
    bool is_mni_image = false;
    bool is_template_space = false;
    bool trackable = true;
public:
    fiber_directions dir;
    odf_data odf;
    connectometry_db db;
    mutable std::vector<item> view_item;
public:
    bool has_high_reso = false;
    std::shared_ptr<fib_data> high_reso;
public:
    int prog;
    tipl::image<3,tipl::vector<3,float> > s2t,t2s;
private:
    mutable tipl::image<3,tipl::vector<3,float> > native_position;
public:
    tipl::shape<3> native_geo;
    tipl::vector<3> native_vs;
    const tipl::image<3,tipl::vector<3,float> >& get_native_position(void) const;
public:
    size_t template_id = 256;
    tipl::vector<3> template_vs;
    tipl::image<3> template_I,template_I2;
    std::vector<std::shared_ptr<atlas> > atlas_list;
    tipl::matrix<4,4> template_to_mni;
    bool has_manual_atlas = false;
    tipl::transformation_matrix<float> manual_template_T;

public:
    std::string t1w_template_file_name,wm_template_file_name,mask_template_file_name;
public:
    std::shared_ptr<TractModel> track_atlas;
    std::string tractography_atlas_file_name;
    std::vector<std::string> tractography_name_list;
    bool recognize(std::shared_ptr<TractModel>& trk,std::vector<unsigned int>& result,float tolerance);
    bool recognize(std::shared_ptr<TractModel>& trk,std::map<float,std::string,std::greater<float> >& result,bool contain);
    void recognize_report(std::shared_ptr<TractModel>& trk,std::string& report);
    unsigned int find_nearest(const float* trk,unsigned int length,bool contain,float false_distance);
public:
    void match_template(void);
    void set_template_id(size_t new_id);
    bool load_template(void);
    bool load_track_atlas(void);
public:
    void run_normalization(bool background,bool inv);
    bool can_map_to_mni(void);
    void temp2sub(tipl::vector<3>& pos);
    void sub2temp(tipl::vector<3>& pos);
    void sub2mni(tipl::vector<3>& pos);
    std::shared_ptr<atlas> get_atlas(const std::string atlas_name);
    bool get_atlas_roi(const std::string& atlas_name,const std::string& region_name,std::vector<tipl::vector<3,short> >& points);
    bool get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,std::vector<tipl::vector<3,short> >& points);
    bool get_atlas_all_roi(std::shared_ptr<atlas> at,std::vector<std::vector<tipl::vector<3,short> > >& points);
    template<tipl::interpolation Type = tipl::interpolation::linear,typename image_type>
    bool mni2sub(image_type& mni_image,const tipl::matrix<4,4>& trans)
    {
        const auto& s2t = get_sub2temp_mapping();
        if(s2t.empty())
        {
            error_msg = "No spatial mapping found for warpping MNI images";
            return false;
        }
        image_type J(s2t.shape()); // subject space image

        // from template space to mni image's space
        auto T = tipl::from_space(template_to_mni).to(trans);
        tipl::par_for(J.size(),[&](size_t index)
        {
            tipl::vector<3> pos(s2t[index]);
            pos.to(T);
            tipl::estimate<Type>(mni_image,pos,J[index]);
        });
        mni_image.swap(J);
        return true;
    }
    const tipl::image<3,tipl::vector<3,float> >& get_sub2temp_mapping(void);

public:
    fib_data(void)
    {
        vs[0] = vs[1] = vs[2] = 1.0;
        trans_to_mni.identity();
    }
    fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_);
    fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_);
public:
    bool load_from_file(const char* file_name);
    bool load_from_mat(void);
    bool save_mapping(const std::string& index_name,const std::string& file_name);
public:
    bool has_odfs(void) const{return odf.has_odfs();}
    const float* get_odf_data(unsigned int index) const{return odf.get_odf_data(index);}
public:
    size_t get_name_index(const std::string& index_name) const;
    void get_index_list(std::vector<std::string>& index_list) const;
public:
    bool add_dT_index(const std::string& index_name);
public:
    std::pair<float,float> get_value_range(const std::string& view_name) const;
    void get_slice(unsigned int view_index,
                   unsigned char d_index,unsigned int pos,
                   tipl::color_image& show_image);
    void get_voxel_info2(int x,int y,int z,std::vector<float>& buf) const;
    void get_voxel_information(int x,int y,int z,std::vector<float>& buf) const;
    void get_iso_fa(tipl::image<3>& iso_fa_) const;
};



template<typename fib_fa_type,typename fun1,typename fun2>
void evaluate_connection(
        const tipl::shape<3>& dim,
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
    const double angular_threshold = 0.984;
    for(unsigned int i = 0;i < 13;++i)
    {
        dis[i] = tipl::vector<3>(dx[i],dy[i],dz[i]);
        dis[i].normalize();
    }
    auto I = tipl::make_image(&fib_fa[0][0],dim);
    I.for_each<tipl::backend::mt>([&](float fa_value,tipl::pixel_index<3> index)
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
                    if(std::abs(dir(index.index(),fib1)*dis[i]) <= angular_threshold)
                        continue;
                    for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                        if(fib_fa[fib2][other_index.index()] > otsu &&
                                std::abs(dir(other_index.index(),fib2)*dis[i]) > angular_threshold)
                            f(index.index(),fib1,other_index.index(),fib2);
                }
                else
                {
                    for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                        if(fib_fa[fib2][other_index.index()] > otsu &&
                                std::abs(dir(other_index.index(),fib2)*dir(index.index(),fib1)) > angular_threshold)
                            f(index.index(),fib1,other_index.index(),fib2);
                }

            }
        }
    });
}


template<typename fib_fa_type,typename fun>
std::pair<float,float> evaluate_fib(
        const tipl::shape<3>& dim,
        float otsu,
        const fib_fa_type& fib_fa,
        fun dir,
        bool check_trajectory = true)
{
    double connection_count = 0.0;
    std::vector<std::vector<unsigned char> > connected(fib_fa.size());
        for(unsigned int index = 0;index < connected.size();++index)
            connected[index].resize(dim.size());

    std::mutex add_mutex;
    evaluate_connection(dim,otsu,fib_fa,dir,[&](unsigned int pos1,unsigned char fib1,unsigned int pos2,unsigned char fib2)
    {
        connected[fib1][pos1] = 1;
        connected[fib2][pos2] = 1;
        auto v = fib_fa[fib2][pos2];
        std::lock_guard<std::mutex> lock(add_mutex);
        connection_count += double(v);
        // no need to add fib1 because it will be counted if fib2 becomes fib1
    },check_trajectory);

    unsigned char num_fib = fib_fa.size();
    double no_connection_count = 0.0;
    for(tipl::pixel_index<3> index(dim);index < dim.size();++index)
    {
        for(unsigned int i = 0;i < num_fib;++i)
            if(fib_fa[i][index.index()] > otsu && !connected[i][index.index()])
                no_connection_count += double(fib_fa[i][index.index()]);
    }

    return std::make_pair(connection_count,no_connection_count);
}

#endif//FIB_DATA_HPP
