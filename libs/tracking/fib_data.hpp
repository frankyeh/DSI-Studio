#ifndef FIB_DATA_HPP
#define FIB_DATA_HPP
#include <fstream>
#include <sstream>
#include <string>
#include "connectometry_db.hpp"
#include "atlas.hpp"

class fib_data;
struct odf_data{
private:
    tipl::image<3,const float*> odf_map;
public:
    std::string error_msg;
    bool read(fib_data& fib);
    bool has_odfs(void) const {return !odf_map.empty();}
    const float* get_odf_data(size_t index){return odf_map[index];}
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
    tipl::shape<3> dim;
    std::vector<const float*> fa;
    float fa_otsu;
    std::vector<tipl::vector<3,float> > odf_table;
    std::vector<tipl::vector<3,unsigned short> > odf_faces;
    unsigned int num_fiber;
    unsigned int half_odf_size;
    std::string error_msg;
public: // for differential tractography
    std::shared_ptr<tipl::image<3> > dt_fa_data;
    std::vector<const float*> dt_fa;
    std::string dt_threshold_name;
public:
    void check_index(unsigned int index);
    bool add_data(fib_data& fib);
    bool set_tracking_index(int new_index);
    bool set_tracking_index(const std::string& name);
    std::string get_threshold_name(void) const{return index_name[uint32_t(cur_index)];}
    tipl::vector<3> get_fib(size_t space_index,unsigned int order) const;
    float cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const;
    float get_track_specific_metrics(size_t space_index,const std::vector<const float*>& index,
                             const tipl::vector<3,float>& dir) const;
};

class tracking_data{
public:
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    unsigned char fib_num;
    float fa_otsu;
    std::string threshold_name,dt_threshold_name;
    std::vector<const float*> dir;
    std::vector<const float*> fa;
    std::vector<const float*> dt_fa;
    std::vector<const short*> findex;
    std::vector<tipl::vector<3,float> > odf_table;
    std::shared_ptr<tipl::image<3> > dt_fa_data;

    const tracking_data& operator=(const tracking_data& rhs) = delete;
public:
    void read(std::shared_ptr<fib_data> fib);
    inline bool get_dir_under_termination_criteria(
                 const tipl::vector<3,float>& position,
                 const tipl::vector<3,float>& ref_dir, // reference direction, should be unit vector
                 tipl::vector<3,float>& result, // reference direction, should be unit vector
                 float threshold,
                 float cull_cos_angle,
                 float dt_threshold) const
    {
        tipl::interpolator::linear<3> tri_interpo;
        if (!tri_interpo.get_location(dim,position))
            return false;
        tipl::vector<3,float> new_dir,main_dir;
        float total_weighting = 0.0f;
        for (unsigned char index = 0;index < 8;++index)
        {
            size_t space_index = tri_interpo.dindex[index];
            float max_value = cull_cos_angle;
            unsigned char fib_order = 0;
            unsigned char reverse = 0;
            for (unsigned char index = 0;index < fib_num && fa[index][space_index] > threshold;++index)
            {
                if (!dt_fa.empty() && dt_fa[index][space_index] <= dt_threshold) // for differential tractography
                    continue;
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
            if (max_value <= cull_cos_angle)
                continue;
            if(!dir.empty())
                main_dir = dir[fib_order] + space_index + (space_index << 1);
            else
                main_dir = odf_table[findex[fib_order][space_index]];
            if(reverse)
            {
                main_dir[0] = -main_dir[0];
                main_dir[1] = -main_dir[1];
                main_dir[2] = -main_dir[2];
            }
            float w = tri_interpo.ratio[index];
            main_dir *= w;
            new_dir += main_dir;
            total_weighting += w;
        }
        if (total_weighting < 0.5f)
            return false;
        new_dir.normalize();
        result = new_dir;
        return true;
    }
    inline tipl::vector<3> get_fib(size_t space_index,unsigned char fib_order) const
    {
        if(!dir.empty())
            return tipl::vector<3>(dir[fib_order] + space_index + (space_index << 1));
        return odf_table[findex[fib_order][space_index]];
    }
    inline float cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const
    {
        if(!dir.empty())
        {
            const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
            return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
        }
        return cur_dir*odf_table[findex[fib_order][space_index]];
    }

};


class fib_data;
struct slice_model
{
private:
    fib_data* handle = nullptr;
    tipl::const_pointer_image<3> image_data;
public:
    template<typename value_type>
    slice_model(const std::string& name_,const value_type* pointer,const tipl::shape<3>& dim_):
                image_data(tipl::make_image(pointer,dim_)),name(name_)
    {
    }
    slice_model(const std::string& name_,fib_data* handle_):
        handle(handle_),name(name_)
    {
    }
    slice_model(const std::string& path_):
        name(std::filesystem::path(path_).stem().stem().string()),path(path_)
    {
    }
    slice_model(const std::string& name_,const std::string& path_):
        name(name_),path(path_)
    {
    }
    tipl::const_pointer_image<3> get_image(void);
    bool image_ready(void){return image_data.data();}
    bool optional(void){return !image_data.data() && !path.empty();}
    void get_image_in_dwi(tipl::image<3>& I);
    void set_image(tipl::const_pointer_image<3> new_image)
    {
        image_data = new_image;
        max_value = 0.0f;
        get_minmax();
    }
public:
    std::string name,path;
    bool registering = false;
    tipl::matrix<4,4> T = tipl::identity_matrix(),iT = tipl::identity_matrix();// T: image->diffusion iT: diffusion->image

public:
    tipl::value_to_color<float> v2c;
    float max_value = 0.0f;
    float min_value = 0.0f;
    float contrast_max = 0.0f;
    float contrast_min = 0.0f;
    unsigned int max_color = 0x00FFFFFF;
    unsigned int min_color = 0;
    tipl::image<3,unsigned int> color_map_buf;

    void get_minmax(void);
    void get_slice(unsigned char d_index,unsigned int pos,
                   tipl::color_image& show_image);
};

class TractModel;
class fib_data
{
public:
    mutable std::string error_msg;
    std::string report,steps,intro,other_images,fib_file_name;
    std::string demo; // used in cli for dT analysis
    tipl::io::gz_mat_read mat_reader;
public:
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4> trans_to_mni;
    bool is_human_data = true;
    bool is_histology = true;
    bool is_mni = false;
    size_t matched_template_id = 0;
    bool trackable = true;
    float min_length(void) const
    {
        if(template_id == 0) // human
            return 30.0f;
        float min_length = dim[0]*vs[0]/4;
        float min_length_digit = float(std::pow(10.0f,std::floor(std::log10(double(min_length)))));
        return int(min_length/min_length_digit)*min_length_digit;

    }
    float max_length(void) const
    {
        if(template_id == 0) // human
            return 200.0f;
        float max_length = dim[1]*vs[1]*1.5;
        float max_length_digit = float(std::pow(10.0f,std::floor(std::log10(double(max_length)))));
        return int(max_length/max_length_digit)*max_length_digit;
    }
public:
    tipl::const_pointer_image<3,unsigned char> mask;
public:
    fiber_directions dir;
    connectometry_db db;
    mutable std::vector<std::shared_ptr<slice_model> > slices;
    void remove_slice(size_t index);
public:
    std::shared_ptr<fib_data> high_reso;
public:
    int prog = 0;
    tipl::image<3,tipl::vector<3,float> > s2t,t2s;
public:
    size_t template_id = 256;
    tipl::vector<3> template_vs;
    tipl::image<3> template_I,template_I2;
    std::vector<std::shared_ptr<atlas> > atlas_list;
    tipl::matrix<4,4> template_to_mni;
    bool has_manual_atlas = false;
    tipl::affine_transform<float> manual_template_T;
    std::vector<std::shared_ptr<std::thread> > reg_threads;
public:
    std::vector<std::string> alternate_mapping;
    size_t alternate_mapping_index = 0;
public:
    std::string t1w_template_file_name,t2w_template_file_name,wm_template_file_name,mask_template_file_name;
    std::string tractography_atlas_file_name;
    std::shared_ptr<atlas> tractography_atlas_roi,tractography_atlas_roa;
public:
    std::vector<std::string> tractography_name_list;
    std::vector<std::string> get_tractography_all_levels(void);
    std::vector<std::string> get_tractography_level0(void);
    std::vector<std::string> get_tractography_level1(const std::string& group);
    std::vector<std::string> get_tractography_level2(const std::string& group1,const std::string& group2);

    std::shared_ptr<TractModel> track_atlas;
    std::vector<float> tract_atlas_min_length,tract_atlas_max_length;
    float tract_atlas_jacobian = 0.0f;
    bool recognize(std::shared_ptr<TractModel>& trk,
                   std::vector<unsigned int>& labels,
                   std::vector<unsigned int>& label_count);
    bool recognize(std::shared_ptr<TractModel>& trk,
                   std::vector<unsigned int>& labels,
                   std::vector<std::string> & label_names);
    bool recognize_and_sort(std::shared_ptr<TractModel>& trk,std::multimap<float,std::string,std::greater<float> >& result);
    void recognize_report(std::shared_ptr<TractModel>& trk,std::string& report);
public:
    void match_template(void);
    void set_template_id(size_t new_id);
    bool load_template(void);
    bool load_track_atlas(void);
    std::vector<size_t> get_track_ids(const std::string& tract_name);
    std::pair<float,float> get_track_minmax_length(const std::string& tract_name);
public:
    bool map_to_mni(bool background = true);
    void temp2sub(std::vector<std::vector<float> >&tracts) const;
    void temp2sub(tipl::vector<3>& pos) const;
    void sub2temp(tipl::vector<3>& pos);
    void sub2mni(tipl::vector<3>& pos);
    void mni2sub(tipl::vector<3>& pos);
    std::shared_ptr<atlas> get_atlas(const std::string atlas_name);
    bool get_atlas_roi(const std::string& atlas_name,const std::string& region_name,std::vector<tipl::vector<3,short> >& points);
    bool get_atlas_roi(std::shared_ptr<atlas> at,const std::string& region_name,std::vector<tipl::vector<3,short> >& points);
    bool get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,
                       const tipl::shape<3>& new_geo,const tipl::matrix<4,4>& new_trans,
                       std::vector<tipl::vector<3,short> >& points);
    bool get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,
                       std::vector<tipl::vector<3,short> >& points)
    {
        return get_atlas_roi(at,roi_index,dim,tipl::matrix<4,4>(tipl::identity_matrix()),points);
    }
    bool get_atlas_all_roi(std::shared_ptr<atlas> at,
                           const tipl::shape<3>& new_geo,const tipl::matrix<4,4>& new_trans,
                           std::vector<std::vector<tipl::vector<3,short> > >& points,
                           std::vector<std::string>& labels);
    template<tipl::interpolation Type = tipl::interpolation::linear,typename image_type>
    bool mni2sub(image_type& mni_image,const tipl::matrix<4,4>& trans,float ratio = 1.0f)
    {
        const auto& s2t = get_sub2temp_mapping();
        if(s2t.empty())
        {
            error_msg = "No spatial mapping found for warping MNI images";
            return false;
        }
        image_type J(s2t.shape()*ratio); // subject space image

        // from template space to mni image's space
        auto T = tipl::from_space(template_to_mni).to(trans);
        tipl::adaptive_par_for(tipl::begin_index(J.shape()),tipl::end_index(J.shape()),
        [&](const tipl::pixel_index<3>& index)
        {
            tipl::vector<3> pos;
            if(ratio == 1.0f)
                pos = s2t[index.index()];
            else
                tipl::estimate<Type>(s2t,tipl::vector<3>(index)/ratio,pos);
            pos.to(T);
            tipl::estimate<Type>(mni_image,pos,J[index.index()]);
        });
        mni_image.swap(J);
        return true;
    }
    const tipl::image<3,tipl::vector<3,float> >& get_sub2temp_mapping(void);
    bool load_mapping(const std::string& file_name);
public:
    fib_data(void)
    {
        vs[0] = vs[1] = vs[2] = 1.0;
        trans_to_mni.identity();
    }
    ~fib_data(void)
    {
        for(auto& each : reg_threads)
            each->join();
    }
    fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_);
    fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_);
public:
    bool load_from_file(const std::string& file_name);
    bool save_to_file(const std::string& file_name);
    bool load_from_mat(void);
    bool save_mapping(const std::string& index_name,const std::string& file_name);
    bool load_at_resolution(const std::string& file_name,float vs);
public:
    bool has_odfs(void) const{return mat_reader.has("odf0");}
public:
    size_t get_name_index(const std::string& index_name) const;
    std::vector<std::string> get_index_list(void) const;
public:
    bool set_dt_index(const std::pair<std::string,std::string>& pair,size_t type);
public:

    void get_voxel_info2(int x,int y,int z,std::vector<float>& buf) const;
    void get_voxel_information(int x,int y,int z,std::vector<float>& buf) const;
    auto get_iso(void) const
    {
        size_t index = get_name_index("iso");
        if(slices.size() == index)
            index = get_name_index("md");
        if(slices.size() == index)
            index = 0;
        return slices[index]->get_image();
    }
    auto get_iso_fa(void) const
    {
        return std::make_pair(slices[0]->get_image(),get_iso());
    }
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
    tipl::adaptive_par_for(tipl::begin_index(dim),tipl::end_index(dim),
                [&](const tipl::pixel_index<3>& index)
    {
        if(fib_fa[0][index.index()] <= otsu)
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
