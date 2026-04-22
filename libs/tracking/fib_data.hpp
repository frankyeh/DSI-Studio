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
    std::vector<std::pair<std::string,std::vector<const float*> > > index_name_data;
    int cur_index = 0;
public:
    tipl::shape<3> dim;
    std::vector<const float*> fa;
    float fa_otsu;
    std::vector<tipl::vector<3,float> > odf_table;
    std::vector<tipl::vector<3,unsigned short> > odf_faces;
    unsigned int num_fiber = 1;
    unsigned int half_odf_size;
    std::string error_msg;
public: // for differential tractography
    tipl::image<3> dt_fa_data;
    std::vector<const float*> dt_fa;
    std::string dt_metrics;
public:
    bool add_data(fib_data& fib);
    bool set_tracking_index(int new_index);
    bool set_tracking_index(const std::string& name);
    std::string get_threshold_name(void) const{return index_name_data[uint32_t(cur_index)].first;}
    tipl::vector<3> get_fib(size_t space_index,unsigned int order) const;
    float cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const;
    float get_track_specific_metrics(size_t space_index,const std::vector<const float*>& index,
                             const tipl::vector<3,float>& dir) const;
};

class tracking_data{
public:
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    unsigned char num_fiber = 1;
    float fa_otsu;
    std::string threshold_name,dt_metrics;
    std::vector<const float*> dir;
    std::vector<const float*> fa;
    std::vector<const float*> dt_fa;
    std::vector<const short*> findex;
    std::vector<tipl::vector<3,float> > odf_table;

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
            for (unsigned char index = 0;index < num_fiber && fa[index][space_index] > threshold;++index)
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
    tipl::image<3> image_buffer;
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
        name(tipl::remove_all_suffix(std::filesystem::path(path_).filename().string())),path(path_)
    {
    }
    slice_model(const std::string& name_,const std::string& path_):
        name(name_),path(path_)
    {
    }
    std::mutex get_image_mutex;
    tipl::const_pointer_image<3> get_image(void);
    bool image_ready(void){return image_data.data();}
    bool optional(void);
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
    tipl::io::gz_mat_read mat_reader;
public:
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4> trans_to_mni;
    template<typename T>
    auto bind(const T& I) const
    {
        return std::tie(vs,trans_to_mni,is_mni,I);
    }
    bool is_human_data = true;
    bool is_histology = true;
    bool is_mni = false;
    bool is_be = false;
    size_t matched_template_id = 0;
    bool trackable = true;
    float default_template_vs(void) const
    {
        const float template_default_vs[] = {
        2.0f,   //  human
        1.25f,  //  neonate
        1.25f,  //  chimpanzee
        1.00f,  //  rhesus
        0.25f,  //  marmoset
        0.25f,  //  rat
        0.1f,   //  mouse
        1.0f,
        1.0f
        };
        return template_default_vs[template_id];
    }
    float default_track_voxel_ratio(void) const
    {
        float ratio = vs[0] / default_template_vs();
        return ratio*ratio;
    }
    float default_min_length(void) const
    {
        return 15.0f*default_template_vs();
    }
    float default_tolerance(void) const
    {
        return 12.0f*default_template_vs();
    }
    float default_max_length(void) const
    {
        return 100.0f*default_template_vs();
    }
public:
    tipl::const_pointer_image<3,unsigned char> mask;
    fiber_directions dir;
    mutable std::vector<std::shared_ptr<slice_model> > slices;
    void remove_slice(size_t index);
public:
    connectometry_db db;
public:
    std::shared_ptr<fib_data> high_reso;
public:
    int prog = 0;
    tipl::image<3,tipl::vector<3,float> > s2t,t2s;
    float R2 = 1.0f;
    unsigned int search_count = 32;
public:
    size_t template_id = 0;
    tipl::vector<3> template_vs;
    tipl::image<3> template_I,template_I2;
public:
    std::vector<std::shared_ptr<atlas> > atlas_list;
    bool add_atlas(const std::string& file_name);
public:
    tipl::matrix<4,4> template_to_mni;
    bool has_manual_atlas = false;
    tipl::affine_param<float> manual_template_T;
    std::vector<std::shared_ptr<std::thread> > reg_threads;
public:
    std::vector<std::string> alternative_mapping;
    size_t alternative_mapping_index = 0;
public:
    std::string t1w_template_file_name,t2w_template_file_name,wm_template_file_name;
    std::vector<std::string> tractography_atlas_list;
    std::string tractography_atlas_file_name;
    std::shared_ptr<atlas> tractography_atlas_roi,tractography_atlas_roa;
public:
    std::vector<std::string> tractography_name_list;
    bool load_tractography_name_list(void);
    std::vector<std::string> get_tractography_all_levels(void);
    std::vector<std::string> get_tractography_level0(void);
    std::vector<std::string> get_tractography_level1(const std::string& group);
    std::vector<std::string> get_tractography_level2(const std::string& group1,const std::string& group2);

    std::shared_ptr<TractModel> track_atlas;
    std::vector<float> tract_atlas_min_length,tract_atlas_median_length,tract_atlas_max_length;
    float tract_atlas_jacobian = 0.0f;
    bool recognize(std::shared_ptr<TractModel>& trk,
                   std::vector<unsigned int>& labels,
                   std::vector<unsigned int>& label_count);
    bool recognize(std::shared_ptr<TractModel>& trk,
                   std::vector<unsigned int>& labels,
                   std::vector<std::string> & label_names);
    std::multimap<float,std::string,std::greater<float> > recognize_and_sort(std::shared_ptr<TractModel> trk);
    void recognize_report(std::shared_ptr<TractModel>& trk,std::string& report);
public:
    void match_template(void);
    void set_tractography_atlas(const std::string& atlas_file_name);
    void set_template_id(size_t new_id);
    bool load_template(void);
    bool load_track_atlas(bool symmetric);
    std::vector<size_t> get_track_ids(const std::string& tract_name);
    std::pair<float,float> get_track_minmax_length(const std::string& tract_name);
    float get_track_median_length(const std::string& tract_name);
public:
    std::string get_mapping_file_name(void) const;
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
        tipl::par_for<tipl::sequential>(J.shape(),[&](const tipl::pixel_index<3>& index)
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
    bool save_slice(const std::string& index_name,const std::string& file_name,bool in_mni = false);
    bool load_template_fib(size_t template_id,float vs);
    bool correct_bias_field(void);
public:
    bool has_odfs(void) const{return mat_reader.has("odf0");}
public:
    size_t get_name_index(const std::string& index_name) const;
    std::vector<std::string> get_index_list() const;
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




template<typename fib_fa_type, typename fun>
float evaluate_fib(
        const tipl::shape<3>& dim,
        float otsu,
        const fib_fa_type& fib_fa,
        fun dir)
{
    std::atomic<double> connection_count{0.0};
    std::atomic<double> total_fa{0.0};
    const double angular_threshold = 0.9659;

    auto add_atomic = [](std::atomic<double>& target, double val)
    {
        double current = target.load();
        while(!target.compare_exchange_weak(current, current + val))
            continue;
    };

    tipl::par_for(dim,[&](const tipl::pixel_index<3>& index)
    {
        auto v1 = fib_fa[index.index()];
        if(v1 <= otsu)
            return;

        auto d = dir(index.index());
        tipl::vector<3,int> oxyz(std::round(d[0]),std::round(d[1]),std::round(d[2]));
        tipl::vector<3,int> pos[2];
        pos[0] = tipl::vector<3,int>(index[0] + oxyz[0], index[1] + oxyz[1], index[2] + oxyz[2]);
        pos[1] = tipl::vector<3,int>(index[0] - oxyz[0], index[1] - oxyz[1], index[2] - oxyz[2]);

        bool connected = false;
        for(int j = 0; j < 2; ++j)
        {
            if(!dim.is_valid(pos[j]))
                continue;
            tipl::pixel_index<3> other_index(pos[j].begin(), dim);
            if(std::abs(dir(other_index.index()) * d) >= angular_threshold)
            {
                connected = true;
                goto end;
            }
        }
    end:
        add_atomic(total_fa, double(v1));
        if(connected)
            add_atomic(connection_count, double(v1));
    });

    double final_total = total_fa.load();
    if(final_total != 0.0)
        return connection_count.load() / final_total;

    return 0.0;
}

#endif//FIB_DATA_HPP
