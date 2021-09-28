#ifndef TRACT_MODEL_HPP
#define TRACT_MODEL_HPP
#include <vector>
#include <iosfwd>
#include "tipl/tipl.hpp"
#include "fib_data.hpp"

class RoiMgr;
void initial_LPS_nifti_srow(tipl::matrix<4,4,float>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
class TractModel{
public:
        std::string report;
        std::string parameter_id;
        bool saved = true;
        bool color_changed = false;
public:
        tipl::shape<3> geo;
        tipl::vector<3> vs;
        tipl::matrix<4,4,float> trans_to_mni;
private:
        std::vector<std::vector<float> > tract_data;
        std::vector<std::vector<float> > deleted_tract_data;
        std::vector<unsigned int> tract_color;
        std::vector<unsigned int> tract_tag;
        std::vector<unsigned int> deleted_tract_color;
        std::vector<unsigned int> deleted_tract_tag;
        std::vector<unsigned int> deleted_count;
        std::vector<char> is_cut;
        unsigned int cur_cut_id = 1;
        std::vector<std::pair<unsigned int,unsigned int> > redo_size;
        // offset, size
        void erase_empty(void);
private:
        // for loading multiple clusters
        std::vector<unsigned int> tract_cluster;
public:
        static bool save_all(const char* file_name,
                             const std::vector<std::shared_ptr<TractModel> >& all,
                             const std::vector<std::string>& name_list);
        const std::vector<unsigned int>& get_cluster_info(void) const{return tract_cluster;}
        std::vector<unsigned int>& get_cluster_info(void) {return tract_cluster;}
        void select(float select_angle,
                    const std::vector<tipl::vector<3,float> >& dirs,
                    const tipl::vector<3,float>& from_pos,std::vector<unsigned int>& selected);
        // selection
        void delete_tracts(const std::vector<unsigned int>& tracts_to_delete);
        void select_tracts(const std::vector<unsigned int>& tracts_to_select);
        void delete_repeated(float d);
        void delete_branch(void);
        void delete_by_length(float length);
public:
        TractModel(std::shared_ptr<fib_data> handle):geo(handle->dim),vs(handle->vs),trans_to_mni(handle->trans_to_mni){}
        TractModel(tipl::shape<3> dim_,tipl::vector<3> vs_):geo(dim_),vs(vs_)
        {
            initial_LPS_nifti_srow(trans_to_mni,geo,vs);
        }
        TractModel(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4,float>& trans_to_mni_)
            :geo(dim_),vs(vs_),trans_to_mni(trans_to_mni_)
        {}

        TractModel(const TractModel& rhs)
        {
            (*this) = rhs;
        }
        const TractModel& operator=(const TractModel& rhs)
        {
            geo = rhs.geo;
            vs = rhs.vs;
            trans_to_mni = rhs.trans_to_mni;
            tract_data = rhs.tract_data;
            tract_color = rhs.tract_color;
            tract_tag = rhs.tract_tag;
            report = rhs.report;
            saved = true;
            return *this;
        }
        void add(const TractModel& rhs);
        bool load_from_file(const char* file_name,bool append = false);

        bool save_tracts_to_file(const char* file_name);
        bool save_tracts_in_native_space(std::shared_ptr<fib_data> handle,const char* file_name);
        bool save_tracts_in_template_space(std::shared_ptr<fib_data> handle,const char* file_name);
        bool save_transformed_tracts_to_file(const char* file_name,tipl::shape<3> new_dim,
                                             tipl::vector<3> new_vs,const tipl::matrix<4,4,float>& T,bool end_point);

        void save_vrml(const std::string& file_name,
                       unsigned char tract_style,
                       unsigned char tract_color_style,
                       float tube_diameter,
                       unsigned char tract_tube_detail,
                       const std::string& surface_text);
        bool save_data_to_file(std::shared_ptr<fib_data> handle,const char* file_name,const std::string& index_name);
        bool save_end_points(const char* file_name) const;

        bool load_tracts_color_from_file(const char* file_name);
        bool save_tracts_color_to_file(const char* file_name);


        void release_tracts(std::vector<std::vector<float> >& released_tracks);
        void clear(void);
        void add_tracts(std::vector<std::vector<float> >& new_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,tipl::rgb color);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,unsigned int length_threshold,tipl::rgb color);
        void filter_by_roi(std::shared_ptr<RoiMgr> roi_mgr);
        void reconnect_track(float distance,float angular_threshold);
        void cull(float select_angle,
                  const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  bool delete_track);
        void cut(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos);
        void cut_by_slice(unsigned int dim, unsigned int pos,bool greater,const tipl::matrix<4,4,float>* T = nullptr);
        void paint(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  unsigned int color);
        void set_color(unsigned int color){std::fill(tract_color.begin(),tract_color.end(),color);color_changed = true;}
        void set_tract_color(std::vector<unsigned int>& new_color){tract_color = new_color;color_changed = true;}
        void cut_by_mask(const char* file_name);
        void clear_deleted(void);
        void undo(void);
        void redo(void);
        bool trim(void);
        void resample(float new_step);
        void get_tract_points(std::vector<tipl::vector<3,float> >& points);
        void get_in_slice_tracts(unsigned char dim,int pos,
                                 tipl::matrix<4,4,float>* T,
                                 std::vector<std::vector<tipl::vector<2,float> > >& lines,
                                 std::vector<unsigned int>& colors,
                                 unsigned int max_count);
        void to_voxel(std::vector<tipl::vector<3,short> >& points,float ratio,int id = -1);
        void to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                std::vector<tipl::vector<3,short> >& points2,float ratio);
        void to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                std::vector<tipl::vector<3,short> >& points2,float ratio,float end_dis);

        size_t get_deleted_track_count(void) const{return deleted_tract_data.size();}
        size_t get_visible_track_count(void) const{return tract_data.size();}
        
        const std::vector<float>& get_tract(unsigned int index) const{return tract_data[index];}
        const std::vector<std::vector<float> >& get_tracts(void) const{return tract_data;}
        std::vector<std::vector<float> >& get_deleted_tracts(void) {return deleted_tract_data;}
        std::vector<std::vector<float> >& get_tracts(void) {return tract_data;}
        unsigned int get_tract_color(unsigned int index) const{return tract_color[index];}
        size_t get_tract_length(unsigned int index) const{return tract_data[index].size();}

public:
        void get_density_map(tipl::image<unsigned int,3>& mapping,
             const tipl::matrix<4,4,float>& transformation,bool endpoint);
        void get_density_map(tipl::image<tipl::rgb,3>& mapping,
             const tipl::matrix<4,4,float>& transformation,bool endpoint);
        static bool export_tdi(const char* file_name,
                          std::vector<std::shared_ptr<TractModel> > tract_models,
                          tipl::shape<3>& dim,
                          tipl::vector<3,float> vs,
                          tipl::matrix<4,4,float> transformation,bool color,bool end_point);
        static bool export_pdi(const char* file_name,
                               const std::vector<std::shared_ptr<TractModel> >& tract_models);
        static bool export_end_pdi(const char* file_name,
                               const std::vector<std::shared_ptr<TractModel> >& tract_models,float end_distance = 3.0f);
public:
        void get_quantitative_info(std::shared_ptr<fib_data> handle,std::string& result);
        tipl::vector<3> get_report(std::shared_ptr<fib_data> handle,
                        unsigned int profile_dir,float band_width,const std::string& index_name,
                        std::vector<float>& values,
                        std::vector<float>& data_profile,
                        std::vector<float>& data_ci1,
                        std::vector<float>& data_ci2);

public:
        void get_tract_data(std::shared_ptr<fib_data> handle,
                            unsigned int fiber_index,
                            unsigned int index_num,
                            std::vector<float>& data) const;
        bool get_tracts_data(std::shared_ptr<fib_data> handle,
                const std::string& index_name,
                std::vector<std::vector<float> >& data) const;
        void get_tracts_data(std::shared_ptr<fib_data> handle,unsigned int index_num,float& mean) const;
public:

        void get_passing_list(const tipl::image<std::vector<short>,3>& region_map,
                              unsigned int region_count,
                                     std::vector<std::vector<short> >& passing_list1,
                                     std::vector<std::vector<short> >& passing_list2) const;
        void get_end_list(const tipl::image<std::vector<short>,3>& region_map,
                                     std::vector<std::vector<short> >& end_list1,
                                     std::vector<std::vector<short> >& end_list2) const;
        void run_clustering(unsigned char method_id,unsigned int cluster_count,float param);

};




class atlas;
class ROIRegion;
class ConnectivityMatrix{
public:

    tipl::image<float,2> matrix_value;
public:
    tipl::image<std::vector<short>,3> region_map;
    size_t region_count = 0;
    std::vector<std::string> region_name;
    std::string error_msg,atlas_name;
    float overlap_ratio;
    bool set_atlas(std::shared_ptr<atlas> data,std::shared_ptr<fib_data> handle);
    void set_regions(const tipl::shape<3>& geo,
                     const std::vector<std::shared_ptr<ROIRegion> >& regions);
public:
    void save_to_image(tipl::color_image& cm);
    void save_to_file(const char* file_name);
    void save_to_connectogram(const char* file_name);
    void save_to_text(std::string& text);
    bool calculate(std::shared_ptr<fib_data> handle,TractModel& tract_model,std::string matrix_value_type,bool use_end_only,float threshold);
    void network_property(std::string& report);
};




#endif//TRACT_MODEL_HPP
