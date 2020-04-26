#ifndef TRACT_MODEL_HPP
#define TRACT_MODEL_HPP
#include <vector>
#include <iosfwd>
#include "tipl/tipl.hpp"
#include "fib_data.hpp"

class RoiMgr;
class TractModel{
public:
        std::string report;
        std::string parameter_id;
        bool saved = true;
private:
        std::shared_ptr<fib_data> handle;
        tipl::geometry<3> geometry;
        tipl::vector<3> vs;
        std::shared_ptr<tracking_data> fib;
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
        unsigned int find_nearest(const float* trk,unsigned int length,bool contain,float false_distance);

public:
        TractModel(std::shared_ptr<fib_data> handle_);
        const TractModel& operator=(const TractModel& rhs)
        {
            geometry = rhs.geometry;
            vs = rhs.vs;
            handle = rhs.handle;
            tract_data = rhs.tract_data;
            tract_color = rhs.tract_color;
            tract_tag = rhs.tract_tag;
            report = rhs.report;
            saved = true;
            return *this;
        }
        std::shared_ptr<fib_data> get_handle(void){return handle;}
        const tracking_data& get_fib(void) const{return *fib.get();}
        tracking_data& get_fib(void){return *fib.get();}
        void add(const TractModel& rhs);
        bool load_from_file(const char* file_name,bool append = false);

        bool save_tracts_in_native_space(const char* file_name,tipl::image<tipl::vector<3,float>,3 > native_position);
        bool save_tracts_to_file(const char* file_name);
        void save_vrml(const char* file_name,
                       unsigned char tract_style,
                       unsigned char tract_color_style,
                       float tube_diameter,
                       unsigned char tract_tube_detail,
                       const std::string& surface_text);
        bool save_transformed_tracts_to_file(const char* file_name,const float* transform,bool end_point);
        bool save_data_to_file(const char* file_name,const std::string& index_name);
        void save_end_points(const char* file_name) const;

        bool load_tracts_color_from_file(const char* file_name);
        bool save_tracts_color_to_file(const char* file_name);


        void release_tracts(std::vector<std::vector<float> >& released_tracks);
        void clear(void);
        void add_tracts(std::vector<std::vector<float> >& new_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,tipl::rgb color);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,unsigned int length_threshold);
        void filter_by_roi(std::shared_ptr<RoiMgr> roi_mgr);
        void reconnect_track(float distance,float angular_threshold);
        void cull(float select_angle,
                  const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  bool delete_track);
        void cut(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos);
        void cut_by_slice(unsigned int dim, unsigned int pos,bool greater);
        void paint(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  unsigned int color);
        void set_color(unsigned int color){std::fill(tract_color.begin(),tract_color.end(),color);}
        void set_tract_color(unsigned int index,unsigned int color){tract_color[index] = color;}
        void set_tract_color(std::vector<unsigned int>& new_color){tract_color = new_color;}
        void cut_by_mask(const char* file_name);
        void clear_deleted(void);
        void undo(void);
        void redo(void);
        bool trim(void);
        void get_tract_points(std::vector<tipl::vector<3,float> >& points);
        void to_voxel(std::vector<tipl::vector<3,short> >& points,float ratio,int id = -1);
        void to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                std::vector<tipl::vector<3,short> >& points2,float ratio);

        size_t get_deleted_track_count(void) const{return deleted_tract_data.size();}
        size_t get_visible_track_count(void) const{return tract_data.size();}
        
        const std::vector<float>& get_tract(unsigned int index) const{return tract_data[index];}
        const std::vector<std::vector<float> >& get_tracts(void) const{return tract_data;}
        const std::vector<std::vector<float> >& get_deleted_tracts(void) const{return deleted_tract_data;}
        std::vector<std::vector<float> >& get_tracts(void) {return tract_data;}
        unsigned int get_tract_color(unsigned int index) const{return tract_color[index];}
        size_t get_tract_length(unsigned int index) const{return tract_data[index].size();}
        void get_density_map(tipl::image<unsigned int,3>& mapping,
             const tipl::matrix<4,4,float>& transformation,bool endpoint);
        void get_density_map(tipl::image<tipl::rgb,3>& mapping,
             const tipl::matrix<4,4,float>& transformation,bool endpoint);
        void save_tdi(const char* file_name,bool sub_voxel,bool endpoint,const tipl::matrix<4,4,float>& tran);

        void get_quantitative_info(std::string& result);
        bool recognize(std::vector<unsigned int>& result);
        bool recognize(std::map<float,std::string,std::greater<float> >& result,bool contain = false);
        void recognize_report(std::string& report);
        void get_report(unsigned int profile_dir,float band_width,const std::string& index_name,
                        std::vector<float>& values,
                        std::vector<float>& data_profile);

public:
        void get_tract_data(unsigned int fiber_index,
                            unsigned int index_num,
                            std::vector<float>& data) const;
        bool get_tracts_data(
                const std::string& index_name,
                std::vector<std::vector<float> >& data) const;
        void get_tracts_data(unsigned int index_num,float& mean) const;
public:

        void get_passing_list(const std::vector<std::vector<short> >& region_map,
                              unsigned int region_count,
                                     std::vector<std::vector<short> >& passing_list1,
                                     std::vector<std::vector<short> >& passing_list2) const;
        void get_end_list(const std::vector<std::vector<short> >& region_map,
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
    std::vector<std::vector<short> > region_map;
    size_t region_count;
    std::vector<std::string> region_name;
    std::string error_msg,atlas_name;
    float overlap_ratio;
    void set_atlas(std::shared_ptr<atlas> data,const tipl::image<tipl::vector<3,float>,3 >& mni_position);
    void set_regions(const tipl::geometry<3>& geo,
                     const std::vector<std::shared_ptr<ROIRegion> >& regions);
public:
    void save_to_image(tipl::color_image& cm);
    void save_to_file(const char* file_name);
    void save_to_connectogram(const char* file_name);
    void save_to_text(std::string& text);
    bool calculate(TractModel& tract_model,std::string matrix_value_type,bool use_end_only,float threshold);
    void network_property(std::string& report);
};




#endif//TRACT_MODEL_HPP
