#ifndef TRACT_MODEL_HPP
#define TRACT_MODEL_HPP
#include <vector>
#include <iosfwd>
#include "image/image.hpp"
#include "fib_data.hpp"

class RoiMgr;
class TractModel{
public:
        std::string report;
private:
        std::shared_ptr<fib_data> handle;
        image::geometry<3> geometry;
        image::vector<3> vs;
        std::auto_ptr<tracking> fib;
private:
        std::vector<std::vector<float> > tract_data;
        std::vector<std::vector<float> > deleted_tract_data;
        std::vector<unsigned int> tract_color;
        std::vector<unsigned int> deleted_tract_color;
        std::vector<unsigned int> deleted_count;
        std::vector<std::pair<unsigned int,unsigned int> > deleted_cut_count;
        std::vector<std::pair<unsigned int,unsigned int> > redo_size;
        // offset, size
private:
        // for loading multiple clusters
        std::vector<unsigned int> tract_cluster;
public:
        static bool save_all(const char* file_name,const std::vector<TractModel*>& all);
        const std::vector<unsigned int>& get_cluster_info(void) const{return tract_cluster;}
        std::vector<unsigned int>& get_cluster_info(void) {return tract_cluster;}
        void select(float select_angle,
                    const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                    const image::vector<3,float>& from_pos,std::vector<unsigned int>& selected);
        // selection
        void delete_tracts(const std::vector<unsigned int>& tracts_to_delete);
        void select_tracts(const std::vector<unsigned int>& tracts_to_select);
        void delete_repeated(void);

public:
        TractModel(std::shared_ptr<fib_data> handle_);
        const TractModel& operator=(const TractModel& rhs)
        {
            geometry = rhs.geometry;
            vs = rhs.vs;
            handle = rhs.handle;
            tract_data = rhs.tract_data;
            tract_color = rhs.tract_color;
            report = rhs.report;
            return *this;
        }
        const tracking& get_fib(void) const{return *fib.get();}
        tracking& get_fib(void){return *fib.get();}
        void add(const TractModel& rhs);
        bool load_from_file(const char* file_name,bool append = false);

        bool save_tracts_to_file(const char* file_name);
        void save_vrml(const char* file_name,
                       unsigned char tract_style,
                       unsigned char tract_color_style,
                       float tube_diameter,
                       unsigned char tract_tube_detail);
        bool save_transformed_tracts_to_file(const char* file_name,const float* transform,bool end_point);
        bool save_data_to_file(const char* file_name,const std::string& index_name);
        void save_end_points(const char* file_name) const;

        bool load_tracts_color_from_file(const char* file_name);
        bool save_tracts_color_to_file(const char* file_name);


        void release_tracts(std::vector<std::vector<float> >& released_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,image::rgb_color color);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,unsigned int length_threshold);
        void filter_by_roi(RoiMgr& roi_mgr);
        void cull(float select_angle,
                  const image::vector<3,float>& from_dir,
                  const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,
                  bool delete_track);
        void cut(float select_angle,const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos);
        void cut_by_slice(unsigned int dim, unsigned int pos,bool greater);
        void paint(float select_angle,const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,
                  unsigned int color);
        void set_color(unsigned int color){std::fill(tract_color.begin(),tract_color.end(),color);}
        void set_tract_color(unsigned int index,unsigned int color){tract_color[index] = color;}
        void cut_by_mask(const char* file_name);
        void clear_deleted(void);
        void undo(void);
        void redo(void);
        bool trim(void);


        void get_end_points(std::vector<image::vector<3,float> >& points);
        void get_end_points(std::vector<image::vector<3,short> >& points);
        void get_tract_points(std::vector<image::vector<3,short> >& points);

        size_t get_deleted_track_count(void) const{return deleted_tract_data.size();}
        size_t get_visible_track_count(void) const{return tract_data.size();}
        
        const std::vector<float>& get_tract(unsigned int index) const{return tract_data[index];}
        const std::vector<std::vector<float> >& get_tracts(void) const{return tract_data;}
        const std::vector<std::vector<float> >& get_deleted_tracts(void) const{return deleted_tract_data;}
        std::vector<std::vector<float> >& get_tracts(void) {return tract_data;}
        unsigned int get_tract_color(unsigned int index) const{return tract_color[index];}
        size_t get_tract_length(unsigned int index) const{return tract_data[index].size();}
        void get_density_map(image::basic_image<unsigned int,3>& mapping,
             const image::matrix<4,4,float>& transformation,bool endpoint);
        void get_density_map(image::basic_image<image::rgb_color,3>& mapping,
             const image::matrix<4,4,float>& transformation,bool endpoint);
        void save_tdi(const char* file_name,bool sub_voxel,bool endpoint,const std::vector<float>& tran);

        void get_quantitative_data(std::vector<float>& data);
        void get_quantitative_info(std::string& result);
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
public:

        void get_passing_list(const std::vector<std::vector<image::vector<3,short> > >& regions,
                                     std::vector<std::vector<unsigned int> >& passing_list,
                                     bool use_end_only) const;

};




class atlas;
class ConnectivityMatrix{
public:
    std::vector<std::vector<unsigned int> > passing_list;
    image::basic_image<float,2> matrix_value;
public:
    std::vector<std::vector<image::vector<3,short> > > regions;
    std::vector<std::string> region_name;
    std::string error_msg;
    void set_atlas(atlas& data,const image::basic_image<image::vector<3,float>,3 >& mni_position);
public:
    void save_to_image(image::color_image& cm);
    void save_to_file(const char* file_name);
    bool calculate(TractModel& tract_model,std::string matrix_value_type,bool use_end_only);
    void network_property(std::string& report);
};




#endif//TRACT_MODEL_HPP
