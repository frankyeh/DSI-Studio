#ifndef TRACT_MODEL_HPP
#define TRACT_MODEL_HPP
#include <vector>
#include <iosfwd>
#include "image/image.hpp"

class ODFModel;
class TractModel{


private:
        image::geometry<3> geometry;
        image::vector<3> vs;
        ODFModel* handle;
private:
        std::vector<std::vector<float> > tract_data;
        std::vector<std::vector<float> > deleted_tract_data;
        std::vector<unsigned int> tract_color;
        std::vector<unsigned int> deleted_tract_color;
        std::vector<unsigned int> deleted_count;
        std::vector<std::pair<unsigned int,unsigned int> > redo_size;
        // offset, size
private:
        // for loading multiple clusters
        std::vector<unsigned int> tract_cluster;
public:
        static bool save_all(const char* file_name,const std::vector<TractModel*>& all);
        const std::vector<unsigned int>& get_cluster_info(void) const{return tract_cluster;}
        std::vector<unsigned int>& get_cluster_info(void) {return tract_cluster;}
private:
        void select(const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,const image::vector<3,float>& to_pos,
                  std::vector<unsigned int>& selected);
        // selection
        void delete_tracts(const std::vector<unsigned int>& tracts_to_delete);
        void select_tracts(const std::vector<unsigned int>& tracts_to_select);
public:
        TractModel(ODFModel* handle_,
                   const image::geometry<3>& geo,
                   const image::vector<3>& vs_):geometry(geo),vs(vs_),handle(handle_){}
        const TractModel& operator=(const TractModel& rhs)
        {
            geometry = rhs.geometry;
            vs = rhs.vs;
            handle = rhs.handle;
            tract_data = rhs.tract_data;
            return *this;
        }
        void add(const TractModel& rhs);
        bool load_from_file(const char* file_name,bool append = false);

        bool save_fa_to_file(const char* file_name,float threshold,float cull_angle_cos);
        bool save_tracts_to_file(const char* file_name);
        bool save_transformed_tracts_to_file(const char* file_name,const float* transform,bool end_point);
        bool save_data_to_file(const char* file_name,const std::string& index_name);
        void save_end_points(const char* file_name) const;

        bool load_tracts_color_from_file(const char* file_name);
        bool save_tracts_color_to_file(const char* file_name);


        void release_tracts(std::vector<std::vector<float> >& released_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks);
        void cull(const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,const image::vector<3,float>& to_pos,
                  bool delete_track);
        void cut(const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,const image::vector<3,float>& to_pos);
        void paint(const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                  const image::vector<3,float>& from_pos,const image::vector<3,float>& to_pos,
                  unsigned int color);
        void set_color(unsigned int color){std::fill(tract_color.begin(),tract_color.end(),color);}
        void cut_by_mask(const char* file_name);
        void undo(void);
        void redo(void);
        void trim(void);


        void get_end_points(std::vector<image::vector<3,short> >& points);
        void get_tract_points(std::vector<image::vector<3,short> >& points);

        unsigned int get_deleted_track_count(void) const{return deleted_tract_data.size();}
        unsigned int get_visible_track_count(void) const{return tract_data.size();}
        
        const std::vector<float>& get_tract(unsigned int index) const{return tract_data[index];}
        const std::vector<std::vector<float> >& get_tracts(void) const{return tract_data;}
        unsigned int get_tract_color(unsigned int index) const{return tract_color[index];}
        unsigned int get_tract_length(unsigned int index) const{return tract_data[index].size();}
        void get_density_map(image::basic_image<unsigned int,3>& mapping,
             const std::vector<float>& transformation,bool endpoint);
        void get_density_map(image::basic_image<image::rgb_color,3>& mapping,
             const std::vector<float>& transformation,bool endpoint);

};




#endif//TRACT_MODEL_HPP
