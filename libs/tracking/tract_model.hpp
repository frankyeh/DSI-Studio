#ifndef TRACT_MODEL_HPP
#define TRACT_MODEL_HPP
#include <vector>
#include <iosfwd>
#include "fib_data.hpp"

class RoiMgr;
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
class TractModel{
public:
        std::string report,name,parameter_id;
        bool saved = true;
public:
        tipl::shape<3> geo;
        tipl::vector<3> vs;
        tipl::matrix<4,4> trans_to_mni;
        bool is_mni = false;
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
public:
        // for loading multiple clusters
        // it can be empty
        std::vector<unsigned int> tract_cluster;
        float loaded_value = 0.0f;
        std::vector<float> loaded_values;
public:
        static bool save_all(const char* file_name,
                             const std::vector<std::shared_ptr<TractModel> >& all);
        void select(float select_angle,
                    const std::vector<tipl::vector<3,float> >& dirs,
                    const tipl::vector<3,float>& from_pos,std::vector<unsigned int>& selected);
        // selection
        bool delete_tracts(const std::vector<unsigned int>& tracts_to_delete);
        bool select_tracts(const std::vector<unsigned int>& tracts_to_select);
        bool delete_repeated(float d);
        bool delete_branch(void);
        bool delete_by_length(float length);
public:
        static auto separate_tracts(std::shared_ptr<TractModel> tract_model,
                        const std::vector<unsigned int>& labels,
                        const std::vector<std::string>& name)
        {
            std::vector<std::shared_ptr<TractModel> > all;
            if(tract_model->tract_cluster.empty())
                return all;
            std::vector<std::vector<float> > all_tract;
            std::vector<unsigned int> all_tract_color(std::move(tract_model->tract_color));
            tract_model->release_tracts(all_tract);
            all.resize(tipl::max_value(labels) + 1);
            tipl::adaptive_par_for(all.size(),[&](size_t cluster_index)
            {
                auto fiber_num = std::count(labels.begin(),labels.end(),cluster_index);
                if(!fiber_num)
                    return;
                std::vector<std::vector<float> > tract(fiber_num);
                std::vector<unsigned int> tract_color(fiber_num);
                for(unsigned int index = 0,i = 0;index < labels.size();++index)
                    if(labels[index] == cluster_index)
                    {
                        tract[i].swap(all_tract[index]);
                        tract_color[i] = all_tract_color[index];
                        ++i;
                    }
                all[cluster_index] = std::make_shared<TractModel>(*tract_model.get());
                all[cluster_index]->add_tracts(tract);
                all[cluster_index]->tract_color.swap(tract_color);
                all[cluster_index]->name = (cluster_index < name.size() && !name[cluster_index].empty() ? name[cluster_index] :
                                            std::string("cluster") + std::to_string(cluster_index));
            });
            return all;
        }
        static auto load_from_file(const char* file_name,std::shared_ptr<fib_data> handle,bool tract_is_mni = false)
        {
            tipl::progress prog("open ",file_name);
            std::vector<std::shared_ptr<TractModel> > all_tracts;
            auto tract_model = std::make_shared<TractModel>(handle);
            if(!tract_model->load_tracts_from_file(file_name,handle.get(),tract_is_mni))
                return all_tracts;
            if(tract_model->tract_cluster.empty())
            {
                tract_model->name = std::filesystem::path(std::string(file_name)).stem().stem().u8string();
                all_tracts.push_back(tract_model);
                return all_tracts;
            }
            std::ifstream in(std::string(file_name)+".txt");
            return separate_tracts(tract_model,tract_model->tract_cluster,
                std::vector<std::string>((std::istream_iterator<std::string>(in)),(std::istream_iterator<std::string>())));
        }

public:
        TractModel(std::shared_ptr<fib_data> handle):geo(handle->dim),vs(handle->vs),trans_to_mni(handle->trans_to_mni),is_mni(handle->is_mni){}
        TractModel(tipl::shape<3> dim_,tipl::vector<3> vs_):geo(dim_),vs(vs_)
        {
            initial_LPS_nifti_srow(trans_to_mni,geo,vs);
        }
        TractModel(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_)
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
            is_mni = rhs.is_mni;
            tract_data = rhs.tract_data;
            tract_color = rhs.tract_color;
            tract_tag = rhs.tract_tag;
            report = rhs.report;
            parameter_id = rhs.parameter_id;
            name = rhs.name;
            saved = true;
            return *this;
        }
        void add(const TractModel& rhs);
        bool load_tracts_from_file(const char* file_name,fib_data* handle,bool tract_is_mni = false);

        bool save_tracts_to_file(const char* file_name);
        bool save_tracts_in_template_space(std::shared_ptr<fib_data> handle,const char* file_name,bool output_mni = false);
        bool save_transformed_tract(const char* file_name,tipl::shape<3> new_dim,
                                             tipl::vector<3> new_vs,const tipl::matrix<4,4>& trans_to_mni,const tipl::matrix<4,4>& T,bool end_point);

        std::string get_obj(unsigned int& coordinate_count,
                       unsigned char tract_style,
                       float tube_diameter,
                       unsigned char tract_tube_detail);
        bool save_data_to_file(std::shared_ptr<fib_data> handle,const char* file_name,const std::string& index_name);
        bool save_end_points(const char* file_name) const;

        bool load_tracts_color_from_file(const char* file_name);
        bool save_tracts_color_to_file(const char* file_name);


        void release_tracts(std::vector<std::vector<float> >& released_tracks);
        void clear(void);
        void add_tracts(std::vector<std::vector<float> >& new_tracks);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,tipl::rgb color);
        void add_tracts(std::vector<std::vector<float> >& new_tracks,unsigned int length_threshold,tipl::rgb color);
        bool filter_by_roi(std::shared_ptr<RoiMgr> roi_mgr);
        bool reconnect_tract(float distance,float angular_threshold);
        bool cull(float select_angle,
                  const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  bool delete_track);
        bool cut(const std::vector<unsigned int>& tract_to_delete,
                 const std::vector<std::vector<float> >& new_tract,
                 const std::vector<unsigned int>& new_tract_color);
        bool cut(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos);
        bool cut_end_portion(float from,float to);
        bool cut_by_slice(unsigned int dim, unsigned int pos,bool greater,const tipl::matrix<4,4>* T = nullptr);
        void cut_by_mask(const char* file_name);
        bool paint(float select_angle,const std::vector<tipl::vector<3,float> > & dirs,
                  const tipl::vector<3,float>& from_pos,
                  unsigned int color);
        void set_color(unsigned int color){std::fill(tract_color.begin(),tract_color.end(),color);}
        void set_tract_color(std::vector<unsigned int>& new_color){tract_color = new_color;}
        void clear_deleted(void);
        bool undo(void);
        bool redo(void);
        bool trim(void);
        void trim(unsigned int iterations);
        void region_pruning(float region_ratio);
        void flip(char dim);

        void resample(float new_step);
        void get_tract_points(std::vector<tipl::vector<3,float> >& points);
        void get_in_slice_tracts(unsigned char dim,int pos,
                                 tipl::matrix<4,4>* T,
                                 std::vector<std::vector<tipl::vector<2,float> > >& lines,
                                 std::vector<std::vector<unsigned int> >& colors,
                                 unsigned int max_count,
                                 int track_color_style,
                                 bool& terminated);
        void to_voxel(std::vector<tipl::vector<3,short> >& points,const tipl::matrix<4,4>& trans = tipl::identity_matrix(),int id = -1);
        void to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                std::vector<tipl::vector<3,short> >& points2,const tipl::matrix<4,4>& trans = tipl::identity_matrix());
        void to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                std::vector<tipl::vector<3,short> >& points2,const tipl::matrix<4,4>& trans,float end_dis);

        size_t get_deleted_track_count(void) const{return deleted_tract_data.size();}
        size_t get_visible_track_count(void) const{return tract_data.size();}
        
        auto get_tract_point(unsigned int index,unsigned int pos) const{return tipl::vector<3>(&tract_data[index][pos + (pos << 1)]);}
        const std::vector<float>& get_tract(unsigned int index) const{return tract_data[index];}
        const std::vector<std::vector<float> >& get_tracts(void) const{return tract_data;}
        std::vector<std::vector<float> >& get_deleted_tracts(void) {return deleted_tract_data;}
        std::vector<std::vector<float> >& get_tracts(void) {return tract_data;}
        unsigned int get_tract_color(unsigned int index) const{return tract_color[index];}
        float get_tract_length_in_mm(unsigned int index) const;
public:
        void get_density_map(tipl::image<3,unsigned int>& mapping,
             const tipl::matrix<4,4>& to_t1t2,bool endpoint);
        void get_density_map(tipl::image<3,tipl::rgb>& mapping,
             const tipl::matrix<4,4>& to_t1t2,bool endpoint);
        static bool export_tdi(const char* file_name,
                          std::vector<std::shared_ptr<TractModel> > tract_models,
                          tipl::shape<3> dim,
                          tipl::vector<3,float> vs,
                          const tipl::matrix<4,4>& trans_to_mni,
                          const tipl::matrix<4,4>& to_t1t2,
                               bool color,bool end_point);
        static bool export_pdi(const char* file_name,
                               const std::vector<std::shared_ptr<TractModel> >& tract_models);
public:
        void get_quantitative_info(std::shared_ptr<fib_data> handle,std::string& result);
        void get_quantitative_info(std::shared_ptr<fib_data> handle,std::vector<std::string>& title,std::vector<float>& results);
        tipl::vector<3> get_report(std::shared_ptr<fib_data> handle,
                        unsigned int profile_dir,float band_width,const std::string& index_name,
                        std::vector<float>& values,
                        std::vector<float>& data_profile,
                        std::vector<float>& data_ci1,
                        std::vector<float>& data_ci2);

public:
        std::vector<float> get_tract_data(std::shared_ptr<fib_data> handle,size_t fiber_index,size_t index_num) const;
        std::vector<std::vector<float> > get_tracts_data(std::shared_ptr<fib_data> handle,const std::string& index_name) const;
        float get_tracts_mean(std::shared_ptr<fib_data> handle,unsigned int index_num) const;
public:

        void get_passing_list(const tipl::image<3,std::vector<short> >& region_map,
                              unsigned int region_count,
                                     std::vector<std::vector<short> >& passing_list1,
                                     std::vector<std::vector<short> >& passing_list2) const;
        void get_end_list(const tipl::image<3,std::vector<short> >& region_map,
                                     std::vector<std::vector<short> >& end_list1,
                                     std::vector<std::vector<short> >& end_list2) const;
        void run_clustering(unsigned char method_id,unsigned int cluster_count,float param);

};




class atlas;
class ROIRegion;
struct Parcellation{
public:
    std::shared_ptr<fib_data> handle;
    std::vector<std::vector<tipl::vector<3,short> > > points;
    std::vector<std::string> labels;
    std::string name;
public:
    mutable std::string error_msg;
public:
    Parcellation(std::shared_ptr<fib_data> handle_):handle(handle_){}
    bool load_from_atlas(std::string atlas_name);
    void load_from_regions(const std::vector<std::shared_ptr<ROIRegion> >& regions);
    std::vector<float> get_t2r_values(std::shared_ptr<TractModel> tract) const;
    std::string get_t2r(const std::vector<std::shared_ptr<TractModel> >& tracts) const;
    bool save_t2r(const std::string& filename,const std::vector<std::shared_ptr<TractModel> >& tracts) const;
};

class ConnectivityMatrix{
public:
    std::vector<std::string> metrics;
    std::vector<std::vector<float> > metrics_data;
public:
    tipl::image<2> matrix_value;
public:
    tipl::image<3,std::vector<short> > region_map;
    size_t region_count = 0;
    std::vector<std::string> region_name;
    std::string error_msg,atlas_name;
    void set_parcellation(const Parcellation& p);

public:
    void save_to_image(tipl::color_image& cm);
    void save_to_file(const char* file_name);
    void save_to_connectogram(const char* file_name);
    void save_to_text(std::string& text);
    bool calculate(std::shared_ptr<fib_data> handle,TractModel& tract_model,std::string matrix_value_type,bool use_end_only,float threshold);
    void network_property(std::string& report);
};




#endif//TRACT_MODEL_HPP
