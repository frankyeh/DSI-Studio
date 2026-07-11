// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include "fib_data.hpp"
#include "opengl/region_render.hpp"
// ---------------------------------------------------------------------------
class SliceModel;
// region_feature
const unsigned char roi_id = 0;
const unsigned char roa_id = 1;
const unsigned char end_id = 2;
const unsigned char seed_id = 3;
const unsigned char term_id = 4;
const unsigned char not_end_id = 5;
const unsigned char limiting_id = 6;
const unsigned char default_id = 7;
class ROIRegion {
public:
        std::string name = "region";
        tipl::shape<3> dim;
        tipl::vector<3> vs;
        tipl::matrix<4,4> trans_to_mni;
        bool is_mni = false;
        template<typename T>
        auto bind(const T& I) const
        {
            return std::tie(vs,trans_to_mni,is_mni,I);
        }
public:
        std::vector<tipl::vector<3,short> > region;
        std::vector<std::vector<tipl::vector<3,short> > > undo_backup;
        std::vector<std::vector<tipl::vector<3,short> > > redo_backup;
public:
        ~ROIRegion();
        bool makeMeshes(unsigned char smooth);
        std::thread mesh_thread;
        std::mutex mesh_lock;
        std::shared_ptr<RegionRender> pending_region_render;
        std::atomic_bool mesh_running = false,mesh_terminated = false;
private:
        tipl::image<3,unsigned int> index_mask; // lazy cache, empty until first add_points()
        size_t index_mask_point_count = 0;    // bookkeeping points count in the mask to avoid mask-region mismatch
        void point2mask(void);
    public:
        bool modified = true;
        void set_modified(void){modified = true;index_mask.clear();index_mask_point_count = 0;}
public:
        bool is_diffusion_space = true;
        tipl::matrix<4,4> to_diffusion_space = tipl::identity_matrix();
public: // rendering options
        std::shared_ptr<RegionRender> region_render;
        unsigned char regions_feature = default_id;

public: // rendering options
        ROIRegion(std::shared_ptr<fib_data> handle):
            dim(handle->dim),vs(handle->vs),trans_to_mni(handle->trans_to_mni),is_mni(handle->is_mni),region_render(new RegionRender){}
        ROIRegion(tipl::shape<3> dim_,tipl::vector<3> vs_):dim(dim_),vs(vs_),region_render(new RegionRender)
        {
            tipl::io::initial_nifti_srow(trans_to_mni,dim,vs);
        }
        ROIRegion(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_)
            :dim(dim_),vs(vs_),trans_to_mni(trans_to_mni_),region_render(new RegionRender){}

        ROIRegion(const ROIRegion& rhs):region_render(new RegionRender)
        {
            (*this) = rhs;
        }

        void copy_space(const ROIRegion & rhs){
            dim = rhs.dim;
            vs = rhs.vs;
            trans_to_mni = rhs.trans_to_mni;
            is_mni = rhs.is_mni;
            is_diffusion_space = rhs.is_diffusion_space;
            to_diffusion_space = rhs.to_diffusion_space;

            region.clear();
            undo_backup.clear();
            redo_backup.clear();

        }
        const ROIRegion& operator = (const ROIRegion & rhs) {
            copy_space(rhs);

            region = rhs.region;
            undo_backup = rhs.undo_backup;
            redo_backup = rhs.redo_backup;

            regions_feature = rhs.regions_feature;
            region_render->color = rhs.region_render->color;
            set_modified();
            return *this;
        }
        void swap(ROIRegion & rhs) {

            dim.swap(rhs.dim);
            std::swap(vs,rhs.vs);
            trans_to_mni.swap(rhs.trans_to_mni);
            region.swap(rhs.region);
            index_mask.swap(rhs.index_mask);
            std::swap(index_mask_point_count,rhs.index_mask_point_count);
            undo_backup.swap(rhs.undo_backup);
            redo_backup.swap(rhs.redo_backup);
            std::swap(regions_feature,rhs.regions_feature);
            region_render.swap(rhs.region_render);
            std::swap(modified,rhs.modified);
            std::swap(is_diffusion_space,rhs.is_diffusion_space);
            std::swap(to_diffusion_space,rhs.to_diffusion_space);
        }

        tipl::vector<3> get_center(void) const
        {
            tipl::vector<3> c;
            if(region.size())
            {
                c = region.front();
                c += region.back();
                c *= 0.5;
                if(!is_diffusion_space)
                    c.to(to_diffusion_space);
            }
            return c;
        }
public:
        void new_from_sphere(tipl::vector<3> mni,float radius);
        std::vector<tipl::vector<3,short> > to_space(const tipl::shape<3>& dim_to,
                                                     const tipl::matrix<4,4>& trans_to) const;
        std::vector<tipl::vector<3,short> > to_space(const tipl::shape<3>& dim_to) const;
        void add_points(std::vector<tipl::vector<3,float> >&& points,bool del = false);
        void add_points(std::vector<tipl::vector<3,short> >&& points,bool del = false);
        void undo(void)
        {
            if(region.empty() && undo_backup.empty())
                return;
            redo_backup.push_back(std::move(region));
            if(!undo_backup.empty())
            {
                region = std::move(undo_backup.back());
                undo_backup.pop_back();
            }
            set_modified();

        }
        bool redo(void)
        {
            if(redo_backup.empty())
                return false;
            undo_backup.push_back(std::vector<tipl::vector<3,short> >());
            undo_backup.back().swap(region);
            region.swap(redo_backup.back());
            redo_backup.pop_back();
            set_modified();
            return true;
        }
        bool save_region_to_file(const std::filesystem::path&);
        bool load_region_from_file(const std::filesystem::path&);
        void flip_region(unsigned int dimension);
        bool shift(tipl::vector<3,float> dx);

        void from_mask(const tipl::image<3,unsigned char>& mask);

        void to_mask(tipl::image<3,unsigned char>& mask) const;
        void to_mask(tipl::image<3,unsigned char>& mask,const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to) const;
        auto to_mask(void) const
        {
            tipl::image<3,unsigned char> mask;
            to_mask(mask);
            return mask;
        }
        auto to_mask(const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to) const
        {
            tipl::image<3,unsigned char> mask;
            to_mask(mask,dim_to,trans_to);
            return mask;
        }
        void perform(const std::string& action);

        template<typename value_type>
        bool has_point(tipl::vector<3,value_type> p) const
        {
            if(!is_diffusion_space)
                p.to(tipl::matrix<4,4>(tipl::inverse(to_diffusion_space)));
            tipl::vector<3,short> q(std::round(p[0]),std::round(p[1]),std::round(p[2]));
            if(!dim.is_valid(q))
                return false;
            return !index_mask.empty() && index_mask_point_count == region.size() ?
                       index_mask.at(q) : std::find(region.begin(),region.end(),q) != region.end();
        }


        bool is_same_space(const ROIRegion& rhs) const
        {   return dim == rhs.dim && to_diffusion_space == rhs.to_diffusion_space;}

    public:
        float get_volume(void) const;
        tipl::vector<3> get_pos(void) const;
        void get_quantitative_data(std::shared_ptr<slice_model> slice,float& mean,float& max_v,float& min_v);
        void get_quantitative_data(std::shared_ptr<fib_data> handle,std::vector<std::string>& titles,std::vector<float>& data);
};

#endif
