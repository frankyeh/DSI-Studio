// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>
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
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
class ROIRegion {
public:
        std::string name = "region";
        tipl::shape<3> dim;
        tipl::vector<3> vs;
        tipl::matrix<4,4> trans_to_mni;
        bool is_mni = false;
public:
        std::vector<tipl::vector<3,short> > region;
        std::vector<std::vector<tipl::vector<3,short> > > undo_backup;
        std::vector<std::vector<tipl::vector<3,short> > > redo_backup;
public:
        bool is_diffusion_space = true;
        tipl::matrix<4,4> to_diffusion_space = tipl::identity_matrix();
public: // rendering options
        std::shared_ptr<RegionRender> region_render;
        unsigned char regions_feature = default_id;
        bool modified = true;
public: // rendering options
        ROIRegion(std::shared_ptr<fib_data> handle):
            dim(handle->dim),vs(handle->vs),trans_to_mni(handle->trans_to_mni),is_mni(handle->is_mni),region_render(new RegionRender){}
        ROIRegion(tipl::shape<3> dim_,tipl::vector<3> vs_):dim(dim_),vs(vs_),region_render(new RegionRender)
        {
            initial_LPS_nifti_srow(trans_to_mni,dim,vs);
        }
        ROIRegion(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_)
            :dim(dim_),vs(vs_),trans_to_mni(trans_to_mni_),region_render(new RegionRender){}

        ROIRegion(const ROIRegion& rhs):region_render(new RegionRender)
        {
            (*this) = rhs;
        }

        const ROIRegion& operator = (const ROIRegion & rhs) {

            dim = rhs.dim;
            vs = rhs.vs;
            trans_to_mni = rhs.trans_to_mni;
            is_mni = rhs.is_mni;

            region = rhs.region;
            undo_backup = rhs.undo_backup;
            redo_backup = rhs.redo_backup;
            regions_feature = rhs.regions_feature;
            region_render->color = rhs.region_render->color;
            modified = true;
            is_diffusion_space = rhs.is_diffusion_space;
            to_diffusion_space = rhs.to_diffusion_space;
            return *this;
        }
        void swap(ROIRegion & rhs) {

            dim.swap(rhs.dim);
            std::swap(vs,rhs.vs);
            trans_to_mni.swap(rhs.trans_to_mni);
            region.swap(rhs.region);
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
        void new_from_mni_sphere(std::shared_ptr<fib_data> handle,tipl::vector<3> mni,float radius);
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
            modified = true;

        }
        bool redo(void)
        {
            if(redo_backup.empty())
                return false;
            undo_backup.push_back(std::vector<tipl::vector<3,short> >());
            undo_backup.back().swap(region);
            region.swap(redo_backup.back());
            redo_backup.pop_back();
            modified = true;
            return true;
        }
        bool save_region_to_file(const char* FileName);
        bool load_region_from_file(const char* FileName);
        void flip_region(unsigned int dimension);
        bool shift(tipl::vector<3,float> dx);

        void load_region_from_buffer(tipl::image<3,unsigned char>& mask);
        void save_region_to_buffer(tipl::image<3,unsigned char>& mask) const;
        void save_region_to_buffer(tipl::image<3,unsigned char>& mask,const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to) const;
        void perform(const std::string& action);
        void makeMeshes(unsigned char smooth);
        template<typename value_type>
        bool has_point(tipl::vector<3,value_type> point_in_dwi_space) const
        {
            if(!is_diffusion_space)
                point_in_dwi_space.to(tipl::matrix<4,4>(tipl::inverse(to_diffusion_space)));
            return std::find(region.begin(),region.end(),
                             tipl::vector<3,short>(std::round(point_in_dwi_space[0]),
                                                   std::round(point_in_dwi_space[1]),
                                                   std::round(point_in_dwi_space[2]))) != region.end();
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
