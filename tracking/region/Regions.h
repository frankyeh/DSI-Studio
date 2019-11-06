// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>

#include "tipl/tipl.hpp"
#include "RegionModel.h"
#include "fib_data.hpp"
// ---------------------------------------------------------------------------
class SliceModel;
// region_feature
const unsigned char roi_id = 0;
const unsigned char roa_id = 1;
const unsigned char end_id = 2;
const unsigned char seed_id = 3;
const unsigned char terminate_id = 4;
const unsigned char not_ending_id = 5;

class ROIRegion {
public:
        std::shared_ptr<fib_data> handle;
        std::vector<tipl::vector<3,short> > region;
        bool modified;
        std::vector<std::vector<tipl::vector<3,short> > > undo_backup;
        std::vector<std::vector<tipl::vector<3,short> > > redo_backup;
public:
        bool super_resolution = false;
        float resolution_ratio = 1.0;
        float opacity = -1.0f;
public: // rendering options
        RegionModel show_region;
        unsigned char regions_feature;

        ROIRegion(const ROIRegion& rhs,float resolution_ratio_ = 1.0) :
            handle(rhs.handle),
            region(rhs.region),super_resolution(resolution_ratio_ != 1.0),
            resolution_ratio(resolution_ratio_),
            regions_feature(rhs.regions_feature), modified(true)
        {
            show_region = rhs.show_region;
        }
        const ROIRegion& operator = (const ROIRegion & rhs) {
            handle = rhs.handle;
            region = rhs.region;
            undo_backup = rhs.undo_backup;
            redo_backup = rhs.redo_backup;
            regions_feature = rhs.regions_feature;
            show_region = rhs.show_region;
            modified = true;
            super_resolution = rhs.super_resolution;
            resolution_ratio = rhs.resolution_ratio;
            return *this;
        }
        void swap(ROIRegion & rhs) {
            handle.swap(rhs.handle);
            region.swap(rhs.region);
            undo_backup.swap(rhs.undo_backup);
            redo_backup.swap(rhs.redo_backup);
            std::swap(regions_feature,rhs.regions_feature);
            show_region.swap(rhs.show_region);
            std::swap(modified,rhs.modified);
            std::swap(super_resolution,rhs.super_resolution);
            std::swap(resolution_ratio,rhs.resolution_ratio);
        }

        ROIRegion(std::shared_ptr<fib_data> handle_)
            : handle(handle_),modified(false){}
        tipl::geometry<3> get_buffer_dim(void) const;
        tipl::vector<3,short> get_region_voxel(unsigned int index) const
        {
            tipl::vector<3,short> result = region[index];
            if(resolution_ratio == 1.0)
                return result;
            result[0] = (float)result[0]/resolution_ratio;
            result[1] = (float)result[1]/resolution_ratio;
            result[2] = (float)result[2]/resolution_ratio;
            return result;
        }
        tipl::vector<3,float> get_center(void) const
        {
            tipl::vector<3,float> c;
            if(region.size())
            {
                c = region.front();
                c += region.back();
                c *= 0.5;
                c /= resolution_ratio;
            }
            return c;
        }
        void get_region_voxels(std::vector<tipl::vector<3,short> >& output) const
        {
            output = region;
            if(resolution_ratio == 1.0)
                return;
            for(int i = 0;i < region.size();++i)
            {
                output[i][0] = (float)region[i][0]/resolution_ratio;
                output[i][1] = (float)region[i][1]/resolution_ratio;
                output[i][2] = (float)region[i][2]/resolution_ratio;
            }
        }
        const std::vector<tipl::vector<3,short> >& get_region_voxels_raw(void) const {return region;}
        void assign(const std::vector<tipl::vector<3,short> >& region_,float r)
        {
            if(!region.empty())
                undo_backup.push_back(region);
            region = region_;
            resolution_ratio = r;
            modified = true;
        }

        bool empty(void) const {return region.empty();}

        void clear(void)
        {
            modified = true;
            region.clear();
        }

        void erase(unsigned int index)
        {
            modified = true;
            region.erase(region.begin()+index);
        }

        unsigned int size(void) const {return (unsigned int)region.size();}
        std::vector<tipl::vector<3,short> >::const_iterator
                begin(void) const {return region.begin();}

public:
        void add(const ROIRegion & rhs)
        {
            std::vector<tipl::vector<3,short> > tmp(rhs.region);
            add_points(tmp,false,rhs.resolution_ratio);
        }
        template<typename value_type>
        void change_resolution(std::vector<tipl::vector<3,value_type> >& points,float point_resolution)
        {
            if(point_resolution == resolution_ratio)
                return;
            float ratio = resolution_ratio/point_resolution;
            if(resolution_ratio > point_resolution)
            {
                short limit = std::ceil(ratio);
                std::vector<tipl::vector<3,short> > new_points;
                for(short dz = -limit;dz <= limit;++dz)
                    for(short dy = -limit;dy <= limit;++dy)
                        for(short dx = -limit;dx <= limit;++dx)
                            new_points.push_back(tipl::vector<3,short>(dx,dy,dz));


                std::vector<tipl::vector<3,value_type> > pp(points.size()*new_points.size());
                tipl::par_for(points.size(),[&](int i)
                {
                    points[i] *= ratio;
                    points[i].round();
                    unsigned int pos = i*new_points.size();
                    for(int j = 0;j < new_points.size();++j)// 1 for skip 0 0 0
                    {
                        tipl::vector<3,short> p(new_points[j]);
                        p += points[i];
                        pp[pos + j] = p;
                    }
                });
                pp.swap(points);
            }
            else
                tipl::multiply_constant(points,ratio);
        }
        void add_points(std::vector<tipl::vector<3,float> >& points,bool del,float point_resolution = 1.0);
        void add_points(std::vector<tipl::vector<3,short> >& points,bool del,float point_resolution = 1.0);
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
        void SaveToFile(const char* FileName);
        bool LoadFromFile(const char* FileName);
        void Flip(unsigned int dimension);
        void shift(tipl::vector<3,float> dx);

        template<class image_type>
        void LoadFromBuffer(const image_type& from,const tipl::matrix<4,4,float>& trans)
        {
            std::vector<tipl::vector<3,float> > points;
            for (tipl::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
            {
                tipl::vector<3> p(index.begin());
                p.to(trans);
                p += 0.5;
                if (from.geometry().is_valid(p) && from.at(p[0],p[1],p[2]) != 0)
                    points.push_back(tipl::vector<3>(index.begin()));
            }
            region.clear();
            add_points(points,false,1.0f);
        }

        template<class image_type>
        void LoadFromBuffer(const image_type& mask)
        {
            modified = true;
            if(!region.empty())
                undo_backup.push_back(std::move(region));
            region.clear();
            std::vector<tipl::vector<3,short> > points;

            if(mask.width() != handle->dim[0])
                resolution_ratio = (float)mask.width()/(float)handle->dim[0];
            if(resolution_ratio < 1.0f)
            {
                for (tipl::pixel_index<3>index(handle->dim);index < handle->dim.size();++index)
                {
                    tipl::vector<3> pos(index);
                    pos *= resolution_ratio;
                    pos.round();
                    if(mask.geometry().is_valid(pos[0],pos[1],pos[2]) &&
                       mask.at(pos[0],pos[1],pos[2]))
                        points.push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
                }
                resolution_ratio = 1.0f;
            }
            else {
                for (tipl::pixel_index<3>index(mask.geometry());index < mask.size();++index)
                    if (mask[index.index()] != 0)
                        points.push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
            }
            region.swap(points);
        }
        void SaveToBuffer(tipl::image<unsigned char, 3>& mask,float target_resolution);
        void SaveToBuffer(tipl::image<unsigned char, 3>& mask){SaveToBuffer(mask,resolution_ratio);}
        void perform(const std::string& action);
        void makeMeshes(unsigned char smooth);
        template<typename value_type>
        bool has_point(const tipl::vector<3,value_type>& point) const
        {
            if(resolution_ratio != 1.0)
            {
                tipl::vector<3,short> p(std::round(point[0]*resolution_ratio),
                                         std::round(point[1]*resolution_ratio),
                                         std::round(point[2]*resolution_ratio));
                return std::binary_search(region.begin(),region.end(),p);
            }
            tipl::vector<3,short> p(std::round(point[0]),
                                     std::round(point[1]),
                                     std::round(point[2]));
            return std::binary_search(region.begin(),region.end(),p);
        }
        template<typename value_type>
        bool has_points(const std::vector<tipl::vector<3,value_type> >& points) const
        {
            for(unsigned int index = 0; index < points.size(); ++index)
                if(has_point(points[index]))
                    return true;
            return false;
        }
        void get_quantitative_data(std::shared_ptr<fib_data> handle,std::vector<std::string>& titles,std::vector<float>& data);
};

#endif
