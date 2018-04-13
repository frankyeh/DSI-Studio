// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>

#include "image/image.hpp"
#include "RegionModel.h"
// ---------------------------------------------------------------------------
class fib_data;
struct SliceModel;
// region_feature
const unsigned int roi_id = 0;
const unsigned int roa_id = 1;
const unsigned int end_id = 2;
const unsigned int seed_id = 3;
const unsigned int terminate_id = 4;

class ROIRegion {
public:
        std::shared_ptr<fib_data> handle;
        std::vector<image::vector<3,short> > region;
        bool modified;
        std::vector<std::vector<image::vector<3,short> > > undo_backup;
        std::vector<std::vector<image::vector<3,short> > > redo_backup;
public:
        bool super_resolution = false;
        float resolution_ratio = 1.0;
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
        image::geometry<3> get_buffer_dim(void) const;
        image::vector<3,short> get_region_voxel(unsigned int index) const
        {
            image::vector<3,short> result = region[index];
            if(resolution_ratio == 1.0)
                return result;
            result[0] = (float)result[0]/resolution_ratio;
            result[1] = (float)result[1]/resolution_ratio;
            result[2] = (float)result[2]/resolution_ratio;
            return result;
        }
        void get_region_voxels(std::vector<image::vector<3,short> >& output) const
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
        const std::vector<image::vector<3,short> >& get_region_voxels_raw(void) const {return region;}
        void assign(const std::vector<image::vector<3,short> >& region_,float r)
        {
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
        std::vector<image::vector<3,short> >::const_iterator
                begin(void) const {return region.begin();}

public:
        void add(const ROIRegion & rhs)
        {
            std::vector<image::vector<3,short> > tmp(rhs.region);
            add_points(tmp,false,rhs.resolution_ratio);
        }
        template<typename value_type>
        void change_resolution(std::vector<image::vector<3,value_type> >& points,float point_resolution)
        {
            if(point_resolution == resolution_ratio)
                return;
            float ratio = resolution_ratio/point_resolution;
            if(resolution_ratio > point_resolution)
            {
                short limit = std::ceil(ratio);
                std::vector<image::vector<3,short> > new_points;
                for(short dz = -limit;dz <= limit;++dz)
                    for(short dy = -limit;dy <= limit;++dy)
                        for(short dx = -limit;dx <= limit;++dx)
                            new_points.push_back(image::vector<3,short>(dx,dy,dz));


                std::vector<image::vector<3,value_type> > pp(points.size()*new_points.size());
                image::par_for(points.size(),[&](int i)
                {
                    points[i] *= ratio;
                    points[i].round();
                    unsigned int pos = i*new_points.size();
                    for(int j = 0;j < new_points.size();++j)// 1 for skip 0 0 0
                    {
                        image::vector<3,short> p(new_points[j]);
                        p += points[i];
                        pp[pos + j] = p;
                    }
                });
                pp.swap(points);
            }
            else
                image::multiply_constant(points,ratio);
        }
        void add_points(std::vector<image::vector<3,float> >& points,bool del,float point_resolution = 1.0);
        void add_points(std::vector<image::vector<3,short> >& points,bool del,float point_resolution = 1.0);
        void undo(void)
        {
            if(region.empty() && undo_backup.empty())
                return;
            redo_backup.push_back(std::vector<image::vector<3,short> >());
            redo_backup.back().swap(region);
            if(!undo_backup.empty())
            {
                region.swap(undo_backup.back());
                undo_backup.pop_back();
            }
            modified = true;

        }
        void redo(void)
        {
            if(redo_backup.empty())
                return;
            undo_backup.push_back(std::vector<image::vector<3,short> >());
            undo_backup.back().swap(region);
            region.swap(redo_backup.back());
            redo_backup.pop_back();
            modified = true;

        }
        void SaveToFile(const char* FileName);
        bool LoadFromFile(const char* FileName);
        void Flip(unsigned int dimension);
        void shift(image::vector<3,float> dx);

        template<class image_type>
        void LoadFromBuffer(const image_type& from,const image::matrix<4,4,float>& trans)
        {
            std::vector<image::vector<3,float> > points;
            float det = std::fabs(trans.det());
            if(det < 8) // from a low resolution image
            for (image::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
            {
                image::vector<3> p(index.begin());
                p.to(trans);
                if (from.geometry().is_valid(p) && from.at(p[0],p[1],p[2]) != 0)
                    points.push_back(image::vector<3>(index.begin()));
            }
            else// from a high resolution image
            {
                image::matrix<4,4,float> inv(trans);
                if(!inv.inv())
                    return;
                for (image::pixel_index<3> index(from.geometry());
                     index < from.geometry().size();++index)
                if(from[index.index()])
                {
                    image::vector<3> p(index.begin());
                    p.to(inv);
                    if (handle->dim.is_valid(p))
                        points.push_back(image::vector<3,float>(p[0],p[1],p[2]));
                }
            }
            det = std::round(det);
            image::multiply_constant(points,det);
            region.clear();
            add_points(points,false,det);
        }

        template<class image_type>
        void LoadFromBuffer(const image_type& mask)
        {
            modified = true;
            if(!region.empty())
                undo_backup.push_back(region);
            region.clear();
            std::vector<image::vector<3,short> > points;
            for (image::pixel_index<3>index(mask.geometry());index < mask.size();++index)
                if (mask[index.index()] != 0)
                    points.push_back(image::vector<3,short>(index.x(), index.y(),index.z()));
            if(mask.width() != handle->dim[0])
                resolution_ratio = (float)mask.width()/(float)handle->dim[0];
            region.swap(points);
        }
        void SaveToBuffer(image::basic_image<unsigned char, 3>& mask,unsigned char value=255);
        void perform(const std::string& action);
        void makeMeshes(unsigned char smooth);
        template<typename value_type>
        bool has_point(const image::vector<3,value_type>& point) const
        {
            if(resolution_ratio != 1.0)
            {
                image::vector<3,short> p(std::round(point[0]*resolution_ratio),
                                         std::round(point[1]*resolution_ratio),
                                         std::round(point[2]*resolution_ratio));
                return std::binary_search(region.begin(),region.end(),p);
            }
            image::vector<3,short> p(std::round(point[0]),
                                     std::round(point[1]),
                                     std::round(point[2]));
            return std::binary_search(region.begin(),region.end(),p);
        }
        template<typename value_type>
        bool has_points(const std::vector<image::vector<3,value_type> >& points) const
        {
            for(unsigned int index = 0; index < points.size(); ++index)
                if(has_point(points[index]))
                    return true;
            return false;
        }
        void get_quantitative_data(std::shared_ptr<fib_data> handle,std::vector<std::string>& titles,std::vector<float>& data);
};

#endif
