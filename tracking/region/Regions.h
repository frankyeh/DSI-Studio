// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>

#include "image/image.hpp"
#include "RegionModel.h"
// ---------------------------------------------------------------------------
class FibData;
class SliceModel;
// region_feature
const unsigned int roi_id = 0;
const unsigned int roa_id = 1;
const unsigned int end_id = 2;
const unsigned int seed_id = 3;
const unsigned int terminate_id = 4;

class ROIRegion {
private:
        std::vector<image::vector<3,short> > region;
        image::geometry<3> geo;
        image::vector<3> vs;
        bool modified;
        std::vector<std::vector<image::vector<3,short> > > undo_backup;
        std::vector<std::vector<image::vector<3,short> > > redo_backup;
private:
        bool has_back_thread;
        unsigned int back_thread_id;
public:
        bool need_update(void);
        bool has_thread(void) const{return has_back_thread;}
public: // rendering options
        RegionModel show_region;
        unsigned char regions_feature;

        ROIRegion(const ROIRegion& rhs) :
            region(rhs.region), geo(rhs.geo),vs(rhs.vs),
            regions_feature(rhs.regions_feature), modified(true),has_back_thread(false)
        {
            show_region = rhs.show_region;
        }
        ~ROIRegion(void);
        const ROIRegion& operator = (const ROIRegion & rhs) {
            region = rhs.region;
            geo = rhs.geo;
            vs = rhs.vs;
            undo_backup = rhs.undo_backup;
            redo_backup = rhs.redo_backup;
            regions_feature = rhs.regions_feature;
            show_region = rhs.show_region;
            modified = true;
            has_back_thread = rhs.has_back_thread;
            back_thread_id = rhs.back_thread_id;
            return *this;
        }
        void swap(ROIRegion & rhs) {
            region.swap(rhs.region);
            geo.swap(rhs.geo);
            std::swap(vs,rhs.vs);
            undo_backup.swap(rhs.undo_backup);
            redo_backup.swap(rhs.redo_backup);
            std::swap(regions_feature,rhs.regions_feature);
            show_region.swap(rhs.show_region);
            std::swap(modified,rhs.modified);
            std::swap(has_back_thread,rhs.has_back_thread);
            std::swap(back_thread_id,rhs.back_thread_id);
        }

        ROIRegion(const image::geometry<3>& geo_, const image::vector<3>& vs_)
            : geo(geo_), vs(vs_),has_back_thread(false),modified(false){}

        const std::vector<image::vector<3,short> >& get(void) const {return region;}
        void assign(const std::vector<image::vector<3,short> >& region_)
        {
            region = region_;
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

        void push_back(const image::vector<3,short>& point)
        {
            region.push_back(point);
            modified = true;
        }

        std::vector<image::vector<3,short> >::const_iterator
                begin(void) const {return region.begin();}

public:
        void add(const ROIRegion & rhs)
        {
            std::vector<image::vector<3,short> > tmp(rhs.region);
            add_points(tmp,false);
        }
        void add_points(std::vector<image::vector<3,short> >& points,bool del);
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
        void SaveToFile(const char* FileName,const std::vector<float>& trans);
        bool LoadFromFile(const char* FileName,const std::vector<float>& trans);
        void Flip(unsigned int dimension);
        void shift(const image::vector<3,short>& dx);

        template<typename image_type,typename trans_type>
        void LoadFromBuffer(const image_type& from,const trans_type& trans)
        {
            std::vector<image::vector<3,short> > points;
            for (image::pixel_index<3> index(geo);index < geo.size();++index)
            {
                image::vector<3> p(index.begin());
                p.to(trans);
                p += 0.5;
                p.floor();
                if (from.geometry().is_valid(p) && from.at(p[0],p[1],p[2]) != 0)
                    points.push_back(image::vector<3,short>(index.begin()));
            }
            region.swap(points);
            std::sort(region.begin(),region.end());
        }

        template<typename image_type>
        void LoadFromBuffer(const image_type& mask)
        {
            modified = true;
            if(!region.empty())
                undo_backup.push_back(region);
            region.clear();
            for (image::pixel_index<3>index(mask.geometry());index < mask.size();++index)
                if (mask[index.index()] != 0)
                region.push_back(image::vector<3,short>(index.x(), index.y(),index.z()));
            std::sort(region.begin(),region.end());
        }
        void SaveToBuffer(image::basic_image<unsigned char, 3>& mask,
                unsigned char value=255);

        void getSlicePosition(SliceModel* slice, unsigned int pindex, int& x, int& y,int& z);

        void makeMeshes(bool smooth);
        bool has_point(const image::vector<3,short>& point);
        bool has_points(const std::vector<image::vector<3,short> >& points);
        void get_quantitative_data(FibData* handle,std::vector<std::string>& titles,std::vector<float>& data);
};

#endif
