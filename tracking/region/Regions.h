// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>
#include <boost/thread/thread.hpp>
#include "image/image.hpp"
#include "RegionModel.h"
// ---------------------------------------------------------------------------
class ODFModel;
class SliceModel;
// region_feature
const unsigned int roi_id = 0;
const unsigned int roa_id = 1;
const unsigned int end_id = 2;
const unsigned int seed_id = 3;
const unsigned int none_roi_id = 4;

class ROIRegion {
private:
        std::vector<image::vector<3,short> > region;
        image::geometry<3> geo;
        image::vector<3> vs;
        bool modified;
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
            regions_feature = rhs.regions_feature;
            show_region = rhs.show_region;
            modified = true;
            has_back_thread = rhs.has_back_thread;
            back_thread_id = rhs.back_thread_id;
            return *this;
        }

        ROIRegion(const image::geometry<3>& geo_, const image::vector<3>& vs_)
            : geo(geo_), vs(vs_),has_back_thread(false),modified(false){}

        const std::vector<image::vector<3,short> >& get(void) const {
        return region;
    }
        void assign(const std::vector<image::vector<3,short> >& region_)
        {
            region = region_;
            modified = true;
        }

    bool empty(void) const {
        return region.empty();
    }

    void clear(void) {
        modified = true;
        region.clear();
    }

        void erase(unsigned int index){
            modified = true;
            region.erase(region.begin()+index);
        }

        unsigned int size(void) const {
        return region.size();
    }

        void push_back(const image::vector<3,short>& point) {
        region.push_back(point);
        modified = true;
    }

        std::vector<image::vector<3,short> >::const_iterator
                begin(void) const {return region.begin();}

public:
        void add_points(std::vector<image::vector<3,short> >& points,bool del);
        void SaveToFile(const char* FileName,const std::vector<float>& trans);
        bool LoadFromFile(const char* FileName,const std::vector<float>& trans);
        void Flip(unsigned int dimension);
        void shift(const image::vector<3,short>& dx);

        template<typename image_type>
        void LoadFromBuffer(const image_type& from,const std::vector<float>& trans)
        {
            std::vector<image::vector<3,short> > points;
            for (image::pixel_index<3> index; index.is_valid(geo);index.next(geo))
            {
                image::vector<3> p(index.begin()),p2;
                image::vector_transformation(p.begin(),p2.begin(),trans.begin(),image::vdim<3>());
                p2 += 0.5;
                p2.floor();
                if (from.geometry().is_valid(p2) && from.at(p2[0],p2[1],p2[2]) != 0)
                    points.push_back(image::vector<3,short>(index.begin()));
            }
            region.swap(points);
            std::sort(region.begin(),region.end());
        }

        template<typename image_type>
        void LoadFromBuffer(const image_type& mask)
        {
            modified = true;region.clear();
            for (image::pixel_index<3>index;index.is_valid(mask.geometry());index.next(mask.geometry()))
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
        void get_quantitative_data(ODFModel* handle,std::vector<float>& data);
};

#endif
