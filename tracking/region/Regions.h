// ---------------------------------------------------------------------------
#ifndef RegionsH
#define RegionsH
#include <vector>
#include <map>
#include <boost/thread/thread.hpp>
#include "image/image.hpp"
#include "RegionModel.h"
// ---------------------------------------------------------------------------

class SliceModel;
// region_feature
const unsigned int roi_id = 0;
const unsigned int roa_id = 1;
const unsigned int end_id = 2;
const unsigned int seed_id = 3;
const unsigned int none_roi_id = 4;

class ROIRegion {
private:
        std::vector<image::vector<3,short> >region;
        image::geometry<3> geo;
        image::vector<3> vs;
        bool modified;
private:
        std::auto_ptr<boost::thread> back_thread;
        std::auto_ptr<RegionModel> back_region;
        void updateMesh(bool smooth);
public:
        bool has_background_thread(void) const{return back_thread.get();}
public: // rendering options
        RegionModel show_region;
        unsigned char regions_feature;

        ROIRegion(const ROIRegion& rhs) : region(rhs.region), geo(rhs.geo),vs(rhs.vs),
        regions_feature(rhs.regions_feature), modified(true) {
                show_region = rhs.show_region;
    }
        const ROIRegion& operator = (const ROIRegion & rhs) {
            region = rhs.region;
            geo = rhs.geo;
            vs = rhs.vs;
            regions_feature = rhs.regions_feature;
            show_region = rhs.show_region;
            modified = true;
            return *this;
        }

        ROIRegion(const image::geometry<3>& geo_, const image::vector<3>& vs_)
            : geo(geo_), vs(vs_),modified(false){}

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

    void LoadFromBuffer(const image::basic_image<unsigned char, 3>& mask);
    void SaveToBuffer(image::basic_image<unsigned char, 3>& mask,
                unsigned char value=255);

        void getSlicePosition(SliceModel* slice, unsigned int pindex, int& x, int& y,
        int& z);

        void makeMeshes(bool smooth);
        bool has_point(const image::vector<3,short>& point);
        bool has_points(const std::vector<image::vector<3,short> >& points);
};

#endif
