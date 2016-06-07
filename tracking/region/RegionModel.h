// ---------------------------------------------------------------------------

#ifndef RegionModelH
#define RegionModelH
#include <vector>
#include <map>
#include "image/image.hpp"
// ---------------------------------------------------------------------------
class RegionModel {
public:
        typedef image::march_cube<image::vector<3,float> >mesh_type;
        std::auto_ptr<mesh_type> object;
        std::vector<std::vector<unsigned int> > sorted_index;
        void sortIndices(void);
public:
        float alpha;
        image::rgb_color color;
        RegionModel(void) {
                alpha = 0.6;
		color = (unsigned int)0x00FFFFFF;
        }
        void swap(RegionModel& rhs) {
                std::swap(alpha,rhs.alpha);
                std::swap(color,rhs.color);
                std::swap(object,rhs.object);
                sorted_index.swap(rhs.sorted_index);
        }
	const RegionModel& operator = (const RegionModel & rhs);
        // bool load_from_file(const char* file_name);
        bool load(const image::basic_image<float, 3>& image_, float threshold_);
        bool load(const std::vector<image::vector<3,short> >& region, double scale,unsigned char smooth);
        //bool load(const image::basic_image<unsigned char, 3>& mask,unsigned char threshold);
        bool load(unsigned int* buffer, image::geometry<3>geo, unsigned int threshold);
        mesh_type* get(void) {return object.get();}
        const std::vector<unsigned int>& getSortedIndex(unsigned char view) const
        {return sorted_index[view];}

        void move_object(const image::vector<3,float>& shift);
};

#endif
