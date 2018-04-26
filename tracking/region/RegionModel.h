// ---------------------------------------------------------------------------

#ifndef RegionModelH
#define RegionModelH
#include <vector>
#include <map>
#include "tipl/tipl.hpp"
// ---------------------------------------------------------------------------
class RegionModel {
public:
        typedef tipl::march_cube<tipl::vector<3,float> >mesh_type;
        std::auto_ptr<mesh_type> object;
        std::vector<std::vector<unsigned int> > sorted_index;
        tipl::vector<3,float> center;
        void sortIndices(void);
public:
        float alpha;
        tipl::rgb color;
        RegionModel(void) {
                alpha = 0.6f;
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
        bool load(const tipl::image<float, 3>& image_, float threshold_);
        bool load(const std::vector<tipl::vector<3,short> >& region, double scale,unsigned char smooth);
        //bool load(const tipl::image<unsigned char, 3>& mask,unsigned char threshold);
        bool load(unsigned int* buffer, tipl::geometry<3>geo, unsigned int threshold);
        mesh_type* get(void) {return object.get();}
        const std::vector<unsigned int>& getSortedIndex(unsigned char view) const
        {return sorted_index[view];}

        void move_object(const tipl::vector<3,float>& shift);
};

#endif
