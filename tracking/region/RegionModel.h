// ---------------------------------------------------------------------------

#ifndef RegionRenderH
#define RegionRenderH
#include <QtOpenGL>
#include <vector>
#include <map>
#include "TIPL/tipl.hpp"

namespace tipl{
    class march_cube;
}
class GLWidget;
// ---------------------------------------------------------------------------
class RegionRender {
private:
        GLWidget* glwidget = nullptr;
        GLuint surface = 0;
public:
        std::shared_ptr<tipl::march_cube> object;
        std::vector<std::vector<unsigned int> > sorted_index;
        tipl::vector<3,float> center;
        void sortIndices(void);
public:
        float alpha = 0.6f;
        tipl::rgb color = uint32_t(0x00FFFFFF);

public:
        ~RegionRender(void);
        void swap(RegionRender& rhs) {
            std::swap(surface,rhs.surface);
            std::swap(object,rhs.object);
            sorted_index.swap(rhs.sorted_index);
            std::swap(center,rhs.center);
            std::swap(alpha,rhs.alpha);
            std::swap(color,rhs.color);
        }
        const RegionRender& operator = (const RegionRender & rhs) = delete;
        // bool load_from_file(const char* file_name);
        bool load(const tipl::image<3>& image_, float threshold_);
        bool load(const std::vector<tipl::vector<3,short> >& region, tipl::matrix<4,4>& trans,unsigned char smooth);
        //bool load(const tipl::image<3,unsigned char>& mask,unsigned char threshold);
        bool load(unsigned int* buffer, tipl::shape<3>geo, unsigned int threshold);
        tipl::march_cube* get(void) {return object.get();}
        const std::vector<unsigned int>& getSortedIndex(unsigned char view) const
        {return sorted_index[view];}

        void move_object(const tipl::vector<3,float>& shift);
        const std::vector<tipl::vector<3> >& point_list(void) const;
        const std::vector<tipl::vector<3> >& normal_list(void) const;
        const std::vector<tipl::vector<3,unsigned int> >& tri_list(void) const;
        void transform_point_list(const tipl::matrix<4,4>& T);
};

#endif
