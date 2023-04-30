// ---------------------------------------------------------------------------

#ifndef RegionRenderH
#define RegionRenderH
#include <QtOpenGL>
#include <vector>
#include <map>
#include "zlib.h"
#include "TIPL/tipl.hpp"

namespace tipl{
    class march_cube;
}
class GLWidget;
// ---------------------------------------------------------------------------
class RegionRender {
public:
        std::shared_ptr<tipl::march_cube> object;
        tipl::vector<3,float> center;
public:
        float alpha = 0.6f;
        tipl::rgb color = uint32_t(0x00FFFFFF);

public:
        ~RegionRender(void);
        void swap(RegionRender& rhs) {
            std::swap(object,rhs.object);
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
        void move_object(const tipl::vector<3,float>& shift);
        void transform_point_list(const tipl::matrix<4,4>& T);
public:
        void draw(unsigned char cur_view,float alpha,int blend1,int blend2);
};

#endif
