#include <memory>
#include <algorithm>
#include <string>
#include <set>

#include "prog_interface_static_link.h"
#include "roi.hpp"
#include "tracking_model.hpp"

#include "tracking_method.hpp"
#include "image/image.hpp"

extern "C"
    void tracking_get_slices_dir_color(ODFModel* odf_model,unsigned short order,unsigned int* pixels)
{
    odf_model->getSlicesDirColor(order,pixels);
}

extern "C"
    const float* get_odf_direction(ODFModel* odf_model,unsigned int index)
{
    return &*odf_model->fib_data.fib.odf_table[index].begin();
}
extern "C"
    const unsigned short* get_odf_faces(ODFModel* odf_model,unsigned int index)
{
    return &*odf_model->fib_data.fib.odf_faces[index].begin();
}

extern "C"
    const float* get_odf_data(ODFModel* odf_model,unsigned int index)
{
    return odf_model->fib_data.fib.get_odf_data(index);
}
