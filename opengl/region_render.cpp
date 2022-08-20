// ---------------------------------------------------------------------------
#include <set>
#include <fstream>
#include <sstream>
#include <iterator>
#include "region_render.hpp"
#include "SliceModel.h"
#include "glwidget.h"

RegionRender::~RegionRender(void)
{
    if(surface)
    {
        glwidget->glDeleteBuffers(1,&surface);
        surface = 0;
    }
}
//---------------------------------------------------------------------------
void RegionRender::sortIndices(void)
        //const unsigned int* meshes,unsigned int mesh_count,const float* vertices)
{
    sorted_index.resize(6);
    auto mesh_count = object->tri_list.size();
    for (unsigned int view_index = 0;view_index < 3;++view_index)
    {
        std::vector<std::pair<float,unsigned int> > index_weighting(mesh_count);
        for (unsigned int index = 0;index < mesh_count;++index)
            index_weighting[index] =
            std::make_pair(object->point_list[object->tri_list[index][0]][view_index],index);

        std::sort(index_weighting.begin(),index_weighting.end());

        sorted_index[view_index].resize(mesh_count*3);
        std::vector<unsigned int>::iterator index_iter1 = sorted_index[view_index].begin();
        for (unsigned int index = 0;index < mesh_count;++index)
        {
            unsigned int new_index = index_weighting[index].second;
            *index_iter1 = object->tri_list[new_index][0];
            ++index_iter1;
            *index_iter1 = object->tri_list[new_index][1];
            ++index_iter1;
            *index_iter1 = object->tri_list[new_index][2];
            ++index_iter1;
        }
        sorted_index[view_index+3].resize(mesh_count*3);
        std::copy(sorted_index[view_index].begin(),sorted_index[view_index].end(),
                    sorted_index[view_index+3].rbegin());
    }
}




bool RegionRender::load(const std::vector<tipl::vector<3,short> >& seeds, tipl::matrix<4,4>& trans,unsigned char smooth)
{
    if(seeds.empty())
    {
        object.reset();
        return false;
    }
    bool need_trans = (trans != tipl::identity_matrix());

    tipl::vector<3,short> max_value(seeds[0]), min_value(seeds[0]);
    tipl::bounding_box_mt(seeds,max_value,min_value);

    center = max_value;
    center += min_value;
    center *= 0.5f;
    max_value += tipl::vector<3,short>(5, 5, 5);
    min_value -= tipl::vector<3,short>(5, 5, 5);
    tipl::shape<3> geo(uint32_t(max_value[0] - min_value[0]),
                          uint32_t(max_value[1] - min_value[1]),
                          uint32_t(max_value[2] - min_value[2]));
    float cur_scale = 1.0f;
    while(geo.width() > 256 || geo.height() > 256 || geo.depth() > 256)
    {
        cur_scale *= 2.0f;
        geo = tipl::shape<3>(geo[0]/2,geo[1]/2,geo[2]/2);
    }


    tipl::image<3,unsigned char> buffer(geo);
    tipl::par_for(seeds.size(),[&](unsigned int index)
    {
        tipl::vector<3,short> point(seeds[index]);
        point -= min_value;
        point /= cur_scale;
        if(buffer.shape().is_valid(point))
            buffer[tipl::pixel_index<3>(point[0], point[1], point[2],
                                     buffer.shape()).index()] = 200;
    });


    while(smooth)
    {
        tipl::filter::mean(buffer);
        --smooth;
    }
    object.reset(new tipl::march_cube(buffer, uint8_t(20)));
    if (object->point_list.empty())
    {
        object.reset();
        return false;
    }
    tipl::vector<3,float> shift(min_value);
    tipl::par_for(object->point_list.size(),[&](unsigned int index)
    {
        if (cur_scale != 1.0f)
            object->point_list[index] *= cur_scale;
        object->point_list[index] += shift;
        if(need_trans)
            object->point_list[index].to(trans);
    });
    sortIndices();
    return object.get();
}

bool RegionRender::load(const tipl::image<3>& image_,
                       float threshold)
{
    tipl::image<3> image_buffer(image_);

    float scale = 1.0f;
    while(image_buffer.width() > 256 || image_buffer.height() > 256 || image_buffer.depth() > 256)
    {
        scale *= 2.0f;
        tipl::downsampling(image_buffer);
    }
    if (threshold == 0.0f)
    {
        float sum = 0;
        unsigned int num = 0;
        auto index_to = (image_buffer.size() >> 1) + image_buffer.shape().plane_size();
        for (auto index = (image_buffer.size() >> 1); index < index_to;++index)
        {
            float g = image_buffer[index];
            if (g > 0)
            {
                sum += g;
                ++num;
            }
        }
        if (!num)
            return false;
        threshold = sum / num * 0.85f;
    }
    object.reset(new tipl::march_cube(image_buffer,threshold));
    if (scale != 1.0f)
        for (unsigned int index = 0; index < object->point_list.size(); ++index)
            object->point_list[index] *= scale;
    sortIndices();
    return object->point_list.size();
}
// ---------------------------------------------------------------------------

bool RegionRender::load(unsigned int* buffer, tipl::shape<3>geo,
                       unsigned int threshold)
{
    tipl::image<3,unsigned char>re_buffer(geo);
    for (unsigned int index = 0; index < re_buffer.size(); ++index)
        re_buffer[index] = buffer[index] > threshold ? 200 : 0;

    tipl::filter::mean(re_buffer);
    object.reset(new tipl::march_cube(re_buffer, 50));
    if (object->point_list.empty())
        object.reset();
    else
        sortIndices();
    return object.get();
}

void RegionRender::move_object(const tipl::vector<3,float>& shift)
{
    if(!object.get())
        return;
    tipl::add_constant(object->point_list,shift);

}

const std::vector<tipl::vector<3> >& RegionRender::point_list(void) const
{
    return object->point_list;
}
const std::vector<tipl::vector<3> >& RegionRender::normal_list(void) const
{
    return object->normal_list;
}
const std::vector<tipl::vector<3,unsigned int> >& RegionRender::tri_list(void) const
{
    return object->tri_list;
}

void RegionRender::trasnform_point_list(const tipl::matrix<4,4>& T)
{
    if(!object.get())
        return;
    auto& point_list = object->point_list;
    tipl::par_for(point_list.size(),[&](unsigned int i){
        point_list[i].to(T);
    });
}
