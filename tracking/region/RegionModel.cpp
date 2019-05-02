// ---------------------------------------------------------------------------
#include <set>
#include <fstream>
#include <sstream>
#include <iterator>
#include "RegionModel.h"
#include "SliceModel.h"
// ---------------------------------------------------------------------------
const RegionModel& RegionModel:: operator = (const RegionModel & rhs)
{
    if (rhs.object.get())
        object.reset(new mesh_type(*rhs.object.get()));
    sorted_index = rhs.sorted_index;
    alpha = rhs.alpha;
    color = rhs.color;
    return *this;
}

//---------------------------------------------------------------------------
void RegionModel::sortIndices(void)
        //const unsigned int* meshes,unsigned int mesh_count,const float* vertices)
{
    sorted_index.resize(6);
    unsigned int mesh_count = object->tri_list.size();
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




bool RegionModel::load(const std::vector<tipl::vector<3,short> >& seeds, double resolution_ratio,unsigned char smooth)
{
    if(seeds.empty())
    {
        object.reset(0);
        return false;
    }
    tipl::vector<3,short> max_value(seeds[0]), min_value(seeds[0]);
    tipl::bounding_box_mt(seeds,max_value,min_value);

    center = max_value;
    center += min_value;
    center *= 0.5;
    center /= resolution_ratio;

    max_value += tipl::vector<3,short>(5, 5, 5);
    min_value -= tipl::vector<3,short>(5, 5, 5);
    tipl::geometry<3> geo(max_value[0] - min_value[0],
            max_value[1] - min_value[1], max_value[2] - min_value[2]);
    float cur_scale = 1.0;
    while(geo.width() > 256 || geo.height() > 256 || geo.depth() > 256)
    {
        cur_scale *= 2.0;
        geo[0] /= 2.0;
        geo[1] /= 2.0;
        geo[2] /= 2.0;
    }


    tipl::image<unsigned char, 3>buffer(geo);
    tipl::par_for(seeds.size(),[&](unsigned int index)
    {
        tipl::vector<3,short> point(seeds[index]);
        point -= min_value;
        point /= cur_scale;
        buffer[tipl::pixel_index<3>(point[0], point[1], point[2],
                                     buffer.geometry()).index()] = 200;
    });


    while(smooth)
    {
        tipl::filter::mean(buffer);
        --smooth;
    }

    object.reset(new tipl::march_cube<tipl::vector<3,float> >(buffer, 20));
    if (object->point_list.empty())
    {
        object.reset(0);
        return false;
    }
    tipl::vector<3,float>shift(min_value);
    if (resolution_ratio != 1.0)
    {
        cur_scale /= resolution_ratio;
        shift /= resolution_ratio;
    }
    tipl::par_for(object->point_list.size(),[&](unsigned int index)
    {
        if (cur_scale != 1.0)
            object->point_list[index] *= cur_scale;
        object->point_list[index] += shift;
    });
    sortIndices();
    return object.get();
}

bool RegionModel::load(const tipl::image<float, 3>& image_,
                       float threshold)
{
    tipl::image<float, 3> image_buffer(image_);

    float scale = 1.0;
    while(image_buffer.width() > 256 || image_buffer.height() > 256 || image_buffer.depth() > 256)
    {
        scale *= 2.0;
        tipl::downsampling(image_buffer);
    }
    if (threshold == 0.0)
    {
        float sum = 0;
        unsigned int num = 0;
        unsigned int index_to = (image_buffer.size() >> 1) + image_buffer.geometry()
                          .plane_size();
        for (unsigned int index = (image_buffer.size() >> 1); index < index_to;++index)
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
        threshold = sum / num * 0.85;
    }
    object.reset(new tipl::march_cube<tipl::vector<3,float> >(image_buffer,
                 threshold));
    if (scale != 1.0)
        for (unsigned int index = 0; index < object->point_list.size(); ++index)
            object->point_list[index] *= scale;
    sortIndices();
    return object->point_list.size();
}
// ---------------------------------------------------------------------------

bool RegionModel::load(unsigned int* buffer, tipl::geometry<3>geo,
                       unsigned int threshold)
{
    tipl::image<unsigned char, 3>re_buffer(geo);
    for (unsigned int index = 0; index < re_buffer.size(); ++index)
        re_buffer[index] = buffer[index] > threshold ? 200 : 0;

    tipl::filter::mean(re_buffer);

    object.reset(new tipl::march_cube<tipl::vector<3,float> >(re_buffer, 50));
    if (object->point_list.empty())
        object.reset(0);
    else
        sortIndices();
    return object.get();
}

void RegionModel::move_object(const tipl::vector<3,float>& shift)
{
    if(!object.get())
        return;
    tipl::add_constant(object->point_list,shift);

}
