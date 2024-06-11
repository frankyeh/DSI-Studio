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
    tipl::bounding_box(seeds,max_value,min_value);

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
    tipl::vector<3,float> shift(min_value);
    tipl::par_for(object->point_list.size(),[&](unsigned int index)
    {
        if (cur_scale != 1.0f)
            object->point_list[index] *= cur_scale;
        object->point_list[index] += shift;
        if(need_trans)
            object->point_list[index].to(trans);
    });
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
        tipl::multiply_constant(object->point_list,scale);
    return object.get();
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
    return object.get();
}

void RegionRender::move_object(const tipl::vector<3,float>& shift)
{
    if(object.get())
        tipl::add_constant(object->point_list,shift);
}

void RegionRender::transform_point_list(const tipl::matrix<4,4>& T)
{
    if(!object.get())
        return;
    auto& point_list = object->point_list;
    tipl::par_for(point_list.size(),[&](unsigned int i){
        point_list[i].to(T);
    });
}
std::string RegionRender::get_obj(unsigned int& coordinate_count,tipl::vector<3> vs)
{
    std::string output;
    for(auto& pos : object->point_list)
    {
        output.push_back('v');
        output.push_back(' ');
        output += std::to_string(pos[0]*vs[0]);
        output.pop_back();
        output.pop_back();
        output.pop_back();
        output.back() = ' ';
        output += std::to_string(pos[2]*vs[2]);
        output.pop_back();
        output.pop_back();
        output.pop_back();
        output.back() = ' ';
        output += std::to_string(-pos[1]*vs[1]);
        output.pop_back();
        output.pop_back();
        output.pop_back();
        output.back() = '\n';
    }
    ++coordinate_count;
    for(auto& each : object->tri_list)
    {
        output.push_back('f');
        output.push_back(' ');
        output += std::to_string(each[0] + coordinate_count);
        output.push_back(' ');
        output += std::to_string(each[1] + coordinate_count);
        output.push_back(' ');
        output += std::to_string(each[2] + coordinate_count);
        output.push_back('\n');
    }
    coordinate_count += object->point_list.size()-1;
    return output;
}
void handleAlpha(tipl::rgb color,float alpha,int blend1,int blend2);
void RegionRender::draw(unsigned char cur_view,float alpha,int blend1,int blend2)
{
    if(!object.get() || object->tri_list.empty())
        return;
    handleAlpha(color,alpha,blend1,blend2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &object->point_list[0]);
    glNormalPointer(GL_FLOAT, 0, &object->normal_list[0]);
    glDrawElements(GL_TRIANGLES, object->sorted_index[cur_view].size(),
                       GL_UNSIGNED_INT,&object->sorted_index[cur_view][0]);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}
