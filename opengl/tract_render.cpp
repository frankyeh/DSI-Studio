#include "glwidget.h"
#include "tract_render.hpp"
#include "tracking/color_bar_dialog.hpp"
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

void TractRenderParam::init(GLWidget* glwidget,
                            tracking_window& cur_tracking_window)
{
    const float detail_option[5] = {1.0f,0.5f,0.25f,0.0f,0.0f};
    tract_alpha = glwidget->tract_alpha;
    tract_color_saturation = glwidget->tract_color_saturation;
    tract_color_brightness = glwidget->tract_color_brightness;
    tube_diameter = glwidget->tube_diameter;

    tract_style = glwidget->tract_style;
    tract_color_style = glwidget->tract_color_style;
    tract_tube_detail = glwidget->tract_tube_detail;
    tract_shader = glwidget->tract_shader;
    end_point_shift = glwidget->end_point_shift;

    tract_color_saturation_base = tract_color_brightness*(1.0f-tract_color_saturation);
    show_end_only = tract_style >= 2;
    tube_detail = tube_diameter*detail_option[tract_tube_detail]*4.0f;
    tract_shaderf = 0.01f*float(tract_shader);


    if(tract_color_style == 2 || // local index
       tract_color_style == 3 || // avg index
       tract_color_style == 5) // max index
    {
        color_map = cur_tracking_window.color_bar->color_map;
        color_r = cur_tracking_window.color_bar->color_r;
        color_min = cur_tracking_window.color_bar->color_min;
    }

    tract_visible_tract = float(cur_tracking_window["tract_visible_tract"].toInt());
    unsigned int track_num_index = cur_tracking_window.handle->get_name_index(
                                           cur_tracking_window.color_bar->get_tract_color_name().toStdString());
}


void TractRenderData::add_tract(const TractRenderParam& param,
                                const std::vector<float>& tract,
                                const TractRenderShader& shader,
                                const tipl::vector<3>& assigned_color,
                                const std::vector<float>& metrics)
{
    const unsigned char end_sequence[8] = {4,3,5,2,6,1,7,0};
    const unsigned char end_sequence2[8] = {7,0,6,1,5,2,4,3};
    std::vector<tipl::vector<3,float> > points(8),previous_points(8),normals(8),previous_normals(8);
    tipl::rgb paint_color;

    unsigned int vertex_count = tract.size()/3;
    const float* data_iter = &tract[0];

    tipl::vector<3,float> last_pos(data_iter),pos,
        vec_a(1,0,0),vec_b(0,1,0),
        vec_n,prev_vec_n,vec_ab,vec_ba,cur_color(assigned_color),previous_color;

    for (unsigned int index = 0; index < vertex_count;data_iter += 3, ++index)
    {
        // skip straight line!
        if (param.tract_style && index != 0 && index+1 != vertex_count)
        {
            tipl::vector<3,float> displacement(data_iter+3);
            displacement -= last_pos;
            displacement -= prev_vec_n*(prev_vec_n*displacement);
            if (displacement.length() < param.tube_detail)
                continue;
        }

        pos[0] = data_iter[0];
        pos[1] = data_iter[1];
        pos[2] = data_iter[2];
        if (index + 1 < vertex_count)
        {
            vec_n[0] = data_iter[3] - data_iter[0];
            vec_n[1] = data_iter[4] - data_iter[1];
            vec_n[2] = data_iter[5] - data_iter[2];
            vec_n.normalize();
        }
        auto abs_vec_n = vec_n;
        abs_vec_n.abs();

        switch(param.tract_color_style)
        {
            case 0://directional
                cur_color = abs_vec_n;
                if(param.tract_color_saturation != 1.0f)
                {
                    cur_color *= param.tract_color_saturation;
                    cur_color += param.tract_color_saturation_base;
                }
                break;
            case 2://local anisotropy
                if(index < metrics.size())
                    cur_color = param.get_color(metrics[index]);
                break;
            default:
                cur_color = assigned_color;
                break;
        }

        if(param.tract_shader)
        {
            cur_color *= 1.0f-std::min<float>(shader.get_shade(pos)*param.tract_shaderf,0.95f);
            cur_color += 0.05f;
        }
        if(param.tract_style)
        {

        if (index == 0 && std::fabs(vec_a*vec_n) > 0.5f)
            std::swap(vec_a,vec_b);

        vec_b = vec_a.cross_product(vec_n);
        vec_a = vec_n.cross_product(vec_b);
        vec_a.normalize();
        vec_b.normalize();
        vec_ba = vec_ab = vec_a;
        vec_ab += vec_b;
        vec_ba -= vec_b;
        vec_ab.normalize();
        vec_ba.normalize();
        // get normals
        {
            normals[0] = vec_a;
            normals[1] = vec_ab;
            normals[2] = vec_b;
            normals[3] = -vec_ba;
            normals[4] = -vec_a;
            normals[5] = -vec_ab;
            normals[6] = -vec_b;
            normals[7] = vec_ba;
        }
        vec_ab *= param.tube_diameter;
        vec_ba *= param.tube_diameter;
        vec_a *= param.tube_diameter;
        vec_b *= param.tube_diameter;

        // add point
        {
            std::fill(points.begin(),points.end(),pos);
            points[0] += vec_a;
            points[1] += vec_ab;
            points[2] += vec_b;
            points[3] -= vec_ba;
            points[4] -= vec_a;
            points[5] -= vec_ab;
            points[6] -= vec_b;
            points[7] += vec_ba;
        }
        if(param.show_end_only)
        {
            // add end
            if (index == 0)
            {
                if(param.tract_style != 3)
                {
                    tipl::vector<3,float> shift(vec_n);
                    shift *= -(int)param.end_point_shift;
                    for (unsigned int k = 0;k < 8;++k)
                    {
                        tipl::vector<3,float> cur_point = points[end_sequence[k]];
                        cur_point += shift;
                        add_tube(cur_point,cur_color,-vec_n,param.tract_alpha);
                    }
                    if(param.tract_style == 2)
                        end_tube_strip();
                }

            }
            else
                if(index + 1 == vertex_count)
                {
                    if(param.tract_style != 4)
                    {
                        tipl::vector<3,float> shift(vec_n);
                        shift *= (int)param.end_point_shift;
                        for (unsigned int k = 0;k < 8;++k)
                        {
                            tipl::vector<3,float> cur_point = points[end_sequence2[k]];
                            cur_point += shift;
                            add_tube(cur_point,cur_color,vec_n,param.tract_alpha);
                        }
                    }
                }
        }
        else
        {
            if (index == 0)
            {
                for (unsigned int k = 0;k < 8;++k)
                    add_tube(points[end_sequence[k]],cur_color,normals[end_sequence[k]],param.tract_alpha);
            }
            else
            {
                add_tube(points[0],cur_color,normals[0],param.tract_alpha);
                for (unsigned int k = 1;k < 8;++k)
                {
                   add_tube(previous_points[k],previous_color,previous_normals[k],param.tract_alpha);
                   add_tube(points[k],cur_color,normals[k],param.tract_alpha);
                }
                add_tube(points[0],cur_color,normals[0],param.tract_alpha);

                if(index +1 == vertex_count)
                {
                    for (unsigned int k = 2;k < 8;++k) // skip 0 and 1 because the tubes have them
                        add_tube(points[end_sequence2[k]],cur_color,normals[end_sequence2[k]],param.tract_alpha);
                }
            }
        }
        previous_points.swap(points);
        previous_normals.swap(normals);
        previous_color = cur_color;
        prev_vec_n = vec_n;
        last_pos = pos;
        }
        else
        {
            add_line(pos,cur_color,param.tract_alpha);
        }
    }
    if(param.tract_style)
        end_tube_strip();
    else
        end_line_strip();
}
void TractRenderData::clear(void)
{
    tube_vertices.clear();
    line_vertices.clear();
    tube_vertices_count = 0;
    line_vertices_count = 0;
    tube_strip_pos.clear();
    line_strip_pos.clear();
    tube_strip_pos.push_back(0);
    line_strip_pos.push_back(0);

}
bool TractRenderData::draw(GLWidget* glwidget,std::chrono::high_resolution_clock::time_point end_time)
{
    if(std::chrono::high_resolution_clock::now() > end_time)
        return false;
    if(!tube_strip_size.empty())
    {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        GLsizei stride = 10*sizeof(float); // 3 vertices + 3 normal + 4 color
        glVertexPointer(3, GL_FLOAT, stride, &tube_vertices[0]);
        glNormalPointer(GL_FLOAT, stride, &tube_vertices[0]+3);
        glColorPointer(3, GL_FLOAT, stride, &tube_vertices[0]+6);

        for(size_t i = 0;i < tube_strip_size.size() && std::chrono::high_resolution_clock::now() < end_time;++i)
            glDrawArrays(GL_TRIANGLE_STRIP,tube_strip_pos[i],tube_strip_size[i]);

        /*
        glwidget->glMultiDrawArrays(GL_TRIANGLE_STRIP,
                            &tube_strip_pos[0],
                            &tube_strip_size[0],
                            tube_strip_size.size());
                            */
        glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
    }

    if(!line_strip_size.empty())
    {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        GLsizei stride = 7*sizeof(float); // 3 vertices + 4 color
        glVertexPointer(3, GL_FLOAT, stride, &line_vertices[0]);
        glColorPointer(3, GL_FLOAT, stride, &line_vertices[0]+3);
        for(size_t i = 0;i < line_strip_size.size();++i)
            glDrawArrays(GL_LINE_STRIP,line_strip_pos[i],line_strip_size[i]);

        /*glwidget->glMultiDrawArrays(GL_LINE_STRIP,
                            &line_strip_pos[0],
                            &line_strip_size[0],
                            line_strip_size.size());*/

        glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
        glDisableClientState(GL_COLOR_ARRAY);
    }
    return std::chrono::high_resolution_clock::now() < end_time;
}

TractRender::TractRender(void)
{
}

TractRender::~TractRender(void)
{
    auto lock = start_writing();
}

float TractRenderShader::get_shade(const tipl::vector<3>& pos) const
{
    float d = 1.0f;
    {
        int x = std::round(pos[0]*to64[0]);
        int y = std::round(pos[1]*to64[1]);
        int z = std::round(pos[2]*to64[2]);
        if(max_x_map.shape().is_valid(y,z))
        {
            size_t p = size_t(y + z*max_x_map.width());
            d += std::min<float>(4.0f,
                                 std::min<float>(std::max<float>(0.0f,max_x_map[p]-pos[0]),
                                            std::max<float>(0.0f,pos[0]-min_x_map[p])));
        }
        if(max_y_map.shape().is_valid(x,z))
        {
            size_t p = size_t(x + z*max_y_map.width());
            d += std::min<float>(4.0f,
                                 std::min<float>(std::max<float>(0.0f,max_y_map[p]-pos[1]),
                                            std::max<float>(0.0f,pos[1]-min_y_map[p])));
        }
        if(max_z_map.shape().is_valid(x,y))
        {
            size_t p = size_t(x + y*max_z_map.width());
            d += std::min<float>(4.0f,
                                 std::min<float>(std::max<float>(0.0f,max_z_map[p]-pos[2]),
                                            std::max<float>(0.0f,pos[2]-min_z_map[p])));
        }
    }
    return d;
}
void TractRenderShader::add_shade(std::shared_ptr<TractModel>& active_tract_model,
                            const std::vector<unsigned int>& visible)
{
    for (unsigned int data_index : visible)
    {
        auto& cur_tract = active_tract_model->get_tract(data_index);
        if(cur_tract.size() < 6)
            continue;
        unsigned int vertex_count = cur_tract.size()/3;
        const float* data_iter = &cur_tract[0];
        for (unsigned int index = 0; index < vertex_count;data_iter += 3, ++index)
        {
            int x = std::round(data_iter[0]*to64[0]);
            int y = std::round(data_iter[1]*to64[1]);
            int z = std::round(data_iter[2]*to64[2]);
            if(max_x_map.shape().is_valid(y,z))
            {
                size_t pos = size_t(y + (z << 6));
                max_x_map[pos] = std::max<float>(max_x_map[pos],data_iter[0]);
                min_x_map[pos] = std::min<float>(min_x_map[pos],data_iter[0]);
            }
            if(max_y_map.shape().is_valid(x,z))
            {
                size_t pos = size_t(x + (z << 6));
                max_y_map[pos] = std::max<float>(max_y_map[pos],data_iter[1]);
                min_y_map[pos] = std::min<float>(min_y_map[pos],data_iter[1]);
            }
            if(max_z_map.shape().is_valid(x,y))
            {
                size_t pos = size_t(x + (y << 6));
                max_z_map[pos] = std::max<float>(max_z_map[pos],data_iter[2]);
                min_z_map[pos] = std::min<float>(min_z_map[pos],data_iter[2]);
            }
        }
    }

    {
        std::thread t1([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(max_x_map);});
        std::thread t2([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(min_x_map);});
        std::thread t3([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(max_y_map);});
        std::thread t4([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(min_y_map);});
        std::thread t5([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(max_z_map);});
        std::thread t6([&](){for(int i = 0;i < 3; ++i)tipl::filter::mean(min_z_map);});
        t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();
    }
}
void TractRender::prepare_update(std::shared_ptr<TractModel>& active_tract_model,
                                 TractRenderParam param,
                                 std::shared_ptr<fib_data> handle)
{
    auto lock = start_reading(false);
    if(!lock.get())
        return;

    auto dim = handle->dim;

    float skip_rate = param.tract_visible_tract/param.total_visible_tract;

    std::vector<unsigned int> visible;
    {
        visible.reserve(param.tract_visible_tract*1.5f);
        tipl::uniform_dist<float> uniform_gen(0.0f,1.0f);
        auto tracks_count = active_tract_model->get_visible_track_count();
        for (unsigned int data_index = 0; data_index < tracks_count && !about_to_write; ++data_index)
        {
            if(skip_rate < 1.0f && uniform_gen() > skip_rate)
                continue;
            if (active_tract_model->get_tract(data_index).size() <= 3)
                continue;
            visible.push_back(data_index);
        }
    }

    std::vector<tipl::vector<3> > assigned_colors;
    // Directional:Assigned:Local Index:Averaged Index:Averaged Directional:Max Index
    if(param.tract_color_style == 1 || // assigned
       param.tract_color_style == 3 || // mean value
       param.tract_color_style == 5)   // max value
    {
        assigned_colors.resize(visible.size());
        for (unsigned int i = 0;i < visible.size();++i)
        {
            if(param.tract_color_style == 1) // assigned
            {
                tipl::rgb paint_color = active_tract_model->get_tract_color(visible[i]);
                assigned_colors[i] = tipl::vector<3,float>(paint_color.r,paint_color.g,paint_color.b);
                assigned_colors[i] /= 255.0f;
                continue;
            }
            std::vector<float> metrics;
            active_tract_model->get_tract_data(handle,visible[i],param.track_num_index,metrics);
            if(param.tract_color_style == 3) // mean values
                assigned_colors[i] = param.get_color(tipl::mean(metrics));
            if(param.tract_color_style == 5) // max values
                assigned_colors[i] = param.get_color(tipl::max_value(metrics));
        };
    }

    TractRenderShader shader(dim);
    if(param.tract_shader)
        shader.add_shade(active_tract_model,visible);


    auto thread_count = std::min<int>(4,tipl::max_thread_count);
    new_data.clear();
    new_data.resize(thread_count);
    tipl::par_for(thread_count,[&](unsigned int thread)
    {
        for(unsigned int i = thread;i < visible.size();i += thread_count)
        {
            if(about_to_write)
                break;
            std::vector<float> metrics;
            if(param.tract_color_style == 2)
                active_tract_model->get_tract_data(handle,visible[i],param.track_num_index,metrics);
            new_data[thread].add_tract(param,active_tract_model->get_tract(visible[i]),shader,
                           assigned_colors.empty() ? tipl::vector<3>() : assigned_colors[i],metrics);
        }
    });
}
bool TractRender::render_tracts(GLWidget* glwidget)
{
    if(need_update)
    {
        need_update = false;
        new_data.swap(data);
        new_data.clear();
    }
    auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(100);
    for(auto& d: data)
    {
        if(!d.draw(glwidget,end_time))
            return false;
    }
    return true;
}
