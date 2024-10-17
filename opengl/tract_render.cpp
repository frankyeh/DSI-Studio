#include "glwidget.h"
#include "tract_render.hpp"
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

void TractRenderData::add_tract(const tracking_window& param,
                                const std::vector<float>& tract,
                                const TractRenderShader& shader,
                                const tipl::vector<3>& assigned_color,
                                const std::vector<float>& metrics)
{
    auto tract_alpha = param["tract_alpha"].toFloat();
    auto tract_color_saturation = param["tract_color_saturation"].toFloat();
    auto tract_color_brightness = param["tract_color_brightness"].toFloat();
    auto tube_diameter = param["tube_diameter"].toFloat();

    auto tract_style = param["tract_style"].toInt();
    auto tract_color_style = param["tract_color_style"].toInt();
    auto tract_tube_detail = param["tract_tube_detail"].toInt();
    auto tract_shader = param["tract_shader"].toInt();
    auto end_point_shift = param["end_point_shift"].toInt();

    auto color_min = param["tract_color_min_value"].toFloat();
    auto color_r = param["tract_color_max_value"].toFloat()-color_min;

    const float detail_option[5] = {1.0f,0.5f,0.25f,0.0f,0.0f};
    auto tract_color_saturation_base = tract_color_brightness*(1.0f-tract_color_saturation);

    auto tube_detail = tube_diameter*detail_option[tract_tube_detail]*4.0f;
    auto tract_shaderf = 0.01f*float(tract_shader);

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
        if (tract_style && index != 0 && index+1 != vertex_count)
        {
            tipl::vector<3,float> displacement(data_iter+3);
            displacement -= last_pos;
            displacement -= prev_vec_n*(prev_vec_n*displacement);
            if (displacement.length() < tube_detail)
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

        switch(tract_color_style)
        {
            case 0://directional
                cur_color = abs_vec_n;
                if(tract_color_saturation != 1.0f)
                {
                    cur_color *= tract_color_saturation;
                    cur_color += tract_color_saturation_base;
                }
                break;
            case 2://local anisotropy
                if(index < metrics.size())
                    cur_color = param.tractWidget->color_map.value2color(metrics[index],color_min,color_r);
                break;
            default:
                cur_color = assigned_color;
                break;
        }

        if(tract_shader)
        {
            cur_color *= 1.0f-std::min<float>(shader.get_shade(pos)*tract_shaderf,0.95f);
            cur_color += 0.05f;
        }
        if(tract_style)
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
        vec_ab *= tube_diameter;
        vec_ba *= tube_diameter;
        vec_a *= tube_diameter;
        vec_b *= tube_diameter;

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
        if(tract_style >= 2) // show_end_only
        {
            // add end
            if (index == 0)
            {
                if(tract_style != 3)
                {
                    tipl::vector<3,float> shift(vec_n);
                    shift *= -(int)end_point_shift;
                    for (unsigned int k = 0;k < 8;++k)
                    {
                        tipl::vector<3,float> cur_point = points[end_sequence[k]];
                        cur_point += shift;
                        add_tube(cur_point,cur_color,-vec_n,tract_alpha);
                    }
                    if(tract_style == 2)
                        end_tube_strip();
                }

            }
            else
                if(index + 1 == vertex_count)
                {
                    if(tract_style != 4)
                    {
                        tipl::vector<3,float> shift(vec_n);
                        shift *= (int)end_point_shift;
                        for (unsigned int k = 0;k < 8;++k)
                        {
                            tipl::vector<3,float> cur_point = points[end_sequence2[k]];
                            cur_point += shift;
                            add_tube(cur_point,cur_color,vec_n,tract_alpha);
                        }
                    }
                }
        }
        else
        {
            if (index == 0)
            {
                for (unsigned int k = 0;k < 8;++k)
                    add_tube(points[end_sequence[k]],cur_color,normals[end_sequence[k]],tract_alpha);
            }
            else
            {
                add_tube(points[0],cur_color,normals[0],tract_alpha);
                for (unsigned int k = 1;k < 8;++k)
                {
                   add_tube(previous_points[k],previous_color,previous_normals[k],tract_alpha);
                   add_tube(points[k],cur_color,normals[k],tract_alpha);
                }
                add_tube(points[0],cur_color,normals[0],tract_alpha);

                if(index +1 == vertex_count)
                {
                    for (unsigned int k = 2;k < 8;++k) // skip 0 and 1 because the tubes have them
                        add_tube(points[end_sequence2[k]],cur_color,normals[end_sequence2[k]],tract_alpha);
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
            add_line(pos,cur_color,tract_alpha);
        }
    }
    if(tract_style)
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

TractRenderShader::TractRenderShader(tracking_window& cur_tracking_window):
    dim(cur_tracking_window.handle->dim),
    min_x_map(tipl::shape<2>(64,64),float(dim.width())),
    min_y_map(tipl::shape<2>(64,64),float(dim.height())),
    min_z_map(tipl::shape<2>(64,64),float(dim.depth())),
    max_x_map(tipl::shape<2>(64,64)),
    max_y_map(tipl::shape<2>(64,64)),
    max_z_map(tipl::shape<2>(64,64))
{
    to64[0] = 64.0f/float(dim[0]);
    to64[1] = 64.0f/float(dim[1]);
    to64[2] = 64.0f/float(dim[2]);

    auto models = cur_tracking_window.tractWidget->get_checked_tracks();
    size_t total_visible_tract = 0;
    for(auto each : models)
        total_visible_tract += each->get_visible_track_count();

    skip_rate = cur_tracking_window["tract_visible_tract"].toFloat()/float(total_visible_tract);

    tipl::adaptive_par_for(models.size(),[&](size_t i)
    {
        auto tracks_count = models[i]->get_visible_track_count();
        tipl::uniform_dist<float> uniform_gen(0.0f,1.0f);
        for (unsigned int data_index = 0; data_index < tracks_count; ++data_index)
        {
            if (models[i]->get_tract(data_index).size() < 6)
                continue;
            if(skip_rate < 1.0f && uniform_gen() > skip_rate)
                continue;
            auto& cur_tract = models[i]->get_tract(data_index);
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
    });

    for(int i = 0;i < 3; ++i)
    {
        tipl::filter::mean(max_x_map);
        tipl::filter::mean(min_x_map);
        tipl::filter::mean(max_y_map);
        tipl::filter::mean(min_y_map);
        tipl::filter::mean(max_z_map);
        tipl::filter::mean(min_z_map);
    }
}
void TractRender::prepare_update(tracking_window& param,
                                 std::shared_ptr<TractModel>& active_tract_model,
                                 const TractRenderShader& shader)
{
    auto lock = start_reading(false);
    if(!lock.get())
        return;

    auto dim = param.handle->dim;

    std::vector<unsigned int> visible;
    {
        auto tracks_count = active_tract_model->get_visible_track_count();
        visible.reserve(tracks_count);
        tipl::uniform_dist<float> uniform_gen(0.0f,1.0f);
        for (unsigned int data_index = 0; data_index < tracks_count; ++data_index)
        {
            if(shader.skip_rate < 1.0f && uniform_gen() > shader.skip_rate)
                continue;
            if (active_tract_model->get_tract(data_index).size() <= 3)
                continue;
            visible.push_back(data_index);
        }
    }
    auto tract_color_style = param["tract_color_style"].toInt();
    auto index_num = param["tract_color_index"].toInt();
    tipl::out() << "rendering using metrics: " << (index_num < param.handle->dir.index_name.size() ?  param.handle->dir.index_name[index_num]:param.handle->slices[index_num]->name);

    std::vector<tipl::vector<3> > assigned_colors;
    // Directional:Assigned:Local Index:Averaged Index:Averaged Directional:Max Index
    if(tract_color_style == 1 || // assigned
       tract_color_style == 3 || // mean value
       tract_color_style == 5)   // max value
    {
        auto color_min = param["tract_color_min_value"].toFloat();
        auto color_r = param["tract_color_max_value"].toFloat()-color_min;
        assigned_colors.resize(visible.size());
        for (unsigned int i = 0;i < visible.size();++i)
        {
            if(tract_color_style == 1) // assigned
            {
                tipl::rgb paint_color = active_tract_model->get_tract_color(visible[i]);
                assigned_colors[i] = tipl::vector<3,float>(paint_color.r,paint_color.g,paint_color.b);
                assigned_colors[i] /= 255.0f;
                continue;
            }
            std::vector<float> metrics(active_tract_model->get_tract_data(param.handle,visible[i],index_num));
            if(tract_color_style == 3) // mean values
                assigned_colors[i] = param.tractWidget->color_map.value2color(tipl::mean(metrics),color_min,color_r);
            if(tract_color_style == 5) // max values
                assigned_colors[i] = param.tractWidget->color_map.value2color(tipl::max_value(metrics),color_min,color_r);
        };
    }




    std::vector<TractRenderData> new_data(data_block_count);
    tipl::par_for(data_block_count,[&](unsigned int thread)
    {
        for(unsigned int i = thread;i < visible.size();i += data_block_count)
        {
            if(about_to_write)
                break;
            std::vector<float> metrics;
            if(tract_color_style == 2) // local values
                metrics = std::move(active_tract_model->get_tract_data(param.handle,visible[i],index_num));
            new_data[thread].add_tract(param,active_tract_model->get_tract(visible[i]),shader,
                           assigned_colors.empty() ? tipl::vector<3>() : assigned_colors[i],metrics);
        }
    },4);

    new_data.swap(data);
    new_data.clear();
    need_update = false;
    update_data_count = 0;
}
bool TractRender::render_tracts(size_t index,
                                GLWidget* glwidget,
                                std::chrono::high_resolution_clock::time_point end_time)
{
    if(index < data.size() && !data[index].draw(glwidget,end_time))
    {
        if(update_data_count < index)
        {
            update_data_count = index;
            return false; // will emit update
        }
    }
    update_data_count = index;
    return true;
}
