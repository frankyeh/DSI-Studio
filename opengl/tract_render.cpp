#include "glwidget.h"
#include "tract_render.hpp"
#include "renderingtablewidget.h"
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

TractRenderSetting::TractRenderSetting(const tracking_window& param):
    alpha(param["tract_alpha"].toFloat()),saturation(param["tract_color_saturation"].toFloat()),
    saturation_base(param["tract_color_brightness"].toFloat()*(1.0f-saturation)),diameter(param["tube_diameter"].toFloat()),
    shader(0.01f*param["tract_shader"].toFloat()),color_min(param["tract_color_min_value"].toFloat()),
    color_r(param["tract_color_max_value"].toFloat()-color_min),style(param["tract_style"].toInt()),
    color_style(param["tract_color_style"].toInt()),end_point_shift(param["end_point_shift"].toInt()),
    color_map(param.tractWidget->color_map)
{
    const float detail_option[5] = {1.0f,0.5f,0.25f,0.0f,0.0f};
    detail = diameter*detail_option[param["tract_tube_detail"].toInt()]*4.0f;
}
void TractRenderData::add_tract(const TractRenderSetting& s,
                                const std::vector<float>& tract,
                                const TractRenderShader& shader,
                                const tipl::vector<3>& assigned_color,
                                const std::vector<float>& metrics)
{
    const unsigned char end_sequence[8] = {4,3,5,2,6,1,7,0};
    const unsigned char end_sequence2[8] = {7,0,6,1,5,2,4,3};
    std::vector<tipl::vector<3,float> > points(8),previous_points(8),normals(8),previous_normals(8);

    unsigned int vertex_count = tract.size()/3;
    const float* data_iter = &tract[0];

    tipl::vector<3,float> last_pos(data_iter),pos,
        vec_a(1,0,0),vec_b(0,1,0),
        vec_n,prev_vec_n,vec_ab,vec_ba,cur_color(assigned_color),previous_color;

    for (unsigned int index = 0; index < vertex_count;data_iter += 3, ++index)
    {
        // skip straight line!
        if (s.style && index != 0 && index+1 != vertex_count)
        {
            tipl::vector<3,float> displacement(data_iter+3);
            displacement -= last_pos;
            displacement -= prev_vec_n*(prev_vec_n*displacement);
            if (displacement.length() < s.detail)
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

        switch(s.color_style)
        {
            case 0://directional
                cur_color = vec_n;
                cur_color.abs();
                if(s.saturation != 1.0f)
                {
                    cur_color *= s.saturation;
                    cur_color += s.saturation_base;
                }
                break;
            case 2://local anisotropy
                if(index < metrics.size())
                    cur_color = s.color_map.value2color(metrics[index],s.color_min,s.color_r);
                break;
            default:
                cur_color = assigned_color;
                break;
        }

        if(s.shader != 0.0f)
        {
            cur_color *= 1.0f-std::min<float>(shader.get_shade(pos)*s.shader,0.95f);
        }
        if(!s.style)
        {
            add_line(pos,cur_color,s.alpha);
            continue;
        }

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
        vec_ab *= s.diameter;
        vec_ba *= s.diameter;
        vec_a *= s.diameter;
        vec_b *= s.diameter;

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
        if(s.style >= 2) // show_end_only
        {
            // add end
            if (index == 0)
            {
                if(s.style != 3)
                {
                    tipl::vector<3,float> shift(vec_n);
                    shift *= -s.end_point_shift;
                    for (unsigned int k = 0;k < 8;++k)
                    {
                        tipl::vector<3,float> cur_point = points[end_sequence[k]];
                        cur_point += shift;
                        add_tube(cur_point,cur_color,-vec_n,s.alpha);
                    }
                    if(s.style == 2)
                        end_strip();
                }

            }
            else
                if(index + 1 == vertex_count)
                {
                    if(s.style != 4)
                    {
                        tipl::vector<3,float> shift(vec_n);
                        shift *= s.end_point_shift;
                        for (unsigned int k = 0;k < 8;++k)
                        {
                            tipl::vector<3,float> cur_point = points[end_sequence2[k]];
                            cur_point += shift;
                            add_tube(cur_point,cur_color,vec_n,s.alpha);
                        }
                    }
                }
        }
        else
        {
            if (index == 0)
            {
                for (unsigned int k = 0;k < 8;++k)
                    add_tube(points[end_sequence[k]],cur_color,normals[end_sequence[k]],s.alpha);
            }
            else
            {
                add_tube(points[0],cur_color,normals[0],s.alpha);
                for (unsigned int k = 1;k < 8;++k)
                {
                   add_tube(previous_points[k],previous_color,previous_normals[k],s.alpha);
                   add_tube(points[k],cur_color,normals[k],s.alpha);
                }
                add_tube(points[0],cur_color,normals[0],s.alpha);

                if(index +1 == vertex_count)
                {
                    for (unsigned int k = 2;k < 8;++k) // skip 0 and 1 because the tubes have them
                        add_tube(points[end_sequence2[k]],cur_color,normals[end_sequence2[k]],s.alpha);
                }
            }
        }
        previous_points.swap(points);
        previous_normals.swap(normals);
        previous_color = cur_color;
        prev_vec_n = vec_n;
        last_pos = pos;
    }
    end_strip();
}
bool TractRenderData::draw(std::chrono::high_resolution_clock::time_point end_time)
{
    if(std::chrono::high_resolution_clock::now() > end_time)
        return false;
    if(strip_size.empty())
        return true;
    if(!buffer.isCreated()) // the first draw uploads the mesh to the gpu and frees it here
    {
        buffer.create();
        buffer.bind();
        buffer.allocate(vertices.data(),int(vertices.size()*sizeof(float)));
        std::vector<float>().swap(vertices);
    }
    else
        buffer.bind();
    GLsizei stride = (tube ? 10 : 7)*sizeof(float); // offsets in the bound buffer: vertex, normal (tube), then color
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, stride, nullptr);
    glColorPointer(3, GL_FLOAT, stride, reinterpret_cast<void*>((tube ? 6 : 3)*sizeof(float)));
    if(tube)
    {
        glEnableClientState(GL_NORMAL_ARRAY);
        glNormalPointer(GL_FLOAT, stride, reinterpret_cast<void*>(3*sizeof(float)));
    }
    auto mode = tube ? GL_TRIANGLE_STRIP : GL_LINE_STRIP;
    for(size_t i = 0;i < strip_size.size() && std::chrono::high_resolution_clock::now() < end_time;++i)
        glDrawArrays(mode,strip_pos[i],strip_size[i]);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    buffer.release();
    return std::chrono::high_resolution_clock::now() < end_time;
}

TractRender::~TractRender(void)
{
    auto lock = start_writing();
}

// the shading at pos, linearly interpolated between cells
float TractRenderShader::get_shade(const tipl::vector<3>& pos) const
{
    return tipl::estimate(shade,tipl::vector<3>(pos[0]*to_cell,pos[1]*to_cell,pos[2]*to_cell));
}
// a random draw of tract_visible_tract tracts over the checked bundles, also drawn by prepare_update, and the density of the cells of
// a 64^3 grid (2^shift voxels each) they pass through, smoothed by three mean filters: tracts inside dense bundles are shaded more,
// gaps and isolated tracts less, and nearby tracts share their shading; the shading is 24*(density/its maximum)^2: the densest core
// reaches black at the default shade (4), the squaring keeps outer tracts bright, and a single thin bundle, whose density never
// reaches that of a whole-brain tractogram, is shaded as much
TractRenderShader::TractRenderShader(tracking_window& cur_tracking_window,const std::vector<std::shared_ptr<TractModel> >& models):
    shade(tipl::shape<3>(64,64,64))
{
    const auto& dim = cur_tracking_window.handle->dim;
    int shift = 0; // a cell: 2^shift voxels
    for(auto max_dim = std::max({dim[0],dim[1],dim[2]});(max_dim >> shift) >= 64;++shift)
        ;
    to_cell = 1.0f/float(1 << shift);

    size_t total_visible_tract = 0;
    for(auto each : models)
        total_visible_tract += each->get_visible_track_count();
    float skip_rate = cur_tracking_window["tract_visible_tract"].toFloat()/float(total_visible_tract);
    for(auto each : models)
    {
        visible.emplace_back();
        tipl::uniform_dist<float> uniform_gen(0.0f,1.0f);
        for(unsigned int data_index = 0,tracks_count = each->get_visible_track_count();data_index < tracks_count;++data_index)
            if((skip_rate >= 1.0f || uniform_gen() <= skip_rate) && each->get_tract(data_index).size() > 3)
                visible.back().push_back(data_index);
    }

    // the occupied cells: each thread fills its own grid over every thread_count-th drawn tract, then the grids are merged
    std::vector<tipl::image<3,unsigned char> > occupied(std::min<int>(tipl::max_thread_count,8),tipl::image<3,unsigned char>(shade.shape()));
    tipl::par_for(occupied.size(),[&](size_t t)
    {
        for(size_t m = 0;m < models.size();++m)
            for(size_t i = t;i < visible[m].size();i += occupied.size())
            {
                auto& cur_tract = models[m]->get_tract(visible[m][i]);
                for(size_t index = 0;index < cur_tract.size();index += 3)
                {
                    int x = int(cur_tract[index]+0.5f) >> shift,y = int(cur_tract[index+1]+0.5f) >> shift,z = int(cur_tract[index+2]+0.5f) >> shift;
                    if(occupied[t].shape().is_valid(x,y,z))
                        occupied[t][size_t(x+((y+(z << 6)) << 6))] = 1;
                }
            }
    },occupied.size());
    for(const auto& each : occupied)
        for(size_t i = 0;i < shade.size();++i)
            if(each[i])
                shade[i] = 1.0f;
    for(int pass = 0;pass < 3;++pass)
        tipl::filter::mean(shade);
    float max_density = tipl::max_value(shade),scale = 24.0f/(max_density*max_density);
    for(auto& each : shade)
        each *= each*scale;
}
void TractRender::prepare_update(tracking_window& param,
                                 std::shared_ptr<TractModel>& active_tract_model,
                                 const TractRenderShader& shader,const std::vector<unsigned int>& visible)
{
    auto lock = start_reading(false);
    if(!lock.get())
        return;

    TractRenderSetting s(param);
    auto index_num = param.handle->get_name_index(
        param.renderWidget->getListValue("tract_color_metrics").toStdString());
    bool voxel_based_metric = index_num >= param.handle->dir.index_name_data.size();
    if(s.color_style >= 2 && s.color_style <= 4 && voxel_based_metric && index_num >= param.handle->slices.size()) // metric coloring
        return;

    std::vector<tipl::vector<3> > assigned_colors;
    // 0:Directional 1:Assigned 2:Local Index 3:Averaged Index 4:Max Index 5:Loaded
    if(s.color_style == 1 || // assigned
       s.color_style == 3 || // mean value
       s.color_style == 4 || // max value
       s.color_style == 5 )  // loaded value
    {
        assigned_colors.resize(visible.size());
        for (unsigned int i = 0;i < visible.size();++i)
        {
            if(s.color_style == 1) // assigned
            {
                tipl::rgb paint_color = active_tract_model->get_tract_color(visible[i]);
                assigned_colors[i] = tipl::vector<3,float>(paint_color.r,paint_color.g,paint_color.b);
                assigned_colors[i] /= 255.0f;
                continue;
            }
            if(s.color_style == 5) // loaded value
            {
                if(active_tract_model->loaded_values.empty())
                    assigned_colors[i] = i ? assigned_colors[0] : s.color_map.value2color(active_tract_model->loaded_value,s.color_min,s.color_r);
                else
                    if(visible[i] < active_tract_model->loaded_values.size())
                        assigned_colors[i] = s.color_map.value2color(active_tract_model->loaded_values[visible[i]],s.color_min,s.color_r);
                continue;
            }
            std::vector<float> metrics(active_tract_model->get_tract_data(param.handle,visible[i],index_num));
            assigned_colors[i] = s.color_map.value2color(s.color_style == 3 ? float(tipl::mean(metrics)) : tipl::max_value(metrics),s.color_min,s.color_r); // mean or max
        }
    }

    std::vector<TractRenderData> new_data(data_block_count);
    if(s.color_style == 2 && voxel_based_metric)
        param.handle->slices[index_num]->get_image();
    tipl::par_for(data_block_count,[&](unsigned int thread)
    {
        for(unsigned int i = thread;i < visible.size();i += data_block_count)
        {
            if(about_to_write)
                break;
            std::vector<float> metrics;
            if(s.color_style == 2) // local values
                metrics = active_tract_model->get_tract_data(param.handle,visible[i],index_num);
            new_data[thread].add_tract(s,active_tract_model->get_tract(visible[i]),shader,
                           assigned_colors.empty() ? tipl::vector<3>() : assigned_colors[i],metrics);
        }
    },4);

    data.swap(new_data);
    need_update = false;
    update_data_count = 0;
}
bool TractRender::render_tracts(size_t index,std::chrono::high_resolution_clock::time_point end_time)
{
    bool done = index >= data.size() || data[index].draw(end_time) || update_data_count >= index; // false: will emit update
    update_data_count = index;
    return done;
}
