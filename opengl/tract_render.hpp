#ifndef TRACT_RENDER_HPP
#define TRACT_RENDER_HPP
#include <QtOpenGL>
#include "tract_model.hpp"
class tracking_window;
class GLWidget;

struct TractRenderShader{
    tipl::shape<3> dim;
    tipl::vector<3> to64;
    tipl::image<2,float> max_z_map,min_z_map,max_x_map,min_x_map,min_y_map,max_y_map;
    float skip_rate = 0.0f;
    TractRenderShader(tracking_window& cur_tracking_window);

    float get_shade(const tipl::vector<3>& pos) const;
};

class TractRenderData{
    std::vector<float> tube_vertices;
    size_t tube_vertices_count = 0;
    std::vector<float> line_vertices;
    size_t line_vertices_count = 0;
    std::vector<GLint> tube_strip_pos;
    std::vector<GLint> line_strip_pos;
    std::vector<GLsizei> tube_strip_size;
    std::vector<GLsizei> line_strip_size;
public:
    bool draw(GLWidget* glwidget,std::chrono::high_resolution_clock::time_point end_time);
public:
    TractRenderData(void)
    {
        tube_strip_pos.push_back(0);
        line_strip_pos.push_back(0);
    }
public:
    void clear(void);
    void add_tube(const tipl::vector<3>& v,const tipl::vector<3>& c,const tipl::vector<3>& n,float alpha)
    {
        tube_vertices.push_back(v[0]);
        tube_vertices.push_back(v[1]);
        tube_vertices.push_back(v[2]);
        tube_vertices.push_back(n[0]);
        tube_vertices.push_back(n[1]);
        tube_vertices.push_back(n[2]);
        tube_vertices.push_back(c[0]);
        tube_vertices.push_back(c[1]);
        tube_vertices.push_back(c[2]);
        tube_vertices.push_back(alpha);
        ++tube_vertices_count;
    }
    void add_line(const tipl::vector<3>& v,const tipl::vector<3>& c,float alpha)
    {
        line_vertices.push_back(v[0]);
        line_vertices.push_back(v[1]);
        line_vertices.push_back(v[2]);
        line_vertices.push_back(c[0]);
        line_vertices.push_back(c[1]);
        line_vertices.push_back(c[2]);
        line_vertices.push_back(alpha);
        ++line_vertices_count;
    }
    inline void end_tube_strip(void)
    {
        tube_strip_size.push_back(tube_vertices_count-tube_strip_pos.back());
        tube_strip_pos.push_back(tube_vertices_count);
    }
    inline void end_line_strip(void)
    {
        line_strip_size.push_back(line_vertices_count-line_strip_pos.back());
        line_strip_pos.push_back(line_vertices_count);
    }
    void add_tract(const tracking_window& param,
                   const std::vector<float>& tract,
                   const TractRenderShader& shader,
                   const tipl::vector<3>& assign_color,
                   const std::vector<float>& metrics);
};

struct TractRender{
public:
    std::vector<TractRenderData> data;
    static const size_t data_block_count = 16;
public:
    bool terminated = false;
    std::shared_ptr<std::thread> calculation_thread;
public:
    bool need_update = true;
    unsigned int update_data_count = 0;
    bool about_to_write = false;
    unsigned int reading_threads = 0;
    bool writing = false;
    std::mutex writing_lock;
    std::mutex reading_lock;    
public:
    struct end_reading
    {
        TractRender& host;
        end_reading(TractRender& host_):host(host_){}
        ~end_reading(void)
        {
            std::lock_guard<std::mutex> lock(host.reading_lock);
            --host.reading_threads;
        }
    };
    auto start_reading(bool wait = true)
    {
        while(writing || about_to_write)
        {
            if(!wait)
                return std::shared_ptr<end_reading>();
            std::this_thread::yield();
        }
        std::lock_guard<std::mutex> lock(reading_lock);
        ++reading_threads;
        return std::make_shared<end_reading>(*this);
    }
    struct end_writing
    {
        TractRender& host;
        end_writing(TractRender& host_):host(host_){}
        ~end_writing(void)
        {
            std::lock_guard<std::mutex> lock(host.writing_lock);
            host.writing = false;
        }
    };
    auto start_writing(bool wait = true)
    {
        std::lock_guard<std::mutex> lock(writing_lock);
        while(reading_threads)
        {
            if(!wait)
                return std::shared_ptr<end_writing>();
            about_to_write = true;
            std::this_thread::yield();
        }
        writing = true;
        about_to_write = false;
        return std::make_shared<end_writing>(*this);
    }
    TractRender(void);
    ~TractRender(void);
    void prepare_update(tracking_window& param,
                        std::shared_ptr<TractModel>& active_tract_model,
                        const TractRenderShader& shader);
    bool render_tracts(size_t index,GLWidget* glwidget,std::chrono::high_resolution_clock::time_point end_time);
};
#endif // TRACT_RENDER_HPP
