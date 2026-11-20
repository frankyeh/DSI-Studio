#ifndef TRACT_RENDER_HPP
#define TRACT_RENDER_HPP
#include <QtOpenGL>
#include "tract_model.hpp"
class tracking_window;
class GLWidget;

struct TractRenderShader{
    float to_cell = 1.0f; // voxel to cell coordinates
    tipl::image<3> shade; // the shading of each cell of a 64^3 grid
    std::vector<std::vector<unsigned int> > visible; // the drawn tracts of each bundle
    TractRenderShader(tracking_window& cur_tracking_window,const std::vector<std::shared_ptr<TractModel> >& models);

    float get_shade(const tipl::vector<3>& pos) const;
};

struct TractRenderSetting{ // the rendering settings, read once per update
    float alpha,saturation,saturation_base,diameter,shader,color_min,color_r,detail;
    int style,color_style,end_point_shift;
    const tipl::color_map& color_map;
    TractRenderSetting(const tracking_window& param);
};

class TractRenderData{ // a block of tract meshes: tubes (vertex, normal, color) or lines (vertex, color) as 3+3+4 or 3+4 floats
    std::vector<float> vertices; // the mesh, moved to the gpu buffer at the first draw
    size_t vertices_count = 0;
    bool tube = false;
    QOpenGLBuffer buffer;
    std::vector<GLint> strip_pos = {0};
    std::vector<GLsizei> strip_size;
public:
    bool draw(std::chrono::high_resolution_clock::time_point end_time);
    void add_tube(const tipl::vector<3>& v,const tipl::vector<3>& c,const tipl::vector<3>& n,float alpha)
    {
        vertices.insert(vertices.end(),{v[0],v[1],v[2],n[0],n[1],n[2],c[0],c[1],c[2],alpha});
        ++vertices_count;
        tube = true;
    }
    void add_line(const tipl::vector<3>& v,const tipl::vector<3>& c,float alpha)
    {
        vertices.insert(vertices.end(),{v[0],v[1],v[2],c[0],c[1],c[2],alpha});
        ++vertices_count;
    }
    void end_strip(void)
    {
        strip_size.push_back(vertices_count-strip_pos.back());
        strip_pos.push_back(vertices_count);
    }
    void add_tract(const TractRenderSetting& s,
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
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        writing = true;
        about_to_write = false;
        return std::make_shared<end_writing>(*this);
    }
    ~TractRender(void);
    void prepare_update(tracking_window& param,
                        std::shared_ptr<TractModel>& active_tract_model,
                        const TractRenderShader& shader,const std::vector<unsigned int>& visible);
    bool render_tracts(size_t index,std::chrono::high_resolution_clock::time_point end_time);
};
#endif // TRACT_RENDER_HPP
