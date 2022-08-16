#ifndef TRACT_RENDER_HPP
#define TRACT_RENDER_HPP
#include "TIPL/tipl.hpp"
#include "tract_model.hpp"
class tracking_window;
class GLWidget;
struct TractRenderParam{

    float tract_alpha = 0.0f;
    float tract_color_saturation = 0.0f;
    float tract_color_brightness = 0.0f;
    float tube_diameter = 0.0f;

    unsigned char tract_alpha_style = 0;
    unsigned char tract_style = 0;
    unsigned char tract_color_style = 0;
    unsigned char tract_tube_detail = 0;
    unsigned char tract_shader = 0;
    unsigned char end_point_shift = 0;


    float alpha;
    float tract_color_saturation_base;
    bool show_end_only;
    float tube_detail;
    float tract_shaderf;

    tipl::color_map color_map;
    double color_r,color_min;
    const tipl::vector<3,float>& get_color(double value) const
    {
        return color_map[uint32_t(std::floor(std::min(1.0,(std::max<double>(value-color_min,0.0))/color_r)*255.0+0.49))];
    }
    void init(GLWidget* glwidget,
              tracking_window& cur_tracking_window,
              bool simple);
};

struct TractRenderShader{
    tipl::shape<3> dim;
    tipl::image<2,float> max_z_map,min_z_map,max_x_map,min_x_map,min_y_map,max_y_map;
    TractRenderShader(tipl::shape<3> dim_):dim(dim_),
        min_x_map(tipl::shape<2>(64,64),float(dim.width())),
        min_y_map(tipl::shape<2>(64,64),float(dim.height())),
        min_z_map(tipl::shape<2>(64,64),float(dim.depth())),
        max_x_map(tipl::shape<2>(64,64)),
        max_y_map(tipl::shape<2>(64,64)),
        max_z_map(tipl::shape<2>(64,64)){}
    void add_shade(std::shared_ptr<TractModel>& active_tract_model,
                   const std::vector<unsigned int>& visible);
    float get_shade(const tipl::vector<3>& pos) const;
};

struct TractRenderData{
    std::vector<float> tube_vertices;
    std::vector<float> tube_normals;
    std::vector<float> tube_colors;
    std::vector<unsigned int> tube_strip_pos;
    std::vector<float> line_vertices;
    std::vector<float> line_colors;
    std::vector<unsigned int> line_strip_pos;
    void clear(void)
    {
        tube_vertices.clear();
        tube_normals.clear();
        tube_colors.clear();
        line_vertices.clear();
        line_colors.clear();
        tube_strip_pos.clear();
        line_strip_pos.clear();
        tube_strip_pos.push_back(0);
        line_strip_pos.push_back(0);
    }
    inline void add_tube(const tipl::vector<3>& v,const tipl::vector<3>& c,const tipl::vector<3>& n)
    {
        tube_vertices.push_back(v[0]);
        tube_vertices.push_back(v[1]);
        tube_vertices.push_back(v[2]);
        tube_normals.push_back(n[0]);
        tube_normals.push_back(n[1]);
        tube_normals.push_back(n[2]);
        tube_colors.push_back(c[0]);
        tube_colors.push_back(c[1]);
        tube_colors.push_back(c[2]);
    }
    inline void add_line(const tipl::vector<3>& v,const tipl::vector<3>& c)
    {
        line_vertices.push_back(v[0]);
        line_vertices.push_back(v[1]);
        line_vertices.push_back(v[2]);
        line_colors.push_back(c[0]);
        line_colors.push_back(c[1]);
        line_colors.push_back(c[2]);
    }
    void end_tube_strip(void)
    {
        tube_strip_pos.push_back(tube_vertices.size());
    }
    void end_line_strip(void)
    {
        line_strip_pos.push_back(line_vertices.size());
    }
    void add_tract(const TractRenderParam& param,
                   const std::vector<float>& tract,bool simple,
                   const TractRenderShader& shader,
                   const tipl::vector<3>& assign_color,
                   const std::vector<float>& metrics);
    void create_list(bool tube);
};

struct TractRender{
public:
    TractRenderParam param;
    unsigned int tracts = 0;
public:
    bool need_update = true;
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
    auto start_writing(void)
    {
        std::lock_guard<std::mutex> lock(writing_lock);
        while(reading_threads)
        {
            about_to_write = true;
            std::this_thread::yield();
        }
        writing = true;
        about_to_write = false;
        return std::make_shared<end_writing>(*this);
    }
    TractRender(void);
    ~TractRender(void);
    void render_tracts(std::shared_ptr<TractModel>& tract_data,GLWidget* glwidget,tracking_window& cur_tracking_window,bool simple);
};
#endif // TRACT_RENDER_HPP
