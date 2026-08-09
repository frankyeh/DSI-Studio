#ifndef IMG_HPP
#define IMG_HPP

#include "zlib.h"
#include "TIPL/tipl.hpp"

class variant_image{
public:
    constexpr static size_t pixelbit[4] = {1,2,4,4};

    tipl::image<3,unsigned char,tipl::buffer_container> I_int8;
    tipl::image<3,unsigned short,tipl::buffer_container> I_int16;
    tipl::image<3,unsigned int,tipl::buffer_container> I_int32;
    tipl::image<3,float,tipl::buffer_container> I_float32;

    enum {int8 = 0,int16 = 1,int32 = 2,float32 = 3} pixel_type = int8;

    tipl::shape<3> shape;
    size_t dim4 = 1;
    bool is_mni = false;
    bool interpolation = true;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4> T;
    std::string error_msg,info;
private:
    tipl::reg::mm_reg<tipl::out> r;
public:
    variant_image(void) = default;
    variant_image(const variant_image&) = default;
    variant_image& operator=(const variant_image&) = default;
    variant_image(variant_image&&) noexcept = default;
    variant_image& operator=(variant_image&&) noexcept = default;

    template<typename U>
    auto bind(const U& I) const
    {
        return std::tie(vs,T,is_mni,I);
    }

    template<typename U>
    auto bind(U& I)
    {
        return std::tie(vs,T,is_mni,I);
    }

    template<typename Fun,typename Arg>
    bool call_function(Fun&& fun,Arg&& arg)
    {
        if constexpr(std::is_void_v<decltype(fun(std::forward<Arg>(arg)))>)
        {
            fun(std::forward<Arg>(arg));
            return true;
        }
        else
            return fun(std::forward<Arg>(arg));
    }

    template<typename T>
    bool apply(T&& fun)
    {
        switch(pixel_type)
        {
        case int8:    return call_function(fun,I_int8);
        case int16:   return call_function(fun,I_int16);
        case int32:   return call_function(fun,I_int32);
        case float32: return call_function(fun,I_float32);
        }
        return false;
    }

    bool read_mat_image(size_t index,
                        tipl::io::gz_mat_read& mat);

    void write_mat_image(size_t index,
                         tipl::io::gz_mat_read& mat);

    void change_type(decltype(pixel_type));

    bool command(std::string cmd,std::string param1);

    bool load_from_file(const std::filesystem::path& file_name);

    bool empty(void) const
    {
        return shape.size() == 0;
    }

    size_t buf_size(void) const
    {
        return shape.size()*pixelbit[pixel_type];
    }
};

// compact, agent-facing summary of a preview capture: cheap, unambiguous facts computed once at a
// fixed internal resolution (independent of whatever display resolution render_art/render_occupancy
// are asked for), meant to be read alongside -- not instead of -- the rendered grids themselves
struct PreviewStats
{
    double coverage = 0.0;                     // fraction of cells classified as foreground
    double x0 = 0.0,y0 = 0.0,x1 = 0.0,y1 = 0.0; // normalized bounding box of foreground, 0..1
    double left_ratio = 0.0,right_ratio = 0.0;  // fraction of foreground cells in the left/right half
    double mean_lum = 0.0,min_lum = 255.0,max_lum = 0.0; // raw-pixel luminance, not downsampled
    int blobs = 0;                             // 4-connected foreground clusters
};

// takes a width, height, and a pixel accessor (int x,int y)->double grayscale 0-255 -- templated on
// the accessor rather than any concrete image type, so this header stays independent of Qt/QImage or
// any other image library; the caller (which already has the real image type and knows how to read a
// grayscale value from it) supplies the accessor as a lambda. Offers two renderings of the same
// underlying data at different sizes -- both digit grids, not character-density art. Symbol ramps
// (".:-=+*#%@") are a human-vision hack: the glyphs are chosen so their *rendered ink density*
// approximates grayscale to an eye, which is meaningless to a model reading tokens, not pixels, and
// risks a run of symbols merging into an unrelated learned token ("###", "===", "***" all mean
// something else). Digits are tokenized predictably (a lone digit between spaces is essentially
// always one token) and carry intrinsic ordinal meaning with no learned convention required. Grids
// are also kept small and row-labeled, since reconstructing 2D structure from flattened text is a
// known weak spot for attention -- the model has to count characters/newlines to track position,
// which degrades over long spans regardless of what symbol set is used; more resolution isn't usable.
template<typename pixel_fun_type>
class TextPreview
{
    int w = 0,h = 0;
    pixel_fun_type get_pixel; // (int x,int y) -> double, grayscale 0..255
    double bg = 0.0;
    static constexpr double fg_threshold = 20.0; // luminance levels, out of 255
public:
    PreviewStats stats;
public:
    // bg is the background luminance, supplied by the caller rather than sampled from this image's
    // own corners: on a zoomed crop, the crop's corners can land anywhere in the source (e.g. the
    // center), which is not necessarily background, so the caller passes the value measured once on
    // the full, un-cropped capture
    TextPreview(int width,int height,pixel_fun_type pixel_fun,double bg_):
        w(width),h(height),get_pixel(pixel_fun),bg(bg_)
    {
        if(w <= 0 || h <= 0)
            return;
        compute_stats();
    }
    bool empty(void) const { return w <= 0 || h <= 0; }
private:
    // downsamples the source into a cols x rows grid, averaging per-pixel `value(x,y)` into each
    // cell -- shared by render_grid (raw luminance, for display) and compute_stats (foreground
    // coverage, for bbox/blob detection) so both use the same grid geometry
    template<typename value_fun_type>
    void downsample(int cols,int rows,std::vector<double>& cell,value_fun_type&& value) const
    {
        cell.assign(size_t(cols)*size_t(rows),0.0);
        std::vector<int> count(cell.size(),0);
        for(int y = 0,ry = 0;y < h;++y,ry += rows)
        {
            int cy = std::min(rows-1,ry/h);
            size_t row_offset = size_t(cy)*size_t(cols);
            for(int x = 0,rx = 0;x < w;++x,rx += cols)
            {
                int cx = std::min(cols-1,rx/w);
                size_t i = row_offset+size_t(cx);
                cell[i] += value(x,y);
                ++count[i];
            }
        }
        for(size_t i = 0;i < cell.size();++i)
            if(count[i])
                cell[i] /= count[i];
    }
    // fills `stats` from a fixed internal resolution, independent of any display choice
    void compute_stats(void)
    {
        constexpr int n = 32;
        int rows = std::max(1,int(double(n)*h/w));
        std::vector<double> cover;
        downsample(n,rows,cover,[&](int x,int y){ return std::fabs(get_pixel(x,y)-bg) > fg_threshold ? 1.0 : 0.0; });

        double mean_lum = 0.0,min_lum = 255.0,max_lum = 0.0;
        for(int y = 0;y < h;++y)
            for(int x = 0;x < w;++x)
            {
                double v = get_pixel(x,y);
                mean_lum += v;
                min_lum = std::min(min_lum,v);
                max_lum = std::max(max_lum,v);
            }
        mean_lum /= double(w)*double(h);

        std::vector<char> fg(cover.size(),0);
        int fg_count = 0,min_cx = n,max_cx = -1,min_cy = rows,max_cy = -1,left_count = 0,right_count = 0;
        int half = n/2;
        for(int y = 0,i = 0;y < rows;++y)
            for(int x = 0;x < n;++x,++i)
            {
                if(cover[i] <= 0.5) // majority of the cell must be foreground to count for bbox/blobs
                    continue;
                fg[i] = 1;
                ++fg_count;
                min_cx = std::min(min_cx,x); max_cx = std::max(max_cx,x);
                min_cy = std::min(min_cy,y); max_cy = std::max(max_cy,y);
                (x < half ? left_count : right_count)++;
            }

        stats.coverage = double(fg_count)/double(n*rows);
        if(fg_count)
        {
            stats.x0 = double(min_cx)/n;   stats.y0 = double(min_cy)/rows;
            stats.x1 = double(max_cx+1)/n; stats.y1 = double(max_cy+1)/rows;
            stats.left_ratio = double(left_count)/fg_count;
            stats.right_ratio = double(right_count)/fg_count;
        }
        stats.mean_lum = mean_lum;
        stats.min_lum = min_lum;
        stats.max_lum = max_lum;

        // 4-connected flood fill to count distinct visible clusters
        std::vector<char> visited(fg.size(),0);
        std::vector<size_t> stack;
        for(size_t i = 0;i < fg.size();++i)
        {
            if(!fg[i] || visited[i])
                continue;
            ++stats.blobs;
            stack.push_back(i);
            visited[i] = 1;
            while(!stack.empty())
            {
                size_t cur = stack.back();
                stack.pop_back();
                int cx = int(cur % size_t(n)),cy = int(cur / size_t(n));
                const int dx[4] = {-1,1,0,0},dy[4] = {0,0,-1,1};
                for(int d = 0;d < 4;++d)
                {
                    int nx = cx+dx[d],ny = cy+dy[d];
                    if(nx < 0 || nx >= n || ny < 0 || ny >= rows)
                        continue;
                    size_t ni = size_t(ny)*size_t(n)+size_t(nx);
                    if(fg[ni] && !visited[ni])
                    {
                        visited[ni] = 1;
                        stack.push_back(ni);
                    }
                }
            }
        }
    }
    // shared renderer: a cols x rows digit grid, 0 = black (luminance 0) to 9 = white (luminance
    // 255) -- a direct, absolute mapping so the same digit always means the same brightness
    // regardless of what background/foreground split applies elsewhere (e.g. `stats`), and so a
    // zoomed-in crop reads consistently with the full view it was cropped from. Every digit,
    // including the first in a row, is preceded by exactly one space -- a uniform " D" pattern per
    // cell rather than "D "/"D" for the last one, since tokenizers commonly treat "space+digit" as
    // one dedicated token; a consistent leading space gives every cell the same shot at landing as
    // exactly one token, with no special-cased first column. Each row is prefixed with its index so
    // the model doesn't have to count newlines to know which row it's reading.
    inline std::string render_grid(int cols,int rows) const
    {
        if(empty())
            return "(empty image)";
        std::vector<double> lum;
        downsample(cols,rows,lum,[&](int x,int y){ return get_pixel(x,y); });
        int row_label_width = int(std::to_string(rows-1).size());
        std::ostringstream out;
        for(int y = 0,i = 0;y < rows;++y)
        {
            out << std::string(size_t(row_label_width-int(std::to_string(y).size())),' ') << y << ":";
            for(int x = 0;x < cols;++x,++i)
                out << ' ' << int(std::min(9.0,lum[i]*10.0/255.0));
            out << '\n';
        }
        return out.str();
    }
public:
    // digit luminance grid at ~cols wide, rows chosen from the image's true aspect ratio (no
    // monospace-font correction -- this is coordinate data meant to line up with the normalized
    // bbox in `stats`, not a picture meant to be visually squinted at). Columns have no index label
    // (unlike rows), so they're the more fragile axis for position-tracking; capped at 20 rather than
    // higher, since text-only grid-reasoning benchmarks (e.g. ARC-style tasks) already show visibly
    // degraded position accuracy by roughly 16-20 cells per unlabeled axis.
    inline std::string render_art(int cols) const
    {
        cols = std::max(6,std::min(cols,20));
        int rows = empty() ? 1 : std::max(1,int(double(cols)*h/w));
        return render_grid(cols,rows);
    }
    // fixed n x n luminance grid (default 8x8): same semantics as render_art, just a small size
    // that's always cheap to include regardless of what render_art is asked for
    inline std::string render_occupancy(int n = 8) const
    {
        n = std::max(2,std::min(n,12));
        return render_grid(n,n);
    }
    inline std::string format_stats(void) const
    {
        std::ostringstream out;
        out << "coverage=" << int(stats.coverage*100) << "%"
            << " bbox=(" << stats.x0 << "," << stats.y0 << ")-(" << stats.x1 << "," << stats.y1 << ")"
            << " left/right=" << int(stats.left_ratio*100) << "%/" << int(stats.right_ratio*100) << "%"
            << " blobs=" << stats.blobs
            << " luminance(min/mean/max)=" << int(stats.min_lum) << "/" << int(stats.mean_lum) << "/" << int(stats.max_lum);
        return out.str();
    }
};

#endif // IMG_HPP
