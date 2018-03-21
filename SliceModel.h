#ifndef SliceModelH
#define SliceModelH
#include <future>
#include "image/image.hpp"
#include "libs/gzip_interface.hpp"

// ---------------------------------------------------------------------------
struct SliceModel {
public:
        bool is_diffusion_space = true;
        image::matrix<4,4,float> transform,invT;
        image::geometry<3>geometry;
        image::vector<3,float>voxel_size;
        SliceModel(void);
public:
        virtual std::pair<float,float> get_value_range(void) const = 0;
        virtual std::pair<float,float> get_contrast_range(void) const = 0;
        virtual std::pair<unsigned int,unsigned int> get_contrast_color(void) const = 0;
        virtual void set_contrast_range(float min_v,float max_v) = 0;
        virtual void set_contrast_color(unsigned int min_c,unsigned int max_c) = 0;
        virtual void get_slice(image::color_image& image,
                               unsigned char,const image::value_to_color<float>& v2c,
                               const SliceModel* overlay,
                               const image::value_to_color<float>& overlay_v2c) const = 0;
        virtual image::const_pointer_image<float, 3> get_source(void) const = 0;

public:

        template<typename value_type1,typename value_type2>
        void toDiffusionSpace(unsigned char cur_dim,value_type1 x, value_type1 y,
                              value_type2& px, value_type2& py, value_type2& pz) const
        {
            if(!is_diffusion_space)
            {
                image::vector<3,float> v;
                image::slice2space(cur_dim, x, y, slice_pos[cur_dim], v[0],v[1],v[2]);
                v.to(transform);
                v.round();
                px = v[0];
                py = v[1];
                pz = v[2];
            }
            else
                image::slice2space(cur_dim, x, y, slice_pos[cur_dim], px, py, pz);
        }
        void toOtherSlice(const SliceModel* other_slice,
                          unsigned char cur_dim,float x,float y,
                          image::vector<3,float>& v) const
        {
            image::slice2space(cur_dim, x, y, slice_pos[cur_dim], v[0],v[1],v[2]);
            if(!is_diffusion_space)
                v.to(transform);
            if(!other_slice->is_diffusion_space)
                v.to(other_slice->invT);
        }
        template<typename value_type>
        bool to3DSpace(unsigned char cur_dim,value_type x, value_type y,
                       value_type& px, value_type& py, value_type& pz) const
        {
            image::slice2space(cur_dim, x, y, slice_pos[cur_dim], px, py, pz);
            return geometry.is_valid(px, py, pz);
        }

public:
        // for directx
        image::vector<3,int> slice_pos;
        bool slice_visible[3];
        void get_texture(unsigned char dim,image::color_image& cur_rendering_image,
                         const image::value_to_color<float>& v2c,
                         const SliceModel* overlay,
                         const image::value_to_color<float>& overlay_v2c)
        {
            get_slice(cur_rendering_image,dim,v2c,overlay,overlay_v2c);
            for(unsigned int index = 0;index < cur_rendering_image.size();++index)
            {
                unsigned char value =
                255-cur_rendering_image[index].data[0];
                if(value >= 230)
                    value -= (value-230)*10;
                cur_rendering_image[index].data[3] = value;
            }
        }

        void get_other_slice_pos(unsigned char cur_dim,int& x, int& y) const {
                x = slice_pos[(cur_dim + 1) % 3];
                y = slice_pos[(cur_dim + 2) % 3];
                if (cur_dim == 1)
                        std::swap(x, y);
        }
        bool set_slice_pos(int x,int y,int z)
        {
            if(!geometry.is_valid(x,y,z))
                return false;
            bool has_updated = false;
            if(slice_pos[0] != x)
            {
                slice_pos[0] = x;
                has_updated = true;
            }
            if(slice_pos[1] != y)
            {
                slice_pos[1] = y;
                has_updated = true;
            }
            if(slice_pos[2] != z)
            {
                slice_pos[2] = z;
                has_updated = true;
            }
            return has_updated;
        }
        void get_slice_positions(unsigned int dim,std::vector<image::vector<3,float> >& points)
        {
            points.resize(4);
            image::get_slice_positions(dim, slice_pos[dim], geometry,points);
            if(!is_diffusion_space)
            for(unsigned int index = 0;index < points.size();++index)
                points[index].to(transform);
        }
        void get_mosaic(image::color_image& image,
                        unsigned int mosaic_size,
                        const image::value_to_color<float>& v2c,
                        unsigned int skip,
                        const SliceModel* overlay,
                        const image::value_to_color<float>& overlay_v2c);
        void apply_overlay(image::color_image& show_image,
                           unsigned char dim,
                           const SliceModel* other_slice,
                           const image::value_to_color<float>& overlay_v2c) const;
};

class fib_data;
class FibSliceModel : public SliceModel{
public:
    std::shared_ptr<fib_data> handle;
    int view_id;
public:
    FibSliceModel(std::shared_ptr<fib_data> new_handle,int view_id_);
public:
    std::pair<float,float> get_value_range(void) const;
    std::pair<float,float> get_contrast_range(void) const;
    std::pair<unsigned int,unsigned int> get_contrast_color(void) const;
    void set_contrast_range(float min_v,float max_v);
    void set_contrast_color(unsigned int min_c,unsigned int max_c);


    void get_slice(image::color_image& image,unsigned char cur_dim,
                   const image::value_to_color<float>& v2c,
                   const SliceModel* overlay,
                   const image::value_to_color<float>& overlay_v2c) const;
    image::const_pointer_image<float, 3> get_source(void) const;

};

class CustomSliceModel : public SliceModel{
public:
    std::string name;
public:
    std::auto_ptr<std::future<void> > thread;

    image::const_pointer_image<float,3> from;
    image::vector<3> from_vs;
    image::basic_image<float,3> source_buf;
    image::vector<3> source_buf_vs;
    float size_ratio = 1.0f;
    image::affine_transform<double> arg_min;
    bool terminated;
    bool ended;
    ~CustomSliceModel(void)
    {
        terminate();
    }

    void terminate(void);
    void argmin(image::reg::reg_type reg_type);
    void update(void);

public:
    image::basic_image<float, 3> source_images;
    float min_value,max_value,scale;
public:
    float contrast_max,contrast_min;
    unsigned int max_color = 0x00FFFFFF;
    unsigned int min_color = 0;
public:
    template<class loader>
    void load(const loader& io)
    {
        io.get_voxel_size(voxel_size.begin());
        io >> source_images;
        init();
    }
    template<class loader>
    void loadLPS(loader& io)
    {
        io.get_voxel_size(voxel_size.begin());
        io.toLPS(source_images);
        init();
    }
    void init(void);
    bool initialize(std::shared_ptr<fib_data> handle,bool is_qsdr,const std::vector<std::string>& files,bool correct_intensity);
public:
    std::pair<float,float> get_value_range(void) const;
    std::pair<float,float> get_contrast_range(void) const;
    std::pair<unsigned int,unsigned int> get_contrast_color(void) const;
    void set_contrast_range(float min_v,float max_v);
    void set_contrast_color(unsigned int min_c,unsigned int max_c);

    void get_slice(image::color_image& image,unsigned char cur_dim,
                   const image::value_to_color<float>& v2c,
                   const SliceModel* overlay,
                   const image::value_to_color<float>& overlay_v2c) const;
    bool stripskull(float qa_threshold);
    image::const_pointer_image<float, 3> get_source(void) const  {return source_images;}
};

#endif
