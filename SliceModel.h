#ifndef SliceModelH
#define SliceModelH
#include <future>
#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"

// ---------------------------------------------------------------------------
class fib_data;
class SliceModel {
public:
    std::shared_ptr<fib_data> handle;
    int view_id;
    bool is_diffusion_space = true;
    tipl::matrix<4,4,float> T,invT; // T: image->diffusion iT: diffusion->image
    tipl::geometry<3>geometry;
    tipl::vector<3,float> voxel_size;
public:
    // for directx
    tipl::vector<3,int> slice_pos;
    bool slice_visible[3];
public:
    SliceModel(std::shared_ptr<fib_data> new_handle,int view_id_);
    virtual ~SliceModel(void){}
public:

    std::pair<float,float> get_value_range(void) const;
    std::pair<float,float> get_contrast_range(void) const;
    std::pair<unsigned int,unsigned int> get_contrast_color(void) const;
    void set_contrast_range(float min_v,float max_v);
    void set_contrast_color(unsigned int min_c,unsigned int max_c);
    void get_slice(tipl::color_image& image,
                           unsigned char,const tipl::value_to_color<float>& v2c,
                           const SliceModel* overlay,
                           const tipl::value_to_color<float>& overlay_v2c) const;
    tipl::const_pointer_image<float, 3> get_source(void) const;

public:
    template<typename value_type1,typename value_type2>
    void toDiffusionSpace(unsigned char cur_dim,value_type1 x, value_type1 y,
                          value_type2& px, value_type2& py, value_type2& pz) const
    {
        if(!is_diffusion_space)
        {
            tipl::vector<3,float> v;
            tipl::slice2space(cur_dim, x, y, slice_pos[cur_dim], v[0],v[1],v[2]);
            v.to(T);
            v.round();
            px = v[0];
            py = v[1];
            pz = v[2];
        }
        else
            tipl::slice2space(cur_dim, x, y, slice_pos[cur_dim], px, py, pz);
    }
    void toOtherSlice(const SliceModel* other_slice,
                      unsigned char cur_dim,float x,float y,
                      tipl::vector<3,float>& v) const
    {
        tipl::slice2space(cur_dim, x, y, slice_pos[cur_dim], v[0],v[1],v[2]);
        if(!is_diffusion_space)
            v.to(T);
        if(!other_slice->is_diffusion_space)
            v.to(other_slice->invT);
    }
    template<typename value_type>
    bool to3DSpace(unsigned char cur_dim,value_type x, value_type y,
                   value_type& px, value_type& py, value_type& pz) const
    {
        tipl::slice2space(cur_dim, x, y, slice_pos[cur_dim], px, py, pz);
        return geometry.is_valid(px, py, pz);
    }


public:
    void get_texture(unsigned char dim,tipl::color_image& cur_rendering_image,
                     const tipl::value_to_color<float>& v2c,
                     const SliceModel* overlay,
                     const tipl::value_to_color<float>& overlay_v2c)
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
    void get_slice_positions(unsigned int dim,std::vector<tipl::vector<3,float> >& points)
    {
        points.resize(4);
        tipl::get_slice_positions(dim, slice_pos[dim], geometry,points);
        if(!is_diffusion_space)
        for(unsigned int index = 0;index < points.size();++index)
            points[index].to(T);
    }
    void get_mosaic(tipl::color_image& image,
                    unsigned int mosaic_size,
                    const tipl::value_to_color<float>& v2c,
                    unsigned int skip,
                    const SliceModel* overlay,
                    const tipl::value_to_color<float>& overlay_v2c);
    void apply_overlay(tipl::color_image& show_image,
                       unsigned char dim,
                       const SliceModel* other_slice,
                       const tipl::value_to_color<float>& overlay_v2c) const;
};

class CustomSliceModel : public SliceModel {
public:
    std::string name,error_msg = "unknown error";
public:
    std::auto_ptr<std::future<void> > thread;

    tipl::const_pointer_image<float,3> from;
    tipl::vector<3> from_vs;
    tipl::affine_transform<double> arg_min;
    bool terminated;
    bool ended;
    CustomSliceModel(std::shared_ptr<fib_data> new_handle);
    ~CustomSliceModel(void)
    {
        terminate();
    }

    void terminate(void);
    void argmin(tipl::reg::reg_type reg_type);
    void update(void);

public:
    tipl::image<float, 3> source_images;
public:

    bool initialize(const std::vector<std::string>& files,bool correct_intensity);
public:
    bool stripskull(void);
};

#endif
