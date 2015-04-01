// ---------------------------------------------------------------------------

#ifndef SliceModelH
#define SliceModelH
#include <boost/lambda/lambda.hpp>
#include <boost/thread/thread.hpp>
#include "image/image.hpp"
#include "libs/gzip_interface.hpp"

// ---------------------------------------------------------------------------
struct SliceModel {
public:
        image::geometry<3>geometry;
        image::vector<3,float>voxel_size;
        image::vector<3,float>center_point;
        SliceModel(void);
public:
        virtual float get_value_range(void) const = 0;
        virtual void get_slice(image::color_image& image,float contrast,float offset) const = 0;
        virtual image::const_pointer_image<float, 3> get_source(void) const = 0;
public:
public:
        template<typename input_type1,typename input_type2>
        bool get3dPosition(input_type1 x, input_type1 y, input_type2& px, input_type2& py, input_type2& pz) const {
                image::slice2space(cur_dim, x, y, slice_pos[cur_dim], px, py, pz);
                return geometry.is_valid(px, py, pz);
        }

        template<typename input_type1,typename input_type2>
        void getSlicePosition(input_type1 px, input_type1 py, input_type1 pz, input_type2& x, input_type2& y,
                input_type2& slice_index) const {
                image::space2slice(cur_dim, px, py, pz, x, y, slice_index);
        }

        template<typename input_type>
        unsigned int get3dPosition(input_type x, input_type y) const {
                int px, py, pz;
                if (!get3dPosition(x, y, px, py, pz))
                        return 0;
                return image::pixel_index<3>(px, py, pz, geometry).index();
        }
public:
        // for directx
        unsigned char cur_dim;
        int slice_pos[3];
        bool slice_visible[3];
        bool texture_need_update[3];
        void get_texture(unsigned char dim,image::color_image& cur_rendering_image,float contrast,float offset)
        {
            unsigned char cur_dim_backup = cur_dim;
            cur_dim = dim;
            get_slice(cur_rendering_image,contrast,offset);
            cur_dim = cur_dim_backup;
            for(unsigned int index = 0;index < cur_rendering_image.size();++index)
            {
                unsigned char value =
                255-cur_rendering_image[index].data[0];
                if(value >= 230)
                    value -= (value-230)*10;
                cur_rendering_image[index].data[3] = value;
            }
            texture_need_update[dim] = false;
        }
        void get_slice_positions(unsigned int dim,std::vector<image::vector<3,float> >& points)
        {
            points.resize(4);
            image::get_slice_positions(dim, slice_pos[dim], geometry,points);
        }
        void get_other_slice_pos(int& x, int& y) const {
                x = slice_pos[(cur_dim + 1) % 3];
                y = slice_pos[(cur_dim + 2) % 3];
                if (cur_dim == 1)
                        std::swap(x, y);
        }
        bool set_slice_pos(int x,int y,int z)
        {
            bool has_updated = false;
            if(slice_pos[0] != x)
            {
                slice_pos[0] = x;
                texture_need_update[0] = has_updated = true;
            }
            if(slice_pos[1] != y)
            {
                slice_pos[1] = y;
                texture_need_update[1] = has_updated = true;
            }
            if(slice_pos[2] != z)
            {
                slice_pos[2] = z;
                texture_need_update[2] = has_updated = true;
            }
            return has_updated;
        }
};

class FibData;
class FibSliceModel : public SliceModel{
public:
    FibData* handle;
    image::const_pointer_image<float,3> source_images;
    std::string view_name,overlay_name;
public:
    FibSliceModel(FibData* new_handle);
    void set_view_name(const std::string& view_name_,const std::string& overlay_name_)
    {
        view_name = view_name_;
        overlay_name = overlay_name_;
    }
public:
    float get_value_range(void) const;
    void get_slice(image::color_image& image,float contrast,float offset) const;
    image::const_pointer_image<float, 3> get_source(void) const{return source_images;}
    void get_mosaic(image::color_image& image,
                    unsigned int mosaic_size,
                    float contrast,
                    float offset,
                    unsigned int skip) const;
};

class CustomSliceModel : public SliceModel{
public:
    std::vector<float> transform,invT;
    image::basic_image<float,3> roi_image;
    float* roi_image_buf;
public:
    std::auto_ptr<boost::thread> thread;
    image::const_pointer_image<float,3> from;
    image::affine_transform<3> arg_min;
    bool terminated;
    bool ended;
    ~CustomSliceModel(void)
    {
        terminate();
    }

    void terminate(void);
    void argmin(int reg_type);
    void update(void);
    void update_roi(void);

public:
    image::basic_image<float, 3> source_images;
    float min_value,max_value,scale;
    template<typename loader>
    void load(const loader& io)
    {
        io.get_voxel_size(voxel_size.begin());
        io >> source_images;
        init();
    }
    template<typename loader>
    void loadLPS(loader& io)
    {
        io.get_voxel_size(voxel_size.begin());
        io.toLPS(source_images);
        init();
    }
    void init(void);
    bool initialize(FibSliceModel& slice,bool is_qsdr,const std::vector<std::string>& files);
public:
    float get_value_range(void) const;
    void get_slice(image::color_image& image,float contrast,float offset) const;
    image::const_pointer_image<float, 3> get_source(void) const  {return source_images;}
};

#endif
