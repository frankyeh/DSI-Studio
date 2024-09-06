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
    bool is_mni = false;
    bool interpolation = true;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4> T;
    std::string error_msg;
public:
    variant_image(void){}
    variant_image(const variant_image& rhs)
    {copy_from(rhs);}
    variant_image& operator=(const variant_image& rhs){copy_from(rhs);return *this;}
private:
    void copy_from(const variant_image& rhs)
    {
        I_float32 = rhs.I_float32;
        I_int32 = rhs.I_int32;
        I_int16 = rhs.I_int16;
        I_int8 = rhs.I_int8;
        pixel_type = decltype(pixel_type)(rhs.pixel_type);
        shape = rhs.shape;
        is_mni = rhs.is_mni;
        vs = rhs.vs;
        T = rhs.T;
    }
public:
    template <typename T>
    void apply(T&& fun)
    {
        switch(pixel_type)
        {
            case int8:fun(I_int8);return;
            case int16:fun(I_int16);return;
            case int32:fun(I_int32);return;
            case float32:fun(I_float32);return;
        }
    }
    bool read_mat_image(size_t index,
                        tipl::io::gz_mat_read& mat);
    void write_mat_image(size_t index,
                        tipl::io::gz_mat_read& mat);
    void change_type(decltype(pixel_type));
    bool command(std::string cmd,std::string param1);
    bool load_from_file(const char* file_name,std::string& info);
    bool empty(void) const{return shape.size() == 0;}
    size_t buf_size(void) const { return shape.size()*pixelbit[pixel_type];}
};
#endif // IMG_HPP
