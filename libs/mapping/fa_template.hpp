#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP

#include "image/image.hpp"
struct fa_template{
    std::string template_file_name;
    image::vector<3> vs,shift;
    image::basic_image<float,3> I;
    image::basic_image<unsigned char,3> mask;
    float tran[16];
    bool load_from_file(void);
    template<typename v_type>
    void to_mni(v_type& p)
    {
        p[0] = p[0]*tran[0];
        p[1] = p[1]*tran[5];
        p[2] = p[2]*tran[10];
        p += shift;
    }
};


#endif // FA_TEMPLATE_HPP
