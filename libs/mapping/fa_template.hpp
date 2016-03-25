#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP

#include "image/image.hpp"
struct fa_template{
    std::string template_file_name;
    image::vector<3> vs,shift;
    image::basic_image<float,3> I;
    float tran[16];
    bool load_from_file(void);
    void to_mni(image::vector<3,float>& p);
};


#endif // FA_TEMPLATE_HPP
