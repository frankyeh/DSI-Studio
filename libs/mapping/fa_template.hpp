#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP
#include "image/image.hpp"

struct fa_template{
    std::vector<float> tran;
    image::basic_image<float,3> I;
    bool load_from_file(const char* file_name);
    void to_mni(image::vector<3,float>& p);
    void to_mni(std::vector<float>& t);
    fa_template(void):tran(16){}
};

#endif // FA_TEMPLATE_HPP
