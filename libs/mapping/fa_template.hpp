#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP

#include "image/image.hpp"
struct fa_template{
    std::string template_file_name,error_msg;
    image::vector<3> vs,shift;
    image::basic_image<float,3> I;
    float tran[16];
    bool load_from_file(void);
    template<typename v_type>
    void to_mni(v_type& p)
    {
        if(tran[0] == -1)
            p[0] = -p[0];
        else
            p[0] = p[0]*tran[0];
        if(tran[5] == -1)
            p[1] = -p[1];
        else
            p[1] = p[1]*tran[5];
        if(tran[10] != 1)
            p[2] = p[2]*tran[10];
        p += shift;
    }
};


#endif // FA_TEMPLATE_HPP
