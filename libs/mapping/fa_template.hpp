#ifndef FA_TEMPLATE_HPP
#define FA_TEMPLATE_HPP

#include "tipl/tipl.hpp"
struct fa_template{
    std::string template_file_name,error_msg;
    tipl::vector<3> vs,shift;
    tipl::image<float,3> I;
    float tran[16];
    bool load_from_file(void);
    template<typename v_type>
    void to_mni(v_type& p)
    {
        if(tran[0] == -1.0f)
            p[0] = -p[0];
        else
            p[0] *= tran[0];
        if(tran[5] == -1.0f)
            p[1] = -p[1];
        else
            p[1] *= tran[5];
        if(tran[10] != 1.0f)
            p[2] *= tran[10];
        p += shift;
    }
    template<typename v_type>
    void from_mni(v_type& p)
    {
        p -= shift;
        if(tran[10] != 1.0f)
            p[2] /= tran[10];

        if(tran[5] == -1.0f)
            p[1] = -p[1];
        else
            p[1] /= tran[5];
        if(tran[0] == -1.0f)
            p[0] = -p[0];
        else
            p[0] /= tran[0];
    }
};


#endif // FA_TEMPLATE_HPP
