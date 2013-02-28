#include "image/image.hpp"
#include "math/matrix_op.hpp"
#include "fa_template.hpp"
#include "libs/gzip_interface.hpp"
bool fa_template::load_from_file(const char* file_name)
{
    gz_nifti read;
    if(read.load_from_file(file_name))
    {
        read >> I;
        read.get_image_transformation(tran.begin());
        tran[15] = 1.0;
        if(tran[0] < 0)
        {
            if(tran[5] > 0)
            {
                image::flip_y(I);
                tran[0] = -tran[0];
                tran[3] -= (I.width()-1)*tran[0];
            }
        }
        else
            image::flip_xy(I);
        return true;
    }
    return false;
}

void fa_template::to_mni(image::vector<3,float>& p)
{
    image::vector<3,float> pp(p);
    pp[0] = I.width()-pp[0]-1;
    pp[1] = I.height()-pp[1]-1;
    image::vector_transformation(pp.begin(),p.begin(),tran.begin(),image::vdim<3>());
}

void fa_template::add_transformation(std::vector<float>& t)
{
    std::vector<float> tt(t);
    tt.resize(16);
    tt[12] = tt[13] = tt[14] = 0;
    tt[15] = 1.0;
    // flip xy
    for(int i = 0;i < 8;++i)
        tt[i] = -tt[i];
    tt[3] += I.width()-1;
    tt[7] += I.height()-1;
    math::matrix_product(tran.begin(),tt.begin(),t.begin(),math::dim<3,4>(),math::dim<4,4>());
 }
