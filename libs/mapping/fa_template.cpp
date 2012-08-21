#include "image/image.hpp"
#include "fa_template.hpp"

bool fa_template::load_from_file(const char* file_name)
{
    image::io::nifti read;
    if(read.load_from_file(file_name))
    {
        read >> I;
        read.get_image_transformation(tran.begin());
        if(tran[0] < 0)
        {
            image::flip_y(I);
            tran[0] = -tran[0];
            tran[3] -= (I.width()-1)*tran[0];
        }
        else
            image::flip_xy(I);
        return true;
    }
    return false;
}

void fa_template::to_mni(image::vector<3,float>& p)
{
    p[0] = I.width()-p[0]-1;
    p[1] = I.height()-p[1]-1;

    p[0] *= tran[0];
    p[1] *= tran[5];
    p[2] *= tran[10];

    p[0] += tran[3];
    p[1] += tran[7];
    p[2] += tran[11];
}
