#include "image/image.hpp"
#include "fa_template.hpp"
#include "libs/gzip_interface.hpp"

#include <QApplication>
#include <QDir>

bool fa_template::load_from_file(void)
{
    std::string fa_template_path = QCoreApplication::applicationDirPath().toLocal8Bit().begin();
    fa_template_path += "/HCP488_QA.nii.gz";
    std::string fa_template_path2 = QDir::currentPath().toLocal8Bit().begin();
    fa_template_path2 += "/HCP488_QA.nii.gz";

    gz_nifti read;
    if((!template_file_name.empty() && read.load_from_file(template_file_name.c_str())) ||
            read.load_from_file(fa_template_path.c_str()) || read.load_from_file(fa_template_path2.c_str()))
    {
        read >> I;
        read.get_image_transformation(tran.begin());
        tran[15] = 1.0;
        if(tran[0] < 0)
        {
            tran[0] = -tran[0];
            tran[3] -= (I.width()-1)*tran[0];
            if(tran[5] > 0)
                image::flip_y(I);
            else
            {
                tran[5] = -tran[5];
                tran[7] -= (I.height()-1)*tran[5];
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
    p[0] = I.width()-p[0]-1;
    p[1] = I.height()-p[1]-1;
    p.to(tran);
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
    image::mat::product(tran.begin(),tt.begin(),t.begin(),image::dim<3,4>(),image::dim<4,4>());
 }
