#include "image/image.hpp"
#include "fa_template.hpp"
#include "libs/gzip_interface.hpp"

#include <QApplication>
#include <QDir>

bool fa_template::load_from_file(void)
{
    std::string fa_template_path = QCoreApplication::applicationDirPath().toLocal8Bit().begin();
    fa_template_path += "/HCP842_1mm_qa.nii.gz";
    std::string fa_template_path2 = QDir::currentPath().toLocal8Bit().begin();
    fa_template_path2 += "/HCP842_1mm_qa.nii.gz";

    gz_nifti read;
    if((!template_file_name.empty() && read.load_from_file(template_file_name.c_str())) ||
            read.load_from_file(fa_template_path.c_str()) || read.load_from_file(fa_template_path2.c_str()))
    {
        read.toLPS(I);
        read.get_image_transformation(tran);
        vs[0] = tran[0];
        vs[1] = tran[5];
        vs[2] = tran[10];
        vs.abs();
        shift[0] = tran[3];
        shift[1] = tran[7];
        shift[2] = tran[11];
        return true;
    }
    return false;
}

void fa_template::to_mni(image::vector<3,float>& p)
{
    p[0] = p[0]*tran[0];
    p[1] = p[1]*tran[5];
    p[2] = p[2]*tran[10];
    p += shift;
}


