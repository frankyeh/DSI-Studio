#include "image/image.hpp"
#include "fa_template.hpp"
#include "libs/gzip_interface.hpp"

#include <QApplication>
#include <QDir>

extern QString fa_template_file_name;
bool fa_template::load_from_file(void)
{
    gz_nifti read;
    if(!fa_template_file_name.isEmpty() && read.load_from_file(fa_template_file_name.toStdString().c_str()))
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




