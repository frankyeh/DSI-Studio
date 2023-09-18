#include <iostream>
#include <QDir>
#include "TIPL/tipl.hpp"
QStringList rename_dicom_at_dir(QString path,QString output);
QStringList GetSubDir(QString Dir,bool recursive = true);
void dicom2src_and_nii(std::string dir_);
int ren(tipl::program_option<tipl::out>& po)
{
    tipl::progress prog("run ren");
    auto source = std::filesystem::path(po.get("source")).string();
    auto output = std::filesystem::path(po.get("output",po.get("source"))).string();
    auto subject_dir = rename_dicom_at_dir(source.c_str(),output.c_str());
    if(po.get("to_src_nii",0))
    {
        for(auto dir : subject_dir)
        {
            tipl::progress prog("Converting DICOM to SRC/NII ",dir.toStdString().c_str());
            dicom2src_and_nii(dir.toStdString().c_str());
        }
    }
    return 0;
}
