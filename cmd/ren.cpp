#include <iostream>
#include <QDir>
#include "program_option.hpp"
void rename_dicom_at_dir(QString path,QString output);
QStringList GetSubDir(QString Dir,bool recursive = true);
void dicom2src(std::string dir_,std::ostream& out);
int ren(program_option& po)
{
    progress p("run ren");
    auto source = std::filesystem::path(po.get("source")).string();
    auto output = std::filesystem::path(po.get("output",po.get("source"))).string();
    rename_dicom_at_dir(source.c_str(),output.c_str());
    if(po.get("to_src_nii",0))
        dicom2src(output.c_str(),std::cout);
    return 0;
}
