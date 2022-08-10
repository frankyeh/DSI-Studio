#include <iostream>
#include <QDir>
#include "program_option.hpp"
void rename_dicom_at_dir(QString path,QString output);
QStringList GetSubDir(QString Dir,bool recursive = true);
void dicom2src(std::string dir_,std::ostream& out);
int ren(program_option& po)
{
    progress p("run ren");
    show_progress() << "current directory is " << std::filesystem::current_path() << std::endl;
    rename_dicom_at_dir(po.get("source").c_str(),
                        po.get("output",po.get("source")).c_str());
    if(po.get("to_src_nii",0))
        dicom2src(po.get("output",po.get("source")).c_str(),std::cout);
    return 0;
}
