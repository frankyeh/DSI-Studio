#include <iostream>
#include <QDir>
#include "program_option.hpp"
bool RenameDICOMToDir(QString FileName, QString ToDir);
QStringList GetSubDir(QString Dir,bool recursive = true);
void dicom2src(std::string dir_,std::ostream& out);
int ren(program_option& po)
{
    QString output;
    if(po.has("output"))
        output = po.get("output").c_str();
    else
        output = po.get("source").c_str();
    QStringList dirs = GetSubDir(po.get("source").c_str());
    for (int i = 0;i < dirs.size();++i)
    {
        QStringList files = QDir(dirs[i]).entryList(QStringList("*"),
                                    QDir::Files | QDir::NoSymLinks);
        for (int j = 0;j < files.size();++j)
        {
            show_progress() << "renaming " << dirs[i].toStdString() << "/" << files[j].toStdString() << std::endl;
            if(!RenameDICOMToDir(dirs[i] + "/" + files[j],output))
                show_progress() << "cannot rename the file." << std::endl;
        }
    }
    if(po.get("to_src_nii",0))
        dicom2src(output.toStdString(),std::cout);
    return 0;
}
