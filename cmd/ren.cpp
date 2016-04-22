#include <iostream>
#include <QDir>
#include "program_option.hpp"
bool RenameDICOMToDir(QString FileName, QString ToDir);
QStringList GetSubDir(QString Dir);
int ren(void)
{
    QString output;
    if(po.has("output"))
        output = po.get("output").c_str();
    else
        output = po.get("source").c_str();
    QStringList dirs = GetSubDir(po.get("source").c_str());
    for (unsigned int i = 0;i < dirs.size();++i)
    {
        QStringList files = QDir(dirs[i]).entryList(QStringList("*"),
                                    QDir::Files | QDir::NoSymLinks);
        for (unsigned int j = 0;j < files.size();++j)
        {
            std::cout << "renaming " << dirs[i].toStdString() << "/" << files[j].toStdString() << std::endl;
            if(!RenameDICOMToDir(dirs[i] + "/" + files[j],output))
                std::cout << "Cannot rename the file." << std::endl;
        }
    }
    return 0;
}


QStringList dirs = GetSubDir(path);
for(unsigned int index = 0;check_prog(index,dirs.size());++index)
{
    QStringList files = QDir(dirs[index]).entryList(QStringList("*"),
                                QDir::Files | QDir::NoSymLinks);
    set_title(QFileInfo(dirs[index]).fileName().toLocal8Bit().begin());
    for(unsigned int j = 0;j < files.size() && check_prog(index,dirs.size());++j)
    {
        set_title(files[j].toLocal8Bit().begin());
        RenameDICOMToDir(dirs[index] + "/" + files[j],path);
    }
}
