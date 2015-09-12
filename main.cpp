#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include "mainwindow.h"
#include "boost/program_options.hpp"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>

#include "cmd/cnt.cpp" // Qt project cannot build cnt.cpp without adding this.

namespace po = boost::program_options;


int rec(int ac, char *av[]);
int trk(int ac, char *av[]);
int src(int ac, char *av[]);
int ana(int ac, char *av[]);
int exp(int ac, char *av[]);
int atl(int ac, char *av[]);
int cnt(int ac, char *av[]);
int vis(int ac, char *av[]);


fa_template fa_template_imp;
std::vector<atlas> atlas_list;
QStringList search_files(QString dir,QString filter)
{
    QStringList dir_list,src_list;
    dir_list << dir;
    for(unsigned int i = 0;i < dir_list.size();++i)
    {
        QDir cur_dir = dir_list[i];
        QStringList new_list = cur_dir.entryList(QStringList(""),QDir::AllDirs|QDir::NoDotAndDotDot);
        for(unsigned int index = 0;index < new_list.size();++index)
            dir_list << cur_dir.absolutePath() + "/" + new_list[index];
        QStringList file_list = cur_dir.entryList(QStringList(filter),QDir::Files|QDir::NoSymLinks);
        for (unsigned int index = 0;index < file_list.size();++index)
            src_list << dir_list[i] + "/" + file_list[index];
    }
    return src_list;
}


void load_atlas(void)
{
    QDir dir = QCoreApplication::applicationDirPath()+ "/atlas";
    QStringList atlas_name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
    atlas_name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    if(atlas_name_list.empty())
    {
        dir = QDir::currentPath()+ "/atlas";
        atlas_name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
        atlas_name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    }
    if(atlas_name_list.empty())
        return;
    atlas_list.resize(atlas_name_list.size());
    for(int index = 0;index < atlas_name_list.size();++index)
    {
        atlas_list[index].name = QFileInfo(atlas_name_list[index]).baseName().toLocal8Bit().begin();
        atlas_list[index].filename = (dir.absolutePath() + "/" + atlas_name_list[index]).toLocal8Bit().begin();
    }

}
image::basic_image<int,3> cerebrum_1mm,cerebrum_2mm;
bool load_cerebrum_mask(void)
{
    if(!cerebrum_1mm.empty() && !cerebrum_2mm.empty())
        return true;
    QString file1 = "/cerebrum1.nii.gz";
    QString file2 = "/cerebrum2.nii.gz";
    if(QFileInfo(QCoreApplication::applicationDirPath() + file1).exists())
        file1 = QCoreApplication::applicationDirPath() + file1;
    else
        if(QFileInfo(QDir::currentPath() + file1).exists())
            file1 = QDir::currentPath() + file1;
        else
            return false;

    if(QFileInfo(QCoreApplication::applicationDirPath() + file2).exists())
        file2 = QCoreApplication::applicationDirPath() + file2;
    else
        if(QFileInfo(QDir::currentPath() + file2).exists())
            file2 = QDir::currentPath() + file2;
        else
            return false;
    {
        gz_nifti n1;
        if(!n1.load_from_file(file1.toLocal8Bit().begin()))
            return false;
        n1.toLPS(cerebrum_1mm);
    }
    {
        gz_nifti n2;
        if(!n2.load_from_file(file2.toLocal8Bit().begin()))
            return false;
        n2.toLPS(cerebrum_2mm);
    }
    return true;
}

int main(int ac, char *av[])
{ 
    if(ac > 2)
    {
        std::auto_ptr<QCoreApplication> cmd;
        {
            for (int i = 1; i < ac; ++i)
                if (std::string(av[i]) == std::string("--action=cnt") ||
                    std::string(av[i]) == std::string("--action=vis"))
                {
                    cmd.reset(new QApplication(ac, av));
                    std::cout << "Starting GUI-based command line interface." << std::endl;
                    break;
                }
            if(!cmd.get())
                cmd.reset(new QCoreApplication(ac, av));
        }
        cmd->setOrganizationName("LabSolver");
        cmd->setApplicationName("DSI Studio");

        try
        {
            std::cout << "DSI Studio " << __DATE__ << ", Fang-Cheng Yeh" << std::endl;

        // options for general options
            po::options_description desc("reconstruction options");
            desc.add_options()
            ("help", "help message")
            ("action", po::value<std::string>(), "rec:diffusion reconstruction trk:fiber tracking")
            ("source", po::value<std::string>(), "assign the .src or .fib file name")
            ;


            po::variables_map vm;
            po::store(po::command_line_parser(ac, av).options(desc).allow_unregistered().run(),vm);
            if (vm.count("help"))
            {
                std::cout << "example: perform reconstruction" << std::endl;
                std::cout << "    --action=rec --source=test.src.gz --method=4 " << std::endl;
                std::cout << "example: perform fiber tracking" << std::endl;
                std::cout << "    --action=trk --source=test.src.gz.fib.gz --method=0 --fiber_count=5000" << std::endl;
                return 1;
            }

            if (!vm.count("action") || !vm.count("source"))
            {
                std::cout << "invalid command, use --help for more detail" << std::endl;
                return 1;
            }
            if(vm["action"].as<std::string>() == std::string("rec"))
                return rec(ac,av);
            if(vm["action"].as<std::string>() == std::string("trk"))
                return trk(ac,av);
            if(vm["action"].as<std::string>() == std::string("src"))
                return src(ac,av);
            if(vm["action"].as<std::string>() == std::string("ana"))
                return ana(ac,av);
            if(vm["action"].as<std::string>() == std::string("exp"))
                return exp(ac,av);
            if(vm["action"].as<std::string>() == std::string("atl"))
                return atl(ac,av);
            if(vm["action"].as<std::string>() == std::string("cnt"))
                return cnt(ac,av);
            if(vm["action"].as<std::string>() == std::string("vis"))
                return vis(ac,av);
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }
        catch(const std::exception& e ) {
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "unknown error occured" << std::endl;
        }

        return 1;
    }
    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);
    // load template
    if(!fa_template_imp.load_from_file())
    {
        QMessageBox::information(0,"Error","Cannot find HCP488_QA.nii.gz in file directory",0);
        return false;
    }
    // load atlas
    load_atlas();

    MainWindow w;
    w.setFont(font);
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}
