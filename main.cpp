#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QCleanlooksStyle>
#include <QMacStyle>
#include <QMetaObject>
#include <QMetaMethod>
#include <QMessageBox>
#include <QDir>
#include "mainwindow.h"
#include "boost/program_options.hpp"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>
namespace po = boost::program_options;

int rec(int ac, char *av[]);
int trk(int ac, char *av[]);
int src(int ac, char *av[]);
int ana(int ac, char *av[]);
int exp(int ac, char *av[]);
int atl(int ac, char *av[]);

fa_template fa_template_imp;
std::vector<atlas> atlas_list;
void prog_debug(const char* file,const char* fun)
{

}

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


std::string get_fa_template_path(void)
{
    std::string fa_template_path = QCoreApplication::applicationDirPath().toLocal8Bit().begin();
    fa_template_path += "/FMRIB58_FA_1mm.nii.gz";
    return fa_template_path;
}

void load_atlas(void)
{
    QDir dir = QCoreApplication::applicationDirPath()+ "/atlas";
    QStringList atlas_name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
    atlas_name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    for(int index = 0;index < atlas_name_list.size();++index)
    {
        atlas_list.push_back(atlas());
        if(!atlas_list.back().load_from_file((dir.absolutePath() + "/" + atlas_name_list[index]).toLocal8Bit().begin()))
            atlas_list.pop_back();
        atlas_list.back().name = QFileInfo(atlas_name_list[index]).baseName().toLocal8Bit().begin();
    }
}



int main(int ac, char *av[])
{

    if(ac > 2)
    {
        QCoreApplication cmd(ac,av);
        cmd.setOrganizationName("LabSolver");
        cmd.setApplicationName("DSI Studio");

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
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }
        catch(const std::exception& e ) {
            std::cerr << e.what() << std::endl;
        }
        return 1;
    }
    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);
    //QApplication::setStyle(new QMacStyle);

    // load template
    if(!fa_template_imp.load_from_file(get_fa_template_path().c_str()))
    {
        std::string error_str = "Cannot find ";
        error_str += get_fa_template_path();
        QMessageBox::information(0,"Error",error_str.c_str(),0);
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
