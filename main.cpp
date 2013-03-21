#include <iostream>
#include <iterator>
#include <string>
#include <QApplication>
#include <QCleanlooksStyle>
#include <QMetaObject>
#include <QMetaMethod>
#include <QMessageBox>
#include <QDir>
#include "mainwindow.h"
#include "mat_file.hpp"
#include "boost/program_options.hpp"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include <iostream>
#include <iterator>
namespace po = boost::program_options;

int rec(int ac, char *av[]);
int trk(int ac, char *av[]);
int src(int ac, char *av[]);
int ana(int ac, char *av[]);
int exp(int ac, char *av[]);

fa_template fa_template_imp;

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


std::string program_base;
bool load_fa_template(void)
{
    std::string fa_template_path = program_base;
    fa_template_path += "FMRIB58_FA_1mm.nii.gz";
    if(!fa_template_imp.load_from_file(fa_template_path.c_str()))
    {
        std::string error_str = "Cannot find the fa template file at ";
        error_str += fa_template_path;
        QMessageBox::information(0,"Error",error_str.c_str(),0);
        return false;
    }
    return true;
}

int main(int ac, char *av[])
{

    {
        int pos = 0;
        for(int index = 0;av[0][index];++index)
            if(av[0][index] == '\\' || av[0][index] == '/')
                pos = index;
        program_base = std::string(&(av[0][0]),&(av[0][0])+pos+1);
    }

    if(ac > 2)
    {
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
                std::cout << "options:" << std::endl;
                rec(0,0);
                std::cout << "example: perform fiber tracking" << std::endl;
                std::cout << "    --action=trk --source=test.src.gz.fib.gz --method=0 --fiber_count=5000" << std::endl;
                std::cout << "options:" << std::endl;
                trk(0,0);
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
        }
        return 1;
    }

    QApplication::setStyle(new QCleanlooksStyle);
    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);

    if(!load_fa_template())
        return -1;

    MainWindow w;
    w.setFont(font);
    w.showMaximized();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}
