#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QMessageBox>
#include <QStyleFactory>
#include <QDir>
#include "mainwindow.h"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>
#include "program_option.hpp"
#include "cmd/cnt.cpp" // Qt project cannot build cnt.cpp without adding this.

track_recognition track_network;
fa_template fa_template_imp;
std::string fa_template_file_name,
        fib_template_file_name,
        t1w_template_file_name,
        t1w_mask_template_file_name;
extern std::vector<atlas> atlas_list;
void load_atlas(void);
int rec(void);
int trk(void);
int src(void);
int ana(void);
int exp(void);
int atl(void);
int cnt(void);
int vis(void);
int ren(void);


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

void load_file_name(void)
{
    QString filename;
    filename = QCoreApplication::applicationDirPath() + "/HCP842_QA.nii.gz";
    if(QFileInfo(filename).exists())
        fa_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/HCP842_QA.nii.gz";
    if(QFileInfo(filename).exists())
        fa_template_file_name = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/HCP842_2mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/HCP842_2mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_template_file_name = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_mask_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_mask_template_file_name = filename.toStdString();
}

void init_application(QApplication& a)
{
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);
    a.setStyle(QStyleFactory::create("Fusion"));
    if(!fa_template_imp.load_from_file())
    {
        QMessageBox::information(0,"Error",QString("Cannot find HCP842_QA.nii.gz in ") + QCoreApplication::applicationDirPath(),0);
        return;
    }


    load_atlas();
}

program_option po;
int run_cmd(int ac, char *av[])
{
    try
    {
        std::cout << "DSI Studio " << __DATE__ << ", Fang-Cheng Yeh" << std::endl;
        po.init(ac,av);
        std::auto_ptr<QApplication> gui;
        std::auto_ptr<QCoreApplication> cmd;
        for (int i = 1; i < ac; ++i)
            if (std::string(av[i]) == std::string("--action=cnt") ||
                std::string(av[i]) == std::string("--action=vis"))
            {
                gui.reset(new QApplication(ac, av));
                init_application(*gui.get());
                std::cout << "Starting GUI-based command line interface." << std::endl;
                break;
            }

        if(!gui.get())
        {
            cmd.reset(new QCoreApplication(ac, av));
            cmd->setOrganizationName("LabSolver");
            cmd->setApplicationName("DSI Studio");
        }
        if (!po.has("action") || !po.has("source"))
        {
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }
        QDir::setCurrent(QFileInfo(po.get("source").c_str()).absolutePath());
        if(po.get("action") == std::string("rec"))
            rec();
        if(po.get("action") == std::string("trk"))
            trk();
        if(po.get("action") == std::string("src"))
            src();
        if(po.get("action") == std::string("ana"))
            ana();
        if(po.get("action") == std::string("exp"))
            exp();
        if(po.get("action") == std::string("atl"))
            atl();
        if(po.get("action") == std::string("cnt"))
            cnt();
        if(po.get("action") == std::string("vis"))
            vis();
        if(po.get("action") == std::string("ren"))
            ren();
        if(gui.get() && po.get("stay_open") == std::string("1"))
            gui->exec();
        return 1;
    }
    catch(const std::exception& e ) {
        std::cout << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "unknown error occured" << std::endl;
    }
    return 0;
}



int main(int ac, char *av[])
{ 
    load_file_name();
    if(ac > 2)
        return run_cmd(ac,av);
    QApplication a(ac,av);
    init_application(a);
    MainWindow w;
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}
