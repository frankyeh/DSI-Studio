#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QMessageBox>
#include <QStyleFactory>
#include <QDir>
#include "mainwindow.h"
#include "tipl/tipl.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>
#include "program_option.hpp"
#include "cmd/cnt.cpp" // Qt project cannot build cnt.cpp without adding this.


track_recognition track_network;
fa_template fa_template_imp;
std::string fa_template_file_name,
        fib_template_file_name_1mm,fib_template_file_name_2mm,
        t1w_template_file_name,wm_template_file_name,
        t1w_mask_template_file_name;
std::vector<std::string> fa_template_list;

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
int cnn(void);
int qc(void);


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
    filename = QCoreApplication::applicationDirPath() + "/template/HCP1021_QA.nii.gz";
    if(QFileInfo(filename).exists())
        fa_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/template/HCP1021_QA.nii.gz";
    if(QFileInfo(filename).exists())
        fa_template_file_name = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/HCP1021.2mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name_2mm = filename.toStdString();
    filename = QDir::currentPath() + "/HCP1021.2mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name_2mm = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/HCP1021.1mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name_1mm = filename.toStdString();
    filename = QDir::currentPath() + "/HCP1021.1mm.fib.gz";
    if(QFileInfo(filename).exists())
        fib_template_file_name_1mm = filename.toStdString();


    filename = QCoreApplication::applicationDirPath() + "/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz";
    if(QFileInfo(filename).exists())
        t1w_template_file_name = filename.toStdString();

    filename = QCoreApplication::applicationDirPath() + "/mni_icbm152_wm_tal_nlin_asym_09c.nii.gz";
    if(QFileInfo(filename).exists())
        wm_template_file_name = filename.toStdString();
    filename = QDir::currentPath() + "/mni_icbm152_wm_tal_nlin_asym_09c.nii.gz";
    if(QFileInfo(filename).exists())
        wm_template_file_name = filename.toStdString();


    QDir dir = QCoreApplication::applicationDirPath()+ "/template";
    if(!dir.exists())
        dir = QDir::currentPath()+ "/template";
    QStringList name_list = dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    for(int i = 0;i < name_list.size();++i)
    {
        std::string full_path = (dir.absolutePath() + "/" + name_list[i]).toStdString();
        if(full_path == fa_template_file_name)
            fa_template_list.insert(fa_template_list.begin(),full_path);
        else
            fa_template_list.push_back(full_path);
    }
}

void init_application(void)
{
    QApplication::setOrganizationName("LabSolver");
    QApplication::setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    QApplication::setFont(font);
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    load_file_name();
    if(!fa_template_imp.load_from_file())
    {
        QMessageBox::information(0,"Error",fa_template_imp.error_msg.c_str(),0);
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
                init_application();
                std::cout << "Starting GUI-based command line interface." << std::endl;
                break;
            }

        if(!gui.get())
        {
            cmd.reset(new QCoreApplication(ac, av));
            load_file_name();
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
            return rec();
        if(po.get("action") == std::string("trk"))
            return trk();
        if(po.get("action") == std::string("src"))
            return src();
        if(po.get("action") == std::string("ana"))
            return ana();
        if(po.get("action") == std::string("exp"))
            return exp();
        if(po.get("action") == std::string("atl"))
            return atl();
        if(po.get("action") == std::string("cnt"))
            return cnt();
        if(po.get("action") == std::string("ren"))
            return ren();
        if(po.get("action") == std::string("cnn"))
            return cnn();
        if(po.get("action") == std::string("qc"))
            return qc();
        if(po.get("action") == std::string("vis"))
        {
            vis();
            if(po.get("stay_open") == std::string("1"))
                gui->exec();
            return 1;
        }
        std::cout << "Unknown action:" << po.get("action") << std::endl;
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
    if(ac > 2)
        return run_cmd(ac,av);
    QApplication a(ac,av);
    init_application();
    MainWindow w;
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}
