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

image::ml::network track_network;
std::vector<std::string> track_network_list;
fa_template fa_template_imp;
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

void load_track_network(void)
{
    QString file_name = QCoreApplication::applicationDirPath()+ "/network.txt";
    QString track_label = QCoreApplication::applicationDirPath()+ "/network_label.txt";

    if(QFileInfo(file_name).exists() && QFileInfo(track_label).exists())
    {
        track_network.load_from_file(file_name.toStdString().c_str());
        std::ifstream in(track_label.toStdString().c_str());
        std::string line;
        while(std::getline(in,line))
            track_network_list.push_back(line);
    }
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


program_option po;
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
            po.init(ac,av);

            if (!po.has("action") || !po.has("source"))
            {
                std::cout << "invalid command, use --help for more detail" << std::endl;
                return 1;
            }
            QDir::setCurrent(QFileInfo(po.get("action").c_str()).absolutePath());
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
            if(po.get("action") == std::string("vis"))
                return vis();
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
    a.setStyle(QStyleFactory::create("Fusion"));
    // load template
    if(!fa_template_imp.load_from_file())
    {
        QMessageBox::information(0,"Error","Cannot find HCP488_QA.nii.gz in file directory",0);
        return false;
    }
    load_atlas();
    load_track_network();

    MainWindow w;
    w.setFont(font);
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}
