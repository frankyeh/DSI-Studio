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
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>
#include "program_option.hpp"
#include "cmd/cnt.cpp" // Qt project cannot build cnt.cpp without adding this.

std::string
        fib_template_file_name_2mm,
        t1w_template_file_name,wm_template_file_name,
        t1w_mask_template_file_name,tractography_atlas_file_name;
std::vector<std::string> fa_template_list,iso_template_list,atlas_file_list,tractography_name_list;
int rec(void);
int trk(void);
int src(void);
int ana(void);
int exp(void);
int atl(void);
int cnt(void);
int cnt_ind(void);
int vis(void);
int ren(void);
int cnn(void);
int qc(void);
int reg(void);


int match_template(float volume)
{
    float min_dif = std::numeric_limits<float>::max();
    int matched_index = 0;
    for(int i = 0;i < fa_template_list.size();++i)
    {
        gz_nifti read;
        if(!read.load_from_file(fa_template_list[i].c_str()))
            continue;
        float v = float(read.nif_header2.dim[1]*read.nif_header2.dim[2]*read.nif_header2.dim[3])*
                read.nif_header2.pixdim[1]*read.nif_header2.pixdim[2]*read.nif_header2.pixdim[3];
        v = std::fabs(v-volume);
        if(v < min_dif)
        {
            min_dif = v;
            matched_index = i;
        }
    }
    return matched_index;
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

std::string find_full_path(QString name)
{
    QString filename = QCoreApplication::applicationDirPath() + name;
    if(QFileInfo(filename).exists())
        return filename.toStdString();
    filename = QDir::currentPath() + name;
    if(QFileInfo(filename).exists())
        return filename.toStdString();
    std::string error_msg = "Cannot find ";
    error_msg += filename.toStdString();
    throw error_msg;
}


void load_file_name(void)
{
    fib_template_file_name_2mm = find_full_path("/HCP1021.2mm.fib.gz");
    t1w_template_file_name = find_full_path("/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz");
    wm_template_file_name = find_full_path("/mni_icbm152_wm_tal_nlin_asym_09c.nii.gz");
    t1w_mask_template_file_name = find_full_path("/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii.gz");
    tractography_atlas_file_name = find_full_path("/atlas/HCP842_tractography.trk.gz");

    std::string tractography_name_list_file_name = find_full_path("/atlas/HCP842_tractography.txt");
    if(!tractography_atlas_file_name.empty() && QFileInfo(tractography_name_list_file_name.c_str()).exists())
    {
        std::ifstream in(tractography_name_list_file_name);
        std::string line;
        while(std::getline(in,line))
        {
            std::istringstream in2(line);
            in2 >> line;
            in2 >> line;
            std::replace(line.begin(),line.end(),'_',' ');
            std::transform(line.begin(), line.end(), line.begin(),::tolower);
            if(line.back() == 'l' && line[line.length()-2] == ' ')
                line = std::string("left ") + line.substr(0,line.length()-2);
            if(line.back() == 'r' && line[line.length()-2] == ' ')
                line = std::string("right ") + line.substr(0,line.length()-2);
            tractography_name_list.push_back(line);
        }
    }
    else
        tractography_atlas_file_name.clear();
    // search for all anisotropy template
    {
        QDir dir = QCoreApplication::applicationDirPath()+ "/template";
        if(!dir.exists())
            dir = QDir::currentPath()+ "/template";
        QStringList name_list = dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);

        // Make HCP1021 the default
        for(int i = 0;i < name_list.size();++i)
        {
            if(name_list[i].contains("HCP"))
            {
                QString item_to_move = name_list[i];
                name_list.erase(name_list.begin()+i);
                name_list.insert(name_list.begin(),item_to_move);
            }
        }
        for(int i = 0;i < name_list.size();++i)
        {
            std::string full_path = (dir.absolutePath() + "/" + name_list[i]).toStdString();
            if(!QFileInfo(name_list[i]).fileName().contains(".QA.nii.gz"))
                continue;
            fa_template_list.push_back(full_path);
            QString iso_file = dir.absolutePath() + "/" + QFileInfo(name_list[i]).baseName() + ".ISO.nii.gz";
            if(QFileInfo(iso_file).exists())
                iso_template_list.push_back(iso_file.toStdString());
            else
                iso_template_list.push_back(std::string());
        }
    }

    QDir dir = QCoreApplication::applicationDirPath()+ "/atlas";
    if(!dir.exists())
        dir = QDir::currentPath()+ "/atlas";
    QStringList name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
    name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    if(name_list.empty())
        return;
    for(int i = 1;i < name_list.size();++i)
        if(name_list[i].contains("tractography"))
        {
            auto str = name_list[i];
            name_list.removeAt(i);
            name_list.insert(0,str);
        }
    for(int index = 0;index < name_list.size();++index)
        atlas_file_list.push_back((dir.absolutePath() + "/" + name_list[index]).toStdString());
}

void init_application(void)
{
    QApplication::setOrganizationName("LabSolver");
    QApplication::setApplicationName("DSI Studio");

    #ifdef __APPLE__
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    QApplication::setFont(font);
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    #endif
    try{
    load_file_name();
    }
    catch(std::string msg)
    {
        QMessageBox::information(0,"Error",QString() + msg.c_str() +
            ". Please download dsi_studio_other_files.zip from DSI Studio website and place them with the DSI Studio executives",0);
    }
}

program_option po;
int run_action(std::shared_ptr<QApplication> gui)
{
    std::string action = po.get("action");
    if(action == std::string("rec"))
        return rec();
    if(action == std::string("trk"))
        return trk();
    if(action == std::string("src"))
        return src();
    if(action == std::string("ana"))
        return ana();
    if(action == std::string("exp"))
        return exp();
    if(action == std::string("atl"))
        return atl();
    if(action == std::string("cnt"))
        return cnt();
    if(action == std::string("cnt_ind"))
        return cnt_ind();
    if(action == std::string("ren"))
        return ren();
    if(action == std::string("cnn"))
        return cnn();
    if(action == std::string("qc"))
        return qc();
    if(action == std::string("reg"))
        return reg();
    if(action == std::string("vis"))
    {
        vis();
        if(po.get("stay_open") == std::string("1"))
            gui->exec();
        return 0;
    }
    std::cout << "Unknown action:" << action << std::endl;
    return 1;
}

int run_cmd(int ac, char *av[])
{
    try
    {
        std::cout << "DSI Studio " << __DATE__ << ", Fang-Cheng Yeh" << std::endl;
        if(!po.parse(ac,av))
        {
            std::cout << po.error_msg << std::endl;
            return 1;
        }
        std::shared_ptr<QApplication> gui;
        std::shared_ptr<QCoreApplication> cmd;
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
            try{
            load_file_name();
            }
            catch(std::string msg)
            {
                std::cout << msg << std::endl;
                return 1;
            }
            cmd->setOrganizationName("LabSolver");
            cmd->setApplicationName("DSI Studio");
        }
        if (!po.has("action"))
        {
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }
        QDir::setCurrent(QFileInfo(po.get("source").c_str()).absolutePath());
        if(po.get("source").find('*') != std::string::npos)
        {
            auto file_list = QDir::current().entryList(QStringList(QFileInfo(po.get("source").c_str()).fileName()),
                                            QDir::Files|QDir::NoSymLinks);
            for (unsigned int index = 0;index < file_list.size();++index)
            {
                QString filename = QDir::current().absoluteFilePath(file_list[index]);
                std::cout << "=======================================" << std::endl;
                std::cout << "Process file:" << filename.toStdString() << std::endl;
                po.set("source",filename.toStdString());
                run_action(gui);
            }
        }
        else
            return run_action(gui);
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

extern bool has_gui;
int main(int ac, char *av[])
{
    if(ac > 2)
        return run_cmd(ac,av);
    QApplication a(ac,av);
    init_application();
    MainWindow w;
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    has_gui = true;
    return a.exec();
}
