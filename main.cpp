#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QMessageBox>
#include <QStyleFactory>
#include <QFileInfo>
#include <QDir>
#include "TIPL/tipl.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
#include "mapping/atlas.hpp"
#include "program_option.hpp"
#include "mainwindow.h"
#include "console.h"

#ifndef QT6_PATCH
#include <QTextCodec>
#endif

std::string
        fib_template_file_name_2mm,
        device_content_file;
std::vector<std::string> fa_template_list,
                         iso_template_list,
                         track_atlas_file_list;
std::vector<std::vector<std::string> > template_atlas_list;


int rec(program_option& po);
int trk(program_option& po);
int src(program_option& po);
int ana(program_option& po);
int exp(program_option& po);
int atl(program_option& po);
int cnt(program_option& po);
int cnt_ind(program_option& po);
int vis(program_option& po);
int ren(program_option& po);
int cnn(program_option& po);
int qc(program_option& po);
int reg(program_option& po);
int atk(program_option& po);
int xnat(program_option& po);


size_t match_volume(float volume)
{
    float min_dif = std::numeric_limits<float>::max();
    size_t matched_index = 0;
    for(size_t i = 0;i < fa_template_list.size();++i)
    {
        gz_nifti read;
        if(!read.load_from_file(fa_template_list[i].c_str()))
            continue;
        float v = float(read.nif_header.dim[1]*read.nif_header.dim[2]*read.nif_header.dim[3])*
                float(read.nif_header.pixdim[1]*read.nif_header.pixdim[2]*read.nif_header.pixdim[3]);
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
    for(int i = 0;i < dir_list.size();++i)
    {
        QDir cur_dir = dir_list[i];
        QStringList new_list = cur_dir.entryList(QStringList(""),QDir::AllDirs|QDir::NoDotAndDotDot);
        for(int index = 0;index < new_list.size();++index)
            dir_list << cur_dir.absolutePath() + "/" + new_list[index];
        QStringList file_list = cur_dir.entryList(QStringList(filter),QDir::Files|QDir::NoSymLinks);
        for (int index = 0;index < file_list.size();++index)
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
    return std::string();
}

bool load_file_name(void)
{
    fib_template_file_name_2mm = find_full_path("/atlas/ICBM152_adult/ICBM152_adult.fib.gz");
    device_content_file = find_full_path("/device.txt");

    //std::cout << "search templates and atlases" << std::endl;
    {
        QDir dir = QCoreApplication::applicationDirPath()+ "/atlas";
        if(!dir.exists())
            dir = QDir::currentPath()+ "/atlas";

        QStringList name_list = dir.entryList(QStringList("*"),QDir::Dirs|QDir::NoSymLinks);

        // Make ICBM152 the default
        for(int i = 0;i < name_list.size();++i)
        {
            if(name_list[i].contains("ICBM"))
            {
                QString item_to_move = name_list[i];
                name_list.erase(name_list.begin()+i);
                name_list.insert(name_list.begin(),item_to_move);
            }
        }
        for(int i = 0;i < name_list.size();++i)
        {
            QDir template_dir = dir.absolutePath() + "/" + name_list[i];
            QString qa_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".QA.nii.gz";
            QString iso_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".ISO.nii.gz";
            QString tt_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".tt.gz";
            if(!QFileInfo(qa_file_path).exists())
                continue;
            // setup QA and ISO template
            fa_template_list.push_back(qa_file_path.toStdString());
            if(QFileInfo(iso_file_path).exists())
                iso_template_list.push_back(iso_file_path.toStdString());
            else
                iso_template_list.push_back(std::string());

            if(QFileInfo(iso_file_path).exists())
                track_atlas_file_list.push_back(tt_file_path.toStdString());
            else
                track_atlas_file_list.push_back(std::string());
            // find related atlases
            {
                QStringList atlas_list = template_dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
                atlas_list << template_dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
                atlas_list.sort();
                std::vector<std::string> atlas_file_list;
                for(int index = 0;index < atlas_list.size();++index)
                    if(QFileInfo(atlas_list[index]).baseName() != name_list[i])
                        atlas_file_list.push_back((template_dir.absolutePath() + "/" + atlas_list[index]).toStdString());
                template_atlas_list.push_back(std::move(atlas_file_list));
            }
        }
        if(fa_template_list.empty())
            return false;
    }
    return true;
}

QString version_string(void)
{
    QString base = "DSI Studio version: ";
    #ifdef CUDA_ARCH
    if constexpr(tipl::use_cuda)
        base = QString("DSI Studio (CUDA SM%1) version: ").arg(CUDA_ARCH);
    #endif

    base += DSISTUDIO_RELEASE_NAME;

    base += "\"";
    unsigned int code = DSISTUDIO_RELEASE_CODE;
    #ifdef QT6_PATCH
        base += QStringDecoder(QStringDecoder::Utf8)(reinterpret_cast<const char*>(&code));
    #else
        base += QTextCodec::codecForName("UTF-8")->toUnicode(reinterpret_cast<const char*>(&code));
    #endif
    base += "\"";
    return base;
}

void init_application(void)
{
    QApplication::setOrganizationName("LabSolver");
    QApplication::setApplicationName(version_string());

    #ifdef __APPLE__
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    QApplication::setFont(font);
    #endif
    QSettings settings;
    QString style = settings.value("styles","Fusion").toString();
    if(style != "default" && !style.isEmpty())
        QApplication::setStyle(style);

    if(!load_file_name())
        QMessageBox::information(nullptr,"Error","Cannot find template data.");
}

int run_action(program_option& po,std::shared_ptr<QApplication> gui)
{
    std::string action = po.get("action");
    if(action == std::string("rec"))
        return rec(po);
    if(action == std::string("trk"))
        return trk(po);
    if(action == std::string("atk"))
        return atk(po);
    if(action == std::string("src"))
        return src(po);
    if(action == std::string("ana"))
        return ana(po);
    if(action == std::string("exp"))
        return exp(po);
    if(action == std::string("atl"))
        return atl(po);
    if(action == std::string("cnt"))
        return cnt(po);
    if(action == std::string("ren"))
        return ren(po);
    if(action == std::string("cnn"))
        return cnn(po);
    if(action == std::string("qc"))
        return qc(po);
    if(action == std::string("reg"))
        return reg(po);
    if(action == std::string("xnat"))
        return xnat(po);
    if(action == std::string("vis"))
    {
        vis(po);
        if(po.get("stay_open") == std::string("1"))
            gui->exec();
        return 0;
    }
    std::cout << "Unknown action:" << action << std::endl;
    return 1;
}
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
bool check_cuda(std::string& error_msg);
bool match_files(const std::string& file_path1,const std::string& file_path2,
                 const std::string& file_path1_others,std::string& file_path2_gen);
int run_cmd(int ac, char *av[])
{
    program_option po;
    try
    {
        std::cout << "DSI Studio \"" << DSISTUDIO_RELEASE_NAME << "\" " << __DATE__ << std::endl;

        if constexpr(tipl::use_cuda)
        {
            std::string msg;
            if(!check_cuda(msg))
            {
                std::cout << "ERROR:" << msg <<std::endl;
                return 1;
            }
        }

        if(!po.parse(ac,av))
        {
            std::cout << po.error_msg << std::endl;
            return 1;
        }
        if(po.has("version"))
            return 0;
        if (!po.has("action"))
        {
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }

        std::string source = po.get("source");
        std::string action = po.get("action");


        std::shared_ptr<QApplication> gui;
        std::shared_ptr<QCoreApplication> cmd;

        if ((action == "cnt" && po.get("no_tractogram",1) == 0) || action == "vis")
        {
            std::cout << "Starting GUI-based command line interface." << std::endl;
            gui.reset(new QApplication(ac, av));
            init_application();
        }
        else
        {
            cmd.reset(new QCoreApplication(ac, av));
            cmd->setOrganizationName("LabSolver");
            cmd->setApplicationName(version_string());
            if(!load_file_name())
            {
                std::cout << "ERROR: Cannot find template data." << std::endl;
                return 1;
            }
        }
        if(action == "atk" || action == "atl" || source.find('*') == std::string::npos) // atk, atl handle * by itself
            return run_action(po,gui);

        // handle source having wildcard
        std::vector<std::string> source_files;
        get_filenames_from(source,source_files);

        std::vector<std::pair<std::string,std::string> > wildcard_list;
        po.get_wildcard_list(wildcard_list);

        for (size_t i = 0;i < source_files.size();++i)
        {
            std::cout << "Process file:" << source_files[i] << std::endl;
            po.set("source",source_files[i]);
            // apply '*' to other arguments
            for(const auto& wildcard : wildcard_list)
            {
                if(wildcard.first == "source")
                    continue;
                std::string apply_wildcard;
                if(!match_files(source,source_files[i],wildcard.second,apply_wildcard))
                {
                    std::cout << "ERROR: cannot translate " << wildcard.second <<
                                 " at --" << wildcard.first << std::endl;
                    return 1;
                }
                std::cout << wildcard.second << "->" << apply_wildcard << std::endl;
                po.set(wildcard.first.c_str(),apply_wildcard);
            }
            po.set_used(0);
            if(run_action(po,gui) == 1)
            {
                std::cout << "Terminated due to error." << std::endl;
                return 1;
            }
        }
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

extern console_stream console;

int main(int ac, char *av[])
{
    if(ac > 2 || QString(av[1]).endsWith(".txt") || QString(av[1]).endsWith(".log"))
        return run_cmd(ac,av);

    // replace default std::cout buffer
    console.attach();

    QApplication a(ac,av);
    init_application();
    MainWindow w;
    w.setWindowTitle(version_string() + " " + __DATE__);
    has_gui = true;

    {
        if constexpr(tipl::use_cuda)
        {
            std::string msg;
            if(!check_cuda(msg))
            {
                QMessageBox::critical(&w,"ERROR",msg.c_str());
                return 1;
            }
        }
    }

    // presentation mode
    QStringList fib_list = QDir(QCoreApplication::applicationDirPath()+ "/presentation").
                            entryList(QStringList("*fib.gz") << QString("*_qa.nii.gz"),QDir::Files|QDir::NoSymLinks);
    if(fib_list.size())
    {
        w.hide();
        w.loadFib(QCoreApplication::applicationDirPath() + "/presentation/" + fib_list[0],true);
    }
    else
        w.show();

    if(ac == 2)
        w.openFile(av[1]);

    return a.exec();
}
