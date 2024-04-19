#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMessageBox>
#include <QStyleFactory>
#include <QFileInfo>
#include <QDir>
#include <QImageReader>
#include "mapping/atlas.hpp"
#include "mainwindow.h"
#include "console.h"

#ifndef QT6_PATCH
#include <QTextCodec>
#else
#include <QStringDecoder>
#endif

std::string device_content_file,topup_param_file;
std::vector<std::string> fa_template_list,
                         iso_template_list,
                         fib_template_list,
                         model_list_t2w;
std::vector<std::vector<std::string> > atlas_file_name_list;


class CustomSliceModel;
std::vector<std::shared_ptr<CustomSliceModel> > other_slices;

int rec(tipl::program_option<tipl::out>& po);
int trk(tipl::program_option<tipl::out>& po);
int src(tipl::program_option<tipl::out>& po);
int ana(tipl::program_option<tipl::out>& po);
int exp(tipl::program_option<tipl::out>& po);
int atl(tipl::program_option<tipl::out>& po);
int cnt(tipl::program_option<tipl::out>& po);
int cnt_ind(tipl::program_option<tipl::out>& po);
int vis(tipl::program_option<tipl::out>& po);
int ren(tipl::program_option<tipl::out>& po);
int cnn(tipl::program_option<tipl::out>& po);
int qc(tipl::program_option<tipl::out>& po);
int reg(tipl::program_option<tipl::out>& po);
int atk(tipl::program_option<tipl::out>& po);
int xnat(tipl::program_option<tipl::out>& po);


size_t match_volume(float volume)
{
    float min_dif = std::numeric_limits<float>::max();
    size_t matched_index = 0;
    for(size_t i = 0;i < fa_template_list.size();++i)
    {
        tipl::io::gz_nifti read;
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

QImage readImage(QString filename,std::string& error)
{
    QImageReader im(filename);
    if(!im.canRead())
    {
        error = im.errorString().toStdString();
        return QImage();
    }
    tipl::out() << "loading image: " << filename.toStdString();
    tipl::out() << "size:" << im.size().width() << " " << im.size().height();
#ifdef QT6_PATCH
    im.setAllocationLimit(0);
#endif
    im.setClipRect(QRect(0,0,im.size().width(),im.size().height()));
    QImage in = im.read();
    if(in.isNull())
    {
        error = im.errorString().toStdString();
        return QImage();
    }
    return in;
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
    QString filename = QCoreApplication::applicationDirPath() + "/" + name;
    if(QFileInfo(filename).exists())
        return filename.toStdString();
    filename = QDir::currentPath() + "/" + name;
    if(QFileInfo(filename).exists())
        return filename.toStdString();
    return name.toStdString();
}

bool load_file_name(void)
{
    device_content_file = find_full_path("device.txt");
    topup_param_file = find_full_path("topup_param.txt");

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
            QString fib_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".fib.gz";
            if(!QFileInfo(qa_file_path).exists())
                continue;
            // setup QA and ISO template        
            fa_template_list.push_back(qa_file_path.toStdString());
            if(QFileInfo(iso_file_path).exists())
                iso_template_list.push_back(iso_file_path.toStdString());
            else
                iso_template_list.push_back(std::string());
            // not all have FIB template
            if(QFileInfo(fib_file_path).exists())
                fib_template_list.push_back(fib_file_path.toStdString());
            else
                fib_template_list.push_back(std::string());

            // find related atlases
            {
                QStringList atlas_list = template_dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
                atlas_list << template_dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
                atlas_list.sort();
                std::vector<std::string> file_list;
                for(auto each : atlas_list)
                    if(QFileInfo(each).baseName() != name_list[i])
                        file_list.push_back((template_dir.absolutePath() + "/" + each).toStdString());
                atlas_file_name_list.push_back(std::move(file_list));
            }

            // find a matching unet model
            {
                model_list_t2w.push_back(std::string());
                auto name = name_list[i].split('_').back();
                if(name == "adult" || name == "neonate")
                    name = "human";
                QDir network_dir = QCoreApplication::applicationDirPath() + "/network";
                for(auto each_model : network_dir.entryList(QStringList("*.net.gz"),QDir::Files|QDir::NoSymLinks))
                    if(QFileInfo(each_model).completeBaseName().contains(name) &&
                       QFileInfo(each_model).completeBaseName().contains("t2w.seg5"))
                    {
                        model_list_t2w.back() = each_model.toStdString();
                        break;
                    }
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

void move_current_dir_to(const std::string& file_name)
{
    auto dir = std::filesystem::path(file_name).parent_path();
    if(dir.empty())
    {
        tipl::out() << "current directory is " << std::filesystem::current_path() << std::endl;
        return;
    }
    tipl::out() << "change current directory to " << dir << std::endl;
    std::filesystem::current_path(dir);
}

int run_action(tipl::program_option<tipl::out>& po)
{
    std::string action = po.get("action");
    tipl::progress prog("run ",action.c_str());
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
        return vis(po);
    tipl::out() << "ERROR: unknown action: " << action << std::endl;
    return 1;
}
int run_action_with_wildcard(tipl::program_option<tipl::out>& po)
{
    std::string source = po.get("source");
    std::string action = po.get("action");
    std::string loop = po.get("loop",source);

    if(action == "atk" || action == "atl" || loop.find('*') == std::string::npos) // atk, atl handle * by itself
    {
        if(run_action(po))
            return 1;
    }
    else
    // loop
    {
        tipl::progress prog("processing loop");
        std::vector<std::string> loop_files;
        if(!tipl::search_filesystem(loop,loop_files))
        {
            tipl::out() << "ERROR: invalid file path " << loop << std::endl;;
            return false;
        }
        tipl::out() << "a total of " << loop_files.size() << " files found" << std::endl;
        std::vector<std::pair<std::string,std::string> > wildcard_list;
        po.get_wildcard_list(wildcard_list);

        tipl::par_for(loop_files.size(),[&](size_t i)
        {
            // clear --other_slices
            other_slices.clear();
            // apply '*' to other arguments
            for(const auto& wildcard : wildcard_list)
            {
                std::istringstream in2(wildcard.second);
                std::string apply_wildcard;
                std::string each;
                while(std::getline(in2,each,','))
                {
                    std::string apply_wildcard_each;
                    if(each.find('*') == std::string::npos)
                        apply_wildcard_each = each;
                    else
                    if(!tipl::match_files(loop,loop_files[i],each,apply_wildcard_each))
                    {
                        tipl::out() << "ERROR: cannot translate " << wildcard.second <<
                                     " at --" << wildcard.first << std::endl;
                        return 1;
                    }
                    if(!apply_wildcard.empty())
                        apply_wildcard += ",";
                    apply_wildcard += apply_wildcard_each;
                }
                tipl::out() << wildcard.second << "->" << apply_wildcard << std::endl;
                po.set(wildcard.first.c_str(),apply_wildcard);
            }
            po.set_used(0);
            po.get("loop");
            if(run_action(po))
                return 1;
            return 0;
        },po.get("loop_thread",1));
    }
    return 0;
}
void check_cuda(std::string& error_msg);
bool has_cuda = false;
int gpu_count = 0;
void init_cuda(void)
{
    if constexpr(tipl::use_cuda)
    {
        std::string cuda_msg;
        check_cuda(cuda_msg);
        if(!has_cuda)
            tipl::out() << cuda_msg << std::endl;
    }
    tipl::out() << version_string().toStdString() << ((has_cuda) ? " CPU/GPU computation enabled " : "") << std::endl;
}
int run_cmd(int ac, char *av[])
{
    tipl::program_option<tipl::out> po;
    try
    {
        tipl::progress prog((version_string().toStdString()+" ").c_str(),__DATE__);
        init_cuda();

        if(!po.parse(ac,av))
        {
            tipl::out() << po.error_msg << std::endl;
            return 1;
        }
        if (!po.check("action"))
            return 1;

        std::shared_ptr<QApplication> gui;
        std::shared_ptr<QCoreApplication> cmd;

        std::string action = po.get("action");
        if ((action == "cnt" && po.get("no_tractogram",1) == 0) || action == "vis")
        {
            tipl::out() << "Starting GUI-based command line interface." << std::endl;
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
                tipl::out() << "ERROR: Cannot find template data." << std::endl;
                return 1;
            }
        }

        if(run_action_with_wildcard(po))
            return 1;
        if(gui.get() && po.get("stay_open",0))
            gui->exec();
    }
    catch(const std::exception& e ) {
        std::cout << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "unknown error occurred" << std::endl;
    }
    return 0;
}

extern console_stream console;

int main(int ac, char *av[])
{
    if(ac > 2)
        return run_cmd(ac,av);
    if(ac == 2 && std::string(av[1]) == "--version")
    {
        std::cout << version_string().toStdString() << " " <<  __DATE__ << std::endl;
        return 1;
    }

    try{
    QApplication a(ac,av);
    if(ac == 2)
    {
        QLocalSocket socket;
        socket.connectToServer("dsi-studio");
        if (socket.waitForConnected(500))
        {
            tipl::out() << "another instance is running, passing file name.";
            socket.write(av[1]);
            socket.flush();
            socket.waitForBytesWritten(500);
            socket.disconnectFromServer();
            return 0;
        }
    }

    tipl::show_prog = true;

    console.attach();

    {
        tipl::progress prog((version_string().toStdString()+" ").c_str(),__DATE__);
        init_cuda();
        init_application();
    }

    MainWindow w;
    w.setWindowTitle(version_string() + " " + __DATE__);
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
        w.openFile(QStringList() << av[1]);

    QLocalServer server;
    if(server.listen("dsi-studio"))
    {
        QObject::connect(&server, &QLocalServer::newConnection, [&server,&w]()
        {
            tipl::out() << "received file name from another instance";
            QLocalSocket *clientSocket = server.nextPendingConnection();
            if (clientSocket)
            {
                clientSocket->waitForReadyRead(500);
                auto file_name = clientSocket->readAll();
                tipl::out() << "file name:" << file_name.begin();
                if(std::filesystem::exists(file_name.begin()))
                {
                    w.openFile(QStringList() << file_name);
                    w.show();
                    w.setWindowState((w.windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
                    w.raise();
                    w.activateWindow();
                }
                clientSocket->disconnectFromServer();
                clientSocket->deleteLater();
            }
        });
    }
    return a.exec();

    }
    catch(const std::bad_alloc&)
    {
        QMessageBox::critical(0,"ERROR","Crash due to insufficient memory.");
    }
    catch(const std::runtime_error& error)
    {
        QMessageBox::critical(0,"ERROR",error.what());
    }
    return 1;
}
