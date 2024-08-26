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
                         t1w_template_list,
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
int img(tipl::program_option<tipl::out>& po);


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

QImage read_qimage(QString filename,std::string& error)
{
    QImageReader im(filename);
    if(!im.canRead())
    {
        error = im.errorString().toStdString();
        tipl::out() << error;
        return QImage();
    }
    tipl::out() << "opening " << filename.toStdString();
    tipl::out() << "size:" << im.size().width() << " " << im.size().height();
#ifdef QT6_PATCH
    im.setAllocationLimit(0);
#endif
    im.setClipRect(QRect(0,0,im.size().width(),im.size().height()));
    QImage in = im.read();
    if(in.isNull())
    {
        error = im.errorString().toStdString();
        tipl::out() << error;
        return QImage();
    }
    return in;
}

tipl::color_image read_color_image(const std::string& filename,std::string& error)
{
    tipl::color_image I;
    auto qimage = read_qimage(filename.c_str(),error);
    if(qimage.isNull())
        return I;
    I << qimage;
    return I;
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
            QString t1w_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".T1W.nii.gz";
            QString tt_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".tt.gz";
            QString fib_file_path = template_dir.absolutePath() + "/" + name_list[i] + ".fz";
            if(!QFileInfo(qa_file_path).exists())
                continue;
            // setup QA and ISO template        
            fa_template_list.push_back(qa_file_path.toStdString());
            if(QFileInfo(iso_file_path).exists())
                iso_template_list.push_back(iso_file_path.toStdString());
            else
                iso_template_list.push_back(std::string());
            if(QFileInfo(t1w_file_path).exists())
                t1w_template_list.push_back(t1w_file_path.toStdString());
            else
                t1w_template_list.push_back(std::string());

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

const char* version_string = "DSI Studio version: Hou \"\xe4\xbe\xaf\"";
int map_ver = 202406;
int src_ver = 202408;
int fib_ver = 202408;

bool init_application(void)
{
    QCoreApplication::setOrganizationName("LabSolver");
    QCoreApplication::setApplicationName(version_string);

    if(tipl::show_prog)
    {
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
        {
            QMessageBox::critical(nullptr,"ERROR","Cannot find template data.");
            return false;
        }

    }
    else
    {
        if(!load_file_name())
        {
            tipl::error() << "cannot find template data." << std::endl;
            return false;
        }
    }
    return true;
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
    if(action == std::string("img"))
        return img(po);
    if(action == std::string("vis"))
        return vis(po);
    tipl::error() << "unknown action: " << action << std::endl;
    return 1;
}
int run_action_with_wildcard(tipl::program_option<tipl::out>& po,int ac, char *av[])
{
    tipl::progress prog("command line");
    std::string source = po.get("source");
    std::string action = po.get("action");


    std::shared_ptr<QCoreApplication> cmd;
    if(av)
    {
        if ((action == "cnt" && po.get("no_tractogram",1) == 0) || action == "vis")
        {
            tipl::out() << "Starting GUI-based command line interface." << std::endl;
            cmd.reset(new QApplication(ac, av));
            tipl::show_prog = true;
        }
        else
            cmd.reset(new QCoreApplication(ac, av));
        if(!init_application())
            return 1;
    }


    std::string loop;
    if(po.has("loop"))
        loop = po.get("loop");
    else
    if(source.find('*') != std::string::npos && action != "atk" && action != "atl" && action != "src" && action != "qc")
        loop = po.get("loop",source);

    if(loop.empty())
    {
        if(run_action(po))
            return 1;
    }
    else
    // loop
    {
        std::vector<std::string> loop_files;
        if(!tipl::search_filesystem<tipl::out,tipl::error>(loop,loop_files))
        {
            tipl::error() << "no file found: " << loop << std::endl;;
            return 1;
        }
        tipl::out() << "a total of " << loop_files.size() << " files found" << std::endl;
        std::vector<std::pair<std::string,std::string> > wildcard_list;
        po.get_wildcard_list(wildcard_list);

        for(size_t i = 0;prog(i,loop_files.size());++i)
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
                        tipl::error() << "cannot translate " << wildcard.second <<
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
            if(run_action(po) == 1)
                return 1;
        }
    }

    if(po.has("stay_open") && cmd.get())
        cmd->exec();
    return prog.aborted() ? 1 : 0;
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
        else
            tipl::out() << "CPU/GPU computation enabled "<< std::endl;
    }
}
extern console_stream console;
int main(int ac, char *av[])
{
    std::cout << version_string << " " << __DATE__ << std::endl;

    if(ac == 2 && std::string(av[1]) == "--version")
        return 0;

    if(ac > 2)
    {
        tipl::program_option<tipl::out> po;
        try
        {
            if(!po.parse(ac,av) || !po.check("action"))
            {
                tipl::error() << po.error_msg << std::endl;
                return 1;
            }
            init_cuda();
            if(run_action_with_wildcard(po,ac,av))
                return 1;
            po.check_end_param<tipl::warning>();
            return 0;
        }
        catch(const std::exception& e ) {
            tipl::error() << e.what() << std::endl;
            return 1;
        }
        catch(...)
        {
            tipl::error() <<"unknown error occurred" << std::endl;
            return 1;
        }
        return 0;
    }


    QApplication a(ac,av);
    try
    {

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

        tipl::progress prog(version_string);


        init_cuda();
        if(!init_application())
            return 1;

        MainWindow w;
        w.setWindowTitle(QString(version_string) + " " + __DATE__);
        // presentation mode
        QStringList fib_list = QDir(QCoreApplication::applicationDirPath()+ "/presentation").
                                entryList(QStringList("*fib.gz") << QString("*.fz"),QDir::Files|QDir::NoSymLinks);
        if(fib_list.size())
        {
            w.hide();
            w.loadFib(QCoreApplication::applicationDirPath() + "/presentation/" + fib_list[0]);
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
        QMessageBox::critical(0,"ERROR","insufficient memory.");
    }
    catch(const std::runtime_error& error)
    {
        QMessageBox::critical(0,"ERROR",error.what());
    }
    return 1;
}
