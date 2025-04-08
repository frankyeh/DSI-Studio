#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <unordered_set>
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


size_t match_volume(tipl::const_pointer_image<3,unsigned char> mask,tipl::vector<3> vs)
{
    if(mask.empty())
        return 0;

    auto get_max_axisl_count = [](tipl::const_pointer_image<3,unsigned char> mask)->size_t
    {
        size_t max_axial_volume = 0;
        for(size_t z = 0;z < mask.depth();++z)
        {
            auto I = mask.slice_at(z);
            size_t count = std::count_if(I.begin(),I.end(),[](unsigned char v){return v > 0;});
            if(count > max_axial_volume)
                max_axial_volume = count;
        }
        return max_axial_volume;
    };
    float cur_volume = get_max_axisl_count(mask)*vs[0]*vs[1];
    std::vector<std::pair<std::string,float> > template_volume = {
        {"human",22086.2f},{"human_neonate",8047.0f},{"rhesus",3556},{"marmoset",646.72},{"rat",367.44},{"mouse",111.75}};

    /*
    for(size_t i = 0;i < fa_template_list.size();++i)
    {
        tipl::io::gz_nifti read;
        tipl::image<3,unsigned char> I;
        if(!read.load_from_file(fa_template_list[i].c_str()))
            continue;
        read >> I;
        auto tvs = read.get_voxel_size<3>();
        tipl::out() << fa_template_list[i] << " : " << get_max_axisl_count(tipl::make_image(I.data(),I.shape()))*tvs[0]*tvs[1];
    }
    */

    std::string best_template_name = template_volume[0].first;
    float best_dif = std::fabs(template_volume[0].second-cur_volume);
    for(size_t i = 1;i < template_volume.size();++i)
    {
        float cur_dif = std::fabs(template_volume[i].second-cur_volume);
        if(cur_dif < best_dif)
        {
            best_dif = cur_dif;
            best_template_name = template_volume[i].first;
        }
    }
    tipl::out() << "match slice volume: " << cur_volume << " → " << best_template_name;
    for(size_t i = 0;i < fa_template_list.size();++i)
        if(tipl::contains(tipl::split(std::filesystem::path(fa_template_list[i]).stem().string(),'.').front(),best_template_name))
            return i;
    return 0;
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

        QStringList dir_list = dir.entryList(QStringList("*"),QDir::Dirs|QDir::NoSymLinks);
        dir_list.sort();
        QStringList name_list;
        for(auto each : {"human","chimpanzee","rhesus","marmoset","rat","mouse"})
        {
            for(size_t i = 0;i < dir_list.size();++i)
                if(dir_list[i].contains(each))
                {
                    name_list << dir_list[i];
                    dir_list[i].clear();
                }
        }
        for(size_t i = 0;i < dir_list.size();++i)
            if(!dir_list[i].isEmpty())
                name_list << dir_list[i];

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
            iso_template_list.push_back(iso_file_path.toStdString());
            t1w_template_list.push_back(t1w_file_path.toStdString());
            fib_template_list.push_back(fib_file_path.toStdString());

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

const char* version_string = "Hou \"\xe4\xbe\xaf\"";
int map_ver = 202408;
int src_ver = 202408;
int fib_ver = 202408;

bool init_application(void)
{
    QCoreApplication::setOrganizationName("LabSolver");
    QCoreApplication::setApplicationName(QString("DSI Studio ") + version_string);

    if(tipl::show_prog)
    {
        #ifdef __APPLE__
        QFont font;
        font.setFamily(QString::fromUtf8("Arial"));
        QApplication::setFont(font);
        #else
        QSettings settings;
        QString style = settings.value("styles","Fusion").toString();
        if(style != "default" && !style.isEmpty())
            QApplication::setStyle(style);

        if(QApplication::palette().color(QPalette::Window).lightnessF() < 0.5)
        {
            // Iterate over available styles to find one that produces a white window background.
            QStringList availableStyles = QStyleFactory::keys();
            bool foundLightStyle = false;
            QString lightStyleName;
            for(const QString &s : availableStyles)
            {
                QApplication::setStyle(s);
                if(QApplication::palette().color(QPalette::Window).lightnessF() >= 0.5)
                {
                    foundLightStyle = true;
                    lightStyleName = s;
                    break;
                }
            }
            // If a light style was found, update the application style and save it to QSettings.
            if(foundLightStyle)
            {
                QApplication::setStyle(lightStyleName);
                settings.setValue("styles", lightStyleName);
            }
        }
        #endif

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


int rec(tipl::program_option<tipl::out>& po);
int trk(tipl::program_option<tipl::out>& po);
int src(tipl::program_option<tipl::out>& po);
int ana(tipl::program_option<tipl::out>& po);
int exp(tipl::program_option<tipl::out>& po);
int atl(tipl::program_option<tipl::out>& po);
int db(tipl::program_option<tipl::out>& po);
int tmp(tipl::program_option<tipl::out>& po);
int cnt(tipl::program_option<tipl::out>& po);
int vis(tipl::program_option<tipl::out>& po);
int ren(tipl::program_option<tipl::out>& po);
int cnn(tipl::program_option<tipl::out>& po);
int qc(tipl::program_option<tipl::out>& po);
int reg(tipl::program_option<tipl::out>& po);
int atk(tipl::program_option<tipl::out>& po);
int xnat(tipl::program_option<tipl::out>& po);
int img(tipl::program_option<tipl::out>& po);

static const std::unordered_map<std::string, int(*)(tipl::program_option<tipl::out>&)> action_map = {
    {"rec", rec},
    {"trk", trk},
    {"src", src},
    {"ana", ana},
    {"exp", exp},
    {"atl", atl},
    {"db", db},
    {"tmp", tmp},
    {"cnt", cnt},
    {"vis", vis},
    {"ren", ren},
    {"cnn", cnn},
    {"qc", qc},
    {"reg", reg},
    {"atk", atk},
    {"xnat", xnat},
    {"img", img}
};


int run_action(tipl::program_option<tipl::out>& po)
{
    std::string action = po.get("action");
    if(!tipl::show_prog && action =="vis")
    {
        tipl::error() << action << " is only supported at GUI's console";
        return 1;
    }
    tipl::progress prog("run ",action.c_str());
    auto it = action_map.find(action);
    if (it != action_map.end())
        return it->second(po);
    tipl::error() << "unknown action: " << action;
    return 1;
}
void check_cuda(std::string& error_msg);
bool has_cuda = false;
int run_action_with_wildcard(tipl::program_option<tipl::out>& po,int ac, char *av[])
{
    tipl::progress prog("command line");
    std::string action = po.get("action");
    std::shared_ptr<QCoreApplication> cmd;
    if(av)
    {
        cmd.reset(new QCoreApplication(ac, av));
        if(!init_application())
            return 1;
        if constexpr(tipl::use_cuda)
        {
            std::string cuda_msg;
            check_cuda(cuda_msg);
            if(has_cuda)
                tipl::out() << "CPU/GPU computation enabled "<< std::endl;
        }
    }


    std::string loop;
    if(po.has("loop"))
        loop = po.get("loop");
    else
    {
        std::unordered_set<std::string> excluded_actions = {"atk", "src", "qc", "db", "tmp"};
        if (excluded_actions.find(action) == excluded_actions.end() &&
            po.has("source") &&
            po.get("source").find('*') != std::string::npos)
            loop = po.get("source");
    }
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
int gpu_count = 0;
extern console_stream console;
int main(int ac, char *av[])
{
    std::string show_ver = std::string("DSI Studio version: ") + version_string + " " + __DATE__;
    std::cout << show_ver << std::endl;

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
            if (socket.waitForConnected(5000))
            {
                tipl::out() << "another instance is running, passing file name.";
                socket.write(av[1]);
                socket.flush();
                socket.waitForBytesWritten(5000);
                if(socket.waitForReadyRead(5000))
                {
                    if (socket.readAll() == "OKAY")
                    {
                        socket.disconnectFromServer();
                        return 0;
                    }
                }
                socket.disconnectFromServer();
            }
        }

        tipl::show_prog = true;
        console.attach();

        tipl::progress prog(show_ver);


        if(!init_application())
            return 1;

        if constexpr(tipl::use_cuda)
        {
            std::string cuda_msg;
            check_cuda(cuda_msg);
            if(!has_cuda)
            {
                QMessageBox::critical(nullptr,"ERROR",cuda_msg.c_str());
                return 1;
            }
            else
                tipl::out() << "CPU/GPU computation enabled "<< std::endl;
        }

        MainWindow w;
        w.setWindowTitle(show_ver.c_str());
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
                    if(!tipl::progress::is_running())
                    {
                        if(std::filesystem::exists(file_name.begin()))
                        {
                            w.openFile(QStringList() << file_name);
                            w.show();
                            w.setWindowState((w.windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
                            w.raise();
                            w.activateWindow();
                        }
                        clientSocket->write("OKAY");
                    }
                    else
                        clientSocket->write("BUSY");
                    clientSocket->flush();
                    clientSocket->waitForBytesWritten(500);
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
