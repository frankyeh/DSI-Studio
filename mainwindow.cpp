#include <QFileDialog>
#include <QDir>
#include <QUrl>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QAction>
#include <QTextStream>
#include <QHeaderView>
#include <QStyleFactory>
#include <QNetworkInterface>
#include <QNetworkRequest>
#include <QSysInfo>
#include <QStandardPaths>
#include <QDialog>
#include <QLineEdit>
#include <QUuid>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPointer>

#include <QJsonDocument>
#include <QMap>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ai_agent.hpp"
#include "regtoolbox.h"
#include "reconstruction/reconstruction_window.h"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"
#include "dicom/dicom_parser.h"
#include "view_image.h"
#include "mapping/atlas.hpp"
#include "fib_data.hpp"
#include "connectometry/group_connectometry_analysis.h"
#include "connectometry/createdbdialog.h"
#include "connectometry/db_window.h"
#include "connectometry/group_connectometry.hpp"
#include "libs/dsi/image_model.hpp"
#include "manual_alignment.h"
#include "auto_track.h"
#include "xnat_dialog.h"
#include "console.h"
#include "fiber_data_hub.hpp"


QString access_token;
extern MainWindow* main_window;
void checkForVersionSpecificBugs(const QString& bugListText)
{
    QDate compDate = QDate::fromString(__DATE__, "MMM dd yyyy");
    if (!compDate.isValid())
        return;

    auto match_date = [&](auto op,auto date) -> bool
    {
        QDate rangeDate = QDate::fromString(date, "M/d/yyyy");
        if (!rangeDate.isValid())
            rangeDate = QDate::fromString(date, "MM/dd/yyyy");
        if (!rangeDate.isValid())
            return false;
        if (op == ">=") return (compDate >= rangeDate);
        else if (op == "<=") return (compDate <= rangeDate);
        else if (op == ">") return (compDate > rangeDate);
        else if (op == "<") return (compDate < rangeDate);
        return false;
    };

    QStringList matchingBugs;
    for (auto line : bugListText.split('\n', Qt::SkipEmptyParts))
    {
        if (!line.contains("versions"))
            continue;
        if (line.contains("windows") && !QSysInfo::productType().contains("windows"))
            continue;
        if (line.contains("macos") && !QSysInfo::productType().contains("macos"))
            continue;
        if (line.contains("ubuntu") && !QSysInfo::productType().contains("ubuntu"))
            continue;

        int start = line.indexOf('['), end = line.indexOf(']');
        if (start == -1 || end == -1 || end <= start)
            continue;
        QString spec = line.mid(start + 1, end - start - 1).trimmed();
        QString desc = line.mid(end + 1).trimmed();
        if (!spec.startsWith("versions ") || desc.isEmpty())
            continue;
        QStringList conds = spec.trimmed().split(' ', Qt::SkipEmptyParts);
        bool match = true;
        for(size_t i = 2;i < conds.size(); i += 2)
            if(!match_date(conds[i-1].trimmed(),conds[i].trimmed()))
            {
                match = false;
                break;
            }
        if (match)
            matchingBugs.append(desc);
    }

    if (!matchingBugs.isEmpty())
        QMessageBox::critical(nullptr, "Program Update Recommended",
                              "This DSI Studio version is affected by the following issues:\n\n- " +
                              matchingBugs.join("\n- ") +
                              "\n\nIt is highly recommended to update DSI Studio to the latest version to avoid these issues.");
}



extern std::vector<std::filesystem::path> fib_template_list;
std::vector<tracking_window*> tracking_windows;
MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);

    ui->styles->addItems(QStringList("default") << QStyleFactory::keys());
    ui->styles->setCurrentText(settings.value("styles","Fusion").toString());

    for(auto* table : {ui->recentSrc,ui->recentFib})
    {
        table->setColumnCount(2);
        table->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
        table->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
        table->setAlternatingRowColors(true);
    }
    QObject::connect(ui->recentFib,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_fib_at(int,int)));
    QObject::connect(ui->recentSrc,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_src_at(int,int)));
    updateRecentList();

    auto workdir_list = settings.value("WORK_PATH").toStringList();
    if (!settings.contains("WORK_PATH"))
        ui->workDir->addItem(QDir::currentPath());

    tipl::qt::working_dirs << QUrl::fromLocalFile(QStandardPaths::writableLocation(QStandardPaths::DesktopLocation));
    tipl::qt::working_dirs << QUrl::fromLocalFile(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    tipl::qt::working_dirs << QUrl::fromLocalFile(QStandardPaths::writableLocation(QStandardPaths::DownloadLocation));

    for(const auto& each : workdir_list)
        if(QFileInfo::exists(each))
        {
            ui->workDir->addItem(each);
            tipl::qt::working_dirs << QUrl::fromLocalFile(each);
        }
    if(!ui->workDir->count())
        ui->workDir->addItem(QDir::currentPath());
    for(auto& each : fib_template_list)
    {
        QString name = std::filesystem::path(each).stem().string().c_str();
        ui->template_list->addItem(name);
    }
    ui->tabWidget->setCurrentIndex(0);
    ui->template_list->setCurrentRow(0);

    {
        news = settings.value("login_news").toString();
        address = settings.value("login_address",QLocale::countryToString(QLocale::system().country())).toString();
        host_name = settings.value("login_hostname",QHostInfo::localHostName().isEmpty() ? QSysInfo::machineHostName() : QHostInfo::localHostName()).toString();
        username = settings.value("login_id",
                QDir(QStandardPaths::writableLocation(QStandardPaths::HomeLocation)).dirName() + "," +
                QUuid::createUuid().toString(QUuid::WithoutBraces) + "," +
                QLocale::countryToString(QLocale::system().country())).toString();
        if(!settings.contains("login_id"))
            settings.setValue("login_id",username);

        {
            QString licenseText;
            {
                QFile licenseFile(QApplication::applicationDirPath() + "/LICENSE");
                if(!licenseFile.open(QIODevice::ReadOnly))
                    throw std::runtime_error("cannot locate license file");
                licenseText = licenseFile.readAll();
            }

            QDialog *dialog = new QDialog(this);
            dialog->setWindowTitle("DSI Studio");
            dialog->setWindowFlags(Qt::Dialog | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
            dialog->setModal(true);

            QHBoxLayout *main_layout = new QHBoxLayout;
            dialog->setLayout(main_layout);
            QVBoxLayout *left_layout = new QVBoxLayout;
            QVBoxLayout *right_layout = new QVBoxLayout;

            {
                auto title = new QLabel("License Information:");
                title->setStyleSheet("font-weight: bold;");
                right_layout->addWidget(title);
            }

            {
                QTextBrowser *licenseBrowser = new QTextBrowser;
                licenseBrowser->setMarkdown(licenseText);
                licenseBrowser->setReadOnly(true);
                licenseBrowser->setOpenExternalLinks(true);
                right_layout->addWidget(licenseBrowser);
            }


            {
                QHBoxLayout *h_layout = new QHBoxLayout;
                h_layout->addWidget(new QLabel("Registering Entity:"));
                auto line_edit = new QLineEdit(username);
                line_edit->setReadOnly(true);
                h_layout->addWidget(line_edit);
                right_layout->addLayout(h_layout);

            }

            {
                auto registry_info = new QLabel("Registering Information: " + host_name + "," + address);
                registry_info->setWordWrap(true);
                right_layout->addWidget(registry_info);
            }

            {
                auto note = new QLabel("By clicking 'Accept & Sign in', you agree to the licensing terms and sign in using the registration registry and information.");
                note->setWordWrap(true);
                note->setStyleSheet("font-weight: bold;");
                right_layout->addWidget(note);
            }

            {
                QPushButton *closeButton = new QPushButton("Accept && Sign in");
                closeButton->setStyleSheet("font-size: 14pt; font-weight: bold;");
                auto h = closeButton->sizeHint().height() * 1.5f;
                closeButton->setFixedHeight(h);
                connect(closeButton, &QPushButton::clicked, dialog, &QDialog::close);
                connect(closeButton, &QPushButton::clicked, this, &MainWindow::login);
                QPushButton *exitButton = new QPushButton("Decline && Exit");
                exitButton->setFixedHeight(h);
                exitButton->setMaximumWidth(100);
                connect(exitButton, &QPushButton::clicked, dialog, &QDialog::close);
                connect(exitButton, &QPushButton::clicked, this, &MainWindow::close);
                QHBoxLayout *h_layout = new QHBoxLayout;
                h_layout->setSpacing(0);
                h_layout->addWidget(closeButton);
                h_layout->addWidget(exitButton);
                right_layout->addLayout(h_layout);
            }


            {
                auto title = new QLabel("News and Updates:");
                title->setStyleSheet("font-weight: bold;");
                left_layout->addWidget(title);
            }

            {
                QTextBrowser *NewsBrowser = new QTextBrowser;
                NewsBrowser->setMarkdown(news);
                NewsBrowser->setReadOnly(true);
                NewsBrowser->setOpenExternalLinks(true);
                left_layout->addWidget(NewsBrowser);
                checkForVersionSpecificBugs(news);
            }

            main_layout->addLayout(left_layout, 1);
            main_layout->addLayout(right_layout, 1);

            dialog->resize(1024,800);
            dialog->show();

        }
    }

    // Connect command buttons
    {
        for(auto* button : findChildren<QPushButton*>())
        {
            QString tip = button->statusTip();
            if(!tip.startsWith("run "))
                continue;
            std::string command_name = tip.mid(4).trimmed().toStdString();
            connect(button,&QPushButton::clicked,this,[this,command_name]
            {
                if(!command({command_name}) && !error_msg.empty())
                    QMessageBox::critical(this,"ERROR",QString::fromStdString(error_msg));
            });
        }
    }

    ai_agent = new AIAgent(this);
}

extern const char* version_string;
void MainWindow::login(void)
{
    setWindowTitle(windowTitle() + "(Offline)");
    QDnsLookup *dns = new QDnsLookup(this);
    dns->setType(QDnsLookup::TXT);
    dns->setName(DSI_STUDIO_LOGIN);
    connect(dns, &QDnsLookup::finished, [=]()
    {
        if (dns->error() != QDnsLookup::NoError)
        {
            qWarning() << "cannot login due to DNS lookup error:" << dns->errorString();
            dns->deleteLater();
            return;
        }
        for (const auto &record : dns->textRecords())
        {
            info = QString(record.values().join("")).split(',');
            break;
        }
        // update news
        if(info.size() >= 1)
        {
            auto reply = get(info[0]);
            connect(reply.get(), &QNetworkReply::finished, this, [=]()
            {
                if (reply->error() == QNetworkReply::NoError)
                {
                    settings.setValue("login_news",QString(reply->readAll()));
                    settings.sync();
                }
            });
        }
        // update registering information
        if(info.size() >= 3)
        {
            auto reply = get(info[1]);
            connect(reply.get(), &QNetworkReply::finished, this, [=]()
            {
                if (reply->error() == QNetworkReply::NoError)
                {
                    auto reply2 = get(info[2].arg(QJsonDocument::fromJson(QString(reply->readAll()).toUtf8()).object().value("ip").toString()));
                    connect(reply2.get(), &QNetworkReply::finished, this, [=]()
                    {
                        if (reply2->error() == QNetworkReply::NoError)
                        {
                            QJsonObject jsonObject = QJsonDocument::fromJson(QString(reply2->readAll()).toUtf8()).object();
                            if(!jsonObject.value("city").toString().isEmpty())
                            {
                                settings.setValue("login_address",jsonObject.value("city").toString() + "," +
                                                                  jsonObject.value("region").toString() + "," +
                                                                  jsonObject.value("countryCode").toString() + " " +
                                                                  jsonObject.value("zip").toString() + " ");
                                settings.setValue("login_hostname",jsonObject.value("as").toString());
                                settings.sync();
                            }

                        }
                    });
                }
            });
        }
        if(info.size() >= 5)
        {
            QNetworkRequest request(QUrl(info[3].toStdString().c_str()));
            request.setRawHeader("Content-Type", "application/json");
            QJsonObject data;
            data["name"] = username;
            data["fn"] = host_name;
            data["os"] = QSysInfo::productType() + QSysInfo::productVersion();
            data["version"] = QString(version_string) + " " + __DATE__;
            data["address"] = address;
            auto reply = manager.post(request, QJsonDocument(data).toJson());
            QObject::connect(reply, &QNetworkReply::finished, [=]()
            {
                if (reply->error() == QNetworkReply::NoError)
                {
                    QString result = reply->readAll();
                    if(result.startsWith('{')) // json format
                    {
                        auto data = QJsonDocument::fromJson(result.toUtf8()).object();
                        if (data.contains("title"))
                            setWindowTitle(windowTitle().remove("(Offline)") + " " + data["title"].toString());
                        if (data.contains("token"))
                            access_token = data["token"].toString();
                        if (data.contains("notice"))
                            QMessageBox::critical(this,"Notice",data["notice"].toString());
                    }
                    else
                        setWindowTitle(windowTitle().remove("(Offline)") + " " + result);
                }
                reply->deleteLater();
            });
        }
        dns->deleteLater();
    });
    dns->lookup();
}
void MainWindow::openFile(QStringList file_names)
{
    if(file_names.isEmpty() || file_names[0].isEmpty())
        return;
    QString file_name = file_names[0];
    auto name = file_name.toLower();
    if(!QFileInfo::exists(file_name))
    {
        if(file_name[0] == '-') // Mac pass a variable
            return;
        QMessageBox::critical(this,"ERROR",QString("Cannot find ") +
        file_name + " at current dir: " + QDir::current().dirName());
    }
    else
    {
        if(name.endsWith(".csv"))
        {
            auto lines = tipl::read_text_file(tipl::qt::to_path(file_name));
            if(lines.empty() || !tipl::begins_with(lines[0],"open_fib,"))
            {
                QMessageBox::critical(this,"ERROR","invalid command csv file");
                return;
            }
            if(!loadFib(QString::fromStdString(tipl::split(lines[0],',')[1])))
                return;
            if(!tracking_windows.empty())
            {
                for(size_t i = 1;i < lines.size();++i)
                    if(!tracking_windows.back()->command(tipl::split(lines[i],',')))
                    {
                        if(!tracking_windows.back()->error_msg.empty())
                        QMessageBox::critical(this,"ERROR",tracking_windows.back()->error_msg.c_str());
                        return;
                    }
            }
        }
        else
        if(name.endsWith(".tt.gz") ||
           name.endsWith(".trk") ||
           name.endsWith(".trk.gz"))
        {
            auto file_list = QFileInfo(file_name).dir().entryList(QStringList("*fz"),QDir::Files|QDir::NoSymLinks);
            file_list << QFileInfo(file_name).dir().entryList(QStringList("*fib.gz"),QDir::Files|QDir::NoSymLinks);
            if(file_list.size() == 1)
            {
                if(loadFib(QFileInfo(file_name).absolutePath() + "/" + file_list[0]))
                    for(const auto& each:file_names)
                        tracking_windows.back()->command({"open_tract",each.toStdString()});
            }
            else
                loadFib(file_name);
        }
        else
        if(name.endsWith("fib.gz") ||
           name.endsWith(".fz") ||
           name.endsWith(".dz") ||
           name.endsWith("tck"))
        {
            if(name.endsWith("db.fib.gz") ||
               name.endsWith("db.fz") ||
               name.endsWith(".dz"))
            {
                std::shared_ptr<group_connectometry_analysis> database(new group_connectometry_analysis);
                if(!database->load_database(file_name.toStdString().c_str()))
                {
                    QMessageBox::critical(
                        this,"ERROR",database->error_msg.c_str());
                    return;
                }

                auto* db = new db_window(this,database);
                db->setWindowTitle(file_name);
                db->setAttribute(Qt::WA_DeleteOnClose);
                db->show();

            }
            else
                loadFib(file_name);
        }
        else
        if(name.endsWith("src.gz") || name.endsWith(".sz"))
        {
            loadSrc(file_names);
        }
        else
        if(name.endsWith(".nhdr") ||
           name.endsWith(".nrrd") ||
           name.endsWith(".nii") ||
           name.endsWith(".nii.gz") ||
                name.endsWith(".dcm") ||
                name.endsWith(".nz") ||
                name.endsWith(".mz"))
        {
            loadNii(file_names);
        }
        else {
            QMessageBox::critical(this,"ERROR","Unsupported file extension");
        }
    }
}
void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->hasUrls())
    {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *event)
{
    event->acceptProposedAction();
    QList<QUrl> droppedUrls = event->mimeData()->urls();
    int droppedUrlCnt = droppedUrls.size();
    QStringList files;
    for(int i = 0; i < droppedUrlCnt; i++)
        files << droppedUrls[i].toLocalFile();
    openFile(files);
}

void MainWindow::open_fib_at(int row,int)
{
    if(auto* item = ui->recentFib->item(row,0))
        loadFib(item->data(Qt::UserRole).toString());
}

void MainWindow::open_src_at(int row,int)
{
    if(auto* item = ui->recentSrc->item(row,0))
        loadSrc({item->data(Qt::UserRole).toString()});
}


void MainWindow::closeEvent(QCloseEvent* event)
{
    auto windows = tracking_windows;
    for(auto* window : windows)
        if(window && !window->close())
        {
            event->ignore();
            return;
        }
    ai_agent->close(); // its own closeEvent stops every running agent process; ~AIAgent() alone would not
    QMainWindow::closeEvent(event);
}


MainWindow::~MainWindow()
{
    console.log_window = nullptr;
    QStringList workdir_list;
    for (int index = 0;index < 10 && index < ui->workDir->count();++index)
        workdir_list << ui->workDir->itemText(index);
    auto current = ui->workDir->currentIndex();
    if(current > 0 && current < workdir_list.size())
        workdir_list.move(current,0);
    settings.setValue("WORK_PATH", workdir_list);
    delete ui;

}


void MainWindow::updateRecentList()
{
    auto update = [this](QTableWidget* table,const char* key)
    {
        auto files = settings.value(key).toStringList();
        table->clearContents();
        table->setRowCount(files.size());

        for(int row = 0;row < files.size();++row)
        {
            QFileInfo info(files[row]);
            table->setRowHeight(row,20);
            table->setItem(row,0,new QTableWidgetItem(info.fileName()));
            table->setItem(row,1,new QTableWidgetItem(info.absolutePath()));

            for(int col = 0;col < 2;++col)
            {
                auto* item = table->item(row,col);
                item->setFlags(item->flags() & ~Qt::ItemIsEditable);
                item->setData(Qt::UserRole,files[row]);
                if(!info.exists())
                    item->setForeground(Qt::gray);
            }
        }
        table->setHorizontalHeaderLabels({"File Name","Directory"});
    };

    update(ui->recentFib,"recentFibFileList");
    update(ui->recentSrc,"recentSrcFileList");
}


void MainWindow::addRecent(QString filename,const char* key)
{
    auto files = settings.value(key).toStringList();
    QFileInfo info(filename);
    for(int index = files.size()-1;index >= 0;--index)
        if(QFileInfo(files[index]) == info)
            files.removeAt(index);
    files.prepend(QDir::toNativeSeparators(info.absoluteFilePath()));
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue(key,files);
    updateRecentList();
}
void MainWindow::addFib(QString filename)
{
    addRecent(filename,"recentFibFileList");
}
void MainWindow::addSrc(QString filename)
{
    addRecent(filename,"recentSrcFileList");
}
void shift_track_for_tck(std::vector<std::vector<float> >& loaded_tract_data,tipl::shape<3>& geo);
extern QByteArray default_geo,default_state;
QString command_window_id(QWidget* window)
{
    // the one place that maps a widget to its AI-addressable window type; everything downstream
    // that needs the type back out of an id string uses command_window_type(const QString&) instead
    const char* type =
        qobject_cast<MainWindow*>(window) ? "main" :
        qobject_cast<tracking_window*>(window) ? "tracking" :
        qobject_cast<reconstruction_window*>(window) ? "recon" :
        qobject_cast<group_connectometry*>(window) ? "connectometry" :
        qobject_cast<view_image*>(window) ? "image" : nullptr;
    return type ? command_window_id(window,type) : QString();
}
QString command_window_type(const QString& id)
{
    if(id == "main")
        return "main";
    if(id.startsWith("tracking"))
        return "tracking";
    if(id.startsWith("recon"))
        return "recon";
    if(id.startsWith("connectometry"))
        return "connectometry";
    if(id.startsWith("image"))
        return "image";
    return QString();
}
void MainWindow::report_and_target_window(QWidget* window)
{
    auto id = command_window_id(window);
    tipl::out() << "window created, id: " << id.toStdString(); // id itself is "<type><hex>", the type needs no separate extraction here
    ai_agent->update_current_window(window);
}
bool MainWindow::loadFib(QString filename)
{
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(tipl::qt::to_path(filename)))
    {
        error_msg = new_handle->error_msg;
        if(!new_handle->error_msg.empty())
            QMessageBox::critical(this,"ERROR",new_handle->error_msg.c_str());
        return false;
    }
    tracking_windows.push_back(new tracking_window(this,new_handle));
    report_and_target_window(tracking_windows.back());
    tracking_windows.back()->setAttribute(Qt::WA_DeleteOnClose);
    tracking_windows.back()->setWindowTitle(filename);
    if(filename.contains("/presentation/"))
    {
        tracking_windows.back()->command({"load_workspace",QFileInfo(filename).absolutePath().toStdString()});
        tracking_windows.back()->command({"presentation_mode"});
    }
    else
    if(!filename.contains(QCoreApplication::applicationDirPath()))
    {
        addFib(filename);
        add_work_dir(QFileInfo(filename).absolutePath());
    }
    tracking_windows.back()->showNormal();
    tracking_windows.back()->resize(1200,700);
    if(tipl::ends_with(filename.toStdString(),{".trk.gz",".trk",".tck",".tt.gz"}))
    {
        tracking_windows.back()->command({"open_tract",filename.toStdString()});
        if(filename.endsWith(".tck"))
        {
            tipl::shape<3> geo;
            shift_track_for_tck(tracking_windows.back()->tractWidget->tract_models.back()->get_tracts(),geo);
        }
    }
    if(!default_geo.size())
        default_geo = tracking_windows.back()->saveGeometry();
    if(!default_state.size())
        default_state = tracking_windows.back()->saveState();

    QFileInfo info(filename);
    auto base = info.completeBaseName();
    if(int p = base.lastIndexOf('_');!filename.endsWith("_dseg.nii.gz",Qt::CaseInsensitive) && p >= 0)
    {
        auto dseg_file = info.dir().filePath(
            base.left(p)+"_dseg.nii.gz");
        if(QFileInfo::exists(dseg_file))
            tracking_windows.back()->command({"open_region",dseg_file.toUtf8().constData()});
    }
    return true;
}
void MainWindow::loadNii(QStringList file_names)
{
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if(!dialog->open(file_names))
    {
        delete dialog;
        return;
    }
    report_and_target_window(dialog);
    dialog->show();
}

bool MainWindow::loadSrc(QStringList filenames)
{
    if(filenames.empty())
    {
        error_msg = "cannot find SRC.gz files in the directory. Please create SRC files first.";
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
        return false;
    }
    try
    {
        tipl::progress prog("SRC reconstruction");
        reconstruction_window* new_mdi = new reconstruction_window(filenames,this);
        report_and_target_window(new_mdi);
        new_mdi->setAttribute(Qt::WA_DeleteOnClose);
        new_mdi->show();
        if(filenames.size() == 1)
        {
            addSrc(filenames[0]);
            add_work_dir(QFileInfo(filenames[0]).absolutePath());
        }
    }
    catch(const std::runtime_error& error)
    {
        error_msg = error.what();
        if(!tipl::prog_aborted)
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
        return false;
    }
    return true;
}

void MainWindow::open_DWI(QStringList filenames)
{
    if(filenames.isEmpty() || filenames[0].isEmpty())
        return;
    tipl::progress prog("Open DWI");
    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    if(QFileInfo(filenames[0]).completeBaseName() == "subject")
    {
        tipl::io::bruker_info subject_file;
        if(!subject_file.load_from_file(filenames[0].toStdString().c_str()))
            return;
        QString dir = QFileInfo(filenames[0]).absolutePath();
        filenames.clear();
        for(unsigned int i = 1;i < 100;++i)
        if(QDir(dir + "/" +QString::number(i)).exists())
        {
            bool is_dwi =false;
            // has dif info in the method file
            {
                tipl::io::bruker_info method_file;
                QString method_name = dir + "/" +QString::number(i)+"/method";
                if(method_file.load_from_file(tipl::qt::to_path(method_name)) &&
                   method_file["PVM_DwEffBval"].length())
                    is_dwi = true;
            }
            // has dif info in the imnd file
            {
                tipl::io::bruker_info imnd_file;
                QString imnd_name = dir + "/" +QString::number(i)+"/imnd";
                if(imnd_file.load_from_file(tipl::qt::to_path(imnd_name)) &&
                   imnd_file["IMND_diff_b_value"].length())
                    is_dwi = true;
            }
            if(is_dwi)
                filenames.push_back(dir + "/" +QString::number(i)+"/pdata/1/2dseq");
        }
        if(filenames.size() == 0)
        {
            QMessageBox::critical(this,"ERROR","No diffusion data in this subject");
            return;
        }
        std::string file_name(subject_file["SUBJECT_study_name"]);
        file_name.erase(std::remove(file_name.begin(),file_name.end(),' '),file_name.end());
        dicom_parser* dp = new dicom_parser(filenames,this);
        dp->set_name(dir + "/" + file_name.c_str() + ".sz");
        dp->setAttribute(Qt::WA_DeleteOnClose);
        dp->showNormal();
        return;
    }

    if(filenames[0].endsWith(".dcm"))
    {
        QString sel = QString("*.")+QFileInfo(filenames[0]).suffix();
        QDir directory = QFileInfo(filenames[0]).absoluteDir();
        QStringList file_list = directory.entryList(QStringList(sel),QDir::Files|QDir::NoSymLinks);
        if(file_list.size() > filenames.size())
        {
            QString msg =
              QString("There are %1 %2 files in the directory. Select all?").arg(file_list.size()).arg(QFileInfo(filenames[0]).suffix());
            int result = QMessageBox::information(this,"Input images",msg,
                                     QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::Yes)
            {
                filenames = file_list;
                for(int index = 0;index < filenames.size();++index)
                    filenames[index] = directory.absolutePath() + "/" + filenames[index];
            }
        }
    }
    dicom_parser* dp = new dicom_parser(filenames,this);
    dp->setAttribute(Qt::WA_DeleteOnClose);
    dp->showNormal();
    if(dp->dwi_files.empty())
        dp->close();
}

std::filesystem::path rename_dicom(const std::filesystem::path& file_name,std::filesystem::path output,std::string& error_msg);

void MainWindow::add_work_dir(QString dir)
{
    if(ui->workDir->findText(dir) != -1)
        ui->workDir->removeItem(ui->workDir->findText(dir));
    ui->workDir->insertItem(0,dir);
    ui->workDir->setCurrentIndex(0);

    if(tipl::qt::working_dirs.indexOf(dir) != -1)
        tipl::qt::working_dirs.remove(tipl::qt::working_dirs.indexOf(dir));
    tipl::qt::working_dirs << dir;

}

QString MainWindow::work_dir() const
{
    return ui->workDir->currentText();
}



std::vector<std::filesystem::path> rename_dicom_at_dir(std::filesystem::path path,
                                                       std::filesystem::path output,std::string& error_msg);

bool parse_dwi(const std::vector<std::filesystem::path>& file_list,
               std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg);
std::filesystem::path get_dicom_output_name(const std::filesystem::path& file_name,
                                            const std::string& file_extension, bool add_path);
QStringList search_files(QString dir,QString filter);

void MainWindow::on_workDir_currentTextChanged(const QString &arg1)
{
    if(!arg1.isEmpty())
        QDir::setCurrent(arg1);
}

bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>& trans);

void MainWindow::on_linear_reg_clicked()
{
    QStringList filename1 = tipl::qt::open_image_files(this,ui->workDir->currentText(),
                                            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename1.isEmpty())
        return;


    QStringList filename2 = tipl::qt::open_image_files(this,QFileInfo(filename1[0]).absolutePath(),
                                            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename2.isEmpty())
        return;


    tipl::image<3> ref1,ref2;
    tipl::vector<3> vs1,vs2;
    tipl::matrix<4,4> t1,t2;
    if(!load_image_from_files(filename1,ref1,vs1,t1) ||
       !load_image_from_files(filename2,ref2,vs2,t2))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,tipl::reg::subject_image_pre(tipl::image<3>(ref1)),tipl::image<3,unsigned char>(),vs1,
                                                                       tipl::reg::template_image_pre(tipl::image<3>(ref2)),tipl::image<3,unsigned char>(),vs2,tipl::reg::affine,tipl::reg::mutual_info));
    manual->from_T = t1;
    manual->to_T = t2;

    if(manual->exec() != QDialog::Accepted)
        return;
}

std::string quality_check_src_files(const std::vector<std::filesystem::path>& file_list,
                                    bool check_btable,bool use_template,unsigned int template_id);
std::string quality_check_fib_files(const std::vector<std::filesystem::path>& file_list);
std::string quality_check_nii_files(const std::vector<std::filesystem::path>& file_list);

std::vector<std::filesystem::path> search_dwi_nii_bids(const std::filesystem::path& dir);
bool nii2src(const std::vector<std::filesystem::path>& dwi_nii_files,
             const std::filesystem::path& output_dir,
             bool is_bids,
             bool overwrite,
             bool topup_eddy,
             std::string& error_msg);
void search_dwi_nii(const std::filesystem::path& dir,std::vector<std::filesystem::path>& dwi_nii_files);

bool dicom2src_and_nii(std::vector<std::filesystem::path> files,bool overwrite,std::string& error_msg)
{
    auto fail = [&](const std::string& msg){error_msg = msg;return tipl::error() << msg,false;};
    if(files.empty())
        return fail("no files provided");
    std::sort(files.begin(),files.end());
    tipl::progress p("processing DICOM at "+files.front().parent_path().u8string());
    std::string manu,make,report,sequence;
    {
        tipl::io::dicom header;
        if(!header.load_from_file(files[0]))
            return fail("cannot read image volume. skip");
        header.get_sequence_id(sequence);
        header.get_text(0x0008,0x0070,manu);//Manufacturer
        header.get_text(0x0008,0x1090,make);
        manu.erase(std::remove(manu.begin(),manu.end(),' '),manu.end());
        make.erase(std::remove(make.begin(),make.end(),' '),make.end());
        std::ostringstream info;
        info << manu.c_str() << " " << make.c_str() << " " << sequence
            << ".TE=" << header.get_float(0x0018,0x0081) << ".TR=" << header.get_float(0x0018,0x0080)  << ".";
        report = info.str();
        if(report.size() < 80)
            report.resize(80);
    }


    std::vector<std::shared_ptr<DwiHeader> > dicom_files;
    std::string parse_error;
    auto nii_file_name = get_dicom_output_name(files[0],"_" + sequence + ".nii.gz",true);

    if(!parse_dwi(files,dicom_files,parse_error) || dicom_files.size() == 1)
    {
        if(tipl::prog_aborted)
            return false;
        if(!parse_error.empty())
            return fail(parse_error);

        if(!overwrite && std::filesystem::exists(nii_file_name))
            return tipl::out() << nii_file_name << " exists. skipping",true;

        tipl::out() << "handled as structure images";
        tipl::image<3> source_images;
        tipl::vector<3> vs;

        if(files.size()==1)
        {
            tipl::io::dicom v;
            if(!v.load_from_file(files[0]))
                return fail("cannot parse dicom file");
            v >> std::tie(source_images,vs);
            if(source_images.empty())
                return fail("cannot read "+files[0].u8string()+" as image, skipping");
        }
        else
        {
            tipl::out() << "parsing " << files.size() << " dicom files";
            tipl::io::dicom_volume v;
            if(!v.load_from_files(files))
                return fail(v.error_msg);
            tipl::out() << "dim: " << v.dim << " vs: " << v.vs;
            tipl::out() << "trans: " << tipl::matrix<3,3,float>(v.orientation_matrix);
            tipl::out() << "dim order: " << tipl::vector<3,int>(v.dim_order);
            tipl::out() << "flipping: " << tipl::vector<3,int>(v.flip);
            v >> source_images;
            v.get_voxel_size(vs);
            if(source_images.empty())
                return fail("cannot read as image volume, skipping");
        }

        tipl::matrix<4,4,float> trans;
        tipl::io::initial_nifti_srow(trans,source_images.shape(),vs);
        return tipl::io::gz_nifti(nii_file_name,std::ios::out) << vs << trans << source_images;
    }

    if(!DwiHeader::has_b_table(dicom_files))
    {
        if(!overwrite && std::filesystem::exists(nii_file_name))
            return tipl::out() << nii_file_name << " exists. skipping",true;
        tipl::out() << "The images do not have b-table. Save as 4D NIFTI" << std::endl;
        auto dicom = dicom_files[0];
        tipl::matrix<4,4> trans;
        tipl::io::initial_nifti_srow(trans,dicom->image.shape(),dicom->voxel_size);

        tipl::image<4,unsigned short> buffer(dicom->image.shape().expand(dicom_files.size()));
        for(unsigned int index = 0;index < dicom_files.size();++index)
        {
            std::copy(dicom_files[index]->image.begin(),
                      dicom_files[index]->image.end(),
                      buffer.begin() + long(index*dicom_files[index]->image.size()));
        }
        tipl::out() << "output 4D NII file";
        return tipl::io::gz_nifti(nii_file_name,std::ios::out) << dicom->voxel_size << trans << report << buffer;
    }

    auto src_name = get_dicom_output_name(files[0],(std::string("_")+sequence+".sz"),true);
    if(!overwrite && std::filesystem::exists(src_name))
        return tipl::out() << src_name << " exists. skipping",true;
    src_data src;
    if(!src.load_from_file(dicom_files,false) ||
       !src.save_to_file(src_name))
        return fail(src.error_msg);
    return true;
}

bool dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite,std::string& error_msg)
{
    tipl::progress prog("convert DICOM to SRC or nifti files");
    std::vector<std::filesystem::path> pending{dir};
    size_t attempted = 0,succeeded = 0;
    for(size_t p = 0,done = 0,total = 0;p < pending.size();++p)
    {
        auto dir_list = tipl::search_dirs(pending[p],std::string());
        total += dir_list.size();
        bool has_dicom = false;
        for(size_t i = 0;i < dir_list.size();++i,++done)
        {
            if(!prog(done,total))
                return false; // aborted by the user; not a reportable failure
            auto dicom_file_list = tipl::search_files(dir_list[i],"*.dcm");
            if(dicom_file_list.empty())
                continue;
            has_dicom = true;
            while(i+1 < dir_list.size() && std::filesystem::exists(dir_list[i+1]/dicom_file_list.front().filename()))
                tipl::search_files(dir_list[++i],"*.dcm",dicom_file_list),++done;
            ++attempted;
            std::string series_error;
            if(dicom2src_and_nii(dicom_file_list,overwrite,series_error))
                ++succeeded;
            else if(!series_error.empty())
                error_msg += (error_msg.empty() ? std::string() : "\n")+series_error;
        }
        if(!has_dicom)
            pending.insert(pending.end(),dir_list.begin(),dir_list.end());
    }
    // some series failing is not itself a failure as long as at least one converted; error_msg still
    // carries every failure so the caller can warn about it even when reporting overall success
    if(attempted == 0)
        error_msg = "no DICOM files found in "+dir.u8string();
    return attempted != 0 && succeeded != 0;
}




void MainWindow::on_styles_activated(int)
{
    if(ui->styles->currentText() != settings.value("styles","Fusion").toString())
    {
        settings.setValue("styles",ui->styles->currentText());
        QMessageBox::information(this,QApplication::applicationName(),"You will need to restart DSI Studio to see the change");
    }
}

void MainWindow::on_recentFib_cellClicked(int row, int column)
{
    ui->open_selected_fib->setEnabled(true);
}

void MainWindow::on_recentSrc_cellClicked(int row, int column)
{
    ui->open_selected_src->setEnabled(true);
}

void MainWindow::on_open_selected_src_clicked()
{
    if(ui->recentSrc->currentRow() >= 0)
        open_src_at(ui->recentSrc->currentRow(),0);
}

void MainWindow::on_open_selected_fib_clicked()
{
    if(ui->recentFib->currentRow() >= 0)
        open_fib_at(ui->recentFib->currentRow(),0);
}


void MainWindow::on_template_list_itemDoubleClicked(QListWidgetItem *item)
{
    open_template(item->text());
}

bool MainWindow::open_template(QString name)
{
    for(auto& each : fib_template_list)
        if(std::filesystem::path(each).stem().u8string() == name.toStdString())
        {
            if(!loadFib(each.u8string().c_str()))
                return false;
            tracking_windows.back()->work_path.clear();
            return true;
        }
    return error_msg = name.toStdString() + " not a valid template",false;
}


QSharedPointer<QNetworkReply> MainWindow::get(QUrl url)
{
    QNetworkRequest request;
    request.setUrl(url);
    if(url.toString().contains("releases/assets/")) // when downloading restricted, the url is replaced by asset id
        request.setRawHeader("Accept", "application/octet-stream");
    else
        request.setRawHeader("Accept", "application/json");
    if(!access_token.isEmpty() && url.toString().contains("restricted"))
        request.setRawHeader("Authorization",QString("token %1").arg(access_token).toUtf8());
    return QSharedPointer<QNetworkReply>(manager.get(request),
                                         [](QNetworkReply* reply)
                                         {
                                             if(reply->isRunning())
                                                 reply->abort();
                                             reply->deleteLater();
                                         });
}

// run_shell (curl) tasks still in flight: id -> original command text; only ever touched on the GUI
// thread (dispatch_cmd's caller, and QProcess's own signals, both run there), so no lock is needed
static QMap<QString,QString> shell_tasks;

// forwards cmd to whichever AI-addressable window type target actually is; false with error left empty means
// target isn't one of these (e.g. it's main itself) and the caller should fall back to its own error_msg
static bool command_window(QWidget* target,const std::vector<std::string>& cmd,command_source source,QString& error)
{
    auto run = [&](auto* window)
    {
        if(window->command(cmd,source))
            return true;
        error = QString::fromUtf8(window->error_msg);
        if(error.isEmpty())
            error = "command failed";
        return false;
    };
    if(auto* w = qobject_cast<tracking_window*>(target))
        return run(w);
    if(auto* w = qobject_cast<reconstruction_window*>(target))
        return run(w);
    if(auto* w = qobject_cast<view_image*>(target))
        return run(w);
    if(auto* w = qobject_cast<group_connectometry*>(target))
        return run(w);
    return false;
}

QJsonObject MainWindow::dispatch_cmd(ai_info& info,const QJsonObject& request)
{
    auto fail = [](const QString& error)
    {
        return QJsonObject{{"status","error"},{"result",QJsonArray{
            QJsonObject{{"status","error"},{"error",error}}}}};
    };
    // removes ANSI escape/color codes from captured command output before it's reported to the AI agent
    auto strip_ansi = [](QString text)
    {
        static const QRegularExpression ansi_escape(
            QStringLiteral("\x1B\\[[0-?]*[ -/]*[@-~]"));
        return text.remove(ansi_escape);
    };
    // a chat/reasoning-only request (no command) is valid: the chat text is already
    // recorded by the caller (AIAgent::ai_request) before this returns, so report success
    // instead of an error
    bool has_chat = !request["chat"].toString().trimmed().isEmpty() ||
                    !request["reasoning"].toString().trimmed().isEmpty();
    // exposes this request's chat text (e.g. to run_shell's confirmation dialog) for the duration of this call;
    // restores the previous value on every exit path, including a reentrant request nested inside a long command
    struct chat_context_guard
    {
        QString& target;
        QString prev;
        chat_context_guard(QString& target_,QString value):target(target_),prev(target_) {target = std::move(value);}
        ~chat_context_guard() {target = prev;}
    } chat_guard(ai_chat_context,request["chat"].toString().trimmed());
    auto no_command_or_fail = [&]{return has_chat ? QJsonObject{{"status","success"},{"result",QJsonArray()}} : fail("missing command field");};
    auto command_json = request["command"];
    if(command_json.isUndefined() || command_json.isNull())
        return no_command_or_fail();

    std::vector<std::vector<std::string>> cmds;
    // prepare cmds
    for(const auto& value :
        (command_json.isArray() ? command_json.toArray() : QJsonArray{command_json}))
    {
        auto object = value.toObject();
        auto& cmd = cmds.emplace_back();
        auto add = [&](const QJsonValue& value){cmd.push_back(value.toVariant().toString().toUtf8().toStdString());};
        add(object["cmd"]);
        if(cmd[0].empty())
            return fail("invalid cmd text");
        auto param = object["param"];
        if(param.isArray())
            for(const auto& value : param.toArray())
                add(value);
        else if(!param.isUndefined() && !param.isNull())
            add(param);
    }
    if(cmds.empty())
        return no_command_or_fail();

    QJsonArray results;

    QPointer<QWidget> locked_target; // releases the locked window (busy) on target switch or batch end; QPointer so a target destroyed mid-batch (e.g. a local user closing it between commands) auto-nulls instead of dangling
    auto unlock_target = [&]
    {
        if(!locked_target)
            return;
        locked_target->setProperty("busy",false);
        locked_target = nullptr;
    };

    // pure lookup, no locking/side effects -- safe to reuse anywhere a window id needs checking (e.g. set_window)
    auto find_window = [&](const QString& id) -> QWidget*
    {
        if(id == "main")
            return this;
        for(auto* each : QApplication::allWidgets())
            if(command_window_id(each) == id)
                return each;
        return nullptr;
    };

    // resolves+locks the current target (unless already locked from a previous command in this same segment);
    // returns false (with `error` set) if it's busy elsewhere or can't be found
    auto resolve_target = [&](QString& error)
    {
        if(locked_target)
            return true;
        auto all_widgets = QApplication::allWidgets();
        if(std::any_of(all_widgets.begin(),all_widgets.end(),
                        [](auto* each){return each->property("busy").toBool();}))
        {
            error = "another CMD is running; check opened windows";
            return false;
        }
        auto* target = find_window(info.current_window);
        if(!target)
        {
            error = "target window not found, terminated by user? Use set_window to select a window first.";
            return false;
        }
        target->setProperty("busy",true);
        locked_target = target;
        return true;
    };

    for(const auto& cmd : cmds)
    {
        auto command_name = QString::fromUtf8(cmd[0]);
        auto window_before = info.current_window;
        QString output,error;
        QString* prev_capture; // a reentrant request handled during this command's processEvents() calls also captures; restore instead of clearing
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            prev_capture = console.capture;
            console.capture = &output;
        }
        auto prev_cwd = QDir::currentPath(); // a reentrant request restores this session's own directory when it's done, same reasoning as prev_capture
        auto session_cwd = info.model_settings["cwd"].toString();
        auto base_cwd = session_cwd.isEmpty() ? prev_cwd : session_cwd;
        if(base_cwd != prev_cwd)
            QDir::setCurrent(base_cwd);
        // reports error/output, whichever applies; caller writes "if(!finish()) break;". Also the shared cleanup
        // for every exit path: stops capturing console output for this command, and releases a stale lock if
        // the command retargeted (open_fib/open_src/open_image/set_window)
        auto finish = [&]
        {
            {
                std::lock_guard<std::mutex> lock(console.edit_buf);
                console.capture = prev_capture;
            }
            if(auto cwd = QDir::currentPath();cwd != base_cwd) // "run_shell cd ..." changed it; remember it for this session
            {
                info.model_settings["cwd"] = cwd;
                info.save_config();
            }
            QDir::setCurrent(prev_cwd);
            if(info.current_window != window_before)
                unlock_target();

            // recorded once, here, for every command regardless of outcome (including unknown/failed ones) --
            // title is whatever window this command's context ends up being (its own destination for
            // set_window), empty for main or when there's no such window
            QString title;
            if(command_window_type(info.current_window) != "main")
                if(auto* target = find_window(info.current_window))
                    title = QFileInfo(target->windowTitle()).fileName();
            QJsonObject entry{{"type","request"},{"text",command_name},{"window",command_window_type(info.current_window)}};
            if(!title.isEmpty())
                entry["title"] = title;
            info.record_history(entry);

            QJsonObject result{{"cmd",command_name},{"status",error.isEmpty() ? "success" : "error"}};
            if(!output.isEmpty())
                result["output"] = output;
            if(!error.isEmpty())
                result["error"] = error+". Read DSI Studio Manuals and retry.";
            results.append(result);
            return error.isEmpty();
        };
        tipl::progress prog(command_record(info.current_window,cmd,command_source::AI));

        try
        {
            if(command_name == "bring_to_front" || command_name == "close" ||
               command_name == "minimize" || command_name == "maximize")
            {
                // instantaneous window-manager calls: no batching benefit from resolve_target's lock, just a plain lookup
                if(auto* target = find_window(info.current_window))
                {
                    if(command_name == "close" && target == this)
                        error = "the main window cannot be closed by AI";
                    else if(command_name == "bring_to_front")
                    {
                        target->showNormal();
                        target->raise();
                        target->activateWindow();
                    }
                    else if(command_name == "minimize")
                        target->showMinimized();
                    else if(command_name == "maximize")
                        target->showMaximized();
                    else // close
                    {
                        if(locked_target == target)
                            locked_target = nullptr; // let the target manage its own lifetime again once this returns, if an earlier command in this batch had it locked
                        target->close(); // non-spontaneous: tracking_window::closeEvent() skips the unsaved-tracts prompt for this
                    }
                }
                else // not an agent mistake: most likely a local user closed it -- informational, not an error
                    output = "target window not found, terminated by user? Use set_window to select a window first.";
            }
            else if(command_name == "set_title")
            {
                if(cmd.size() != 2 || cmd[1].empty())
                    error = "usage: set_title <title>";
                else if(!info.save_title(QString::fromStdString(cmd[1]).simplified()))
                    error = "cannot save title";
            }
            else if(command_name == "log")
            {
                std::lock_guard<std::mutex> lock(console.edit_buf);
                if(info.log_position == quint64(-1))
                    info.log_position = console.total_size; // first ever log read for this session: start from now, not from the console's whole history
                auto end = console.total_size;
                auto first = end-quint64(console.history.size());
                // any parameter (cmd[1] or cmd[2]) pulls everything still retained in the console
                // buffer instead of just what's new since this session's own cursor
                bool full = (cmd.size() > 1 && !cmd[1].empty()) || (cmd.size() > 2 && !cmd[2].empty());
                auto begin = full ? first : std::max(info.log_position,first);
                bool capped = end-begin > 16*1024;
                if(capped)
                    begin = end-16*1024;
                auto text = console.history.mid(qsizetype(begin-first));
                if(capped)
                    text.remove(0,text.indexOf('\n')+1);
                text = strip_ansi(text);
                QStringList lines;
                for(const auto& line : text.split('\n'))
                    if(!line.contains("[DEBUG]"))
                        lines << line;
                output = lines.join('\n').right(4*1024);
                info.log_position = end;
            }
            else if(command_name == "set_window")
            {
                auto param = cmd.size() > 1 ? QString::fromStdString(cmd[1]) : QString();
                if(param.isEmpty() || find_window(param))
                {
                    info.current_window = param.isEmpty() ? "main" : param; // finish()'s generic window_before check below releases the previous target's lock
                    output = "current window: "+info.current_window;
                }
                else // not an error: informational, current window is left unchanged
                    output = "window \""+param+"\" not found, terminated by user? current window remains: "+info.current_window;
            }
            else if(command_name == "list_window")
            {
                auto* modal = QApplication::activeModalWidget();
                bool application_busy = tipl::status_list.size() > 1;
                QJsonObject windows;
                for(auto* each : QApplication::allWidgets())
                {
                    auto id = command_window_id(each);
                    if(id.isEmpty())
                        continue;
                    bool busy = each->property("busy").toBool();
                    if(auto* tracking = qobject_cast<tracking_window*>(each))
                        busy |= tracking->history.running_commands ||
                                (tracking->tractWidget &&
                                 std::any_of(
                                     tracking->tractWidget->thread_data.begin(),
                                     tracking->tractWidget->thread_data.end(),
                                     [](const auto& thread){return bool(thread);})) ||
                                std::any_of(
                                    tracking->slices.begin(),tracking->slices.end(),
                                    [](const auto& slice)
                                    {
                                        auto custom = std::dynamic_pointer_cast<CustomSliceModel>(slice);
                                        return custom && custom->running;
                                    });
                    else if(auto* gc = qobject_cast<group_connectometry*>(each))
                        busy |= bool(gc->timer);
                    bool waiting = modal && (modal == each || each->isAncestorOf(modal));
                    windows[id] = QJsonObject{
                        {"status",waiting ? "waiting" : busy ? "busy" : "idle"},
                        {"title",QDir::fromNativeSeparators(each->windowTitle())}
                    };
                    application_busy |= busy;
                }
                for(auto it = shell_tasks.constBegin();it != shell_tasks.constEnd();++it)
                    windows[it.key()] = QJsonObject{{"status","busy"},{"title",it.value()}};
                application_busy |= !shell_tasks.isEmpty();
                // active tipl::progress operations, outermost to innermost; index 0 is the app-lifetime citation entry, skipped
                QJsonArray progress;
                for(size_t i = 1;i < tipl::status_list.size();++i)
                {
                    const auto& s = tipl::status_list[i];
                    progress.append(QJsonObject{
                        {"status",QString::fromStdString(s.status)},
                        {"now",int(s.now)},
                        {"total",int(s.total)},
                        {"at",QString::fromStdString(s.at)}});
                }
                output = QString::fromUtf8(QJsonDocument(QJsonObject{
                    {"application",QJsonObject{{"status",modal ? "waiting" : application_busy ? "busy" : "idle"}}},
                    {"current_window",info.current_window},
                    {"progress",progress},
                    {"windows",windows}}).toJson(QJsonDocument::Compact));
            }
            else if(command(cmd,command_source::AI)) {} // handled by MainWindow directly: nothing more to do
            else if(error_msg == "unknown command: "+cmd[0])
            {
                if(!resolve_target(error))
                {
                    output = error; // busy-elsewhere / window-not-found: a status report, not an agent mistake
                    error.clear();
                }
                else if(!command_window(locked_target,cmd,command_source::AI,error) && error.isEmpty())
                    error = QString::fromStdString(error_msg); // target is main itself; nothing else to try
            }
            else // recognized by MainWindow but failed for a real reason
                error = QString::fromStdString(error_msg);
        }
        catch(const std::exception& e){error = e.what();}
        catch(...){error = "unknown error";}

        output = strip_ansi(output);
        error = strip_ansi(error);

        if(!finish())
            break;
    }
    unlock_target(); // release whatever is still locked when the batch finishes normally

    return QJsonObject{
        {"status",results.last().toObject()["status"]},{"result",results}};
}

int run_action_with_wildcard(tipl::program_option<tipl::out>&);
bool MainWindow::command(const std::vector<std::string>& cmd)
{
    return command(cmd,command_source::User);
}
bool MainWindow::command(const std::vector<std::string>& cmd,
                         command_source source)
{
    error_msg.clear();
    auto fail = [&](const std::string& msg){error_msg = msg;return false;};
    // for a batch command run to completion in this function: err empty means success, prompting "Finished." for a local user; non-empty means fail(err)
    auto finish = [&](const std::string& err)
    {
        if(!err.empty())
            return fail(err);
        if(source == command_source::User)
            QMessageBox::information(this,QApplication::applicationName(),"Finished.");
        return true;
    };
    if(cmd.empty())
        return fail("empty command");

    if(cmd[0] == "open_hub" || tipl::begins_with(cmd[0],"hub_"))
    {
        if(cmd[0] == "open_hub" && cmd.size() != 1)
            return fail("open_hub takes no arguments");
        if(!fiber_data_hub)
            fiber_data_hub = new FiberDataHub(this);
        fiber_data_hub->showNormal();
        fiber_data_hub->raise();
        fiber_data_hub->activateWindow();
        if(cmd[0] == "open_hub")
            return true;
        if(!fiber_data_hub->command(cmd))
            return fail(fiber_data_hub->error_msg);
        return true;
    }


    auto get_files = [&](size_t begin = 1)
    {
        QStringList files;
        for(size_t i = begin;i < cmd.size();++i)
            files << QString::fromUtf8(cmd[i].c_str());
        return files;
    };
    auto select_dir = [&](const QString& title,QString initial = {})
    {
        if(cmd.size() >= 2) // >= rather than == so commands taking a second, separate parameter after the directory (e.g. bids_to_src's output folder) still work
            return QString::fromUtf8(cmd[1]);
        return QFileDialog::getExistingDirectory(
            this,title,initial.isEmpty() ? work_dir() : initial);
    };
    auto select_images = [&](const QString& filter,bool single = false)
    {
        if(cmd.size() >= 2)
            return get_files();
        if(!single)
            return tipl::qt::open_image_files(this,work_dir(),filter);
        auto file = tipl::qt::open_image_file(this,work_dir(),filter);
        return file.isEmpty() ? QStringList() : QStringList{file};
    };

    if(cmd[0] == "list_recent_fib" ||
       cmd[0] == "list_recent_src")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        auto files = settings.value(
            cmd[0] == "list_recent_fib" ?
                "recentFibFileList" : "recentSrcFileList").toStringList();
        for(auto file : files)
            if(QFileInfo::exists(
                   file = QDir::fromNativeSeparators(file)))
                tipl::out() << file.toStdString();
        return true;
    }

    if(cmd[0] == "reset_settings")
    {
        if(cmd.size() != 1)
            return fail("reset_settings takes no arguments");
        settings.clear();
        settings.sync();
        return true;
    }

    if(cmd[0] == "set_work_dir")
    {
        auto dir = select_dir(
            "Browse Directory",ui->workDir->currentText());
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        return true;
    }

    if(cmd[0] == "rename_dicom")
    {
        QStringList files;
        if(cmd.size() >= 2)
            files = get_files();
        else
        {
            files = QFileDialog::getOpenFileNames(
                         this,"Open DICOM files",work_dir(),"All files (*)");
            if(files.isEmpty())
                return true;
        }
        add_work_dir(QFileInfo(files[0]).absolutePath());
        tipl::progress prog("Rename DICOM Files");
        bool result = true;
        std::string dicom_error;
        for(int index = 0;prog(index,files.size());++index)
        {
            auto file = tipl::qt::to_path(files[index]);
            std::string error;
            if(rename_dicom(file,file.parent_path(),error).empty())
            {
                result = false;
                if(!error.empty())
                    dicom_error += (dicom_error.empty() ? std::string() : "\n")+error;
            }
        }
        if(tipl::prog_aborted)
            return true;
        return finish(result ? std::string() :
            dicom_error.empty() ? "one or more DICOM files could not be renamed" : dicom_error);
    }

    if(cmd[0] == "rename_dicom_dir")
    {
        auto dir = select_dir("Browse Directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        std::string dicom_error;
        rename_dicom_at_dir(tipl::qt::to_path(dir),tipl::qt::to_path(dir),dicom_error);
        return finish(dicom_error);
    }

    if(cmd[0] == "convert_dicom_dir")
    {
        auto dir = select_dir("Open directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        std::string dicom_error;
        if(!dicom2src_and_nii(tipl::qt::to_path(dir),false,dicom_error))
            return fail(dicom_error.empty() ? "DICOM conversion failed" : dicom_error);
        if(source == command_source::User) // some series may have failed even on overall success; say so instead of a bare "Finished."
            QMessageBox::information(this,QApplication::applicationName(),
                dicom_error.empty() ? "Finished." : QString::fromStdString("Finished with warnings:\n"+dicom_error));
        return true;
    }

    if(cmd[0] == "bids_to_src")
    {
        auto dir = select_dir("Open BIDS Folder");
        if(dir.isEmpty())
            return true;
        auto output_dir = cmd.size() >= 3 ? QString::fromUtf8(cmd[2]) :
                              QFileDialog::getExistingDirectory(
                                  this,"Please Specify the Output Folder",
                                  QDir(dir).path()+"/derivatives");
        if(output_dir.isEmpty())
            return true;
        if(!QDir(output_dir).exists() && !QDir().mkpath(output_dir))
            return fail("cannot create output folder: "+output_dir.toStdString());
        add_work_dir(dir);
        auto files = search_dwi_nii_bids(tipl::qt::to_path(dir));
        if(files.empty())
        {
            std::string message("cannot find bids nifti data");
            return fail(message);
        }
        std::sort(files.begin(),files.end());
        std::string src_error;
        nii2src(files,tipl::qt::to_path(output_dir),true,true,false,src_error);
        if(tipl::prog_aborted)
            return true;
        return finish(src_error);
    }

    if(cmd[0] == "nifti_dir_to_src")
    {
        auto dir = select_dir("Open directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);

        std::vector<std::filesystem::path> files;
        search_dwi_nii(tipl::qt::to_path(dir),files);
        if(files.empty())
            return fail("cannot find nifti data");

        std::vector<std::filesystem::path> selected;
        selected.reserve(files.size());
        bool yes_to_all = false,no_to_all = false;
        auto output_dir = tipl::qt::to_path(dir);
        for(const auto& nii : files)
        {
            auto src = output_dir/tipl::remove_all_suffix(nii.filename());
            src += ".sz";
            if(std::filesystem::exists(src) && !yes_to_all)
            {
                if(no_to_all)
                    continue;
                auto result = QMessageBox::information(
                    this,QApplication::applicationName(),
                    QString("%1 exists, overwrite?").arg(
                        QString::fromUtf8(src.filename().u8string().c_str())),
                    QMessageBox::Yes|QMessageBox::YesToAll|
                    QMessageBox::No|QMessageBox::NoToAll|QMessageBox::Cancel);
                if(result == QMessageBox::Cancel)
                    return true;
                if(result == QMessageBox::YesToAll)
                    yes_to_all = true;
                if(result == QMessageBox::NoToAll)
                    no_to_all = true;
                if(result == QMessageBox::No || result == QMessageBox::NoToAll)
                    continue;
            }
            selected.push_back(nii);
        }
        std::string src_error;
        nii2src(selected,output_dir,false,true,false,src_error);
        if(tipl::prog_aborted)
            return true;
        return finish(src_error);
    }

    if(cmd[0] == "collect_network_measures")
    {
        QStringList files;
        if(cmd.size() >= 2)
            files = get_files();
        else
        {
            files = QFileDialog::getOpenFileNames(
                         this,"Open Network Measures",work_dir(),
                         "Text files (*.txt);;All files (*)");
            if(files.isEmpty())
                return true;
        }

        QStringList fields;
        QMap<QString,QStringList> values;
        auto add = [&](const QString& field,const QString& value,int column)
        {
            if(!values.contains(field))
            {
                QStringList row;
                row.fill({},files.size());
                values[field] = row;
                fields << field;
            }
            values[field][column] = value;
        };

        for(int column = 0;column < files.size();++column)
        {
            std::ifstream in(tipl::qt::to_path(files[column]));
            std::vector<std::string> nodes;
            std::string name,value;
            while(in >> name)
            {
                if(name == "network_measures")
                {
                    std::string line;
                    std::getline(in,line);
                    std::istringstream stream(line);
                    nodes.assign(std::istream_iterator<std::string>(stream),{});
                    break;
                }
                if(!(in >> value))
                    break;
                add(QString::fromStdString(name),
                    QString::fromStdString(value),column);
            }

            std::string line;
            while(std::getline(in,line))
            {
                std::istringstream stream(line);
                if(!(stream >> name) || name.empty() || name[0] == '#')
                    continue;
                for(const auto& node : nodes)
                {
                    value.clear();
                    stream >> value;
                    add(QString::fromStdString(name+"_"+node),
                        QString::fromStdString(value),column);
                }
            }
        }

        auto output = files[0]+".collected.txt";
        QFile file(output);
        if(!file.open(QIODevice::WriteOnly|QIODevice::Text))
            return fail("cannot write "+output.toStdString());

        QTextStream out(&file);
        out << "Field";
        for(const auto& input : files)
            out << '\t' << QFileInfo(input).baseName();
        out << '\n';
        for(const auto& field : fields)
            out << field << '\t' << values[field].join('\t') << '\n';
        if(cmd.size() == 1)
            QMessageBox::information(this,QApplication::applicationName(),"File saved to "+output);
        return true;
    }

    if(cmd[0] == "open_src")
    {
        auto files = select_images(
            "Src files (*.sz *src.gz);;Histology images (*.jpg *.tif);;All files (*)");
        if(files.isEmpty())
            return true;
        add_work_dir(QFileInfo(files[0]).absolutePath());
        if(!loadSrc(files))
            return fail(error_msg);
        return true;
    }

    if(cmd[0] == "open_dwi_nifti" ||
       cmd[0] == "open_dwi_dicom" ||
       cmd[0] == "open_dwi_2dseq")
    {
        bool nifti = cmd[0] == "open_dwi_nifti";
        auto files = select_images(
            nifti ? "NIFTI files (*.nii *.nii.gz);;All files (*)" :
            cmd[0] == "open_dwi_dicom" ?
                "DICOM files (*.dcm);;All files (*)" :
                "2dseq files (2dseq);;FDF files (*.fdf);;NRRD Files (*.nrrd);;All files (*)",
            nifti);
        if(files.isEmpty())
            return true;
        open_DWI(files);
        return true;
    }

    if(cmd[0] == "open_src_dir")
    {
        auto dir = select_dir("Open directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        if(!loadSrc(search_files(dir,"*src.gz") << search_files(dir,"*.sz")))
            return fail(error_msg);
        return true;
    }

    if(cmd[0] == "open_fib" || cmd[0] == "open_structural_tracking")
    {
        QString file;
        if(cmd.size() == 2)
            file = QString::fromUtf8(cmd[1]);
        else
        {
            auto filter = cmd[0] == "open_fib" ?
                          "Fib files (*.fz *fib.gz *.dz);;All files (*)" :
                          "Image files (*nii.gz *.nii 2dseq);;All files (*)";
            file = tipl::qt::open_image_file(
                            this,ui->workDir->currentText(),filter);
            if(file.isEmpty())
                return true;
            add_work_dir(QFileInfo(file).absolutePath());
        }
        return loadFib(file);
    }

    if(cmd[0] == "open_template")
    {
        QString template_name;
        if(cmd.size() == 2)
            template_name = QString::fromUtf8(cmd[1]);
        else
        {
            auto* item = ui->template_list->currentItem();
            if(!item)
                return true;
            template_name = item->text();
        }
        return open_template(template_name);
    }

    if(cmd[0] == "create_db" || cmd[0] == "create_average")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        auto* window = new CreateDBDialog(this,cmd[0] == "create_db");
        window->setAttribute(Qt::WA_DeleteOnClose);
        window->show();
        return true;
    }

    if(cmd[0] == "open_db" || cmd[0] == "open_connectometry")
    {
        QString file;
        if(cmd.size() == 2)
            file = QString::fromUtf8(cmd[1]);
        else
        {
            file = tipl::qt::open_image_file(this,ui->workDir->currentText(),"Database (*.dz *db.fz *db?fib.gz);;All files (*)");
            if (file.isEmpty())
                return true;
        }
        std::shared_ptr<group_connectometry_analysis> database;
        {
            add_work_dir(QFileInfo(file).absolutePath());
            database = std::make_shared<group_connectometry_analysis>();
            tipl::progress prog_("reading connectometry db");
            if(!database->load_database(file.toStdString().c_str()))
                return fail(database->error_msg);
        }

        if(cmd[0] == "open_db")
        {
            auto* window = new db_window(this,database);
            window->setWindowTitle(file);
            window->setAttribute(Qt::WA_DeleteOnClose);
            window->show();
        }
        else
        {
            auto* window = new group_connectometry(this,database,file);
            window->setWindowTitle(file);
            report_and_target_window(window);
            window->setAttribute(Qt::WA_DeleteOnClose);
            window->show();
        }
        return true;
    }

    if(cmd[0] == "open_auto_track" ||
       cmd[0] == "open_nonlinear_registration" ||
       cmd[0] == "open_xnat")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        QWidget* window =
            cmd[0] == "open_auto_track" ?
                static_cast<QWidget*>(new auto_track(this)) :
            cmd[0] == "open_nonlinear_registration" ?
                static_cast<QWidget*>(new RegToolBox(this)) :
                static_cast<QWidget*>(new xnat_dialog(this));
        window->setAttribute(Qt::WA_DeleteOnClose);
        window->showNormal();
        return true;
    }

    if(cmd[0] == "open_console")
    {
        if(cmd.size() != 1)
            return fail("open_console takes no arguments");
        static Console* console_window = nullptr;
        if(!console_window)
            console_window = new Console(this);
        console_window->showNormal();
        return true;
    }

    if(cmd[0] == "clear_recent_src" || cmd[0] == "clear_recent_fib")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        bool src = cmd[0] == "clear_recent_src";
        (src ? ui->recentSrc : ui->recentFib)->setRowCount(0);
        (src ? ui->open_selected_src : ui->open_selected_fib)->setEnabled(false);
        settings.setValue(src ? "recentSrcFileList" : "recentFibFileList",
                          QStringList());
        return true;
    }

    if(cmd[0] == "show_qc_nii" || cmd[0] == "show_qc_src" || cmd[0] == "show_qc_fib" ||
       cmd[0] == "save_qc_nii" || cmd[0] == "save_qc_src" || cmd[0] == "save_qc_fib")
    {
        bool save = tipl::begins_with(cmd[0],"save_");
        bool nii = tipl::contains(cmd[0],"nii"),src = tipl::contains(cmd[0],"src");
        QString filter = nii ? "NIFTI files (*.nii *nii.gz);;All files (*)" :
                          src ? "Src files (*.sz *src.gz);;All files (*)" :
                                "Fib files (*.fz *fib.gz);;All files (*)";
        std::string save_path;
        QStringList filenames;
        if(save)
        {
            if(cmd.size() < 2 || cmd[1].empty())
                return fail("usage: "+cmd[0]+" <output file path> [input files...]");
            save_path = cmd[1];
            filenames = cmd.size() >= 3 ? get_files(2) : tipl::qt::open_image_files(this,work_dir(),filter);
        }
        else
            filenames = select_images(filter);
        if(filenames.isEmpty())
            return true;
        std::vector<std::filesystem::path> files;
        files.reserve(filenames.size());
        for(const auto& file : filenames)
            files.push_back(tipl::qt::to_path(file));
        tipl::progress prog(nii ? "checking NIFTI files" :
                            src ? "checking SRC files" : "checking FIB files");
        auto result = nii ? quality_check_nii_files(files) :
                      src ? quality_check_src_files(files,false,false,0) :
                            quality_check_fib_files(files);
        if(save)
        {
            tipl::out() << "save " << save_path;
            if(!tipl::write_text_file(save_path,result,tipl::error()))
                return fail("cannot write to "+save_path);
        }
        else if(source == command_source::AI)
            tipl::out() << result;
        else
            show_info_dialog(nii ? "NIFTI report" : src ? "SRC report" : "FIB report",result);
        return true;
    }

    if(cmd[0] == "run_cli")
    {
        if(cmd.size() != 2 || cmd[1].empty())
            return fail("usage: run_cli <command line>");
        auto command_line = cmd[1];
        if(tipl::begins_with(command_line,"dsi_studio "))
            command_line.erase(0,11);
        tipl::program_option<tipl::out> po;
        if(!po.parse(command_line))
            return fail(po.error_msg);
        if(!po.has("action"))
            po.set("action","vis");
        if(run_action_with_wildcard(po))
            return fail("command line failed");
        po.check_end_param<tipl::warning>();
        return true;
    }

    if(cmd[0] == "open_image")
    {
        auto files = select_images(
            "image files (*.nii *nii.gz *.dcm *.nhdr *.nrrd 2dseq);;All files (*)");
        if(files.isEmpty())
            return true;
        add_work_dir(QFileInfo(files[0]).absolutePath());
        auto* window = new view_image(this);
        window->setAttribute(Qt::WA_DeleteOnClose);
        if(!window->open(files))
        {
            auto message = window->error_msg;
            delete window;
            return fail(message);
        }
        report_and_target_window(window);
        window->show();
        return true;
    }
    if(cmd[0] == "open_ai")
    {
        if(cmd.size() != 1)
            return fail("open_ai takes no arguments");
        ai_agent->showNormal();
        ai_agent->raise();
        ai_agent->activateWindow();
        return true;
    }
    if(cmd[0] == "voice")
    {
        if(cmd.size() != 2 || cmd[1].empty())
            return fail("usage: voice <text>");
#ifdef Q_OS_WIN
        QProcess process;
        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("DSI_VOICE_TEXT",QString::fromUtf8(cmd[1].c_str()));
        process.setProcessEnvironment(env);
        process.setProgram("powershell.exe");
        process.setArguments({
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$v=New-Object -ComObject SAPI.SpVoice;"
                "[void]$v.Speak($env:DSI_VOICE_TEXT)"
            });
        // detach: DSI Studio returns immediately and does not wait for powershell to finish speaking
        return process.startDetached() || fail("cannot start Windows speech");
#else
        return fail("voice is available only on Windows");
#endif
    }
    if(cmd[0] == "run_shell")
    {
        if(cmd.size() != 2 || cmd[1].empty())
            return fail("usage: run_shell <command>");
        QString text = QString::fromUtf8(cmd[1].c_str());
        QString program = text.section(' ',0,0);
        if(!program.compare("cd",Qt::CaseInsensitive))
        {
            // "cd" is a shell builtin: change DSI Studio's own working directory so it persists across calls
            QString path = text.mid(program.length()).trimmed();
            if(path.size() >= 2 && path.startsWith('"') && path.endsWith('"'))
                path = path.mid(1,path.size()-2);
            if(!path.isEmpty() && !QDir::setCurrent(path))
                return fail(("cannot change directory to: "+path).toStdString());
            tipl::out() << QDir::currentPath().toStdString();
            return true;
        }
        // runs "text" through a real shell so pipes/redirection/&&/globbing behave the same on every OS
        auto configure_process = [](QProcess& process,const QString& shell_text)
        {
#ifdef Q_OS_WIN
            process.setProgram("cmd.exe");
            process.setNativeArguments("/c " + shell_text);
#else
            process.setProgram("/bin/sh");
            process.setArguments({"-c",shell_text});
#endif
        };
        bool is_curl = !program.compare("curl",Qt::CaseInsensitive);
        if(is_curl) // curl's default progress meter redraws one line via \r for an interactive terminal;
            text.insert(program.length()," -s -S"); // captured non-interactively, that just floods the log. -s hides it, -S still shows real errors
        // a plain single command (e.g. a bare "dir"/"ls") runs unconfirmed; anything containing a
        // character a real shell treats specially -- chaining (&&/||/;/&), piping (|), substitution
        // (`/$()), or redirection (>/<) -- still requires confirmation, since the full text is what
        // actually runs, not just what the visible command looks like at a glance
        static const QStringList shell_special{"&","|",";","`","$(",">","<"};
        bool needs_confirm = std::any_of(shell_special.begin(),shell_special.end(),
            [&](const QString& token){return text.contains(token);});
        if(source == command_source::AI && needs_confirm)
        {
            QString message;
            if(!ai_chat_context.isEmpty()) // explains why, using the agent's own accompanying chat message, when one was sent with this request
                message = "The AI agent says:\n\n"+ai_chat_context+"\n\n";
            message += "The AI agent wants to run this shell command:\n\n"+text;
            if(QMessageBox::question(this,"AI Shell Command Request",message,
                   QMessageBox::Yes|QMessageBox::No,QMessageBox::No) != QMessageBox::Yes)
                return fail("user declined to run this shell command");
        }
        if(!is_curl) // not curl: assume it's fast, wait for it, but bound the wait so a stuck program cannot deadlock this request forever
        {
            QProcess process;
            configure_process(process,text);
            process.start();
            if(!process.waitForStarted(3000))
                return fail("cannot start command");
            constexpr int shell_timeout_ms = 600000; // 10 minutes; longer operations should use curl's async path instead
            if(!process.waitForFinished(shell_timeout_ms))
            {
                process.kill();
                process.waitForFinished(3000);
                return fail("command timed out after 10 minutes");
            }
            tipl::out() << process.readAllStandardOutput().toStdString();
            auto err = process.readAllStandardError().toStdString();
            if(!err.empty())
                tipl::error() << err;
            if(process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
                return fail("command exited with code "+std::to_string(process.exitCode()));
            return true;
        }
        // curl can hang, so run it as a self-owned asynchronous QProcess instead of blocking any thread;
        // "list_window" shows the id as busy until it finishes (there is intentionally no completion timeout here)
        static int next_curl_id = 0;
        QString id = "curl"+QString::number(++next_curl_id);
        auto* process = new QProcess(this); // falls back to MainWindow as owner if it outlives this call
        configure_process(*process,text);
        auto cleanup = [process,id] // every terminal path removes the task and releases the process the same way
        {
            shell_tasks.remove(id);
            process->deleteLater();
        };
        QObject::connect(process,&QProcess::errorOccurred,[id,text,cleanup](QProcess::ProcessError error)
        {
            if(error != QProcess::FailedToStart) // Crashed etc. still emits finished; that path already cleans up
                return;
            tipl::error() << (id+" cannot start: "+text).toStdString();
            cleanup();
        });
        QObject::connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            [process,id,text,cleanup](int exit_code,QProcess::ExitStatus exit_status)
        {
            tipl::out() << process->readAllStandardOutput().toStdString();
            auto err = process->readAllStandardError().toStdString();
            if(!err.empty())
                tipl::error() << err;
            if(exit_status != QProcess::NormalExit || exit_code != 0)
                tipl::error() << (id+" exited with code "+QString::number(exit_code)+": "+text).toStdString();
            cleanup();
        });
        shell_tasks[id] = text;
        process->start();
        tipl::out() << ("started "+id+": "+text).toStdString();
        return true;
    }
    return fail("unknown command: "+cmd[0]);
}
