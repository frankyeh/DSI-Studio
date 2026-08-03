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

#include <QJsonDocument>
#include <QMap>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <thread>
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
    report_window_created(tracking_windows.back(),"tracking");
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
    report_window_created(dialog,"image");
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
        report_window_created(new_mdi,"recon");
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

std::filesystem::path rename_dicom(const std::filesystem::path& file_name,std::filesystem::path output);

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
                                                       std::filesystem::path output);

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

bool get_pe_dir(const std::string& nii_name,size_t& pe_dir,bool& is_neg)
{
    QFile file(QString::fromUtf8(
        (tipl::remove_all_suffix(nii_name)+".json").c_str()));
    if(!file.open(QIODevice::ReadOnly))
        return false;

    auto value = QJsonDocument::fromJson(file.readAll()).
                 object()["PhaseEncodingDirection"].toString();
    auto axis = QString("ijk").indexOf(value.left(1));
    if(axis < 0 || value.size() > 2 ||
        (value.size() == 2 && value[1] != '-'))
        return false;

    pe_dir = size_t(axis);
    is_neg = value.endsWith('-');
    return true;
}

std::vector<std::filesystem::path> search_dwi_nii_bids(const std::filesystem::path& dir);
bool nii2src(const std::vector<std::filesystem::path>& dwi_nii_files,
             const std::filesystem::path& output_dir,
             bool is_bids,
             bool overwrite,
             bool topup_eddy,
             const char* progress_name = "convert nifti to src files");
void search_dwi_nii(const std::filesystem::path& dir,std::vector<std::filesystem::path>& dwi_nii_files);

bool dicom2src_and_nii(std::vector<std::filesystem::path> files,bool overwrite)
{
    if(files.empty())
        return false;
    std::sort(files.begin(),files.end());
    tipl::progress p("processing DICOM at "+files.front().parent_path().u8string());
    std::string manu,make,report,sequence;
    {
        tipl::io::dicom header;
        if(!header.load_from_file(files[0]))
            return tipl::error() << "cannot read image volume. skip",false;
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
    std::string error_msg;
    auto nii_file_name = get_dicom_output_name(files[0],"_" + sequence + ".nii.gz",true);

    if(!parse_dwi(files,dicom_files,error_msg) || dicom_files.size() == 1)
    {
        if(tipl::prog_aborted)
            return false;
        if(!error_msg.empty())
            return tipl::error() << error_msg,false;

        if(!overwrite && std::filesystem::exists(nii_file_name))
            return tipl::out() << nii_file_name << " exists. skipping",true;

        tipl::out() << "handled as structure images";
        tipl::image<3> source_images;
        tipl::vector<3> vs;

        if(files.size()==1)
        {
            tipl::io::dicom v;
            if(!v.load_from_file(files[0]))
                return tipl::error() << "cannot parse dicom file",false;
            v >> std::tie(source_images,vs);
            if(source_images.empty())
                return tipl::warning() << "cannot read " << files[0] << " as image, skipping",false;
        }
        else
        {
            tipl::out() << "parsing " << files.size() << " dicom files";
            tipl::io::dicom_volume v;
            if(!v.load_from_files(files))
                return tipl::out() << v.error_msg,false;
            tipl::out() << "dim: " << v.dim << " vs: " << v.vs;
            tipl::out() << "trans: " << tipl::matrix<3,3,float>(v.orientation_matrix);
            tipl::out() << "dim order: " << tipl::vector<3,int>(v.dim_order);
            tipl::out() << "flipping: " << tipl::vector<3,int>(v.flip);
            v >> source_images;
            v.get_voxel_size(vs);
            if(source_images.empty())
                return tipl::warning() << "cannot read as image volume, skipping",false;
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
        return tipl::error() << src.error_msg,false;
    return true;
}

bool dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite)
{
    tipl::progress prog("convert DICOM to SRC or nifti files");
    std::vector<std::filesystem::path> pending{dir};
    bool result = true;
    for(size_t p = 0,done = 0,total = 0;p < pending.size();++p)
    {
        auto dir_list = tipl::search_dirs(pending[p],std::string());
        total += dir_list.size();
        bool has_dicom = false;
        for(size_t i = 0;i < dir_list.size();++i,++done)
        {
            if(!prog(done,total))
                return false;
            auto dicom_file_list = tipl::search_files(dir_list[i],"*.dcm");
            if(dicom_file_list.empty())
                continue;
            has_dicom = true;
            while(i+1 < dir_list.size() && std::filesystem::exists(dir_list[i+1]/dicom_file_list.front().filename()))
                tipl::search_files(dir_list[++i],"*.dcm",dicom_file_list),++done;
            if(!dicom2src_and_nii(dicom_file_list,overwrite))
                result = false;
        }
        if(!has_dicom)
            pending.insert(pending.end(),dir_list.begin(),dir_list.end());
    }
    return result;
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

static std::mutex shell_tasks_mutex; // run_shell (curl) tasks still in flight: id -> original command text
static QMap<QString,QString> shell_tasks;

QJsonObject MainWindow::dispatch_cmd(ai_info& info,const QJsonObject& request)
{
    auto fail = [](const QString& error)
    {
        return QJsonObject{{"status","error"},{"result",QJsonArray{
            QJsonObject{{"status","error"},{"error",error}}}}};
    };
    // "main"/"trackingXXXX"/"reconXXXX"/"imageXXXX", or empty if not an AI-addressable window
    auto ai_window_id = [](QWidget* window)
    {
        if(qobject_cast<MainWindow*>(window))
            return command_window_id(window,"main");
        if(qobject_cast<tracking_window*>(window))
            return command_window_id(window,"tracking");
        if(qobject_cast<reconstruction_window*>(window))
            return command_window_id(window,"recon");
        return qobject_cast<view_image*>(window) ?
            command_window_id(window,"image") : QString();
    };
    // removes ANSI escape/color codes from captured command output before it's reported to the AI agent
    auto strip_ansi = [](QString text)
    {
        static const QRegularExpression ansi_escape(
            QStringLiteral("\x1B\\[[0-?]*[ -/]*[@-~]"));
        return text.remove(ansi_escape);
    };
    auto command_json = request["command"];
    if(command_json.isUndefined() || command_json.isNull())
        return fail("missing command field");
    std::vector<std::vector<std::string>> cmds;
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
        return fail("missing command field");

    QJsonArray results;
    QString ai_current_window = "main"; // local to this one dispatch call: "set_window" only retargets later commands in this same batch, nothing persists between calls

    QWidget* locked_target = nullptr; // releases the locked window (setUpdatesEnabled/busy) on target switch or batch end
    bool locked_updates_enabled = true;
    auto unlock_target = [&]
    {
        if(!locked_target)
            return;
        locked_target->setProperty("busy",false);
        locked_target->setUpdatesEnabled(locked_updates_enabled);
        if(auto* window = qobject_cast<tracking_window*>(locked_target))
        {
            window->slice_need_update = true;
            window->glWidget->update_slice();
        }
        else
            locked_target->update();
        locked_target = nullptr;
    };

    // resolves+locks the current target (unless already locked from a previous command in this same segment);
    // returns false (with `error` set) if it's busy elsewhere or can't be found
    auto resolve_target = [&](QString& error)
    {
        if(locked_target)
            return true;
        QWidget* target = ai_current_window == "main" ? static_cast<QWidget*>(this) : nullptr;
        bool busy_elsewhere = false;
        for(auto* each : QApplication::allWidgets())
        {
            if(each->property("busy").toBool())
                busy_elsewhere = true;
            if(!target && ai_window_id(each) == ai_current_window)
                target = each;
        }
        if(busy_elsewhere || !target)
        {
            error = busy_elsewhere ? "another CMD is running; check opened windows" :
                                     "target window not found, terminated by user? Use set_window to select a window first.";
            return false;
        }
        locked_updates_enabled = target->updatesEnabled();
        target->setUpdatesEnabled(false);
        target->setProperty("busy",true);
        locked_target = target;
        return true;
    };

    for(const auto& cmd : cmds)
    {
        auto command_name = QString::fromUtf8(cmd[0]);
        auto command_result = [&](bool ok,const QString& output = {},const QString& error = {}) // avoids repeating {"cmd",command_name} everywhere
        {
            QJsonObject result{{"cmd",command_name},{"status",ok ? "success" : "error"}};
            if(!output.isEmpty())
                result["output"] = output;
            if(!error.isEmpty())
                result["error"] = error;
            return result;
        };

        // bring_to_front/close/minimize/maximize: generic window control, works on whichever window is targeted;
        // not left duplicated per window (every command() implementation had its own identical copy)
        if(command_name == "bring_to_front" || command_name == "close" ||
           command_name == "minimize" || command_name == "maximize")
        {
            QString error;
            if(!resolve_target(error))
            {
                results.append(command_result(false,{},error));
                break;
            }
            if(command_name == "close" && locked_target == this)
            {
                results.append(command_result(false,{},"the main window cannot be closed by AI"));
                break;
            }
            if(command_name == "bring_to_front")
            {
                locked_target->showNormal();
                locked_target->raise();
                locked_target->activateWindow();
            }
            else if(command_name == "minimize")
                locked_target->showMinimized();
            else if(command_name == "maximize")
                locked_target->showMaximized();
            else // close
            {
                auto* target = locked_target;
                locked_target = nullptr; // let the target manage its own lifetime again once this returns
                target->close(); // non-spontaneous: tracking_window::closeEvent() skips the unsaved-tracts prompt for this
            }
            results.append(command_result(true));
            continue;
        }

        // set_title/log/set_window: need this call's ai_info/ai_current_window directly, which are local to
        // dispatch_cmd() -- meaningless outside an AI request, so handled here rather than in command()
        if(command_name == "set_title" || command_name == "log" || command_name == "set_window")
        {
            QString output,error;
            if(command_name == "set_title")
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
                auto begin = std::max(info.log_position,first);
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
            else // set_window
            {
                auto param = cmd.size() > 1 ? QString::fromStdString(cmd[1]) : QString();
                QString new_window = "main";
                if(!param.isEmpty())
                {
                    bool bare_type = (param == "tracking" || param == "recon" || param == "image");
                    if(bare_type)
                    {
                        auto file_name = cmd.size() > 2 ? QString::fromStdString(cmd[2]) : QString();
                        if(file_name.isEmpty())
                            error = "set_window: a "+param+" window needs a file name (2nd param) to tell it apart from other open "+param+" windows";
                        else
                        {
                            QWidget* found = nullptr;
                            for(auto* each : QApplication::allWidgets())
                            {
                                auto id = ai_window_id(each);
                                if(id.startsWith(param) && id != param &&
                                   QFileInfo(each->windowTitle()).fileName().contains(file_name,Qt::CaseInsensitive))
                                {
                                    found = each;
                                    break;
                                }
                            }
                            if(!found)
                                error = "set_window: no "+param+" window matching \""+file_name+"\" is open";
                            else
                                new_window = ai_window_id(found);
                        }
                    }
                    else
                    {
                        bool exists = param == "main";
                        for(auto* each : QApplication::allWidgets())
                            if(!exists && ai_window_id(each) == param)
                                exists = true;
                        if(!exists)
                            error = "set_window: window \""+param+"\" not found, terminated by user?";
                        else
                            new_window = param;
                    }
                }
                if(error.isEmpty())
                {
                    if(new_window != ai_current_window)
                        unlock_target(); // retargeting: release whatever was locked for the previous window
                    ai_current_window = new_window;
                    output = "current window: "+new_window;
                }
            }
            if(!error.isEmpty())
            {
                results.append(command_result(false,{},error));
                break;
            }
            results.append(command_result(true,output));
            continue;
        }

        // list_window: enumerates all AI-addressable windows and their busy/idle/waiting status
        if(command_name == "list_window")
        {
            auto* modal = QApplication::activeModalWidget();
            bool application_busy = tipl::status_list.size() > 1;
            QJsonObject windows;
            for(auto* each : QApplication::allWidgets())
            {
                auto id = ai_window_id(each);
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
                bool waiting = modal && (modal == each || each->isAncestorOf(modal));
                windows[id] = QJsonObject{
                    {"status",waiting ? "waiting" : busy ? "busy" : "idle"},
                    {"title",QDir::fromNativeSeparators(each->windowTitle())}
                };
                application_busy |= busy;
            }
            {
                std::lock_guard<std::mutex> lock(shell_tasks_mutex);
                for(auto it = shell_tasks.constBegin();it != shell_tasks.constEnd();++it)
                    windows[it.key()] = QJsonObject{{"status","busy"},{"title",it.value()}};
                application_busy |= !shell_tasks.isEmpty();
            }
            results.append(command_result(true,QString::fromUtf8(QJsonDocument(QJsonObject{
                {"application",QJsonObject{{"status",modal ? "waiting" : application_busy ? "busy" : "idle"}}},
                {"windows",windows}}).toJson(QJsonDocument::Compact))));
            continue;
        }

        // everything else is offered to MainWindow first, with no target needed for that attempt -- this is
        // how voice/run_shell (implemented only in MainWindow::command()) just work, and also
        // covers ambiguous names like "open_fib" that exist differently elsewhere too.
        // only if MainWindow doesn't recognize the command at all does it fall through to the real target
        QString output,error;
        auto manual_hint = [](QString msg){return msg+". Read ai/DSI_STUDIO_AI_MANUAL.md and retry.";};
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            console.capture = &output;
        }
        try
        {
            bool handled_by_main = command(cmd,command_source::AI);
            if(!handled_by_main && error_msg == "unknown command: "+cmd[0])
            {
                if(resolve_target(error))
                {
                    auto target_type = ai_current_window == "main" ? QString("main") :
                                       ai_current_window.startsWith("tracking") ? "tracking" :
                                       ai_current_window.startsWith("recon") ? "recon" : "image";
                    auto target_title = target_type == "main" ? QString() :
                                        QFileInfo(locked_target->windowTitle()).fileName();
                    info.record_history(QJsonObject{
                        {"type","request"},
                        {"text",command_name+" → "+target_type+" window "+target_title},
                        {"window",ai_current_window}});

                    auto execute = [&](auto* window,bool success)
                    {
                        if(!success)
                        {
                            auto window_error = QString::fromUtf8(window->error_msg);
                            error = manual_hint(window_error.isEmpty() ? "command failed" : window_error);
                        }
                    };
                    if(auto* window = qobject_cast<tracking_window*>(locked_target))
                        execute(window,window->command(cmd,command_source::AI));
                    else if(auto* window = qobject_cast<reconstruction_window*>(locked_target))
                        execute(window,window->command(cmd,command_source::AI));
                    else if(auto* window = qobject_cast<view_image*>(locked_target))
                        execute(window,window->command(cmd,command_source::AI));
                    else
                        error = manual_hint(QString::fromStdString(error_msg)); // target is main itself; nothing else to try
                }
            }
            else if(!handled_by_main)
                error = manual_hint(QString::fromStdString(error_msg));
        }
        catch(const std::exception& e){error = e.what();}
        catch(...){error = "unknown error";}

        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            console.capture = nullptr;
        }

        output = strip_ansi(output);
        error = strip_ansi(error);

        results.append(command_result(error.isEmpty(),output,error));
        if(!error.isEmpty())
        {
            unlock_target();
            break;
        }
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
    if(cmd.empty())
        return fail("empty command");
    command_report report(this,"main",cmd,source);

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
        if(cmd.size() == 2)
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
        for(int index = 0;prog(index,files.size());++index)
        {
            auto file = tipl::qt::to_path(files[index]);
            if(rename_dicom(file,file.parent_path()).empty())
                result = false;
        }
        if(tipl::prog_aborted)
            return true;
        return result || fail("one or more DICOM files could not be renamed");
    }

    if(cmd[0] == "rename_dicom_dir")
    {
        auto dir = select_dir("Browse Directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        rename_dicom_at_dir(tipl::qt::to_path(dir),tipl::qt::to_path(dir));
        return true;
    }

    if(cmd[0] == "convert_dicom_dir")
    {
        auto dir = select_dir("Open directory");
        if(dir.isEmpty())
            return true;
        add_work_dir(dir);
        return dicom2src_and_nii(tipl::qt::to_path(dir),false) ||
               fail("DICOM conversion failed");
    }

    if(cmd[0] == "bids_to_src")
    {
        auto dir = select_dir("Open BIDS Folder");
        if(dir.isEmpty())
            return true;
        auto output_dir = QFileDialog::getExistingDirectory(
                              this,"Please Specify the Output Folder",
                              QDir(dir).path()+"/derivatives");
        if(output_dir.isEmpty())
            return true;
        add_work_dir(dir);
        auto files = search_dwi_nii_bids(tipl::qt::to_path(dir));
        if(files.empty())
        {
            std::string message("cannot find bids nifti data");
            return fail(message);
        }
        std::sort(files.begin(),files.end());
        return nii2src(files,tipl::qt::to_path(output_dir),true,true,false) ||
               fail("BIDS to SRC conversion failed");
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
        return nii2src(selected,output_dir,false,true,false,
                       "batch creating src") ||
               fail("NIfTI to SRC conversion failed");
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
        static Console* console = nullptr;
        if(!console)
            console = new Console(this);
        console->showNormal();
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

    if(cmd[0] == "qc_nii" || cmd[0] == "qc_src" || cmd[0] == "qc_fib")
    {
        bool nii = cmd[0] == "qc_nii",src = cmd[0] == "qc_src";
        auto filenames = select_images(
            nii ? "NIFTI files (*.nii *nii.gz);;All files (*)" :
            src ? "Src files (*.sz *src.gz);;All files (*)" :
                  "Fib files (*.fz *fib.gz);;All files (*)");
        if(filenames.isEmpty())
            return true;
        std::vector<std::filesystem::path> files;
        files.reserve(filenames.size());
        for(const auto& file : filenames)
            files.push_back(tipl::qt::to_path(file));
        tipl::progress prog(nii ? "checking NIFTI files" :
                            src ? "checking SRC files" : "checking FIB files");
        show_info_dialog(nii ? "NIFTI report" : src ? "SRC report" : "FIB report",
                         nii ? quality_check_nii_files(files) :
                         src ? quality_check_src_files(files,false,false,0) :
                               quality_check_fib_files(files));
        return true;
    }

    if(cmd[0] == "run_cli")
    {
        if(cmd.size() != 2)
            return fail("usage: run_cli <command line>");
        tipl::program_option<tipl::out> po;
        if(!po.parse(cmd[1]) || !po.check("action"))
            return fail(po.error_msg);
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
        report_window_created(window,"image");
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
        if(program.compare("dir",Qt::CaseInsensitive) &&
           program.compare("curl",Qt::CaseInsensitive))
            return fail("run_shell only allows dir, curl, and cd commands");
        for(auto c : QString("&|;<>^`\n\r"))
            if(text.contains(c))
                return fail("run_shell command contains disallowed characters");
        if(program.compare("curl",Qt::CaseInsensitive)) // dir: local and fast, just wait for it
        {
            QProcess process;
#ifdef Q_OS_WIN
            process.start("cmd.exe",QStringList() << "/c" << text);
#else
            process.start(text);
#endif
            if(!process.waitForStarted(3000))
                return fail("cannot start command");
            if(!process.waitForFinished(-1)) // no timeout: wait until it actually completes
                return fail("command did not finish");
            tipl::out() << process.readAllStandardOutput().toStdString();
            auto err = process.readAllStandardError().toStdString();
            if(!err.empty())
                tipl::error() << err;
            return true;
        }
        // curl can hang, so run it detached; "list_window" shows the id as busy until it finishes
        static std::atomic<int> next_curl_id{0};
        QString id = "curl"+QString::number(++next_curl_id);
        {
            std::lock_guard<std::mutex> lock(shell_tasks_mutex);
            shell_tasks[id] = text;
        }
        std::thread([text,id]
        {
            QProcess process;
#ifdef Q_OS_WIN
            process.start("cmd.exe",QStringList() << "/c" << text);
#else
            process.start(text);
#endif
            bool started = process.waitForStarted(3000);
            bool finished = started && process.waitForFinished(-1);
            if(finished)
                tipl::out() << process.readAllStandardOutput().toStdString();
            if(!started)
                tipl::error() << (id+" cannot start: "+text).toStdString();
            else if(!finished)
                tipl::error() << (id+" did not finish: "+text).toStdString();
            auto err = process.readAllStandardError().toStdString();
            if(!err.empty())
                tipl::error() << err;
            std::lock_guard<std::mutex> lock(shell_tasks_mutex);
            shell_tasks.remove(id);
        }).detach();
        tipl::out() << ("started "+id+": "+text).toStdString();
        return true;
    }
    return fail("unknown command: "+cmd[0]);
}
