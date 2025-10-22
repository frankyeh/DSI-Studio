#include <QFileDialog>
#include <QDateTime>
#include <QUrl>
#include <QMessageBox>
#include <QProgressDialog>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QAction>
#include <QStyleFactory>
#include <QNetworkInterface>
#include <QSysInfo>

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QRegularExpression>

#include <filesystem>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "regtoolbox.h"
#include "reconstruction/reconstruction_window.h"
#include "tracking/tracking_window.h"
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

QString access_token;

void checkForVersionSpecificBugs_Minimal(const QString& bugListText)
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
                              "Your program version is affected by the following known issues:\n\n- " +
                              matchingBugs.join("\n- ") +
                              "\n\nIt is highly recommended to update your program to the latest version to avoid these bugs.");
}

extern std::vector<std::string> fib_template_list;
std::vector<tracking_window*> tracking_windows;
MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);
    ui->styles->addItems(QStringList("default") << QStyleFactory::keys());
    ui->styles->setCurrentText(settings.value("styles","Fusion").toString());

    ui->recentFib->setColumnCount(3);
    ui->recentFib->setColumnWidth(0,300);
    ui->recentFib->setColumnWidth(1,250);
    ui->recentFib->setColumnWidth(2,150);
    ui->recentFib->setAlternatingRowColors(true);
    ui->recentSrc->setColumnCount(3);
    ui->recentSrc->setColumnWidth(0,300);
    ui->recentSrc->setColumnWidth(1,250);
    ui->recentSrc->setColumnWidth(2,150);
    ui->recentSrc->setAlternatingRowColors(true);
    QObject::connect(ui->recentFib,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_fib_at(int,int)));
    QObject::connect(ui->recentSrc,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_src_at(int,int)));
    updateRecentList();

    if (settings.contains("WORK_PATH"))
        ui->workDir->addItems(settings.value("WORK_PATH").toStringList());
    else
        ui->workDir->addItem(QDir::currentPath());
    ui->download_dir->setText(ui->workDir->currentText());

    for(auto& each : fib_template_list)
    {
        QString name = std::filesystem::path(each).stem().string().c_str();
        ui->template_list->addItem(name);
    }
    ui->tabWidget->setCurrentIndex(0);
    ui->github_release_note->setCurrentIndex(0);
    ui->github_release_note->setTabVisible(1,false);
    ui->github_open_file->setVisible(false);
    ui->github_open_file_mode->setVisible(false);
    ui->template_list->setCurrentRow(0);




    {
        auto reply = get(QString("https://raw.githubusercontent.com/frankyeh/DSI-Studio-Website/main/news.md"));
        while (!reply->isFinished())
            qApp->processEvents();
        QString news = reply->readAll();
        reply = get(QString("https://freegeoip.app/json/"));
        while (!reply->isFinished())
            qApp->processEvents();
        reply = get(QString("http://ip-api.com/json/%1").arg(QJsonDocument::fromJson(QString(reply->readAll()).toUtf8()).object().value("ip").toString()));
        while (!reply->isFinished())
            qApp->processEvents();

        {
            // Get the vcardArray
            QJsonObject jsonObject = QJsonDocument::fromJson(QString(reply->readAll()).toUtf8()).object();
            if(!jsonObject.isEmpty())
            {
                adrValue = jsonObject.value("city").toString() + "," +
                           jsonObject.value("region").toString() + "," +
                           jsonObject.value("countryCode").toString() + " " +
                           jsonObject.value("zip").toString() + " ";
                fnValue = jsonObject.value("as").toString();
            }
        }

        {
            QString licenseText;
            {
                QFile licenseFile(QApplication::applicationDirPath() + "/LICENSE");
                if (!licenseFile.open(QIODevice::ReadOnly))
                {
                    QMessageBox::critical(this,"ERROR","cannot locate license file");
                    return;
                }
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
                auto line_edit = new QLineEdit(QDir(QStandardPaths::writableLocation(QStandardPaths::HomeLocation)).dirName() + "," + fnValue + "," + adrValue);
                line_edit->setReadOnly(true);
                h_layout->addWidget(line_edit);
                right_layout->addLayout(h_layout);

            }

            if((fnValue.contains(" LLC") || fnValue.contains(" L.L.C") || fnValue.contains(" Inc") || fnValue.contains(" INC")) &&
                !fnValue.toLower().contains("com") && !fnValue.toLower().contains("tel"))
            {
                auto notice = new QLabel("This license agreement does not cover commercial use. For commercial entities, please contact frank.yeh@gmail.com to obtain a commercial license.");
                notice->setWordWrap(true);
                notice->setStyleSheet("color: red; font-weight: bold;");
                right_layout->addWidget(notice);
            }

            {
                auto note = new QLabel("By clicking 'Accept & Sign in', you agree to the licensing terms and will sign in using the registration information.");
                note->setWordWrap(true);
                note->setStyleSheet("font-weight: bold;");
                right_layout->addWidget(note);
            }

            {
                // MODIFYING REGISTRATION CODE INVALIDATES LICENSING AGREEMENT
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
                checkForVersionSpecificBugs_Minimal(news);
            }

            main_layout->addLayout(left_layout, 1);
            main_layout->addLayout(right_layout, 1);

            dialog->resize(1024,800);
            dialog->show();
        }
    }

    // load fiber data hub file list
    {
        QJsonDocument doc = QJsonDocument::fromJson(settings.value("file data hub").toString().toUtf8());
        QJsonObject root = doc.object();
        if (root.contains("dates"))
        {
            QJsonObject datesObj = root["dates"].toObject();
            for (const QString& key : datesObj.keys()) dates[key] = datesObj[key].toString();
        }
        if (root.contains("tags"))
        {
            QJsonObject tagsObj = root["tags"].toObject();
            for (const QString& key : tagsObj.keys()) tags[key] = tagsObj[key].toArray();
        }
    }

}

extern const char* version_string;
void MainWindow::login()
{
    // MODIFYING REGISTRATION CODE INVALIDATES LICENSING AGREEMENT
    setWindowTitle(windowTitle() + "(Offline)");
    QDnsLookup *dns = new QDnsLookup(this);
    dns->setType(QDnsLookup::TXT);
    dns->setName(DSI_STUDIO_LOGIN);
    connect(dns, &QDnsLookup::finished, this, [=]()
    {
        if (dns->error() == QDnsLookup::NoError)
            for (const auto &record : dns->textRecords())
            {
                login_with_param(QString(record.values().join("")).split(","));
                break;
            }
        dns->deleteLater();
    });
    dns->lookup();
}

void MainWindow::login_with_param(QStringList param)
{
    // MODIFYING REGISTRATION CODE INVALIDATES LICENSING AGREEMENT
    if(param.size() < 6)
        return;
    QNetworkRequest request(QUrl(param[0].toStdString().c_str()));
    request.setRawHeader("Content-Type", "application/json");
    QJsonObject data;
    data[param[1]] = QDir(QStandardPaths::writableLocation(QStandardPaths::HomeLocation)).dirName();
    data[param[2]] = fnValue;
    data[param[3]] = QSysInfo::productType() + QSysInfo::productVersion();
    data[param[4]] = QString(version_string) + " " + __DATE__;
    data[param[5]] = adrValue;
    QNetworkReply *reply = manager.post(request, QJsonDocument(data).toJson());
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
void MainWindow::openFile(QStringList file_names)
{
    QString file_name = file_names[0];
    if(!QFileInfo(file_name).exists())
    {
        if(file_name[0] == '-') // Mac pass a variable
            return;
        QMessageBox::critical(this,"ERROR",QString("Cannot find ") +
        file_name + " at current dir: " + QDir::current().dirName());
    }
    else
    {
        if(QString(file_name).endsWith(".csv"))
        {
            auto lines = tipl::read_text_file(file_name.toStdString());
            if(lines.empty() || !tipl::begins_with(lines[0],"open_fib,"))
            {
                QMessageBox::critical(this,"ERROR","invalid command csv file");
                return;
            }
            loadFib(QString::fromStdString(tipl::split(lines[0],',')[1]));
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
        if(QString(file_name).endsWith(".tt.gz") ||
           QString(file_name).endsWith(".trk") ||
           QString(file_name).endsWith(".trk.gz"))
        {
            auto file_list = QFileInfo(file_name).dir().entryList(QStringList("*fz"),QDir::Files|QDir::NoSymLinks);
            file_list << QFileInfo(file_name).dir().entryList(QStringList("*fib.gz"),QDir::Files|QDir::NoSymLinks);
            if(file_list.size() == 1)
            {
                loadFib(QFileInfo(file_name).absolutePath() + "/" + file_list[0]);
                for(auto each:file_names)
                    tracking_windows.back()->command({"open_tract",each.toStdString()});
            }
            else
                loadFib(file_name);
        }
        else
        if(QString(file_name).endsWith("fib.gz") ||
           QString(file_name).endsWith(".fz") ||
           QString(file_name).endsWith(".dz") ||
           QString(file_name).endsWith("tck"))
        {
            if(QString(file_name).endsWith("db.fib.gz") ||
               QString(file_name).endsWith("db.fz") ||
               QString(file_name).endsWith(".dz"))
            {
                std::shared_ptr<group_connectometry_analysis> database(new group_connectometry_analysis);
                if(database->load_database(file_name.toStdString().c_str()))
                {
                    db_window* db = new db_window(this,database);
                    db->setWindowTitle(file_name);
                    db->setAttribute(Qt::WA_DeleteOnClose);
                    db->show();
                }
            }
            else
                loadFib(file_name);
        }
        else
        if(QString(file_name).endsWith("src.gz") || QString(file_name).endsWith(".sz"))
        {
            loadSrc(file_names);
        }
        else
        if(QString(file_name).endsWith(".nhdr") ||
           QString(file_name).endsWith(".nrrd") ||
           QString(file_name).endsWith(".nii") ||
           QString(file_name).endsWith(".nii.gz") ||
                QString(file_name).endsWith(".dcm"))
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
    loadFib(ui->recentFib->item(row,1)->text() + "/" +
            ui->recentFib->item(row,0)->text());
}

void MainWindow::open_src_at(int row,int)
{
    loadSrc(QStringList() << (ui->recentSrc->item(row,1)->text() + "/" +
            ui->recentSrc->item(row,0)->text()));
}


void MainWindow::closeEvent(QCloseEvent *event)
{
    for(size_t index = 0;index < tracking_windows.size();++index)
    if(tracking_windows[index])
        {
            tracking_windows[index]->closeEvent(event);
            if(!event->isAccepted())
                return;
            delete tracking_windows[index];
        }
    QMainWindow::closeEvent(event);
}
MainWindow::~MainWindow()
{
    // save fiber data hub file list
    {
        QJsonObject root, datesObj, tagsObj;
        for (const auto& pair : dates) datesObj[pair.first] = pair.second;
        for (const auto& pair : tags) tagsObj[pair.first] = pair.second;
        root["dates"] = datesObj;
        root["tags"] = tagsObj;
        settings.setValue("file data hub", QJsonDocument(root).toJson(QJsonDocument::Compact));
    }

    console.log_window = nullptr;
    QStringList workdir_list;
    for (int index = 0;index < 10 && index < ui->workDir->count();++index)
        workdir_list << ui->workDir->itemText(index);
    std::swap(workdir_list[0],workdir_list[ui->workDir->currentIndex()]);
    settings.setValue("WORK_PATH", workdir_list);
    delete ui;

}


void MainWindow::updateRecentList(void)
{
    {
        QStringList file_list = settings.value("recentFibFileList").toStringList();
        ui->recentFib->clear();
        ui->recentFib->setRowCount(file_list.size());
        for (int index = 0;index < file_list.size();++index)
        {
            ui->recentFib->setRowHeight(index,20);
            ui->recentFib->setItem(index, 0, new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).filename().string().c_str()));
            ui->recentFib->setItem(index, 1, new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).parent_path().string().c_str()));
            ui->recentFib->item(index,0)->setFlags(ui->recentFib->item(index,0)->flags() & ~Qt::ItemIsEditable);
            ui->recentFib->item(index,1)->setFlags(ui->recentFib->item(index,1)->flags() & ~Qt::ItemIsEditable);
        }
    }
    {
        QStringList file_list = settings.value("recentSrcFileList").toStringList();
        ui->recentSrc->clear();
        ui->recentSrc->setRowCount(file_list.size());
        for (int index = 0;index < file_list.size();++index)
        {
            ui->recentSrc->setRowHeight(index,20);
            ui->recentSrc->setItem(index, 0, new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).filename().string().c_str()));
            ui->recentSrc->setItem(index, 1, new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).parent_path().string().c_str()));
            ui->recentSrc->item(index,0)->setFlags(ui->recentSrc->item(index,0)->flags() & ~Qt::ItemIsEditable);
            ui->recentSrc->item(index,1)->setFlags(ui->recentSrc->item(index,1)->flags() & ~Qt::ItemIsEditable);
        }
    }
    QStringList header;
    header << "File Name" << "Directory";
    ui->recentFib->setHorizontalHeaderLabels(header);
    ui->recentSrc->setHorizontalHeaderLabels(header);
}

void MainWindow::addFib(QString filename)
{
    // update recent file list
    QStringList files = settings.value("recentFibFileList").toStringList();
    files.removeAll(filename);
    files.prepend(filename);
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue("recentFibFileList", files);
    updateRecentList();
}

void MainWindow::addSrc(QString filename)
{
    // update recent file list
    QStringList files = settings.value("recentSrcFileList").toStringList();
    files.removeAll(filename);
    files.prepend(filename);
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue("recentSrcFileList", files);
    updateRecentList();
}
void shift_track_for_tck(std::vector<std::vector<float> >& loaded_tract_data,tipl::shape<3>& geo);
extern QByteArray default_geo,default_state;
void MainWindow::loadFib(QString filename)
{
    std::string file_name = filename.toStdString();
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        if(!new_handle->error_msg.empty())
            QMessageBox::critical(this,"ERROR",new_handle->error_msg.c_str());
        return;
    }
    tracking_windows.push_back(new tracking_window(this,new_handle));
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
    if(filename.endsWith("trk.gz") || filename.endsWith("trk") || filename.endsWith("tck") || filename.endsWith("tt.gz"))
    {
        tracking_windows.back()->command({"open_tract",filename.toStdString()});
        if(filename.endsWith("tck"))
        {
            tipl::shape<3> geo;
            shift_track_for_tck(tracking_windows.back()->tractWidget->tract_models.back()->get_tracts(),geo);
        }
    }
    if(!default_geo.size())
        default_geo = tracking_windows.back()->saveGeometry();
    if(!default_state.size())
        default_state = tracking_windows.back()->saveState();
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
    dialog->show();
}

void MainWindow::loadSrc(QStringList filenames)
{
    if(filenames.empty())
    {
        QMessageBox::critical(this,"ERROR","Cannot find SRC.gz files in the directory. Please create SRC files first.");
        return;
    }
    try
    {
        tipl::progress prog("[Step T2][Reconstruction]");
        reconstruction_window* new_mdi = new reconstruction_window(filenames,this);
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
        if(!tipl::prog_aborted)
            QMessageBox::critical(this,"ERROR",error.what());
    }

}


void MainWindow::openRecentFibFile(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    loadFib(action->data().toString());
}
void MainWindow::openRecentSrcFile(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    loadSrc(QStringList() << action->data().toString());
}

void MainWindow::open_DWI(QStringList filenames)
{
    if ( filenames.isEmpty() )
        return;
    tipl::progress prog("[Step T1][Open Source Images]");
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
                if(method_file.load_from_file(method_name.toStdString().c_str()) &&
                   method_file["PVM_DwEffBval"].length())
                    is_dwi = true;
            }
            // has dif info in the imnd file
            {
                tipl::io::bruker_info imnd_file;
                QString imnd_name = dir + "/" +QString::number(i)+"/imnd";
                if(imnd_file.load_from_file(imnd_name.toStdString().c_str()) &&
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

void MainWindow::on_Reconstruction_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                           this,
                           "Open Src files",
                           ui->workDir->currentText(),
                           "Src files (*.sz *src.gz);;Histology images (*.jpg *.tif);;All files (*)" );
    if (filenames.isEmpty())
        return;
    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    loadSrc(filenames);
}

void MainWindow::on_FiberTracking_clicked()
{

    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Fib files",
                           ui->workDir->currentText(),
                           "Fib files (*.fz *fib.gz *.dz);;All files (*)");
    if (filename.isEmpty())
        return;
    add_work_dir(QFileInfo(filename).absolutePath());
    loadFib(filename);
}

void MainWindow::on_T1WFiberTracking_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open T1W files",
                           ui->workDir->currentText(),
                           "Image files (*nii.gz *.nii 2dseq);;All files (*)");
    if (filename.isEmpty())
        return;
    add_work_dir(QFileInfo(filename).absolutePath());
    loadFib(filename);
}



void check_name(std::string& name)
{
    for(unsigned int index = 0;index < name.size();++index)
        if((name[index] < '0' || name[index] > '9') &&
           (name[index] < 'a' || name[index] > 'z') &&
           (name[index] < 'A' || name[index] > 'Z') &&
                name[index] != '.')
            name[index] = '_';
}

QString RenameDICOMToDir(QString FileName, QString ToDir)
{
    QString NewName;
    {
        std::string person, sequence, imagename;
        {
            tipl::io::dicom header;
            if (!header.load_from_file(FileName.toStdString().c_str()))
            {
                tipl::out() << "not a DICOM file. Skipping" << std::endl;
                return QString();
            }
            header.get_patient(person);
            header.get_sequence(sequence);
            header.get_image_name(imagename);
        }
        check_name(person);
        check_name(sequence);
        check_name(imagename);

        QString Person(person.c_str()), Sequence(sequence.c_str()),
        ImageName(imagename.c_str());
        ToDir += "/";
        ToDir += Person;
        if (!QDir(ToDir).exists() && !std::filesystem::create_directory(std::filesystem::path(ToDir.toStdString())))
            tipl::error() << "cannot create dir " << ToDir.toStdString() << std::endl;
        ToDir += "/";
        ToDir += Sequence;
        if (!QDir(ToDir).exists() && !std::filesystem::create_directory(std::filesystem::path(ToDir.toStdString())))
            tipl::error() << "cannot create dir " << ToDir.toStdString() << std::endl;
        ToDir += "/";
        ToDir += ImageName;
        NewName = ToDir;
    }
    if(FileName != NewName)
    {
        tipl::out() << FileName.toStdString() << "->" << NewName.toStdString() << std::endl;
        if(!QFile::rename(FileName,NewName))
            tipl::error() << "cannot rename the file." << std::endl;
    }
    return NewName;
}

void MainWindow::on_RenameDICOM_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,
                                "Open DICOM files",
                                ui->workDir->currentText(),
                                "All files (*)" );
    if ( filenames.isEmpty() )
        return;
    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    tipl::progress prog("Rename DICOM Files");
    for (unsigned int index = 0;prog(index,filenames.size());++index)
        RenameDICOMToDir(filenames[index],QFileInfo(filenames[index]).absolutePath());
}


void MainWindow::add_work_dir(QString dir)
{
    if(ui->workDir->findText(dir) != -1)
        ui->workDir->removeItem(ui->workDir->findText(dir));
    ui->workDir->insertItem(0,dir);
    ui->workDir->setCurrentIndex(0);
}



void MainWindow::on_browseDir_clicked()
{
    QString filename =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( filename.isEmpty() )
        return;
    add_work_dir(filename);
}


QStringList GetSubDir(QString Dir,bool recursive = true)
{
    QStringList sub_dirs;
    QStringList dirs = QDir(Dir).entryList(QStringList("*"),
                                            QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);
    if(recursive)
        sub_dirs << Dir;
    for(int index = 0;index < dirs.size();++index)
    {
        QString new_dir = Dir + "/" + dirs[index];
        if(recursive)
            sub_dirs << GetSubDir(new_dir,recursive);
        else
            sub_dirs << new_dir;
    }
    return sub_dirs;
}

QStringList rename_dicom_at_dir(QString path,QString output)
{
    tipl::progress prog("Renaming DICOM",true);
    tipl::out() << "current directory is " << std::filesystem::current_path() << std::endl
                    << "source directory is " << path.toStdString() << std::endl
                    << "output directory is " << output.toStdString() << std::endl;
    QStringList dirs = GetSubDir(path);
    QStringList subject_dirs;
    subject_dirs.resize(dirs.size());
    tipl::par_for(dirs.size(),[&](size_t index)
    {
        QStringList files = QDir(dirs[index]).entryList(QStringList("*"),QDir::Files | QDir::NoSymLinks);
        for(int j = 0;j < files.size() && index < dirs.size();++j)
        {
            auto dir = QFileInfo(RenameDICOMToDir(dirs[index] + "/" + files[j],output)).absoluteDir();
            dir.cdUp();
            subject_dirs[index] = dir.absolutePath();
        }
    });
    subject_dirs.removeDuplicates();
    return subject_dirs;
}
void MainWindow::on_RenameDICOMDir_clicked()
{
    QString path =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( path.isEmpty() )
        return;
    add_work_dir(path);
    rename_dicom_at_dir(path,path);
    QMessageBox::information(this,QApplication::applicationName(),"renaming complete");
}

void MainWindow::on_vbc_clicked()
{
    CreateDBDialog* new_mdi = new CreateDBDialog(this,true);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->show();
}

void MainWindow::on_averagefib_clicked()
{
    CreateDBDialog* new_mdi = new CreateDBDialog(this,false);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->show();
}

bool parse_dwi(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg);
QString get_dicom_output_name(QString file_name,QString file_extension,bool add_path);
QStringList search_files(QString dir,QString filter);
void MainWindow::on_batch_reconstruction_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);
    loadSrc(search_files(dir,"*src.gz") << search_files(dir,"*.sz"));
}

void MainWindow::on_view_image_clicked()
{
    QStringList filename = QFileDialog::getOpenFileNames(
                                this,
                                "Open Image",
                                ui->workDir->currentText(),
                                "image files (*.nii *nii.gz *.dcm *.nhdr *.nrrd 2dseq *.fz *.dz *fib.gz *.sz *src.gz)" );
    if(filename.isEmpty())
        return;
    add_work_dir(QFileInfo(filename[0]).absolutePath());
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if(!dialog->open(filename))
    {
        QMessageBox::critical(this,"ERROR",dialog->error_msg.c_str());
        delete dialog;
        return;
    }
    dialog->show();
}

void MainWindow::on_workDir_currentTextChanged(const QString &arg1)
{
    if(!arg1.isEmpty())
        QDir::setCurrent(arg1);
}

bool MainWindow::load_db(std::shared_ptr<group_connectometry_analysis>& database,QString& filename)
{
    filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Database files",
                           ui->workDir->currentText(),
                           "Database (*.dz *db.fz *db?fib.gz);;All files (*)");
    if (filename.isEmpty())
        return false;
    add_work_dir(QFileInfo(filename).absolutePath());
    database = std::make_shared<group_connectometry_analysis>();
    tipl::progress prog_("reading connectometry db");
    if(!database->load_database(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR",database->error_msg.c_str());
        return false;
    }
    return true;
}

void MainWindow::on_open_db_clicked()
{
    QString filename;
    std::shared_ptr<group_connectometry_analysis> database;
    if(!load_db(database,filename))
        return;
    db_window* db = new db_window(this,database);
    db->setWindowTitle(filename);
    db->setAttribute(Qt::WA_DeleteOnClose);
    db->show();
}

void MainWindow::on_group_connectometry_clicked()
{
    QString filename;
    std::shared_ptr<group_connectometry_analysis> database;
    if(!load_db(database,filename))
        return;
    group_connectometry* group_cnt = new group_connectometry(this,database,filename);
    group_cnt->setAttribute(Qt::WA_DeleteOnClose);
    group_cnt->show();
}


bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>& trans);

void MainWindow::on_linear_reg_clicked()
{
    QStringList filename1 = QFileDialog::getOpenFileNames(
            this,"Open Subject Image",ui->workDir->currentText(),
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename1.isEmpty())
        return;


    QStringList filename2 = QFileDialog::getOpenFileNames(
            this,"Open Template Image",QFileInfo(filename1[0]).absolutePath(),
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename2.isEmpty())
        return;


    tipl::image<3> ref1,ref2;
    tipl::vector<3> vs1,vs2;
    tipl::matrix<4,4> t1,t2;
    if(!load_image_from_files(filename1,ref1,vs1,t1) ||
       !load_image_from_files(filename2,ref2,vs2,t2))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,subject_image_pre(tipl::image<3>(ref1)),tipl::image<3,unsigned char>(),vs1,
                                                                       template_image_pre(tipl::image<3>(ref2)),tipl::image<3,unsigned char>(),vs2,tipl::reg::affine,tipl::reg::mutual_info));
    manual->from_T = t1;
    manual->to_T = t2;

    if(manual->exec() != QDialog::Accepted)
        return;
}

void MainWindow::on_nonlinear_reg_clicked()
{
    RegToolBox* rt = new RegToolBox(this);
    rt->setAttribute(Qt::WA_DeleteOnClose);
    rt->showNormal();
}

std::string quality_check_src_files(const std::vector<std::string>& file_list,
                                    bool check_btable,bool use_template,unsigned int template_id);
std::string quality_check_fib_files(const std::vector<std::string>& file_list);
std::string quality_check_nii_files(const std::vector<std::string>& file_list);

void MainWindow::on_SRC_qc_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                           this,"Open SRC files",
                           ui->workDir->currentText(),
                           "Src files (*.sz *src.gz);;All files (*)" );
    if (filenames.isEmpty())
        return;
    std::vector<std::string> files;
    for(auto each : filenames)
        files.push_back(each.toStdString());
    tipl::progress prog("checking SRC files");
    show_info_dialog("SRC report",quality_check_src_files(files,false,false,0));
}


void MainWindow::on_NII_qc_clicked()
{
    auto filenames = QFileDialog::getOpenFileNames(
                           this,"Open NIFTI files",
                           ui->workDir->currentText(),
                           "NIFTI files (*.nii *nii.gz);;All files (*)");
    if (filenames.isEmpty())
        return;
    std::vector<std::string> files;
    for(auto each : filenames)
        files.push_back(each.toStdString());
    tipl::progress prog("checking NIFTI files");
    show_info_dialog("NIFTI report",quality_check_nii_files(files));
}



void MainWindow::on_FIB_qc_clicked()
{
    auto filenames = QFileDialog::getOpenFileNames(
                           this,"Open FIB files",
                           ui->workDir->currentText(),
                           "Fib files (*.fz *fib.gz);;All files (*)");

    if (filenames.isEmpty())
        return;
    std::vector<std::string> files;
    for(auto each : filenames)
        files.push_back(each.toStdString());
    tipl::progress prog("checking FIB files");
    show_info_dialog("FIB report",quality_check_fib_files(files));
}

void MainWindow::on_parse_network_measures_clicked()
{
    QStringList filename = QFileDialog::getOpenFileNames(
            this,"Open Network Measures",ui->workDir->currentText(),
            "Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    std::ofstream out((filename[0]+".collected.txt").toStdString().c_str());
    out << "Field\t";
    for(int i = 0;i < filename.size();++i)
        out << QFileInfo(filename[i]).baseName().toStdString() << "\t";
    out << std::endl;

    std::vector<std::string> line_output;
    for(int i = 0;i < filename.size();++i)
    {
        std::ifstream in(filename[i].toStdString().c_str());
        std::vector<std::string> node_list;
        // global measures
        size_t line_index = 0;
        while(in)
        {
            std::string t1,t2;
            in >> t1;
            if(t1 == "network_measures")
            {
                std::string nodes;
                std::getline(in,nodes);
                std::istringstream nodestream(nodes);
                std::copy(std::istream_iterator<std::string>(nodestream),
                          std::istream_iterator<std::string>(),std::back_inserter(node_list));
                break;
            }
            in >> t2;
            if(i == 0)
            {
                line_output.push_back(t1);
                line_output.back() += "\t";
            }
            line_output[line_index] += t2;
            line_output[line_index] += "\t";
            ++line_index;
        }
        // nodal measures
        std::string line;
        while(std::getline(in,line))
        {
            std::istringstream in2(line);
            std::string t1;
            in2 >> t1;
            if(t1[0] == '#' || t1[0] == ' ')
                continue;
            for(size_t k = 0;k < node_list.size();++k,++line_index)
            {
                std::string t2;
                in2 >> t2;
                if(i==0)
                {
                    line_output.push_back(t1);
                    line_output.back() += "_";
                    line_output.back() += node_list[k];
                    line_output.back() += "\t";
                }
                line_output[line_index] += t2;
                line_output[line_index] += "\t";
            }
        }
    }
    for(size_t i = 0;i < line_output.size();++i)
        out << line_output[i] << std::endl;

    QMessageBox::information(this,QApplication::applicationName(),QString("File saved to")+filename[0]+".collected.txt");

}

void MainWindow::on_auto_track_clicked()
{
    auto_track* at = new auto_track(this);
    at->setAttribute(Qt::WA_DeleteOnClose);
    at->showNormal();
}



bool get_pe_dir(const std::string& nii_name,size_t& pe_dir,bool& is_neg)
{
    const char pe_coding[3][2][5] = { { "\"i\"","\"i-\"" },
                                       { "\"j\"","\"j-\"" },
                                       { "\"k\"","\"k-\"" }};
    std::string json_name(nii_name);
    tipl::remove_suffix(json_name,".nii.gz");
    tipl::remove_suffix(json_name,".nii");
    json_name += ".json";
    if(!std::filesystem::exists(json_name))
        return false;

    std::stringstream buffer;
    buffer << std::ifstream(json_name).rdbuf();
    std::string json_content(buffer.str());
    for(pe_dir = 0;pe_dir < 3;++pe_dir)
    {
        if(json_content.find(pe_coding[pe_dir][0]) != std::string::npos)
        {
            is_neg = false;
            return true;
        }
        if(json_content.find(pe_coding[pe_dir][1]) != std::string::npos)
        {
            is_neg = true;
            return true;
        }
    }
    return false;
}
std::vector<std::string> search_dwi_nii_bids(const std::string& dir);
void search_dwi_nii(const std::string& dir,std::vector<std::string>& dwi_nii_files);
void MainWindow::batch_create_src(const std::vector<std::string>& dwi_nii_files,const std::string& output_dir)
{
    if(dwi_nii_files.empty())
    {
        QMessageBox::critical(this,"ERROR","no dwi nifti files found");
        return;
    }
    bool no_to_all = false;
    bool yes_to_all = false;
    tipl::progress prog("batch creating src");
    std::deque<std::string> nii_list,src_list;
    size_t nii_count = 0;
    std::mutex access_list;
    bool ended = false;
    tipl::par_for<tipl::sequential_with_id>(8,[&](unsigned int index,unsigned int id)
    {
        if(id == 0) // main thread
        {
            for(int j = 0;j < dwi_nii_files.size();++j)
            {
                std::string nii_name = dwi_nii_files[j];
                std::string src_name = output_dir + "/" + std::filesystem::path(nii_name).stem().stem().u8string() + ".sz";
                std::vector<std::shared_ptr<DwiHeader> > dwi_files;

                if(std::filesystem::exists(src_name) && !yes_to_all)
                {
                    if(no_to_all)
                        continue;
                    int result = QMessageBox::information(this,QApplication::applicationName(),
                                    QString("%1 exists, overwrite?").arg(std::filesystem::path(src_name).filename().c_str()),
                                    QMessageBox::Yes|QMessageBox::YesToAll|QMessageBox::No|QMessageBox::NoToAll|QMessageBox::Cancel);
                    if(result == QMessageBox::Cancel)
                        return;
                    if(result == QMessageBox::YesToAll)
                        yes_to_all = true;
                    if(result == QMessageBox::NoToAll)
                    {
                        no_to_all = true;
                        continue;
                    }
                    if(result == QMessageBox::No)
                        continue;
                }
                std::lock_guard<std::mutex> lock(access_list);
                nii_list.push_back(nii_name);
                src_list.push_back(src_name);
                ++nii_count;
            }
            ended = true;
        }
        while(!prog.aborted() && !(ended && nii_count == 0))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if(id == 0)
                prog(dwi_nii_files.size()-nii_list.size(),dwi_nii_files.size());

            std::string nii_name,src_name;
            {
                std::lock_guard<std::mutex> lock(access_list);
                if(!nii_count)
                    continue;
                nii_name = nii_list.front();
                src_name = src_list.front();
                nii_list.pop_front();
                src_list.pop_front();
                --nii_count;
            }
            tipl::out() << "processing " << nii_name << std::endl;
            src_data src;
            if(!src.load_from_file(std::vector<std::string>({nii_name}),true) ||
               !src.save_to_file(src_name))
                tipl::warning() << src.error_msg;
        }
    },8);
}
bool nii2src(const std::vector<std::string>& dwi_nii_files,
             const std::string& output_dir,
             bool is_bids,
             bool overwrite,
             bool topup_eddy);
void MainWindow::on_nii2src_bids_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                    this,
                                    "Open BIDS Folder",
                                    ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    QString output_dir = QFileDialog::getExistingDirectory(
                                    this,
                                    "Please Specify the Output Folder",
                                    QDir(dir).path()+"/derivatives");
    if(output_dir.isEmpty())
        return;
    add_work_dir(dir);
    auto dwi_nii_files = search_dwi_nii_bids(dir.toStdString());
    if(dwi_nii_files.empty())
    {
        QMessageBox::critical(this,"ERROR","cannot find bids nifti data");
        return;
    }
    std::sort(dwi_nii_files.begin(),dwi_nii_files.end());
    nii2src(dwi_nii_files,output_dir.toStdString(),true,true,false);
}
void MainWindow::on_nii2src_sf_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                    this,
                                    "Open directory",
                                    ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);
    std::vector<std::string> dwi_nii_files;
    search_dwi_nii(dir.toStdString(),dwi_nii_files);
    if(dwi_nii_files.empty())
    {
        QMessageBox::critical(this,"ERROR","cannot find nifti data");
        return;
    }
    batch_create_src(dwi_nii_files,dir.toStdString());
}

bool dcm2src_and_nii(QStringList files,bool overwrite)
{
    if(files.empty())
        return false;
    files.sort();
    tipl::progress p("processing DICOM at "+std::filesystem::path(files[0].toStdString()).parent_path().string());
    // extract information
    std::string manu,make,report,sequence;
    {
        tipl::io::dicom header;
        if(!header.load_from_file(files[0].toStdString().c_str()))
        {
            tipl::error() << "cannot read image volume. skip" << std::endl;
            return false;
        }
        header.get_sequence_id(sequence);
        header.get_text(0x0008,0x0070,manu);//Manufacturer
        header.get_text(0x0008,0x1090,make);
        std::replace(manu.begin(),manu.end(),' ',char(0));
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
    auto nii_file_name = get_dicom_output_name(files[0],("_" + sequence + ".nii.gz").c_str(),true).toStdString();

    if(!parse_dwi(files,dicom_files,error_msg) || dicom_files.size() == 1)
    {
        if(tipl::prog_aborted)
            return false;
        if(!error_msg.empty())
        {
            tipl::error() << error_msg;
            return false;
        }

        if(!overwrite && std::filesystem::exists(nii_file_name))
        {
            tipl::out() << nii_file_name << " exists. skipping";
            return true;
        }

        tipl::out() << "handled as structure images";
        tipl::image<3> source_images;
        tipl::vector<3> vs;

        if(files.size()==1)
        {
            tipl::io::dicom v;
            if(!v.load_from_file(files[0].toStdString()))
            {
                tipl::error() << "cannot parse dicom file" << std::endl;
                return false;
            }
            v >> source_images;
            v.get_voxel_size(vs);
        }
        else
        {
            tipl::out() << "parsing " << files.size() << " dicom files";
            std::sort(files.begin(),files.end(),compare_qstring());
            tipl::io::dicom_volume v;
            std::vector<std::string> file_list;
            for(int index = 0;index < files.size();++index)
                file_list.push_back(files[index].toStdString().c_str());
            if(!v.load_from_files(file_list))
            {
                tipl::out() << v.error_msg.c_str() << std::endl;
                return false;
            }
            tipl::out() << "dim: " << v.dim;
            tipl::out() << "vs: " << v.vs;
            tipl::out() << "trans: " << tipl::matrix<3,3,float>(v.orientation_matrix);
            tipl::out() << "dim order: " << tipl::vector<3,int>(v.dim_order);
            tipl::out() << "flipping: " << tipl::vector<3,int>(v.flip);
            v >> source_images;
            v.get_voxel_size(vs);
        }
        if(source_images.empty())
        {
            tipl::error() << "cannot parse as image volume";
            return false;
        }
        tipl::matrix<4,4,float> trans;
        initial_LPS_nifti_srow(trans,source_images.shape(),vs);
        tipl::out() << "converted to NIFTI: " << std::filesystem::path(nii_file_name).filename().u8string() << std::endl;
        std::string nii_file_name_tmp = nii_file_name + ".tmp";
        if(!tipl::io::gz_nifti::save_to_file(nii_file_name_tmp,source_images,vs,trans))
        {
            tipl::error() << "cannot write to a temporary file " + nii_file_name_tmp;
            return false;
        }
        if(std::filesystem::exists(nii_file_name))
            std::filesystem::remove(nii_file_name);
        std::filesystem::rename(nii_file_name_tmp,nii_file_name);
        return true;
    }

    if(!DwiHeader::has_b_table(dicom_files))
    {
        if(!overwrite && std::filesystem::exists(nii_file_name))
        {
            tipl::out() << nii_file_name << " exists. skipping";
            return true;
        }
        tipl::out() << "The images do not have b-table. Save as 4D NIFTI" << std::endl;
        auto dicom = dicom_files[0];
        tipl::matrix<4,4> trans;
        initial_LPS_nifti_srow(trans,dicom->image.shape(),dicom->voxel_size);

        tipl::shape<4> nifti_dim;
        std::copy(dicom->image.shape().begin(),
                  dicom->image.shape().end(),nifti_dim.begin());
        nifti_dim[3] = uint32_t(dicom_files.size());

        tipl::image<4,unsigned short> buffer(nifti_dim);
        for(unsigned int index = 0;index < dicom_files.size();++index)
        {
            std::copy(dicom_files[index]->image.begin(),
                      dicom_files[index]->image.end(),
                      buffer.begin() + long(index*dicom_files[index]->image.size()));
        }
        tipl::out() << "Create 4D NII file: " << nii_file_name << std::endl;
        std::string nii_file_name_tmp = nii_file_name + ".tmp";
        if(!tipl::io::gz_nifti::save_to_file(nii_file_name_tmp,buffer,dicom->voxel_size,trans,false,report.c_str()))
        {
            tipl::error() << "cannot write to a temporary file " + nii_file_name_tmp;
            return false;
        }
        if(std::filesystem::exists(nii_file_name))
            std::filesystem::remove(nii_file_name);
        std::filesystem::rename(nii_file_name_tmp,nii_file_name);
        return true;
    }

    auto src_name = get_dicom_output_name(files[0],(std::string("_")+sequence+".sz").c_str(),true).toStdString();
    if(!overwrite && std::filesystem::exists(src_name))
    {
        tipl::out() << src_name << " exists. skipping";
        return true;
    }
    src_data src;
    if(!src.load_from_file(dicom_files,false) ||
       !src.save_to_file(src_name))
    {
        tipl::error() << src.error_msg;
        return false;
    }
    return true;
}

void dicom2src_and_nii(std::string dir_,bool overwrite)
{
    tipl::progress prog("convert DICOM to NIFTI/SRC");
    QStringList dir_list = GetSubDir(dir_.c_str(),false);
    bool has_dicom = false;
    for(int i = 0;prog(i,dir_list.size());++i)
    {
        QDir cur_dir = dir_list[i];
        QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        if(dicom_file_list.empty())
            continue;
        has_dicom = true;
        // aggregate DWI with identical names from consecutive folders
        QStringList aggregated_file_list;
        for(;prog(i,dir_list.size());++i)
        {
            for (int index = 0;index < dicom_file_list.size();++index)
                aggregated_file_list << dir_list[i] + "/" + dicom_file_list[index];
            if(i+1 < dir_list.size() && !QFileInfo(dir_list[i+1] + "/" + dicom_file_list[0]).exists())
                break;
        }
        dcm2src_and_nii(aggregated_file_list,overwrite);
    }
    if(!has_dicom)
        for(auto dir : dir_list)
            dicom2src_and_nii(dir.toStdString(),overwrite);
}

void MainWindow::on_dicom2nii_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);
    dicom2src_and_nii(dir.toStdString(),false);
}




void MainWindow::on_xnat_download_clicked()
{
    auto* xnat = new xnat_dialog(this);
    xnat->setAttribute(Qt::WA_DeleteOnClose);
    xnat->showNormal();
}


void MainWindow::on_styles_activated(int)
{
    if(ui->styles->currentText() != settings.value("styles","Fusion").toString())
    {
        settings.setValue("styles",ui->styles->currentText());
        QMessageBox::information(this,QApplication::applicationName(),"You will need to restart DSI Studio to see the change");
    }
}

void MainWindow::on_clear_settings_clicked()
{
    QSettings(QSettings::SystemScope,"LabSolver").clear();
    QMessageBox::information(this,QApplication::applicationName(),"Setting Cleared");
}


void MainWindow::on_console_clicked()
{
    static Console* con(0);
    if(!con)
        con = new Console(this);
    con->showNormal();
}





void MainWindow::on_recentFib_cellClicked(int row, int column)
{
    ui->open_selected_fib->setEnabled(true);
}

void MainWindow::on_recentSrc_cellClicked(int row, int column)
{
    ui->open_selected_src->setEnabled(true);
}

void MainWindow::on_clear_src_history_clicked()
{
    ui->recentSrc->setRowCount(0);
    ui->open_selected_src->setEnabled(false);
    settings.setValue("recentSRCFileList", QStringList());
}

void MainWindow::on_open_selected_src_clicked()
{
     open_src_at(ui->recentSrc->currentRow(),0);
}

void MainWindow::on_clear_fib_history_clicked()
{
    ui->recentFib->setRowCount(0);
    ui->open_selected_fib->setEnabled(false);
    settings.setValue("recentFibFileList", QStringList());
}

void MainWindow::on_open_selected_fib_clicked()
{
    open_fib_at(ui->recentFib->currentRow(),0);
}


void MainWindow::on_template_list_itemDoubleClicked(QListWidgetItem *item)
{
    open_template(item->text());
}

void MainWindow::open_template(QString name)
{
    for(auto& each : fib_template_list)
        if(std::filesystem::path(each).stem().string() == name.toStdString())
        {
            loadFib(each.c_str());
            tracking_windows.back()->work_path.clear();
            return;
        }
}


void MainWindow::on_TemplateFiberTracking_clicked()
{
    if(ui->template_list->currentRow() >= 0)
        open_template(ui->template_list->item(ui->template_list->currentRow())->text());
}



void MainWindow::on_OpenDWI_NIFTI_clicked()
{
    open_DWI(QStringList() << QFileDialog::getOpenFileName(
                         this,
                         "Open NIFTI file",
                         ui->workDir->currentText(),
                         "NIFTI files (*.nii *.nii.gz);;All files (*)" ));
}


void MainWindow::on_OpenDWI_DICOM_clicked()
{
    open_DWI(QFileDialog::getOpenFileNames(
                         this,
                         "Open DICOM files",
                         ui->workDir->currentText(),
                         "DICOM files (*.dcm);;All files (*)" ));
}


void MainWindow::on_OpenDWI_2dseq_clicked()
{
    open_DWI(QFileDialog::getOpenFileNames(
                         this,
                         "Open 2dseq or Variant files",
                         ui->workDir->currentText(),
                         "2dseq files (2dseq);;FDF files (*.fdf);;NRRD Files (*.nrrd);;All files (*)" ));
}
void MainWindow::on_tabWidget_currentChanged(int index)
{
    if(index == 4 && !ui->github_tags->rowCount() && !fetch_github)
    {
        auto reply = get(QString("https://raw.githubusercontent.com/frankyeh/Brain-Data/gh-pages/index.md"));
        while (!reply->isFinished())
            qApp->processEvents();
        QString content = reply->readAll();
        QStringList lines = content.split('\n');
        lines.removeFirst();
        QStringList filteredLines;
        for (const QString &line : lines) {
            if (!line.trimmed().startsWith("<img src"))
                filteredLines.append(line);
            if(line.startsWith("- "))
            {
                QRegularExpression re("\\[([^\\]]+)\\]\\(https://github\\.com/([^/]+/[^/]+)/");
                QRegularExpressionMatch match = re.match(line);
                if (match.hasMatch())
                {
                    QString label = match.captured(1);     // "HCP lifespan studies"
                    QString userRepo = match.captured(2);  // "data-hcp/lifespan"
                    ui->github_repo->addItem(label,userRepo);
                }
            }
        }

        ui->github_note->setMarkdown(filteredLines.join("\n"));
        ui->github_note->setReadOnly(true);
        ui->github_note->setOpenExternalLinks(true);
        fetch_github = true;
        on_github_repo_currentIndexChanged(0);
    }
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

void MainWindow::on_github_repo_currentIndexChanged(int index)
{
    if(ui->github_repo->currentIndex() < 0 || !fetch_github)
        return;
    QString repo = ui->github_repo->currentData().toString();
    if(tags.find(repo) == tags.end())
    {
        tags[repo] = QJsonArray();
        dates[repo] = QString();
        on_load_tags_clicked();
        return;
    }

    notes.clear();
    assets.clear();
    ui->github_tags->setSortingEnabled(false);
    ui->github_tags->setRowCount(0);

    std::map<QString, std::pair<QString,std::set<std::string> > > agg;
    foreach (const QJsonValue& release, tags[repo])
    {
        auto object = release.toObject();
        auto tag = object.value("tag_name").toString();
        if(tag.length() > 2 && tag[tag.length()-2] == '_' && tag[tag.length()-1] >= '1' && tag[tag.length()-1] <= '9')
            tag.chop(2);

        notes[tag] = object.value("body").toString();
        auto& agg_at_tag = agg[tag];
        agg_at_tag.first = object.value("name").toString();
        auto& names = agg_at_tag.second;
        foreach (const auto& each, object.value("assets").toArray())
        {
            auto fn = each.toObject().value("name").toString().toStdString();
            if (fn.empty() || fn.back()!='z' || tipl::ends_with(fn,".db.fz") || tipl::ends_with(fn,".dz"))
                continue;
            names.insert(fn.substr(0, std::min(fn.find('_'), fn.find('.'))));
            assets[tag].append(each);
        }
    }
    if(dates[repo].isEmpty())
        ui->tag_date->setText("Loading...");
    else
        ui->tag_date->setText("Last sync:" + dates[repo]);

    for (const auto& each : agg)
    {
        int row=ui->github_tags->rowCount();
        ui->github_tags->insertRow(row);
        ui->github_tags->setItem(row,0,new QTableWidgetItem(each.first));
        ui->github_tags->setItem(row,1,new QTableWidgetItem(QString::number(each.second.second.size())));
        ui->github_tags->setItem(row,2,new QTableWidgetItem(QString::number(assets[each.first].size())));
        ui->github_tags->setItem(row,3,new QTableWidgetItem(each.second.first));
    }

    ui->github_tags->sortByColumn(0,Qt::AscendingOrder);
    ui->github_tags->setSortingEnabled(true);
    ui->github_tags->resizeRowsToContents();
    ui->github_tags->resizeColumnToContents(0);
    ui->github_tags->resizeColumnToContents(1);
    ui->github_tags->resizeColumnToContents(2);
}


void MainWindow::on_load_tags_clicked()
{
    if(ui->github_repo->currentIndex() < 0 || !fetch_github)
        return;
    QString repo = ui->github_repo->currentData().toString();
    QString url = QString("https://api.github.com/repos/%1/releases").arg(repo);
    ui->github_tags->setSortingEnabled(false);
    ui->github_tags->setRowCount(0);
    ui->tag_date->setText("Loading...");
    ui->load_tags->setEnabled(false);
    notes.clear();
    assets.clear();
    std::vector<int> per_page = {64,32,16,8,4};
    QTimer::singleShot(0,this, [=](){loadTags(QUrl(url), repo, QJsonArray(), per_page[std::min<int>(per_page.size()-1,github_api_rate_limit/15)]);});
}

QString showQNetworkReplyError(QNetworkReply* reply)
{
    int http_error = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if(http_error)
        return QMap<int, QString>({
            {301, "Moved Permanently - The requested resource has been permanently moved to a new location."},
            {302, "Found - The requested resource resides temporarily under a different URI."},
            {304, "Not Modified - The server has fulfilled the request, but the document has not been modified."},
            {400, "Bad Request - The request was invalid."},
            {401, "Unauthorized - Valid authentication credentials are required."},
            {404, "Permission Needed - The resource requires access permission."},
            {405, "Method Not Allowed - The request method is not supported for the requested resource."},
            {408, "Request Timeout - The server timed out waiting for the request."},
            {500, "Internal Server Error - The server encountered an unexpected condition."},
            {502, "Bad Gateway - The server received an invalid response from an upstream server."},
            {503, "Service Unavailable - The server is currently unable to handle the request."},
            {504, "Gateway Timeout - The server did not receive a timely response from an upstream server."},
                                      }).value(http_error,"error code: " + QString::number(http_error));

    return reply->errorString();
}

void MainWindow::update_rate_limit(QSharedPointer<QNetworkReply> reply)
{
    if(reply->rawHeader("X-RateLimit-Remaining").toInt() == 0)
        return;
    tipl::out() << "api rate limit: " << (github_api_rate_limit = reply->rawHeader("X-RateLimit-Remaining").toInt());
}

void MainWindow::loadTags(QUrl url,QString repo,QJsonArray array,int per_page)
{
    static int retryCount = 0;
    {
        QUrlQuery q(url.query());
        q.removeAllQueryItems("per_page");
        q.addQueryItem("per_page", repo.contains("restricted") ? "64" : QString::number(per_page).toStdString().c_str());
        url.setQuery(q);
    }

    tags[repo] = array;
    if (!array.isEmpty() && repo == ui->github_repo->currentData().toString())
        QTimer::singleShot(0, this, [this]() {on_github_repo_currentIndexChanged(0);});

    tipl::out() << "loading " << url.toString().toStdString();
    auto reply = get(url);
    connect(reply.get(), &QNetworkReply::finished, this, [=]() mutable {
        if (reply->error() != QNetworkReply::NoError)
        {
            if (reply->error() != QNetworkReply::OperationCanceledError) {
                int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
                if (status!=401 && status!=404 && status!=403 && retryCount<5) {
                    int waitTime = 2 << retryCount;  // 2,4,8,16,32s
                    QTimer::singleShot(waitTime*1000, this, [=]() {
                        ++retryCount;
                        loadTags(url, repo, array,per_page);
                    });
                } else {
                    QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
                }
            }
        }
        else
        {
            update_rate_limit(reply);
            retryCount = 0;
            foreach (const QJsonValue& release , QJsonDocument::fromJson(QString(reply->readAll()).toUtf8()).array())
                array.append(release);

            // next‑page?
            auto m = QRegularExpression("<([^>]+)>; rel=\"next\"").match(reply->rawHeader("Link"));
            if (m.hasMatch())
            {
                QUrl nextPg = m.captured(1);
                if (nextPg.isValid())
                {
                    int delay_time = 0;
                    if(github_api_rate_limit < 40)
                        delay_time = 1000;
                    if(github_api_rate_limit < 20)
                        delay_time = 5000;
                    QTimer::singleShot(delay_time, this, [=]() {loadTags(nextPg, repo, array , per_page);});
                    return;
                }
            }
            tags[repo] = array;
            dates[repo] = QDate::currentDate().toString("yyyy/MM/dd");
            if (!array.isEmpty() && repo == ui->github_repo->currentData().toString())
                QTimer::singleShot(0, this, [this]() {on_github_repo_currentIndexChanged(0);});
        }
        ui->load_tags->setEnabled(true);
        reply->deleteLater();
    });
}


void MainWindow::loadFiles()
{
    bool is_restricted = ui->github_repo->currentText().contains("restricted");
    ui->github_release_files->setSortingEnabled(false);
    ui->github_release_files->setUpdatesEnabled(false);
    ui->github_release_files->setRowCount(0);


    for(int tab = ui->github_release_note->count()-1;tab > 0;--tab)
        ui->github_release_note->removeTab(tab);
    github_tsv_link.resize(1);

    QStringList units = {" b", " kb", " mb", " gb"};
    foreach (const QJsonValue& asset,assets[cur_tag])
    {
        QJsonObject assetObject = asset.toObject();
        size_t size = assetObject.value("size").toInteger();
        int i = 0;
        while (size >= 1024 && i < units.size() - 1)
        {
            size /= 1024;
            i++;
        }
        int row = ui->github_release_files->rowCount();
        auto file_name = assetObject.value("name").toString();
        ui->github_release_files->insertRow(row);
        ui->github_release_files->setItem(row, 0, new QTableWidgetItem(file_name));
        ui->github_release_files->setItem(row, 1, new QTableWidgetItem(QString::number(size)+units[i]));
        ui->github_release_files->setItem(row, 2, new QTableWidgetItem(assetObject.value("created_at").toString()));
        ui->github_release_files->setItem(row, 3, new QTableWidgetItem(QString::number(assetObject.value("download_count").toInteger())));
        if(is_restricted)
            ui->github_release_files->setItem(row, 4, new QTableWidgetItem(assetObject.value("url").toString()));
        else
            ui->github_release_files->setItem(row, 4, new QTableWidgetItem(assetObject.value("browser_download_url").toString()));
        ui->github_release_files->item(row,1)->setData(Qt::UserRole, assetObject.value("size").toInteger()); // Save the original size
        if(file_name.contains(".tsv"))
        {
            ui->github_release_note->addTab(new QWidget(ui->github_release_note),file_name.remove(".tsv"));
            github_tsv_link.push_back(assetObject.value("browser_download_url").toString());
        }
    }
    ui->github_release_files->sortByColumn(0,Qt::AscendingOrder);
    ui->github_release_files->setUpdatesEnabled(true);
    ui->github_release_files->resizeColumnToContents(0);
    ui->github_release_files->resizeColumnToContents(1);
    ui->github_release_files->resizeColumnToContents(2);
    ui->github_release_files->setColumnWidth(3,50);
    ui->github_release_files->setSortingEnabled(true);

    ui->file_count->setText(QString("%1 files").arg(ui->github_release_files->rowCount()));

}


void MainWindow::on_github_release_note_currentChanged(int index)
{
    if(index && index < github_tsv_link.size())
    {

        if(!github_tsv_link[index].isEmpty())
        {
            tipl::out() << "downloading " << github_tsv_link[index].toStdString().c_str();
            auto reply = get(github_tsv_link[index]);
            QEventLoop loop;
            QObject::connect(reply.get(), &QNetworkReply::finished, this, [&loop, this, reply, index]()
            {
                loop.quit();
                if (reply->error() == QNetworkReply::NoError &&
                    index < github_tsv_link.size() &&
                    !github_tsv_link[index].isEmpty())
                {
                    github_tsv_link[index].clear();
                    auto tableWidget = new QTableWidget(ui->github_release_note->widget(index));
                    auto layout = new QVBoxLayout(ui->github_release_note->widget(index));
                    layout->addWidget(tableWidget);

                    QString data = reply->readAll();
                    QStringList rows = data.split("\n");
                    while(rows.count() && rows.back().isEmpty())
                        rows.pop_back();
                    QStringList headers = rows.takeFirst().split("\t");
                    tableWidget->setRowCount(rows.size());
                    tableWidget->setColumnCount(headers.size());
                    tableWidget->setHorizontalHeaderLabels(headers);

                    for (int i = 0; i < rows.size(); ++i) {
                        QStringList cols = rows.at(i).split("\t");
                        for (int j = 0; j < cols.size(); ++j) {
                            QTableWidgetItem* item = new QTableWidgetItem;
                            bool ok;
                            double val = cols.at(j).toDouble(&ok);
                            if (ok)
                                item->setData(Qt::DisplayRole, val);
                            else
                                item->setText(cols.at(j));
                            tableWidget->setItem(i, j, item);
                        }
                    }
                    tableWidget->setSortingEnabled(true);
                }
            });
            loop.exec();

        }
    }
}


void MainWindow::on_github_tags_itemSelectionChanged()
{
    if(ui->github_tags->currentRow() >= 0 && ui->github_tags->rowCount())
    {
        cur_tag = ui->github_tags->item(ui->github_tags->currentRow(), 0)->text();
        QString title = ui->github_tags->item(ui->github_tags->currentRow(), 3)->text();
        ui->github_repo_title->setText(title);
        auto content = notes[cur_tag].split('\n');
        if(!content.empty() && content[0].contains(title))
            content.remove(0);
        ui->github_note->setMarkdown(content.join('\n'));
        ui->github_release_note->setCurrentIndex(0);
        loadFiles();
    }
    ui->github_tags->setColumnWidth(3,50);
}

void MainWindow::on_browseDownloadDir_clicked()
{
    QString filename =
        QFileDialog::getExistingDirectory(this,"Browse Download Directory",
                                          ui->workDir->currentText());
    if ( filename.isEmpty() )
        return;
    ui->download_dir->setText(filename);
}


void MainWindow::on_github_release_files_itemSelectionChanged()
{
    int selectedRows = ui->github_release_files->selectionModel()->selectedRows().size();
    ui->github_release_files->setColumnWidth(3,50);
    ui->github_download->setEnabled(selectedRows > 0);
    if(selectedRows == 1 && ui->github_release_files->currentRow() >= 0)
    {
        auto file_name = ui->github_release_files->item(ui->github_release_files->currentRow(),0)->text();
        ui->github_open_file->setText(QString("Open %1 at").arg(file_name));
        ui->github_open_file->setVisible(true);
        ui->github_open_file_mode->setVisible(true);
        ui->github_open_file_mode->clear();
        ui->github_open_file_mode->addItem("O1: View Image");
        ui->github_open_file_mode->addItem(file_name.endsWith(".src.gz") || file_name.endsWith(".sz") ? "T2: Reconstruction": "T3: Fiber Tracking");
        ui->github_open_file_mode->setCurrentIndex(file_name.endsWith(".nii.gz") || file_name.endsWith(".nii") ? 0 : 1 );
        if(file_name.endsWith(".db.fz") || file_name.endsWith(".db.fib.gz") || file_name.endsWith(".dz"))
        {
            ui->github_open_file_mode->addItems({"C2: View Database","C3: Correlational Tracking"});
            ui->github_open_file_mode->setCurrentIndex(2);
        }
    }
    else
    {
        ui->github_open_file->setVisible(false);
        ui->github_open_file_mode->setVisible(false);
    }
    ui->github_download->setText(selectedRows > 0 ? QString("Download %1 File(s)...").arg(selectedRows) : QString("Download"));
    ui->file_count->setText(QString("%1/%2 files").arg(selectedRows).arg(ui->github_release_files->rowCount()));
}


void MainWindow::on_github_select_all_clicked()
{
    ui->github_release_files->selectAll();
}


void MainWindow::on_github_download_clicked()
{
    QList<QTableWidgetSelectionRange> ranges = ui->github_release_files->selectedRanges();
    if (ranges.isEmpty()){
        QMessageBox::critical(this, "ERROR", "No files selected for download");
        return;
    }

    std::vector<int> row_list;
    for (int i = 0; i < ranges.size();++i)
        for (int row = ranges[i].topRow(); row <= ranges[i].bottomRow(); ++row)
            row_list.push_back(row);

    tipl::progress p("downloading...",true);
    for (int i = 0; p(i,row_list.size());++i)
    {
        qint64 startTime = QDateTime::currentMSecsSinceEpoch();

        QString url = ui->github_release_files->item(row_list[i], 4)->text();
        QString filePath = ui->download_dir->text() + "/" + ui->github_release_files->item(row_list[i], 0)->text();
        if (QFile::exists(filePath) && !ui->download_overwrite->isChecked())
        {
            tipl::out() << filePath.toStdString() << " exists...skipping";
            continue;
        }

        tipl::out() << url.toStdString();

        QSharedPointer<QNetworkReply> reply;
        int retry = 0;
        const int max_retry = 5;
        while (retry < max_retry)
        {
            reply = get(url);
            qint64 bytesTotal = ui->github_release_files->item(row_list[i], 1)->data(Qt::UserRole).toLongLong();
            while (!reply->isFinished() && !p.aborted())
            {
                QCoreApplication::processEvents();
                QThread::msleep(100); // Check every 100ms
            }
            if (reply->error() == QNetworkReply::NoError)
                break;
            retry++;
            QThread::sleep(3);
        }

        if (retry >= max_retry)
        {
            QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
            return;
        }
        if (p.aborted())
            return;

        {
            auto file = std::make_shared<QFile>(filePath);
            if (!file->open(QFile::WriteOnly))
            {
                QMessageBox::critical(this, "ERROR", "Failed to save file to disk");
                return;
            }
            QTimer::singleShot(0, this, [file, reply]()
            {
                file->write(reply->readAll());
            });
        }
    }
}


void MainWindow::on_github_select_matching_clicked()
{
    tipl::progress p("select matching");
    QString pattern = QInputDialog::getText(this, "Select Matching", "Enter a sub text (fib.gz), wild card (*.fib.gz) or regex pattern:");
    if (pattern.isEmpty())
        return;
    Qt::MatchFlag flags = Qt::MatchContains;
    if(pattern.contains("*"))
        flags = Qt::MatchWildcard;
    else
    if(pattern.contains(QRegularExpression("[.^$|()\\[\\]{}*+?\\\\]")))
    {
        QRegularExpression regex(pattern);
        if (regex.isValid())
            flags = Qt::MatchRegularExpression;
        else
        {
            QMessageBox::critical(this,"ERROR","Invalid regular expression pattern");
            return;
        }
    }
    QList<QTableWidgetItem*> items = ui->github_release_files->findItems(pattern, flags);
    ui->github_release_files->blockSignals(true);
    ui->github_release_files->clearSelection();
    for (int i = 0; p(i, items.size()); ++i)
        ui->github_release_files->setRangeSelected(QTableWidgetSelectionRange(items[i]->row(), 0, items[i]->row(), ui->github_release_files->columnCount() - 1), true);
    ui->github_release_files->blockSignals(false);
    on_github_release_files_itemSelectionChanged();
}





void MainWindow::on_github_open_file_clicked()
{
    auto row = ui->github_release_files->currentRow();
    if(row < 0)
        return;
    QDir dir(QDir::tempPath() + "/" + cur_tag);
    if (!dir.exists())
    {
        if(!dir.mkpath("."))
        {
            QMessageBox::critical(this,"ERROR","cannot create a temporary directory to store file");
            return;
        }
    }

    QString filePath = dir.path()+ "/" + ui->github_release_files->item(row, 0)->text();
    auto git_open = [this,filePath](void)
    {
        if(filePath.endsWith(".nii.gz") || filePath.endsWith(".nii") ||
           filePath.endsWith(".fib.gz") || filePath.endsWith(".fz") || filePath.endsWith(".dz"))
        {

            if(ui->github_open_file_mode->currentIndex() == 0)
                loadNii(QStringList() << filePath);
            else
            if(ui->github_open_file_mode->currentIndex() == 1)
                loadFib(filePath);
            else
            if(ui->github_open_file_mode->currentIndex() > 1) // open db
            {
                auto database = std::make_shared<group_connectometry_analysis>();
                tipl::progress prog("reading connectometry db");
                if(!database->load_database(filePath.toStdString().c_str()))
                {
                    QMessageBox::critical(this,"ERROR",database->error_msg.c_str());
                    return;
                }
                if(ui->github_open_file_mode->currentIndex() == 2)
                {
                    auto db = new db_window(this,database);
                    db->setWindowTitle(filePath);
                    db->setAttribute(Qt::WA_DeleteOnClose);
                    db->show();
                }
                else
                {
                    auto group_cnt = new group_connectometry(this,database,filePath);
                    group_cnt->setAttribute(Qt::WA_DeleteOnClose);
                    group_cnt->show();
                }
            }
        }
        else
        {
            if(ui->github_open_file_mode->currentIndex() == 0)
                loadNii(QStringList() << filePath);
            else
                openFile(QStringList() << filePath);
        }
    };


    qint64 bytesTotal = ui->github_release_files->item(row, 1)->data(Qt::UserRole).toLongLong();
    if (QFile::exists(filePath))
    {
        git_open();
        return;
    }
    tipl::out() << "download file to " << filePath.toStdString();
    auto reply = get(ui->github_release_files->item(row, 4)->text());

    // Create a progress dialog
    QProgressDialog progressDialog("Downloading...", "Cancel", 0, 100, this);
    progressDialog.setModal(true);
    progressDialog.show();
    qint64 bytesReceived = 0;
    QEventLoop loop;

    QObject::connect(reply.get(), &QNetworkReply::readyRead, this,
                     [this, &progressDialog, &bytesReceived, bytesTotal,reply]()
    {
        progressDialog.setValue((reply->bytesAvailable() * 100) / (bytesTotal));
    });
    QObject::connect(reply.get(), &QNetworkReply::finished, this,
                     [this, filePath, git_open, &progressDialog, &loop,reply]() // Pass the loop to the lambda
    {
        if (reply->error() != QNetworkReply::NoError)
        {
            if(reply->error() != QNetworkReply::OperationCanceledError)
                QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
        }
        else
        {
            auto downloadFile = std::make_shared<QFile>(filePath);
            if (!downloadFile->open(QFile::WriteOnly))
            {
                QMessageBox::critical(this, "ERROR", "Failed to open file for writing");
                return;
            }
            downloadFile->write(reply->readAll());
            downloadFile->close();
            QTimer::singleShot(0, this, [git_open](){git_open();});
        }
        progressDialog.close();
        loop.quit();
    });

    QObject::connect(&progressDialog, &QProgressDialog::canceled, this, [this,&loop,reply]() // Pass the loop to the lambda
    {
        if (reply && reply->isRunning())
            reply->abort();
    });

    loop.exec();
}

