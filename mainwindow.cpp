#include <QFileDialog>
#include <QDateTime>
#include <QDir>
#include <QInputDialog>
#include <QMenu>
#include <QUrl>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QAction>
#include <QTextStream>
#include <QHeaderView>
#include <QStyleFactory>
#include <QNetworkInterface>
#include <QSysInfo>
#include <QStandardPaths>
#include <QShortcut>
#include <QStandardItemModel>
#include <QDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QUuid>
#include <QEvent>

#include <QJsonDocument>
#include <QJsonArray>
#include <QMap>
#include <QJsonObject>

#include <filesystem>
#include "mainwindow.h"
#include "ui_mainwindow.h"
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
    auto* agents = qobject_cast<QStandardItemModel*>(
                       ui->ai_agent_selector->model());
    auto set_agent = [&](ai_provider provider,const QString& path,
                          const QStringList& models,QJsonObject profiles = {})
    {
        auto index = int(provider);
        auto agent = ui->ai_agent_selector->itemText(index);
        auto* item = agents->item(index);
        item->setText(agent+(path.isEmpty() ? " (not found)" : ""));
        item->setEnabled(!path.isEmpty());
        ui->ai_agent_selector->setItemData(index,path,Qt::UserRole+1);
        ui->ai_agent_selector->setItemData(index,models);
        ui->ai_agent_selector->setItemData(index,QVariant::fromValue(profiles),Qt::UserRole+2);
        ai_log(path.isEmpty() ? agent+" not found" : agent+": "+path);
        if(!path.isEmpty())
            ai_log(agent+" models: "+(models.isEmpty() ? "none detected" : models.join(", ")));
    };
    QString codex_path,claude_path;
    {
        // find Codex executable and models
        codex_path = QStandardPaths::findExecutable("codex");
        if(codex_path.isEmpty())
        {
            QDir dir(QStandardPaths::writableLocation(
                         QStandardPaths::GenericDataLocation)+"/OpenAI/Codex/bin");
            for(const auto& name : dir.entryList(
                    QDir::Dirs|QDir::NoDotAndDotDot,QDir::Time))
                if(QFileInfo::exists(codex_path = dir.filePath(name+"/codex.exe")))
                    break;
        }
        if(!QFileInfo::exists(codex_path))
            codex_path.clear();
        set_agent(ai_provider::Codex,codex_path,{});
        refresh_codex_models(codex_path);
    }
    {
        // find Claude executable and models
        claude_path = QStandardPaths::findExecutable("claude");
#ifdef Q_OS_WIN
        if(claude_path.isEmpty())
            claude_path = QDir::homePath()+"/.local/bin/claude.exe";
#endif
        if(!QFileInfo::exists(claude_path))
            claude_path.clear();
        set_agent(ai_provider::Claude,claude_path,{});
        refresh_ollama_models();
    }

    if(codex_path.isEmpty() && !claude_path.isEmpty())
        ui->ai_agent_selector->setCurrentIndex(int(ai_provider::Claude));

    ui->ai_agent_selector->setEnabled(
        !codex_path.isEmpty() || !claude_path.isEmpty());
    auto update_models = [this]
    {
        ui->ai_model_selector->clear();
        ui->ai_model_selector->addItem("default");
        auto profiles = ui->ai_agent_selector->currentData(Qt::UserRole+2).
                        toJsonObject();
        for(const auto& model : ui->ai_agent_selector->currentData().toStringList())
            ui->ai_model_selector->addItem(model,
                QVariant::fromValue(profiles[model].toObject()));
    };
    connect(ui->ai_agent_selector,QOverload<int>::of(&QComboBox::currentIndexChanged),this,
            [update_models] { update_models(); });
    update_models();

    {
        auto update_agent_name = [this]
        {
            auto index = ui->ai_agent_selector->currentIndex();
            QString name = index == int(ai_provider::Codex) ? "Codex" : "Claude";
            if(ui->ai_model_selector->currentData().toJsonObject()["provider"].toInt() ==
                int(ai_model_provider::Ollama))
            {
                auto host = settings.value("ai/ollama_host","localhost").
                            toString().trimmed();
                if(!host.contains("://"))
                    host.prepend("http://");
                name += "/Ollama(" + QUrl(host).host() + ")";
            }
            ui->ai_agent_selector->setItemText(index,name);
        };
        connect(ui->ai_model_selector,&QComboBox::currentTextChanged,
                this,[update_agent_name]{update_agent_name();});
        update_agent_name();
    }


    auto default_agent = settings.value(
        "ai/default_agent",ui->ai_agent_selector->currentIndex());
    bool okay;
    auto default_index = default_agent.toInt(&okay);
    if(!okay)
        default_index = int(ai_info::identify_provider(default_agent.toString()));
    if(default_index >= 0 &&
       default_index < ui->ai_agent_selector->count())
        ui->ai_agent_selector->setCurrentIndex(default_index);



    ui->ai_model_selector->setCurrentText(
        settings.value("ai/default_model",ui->ai_model_selector->currentText()).toString());


    connect(ui->tabWidget,&QTabWidget::currentChanged,this,[this]
    {
        if(ui->tabWidget->currentWidget() == ui->tab_8)
            stop_ai_blink();
    });
    auto* send = new QShortcut(QKeySequence(Qt::CTRL|Qt::Key_Return),
                               ui->ai_chat_input);
    send->setContext(Qt::WidgetShortcut);
    connect(send,&QShortcut::activated,
            ui->ai_send_message,&QPushButton::click);

    ai_project_dir = QStandardPaths::writableLocation(
                         QStandardPaths::AppLocalDataLocation)+"/ai_projects";
    QDir dir(ai_project_dir);
    dir.mkpath(".");

    ai_project_menu = new QMenu(this);
    ai_project_menu->setStyleSheet(
        "QMenu{background:#fff;border:1px solid #d9d9dc;padding:4px;}"
        "QMenu::item{padding:6px 24px 6px 10px;border-radius:4px;}"
        "QMenu::item:selected{background:#e9e9eb;}"
        "QMenu::item:disabled{color:#9a9a9e;}"
        "QMenu::separator{height:1px;background:#dedee1;margin:4px;}");
    connect(ai_project_menu->addAction("Rename"),&QAction::triggered,this,[this]
    {
        auto* item = ui->ai_project_list->currentItem();
        if(!item)
            return;
        auto session = item->data(Qt::UserRole).toString();
        bool okay;
        auto title = QInputDialog::getText(
            this,"Rename Chat","Chat name:",QLineEdit::Normal,
            ai_infos[session].title(session),&okay);
        if(okay && !set_ai_title(session,title))
            QMessageBox::warning(
                this,"Rename Chat","The chat name could not be saved.");
    });

    connect(ai_project_menu->addAction("Details..."),&QAction::triggered,this,[this]
    {
        auto* item = ui->ai_project_list->currentItem();
        if(!item)
            return;
        auto session = item->data(Qt::UserRole).toString();
        QMessageBox::information(
            this,"Chat Details",ai_infos[session].details(session));
    });
    ai_project_menu->addSeparator();

    connect(ai_project_menu->addAction("Remove"),&QAction::triggered,this,[this]
    {
        auto* item = ui->ai_project_list->currentItem();
        if(!item)
            return;
        auto session = item->data(Qt::UserRole).toString();
        if(ai_infos[session].processes)
        {
            QMessageBox::information(this,"Remove Project","Wait for the AI agent to finish first.");
            return;
        }
        if(QMessageBox::question(this,"Remove Project","Remove this project and its saved history?") != QMessageBox::Yes)
            return;

        QFile::remove(ai_project_dir+"/"+QString::fromLatin1(
                          QUrl::toPercentEncoding(session))+".jsonl");
        auto agent_name = ai_infos[session].agent_name;
        if(!agent_name.isEmpty())
            QFile::remove(ai_project_dir+"/"+QString::fromLatin1(
                QUrl::toPercentEncoding(agent_name+"@"+session))+".jsonl");
        ai_infos.erase(session);
        ai_log_positions.remove(session);
        ui->ai_project_list->setCurrentItem(nullptr);
        delete item;

        if(ui->ai_project_list->count())
            ui->ai_project_list->setCurrentRow(0);
        else
            ui->ai_chat_history->clear();
    });

    connect(ui->ai_project_list,&QListWidget::currentItemChanged,this,
            [this](QListWidgetItem* item,QListWidgetItem* previous)
    {
        for(auto* i : {previous,item})
            if(i)
                ui->ai_project_list->itemWidget(i)->
                    findChild<QPushButton*>("ai_project_title")->
                    setStyleSheet(i == item ? "background:#dce9f9;" : "");
        if(item)
        {
            stop_ai_blink();
            auto session = item->data(Qt::UserRole).toString();
            select_agent_model(ai_infos[session]);
            show_ai_project(session);
        }
        else
            ui->ai_chat_history->clear();
    });

    for(const auto& info : dir.entryInfoList(
            {"*.jsonl"},QDir::Files,QDir::Time|QDir::Reversed))
    {
        auto legacy_key = QUrl::fromPercentEncoding(
                              info.completeBaseName().toLatin1());
        auto session = legacy_key.section('@',-1);
        auto agent_name = legacy_key.contains('@') ?
                          legacy_key.section('@',0,0) : QString();
        QFile file(info.filePath());
        if(!file.open(QIODevice::ReadOnly))
            continue;

        QJsonArray loaded_history;
        QString project_title,cwd;
        while(!file.atEnd())
        {
            auto doc = QJsonDocument::fromJson(file.readLine());
            if(doc.isObject())
            {
                auto entry = doc.object();
                if(entry["type"] == "title")
                {
                    project_title = entry["text"].toString();
                    continue;
                }
                loaded_history.append(entry);
                if(entry["type"] == "request")
                {
                    auto request = QJsonDocument::fromJson(
                                       entry["text"].toString().toUtf8()).object();
                    if(!request["session"].toString().isEmpty())
                        session = request["session"].toString();
                    if(!request["agent"].toString().isEmpty())
                        agent_name = request["agent"].toString();
                    if(QDir(request["cwd"].toString()).exists())
                        cwd = request["cwd"].toString();
                }
            }
        }

        if(loaded_history.isEmpty() || session.isEmpty())
            continue;
        auto& ai = ai_infos[session];
        ai.update(agent_name,cwd);
        if(!project_title.isEmpty())
            ai.project_titles = project_title;
        for(const auto& entry : loaded_history)
            ai.projects.append(entry);
        show_ai_project(session);
    }

    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);

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
                checkForVersionSpecificBugs_Minimal(news);
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
                    {command({command_name});});
        }
    }
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
bool MainWindow::loadFib(QString filename)
{
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(tipl::qt::to_path(filename)))
    {
        if(!new_handle->error_msg.empty())
            QMessageBox::critical(this,"ERROR",new_handle->error_msg.c_str());
        return false;
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
    if(filenames.isEmpty() || filenames[0].isEmpty())
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
        rename_dicom(tipl::qt::to_path(filenames[index]),tipl::qt::to_path(filenames[index]).parent_path());
}


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
void MainWindow::on_RenameDICOMDir_clicked()
{
    QString path =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( path.isEmpty() )
        return;
    add_work_dir(path);
    rename_dicom_at_dir(tipl::qt::to_path(path),tipl::qt::to_path(path));
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

bool parse_dwi(const std::vector<std::filesystem::path>& file_list,
               std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg);
std::filesystem::path get_dicom_output_name(const std::filesystem::path& file_name,
                                            const std::string& file_extension, bool add_path);
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



void MainWindow::on_workDir_currentTextChanged(const QString &arg1)
{
    if(!arg1.isEmpty())
        QDir::setCurrent(arg1);
}

bool MainWindow::load_db(std::shared_ptr<group_connectometry_analysis>& database,QString& filename)
{
    filename = tipl::qt::open_image_file(this,ui->workDir->currentText(),"Database (*.dz *db.fz *db?fib.gz);;All files (*)");
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

void MainWindow::on_SRC_qc_clicked()
{
    QStringList filenames = tipl::qt::open_image_files(this,ui->workDir->currentText(),"Src files (*.sz *src.gz);;All files (*)" );
    if (filenames.isEmpty())
        return;
    std::vector<std::filesystem::path> files;
    for(const auto& each : filenames)
        files.push_back(tipl::qt::to_path(each));
    tipl::progress prog("checking SRC files");
    show_info_dialog("SRC report",quality_check_src_files(files,false,false,0));
}


void MainWindow::on_NII_qc_clicked()
{
    auto filenames = tipl::qt::open_image_files(this,ui->workDir->currentText(),"NIFTI files (*.nii *nii.gz);;All files (*)");
    if (filenames.isEmpty())
        return;
    std::vector<std::filesystem::path> files;
    for(const auto& each : filenames)
        files.push_back(tipl::qt::to_path(each));
    tipl::progress prog("checking NIFTI files");
    show_info_dialog("NIFTI report",quality_check_nii_files(files));
}



void MainWindow::on_FIB_qc_clicked()
{
    auto filenames = tipl::qt::open_image_files(this,ui->workDir->currentText(),"Fib files (*.fz *fib.gz);;All files (*)");

    if (filenames.isEmpty())
        return;
    std::vector<std::filesystem::path> files;
    for(const auto& each : filenames)
        files.push_back(tipl::qt::to_path(each));
    tipl::progress prog("checking FIB files");
    show_info_dialog("FIB report",quality_check_fib_files(files));
}

void MainWindow::on_parse_network_measures_clicked()
{
    auto files = QFileDialog::getOpenFileNames(
        this,"Open Network Measures",ui->workDir->currentText(),
        "Text files (*.txt);;All files (*)");
    if(files.isEmpty())
        return;

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
    {
        QMessageBox::critical(this,"ERROR","cannot write "+output);
        return;
    }

    QTextStream out(&file);
    out << "Field";
    for(const auto& input : files)
        out << '\t' << QFileInfo(input).baseName();
    out << '\n';

    for(const auto& field : fields)
        out << field << '\t' << values[field].join('\t') << '\n';

    QMessageBox::information(
        this,QApplication::applicationName(),"File saved to "+output);
}

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
    auto dwi_nii_files = search_dwi_nii_bids(tipl::qt::to_path(dir));
    if(dwi_nii_files.empty())
    {
        QMessageBox::critical(this,"ERROR","cannot find bids nifti data");
        return;
    }
    std::sort(dwi_nii_files.begin(),dwi_nii_files.end());
    nii2src(dwi_nii_files,tipl::qt::to_path(output_dir),true,true,false);
}
void search_dwi_nii(const std::filesystem::path& dir,std::vector<std::filesystem::path>& dwi_nii_files);
void MainWindow::on_nii2src_sf_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
        this,"Open directory",ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);

    std::vector<std::filesystem::path> files;
    search_dwi_nii(tipl::qt::to_path(dir),files);
    if(files.empty())
    {
        QMessageBox::critical(this,"ERROR","cannot find nifti data");
        return;
    }

    std::vector<std::pair<std::filesystem::path,std::filesystem::path>> jobs;
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
                QString("%1 exists, overwrite?").
                arg(QString::fromUtf8(src.filename().u8string().c_str())),
                QMessageBox::Yes|QMessageBox::YesToAll|
                    QMessageBox::No|QMessageBox::NoToAll|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::YesToAll)
                yes_to_all = true;
            if(result == QMessageBox::NoToAll)
                no_to_all = true;
            if(result == QMessageBox::No || result == QMessageBox::NoToAll)
                continue;
        }
        jobs.emplace_back(nii,src);
    }

    tipl::progress prog("batch creating src");
    std::atomic_size_t done = 0;
    tipl::par_for(jobs.size(),[&](size_t index)
    {
        if(!prog(done.fetch_add(1),jobs.size()))
            return;
        src_data src;
        if(!src.load_from_file({jobs[index].first},true) ||
            !src.save_to_file(jobs[index].second))
            tipl::warning() << src.error_msg;
    });
}

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

void dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite)
{
    tipl::progress prog("convert DICOM to SRC or nifti files");
    std::vector<std::filesystem::path> pending{dir};
    for(size_t p = 0,done = 0,total = 0;p < pending.size();++p)
    {
        auto dir_list = tipl::search_dirs(pending[p],std::string());
        total += dir_list.size();
        bool has_dicom = false;
        for(size_t i = 0;i < dir_list.size();++i,++done)
        {
            if(!prog(done,total))
                return;
            auto dicom_file_list = tipl::search_files(dir_list[i],"*.dcm");
            if(dicom_file_list.empty())
                continue;
            has_dicom = true;
            while(i+1 < dir_list.size() && std::filesystem::exists(dir_list[i+1]/dicom_file_list.front().filename()))
                tipl::search_files(dir_list[++i],"*.dcm",dicom_file_list),++done;
            dicom2src_and_nii(dicom_file_list,overwrite);
        }
        if(!has_dicom)
            pending.insert(pending.end(),dir_list.begin(),dir_list.end());
    }
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
    dicom2src_and_nii(tipl::qt::to_path(dir),false);
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
    settings.clear();
    settings.sync();
    QMessageBox::information(this,QApplication::applicationName(),"Setting Cleared");
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

void MainWindow::open_template(QString name)
{
    for(auto& each : fib_template_list)
        if(std::filesystem::path(each).stem().u8string() == name.toStdString())
        {
            if(loadFib(each.u8string().c_str()))
                tracking_windows.back()->work_path.clear();
            return;
        }
}


void MainWindow::on_OpenDWI_NIFTI_clicked()
{
    open_DWI(QStringList() << tipl::qt::open_image_file(this,ui->workDir->currentText(),"NIFTI files (*.nii *.nii.gz);;All files (*)" ));
}


void MainWindow::on_OpenDWI_DICOM_clicked()
{
    open_DWI(tipl::qt::open_image_files(this,ui->workDir->currentText(),"DICOM files (*.dcm);;All files (*)" ));
}


void MainWindow::on_OpenDWI_2dseq_clicked()
{
    open_DWI(tipl::qt::open_image_files(this,ui->workDir->currentText(),"2dseq files (2dseq);;FDF files (*.fdf);;NRRD Files (*.nrrd);;All files (*)" ));
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

int run_action_with_wildcard(tipl::program_option<tipl::out>&);
bool MainWindow::command(const std::vector<std::string>& cmd)
{
    error_msg.clear();
    auto fail = [&](const std::string& msg){error_msg = msg;return false;};
    if(cmd.empty())
        return fail("empty command");

    if(cmd[0] == "list_recent_fib")
    {
        if(cmd.size() != 1)
            return fail("list_recent_fib takes no arguments");
        for(const auto& file : settings.value("recentFibFileList").toStringList())
            tipl::out() << QDir::fromNativeSeparators(file).toStdString();
        return true;
    }

    if(cmd[0] == "list_recent_src")
    {
        if(cmd.size() != 1)
            return fail("list_recent_src takes no arguments");
        for(const auto& file : settings.value("recentSrcFileList").toStringList())
            tipl::out() << QDir::fromNativeSeparators(file).toStdString();
        return true;
    }

    if(cmd[0] == "set_work_dir")
    {
        if(cmd.size() != 1)
            return fail("set_work_dir takes no arguments");
        auto dir = QFileDialog::getExistingDirectory(
                       this,"Browse Directory",ui->workDir->currentText());
        if(!dir.isEmpty())
            add_work_dir(dir);
        return true;
    }

    if(cmd[0] == "open_src")
    {
        if(cmd.size() != 1)
            return fail("open_src takes no arguments");
        auto files = tipl::qt::open_image_files(
                         this,ui->workDir->currentText(),
                         "Src files (*.sz *src.gz);;Histology images (*.jpg *.tif);;All files (*)");
        if(files.isEmpty())
            return true;
        add_work_dir(QFileInfo(files[0]).absolutePath());
        loadSrc(files);
        return true;
    }

    if(cmd[0] == "open_fib" || cmd[0] == "open_structural_tracking")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        auto filter = cmd[0] == "open_fib" ?
                      "Fib files (*.fz *fib.gz *.dz);;All files (*)" :
                      "Image files (*nii.gz *.nii 2dseq);;All files (*)";
        auto file = tipl::qt::open_image_file(
                        this,ui->workDir->currentText(),filter);
        if(file.isEmpty())
            return true;
        add_work_dir(QFileInfo(file).absolutePath());
        return loadFib(file);
    }

    if(cmd[0] == "open_template")
    {
        if(cmd.size() != 1)
            return fail("open_template takes no arguments");
        auto* item = ui->template_list->currentItem();
        if(!item)
            return fail("no template selected");
        open_template(item->text());
        return true;
    }

    if(cmd[0] == "open_db" || cmd[0] == "open_connectometry")
    {
        if(cmd.size() != 1)
            return fail(cmd[0]+" takes no arguments");
        QString file;
        std::shared_ptr<group_connectometry_analysis> database;
        if(!load_db(database,file))
            return true;
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

    if(cmd[0] == "open_auto_track")
    {
        if(cmd.size() != 1)
            return fail("open_auto_track takes no arguments");
        auto* window = new auto_track(this);
        window->setAttribute(Qt::WA_DeleteOnClose);
        window->showNormal();
        return true;
    }

    if(cmd[0] == "open_nonlinear_registration")
    {
        if(cmd.size() != 1)
            return fail("open_nonlinear_registration takes no arguments");
        auto* window = new RegToolBox(this);
        window->setAttribute(Qt::WA_DeleteOnClose);
        window->showNormal();
        return true;
    }

    if(cmd[0] == "open_xnat")
    {
        if(cmd.size() != 1)
            return fail("open_xnat takes no arguments");
        auto* window = new xnat_dialog(this);
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
        if(cmd.size() == 1)
        {
            auto files = tipl::qt::open_image_files(
                             this,ui->workDir->currentText(),
                             "image files (*.nii *nii.gz *.dcm *.nhdr *.nrrd 2dseq);;All files (*)");
            if(files.isEmpty())
                return true;
            add_work_dir(QFileInfo(files[0]).absolutePath());
            auto* window = new view_image(this);
            window->setAttribute(Qt::WA_DeleteOnClose);
            if(!window->open(files))
                return QMessageBox::critical(
                           this,"ERROR",window->error_msg.c_str()),
                       delete window,false;
            window->show();
            return true;
        }
        QStringList files;
        for(size_t i = 1;i < cmd.size();++i)
            files << QString::fromUtf8(cmd[i]);
        loadNii(files);
        return true;
    }

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

    return fail("unknown command: "+cmd[0]);
}
