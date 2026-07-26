#include <QFileDialog>
#include <QDateTime>
#include <QDir>
#include <QMenu>
#include <QUrl>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QAction>
#include <QHeaderView>
#include <QStyleFactory>
#include <QNetworkInterface>
#include <QSysInfo>
#include <QStandardPaths>
#include <QShortcut>
#include <QProcess>
#include <QUuid>
#include <QTimer>
#include <QScrollBar>

#include <QJsonDocument>
#include <QJsonArray>
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

static QString find_codex_executable()
{
    auto executable = QStandardPaths::findExecutable("codex");
    if(!executable.isEmpty())
        return executable;

    QDir dir(QStandardPaths::writableLocation(
                 QStandardPaths::GenericDataLocation)+"/OpenAI/Codex/bin");
    for(const auto& name : dir.entryList(
            QDir::Dirs|QDir::NoDotAndDotDot,QDir::Time))
        if(QFileInfo::exists(executable = dir.filePath(name+"/codex.exe")))
            return executable;
    return {};
}

static void ai_reply(QLocalSocket* socket,const QString& agent,
                     QByteArray reply,QJsonArray* results = nullptr)
{
    auto& prompts = main_window->ai_prompts[agent];
    if(results)
    {
        if(!prompts.isEmpty())
        {
            auto result = results->last().toObject();
            result["prompt"] = prompts;
            results->replace(results->size()-1,result);
        }
        reply = QJsonDocument(*results).toJson(QJsonDocument::Compact);
    }
    else if(!prompts.isEmpty())
    {
        auto payload = "PROMPT\t" +
                       QJsonDocument(prompts).toJson(QJsonDocument::Compact) + '\n';
        int pos = reply.indexOf('\n');
        if(pos < 0)
            reply.append('\n').append(payload);
        else
            reply.insert(pos+1,payload);
    }
    if(socket->write(reply) == reply.size())
        prompts = {};
}

static void ai_request_list(QLocalSocket* socket,const QString& agent)
{
    static quint64 next_id = 0;
    QStringList result;
    for(auto* window : QApplication::allWidgets())
    {
        QString type;
        if(qobject_cast<MainWindow*>(window))
            type = "main";
        else if(qobject_cast<tracking_window*>(window))
            type = "tracking";
        else if(qobject_cast<view_image*>(window))
            type = "image";
        else
            continue;

        if(!window->property("remote_id").isValid())
            window->setProperty("remote_id",++next_id);

        result << QString("%1\t%2\t%3")
                      .arg(type)
                      .arg(window->property("remote_id").toULongLong())
                      .arg(window->windowTitle());
    }
    ai_reply(socket,agent,"OKAY\n" + result.join('\n').toUtf8());
}

static void ai_request_command(QLocalSocket* socket,const QString& agent,
                               const QJsonObject& request)
{
    auto id = request["window"].toVariant().toString();
    auto commands = request["command"].toArray();
    if(id.isEmpty() || commands.isEmpty())
        return ai_reply(socket,agent,"ERROR\tinvalid command");
    if(commands[0].isString())
    {
        QJsonArray batch;
        batch.append(commands);
        commands = batch;
    }

    QWidget* target = nullptr;
    for(auto* window : QApplication::allWidgets())
        if(window->property("remote_id").toString() == id)
            target = window;
    if(!target)
        return ai_reply(socket,agent,"ERROR\twindow not found");

    auto run = [&](const std::vector<std::string>& cmd,QString& output,QString& error)
    {
        if(cmd.empty() || cmd[0].empty())
            return error = "empty command",false;

        bool okay = false;
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            console.capture = &output;
        }
        try
        {
            if(auto* window = qobject_cast<MainWindow*>(target))
                okay = window->command(cmd),
                    error = QString::fromStdString(window->error_msg);
            else if(auto* window = qobject_cast<tracking_window*>(target))
                okay = window->command(cmd),
                    error = QString::fromStdString(window->error_msg);
            else if(auto* window = qobject_cast<view_image*>(target))
                if(cmd.size() > 2)
                    error = "too many parameters";
                else
                    okay = window->command(cmd[0],cmd.size() == 2 ? cmd[1] : ""),
                        error = QString::fromStdString(window->error_msg);
            else
                error = "unsupported window";
        }
        catch(const std::exception& e)
        {
            error = e.what();
        }
        catch(...)
        {
            error = "unknown error";
        }
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            console.capture = nullptr;
        }
        return okay;
    };

    bool updates_enabled = target->updatesEnabled();
    target->setUpdatesEnabled(false);
    QJsonArray results;

    for(int index = 0;index < commands.size();++index)
    {
        QJsonObject result{{"index",index}};
        QString output,error;
        std::vector<std::string> cmd;
        auto args = commands[index].toArray();

        if(!commands[index].isArray() ||
            !std::all_of(args.begin(),args.end(),
                         [](const auto& value){return value.isString();}))
            error = "command and parameters must be strings in an array";
        else
            for(const auto& value : args)
                cmd.push_back(value.toString().toUtf8().toStdString());

        bool okay = error.isEmpty() && run(cmd,output,error);
        if(!okay && error.isEmpty())
            error = "command failed";

        result["okay"] = okay;
        result["output"] = output;
        if(!okay)
            result["error"] = error;
        results.append(result);

        if(!okay)
            break;
    }

    target->setUpdatesEnabled(updates_enabled);
    if(auto* window = qobject_cast<tracking_window*>(target))
    {
        window->slice_need_update = true;
        window->glWidget->update_slice();
    }
    else
        target->update();

    ai_reply(socket,agent,QByteArray(),&results);
}

static void ai_request_log(QLocalSocket* socket,const QString& agent)
{
    QByteArray output;
    {
        std::lock_guard<std::mutex> lock(console.edit_buf);
        output = console.history.toUtf8();
    }
    ai_reply(socket,agent,"OKAY\n" + output);
}
void ai_request(QLocalSocket* socket,const QByteArray& data)
{
    QJsonParseError error;
    auto doc = QJsonDocument::fromJson(data,&error);
    if(!doc.isObject())
        return ai_reply(socket,{},("ERROR\tinvalid JSON: " +
                                     error.errorString()).toUtf8());

    auto request = doc.object();
    auto agent = request["agent"].toString();
    auto type = request["request"].toString().toUpper();
    if(!agent.startsWith('@'))
        return ai_reply(socket,agent,"ERROR\tinvalid agent");
    auto session = request["session"].toString().trimmed();
    if(agent.startsWith("@C") && QUuid(session).isNull())
        return ai_reply(socket,agent,"ERROR\tinvalid Codex session");
    if(!session.isEmpty())
        main_window->ai_sessions[agent] = session;
    auto cwd = request["cwd"].toString();
    if(QDir(cwd).exists())
        main_window->ai_work_dirs[agent] = cwd;

    auto activity = request;
    activity.remove("chat");
    main_window->add_ai_history(
        agent,"request",
        QJsonDocument(activity).toJson(QJsonDocument::Compact));

    auto chat = request["chat"].toString().trimmed();
    if(!chat.isEmpty())
        main_window->add_ai_history(agent,"assistant",chat);

    if(type == "LIST")
        return ai_request_list(socket,agent);
    if(type == "LOG")
        return ai_request_log(socket,agent);
    if(type == "CMD")
        return ai_request_command(socket,agent,request);
    ai_reply(socket,agent,"ERROR\tunknown request");
}

void MainWindow::show_ai_project(const QString& agent,QJsonObject added)
{
    const auto& history = ai_projects[agent];
    if(history.isEmpty())
        return;

    QString name = agent.startsWith("@C") ? "Codex" :
                   agent.startsWith("@A") ? "Claude Code" : "AI Agent";
    QString project_title = name+" · "+agent;

    auto* item = ai_project_items.value(agent);
    if(!item)
    {
        item = new QListWidgetItem;
        item->setData(Qt::UserRole,agent);
        ui->ai_project_list->insertItem(0,item);
        ai_project_items[agent] = item;

        auto* row = new QWidget;
        auto* title = new QPushButton(row);
        title->setObjectName("ai_project_title");
        title->setFlat(true);

        auto* button = new QToolButton(row);
        button->setObjectName("ai_project_menu_button");
        button->setText("...");
        button->setToolTip("Project actions");
        button->setFixedSize(28,28);
        button->setPopupMode(QToolButton::InstantPopup);
        button->setMenu(ai_project_menu);

        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(6,2,2,2);
        layout->setSpacing(2);
        layout->addWidget(title,1);
        layout->addWidget(button);
        ui->ai_project_list->setItemWidget(item,row);

        connect(title,&QPushButton::clicked,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
        connect(button,&QToolButton::pressed,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
    }

    item->setText(project_title);
    auto* row = ui->ai_project_list->itemWidget(item);
    auto* title = row->findChild<QPushButton*>("ai_project_title");
    title->setText(project_title);
    title->setToolTip(project_title);
    item->setSizeHint(row->sizeHint());
    item->setHidden(!project_title.contains(
        ui->ai_project_filter->text(),Qt::CaseInsensitive));

    if(!ui->ai_project_list->currentItem())
    {
        ui->ai_project_list->setCurrentItem(item);
        return;
    }
    if(ui->ai_project_list->currentItem() != item)
        return;

    auto append = [&](const QJsonObject& entry)
    {
        auto type = entry["type"].toString();
        bool user = type == "user",request = type == "request";
        auto content = entry["text"].toString().
                       toHtmlEscaped().replace('\n',"<br>");
        auto cell = QString(
            "<td bgcolor=\"%1\"><b>%2 · %3</b> "
            "<font color=\"#80868b\">%4</font><br>%5</td>")
            .arg(request ? "#f1f3f4" : user ? "#e8f0fe" : "#e8f5e9",
                 request ? "Activity" : user ? "You" : name,
                 agent.toHtmlEscaped(),
                 QDateTime::fromString(
                     entry["time"].toString(),Qt::ISODate).
                     toString("MM/dd HH:mm:ss"),
                 request ? "<code>"+content+"</code>" : content);

        ui->ai_chat_history->append(
            QString("<table width=\"100%\" cellspacing=\"3\" "
                    "cellpadding=\"7\"><tr>%1</tr></table>")
            .arg(request ? cell :
                 user ? "<td width=\"20%\"></td>"+cell :
                        cell+"<td width=\"20%\"></td>"));
    };

    if(added.isEmpty())
    {
        ui->ai_chat_history->clear();
        for(const auto& entry : history)
            append(entry.toObject());
    }
    else
        append(added);

    ui->ai_chat_history->ensureCursorVisible();
    QTimer::singleShot(0,ui->ai_chat_history,[this]
    {
        auto* bar = ui->ai_chat_history->verticalScrollBar();
        bar->setValue(bar->maximum());
    });
    ui->ai_connected_agent->setText("Agent: "+name+" "+agent);
    ui->ai_control_status->setText("● Active");
}

void MainWindow::add_ai_history(const QString& agent,const QString& type,
                                const QString& text)
{
    QJsonObject entry{
        {"type",type},{"text",text},
        {"time",QDateTime::currentDateTime().toString(Qt::ISODate)}
    };
    ai_projects[agent].append(entry);

    QFile file(ai_project_dir+"/"+QString::fromLatin1(
                   QUrl::toPercentEncoding(agent))+".jsonl");
    if(file.open(QIODevice::WriteOnly|QIODevice::Append))
        file.write(QJsonDocument(entry).toJson(QJsonDocument::Compact)+'\n');

    show_ai_project(agent,entry);
}

void MainWindow::on_ai_send_message_clicked()
{
    auto* item = ui->ai_project_list->currentItem();
    auto text = ui->ai_chat_input->toPlainText().trimmed();
    if(!item || text.isEmpty())
        return;

    auto agent = item->data(Qt::UserRole).toString();
    add_ai_history(agent,"user",text);
    ui->ai_chat_input->clear();
    auto session = ai_sessions.value(agent);
    auto codex = find_codex_executable();
    if(!agent.startsWith("@C") || session.isEmpty() || codex.isEmpty() ||
       ai_processes.contains(agent))
    {
        ai_prompts[agent].append(text);
        ui->ai_control_status->setText("● Queued");
        return;
    }

    auto* process = new QProcess(this);
    ai_processes[agent] = process;
    if(QDir(ai_work_dirs.value(agent)).exists())
        process->setWorkingDirectory(ai_work_dirs[agent]);

    connect(process,&QProcess::readyReadStandardOutput,this,
            [=]{process->readAllStandardOutput();});
    connect(process,&QProcess::readyReadStandardError,this,
            [=]{process->readAllStandardError();});
    connect(process,&QProcess::started,this,
            [=]{ui->ai_control_status->setText("● Codex running");});
    connect(process,&QProcess::errorOccurred,this,[=](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
            return;
        ai_processes.remove(agent);
        ai_prompts[agent].append(text);
        add_ai_history(agent,"activity","Cannot start Codex: "+
                       process->errorString());
        ui->ai_control_status->setText("● Queued");
        process->deleteLater();
    });
    connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this,[=]
    {
        ai_processes.remove(agent);
        ui->ai_control_status->setText("● Ready");
        process->deleteLater();
    });
    ui->ai_control_status->setText("● Starting Codex");
    auto prompt = text+
        "\n\n[DSI Studio] Reply through the DSI Studio local server using "
        "agent "+agent+" and session "+session+". Send user-facing progress "
        "with the JSON chat field and attach the final answer once to a LOG "
        "request. Do not use Codex CLI output as the reply.";
    process->start(codex,{"exec","resume",session,prompt});
}


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
    for(auto name : {"Rename","Share"})
        ai_project_menu->addAction(name)->setEnabled(false);
    ai_project_menu->addSeparator();

    connect(ai_project_menu->addAction("Remove"),&QAction::triggered,this,[this]
    {
        auto* item = ui->ai_project_list->currentItem();
        if(!item || QMessageBox::question(
               this,"Remove Project",
               "Remove this project and its saved history?") != QMessageBox::Yes)
            return;

        auto agent = item->data(Qt::UserRole).toString();
        QFile::remove(ai_project_dir+"/"+QString::fromLatin1(
                          QUrl::toPercentEncoding(agent))+".jsonl");
        ai_projects.remove(agent);
        ai_project_items.remove(agent);
        ai_sessions.remove(agent);
        ai_work_dirs.remove(agent);
        delete item;

        if(ui->ai_project_list->count())
            ui->ai_project_list->setCurrentRow(0);
        else
        {
            ui->ai_chat_history->clear();
            ui->ai_connected_agent->setText("Agent: None");
            ui->ai_control_status->setText("● Ready");
        }
    });

    connect(ui->ai_project_list,&QListWidget::currentItemChanged,this,
            [this](QListWidgetItem* item,QListWidgetItem*)
    {
        if(item)
            show_ai_project(item->data(Qt::UserRole).toString());
        else
            ui->ai_chat_history->clear();
    });

    connect(ui->ai_project_filter,&QLineEdit::textChanged,this,
            [this](const QString& text)
    {
        for(auto* item : ai_project_items)
            item->setHidden(
                !item->text().contains(text,Qt::CaseInsensitive));
    });

    for(const auto& info : dir.entryInfoList(
            {"*.jsonl"},QDir::Files,QDir::Time|QDir::Reversed))
    {
        auto agent = QUrl::fromPercentEncoding(
                         info.completeBaseName().toLatin1());
        QFile file(info.filePath());
        if(!file.open(QIODevice::ReadOnly))
            continue;

        auto& history = ai_projects[agent];
        while(!file.atEnd())
        {
            auto doc = QJsonDocument::fromJson(file.readLine());
            if(doc.isObject())
            {
                auto entry = doc.object();
                history.append(entry);
                if(entry["type"] == "request")
                {
                    auto request = QJsonDocument::fromJson(
                                       entry["text"].toString().toUtf8()).object();
                    auto session = request["session"].toString();
                    if(!session.isEmpty())
                        ai_sessions[agent] = session;
                    auto cwd = request["cwd"].toString();
                    if(QDir(cwd).exists())
                        ai_work_dirs[agent] = cwd;
                }
            }
        }

        if(history.isEmpty())
            ai_projects.remove(agent);
        else
            show_ai_project(agent);
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

    for(auto each : workdir_list)
        if(QFileInfo(each).exists())
        {
            ui->workDir->addItem(each);
            tipl::qt::working_dirs << QUrl::fromLocalFile(each);
        }

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
                reply->deleteLater();
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
                        reply2->deleteLater();
                    });
                }
                reply->deleteLater();
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
            auto lines = tipl::read_text_file(tipl::qt::to_path(file_name));
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
                QString(file_name).endsWith(".dcm") ||
                QString(file_name).endsWith(".nz") ||
                QString(file_name).endsWith(".mz"))
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
        for(int index = 0;index < file_list.size();++index)
        {
            ui->recentFib->setRowHeight(index,20);
            ui->recentFib->setItem(index,0,new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).filename().string().c_str()));
            ui->recentFib->setItem(index,1,new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).parent_path().string().c_str()));
            for(int col = 0;col < 2;++col)
            {
                auto item = ui->recentFib->item(index,col);
                item->setFlags(item->flags() & ~Qt::ItemIsEditable);
                if(!QFileInfo::exists(file_list[index]))
                    item->setForeground(Qt::gray);
            }
        }
    }
    {
        QStringList file_list = settings.value("recentSrcFileList").toStringList();
        ui->recentSrc->clear();
        ui->recentSrc->setRowCount(file_list.size());
        for(int index = 0;index < file_list.size();++index)
        {
            ui->recentSrc->setRowHeight(index,20);
            ui->recentSrc->setItem(index,0,new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).filename().string().c_str()));
            ui->recentSrc->setItem(index,1,new QTableWidgetItem(std::filesystem::path(file_list[index].toStdString()).parent_path().string().c_str()));
            for(int col = 0;col < 2;++col)
            {
                auto item = ui->recentSrc->item(index,col);
                item->setFlags(item->flags() & ~Qt::ItemIsEditable);
                if(!QFileInfo::exists(file_list[index]))
                    item->setForeground(Qt::gray);
            }
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
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(tipl::qt::to_path(filename)))
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

    if(int p = filename.lastIndexOf('_');!filename.endsWith("_dseg.nii.gz") && p != -1)
    {
        auto dseg_file = filename.left(p) + "_dseg.nii.gz";
        if(QFileInfo(dseg_file).exists())
            tracking_windows.back()->command({"open_region",dseg_file.toUtf8().constData()});
    }
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

void MainWindow::on_Reconstruction_clicked()
{
    QStringList filenames = tipl::qt::open_image_files(this,ui->workDir->currentText(),
                                            "Src files (*.sz *src.gz);;Histology images (*.jpg *.tif);;All files (*)" );
    if (filenames.isEmpty())
        return;
    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    loadSrc(filenames);
}

void MainWindow::on_FiberTracking_clicked()
{
    QString filename = tipl::qt::open_image_file(this,ui->workDir->currentText(),"Fib files (*.fz *fib.gz *.dz);;All files (*)");
    if (filename.isEmpty())
        return;
    add_work_dir(QFileInfo(filename).absolutePath());
    loadFib(filename);
}

void MainWindow::on_T1WFiberTracking_clicked()
{
    QString filename = tipl::qt::open_image_file(this,ui->workDir->currentText(),"Image files (*nii.gz *.nii 2dseq);;All files (*)");
    if (filename.isEmpty())
        return;
    add_work_dir(QFileInfo(filename).absolutePath());
    loadFib(filename);
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



void MainWindow::on_view_image_clicked()
{
    QStringList filename = tipl::qt::open_image_files(this,ui->workDir->currentText(),
                                           "image files (*.nii *nii.gz *.dcm *.nhdr *.nrrd 2dseq);;All files (*)");
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

void MainWindow::on_nonlinear_reg_clicked()
{
    RegToolBox* rt = new RegToolBox(this);
    rt->setAttribute(Qt::WA_DeleteOnClose);
    rt->showNormal();
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
    QStringList filename = QFileDialog::getOpenFileNames(
            this,"Open Network Measures",ui->workDir->currentText(),
            "Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    std::ofstream out((filename[0]+".collected.txt").toStdString());
    out << "Field\t";
    for(int i = 0;i < filename.size();++i)
        out << QFileInfo(filename[i]).baseName().toStdString() << "\t";
    out << std::endl;

    std::vector<std::string> line_output;
    for(int i = 0;i < filename.size();++i)
    {
        std::ifstream in(filename[i].toStdString());
        std::vector<std::string> node_list;
        // global measures
        size_t line_index = 0;
        while(in)
        {
            std::string t1,t2;
            if(!(in >> t1))
                break;
            if(t1 == "network_measures")
            {
                std::string nodes;
                std::getline(in,nodes);
                std::istringstream nodestream(nodes);
                std::copy(std::istream_iterator<std::string>(nodestream),
                          std::istream_iterator<std::string>(),std::back_inserter(node_list));
                break;
            }
            if(!(in >> t2))
                break;
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
            if(t1.empty() || t1[0] == '#' || t1[0] == ' ')
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
    std::string json_name(tipl::remove_all_suffix(nii_name) + ".json");
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
                                    this,
                                    "Open directory",
                                    ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);
    std::vector<std::filesystem::path> dwi_nii_files;
    search_dwi_nii(tipl::qt::to_path(dir),dwi_nii_files);
    if(dwi_nii_files.empty())
    {
        QMessageBox::critical(this,"ERROR","cannot find nifti data");
        return;
    }
    auto output_dir = tipl::qt::to_path(dir);


    bool no_to_all = false;
    bool yes_to_all = false;
    tipl::progress prog("batch creating src");
    std::deque<std::filesystem::path> nii_list,src_list;
    size_t nii_count = 0;
    std::mutex access_list;
    bool ended = false;
    tipl::par_for(8,[&](size_t index)
        {
            if(tipl::is_main_thread())
            {
                for(int j = 0;j < dwi_nii_files.size();++j)
                {
                    auto nii_name = dwi_nii_files[j];
                    auto src_name = output_dir/tipl::remove_all_suffix(nii_name.filename());
                    src_name += ".sz";
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
                prog(dwi_nii_files.size()-nii_list.size(),dwi_nii_files.size());

                std::filesystem::path nii_name,src_name;
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
                tipl::out() << "processing " << nii_name;
                src_data src;
                if(!src.load_from_file({nii_name},true) || !src.save_to_file(src_name))
                    tipl::warning() << src.error_msg;
            }
        },8);
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

void MainWindow::on_fiber_data_hub_clicked()
{
    if(!fiber_data_hub)
        fiber_data_hub = new FiberDataHub(this);
    fiber_data_hub->showNormal();
    fiber_data_hub->raise();
    fiber_data_hub->activateWindow();
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
    settings.setValue("recentSrcFileList", QStringList());
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
        if(std::filesystem::path(each).stem().u8string() == name.toStdString())
        {
            loadFib(each.u8string().c_str());
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
    auto fail = [this](std::string error)
    {
        error_msg = error;
        return false;
    };
    const std::string usage =
        "list_recent | run_cli <command line> | hub repos | hub tags <repo> | "
        "hub files <repo> <tag> [text] [offset] [limit] | hub open <repo> <tag> <file> | "
        "hub download <repo> <tag> <file> <dir>";

    if(cmd.size() == 1 && cmd[0] == "list_recent")
    {
        for(const auto& file : settings.value("recentSrcFileList").toStringList() +
                                    settings.value("recentFibFileList").toStringList())
            if(file.endsWith(".sz",Qt::CaseInsensitive) ||
                file.endsWith(".fz",Qt::CaseInsensitive))
                tipl::out() << file.toStdString();
        return true;
    }

    if(cmd.size() == 2 && cmd[0] == "run_cli")
    {
        tipl::program_option<tipl::out> po;
        if(!po.parse(cmd[1]) || !po.check("action"))
            return fail(po.error_msg);
        if(run_action_with_wildcard(po))
            return fail("command line failed");
        po.check_end_param<tipl::warning>();
        return true;
    }
    if(cmd.empty() || cmd[0] != "hub")
        return fail(usage);
    on_fiber_data_hub_clicked();
    if(!fiber_data_hub->command(cmd))
        return fail(fiber_data_hub->error_msg);
    return true;
}
