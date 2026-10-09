#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMenu>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QShortcut>
#include <QSharedPointer>
#include <QShowEvent>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTextFrame>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QUuid>
#include <QUrl>
#include <QUrlQuery>
#include <QVBoxLayout>

#include <algorithm>
#include "ai_agent.hpp"
#include "cmd/ai.hpp"
#include "ui_ai_agent.h"
#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "TIPL/tipl.hpp"

constexpr qsizetype ai_debug_truncate_length = 300; // level 1 (truncated) caps each logged line to this many characters
static const QStringList local_agents{"Codex","Claude","Muse","Antigravity","Grok"}; // every local CLI agent, in settings order
QProcessEnvironment agent_environment(const QString& provider)
{
    auto env = QProcessEnvironment::systemEnvironment();
#if defined(Q_OS_WIN) || defined(Q_OS_LINUX)
    if(provider == "Muse" && !env.contains("TBH_CREDENTIAL_BACKEND"))
        env.insert("TBH_CREDENTIAL_BACKEND","file");
#endif
    return env;
}
void start_process(QProcess& process,const QString& executable,QStringList args)
{
#ifdef Q_OS_WIN
    if(executable.endsWith(".cmd",Qt::CaseInsensitive) ||
       executable.endsWith(".bat",Qt::CaseInsensitive))
    {
        process.start(qEnvironmentVariable("ComSpec","cmd.exe"),QStringList{"/d","/c",executable}+args);
        return;
    }
#endif
    process.start(executable,args);
}
void kill_process_tree(QProcess* process) // kill(): a windowless console child never sees terminate()'s WM_CLOSE
{
    if(!process || process->state() == QProcess::NotRunning)
        return;
#ifdef Q_OS_WIN
    if(auto pid = process->processId()) // taskkill /T first: it walks descendants (e.g. node under the cmd.exe wrapper) only while the parent PID lives
    {
        QProcess taskkill; // not QProcess::execute(): that forwards taskkill's "SUCCESS: ..." lines to DSI Studio's console
        taskkill.start("taskkill",{"/PID",QString::number(pid),"/T","/F"});
        taskkill.waitForFinished();
    }
#endif
    if(process->state() != QProcess::NotRunning)
    {
        process->kill();
        // reap only a parentless (stack) QProcess; a parented one finishes asynchronously, never inside the caller's stdout loop
        if(!process->parent())
            process->waitForFinished(1000);
    }
}
void fail_agent_process(QProcess* process,const QString& message) // a provider-protocol failure: the finished handler reports fatal_error ahead of stderr
{
    process->setProperty("fatal_error",message);
    kill_process_tree(process);
}
QString muse_uuid_v7()
{
    auto bytes = QUuid::createUuid().toRfc4122();
    auto ms = quint64(QDateTime::currentMSecsSinceEpoch());
    for(int i = 5;i >= 0;--i,ms >>= 8)
        bytes[i] = char(ms);
    bytes[6] = char((quint8(bytes[6])&0x0f)|0x70);
    bytes[8] = char((quint8(bytes[8])&0x3f)|0x80);
    return QUuid::fromRfc4122(bytes).toString(QUuid::WithoutBraces);
}
QByteArray muse_command(const QString& id,const QString& method,QJsonObject params)
{
    params["commandId"] = id;
    return json_line({{"jsonrpc","2.0"},{"id",id},
        {"method",method},{"params",params}});
}
QByteArray muse_initialize() // one handshake for the status probe, model list and chat; the probe checks experimentalApi in the reply
{
    return json_line({{"jsonrpc","2.0"},{"id","initialize"},{"method","initialize"},
        {"params",QJsonObject{{"clientInfo",QJsonObject{{"name","dsi_studio"},{"title","DSI Studio"},{"version","1.0"}}},
            {"capabilities",QJsonObject{{"experimentalApi",true},{"userInputDialogs",false}}}}}});
}
QByteArray grok_initialize()
{
    return json_line({{"jsonrpc","2.0"},{"id","initialize"},{"method","initialize"},
        {"params",QJsonObject{{"protocolVersion",1},{"clientCapabilities",QJsonObject()}}}});
}

void AIAgent::ai_log(QString text)
{
    if(ai_debug_level <= 0)
        return;
    if(ai_debug_level == 1 && text.size() > ai_debug_truncate_length)
        text = text.left(ai_debug_truncate_length)+"...";
    auto prefix = QString("[DEBUG] ");
    tipl::out() << (prefix+text.remove('\r').
                    replace('\n',"\n"+prefix)).toStdString();
}
QJsonObject AIAgent::next_json_line(QProcess* process)
{
    auto line = process->readLine();
    ai_log("stdout:"+QString::fromUtf8(line).trimmed());
    return QJsonDocument::fromJson(line).object();
}
AIAgent::AIAgent(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::AIAgent)
{
    ui->setupUi(this);
    ai_debug_level = settings.value("ai/debug",0).toInt();
    ui->ai_work_dir->setText(main_window.work_dir());
    // the selected chat's dispatch directory (model_settings["cwd"], also run_shell's "cd" target)
    auto sync_work_dir = [this]
    {
        auto* info = selected_info();
        if(!info)
            return;
        auto cwd = ui->ai_work_dir->text().trimmed();
        if(info->model_settings["cwd"].toString() != cwd)
        {
            info->model_settings["cwd"] = cwd;
            info->save_config();
        }
    };
    connect(ui->ai_work_dir,&QLineEdit::editingFinished,this,sync_work_dir);
    connect(ui->ai_browse_work_dir,&QPushButton::clicked,this,[this,sync_work_dir]
    {
        auto path = QFileDialog::getExistingDirectory(
            this,"Select AI Work Directory",ui->ai_work_dir->text());
        if(!path.isEmpty())
        {
            ui->ai_work_dir->setText(QDir::toNativeSeparators(path));
            sync_work_dir();
        }
    });
    ai_status_timer = new QTimer(this);
    connect(ai_status_timer,&QTimer::timeout,this,[this]
    {
        bool running = false;
        for(auto& entry : ai_infos)
        {
            auto& info = entry.second;
            if(info.is_running() || web_connected(info)) // a polling Web chat pulses green
            {
                running = true;
                if(auto* row = ui->ai_project_list->itemWidget(info.project_items))
                    update_status_dot(row->findChild<QLabel*>("ai_project_status_dot"),
                                      info.status,true);
            }
        }
        if(!running)
            ai_status_timer->stop();
        if(auto* info = selected_info();info && info->is_running())
        {
            auto status = ui->ai_status->text();
            ui->ai_status->setText(
                status.endsWith("...") ? status.chopped(2) : status+".");
            ui->ai_status->repaint();
        }
    });
    ui->ai_status->hide();

    web_timer.setSingleShot(true);
    web_manager.setTransferTimeout(30000); // a hung Google request would otherwise stall polling for good
    connect(&web_timer,&QTimer::timeout,this,&AIAgent::poll_web);

    refresh_agent_executables();
    refresh_agent_status();
    for(const auto& provider : local_agents) // the first installed agent; Codex when none is
        if(!agent_entries[provider].executable.isEmpty())
        {
            current_agent = provider;
            break;
        }
    update_agent_status_label();
    auto* send = new QShortcut(
        QKeySequence(Qt::CTRL|Qt::Key_Return),ui->ai_chat_input);
    send->setContext(Qt::WidgetShortcut);
    connect(send,&QShortcut::activated,this,[this]
    {
        // a fixed "submit" shortcut, never Stop/Resume regardless of what the button currently shows
        if(current_send_action() == send_action::Send)
            ui->ai_send_message->click();
    });
    connect(ui->ai_chat_input,&QPlainTextEdit::textChanged,
            this,&AIAgent::update_send_button);



    ai_project_menu = new QMenu(this);
    ai_project_menu->setStyleSheet(
        "QMenu{background:#fff;border:1px solid #d9d9dc;padding:4px;}"
        "QMenu::item{padding:6px 24px 6px 10px;border-radius:4px;}"
        "QMenu::item:selected{background:#e9e9eb;}"
        "QMenu::item:disabled{color:#9a9a9e;}"
        "QMenu::separator{height:1px;background:#dedee1;margin:4px;}");
    connect(ai_project_menu->addAction("Rename"),&QAction::triggered,this,[this]
    {
        auto* info = selected_info();
        if(!info) // menu can only be reached via a row's own "..." button, but guard against it anyway rather than trust that indirectly
            return;
        bool okay;
        auto title = QInputDialog::getText(
            this,"Rename Chat","Chat name:",QLineEdit::Normal,
            info->title(),&okay);
        if(okay && info->save_title(title))
            show_ai_project(*info);
        else if(okay)
            QMessageBox::warning(
                this,"Rename Chat","The chat name could not be saved.");
    });
    connect(ai_project_menu->addAction("Details..."),
            &QAction::triggered,this,[this]
    {
        auto* info = selected_info();
        if(!info)
            return;
        QMessageBox details(
            QMessageBox::Information,"Chat Details",
            info->details(),
            QMessageBox::Ok,this);
        details.setTextInteractionFlags(
            Qt::TextSelectableByMouse|Qt::TextSelectableByKeyboard);
        details.exec();
    });
    ai_project_menu->addSeparator();
    connect(ai_project_menu->addAction("Remove"),&QAction::triggered,this,[this]
    {
        auto row = ui->ai_project_list->currentRow();
        if(row < 0)
            return;
        auto session = ui->ai_project_list->item(row)->data(Qt::UserRole).toString();
        if(session == web_session_id)
            stop_web();
        if(auto* found = ai_info::find(session))
        {
            if(auto* process = found->processes)
            {
                process->disconnect(); kill_process_tree(process); process->waitForFinished(1000); process->deleteLater();
            }
            if(found->provider == "Web") // trash its mailbox Doc so "DSI Studio AI" does not accumulate them
                google_api("PATCH","https://www.googleapis.com/drive/v3/files/"+found->model_settings["google_file_id"].toString(),
                           {{"trashed",true}},[](QJsonObject){});
        }
        QFile::remove(ai_info::history_file(session));
        QFile::remove(ai_info::config_file(session));
        settings.remove("ai/title/"+session);
        ai_infos.erase(session);
        auto* taken_item = ui->ai_project_list->takeItem(row); // defer delete: its row widget owns the "..." button whose menu action is still running
        QTimer::singleShot(0,this,[taken_item]{delete taken_item;});

        // keep a chat selected whenever one exists
        if(ui->ai_project_list->count())
            ui->ai_project_list->setCurrentRow(std::min(row,ui->ai_project_list->count()-1));
    });

    connect(ui->ai_project_list,&QListWidget::currentItemChanged,this,
            [this](QListWidgetItem* item,QListWidgetItem* previous)
    {
        for(auto* i : {previous,item})
        {
            if(!i)
                continue;
            auto* widget = ui->ai_project_list->itemWidget(i);
            if(!widget) // null for an item already detached from the list (e.g. mid-removal)
                continue;
            if(auto* button = widget->findChild<QPushButton*>())
                button->setStyleSheet(i == item ?
                    "color:#202124;background:#dce9f9;" : "");
        }
        if(!item)
        {
            ui->ai_chat_history->clear();
            ui->ai_chat_history->setEnabled(false);
            ui->ai_chat_input->setEnabled(false);
            update_send_button();
            return ui->ai_status->hide();
        }

        stop_blink(ui->ai_project_list->itemWidget(item));
        auto* info = selected_info();
        if(!info) // item is a real row, but guard anyway rather than trust that indirectly
            return;
        ui->ai_chat_history->setEnabled(true);
        ui->ai_chat_input->setEnabled(true);
        ui->ai_work_dir->setText(info->model_settings.contains("cwd") ?
            info->model_settings["cwd"].toString() : main_window.work_dir());
        update_agent_status_label();
        show_ai_project(*info);
    });

    for(const auto& info : QDir(ai_project_dir).entryInfoList(
            {"*.jsonl"},QDir::Files,QDir::Time|QDir::Reversed))
    {
        auto session = QUrl::fromPercentEncoding(
                           info.completeBaseName().toLatin1());
        QList<QJsonObject> history;
        QFile file(info.filePath());
        if(!file.open(QIODevice::ReadOnly))
            continue;
        while(!file.atEnd())
            if(auto doc = QJsonDocument::fromJson(file.readLine());doc.isObject())
                history.append(doc.object());
        if(history.isEmpty() || session.isEmpty())
            continue;
        QJsonObject config;
        if(QFile config_file(ai_info::config_file(session));config_file.open(QIODevice::ReadOnly))
            config = QJsonDocument::fromJson(config_file.readAll()).object();
        auto* ai = ai_info::create(
            session,config["provider"].toString(),config["agent"].toString());
        if(!ai)
            continue;
        bool established = config["established"].toBool();
        set_ai_status(ai->sessions,established ? session_status::Completed : session_status::New,
                      established ? "Previous chat loaded." : "Previous attempt never connected.");
        ai->model_settings = config["model_settings"].toObject();
        ai->project_titles = settings.value("ai/title/"+session).toString();
        ai->projects = std::move(history);
        show_ai_project(*ai);
    }
    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);
    else // no chat at all: leave the right pane disabled until New Chat creates one
    {
        ui->ai_chat_history->setEnabled(false);
        ui->ai_chat_input->setEnabled(false);
        update_send_button();
    }
    // Web chats are never auto-reconnected at startup; use Resume to reconnect a chat explicitly
}

AIAgent::~AIAgent()
{
    // app exit can end before closeEvent()'s 5 s fallback: tree-kill what is still live, disconnected from this half-destroyed object
    for(auto& entry : ai_infos)
        if(auto* process = entry.second.processes)
        {
            process->disconnect();
            kill_process_tree(process);
            process->waitForFinished(1000);
            entry.second.processes = nullptr; // ai_infos outlives this window; its child QProcess does not
        }
    delete ui;
}

void AIAgent::add_ai_reply(QProcess* process,const QString& chat,const QString& reasoning) // a parsed agent reply; an empty one is ignored
{
    if(chat.isEmpty() && reasoning.isEmpty())
        return;
    if(auto* info = ai_info::find(process->objectName()))
    {
        process->setProperty("had_reply",true);
        add_ai_reply(*info,chat,reasoning);
    }
}
void AIAgent::add_ai_reply(ai_info& info,const QString& chat,const QString& reasoning)
{
    auto entry = info.record_reply(chat,reasoning);
    set_ai_status(info.sessions,chat.isEmpty() ? session_status::Thinking : session_status::WaitingUser,
                  chat.isEmpty() ? "Agent is thinking" : "Agent replied; waiting for your message.");
    show_ai_project(info,entry); // pass the entry so show_ai_project can see it's a new non-user reply and blink
}

void AIAgent::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    refresh_agent_executables(); // picks up a CLI installed since the window was last shown, before the refreshes below read agent_entries[...].executable
    for(const auto& provider : local_agents)
        refresh_agent_models(provider);
    refresh_ollama_models(); // one /api/tags request feeds both Claude and Codex
    auto* item = ui->ai_project_list->currentItem();
    stop_blink(item ? ui->ai_project_list->itemWidget(item) : nullptr);
}

void AIAgent::closeEvent(QCloseEvent* event)
{
    // each process's finished handler (prepare_ai()) classifies New/Completed/Failed itself
    for(auto& entry : ai_infos)
        if(auto* process = entry.second.processes)
        {
            process->setProperty("user_stopped",true); // finished() reports a user stop, not a failure
            process->closeWriteChannel();
            QTimer::singleShot(5000,process,[process]
            {
                kill_process_tree(process);
            });
        }
    stop_web();
    QMainWindow::closeEvent(event);
}

void AIAgent::set_ai_status(const QString& session,session_status status,QString message)
{
    auto* info = ai_info::find(session);
    if(!info)
        return;
    info->status = status;
    info->status_message = std::move(message);
    if(ai_debug_level)
        tipl::out() << "[DEBUG] " << info->agent_name.toStdString() << "@"
                    << session.toStdString() << " " << session_status_text(status).toStdString()
                    << ": " << info->status_message.toStdString();
    update_ai_status(*info,true);
}

void AIAgent::update_ai_status(const ai_info& info,bool pulse)
{
    bool running = info.is_running() || web_connected(info); // a polling Web chat pulses green
    if(info.project_items)
    {
        auto* row = ui->ai_project_list->itemWidget(info.project_items);
        update_status_dot(row ? row->findChild<QLabel*>("ai_project_status_dot") : nullptr,
                          info.status,pulse && running);
    }

    if(running && !ai_status_timer->isActive())
        ai_status_timer->start(500);
    if(selected_info() != &info)
        return;
    // one line -- a multi-line stderr dump (Failed) must not grow the composer's height
    auto text = (session_status_text(info.status)+": "+info.status_message).simplified();
    if(running && !text.endsWith('.'))
        text += ".";
    ui->ai_status->show();
    ui->ai_status->setToolTip(text); // full message on hover -- the label itself may show a truncated "..." version
    ui->ai_status->setText(QFontMetrics(ui->ai_status->font()).elidedText(
        text,Qt::ElideRight,ui->ai_status->maximumWidth()-30)); // truncate -- an unbounded message here was pushing the whole window wider
    ui->ai_status->repaint();
    update_send_button(); // Stop/Send now follow the session status
}

void AIAgent::ai_request(const QByteArray& data,QByteArray& reply)
{
    auto status_reply = [](QString status,QString error = {})
    {
        QJsonObject reply{{"status",status}};
        if(!error.isEmpty())
            reply["error"] = error;
        return QJsonDocument(reply).toJson(QJsonDocument::Compact);
    };
    QJsonParseError parse_error;
    auto doc = QJsonDocument::fromJson(data,&parse_error);
    auto request = doc.object();
    auto session = request["session"].toString().trimmed();
    if(!doc.isObject())
        return void(reply = status_reply("error","invalid JSON: "+parse_error.errorString()));
    if(session.isEmpty())
        return void(reply = status_reply("error","missing session: provide resumable provider thread ID"));
    if(!is_valid_session_id(session))
        return void(reply = status_reply("error","invalid session: provide resumable provider thread ID"));

    auto* found = ai_info::find(session);
    // a lone `dsi.sh new_chat` (no param, batch or chat text) is the one way an unknown UUID joins: an external agent registers its own session
    if(auto command = request["command"].toObject();request["command"].isObject() && command.size() == 1 &&
       command["cmd"].toString() == "new_chat" && !request.contains("chat") && !request.contains("reasoning"))
    {
        auto agent = request["agent"].toString().trimmed();
        if(found)
            return void(reply = status_reply("error","session already exists"));
        if(!local_agents.contains(agent))
            return void(reply = status_reply("error","invalid agent"));
        found = ai_info::create(session,"AgentServer",agent); // AgentServer: not a DSI-launched process, so no Send/Stop/model control
        set_ai_status(session,session_status::WaitingUser,"External agent session connected.");
        add_ai_history(*found,"activity","External agent session connected."); // a recorded entry is what lets save_config() persist it
        found->save_config();
        ai_log("external agent session connected: "+agent+"@"+session);
        return void(reply = status_reply("success"));
    }
    if(!found) // any other request must match an existing chat; a removed chat's orphan fails here instead of recreating it
        return void(reply = status_reply("error","session not found"));
    ai_info& info = *found;
    set_ai_status(session,session_status::Thinking,"Processing agent request");

    ai_log("received: "+QString::fromUtf8(data));
    auto chat = request["chat"].toString().trimmed();
    auto reasoning = request["reasoning"].toString().trimmed();
    dispatching_info = &info;
    auto result = main_window.dispatch_cmd(info,request); // MainWindow's command center handles everything
    dispatching_info = nullptr;
    // a live local process is still mid-turn (this was one of its tool calls); Web or no transport is idle now
    if(info.processes && info.processes->state() != QProcess::NotRunning)
        set_ai_status(session,session_status::Thinking,
                      "Command completed; waiting for agent input");
    else
        set_ai_status(session,session_status::WaitingUser,"Request completed; waiting for next request.");

    auto entry = info.record_reply(chat,reasoning);
    reply = QJsonDocument(result).toJson(QJsonDocument::Compact);
    ai_log(QString("reply for %1@%2: %3 ...")
               .arg(info.agent_name,session,
                    QString::fromUtf8(reply).left(32)));
    show_ai_project(info,entry);
}

void AIAgent::update_current_window(QWidget* window)
{
    if(dispatching_info)
        dispatching_info->current_window = command_window_id(window);
}

void AIAgent::show_ai_project(ai_info& info,QJsonObject added_entry)
{
    auto* item = info.project_items;
    if(!item)
    {
        item = new QListWidgetItem;
        item->setData(Qt::UserRole,info.sessions);
        ui->ai_project_list->insertItem(0,item);
        info.project_items = item;

        auto* row = new QWidget;
        auto* status_dot = new QLabel(row);
        status_dot->setObjectName("ai_project_status_dot");
        status_dot->setFixedSize(10,10);

        auto* title = new QPushButton(row);
        title->setFlat(true);
        title->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Preferred);

        auto* button = new QToolButton(row);
        button->setObjectName("ai_project_menu_button");
        button->setText("...");
        button->setToolTip("Project actions");
        button->setFixedSize(28,28);
        button->setPopupMode(QToolButton::InstantPopup);
        button->setMenu(ai_project_menu);

        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(6,2,2,2);
        layout->setSpacing(6);
        layout->addWidget(status_dot);
        layout->addWidget(title,1);
        layout->addWidget(button);
        ui->ai_project_list->setItemWidget(item,row);

        auto* blink = new QTimer(row);
        blink->setInterval(500);
        connect(blink,&QTimer::timeout,row,[row]
        {
            row->setStyleSheet(row->styleSheet().isEmpty() ?
                "background:#ffe082;border-radius:5px;" : "");
        });

        auto select = [this,item]{ui->ai_project_list->setCurrentItem(item);};
        connect(title,&QPushButton::clicked,this,select);
        connect(button,&QToolButton::pressed,this,select);
    }

    auto* row = ui->ai_project_list->itemWidget(item);
    auto* title = row->findChild<QPushButton*>();
    // never touched (no content, no title), not merely New, which a reconnecting chat also is
    auto chat_title = info.projects.isEmpty() && info.project_titles.isEmpty() ?
        "New "+info.agent_name+" Chat" : info.title();
    title->setText((info.provider == "Web" ? QString("🌐 ") : QString())+chat_title);
    title->setToolTip(title->text());
    title->repaint();
    item->setSizeHint(QSize(0,row->sizeHint().height()));

    update_ai_status(info);

    auto* current = ui->ai_project_list->currentItem();
    const auto added_type = added_entry["type"].toString();
    if(!current && added_type == "user") // the user just started this chat themselves (nothing else was selected): bring it up
    {
        ui->ai_project_list->setCurrentItem(item);
        return; // currentItemChanged already rebuilt this chat's complete history
    }

    if(!added_type.isEmpty() && added_type != "user" &&
       (current != item || !isVisible()))
    {
        row->setStyleSheet("background:#ffe082;border-radius:5px;");
        row->findChild<QTimer*>()->start();
    }

    if(current != item)
        return;

    show_ai_history(info,std::move(added_entry));
}

void AIAgent::show_ai_history(ai_info& info,QJsonObject added_entry)
{
    auto* bar = ui->ai_chat_history->verticalScrollBar();
    const auto scroll_position = bar->value();
    const bool follow_latest = bar->maximum()-scroll_position <= bar->singleStep();
    const auto& history = info.projects;
    const auto added_type = added_entry["type"].toString();
    auto to_html = [](QString text)
    {
        return text.toHtmlEscaped().replace('\n',"<br>");
    };
    // renders chat/reasoning text as Markdown (bold, lists, code, links, ...) instead of plain escaped text;
    // falls back to plain escaping if the body can't be extracted from QTextDocument's generated HTML
    auto markdown_to_html = [&](const QString& text)
    {
        QTextDocument doc;
        doc.setMarkdown(text);
        auto html = doc.toHtml();
        auto begin = html.indexOf("<body");
        begin = begin < 0 ? -1 : html.indexOf('>',begin);
        auto end = html.lastIndexOf("</body>");
        if(begin < 0 || end < 0 || end <= begin)
            return to_html(text);
        static const QRegularExpression loose_margins(
            "margin-top:\\d+px; margin-bottom:\\d+px;");
        return html.mid(begin+1,end-begin-1).trimmed().replace(
            loose_margins,"margin-top:0px; margin-bottom:6px;");
    };
    const bool show_reasoning = settings.value("ai/show_reasoning",false).toBool(); // read once: append() runs per history entry
    auto hidden_assistant = [&](const QJsonObject& entry)
    {
        return entry["type"] == "assistant" &&
               entry["text"].toString().trimmed().isEmpty() &&
               (!show_reasoning || entry["reasoning"].toString().trimmed().isEmpty());
    };
    auto append = [&](const QJsonObject& entry,const QStringList& activities = {})
    {
        bool user = entry["type"] == "user";
        auto content = entry["text"].toString();
        auto reasoning = show_reasoning ? entry["reasoning"].toString().trimmed() : QString();
        if(content.trimmed().isEmpty() && reasoning.isEmpty() && activities.isEmpty())
            return;

        content = content.trimmed().isEmpty() ? QString() : markdown_to_html(content);
        if(!reasoning.isEmpty())
            content = "<span style=\"color:#5f6368;\">"+markdown_to_html(reasoning)+"</span>"+
                      (content.isEmpty() ? "" : "<br>"+content);

        if(!activities.isEmpty())
            content += QString("<div style=\"margin:0;color:#5f6368;font-size:9pt;\">") +
                       activities.join("<br>") + "</div>";

        auto type = entry["type"].toString();
        auto color = user ? "#e8f0fe" : type == "error" ? "#fce8e6" : type == "activity" ? "#f1f3f4" : "#e8f5e9";
        auto time = QDateTime::fromString(entry["time"].toString(),Qt::ISODate).
                    toString("MM/dd HH:mm:ss");
        auto cell = QString(
                        "<td bgcolor=\"%1\"><b style=\"background-color:%1\">%2</b>"
                        "<font color=\"#80868b\">%3</font><br>%4</td>")
                        .arg(color,(user ? QString("You") : info.agent_name).toHtmlEscaped()+" &middot; ",time,content);

        auto cursor = ui->ai_chat_history->document()->
                      rootFrame()->lastCursorPosition();
        cursor.insertHtml(
            QString("<table width=\"100%\" cellspacing=\"3\" "
                    "cellpadding=\"7\"><tr>%1</tr></table>")
                .arg(user ? "<td width=\"20%\"></td>"+cell :
                         cell+"<td width=\"20%\"></td>"));
    };

    if(added_type.isEmpty() || added_type == "request" ||
       (added_type == "assistant" &&
        history.size() > 1 &&
        history[history.size()-2]["type"] == "request"))
    {
        ui->ai_chat_history->clear();
        for(int index = 0;index < history.size();)
        {
            const auto& entry = history[index];

            if(hidden_assistant(entry))
            {
                ++index;
                continue;
            }

            QJsonObject owner;
            if(entry["type"] == "assistant")
            {
                owner = entry;
                ++index;

                while(index < history.size() &&
                      hidden_assistant(history[index]))
                    ++index;

                if(index == history.size() ||
                   history[index]["type"] != "request")
                {
                    append(owner);
                    continue;
                }
            }
            else if(entry["type"] == "request")
                owner = QJsonObject{
                    {"type","assistant"},
                    {"time",entry["time"]}
                };
            else
            {
                append(entry);
                ++index;
                continue;
            }

            QStringList activities,commands;
            QString target;

            auto add_activity = [&]
            {
                if(commands.isEmpty())
                    return;
                activities << "<b>"+to_html(target)+"</b>: "+
                                  commands.join(" &rarr; ");
                commands.clear();
            };

            for(;index < history.size();++index)
            {
                const auto& request = history[index];
                if(hidden_assistant(request))
                    continue;
                if(request["type"] != "request")
                    break;

                auto request_target = request["title"].toString();
                request_target +=
                    (request_target.isEmpty() ? "" : " · ")+
                    request["window"].toString();

                if(!commands.isEmpty() && target != request_target)
                    add_activity();
                target = request_target;

                auto command =
                    "<code>"+to_html(request["text"].toString())+"</code>";

                if(index+1 < history.size())
                {
                    if(auto duration = QDateTime::fromString(
                                            request["time"].toString(),Qt::ISODate).msecsTo(
                                                QDateTime::fromString(
                                                    history[index+1]["time"].toString(),
                                                    Qt::ISODate));duration >= 0)
                    {
                        auto seconds =
                            QString::number(duration/1000.0,'f',1);
                        if(seconds.endsWith(".0"))
                            seconds.chop(2);
                        command += " ("+seconds+"s)";
                    }
                }

                commands << command;
            }

            add_activity();
            append(owner,activities);
        }
    }
    else
        append(added_entry);

    // Rebuilding the document resets its scroll position; preserve readers browsing earlier replies.
    if(follow_latest)
        QTimer::singleShot(
            0,bar,[bar]{bar->setValue(bar->maximum());});
    else
        bar->setValue(scroll_position);
    ui->ai_chat_history->viewport()->repaint();
}

void AIAgent::update_agent_models(
    const QString& agent,const QStringList& names,bool ollama)
{
    auto& profiles = agent_entries[agent].profiles;
    auto previous = profiles;
    for(auto i = profiles.begin();i != profiles.end();)
        if(i.value().toObject().contains("provider") == ollama)
            i = profiles.erase(i);
        else
            ++i;
    for(const auto& name : names)
        profiles[name] = ollama ? // "url": the server this model was discovered on, so a chat stays pinned to it
            QJsonObject{{"provider",true},{"url",ai_ollama_url(settings).first.toString()}} : previous[name].toObject();

    if(current_agent == agent)
    {
        // refresh the default model's cached profile; value(), not [], which would insert a blank profile
        if(current_model_name.isEmpty() || profiles.contains(current_model_name))
            current_model_info = profiles.value(current_model_name).toObject();
        update_agent_status_label();
    }
    if(profiles != previous)
        emit agent_models_changed(agent);
}
void AIAgent::refresh_agent_executables() // re-discover CLIs, so one installed while DSI Studio runs is picked up
{
    auto set_executable = [this](const QString& provider,const QString& name,QString fallback)
    {
        auto path = QStandardPaths::findExecutable(name);
        if(path.isEmpty() && QFileInfo::exists(fallback))
            path = fallback;
        agent_entries[provider].executable = path;
        ai_log(path.isEmpty() ? provider+" not found" : provider+": "+path);
    };

    QString codex_path;
    if(QStandardPaths::findExecutable("codex").isEmpty())
    {
        QDir dir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation)+"/OpenAI/Codex/bin");
        for(const auto& name : dir.entryList(QDir::Dirs|QDir::NoDotAndDotDot,QDir::Time))
            if(QFileInfo::exists(codex_path = dir.filePath(name+"/codex.exe")))
                break;
    }
    set_executable("Codex","codex",codex_path);
#ifdef Q_OS_WIN
    auto local_app_data = qEnvironmentVariable("LOCALAPPDATA");
    set_executable("Claude","claude",QDir::homePath()+"/.local/bin/claude.exe");
    set_executable("Muse","muse",local_app_data.isEmpty() ? QString() : local_app_data+"/Programs/muse/muse.cmd");
    set_executable("Antigravity","agy",local_app_data+"/agy/bin/agy.exe");
    set_executable("Grok","grok",QDir::homePath()+"/.grok/bin/grok.exe");
#else
    set_executable("Claude","claude",{});
    set_executable("Muse","muse",QDir::homePath()+"/.local/bin/muse");
    set_executable("Antigravity","agy",QDir::homePath()+"/.local/bin/agy");
    set_executable("Grok","grok",QDir::homePath()+"/.grok/bin/grok");
#endif

    if(!agent_entries["Claude"].executable.isEmpty())
    {
        // claude has no equivalent of "codex debug models" to query live, so use its known model aliases
        static const QStringList claude_models{"sonnet","fable","opus","haiku"};
        update_agent_models("Claude",claude_models,false);
        ai_log("Claude models: "+claude_models.join(", "));
    }
}
void AIAgent::refresh_agent_models(const QString& provider) // Claude's list is fixed (refresh_agent_executables()); Ollama models are refreshed separately, for every agent that can use them
{
    if(provider == "Claude")
        return;
    auto path = agent_entries[provider].executable;
    if(path.isEmpty())
        return;

    auto* process = new QProcess(this);
    process->setProcessEnvironment(agent_environment(provider));
    connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),process,&QObject::deleteLater);
    auto set_models = [=](const QJsonArray& list,const QString& key)
    {
        QStringList models;
        for(const auto& value : list)
            if(auto model = value.toObject()[key].toString();!model.isEmpty())
                models << model;
        update_agent_models(provider,models,false);
        ai_log(provider+" models: "+models.join(", "));
    };

    if(provider == "Codex")
    {
        connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
                this,[=](int exit_code,QProcess::ExitStatus exit_status)
        {
            auto doc = QJsonDocument::fromJson(process->readAllStandardOutput());
            // a failed/timed-out/unrecognized query says nothing about the account: keep the last valid list
            if(exit_code || exit_status != QProcess::NormalExit || !(doc.isArray() || doc.object().value("models").isArray()))
                return;
            QStringList models;
            for(const auto& value : doc.isArray() ? doc.array() : doc.object()["models"].toArray())
            {
                auto object = value.toObject();
                auto model = object["slug"].toString();
                if(model.isEmpty()) model = object["model"].toString();
                if(model.isEmpty()) model = object["id"].toString();
                if(!model.isEmpty()) models << model;
            }
            update_agent_models("Codex",models,false);
        });
        start_process(*process,path,{"debug","models"});
    }
    else if(provider == "Antigravity")
    {
        connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),this,[=]
        {
            if(process->exitStatus() == QProcess::NormalExit && process->exitCode() == 0)
                set_models(QJsonDocument::fromJson(process->readAllStandardOutput()).
                           object()["command"].toObject()["data"].toObject()["models"].toArray(),"id");
        });
        connect(process,&QProcess::started,process,&QProcess::closeWriteChannel);
        start_process(*process,path,{"--output-format","json","models"});
    }
    else if(provider == "Muse")
    {
        connect(process,&QProcess::readyReadStandardOutput,this,[=]
        {
            while(process->canReadLine())
            {
                auto msg = next_json_line(process);
                auto id = msg["id"].toString();
                if(id == "initialize")
                {
                    if(msg.contains("error"))
                        return kill_process_tree(process);
                    process->write(json_line({{"jsonrpc","2.0"},{"method","initialized"}}));
                    process->write(json_line({{"jsonrpc","2.0"},{"id","model_list"},{"method","model/list"},{"params",QJsonObject()}}));
                }
                else if(id == "model_list")
                {
                    if(!msg.contains("error"))
                        set_models(msg["result"].toObject()["models"].toArray(),"modelId");
                    process->closeWriteChannel();
                }
            }
        });
        connect(process,&QProcess::started,process,[=]{process->write(muse_initialize());});
        start_process(*process,path,{"serve"});
    }
    else // Grok: ACP initialize returns _meta.modelState.availableModels without a session or sign-in
    {
        connect(process,&QProcess::readyReadStandardOutput,this,[=]
        {
            while(process->canReadLine())
            {
                auto msg = next_json_line(process);
                if(msg["id"].toString() != "initialize")
                    continue;
                // a failed or unrecognized reply keeps the last valid list
                if(auto list = msg["result"].toObject()["_meta"].toObject()["modelState"].toObject().value("availableModels");list.isArray())
                    set_models(list.toArray(),"modelId");
                return kill_process_tree(process); // one reply is all this probe needs
            }
        });
        connect(process,&QProcess::started,process,[=]{process->write(grok_initialize());});
        start_process(*process,path,{"agent","stdio"});
    }
    QTimer::singleShot(provider == "Codex" ? 5000 : provider == "Antigravity" ? 10000 : 15000,process,[process]{kill_process_tree(process);});
}
void AIAgent::refresh_ollama_models()
{
    // one discovery feeds every agent that can route to Ollama: Claude (Anthropic API) and Codex (Responses API)
    auto set_models = [this](const QStringList& models)
    {
        for(const auto& agent : {QString("Claude"),QString("Codex")})
            if(!agent_entries[agent].executable.isEmpty())
                update_agent_models(agent,models,true);
    };

    auto ollama = ai_ollama_url(settings);
    if(!ollama.second)
        return set_models({});

    auto url = ollama.first;
    url.setPath("/api/tags");

    auto* network = new QNetworkAccessManager(this);
    network->setProxy(QNetworkProxy::NoProxy);
    QNetworkRequest request(url);
    request.setTransferTimeout(10000);
    auto* reply = network->get(request);

    connect(reply,&QNetworkReply::finished,this,
            [=]
            {
                QStringList models;
                auto list = QJsonDocument::fromJson(reply->readAll()).object().value("models");
                bool okay = reply->error() == QNetworkReply::NoError && list.isArray(); // the list means "models on the current server": a failed or non-Ollama reply clears it
                for(const auto& value : list.toArray())
                    if(auto name = value.toObject()["name"].toString();!name.isEmpty())
                        models << name;
                ai_log("Ollama "+url.toString()+" "+ (okay ? "connected" : reply->error() ? reply->errorString() : QString("is not an Ollama server")));
                if(ai_ollama_url(settings).first == ollama.first) // drop a stale reply after the host changed
                    set_models(okay ? models : QStringList());
                reply->deleteLater();
                network->deleteLater();
            });
}
void AIAgent::add_ai_history(ai_info& info,const QString& type,const QString& text)
{
    show_ai_project(info,info.record_history(QJsonObject{{"type",type},{"text",text}}));
}

static ai_agent_status check_agent_status(const QString& provider,const QString& executable,QString& info)
{
    info.clear();
    if(!local_agents.contains(provider))
        return ai_agent_status::Error;
    if(executable.isEmpty())
        return ai_agent_status::NotInstalled;

    QProcess process;
    process.setProcessEnvironment(agent_environment(provider));
    start_process(process,executable,provider == "Grok" ?        QStringList{"agent","stdio"} :
                                     provider == "Antigravity" ? QStringList{"--output-format","json","models"} :
                                     provider == "Muse" ?        QStringList{"serve"} :
                                     provider == "Codex" ?       QStringList{"login","status"} : QStringList{"auth","status"});
    if(!process.waitForStarted(3000))
        return kill_process_tree(&process),ai_agent_status::Error;

    if(provider == "Grok") // ACP initialize reports the agent's own credential choice: _meta.defaultAuthMethodId, null when none is usable
    {
        process.write(grok_initialize());
        QJsonObject reply;
        for(auto deadline = QDateTime::currentMSecsSinceEpoch()+10000;reply.isEmpty() &&
            (process.state() != QProcess::NotRunning || process.canReadLine()) && QDateTime::currentMSecsSinceEpoch() < deadline;)
        {
            if(!process.canReadLine())
                process.waitForReadyRead(int(deadline-QDateTime::currentMSecsSinceEpoch()));
            while(process.canReadLine() && reply.isEmpty())
                if(auto msg = QJsonDocument::fromJson(process.readLine()).object();msg["id"].toString() == "initialize")
                    reply = msg;
        }
        process.closeWriteChannel();
        if(!process.waitForFinished(1000))
            kill_process_tree(&process);
        if(!reply.contains("result"))
            return ai_agent_status::Error;
        auto method = reply["result"].toObject()["_meta"].toObject()["defaultAuthMethodId"].toString();
        if(method.isEmpty())
            return ai_agent_status::SignInRequired;
        info = method == "xai.api_key" ? "API key" : "Signed in";
        return ai_agent_status::Ready;
    }

    if(provider == "Antigravity")
    {
        process.closeWriteChannel();
        // agy started, so a failed or stalled "models" means no usable credential (after /logout it waits for a browser sign-in);
        // Sign In then opens agy in a terminal, which shows any other error itself
        if(!process.waitForFinished(10000) || process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
            return kill_process_tree(&process),ai_agent_status::SignInRequired;
        info = "Signed in";
        return ai_agent_status::Ready;
    }

    if(provider == "Muse")
    {
        auto deadline = QDateTime::currentMSecsSinceEpoch()+5000;
        auto read_response = [&](const QString& id)
        {
            while(process.state() != QProcess::NotRunning &&
                  QDateTime::currentMSecsSinceEpoch() < deadline)
            {
                if(!process.canReadLine())
                    process.waitForReadyRead(int(deadline-QDateTime::currentMSecsSinceEpoch()));
                while(process.canReadLine())
                {
                    auto msg = QJsonDocument::fromJson(process.readLine()).object();
                    if(msg["id"].toString() == id)
                        return msg;
                }
            }
            return QJsonObject();
        };

        process.write(muse_initialize());
        auto initialized = read_response("initialize");
        auto status = ai_agent_status::Error;
        if(initialized["result"].toObject()["experimentalApi"].toBool())
        {
            process.write(json_line({{"jsonrpc","2.0"},{"method","initialized"}}));
            process.write(json_line({{"jsonrpc","2.0"},{"id","account_read"},{"method","account/read"}}));
            auto account = read_response("account_read")["result"].toObject();
            if(!account.isEmpty())
            {
                auto state = account["state"].toString();
                bool credential_required = account["credentialRequired"].toBool();
                if(credential_required && state == "loggedOut")
                    status = ai_agent_status::SignInRequired;
                else
                {
                    info = account["label"].toString().trimmed();
                    if(info.isEmpty())
                        info = state == "apiKey" ? "API key" :
                               state == "envKey" ? "Environment key" :
                               state == "accountLogin" ? "Signed in" :
                               credential_required ? "Signed in" : "Ready";
                    status = ai_agent_status::Ready;
                }
            }
        }
        process.closeWriteChannel();
        if(!process.waitForFinished(1000))
            kill_process_tree(&process);
        return status;
    }

    if(!process.waitForFinished(10000))
        return kill_process_tree(&process),ai_agent_status::Error; // the stack QProcess destructor would kill only the .cmd wrapper
    if(process.exitStatus() != QProcess::NormalExit)
        return ai_agent_status::Error;
    if(provider == "Codex")
    {
        if(process.exitCode() != 0)
            return QString::fromUtf8(process.readAllStandardError()).contains(
                       "Not logged in",Qt::CaseInsensitive) ?
                       ai_agent_status::SignInRequired : ai_agent_status::Error;
        // codex login status has no --json/structured output (email/plan aren't exposed), only this
        // free-text auth-method line -- see https://github.com/openai/codex/issues/19866
        auto output = QString::fromUtf8(process.readAllStandardOutput());
        info = output.contains("API key",Qt::CaseInsensitive) ? "API key" :
               output.contains("ChatGPT",Qt::CaseInsensitive) ? "ChatGPT" :
               output.contains("Agent Identity",Qt::CaseInsensitive) ? "Agent Identity" : "Signed in";
        return ai_agent_status::Ready;
    }
    if(process.exitCode() != 0)
        return ai_agent_status::Error;
    auto object = QJsonDocument::fromJson(process.readAllStandardOutput()).object(); // unparsable or not an object: empty
    if(!object["loggedIn"].isBool())
        return ai_agent_status::Error;
    if(!object["loggedIn"].toBool())
        return ai_agent_status::SignInRequired;
    auto email = object["email"].toString();
    if(!email.isEmpty())
    {
        auto tier = object["subscriptionType"].toString();
        info = tier.isEmpty() ? email : email+" · "+tier.left(1).toUpper()+tier.mid(1);
    }
    else
    {
        auto api_provider = object["apiProvider"].toString();
        info = api_provider.isEmpty() ? "API key" : "API key · "+api_provider;
    }
    return ai_agent_status::Ready;
}

void AIAgent::refresh_agent_status(const QString& provider)
{
    auto check = [this](const QString& provider)
    {
        auto& entry = agent_entries[provider];
        auto check_id = ++entry.status_check_id;
        entry.status_info.clear();
        entry.status = entry.executable.isEmpty() ? ai_agent_status::NotInstalled : ai_agent_status::Checking;
        emit agent_status_changed(provider);
        if(entry.executable.isEmpty())
            return;
        auto executable = entry.executable;
        struct status_result
        {
            ai_agent_status status = ai_agent_status::Error;
            QString info;
        };
        auto result = QSharedPointer<status_result>::create();
        auto* worker = QThread::create([provider,executable,result]
        {
            result->status = check_agent_status(provider,executable,result->info);
        });
        connect(worker,&QThread::finished,this,[this,provider,executable,check_id,result]
        {
            auto& entry = agent_entries[provider];
            if(entry.status_check_id != check_id || entry.executable != executable)
                return;
            entry.status = result->status;
            entry.status_info = result->info;
            emit agent_status_changed(provider);
        });
        connect(worker,&QThread::finished,worker,&QObject::deleteLater);
        worker->start();
    };

    for(const auto& each : provider.isEmpty() ? local_agents : QStringList{provider})
        check(each);
}

bool AIAgent::run_agent_login(const QString& provider)
{
    if(!local_agents.contains(provider))
        return false;
    const auto& executable = agent_entries[provider].executable;
    if(executable.isEmpty())
        return false;

    // agy reuses a stored credential silently; its only sign-out is the interactive /logout
    if(provider == "Antigravity" && agent_entries[provider].status == ai_agent_status::Ready)
    {
        QMessageBox::information(this,"AI Agent","Antigravity is currently signed in. In the terminal, run agy, "
                                 "and type /logout, and come back to sign in again here.");
        return false;
    }

    // every agent but Muse signs in through its own CLI in a visible terminal; Antigravity's CLI starts its sign-in on launch
    auto args = provider == "Claude" ? QStringList{"auth","login"} :
                provider == "Antigravity" ? QStringList() : QStringList{"login"};
    QProcess terminal,muse_login; // muse login prints a device-code URL instead of opening the browser
    bool started = false;
    if(provider == "Muse")
    {
        muse_login.setProcessEnvironment(agent_environment(provider)); // Muse's file credential backend must match status/chat
        muse_login.setProcessChannelMode(QProcess::MergedChannels);
        start_process(muse_login,executable,args);
        if(!(started = muse_login.waitForStarted(3000)))
            kill_process_tree(&muse_login);
    }
    else
    {
        terminal.setProcessEnvironment(agent_environment(provider));
        terminal.setWorkingDirectory(ui->ai_work_dir->text());
        auto start_terminal = [&](const QString& program,const QStringList& terminal_args)
        {
            terminal.setProgram(program);
            terminal.setArguments(terminal_args);
            return terminal.startDetached();
        };
#ifdef Q_OS_WIN
        started = start_terminal(qEnvironmentVariable("ComSpec","cmd.exe"),
                                 QStringList{"/k",QDir::toNativeSeparators(executable)}+args);
#elif defined(Q_OS_MACOS)
        auto command = "'"+executable+"' "+args.join(" ");
        command.replace("\\","\\\\").replace("\"","\\\"");
        started = start_terminal("osascript",{"-e","tell application \"Terminal\" to do script \""+command+"\""});
#else
        for(const auto& name : {QString("x-terminal-emulator"),QString("gnome-terminal"),
                                QString("konsole"),QString("xterm")})
            if(auto program = QStandardPaths::findExecutable(name);!program.isEmpty() &&
               (started = start_terminal(program,QStringList{name == "gnome-terminal" ? "--" : "-e",executable}+args)))
                break;
#endif
    }
    if(!started)
    {
        QMessageBox::warning(this,"AI Agent","Cannot start "+provider+" sign-in.");
        return false;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(provider+" Sign In");
    QVBoxLayout layout(&dialog);
    QLabel status("Complete sign-in in the "+provider+" terminal/browser, then click Done.");
    status.setWordWrap(true);
    status.setFixedWidth(420);
    layout.addWidget(&status);
    QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    auto* done = buttons.addButton("Done",QDialogButtonBox::AcceptRole);
    layout.addWidget(&buttons);
    connect(done,&QPushButton::clicked,&dialog,[&]
    {
        QString info;
        auto agent_status = check_agent_status(provider,executable,info);
        if(agent_status == ai_agent_status::Ready)
            dialog.accept();
        else
            status.setText(agent_status == ai_agent_status::SignInRequired ?
                "Sign-in not detected yet. Complete sign-in, then click Done again." :
                "Could not verify "+provider+" sign-in.");
    });
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
    connect(&muse_login,&QProcess::readyRead,&dialog,[&]
    {
        auto text = QString::fromUtf8(muse_login.readAll());
        if(auto match = QRegularExpression("https?://\\S+").match(text);match.hasMatch())
        {
            QDesktopServices::openUrl(QUrl(match.captured()));
            status.setText(text.trimmed()); // the code the browser page must match
        }
    });
    bool accepted = dialog.exec() == QDialog::Accepted;
    kill_process_tree(&muse_login); // no-op when muse login already finished or never started
    if(!accepted)
        return false;
    refresh_agent_models(provider);
    return true;
}

void AIAgent::update_agent_status_label()
{
    static const QString dot = QString(" ")+QChar(0x00B7)+" "; // middle dot separator
    auto* info = selected_info();
    update_send_button(); // the send button depends on the same selected chat
    ui->ai_agent_status->setVisible(!info || info->provider != "AgentServer"); // a log/routing record has no agent/model to show
    if(info && info->provider == "AgentServer")
        return;
    if(info && info->provider == "Web")
        return ui->ai_agent_status->setText("Web");
    // a local chat's own model, or with nothing selected the default the next New Chat starts with
    auto model_name = info ? info->model_settings["model"].toString() : current_model_name;
    auto model_info = info ? info->model_settings["info"].toObject() : current_model_info;
    auto text = (info ? info->provider : current_agent)+dot+(model_name.isEmpty() ? QString("default") : model_name);
    if(model_info.contains("provider"))
        text += dot+"Ollama@"+(model_info.contains("url") ? QUrl(model_info["url"].toString()).host() :
                                                            ai_ollama_url(settings).first.host());
    ui->ai_agent_status->setText(text);
}

ai_info* AIAgent::selected_info() const
{
    auto* item = ui->ai_project_list->currentItem();
    // find(), not ai_infos[id], which would create a blank chat for a stale id
    return item ? ai_info::find(item->data(Qt::UserRole).toString()) : nullptr;
}

bool AIAgent::web_connected(const ai_info& info) const
{
    return info.sessions == web_session_id && !google_file_id.isEmpty();
}

AIAgent::send_action AIAgent::current_send_action() const
{
    auto* info = selected_info();
    bool has_input = !ui->ai_chat_input->toPlainText().trimmed().isEmpty();
    // New chats start only from the New Chat button; AgentServer is a log/routing record with no local subprocess.
    if(!info || info->provider == "AgentServer")
        return send_action::Disabled;
    if(info->provider == "Web")
        return web_connected(*info) ? send_action::Stop : send_action::Resume;
    if(!info->processes) // never launched (or a prior attempt cleanly ended): a fresh launch, always a real send
        return has_input ? send_action::Send : send_action::Disabled;
    if(!has_input) // a live local process can always be stopped: a reply can arrive (WaitingUser) before its turn ends
        return send_action::Stop;
    // a message can only be written to a running process with an established session; while starting or exiting, Send waits
    return info->processes->state() == QProcess::Running && info->status != session_status::New ?
           send_action::Send : send_action::Disabled;
}

void AIAgent::update_send_button()
{
    auto action = current_send_action();
    ui->ai_send_message->setEnabled(action != send_action::Disabled);
    ui->ai_send_message->setText(
        action == send_action::Stop ? "Stop" :
        action == send_action::Resume ? "Resume" : "Send");
}

void AIAgent::google_token_post(QList<QPair<QString,QString>> form,std::function<void(QString)> done)
{
    form.emplaceBack("client_id",GOOGLE_CLIENT_ID);
    form.emplaceBack("client_secret",GOOGLE_CLIENT_SECRET); // bundled in a desktop binary, so not confidential: PKCE + state protect the flow
    QByteArray body;
    for(const auto& [key,value] : form)
        body += key.toUtf8()+"="+QUrl::toPercentEncoding(value)+"&";
    QNetworkRequest request(QUrl("https://oauth2.googleapis.com/token"));
    request.setHeader(QNetworkRequest::ContentTypeHeader,"application/x-www-form-urlencoded");
    auto* reply = web_manager.post(request,body);
    connect(reply,&QNetworkReply::finished,this,[this,reply,done]
    {
        reply->deleteLater();
        auto json = QJsonDocument::fromJson(reply->readAll()).object();
        if(!json.contains("access_token"))
        {
            if(json["error"].toString() == "invalid_grant") // revoked: sign out; a network or server error keeps the saved token
                settings.setValue("ai/google_refresh_token",google_refresh_token = QString());
            return done(json["error"].toString(reply->errorString()));
        }
        google_access_token = json["access_token"].toString();
        google_token_expiry = QDateTime::currentDateTimeUtc().addSecs(json["expires_in"].toInt());
        if(json.contains("refresh_token")) // returned with consent; a refresh keeps the saved one
            settings.setValue("ai/google_refresh_token",google_refresh_token = json["refresh_token"].toString());
        done({});
    });
}
void AIAgent::with_google_token(std::function<void(QString)> call)
{
    if(QDateTime::currentDateTimeUtc().secsTo(google_token_expiry) > 60)
        return call(google_access_token);
    if(google_refresh_token.isEmpty()) // signed out: never hand on an expired token
    {
        google_error = "signed out of Google";
        return call({});
    }
    google_token_post({{"grant_type","refresh_token"},{"refresh_token",google_refresh_token}},[this,call](QString error)
    {
        if(!error.isEmpty())
            google_error = "Google sign-in failed: "+error;
        call(error.isEmpty() ? google_access_token : QString());
    });
}
void AIAgent::google_api(const QByteArray& verb,const QString& url,const QJsonObject& body,std::function<void(QJsonObject)> done)
{
    auto failed = [this,done]
    {
        if(!google_file_id.isEmpty()) // a live Web chat shows it; polling retries until the idle limit stops it
            set_ai_status(web_session_id,session_status::Failed,"Web: "+google_error+"; retrying.");
        done({});
    };
    with_google_token([this,verb,url,body,done,failed](QString token)
    {
        if(token.isEmpty()) // signed out: fail without a doomed request
            return failed();
        QNetworkRequest request{QUrl(url)};
        request.setRawHeader("Authorization","Bearer "+token.toUtf8());
        request.setHeader(QNetworkRequest::ContentTypeHeader,"application/json");
        auto* reply = web_manager.sendCustomRequest(request,verb,body.isEmpty() ? QByteArray() : QJsonDocument(body).toJson(QJsonDocument::Compact));
        connect(reply,&QNetworkReply::finished,this,[this,reply,done,failed]
        {
            reply->deleteLater();
            auto json = QJsonDocument::fromJson(reply->readAll()).object();
            if(reply->error() == QNetworkReply::NoError)
                return done(json);
            google_error = json["error"].toObject()["message"].toString(reply->errorString()); // Google's own reason when it gives one
            failed();
        });
    });
}
void AIAgent::write_google_doc(const QJsonObject& doc,const QJsonObject& message,std::function<void(bool)> done)
{
    auto content = doc["body"].toObject()["content"].toArray();
    auto end = content.isEmpty() ? 0 : content.last().toObject()["endIndex"].toInt();
    QJsonArray requests;
    if(end > 2) // everything except the final newline every Doc keeps
        requests.append(QJsonObject{{"deleteContentRange",QJsonObject{{"range",QJsonObject{{"startIndex",1},{"endIndex",end-1}}}}}});
    requests.append(QJsonObject{{"insertText",QJsonObject{{"location",QJsonObject{{"index",1}}},
        {"text",QString(QJsonDocument(message).toJson(QJsonDocument::Compact))}}}});
    QJsonObject body{{"requests",requests}};
    if(doc.contains("revisionId")) // fails instead of overwriting a message written after our read
        body["writeControl"] = QJsonObject{{"requiredRevisionId",doc["revisionId"]}};
    google_api("POST","https://docs.googleapis.com/v1/documents/"+google_file_id+":batchUpdate",body,
               [done](QJsonObject reply){done(!reply.isEmpty());});
}
void AIAgent::create_web_session()
{
    if(google_refresh_token.isEmpty() && !sign_in_google())
        return;
    if(google_folder_id.isEmpty()) // resolved once (found or created), then reused by its ID
        return google_api("GET","https://www.googleapis.com/drive/v3/files?fields=files(id)&q="+QString::fromLatin1(QUrl::toPercentEncoding(
                          "name='DSI Studio AI' and mimeType='application/vnd.google-apps.folder' and trashed=false")),{},[this](QJsonObject found)
        {
            auto use = [this](QJsonObject folder)
            {
                if(folder["id"].toString().isEmpty())
                    return void(QMessageBox::warning(this,"AI Agent","Cannot create the DSI Studio AI folder in Google Drive."));
                settings.setValue("ai/google_folder_id",google_folder_id = folder["id"].toString());
                create_web_session();
            };
            if(!found.contains("files")) // a failed search is not "no folder": never create a duplicate
                return void(QMessageBox::warning(this,"AI Agent","Cannot reach Google Drive."));
            if(auto files = found["files"].toArray();!files.isEmpty())
                return use(files[0].toObject());
            google_api("POST","https://www.googleapis.com/drive/v3/files?fields=id",
                       {{"name","DSI Studio AI"},{"mimeType","application/vnd.google-apps.folder"}},use);
        });
    auto session = QUuid::createUuid().toString(QUuid::WithoutBraces);
    google_api("POST","https://www.googleapis.com/drive/v3/files?fields=id",
               {{"name","DSI Studio "+session},{"mimeType","application/vnd.google-apps.document"},{"parents",QJsonArray{google_folder_id}}},
               [this,session](QJsonObject file)
    {
        if(file["id"].toString().isEmpty())
        {
            settings.setValue("ai/google_folder_id",google_folder_id = QString()); // the folder may have been deleted: recreate it next time
            return void(QMessageBox::warning(this,"AI Agent","Cannot create the Web session."));
        }
        auto* info = ai_info::create(session,"Web","Web"); // agent agnostic: any web-based agent can join
        info->model_settings["google_file_id"] = file["id"].toString();
        add_ai_history(*info,"activity","Web session started.");
        ui->ai_project_list->setCurrentItem(info->project_items);
        start_web(*info); // the first poll writes "ready" into the empty Doc
        info->save_config(); // after start_web(), so the chat is saved as established
    });
}
void AIAgent::start_web(ai_info& info)
{
    stop_web();
    if(info.sessions != web_session_id) // a pending result belongs to its own session only
    {
        web_session_id = info.sessions;
        web_last_id = 0; // a claimed request is always overwritten by "processing", so it is never re-run
        web_pending_result = QJsonObject();
    }
    google_file_id = info.model_settings["google_file_id"].toString();
    google_error = "the session Doc was never read"; // the first good read clears it and shows the chat connected
    set_ai_status(info.sessions,session_status::WaitingUser,"Web: reading the session Doc.");
    web_idle.start();
    web_timer.start(0);
    QApplication::clipboard()->setText( // on every start and Resume: the agent may be a new chat
        "Connect to DSI Studio. First read the public GitHub file "
        "frankyeh/DSI-Studio-AI/DSI_STUDIO_AI_SKILL_WEB.md and follow it. "
        "Session document: https://docs.google.com/document/d/"+google_file_id+"/edit");
    QMessageBox::information(this,"Web","The connection prompt is copied. Paste it into any web-based AI agent and send.");
}
void AIAgent::stop_web(const QString& message)
{
    if(google_file_id.isEmpty())
        return;
    google_file_id.clear(); // callbacks still in flight see this and stop
    web_timer.stop();
    set_ai_status(web_session_id,session_status::Completed,message);
}
QString AIAgent::google_doc_url() const // only what write_google_doc() and the mailbox need
{
    return "https://docs.googleapis.com/v1/documents/"+google_file_id+
           "?fields=revisionId,body(content(endIndex,paragraph(elements(textRun(content)))))";
}
static QString web_text(const QJsonObject& doc) // the mailbox message: the Doc body as one trimmed string
{
    QString text;
    for(const auto& block : doc["body"].toObject()["content"].toArray())
        for(const auto& element : block.toObject()["paragraph"].toObject()["elements"].toArray())
            text += element.toObject()["textRun"].toObject()["content"].toString();
    return text.trimmed();
}
void AIAgent::poll_web()
{
    if(google_file_id.isEmpty())
        return;
    if(!web_pending_result.isEmpty())
        return publish_web_result(); // a previous result write failed; retry it, never re-execute
    if(web_idle.hasExpired(180000)) // also bounds failing reads
        return stop_web(google_error.isEmpty() ? QString("Web stopped after 3 minutes without a request; press Resume to continue.") :
                        "Web stopped: "+google_error+"; press Resume to retry.");
    // the body is one tiny JSON message, so it is read directly (Drive file.version proved an unreliable doorbell)
    google_api("GET",google_doc_url(),{},[this,file = google_file_id](QJsonObject doc)
    {
        if(file != google_file_id) // stopped or switched while in flight
            return;
        if(doc.isEmpty()) // failed, and shown by google_api()
            return web_timer.start(5000);
        if(!google_error.isEmpty()) // the first good read after a start, Resume, or failure
        {
            google_error.clear();
            set_ai_status(web_session_id,session_status::WaitingUser,"Web connected; waiting for a request.");
        }
        auto text = web_text(doc);
        if(text.isEmpty()) // a new Doc, or a "ready" write that failed: retried until it lands
            return write_google_doc(doc,{{"dsi_bridge",true},{"session",web_session_id},{"from","dsi"},{"state","ready"}},
                                    [this,file](bool written){if(file == google_file_id) web_timer.start(written ? 500 : 5000);});
        QJsonParseError parse_error;
        auto request = QJsonDocument::fromJson(text.toUtf8(),&parse_error).object();
        auto id = request["id"].toInteger();
        if(request["session"].toString() == web_session_id && request["from"].toString() == "dsi" &&
           request["state"].toString() == "processing" && id > web_last_id)
        {
            // claimed but never finished (DSI Studio stopped mid-request): report it instead of re-running it
            web_last_id = id;
            web_pending_result = QJsonObject{{"dsi_bridge",true},{"session",web_session_id},{"id",id},{"from","dsi"},{"state","error"},
                {"response",QJsonObject{{"status","error"},{"error","outcome unknown: DSI Studio stopped while running this request; check with list_window"}}}};
            return publish_web_result();
        }
        if(request["from"].toString() == "dsi")
            return web_timer.start(500); // our own write
        // anything else is a request: a malformed one is answered, or the agent waits forever
        auto error = parse_error.error != QJsonParseError::NoError ?
                        "invalid JSON (" + parse_error.errorString() + " at offset " + QString::number(parse_error.offset) + ")" :
                     request["session"].toString() != web_session_id ? QString("wrong session") :
                     request["from"].toString() != "agent" || request["state"].toString() != "request" ?
                        QString(R"(needs "from":"agent" and "state":"request")") :
                     id <= web_last_id ? "id must be greater than " + QString::number(web_last_id) : QString();
        if(!error.isEmpty()) // nothing ran, so written with this read's revision (never over a newer message) and re-checked if it fails
            return write_google_doc(doc,{{"dsi_bridge",true},{"session",web_session_id},{"id",id > web_last_id ? id : web_last_id+1},
                {"from","dsi"},{"state","error"},{"response",QJsonObject{{"status","error"},
                {"error",error + "; nothing was run. Resend the whole request as one line of compact JSON"}}}},
                [this,file](bool written){if(file == google_file_id) web_timer.start(written ? 500 : 5000);});
        // claim with the revision we read: a crash after it never re-runs the command
        write_google_doc(doc,{{"dsi_bridge",true},{"session",web_session_id},{"id",id},{"from","dsi"},{"state","processing"}},
                         [this,file,id,request](bool claimed)
        {
            if(file != google_file_id)
                return;
            if(!claimed) // not claimed, so not executed: read again
                return web_timer.start(5000);
            web_last_id = id;
            auto forwarded = request;
            for(auto key : {"dsi_bridge","id","from","state"})
                forwarded.remove(key);
            QByteArray reply_bytes;
            ai_request(QJsonDocument(forwarded).toJson(QJsonDocument::Compact),reply_bytes);
            auto response = QJsonDocument::fromJson(reply_bytes).object();
            web_pending_result = QJsonObject{{"dsi_bridge",true},{"session",web_session_id},{"id",id},{"from","dsi"},
                {"state",response["status"].toString() == "error" ? "error" : "done"},{"response",response}};
            publish_web_result();
        });
    });
}
void AIAgent::publish_web_result()
{
    google_api("GET",google_doc_url(),{},[this,file = google_file_id](QJsonObject doc)
    {
        if(file != google_file_id)
            return;
        if(doc.isEmpty()) // without the current body the write would append instead of replace
            return web_timer.start(5000);
        auto current = QJsonDocument::fromJson(web_text(doc).toUtf8()).object();
        if(current["from"].toString() != "dsi" || current["state"].toString() != "processing" ||
           current["id"].toInteger() != web_pending_result["id"].toInteger())
        {
            // only our own "processing" is replaced: anything else means the result already landed (its acknowledgement
            // was lost) or the agent has moved on, and a newer request must never be overwritten
            web_pending_result = QJsonObject();
            web_idle.start();
            return web_timer.start(500);
        }
        write_google_doc(doc,web_pending_result,[this,file](bool published)
        {
            if(file != google_file_id)
                return;
            if(published)
            {
                web_pending_result = QJsonObject();
                web_idle.start(); // the 3-minute idle limit counts from the last result
            }
            web_timer.start(published ? 500 : 5000);
        });
    });
}
bool AIAgent::sign_in_google()
{
    auto random_text = [](int bytes)
    {
        QByteArray data(bytes,0);
        for(auto& c : data)
            c = char(QRandomGenerator::system()->bounded(256));
        return QString(data.toBase64(QByteArray::Base64UrlEncoding|QByteArray::OmitTrailingEquals));
    };
    auto state = random_text(16),verifier = random_text(48);
    QTcpServer server; // lives only for this sign-in
    server.listen(QHostAddress::LocalHost);
    auto redirect = QString("http://127.0.0.1:%1/").arg(server.serverPort());
    QUrlQuery query;
    for(const auto& [key,value] : QList<QPair<QString,QString>>{
            {"client_id",GOOGLE_CLIENT_ID},{"redirect_uri",redirect},{"response_type","code"},
            {"scope","https://www.googleapis.com/auth/drive.file"},{"state",state},{"code_challenge_method","S256"},
            {"code_challenge",QCryptographicHash::hash(verifier.toUtf8(),QCryptographicHash::Sha256).toBase64(
                QByteArray::Base64UrlEncoding|QByteArray::OmitTrailingEquals)},
            {"access_type","offline"},{"prompt","select_account consent"}}) // consent on every sign-in: Google returns a refresh token (for the chosen account) only with consent
        query.addQueryItem(key,QUrl::toPercentEncoding(value));
    QUrl url("https://accounts.google.com/o/oauth2/v2/auth");
    url.setQuery(query);

    QMessageBox dialog(QMessageBox::NoIcon,"Google Sign In","Complete Google sign-in in your browser.",QMessageBox::Cancel,this);
    QString error;
    connect(&server,&QTcpServer::newConnection,&dialog,[&]
    {
        auto* socket = server.nextPendingConnection();
        connect(socket,&QTcpSocket::readyRead,&dialog,[&,socket]
        {
            QUrlQuery params(QUrl(QString::fromLatin1(socket->readLine()).section(' ',1,1)).query()); // "GET /?state=..&code=.. HTTP/1.1"
            socket->write("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nYou may close this window.");
            socket->disconnectFromHost();
            if(params.queryItemValue("state",QUrl::FullyDecoded) != state) // also skips the favicon request
                return;
            google_token_post({{"grant_type","authorization_code"},{"code",params.queryItemValue("code",QUrl::FullyDecoded)},
                               {"code_verifier",verifier},{"redirect_uri",redirect}},
                              [&,alive = QPointer<QMessageBox>(&dialog)](QString token_error)
            {
                if(alive) // a reply after Cancel must not touch these locals
                    error = token_error,dialog.done(0);
            });
        });
    });
    QDesktopServices::openUrl(url);
    if(dialog.exec() == QMessageBox::Cancel)
        return false;
    if(!error.isEmpty())
        return QMessageBox::warning(this,"AI Agent","Google sign-in failed: "+error),false;
    settings.setValue("ai/google_folder_id",google_folder_id = QString()); // the account may have changed: resolve its folder again
    return true;
}
bool AIAgent::run_new_chat_dialog(const QString& title,const QString& accept_text)
{
    QDialog dialog(this);
    dialog.setWindowTitle(title);
    dialog.setMinimumWidth(440);
    dialog.setStyleSheet(ai_dialog_style());
    QFormLayout layout(&dialog);
    layout.setSpacing(10);
    layout.setContentsMargins(20,18,20,16);

    QComboBox agent;
    for(auto name : {"Codex","Claude","Muse","Antigravity","Grok","Web"})
        agent.addItem(name,name);
    auto update_agent = [&](const QString& provider)
    {
        const auto& entry = agent_entries[provider];
        bool is_ready = !entry.executable.isEmpty() && (entry.status == ai_agent_status::Ready ||
                        std::any_of(entry.profiles.begin(),entry.profiles.end(),[](const QJsonValue& profile){return profile.toObject().contains("provider");}));
        bool checking = entry.status == ai_agent_status::Unknown || entry.status == ai_agent_status::Checking;
        auto* item = static_cast<QStandardItemModel*>(agent.model())->item(agent.findData(provider));
        item->setText(is_ready ? provider :
                      provider+(checking ? " (checking...)" : " (setup required)"));
        item->setToolTip(is_ready ? QString() : checking ?
                         "Checking local agent status..." :
                         "Open Settings (⚙) to install or sign in.");
    };
    for(const auto& provider : local_agents)
        update_agent(provider);

    agent.setCurrentIndex(agent.findData(current_agent));
    layout.addRow("Agent:",&agent);

    QWidget local; // declared before its would-be children below, so it is destroyed after them
    QComboBox model; // not editable -- same as the Agent combo above, which never had the popup-visibility problem an editable combo did
    model.setMaximumHeight(model.sizeHint().height());
    auto* local_layout = new QFormLayout(&local);
    local_layout->setContentsMargins(0,0,0,0);
    local_layout->addRow("Model:",&model);
    layout.addRow(&local);

    auto update_field = [&]
    {
        auto provider = agent.currentData().toString();
        local.setVisible(provider != "Web"); // Web has no model choice; Google sign-in is asked at Start when needed
        if(provider != "Web")
            set_model_selector(model,agent_entries[provider].profiles,
                // only the agent that's actually active right now keeps its remembered model; switching to a different agent resets to that agent's own "default"
                provider == current_agent ? current_model_name : QString(),{},
                provider == current_agent ? current_model_info : QJsonObject());
        dialog.adjustSize();
    };
    update_field();
    connect(&agent,QOverload<int>::of(&QComboBox::currentIndexChanged),&dialog,[&](int){update_field();});
    QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    auto* accept = buttons.addButton(accept_text,QDialogButtonBox::AcceptRole);
    accept->setObjectName("ai_primary_button");
    layout.addRow(&buttons);
    connect(this,&AIAgent::agent_status_changed,&dialog,update_agent);
    connect(this,&AIAgent::agent_models_changed,&dialog,[&](const QString& provider)
    {
        update_agent(provider);
        if(agent.currentData().toString() == provider)
            set_model_selector(model,agent_entries[provider].profiles,model_combo_key(model),{},model.currentData().toJsonObject());
    });
    connect(accept,&QPushButton::clicked,&dialog,[&]
    {
        auto provider = agent.currentData().toString();
        if(provider == "Web" || (!agent_entries[provider].executable.isEmpty() &&
           (agent_entries[provider].status == ai_agent_status::Ready || model.currentData().toJsonObject().contains("provider"))))
            return dialog.accept();
        dialog.reject();
        on_ai_quick_settings_clicked();
    });
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);

    if(dialog.exec() != QDialog::Accepted)
        return false;
    if(agent.currentData().toString() == "Web")
        return create_web_session(),false;
    current_agent = agent.currentData().toString();
    current_model_name = model_combo_key(model);
    current_model_info = model.currentData().toJsonObject(); // the chosen entry's own profile, incl. its Ollama server
    return true;
}

void AIAgent::create_new_chat(const QString& provider)
{
    // drop any never-used placeholder left behind by an abandoned "New Chat" attempt before adding another
    for(auto it = ai_infos.begin();it != ai_infos.end();)
        if(it->second.status == session_status::New && it->second.projects.isEmpty() && !it->second.processes)
        {
            if(auto* item = it->second.project_items)
            {
                auto* taken_item = ui->ai_project_list->takeItem(ui->ai_project_list->row(item));
                QTimer::singleShot(0,this,[taken_item]{delete taken_item;});
            }
            it = ai_infos.erase(it);
        }
        else
            ++it;

    auto* info = ai_info::create(
        provider == "Muse" ? muse_uuid_v7() :
        QUuid::createUuid().toString(QUuid::WithoutBraces),provider); // status defaults to New; no "new:"/other marker on the id itself
    info->model_settings = QJsonObject{{"model",current_model_name},{"info",current_model_info}};
    set_ai_status(info->sessions,session_status::New,"Ready for a message.");
    show_ai_project(*info);
    ui->ai_project_list->setCurrentItem(info->project_items);
}

void AIAgent::on_ai_new_chat_clicked()
{
    if(!run_new_chat_dialog("New Chat","Start"))
        return;
    create_new_chat(current_agent); // selecting the new chat refreshes the status label and send button
    ui->ai_chat_input->clear();
    ui->ai_chat_input->setFocus();
}

void AIAgent::on_ai_agent_status_clicked()
{
    if(auto* info = selected_info())
    {
        if(info->provider == "Web" || info->provider == "AgentServer") // bound to its one session Doc / a log record: no agent or model to change
            return;
        if(info->processes) // a running agent keeps the model it was launched with
            return void(QMessageBox::information(this,"Change Model","Stop the agent before changing its model."));
        QDialog dialog(this);
        dialog.setWindowTitle("Change Model");
        QFormLayout layout(&dialog);
        QLabel agent_label(info->provider);
        QComboBox model;
        set_model_selector(model,agent_entries[info->provider].profiles,info->model_settings["model"].toString(),{},
                           info->model_settings["info"].toObject());
        layout.addRow("Agent:",&agent_label);
        layout.addRow("Model:",&model);
        QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
        layout.addRow(&buttons);
        connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
        connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
        if(dialog.exec() != QDialog::Accepted)
            return;

        info->model_settings["model"] = model_combo_key(model);
        info->model_settings["info"] = model.currentData().toJsonObject(); // the chosen entry's own profile, incl. its Ollama server
        info->save_config();
        update_agent_status_label();
        return;
    }

    if(run_new_chat_dialog("Change Agent/Model","Save"))
        update_agent_status_label();
}

void AIAgent::on_ai_quick_settings_clicked()
{
    QDialog dialog(this);
    dialog.setWindowTitle("AI Settings");
    dialog.setMinimumWidth(420);
    dialog.setStyleSheet(ai_dialog_style());

    auto* root = new QVBoxLayout(&dialog);
    root->setSpacing(14);
    root->setContentsMargins(20,20,20,16);

    // a settings card appended to the dialog (titled unless heading is empty); returns its layout for the section-specific controls
    auto add_card = [root](const QString& heading)
    {
        auto* card = new QFrame;
        card->setObjectName("ai_step_card");
        auto* layout = new QVBoxLayout(card);
        layout->setContentsMargins(14,12,14,12);
        layout->setSpacing(8);
        if(!heading.isEmpty())
        {
            auto* label = new QLabel(heading);
            label->setObjectName("ai_step_heading");
            layout->addWidget(label);
        }
        root->addWidget(card);
        return layout;
    };

    auto* agent_layout = add_card("Agents");

    // one row per agent: name and colored status on the left, a single action button on the right
    QHash<QString,QPair<QLabel*,QPushButton*> > agent_rows;
    auto refresh_agent_row = [this](const QString& provider,QLabel* label,QPushButton* button)
    {
        const auto& entry = agent_entries[provider];
        QString color = "#9aa0a6",text = "Not installed",action = "Install";
        switch(entry.status)
        {
        case ai_agent_status::NotInstalled:
            break;
        case ai_agent_status::Checking:
            color = "#4285f4";
            text = "Checking...";
            action = "Sign In";
            break;
        case ai_agent_status::SignInRequired:
            color = "#f9ab00";
            text = "Not signed in";
            action = "Sign In";
            break;
        case ai_agent_status::Ready:
            color = "#34a853";
            text = entry.status_info.isEmpty() ? "Ready" : "Ready · "+entry.status_info;
            action = "Sign In Again"; // switch accounts or recover stale credentials
            break;
        case ai_agent_status::Unknown:
        case ai_agent_status::Error:
            color = "#ea4335";
            text = "Status unavailable";
            action = "Check Status";
            break;
        }
        label->setText("<b>"+provider+"</b><br><span style='color:"+color+";'>&#9679;</span> "
                       "<span style='color:#5f6368;'>"+text.toHtmlEscaped()+"</span>");
        button->setText(action);
        button->setEnabled(entry.status != ai_agent_status::Checking);
    };
    auto add_row = [agent_layout]
    {
        auto* label = new QLabel;
        auto* button = new QPushButton;
        button->setMinimumWidth(110);
        auto* row = new QHBoxLayout;
        row->addWidget(label,1);
        row->addWidget(button);
        agent_layout->addLayout(row);
        return QPair<QLabel*,QPushButton*>{label,button};
    };
    for(const auto& provider : local_agents)
    {
        auto [label,button] = agent_rows[provider] = add_row();
        refresh_agent_row(provider,label,button);
        connect(button,&QPushButton::clicked,&dialog,[&,provider]
        {
            auto status = agent_entries[provider].status;
            if(status == ai_agent_status::NotInstalled)
            {
                refresh_agent_executables();
                if(agent_entries[provider].executable.isEmpty())
                    QDesktopServices::openUrl(agent_install_url(provider));
                else
                {
                    refresh_agent_status(provider);
                    refresh_agent_models(provider);
                    if(provider == "Claude" || provider == "Codex") // the agents that can use Ollama; a failed query would clear their Ollama models
                        refresh_ollama_models();
                }
            }
            else if(status == ai_agent_status::SignInRequired || status == ai_agent_status::Ready)
            {
                if(run_agent_login(provider))
                    refresh_agent_status(provider);
            }
            else if(status == ai_agent_status::Unknown || status == ai_agent_status::Error)
                refresh_agent_status(provider);
        });
    }
    // Web: the last row, signed in through Google like the local agents' own sign-in
    auto [web_label,web_button] = add_row();
    auto refresh_web_row = [this,web_label = web_label,web_button = web_button]
    {
        bool ready = !google_refresh_token.isEmpty();
        web_label->setText(QString("<b>Web</b><br><span style='color:%1;'>&#9679;</span> "
                                   "<span style='color:#5f6368;'>%2</span>").arg(ready ? "#34a853" : "#f9ab00",ready ? "Ready · Google Drive › DSI Studio AI" : "Not signed in · Google"));
        web_button->setText(ready ? "Sign In Again" : "Sign In");
    };
    refresh_web_row();
    connect(web_button,&QPushButton::clicked,&dialog,[this,refresh_web_row]
    {
        sign_in_google();
        refresh_web_row();
    });
    connect(this,&AIAgent::agent_status_changed,&dialog,
            [&,refresh_agent_row](const QString& provider)
    {
        if(auto it = agent_rows.find(provider);it != agent_rows.end())
            refresh_agent_row(provider,it->first,it->second);
    });

    auto* ollama_layout = add_card("Ollama connection");
    auto* ollama_form = new QFormLayout;
    ollama_form->setContentsMargins(0,0,0,0);
    QLineEdit host(settings.value("ai/ollama_host","localhost").toString());
    QSpinBox port;
    port.setRange(1,65535);
    port.setValue(settings.value("ai/ollama_port",11434).toInt());
    ollama_form->addRow("Host/IP:",&host);
    ollama_form->addRow("Port:",&port);
    ollama_layout->addLayout(ollama_form);

    QPushButton check_ollama("Check connection");
    QLabel ollama_status;
    ollama_status.setObjectName("ai_step_body");
    auto* ollama_button_row = new QHBoxLayout;
    ollama_button_row->addWidget(&check_ollama);
    ollama_button_row->addWidget(&ollama_status);
    ollama_button_row->addStretch();
    ollama_layout->addLayout(ollama_button_row);

    auto* ollama_network = new QNetworkAccessManager(&dialog);
    ollama_network->setProxy(QNetworkProxy::NoProxy);
    connect(&check_ollama,&QPushButton::clicked,&dialog,[&]
    {
        auto value = host.text().trimmed();
        if(value.isEmpty())
            return ollama_status.setText("Host required");
        if(!value.contains("://"))
            value.prepend("http://");

        QUrl url(value);
        url.setPort(port.value());
        url.setPath("/api/tags");
        check_ollama.setEnabled(false);
        ollama_status.setText("Checking...");

        QNetworkRequest request(url);
        request.setTransferTimeout(10000);
        auto* reply = ollama_network->get(request);
        connect(reply,&QNetworkReply::finished,&dialog,[&,reply]
        {
            check_ollama.setEnabled(true);
            auto models = QJsonDocument::fromJson(reply->readAll()).object().value("models");
            // setTransferTimeout() aborts the reply; depending on the Qt version this reports TimeoutError or OperationCanceledError
            if(reply->error() == QNetworkReply::TimeoutError || reply->error() == QNetworkReply::OperationCanceledError)
                ollama_status.setText("No response within 10 s (server asleep or port blocked?)");
            else if(reply->error() != QNetworkReply::NoError)
                ollama_status.setText("Unavailable: "+reply->errorString());
            else if(!models.isArray())
                ollama_status.setText("Reachable, but not an Ollama server");
            else if(models.toArray().isEmpty())
                ollama_status.setText("Connected · no models installed");
            else
                ollama_status.setText("Connected · "+QString::number(models.toArray().size())+" models");
            reply->deleteLater();
        });
    });

    auto* chat_layout = add_card({});
    QCheckBox history("Keep AI chat history");
    history.setChecked(settings.value("ai/keep_history",true).toBool());
    QCheckBox show_reasoning("Show reasoning");
    show_reasoning.setToolTip("Show AI reasoning messages in chat history");
    show_reasoning.setChecked(settings.value("ai/show_reasoning",false).toBool());
    chat_layout->addWidget(&history);
    chat_layout->addWidget(&show_reasoning);
    auto* debug_row = new QHBoxLayout;
    QComboBox debug;
    debug.addItems({"Disabled","Enabled (truncated)","Enabled (complete)"});
    debug.setCurrentIndex(settings.value("ai/debug",0).toInt());
    debug_row->addWidget(new QLabel("Debug mode:"));
    debug_row->addWidget(&debug,1);
    chat_layout->addLayout(debug_row);

    QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
    buttons.button(QDialogButtonBox::Save)->setObjectName("ai_primary_button");
    root->addWidget(&buttons);
    connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
    if(dialog.exec() != QDialog::Accepted)
        return;

    settings.setValue("ai/ollama_host",host.text().trimmed());
    settings.setValue("ai/ollama_port",port.value());
    settings.setValue("ai/keep_history",history.isChecked());
    bool reasoning_changed = show_reasoning.isChecked() != settings.value("ai/show_reasoning",false).toBool();
    settings.setValue("ai/show_reasoning",show_reasoning.isChecked());
    settings.setValue("ai/debug",debug.currentIndex());
    ai_debug_level = debug.currentIndex();
    if(reasoning_changed)
        if(auto* info = selected_info())
            show_ai_project(*info);

    refresh_ollama_models();
}

QString AIAgent::prepare_ai(ai_info& info)
{
    // fresh state each attempt -- a field this attempt doesn't set (e.g. launch_model_url when not using Ollama) must not carry over a stale value from the last one
    info.launch_name.clear();
    info.launch_model.clear();
    info.launch_model_url.clear();
    auto provider = info.provider;
    auto session = info.sessions; // captured by value below for every async handler -- info itself must never be captured across them (Codex can still rename/rekey the session)

    auto fail_launch = [&](QString message)
    {
        if(!message.startsWith("ERROR:"))
            message.prepend("ERROR: ");
        set_ai_status(session,info.status == session_status::New ?
                      session_status::New : session_status::Failed,message);
        add_ai_history(info,"error",message);
        info.save_config();
        return QString();
    };

    // Resolve agent
    info.launch_name = provider;
    if(agent_entries[provider].executable.isEmpty()) // stale showEvent() check -- the window may have stayed open since before an install finished, so retry once before assuming it's still missing
        refresh_agent_executables();
    auto executable = agent_entries[provider].executable;
    if(executable.isEmpty())
    {
        QDesktopServices::openUrl(agent_install_url(provider)); // same as the sidebar's Install button
        return fail_launch(info.launch_name+" is not installed. Opening the install page...");
    }

    // Resolve work directory
    auto project_dir = ui->ai_work_dir->text().trimmed();
    ui->ai_work_dir->setText(
        project_dir.isEmpty() ? main_window.work_dir() : project_dir);

    info.launch_model = info.model_settings["model"].toString().trimmed();
    if(auto model_info = info.model_settings["info"].toObject();model_info.contains("provider"))
    {
        // the chat's own Ollama server, saved with its model; chats saved before servers were recorded use the current setting
        auto [url,configured] = model_info.contains("url") ? QPair<QUrl,bool>{QUrl(model_info["url"].toString()),true} :
                                                             ai_ollama_url(settings);
        info.launch_model_url = url;
        info.launch_name += "/Ollama("+info.launch_model_url.host()+")";
        if(!configured)
            return fail_launch("Set the Ollama host/IP in AI Settings first.");
        if(url.host().isEmpty()) // never let a damaged Ollama profile fall through to native/cloud routing
            return fail_launch("This chat's saved Ollama server is invalid; choose its model again in Change Model.");
    }
    else if(agent_entries[provider].status == ai_agent_status::SignInRequired)
    {
        if(!run_agent_login(provider))
            return fail_launch(info.launch_name+" sign-in was not completed.");
        refresh_agent_status(provider);
    }
    auto* process = new QProcess(this);
    process->setObjectName(session);
    process->setWorkingDirectory(provider == "Antigravity" ?
                                 ui->ai_work_dir->text() :
                                 QApplication::applicationDirPath()+"/ai");
    auto env = agent_environment(provider);
    if(!info.launch_model_url.isEmpty()) // agents inherit HTTP(S)_PROXY; keep LAN Ollama traffic direct
        env.insert("NO_PROXY",info.launch_model_url.host()+
                   (env.contains("NO_PROXY") ? ","+env.value("NO_PROXY") : QString()));
    env.insert("DSI_STUDIO_AGENT",provider);
    if(provider == "Muse")
        env.insert("MUSE_SESSION_ID",session);
#ifdef Q_OS_WIN
    // locate bash for windows
    for(const auto& path : {qEnvironmentVariable("ProgramFiles") + "/Git/bin",
                            qEnvironmentVariable("LOCALAPPDATA") + "/Programs/Git/bin"})
        if(QFile::exists(path + "/bash.exe"))
        {
            ai_log("bash found: "+path+"/bash.exe");
            env.insert("PATH",path + ";" + env.value("PATH"));
            break;
        }
#endif
    process->setProcessEnvironment(env);

    info.processes = process;
    auto name = info.launch_name; // a plain value copy for the async handlers below -- never info itself
    const bool first_launch = info.status == session_status::New; // pre-launch status: a never-established chat returns to New, not Failed


    // one ending for a failed start and for an exit: an unestablished first launch has no real id to preserve,
    // so it returns to New (its recorded message stays); a reconnect keeps its id
    auto end_process = [=](bool failed,bool user_stopped,QString message)
    {
        if(failed)
            ai_log(message);
        if(auto* info = ai_info::find(process->objectName()))
        {
            info->processes = nullptr;
            if(first_launch && info->status == session_status::New)
            {
                if(!failed && !user_stopped)
                    message = "ERROR: AI agent ended before creating a new chat.";
                set_ai_status(info->sessions,session_status::New,message);
                add_ai_history(*info,user_stopped ? "activity" : "error",message);
                info->save_config(); // so the recorded messages have a config.json naming their agent on reload
            }
            else if(failed || user_stopped)
            {
                set_ai_status(info->sessions,failed ? session_status::Failed : session_status::Completed,message);
                add_ai_history(*info,failed ? "error" : "activity",message);
            }
            else if(!process->property("had_reply").toBool())
            {
                set_ai_status(info->sessions,session_status::Completed,"No reply from AI agent.");
                add_ai_history(*info,"activity","No reply from AI agent.");
            }
            else
            {
                set_ai_status(info->sessions,session_status::Completed,"Agent process finished.");
                show_ai_project(*info);
            }
        }
        update_send_button();
        process->deleteLater();
    };

    connect(process,&QProcess::readyReadStandardError,this,[=]
    {
        auto error = process->property("stderr").toByteArray()+
                     process->readAllStandardError();
        process->setProperty("stderr",error.right(8*1024));
    });

    connect(process,&QProcess::started,this,[=]
    {
        // stdin stays open for every local provider: Codex app-server, Claude stream-json, and Muse MSP
        auto session = process->objectName();
        ai_log("connecting to "+ name + "@" + session+
            " pid:"+QString::number(process->processId()));
        // only the provider's own establish event confirms a session: a first launch stays New until then, while a
        // reconnect shows Thinking so a save mid-reconnect cannot persist established:false (see save_config())
        if(auto* info = ai_info::find(session))
        {
            set_ai_status(session,first_launch ? session_status::New : session_status::Thinking,
                          "Waiting for "+name+" connection");
            show_ai_project(*info);
        }
        update_send_button();
    });

    connect(process,&QProcess::errorOccurred,this,
            [=](QProcess::ProcessError error)
    {
        if(error == QProcess::FailedToStart)
            end_process(true,false,"ERROR: Cannot start "+name+": "+process->errorString());
    });

    connect(process,
            QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this,[=](int exit_code,QProcess::ExitStatus exit_status)
    {
        bool user_stopped = process->property("user_stopped").toBool();
        ai_log(name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        auto fatal_error = process->property("fatal_error").toString();
        bool failed = !user_stopped && (!fatal_error.isEmpty() || exit_code ||
                                         exit_status == QProcess::CrashExit);
        auto error_message = user_stopped ? QString("Stopped by user.") :
                             !fatal_error.isEmpty() ? "ERROR: "+fatal_error :
                             !error.isEmpty() ? "ERROR: "+QString::fromUtf8(error) :
                             "ERROR: "+name+" exited with code "+QString::number(exit_code)+".";
        end_process(failed,user_stopped,error_message);
    });
    return executable;
}

void write_agent_input(const ai_info& info,const QString& text) // the one stdin boundary for every local agent's user input; records nothing
{
    auto* process = info.processes;
    auto session = process->objectName(); // the established id, in case establishment renamed it
    if(info.provider == "Claude")
        process->write(json_line({{"type","user"},{"message",QJsonObject{{"role","user"},
            {"content",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}}}}));
    else if(info.provider == "Muse")
        process->write(muse_command(muse_uuid_v7(),"turn/start",{{"sessionId",session},{"ifBusy","steer"},
            {"input",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}}));
    else if(info.provider == "Antigravity")
        process->write(json_line({{"event","user"},{"message",QJsonObject{{"content",text}}}}));
    else if(info.provider == "Grok")
    {
        process->setProperty("turn_active",true); // Grok has no turn id: Stop cancels in-protocol only while a prompt is in flight
        process->write(json_line({{"jsonrpc","2.0"},{"id","prompt"},{"method","session/prompt"},
            {"params",QJsonObject{{"sessionId",session},
                {"prompt",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}}}}));
    }
    else // Codex app-server: steer into the currently active turn, or start a fresh one if idle
    {
        auto turn_id = process->property("turn_id").toString();
        QJsonObject params{{"threadId",session},{"input",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}};
        if(!turn_id.isEmpty())
            params["expectedTurnId"] = turn_id;
        process->write(json_line({{"id",turn_id.isEmpty() ? "turn_start" : "turn_steer"},
                                  {"method",turn_id.isEmpty() ? "turn/start" : "turn/steer"},{"params",params}}));
    }
}
bool cancel_agent_turn(const ai_info& info) // in-protocol cancel of the active turn; false when there is none to cancel (Claude, Antigravity, idle)
{
    auto* process = info.processes;
    auto session = process->objectName();
    auto turn_id = process->property("turn_id").toString();
    if(info.provider == "Muse" && !turn_id.isEmpty())
        process->write(muse_command(muse_uuid_v7(),"turn/cancel",{{"sessionId",session},{"turnId",turn_id}}));
    else if(info.provider == "Codex" && !turn_id.isEmpty())
        process->write(json_line({{"id","turn_interrupt"},{"method","turn/interrupt"},
            {"params",QJsonObject{{"threadId",session},{"turnId",turn_id}}}}));
    else if(info.provider == "Grok" && process->property("turn_active").toBool()) // the prompt reply then reports "cancelled"
        process->write(json_line({{"jsonrpc","2.0"},{"method","session/cancel"},
            {"params",QJsonObject{{"sessionId",session}}}}));
    else
        return false;
    return true;
}
void AIAgent::finish_agent_turn(QProcess* process,QString error,bool cancelled) // one turn-end rule for every agent whose protocol reports it
{
    process->setProperty("turn_active",false);
    process->setProperty("turn_id",QString());
    auto* info = ai_info::find(process->objectName());
    if(!info)
        return;
    if(error.isEmpty())
        return set_ai_status(info->sessions,session_status::WaitingUser,cancelled ? "Stopped by user." : "Waiting for user");
    if(!error.startsWith("ERROR:"))
        error.prepend("ERROR: ");
    set_ai_status(info->sessions,session_status::Failed,error);
    add_ai_history(*info,"error",error);
}
ai_info* AIAgent::establish_agent_session(QProcess* process,QString new_session)
{
    // a fresh launch adopts the agent's own id (empty: the agent runs under DSI Studio's id); a resume must keep its id
    auto old_session = process->objectName();
    auto* info = ai_info::find(old_session);
    if(!info) // the chat was deleted while launching -- deletion stays deleted
        return kill_process_tree(process),nullptr;
    if(new_session.isEmpty())
        new_session = old_session;
    if(!is_valid_session_id(new_session))
        return fail_agent_process(process,"Agent returned an invalid session ID."),nullptr;
    if(new_session != old_session)
    {
        if(info->status != session_status::New)
            return fail_agent_process(process,"Agent resumed a different session."),nullptr;
        if(!(info = assign_ai_session(old_session,new_session)))
            return fail_agent_process(process,"Agent session could not be assigned."),nullptr;
        process->setObjectName(new_session);
        info->processes = process;
    }
    set_ai_status(info->sessions,session_status::Thinking,"Session started; waiting for agent");
    info->save_config();
    return info;
}
QStringList AIAgent::configure_claude(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    static const char* ollama_model_vars[] = {
        "ANTHROPIC_DEFAULT_HAIKU_MODEL","ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL","CLAUDE_CODE_SUBAGENT_MODEL"};
    auto env = process->processEnvironment();
    if(!info.launch_model_url.isEmpty())
    {
        env.insert("ANTHROPIC_BASE_URL",info.launch_model_url.toString());
        env.insert("ANTHROPIC_AUTH_TOKEN","ollama");
        env.insert("ANTHROPIC_API_KEY","");
        env.insert("CLAUDE_CODE_USE_POWERSHELL_TOOL","1");
        if(!info.launch_model.isEmpty())
            for(auto name : ollama_model_vars)
                env.insert(name,info.launch_model);
    }
    else // real Anthropic model: strip any Ollama redirect inherited from the system environment
    {
        for(auto name : {"ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","CLAUDE_CODE_USE_POWERSHELL_TOOL"})
            env.remove(name);
        for(auto name : ollama_model_vars)
            env.remove(name);
    }
    process->setProcessEnvironment(env);

    connect(process,&QProcess::readyReadStandardOutput,this,
            [=]
            {
                while(process->canReadLine())
                {
                    auto event = next_json_line(process);
                    if(event.contains("error"))
                    {
                        if(auto* info = ai_info::find(process->objectName()))
                        {
                            auto details = "Error: "+QString::fromUtf8(QJsonDocument(event).toJson(QJsonDocument::Compact));
                            set_ai_status(info->sessions,session_status::Failed,details); // red dot; a later reply restores the status
                            add_ai_history(*info,"error",details);
                            // an expired OAuth login is not detected by "claude auth status"; require sign-in before the next launch
                            if(event["error"].toString() == "authentication_failed" &&
                               details.contains("OAuth",Qt::CaseInsensitive))
                            {
                                auto& entry = agent_entries["Claude"];
                                ++entry.status_check_id;
                                entry.status = ai_agent_status::SignInRequired;
                                entry.status_info.clear();
                                emit agent_status_changed("Claude");
                            }
                        }
                        continue;
                    }
                    auto event_type = event["type"].toString();
                    if(event_type == "system")
                    {
                        auto subtype = event["subtype"].toString();
                        if(subtype == "init") // the session-established event; Claude runs under DSI Studio's own --session-id/--resume id
                            establish_agent_session(process);
                        else if(subtype == "thinking_tokens")
                        {
                            if(auto* info = ai_info::find(process->objectName());
                               info && info->status != session_status::Thinking)
                                set_ai_status(info->sessions,session_status::Thinking,
                                              "Agent is thinking");
                        }
                        continue;
                    }

                    if(event_type != "assistant")
                        continue;

                    auto message = event["message"].toObject();
                    QStringList chats,reasonings;
                    for(const auto& value : message["content"].toArray())
                    {
                        auto content = value.toObject();
                        auto type = content["type"].toString();
                        if(type == "text")
                            chats << content["text"].toString();
                        else if(type == "thinking" || type == "reasoning")
                        {
                            auto text = content[type].toString();
                            reasonings << (text.isEmpty() ? content["text"].toString() : text);
                        }
                    }
                    auto chat_text = chats.join('\n').trimmed();
                    auto reasoning_text = reasonings.join('\n').trimmed();
                    if(!chat_text.isEmpty() || !reasoning_text.isEmpty())
                        process->setProperty("had_reply",true);
                    if(auto* info = ai_info::find(process->objectName()))
                        add_ai_reply(*info,chat_text,reasoning_text);
                }
            });
    connect(process,&QProcess::started,process,[process,text] // Claude's system/init only arrives after its first input
    {
        if(auto* info = ai_info::find(process->objectName()))
            write_agent_input(*info,text);
    });
    QStringList args{
        "-p",
        "--input-format","stream-json",
        "--output-format","stream-json",
        "--verbose",
        "--add-dir",ui->ai_work_dir->text(),
        "--allowedTools","Bash(bash ./dsi.sh:*),PowerShell(./dsi.ps1:*),WebFetch,WebSearch,Read,Glob,Grep",
        info.status == session_status::New ? "--session-id" : "--resume",info.sessions};
    // an absent --model falls back to whatever the Claude CLI last remembered from an unrelated session, not a real default
    args << "--model" << (info.launch_model.isEmpty() ? "sonnet" : info.launch_model);
    return args;
}
QStringList AIAgent::configure_muse(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    auto session = info.sessions;
    auto status = info.status;
    auto model = info.launch_model;
    auto workspace = info.model_settings["cwd"].toString();
    if(workspace.isEmpty())
        workspace = ui->ai_work_dir->text();

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        while(process->canReadLine())
        {
            auto msg = next_json_line(process);
            auto id = msg["id"].toString();
            auto session_request = process->property("muse_session_request").toString();
            if(msg.contains("error"))
            {
                auto message = msg["error"].toObject()["message"].toString().trimmed();
                message = "Muse "+(message.isEmpty() ? QString("request failed.") : message);
                if(id == "initialize" || id == session_request)
                    fail_agent_process(process,message);
                else
                    finish_agent_turn(process,message);
                continue;
            }

            if(id == "initialize")
            {
                process->write(json_line({{"jsonrpc","2.0"},{"method","initialized"}}));
                auto request_id = muse_uuid_v7();
                process->setProperty("muse_session_request",request_id);
                QJsonObject params{{"sessionId",session}};
                if(status == session_status::New)
                {
                    params["approvalMode"] = "allowAll";
                    params["workspaceRoot"] = workspace;
                    if(!model.isEmpty())
                        params["modelId"] = model;
                }
                process->write(muse_command(request_id,status == session_status::New ? "session/start" : "session/resume",params));
                continue;
            }

            if(id == session_request)
            {
                if(auto* current = establish_agent_session(process,msg["result"].toObject()["session"].toObject()["sessionId"].toString()))
                    write_agent_input(*current,text);
                continue;
            }

            auto result = msg["result"].toObject();
            if(result.contains("turnId"))
                process->setProperty("turn_id",result["turnId"].toString());

            auto method = msg["method"].toString();
            if(method == "turn/started")
                process->setProperty("turn_id",msg["params"].toObject()["turnId"].toString());
            else if(method == "item/completed")
            {
                auto item = msg["params"].toObject()["item"].toObject();
                auto kind = item["kind"].toString();
                auto value = item["text"].toString().trimmed();
                if(kind == "agentMessage" || kind == "reasoning")
                    add_ai_reply(process,kind == "agentMessage" ? value : QString(),kind == "reasoning" ? value : QString());
            }
            else if(method == "turn/completed")
            {
                auto params = msg["params"].toObject();
                auto terminal = params["terminal"].toString();
                auto error = params["error"].toObject()["message"].toString();
                if(error.isEmpty())
                    error = params["reason"].toString();
                finish_agent_turn(process,terminal != "failed" ? QString() : error.isEmpty() ? "Muse turn failed." : error,
                                  terminal == "cancelled");
            }
        }
    });

    connect(process,&QProcess::started,process,[=]
    {
        process->write(muse_initialize());
    });
    return {"serve","--trust-workspace","--disable-sandbox"};
}

QStringList AIAgent::configure_antigravity(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    bool resuming = info.status != session_status::New;
    auto workspace = ui->ai_work_dir->text();

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        while(process->canReadLine())
        {
            auto msg = next_json_line(process);
            auto event = msg["event"].toString();
            if(event == "init")
            {
                if(auto* current = establish_agent_session(process,msg["conversation_id"].toString()))
                    write_agent_input(*current,text);
                continue;
            }
            if(event != "result")
                continue;

            auto result = msg["result"].toObject();
            auto terminal = result["status"].toString();
            bool cancelled = terminal == "CANCELED" || terminal == "INTERRUPTED";
            auto reply = result["response"].toString().trimmed();
            if(terminal == "SUCCESS")
                add_ai_reply(process,reply,{});
            auto error = result["error"].toString().trimmed();
            finish_agent_turn(process,terminal == "SUCCESS" || cancelled ? QString() :
                              "Antigravity "+(!error.isEmpty() ? error : terminal.isEmpty() ? QString("request failed.") : terminal.toLower()+"."),
                              cancelled);
        }
    });

    QStringList args{"--input-format","stream-json",
                     "--output-format","stream-json",
                     "--dangerously-skip-permissions",
                     "--add-dir",workspace};
    if(!info.launch_model.isEmpty())
        args << "--model" << info.launch_model;
    if(resuming)
        args << "--conversation" << info.sessions;
    return args;
}

QStringList AIAgent::configure_grok(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    auto session = info.sessions; // DSI Studio's chat ID becomes Grok's session ID (_meta.sessionId), exported to tools as GROK_SESSION_ID
    bool resuming = info.status != session_status::New;
    auto model = info.launch_model;
    auto cwd = process->workingDirectory(); // the fixed ai folder: Grok groups persisted sessions by cwd
    auto send_prompt = [=]
    {
        if(auto* current = ai_info::find(process->objectName()))
            write_agent_input(*current,text);
    };

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        while(process->canReadLine())
        {
            auto msg = next_json_line(process);
            auto method = msg["method"].toString();
            if(method == "session/update")
            {
                // streamed chunks are collected and recorded once per turn: each add_ai_reply() is a persisted history entry
                auto update = msg["params"].toObject()["update"].toObject();
                auto type = update["sessionUpdate"].toString();
                auto value = update["content"].toObject()["text"].toString();
                if(type == "agent_message_chunk" || type == "agent_thought_chunk")
                {
                    auto key = type == "agent_message_chunk" ? "grok_chat" : "grok_reasoning";
                    process->setProperty(key,process->property(key).toString()+value);
                }
                else if(auto title = update["title"].toString().trimmed();type == "tool_call" && !title.isEmpty())
                    set_ai_status(process->objectName(),session_status::Thinking,title);
                continue;
            }
            if(method == "session/request_permission") // not expected with --always-approve; an unanswered reverse request would stall the turn
            {
                process->write(json_line({{"jsonrpc","2.0"},{"id",msg.value("id")},
                       {"result",QJsonObject{{"outcome",QJsonObject{{"outcome","cancelled"}}}}}}));
                if(auto* current = ai_info::find(process->objectName()))
                    add_ai_history(*current,"error","ERROR: Grok requested an interactive permission; the request was cancelled.");
                continue;
            }
            auto id = msg["id"].toString();
            if(id.isEmpty())
                continue; // other Grok notifications are not used
            if(msg.contains("error"))
            {
                auto error = msg["error"].toObject()["message"].toString().trimmed();
                auto message = "Grok "+id+" failed: "+(error.isEmpty() ? QString("request failed.") : error);
                if(id != "prompt")
                    fail_agent_process(process,message);
                else
                {
                    process->setProperty("grok_chat",QString());
                    process->setProperty("grok_reasoning",QString());
                    finish_agent_turn(process,message);
                }
                continue;
            }
            if(id == "initialize") // authenticate as Grok's own headless client does, failing closed without a usable credential
            {
                auto auth = msg["result"].toObject()["_meta"].toObject()["defaultAuthMethodId"].toString();
                if(auth.isEmpty())
                {
                    auto& entry = agent_entries["Grok"];
                    ++entry.status_check_id;
                    entry.status = ai_agent_status::SignInRequired;
                    entry.status_info.clear();
                    emit agent_status_changed("Grok");
                    fail_agent_process(process,"Grok is not signed in.");
                    continue;
                }
                process->write(json_line({{"jsonrpc","2.0"},{"id","authenticate"},{"method","authenticate"},
                       {"params",QJsonObject{{"methodId",auth}}}}));
            }
            else if(id == "authenticate")
            {
                QJsonObject meta{{"sessionId",session}};
                if(resuming)
                    meta = QJsonObject{{"noReplay",true}}; // DSI Studio already holds the transcript
                else if(!model.isEmpty())
                    meta["modelId"] = model;
                QJsonObject params{{"cwd",cwd},{"mcpServers",QJsonArray()},{"_meta",meta}};
                if(resuming)
                    params["sessionId"] = session;
                process->write(json_line({{"jsonrpc","2.0"},{"id","session"},{"method",resuming ? "session/load" : "session/new"},{"params",params}}));
            }
            else if(id == "session")
            {
                if(!establish_agent_session(process,msg["result"].toObject()["sessionId"].toString()))
                    continue;
                if(resuming && !model.isEmpty()) // a new session got its model through _meta.modelId
                    process->write(json_line({{"jsonrpc","2.0"},{"id","set_model"},{"method","session/set_config_option"},
                           {"params",QJsonObject{{"sessionId",process->objectName()},{"configId","model"},{"value",model}}}}));
                else
                    send_prompt();
            }
            else if(id == "set_model")
                send_prompt();
            else if(id == "prompt")
            {
                auto chat = process->property("grok_chat").toString().trimmed();
                auto reasoning = process->property("grok_reasoning").toString().trimmed();
                process->setProperty("grok_chat",QString());
                process->setProperty("grok_reasoning",QString());
                add_ai_reply(process,chat,reasoning);
                finish_agent_turn(process,{},msg["result"].toObject()["stopReason"].toString() == "cancelled");
            }
        }
    });
    connect(process,&QProcess::started,process,[=]
    {
        process->write(grok_initialize());
    });
    return {"agent","--always-approve","stdio"};
}
QStringList AIAgent::configure_codex(const ai_info& info,const QString& text)
{
    // app-server JSON-RPC over stdio: initialize -> initialized -> thread/start|resume -> turn/start;
    // item/completed carries each item's complete text, so no delta accumulation
    auto* process = info.processes;
    auto session = info.sessions; // captured by value into the async handler below -- never info itself (Codex renames/rekeys the session there)
    auto model = info.launch_model;
    auto work_dir = ui->ai_work_dir->text(); // NOT the thread's cwd (that stays prepare_ai()'s applicationDirPath()+"/ai", where AGENTS.md lives) -- granted as extra sandbox access instead, same role "--add-dir" played for the old codex exec launch
    bool resuming = info.status != session_status::New; // pre-launch status: New means never established (open a fresh thread), anything else means session already names a real Codex thread id to resume
    QStringList args;
    if(!info.launch_model_url.isEmpty()) // Ollama: a custom provider id -- Codex forces localhost for its reserved "ollama"/"oss" ids (openai/codex#8240)
    {
        auto endpoint = info.launch_model_url;
        endpoint.setPath("/v1");
        // every -c goes before app-server
        args << "-c" << "model_providers.dsi_ollama.name=\"DSI Studio Ollama\""
             << "-c" << "model_providers.dsi_ollama.base_url=\""+endpoint.toString()+"\""
             << "-c" << "model_providers.dsi_ollama.wire_api=\"responses\""
             << "-c" << "model_providers.dsi_ollama.requires_openai_auth=false"
             << "-c" << "model_provider=\"dsi_ollama\"";
    }

    // our own request's reply: {"id":...,"result":...} or {"id":...,"error":...}, never has "method"
    auto handle_response = [=](const QJsonObject& msg)
    {
        auto id = msg["id"].toString();
        if(msg.contains("error"))
        {
            auto error = msg["error"].toObject()["message"].toString().trimmed();
            auto message = "Codex "+id+" failed: "+(error.isEmpty() ? "Unknown error." : error);
            if(auto* info = ai_info::find(process->objectName()))
            {
                set_ai_status(info->sessions,info->status == session_status::New ? session_status::New :
                              process->property("turn_id").toString().isEmpty() ? session_status::Failed :
                              session_status::Thinking,message);
                if(id == "initialize" || id == "thread_start" || id == "thread_resume")
                {
                    // The finish handler records the failure and releases the process for a fresh attempt.
                    process->setProperty("stderr",process->property("stderr").toByteArray()+'\n'+message.toUtf8());
                    return kill_process_tree(process);
                }
                add_ai_history(*info,"error",message);
            }
            return;
        }
        if(id == "initialize")
        {
            process->write(json_line({{"method","initialized"}}));
            // "never": an unanswered approval request would hang the turn. No "cwd": the thread stays in the ai folder
            // (prepare_ai()) so Codex finds AGENTS.md; work_dir is granted as extra sandbox access instead
            QJsonObject params{{"approvalPolicy","never"},
                {"sandboxPolicy",QJsonObject{{"type","workspaceWrite"},
                    {"writableRoots",QJsonArray{work_dir}}}}};
            if(!model.isEmpty() && model != "default") // Codex's own code-assigned alias for "no explicit choice" -- omit the field instead
                params["model"] = model;
            if(resuming)
                params["threadId"] = session;
            process->write(json_line({{"id",resuming ? "thread_resume" : "thread_start"},
                                      {"method",resuming ? "thread/resume" : "thread/start"},{"params",params}}));
            return;
        }
        if(id != "thread_start" && id != "thread_resume")
            return;

        if(auto* info = establish_agent_session(process,msg["result"].toObject()["thread"].toObject()["id"].toString()))
            write_agent_input(*info,text);
    };

    // a server-initiated notification: {"method":...,"params":...}, no "id", no reply expected
    auto handle_notification = [=](const QJsonObject& msg)
    {
        auto method = msg["method"].toString();
        if(method == "turn/started") // tracks the active turn id -- start_ai()'s Codex send path needs it to steer into a running turn instead of starting a conflicting one
            process->setProperty("turn_id",msg["params"].toObject()["turn"].toObject()["id"].toString());
        else if(method == "item/completed")
        {
            auto item = msg["params"].toObject()["item"].toObject();
            auto type = item["type"].toString();
            if(type == "agentMessage")
                add_ai_reply(process,item["text"].toString().trimmed(),QString());
            else if(type == "reasoning")
            {
                QStringList lines;
                for(auto key : {"summary","content"})
                    for(auto v : item[key].toArray())
                        lines << v.toString();
                add_ai_reply(process,QString(),lines.join('\n').trimmed());
            }
        }
        else if(method == "turn/completed")
        {
            // idle again: the next Codex send starts a fresh turn instead of steering into this one
            auto turn = msg["params"].toObject()["turn"].toObject();
            auto turn_status = turn["status"].toString();
            auto error = turn["error"].toObject()["message"].toString();
            finish_agent_turn(process,turn_status != "failed" ? QString() : error.isEmpty() ? "Turn failed" : error,
                              turn_status == "interrupted");
        }
        else if(method == "error")
        {
            auto error = msg["params"].toObject()["error"].toObject()["message"].toString().trimmed();
            auto message = "Codex error: "+(error.isEmpty() ? QString("Unknown error.") : error);
            if(auto* info = ai_info::find(process->objectName()))
            {
                // An error notification can precede recovery; turn/completed owns the final status.
                set_ai_status(info->sessions,info->status,message);
                add_ai_history(*info,"error",message);
            }
        }
    };

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        while(process->canReadLine())
        {
            auto msg = next_json_line(process);
            if(!msg.contains("method") && (msg.contains("result") || msg.contains("error")))
                handle_response(msg);
            else
                handle_notification(msg);
        }
    });

    connect(process,&QProcess::started,process,[=]
    {
        process->write(json_line({{"id","initialize"},{"method","initialize"},
            {"params",QJsonObject{{"clientInfo",QJsonObject{
                {"name","DSI Studio"},{"version","1.0"}}}}}}));
    });

    return args << "app-server";
}

void AIAgent::start_ai(ai_info& info,const QString& text)
{
    if(!local_agents.contains(info.provider))
        return;

    bool launching = !info.processes;
    QString executable;
    if(launching && (executable = prepare_ai(info)).isEmpty()) // failed before creating a process: nothing was sent
        return;

    // recorded once, here, before the launch or write -- never replayed from an async establishment event
    add_ai_history(info,"user",text);
    info.save_config();
    ui->ai_chat_input->clear();
    if(!launching)
    {
        write_agent_input(info,text);
        set_ai_status(info.sessions,session_status::Thinking,"Message sent; waiting for agent");
        return;
    }

    // the first message of every launch/resume carries the same instructions; later messages are the raw text
    auto ai_dir = QApplication::applicationDirPath()+"/ai";
    auto workspace = info.model_settings["cwd"].toString().trimmed();
    if(workspace.isEmpty())
        workspace = ui->ai_work_dir->text().trimmed();
    // relative wherever the agent runs in the ai folder: Claude's --allowedTools pre-approves only "bash ./dsi.sh"
    auto dsi_sh = QDir::cleanPath(info.processes->workingDirectory()) == QDir::cleanPath(ai_dir) ?
                  QString("./dsi.sh") : "\""+ai_dir+"/dsi.sh\"";
    auto prompt = "Read and follow "+QDir::toNativeSeparators(ai_dir+"/AGENTS.md")+" before handling this request. "
                  "Use `bash "+dsi_sh+"` for DSI Studio commands. "
                  "The selected DSI Studio work directory is "+QDir::toNativeSeparators(workspace)+".\n\n"+text;
    QStringList args;
    if(info.provider == "Codex")
        args = configure_codex(info,prompt);
    else if(info.provider == "Muse")
        args = configure_muse(info,prompt);
    else if(info.provider == "Antigravity")
        args = configure_antigravity(info,prompt);
    else if(info.provider == "Grok")
        args = configure_grok(info,prompt);
    else
        args = configure_claude(info,prompt);
    ai_log("start " + executable +
           " args: " + args.join(" ").remove("\n"));
    // New only for a never-established launch; a resumed session shows Thinking (see the started handler in prepare_ai())
    set_ai_status(info.sessions,info.status == session_status::New ?
                  session_status::New : session_status::Thinking,
                  "Starting "+info.launch_name);
    start_process(*info.processes,executable,args);
}

void AIAgent::on_ai_send_message_clicked()
{
    auto* info = selected_info();
    auto text = ui->ai_chat_input->toPlainText().trimmed();

    // executes whatever current_send_action() reports, so the click always does what the label says
    switch(current_send_action())
    {
    case send_action::Disabled:
        return;
    case send_action::Resume: // only reachable for a Web chat, see current_send_action()
        if(!google_refresh_token.isEmpty() || sign_in_google())
            start_web(*info);
        return;
    case send_action::Stop: // only reachable when info exists, see current_send_action()
        if(info->provider == "Web")
            stop_web();
        else if(!cancel_agent_turn(*info))
        {
            info->processes->setProperty("user_stopped",true); // finished() reports a user stop, not a failure
            kill_process_tree(info->processes);
        }
        return;
    case send_action::Send: // only reachable when info exists and isn't AgentServer, see current_send_action()
        start_ai(*info,text);
        update_send_button();
        return;
    }
}
