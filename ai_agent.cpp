#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMenu>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QShortcut>
#include <QShowEvent>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QTextFrame>
#include <QTimer>
#include <QToolButton>
#include <QUuid>
#include <QUrl>

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "ai_agent.hpp"
#include "ui_ai_agent.h"
#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"
#include "view_image.h"
#include "console.h"

std::unordered_map<QString,ai_info> ai_infos;
QMap<QString,quint64> ai_log_positions;
QString ai_project_dir;

struct ai_launch{
    QString session,name,executable,model,profile,prompt;
    QUrl model_url;
    QJsonObject model_setting;
    QProcess* process = nullptr;
    ai_model_provider model_provider = ai_model_provider::Native;
    bool new_session = false;
};

ai_provider ai_info::identify_provider(const QString& name)
{
    return name.contains("codex",Qt::CaseInsensitive) ? ai_provider::Codex :
           name.contains("claude",Qt::CaseInsensitive) ? ai_provider::Claude :
           ai_provider::Unknown;
}
QString ai_info::details(const QString& session) const
{
    int user = 0,assistant = 0,activity = 0;
    for(const auto& value : projects)
    {
        auto type = value.toObject()["type"].toString();
        user += type == "user";
        assistant += type == "assistant";
        activity += type == "request" || type == "activity";
    }
    auto time = [](const QJsonValue& value) {
        return QDateTime::fromString(value.toString(),Qt::ISODate).toString(
                   "yyyy-MM-dd HH:mm:ss");};
    return QString("<b>%1</b><br><br>Agent: %2<br>Session: %3<br>Status: %4<br>"
        "Messages: %5 (%6 you, %7 AI)<br>Activities: %8<br>"
        "Created: %9<br>Updated: %10")
        .arg(title(session).toHtmlEscaped(),agent_name.toHtmlEscaped(),session.toHtmlEscaped(),processes ? "Working" : "Idle")
        .arg(user+assistant).arg(user).arg(assistant).arg(activity)
        .arg(time(projects.first().toObject()["time"]),
             time(projects.last().toObject()["time"]));
}
void ai_info::set_agent_name(const QString& name)
{
    if(!name.isEmpty()) set_provider(identify_provider(name),name);
}
void ai_info::set_provider(ai_provider value,const QString& name)
{
    provider = value; agent_name = name;
}
void ai_info::set_process(QProcess* process)
{
    processes = process;
}
void ai_log(QString text)
{
    tipl::out() << ("[AI AGENT] "+text.remove('\r').replace('\n',"\n[AI AGENT] ")).toStdString();
}
QPair<QUrl,bool> ai_ollama_url(const QSettings& settings)
{
    auto host = settings.value("ai/ollama_host","localhost").toString().trimmed();
    bool configured = !host.isEmpty();
    if(!host.contains("://"))
        host.prepend("http://");
    QUrl url(host);
    url.setPort(settings.value("ai/ollama_port",11434).toInt());
    return {url,configured};
}

AIAgent::AIAgent(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::AIAgent)
{
    ui->setupUi(this);
    ai_status_timer = new QTimer(this);
    ai_status_timer->setInterval(500);
    connect(ai_status_timer,&QTimer::timeout,this,[this]
    {
        if(ai_status_delay)
        {
            if(!--ai_status_delay)
                set_ai_status();
            return;
        }
        ai_status_dots = ai_status_dots%3+1;
        ui->ai_status->setText(ai_status_waiting+QString(ai_status_dots,'.'));
        ui->ai_status->repaint();
    });
    set_ai_status();

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
        ui->ai_agent_selector->setItemData(
            index,QVariant::fromValue(profiles),Qt::UserRole+2);
        ai_log(path.isEmpty() ? agent+" not found" : agent+": "+path);
        if(!path.isEmpty())
            ai_log(agent+" models: "+
                   (models.isEmpty() ? "none detected" : models.join(", ")));
    };
    QString codex_path,claude_path;
    {
        // Find Codex executable and models
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
        // Find Claude executable and models
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
        update_model_selector(ui->ai_agent_selector->currentIndex());
    };
    connect(ui->ai_agent_selector,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,[update_models]{update_models();});
    update_models();

    {
        auto update_agent_name = [this]
        {
            auto index = ui->ai_agent_selector->currentIndex();
            QString name = index == int(ai_provider::Codex) ? "Codex" : "Claude";
            if(ui->ai_model_selector->currentData().toJsonObject()[
                    "provider"].toInt() == int(ai_model_provider::Ollama))
                name += "/Ollama("+ai_ollama_url(settings).first.host()+")";
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
    if(default_index >= 0 && default_index < ui->ai_agent_selector->count())
        ui->ai_agent_selector->setCurrentIndex(default_index);
    ui->ai_model_selector->setCurrentText(
        settings.value("ai/default_model",
                       ui->ai_model_selector->currentText()).toString());

    auto* send = new QShortcut(
        QKeySequence(Qt::CTRL|Qt::Key_Return),ui->ai_chat_input);
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
    connect(ai_project_menu->addAction("Details..."),
            &QAction::triggered,this,[this]
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
            QMessageBox::information(
                this,"Remove Project","Wait for the AI agent to finish first.");
            return;
        }
        if(QMessageBox::question(
               this,"Remove Project",
               "Remove this project and its saved history?") != QMessageBox::Yes)
            return;

        QFile::remove(ai_project_dir+"/"+QString::fromLatin1(
                          QUrl::toPercentEncoding(session))+".jsonl");
        settings.remove("ai/title/"+session);
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
                    setStyleSheet(i == item ?
                        "color:#202124;background:#dce9f9;" : "");
        if(item)
        {
            stop_ai_blink();
            auto session = item->data(Qt::UserRole).toString();
            const auto& info = ai_infos[session];
            auto index = int(info.provider);
            if(index >= 0)
                ui->ai_agent_selector->setCurrentIndex(index);
            auto model = info.model_settings.value("model").toString();
            if(model.isEmpty() || ui->ai_model_selector->findText(model) < 0)
                model = "default";
            ui->ai_model_selector->setCurrentText(model);
            show_ai_project(session);
        }
        else
            ui->ai_chat_history->clear();
    });

    for(const auto& info : dir.entryInfoList(
            {"*.jsonl"},QDir::Files,QDir::Time|QDir::Reversed))
    {
        auto session = QUrl::fromPercentEncoding(
                           info.completeBaseName().toLatin1());
        QFile file(info.filePath());
        if(!file.open(QIODevice::ReadOnly))
            continue;

        QJsonArray loaded_history;
        QString agent_name;
        QJsonObject model_settings;
        while(!file.atEnd())
        {
            auto doc = QJsonDocument::fromJson(file.readLine());
            if(!doc.isObject())
                continue;
            auto entry = doc.object();
            if(loaded_history.isEmpty())
            {
                agent_name = entry["agent"].toString();
                model_settings = entry["model_settings"].toObject();
            }
            loaded_history.append(entry);
        }

        if(loaded_history.isEmpty() || session.isEmpty())
            continue;
        auto* ai = create_ai_info(session,agent_name);
        if(!ai)
            continue;
        ai->model_settings = model_settings;
        ai->project_titles = settings.value("ai/title/"+session).toString();
        for(const auto& entry : loaded_history)
            ai->projects.append(entry);
        show_ai_project(session);
    }
    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);
}

AIAgent::~AIAgent()
{
    delete ui;
}

void AIAgent::refresh_ai_info(ai_info& info)
{
    show_ai_project(info.sessions);
    set_ai_status("Agent request completed.",true);
}
void AIAgent::add_ai_reply(ai_info& info,const QString& chat,const QString& reasoning)
{
    QByteArray reply;
    ai_command(info,QJsonDocument(QJsonObject{
        {"request","CHAT"},{"reasoning",reasoning},{"chat",chat}}).
        toJson(QJsonDocument::Compact),reply);
    refresh_ai_info(info);
}

ai_info* find_ai_info(const QString& session)
{
    auto found = ai_infos.find(session);
    return found == ai_infos.end() ? nullptr : &found->second;
}
ai_info* create_ai_info(QString session,QString agent)
{
    if(session.isEmpty())
        return nullptr;
    if(auto* info = find_ai_info(session))
        return info;
    auto provider = ai_info::identify_provider(agent);
    if(provider == ai_provider::Unknown)
        return nullptr;
    auto& info = ai_infos[session];
    info.sessions = std::move(session);
    info.set_provider(provider,agent);
    return &info;
}
bool save_ai_title(ai_info& info,QString title)
{
    title = title.simplified();
    if(title.isEmpty())
        return false;
    if(title == info.project_titles)
        return true;
    QSettings settings;
    settings.setValue("ai/title/"+info.sessions,title);
    settings.sync();
    if(settings.status() != QSettings::NoError)
        return false;
    info.project_titles = title;
    return true;
}
void record_ai_history(ai_info& info,QJsonObject entry)
{
    if(info.projects.isEmpty())
    {
        entry["agent"] = info.agent_name;
        entry["model_settings"] = info.model_settings;
    }
    entry["time"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    info.projects.append(entry);

    QSettings settings;
    if(!settings.value("ai/keep_history",true).toBool())
        return;
    QFile file(ai_project_dir+"/"+QString::fromLatin1(
                   QUrl::toPercentEncoding(info.sessions))+".jsonl");
    if(!file.open(QIODevice::WriteOnly|QIODevice::Append) ||
       file.write(QJsonDocument(entry).toJson(
                      QJsonDocument::Compact)+'\n') < 0)
        tipl::warning() << "cannot write ai history to "
                        << ai_project_dir.toStdString();
}

void AIAgent::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    stop_ai_blink();
}

void AIAgent::set_ai_status(QString status,bool temporary)
{
    ai_status_timer->stop();
    ai_status_delay = 0;
    if(!status.isEmpty())
        ai_status_activity = status;
    if(active_ai_processes && (status.isEmpty() || temporary))
    {
        ai_status_waiting = ai_status_activity;
        if(ai_status_waiting.endsWith('.'))
            ai_status_waiting.chop(1);
        ai_status_waiting += ", waiting for agent";
        ai_status_dots = 1;
        status = ai_status_waiting+".";
        ai_status_timer->start();
    }
    else
    if(status.isEmpty())
        status = "Current task complete.";
    ui->ai_status->setText(status);
    ui->ai_status->repaint();
    if(temporary && !active_ai_processes)
    {
        ai_status_delay = 4;
        ai_status_timer->start();
    }
}

void ai_command(ai_info& info,const QByteArray& data,QByteArray& reply)
{
    const auto& session = info.sessions;
    reply.clear();
    ai_log("received: "+QString::fromUtf8(data));
    static const QRegularExpression ansi_escape(
        QStringLiteral("\x1B\\[[0-?]*[ -/]*[@-~]"));
    QString chat,reasoning;
    auto set_reply = [&](QByteArray result)
    {
        if(!chat.isEmpty() || !reasoning.isEmpty())
        {
            QJsonObject entry{{"type","assistant"},{"text",chat}};
            if(!reasoning.isEmpty())
                entry["reasoning"] = reasoning;
            record_ai_history(info,entry);
        }
        reply = std::move(result);
        ai_log(QString("reply for %1@%2: %3 ...")
                   .arg(info.agent_name,session,
                        QString::fromUtf8(reply).left(32)));
        info.prompts = {};
    };
    auto add_prompts = [&](QJsonObject result)
    {
        if(!info.prompts.isEmpty())
            result["prompt"] = info.prompts;
        return result;
    };
    auto reply_object = [&](QJsonObject result)
    {
        set_reply(QJsonDocument(add_prompts(result)).toJson(QJsonDocument::Compact));
    };
    auto reply_error = [&](const QString& error)
    {
        reply_object(QJsonObject{{"status","error"},{"error",error}});
    };
    auto reply_results = [&](QJsonArray results)
    {
        bool success = std::all_of(results.begin(),results.end(),
            [](const auto& result)
            {
                return result.toObject()["status"] == "success";
            });
        reply_object(QJsonObject{
            {"status",success ? "success" : "error"},{"result",results}});
    };

    // Parse request
    QJsonParseError error;
    auto doc = QJsonDocument::fromJson(data,&error);
    if(!doc.isObject())
        return reply_error("invalid JSON: "+error.errorString());

    auto request = doc.object();
    auto type = request["request"].toString().toUpper();

    // Initialize log position
    {
        if(!ai_log_positions.contains(session))
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            ai_log_positions[session] = console.total_size;
        }
    }

    auto get_window_id = [](QWidget* window)
    {
        if(qobject_cast<MainWindow*>(window))
            return QString("main");
        auto address = QString::number(reinterpret_cast<quintptr>(window),16);
        if(qobject_cast<tracking_window*>(window))
            return "tracking"+address;
        if(qobject_cast<view_image*>(window))
            return "image"+address;
        return QString();
    };

    chat = request["chat"].toString().trimmed();
    reasoning = request["reasoning"].toString().trimmed();
    if(type == "TITLE")
    {
        auto title = request["title"].toString().simplified();
        if(title.isEmpty())
            return reply_error("missing title");
        if(!save_ai_title(info,title))
            return reply_error("cannot save title");
        return reply_object(QJsonObject{{"status","success"}});
    }
    if(request.contains("title"))
        return reply_error("title is valid only for TITLE");

    if(type == "CMD")
    {
        auto fail = [&](const QString& error)
        {
            reply_results(QJsonArray{QJsonObject{
                {"status","error"},{"error",error}}});
        };
        auto window = request["window"].toString();
        auto command = request["command"];
        if(command.isUndefined() || command.isNull())
            return fail("missing command field");
        if(window.isEmpty())
            return fail("missing target window field");
        std::vector<std::vector<std::string>> cmds;
        std::vector<std::string> cmd0_list;
        {
            for(const auto& value : (command.isArray() ? command.toArray() : QJsonArray{command}))
            {
                auto object = value.toObject();
                auto& cmd = cmds.emplace_back();
                auto add = [&](const QJsonValue& value){cmd.push_back(value.toVariant().toString().toUtf8().toStdString());};
                add(object["cmd"]);
                if(cmd[0].empty())
                    return fail("invalid cmd text");
                cmd0_list.push_back(cmd[0]);
                auto param = object["param"];
                if(param.isArray())
                    for(const auto& value : param.toArray())
                        add(value);
                else if(!param.isUndefined() && !param.isNull())
                    add(param);
            }
        }
        if(cmds.empty())
            return fail("missing command field");

        QWidget* target = nullptr;
        {
            QString target_type,target_title;
            auto windows = QApplication::allWidgets();
            for(auto* each : windows)
            {
                if(each->property("busy").toBool())
                    return fail("another CMD is running; check opened windows");
                if(get_window_id(each) == window)
                {
                    target = each;
                    target_type =
                        qobject_cast<MainWindow*>(each) ? "main" :
                            qobject_cast<tracking_window*>(each) ? "tracking" : "image";
                    if(target_type != "main")
                        target_title = QFileInfo(each->windowTitle()).fileName();
                }
            }
            if(!target)
                return fail("target window not found, terminated by user?");
            auto compact = QString::fromUtf8(tipl::merge(cmd0_list,','));
            record_ai_history(info,QJsonObject{
                    {"type","request"},
                    {"text",compact + " \u2192 "+ target_type + " window "+target_title},
                    {"compact",compact},
                    {"window",window}});
        }


        bool updates_enabled = target->updatesEnabled();
        target->setUpdatesEnabled(false);
        target->setProperty("busy",true);

        QJsonArray results;
        for(const auto& cmd : cmds)
        {
            QString output,error,command_name = QString::fromUtf8(cmd[0]);
            QJsonObject result{{"cmd",command_name}};
            {
                std::lock_guard<std::mutex> lock(console.edit_buf);
                console.capture = &output;
            }
            try
            {
                auto execute = [&](auto* window)
                {
                    if(!window->command(cmd))
                    {
                        error = QString::fromUtf8(window->error_msg);
                        error = (error.isEmpty() ? "command failed" : error)+
                                ". Read ai/DSI_STUDIO_AI_MANUAL.md and retry.";
                    }
                };
                if(auto* window = qobject_cast<MainWindow*>(target))
                    execute(window);
                else if(auto* window = qobject_cast<tracking_window*>(target))
                    execute(window);
                else if(auto* window = qobject_cast<view_image*>(target))
                    execute(window);
            }
            catch(const std::exception& e){error = e.what();}
            catch(...){error = "unknown error";}

            {
                std::lock_guard<std::mutex> lock(console.edit_buf);
                console.capture = nullptr;
            }

            output.remove(ansi_escape);
            error.remove(ansi_escape);

            if(!output.isEmpty())
                result["output"] = output;
            if(!error.isEmpty())
                result["error"] = error;
            result["status"] = error.isEmpty() ? "success" : "error";

            results.append(result);
            if(!error.isEmpty())
                break;
        }

        target->setProperty("busy",false);
        target->setUpdatesEnabled(updates_enabled);
        if(auto* window = qobject_cast<tracking_window*>(target))
        {
            window->slice_need_update = true;
            window->glWidget->update_slice();
        }
        else
            target->update();

        return reply_results(results);
    }

    if(type == "CHAT")
    {
        if(chat.isEmpty() && reasoning.isEmpty())
            return reply_error("missing chat or reasoning");
        return reply_object(QJsonObject{{"status","success"}});
    }

    if(type == "LIST")
    {
        auto status = [](bool waiting,bool busy)
        {
            return waiting ? "waiting" : busy ? "busy" : "idle";
        };

        auto* modal = QApplication::activeModalWidget();
        bool application_busy = tipl::status_list.size() > 1;
        QJsonObject windows;

        for(auto* window : QApplication::allWidgets())
        {
            auto id = get_window_id(window);
            if(id.isEmpty())
                continue;

            bool busy = window->property("busy").toBool();
            if(auto* tracking = qobject_cast<tracking_window*>(window))
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

            bool waiting = modal && (modal == window || window->isAncestorOf(modal));
            windows[id] = QJsonObject{
                {"status",status(waiting,busy)},
                {"title",QDir::fromNativeSeparators(window->windowTitle())}
            };
            application_busy |= busy;
        }

        return reply_object(QJsonObject{
            {"status","success"},
            {"application",QJsonObject{
                {"status",status(bool(modal),application_busy)}}},
            {"windows",windows}});
    }

    if(type == "LOG")
    {
        QByteArray output;
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            auto end = console.total_size;
            auto first = end-quint64(console.history.size());
            auto begin = std::max(ai_log_positions.value(session),first);
            bool capped = end-begin > 16*1024;
            if(capped)
                begin = end-16*1024;
            auto text = console.history.mid(qsizetype(begin-first));
            if(capped)
                text.remove(0,text.indexOf('\n')+1);
            text.remove(ansi_escape);
            QStringList lines;
            for(const auto& line : text.split('\n'))
                if(!line.contains("[AI AGENT]"))
                    lines << line;
            output = lines.join('\n').right(4*1024).toUtf8();
            ai_log_positions[session] = end;
        }
        return reply_object(QJsonObject{
            {"status","success"},{"output",QString::fromUtf8(output)}});
    }

    reply_error("unknown request");
}

void AIAgent::show_ai_project(const QString& session)
{
    show_ai_project(session,{});
}
void AIAgent::show_ai_project(const QString& session,QJsonObject added_entry)
{
    auto& info = ai_infos[session];
    const auto& history = info.projects;
    if(history.isEmpty())
        return;

    auto agent_name = info.agent_name;
    auto project_title = info.title(session);
    auto* item = info.project_items;
    if(!item)
    {
        item = new QListWidgetItem;
        item->setData(Qt::UserRole,session);
        ui->ai_project_list->insertItem(0,item);
        info.project_items = item;

        auto* row = new QWidget;
        auto* title = new QPushButton(row);
        title->setObjectName("ai_project_title");
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
        layout->setSpacing(2);
        layout->addWidget(title,1);
        layout->addWidget(button);
        ui->ai_project_list->setItemWidget(item,row);

        auto* blink = new QTimer(row);
        blink->setObjectName("ai_chat_blink");
        blink->setInterval(500);
        connect(blink,&QTimer::timeout,row,[row]
        {
            row->setStyleSheet(row->styleSheet().isEmpty() ?
                "background:#ffe082;border-radius:5px;" : "");
        });

        connect(title,&QPushButton::clicked,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
        connect(button,&QToolButton::pressed,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
    }

    auto* row = ui->ai_project_list->itemWidget(item);
    // Update project title
    {
        auto* title = row->findChild<QPushButton*>("ai_project_title");
        item->setText({});
        title->setText(project_title);
        title->setToolTip(project_title);
        item->setSizeHint(QSize(0,row->sizeHint().height()));
    }

    auto* current = ui->ai_project_list->currentItem();
    if(!current)
    {
        ui->ai_project_list->setCurrentItem(item);
        return;
    }

    bool visible = current == item && isVisible();
    const auto added_type = added_entry["type"].toString();

    if(!added_type.isEmpty() && added_type != "user" && !visible)
    {
        row->setStyleSheet("background:#ffe082;border-radius:5px;");
        row->findChild<QTimer*>("ai_chat_blink")->start();
    }

    if(current != item)
        return;

    struct request_text{QString full,compact;};
    auto request_content = [](const QJsonObject& entry)
    {
        auto full = entry["text"].toString();
        auto compact = entry["compact"].toString();
        return request_text{full,compact.isEmpty() ? full : compact};
    };
    auto to_html = [](QString text)
    {
        return text.toHtmlEscaped().replace('\n',"<br>");
    };
    auto display_time = [](const QJsonValue& value)
    {
        return QDateTime::fromString(value.toString(),Qt::ISODate).
               toString("MM/dd HH:mm:ss");
    };

    auto append = [&](const QJsonObject& entry,const QString& activity,
                      const QString& end_time = {})
    {
        auto type = entry["type"].toString();
        bool user = type == "user",request = type == "request";
        auto content = request ? request_content(entry).full : entry["text"].toString();
        auto reasoning = entry["reasoning"].toString().trimmed();
        if(content.trimmed().isEmpty() && reasoning.isEmpty())
            return;

        content = to_html(content);
        if(!reasoning.isEmpty())
            content = "<span style=\"color:#5f6368;\">"+to_html(reasoning)+"</span>"+
                      (content.isEmpty() ? "" : "<br>"+content);

        if(!activity.isEmpty())
            content +=
                "<br><span style=\"color:#5f6368;font-size:9pt;\">" +
                to_html(activity) +
                "</span>";

        if(request)
            content = "<span style=\"color:#5f6368;\">"+content+"</span>";

        auto color = request ? "#f1f3f4" : user ? "#e8f0fe" : "#e8f5e9";
        auto time = display_time(entry["time"]);
        if(!end_time.isEmpty())
            time += "\u2013"+display_time(end_time);
        auto cell = QString(
                        "<td bgcolor=\"%1\"><b style=\"background-color:%1\">%2</b>"
                        "<font color=\"#80868b\">%3</font><br>%4</td>")
                        .arg(color,(user ? QString("You") : agent_name).toHtmlEscaped()+" &middot; ",time,content);

        auto cursor = ui->ai_chat_history->document()->
                      rootFrame()->lastCursorPosition();
        cursor.insertHtml(
            QString("<table width=\"100%\" cellspacing=\"3\" "
                    "cellpadding=\"7\"><tr>%1</tr></table>")
                .arg(user ? "<td width=\"20%\"></td>"+cell :
                         cell+"<td width=\"20%\"></td>"));
    };

    const bool paired = added_type == "assistant" && history.size() > 1 &&
            history[history.size()-2].toObject()["type"] == "request";
    const bool rebuild = added_type.isEmpty() || added_type == "request" || paired;

    if(rebuild)
    {
        auto standalone_request = [&](int index)
        {
            return history[index].toObject()["type"] == "request" &&
                   (index+1 == history.size() || history[index+1].toObject()["type"] != "assistant");
        };
        auto request_window = [&](const QJsonObject& entry)
        {
            return entry["window"].toVariant().toString();
        };

        ui->ai_chat_history->clear();
        for(int index = 0;index < history.size();++index)
        {
            auto entry = history[index].toObject();
            auto type = entry["type"].toString();
            if(type == "request")
            {
                if(!standalone_request(index))
                    continue;
                auto combined = entry;
                auto content = request_content(entry);
                QStringList activities{content.compact};
                auto window = request_window(entry);
                auto end = index;
                while(!window.isEmpty() && end+1 < history.size() &&
                      standalone_request(end+1) &&
                      request_window(history[end+1].toObject()) == window)
                    activities << request_content(history[++end].toObject()).compact;
                if(end != index)
                {
                    auto target = content.full;
                    target = target.mid(target.lastIndexOf(" \u2192 ")+3);
                    combined["text"] = activities.join(", ")+" \u2192 "+target;
                }
                append(combined,{},end == index ? QString() :
                       history[end].toObject()["time"].toString());
                index = end;
                continue;
            }
            auto activity = type == "assistant" && index &&
                            history[index-1].toObject()["type"] == "request" ?
                            request_content(history[index-1].toObject()).full :
                            QString();
            append(entry,activity);
        }
    }
    else
        append(added_entry,{});

    ui->ai_chat_history->ensureCursorVisible();
    QTimer::singleShot(0,ui->ai_chat_history,[this]
    {
        auto* bar = ui->ai_chat_history->verticalScrollBar();
        bar->setValue(bar->maximum());
    });
}

void AIAgent::stop_ai_blink()
{
    auto* item = ui->ai_project_list->currentItem();
    auto* row = item ? ui->ai_project_list->itemWidget(item) : nullptr;
    if(!row)
        return;
    row->findChild<QTimer*>("ai_chat_blink")->stop();
    row->setStyleSheet({});
}
void AIAgent::update_model_selector(int index,QString selected)
{
    auto profiles = ui->ai_agent_selector->itemData(
                        index,Qt::UserRole+2).toJsonObject();
    ui->ai_model_selector->clear();
    ui->ai_model_selector->addItem("default");
    for(const auto& model :
        ui->ai_agent_selector->itemData(index).toStringList())
        ui->ai_model_selector->addItem(
            model,QVariant::fromValue(profiles[model].toObject()));
    if(selected.isEmpty())
        return;
    auto selected_index = ui->ai_model_selector->findText(selected);
    if(selected_index < 0)
        selected_index = ui->ai_model_selector->findText(
            settings.value("ai/default_model").toString());
    ui->ai_model_selector->setCurrentIndex(std::max(0,selected_index));
}
void AIAgent::refresh_codex_models(const QString& path)
{
    if(path.isEmpty())
        return;

    auto* process = new QProcess(this);
    connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this,[=]
    {
        QStringList models;
        auto doc = QJsonDocument::fromJson(process->readAllStandardOutput());
        auto list = doc.isArray() ? doc.array() :
                        doc.object()["models"].toArray();
        for(const auto& value : list)
        {
            auto object = value.toObject();
            auto model = object["slug"].toString();
            if(model.isEmpty()) model = object["model"].toString();
            if(model.isEmpty()) model = object["id"].toString();
            if(!model.isEmpty()) models << model;
        }

        models.removeDuplicates();
        models.sort(Qt::CaseInsensitive);
        auto index = int(ai_provider::Codex);
        ui->ai_agent_selector->setItemData(index,models);
        if(ui->ai_agent_selector->currentIndex() == index)
            update_model_selector(index);
        refresh_ollama_models();
        process->deleteLater();
    });

    process->start(path,{"debug","models"});
    QTimer::singleShot(5000,process,[process]
    {
        if(process->state() != QProcess::NotRunning)
            process->kill();
    });
}
void AIAgent::refresh_ollama_models()
{
    auto set_models = [this](QStringList ollama_models)
    {
        for(auto index : {int(ai_provider::Codex),int(ai_provider::Claude)})
        {
            if(ui->ai_agent_selector->itemData(index,Qt::UserRole+1).toString().isEmpty())
                continue;

            auto models = ui->ai_agent_selector->itemData(index).toStringList();
            auto profiles = ui->ai_agent_selector->itemData(
                                                     index,Qt::UserRole+2).toJsonObject();

            for(auto i = models.size();i--;)
                if(profiles[models[i]].toObject()["provider"].toInt() ==
                    int(ai_model_provider::Ollama))
                    profiles.remove(models.takeAt(i));

            for(const auto& model : ollama_models)
            {
                models << model;
                profiles[model] =
                    QJsonObject{{"provider",int(ai_model_provider::Ollama)}};
            }

            models.removeDuplicates();
            models.sort(Qt::CaseInsensitive);
            ui->ai_agent_selector->setItemData(index,models);
            ui->ai_agent_selector->setItemData(
                index,QVariant::fromValue(profiles),Qt::UserRole+2);

            if(ui->ai_agent_selector->currentIndex() == index)
                update_model_selector(
                    index,ui->ai_model_selector->currentText());
        }
    };

    auto [url,configured] = ai_ollama_url(settings);
    if(!configured)
        return set_models({});

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
                bool okay = reply->error() == QNetworkReply::NoError;
                if(okay)
                    for(const auto& value :
                         QJsonDocument::fromJson(reply->readAll()).
                         object()["models"].toArray())
                        models << value.toObject()["name"].toString();
                ai_log("Ollama "+url.toString()+" "+ (okay ? "connected" : reply->errorString()));
                set_models(okay ? models : QStringList());
                reply->deleteLater();
                network->deleteLater();
            });
}
bool AIAgent::set_ai_title(const QString& session,QString title)
{
    auto* info = find_ai_info(session);
    if(!info || !save_ai_title(*info,title))
        return false;
    show_ai_project(session);
    return true;
}

void AIAgent::add_ai_history(const QString& session,const QString& type,const QString& text)
{
    add_ai_history(session,QJsonObject{{"type",type},{"text",text}});
}
void AIAgent::add_ai_history(const QString& session,QJsonObject entry)
{
    auto* info = find_ai_info(session);
    if(!info)
        return;
    record_ai_history(*info,entry);
    show_ai_project(session,entry);
}

void AIAgent::on_ai_new_chat_clicked()
{
    ui->ai_project_list->setCurrentItem(nullptr);
    ui->ai_chat_history->clear();
    ui->ai_chat_input->clear();
    ui->ai_chat_input->setFocus();
    set_ai_status();
}

void AIAgent::on_ai_quick_settings_clicked()
{
    QDialog dialog(this);
    dialog.setWindowTitle("AI Settings");
    QFormLayout layout(&dialog);
    QLineEdit host(settings.value("ai/ollama_host","localhost").toString());
    QSpinBox port;
    port.setRange(1,65535);
    port.setValue(settings.value("ai/ollama_port",11434).toInt());
    QComboBox agent,model;
    for(int index = 0;index < ui->ai_agent_selector->count();++index)
        agent.addItem(ui->ai_agent_selector->itemText(index),index);
    agent.setCurrentIndex(agent.findData(ui->ai_agent_selector->currentIndex()));
    auto update_models = [&]
    {
        auto index = agent.currentData().toInt();
        auto profiles = ui->ai_agent_selector->itemData(index,Qt::UserRole+2).
                        toJsonObject();
        model.clear();
        model.addItem("default");
        for(const auto& name : ui->ai_agent_selector->itemData(index).toStringList())
            model.addItem(name,profiles[name].toObject());
    };
    update_models();
    model.setCurrentText(ui->ai_model_selector->currentText());
    QCheckBox history("Keep AI chat history");
    history.setChecked(settings.value("ai/keep_history",true).toBool());
    layout.addRow("Ollama host/IP:",&host);
    layout.addRow("Ollama port:",&port);
    layout.addRow("Default agent:",&agent);
    layout.addRow("Default model:",&model);
    layout.addRow(&history);
    QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
    layout.addRow(&buttons);
    connect(&agent,QOverload<int>::of(&QComboBox::currentIndexChanged),&dialog,[&](int)
    {
        update_models();
    });
    connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
    if(dialog.exec() != QDialog::Accepted)
        return;

    settings.setValue("ai/ollama_host",host.text().trimmed());
    settings.setValue("ai/ollama_port",port.value());
    settings.setValue("ai/keep_history",history.isChecked());
    settings.setValue("ai/default_agent",agent.currentData().toInt());
    settings.setValue("ai/default_model",model.currentText());
    ui->ai_agent_selector->setCurrentIndex(agent.currentData().toInt());
    ui->ai_model_selector->setCurrentText(model.currentText());

    refresh_ollama_models();
}

ai_launch AIAgent::prepare_ai(ai_provider provider,QString session,
                                 const QString& text,ai_input input)
{
    ai_launch launch;

    // Resolve agent and session
    {
        launch.session = session;
        launch.new_session = session.isEmpty();
        launch.name = provider == ai_provider::Codex ? "Codex" : "Claude";
        launch.executable = ui->ai_agent_selector->itemData(
                                int(provider),Qt::UserRole+1).toString();
        if(launch.executable.isEmpty())
        {
            if(input == ai_input::Pending && !session.isEmpty())
                ai_infos[session].prompts.append(text);
            set_ai_status("AI agent is unavailable.",true);
            QMessageBox::warning(
                this,"AI Agent","AI agent is not installed or cannot be located.");
            return launch;
        }

        if(session.isEmpty() && provider == ai_provider::Claude)
        {
            session = QUuid::createUuid().toString(QUuid::WithoutBraces);
            create_ai_info(session,launch.name);
            launch.session = session;
        }


    }

    // Resolve model
    {
        if(!session.isEmpty())
            launch.model_setting = ai_infos[session].model_settings;
        if(launch.model_setting.isEmpty())
        {
            launch.model_setting["model"] = ui->ai_model_selector->currentText();
            launch.model_setting["info"] =
                ui->ai_model_selector->currentData().toJsonObject();
        }

        launch.model = launch.model_setting["model"].toString().trimmed();
        auto model_info = launch.model_setting["info"].toObject();
        launch.profile = model_info["profile"].toString();
        launch.model_provider =
            ai_model_provider(model_info["provider"].toInt());
        if(launch.model_provider == ai_model_provider::Ollama)
        {
            auto [url,configured] = ai_ollama_url(settings);
            launch.model_url = url;
            launch.name += "/Ollama("+launch.model_url.host()+")";
            if(!configured)
            {
                set_ai_status("Ollama is not configured.",true);
                QMessageBox::warning(
                    this,"AI Agent","Set the Ollama host/IP in AI Settings first.");
                return launch;
            }
        }
        if(launch.model.startsWith("default",Qt::CaseInsensitive))
            launch.model.clear();
        if(!session.isEmpty())
            ai_infos[session].model_settings = launch.model_setting;
    }

    auto set_ai_enabled = [this](bool enabled)
    {
        for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
            button->setEnabled(enabled);
    };
    auto* process = new QProcess(this);
    launch.process = process;
    process->setObjectName(session);
    process->setWorkingDirectory(QApplication::applicationDirPath()+"/ai");

    if(!session.isEmpty())
        ai_infos[session].set_process(process);
    else
        set_ai_enabled(false);

    if(input == ai_input::User)
    {
        if(!session.isEmpty())
            add_ai_history(session,"user",text);
        ui->ai_chat_input->clear();
    }

    connect(process,&QProcess::readyReadStandardError,this,[=]
    {
        auto error = process->property("stderr").toByteArray()+
                     process->readAllStandardError();
        process->setProperty("stderr",error.right(8*1024));
    });

    connect(process,&QProcess::started,this,[=]
    {
        ++active_ai_processes;
        if(provider != ai_provider::Claude)
            process->closeWriteChannel();
        auto session = process->objectName();
        ai_log("connecting to "+ launch.name + "@" +
            (session.isEmpty() ? QString("new") : session)+
            " pid:"+QString::number(process->processId()));
        set_ai_status();
        if(!session.isEmpty())
            show_ai_project(session);
    });

    connect(process,&QProcess::errorOccurred,this,
            [=](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
            return;

        auto session = process->objectName();
        ai_log(launch.name + " error:"+process->errorString());
        set_ai_status("Could not start "+launch.name+".",true);

        if(session.isEmpty())
        {
            set_ai_enabled(true);
            ui->ai_chat_input->setPlainText(text);

            auto message = "Cannot start AI agent: "+process->errorString();
            ui->ai_chat_history->setPlainText(message);
            QMessageBox::warning(this,"AI Agent",message);
        }
        else
        {
            ai_infos[session].set_process(nullptr);
            ai_infos[session].prompts.append(text);
            add_ai_history(session,"activity",
                           "Cannot start AI agent: "+
                           process->errorString());
        }
        process->deleteLater();
    });

    connect(process,
            QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this,[=](int exit_code,QProcess::ExitStatus exit_status)
    {
        active_ai_processes = std::max(0,active_ai_processes-1);
        set_ai_status(launch.name+" finished.",true);
        auto session = process->objectName();
        ai_log(launch.name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        bool failed = exit_code || exit_status == QProcess::CrashExit;
        if(failed)
            ai_log("error code:"+QString::number(exit_code)+" "+QString::fromUtf8(error));

        if(session.isEmpty())
        {
            set_ai_enabled(true);
            ui->ai_chat_input->setPlainText(text);
            QMessageBox::warning(
                this,"AI Agent",
                "AI agent ended before creating a new chat. Check the console.");
        }
        else
        {
            auto& info = ai_infos[session];
            info.set_process(nullptr);

            QString pending;
            for(int index = 0;index < info.prompts.size();++index)
            {
                if(index)
                    pending += "\n\n";
                pending += info.prompts[index].toString();
            }
            info.prompts = {};
            if(!pending.isEmpty())
            {
                process->deleteLater();
                start_ai(session,pending,ai_input::Pending);
                return;
            }

            auto history_size = process->property("history_size");
            bool no_reply = history_size.isValid() && info.projects.size() == history_size.toInt();
            if(failed)
                add_ai_history(session,"activity","AI agent failed.");
            else if(no_reply)
                add_ai_history(session,"activity","No reply from AI agent.");
            else
                show_ai_project(session);
        }
        process->deleteLater();
    });

    // Build agent prompt
    {
        QString prompt = text;
        if(launch.new_session || (input == ai_input::Pending &&
                                   ai_infos[session].projects.size() == 1))
        {
            QDir app(QApplication::applicationDirPath());
            prompt +=
                "\n\n[DSI Studio] Read \""+
                QDir::toNativeSeparators(
                    app.filePath("ai/DSI_STUDIO_AI_SETUP.md"))+
                "\" completely. In \""+
                QDir::toNativeSeparators(
                    app.filePath("ai/DSI_STUDIO_AI_MANUAL.md"))+
                "\", read the operating rules and common syntax, then search the "
                "command inventory only for commands relevant to this request. ";
        }
        launch.prompt = prompt;
    }
    return launch;
}

void AIAgent::run_ai(const ai_launch& launch,QStringList args)
{
    ai_log("start " + launch.executable + " args: " + args.join(" ").remove("\n"));
    set_ai_status("Starting "+launch.name+"...");
    launch.process->start(launch.executable,args);
}
void AIAgent::start_claude(QString session,const QString& text,ai_input input)
{
    auto claude_input = [](const QString& text)
    {
        return QJsonDocument(QJsonObject{
            {"type","user"},{"message",QJsonObject{
                {"role","user"},{"content",QJsonArray{QJsonObject{
                    {"type","text"},{"text",text}}}}}}}).
            toJson(QJsonDocument::Compact)+'\n';
    };
    if(auto* info = find_ai_info(session);
       info && info->processes &&
       info->processes->state() == QProcess::Running)
    {
        if(input == ai_input::User)
        {
            add_ai_history(session,"user",text);
            ui->ai_chat_input->clear();
        }
        info->processes->write(claude_input(text));
        set_ai_status("Message sent to Claude.");
        return;
    }

    auto launch = prepare_ai(ai_provider::Claude,session,text,input);
    if(!launch.process)
        return;
    auto* process = launch.process;
    if(launch.session.isEmpty())
    {
        add_ai_history(launch.session,"activity","Cannot start AI agent: missing session ID.");
        return;
    }
    launch.prompt.prepend("[DSI Studio] Session ID: "+launch.session+"\n\n");
    if(launch.model_provider == ai_model_provider::Ollama)
    {
        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("ANTHROPIC_BASE_URL",launch.model_url.toString());
        env.insert("ANTHROPIC_AUTH_TOKEN","ollama");
        env.insert("ANTHROPIC_API_KEY","");
        env.insert("CLAUDE_CODE_USE_POWERSHELL_TOOL","1");
        if(!launch.model.isEmpty())
            for(auto name : {"ANTHROPIC_DEFAULT_HAIKU_MODEL",
                              "ANTHROPIC_DEFAULT_SONNET_MODEL",
                              "ANTHROPIC_DEFAULT_OPUS_MODEL",
                              "CLAUDE_CODE_SUBAGENT_MODEL"})
                env.insert(name,launch.model);

        process->setProcessEnvironment(env);
    }

    process->setProperty("history_size",ai_infos[launch.session].projects.size());

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
            {
                auto output = process->readAllStandardOutput();
                tipl::out() << "stdout:" << output.toStdString();
                auto buffer = process->property("stdout_buffer").toByteArray()+output;
                for(int pos;(pos = buffer.indexOf('\n')) >= 0;)
                {
                    auto event = QJsonDocument::fromJson(buffer.left(pos)).object();
                    buffer.remove(0,pos+1);
                    auto event_type = event["type"].toString();
                    if(event_type == "system")
                    {
                        if(event["subtype"] == "thinking_tokens" &&
                           ai_status_activity != "Thinking")
                            set_ai_status("Thinking");
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
                    auto chat = chats.join('\n').trimmed();
                    auto reasoning = reasonings.join('\n').trimmed();
                    if(auto* info = find_ai_info(process->objectName()))
                        add_ai_reply(*info,chat,reasoning);
                }
                process->setProperty("stdout_buffer",buffer);
            });
    connect(process,&QProcess::started,process,
            [process,prompt = launch.prompt,claude_input]
            {process->write(claude_input(prompt));});

    QStringList args{
        "-p",
        "--input-format","stream-json",
        "--output-format","stream-json",
        "--verbose",
        "--disallowedTools","Bash",
        "--allowedTools","PowerShell(./dsi_agent.ps1 -Agent Claude -Session " + launch.session + " -Target *)",
        launch.new_session ? "--session-id" : "--resume",launch.session};
    if(!launch.model.isEmpty())
        args << "--model" << launch.model;
    run_ai(launch,args);
}
void AIAgent::start_codex(QString session,const QString& text,ai_input input)
{
    auto launch = prepare_ai(ai_provider::Codex,session,text,input);
    if(!launch.process)
        return;
    auto* process = launch.process;
    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        auto buffer = process->property("stdout_buffer").toByteArray()+
                      process->readAllStandardOutput();
        tipl::out() << "stdout:" << buffer.toStdString();
        for(int pos;(pos = buffer.indexOf('\n')) >= 0;)
        {
            auto event = QJsonDocument::fromJson(buffer.left(pos)).object();
            buffer.remove(0,pos+1);
            if(event["type"] == "thread.started")
            {
                auto session = event["thread_id"].toString();
                if(session.isEmpty())
                    continue;
                bool started_new_session = process->objectName().isEmpty();
                auto* info = create_ai_info(session,launch.name);
                if(!info)
                    continue;
                info->model_settings = launch.model_setting;
                if(started_new_session)
                {
                    process->setObjectName(session);
                    info->set_process(process);
                    add_ai_history(session,"user",text);
                    set_ai_status("Agent session ready.",true);
                    for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
                        button->setEnabled(true);
                }
                continue;
            }

            auto item = event["item"].toObject();
            auto type = item["type"].toString();
            if(type == "agent_message" || type == "reasoning")
            {
                auto text = item["text"].toString().trimmed();
                if(text.isEmpty())
                    continue;
                if(auto* info = find_ai_info(process->objectName()))
                {
                    bool reasoning = type == "reasoning";
                    add_ai_reply(*info,reasoning ? QString() : text,
                                 reasoning ? text : QString());
                }
            }
        }

        process->setProperty("stdout_buffer",buffer);
    });

    QStringList args{"exec"};
    if(launch.model_provider == ai_model_provider::Ollama)
    {
        auto url = launch.model_url;
        url.setPath("/v1");

        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("CODEX_OSS_BASE_URL",url.toString());
        launch.process->setProcessEnvironment(env);

        args << "--oss" << "--local-provider=ollama";
    }
    if(!launch.model.isEmpty())
    {
        args << "--model" << launch.model;
        if(!launch.profile.isEmpty())
            args << "--profile" << launch.profile;
    }
    if(launch.session.isEmpty())
        args << "--json" << "--skip-git-repo-check";
    else
        args << "resume" << launch.session << "--json"
             << "--skip-git-repo-check";
    args << launch.prompt;
    run_ai(launch,args);
}
void AIAgent::start_ai(QString session,const QString& text,ai_input input)
{
    auto provider = session.isEmpty() ?
        ai_provider(ui->ai_agent_selector->currentIndex()) :
        ai_infos[session].provider;
    switch(provider)
    {
    case ai_provider::Codex:
        return start_codex(session,text,input);
    case ai_provider::Claude:
        return start_claude(session,text,input);
    default:
        set_ai_status("Unsupported AI provider.",true);
        QMessageBox::warning(this,"AI Agent","Unsupported AI provider.");
    }
}

void AIAgent::on_ai_send_message_clicked()
{
    auto text = ui->ai_chat_input->toPlainText().trimmed();
    if(text.isEmpty())
        return;
    auto* item = ui->ai_project_list->currentItem();
    auto session = item ? item->data(Qt::UserRole).toString() : QString();
    if(!session.isEmpty())
    {
        auto& info = ai_infos[session];
        if((info.processes &&
            (info.provider != ai_provider::Claude ||
             info.processes->state() != QProcess::Running)) ||
           info.provider == ai_provider::Unknown)
        {
            info.prompts.append(text);
            add_ai_history(session,"user",text);
            ui->ai_chat_input->clear();
            set_ai_status("Message queued for the AI agent.",true);
            return;
        }
    }
    start_ai(session,text,ai_input::User);
}
