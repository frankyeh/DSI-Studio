#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QLineEdit>
#include <QLocalSocket>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSpinBox>
#include <QTextFrame>
#include <QLabel>
#include <QMovie>
#include <QTimer>
#include <QToolButton>
#include <QUuid>
#include <QUrl>

#include <algorithm>
#include <mutex>

#include "../mainwindow.h"
#include "ui_mainwindow.h"
#include "../tracking/tracking_window.h"
#include "../opengl/glwidget.h"
#include "../view_image.h"
#include "../console.h"

extern MainWindow* main_window;

std::unordered_map<QString,ai_info> ai_infos;

struct ai_launch{
    QString session,text,name,executable,model,profile,prompt;
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
        "Created: %9<br>Updated: %10<br>Working folder: %11")
        .arg(title(session).toHtmlEscaped(),
             (agent_name.isEmpty() ? QString("Not available") : agent_name).toHtmlEscaped(),
             session.toHtmlEscaped(),processes ? "Working" : "Idle")
        .arg(user+assistant).arg(user).arg(assistant).arg(activity)
        .arg(time(projects.first().toObject()["time"]),
             time(projects.last().toObject()["time"]),
             (work_dirs.isEmpty() ? QString("Not available") : work_dirs).toHtmlEscaped());
}
void ai_info::update(const QString& name,const QString& cwd)
{
    if(!name.isEmpty()) set_provider(identify_provider(name),name);
    if(QDir(cwd).exists()) work_dirs = cwd;
}
void ai_info::set_provider(ai_provider value,const QString& name)
{
    provider = value; agent_name = name;
}
void ai_info::set_process(QProcess* process)
{
    processes = process;
    if(process) work_dirs = process->workingDirectory();
}
QString ai_info::take_prompts(void)
{
    QStringList result;
    for(const auto& prompt : prompts) result << prompt.toString();
    prompts = {}; return result.join("\n\n");
}
QByteArray ai_info::prepare_reply(QByteArray reply,QJsonArray* results) const
{
    if(results)
    {
        if(!prompts.isEmpty())
        {
            auto result = results->last().toObject();
            result["prompt"] = prompts;
            results->replace(results->size()-1,result);
        }
        return QJsonDocument(*results).toJson(QJsonDocument::Compact);
    }
    if(prompts.isEmpty())
        return reply;
    auto payload = "PROMPT\t" +
                   QJsonDocument(prompts).toJson(QJsonDocument::Compact) + '\n';
    auto pos = reply.indexOf('\n');
    return pos < 0 ? reply.append('\n').append(payload) :
                     reply.insert(pos+1,payload);
}

void ai_log(QString text)
{
    tipl::out() << ("[AI AGENT] "+text.remove('\r').replace('\n',"\n[AI AGENT] ")).toStdString();
}

QRegularExpression ansi_escape(QStringLiteral("\x1B\\[[0-?]*[ -/]*[@-~]"));

QString ai_window_status(QWidget* window,bool& busy,int& jobs)
{
    busy = window->property("busy").toBool();
    jobs = 0;

    if(qobject_cast<MainWindow*>(window))
        return "main";

    if(auto* w = qobject_cast<tracking_window*>(window))
    {
        if(w->tractWidget)
            jobs = int(std::count_if(
                w->tractWidget->thread_data.begin(),
                w->tractWidget->thread_data.end(),
                [](const auto& thread){return bool(thread);}));

        busy |= jobs || w->history.running_commands ||
                std::any_of(w->slices.begin(),w->slices.end(),
                            [](const auto& slice)
                            {
                                auto custom = std::dynamic_pointer_cast<CustomSliceModel>(slice);
                                return custom && custom->running;
                            });
        return "tracking";
    }

    if(qobject_cast<view_image*>(window))
        return "image";

    return {};
}

void ai_reply(QLocalSocket* socket,const QString& session,
                      QByteArray reply,QJsonArray* results = nullptr)
{
    auto& info = ai_infos[session];
    reply = info.prepare_reply(reply,results);
    auto written = socket->write(reply);
    ai_log(QString("DSI Studio replied " + info.agent_name + "@%1").arg(session));
    if(written == reply.size())
        info.prompts = {};
}

void ai_request_list(QLocalSocket* socket,const QString& session)
{
    static quint64 next_id = 0;

    int level = std::max(0,int(tipl::status_list.size())-1);
    bool global_busy = level != 0;
    bool has_tracking = false;
    QString status;

    if(level)
    {
        const auto& prog = tipl::status_list.back();
        status = QString::fromStdString(prog.status).section('\n',0,0).replace('\t',' ');
        if(!prog.at.empty())
            status += " " + QString::fromStdString(prog.at).section('\n',0,0).replace('\t',' ');
    }

    QStringList result;
    for(auto* window : QApplication::allWidgets())
    {
        bool busy;
        int jobs;
        auto type = ai_window_status(window,busy,jobs);
        if(type.isEmpty())
            continue;

        if(!window->property("remote_id").isValid())
            window->setProperty("remote_id",++next_id);

        auto command =
            window->property("command").toString();
        if(level == 1 &&
            status.startsWith("[AI REQUEST]") &&
            !command.isEmpty())
            status = command;

        auto title =
            QDir::fromNativeSeparators(window->windowTitle());
        title.replace('\t',' ').replace('\n',' ');

        result << QString("%1\t%2\t%3\t%4\t%5").arg(type).arg(window->property("remote_id").toULongLong())
                      .arg(int(busy)).arg(jobs).arg(title);

        global_busy |= busy;
        has_tracking |= jobs != 0;
    }

    if(!level && global_busy)
    {
        level = 1;
        status = has_tracking ? "fiber tracking" : "working";
    }

    result.prepend(QString("OKAY\t%1\t%2\t%3").arg(int(global_busy)).arg(level).arg(status));
    ai_reply(socket,session,result.join('\n').toUtf8());
}

void ai_request_command(QLocalSocket* socket,const QString& session,
                                const QJsonObject& request)
{
    auto fail = [&](const QString& error)
    {
        QJsonArray results{QJsonObject{
            {"index",0},{"okay",false},{"output",""},{"error",error}}};
        ai_reply(socket,session,{},&results);
    };
    auto id = request["window"].toVariant().toString();
    auto commands = request["command"].toArray();
    if(id.isEmpty() || commands.isEmpty())
        return fail("invalid CMD. Read ai/DSI_STUDIO_AI_MANUAL.md before retry.");
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
        return fail("window not found");

    for(auto* window : QApplication::allWidgets())
        if(window->property("busy").toBool())
            return fail("another CMD is running; use LIST");

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
    target->setProperty("busy",true);
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

        target->setProperty("command",cmd.empty() ? QString() : QString::fromStdString(cmd[0]));

        bool okay = error.isEmpty() && run(cmd,output,error);

        target->setProperty("command",QVariant());

        output.remove(ansi_escape);
        error.remove(ansi_escape);
        if(!okay)
            error = (error.isEmpty() ? "command failed" : error) +
                    ". Read ai/DSI_STUDIO_AI_MANUAL.md and retry.";

        result["okay"] = okay;
        result["output"] = output;
        if(!okay)
            result["error"] = error;
        results.append(result);

        if(!okay)
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

    ai_reply(socket,session,QByteArray(),&results);
}

QMap<QString,quint64> ai_log_positions;
void ai_request_log(QLocalSocket* socket,const QString& session,
                            bool include)
{
    QByteArray output;
    {
        std::lock_guard<std::mutex> lock(console.edit_buf);
        auto end = console.total_size;
        auto first = end-quint64(console.history.size());
        auto begin = std::max(ai_log_positions.value(session),first);
        if(include)
        {
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
        }
        ai_log_positions[session] = end;
    }
    ai_reply(socket,session,include ? QByteArray("OKAY\n"+output) : QByteArray("OKAY"));
}
void ai_request(QLocalSocket* socket,const QByteArray& data)
{
    QJsonParseError error;
    auto doc = QJsonDocument::fromJson(data,&error);
    if(!doc.isObject())
        return ai_reply(socket,{},("ERROR\tinvalid JSON: " +
                                     error.errorString()).toUtf8());

    auto request = doc.object();
    auto agent_name = request["agent"].toString().trimmed();
    auto type = request["request"].toString().toUpper();
    auto session = request["session"].toString().trimmed();

    if(agent_name.isEmpty())
        return ai_reply(socket,{},"ERROR\tmissing agent: provide a provider-tagged agent name and reuse it for the entire conversation");
    auto provider = ai_info::identify_provider(agent_name);
    if(provider == ai_provider::Unknown)
        return ai_reply(socket,{},"ERROR\tinvalid agent: include Codex or Claude in the agent name");
    if(session.isEmpty())
        return ai_reply(socket,{},"ERROR\tmissing session: provide the initiating-chat session ID and reuse it for the entire conversation");
    if(QUuid(session).toString(QUuid::WithoutBraces).compare(session,Qt::CaseInsensitive))
        return ai_reply(socket,{},"ERROR\tinvalid session: read DSI_STUDIO_AI_SETUP.md and obtain the correct resumable provider thread ID");

    auto index = int(provider);
    if(index >= 0 && main_window->ui->ai_agent_selector->model()->flags(
           main_window->ui->ai_agent_selector->model()->index(index,0)).testFlag(
           Qt::ItemIsEnabled))
        main_window->ui->ai_agent_selector->setCurrentIndex(index);

    if(!ai_log_positions.contains(session))
    {
        std::lock_guard<std::mutex> lock(console.edit_buf);
        ai_log_positions[session] = console.total_size;
    }

    tipl::progress p;
    if(type == "CMD")
    {
        auto msg = QString("[AI REQUEST] ")+type+" from "+agent_name+"@"+session;
        p = tipl::progress(msg.remove('\r').replace('\n',' ').toStdString());
    }

    ai_infos[session].update(agent_name,request["cwd"].toString());


    auto chat = request["chat"].toString().trimmed();
    auto activity = request;
    activity.remove("chat");
    auto json = QString::fromUtf8(QJsonDocument(activity).toJson(QJsonDocument::Compact));

    ai_log(json);

    if(type == "CMD")
        for(auto* window : QApplication::allWidgets())
            if(window->property("remote_id").toString() ==
               request["window"].toVariant().toString())
            {
                QString target_type =
                    qobject_cast<MainWindow*>(window) ? "main" :
                    qobject_cast<tracking_window*>(window) ? "tracking" :
                    qobject_cast<view_image*>(window) ? "image" : "unknown";
                activity["_target_type"] = target_type;
                if(target_type != "main")
                    activity["_target_title"] =
                        QFileInfo(window->windowTitle()).fileName();
                break;
            }
    json = QString::fromUtf8(QJsonDocument(activity).toJson(
                                 QJsonDocument::Compact));

    if(type != "LIST" || !chat.isEmpty())
        main_window->add_ai_history(session,"request",json);

    if(!chat.isEmpty())
        main_window->add_ai_history(session,"assistant",chat);

    if(type == "TITLE")
        return ai_reply(socket,session,
                        main_window->set_ai_title(
                            session,request["title"].toString()) ?
                            "OKAY" : "ERROR\tinvalid title");
    if(type == "LIST")
        return ai_request_list(socket,session);
    if(type == "LOG")
        return ai_request_log(socket,session,chat.isEmpty());
    if(type == "CHAT")
        return ai_reply(socket,session,
                        chat.isEmpty() ? "ERROR\tmissing chat" : "OKAY");
    if(type == "CMD")
        return ai_request_command(socket,session,request);
    ai_reply(socket,session,"ERROR\tunknown request");
}

void MainWindow::select_agent_model(const ai_info& info)
{
    auto index = int(info.provider);
    if(index >= 0)
        ui->ai_agent_selector->setCurrentIndex(index);
    auto model = info.model_settings.value("model").toString();
    if(model.isEmpty() || ui->ai_model_selector->findText(model) < 0)
        model = "default";
    ui->ai_model_selector->setCurrentText(model);
}
void MainWindow::show_ai_project(const QString& session,QJsonObject added)
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

        item->setSizeHint(QSize(0,row->sizeHint().height()));

        connect(title,&QPushButton::clicked,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
        connect(button,&QToolButton::pressed,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
    }

    auto* title = ui->ai_project_list->itemWidget(item)->
                  findChild<QPushButton*>("ai_project_title");
    item->setText(project_title);
    title->setText(project_title);
    title->setToolTip(project_title);
    item->setSizeHint(
        QSize(0,title->parentWidget()->sizeHint().height()));

    auto* current = ui->ai_project_list->currentItem();
    if(!current)
    {
        ui->ai_project_list->setCurrentItem(item);
        return;
    }

    bool visible = current == item &&
                   ui->tabWidget->currentWidget() == ui->tab_8;
    const auto added_type = added["type"].toString();

    if(!added_type.isEmpty() && added_type != "user" && !visible)
    {
        auto* row = ui->ai_project_list->itemWidget(item);
        row->setStyleSheet("background:#ffe082;border-radius:5px;");
        row->findChild<QTimer*>("ai_chat_blink")->start();
    }

    if(current != item)
        return;

    // show running gif
    {
        auto* running =
            ui->ai_chat_composer->findChild<QLabel*>("ai_running");
        if(!running)
        {
            running = new QLabel(ui->ai_chat_composer);
            running->setObjectName("ai_running");
            running->setFixedSize(24,24);

            auto* movie = new QMovie(
                ":/icons/icons/ajax-loader.gif",{},running);
            movie->setScaledSize(QSize(20,20));
            running->setMovie(movie);
            ui->ai_chat_composer_layout->addWidget(running,1,2);
        }

        running->setVisible(info.processes);
        if(info.processes)
            running->movie()->start();
        else
            running->movie()->stop();
    }

    auto request_content = [](const QJsonObject& entry,bool compact = false)
    {
        auto summary = entry["_summary"].toString();
        if(!summary.isEmpty())
            return summary;
        auto detail = QJsonDocument::fromJson(
                          entry["text"].toString().toUtf8()).object();
        auto action = detail["request"].toString().toUpper();
        if(action == "CMD")
        {
            auto commands = detail["command"].toArray();
            auto command_name = [](const QJsonArray& command)
            {
                auto name = command[0].toString();
                if(name == "hub" && command.size() > 1)
                    name += " "+command[1].toString();
                return name;
            };
            QStringList names;
            if(!commands.isEmpty())
                if(commands[0].isArray())
                    for(const auto& command : commands)
                        names << command_name(command.toArray());
                else
                    names << command_name(commands);
            auto target = detail["_target_type"].toString();
            auto destination = target.isEmpty() ?
                "window "+detail["window"].toVariant().toString() :
                target+" window";
            auto title = detail["_target_title"].toString();
            if(compact)
                return names.join(", ");
            return (names.isEmpty() ? "unknown" : names.join(", "))+
                   " \u2192 "+destination+(title.isEmpty() ? "" : " "+title);
        }
        return action == "LIST" ? "checked opened windows" :
               action == "LOG" ? "read new console output" :
               action+" request";
    };

    auto append = [&](const QJsonObject& entry,const QString& activity = {})
    {
        auto type = entry["type"].toString();
        bool user = type == "user",request = type == "request";
        auto content = request ? request_content(entry) : entry["text"].toString();
        if(content.trimmed().isEmpty())
            return;

        content = content.toHtmlEscaped().replace('\n',"<br>");

        if(!activity.isEmpty())
            content +=
                "<br><span style=\"color:#5f6368;font-size:9pt;\">" +
                activity.toHtmlEscaped().replace('\n',"<br>") +
                "</span>";

        if(request)
            content = "<span style=\"color:#5f6368;\">"+content+"</span>";

        auto color = request ? "#f1f3f4" : user ? "#e8f0fe" : "#e8f5e9";
        auto time = QDateTime::fromString(entry["time"].toString(),Qt::ISODate).
                    toString("MM/dd HH:mm:ss");
        auto end_time = entry["_end_time"].toString();
        if(!end_time.isEmpty())
            time += "\u2013"+QDateTime::fromString(end_time,Qt::ISODate).
                    toString("MM/dd HH:mm:ss");
        auto cell = QString(
                        "<td bgcolor=\"%1\"><b style=\"background-color:%1\">%2</b>"
                        "<font color=\"#80868b\">%3</font><br>%4</td>")
                        .arg(color,
                             (user ? QString("You") : agent_name).toHtmlEscaped()+" &middot; ",
                             time,
                             content);

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
        auto request_window = [](const QJsonObject& entry)
        {
            return QJsonDocument::fromJson(entry["text"].toString().toUtf8()).object()["window"].toVariant().toString();
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
                QStringList activities{request_content(entry,true)};
                auto window = request_window(entry);
                auto end = index;
                while(!window.isEmpty() && end+1 < history.size() &&
                      standalone_request(end+1) &&
                      request_window(history[end+1].toObject()) == window)
                    activities << request_content(history[++end].toObject(),true);
                if(end != index)
                {
                    auto target = request_content(entry);
                    target = target.mid(target.lastIndexOf(" \u2192 ")+3);
                    combined["_summary"] = activities.join(", ")+" \u2192 "+target;
                    combined["_end_time"] = history[end].toObject()["time"];
                }
                append(combined);
                index = end;
                continue;
            }
            QString activity;
            if(type == "assistant" && index)
            {
                auto previous = history[index-1].toObject();
                if(previous["type"] == "request")
                {
                    auto request = QJsonDocument::fromJson(
                                       previous["text"].toString().toUtf8()).object();
                    if(request["request"].toString().compare(
                            "CMD",Qt::CaseInsensitive) == 0)
                        activity = request_content(previous);
                }
            }
            append(entry,activity);
        }
    }
    else
        append(added);

    ui->ai_chat_history->ensureCursorVisible();
    QTimer::singleShot(0,ui->ai_chat_history,[this]
    {
        auto* bar = ui->ai_chat_history->verticalScrollBar();
        bar->setValue(bar->maximum());
    });
}

void MainWindow::stop_ai_blink()
{
    auto* item = ui->ai_project_list->currentItem();
    auto* row = item ? ui->ai_project_list->itemWidget(item) : nullptr;
    if(!row)
        return;
    row->findChild<QTimer*>("ai_chat_blink")->stop();
    row->setStyleSheet({});
}
void MainWindow::refresh_codex_models(const QString& path)
{
    if(path.isEmpty())
        return;

    auto* process = new QProcess(this);
    connect(process,
    QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
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
        {
            ui->ai_model_selector->clear();
            ui->ai_model_selector->addItem("default");
            ui->ai_model_selector->addItems(models);
        }
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
void MainWindow::refresh_ollama_models()
{
    auto index = int(ai_provider::Claude);
    auto set_models = [this,index](QStringList ollama_models)
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
            {
                auto selected = ui->ai_model_selector->currentText();
                ui->ai_model_selector->clear();
                ui->ai_model_selector->addItem("default");

                for(const auto& model : models)
                    ui->ai_model_selector->addItem(
                        model,QVariant::fromValue(profiles[model].toObject()));

                auto selected_index =
                    ui->ai_model_selector->findText(selected);
                if(selected_index < 0)
                    selected_index = ui->ai_model_selector->findText(
                        settings.value("ai/default_model").toString());
                ui->ai_model_selector->setCurrentIndex(
                    std::max(0,selected_index));
            }
        }
    };

    auto host = settings.value("ai/ollama_host","localhost").
                toString().trimmed();
    if(host.isEmpty())
        return set_models({});

    if(!host.contains("://"))
        host = "http://"+host;
    QUrl url(host);
    url.setPort(settings.value("ai/ollama_port",11434).toInt());
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
bool MainWindow::save_ai_entry(const QString& session,const QJsonObject& entry)
{
    if(!settings.value("ai/keep_history",true).toBool())
        return true;
    QFile file(ai_project_dir+"/"+QString::fromLatin1(
                   QUrl::toPercentEncoding(session))+".jsonl");
    return file.open(QIODevice::WriteOnly|QIODevice::Append) &&
           file.write(QJsonDocument(entry).toJson(
                          QJsonDocument::Compact)+'\n') >= 0;
}

bool MainWindow::set_ai_title(const QString& session,QString title)
{
    if(session.isEmpty())
        return false;
    title = title.simplified();
    auto& info = ai_infos[session];
    if(title.isEmpty())
        return false;
    if(title == info.project_titles)
        return true;

    QJsonObject entry{{"type","title"},{"text",title},
                      {"time",QDateTime::currentDateTime().toString(Qt::ISODate)}};
    if(!save_ai_entry(session,entry))
        return false;

    info.project_titles = title;
    show_ai_project(session);
    return true;
}

void MainWindow::add_ai_history(const QString& session,const QString& type,
                                const QString& text)
{
    if(session.isEmpty())
        return;
    QJsonObject entry{{"type",type},{"text",text},
                      {"time",QDateTime::currentDateTime().toString(Qt::ISODate)}};
    ai_infos[session].projects.append(entry);
    if(!save_ai_entry(session,entry))
        tipl::warning() << "cannot write ai history to "
                        << ai_project_dir.toStdString();
    show_ai_project(session,entry);
}

void MainWindow::on_ai_new_chat_clicked()
{
    ui->ai_project_list->setCurrentItem(nullptr);
    ui->ai_chat_history->clear();
    ui->ai_chat_input->clear();
    ui->ai_chat_input->setFocus();
    if(auto* running = ui->ai_chat_composer->findChild<QLabel*>("ai_running"))
    {
        running->hide();
        running->movie()->stop();
    }
}

void MainWindow::on_ai_quick_settings_clicked()
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

ai_launch MainWindow::prepare_ai(ai_provider provider,QString session,
                                 const QString& text,bool add_history)
{
    ai_launch launch;
    launch.session = session;
    launch.text = text;
    launch.new_session = session.isEmpty();
    launch.name = provider == ai_provider::Codex ? "Codex" : "Claude";
    launch.executable = ui->ai_agent_selector->itemData(
        int(provider == ai_provider::Codex ?
            ai_provider::Codex : ai_provider::Claude),
        Qt::UserRole+1).toString();
    if(launch.executable.isEmpty())
    {
        if(!add_history && !session.isEmpty())
            ai_infos[session].prompts.append(text);
        QMessageBox::warning(this,"AI Agent","AI agent is not installed or cannot be located.");
        return launch;
    }

    if(session.isEmpty() && provider == ai_provider::Claude)
    {
        session = QUuid::createUuid().toString(QUuid::WithoutBraces);
        ai_infos[session].set_provider(provider,launch.name);
        launch.session = session;
    }

    QString cwd;
    if(!session.isEmpty())
        cwd = ai_infos[session].work_dirs;
    if(!QDir(cwd).exists())
        cwd = work_dir();
    if(!QDir(cwd).exists())
        cwd = QApplication::applicationDirPath();


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
    launch.model_provider = ai_model_provider(model_info["provider"].toInt());
    if(launch.model_provider == ai_model_provider::Ollama)
    {
        auto host = settings.value("ai/ollama_host","localhost").
                    toString().trimmed();
        if(!host.contains("://"))
            host.prepend("http://");
        launch.name += "/Ollama(" + QUrl(host).host() + ")";
    }
    if(launch.model_provider == ai_model_provider::Ollama &&
        settings.value("ai/ollama_host","localhost").toString().trimmed().isEmpty())
    {
        QMessageBox::warning(
            this,"AI Agent","Set the Ollama host/IP in AI Settings first.");
        return launch;
    }

    if(launch.model.startsWith("default",Qt::CaseInsensitive))
        launch.model.clear();

    if(!session.isEmpty())
        ai_infos[session].model_settings = launch.model_setting;


    auto* process = new QProcess(this);
    launch.process = process;
    process->setObjectName(session);
    process->setWorkingDirectory(cwd);

    if(!session.isEmpty())
    {
        ai_infos[session].set_process(process);
        if(add_history)
            add_ai_history(session,"user",text);
    }
    else
        for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
            button->setEnabled(false);

    if(add_history)
        ui->ai_chat_input->clear();

    connect(process,&QProcess::readyReadStandardError,this,[=]
    {
        auto error = process->property("stderr").toByteArray()+
                     process->readAllStandardError();
        process->setProperty("stderr",error.right(8*1024));
    });

    connect(process,&QProcess::started,this,[=]
    {
        process->closeWriteChannel();
        auto session = process->objectName();
        ai_log("Connecting to "+ launch.name + " " +
            (session.isEmpty() ? QString("new") : session)+
            " pid:"+QString::number(process->processId()));
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

        if(session.isEmpty())
        {
            for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
                button->setEnabled(true);
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
        auto session = process->objectName();
        ai_log(launch.name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        auto output = (process->property("stdout").toByteArray()+
                       process->readAllStandardOutput()).trimmed();
        if(exit_code || exit_status == QProcess::CrashExit)
            ai_log("error code:"+QString::number(exit_code)+" "+QString::fromUtf8(error));

        if(session.isEmpty())
        {
            for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
                button->setEnabled(true);
            ui->ai_chat_input->setPlainText(text);
            QMessageBox::warning(
                this,"AI Agent",
                "AI agent ended before creating a new chat. Check the console.");
        }
        else
        {
            auto& info = ai_infos[session];
            info.set_process(nullptr);

            if(auto pending = info.take_prompts(); !pending.isEmpty())
            {
                process->deleteLater();
                start_ai(session,pending,false);
                return;
            }

            auto history_size = process->property("history_size");
            bool no_reply = history_size.isValid() && info.projects.size() == history_size.toInt();
            if(no_reply && !output.isEmpty())
                add_ai_history(session,"assistant",QString::fromUtf8(output));

            if(exit_code || exit_status == QProcess::CrashExit)
                add_ai_history(session,"activity","AI agent failed.");
            else if(no_reply && output.isEmpty())
                add_ai_history(session,"activity","No reply from AI agent.");
            else
                show_ai_project(session);
        }
        process->deleteLater();
    });

    bool initial_task = !session.isEmpty() && !add_history &&
                        ai_infos[session].projects.size() == 1;
    QString prompt = text;
    if(launch.new_session || initial_task)
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
            "command inventory only for commands relevant to this request. "
            "Use the local server and keep the generated identity. Use GUI "
            "control by default and run_cli only when explicitly requested. "
            "Attach only new user-facing chat and send the final answer once "
            "with CHAT. Process every returned PROMPT.";
    }
    if(launch.new_session && provider == ai_provider::Codex)
        prompt +=
            "\n\n[DSI Studio] Use \""+launch.name+
            "\" as agent and CODEX_THREAD_ID as session in every "
            "local-server request.";
    else if(!session.isEmpty())
        prompt +=
            "\n\n[DSI Studio] Continue through agent "+
            ai_infos[session].agent_name+" using session "+
            session+". Use this exact value as session in every local-server "
                      "request. Send new user-facing text and the final reply through "
                      "the named pipe.";
    launch.prompt = prompt;
    return launch;
}

void MainWindow::run_ai(const ai_launch& launch,QStringList args)
{
    ai_log("start " + launch.executable + " args: " + args.join(" ").remove("\n"));
    launch.process->start(launch.executable,args);

}
void MainWindow::start_claude(QString session,const QString& text,bool add_history)
{
    auto launch = prepare_ai(ai_provider::Claude,session,text,add_history);
    if(!launch.process)
        return;
    if(launch.session.isEmpty())
    {
        add_ai_history(launch.session,"activity","Cannot start AI agent: missing session ID.");
        return;
    }

    if(launch.model_provider == ai_model_provider::Ollama)
    {
        auto host = settings.value(
            "ai/ollama_host","localhost").toString().trimmed();
        if(!host.contains("://"))
            host = "http://"+host;
        QUrl url(host);
        url.setPort(settings.value("ai/ollama_port",11434).toInt());

        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("ANTHROPIC_BASE_URL",url.toString());
        env.insert("ANTHROPIC_AUTH_TOKEN","ollama");
        env.insert("ANTHROPIC_API_KEY","");

        if(!launch.model.isEmpty())
            for(auto name : {"ANTHROPIC_DEFAULT_HAIKU_MODEL",
                              "ANTHROPIC_DEFAULT_SONNET_MODEL",
                              "ANTHROPIC_DEFAULT_OPUS_MODEL",
                              "CLAUDE_CODE_SUBAGENT_MODEL"})
                env.insert(name,launch.model);

        launch.process->setProcessEnvironment(env);
    }

    launch.process->setProperty(
        "history_size",ai_infos[launch.session].projects.size());

    connect(launch.process,&QProcess::readyReadStandardOutput,
            launch.process,[process = launch.process]
            {
                auto output = process->property("stdout").toByteArray()+
                              process->readAllStandardOutput();
                process->setProperty("stdout",output.right(64*1024));
            });

    QStringList args{"-p",launch.new_session ? "--session-id" : "--resume",launch.session};
    if(!launch.model.isEmpty())
        args << "--model" << launch.model;
    args << launch.prompt;
    run_ai(launch,args);
}
void MainWindow::start_codex(QString session,const QString& text,bool add_history)
{
    auto launch = prepare_ai(ai_provider::Codex,session,text,add_history);
    if(!launch.process)
        return;
    auto* process = launch.process;
    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        auto buffer = process->property("stdout_buffer").toByteArray()+
                      process->readAllStandardOutput();
        for(int pos;(pos = buffer.indexOf('\n')) >= 0;)
        {
            auto event = QJsonDocument::fromJson(buffer.left(pos)).object();
            buffer.remove(0,pos+1);
            if(event["type"] != "thread.started")
                continue;
            auto session = event["thread_id"].toString();
            if(session.isEmpty())
                continue;
            bool started_new_session = process->objectName().isEmpty();
            if(started_new_session)
            {
                process->setObjectName(session);
                ai_infos[session].set_provider(ai_provider::Codex,launch.name);
                ai_infos[session].set_process(process);
            }
            ai_infos[session].model_settings = launch.model_setting;
            if(started_new_session)
            {
                add_ai_history(session,"user",text);
                for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
                    button->setEnabled(true);
            }
            ai_log("start " + ai_infos[session].agent_name + " session:"+session);
        }
        process->setProperty("stdout_buffer",buffer);
    });

    QStringList args{"exec"};
    if(launch.model_provider == ai_model_provider::Ollama)
    {
        auto host = settings.value("ai/ollama_host","localhost").toString().trimmed();
        if(!host.contains("://"))
            host = "http://"+host;

        QUrl url(host);
        url.setPort(settings.value("ai/ollama_port",11434).toInt());
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
void MainWindow::start_ai(QString session,const QString& text,bool add_history)
{
    auto provider = session.isEmpty() ?
        ai_provider(ui->ai_agent_selector->currentIndex()) :
        ai_infos[session].provider;
    switch(provider)
    {
    case ai_provider::Codex:
        return start_codex(session,text,add_history);
    case ai_provider::Claude:
        return start_claude(session,text,add_history);
    default:
        QMessageBox::warning(this,"AI Agent","Unsupported AI provider.");
    }
}

void MainWindow::on_ai_send_message_clicked()
{
    auto text = ui->ai_chat_input->toPlainText().trimmed();
    if(text.isEmpty())
        return;
    auto* item = ui->ai_project_list->currentItem();
    auto session = item ? item->data(Qt::UserRole).toString() : QString();
    if(!session.isEmpty() && (ai_infos[session].processes ||
                              ai_infos[session].provider == ai_provider::Unknown))
    {
        ai_infos[session].prompts.append(text);
        add_ai_history(session,"user",text);
        ui->ai_chat_input->clear();
        return;
    }
    start_ai(session,text,true);
}
