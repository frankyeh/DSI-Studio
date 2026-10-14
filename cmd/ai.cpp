// ai_info's data layer (session registry, history/config persistence), free helpers, the Web channel, and the headless --action=ai entry point
#include <QColor>
#include <QComboBox>
#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QListWidgetItem>
#include <QSettings>
#include <QTimer>
#include <QUrl>
#include <QUuid>
#include <QWidget>

#include <algorithm>
#include <utility>

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QMimeDatabase>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>

#include <iostream>
#include <sstream>

#include "ai.hpp"
#include "TIPL/tipl.hpp"

std::unordered_map<QString,ai_info> ai_infos;
extern QString ai_project_dir;

bool is_valid_session_id(const QString& id)
{
    return !QUuid(id).toString(QUuid::WithoutBraces).compare(id,Qt::CaseInsensitive);
}

QString session_status_text(session_status status)
{
    switch(status)
    {
    case session_status::New:          return "New";
    case session_status::WaitingUser:  return "Waiting for user";
    case session_status::Thinking:     return "Thinking";
    case session_status::Completed:    return "Completed";
    case session_status::Failed:       return "Failed";
    }
    return {};
}

ai_info* assign_ai_session(const QString& from,const QString& to)
{
    if(from == to)
        return ai_info::find(to);
    if(ai_info::find(to)) // never re-key onto an existing chat: the failed insert would destroy the source
        return nullptr;
    auto node = ai_infos.extract(from);
    if(node.empty())
        return nullptr;
    node.key() = to;
    node.mapped().sessions = to;
    if(node.mapped().project_items)
        node.mapped().project_items->setData(Qt::UserRole,to);
    auto inserted = ai_infos.insert(std::move(node));
    auto move_file = [](const QString& source,const QString& target)
    {
        if(QFile::exists(source) && !QFile::rename(source,target))
            tipl::warning() << "cannot move " << source.toStdString()
                            << " to " << target.toStdString();
    };
    move_file(ai_info::history_file(from),ai_info::history_file(to));
    move_file(ai_info::config_file(from),ai_info::config_file(to));
    QSettings settings;
    if(!inserted.position->second.project_titles.isEmpty())
        settings.setValue("ai/title/"+to,inserted.position->second.project_titles);
    settings.remove("ai/title/"+from);
    return &inserted.position->second;
}

QUrl agent_install_url(const QString& provider) // shared by the sidebar's Install button and a launch that finds the CLI missing, so the two can't drift apart
{
    return QUrl(provider == "Codex" ? "https://chatgpt.com/codex" :
                provider == "Claude" ? "https://claude.com/product/claude-code" :
                provider == "Muse" ? "https://dev.meta.ai/docs/muse-code" :
                provider == "Antigravity" ? "https://antigravity.google/docs/cli/install/" :
                provider == "Grok" ? "https://github.com/xai-org/grok-build" :
                QString());
}

void stop_blink(QWidget* row)
{
    if(!row)
        return;
    row->findChild<QTimer*>()->stop();
    row->setStyleSheet({});
}

void update_status_dot(QLabel* dot,session_status status,bool pulse)
{
    if(!dot)
        return;
    // pulse advances the animation, otherwise it resets; the caller decides when (ai_info::is_running())
    int phase = dot->property("pulse").toInt();
    phase = pulse ? (phase+1)%24 : 0;
    dot->setProperty("pulse",phase);

    QColor color;
    switch(status)
    {
    case session_status::New:          color = "#9aa0a6"; break;
    case session_status::WaitingUser:  color = "#34a853"; break;
    case session_status::Thinking:     color = "#4285f4"; break;
    case session_status::Completed:    color = "#9aa0a6"; break;
    case session_status::Failed:       color = "#ea4335"; break;
    }
    int intensity = phase <= 12 ? phase : 24-phase;
    color = color.lighter(100+intensity*3);
    dot->setStyleSheet(QString("background-color:%1;border-radius:5px;").arg(color.name()));
    dot->setToolTip(session_status_text(status));
}

// shared look for the new-chat/settings dialogs, scoped by objectName so it cannot bleed into other dialogs
QString ai_dialog_style()
{
    return
        "QLabel#ai_dialog_title{font-size:15px;font-weight:600;color:#202124;}"
        "QLabel#ai_dialog_subtitle{color:#5f6368;}"
        "QFrame#ai_step_card{background-color:#f7f7f8;border:1px solid #dddddd;border-radius:10px;}"
        "QLabel#ai_step_heading{font-weight:600;color:#202124;}"
        "QLabel#ai_step_body{color:#3c4043;}"
        "QFrame#ai_field_frame{border:1px solid #d9d9dc;border-radius:10px;background-color:#ffffff;}"
        "QFrame#ai_field_frame QLineEdit{border:none;background:transparent;padding:6px 4px;}"
        "QLabel#ai_helper{color:#1a73e8;}"
        "QLineEdit{border:1px solid #d9d9dc;border-radius:7px;padding:5px 8px;background-color:#f7f7f8;}"
        "QLineEdit:focus{border-color:#8a8a8f;}"
        "QComboBox{border:1px solid #d9d9dc;border-radius:7px;padding:4px 24px 4px 8px;background-color:#f7f7f8;min-height:22px;}"
        "QComboBox:hover{background-color:#eeeeef;border-color:#c8c8cc;}"
        "QComboBox:focus{border-color:#8a8a8f;}"
        "QComboBox::drop-down{border:0;width:22px;}" // otherwise Qt draws the platform's native (raised/beveled) button here
        "QComboBox QAbstractItemView{background-color:#ffffff;border:1px solid #d9d9dc;outline:0;padding:2px;selection-background-color:#e5e5e7;selection-color:#202124;}"
        "QPushButton{color:#202124;background-color:#f4f4f5;border:1px solid #d9d9dc;border-radius:7px;padding:6px 14px;}"
        "QPushButton:hover{background-color:#e9e9eb;border-color:#c8c8cc;}"
        "QPushButton:pressed{background-color:#dddddf;}"
        "QPushButton:disabled{color:#9aa0a6;background-color:#f1f1f2;border-color:#e4e4e6;}"
        "QPushButton#ai_primary_button{background-color:#1a73e8;color:#ffffff;border:none;font-weight:600;}"
        "QPushButton#ai_primary_button:hover{background-color:#1765cc;}"
        "QPushButton#ai_primary_button:pressed{background-color:#175dc1;}"
        "QPushButton#ai_primary_button:disabled{background-color:#a8c7f0;color:#eef3fc;}";
}

QByteArray json_line(const QJsonObject& message)
{
    return QJsonDocument(message).toJson(QJsonDocument::Compact)+'\n';
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

QString model_combo_key(const QComboBox& model) // strips the " (Ollama@host)" suffix off an Ollama model's display text; "default" is a UI label only -- its data value is empty, the one universal representation of "no explicit choice"
{
    auto key = model.currentText().section(" (Ollama@",0,0);
    return key == "default" ? QString() : key;
}

void set_model_selector(QComboBox& model,const QJsonObject& profiles,
                        QString selected,QString fallback,
                        QJsonObject selected_info)
{
    // grouped, not one alphabetical sort: native models first, then Ollama models together as their own block
    QStringList native_names,ollama_names;
    for(const auto& name : profiles.keys())
        (profiles[name].toObject().contains("provider") ? ollama_names : native_names) << name;
    native_names.sort(Qt::CaseInsensitive);
    ollama_names.sort(Qt::CaseInsensitive);

    auto ollama_host = ai_ollama_url(QSettings()).first.host();
    auto display_text = [&](const QString& name,const QJsonObject& info)
    {
        return info.contains("provider") ?
               name+" (Ollama@"+(info.contains("url") ? QUrl(info["url"].toString()).host() : ollama_host)+")" : name;
    };
    model.clear();
    model.addItem("default");
    for(const auto& name : native_names+ollama_names)
        model.addItem(display_text(name,profiles[name].toObject()),profiles[name].toObject());

    auto target = selected.isEmpty() ? fallback : selected;
    int selected_index = -1;
    for(int i = 0;i < model.count() && selected_index < 0;++i)
        if((model.itemText(i) == target || model.itemText(i).startsWith(target+" (")) &&
           (selected_info.isEmpty() || model.itemData(i).toJsonObject().value("url") == selected_info.value("url"))) // same name on another Ollama server is a different model
            selected_index = i;
    if(selected_index < 0 && !selected.isEmpty())
    {
        model.addItem(display_text(selected,selected_info),selected_info);
        selected_index = model.count()-1;
    }
    model.setCurrentIndex(std::max(0,selected_index));
}

QString ai_info::history_file(const QString& session)
{
    return ai_project_dir+"/"+QString::fromLatin1(
               QUrl::toPercentEncoding(session))+".jsonl";
}
QString ai_info::config_file(const QString& session)
{
    return ai_project_dir+"/"+QString::fromLatin1(
               QUrl::toPercentEncoding(session))+".json";
}
void ai_info::save_config() const
{
    // gated on content, not status: a reconnecting chat is New again but must still save; files under a placeholder
    // id are migrated by assign_ai_session()
    if(projects.isEmpty() || !QSettings().value("ai/keep_history",true).toBool())
        return;
    QFile file(config_file(sessions));
    if(file.open(QIODevice::WriteOnly|QIODevice::Truncate))
        file.write(QJsonDocument(QJsonObject{
            {"agent",agent_name},
            {"provider",provider},
            {"model_settings",model_settings},
            // reload trusts this rather than assuming a never-established id is resumable
            {"established",status != session_status::New}}).toJson(QJsonDocument::Compact));
}

QString ai_info::details() const
{
    int user = 0,assistant = 0,activity = 0;
    for(const auto& value : projects)
    {
        auto type = value["type"].toString();
        user += type == "user";
        assistant += type == "assistant";
        activity += type == "request" || type == "activity" || type == "error";
    }
    auto time = [](const QJsonValue& value) {
        return QDateTime::fromString(value.toString(),Qt::ISODate).toString(
                   "yyyy-MM-dd HH:mm:ss");};
    auto created = projects.isEmpty() ? QString() : time(projects.first()["time"]);
    auto updated = projects.isEmpty() ? QString() : time(projects.last()["time"]);
    return QString("<b>%1</b><br><br>Agent: %2<br>Session: %3<br>Status: %4<br>"
        "Messages: %5 (%6 you, %7 AI)<br>Activities: %8<br>"
        "Created: %9<br>Updated: %10")
        .arg(title().toHtmlEscaped(),agent_name.toHtmlEscaped(),sessions.toHtmlEscaped(),session_status_text(status))
        .arg(user+assistant).arg(user).arg(assistant).arg(activity)
        .arg(created,updated);
}
bool ai_info::save_title(QString title)
{
    title = title.simplified();
    if(title.isEmpty())
        return false;
    if(title == project_titles)
        return true;
    QSettings settings;
    settings.setValue("ai/title/"+sessions,title);
    settings.sync();
    if(settings.status() != QSettings::NoError)
        return false;
    project_titles = title;
    return true;
}

ai_info* ai_info::find(const QString& session)
{
    auto found = ai_infos.find(session);
    return found == ai_infos.end() ? nullptr : &found->second;
}
ai_info* ai_info::create(QString session,QString provider,QString agent) // the one constructor for the whole registry
{
    if(session.isEmpty() || provider.isEmpty())
        return nullptr;
    if(auto* info = find(session))
        return info;
    if(agent.isEmpty())
        agent = provider;
    auto& info = ai_infos[session];
    info.sessions = std::move(session);
    info.provider = std::move(provider);
    info.agent_name = std::move(agent);
    return &info;
}

QJsonObject ai_info::record_history(QJsonObject entry)
{
    // written immediately regardless of status: a placeholder id's file is migrated by assign_ai_session()
    entry["time"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    projects.append(entry);
    if(QSettings().value("ai/keep_history",true).toBool())
    {
        QFile file(history_file(sessions));
        if(!file.open(QIODevice::WriteOnly|QIODevice::Append) ||
           file.write(json_line(entry)) < 0)
            tipl::warning() << "cannot write ai history : " << file.errorString().toStdString();
    }
    return entry;
}
QJsonObject ai_info::record_reply(const QString& chat,const QString& reasoning)
{
    if(chat.isEmpty() && reasoning.isEmpty())
        return {};
    QJsonObject entry{{"type","assistant"},{"text",chat}};
    if(!reasoning.isEmpty())
        entry["reasoning"] = reasoning;
    return record_history(entry);
}
QMap<QString,QString> shell_tasks;
bool ai_commands(const QJsonValue& command,std::vector<std::vector<std::string> >& cmds)
{
    for(const auto& value : (command.isArray() ? command.toArray() : QJsonArray{command}))
    {
        auto object = value.toObject();
        auto& cmd = cmds.emplace_back();
        auto add = [&](const QJsonValue& value){cmd.push_back(value.toVariant().toString().toUtf8().toStdString());};
        add(object["cmd"]);
        if(cmd[0].empty())
            return false;
        auto param = object["param"];
        if(param.isArray())
            for(const auto& value : param.toArray())
                add(value);
        else if(!param.isUndefined() && !param.isNull())
            add(param);
    }
    return true;
}
int run_action_with_wildcard(tipl::program_option<tipl::out>&);
bool ai_run_cli(const std::vector<std::string>& cmd,bool headless,std::string& error)
{
    auto fail = [&](const std::string& msg){error = msg;return false;};
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
    if(headless && (po.get("action") == "vis" || po.get("action") == "cnt" || po.get("action") == "ai"))
        return fail("--action=vis, cnt, and ai are not available in headless mode");
    if(run_action_with_wildcard(po))
        return fail("command line failed");
    po.check_end_param<tipl::warning>();
    return true;
}
bool ai_write_text_file(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error)
{
    auto fail = [&](const std::string& msg){error = msg;return false;};
    if(cmd.size() != 3 || cmd[1].empty())
        return fail("usage: write_text_file <filename> <content>");
    auto file_name = QFileInfo(QString::fromUtf8(cmd[1].c_str())).absoluteFilePath(); // relative to this chat's directory
    if(!approve("AI File Write Request","The AI agent wants to write "+QString::number(cmd[2].size())+
                   " bytes to this file, replacing it if it exists:\n\n"+file_name))
        return fail("user declined to write this file");
    QFile file(file_name); // binary: the UTF-8 content and its newlines are written exactly, no BOM
    if(!file.open(QIODevice::WriteOnly) || file.write(cmd[2].data(),qint64(cmd[2].size())) != qint64(cmd[2].size()))
        return fail("cannot write to "+file_name.toStdString());
    tipl::out() << "wrote " << cmd[2].size() << " bytes to " << file_name.toStdString();
    return true;
}
bool ai_run_shell(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error)
{
    auto fail = [&](const std::string& msg){error = msg;return false;};
    if(cmd.size() != 2 || cmd[1].empty())
        return fail("usage: run_shell <command>");
    QString text = QString::fromUtf8(cmd[1].c_str());
    QString program = text.section(' ',0,0);
    bool is_curl = !program.compare("curl",Qt::CaseInsensitive);
    if(is_curl)
        text.insert(program.length()," -s -S");
    if(!approve("AI Shell Command Request","The AI agent wants to run this shell command:\n\n"+text))
        return fail("user declined to run this shell command");
    if(!program.compare("cd",Qt::CaseInsensitive))
    {
        QString path = text.mid(program.length()).trimmed();
        if(path.size() >= 2 && path.startsWith('"') && path.endsWith('"'))
            path = path.mid(1,path.size()-2);
        if(!path.isEmpty() && !QDir::setCurrent(path))
            return fail(("cannot change directory to: "+path).toStdString());
        tipl::out() << QDir::currentPath().toStdString();
        return true;
    }
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
    if(!is_curl)
    {
        QProcess process;
        configure_process(process,text);
        process.start();
        if(!process.waitForStarted(3000))
            return fail("cannot start command");
        constexpr int shell_timeout_ms = 600000;
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
    static int next_curl_id = 0;
    QString id = "curl"+QString::number(++next_curl_id);
    auto* process = new QProcess;
    configure_process(*process,text);
    auto cleanup = [process,id]
    {
        shell_tasks.remove(id);
        process->deleteLater();
    };
    QObject::connect(process,&QProcess::errorOccurred,[id,text,cleanup](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
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
web_bridge::web_bridge(void)
{
    web_timer.setSingleShot(true);
    web_manager.setTransferTimeout(30000); // a hung Google request would otherwise stall polling for good
    connect(&web_timer,&QTimer::timeout,this,[this]{poll_web();});
}
void web_bridge::google_token_post(QList<QPair<QString,QString>> form,std::function<void(QString)> done)
{
    form.emplaceBack("client_id",google_device_auth ? GOOGLE_DEVICE_CLIENT_ID : GOOGLE_CLIENT_ID);
    form.emplaceBack("client_secret",google_device_auth ? GOOGLE_DEVICE_CLIENT_SECRET : GOOGLE_CLIENT_SECRET); // bundled in a desktop binary, so not confidential: PKCE + state (or the user code) protect the flow
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
        {
            settings.setValue("ai/google_refresh_token",google_refresh_token = json["refresh_token"].toString());
            settings.setValue("ai/google_device_auth",google_device_auth); // the client that refreshes this token
        }
        done({});
    });
}
void web_bridge::with_google_token(std::function<void(QString)> call)
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
void web_bridge::google_api(const QByteArray& verb,const QString& url,const QJsonObject& body,std::function<void(QJsonObject)> done)
{
    auto failed = [this,done]
    {
        if(!google_file_id.isEmpty()) // a live Web chat shows it; polling retries until the idle limit stops it
            report(session_status::Failed,"Web: "+google_error+"; retrying.");
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
void web_bridge::write_google_doc(const QJsonObject& doc,const QJsonObject& message,std::function<void(bool)> done)
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
void web_bridge::create_web_session(std::function<void(QString file,QString session)> done)
{
    if(google_folder_id.isEmpty()) // resolved once (found or created), then reused by its ID
        return google_api("GET","https://www.googleapis.com/drive/v3/files?fields=files(id)&q="+QString::fromLatin1(QUrl::toPercentEncoding(
                          "name='DSI Studio AI' and mimeType='application/vnd.google-apps.folder' and trashed=false")),{},[this,done](QJsonObject found)
        {
            auto use = [this,done](QJsonObject folder)
            {
                if(folder["id"].toString().isEmpty())
                    return warn("Cannot create the DSI Studio AI folder in Google Drive.");
                settings.setValue("ai/google_folder_id",google_folder_id = folder["id"].toString());
                create_web_session(done);
            };
            if(!found.contains("files")) // a failed search is not "no folder": never create a duplicate
                return warn("Cannot reach Google Drive.");
            if(auto files = found["files"].toArray();!files.isEmpty())
                return use(files[0].toObject());
            google_api("POST","https://www.googleapis.com/drive/v3/files?fields=id",
                       {{"name","DSI Studio AI"},{"mimeType","application/vnd.google-apps.folder"}},use);
        });
    auto session = QUuid::createUuid().toString(QUuid::WithoutBraces);
    google_api("POST","https://www.googleapis.com/drive/v3/files?fields=id",
               {{"name","DSI Studio "+session},{"mimeType","application/vnd.google-apps.document"},{"parents",QJsonArray{google_folder_id}}},
               [this,session,done](QJsonObject file)
    {
        if(file["id"].toString().isEmpty())
        {
            settings.setValue("ai/google_folder_id",google_folder_id = QString()); // the folder may have been deleted: recreate it next time
            return warn("Cannot create the Web session.");
        }
        done(file["id"].toString(),session);
    });
}
void web_bridge::start_web(const QString& file,const QString& session)
{
    stop_web();
    if(session != web_session_id) // a pending result belongs to its own session only
    {
        web_session_id = session;
        web_last_id = 0; // a claimed request is always overwritten by "processing", so it is never re-run
        web_pending_result = QJsonObject();
    }
    google_file_id = file;
    google_error = "the session Doc was never read"; // the first good read clears it and shows the chat connected
    report(session_status::WaitingUser,"Web: reading the session Doc.");
    web_idle.start();
    web_timer.start(0);
}
QString web_bridge::prompt() const
{
    return "Connect to DSI Studio. First read the public GitHub file "
           "frankyeh/DSI-Studio-AI/DSI_STUDIO_AI_SKILL_WEB.md and follow it. "
           "Session document: https://docs.google.com/document/d/"+google_file_id+"/edit";
}
void web_bridge::stop_web(const QString& message)
{
    if(google_file_id.isEmpty())
        return;
    google_file_id.clear(); // callbacks still in flight see this and stop
    web_timer.stop();
    report(session_status::Completed,message);
}
QString web_bridge::google_doc_url() const // only what write_google_doc() and the mailbox need
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
void web_bridge::poll_web(const QString& modified_time)
{
    if(google_file_id.isEmpty())
        return;
    if(!web_pending_result.isEmpty())
        return publish_web_result(); // a previous result write failed; retry it, never re-execute
    auto idle = web_idle.elapsed();
    if(stop_when_idle && idle > 3600000) // also bounds failing reads; headless Web serves until the process ends
        return stop_web(google_error.isEmpty() ? QString("Web stopped after 1 hour without a request; press Resume to continue.") :
                        "Web stopped: "+google_error+"; press Resume to retry.");
    // idle for 3 minutes: Drive modifiedTime (not file.version) is polled, and the body is read only when it changes; the first
    // idle poll always reads, since web_modified_time is empty after any fast-mode read. A Docs body edit can take minutes to
    // reach modifiedTime, but a rename reaches it at once, so an agent that gets no reply renames the Doc to wake DSI Studio
    if(modified_time.isEmpty() && idle >= 180000)
        return google_api("GET","https://www.googleapis.com/drive/v3/files/"+google_file_id+"?fields=modifiedTime",{},
                          [this,file = google_file_id,idle](QJsonObject meta)
        {
            if(file != google_file_id)
                return;
            if(meta.isEmpty())
                return web_timer.start(5000);
            if(meta["modifiedTime"].toString() != web_modified_time || !google_error.isEmpty()) // a failure clears on the next body read
                return poll_web(meta["modifiedTime"].toString());
            web_timer.start(idle < 300000 ? 5000 : 10000);
        });
    // the body is one tiny JSON message, so during activity it is read directly
    google_api("GET",google_doc_url(),{},[this,file = google_file_id,modified_time](QJsonObject doc)
    {
        if(file != google_file_id) // stopped or switched while in flight
            return;
        if(doc.isEmpty()) // failed, and shown by google_api()
            return web_timer.start(5000);
        if(!google_error.isEmpty()) // the first good read after a start, Resume, or failure
        {
            google_error.clear();
            report(session_status::WaitingUser,"Web connected; waiting for a request.");
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
        if(request["from"].toString() == "dsi") // our own write, nothing to do: only now is the Drive change consumed
        {
            web_modified_time = modified_time;
            return web_timer.start(web_idle.elapsed() < 60000 ? 500 : web_idle.elapsed() < 120000 ? 1000 : 2000); // 500 ms, 1 s, then 2 s per minute until idle
        }
        web_idle.start(); // an agent message: fast polling while it is claimed or answered
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
            auto response = dispatch(forwarded);
            web_pending_result = QJsonObject{{"dsi_bridge",true},{"session",web_session_id},{"id",id},{"from","dsi"},
                {"state",response["status"].toString() == "error" ? "error" : "done"},{"response",response}};
            publish_web_result();
        });
    });
}
void web_bridge::publish_web_result()
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
                web_idle.start(); // fast polling for 3 minutes after a result
            }
            web_timer.start(published ? 500 : 5000);
        });
    });
}
QJsonObject web_bridge::upload_web_file(const QString& file_name,QString& error)
{
    QFileInfo file_info(file_name); // relative to the current (chat) working directory
    QFile file(file_info.absoluteFilePath());
    if(google_file_id.isEmpty() || google_folder_id.isEmpty())
        error = "web_upload is available only in the active Web session";
    else if(!file_info.exists())
        error = "file does not exist: "+file_name;
    else if(!file_info.isFile())
        error = "not a regular file: "+file_name;
    else if(!file.open(QIODevice::ReadOnly))
        error = "cannot open file: "+file_name;
    if(!error.isEmpty())
        return {};
    auto mime = QMimeDatabase().mimeTypeForFile(file_info).name(); // application/octet-stream when unknown
    // one resumable upload for every size: the file is streamed from disk, never held in memory
    QJsonObject uploaded;
    bool done = false;
    QEventLoop loop;
    auto finish = [&](QNetworkReply* reply)
    {
        auto json = QJsonDocument::fromJson(reply ? reply->readAll() : QByteArray()).object();
        if(!reply || reply->error() != QNetworkReply::NoError)
            error = "Google Drive upload failed: "+(reply ? json["error"].toObject()["message"].toString(reply->errorString()) : google_error);
        uploaded = json;
        done = true;
        loop.quit();
    };
    with_google_token([&](QString token)
    {
        if(token.isEmpty())
            return finish(nullptr);
        QNetworkRequest start(QUrl("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id"));
        start.setRawHeader("Authorization","Bearer "+token.toUtf8());
        start.setHeader(QNetworkRequest::ContentTypeHeader,"application/json; charset=UTF-8");
        start.setRawHeader("X-Upload-Content-Type",mime.toUtf8());
        start.setRawHeader("X-Upload-Content-Length",QByteArray::number(file.size()));
        auto* session = web_manager.post(start,QJsonDocument(QJsonObject{{"name",web_session_id+"_"+file_info.fileName()},
                                                                         {"parents",QJsonArray{google_folder_id}}}).toJson(QJsonDocument::Compact));
        connect(session,&QNetworkReply::finished,this,[&,session,token]
        {
            session->deleteLater();
            if(session->error() != QNetworkReply::NoError || !session->hasRawHeader("Location"))
                return finish(session);
            QNetworkRequest put{QUrl::fromEncoded(session->rawHeader("Location"))};
            put.setRawHeader("Authorization","Bearer "+token.toUtf8());
            put.setHeader(QNetworkRequest::ContentTypeHeader,mime);
            auto* upload = web_manager.put(put,&file); // the 30 s timeout counts idle time, so a long upload still completes
            connect(upload,&QNetworkReply::finished,this,[&,upload]
            {
                upload->deleteLater();
                finish(upload);
            });
        });
    });
    if(!done) // with_google_token() may already have finished
        loop.exec();
    if(!error.isEmpty())
        return {};
    auto id = uploaded["id"].toString();
    return {{"name",file_info.fileName()},{"drive_id",id},{"mime_type",mime},{"size",file.size()},
            {"url","https://drive.google.com/file/d/"+id+"/view"}};
}
bool web_bridge::sign_in_device() // Google's device flow, authorized on any computer or phone; no callback port
{
    QString error;
    google_device_auth = true; // the device client issues, and later refreshes, this token
    QNetworkRequest request(QUrl("https://oauth2.googleapis.com/device/code"));
    request.setHeader(QNetworkRequest::ContentTypeHeader,"application/x-www-form-urlencoded");
    auto* reply = web_manager.post(request,"client_id="+QUrl::toPercentEncoding(GOOGLE_DEVICE_CLIENT_ID)+
                                   "&scope="+QUrl::toPercentEncoding("https://www.googleapis.com/auth/drive.file"));
    QEventLoop loop;
    connect(reply,&QNetworkReply::finished,&loop,&QEventLoop::quit);
    loop.exec();
    reply->deleteLater();
    auto code = QJsonDocument::fromJson(reply->readAll()).object(); // device_code is never printed
    if(!code.contains("device_code"))
        return warn("Google sign-in failed: "+code["error"].toString(reply->errorString())),false;
    std::cout << "Google authorization required.\n\nOpen this URL on your personal computer:\n" << code["verification_url"].toString().toStdString()
              << "\n\nCode:\n" << code["user_code"].toString().toStdString() << "\n\nWaiting for Google authorization..." << std::endl;
    auto interval = code["interval"].toInt(5);
    auto deadline = QDateTime::currentDateTimeUtc().addSecs(code["expires_in"].toInt(1800));
    do
    {
        QTimer::singleShot(interval*1000,&loop,&QEventLoop::quit);
        loop.exec();
        google_token_post({{"grant_type","urn:ietf:params:oauth:grant-type:device_code"},{"device_code",code["device_code"].toString()}},
                          [&](QString token_error){error = token_error;loop.quit();});
        loop.exec();
        if(error == "slow_down")
            interval += 5;
    }
    while((error == "authorization_pending" || error == "slow_down") && QDateTime::currentDateTimeUtc() < deadline);
    if(!error.isEmpty())
        return warn("Google sign-in failed: "+error),false;
    std::cout << "Google authorization completed." << std::endl;
    settings.setValue("ai/google_folder_id",google_folder_id = QString()); // the account may have changed: resolve its folder again
    return true;
}
int ai(tipl::program_option<tipl::out>& po) // --action=ai --agent=web: a headless Web session served until Ctrl+C; no window is created
{
    if(po.get("agent") != "web")
        return tipl::error() << "--action=ai supports only --agent=web",1;
    web_bridge web;
    web.stop_when_idle = false;
    web.status = [](session_status,const QString& message){tipl::out() << message.toStdString();};
    web.warn = [](const QString& text){tipl::error() << text.toStdString();QCoreApplication::exit(1);};
    web.dispatch = [&web](const QJsonObject& request)
    {
        if(auto chat = request["chat"].toString().trimmed();!chat.isEmpty())
            std::cout << "\nThe AI agent says: " << chat.toStdString() << std::endl;
        auto approve = [&](const QString& title,const QString& text) // the terminal user approves; no terminal declines
        {
            std::cerr << "\n" << title.toStdString() << "\n" << text.toStdString() << "\nApprove? [y/N] " << std::flush;
            std::string answer;
            return std::getline(std::cin,answer) && (answer == "y" || answer == "Y");
        };
        std::vector<std::vector<std::string> > cmds;
        QJsonArray results;
        if(auto command = request["command"];!command.isUndefined() && !command.isNull() && !ai_commands(command,cmds))
            results.append(QJsonObject{{"status","error"},{"error","invalid cmd text"}});
        for(const auto& cmd : cmds)
        {
            std::string error;
            QString upload_error;
            QJsonObject attachment;
            std::ostringstream out;
            auto* cout_buf = std::cout.rdbuf(out.rdbuf()); // the command's output is returned to the agent
            bool ok = true;
            if(cmd[0] == "run_cli")
                ok = ai_run_cli(cmd,true,error);
            else if(cmd[0] == "run_shell")
                ok = ai_run_shell(cmd,approve,error);
            else if(cmd[0] == "write_text_file")
                ok = ai_write_text_file(cmd,approve,error);
            else if(cmd[0] == "web_upload")
                ok = cmd.size() == 2 && !(attachment = web.upload_web_file(QString::fromStdString(cmd[1]),upload_error)).isEmpty();
            else if(cmd[0] == "set_title") // saved like a GUI chat's title
            {
                if(!(ok = cmd.size() == 2 && ai_info::create(web.web_session_id,"Web","Web")->save_title(QString::fromStdString(cmd[1]))))
                    error = "usage: set_title <title>";
            }
            else
                ok = false,error = "not available in headless mode (--action=ai); use run_cli, run_shell, write_text_file, web_upload, or set_title";
            std::cout.rdbuf(cout_buf);
            std::cout << out.str() << std::flush;
            if(cmd[0] == "web_upload" && !ok)
                error = cmd.size() == 2 ? upload_error.toStdString() : "usage: web_upload <local file>";
            QJsonObject result{{"cmd",QString::fromStdString(cmd[0])},{"status",ok ? "success" : "error"}};
            if(!out.str().empty())
                result["output"] = QString::fromStdString(out.str());
            if(!attachment.isEmpty())
                result["attachment"] = attachment;
            if(!ok)
                result["error"] = QString::fromStdString(error)+". Read DSI Studio Manuals and retry.";
            results.append(result);
            if(!ok)
                break;
        }
        return QJsonObject{{"status",results.isEmpty() ? "success" : results.last().toObject()["status"].toString()},{"result",results}};
    };
    if(web.google_refresh_token.isEmpty() && !web.sign_in_device())
        return 1;
    web.create_web_session([&web](QString file,QString session)
    {
        web.start_web(file,session);
        std::cout << "\nDSI Studio Web AI session ready.\n\nPaste this prompt into a web AI agent:\n\n" << web.prompt().toStdString()
                  << "\n\nDSI Studio is running. Press Ctrl+C to stop." << std::endl;
    });
    return QCoreApplication::exec();
}
