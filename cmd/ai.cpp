// ai_info's own data-layer implementation (declared in ai.hpp): session registry, on-disk history/config
// persistence, and history-entry recording, plus free helpers with no AIAgent/MainWindow dependency (they
// take/return plain Qt types like QWidget*/QLabel*, never Ui::AIAgent or an AIAgent member) -- everything that
// actually touches ui->/agent_entries/etc. stays in ai_agent.cpp (which includes ai_agent.hpp, not this file),
// and dispatch_cmd() builds "request" entries itself now.
// Global (free) functions first, ai_info's own member functions at the back.
#include <QColor>
#include <QComboBox>
#include <QDateTime>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QListWidgetItem>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSettings>
#include <QTimer>
#include <QUrl>
#include <QWidget>

#include <algorithm>
#include <cstring>
#include <utility>

#include "ai.hpp"
#include "TIPL/tipl.hpp"

std::unordered_map<QString,ai_info> ai_infos;
extern QString ai_project_dir;

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
    auto node = ai_infos.extract(from);
    if(node.empty())
        return ai_info::find(to);
    Q_ASSERT(!ai_info::find(to));
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

QUrl agent_install_url(ai_provider provider) // shared by the sidebar's Install button and a launch that finds the CLI missing, so the two can't drift apart
{
    return QUrl(provider == ai_provider::Codex ?
        "https://chatgpt.com/codex" : "https://claude.com/product/claude-code");
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
    // purely presentational: pulse means "advance," its absence means "reset to steady" -- whether that's
    // actually appropriate for this status is the caller's call (see ai_info::is_running()), not this function's
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

// shared look for the GitHub-setup/new-chat dialogs: Google-style cards/fields/buttons, scoped by objectName
// so it can't bleed into unrelated dialogs. Widgets sharing an objectName (e.g. every step card) all match
// the same rule -- that's normal Qt stylesheet behavior, not a lookup key.
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

// 401/403/404/410/422 mean the token/permissions/resource is wrong (retrying can't fix it); checked after github_retry_delay() since 403 can also mean rate limiting
bool github_permanent_failure(int status)
{
    return status == 401 || status == 403 || status == 404 ||
           status == 410 || status == 422;
}

// returns the wait time in ms if GitHub signals rate limiting (429, or 403 meaning the same), else 0
int github_retry_delay(QNetworkReply* reply,const QByteArray& data)
{
    bool ok = false;
    int seconds = reply->rawHeader("Retry-After").toInt(&ok);
    if(ok && seconds > 0)
        return seconds*1000;
    if(reply->rawHeader("X-RateLimit-Remaining") == "0")
    {
        qint64 reset = reply->rawHeader("X-RateLimit-Reset").toLongLong(&ok);
        if(ok)
            return int(std::max<qint64>(1000,reset*1000-QDateTime::currentMSecsSinceEpoch()));
    }
    auto status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if(status == 429 || (status == 403 && data.contains("rate limit")))
        return 60000;
    return 0;
}

// blocking helper: connect_github_issue is one-shot and user-initiated, so a short local event loop keeps its bool/error interface synchronous without added state
QByteArray github_blocking(QNetworkAccessManager& manager,
                            const QNetworkRequest& request,
                            const char* verb,const QByteArray& body,
                            bool& ok,QString& error)
{
    QEventLoop loop;
    QNetworkReply* reply =
        !strcmp(verb,"POST") ? manager.post(request,body) :
        !strcmp(verb,"PATCH") ? manager.sendCustomRequest(request,"PATCH",body) :
        manager.get(request);
    QObject::connect(reply,&QNetworkReply::finished,&loop,&QEventLoop::quit);
    loop.exec();
    ok = reply->error() == QNetworkReply::NoError;
    auto data = reply->readAll();
    if(!ok)
        error = reply->errorString();
    reply->deleteLater();
    return data;
}

QByteArray claude_input(const QString& text)
{
    return QJsonDocument(QJsonObject{
        {"type","user"},{"message",QJsonObject{
            {"role","user"},{"content",QJsonArray{QJsonObject{
                {"type","text"},{"text",text}}}}}}}).
        toJson(QJsonDocument::Compact)+'\n';
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
        return info.contains("provider") ? name+" (Ollama@"+ollama_host+")" : name;
    };
    model.clear();
    model.addItem("default");
    for(const auto& name : native_names+ollama_names)
        model.addItem(display_text(name,profiles[name].toObject()),profiles[name].toObject());

    auto target = selected.isEmpty() ? fallback : selected;
    int selected_index = -1;
    for(int i = 0;i < model.count() && selected_index < 0;++i)
        if(model.itemText(i) == target || model.itemText(i).startsWith(target+" ("))
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
    // gated on real content (projects), not live status -- status alone can't tell "never touched, nothing
    // to persist" apart from "was established, currently New again while reconnecting" (see session_status),
    // and skipping the latter meant edits made mid-reconnect (rename, cwd change) silently didn't persist,
    // and a first message that failed before establishing left a history file config.json couldn't explain
    // on reload. Files written under a still-New Codex placeholder are migrated to its real thread ID by
    // assign_ai_session(), so writing early under the placeholder id is safe
    if(projects.isEmpty() || !QSettings().value("ai/keep_history",true).toBool())
        return;
    QFile file(config_file(sessions));
    if(file.open(QIODevice::WriteOnly|QIODevice::Truncate))
        file.write(QJsonDocument(QJsonObject{
            {"agent",agent_name},
            {"provider",int(provider)}, // reload must trust this, not re-guess from agent_name -- that misclassifies an AgentServer session (or fails outright for a name identify_provider() doesn't recognize)
            {"model_settings",model_settings},
            // never reverts to false once true (New is the only live status this ever sees) -- reload trusts
            // this instead of assuming Completed for a session id that never actually got a real backend thread
            {"established",status != session_status::New}}).toJson(QJsonDocument::Compact));
}

ai_provider ai_info::identify_provider(const QString& name)
{
    // "chatgpt" checked first: the web agent's name is "Codex/ChatGPT-GitHub", which also contains "codex"
    return name.contains("chatgpt",Qt::CaseInsensitive) ? ai_provider::ChatGPT :
           name.contains("codex",Qt::CaseInsensitive) ? ai_provider::Codex :
           name.contains("claude",Qt::CaseInsensitive) ? ai_provider::Claude :
           ai_provider::Unknown;
}
QString ai_info::details() const
{
    int user = 0,assistant = 0,activity = 0;
    for(const auto& value : projects)
    {
        auto type = value["type"].toString();
        user += type == "user";
        assistant += type == "assistant";
        activity += type == "request" || type == "activity";
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
    if(projects.isEmpty()) // see save_config()
    {
        project_titles = title;
        return true;
    }
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
ai_info* ai_info::create(QString session,QString agent,ai_provider provider) // the one constructor for the whole registry
{
    if(provider == ai_provider::Infer)
        provider = identify_provider(agent);
    if(session.isEmpty() || provider < ai_provider::Codex ||
       provider > ai_provider::AgentServer)
        return nullptr;
    if(auto* info = find(session))
        return info;
    auto& info = ai_infos[session];
    info.sessions = std::move(session);
    info.provider = provider;
    info.agent_name = std::move(agent);
    return &info;
}

QJsonObject ai_info::record_history(QJsonObject entry)
{
    // written immediately regardless of status -- a message that's actually been recorded is real content,
    // not provisional. A New session's id is never reused for anything else even before it's confirmed
    // established (Codex's own placeholder gets renamed, not discarded, and assign_ai_session() already
    // migrates this exact file to the new name when that happens), so there's no "wrong file" risk to guard
    // against by waiting
    entry["time"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    projects.append(entry);
    if(QSettings().value("ai/keep_history",true).toBool())
    {
        QFile file(history_file(sessions));
        if(!file.open(QIODevice::WriteOnly|QIODevice::Append) ||
           file.write(QJsonDocument(entry).toJson(QJsonDocument::Compact)+'\n') < 0)
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
