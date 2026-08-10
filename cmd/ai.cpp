// ai_info's own data-layer implementation: session registry, on-disk history/config persistence, and
// history-entry recording. No UI/AIAgent dependency -- everything that touches ui->/agent_entries/etc.
// stays in ai_agent.cpp.
#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QSettings>
#include <QUrl>

#include "ai_agent.hpp"
#include "mainwindow.h"
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
            {"model_settings",model_settings}}).toJson(QJsonDocument::Compact));
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
QJsonObject ai_info::record_request(const QString& command_name,QString title)
{
    QJsonObject entry{{"type","request"},{"text",command_name},{"window",command_window_type(current_window)}};
    if(!title.isEmpty())
        entry["title"] = title;
    return record_history(entry);
}
