// ai_info's data layer (session registry, history/config persistence) and free helpers with no AIAgent/MainWindow dependency
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

QByteArray claude_input(const QString& text)
{
    return json_line({
        {"type","user"},{"message",QJsonObject{
            {"role","user"},{"content",QJsonArray{QJsonObject{
                {"type","text"},{"text",text}}}}}}});
}

QByteArray codex_turn_start(const QString& id,const QString& thread_id,const QString& text)
{
    return json_line({{"id",id},{"method","turn/start"},
        {"params",QJsonObject{{"threadId",thread_id},
            {"input",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}}}});
}

QByteArray codex_turn_steer(const QString& thread_id,const QString& turn_id,const QString& text)
{
    return json_line({{"id","turn_steer"},{"method","turn/steer"},
        {"params",QJsonObject{{"threadId",thread_id},{"expectedTurnId",turn_id},
            {"input",QJsonArray{QJsonObject{{"type","text"},{"text",text}}}}}}});
}

QByteArray codex_turn_interrupt(const QString& thread_id,const QString& turn_id)
{
    return json_line({{"id","turn_interrupt"},{"method","turn/interrupt"},
        {"params",QJsonObject{{"threadId",thread_id},{"turnId",turn_id}}}});
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