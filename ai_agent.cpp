#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
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
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QShortcut>
#include <QShowEvent>
#include <QSpinBox>
#include <QStackedLayout>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QTextFrame>
#include <QTimer>
#include <QToolButton>
#include <QUuid>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "ai_agent.hpp"
#include "ui_ai_agent.h"
#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "TIPL/tipl.hpp"

std::unordered_map<QString,ai_info> ai_infos;
QString ai_project_dir;
constexpr qsizetype ai_debug_truncate_length = 300; // level 1 (truncated) caps each logged line to this many characters
// "ai/debug" setting: 0 = disabled, 1 = enabled (truncated), 2 = enabled (complete)
int& ai_debug_level()
{
    static int level = QSettings().value("ai/debug",0).toInt();
    return level;
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
static bool is_new_chat(const QString& session)
{
    return session.startsWith("new:");
}
void ai_info::save_config() const
{
    if(is_new_chat(sessions) || !QSettings().value("ai/keep_history",true).toBool())
        return;
    QFile file(config_file(sessions));
    if(file.open(QIODevice::WriteOnly|QIODevice::Truncate))
        file.write(QJsonDocument(QJsonObject{
            {"agent",agent_name},
            {"model_settings",model_settings}}).toJson(QJsonDocument::Compact));
}
static ai_info* assign_ai_session(const QString& from,const QString& to)
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
    if(!inserted.position->second.project_titles.isEmpty())
        QSettings().setValue("ai/title/"+to,inserted.position->second.project_titles);
    return &inserted.position->second;
}
struct ai_launch{
    QString name,executable,model;
    QUrl model_url;
    QJsonObject model_setting;
    QProcess* process = nullptr;
};
QByteArray claude_input(const QString& text)
{
    return QJsonDocument(QJsonObject{
        {"type","user"},{"message",QJsonObject{
            {"role","user"},{"content",QJsonArray{QJsonObject{
                {"type","text"},{"text",text}}}}}}}).
        toJson(QJsonDocument::Compact)+'\n';
}
static void stop_blink(QWidget* row)
{
    if(!row)
        return;
    row->findChild<QTimer*>()->stop();
    row->setStyleSheet({});
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
        .arg(title().toHtmlEscaped(),agent_name.toHtmlEscaped(),sessions.toHtmlEscaped(),processes ? "Working" : "Idle")
        .arg(user+assistant).arg(user).arg(assistant).arg(activity)
        .arg(created,updated);
}
void ai_log(QString text)
{
    if(ai_debug_level() <= 0)
        return;
    if(ai_debug_level() == 1 && text.size() > ai_debug_truncate_length)
        text = text.left(ai_debug_truncate_length)+"...";
    auto prefix = QString("[DEBUG] ");
    tipl::out() << (prefix+text.remove('\r').
                    replace('\n',"\n"+prefix)).toStdString();
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
QString model_combo_key(const QComboBox& model) // strips the " (Ollama@host)" suffix off an Ollama model's display text
{
    return model.currentText().section(" (Ollama@",0,0);
}
void set_model_selector(QComboBox& model,const QJsonObject& profiles,
                        QString selected = {},QString fallback = {},
                        QJsonObject selected_info = {})
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
// headless equivalent of set_model_selector's findText-then-fallback-then-default logic, without building a list
QPair<QString,QJsonObject> resolve_model(const QJsonObject& profiles,
                                         const QString& selected,const QString& fallback,
                                         const QJsonObject& selected_info)
{
    auto search = selected.isEmpty() ? fallback : selected;
    if(search == "default" || (!search.isEmpty() && profiles.contains(search)))
        return {search,search == "default" ? QJsonObject() : profiles[search].toObject()};
    if(!selected.isEmpty())
        return {selected,selected_info};
    return {"default",QJsonObject()};
}
bool ai_info::save_title(QString title)
{
    title = title.simplified();
    if(title.isEmpty())
        return false;
    if(title == project_titles)
        return true;
    if(is_new_chat(sessions))
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
AIAgent::AIAgent(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::AIAgent)
{
    ui->setupUi(this);
    ui->ai_work_dir->setText(main_window.work_dir());
    // keeps the field in sync with the selected chat's own dispatch directory (model_settings["cwd"]),
    // the same value run_shell's "cd" updates; also used as --add-dir when launching Codex/Claude
    auto sync_work_dir = [this]
    {
        auto* item = ui->ai_project_list->currentItem();
        if(!item)
            return;
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        auto cwd = ui->ai_work_dir->text().trimmed();
        if(info.model_settings["cwd"].toString() != cwd)
        {
            info.model_settings["cwd"] = cwd;
            info.save_config();
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
        if(ai_status_timer->isSingleShot())
            return set_ai_status();
        auto status = ui->ai_status->text();
        ui->ai_status->setText(
            status.endsWith("...") ? status.chopped(2) : status+".");
        ui->ai_status->repaint();
    });
    set_ai_status();

    github_timer.setSingleShot(true);
    connect(&github_timer,&QTimer::timeout,this,&AIAgent::poll_github_issue);

    QString codex_path,claude_path;
    {
        // Find Codex executable
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
    }
    {
        // Find Claude executable
        claude_path = QStandardPaths::findExecutable("claude");
#ifdef Q_OS_WIN
        if(claude_path.isEmpty())
            claude_path = QDir::homePath()+"/.local/bin/claude.exe";
#endif
        if(!QFileInfo::exists(claude_path))
            claude_path.clear();
    }
    static const char* agent_names[] = {"Codex","Claude"};
    for(auto provider : {ai_provider::Codex,ai_provider::Claude})
    {
        auto index = int(provider);
        const auto& path =
            provider == ai_provider::Codex ? codex_path : claude_path;
        QString agent = agent_names[index];
        agent_entries[index].executable = path;
        ai_log(path.isEmpty() ? agent+" not found" : agent+": "+path);
    }

    if(!claude_path.isEmpty())
    {
        // claude has no equivalent of "codex debug models" to query live, so use its known model aliases
        static const QStringList claude_models{"fable","opus","sonnet","haiku"};
        update_agent_models(int(ai_provider::Claude),claude_models,false);
        ai_log("Claude models: "+claude_models.join(", "));
    }

    if(codex_path.isEmpty() && !claude_path.isEmpty())
        current_agent_index = int(ai_provider::Claude);
    update_agent_status_label();

    auto* send = new QShortcut(
        QKeySequence(Qt::CTRL|Qt::Key_Return),ui->ai_chat_input);
    send->setContext(Qt::WidgetShortcut);
    connect(send,&QShortcut::activated,
            ui->ai_send_message,&QPushButton::click);
    connect(ui->ai_chat_input,&QPlainTextEdit::textChanged,
            this,&AIAgent::update_send_button);

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
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        bool okay;
        auto title = QInputDialog::getText(
            this,"Rename Chat","Chat name:",QLineEdit::Normal,
            info.title(),&okay);
        if(okay && info.save_title(title))
            show_ai_project(info);
        else if(okay)
            QMessageBox::warning(
                this,"Rename Chat","The chat name could not be saved.");
    });
    connect(ai_project_menu->addAction("Details..."),
            &QAction::triggered,this,[this]
    {
        auto* item = ui->ai_project_list->currentItem();
        QMessageBox details(
            QMessageBox::Information,"Chat Details",
            ai_infos[item->data(Qt::UserRole).toString()].details(),
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
        if(auto* process = ai_infos[session].processes)
        {
            process->disconnect(); process->kill(); process->deleteLater(); // kill(): a windowless console child never sees terminate()'s WM_CLOSE
            active_ai_processes = std::max(0,active_ai_processes-1);
            set_ai_status();
        }
        if(session == web_agent_session_id && !github_issue_api.isEmpty())
            disconnect_github_issue(); // otherwise the channel keeps polling and recreates this chat on the next request
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
            if(i) // itemWidget() is null for an item already detached from the list (e.g. mid-removal), so guard both calls
                if(auto* widget = ui->ai_project_list->itemWidget(i))
                    if(auto* button = widget->findChild<QPushButton*>())
                        button->setStyleSheet(i == item ?
                            "color:#202124;background:#dce9f9;" : "");
        if(!item)
        {
            ui->ai_chat_history->clear();
            update_send_button();
            return set_ai_status();
        }

        stop_blink(ui->ai_project_list->itemWidget(item));
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        ui->ai_work_dir->setText(info.model_settings.contains("cwd") ?
            info.model_settings["cwd"].toString() : main_window.work_dir());
        if(info.provider != ai_provider::ChatGPT) // a web chat has no agent/model of its own to adopt as "current"
        {
            current_agent_index = int(info.provider);
            auto resolved = resolve_model(
                agent_entries[current_agent_index].profiles,
                info.model_settings["model"].toString(),{},
                info.model_settings["info"].toObject());
            current_model_name = resolved.first;
            current_model_info = resolved.second;
        }
        update_agent_status_label();
        show_ai_project(info);
        update_send_button();
        set_ai_status();
    });

    QString resume_session,resume_url;
    qint64 resume_config_time = 0;
    for(const auto& info : dir.entryInfoList(
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
        auto first = history.first();
        QJsonObject config;
        QFileInfo config_info(ai_info::config_file(session));
        if(QFile config_file(config_info.filePath());config_file.open(QIODevice::ReadOnly))
            config = QJsonDocument::fromJson(config_file.readAll()).object();
        // config_file() is the current source of truth; fall back to the legacy fields once
        // embedded in the first history entry, for chats saved before this file existed
        auto* ai = ai_info::create(session,
            config.contains("agent") ? config["agent"].toString() : first["agent"].toString());
        if(!ai)
            continue;
        ai->model_settings = config.contains("model_settings") ?
            config["model_settings"].toObject() : first["model_settings"].toObject();
        ai->project_titles = settings.value("ai/title/"+session).toString();
        ai->projects = std::move(history);
        show_ai_project(*ai);

        // among all sessions with a bound issue, resume whichever was bound most recently
        // (config_file() is rewritten only on a deliberate agent/model/channel change, never
        // by ordinary chat activity, so its mtime is a clean "last touched" signal)
        auto issue_url = ai->model_settings["github_issue_url"].toString();
        if(!issue_url.isEmpty() && config_info.lastModified().toMSecsSinceEpoch() > resume_config_time)
        {
            resume_session = session;
            resume_url = issue_url;
            resume_config_time = config_info.lastModified().toMSecsSinceEpoch();
        }
    }
    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);

    // auto-resume the GitHub issue channel that was still connected when DSI Studio last closed
    if(!resume_session.isEmpty())
        QTimer::singleShot(0,this,[this,resume_session,resume_url] // deferred: avoids blocking startup on the network round-trips inside connect_github_issue
        {
            web_agent_session_id = resume_session;
            try_connect_github_issue(resume_url,true);
        });
}

AIAgent::~AIAgent()
{
    delete ui;
}

// GitHub issue channel: the issue body carries the next request; one pinned comment (marked "dsi_session_result":true) carries the result
QNetworkRequest AIAgent::github_request(const QUrl& url) const
{
    QNetworkRequest request(url);
    request.setRawHeader("Authorization",("Bearer "+github_token).toUtf8());
    request.setRawHeader("Accept","application/vnd.github+json");
    request.setRawHeader("X-GitHub-Api-Version","2022-11-28");
    request.setRawHeader("User-Agent","DSI-Studio");
    request.setTransferTimeout(15000); // applies to every GET/POST/PATCH, blocking or async
    return request;
}

// 401/403/404/410/422 mean the token/permissions/resource is wrong (retrying can't fix it); checked after github_retry_delay() since 403 can also mean rate limiting
static bool github_permanent_failure(int status)
{
    return status == 401 || status == 403 || status == 404 ||
           status == 410 || status == 422;
}

// returns the wait time in ms if GitHub signals rate limiting (429, or 403 meaning the same), else 0
static int github_retry_delay(QNetworkReply* reply,const QByteArray& data)
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
static QByteArray github_blocking(QNetworkAccessManager& manager,
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

bool AIAgent::connect_github_issue(const QString& url_text,QString& error)
{
    // snapshot now, so a later AI Settings edit can't swap the identity mid-poll (github_request() uses this member for the whole session)
    github_token = settings.value("ai/github_token").toString().trimmed();
    if(github_token.isEmpty())
        return error = "no GitHub token configured; set one in AI Settings first "
                        "(GitHub requires an authenticated request for every write, "
                        "including editing a comment on a public issue)",false;

    QUrl url(url_text.trimmed());
    if(!url.isValid() || url.scheme().compare("https",Qt::CaseInsensitive) ||
       url.host().compare("github.com",Qt::CaseInsensitive))
        return error = "expected an https://github.com/... issue link",false;

    auto parts = url.path().split('/',Qt::SkipEmptyParts);
    bool number_ok = false;
    qint64 issue_number = parts.size() == 4 ? parts[3].toLongLong(&number_ok) : 0;
    if(parts.size() != 4 || parts[2] != "issues" || !number_ok || issue_number <= 0)
        return error = "expected the form https://github.com/<owner>/<repository>/issues/<number>",false;

    QString owner = parts[0];
    QUrl issue_api("https://api.github.com/repos/"+owner+"/"+parts[1]+
                   "/issues/"+QString::number(issue_number));
    ai_log("github connect: verifying token");

    // identify who the token belongs to (need not be the repo owner); result-comment ownership is checked against this identity, not the issue's owner
    bool ok = false;
    auto authenticated_user = QJsonDocument::fromJson(
        github_blocking(github_manager,github_request(QUrl("https://api.github.com/user")),
                         "GET",{},ok,error)).object()["login"].toString();
    if(!ok)
        return error = "cannot verify GitHub token: "+error,false;
    if(authenticated_user.isEmpty())
        return error = "cannot verify GitHub token: unexpected response from GitHub",false;
    ai_log("github connect: token belongs to "+authenticated_user+"; fetching issue "+issue_api.toString());

    auto issue = QJsonDocument::fromJson(
        github_blocking(github_manager,github_request(issue_api),"GET",{},ok,error)).object();
    if(!ok)
    {
        ai_log("github connect: fetching issue failed: "+error);
        error += " (check that this token has access to this specific repository, e.g. a fine-grained PAT scoped to a different repo)";
        return false;
    }
    ai_log("github connect: issue fetched, state="+issue["state"].toString()+
           " owner="+issue["user"].toObject()["login"].toString());

    if(issue.contains("pull_request"))
        return error = "the link points to a pull request, not an issue",false;
    if(issue["state"].toString() != "open")
        return error = "issue is not open",false;
    if(issue["user"].toObject()["login"].toString().compare(owner,Qt::CaseInsensitive))
        return error = "issue creator must be the repository owner",false;
    if(!issue["title"].toString().startsWith("DSI Studio session"))
        return error = "issue title must start with \"DSI Studio session\"",false;

    ai_log("github connect: fetching comments");
    QJsonArray comments;
    for(int page = 1;;++page)
    {
        auto batch = QJsonDocument::fromJson(
            github_blocking(github_manager,github_request(QUrl(
                issue_api.toString()+"/comments?per_page=100&page="+QString::number(page))),
                "GET",{},ok,error)).array();
        if(!ok)
        {
            ai_log("github connect: fetching comments failed: "+error);
            error += " (check that this token has access to this specific repository)";
            return false;
        }
        for(const auto& comment : batch)
            comments.append(comment);
        if(batch.size() < 100)
            break;
    }
    ai_log("github connect: "+QString::number(comments.size())+" comment(s) fetched");

    // find our own result comment (author must match the token's identity); if more than one matches, keep the highest last_id rather than just the first
    QUrl result_api;
    qint64 last_id = -1;
    for(const auto& each : comments)
    {
        auto comment = each.toObject();
        if(comment["user"].toObject()["login"].toString().compare(authenticated_user,Qt::CaseInsensitive))
            continue;
        auto body = QJsonDocument::fromJson(comment["body"].toString().toUtf8());
        if(!body.isObject() || !body.object()["dsi_session_result"].toBool())
            continue;
        auto candidate_id = body.object()["last_id"].toInteger();
        if(candidate_id > last_id)
        {
            last_id = candidate_id;
            result_api = QUrl(comment["url"].toString());
        }
    }
    if(result_api.isEmpty())
    {
        last_id = 0; // fresh session, no matching comment found
        QJsonObject initial{{"state","idle"},{"last_id",0},{"dsi_session_result",true},{"issue",issue_number}};
        QJsonObject post_body{{"body",QString::fromUtf8(QJsonDocument(initial).toJson(QJsonDocument::Compact))}};
        auto post_request = github_request(QUrl(issue_api.toString()+"/comments"));
        post_request.setRawHeader("Content-Type","application/json");
        auto created = QJsonDocument::fromJson(
            github_blocking(github_manager,post_request,"POST",
                             QJsonDocument(post_body).toJson(QJsonDocument::Compact),ok,error)).object();
        if(!ok)
        {
            ai_log("github connect: creating result comment failed: "+error);
            error += " (check that this token has write access to this specific repository)";
            return false;
        }
        result_api = QUrl(created["url"].toString()); // GitHub's canonical .../issues/comments/<id> form
        if(result_api.isEmpty())
            return error = "cannot create the result comment",false;
    }

    ++github_connection_id; // supersedes any callback still in flight from before
    github_issue_api = issue_api;
    github_result_api = result_api;
    github_etag.clear();
    github_last_id = last_id;
    github_pending_result = QJsonObject();
    github_timer.start(500);
    update_send_button();
    return true;
}

void AIAgent::disconnect_github_issue()
{
    ++github_connection_id; // reject any callback still in flight from this connection
    github_timer.stop();
    github_issue_api.clear();
    github_result_api.clear();
    github_etag.clear();
    github_token.clear();
    github_last_id = 0;
    github_pending_result = QJsonObject();
    update_send_button(); // flips to "Resume" if still in a web-agent session
    if(auto* info = ai_info::find(web_agent_session_id)) // dot reverts from Active immediately, not just on the next poll
        show_ai_project(*info);
}

void AIAgent::poll_github_issue()
{
    if(github_issue_api.isEmpty())
        return;

    github_timer.stop(); // at most one poll or publish in flight at a time

    if(!github_pending_result.isEmpty())
        return send_pending_result(); // a previous PATCH failed; retry it, never re-execute

    auto request = github_request(github_issue_api);
    if(!github_etag.isEmpty())
        request.setRawHeader("If-None-Match",github_etag);

    auto connection_id = github_connection_id;
    auto* reply = github_manager.get(request);
    connect(reply,&QNetworkReply::finished,this,[this,reply,connection_id]()
    {
        reply->deleteLater();
        if(connection_id != github_connection_id)
            return; // this connection was superseded (disconnect, or a fresh reconnect)

        auto restart = [this](int delay_ms = 500)
        {if(!github_issue_api.isEmpty()) github_timer.start(delay_ms);};

        auto status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        auto data = reply->readAll();
        if(auto delay = github_retry_delay(reply,data))
            return restart(delay); // rate limited (429, or 403 that means the same thing)
        if(github_permanent_failure(status))
        {
            disconnect_github_issue(); // clear state first, so set_ai_status()'s "ongoing" check sees it disconnected and this message decays instead of animating forever
            return set_ai_status("GitHub issue channel authorization failed.",true);
        }
        if(reply->error() != QNetworkReply::NoError && status != 304)
            return restart(5000); // transient network error: back off, retry later
        if(status == 304)
            return restart(); // not modified

        if(auto etag = reply->rawHeader("ETag");!etag.isEmpty())
            github_etag = etag;

        auto issue = QJsonDocument::fromJson(data).object(); // body already read above
        if(issue["state"].toString() != "open")
            return disconnect_github_issue(); // closed directly on GitHub; stop without republishing

        auto envelope = QJsonDocument::fromJson(issue["body"].toString().toUtf8());
        if(!envelope.isObject())
            return restart(); // no command posted yet

        auto request_obj = envelope.object();
        auto id = request_obj["id"].toInteger();
        if(id <= github_last_id)
            return restart(); // already handled or not a new request

        auto issue_number = issue["number"];
        auto stamp = [&](QJsonObject result)
        {
            result["id"] = id;
            result["last_id"] = id;
            result["dsi_session_result"] = true;
            result["issue"] = issue_number;
            result["updated_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
            return result;
        };

        if(request_obj["request"].toString() == "close")
            // goes through the same retrying publish path as any result; send_pending_result() disconnects once this is confirmed published
            return publish_github_result(stamp(QJsonObject{{"state","closed"}}));

        if(request_obj["session"].toString().isEmpty())
            return publish_github_result(stamp(QJsonObject{
                {"state","error"},
                {"response",QJsonObject{{"status","error"},
                    {"error","malformed request: missing session"}}}}));

        bool include_log = request_obj["include_log"].toBool();
        request_obj.remove("id");
        request_obj.remove("include_log");
        request_obj["agent"] = "Codex/ChatGPT-GitHub";

        auto session_id = request_obj["session"].toString();
        bool new_session = !ai_info::find(session_id);
        if(is_new_chat(web_agent_session_id))
            assign_ai_session(web_agent_session_id,session_id);
        web_agent_session_id = session_id;
        if(auto* info = ai_info::create(session_id,"Codex/ChatGPT-GitHub")) // records which issue this session is bound to, so a restart can auto-resume polling it
        {
            info->model_settings["github_issue_url"] = github_issue_api.toString();
            info->save_config();
        }

        auto started = QDateTime::currentMSecsSinceEpoch();
        QByteArray reply_bytes;
        ai_request(QJsonDocument(request_obj).toJson(QJsonDocument::Compact),reply_bytes);
        auto response = QJsonDocument::fromJson(reply_bytes).object();

        auto run_ai_command = [&](const QString& session,const QString& cmd_name,const QJsonValue& param = {})
        {
            QJsonObject command{{"cmd",cmd_name}};
            if(!param.isUndefined())
                command["param"] = param;
            QByteArray bytes;
            ai_request(QJsonDocument(QJsonObject{
                {"session",session},{"command",command}}).toJson(QJsonDocument::Compact),bytes);
            return QJsonDocument::fromJson(bytes).object();
        };

        if(new_session) // omits "agent" so set_title itself can never create a session
            run_ai_command(session_id,"set_title",issue["title"].toString());

        if(include_log)
            response["log"] = run_ai_command(request_obj["session"].toString(),"log");

        auto succeeded = [](const QJsonObject& reply)
        {
            if(reply["status"].toString() == "error")
                return false;
            for(const auto& value : reply["result"].toArray())
                if(value.toObject()["status"].toString() == "error")
                    return false;
            return true;
        };

        publish_github_result(stamp(QJsonObject{
            {"state",succeeded(response) ? "done" : "error"},
            {"duration_ms",QDateTime::currentMSecsSinceEpoch()-started},
            {"response",response}}));
    });
}

void AIAgent::publish_github_result(QJsonObject result)
{
    constexpr qsizetype size_limit = 60*1024;
    auto original_size = QJsonDocument(result).toJson(QJsonDocument::Compact).size();
    if(original_size > size_limit)
    {
        result["state"] = "error";
        result["response"] = QJsonObject{
            {"error","response truncated: exceeds GitHub comment size limit"},
            {"original_bytes",original_size}};
    }

    github_pending_result = result; // staged until PATCH is confirmed; retried, never re-executed
    send_pending_result();
}

void AIAgent::send_pending_result()
{
    if(github_result_api.isEmpty() || github_pending_result.isEmpty())
        return;

    auto connection_id = github_connection_id;
    auto pending_id = github_pending_result["id"].toInteger();

    QJsonObject body{{"body",QString::fromUtf8(
        QJsonDocument(github_pending_result).toJson(QJsonDocument::Compact))}};
    auto request = github_request(github_result_api);
    request.setRawHeader("Content-Type","application/json");

    auto* reply = github_manager.sendCustomRequest(request,"PATCH",QJsonDocument(body).toJson(QJsonDocument::Compact));
    connect(reply,&QNetworkReply::finished,this,[this,reply,connection_id,pending_id]()
    {
        reply->deleteLater();
        if(connection_id != github_connection_id)
            return; // this connection was superseded (disconnect, or a fresh reconnect)

        auto status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if(auto delay = github_retry_delay(reply,reply->readAll()))
        {
            if(!github_issue_api.isEmpty())
                github_timer.start(delay); // rate limited (429, or 403 that means the same thing)
            return;
        }
        if(github_permanent_failure(status))
        {
            disconnect_github_issue(); // clear state first, so set_ai_status()'s "ongoing" check sees it disconnected and this message decays instead of animating forever
            return set_ai_status("GitHub issue channel authorization failed.",true);
        }

        if(reply->error() != QNetworkReply::NoError)
        {
            tipl::warning() << "cannot publish result to GitHub issue comment: "
                            << reply->errorString().toStdString();
            if(!github_issue_api.isEmpty())
                github_timer.start(5000); // back off; the pending result is retried, not lost
            return;
        }

        bool closed = github_pending_result["state"].toString() == "closed";
        github_last_id = pending_id;
        github_pending_result = QJsonObject();
        if(closed)
            return disconnect_github_issue();
        if(!github_issue_api.isEmpty())
            github_timer.start(500);
    });
}

void AIAgent::add_ai_reply(ai_info& info,const QString& chat,const QString& reasoning)
{
    auto entry = info.record_reply(chat,reasoning);
    show_ai_project(info,entry); // pass the entry so show_ai_project can see it's a new non-user reply and blink
    if(is_status_target(info.sessions)) // a background chat's reply must not steal the status bar from whatever is currently selected
        set_ai_status("Agent request completed.",true);
}

ai_info* ai_info::find(const QString& session)
{
    auto found = ai_infos.find(session);
    return found == ai_infos.end() ? nullptr : &found->second;
}
ai_info* ai_info::create(QString session,QString agent)
{
    if(session.isEmpty())
        return nullptr;
    if(auto* info = find(session))
        return info;
    auto provider = ai_info::identify_provider(agent);
    if(provider == ai_provider::Unknown)
        return nullptr;
    auto& info = ai_infos[session];
    info.sessions = std::move(session);
    info.provider = provider; info.agent_name = agent;
    return &info;
}

void write_history(const ai_info& info,QIODevice::OpenMode mode,
                   const QList<QJsonObject>& entries)
{
    if(is_new_chat(info.sessions) ||
       !QSettings().value("ai/keep_history",true).toBool())
        return;
    QFile file(ai_info::history_file(info.sessions));
    bool okay = file.open(QIODevice::WriteOnly|mode);
    for(const auto& entry : entries)
        okay = okay && file.write(QJsonDocument(entry).toJson(
                                      QJsonDocument::Compact)+'\n') >= 0;
    if(!okay)
        tipl::warning() << "cannot write ai history : "
                        << file.errorString().toStdString();
}
QJsonObject ai_info::record_history(QJsonObject entry)
{
    entry["time"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    projects.append(entry);
    write_history(*this,QIODevice::Append,QList<QJsonObject>{entry});
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
QJsonObject ai_info::record_request(const QString& command_name,QWidget* target)
{
    auto window_type = current_window == "main" ? QString("main") :
                       current_window.startsWith("tracking") ? "tracking" :
                       current_window.startsWith("recon") ? "recon" : "image";
    QJsonObject entry{{"type","request"},{"text",command_name},{"window",window_type}};
    if(target && window_type != "main")
        entry["title"] = QFileInfo(target->windowTitle()).fileName();
    return record_history(entry);
}
void AIAgent::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    refresh_codex_models();
    auto* item = ui->ai_project_list->currentItem();
    stop_blink(item ? ui->ai_project_list->itemWidget(item) : nullptr);
}

void AIAgent::closeEvent(QCloseEvent* event)
{
    for(auto& entry : ai_infos)
        if(auto* process = entry.second.processes)
        {
            entry.second.processes = nullptr;
            process->disconnect();
            process->kill();
            process->deleteLater();
        }
    active_ai_processes = 0;
    if(!github_issue_api.isEmpty())
        disconnect_github_issue();
    web_agent_active_session = false;
    update_send_button();
    QMainWindow::closeEvent(event);
}

void AIAgent::set_ai_status(QString status,bool temporary)
{
    ai_status_timer->stop();
    if(!status.isEmpty())
        ai_status_activity = status;
    // reflects only the currently selected chat (or, if none, a brand-new chat's own launch), not any other chat's activity
    bool ongoing = (web_agent_active_session && !github_issue_api.isEmpty()) ||
                   [this]
                   {
                       auto* item = ui->ai_project_list->currentItem();
                       return item ? bool(ai_infos[item->data(Qt::UserRole).toString()].processes)
                                   : active_ai_processes > 0;
                   }();
    if(ongoing && (status.isEmpty() || temporary))
    {
        status = ai_status_activity;
        if(status.endsWith('.'))
            status.chop(1);
        status += ", waiting for agent.";
        ai_status_timer->setSingleShot(false);
        ai_status_timer->start(500);
    }

    ui->ai_status->setVisible(!status.isEmpty());
    ui->ai_status->setText(status);
    ui->ai_status->repaint();

    if(temporary && !ongoing)
    {
        ai_status_timer->setSingleShot(true);
        ai_status_timer->start(2000);
    }
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
    if(QUuid(session).toString(QUuid::WithoutBraces).compare(session,Qt::CaseInsensitive))
        return void(reply = status_reply("error","invalid session: provide resumable provider thread ID"));

    auto* found = ai_info::find(session);
    if(!found)
    {
        auto agent = request["agent"].toString().trimmed();
        if(agent.isEmpty())
            return void(reply = status_reply("error","missing agent for new session"));
        if(!(found = ai_info::create(session,agent)))
            return void(reply = status_reply("error","invalid agent: include Codex, Claude, or ChatGPT in the agent name"));
        if(auto model = request["model"].toString().trimmed();!model.isEmpty())
            found->model_settings["model"] = model;
        found->save_config();
    }
    ai_info& info = *found;

    reply.clear();
    ai_log("received: "+QString::fromUtf8(data));
    auto chat = request["chat"].toString().trimmed();
    auto reasoning = request["reasoning"].toString().trimmed();
    auto reply_object = [&](QJsonObject result)
    {
        auto entry = info.record_reply(chat,reasoning);
        if(!info.prompts.isEmpty())
            result["prompt"] = QJsonArray::fromStringList(info.prompts);
        reply = QJsonDocument(result).toJson(QJsonDocument::Compact);
        ai_log(QString("reply for %1@%2: %3 ...")
                   .arg(info.agent_name,session,
                        QString::fromUtf8(reply).left(32)));
        info.prompts.clear();
        show_ai_project(info,entry);
        if(is_status_target(session)) // a background chat's reply must not steal the status bar from whatever is currently selected
            set_ai_status("Agent request completed.",true);
    };
    dispatching_info = &info;
    reply_object(main_window.dispatch_cmd(info,request)); // MainWindow's command center handles everything
    dispatching_info = nullptr;
}

void AIAgent::update_current_window(QWidget* window,const char* type)
{
    if(dispatching_info)
        dispatching_info->current_window = command_window_id(window,type);
}

void AIAgent::show_ai_project(ai_info& info,QJsonObject added_entry)
{
    const auto& history = info.projects;
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

        connect(title,&QPushButton::clicked,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
        connect(button,&QToolButton::pressed,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
    }

    auto* row = ui->ai_project_list->itemWidget(item);
    auto* title = row->findChild<QPushButton*>();
    item->setText({});
    auto chat_title = is_new_chat(info.sessions) && info.project_titles.isEmpty() ?
        "New "+info.agent_name+" Chat" : info.title();
    title->setText((info.provider == ai_provider::ChatGPT ? QString("🌐 ") : QString())+chat_title);
    title->setToolTip(title->text());
    item->setSizeHint(QSize(0,row->sizeHint().height()));

    auto* status_dot = row->findChild<QLabel*>("ai_project_status_dot");
    bool active = info.provider == ai_provider::ChatGPT ?
        (web_agent_active_session && !github_issue_api.isEmpty() && info.sessions == web_agent_session_id) :
        bool(info.processes);
    auto [status_color,status_text] = active ?
        std::make_pair("#34a853","Active") : info.has_error ?
        std::make_pair("#ea4335","Error") : std::make_pair("#9aa0a6","Inactive");
    status_dot->setStyleSheet(QString("background-color:%1;border-radius:5px;").arg(status_color));
    status_dot->setToolTip(status_text);

    auto* current = ui->ai_project_list->currentItem();
    const auto added_type = added_entry["type"].toString();
    if(!current && added_type == "user") // the user just started this chat themselves (nothing else was selected): bring it up
    {
        ui->ai_project_list->setCurrentItem(item);
        return; // currentItemChanged already rebuilt this chat's complete history
    }

    bool visible = current == item && isVisible();

    if(!added_type.isEmpty() && added_type != "user" && !visible)
    {
        row->setStyleSheet("background:#ffe082;border-radius:5px;");
        row->findChild<QTimer*>()->start();
    }

    if(current != item)
        return;

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
    auto display_time = [](const QJsonValue& value)
    {
        return QDateTime::fromString(value.toString(),Qt::ISODate).
               toString("MM/dd HH:mm:ss");
    };

    const bool show_reasoning = settings.value("ai/show_reasoning",false).toBool(); // read once: append() runs per history entry
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

        auto color = user ? "#e8f0fe" : "#e8f5e9";
        auto time = display_time(entry["time"]);
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
       (added_type == "assistant" && history.size() > 1 &&
        history[history.size()-2]["type"] == "request"))
    {
        ui->ai_chat_history->clear();
        auto is_leader = [&](const QJsonObject& entry)
        {
            auto type = entry["type"].toString();
            return type != "request" &&
                   (type != "assistant" ||
                    !entry["text"].toString().trimmed().isEmpty() ||
                    (show_reasoning &&
                     !entry["reasoning"].toString().trimmed().isEmpty()));
        };
        for(int index = 0;index < history.size();++index)
        {
            const auto& entry = history[index];
            if(!is_leader(entry))
                continue;

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
            for(auto end = index+1;end < history.size();++end)
            {
                const auto& request = history[end];
                if(is_leader(request))
                    break;
                if(request["type"] != "request")
                    continue;

                auto request_target = request["title"].toString();
                request_target += (request_target.isEmpty() ? "" : " · ")+
                                  request["window"].toString();
                if(!commands.isEmpty() && target != request_target)
                    add_activity();
                target = request_target;

                auto command = "<code>"+to_html(request["text"].toString())+"</code>";
                if(end+1 < history.size())
                {
                    if(auto duration = QDateTime::fromString(
                            request["time"].toString(),Qt::ISODate).msecsTo(
                            QDateTime::fromString(history[end+1]["time"].toString(),
                                                  Qt::ISODate));duration >= 0)
                    {
                        auto seconds = QString::number(duration/1000.0,'f',1);
                        if(seconds.endsWith(".0"))
                            seconds.chop(2);
                        command += " ("+seconds+"s)";
                    }
                }
                commands << command;
            }
            add_activity();

            if(entry["type"] == "user")
            {
                append(entry);
                if(!activities.isEmpty())
                    append(QJsonObject{{"type","assistant"},{"time",entry["time"]}},
                           activities);
            }
            else
                append(entry,activities);
        }
    }
    else
        append(added_entry);

    ui->ai_chat_history->ensureCursorVisible();
    auto* bar = ui->ai_chat_history->verticalScrollBar();
    QTimer::singleShot(
        0,bar,[bar]{bar->setValue(bar->maximum());});
}

void AIAgent::update_agent_models(
    int index,const QStringList& names,bool ollama)
{
    auto& profiles = agent_entries[index].profiles;
    auto previous = profiles;
    for(auto i = profiles.begin();i != profiles.end();)
        if(i.value().toObject().contains("provider") == ollama)
            i = profiles.erase(i);
        else
            ++i;
    for(const auto& name : names)
        profiles[name] = ollama ?
            QJsonObject{{"provider",true}} : previous[name].toObject();

    if(current_agent_index == index)
    {
        auto resolved = resolve_model(profiles,current_model_name,{},current_model_info);
        current_model_name = resolved.first;
        current_model_info = resolved.second;
        update_agent_status_label();
    }
}
void AIAgent::refresh_codex_models()
{
    auto path = agent_entries[int(ai_provider::Codex)].executable;
    if(path.isEmpty())
        return refresh_ollama_models();

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

        update_agent_models(int(ai_provider::Codex),models,false);
        refresh_ollama_models();
        process->deleteLater();
    });

    process->start(path,{"debug","models"});
    QTimer::singleShot(5000,process,&QProcess::kill);
}
void AIAgent::refresh_ollama_models()
{
    auto set_models = [this](const QStringList& models)
    {
        for(auto index : {int(ai_provider::Codex),int(ai_provider::Claude)})
            if(!agent_entries[index].executable.isEmpty())
                update_agent_models(index,models,true);
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
void AIAgent::add_ai_history(ai_info& info,const QString& type,const QString& text)
{
    show_ai_project(info,info.record_history(QJsonObject{{"type",type},{"text",text}}));
}

bool AIAgent::agent_logged_in(ai_provider provider)
{
    const auto& executable = agent_entries[int(provider)].executable;
    if(executable.isEmpty())
        return false;
    bool is_codex = provider == ai_provider::Codex;
    QProcess process;
    process.start(executable,is_codex ? QStringList{"login","status"} : QStringList{"auth","status"});
    if(!process.waitForStarted(3000) || !process.waitForFinished(10000))
        return false;
    if(is_codex)
        return process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0;
    return QJsonDocument::fromJson(process.readAllStandardOutput()).object()["loggedIn"].toBool();
}

bool AIAgent::run_agent_login(ai_provider provider)
{
    const auto& executable = agent_entries[int(provider)].executable;
    if(executable.isEmpty())
        return false;

    bool is_codex = provider == ai_provider::Codex;
    auto* process = new QProcess(this);
    process->setProcessChannelMode(QProcess::MergedChannels);

    QDialog dialog(this);
    dialog.setWindowTitle((is_codex ? QString("Codex") : QString("Claude"))+" Login");
    QVBoxLayout layout(&dialog);
    QLabel status("Starting sign-in...");
    status.setWordWrap(true);
    status.setFixedWidth(420);
    layout.addWidget(&status);

    QLineEdit code;
    code.setPlaceholderText("Paste the code here after signing in");
    QPushButton submit("Submit Code");
    code.setVisible(false);
    submit.setVisible(false);
    if(!is_codex)
    {
        layout.addWidget(&code);
        layout.addWidget(&submit);
    }
    QPushButton cancel("Cancel");
    layout.addWidget(&cancel);

    bool opened_url = false,succeeded = false;
    connect(process,&QProcess::readyReadStandardOutput,&dialog,[&]
    {
        auto text = QString::fromUtf8(process->readAllStandardOutput());
        status.setText(status.text()+text);
        static const QRegularExpression url_pattern("https?://\\S+");
        if(auto match = url_pattern.match(text);!opened_url && match.hasMatch())
        {
            QDesktopServices::openUrl(QUrl(match.captured()));
            opened_url = true;
            code.setVisible(!is_codex);
            submit.setVisible(!is_codex);
        }
    });
    connect(&submit,&QPushButton::clicked,&dialog,[&]
    {
        process->write(code.text().trimmed().toUtf8()+"\n");
        code.setEnabled(false);
        submit.setEnabled(false);
    });
    connect(&cancel,&QPushButton::clicked,&dialog,[&]
    {
        process->kill();
        dialog.reject();
    });
    connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),&dialog,
        [&](int exit_code,QProcess::ExitStatus exit_status)
    {
        succeeded = exit_code == 0 && exit_status == QProcess::NormalExit;
        dialog.accept();
    });
    connect(process,&QProcess::errorOccurred,&dialog,[&](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
            return;
        status.setText("Cannot start "+executable+": "+process->errorString());
        succeeded = false;
        dialog.reject();
    });

    process->start(executable,is_codex ? QStringList{"login"} : QStringList{"auth","login"});

    dialog.exec();
    if(process->state() != QProcess::NotRunning)
    {
        process->kill();
        process->waitForFinished(3000);
    }
    process->deleteLater();

    if(!succeeded)
        QMessageBox::warning(this,"AI Agent",(is_codex ? "Codex" : "Claude")+QString(" sign-in was not completed."));
    return succeeded;
}

bool AIAgent::try_connect_github_issue(const QString& url,bool resume)
{
    web_agent_active_session = true; // reflects the chosen mode even if the connection below fails, so the label/Resume button stay accurate
    update_send_button();
    update_agent_status_label();
    set_ai_status("Connecting to "+url+"...");
    tipl::out() << "connecting to GitHub issue: " << url.toStdString();

    QString error;
    if(!connect_github_issue(url,error))
    {
        set_ai_status("GitHub issue connect failed: "+error,true);
        tipl::out() << "GitHub issue connect failed: " << error.toStdString();
        // only a resume targets an already-known chat; web_agent_session_id (not sidebar selection) is the reliable way to find it
        if(resume)
            if(auto* info = ai_info::find(web_agent_session_id))
            {
                info->has_error = true;
                show_ai_project(*info);
            }
        return false;
    }
    update_send_button();
    update_agent_status_label();
    set_ai_status("Connected to "+url,true);
    tipl::out() << "connected to GitHub issue: " << url.toStdString();
    if(resume) // dot shows Active immediately, not just once the next poll cycle refreshes it
        if(auto* info = ai_info::find(web_agent_session_id))
            show_ai_project(*info);
    return true;
}

void AIAgent::update_agent_status_label()
{
    auto* item = ui->ai_project_list->currentItem();
    bool web = item ? ai_infos[item->data(Qt::UserRole).toString()].provider == ai_provider::ChatGPT
                     : web_agent_active_session;
    if(web)
        return void(ui->ai_agent_status->setText("ChatGPT(Web)"));

    static const QString dot = QString(" ")+QChar(0x00B7)+" "; // middle dot separator
    QString text = (current_agent_index == int(ai_provider::Codex) ? "Codex" : "Claude") +
                   dot + current_model_name;
    if(current_model_info.contains("provider"))
        text += dot+"Ollama@"+ai_ollama_url(settings).first.host();
    ui->ai_agent_status->setText(text);
}

void AIAgent::try_set_current_model(const QString& name) // accepts any non-empty name since the New Chat model field is editable
{
    if(name.isEmpty())
        return;
    if(name == "default")
    {
        current_model_name = "default";
        current_model_info = QJsonObject();
        return;
    }
    const auto& profiles = agent_entries[current_agent_index].profiles;
    current_model_name = name;
    current_model_info = profiles.contains(name) ? profiles[name].toObject() : QJsonObject();
}

void AIAgent::update_send_button()
{
    auto* item = ui->ai_project_list->currentItem();
    auto* info = item ? &ai_infos[item->data(Qt::UserRole).toString()] : nullptr;
    // selection wins whenever there is one, matching update_agent_status_label()/show_ai_project(): web_agent_active_session
    // only stands in for "no chat selected yet" (e.g. mid New Chat before its sidebar item exists)
    if(info ? info->provider == ai_provider::ChatGPT : web_agent_active_session)
    {
        bool connected = !github_issue_api.isEmpty() && (!info || info->sessions == web_agent_session_id);
        ui->ai_send_message->setText(connected ? "Stop" : "Resume");
        return;
    }
    bool running = info && info->processes;
    bool has_input = !ui->ai_chat_input->toPlainText().trimmed().isEmpty();
    ui->ai_send_message->setText(running && !has_input ? "Stop" : "Send");
}

bool AIAgent::is_status_target(const QString& session) const
{
    if(auto* item = ui->ai_project_list->currentItem())
        return item->data(Qt::UserRole).toString() == session;
    return session.isEmpty(); // nothing selected: only the still-anonymous chat currently being set up counts
}

// resume only ever applies to the web agent: the Agent combo is locked to ChatGPT and disabled, only the issue URL (defaulted to the last one) can still be changed
bool AIAgent::run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                                   bool& web,int& agent_index,QString& model_name,QString& issue_url)
{
    QDialog dialog(this);
    dialog.setWindowTitle(title);
    QFormLayout layout(&dialog);

    QComboBox agent;
    agent.addItem("Codex");
    agent.addItem("Claude");
    agent.addItem("ChatGPT (Web)");
    bool has_token = !settings.value("ai/github_token").toString().isEmpty();
    if(auto* item_model = qobject_cast<QStandardItemModel*>(agent.model()))
    {
        auto disable = [&](int index,bool available,const QString& reason)
        {
            if(available)
                return;
            item_model->item(index)->setEnabled(false);
            item_model->item(index)->setToolTip(reason);
        };
        disable(int(ai_provider::Codex),!agent_entries[int(ai_provider::Codex)].executable.isEmpty(),"Codex was not found");
        disable(int(ai_provider::Claude),!agent_entries[int(ai_provider::Claude)].executable.isEmpty(),"Claude was not found");
        disable(int(ai_provider::ChatGPT),has_token,"Set up a GitHub token in AI Settings first");
    }
    agent.setCurrentIndex(resume ? int(ai_provider::ChatGPT) : current_agent_index);
    agent.setEnabled(!resume);
    layout.addRow("Agent:",&agent);

    QWidget field_container; // declared before its would-be children below, so it is destroyed after them (reverse-declaration-order unwind), not before
    auto* field_stack = new QStackedLayout(&field_container);
    QComboBox model;
    model.setEditable(true); // lets the user type a specific model name (e.g. a dated Claude model), not just pick a known alias
    // web_agent_session_id (not sidebar selection) is the reliable way to find which chat is being resumed
    auto* resume_info = resume ? ai_info::find(web_agent_session_id) : nullptr;
    QLineEdit issue_url_edit(resume_info ? resume_info->model_settings["github_issue_url"].toString() : QString());
    issue_url_edit.setPlaceholderText("https://github.com/owner/repo/issues/1");
    QLabel field_label;
    field_stack->addWidget(&model);
    field_stack->addWidget(&issue_url_edit);
    layout.addRow(&field_label,&field_container);

    auto update_field = [&]()
    {
        bool chatgpt = agent.currentIndex() == int(ai_provider::ChatGPT);
        field_label.setText(chatgpt ? "Issue URL:" : "Model:");
        field_stack->setCurrentWidget(chatgpt ? static_cast<QWidget*>(&issue_url_edit) : static_cast<QWidget*>(&model));
        if(!chatgpt)
            set_model_selector(model,agent_entries[agent.currentIndex()].profiles,
                // only the agent that's actually active right now keeps its remembered model; switching to a different agent resets to that agent's own "default"
                agent.currentIndex() == current_agent_index ? current_model_name : QString());
    };
    update_field();
    connect(&agent,QOverload<int>::of(&QComboBox::currentIndexChanged),&dialog,[&](int){update_field();});

    QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    buttons.addButton(accept_text,QDialogButtonBox::AcceptRole);
    layout.addRow(&buttons);
    connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);

    if(dialog.exec() != QDialog::Accepted)
        return false;

    web = agent.currentIndex() == int(ai_provider::ChatGPT);
    agent_index = agent.currentIndex();
    model_name = web ? QString() : model_combo_key(model);
    issue_url = issue_url_edit.text().trimmed();
    return true;
}

void AIAgent::new_chat_dialog(bool resume)
{
    // resuming an already-known chat: reconnect with its saved issue link directly, no dialog
    if(resume)
        if(auto* info = ai_info::find(web_agent_session_id))
            if(auto url = info->model_settings["github_issue_url"].toString();!url.isEmpty())
            {
                try_connect_github_issue(url,true);
                return;
            }

    bool web = false;
    int agent_index = 0;
    QString model_name,issue_url;
    if(!run_new_chat_dialog(resume,resume ? "Resume Chat" : "New Chat",resume ? "Resume" : "Start",
                             web,agent_index,model_name,issue_url))
        return;
    if(!resume)
        web_agent_session_id.clear(); // starting fresh: no longer tied to whatever chat the old web session was

    auto create_chat = [&](const QString& agent)
    {
        // drop any never-used placeholder left behind by an abandoned "New Chat" attempt before adding another
        for(auto it = ai_infos.begin();it != ai_infos.end();)
            if(is_new_chat(it->first) && it->second.projects.isEmpty() && !it->second.processes)
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
            "new:"+QUuid::createUuid().toString(QUuid::WithoutBraces),agent);
        if(info->provider == ai_provider::ChatGPT)
            web_agent_session_id = info->sessions;
        else
            info->model_settings = QJsonObject{
                {"model",current_model_name},{"info",current_model_info}};
        show_ai_project(*info);
        ui->ai_project_list->setCurrentItem(info->project_items);
    };

    if(web)
    {
        if(!github_issue_api.isEmpty())
            disconnect_github_issue(); // leave the old channel cleanly before attempting a different one
        if(try_connect_github_issue(issue_url,resume) && !resume)
            create_chat("ChatGPT(Web)");
        return;
    }

    if(!github_issue_api.isEmpty())
        disconnect_github_issue(); // leaving web-agent mode for a local chat
    web_agent_active_session = false;
    update_send_button();

    current_agent_index = agent_index;
    try_set_current_model(model_name);
    update_agent_status_label();
    create_chat(current_agent_index == int(ai_provider::Codex) ? "Codex" : "Claude");
    ui->ai_chat_input->clear();
    ui->ai_chat_input->setFocus();
    set_ai_status();
}

void AIAgent::on_ai_new_chat_clicked()
{
    new_chat_dialog(false);
}

void AIAgent::on_ai_agent_status_clicked()
{
    if(auto* item = ui->ai_project_list->currentItem())
    {
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        if(info.provider == ai_provider::ChatGPT) // no agent/model to change here; the only meaningful action is reconnecting
        {
            web_agent_session_id = info.sessions; // resume must target the selected chat, not whatever session was last active
            new_chat_dialog(true);
            return;
        }
        QDialog dialog(this);
        dialog.setWindowTitle("Change Model");
        QFormLayout layout(&dialog);
        QLabel agent_label(info.provider == ai_provider::Codex ? "Codex" : "Claude");
        QComboBox model;
        set_model_selector(model,agent_entries[int(info.provider)].profiles,current_model_name);
        layout.addRow("Agent:",&agent_label);
        layout.addRow("Model:",&model);
        QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
        layout.addRow(&buttons);
        connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
        connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
        if(dialog.exec() != QDialog::Accepted)
            return;

        current_agent_index = int(info.provider);
        try_set_current_model(model_combo_key(model));
        update_agent_status_label();
        return;
    }

    bool web = false;
    int agent_index = 0;
    QString model_name,issue_url;
    if(!run_new_chat_dialog(false,"Change Agent/Model","Save",web,agent_index,model_name,issue_url))
        return;

    if(web)
    {
        try_connect_github_issue(issue_url,false);
        return;
    }

    current_agent_index = agent_index;
    try_set_current_model(model_name);
    update_agent_status_label();
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
    QWidget login_row;
    QHBoxLayout login_layout(&login_row);
    login_layout.setContentsMargins(0,0,0,0);
    QPushButton codex_login("Sign in to Codex..."),claude_login("Sign in to Claude...");
    codex_login.setEnabled(!agent_entries[int(ai_provider::Codex)].executable.isEmpty());
    claude_login.setEnabled(!agent_entries[int(ai_provider::Claude)].executable.isEmpty());
    login_layout.addWidget(&codex_login);
    login_layout.addWidget(&claude_login);
    connect(&codex_login,&QPushButton::clicked,&dialog,[this]{run_agent_login(ai_provider::Codex);});
    connect(&claude_login,&QPushButton::clicked,&dialog,[this]{run_agent_login(ai_provider::Claude);});
    QCheckBox history("Keep AI chat history");
    history.setChecked(settings.value("ai/keep_history",true).toBool());
    QCheckBox show_reasoning("Show reasoning");
    show_reasoning.setToolTip("Show AI reasoning messages in chat history");
    show_reasoning.setChecked(settings.value("ai/show_reasoning",false).toBool());
    QComboBox debug;
    debug.addItem("Disabled");
    debug.addItem("Enabled (truncated)");
    debug.addItem("Enabled (complete)");
    debug.setCurrentIndex(settings.value("ai/debug",0).toInt());
    QLineEdit github_pat(settings.value("ai/github_token").toString());
    github_pat.setEchoMode(QLineEdit::Password);
    github_pat.setPlaceholderText("required to connect a GitHub issue");
    layout.addRow("Ollama host/IP:",&host);
    layout.addRow("Ollama port:",&port);
    layout.addRow(&login_row);
    layout.addRow(&history);
    layout.addRow(&show_reasoning);
    layout.addRow("Debug mode:",&debug);
    layout.addRow("GitHub token (issue channel):",&github_pat);
    QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
    layout.addRow(&buttons);
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
    settings.setValue("ai/github_token",github_pat.text().trimmed());
    ai_debug_level() = debug.currentIndex();
    if(reasoning_changed)
        if(auto* item = ui->ai_project_list->currentItem())
            show_ai_project(ai_infos[item->data(Qt::UserRole).toString()]);

    refresh_ollama_models();
}

ai_launch AIAgent::prepare_ai(ai_provider provider,QString& session,
                                 const QString& text,ai_input input)
{
    ai_launch launch;

    // Resolve agent
    launch.name = provider == ai_provider::Codex ? "Codex" : "Claude";
    launch.executable = agent_entries[int(provider)].executable;
    if(launch.executable.isEmpty())
    {
        if(input == ai_input::Pending && !session.isEmpty())
            ai_infos[session].prompts.append(text);
        auto message = launch.name+" executable was not found.";
        set_ai_status(message,true);
        QMessageBox::warning(this,"AI Agent",message);
        return launch;
    }

    // Resolve work directory
    auto project_dir = ui->ai_work_dir->text().trimmed();
    ui->ai_work_dir->setText(
        project_dir.isEmpty() ? main_window.work_dir() : project_dir);

    auto* info = session.isEmpty() ? nullptr : &ai_infos[session];
    if(info)
        info->has_error = false; // a new attempt clears the sidebar's error dot

    // Resolve model
    QJsonObject selected{
        {"model",current_model_name},
        {"info",current_model_info}};
    launch.model_setting =
        info && selected["model"].toString() ==
                info->model_settings["model"].toString() ?
        info->model_settings : selected;

    launch.model = launch.model_setting["model"].toString().trimmed();
    if(launch.model_setting["info"].toObject().contains("provider"))
    {
        auto [url,configured] = ai_ollama_url(settings);
        launch.model_url = url;
        launch.name += "/Ollama("+launch.model_url.host()+")";
        if(!configured)
        {
            if(input == ai_input::Pending && !session.isEmpty())
                ai_infos[session].prompts.append(text);
            set_ai_status("Ollama is not configured.",true);
            QMessageBox::warning(
                this,"AI Agent","Set the Ollama host/IP in AI Settings first.");
            return launch;
        }
    }
    else if(!agent_logged_in(provider))
    {
        set_ai_status(launch.name+" needs sign-in: check your browser.",true);
        if(!run_agent_login(provider))
        {
            if(input == ai_input::Pending && !session.isEmpty())
                ai_infos[session].prompts.append(text);
            set_ai_status(launch.name+" is not signed in.",true);
            return launch;
        }
    }
    if(launch.model.startsWith("default",Qt::CaseInsensitive))
        launch.model.clear();

    if((session.isEmpty() || is_new_chat(session)) &&
       provider == ai_provider::Claude)
    {
        auto assigned = session.isEmpty() ?
            QUuid::createUuid().toString(QUuid::WithoutBraces) : session.mid(4);
        info = session.isEmpty() ? ai_info::create(assigned,launch.name) :
                                  assign_ai_session(session,assigned);
        session = assigned;
    }
    if(info) // always save once info is known: a brand-new session must persist its agent even if the model happens to match the default
    {
        info->model_settings = launch.model_setting;
        info->save_config();
    }

    auto* process = new QProcess(this);
    launch.process = process;
    process->setObjectName(session);
    process->setWorkingDirectory(QApplication::applicationDirPath()+"/ai");
    auto env = QProcessEnvironment::systemEnvironment();
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

    if(info)
        info->processes = process; // a fresh Codex session has no info yet (assigned on "thread.started"); nothing gets disabled meanwhile

    if(input == ai_input::User)
    {
        if(info)
            add_ai_history(*info,"user",text);
        ui->ai_chat_input->clear();
    }

    auto restore_new_chat = [=](const QString& message,bool show_history)
    {
        QMessageBox::warning(this,"AI Agent",message);
        if(auto* info = ai_info::find(process->objectName()))
        {
            info->processes = nullptr;
            info->has_error = true;
            show_ai_project(*info);
        }
        if(is_status_target(process->objectName()))
        {
            ui->ai_chat_input->setPlainText(text);
            if(show_history)
                ui->ai_chat_history->setPlainText(message);
        }
    };

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
        if(auto* info = ai_info::find(session))
            show_ai_project(*info);
        update_send_button();
    });

    connect(process,&QProcess::errorOccurred,this,
            [=](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
            return;

        auto session = process->objectName();
        auto message = "Cannot start "+launch.name+": "+process->errorString();
        ai_log(message);
        if(is_status_target(session))
            set_ai_status(message,true);

        if(session.isEmpty() || is_new_chat(session))
            restore_new_chat(message,true);
        else
        {
            auto& info = ai_infos[session];
            info.processes = nullptr;
            info.has_error = true;
            if(input == ai_input::Pending)
                info.prompts.append(text);
            else if(auto* item = ui->ai_project_list->currentItem();
                    item && item->data(Qt::UserRole).toString() == session &&
                    ui->ai_chat_input->toPlainText().trimmed().isEmpty())
                ui->ai_chat_input->setPlainText(text);
            add_ai_history(info,"activity",message);
        }
        update_send_button();
        process->deleteLater();
    });

    connect(process,
            QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this,[=](int exit_code,QProcess::ExitStatus exit_status)
    {
        active_ai_processes = std::max(0,active_ai_processes-1);
        bool user_stopped = process->property("user_stopped").toBool();
        auto session = process->objectName();
        if(is_status_target(session))
            set_ai_status((user_stopped ? launch.name+" stopped." : launch.name+" finished."),true);
        ai_log(launch.name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        bool failed = !user_stopped && (exit_code || exit_status == QProcess::CrashExit);
        auto error_message = user_stopped ? QString("Stopped by user.") :
                              ("error code:"+QString::number(exit_code)+" "+
                              QString::fromUtf8(error)).trimmed();
        if(failed)
            ai_log(error_message);

        if(session.isEmpty() || is_new_chat(session))
        {
            auto message = failed ? error_message :
                           "AI agent ended before creating a new chat.";
            restore_new_chat(message,false);
        }
        else
        {
            auto& info = ai_infos[session];
            info.processes = nullptr;
            info.has_error = failed;

            auto pending = info.prompts.join("\n\n");
            info.prompts.clear();
            if(!pending.isEmpty())
                start_ai(session,pending,ai_input::Pending);
            else if(failed || user_stopped)
                add_ai_history(info,"activity",error_message);
            else if(!process->property("had_reply").toBool())
                add_ai_history(info,"activity","No reply from AI agent.");
            else
                show_ai_project(info);
        }
        update_send_button();
        process->deleteLater();
    });
    return launch;
}

QStringList AIAgent::configure_claude(
    const ai_launch& launch,QString session,const QString& text,bool new_session)
{
    auto* process = launch.process;
    static const char* ollama_model_vars[] = {
        "ANTHROPIC_DEFAULT_HAIKU_MODEL","ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL","CLAUDE_CODE_SUBAGENT_MODEL"};
    auto env = process->processEnvironment();
    if(!launch.model_url.isEmpty())
    {
        env.insert("ANTHROPIC_BASE_URL",launch.model_url.toString());
        env.insert("ANTHROPIC_AUTH_TOKEN","ollama");
        env.insert("ANTHROPIC_API_KEY","");
        env.insert("CLAUDE_CODE_USE_POWERSHELL_TOOL","1");
        if(!launch.model.isEmpty())
            for(auto name : ollama_model_vars)
                env.insert(name,launch.model);
    }
    else // real Anthropic model: strip any Ollama redirect inherited from the system environment
    {
        for(auto name : {"ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","CLAUDE_CODE_USE_POWERSHELL_TOOL"})
            env.remove(name);
        for(auto name : ollama_model_vars)
            env.remove(name);
    }
    process->setProcessEnvironment(env);

    connect(process,&QProcess::readyReadStandardOutput,this,[=]
            {
                while(process->canReadLine())
                {
                    auto line = process->readLine();
                    ai_log("stdout:"+QString::fromUtf8(line).trimmed());
                    auto event = QJsonDocument::fromJson(line).object();
                    auto event_type = event["type"].toString();
                    if(event_type == "system" &&
                       event["subtype"] == "thinking_tokens" &&
                       ai_status_activity != "Thinking" &&
                       is_status_target(process->objectName()))
                        set_ai_status("Thinking");

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
    // Prepend a system prompt to the initial text here if needed.
    connect(process,&QProcess::started,process,
            [process,text]
            {process->write(claude_input(text));});
    QStringList args{
        "-p",
        "--input-format","stream-json",
        "--output-format","stream-json",
        "--verbose",
        "--add-dir",ui->ai_work_dir->text(),
        "--allowedTools","Bash(bash ./dsi.sh:*),PowerShell(./dsi.ps1:*),WebFetch,WebSearch,Read,Glob,Grep",
        new_session ? "--session-id" : "--resume",session};
    if(!launch.model.isEmpty())
        args << "--model" << launch.model;
    return args;
}
QStringList AIAgent::configure_codex(
    const ai_launch& launch,QString session,const QString& text)
{
    auto* process = launch.process;
    connect(process,&QProcess::readyReadStandardOutput,this,[=]
    {
        while(process->canReadLine())
        {
            auto line = process->readLine();
            ai_log("stdout:"+QString::fromUtf8(line).trimmed());
            auto event = QJsonDocument::fromJson(line).object();
            if(event["type"] == "thread.started")
            {
                auto old_session = process->objectName();
                auto session = event["thread_id"].toString();
                auto* info = is_new_chat(old_session) ?
                    assign_ai_session(old_session,session) :
                    ai_info::create(session,launch.name);
                if(info)
                {
                    info->model_settings = launch.model_setting;
                    info->save_config();
                }
                if(info && old_session != session)
                {
                    process->setObjectName(session);
                    info->processes = process;
                    add_ai_history(*info,"user",text);
                    if(is_status_target(session))
                        set_ai_status("Agent session ready.",true);
                }
                continue;
            }

            auto item = event["item"].toObject();
            auto type = item["type"].toString();
            bool reasoning = type == "reasoning";
            if(type != "agent_message" && !reasoning)
                continue;

            auto text = item["text"].toString().trimmed();
            if(text.isEmpty())
                continue;
            process->setProperty("had_reply",true);
            if(auto* info = ai_info::find(process->objectName()))
                add_ai_reply(*info,reasoning ? QString() : text,
                             reasoning ? text : QString());
        }
    });

    QStringList args{"exec","--search","--add-dir",ui->ai_work_dir->text()};
    if(!launch.model_url.isEmpty())
    {
        auto url = launch.model_url;
        url.setPath("/v1");

        auto env = launch.process->processEnvironment();
        env.insert("CODEX_OSS_BASE_URL",url.toString());
        launch.process->setProcessEnvironment(env);

        args << "--oss" << "--local-provider=ollama";
    }
    if(!launch.model.isEmpty())
    {
        args << "--model" << launch.model;
        if(auto profile = launch.model_setting["info"].toObject()["profile"].toString();
           !profile.isEmpty())
            args << "--profile" << profile;
    }
    if(!session.isEmpty())
        args << "resume" << session;
    args << "--json" << "--skip-git-repo-check";
    args << text;
    return args;
}
void AIAgent::start_ai(QString session,const QString& text,ai_input input)
{
    auto* info = ai_info::find(session);
    if(info && info->processes)
    {
        add_ai_history(*info,"user",text);
        ui->ai_chat_input->clear();

        bool send = info->provider == ai_provider::Claude &&
                    info->processes->state() == QProcess::Running;
        if(send)
            info->processes->write(claude_input(text));
        else
            info->prompts.append(text);

        set_ai_status(send ? "Message sent to Claude." :
                             "Message queued for the AI agent.",!send);
        return;
    }

    auto provider = info ? info->provider :
        ai_provider(current_agent_index);
    Q_ASSERT(provider == ai_provider::Codex || provider == ai_provider::Claude); // never ChatGPT: callers must intercept a web chat before reaching here

    bool new_session = session.isEmpty() || is_new_chat(session);
    auto launch_session = new_session && provider == ai_provider::Codex ?
        QString() : session;
    auto launch = prepare_ai(provider,launch_session,text,input);
    if(!launch.process)
    {
        if(info)
        {
            info->has_error = true;
            show_ai_project(*info);
        }
        return;
    }
    if(is_new_chat(session) && provider == ai_provider::Codex)
    {
        launch.process->setObjectName(session);
        info->processes = launch.process;
    }
    auto args = provider == ai_provider::Codex ?
        configure_codex(launch,launch_session,text) :
        configure_claude(launch,launch_session,text,new_session);
    ai_log("start " + launch.executable +
           " args: " + args.join(" ").remove("\n"));
    set_ai_status("Starting "+launch.name+"...");
    launch.process->start(launch.executable,args);
}

void AIAgent::on_ai_send_message_clicked()
{
    auto* item = ui->ai_project_list->currentItem();
    auto* info = item ? &ai_infos[item->data(Qt::UserRole).toString()] : nullptr;
    auto text = ui->ai_chat_input->toPlainText().trimmed();

    // selection wins whenever there is one; matches update_send_button() so the click always does what the label says
    if(info ? info->provider == ai_provider::ChatGPT : web_agent_active_session)
    {
        if(!github_issue_api.isEmpty() && (!info || info->sessions == web_agent_session_id))
        {
            disconnect_github_issue();
            set_ai_status("GitHub issue channel stopped.",true);
            return;
        }
        if(info)
            web_agent_session_id = info->sessions; // resume must target the selected chat, not whatever session was last active
        new_chat_dialog(true);
        return;
    }

    if(info)
    {
        if(info->processes)
        {
            if(!text.isEmpty())
            {
                start_ai(info->sessions,text,ai_input::User);
                update_send_button();
                return;
            }
            info->prompts.clear(); // stop means stop: no auto-continue into a queued message
            info->processes->setProperty("user_stopped",true);
            info->processes->kill(); // kill(): a windowless console child never sees terminate()'s WM_CLOSE
            return;
        }
    }

    if(text.isEmpty())
        return;

    start_ai(item ? item->data(Qt::UserRole).toString() : QString(),
             text,ai_input::User);
    update_send_button();
}
