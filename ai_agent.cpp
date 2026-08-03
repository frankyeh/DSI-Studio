#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
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
#include <QRadioButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QShortcut>
#include <QShowEvent>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTextFrame>
#include <QTimer>
#include <QToolButton>
#include <QUuid>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "ai_agent.hpp"
#include "ui_ai_agent.h"
#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"
#include "view_image.h"
#include "reconstruction/reconstruction_window.h"
#include "console.h"

std::unordered_map<QString,ai_info> ai_infos;
QString ai_project_dir;
constexpr auto ai_debug_tag = "[DEUBG]";
bool& ai_debug_enabled()
{
    static bool enabled = QSettings().value("ai/debug").toBool();
    return enabled;
}
QString ai_info::history_file(const QString& session)
{
    return ai_project_dir+"/"+QString::fromLatin1(
               QUrl::toPercentEncoding(session))+".jsonl";
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
    return name.contains("codex",Qt::CaseInsensitive) ? ai_provider::Codex :
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
    return QString("<b>%1</b><br><br>Agent: %2<br>Session: %3<br>Status: %4<br>"
        "Messages: %5 (%6 you, %7 AI)<br>Activities: %8<br>"
        "Created: %9<br>Updated: %10")
        .arg(title().toHtmlEscaped(),agent_name.toHtmlEscaped(),sessions.toHtmlEscaped(),processes ? "Working" : "Idle")
        .arg(user+assistant).arg(user).arg(assistant).arg(activity)
        .arg(time(projects.first()["time"]),time(projects.last()["time"]));
}
void ai_log(QString text)
{
    if(ai_debug_enabled())
    {
        auto prefix = QString(ai_debug_tag)+" ";
        tipl::out() << (prefix+text.remove('\r').
                        replace('\n',"\n"+prefix)).toStdString();
    }
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
void set_model_selector(QComboBox& model,const QJsonObject& profiles,
                        QString selected = {},QString fallback = {},
                        QJsonObject selected_info = {})
{
    auto names = profiles.keys();
    names.sort(Qt::CaseInsensitive);
    model.clear();
    model.addItem("default");
    for(const auto& name : names)
        model.addItem(name,profiles[name].toObject());
    auto selected_index = model.findText(selected.isEmpty() ? fallback : selected);
    if(selected_index < 0 && !selected.isEmpty())
    {
        model.addItem(selected,selected_info);
        selected_index = model.count()-1;
    }
    model.setCurrentIndex(std::max(0,selected_index));
}
// headless equivalent of set_model_selector's selection logic (no list to
// build/show): resolves which model name+info should now be current, given
// the available profiles and a preferred selected/fallback name; mirrors
// set_model_selector's findText()-then-fallback-then-default decision tree,
// including that "default" is always resolvable even though it is not a
// profiles key (it is the sentinel entry set_model_selector always adds first)
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
bool ai_info::save_title(ai_info& info,QString title)
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
AIAgent::AIAgent(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::AIAgent)
{
    ui->setupUi(this);
    ui->ai_work_dir->setText(main_window.work_dir());
    connect(ui->ai_browse_work_dir,&QPushButton::clicked,this,[this]
    {
        auto path = QFileDialog::getExistingDirectory(
            this,"Select AI Work Directory",ui->ai_work_dir->text());
        if(!path.isEmpty())
            ui->ai_work_dir->setText(QDir::toNativeSeparators(path));
    });
    connect(ui->ai_show_reasoning,&QCheckBox::toggled,this,[this]
    {
        if(auto* item = ui->ai_project_list->currentItem())
            show_ai_project(ai_infos[item->data(Qt::UserRole).toString()]);
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
        if(!path.isEmpty())
            ai_log(agent+" models: none detected");
    }
    refresh_codex_models(codex_path);

    if(codex_path.isEmpty() && !claude_path.isEmpty())
        current_agent_index = int(ai_provider::Claude);

    auto default_index = settings.value(
        "ai/default_agent",current_agent_index).toInt();
    if(default_index >= 0 && default_index < int(agent_entries.size()))
        current_agent_index = default_index;
    try_set_current_model(settings.value(
        "ai/default_model",current_model_name).toString());
    update_agent_status_label();

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
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        bool okay;
        auto title = QInputDialog::getText(
            this,"Rename Chat","Chat name:",QLineEdit::Normal,
            info.title(),&okay);
        if(okay && ai_info::save_title(info,title))
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
            process->disconnect(); process->terminate(); process->deleteLater();
            active_ai_processes = std::max(0,active_ai_processes-1);
            set_ai_status();
        }
        QFile::remove(ai_info::history_file(session));
        settings.remove("ai/title/"+session);
        ai_infos.erase(session);
        delete ui->ai_project_list->takeItem(row); // detach then delete: keeps count()/rows consistent

        // keep a chat selected whenever one exists; only New Chat clears the selection
        if(ui->ai_project_list->count())
            ui->ai_project_list->setCurrentRow(std::min(row,ui->ai_project_list->count()-1));
    });

    connect(ui->ai_project_list,&QListWidget::currentItemChanged,this,
            [this](QListWidgetItem* item,QListWidgetItem* previous)
    {
        for(auto* i : {previous,item})
            if(i)
                ui->ai_project_list->itemWidget(i)->
                    findChild<QPushButton*>()->
                    setStyleSheet(i == item ?
                        "color:#202124;background:#dce9f9;" : "");
        if(!item)
        {
            ui->ai_chat_history->clear();
            return update_send_button();
        }

        stop_blink(ui->ai_project_list->itemWidget(item));
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        current_agent_index = int(info.provider);
        auto resolved = resolve_model(
            agent_entries[current_agent_index].profiles,
            info.model_settings["model"].toString(),{},
            info.model_settings["info"].toObject());
        current_model_name = resolved.first;
        current_model_info = resolved.second;
        update_agent_status_label();
        show_ai_project(info);
        update_send_button();
    });

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
        auto* ai = ai_info::create(session,first["agent"].toString());
        if(!ai)
            continue;
        ai->model_settings = first["model_settings"].toObject();
        ai->project_titles = settings.value("ai/title/"+session).toString();
        ai->projects = std::move(history);
        show_ai_project(*ai);
    }
    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);
}

AIAgent::~AIAgent()
{
    delete ui;
}

// GitHub issue channel: the issue body carries the next request, and one
// pinned comment (marked "dsi_session_result":true) carries the result
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

// 401/403/404/410/422: the token, its permissions, or the resource itself is
// wrong; retrying on a timer cannot fix these, only spam GitHub indefinitely.
// Checked only after github_retry_delay() below, since GitHub can also
// return a 403 for rate limiting, which is not a permanent failure.
static bool github_permanent_failure(int status)
{
    return status == 401 || status == 403 || status == 404 ||
           status == 410 || status == 422;
}

// GitHub signals both primary and secondary rate limiting with 429 or 403;
// returns the wait time in ms if so, or 0 if this is not a rate-limit response
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

// blocking helper: connect_github_issue is a one-shot, user-initiated
// action, so a short local event loop keeps its bool/error-message
// interface synchronous without adding pending-connection state
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
    // snapshot now: github_request() uses this member for the whole session,
    // so a later edit to AI Settings cannot swap the identity mid-poll
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

    // identify who the token itself belongs to: it need not be the repo
    // owner (e.g. a bot/collaborator account), and result-comment ownership
    // must be checked against that identity, not the issue's owner
    bool ok = false;
    auto authenticated_user = QJsonDocument::fromJson(
        github_blocking(github_manager,github_request(QUrl("https://api.github.com/user")),
                         "GET",{},ok,error)).object()["login"].toString();
    if(!ok)
        return error = "cannot verify GitHub token: "+error,false;
    if(authenticated_user.isEmpty())
        return error = "cannot verify GitHub token: unexpected response from GitHub",false;

    auto issue = QJsonDocument::fromJson(
        github_blocking(github_manager,github_request(issue_api),"GET",{},ok,error)).object();
    if(!ok)
        return false;

    if(issue.contains("pull_request"))
        return error = "the link points to a pull request, not an issue",false;
    if(issue["state"].toString() != "open")
        return error = "issue is not open",false;
    if(issue["user"].toObject()["login"].toString().compare(owner,Qt::CaseInsensitive))
        return error = "issue creator must be the repository owner",false;
    if(!issue["title"].toString().startsWith("DSI Studio session"))
        return error = "issue title must start with \"DSI Studio session\"",false;

    auto comments = QJsonDocument::fromJson(
        github_blocking(github_manager,github_request(QUrl(issue_api.toString()+"/comments?per_page=100")),
                        "GET",{},ok,error)).array();
    if(!ok)
        return false;

    // find our own existing result comment (author must match the token's
    // own identity, not necessarily the issue/repository owner); if more
    // than one matches (e.g. left over from an earlier bug or a race),
    // keep the one with the highest last_id rather than just the first
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
            return false;
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
            disconnect_github_issue(); // clear connection state first: set_ai_status()'s
                                        // "ongoing" check must see the disconnected state
                                        // so this message decays instead of animating forever
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
            // goes through the same retrying publish path as any other result;
            // send_pending_result() disconnects once this is confirmed published
            return publish_github_result(stamp(QJsonObject{{"state","closed"}}));

        if(request_obj["request"].toString().isEmpty() || request_obj["session"].toString().isEmpty())
            return publish_github_result(stamp(QJsonObject{
                {"state","error"},
                {"response",QJsonObject{{"status","error"},
                    {"error","malformed request: missing request or session"}}}}));

        bool include_log = request_obj["include_log"].toBool();
        request_obj.remove("id");
        request_obj.remove("include_log");
        request_obj["agent"] = "Codex/ChatGPT-GitHub";
        request_obj["title"] = issue["title"].toString(); // only applied by
            // MainWindow::ai_request() when this session is being created

        auto started = QDateTime::currentMSecsSinceEpoch();
        QByteArray reply_bytes;
        main_window.ai_request(QJsonDocument(request_obj).toJson(QJsonDocument::Compact),reply_bytes);
        auto response = QJsonDocument::fromJson(reply_bytes).object();

        if(include_log)
        {
            QByteArray log_reply_bytes;
            main_window.ai_request(QJsonDocument(QJsonObject{
                {"request","LOG"},{"session",request_obj["session"]}}).toJson(QJsonDocument::Compact),
                log_reply_bytes);
            response["log"] = QJsonDocument::fromJson(log_reply_bytes).object();
        }

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
    if(QJsonDocument(result).toJson(QJsonDocument::Compact).size() > size_limit)
        result["response"] = QJsonObject{{"error","response truncated: exceeds GitHub comment size limit"}};

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
            disconnect_github_issue(); // clear connection state first: set_ai_status()'s
                                        // "ongoing" check must see the disconnected state
                                        // so this message decays instead of animating forever
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
    ai_info::record_reply(info,chat,reasoning);
    refresh_ai_info(info);
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
    if(!QSettings().value("ai/keep_history",true).toBool())
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
void ai_info::record_history(ai_info& info,QJsonObject entry)
{
    if(info.projects.isEmpty())
    {
        entry["agent"] = info.agent_name;
        entry["model_settings"] = info.model_settings;
    }
    entry["time"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    info.projects.append(entry);
    write_history(info,QIODevice::Append,QList<QJsonObject>{entry});
}
void ai_info::record_reply(
    ai_info& info,const QString& chat,const QString& reasoning)
{
    if(chat.isEmpty() && reasoning.isEmpty())
        return;
    QJsonObject entry{{"type","assistant"},{"text",chat}};
    if(!reasoning.isEmpty())
        entry["reasoning"] = reasoning;
    record_history(info,entry);
}
void AIAgent::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    auto* item = ui->ai_project_list->currentItem();
    stop_blink(item ? ui->ai_project_list->itemWidget(item) : nullptr);
}

void AIAgent::set_ai_status(QString status,bool temporary)
{
    ai_status_timer->stop();
    if(!status.isEmpty())
        ai_status_activity = status;
    // a connected web-agent session has no QProcess to track, but is just as
    // "ongoing" as a local agent process: keep the status animating instead
    // of letting it decay to "Current task complete." while still connected
    bool ongoing = active_ai_processes ||
                   (web_agent_active_session && !github_issue_api.isEmpty());
    if(ongoing && (status.isEmpty() || temporary))
    {
        status = ai_status_activity;
        if(status.endsWith('.'))
            status.chop(1);
        status += ", waiting for agent.";
        ai_status_timer->setSingleShot(false);
        ai_status_timer->start(500);
    }
    else if(status.isEmpty())
        status = "Current task complete.";

    ui->ai_status->setText(status);
    ui->ai_status->repaint();

    if(temporary && !ongoing)
    {
        ai_status_timer->setSingleShot(true);
        ai_status_timer->start(2000);
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
    auto reply_object = [&](QJsonObject result)
    {
        ai_info::record_reply(info,chat,reasoning);
        if(!info.prompts.isEmpty())
            result["prompt"] = QJsonArray::fromStringList(info.prompts);
        reply = QJsonDocument(result).toJson(QJsonDocument::Compact);
        ai_log(QString("reply for %1@%2: %3 ...")
                   .arg(info.agent_name,session,
                        QString::fromUtf8(reply).left(32)));
        info.prompts.clear();
    };
    auto reply_error = [&](const QString& error)
    {
        reply_object(QJsonObject{{"status","error"},{"error",error}});
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
        if(info.log_position == quint64(-1))
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            info.log_position = console.total_size;
        }
    }

    auto get_window_id = [](QWidget* window)
    {
        if(qobject_cast<MainWindow*>(window))
            return command_window_id(window,"main");
        if(qobject_cast<tracking_window*>(window))
            return command_window_id(window,"tracking");
        if(qobject_cast<reconstruction_window*>(window))
            return command_window_id(window,"recon");
        return qobject_cast<view_image*>(window) ?
            command_window_id(window,"image") : QString();
    };

    chat = request["chat"].toString().trimmed();
    reasoning = request["reasoning"].toString().trimmed();
    if(type == "TITLE")
    {
        auto title = request["title"].toString().simplified();
        if(title.isEmpty())
            return reply_error("missing title");
        if(!ai_info::save_title(info,title))
            return reply_error("cannot save title");
        return reply_object(QJsonObject{{"status","success"}});
    }
    if(request.contains("title"))
        return reply_error("title is valid only for TITLE");

    if(type == "CMD")
    {
        auto fail = [&](const QString& error)
        {
            reply_object(QJsonObject{{"status","error"},{"result",QJsonArray{
                QJsonObject{{"status","error"},{"error",error}}}}});
        };
        auto window = request["window"].toString();
        auto command = request["command"];
        if(command.isUndefined() || command.isNull())
            return fail("missing command field");
        if(window.isEmpty())
            return fail("missing target window field");
        std::vector<std::vector<std::string>> cmds;
        std::vector<std::string> cmd0_list;
        for(const auto& value :
            (command.isArray() ? command.toArray() : QJsonArray{command}))
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
        if(cmds.empty())
            return fail("missing command field");

        QWidget* target = nullptr;
        MainWindow* main_target = nullptr;
        for(auto* each : QApplication::allWidgets())
        {
            if(each->property("busy").toBool())
                return fail("another CMD is running; check opened windows");
            if(auto* main = qobject_cast<MainWindow*>(each))
                main_target = main;
            if(get_window_id(each) == window)
                target = each;
        }
        if(!target)
            return fail("target window not found, terminated by user?");

        auto target_type = window == "main" ? QString("main") :
                           window.startsWith("tracking") ? "tracking" :
                           window.startsWith("recon") ? "recon" : "image";
        auto target_title = target_type == "main" ? QString() :
                            QFileInfo(target->windowTitle()).fileName();
        auto compact = QString::fromUtf8(tipl::merge(cmd0_list,','));
        ai_info::record_history(info,QJsonObject{
            {"type","request"},
            {"text",compact+" \u2192 "+target_type+" window "+target_title},
            {"window",window}});


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
                auto execute = [&](auto* window,bool success)
                {
                    if(!success)
                    {
                        error = QString::fromUtf8(window->error_msg);
                        error = (error.isEmpty() ? "command failed" : error)+
                                ". Read ai/DSI_STUDIO_AI_MANUAL.md and retry.";
                    }
                };
                if(command_name == "voice")
                {
                    if(main_target)
                        execute(main_target,main_target->command(cmd,command_source::AI));
                    else
                        error = "main window not found";
                }
                else if(auto* window = qobject_cast<MainWindow*>(target))
                    execute(window,window->command(cmd,command_source::AI));
                else if(auto* window = qobject_cast<tracking_window*>(target))
                    execute(window,window->command(cmd,command_source::AI));
                else if(auto* window = qobject_cast<reconstruction_window*>(target))
                    execute(window,window->command(cmd,command_source::AI));
                else if(auto* window = qobject_cast<view_image*>(target))
                    execute(window,window->command(cmd,command_source::AI));
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

        return reply_object(QJsonObject{
            {"status",results.last().toObject()["status"]},{"result",results}});
    }

    if(type == "CHAT")
    {
        if(chat.isEmpty() && reasoning.isEmpty())
            return reply_error("missing chat or reasoning");
        return reply_object(QJsonObject{{"status","success"}});
    }

    if(type == "LIST")
    {
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
                {"status",waiting ? "waiting" : busy ? "busy" : "idle"},
                {"title",QDir::fromNativeSeparators(window->windowTitle())}
            };
            application_busy |= busy;
        }

        return reply_object(QJsonObject{
            {"status","success"},
            {"application",QJsonObject{
                {"status",modal ? "waiting" :
                          application_busy ? "busy" : "idle"}}},
            {"windows",windows}});
    }

    if(type == "LOG")
    {
        QByteArray output;
        {
            std::lock_guard<std::mutex> lock(console.edit_buf);
            auto end = console.total_size;
            auto first = end-quint64(console.history.size());
            auto begin = std::max(info.log_position,first);
            bool capped = end-begin > 16*1024;
            if(capped)
                begin = end-16*1024;
            auto text = console.history.mid(qsizetype(begin-first));
            if(capped)
                text.remove(0,text.indexOf('\n')+1);
            text.remove(ansi_escape);
            QStringList lines;
            for(const auto& line : text.split('\n'))
                if(!line.contains(ai_debug_tag))
                    lines << line;
            output = lines.join('\n').right(4*1024).toUtf8();
            info.log_position = end;
        }
        return reply_object(QJsonObject{
            {"status","success"},{"output",QString::fromUtf8(output)}});
    }

    reply_error("unknown request");
}

void AIAgent::show_ai_project(ai_info& info,QJsonObject added_entry)
{
    const auto& history = info.projects;
    if(history.isEmpty())
        return;

    auto* item = info.project_items;
    if(!item)
    {
        item = new QListWidgetItem;
        item->setData(Qt::UserRole,info.sessions);
        ui->ai_project_list->insertItem(0,item);
        info.project_items = item;

        auto* row = new QWidget;
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
        layout->setSpacing(2);
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
    title->setText(info.title());
    title->setToolTip(title->text());
    item->setSizeHint(QSize(0,row->sizeHint().height()));

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
        row->findChild<QTimer*>()->start();
    }

    if(current != item)
        return;

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
        auto content = entry["text"].toString();
        auto reasoning = ui->ai_show_reasoning->isChecked() ?
                         entry["reasoning"].toString().trimmed() : QString();
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
                        .arg(color,(user ? QString("You") : info.agent_name).toHtmlEscaped()+" &middot; ",time,content);

        auto cursor = ui->ai_chat_history->document()->
                      rootFrame()->lastCursorPosition();
        cursor.insertHtml(
            QString("<table width=\"100%\" cellspacing=\"3\" "
                    "cellpadding=\"7\"><tr>%1</tr></table>")
                .arg(user ? "<td width=\"20%\"></td>"+cell :
                         cell+"<td width=\"20%\"></td>"));
    };

    const bool paired = added_type == "assistant" && history.size() > 1 &&
            history[history.size()-2]["type"] == "request";
    const bool rebuild = added_type.isEmpty() || added_type == "request" || paired;

    if(rebuild)
    {
        auto standalone_request = [&](int index)
        {
            return history[index]["type"] == "request" &&
                   (index+1 == history.size() || history[index+1]["type"] != "assistant");
        };
        ui->ai_chat_history->clear();
        for(int index = 0;index < history.size();++index)
        {
            auto entry = history[index];
            auto type = entry["type"].toString();
            if(type == "request")
            {
                if(!standalone_request(index))
                    continue;
                auto combined = entry;
                QStringList activities{
                    entry["text"].toString().section(" \u2192 ",0,0)};
                auto window = entry["window"].toVariant().toString();
                auto end = index;
                while(!window.isEmpty() && end+1 < history.size() &&
                      standalone_request(end+1) &&
                      history[end+1]["window"].toVariant().toString() == window)
                    activities << history[++end]["text"].toString().
                                  section(" \u2192 ",0,0);
                if(end != index)
                {
                    auto target =
                        entry["text"].toString().section(" \u2192 ",1);
                    combined["text"] = activities.join(", ")+" \u2192 "+target;
                }
                append(combined,{},end == index ? QString() :
                       history[end]["time"].toString());
                index = end;
                continue;
            }
            auto activity = type == "assistant" && index &&
                            history[index-1]["type"] == "request" ?
                            history[index-1]["text"].toString() :
                            QString();
            append(entry,activity);
        }
    }
    else
        append(added_entry,{});

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
        auto resolved = resolve_model(profiles,current_model_name,
            settings.value("ai/default_model").toString(),current_model_info);
        current_model_name = resolved.first;
        current_model_info = resolved.second;
        update_agent_status_label();
    }
}
void AIAgent::refresh_codex_models(const QString& path)
{
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
    QJsonObject entry{{"type",type},{"text",text}};
    ai_info::record_history(info,entry);
    show_ai_project(info,entry);
}

void AIAgent::init_agent_model_combo(QComboBox& agent,QComboBox& model,QObject* context)
{
    agent.addItem("Codex");
    agent.addItem("Claude");
    agent.setCurrentIndex(current_agent_index);
    set_model_selector(model,agent_entries[agent.currentIndex()].profiles);
    model.setCurrentText(current_model_name);
    connect(&agent,QOverload<int>::of(&QComboBox::currentIndexChanged),
            context,[this,&model](int index){set_model_selector(model,agent_entries[index].profiles);});
}

bool AIAgent::try_connect_github_issue(const QString& url)
{
    ui->ai_project_list->setCurrentItem(nullptr);
    ui->ai_chat_history->setPlainText("Connecting to "+url+"...");

    QString error;
    if(!connect_github_issue(url,error))
    {
        ui->ai_chat_history->append("Connect failed: "+error);
        set_ai_status("GitHub issue connect failed: "+error,true);
        return false;
    }
    web_agent_active_session = true;
    github_last_issue_url = url;
    update_send_button();
    ui->ai_chat_history->setPlainText("Connected to "+url);
    set_ai_status("Connected to GitHub issue.",true);
    return true;
}

void AIAgent::update_agent_status_label()
{
    static const QString dot = QString(" ")+QChar(0x00B7)+" "; // middle dot separator
    QString text = (current_agent_index == int(ai_provider::Codex) ? "Codex" : "Claude") +
                   dot + current_model_name;
    if(current_model_info.contains("provider"))
        text += dot+"Ollama@"+ai_ollama_url(settings).first.host();
    ui->ai_agent_status->setText(text);
}

// mirrors non-editable QComboBox::setCurrentText: silently does nothing if
// name isn't "default" and isn't a known profile for the current agent
void AIAgent::try_set_current_model(const QString& name)
{
    if(name == "default")
    {
        current_model_name = "default";
        current_model_info = QJsonObject();
        return;
    }
    const auto& profiles = agent_entries[current_agent_index].profiles;
    if(profiles.contains(name))
    {
        current_model_name = name;
        current_model_info = profiles[name].toObject();
    }
}

void AIAgent::update_send_button()
{
    ui->ai_show_reasoning->setVisible(!web_agent_active_session);
    if(web_agent_active_session)
    {
        ui->ai_send_message->setText(github_issue_api.isEmpty() ? "Resume" : "Stop");
        return;
    }
    auto* item = ui->ai_project_list->currentItem();
    bool running = item && ai_infos[item->data(Qt::UserRole).toString()].processes;
    ui->ai_send_message->setText(running ? "Stop" : "Send");
}

// resume: reopens the same dialog for an existing web-agent session — locked
// to "Web agent" (radios disabled) and the local agent/model panel disabled,
// only the issue URL (defaulted to the last one) can still be changed
bool AIAgent::run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                                   bool& web,int& agent_index,QString& model_name,QString& issue_url)
{
    QDialog dialog(this);
    dialog.setWindowTitle(title);
    QVBoxLayout layout(&dialog);

    QRadioButton local_radio("Local agent");
    QRadioButton web_radio("Web agent (ChatGPT via GitHub issue)");

    QWidget local_panel;
    QFormLayout local_layout(&local_panel);
    QComboBox agent,model;
    init_agent_model_combo(agent,model,&dialog);
    local_layout.addRow("Agent:",&agent);
    local_layout.addRow("Model:",&model);

    bool has_token = !settings.value("ai/github_token").toString().isEmpty();
    QWidget web_panel;
    QFormLayout web_layout(&web_panel);
    QLabel web_agent_label("ChatGPT");
    QLineEdit issue_url_edit(resume ? github_last_issue_url : QString());
    issue_url_edit.setPlaceholderText("https://github.com/owner/repo/issues/1");
    web_layout.addRow("Agent:",&web_agent_label);
    web_layout.addRow("Issue URL:",&issue_url_edit);
    QLabel hint(has_token ? QString() :
        "Set up a GitHub token in AI Settings first.");
    hint.setStyleSheet("color:#b00020;");
    hint.setVisible(!has_token);

    if(resume)
    {
        web_radio.setChecked(true);
        local_radio.setEnabled(false);
        web_radio.setEnabled(false);
        local_panel.setEnabled(false);
    }
    else
    {
        local_radio.setChecked(true);
        web_radio.setEnabled(has_token);
    }

    layout.addWidget(&local_radio);
    layout.addWidget(&local_panel);
    layout.addWidget(&web_radio);
    layout.addWidget(&web_panel);
    layout.addWidget(&hint);

    auto update_panels = [&]()
    {
        local_panel.setVisible(local_radio.isChecked());
        web_panel.setVisible(web_radio.isChecked());
    };
    update_panels();
    connect(&local_radio,&QRadioButton::toggled,&dialog,[&](bool){update_panels();});
    connect(&web_radio,&QRadioButton::toggled,&dialog,[&](bool){update_panels();});

    QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    buttons.addButton(accept_text,QDialogButtonBox::AcceptRole);
    layout.addWidget(&buttons);
    connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);

    if(dialog.exec() != QDialog::Accepted)
        return false;

    web = web_radio.isChecked();
    agent_index = agent.currentIndex();
    model_name = model.currentText();
    issue_url = issue_url_edit.text().trimmed();
    return true;
}

void AIAgent::new_chat_dialog(bool resume)
{
    bool web = false;
    int agent_index = 0;
    QString model_name,issue_url;
    if(!run_new_chat_dialog(resume,resume ? "Resume Chat" : "New Chat",resume ? "Resume" : "Start",
                             web,agent_index,model_name,issue_url))
        return;

    if(web)
    {
        try_connect_github_issue(issue_url);
        return;
    }

    if(!github_issue_api.isEmpty())
        disconnect_github_issue(); // leaving web-agent mode for a local chat
    web_agent_active_session = false;
    update_send_button();

    current_agent_index = agent_index;
    try_set_current_model(model_name);
    update_agent_status_label();
    ui->ai_project_list->setCurrentItem(nullptr); // no-op (no signal) if already unselected,
                                                   // so the history clear below can't be skipped
    ui->ai_chat_history->clear();
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
    bool web = false;
    int agent_index = 0;
    QString model_name,issue_url;
    if(!run_new_chat_dialog(false,"Change Agent/Model","Save",web,agent_index,model_name,issue_url))
        return;

    if(web)
    {
        try_connect_github_issue(issue_url);
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
    QComboBox agent,model;
    init_agent_model_combo(agent,model,&dialog);
    QCheckBox history("Keep AI chat history");
    history.setChecked(settings.value("ai/keep_history",true).toBool());
    QCheckBox debug("Enable debug mode");
    debug.setChecked(settings.value("ai/debug").toBool());
    QLineEdit github_pat(settings.value("ai/github_token").toString());
    github_pat.setEchoMode(QLineEdit::Password);
    github_pat.setPlaceholderText("required to connect a GitHub issue");
    layout.addRow("Ollama host/IP:",&host);
    layout.addRow("Ollama port:",&port);
    layout.addRow("Default agent:",&agent);
    layout.addRow("Default model:",&model);
    layout.addRow(&history);
    layout.addRow(&debug);
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
    settings.setValue("ai/debug",debug.isChecked());
    settings.setValue("ai/github_token",github_pat.text().trimmed());
    ai_debug_enabled() = debug.isChecked();
    settings.setValue("ai/default_agent",agent.currentIndex());
    settings.setValue("ai/default_model",model.currentText());
    if(!ui->ai_project_list->currentItem())
    {
        current_agent_index = agent.currentIndex();
        try_set_current_model(model.currentText());
        update_agent_status_label();
    }

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
            set_ai_status("Ollama is not configured.",true);
            QMessageBox::warning(
                this,"AI Agent","Set the Ollama host/IP in AI Settings first.");
            return launch;
        }
    }
    if(launch.model.startsWith("default",Qt::CaseInsensitive))
        launch.model.clear();

    if(session.isEmpty() && provider == ai_provider::Claude)
    {
        session = QUuid::createUuid().toString(QUuid::WithoutBraces);
        info = ai_info::create(session,launch.name);
    }
    if(info && info->model_settings != launch.model_setting)
    {
        info->model_settings = launch.model_setting;
        if(!info->projects.isEmpty())
        {
            info->projects[0]["model_settings"] = info->model_settings;
            write_history(*info,QIODevice::Truncate,info->projects);
        }
    }

    auto restore_new_chat = [=](const QString& message,bool show_history)
    {
        for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
            button->setEnabled(true);
        ui->ai_chat_input->setPlainText(text);
        if(show_history)
            ui->ai_chat_history->setPlainText(message);
        QMessageBox::warning(this,"AI Agent",message);
    };
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
        info->processes = process;
    else
        for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
            button->setEnabled(false);

    if(input == ai_input::User)
    {
        if(info)
            add_ai_history(*info,"user",text);
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
        set_ai_status(message,true);

        if(session.isEmpty())
            restore_new_chat(message,true);
        else
        {
            auto& info = ai_infos[session];
            info.processes = nullptr;
            info.prompts.append(text);
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
        set_ai_status((user_stopped ? launch.name+" stopped." : launch.name+" finished."),true);
        auto session = process->objectName();
        ai_log(launch.name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        bool failed = !user_stopped && (exit_code || exit_status == QProcess::CrashExit);
        auto error_message = user_stopped ? QString("Stopped by user.") :
                              ("error code:"+QString::number(exit_code)+" "+
                              QString::fromUtf8(error)).trimmed();
        if(failed)
            ai_log(error_message);

        if(session.isEmpty())
        {
            auto message = failed ? error_message :
                           "AI agent ended before creating a new chat.";
            restore_new_chat(message,false);
        }
        else
        {
            auto& info = ai_infos[session];
            info.processes = nullptr;

            auto pending = info.prompts.join("\n\n");
            info.prompts.clear();
            if(!pending.isEmpty())
                start_ai(session,pending,ai_input::Pending);
            else if(failed || user_stopped)
                add_ai_history(info,"activity",error_message);
            else if(auto history_size = process->property("history_size");
                    history_size.isValid() &&
                    info.projects.size() == history_size.toInt())
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
    ai_launch launch,QString session,const QString& text,bool new_session)
{
    auto* process = launch.process;
    if(!launch.model_url.isEmpty())
    {
        auto env = process->processEnvironment();
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

    process->setProperty("history_size",ai_infos[session].projects.size());

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
                       ai_status_activity != "Thinking")
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
                    if(auto* info = ai_info::find(process->objectName()))
                        add_ai_reply(*info,chats.join('\n').trimmed(),
                                     reasonings.join('\n').trimmed());
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
        "--allowedTools","Bash(bash ./dsi.sh:*),PowerShell(./dsi.ps1:*)",
        new_session ? "--session-id" : "--resume",session};
    if(!launch.model.isEmpty())
        args << "--model" << launch.model;
    return args;
}
QStringList AIAgent::configure_codex(
    ai_launch launch,QString session,const QString& text)
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
                auto* info = ai_info::create(
                    event["thread_id"].toString(),launch.name);
                if(info)
                    info->model_settings = launch.model_setting;
                if(info && process->objectName().isEmpty())
                {
                    process->setObjectName(info->sessions);
                    info->processes = process;
                    add_ai_history(*info,"user",text);
                    set_ai_status("Agent session ready.",true);
                    for(auto* button : {ui->ai_new_chat,ui->ai_send_message})
                        button->setEnabled(true);
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
            if(auto* info = ai_info::find(process->objectName()))
                add_ai_reply(*info,reasoning ? QString() : text,
                             reasoning ? text : QString());
        }
    });

    QStringList args{"exec","--add-dir",ui->ai_work_dir->text()};
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
    Q_ASSERT(provider != ai_provider::Unknown);

    bool new_session = session.isEmpty();
    auto launch = prepare_ai(provider,session,text,input);
    if(!launch.process)
        return;
    auto args = provider == ai_provider::Codex ?
        configure_codex(launch,session,text) :
        configure_claude(launch,session,text,new_session);
    ai_log("start " + launch.executable +
           " args: " + args.join(" ").remove("\n"));
    set_ai_status("Starting "+launch.name+"...");
    launch.process->start(launch.executable,args);
}

void AIAgent::on_ai_send_message_clicked()
{
    if(web_agent_active_session)
    {
        if(!github_issue_api.isEmpty())
        {
            disconnect_github_issue();
            set_ai_status("GitHub issue channel stopped.",true);
            return;
        }
        new_chat_dialog(true);
        return;
    }

    auto* item = ui->ai_project_list->currentItem();
    if(item)
    {
        auto& info = ai_infos[item->data(Qt::UserRole).toString()];
        if(info.processes)
        {
            info.prompts.clear(); // stop means stop: no auto-continue into a queued message
            info.processes->setProperty("user_stopped",true);
            info.processes->terminate();
            return;
        }
    }

    auto text = ui->ai_chat_input->toPlainText().trimmed();
    if(text.isEmpty())
        return;

    start_ai(item ? item->data(Qt::UserRole).toString() : QString(),
             text,ai_input::User);
    update_send_button();
}
