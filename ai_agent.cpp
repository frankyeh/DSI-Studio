#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QColor>
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
#include <QVBoxLayout>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "ai_agent.hpp"
#include "cmd/ai.hpp"
#include "ui_ai_agent.h"
#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "TIPL/tipl.hpp"

constexpr qsizetype ai_debug_truncate_length = 300; // level 1 (truncated) caps each logged line to this many characters
bool is_valid_session_id(const QString& id)
{
    return !QUuid(id).toString(QUuid::WithoutBraces).compare(id,Qt::CaseInsensitive);
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
AIAgent::AIAgent(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::AIAgent)
{
    ui->setupUi(this);
    ai_debug_level = settings.value("ai/debug",0).toInt();
    ui->ai_work_dir->setText(main_window.work_dir());
    // keeps the field in sync with the selected chat's own dispatch directory (model_settings["cwd"]),
    // the same value run_shell's "cd" updates; also used as --add-dir when launching Codex/Claude
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
            if(info.is_running())
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

    github_timer.setSingleShot(true);
    connect(&github_timer,&QTimer::timeout,this,&AIAgent::poll_github_issue);

    refresh_agent_executables();
    if(agent_entries[int(ai_provider::Codex)].executable.isEmpty() &&
       !agent_entries[int(ai_provider::Claude)].executable.isEmpty())
        current_agent_index = int(ai_provider::Claude);
    update_agent_status_label();
    // not refreshed here: agent_login_info() runs a blocking CLI subprocess per provider, and AIAgent is
    // constructed eagerly at MainWindow startup whether or not this window is ever opened. showEvent()
    // refreshes it before the buttons are ever actually seen -- the .ui defaults ("Sign in to Codex...",
    // "Sign in to Claude...") are shown only in that brief unshown window, never rendered to the user.
    for(auto [button,provider] : {std::pair{ui->ai_codex_login,ai_provider::Codex},
                                   std::pair{ui->ai_claude_login,ai_provider::Claude}})
        connect(button,&QPushButton::clicked,this,[this,provider]
        {
            if(agent_entries[int(provider)].executable.isEmpty()) // stale showEvent() check -- the window may have stayed open since before an install finished, so retry once before assuming it's still missing
                refresh_agent_executables();
            if(agent_entries[int(provider)].executable.isEmpty()) // still not installed -- nothing to sign into yet
                QDesktopServices::openUrl(agent_install_url(provider));
            else
                run_agent_login(provider);
            refresh_login_buttons();
        });

    auto* send = new QShortcut(
        QKeySequence(Qt::CTRL|Qt::Key_Return),ui->ai_chat_input);
    send->setContext(Qt::WidgetShortcut);
    connect(send,&QShortcut::activated,this,[this]
    {
        // a fixed "submit" shortcut, never Stop/Resume regardless of what the button currently shows
        auto action = current_send_action();
        if(action == send_action::Send || action == send_action::Queue)
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
        if(auto* found = ai_info::find(session);found && found->processes)
        {
            auto* process = found->processes;
            process->disconnect(); process->kill(); process->deleteLater(); // kill(): a windowless console child never sees terminate()'s WM_CLOSE
        }
        if(session == web_agent_session_id)
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
            return ui->ai_status->hide();
        }

        stop_blink(ui->ai_project_list->itemWidget(item));
        auto* info = selected_info();
        if(!info) // item is a real row, but guard anyway rather than trust that indirectly
            return;
        ui->ai_work_dir->setText(info->model_settings.contains("cwd") ?
            info->model_settings["cwd"].toString() : main_window.work_dir());
        // no longer copies the selected chat's agent/model into the app-wide default: update_agent_status_label()
        // reads this chat's own model_settings directly, and merely looking at a chat shouldn't change what the
        // next New Chat starts with
        update_agent_status_label();
        show_ai_project(*info);
        update_send_button();
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
        auto first = history.first();
        QJsonObject config;
        if(QFile config_file(ai_info::config_file(session));config_file.open(QIODevice::ReadOnly))
            config = QJsonDocument::fromJson(config_file.readAll()).object();
        // config_file() is the current source of truth; fall back to the legacy fields once
        // embedded in the first history entry, for chats saved before this file existed
        auto agent = config.contains("agent") ? config["agent"].toString() : first["agent"].toString();
        // never re-guess the provider from the name once it's been persisted -- that's exactly what
        // misclassifies an AgentServer session; only a legacy config predating persistence falls back to a guess
        auto* ai = config.contains("provider") ?
            ai_info::create(session,agent,ai_provider(config["provider"].toInt())) :
            ai_info::create(session,agent,ai_provider::Infer);
        if(!ai)
            continue;
        // absent "established" means a config predating this field, from back when save_config() itself
        // only ever wrote for an established session -- so absent defaults to true, same as those old files
        // always implied. Present-and-false means a real, current record of a session that never got a
        // backend thread; loading it as Completed/resumable would try to --resume an id nothing ever confirmed
        bool established = config["established"].toBool(true);
        set_ai_status(ai->sessions,established ? session_status::Completed : session_status::New,
                      established ? "Previous chat loaded." : "Previous attempt never connected.");
        ai->model_settings = config.contains("model_settings") ?
            config["model_settings"].toObject() : first["model_settings"].toObject();
        ai->project_titles = settings.value("ai/title/"+session).toString();
        ai->projects = std::move(history);
        show_ai_project(*ai);
    }
    if(ui->ai_project_list->count())
        ui->ai_project_list->setCurrentRow(0);
    // GitHub issue channels are never auto-reconnected at startup; use Resume to reconnect a chat explicitly
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


bool AIAgent::connect_github_issue(const QString& url_text,QString& error)
{
    // snapshot now, so a later new-chat edit cannot swap the identity mid-poll (github_request() uses this member for the whole session)
    github_token = settings.value("ai/github_token").toString().trimmed();
    if(github_token.isEmpty())
        return error = "no GitHub token configured; set one when starting the ChatGPT Web agent "
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

    // a request can have been executed (side effects already ran) without its result ever being
    // confirmed published, e.g. DSI Studio exited in between; the durable marker written just before
    // execution survives that, so report the outcome as unknown here instead of silently re-running it
    {
        QSettings settings;
        auto pending_issue = settings.value("ai/github_pending_issue").toString();
        auto pending_id = settings.value("ai/github_pending_id",0).toLongLong();
        if(!pending_issue.isEmpty() && pending_issue == issue_api.toString() && pending_id > last_id)
        {
            ai_log("github connect: request "+QString::number(pending_id)+
                   " was executing when DSI Studio last stopped; publishing an unknown-outcome result instead of re-running it");
            settings.remove("ai/github_pending_issue");
            settings.remove("ai/github_pending_id");
            publish_github_result(QJsonObject{
                {"id",pending_id},{"last_id",pending_id},{"dsi_session_result",true},{"issue",issue_number},
                {"state","error"},
                {"response",QJsonObject{{"status","error"},
                    {"error","previous execution outcome unknown after a DSI Studio restart; "
                             "the command may or may not have completed - verify manually before resending"}}}});
        }
    }
    return true;
}

void AIAgent::disconnect_github_issue()
{
    if(github_issue_api.isEmpty()) // nothing to do; callers no longer need to check this themselves
        return;
    ++github_connection_id; // reject any callback still in flight from this connection
    github_timer.stop();
    github_issue_api.clear();
    github_result_api.clear();
    github_etag.clear();
    github_token.clear();
    github_last_id = 0;
    github_pending_result = QJsonObject();
    update_send_button(); // flips to "Resume" if still in a web-agent session
    if(auto* info = ai_info::find(web_agent_session_id))
        if(info->status != session_status::Failed) // a deliberate disconnect never replaces a real failure
            set_ai_status(info->sessions,session_status::Completed,"GitHub issue channel stopped.");
}

bool AIAgent::handle_github_reply(QNetworkReply* reply,quint64 connection_id,int& status,QByteArray& data)
{
    reply->deleteLater();
    if(connection_id != github_connection_id)
        return false; // this connection was superseded (disconnect, or a fresh reconnect)

    status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    data = reply->readAll();
    if(auto delay = github_retry_delay(reply,data))
    {
        if(!github_issue_api.isEmpty())
            github_timer.start(delay); // rate limited (429, or 403 that means the same thing)
        return false;
    }
    if(github_permanent_failure(status))
    {
        disconnect_github_issue();
        if(auto* info = ai_info::find(web_agent_session_id)) // disconnect_github_issue() already set Completed -- this is a real failure, not a deliberate stop
            set_ai_status(info->sessions,session_status::Failed,
                          "GitHub issue channel authorization failed.");
        return false;
    }
    return true;
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
        int status = 0;
        QByteArray data;
        if(!handle_github_reply(reply,connection_id,status,data))
            return;

        auto restart = [this](int delay_ms = 500)
        {if(!github_issue_api.isEmpty()) github_timer.start(delay_ms);};

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
        if(!is_valid_session_id(request_obj["session"].toString()))
            return publish_github_result(stamp(QJsonObject{
                {"state","error"},
                {"response",QJsonObject{{"status","error"},
                    {"error","malformed request: session must be a UUID"}}}}));

        bool include_log = request_obj["include_log"].toBool();
        request_obj.remove("id");
        request_obj.remove("include_log");
        request_obj["agent"] = "Codex/ChatGPT-GitHub";

        auto session_id = request_obj["session"].toString();
        bool set_title = !ai_info::find(session_id);
        auto* web_info = ai_info::find(web_agent_session_id);
        if(web_info && web_info->status == session_status::New)
            assign_ai_session(web_agent_session_id,session_id);
        web_agent_session_id = session_id;
        if(auto* info = ai_info::create(session_id,"Codex/ChatGPT-GitHub",ai_provider::Infer)) // records which issue this session is bound to, so a restart can auto-resume polling it
        {
            set_ai_status(info->sessions,session_status::Thinking,"GitHub request received");
            // stored as "<owner>/<repo>/issues/<number>"; github_issue_api is always
            // "https://api.github.com/repos/<owner>/<repo>/issues/<number>", built by DSI Studio itself
            info->model_settings["github_issue_url"] =
                QString(github_issue_api.toString()).remove("https://api.github.com/repos/");
            info->save_config();
        }

        // durable marker, survives a crash: if DSI Studio exits before the result below is confirmed
        // published, the next connect_github_issue() sees this and reports the outcome as unknown
        // instead of re-executing the same request
        QSettings().setValue("ai/github_pending_issue",github_issue_api.toString());
        QSettings().setValue("ai/github_pending_id",id);

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

        if(set_title) // omits "agent" so set_title itself can never create a session
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
        int status = 0;
        QByteArray data;
        if(!handle_github_reply(reply,connection_id,status,data))
            return;

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
        {
            QSettings settings;
            if(settings.value("ai/github_pending_id",0).toLongLong() == pending_id)
            {
                settings.remove("ai/github_pending_issue");
                settings.remove("ai/github_pending_id");
            }
        }
        if(closed)
            return disconnect_github_issue();
        if(!github_issue_api.isEmpty())
        {
            // published successfully and the channel stays open -- back to idle, waiting for the next
            // request; otherwise this would sit in Thinking indefinitely with the animation still running
            if(auto* info = ai_info::find(web_agent_session_id))
                set_ai_status(info->sessions,session_status::WaitingUser,"Result published; monitoring GitHub issue");
            github_timer.start(500);
        }
    });
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
    refresh_agent_executables(); // picks up a CLI installed since the window was last shown, before the two refreshes below read agent_entries[...].executable
    refresh_codex_models();
    refresh_login_buttons(); // re-checked every time this window is shown, so a login/logout done outside DSI Studio is picked up
    auto* item = ui->ai_project_list->currentItem();
    stop_blink(item ? ui->ai_project_list->itemWidget(item) : nullptr);
}

void AIAgent::closeEvent(QCloseEvent* event)
{
    // let each process's own QProcess::finished handler (in prepare_ai()) run the real finish lifecycle --
    // it already knows how to tell a fresh, never-established launch (reverts to New) from an established
    // session being stopped (Completed) or a genuine crash (Failed), and handles pending prompts/history/UI.
    // Setting this window's ai_infos to Completed unconditionally here bypassed all of that, e.g. wrongly
    // marking a still-New placeholder (never a real Codex/Claude thread) as resumable
    for(auto& entry : ai_infos)
        if(auto* process = entry.second.processes)
        {
            process->setProperty("user_stopped",true); // finished()'s own handler clears queued prompts for a user_stopped session -- no auto-continue into a queued message right after this window tried to shut everything down
            process->kill(); // kill(): a windowless console child never sees terminate()'s WM_CLOSE
        }
    disconnect_github_issue();
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
    bool running = info.is_running();
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
    if(running)
    {
        if(!text.endsWith('.'))
            text += ".";
    }
    ui->ai_status->show();
    ui->ai_status->setToolTip(text); // full message on hover -- the label itself may show a truncated "..." version
    ui->ai_status->setText(QFontMetrics(ui->ai_status->font()).elidedText(
        text,Qt::ElideRight,ui->ai_status->maximumWidth()-30)); // truncate -- an unbounded message here was pushing the whole window wider
    ui->ai_status->repaint();
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
    if(!found)
    {
        auto agent = request["agent"].toString().trimmed();
        if(agent.isEmpty())
            return void(reply = status_reply("error","missing agent for new session"));
        // AgentServer, never derived from the calling agent's own name: a pipe-dispatched session is always a
        // log/routing record for this dispatcher, never a real local Codex/Claude subprocess, regardless of
        // what the caller names itself -- it can't send a live chat message or have its model changed from
        // the GUI (see current_send_action()/on_ai_agent_status_clicked())
        found = ai_info::create(session,agent,ai_provider::AgentServer);
        set_ai_status(found->sessions,session_status::Thinking,"Agent request received"); // save_config() skips a still-New session
        if(auto model = request["model"].toString().trimmed();!model.isEmpty())
            found->model_settings["model"] = model;
        found->save_config();
    }
    ai_info& info = *found;
    set_ai_status(session,session_status::Thinking,"Processing agent request");

    reply.clear();
    ai_log("received: "+QString::fromUtf8(data));
    auto chat = request["chat"].toString().trimmed();
    auto reasoning = request["reasoning"].toString().trimmed();
    dispatching_info = &info;
    auto result = main_window.dispatch_cmd(info,request); // MainWindow's command center handles everything
    dispatching_info = nullptr;
    // dispatch_cmd() only finished the DSI command itself -- a live local process is still mid-turn (it
    // dispatched this as one of its own tool calls and is waiting on our reply to continue), so that's
    // still Thinking, not idle. Only a transport with no ongoing turn of its own (GitHub, or nothing at
    // all) settles to WaitingUser here
    if(info.processes && info.processes->state() != QProcess::NotRunning)
        set_ai_status(session,session_status::Thinking,
                      "Command completed; waiting for agent input");
    else if(github_connected(info))
        set_ai_status(session,session_status::WaitingUser,
                      "Request completed; monitoring GitHub issue");
    else
        set_ai_status(session,session_status::WaitingUser,
                      "Request completed; waiting for next request.");

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
    }
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

        connect(title,&QPushButton::clicked,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
        connect(button,&QToolButton::pressed,this,
                [this,item]{ui->ai_project_list->setCurrentItem(item);});
    }

    auto* row = ui->ai_project_list->itemWidget(item);
    auto* title = row->findChild<QPushButton*>();
    item->setText({});
    // never touched (no content, no title) -- not "currently New", which a reconnecting, previously-used
    // chat also transiently is (see session_status), and shouldn't flash back to this placeholder label for
    auto chat_title = info.projects.isEmpty() && info.project_titles.isEmpty() ?
        "New "+info.agent_name+" Chat" : info.title();
    title->setText((info.provider == ai_provider::ChatGPT ? QString("🌐 ") : QString())+chat_title);
    title->setToolTip(title->text());
    item->setSizeHint(QSize(0,row->sizeHint().height()));

    update_ai_status(info);

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

    show_ai_history(info,std::move(added_entry));
}

void AIAgent::show_ai_history(ai_info& info,QJsonObject added_entry)
{
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
        // the current default model's own profile may have just changed (or disappeared) -- refresh its
        // cached info; an unrecognized name is left exactly as it was. profiles.value(), not profiles[] --
        // operator[] on this non-const profiles would silently insert a spurious entry for a missing key
        // (e.g. an empty current_model_name inserting a bogus "" profile that then shows up as a blank
        // entry in the model dropdown) the same way std::map::operator[] does
        if(current_model_name.isEmpty() || profiles.contains(current_model_name))
            current_model_info = profiles.value(current_model_name).toObject();
        update_agent_status_label();
    }
}
void AIAgent::refresh_agent_executables() // re-run discovery so an install completed after DSI Studio was already running (e.g. via the sidebar's Install button) is picked up without a restart -- called from the constructor and showEvent()
{
    QString codex_path = QStandardPaths::findExecutable("codex");
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

    QString claude_path = QStandardPaths::findExecutable("claude");
#ifdef Q_OS_WIN
    if(claude_path.isEmpty())
        claude_path = QDir::homePath()+"/.local/bin/claude.exe";
#endif
    if(!QFileInfo::exists(claude_path))
        claude_path.clear();

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
        static const QStringList claude_models{"sonnet","fable","opus","haiku"};
        update_agent_models(int(ai_provider::Claude),claude_models,false);
        ai_log("Claude models: "+claude_models.join(", "));
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

QString AIAgent::agent_login_info(ai_provider provider)
{
    const auto& executable = agent_entries[int(provider)].executable;
    if(executable.isEmpty())
        return {};
    bool is_codex = provider == ai_provider::Codex;
    QProcess process;
    process.start(executable,is_codex ? QStringList{"login","status"} : QStringList{"auth","status"});
    if(!process.waitForStarted(3000) || !process.waitForFinished(10000))
        return {};
    if(is_codex)
    {
        if(process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
            return {};
        // codex login status has no --json/structured output (email/plan aren't exposed), only this
        // free-text auth-method line -- see https://github.com/openai/codex/issues/19866
        auto output = QString::fromUtf8(process.readAllStandardOutput());
        return output.contains("API key",Qt::CaseInsensitive) ? "API key" :
               output.contains("ChatGPT",Qt::CaseInsensitive) ? "ChatGPT" :
               output.contains("Agent Identity",Qt::CaseInsensitive) ? "Agent Identity" : "Signed in";
    }
    auto object = QJsonDocument::fromJson(process.readAllStandardOutput()).object();
    if(!object["loggedIn"].toBool())
        return {};
    auto email = object["email"].toString();
    if(!email.isEmpty())
    {
        auto tier = object["subscriptionType"].toString();
        return tier.isEmpty() ? email : email+" · "+tier.left(1).toUpper()+tier.mid(1);
    }
    auto api_provider = object["apiProvider"].toString();
    return api_provider.isEmpty() ? "API key" : "API key · "+api_provider;
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

void AIAgent::refresh_login_buttons()
{
    auto refresh = [this](ai_provider provider,QPushButton* button,const QString& name)
    {
        if(agent_entries[int(provider)].executable.isEmpty())
        {
            button->setEnabled(true); // clicking opens the CLI's install page
            button->setText("Install "+name);
            return;
        }
        auto info = agent_login_info(provider);
        button->setEnabled(info.isEmpty()); // clickable only while not signed in
        button->setText(info.isEmpty() ? "Sign in to "+name+"..." : name+": "+info);
    };
    refresh(ai_provider::Codex,ui->ai_codex_login,"Codex");
    refresh(ai_provider::Claude,ui->ai_claude_login,"Claude");
}

bool AIAgent::try_connect_github_issue(const QString& url)
{
    if(auto* info = ai_info::find(web_agent_session_id))
        // New only for a genuinely never-established chat -- an already-established chat reconnecting stays
        // "established" (see save_config()) through the attempt, so a save mid-connect can't wrongly persist
        // established:false over it
        set_ai_status(info->sessions,info->status == session_status::New ?
                      session_status::New : session_status::Thinking,
                      "Connecting to "+url);
    update_send_button();
    update_agent_status_label();
    tipl::out() << "connecting to GitHub issue: " << url.toStdString();

    QString error;
    if(!connect_github_issue(url,error))
    {
        tipl::out() << "GitHub issue connect failed: " << error.toStdString();
        // web_agent_session_id (not sidebar selection) is the reliable way to find the chat this connection
        // belongs to; the caller guarantees it already refers to a real chat (created fresh, or being resumed)
        if(auto* info = ai_info::find(web_agent_session_id))
            set_ai_status(info->sessions,session_status::Failed,
                          "GitHub issue connection failed: "+error);
        return false;
    }
    tipl::out() << "connected to GitHub issue: " << url.toStdString();
    if(auto* info = ai_info::find(web_agent_session_id))
    {
        // status flips to WaitingUser (established) BEFORE save_config() -- save_config()'s "established"
        // field reads status at the moment it's called, and this chat may be an already-established one just
        // reconnecting (status still New here from the "Connecting..." update above); saving while still New
        // would wrongly persist established:false over a genuinely established chat
        set_ai_status(info->sessions,session_status::WaitingUser, // established and idle -- Thinking is reserved for actually processing a request (see poll_github_issue())
                      "Connected; monitoring GitHub issue");
        // bound the moment the connection succeeds, not deferred until a request happens to arrive
        // (poll_github_issue() also does this for the reactive/resume case) -- the chat's own record is
        // now always current, so update_agent_status_label() never needs to prefer github_issue_api over it
        info->model_settings["github_issue_url"] =
            QString(github_issue_api.toString()).remove("https://api.github.com/repos/");
        info->save_config();
    }
    update_send_button();
    update_agent_status_label();
    return true;
}

void AIAgent::update_agent_status_label()
{
    static const QString dot = QString(" ")+QChar(0x00B7)+" "; // middle dot separator
    auto* info = selected_info();
    bool agent_server = info && info->provider == ai_provider::AgentServer; // a log/routing record, no agent/model of its own to show or change
    ui->ai_agent_status->setVisible(!agent_server);
    if(!agent_server)
    {
        if(info && info->provider == ai_provider::ChatGPT)
        {
            // model_settings["github_issue_url"] is bound the moment a connection succeeds (see
            // try_connect_github_issue()), so this chat's own record is always current -- no need to prefer
            // the live github_issue_api over it
            auto path = info->model_settings["github_issue_url"].toString();
            ui->ai_agent_status->setText(path.isEmpty() ? "ChatGPT(Web)" : "ChatGPT(Web)"+dot+path);
        }
        else // a local chat (its own model, since it can differ from the app-wide default once changed) or
             // nothing selected (the app-wide default that the next New Chat will start with) -- same formatting
        {
            auto format = [&](bool is_codex,const QString& model_name,const QJsonObject& model_info)
            {
                QString text = (is_codex ? "Codex" : "Claude") +
                               dot + (model_name.isEmpty() ? QString("default") : model_name);
                if(model_info.contains("provider"))
                    text += dot+"Ollama@"+ai_ollama_url(settings).first.host();
                return text;
            };
            ui->ai_agent_status->setText(info ?
                format(info->provider == ai_provider::Codex,info->model_settings["model"].toString(),
                       info->model_settings["info"].toObject()) :
                format(current_agent_index == int(ai_provider::Codex),current_model_name,current_model_info));
        }
    }
    // the send button's enabled state/label depends on the same selected-chat context above, so it's
    // refreshed here on every call rather than relying on each call site to also remember it
    update_send_button();
}

void AIAgent::try_set_current_model(const QString& name) // writes the app-wide default (see the member declaration); name is empty for "default" (model_combo_key()'s data value, not the "default" UI label) or a specific model name -- both are always meaningful, never a no-op
{
    const auto& profiles = agent_entries[current_agent_index].profiles;
    current_model_name = name;
    current_model_info = profiles.contains(name) ? profiles[name].toObject() : QJsonObject();
}

void AIAgent::set_chat_model(ai_info& info,const QString& name) const // writes directly into this chat's own model_settings; same name resolution as try_set_current_model()
{
    const auto& profiles = agent_entries[int(info.provider)].profiles;
    info.model_settings["model"] = name;
    info.model_settings["info"] = profiles.contains(name) ? profiles[name].toObject() : QJsonObject();
    info.save_config();
}

ai_info* AIAgent::selected_info() const
{
    auto* item = ui->ai_project_list->currentItem();
    // find(), not ai_infos[id] -- this is meant to resolve an existing chat, never manufacture a blank one
    // for a stale/unrecognized id
    return item ? ai_info::find(item->data(Qt::UserRole).toString()) : nullptr;
}

bool AIAgent::github_connected(const ai_info& info) const
{
    return info.provider == ai_provider::ChatGPT &&
           info.sessions == web_agent_session_id &&
           !github_issue_api.isEmpty();
}

AIAgent::send_action AIAgent::current_send_action() const
{
    auto* info = selected_info();
    bool has_input = !ui->ai_chat_input->toPlainText().trimmed().isEmpty();
    // nothing selected: still lets a typed message start a chat directly (same process as New Chat, minus the
    // dialog -- current_agent_index/current_model_name, the app-wide default, pick the agent/model)
    if(!info)
        return has_input ? send_action::Send : send_action::Disabled;
    if(info->provider == ai_provider::AgentServer) // a log/routing record, no local subprocess to send to
        return send_action::Disabled;
    if(info->provider == ai_provider::ChatGPT)
        return github_connected(*info) ? send_action::Stop : send_action::Resume;
    if(!info->processes) // never launched (or a prior attempt cleanly ended): a fresh launch, always a real send
        return has_input ? send_action::Send : send_action::Disabled;
    if(!has_input)
        return send_action::Stop;
    // start_ai()'s own real distinction: only a live Claude process gets a stdin write; Codex (never mid-turn
    // writable) and a still-connecting Claude both actually queue -- match that here so the label is never a lie
    return (info->provider == ai_provider::Claude && info->processes->state() == QProcess::Running) ?
                send_action::Send : send_action::Queue;
}

void AIAgent::update_send_button()
{
    auto action = current_send_action();
    ui->ai_send_message->setEnabled(action != send_action::Disabled);
    ui->ai_send_message->setText(
        action == send_action::Queue ? "Queue" :
        action == send_action::Stop ? "Stop" :
        action == send_action::Resume ? "Resume" : "Send");
}

bool AIAgent::setup_github_token()
{
    QDialog dialog(this);
    dialog.setWindowTitle("Set Up GitHub Access");
    dialog.setMinimumWidth(460);
    dialog.setStyleSheet(ai_dialog_style());

    auto* root = new QVBoxLayout(&dialog);
    root->setSpacing(12);
    root->setContentsMargins(20,20,20,16);

    auto* title = new QLabel("Connect a GitHub issue channel");
    title->setObjectName("ai_dialog_title");
    auto* subtitle = new QLabel(
        "DSI Studio sends and receives ChatGPT (Web) requests through a private repository issue. "
        "Set this up once: a repository, then a token scoped to it.");
    subtitle->setObjectName("ai_dialog_subtitle");
    subtitle->setWordWrap(true);
    root->addWidget(title);
    root->addWidget(subtitle);

    // a titled, bordered card with a short body line and a single left-aligned action button below it --
    // used only here, for this dialog's two setup steps
    auto add_step_card = [root](const QString& heading,const QString& body,
                                 QPushButton*& action,const QString& action_text)
    {
        auto* card = new QFrame;
        card->setObjectName("ai_step_card");
        auto* card_layout = new QVBoxLayout(card);
        card_layout->setContentsMargins(14,12,14,12);
        card_layout->setSpacing(6);
        auto* heading_label = new QLabel(heading);
        heading_label->setObjectName("ai_step_heading");
        auto* body_label = new QLabel(body);
        body_label->setObjectName("ai_step_body");
        body_label->setWordWrap(true);
        card_layout->addWidget(heading_label);
        card_layout->addWidget(body_label);
        action = new QPushButton(action_text);
        auto* button_row = new QHBoxLayout;
        button_row->addWidget(action);
        button_row->addStretch();
        card_layout->addLayout(button_row);
        root->addWidget(card);
    };

    QPushButton* setup_repo = nullptr;
    add_step_card("Step 1 · Create a private repository",
        "<ol style='margin-left:-20px;'>"
        "<li><b>Repository name*</b> = <i>[any name]</i>, e.g. DSI-Studio-Connect</li>"
        "<li>Choose visibility &rarr; <b>Private</b></li>"
        "<li>Click <b>Create repository</b></li></ol>",
        setup_repo,"Create private repository");

    QPushButton* setup_token = nullptr;
    add_step_card("Step 2 · Create an access token",
        "<ol style='margin-left:-20px;'>"
        "<li><b>Token name*</b> = <i>[any name]</i></li>"
        "<li>Expiration &rarr; select an appropriate duration</li>"
        "<li>Repository access &rarr; <b>Only select repositories</b> &rarr; the repository you just created</li>"
        "<li>Permissions &rarr; Add permissions &rarr; check <b>Issues</b></li>"
        "<li>Issues access &rarr; <b>Read and write</b></li>"
        "<li>Click <b>Generate token</b></li>"
        "<li>Copy the token, then click <b>Paste</b> below</li></ol>",
        setup_token,"Create token");

    auto* token_label = new QLabel("Access token");
    token_label->setObjectName("ai_dialog_subtitle");
    root->addWidget(token_label);
    auto* token_frame = new QFrame;
    token_frame->setObjectName("ai_field_frame");
    auto* token_row = new QHBoxLayout(token_frame);
    token_row->setContentsMargins(10,2,4,2);
    QLineEdit token(settings.value("ai/github_token").toString()); // declared after dialog/token_frame so it is destroyed before them
    token.setEchoMode(QLineEdit::Password);
    token.setPlaceholderText("Paste the token here");
    QPushButton paste("Paste");
    token_row->addWidget(&token,1);
    token_row->addWidget(&paste);
    root->addWidget(token_frame);

    QLabel helper;
    helper.setObjectName("ai_helper");
    helper.setWordWrap(true);
    root->addWidget(&helper);

    QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
    buttons.button(QDialogButtonBox::Save)->setObjectName("ai_primary_button");
    root->addWidget(&buttons);

    auto set_helper = [&](const QString& text)
    {
        helper.setText(text);
        helper.updateGeometry();
        dialog.adjustSize();
    };
    connect(setup_repo,&QPushButton::clicked,&dialog,[&]
    {
        QDesktopServices::openUrl(QUrl("https://github.com/new"));
        set_helper("Create DSI-Studio-Connect as a Private repository, then continue to step 2.");
    });
    connect(setup_token,&QPushButton::clicked,&dialog,[&]
    {
        QApplication::clipboard()->setText(
            "Create a fine-grained GitHub personal access token for DSI Studio:\n"
            "1. Token name*: enter any name.\n"
            "2. Expiration: select an appropriate duration.\n"
            "3. Repository access: choose Only select repositories, then select the private repository created for DSI Studio.\n"
            "4. Permissions: choose Add permissions, then Issues.\n"
            "5. Issues access: choose Read and write.\n"
            "6. Click Generate token.\n"
            "7. Copy the token and click Paste in the DSI Studio dialog.\n"
            "Never paste the token into ChatGPT or a GitHub issue.");
        QDesktopServices::openUrl(QUrl("https://github.com/settings/personal-access-tokens/new"));
        set_helper("Instructions copied. Create and copy the token in GitHub, then return here and click Paste.");
    });
    connect(&paste,&QPushButton::clicked,&dialog,[&]
    {
        auto match = QRegularExpression(
            R"((github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+))").
            match(QApplication::clipboard()->text());
        if(match.hasMatch())
            token.setText(match.captured()),set_helper("Token pasted. Click Save to continue.");
        else
            set_helper("No GitHub token was found in the clipboard.");
    });
    connect(&buttons,&QDialogButtonBox::accepted,&dialog,[&]
    {
        if(token.text().trimmed().isEmpty())
            return set_helper("Create or paste a GitHub token before saving.");
        settings.setValue("ai/github_token",token.text().trimmed());
        dialog.accept();
    });
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
    return dialog.exec() == QDialog::Accepted;
}

// resume only ever applies to the web agent: the Agent combo is locked to ChatGPT and disabled, only the issue URL (defaulted to the last one) can still be changed
bool AIAgent::run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                                   int& agent_index,QString& value)
{
    QDialog dialog(this);
    dialog.setWindowTitle(title);
    dialog.setMinimumWidth(440);
    dialog.setStyleSheet(ai_dialog_style());
    QFormLayout layout(&dialog);
    layout.setSpacing(10);
    layout.setContentsMargins(20,18,20,16);

    QComboBox agent;
    agent.addItem("Codex");
    agent.addItem("Claude");
    agent.addItem("ChatGPT (Web)");
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
    }
    agent.setCurrentIndex(resume ? int(ai_provider::ChatGPT) : current_agent_index);
    agent.setEnabled(!resume);
    layout.addRow("Agent:",&agent);

    QWidget field_container,local,web; // declared before their would-be children below, so they are destroyed after them
    auto* field_layout = new QVBoxLayout(&field_container);
    field_layout->setContentsMargins(0,0,0,0);
    QComboBox model; // not editable -- same as the Agent combo above, which never had the popup-visibility problem an editable combo did
    model.setMaximumHeight(model.sizeHint().height());
    auto* local_layout = new QFormLayout(&local);
    local_layout->setContentsMargins(0,0,0,0);
    local_layout->addRow("Model:",&model);
    // web_agent_session_id (not sidebar selection) is the reliable way to find which chat is being resumed
    auto* resume_info = resume ? ai_info::find(web_agent_session_id) : nullptr;
    auto resume_issue_path = resume_info ? resume_info->model_settings["github_issue_url"].toString() : QString();
    auto* web_layout = new QVBoxLayout(&web);
    web_layout->setContentsMargins(0,0,0,0);
    web_layout->setSpacing(10);

    QPushButton setup_token("Set up GitHub token"); // becomes a disabled "GitHub token ready" status readout once configured -- see update_web()
    auto* token_row = new QHBoxLayout;
    token_row->addWidget(&setup_token);
    token_row->addStretch();
    web_layout->addLayout(token_row);

    auto* issue_card = new QFrame;
    issue_card->setObjectName("ai_step_card");
    auto* issue_card_layout = new QVBoxLayout(issue_card);
    issue_card_layout->setContentsMargins(14,12,14,12);
    issue_card_layout->setSpacing(8);
    auto* issue_heading = new QLabel("Session issue");
    issue_heading->setObjectName("ai_step_heading");
    auto* issue_body = new QLabel("Ask ChatGPT to create the issue, then paste its URL below.");
    issue_body->setObjectName("ai_step_body");
    issue_body->setWordWrap(true);
    issue_card_layout->addWidget(issue_heading);
    issue_card_layout->addWidget(issue_body);

    auto* issue_field_frame = new QFrame;
    issue_field_frame->setObjectName("ai_field_frame");
    auto* issue_row = new QHBoxLayout(issue_field_frame);
    issue_row->setContentsMargins(10,2,4,2);
    QLineEdit issue_url_edit(resume_issue_path.isEmpty() ? QString() : "https://github.com/"+resume_issue_path);
    issue_url_edit.setPlaceholderText("https://github.com/owner/repo/issues/1");
    QPushButton paste_issue("Paste");
    issue_row->addWidget(&issue_url_edit,1);
    issue_row->addWidget(&paste_issue);
    issue_card_layout->addWidget(issue_field_frame);

    QPushButton setup_issue("Ask ChatGPT...");
    auto* issue_button_row = new QHBoxLayout;
    issue_button_row->addWidget(&setup_issue);
    issue_button_row->addStretch();
    issue_card_layout->addLayout(issue_button_row);
    web_layout->addWidget(issue_card);

    QLabel helper;
    helper.setObjectName("ai_helper");
    helper.setWordWrap(true);
    helper.setMinimumWidth(380);
    web_layout->addWidget(&helper);

    field_layout->addWidget(&local);
    field_layout->addWidget(&web);
    layout.addRow(&field_container);

    auto set_helper = [&](const QString& text)
    {
        helper.setText(text);
        helper.updateGeometry();
        dialog.adjustSize();
    };
    auto update_web = [&]
    {
        bool has_token = !settings.value("ai/github_token").toString().trimmed().isEmpty();
        setup_token.setEnabled(!has_token);
        setup_token.setText(has_token ? "GitHub token ready ✓" : "Set up GitHub token");
        for(auto* widget : {static_cast<QWidget*>(&issue_url_edit),
                            static_cast<QWidget*>(&paste_issue),
                            static_cast<QWidget*>(&setup_issue)})
            widget->setEnabled(has_token);
        set_helper(has_token ?
            "GitHub access is ready. Create or paste the session issue URL." :
            "A GitHub token is required. Click Set up GitHub token.");
    };
    auto update_field = [&]
    {
        bool chatgpt = agent.currentIndex() == int(ai_provider::ChatGPT);
        local.setVisible(!chatgpt);
        web.setVisible(chatgpt);
        if(chatgpt)
        {
            if(settings.value("ai/github_token").toString().trimmed().isEmpty())
                setup_github_token();
            update_web();
        }
        if(!chatgpt)
            set_model_selector(model,agent_entries[agent.currentIndex()].profiles,
                // only the agent that's actually active right now keeps its remembered model; switching to a different agent resets to that agent's own "default"
                agent.currentIndex() == current_agent_index ? current_model_name : QString());
    };
    update_field();
    connect(&agent,QOverload<int>::of(&QComboBox::currentIndexChanged),&dialog,[&](int){update_field();});
    connect(&setup_token,&QPushButton::clicked,&dialog,[&]
    {
        setup_github_token();
        update_web();
    });
    connect(&paste_issue,&QPushButton::clicked,&dialog,[&]
    {
        auto match = QRegularExpression(
            R"(https://github\.com/[^\s/]+/[^\s/]+/issues/\d+)",
            QRegularExpression::CaseInsensitiveOption).
            match(QApplication::clipboard()->text());
        if(match.hasMatch())
        {
            issue_url_edit.setText(match.captured());
            set_helper("Issue URL pasted. Click Start to connect.");
        }
        else
            set_helper("No GitHub issue URL was found in the clipboard.");
    });
    connect(&setup_issue,&QPushButton::clicked,&dialog,[&]
    {
        QApplication::clipboard()->setText(
            "I want to connect ChatGPT (Web) to DSI Studio.\n\n"
            "First read the public GitHub file:\n\n"
            "frankyeh/DSI-Studio-AI/DSI_STUDIO_AI_SKILL_GITHUB_ISSUE_SESSION.md\n\n"
            "Follow its instructions for starting a new ChatGPT (Web) GitHub issue session. "
            "Use the GitHub tools available in ChatGPT. If GitHub is unavailable, guide me through enabling it first. "
            "Create or select an appropriate private personal GitHub repository, preferably DSI-Studio-Connect, "
            "create the required session issue, and clearly give me the complete Issue URL to paste into DSI Studio.\n\n"
            "Do not send DSI Studio commands until I confirm that DSI Studio is connected to the issue.");
        QDesktopServices::openUrl(QUrl("https://chatgpt.com/"));
        set_helper("Setup instructions copied. Paste them into ChatGPT. When ChatGPT gives you an Issue URL, return here and click Paste.");
    });
    QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    auto* accept = buttons.addButton(accept_text,QDialogButtonBox::AcceptRole);
    accept->setObjectName("ai_primary_button");
    layout.addRow(&buttons);
    connect(accept,&QPushButton::clicked,&dialog,[&]
    {
        if(agent.currentIndex() == int(ai_provider::ChatGPT))
        {
            if(settings.value("ai/github_token").toString().trimmed().isEmpty())
                return setup_token.click();
            if(issue_url_edit.text().trimmed().isEmpty())
                return setup_issue.click();
        }
        dialog.accept();
    });
    connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);

    if(dialog.exec() != QDialog::Accepted)
        return false;

    agent_index = agent.currentIndex();
    value = (agent_index == int(ai_provider::ChatGPT)) ? issue_url_edit.text().trimmed() : model_combo_key(model);
    return true;
}

ai_info* AIAgent::create_new_chat(const QString& agent)
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
        QUuid::createUuid().toString(QUuid::WithoutBraces),agent,ai_provider::Infer); // status defaults to New; no "new:"/other marker on the id itself
    if(info->provider == ai_provider::ChatGPT)
        web_agent_session_id = info->sessions;
    else
        info->model_settings = QJsonObject{
            {"model",current_model_name},{"info",current_model_info}};
    set_ai_status(info->sessions,session_status::New,"Ready for a message.");
    show_ai_project(*info);
    ui->ai_project_list->setCurrentItem(info->project_items);
    return info;
}

void AIAgent::new_chat_dialog(bool resume)
{
    // resuming an already-known chat: reconnect with its saved issue link directly, no dialog
    if(resume)
        if(auto* info = ai_info::find(web_agent_session_id))
            if(auto path = info->model_settings["github_issue_url"].toString();!path.isEmpty())
            {
                try_connect_github_issue("https://github.com/"+path);
                return;
            }

    int agent_index = 0;
    QString value;
    if(!run_new_chat_dialog(resume,resume ? "Resume Chat" : "New Chat",resume ? "Resume" : "Start",
                             agent_index,value))
        return;
    bool web = agent_index == int(ai_provider::ChatGPT);
    // no early web_agent_session_id.clear() here: disconnect_github_issue() (below, and inside
    // start_new_local_chat()) needs it to still name the old chat so that chat gets marked Completed;
    // create_new_chat("ChatGPT(Web)") already reassigns it for a fresh (non-resume) web chat, and
    // start_new_local_chat() clears it itself once the old channel is actually disconnected

    if(web)
    {
        disconnect_github_issue(); // leave the old channel cleanly before attempting a different one
        if(!resume)
            create_new_chat("ChatGPT(Web)"); // exists immediately, even if the connection below fails -- a failed connection is then just this chat's own Error state, like a local chat's own Stop/error state
        try_connect_github_issue(value);
        return;
    }

    current_agent_index = agent_index;
    try_set_current_model(value);
    start_new_local_chat();
}

ai_info* AIAgent::start_new_local_chat() // shared by new_chat_dialog() and Send-with-nothing-selected: creates a fresh chat with the current default agent/model and prepares the compose box for it
{
    disconnect_github_issue(); // leaving web-agent mode for a local chat -- marks the old web chat Completed via web_agent_session_id, so clear that only after
    web_agent_session_id.clear();
    // update_send_button()/update_agent_status_label() are skipped here: create_new_chat() below selects the
    // new chat, and the sidebar's own currentItemChanged handler already refreshes both for any new selection
    auto* info = create_new_chat(current_agent_index == int(ai_provider::Codex) ? "Codex" : "Claude");
    ui->ai_chat_input->clear();
    ui->ai_chat_input->setFocus();
    return info;
}

void AIAgent::on_ai_new_chat_clicked()
{
    new_chat_dialog(false);
}

void AIAgent::on_ai_agent_status_clicked()
{
    if(auto* info = selected_info())
    {
        if(info->provider == ai_provider::ChatGPT) // change or reconnect using a possibly different issue link
        {
            web_agent_session_id = info->sessions; // resume must target the selected chat, not whatever session was last active
            int agent_index = 0;
            QString value;
            if(!run_new_chat_dialog(true,"Change Issue Link","Reconnect",agent_index,value))
                return;
            disconnect_github_issue(); // leave the old channel cleanly before attempting a different one
            try_connect_github_issue(value);
            return;
        }
        if(info->provider == ai_provider::AgentServer) // no local agent/model of its own to change
            return;
        QDialog dialog(this);
        dialog.setWindowTitle("Change Model");
        QFormLayout layout(&dialog);
        QLabel agent_label(info->provider == ai_provider::Codex ? "Codex" : "Claude");
        QComboBox model;
        set_model_selector(model,agent_entries[int(info->provider)].profiles,info->model_settings["model"].toString());
        layout.addRow("Agent:",&agent_label);
        layout.addRow("Model:",&model);
        QDialogButtonBox buttons(QDialogButtonBox::Cancel|QDialogButtonBox::Save);
        layout.addRow(&buttons);
        connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);
        connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
        if(dialog.exec() != QDialog::Accepted)
            return;

        set_chat_model(*info,model_combo_key(model)); // this chat's own model, not the app-wide default
        update_agent_status_label();
        return;
    }

    int agent_index = 0;
    QString value;
    if(!run_new_chat_dialog(false,"Change Agent/Model","Save",agent_index,value))
        return;

    if(agent_index == int(ai_provider::ChatGPT))
    {
        // same ownership setup new_chat_dialog() does for a fresh web chat -- try_connect_github_issue()
        // assumes web_agent_session_id already names a real chat, which nothing else here would have arranged
        disconnect_github_issue(); // leave any old channel cleanly before attempting a different one
        create_new_chat("ChatGPT(Web)");
        try_connect_github_issue(value);
        return;
    }

    current_agent_index = agent_index;
    try_set_current_model(value);
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

    auto* title = new QLabel("AI Settings");
    title->setObjectName("ai_dialog_title");
    root->addWidget(title);

    auto* ollama_card = new QFrame;
    ollama_card->setObjectName("ai_step_card");
    auto* ollama_layout = new QVBoxLayout(ollama_card);
    ollama_layout->setContentsMargins(14,12,14,12);
    ollama_layout->setSpacing(8);
    auto* ollama_heading = new QLabel("Ollama connection");
    ollama_heading->setObjectName("ai_step_heading");
    ollama_layout->addWidget(ollama_heading);
    auto* ollama_form = new QFormLayout;
    ollama_form->setContentsMargins(0,0,0,0);
    QLineEdit host(settings.value("ai/ollama_host","localhost").toString());
    QSpinBox port;
    port.setRange(1,65535);
    port.setValue(settings.value("ai/ollama_port",11434).toInt());
    ollama_form->addRow("Host/IP:",&host);
    ollama_form->addRow("Port:",&port);
    ollama_layout->addLayout(ollama_form);
    root->addWidget(ollama_card);

    auto* github_card = new QFrame;
    github_card->setObjectName("ai_step_card");
    auto* github_layout = new QVBoxLayout(github_card);
    github_layout->setContentsMargins(14,12,14,12);
    github_layout->setSpacing(8);
    auto* github_heading = new QLabel("GitHub access");
    github_heading->setObjectName("ai_step_heading");
    auto* github_body = new QLabel("Required to connect a ChatGPT (Web) session through a GitHub issue.");
    github_body->setObjectName("ai_step_body");
    github_body->setWordWrap(true);
    github_layout->addWidget(github_heading);
    github_layout->addWidget(github_body);
    QPushButton github_button("Set up GitHub token"); // stays enabled even once configured -- unlike Codex/Claude sign-in, a token can't be re-checked live, so re-opening this is the only way to replace/reset it
    auto* github_button_row = new QHBoxLayout;
    github_button_row->addWidget(&github_button);
    github_button_row->addStretch();
    github_layout->addLayout(github_button_row);
    root->addWidget(github_card);

    auto update_github_button = [&]
    {
        bool has_token = !settings.value("ai/github_token").toString().trimmed().isEmpty();
        github_button.setText(has_token ? "GitHub token ready ✓ · Configure..." : "Set up GitHub token");
    };
    update_github_button();
    connect(&github_button,&QPushButton::clicked,&dialog,[&]
    {
        setup_github_token();
        update_github_button();
    });

    auto* chat_card = new QFrame;
    chat_card->setObjectName("ai_step_card");
    auto* chat_layout = new QVBoxLayout(chat_card);
    chat_layout->setContentsMargins(14,12,14,12);
    chat_layout->setSpacing(8);
    auto* chat_heading = new QLabel("Chat behavior");
    chat_heading->setObjectName("ai_step_heading");
    chat_layout->addWidget(chat_heading);
    QCheckBox history("Keep AI chat history");
    history.setChecked(settings.value("ai/keep_history",true).toBool());
    QCheckBox show_reasoning("Show reasoning");
    show_reasoning.setToolTip("Show AI reasoning messages in chat history");
    show_reasoning.setChecked(settings.value("ai/show_reasoning",false).toBool());
    chat_layout->addWidget(&history);
    chat_layout->addWidget(&show_reasoning);
    auto* debug_row = new QHBoxLayout;
    auto* debug_label = new QLabel("Debug mode:");
    QComboBox debug;
    debug.addItem("Disabled");
    debug.addItem("Enabled (truncated)");
    debug.addItem("Enabled (complete)");
    debug.setCurrentIndex(settings.value("ai/debug",0).toInt());
    debug_row->addWidget(debug_label);
    debug_row->addWidget(&debug,1);
    chat_layout->addLayout(debug_row);
    root->addWidget(chat_card);

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

void AIAgent::prepare_ai(ai_info& info,const QString& text,ai_input input)
{
    // fresh state each attempt -- a field this attempt doesn't set (e.g. launch_model_url when not using Ollama) must not carry over a stale value from the last one
    info.launch_name.clear();
    info.launch_executable.clear();
    info.launch_model.clear();
    info.launch_model_url.clear();
    auto provider = info.provider;
    auto session = info.sessions; // captured by value below for every async handler -- info itself must never be captured across them (Codex can still rename/rekey the session)

    // a failed launch still owes the caller's message a home: queued as a pending prompt on the
    // session so the next successful run picks it up, rather than silently dropping it
    auto preserve_pending = [&]()
    {
        if(input == ai_input::Pending)
            info.prompts.append(text);
    };
    auto fail_launch = [&](const QString& message,bool warn = true)
    {
        preserve_pending();
        set_ai_status(session,info.status == session_status::New ?
                      session_status::New : session_status::Failed,message);
        show_ai_project(info);
        if(warn)
            QMessageBox::warning(this,"AI Agent",message);
    };

    // Resolve agent
    info.launch_name = provider == ai_provider::Codex ? "Codex" : "Claude";
    if(agent_entries[int(provider)].executable.isEmpty()) // stale showEvent() check -- the window may have stayed open since before an install finished, so retry once before assuming it's still missing
        refresh_agent_executables();
    info.launch_executable = agent_entries[int(provider)].executable;
    if(info.launch_executable.isEmpty())
    {
        QDesktopServices::openUrl(agent_install_url(provider)); // same as the sidebar's Install button
        return fail_launch(info.launch_name+" is not installed. Opening the install page...");
    }

    // Resolve work directory
    auto project_dir = ui->ai_work_dir->text().trimmed();
    ui->ai_work_dir->setText(
        project_dir.isEmpty() ? main_window.work_dir() : project_dir);

    info.launch_model = info.model_settings["model"].toString().trimmed();
    if(info.model_settings["info"].toObject().contains("provider"))
    {
        auto [url,configured] = ai_ollama_url(settings);
        info.launch_model_url = url;
        info.launch_name += "/Ollama("+info.launch_model_url.host()+")";
        if(!configured)
            return fail_launch("Set the Ollama host/IP in AI Settings first.");
    }
    else if(agent_login_info(provider).isEmpty())
    {
        if(!run_agent_login(provider))
            return fail_launch(info.launch_name+" sign-in was not completed.",false);
        refresh_login_buttons();
    }
    info.save_config(); // model_settings itself is untouched by this launch -- nothing new to persist here, just re-confirming it under the now-current status

    auto* process = new QProcess(this);
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

    info.processes = process;
    auto name = info.launch_name; // a plain value copy for the async handlers below -- never info itself

    if(input == ai_input::User)
    {
        // recorded here, once, unconditionally, the moment it's sent -- not deferred to whichever async
        // establishment event (Codex "thread.started", Claude stream-json "system"/"init") happens to
        // confirm the session later. That deferral was the actual bug: it required stashing this text in
        // the async handler's own closure to "replay" once establishment confirmed, and if the backend ever
        // sent that one-time event more than once (observed with an Ollama-routed session), the stale
        // stashed text got replayed again too, duplicating the opening message into the chat history
        add_ai_history(info,"user",text);
        ui->ai_chat_input->clear();
    }

    // this session was never established, so it has no real id worth preserving -- back to New entirely,
    // as if this attempt never happened, rather than left marked Failed. The message itself stays recorded
    // (it really was sent) -- this just explains what happened to it, the same as any other failed launch
    auto restore_new_chat = [=](const QString& message)
    {
        QMessageBox::warning(this,"AI Agent",message);
        if(auto* info = ai_info::find(process->objectName()))
        {
            info->processes = nullptr;
            set_ai_status(info->sessions,session_status::New,message);
            add_ai_history(*info,"activity",message);
            info->save_config(); // projects is non-empty now (the recorded messages), so this actually
                                  // writes -- without it, the .jsonl this just wrote would have no config.json
                                  // to explain its agent/provider on the next reload
        }
    };

    connect(process,&QProcess::readyReadStandardError,this,[=]
    {
        auto error = process->property("stderr").toByteArray()+
                     process->readAllStandardError();
        process->setProperty("stderr",error.right(8*1024));
    });

    connect(process,&QProcess::started,this,[=,status = info.status]
    {
        if(provider != ai_provider::Claude)
            process->closeWriteChannel();
        auto session = process->objectName();
        ai_log("connecting to "+ name + "@" + session+
            " pid:"+QString::number(process->processId()));
        // the OS process starting proves nothing about the backend conversation itself -- only this provider's
        // own established-session event (configure_codex's "thread.started", configure_claude's "system"/
        // "init") confirms the session, so a genuine first launch stays New until then. A reconnect of an
        // already-established session (pre-launch status captured above, same as errorOccurred/finished use to
        // tell the two apart) shows Thinking instead -- staying New here too would let a save mid-reconnect
        // wrongly persist established:false over it (see save_config())
        if(auto* info = ai_info::find(session))
        {
            set_ai_status(session,status == session_status::New ?
                          session_status::New : session_status::Thinking,
                          "Waiting for "+name+" connection");
            show_ai_project(*info);
        }
        update_send_button();
    });

    connect(process,&QProcess::errorOccurred,this,
            [=,status = info.status](QProcess::ProcessError error)
    {
        if(error != QProcess::FailedToStart)
            return;

        auto session = process->objectName();
        auto message = "Cannot start "+name+": "+process->errorString();
        ai_log(message);

        auto* found = ai_info::find(session);
        // A first launch that was never established has no real id to preserve; a reconnect keeps its id.
        if(!found || (status == session_status::New && found->status == session_status::New))
            restore_new_chat(message);
        else
        {
            auto& info = *found;
            info.processes = nullptr;
            set_ai_status(session,session_status::Failed,message);
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
            this,[=,status = info.status]
            (int exit_code,QProcess::ExitStatus exit_status)
    {
        bool user_stopped = process->property("user_stopped").toBool();
        auto session = process->objectName();
        ai_log(name + " finished session ");
        auto error = (process->property("stderr").toByteArray()+
                      process->readAllStandardError()).trimmed();
        bool failed = !user_stopped && (exit_code || exit_status == QProcess::CrashExit);
        auto error_message = user_stopped ? QString("Stopped by user.") :
                              ("error code:"+QString::number(exit_code)+" "+
                              QString::fromUtf8(error)).trimmed();
        if(failed)
            ai_log(error_message);

        auto* found = ai_info::find(session);
        // Same reasoning as errorOccurred: only an unestablished first launch returns to New.
        if(!found || (status == session_status::New && found->status == session_status::New))
        {
            auto message = found && found->status == session_status::New ?
                           found->status_message : failed ? error_message :
                           "AI agent ended before creating a new chat.";
            restore_new_chat(message);
        }
        else
        {
            auto& info = *found;
            info.processes = nullptr;

            if(user_stopped) // stop means stop: no auto-continue into a queued message -- caller only marks
                info.prompts.clear(); // the process user_stopped and kills it, this is the one place that decides what that implies for queued prompts
            auto pending = info.prompts.join("\n\n");
            info.prompts.clear();
            if(!pending.isEmpty())
                start_ai(info,pending,ai_input::Pending);
            else if(failed || user_stopped)
            {
                set_ai_status(session,failed ? session_status::Failed : session_status::Completed,
                              error_message);
                add_ai_history(info,"activity",error_message);
            }
            else if(!process->property("had_reply").toBool())
            {
                set_ai_status(session,session_status::Completed,"No reply from AI agent.");
                add_ai_history(info,"activity","No reply from AI agent.");
            }
            else
            {
                set_ai_status(session,session_status::Completed,"Agent process finished.");
                show_ai_project(info);
            }
        }
        update_send_button();
        process->deleteLater();
    });
}

QStringList AIAgent::configure_claude(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    auto session = info.sessions; // captured by value into the async handlers below -- never info itself
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
                    auto line = process->readLine();
                    ai_log("stdout:"+QString::fromUtf8(line).trimmed());
                    auto event = QJsonDocument::fromJson(line).object();
                    auto event_type = event["type"].toString();
                    if(event_type == "system")
                    {
                        auto subtype = event["subtype"].toString();
                        if(subtype == "init")
                        {
                            // the session-established event: Claude's own stream-json protocol confirms the
                            // conversation actually initialized, not just that the OS process started -- the
                            // Codex equivalent is "thread.started" in configure_codex(). Status/config only --
                            // the opening message was already recorded, once, synchronously, when it was sent
                            // (see prepare_ai()); this event has no content-recording role at all anymore
                            if(auto* info = ai_info::find(process->objectName()))
                            {
                                set_ai_status(info->sessions,session_status::Thinking,
                                              "Session started; waiting for agent input");
                                info->save_config();
                            }
                        }
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
        info.status == session_status::New ? "--session-id" : "--resume",session};
    // an absent --model falls back to whatever the Claude CLI last remembered from an unrelated session, not a real default
    args << "--model" << (info.launch_model.isEmpty() ? "sonnet" : info.launch_model);
    return args;
}
QStringList AIAgent::configure_codex(const ai_info& info,const QString& text)
{
    auto* process = info.processes;
    auto session = info.sessions; // captured by value into the async handler below -- never info itself (Codex renames/rekeys the session there)
    auto name = info.launch_name;
    connect(process,&QProcess::readyReadStandardOutput,this,
            [=,status = info.status]
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
                if(!is_valid_session_id(session))
                {
                    // Codex's own protocol contract broke -- do not let a malformed id corrupt ai_infos'
                    // keying; surface it immediately instead of silently accepting it
                    ai_log("invalid thread_id from Codex (not a UUID): "+session);
                    if(auto* info = ai_info::find(old_session))
                    {
                        set_ai_status(info->sessions,status == session_status::New ?
                                      session_status::New : session_status::Failed,
                                      "Codex returned an invalid thread ID.");
                        show_ai_project(*info);
                    }
                    return process->kill();
                }
                auto* old_info = ai_info::find(old_session);
                // never established before -- still just DSI Studio's own placeholder, safe to rename in place
                bool still_placeholder = status == session_status::New && old_info &&
                                         old_info->status == session_status::New;
                auto* info = still_placeholder ?
                    assign_ai_session(old_session,session) :
                    ai_info::create(session,name,ai_provider::Codex); // already known, not inferred: this whole handler is Codex-specific
                if(info)
                {
                    set_ai_status(info->sessions,session_status::Thinking,
                                  "Session started; waiting for agent input");
                    if(!still_placeholder) // a genuinely different/new entry -- old_info (still alive, just not renamed) is the only place its settings still exist
                        info->model_settings = old_info ? old_info->model_settings : QJsonObject();
                    info->save_config();
                }
                // status/rename only -- the opening message was already recorded, once, synchronously, when
                // it was sent (see prepare_ai()); this event has no content-recording role at all anymore
                if(info && old_session != session)
                {
                    process->setObjectName(session);
                    info->processes = process;
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

    QStringList args{"exec","--add-dir",ui->ai_work_dir->text()};
    if(!info.launch_model_url.isEmpty())
    {
        auto url = info.launch_model_url;
        url.setPath("/v1");

        auto env = process->processEnvironment();
        env.insert("CODEX_OSS_BASE_URL",url.toString());
        process->setProcessEnvironment(env);

        args << "--oss" << "--local-provider=ollama";
    }
    if(!info.launch_model.isEmpty() && info.launch_model != "default") // Codex's own code-assigned alias for "no explicit choice" -- its CLI has no --model value named this, omit the flag instead
    {
        args << "--model" << info.launch_model;
        if(auto profile = info.model_settings["info"].toObject()["profile"].toString();
           !profile.isEmpty())
            args << "--profile" << profile;
    }
    if(info.status != session_status::New) // Completed or Failed: already established, resumable regardless of how the last run ended
        args << "resume" << session;
    args << "--json" << "--skip-git-repo-check";
    args << text;
    return args;
}
void AIAgent::start_ai(ai_info& info,const QString& text,ai_input input)
{
    if(info.processes)
    {
        add_ai_history(info,"user",text);
        ui->ai_chat_input->clear();

        bool send = info.provider == ai_provider::Claude &&
                    info.processes->state() == QProcess::Running;
        if(send)
            info.processes->write(claude_input(text));
        else
            info.prompts.append(text);

        set_ai_status(info.sessions,send ? session_status::Thinking : info.status,send ?
                      "Message sent; waiting for agent" : "Message queued for the AI agent.");
        return;
    }

    Q_ASSERT(info.provider == ai_provider::Codex || info.provider == ai_provider::Claude); // never ChatGPT: callers must intercept a web chat before reaching here

    prepare_ai(info,text,input);
    if(!info.processes) // prepare_ai() failed before ever creating a process
        return;
    auto args = info.provider == ai_provider::Codex ?
        configure_codex(info,text) :
        configure_claude(info,text);
    ai_log("start " + info.launch_executable +
           " args: " + args.join(" ").remove("\n"));
    // New only for a genuinely never-established launch; an already-established session being resumed (info.status
    // here is still the pre-launch value -- configure_codex()/configure_claude() above only read it) shows
    // Thinking instead, so a save mid-reconnect can't wrongly persist established:false over it (see save_config())
    set_ai_status(info.sessions,info.status == session_status::New ?
                  session_status::New : session_status::Thinking,
                  "Starting "+info.launch_name);
    info.processes->start(info.launch_executable,args);
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
    case send_action::Resume: // only reachable when info exists, see current_send_action()
        web_agent_session_id = info->sessions; // resume must target the selected chat, not whatever session was last active
        new_chat_dialog(true);
        return;
    case send_action::Stop: // only reachable when info exists, see current_send_action()
        if(info->provider == ai_provider::ChatGPT)
            disconnect_github_issue();
        else
        {
            info->processes->setProperty("user_stopped",true); // finished()'s own handler clears queued prompts for a user_stopped session -- no auto-continue into a queued message
            info->processes->kill(); // kill(): a windowless console child never sees terminate()'s WM_CLOSE
        }
        return;
    case send_action::Send: // reachable when info exists and isn't AgentServer, or when nothing is selected but there's text to send, see current_send_action()
    case send_action::Queue: // start_ai() itself decides send-vs-queue (only a live Claude process gets a real write); this case just gets it there
        start_ai(*(info ? info : start_new_local_chat()),text,ai_input::User);
        update_send_button();
        return;
    }
}
