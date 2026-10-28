// ai_info's data layer (session registry, history/config persistence), free helpers, the Web channel, and the headless --action=ai entry point
#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSettings>
#include <QTimer>
#include <QUrl>
#include <QUuid>

#include <algorithm>
#include <climits>
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
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSaveFile>
#include <QStandardPaths>

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

// local-agent discovery and noninteractive readiness probes, shared by the GUI Settings and headless --action=ai
const QStringList local_agents{"Codex","Claude","Muse","Antigravity","Grok"}; // every local CLI agent, in settings order
QProcessEnvironment agent_environment(const QString& provider)
{
    auto env = QProcessEnvironment::systemEnvironment();
#if defined(Q_OS_WIN) || defined(Q_OS_LINUX)
    if(provider == "Muse" && !env.contains("TBH_CREDENTIAL_BACKEND"))
        env.insert("TBH_CREDENTIAL_BACKEND","file");
#endif
    if(provider == "Claude") // real Anthropic baseline for status, sign-in and chat: drop any inherited Ollama redirect; an Ollama launch re-adds its own
        for(auto var : {"ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","CLAUDE_CODE_USE_POWERSHELL_TOOL",
                        "ANTHROPIC_DEFAULT_HAIKU_MODEL","ANTHROPIC_DEFAULT_SONNET_MODEL",
                        "ANTHROPIC_DEFAULT_OPUS_MODEL","CLAUDE_CODE_SUBAGENT_MODEL"})
            env.remove(var);
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
QByteArray jsonrpc(const QString& id,const QString& method,const QJsonObject& params)
{
    QJsonObject msg{{"jsonrpc","2.0"},{"method",method}};
    if(!id.isEmpty())
        msg["id"] = id;
    if(!params.isEmpty())
        msg["params"] = params;
    return json_line(msg);
}
QByteArray muse_initialize() // one handshake for the status probe, model list and chat; the probe checks experimentalApi in the reply
{
    return jsonrpc("initialize","initialize",
        {{"clientInfo",QJsonObject{{"name","dsi_studio"},{"title","DSI Studio"},{"version","1.0"}}},
         {"capabilities",QJsonObject{{"experimentalApi",true},{"userInputDialogs",false}}}});
}
QByteArray grok_initialize()
{
    return jsonrpc("initialize","initialize",{{"protocolVersion",1},{"clientCapabilities",QJsonObject()}});
}

QString find_agent_executable(const QString& provider)
{
    if(!local_agents.contains(provider))
        return {};
    if(auto path = QStandardPaths::findExecutable(provider == "Antigravity" ? "agy" : provider.toLower());!path.isEmpty())
        return path;
    QString fallback;
    if(provider == "Codex")
    {
        QDir dir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation)+"/OpenAI/Codex/bin");
        for(const auto& name : dir.entryList(QDir::Dirs|QDir::NoDotAndDotDot,QDir::Time))
            if(QFileInfo::exists(fallback = dir.filePath(name+"/codex.exe")))
                break;
    }
#ifdef Q_OS_WIN
    auto local_app_data = qEnvironmentVariable("LOCALAPPDATA");
    if(provider == "Claude")
        fallback = QDir::homePath()+"/.local/bin/claude.exe";
    if(provider == "Muse")
        fallback = local_app_data.isEmpty() ? QString() : local_app_data+"/Programs/muse/muse.cmd";
    if(provider == "Antigravity")
        fallback = local_app_data+"/agy/bin/agy.exe";
    if(provider == "Grok")
        fallback = QDir::homePath()+"/.grok/bin/grok.exe";
#else
    if(provider == "Muse")
        fallback = QDir::homePath()+"/.local/bin/muse";
    if(provider == "Antigravity")
        fallback = QDir::homePath()+"/.local/bin/agy";
    if(provider == "Grok")
        fallback = QDir::homePath()+"/.grok/bin/grok";
#endif
    return QFileInfo::exists(fallback) ? fallback : QString();
}
ai_agent_status check_agent_status(const QString& provider,const QString& executable,QString& info)
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

    if(provider == "Grok" || provider == "Muse")
    {
        auto deadline = QDateTime::currentMSecsSinceEpoch()+(provider == "Grok" ? 10000 : 5000);
        auto read_response = [&](const QString& id)
        {
            while((process.state() != QProcess::NotRunning || process.canReadLine()) &&
                  QDateTime::currentMSecsSinceEpoch() < deadline)
            {
                if(!process.canReadLine())
                    process.waitForReadyRead(int(deadline-QDateTime::currentMSecsSinceEpoch()));
                while(process.canReadLine())
                    if(auto msg = QJsonDocument::fromJson(process.readLine()).object();msg["id"].toString() == id)
                        return msg;
            }
            return QJsonObject();
        };
        auto status = ai_agent_status::Error;
        if(provider == "Grok") // ACP initialize reports the agent's own credential choice: _meta.defaultAuthMethodId, null when none is usable
        {
            process.write(grok_initialize());
            if(auto reply = read_response("initialize");reply.contains("result"))
            {
                auto method = reply["result"].toObject()["_meta"].toObject()["defaultAuthMethodId"].toString();
                if(method.isEmpty())
                    status = ai_agent_status::SignInRequired;
                else
                {
                    info = method == "xai.api_key" ? "API key" : "Signed in";
                    status = ai_agent_status::Ready;
                }
            }
        }
        else
        {
            process.write(muse_initialize());
            if(read_response("initialize")["result"].toObject()["experimentalApi"].toBool())
            {
                process.write(jsonrpc({},"initialized"));
                process.write(jsonrpc("account_read","account/read"));
                if(auto account = read_response("account_read")["result"].toObject();!account.isEmpty())
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
        // codex login status has no --json/structured output (email/plan aren't exposed), only this
        // free-text auth-method line -- see https://github.com/openai/codex/issues/19866 -- on stdout or stderr depending on version
        auto output = QString::fromUtf8(process.readAllStandardOutput()+process.readAllStandardError());
        if(process.exitCode() != 0)
            return output.contains("Not logged in",Qt::CaseInsensitive) ?
                       ai_agent_status::SignInRequired : ai_agent_status::Error;
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

QList<ai_info*> load_ai_projects()
{
    QList<ai_info*> loaded;
    QSettings settings;
    for(const auto& info : QDir(ai_project_dir).entryInfoList(
            {"*.jsonl"},QDir::Files,QDir::Time|QDir::Reversed))
    {
        auto session = QUrl::fromPercentEncoding(
                           info.completeBaseName().toLatin1());
        if(session.isEmpty() || ai_info::find(session)) // a live chat is never reloaded over
            continue;
        QList<QJsonObject> history;
        QFile file(info.filePath());
        if(!file.open(QIODevice::ReadOnly))
            continue;
        while(!file.atEnd())
            if(auto doc = QJsonDocument::fromJson(file.readLine());doc.isObject())
                history.append(doc.object());
        if(history.isEmpty())
            continue;
        QJsonObject config;
        if(QFile config_file(ai_info::config_file(session));config_file.open(QIODevice::ReadOnly))
            config = QJsonDocument::fromJson(config_file.readAll()).object();
        auto* ai = ai_info::create(
            session,config["provider"].toString(),config["agent"].toString());
        if(!ai)
            continue;
        bool established = config["established"].toBool();
        ai->status = established ? session_status::Completed : session_status::New;
        ai->status_message = established ? "Previous chat loaded." : "Previous attempt never connected.";
        ai->model_settings = config["model_settings"].toObject();
        ai->project_titles = settings.value("ai/title/"+session).toString();
        ai->projects = std::move(history);
        loaded << ai;
    }
    return loaded;
}

QByteArray json_line(const QJsonObject& message)
{
    return QJsonDocument(message).toJson(QJsonDocument::Compact)+'\n';
}

QPair<QUrl,bool> ai_ollama_url(QString host,int port)
{
    host = host.trimmed();
    bool configured = !host.isEmpty();
    if(!host.contains("://"))
        host.prepend("http://");
    QUrl url(host);
    url.setPort(port);
    return {url,configured};
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
bool ai_run_cli(const std::vector<std::string>& cmd,std::string& error)
{
    if(cmd.size() != 2 || cmd[1].empty())
        return error = "usage: run_cli <command line>",false;
    auto command_line = cmd[1];
    if(tipl::begins_with(command_line,"dsi_studio "))
        command_line.erase(0,11);
    tipl::program_option<tipl::out> po;
    if(!po.parse(command_line))
        return error = po.error_msg,false;
    if(!po.has("action"))
        po.set("action","vis");
    if(run_action_with_wildcard(po))
        return error = "command line failed",false;
    po.check_end_param<tipl::warning>();
    return true;
}
static bool read_only_shell(const QString& text) // conservative: one simple command from a short list, without shell syntax or writing options
{
    static const QString shell_syntax(";&|<>`$(){}\\\n\r");
    static const QStringList commands{"pwd","ls","cat","head","tail","grep","wc","stat","du","df","basename","dirname","readlink","realpath","find","rg","file","dir","type"},
                             writing{"-exec","-execdir","-ok","-okdir","-delete","-fprint","-fprint0","-fprintf","-fls"};
    auto args = text.split(' ',Qt::SkipEmptyParts);
    if(args.isEmpty() || !commands.contains(args[0]) || std::any_of(text.begin(),text.end(),[](QChar c){return shell_syntax.contains(c);}))
        return false;
    bool options_matter = args[0] == "find" || args[0] == "rg" || args[0] == "file"; // the commands with writing or command-running options
    return std::none_of(args.begin()+1,args.end(),[&](QString arg)
    {
        // an unquoted wildcard could expand to a file named like an option
        if(options_matter && arg.contains(QRegularExpression("[*?\\[]")) && !arg.startsWith('\'') && !arg.startsWith('"'))
            return true;
        arg.remove('\'').remove('"');
        return writing.contains(arg) || arg.startsWith("--pre") || // file -C/--compile writes, also bundled (-bC) or abbreviated (--comp)
               (args[0] == "file" && (arg.startsWith("--co") || QRegularExpression("^-[^-]*C").match(arg).hasMatch()));
    });
}
bool ai_write_text_file(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error)
{
    if(cmd.size() != 3 || cmd[1].empty())
        return error = "usage: write_text_file <filename> <content>",false;
    auto file_name = QFileInfo(QString::fromUtf8(cmd[1].c_str())).absoluteFilePath(); // relative to the working directory (set_work_dir)
    if(!approve("AI File Write Request","The AI agent wants to write "+QString::number(cmd[2].size())+
                   " bytes to this file, replacing it if it exists:\n\n"+file_name,false))
        return error = "user declined to write this file",false;
    QFile file(file_name); // binary: the UTF-8 content and its newlines are written exactly, no BOM
    if(!file.open(QIODevice::WriteOnly) || file.write(cmd[2].data(),qint64(cmd[2].size())) != qint64(cmd[2].size()))
        return error = "cannot write to "+file_name.toStdString(),false;
    tipl::out() << "wrote " << cmd[2].size() << " bytes to " << file_name.toStdString();
    return true;
}
bool ai_run_shell(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error,
                  const std::function<void(qint64)>& started)
{
    if(cmd.size() != 2 || cmd[1].empty())
        return error = "usage: run_shell <command>",false;
    QString text = QString::fromUtf8(cmd[1].c_str());
    QString program = text.section(' ',0,0);
    bool is_curl = !program.compare("curl",Qt::CaseInsensitive);
    if(is_curl)
        text.insert(program.length()," -s -S");
    if(!approve("AI Shell Command Request","The AI agent wants to run this shell command:\n\n"+text,read_only_shell(text)))
        return error = "user declined to run this shell command",false;
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
    auto log_output = [](QProcess& process)
    {
        tipl::out() << process.readAllStandardOutput().toStdString();
        if(auto err = process.readAllStandardError().toStdString();!err.empty())
            tipl::error() << err;
    };
    if(!is_curl)
    {
        QProcess process;
        configure_process(process,text);
        process.start();
        if(!process.waitForStarted(3000))
            return error = "cannot start command",false;
        if(started)
            started(process.processId());
        constexpr int shell_timeout_ms = 600000;
        if(!process.waitForFinished(shell_timeout_ms) && process.state() != QProcess::NotRunning) // false also when it already finished, e.g. during report_step
        {
            process.kill();
            process.waitForFinished(3000);
            return error = "command timed out after 10 minutes",false;
        }
        log_output(process);
        if(process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
            return error = "command exited with code "+std::to_string(process.exitCode()),false;
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
        [process,id,text,cleanup,log_output](int exit_code,QProcess::ExitStatus exit_status)
    {
        log_output(*process);
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
            status(session_status::Failed,"Web: "+google_error+"; retrying.");
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
void web_bridge::write_google_doc(const QJsonObject& doc,QJsonObject message,std::function<void(bool)> done)
{
    message["dsi_bridge"] = true;
    message["session"] = web_session_id;
    message["from"] = "dsi";
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
    auto url = "https://docs.googleapis.com/v1/documents/"+google_file_id+":batchUpdate";
    if(!done)
        google_api_wait("POST",url,body);
    else
        google_api("POST",url,body,[done](QJsonObject reply){done(!reply.isEmpty());});
}
static const QString web_title_separator = QString(" ")+QChar(0x00B7)+" DSI Studio "; // a Doc named "<title> <middle dot> DSI Studio <codename>"
static const QString web_folder_search = "https://www.googleapis.com/drive/v3/files?fields=files(id)&q="+QString::fromLatin1(QUrl::toPercentEncoding(
                                         "name='DSI Studio AI' and mimeType='application/vnd.google-apps.folder' and trashed=false"));
QJsonArray web_bridge::list_web_sessions(void)
{
    if(google_folder_id.isEmpty()) // found, never created: no folder means no chats
    {
        auto folders = google_api_wait("GET",web_folder_search)["files"].toArray();
        if(folders.isEmpty())
            return {};
        settings.setValue("ai/google_folder_id",google_folder_id = folders[0].toObject()["id"].toString());
    }
    // drive.file lists only files DSI Studio created, so other Docs never appear; web_download slots are not Docs
    return google_api_wait("GET","https://www.googleapis.com/drive/v3/files?pageSize=15&orderBy=modifiedTime%20desc&fields=files(id,name,modifiedTime,appProperties)&q="+
                           QString::fromLatin1(QUrl::toPercentEncoding("'"+google_folder_id+"' in parents and mimeType='application/vnd.google-apps.document' and trashed=false")))["files"].toArray();
}
bool web_bridge::remove_web_session(const QString& file_id,const QString& session)
{
    QFile::remove(ai_info::history_file(session));
    QFile::remove(ai_info::config_file(session));
    settings.remove("ai/title/"+session);
    return file_id.isEmpty() || !google_api_wait("PATCH","https://www.googleapis.com/drive/v3/files/"+file_id,{{"trashed",true}}).isEmpty();
}
void web_bridge::set_web_title(const QString& title)
{
    if(!google_file_id.isEmpty()) // the codename stays at the end of the name, so an agent can still find the Doc by it
        google_api("PATCH","https://www.googleapis.com/drive/v3/files/"+google_file_id,{{"name",title+web_title_separator+web_code}},[](QJsonObject){});
}
void web_bridge::create_web_session(std::function<void(QString file,QString session)> done)
{
    if(google_folder_id.isEmpty()) // resolved once (found or created), then reused by its ID
        return google_api("GET",web_folder_search,{},[this,done](QJsonObject found)
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
    // the codename is how a user or agent names the session; the UUID in appProperties stays the protocol session ID
    static const QStringList word1{"amber","arctic","azure","blaze","cedar","coral","crystal","delta","ember","frost","gold","green","jade","lunar","maple","pine",
                                   "neon","nova","ocean","onyx","polar","river","sage","silver","solar","swift","storm","sunset","terra","violet","wild","zen"},
                             word2{"bison","comet","eagle","falcon","finch","fox","hawk","heron","koala","lion","moose","meteor","otter","owl","panda","raven",
                                   "rocket","seal","shark","star","swan","tiger","orca","wolf","zebra","cloud","flame","leaf","moon","stone","pike","spark"};
    auto random_code = []{return word1[QRandomGenerator::global()->bounded(int(word1.size()))]+"-"+word2[QRandomGenerator::global()->bounded(int(word2.size()))];};
    auto code = random_code();
    for(int tries = 1;;++tries) // a fresh pair is easier to remember than a number, until pairs keep colliding
    {
        auto found = google_api_wait("GET","https://www.googleapis.com/drive/v3/files?fields=files(id)&q="+QString::fromLatin1(QUrl::toPercentEncoding(
                "'"+google_folder_id+"' in parents and appProperties has {key='dsi_code' and value='"+code+"'} and trashed=false")));
        if(!found.contains("files")) // a failed search is not "free": never publish an unverified codename
            return warn("Cannot reach Google Drive.");
        if(found["files"].toArray().isEmpty())
            break;
        code = random_code()+(tries < 5 ? QString() : "-"+QString::number(tries));
    }
    auto session = QUuid::createUuid().toString(QUuid::WithoutBraces);
    google_api("POST","https://www.googleapis.com/drive/v3/files?fields=id",
               {{"name","DSI Studio "+code},{"mimeType","application/vnd.google-apps.document"},{"parents",QJsonArray{google_folder_id}},
                {"appProperties",QJsonObject{{"dsi_session",session},{"dsi_code",code}}}},
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
static QString web_text(const QJsonObject& doc) // the mailbox message: the Doc body as one trimmed string
{
    QString text;
    for(const auto& block : doc["body"].toObject()["content"].toArray())
        for(const auto& element : block.toObject()["paragraph"].toObject()["elements"].toArray())
            text += element.toObject()["textRun"].toObject()["content"].toString();
    return text.trimmed();
}
static QJsonObject web_json(const QJsonObject& doc,QJsonParseError* error = nullptr)
{
    return QJsonDocument::fromJson(web_text(doc).toUtf8(),error).object();
}
void web_bridge::start_web(const QString& file,const QString& session)
{
    stop_web();
    if(session != web_session_id) // a pending result belongs to its own session only
    {
        web_session_id = session;
        web_last_id = 0;
        web_pending_result = QJsonObject();
    }
    google_file_id = file;
    web_code = google_api_wait("GET","https://www.googleapis.com/drive/v3/files/"+file+"?fields=appProperties")["appProperties"].toObject()["dsi_code"].toString();
    if(web_code.isEmpty()) // a Doc from before codenames keeps its UUID in the name
        web_code = session;
    // continue after the Doc's last message: an unclaimed old request is answered "id must be greater" instead of run,
    // and a "processing" left by a stopped DSI Studio still gets "outcome unknown"
    if(auto doc = google_api_wait("GET",google_doc_url());!doc.isEmpty())
    {
        auto current = web_json(doc);
        web_last_id = current["id"].toInteger()-(current["from"].toString() == "dsi" && current["state"].toString() == "processing" ? 1 : 0);
    }
    google_error = "the session Doc was never read"; // the first good read clears it and shows the chat connected
    status(session_status::WaitingUser,"Web: reading the session Doc.");
    web_idle.start();
    web_timer.start(0);
}
QString web_bridge::prompt() const
{
    return "Read the public GitHub file frankyeh/DSI-Studio-AI/DSI_STUDIO_AI_SKILL_WEB.md and follow it "
           "to connect to DSI Studio "+web_code+"."; // the agent finds the Doc by its codename in the DSI Studio AI folder
}
void web_bridge::stop_web(const QString& message)
{
    if(google_file_id.isEmpty())
        return;
    google_file_id.clear(); // callbacks still in flight see this and stop
    web_timer.stop();
    status(session_status::Completed,message);
}
QString web_bridge::google_doc_url() const // only what write_google_doc() and the mailbox need
{
    return "https://docs.googleapis.com/v1/documents/"+google_file_id+
           "?fields=revisionId,body(content(endIndex,paragraph(elements(textRun(content)))))";
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
            status(session_status::WaitingUser,"Web connected; waiting for a request.");
        }
        if(web_text(doc).isEmpty()) // a new Doc, or a "ready" write that failed: retried until it lands
            return write_google_doc(doc,{{"code",web_code},{"state","ready"}},
                                    [this,file](bool written){if(file == google_file_id) web_timer.start(written ? 500 : 5000);});
        QJsonParseError parse_error;
        auto request = web_json(doc,&parse_error);
        auto id = request["id"].toInteger();
        if(request["session"].toString() == web_session_id && request["from"].toString() == "dsi" &&
           request["state"].toString() == "processing" && id > web_last_id)
        {
            // claimed but never finished (DSI Studio stopped mid-request): report it instead of re-running it
            web_last_id = id;
            QString error("outcome unknown: DSI Studio stopped while running this request; check with list_window");
            if(auto pid = request["pid"].toInteger()) // DSI Studio cannot control it, but the agent can inspect it before repeating
                error += "; its shell command may still be running as pid "+QString::number(pid);
            web_pending_result = QJsonObject{{"id",id},{"state","error"},
                {"response",QJsonObject{{"status","error"},{"error",error}}}};
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
            return write_google_doc(doc,{{"id",id > web_last_id ? id : web_last_id+1},
                {"state","error"},{"response",QJsonObject{{"status","error"},
                {"error",error + "; nothing was run. Resend the whole request as one line of compact JSON"}}}},
                [this,file](bool written){if(file == google_file_id) web_timer.start(written ? 500 : 5000);});
        // claim with the revision we read: a crash after it never re-runs the command
        write_google_doc(doc,{{"id",id},{"state","processing"}},
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
            web_dispatching = true;
            auto response = dispatch(forwarded);
            web_dispatching = false;
            web_pending_result = QJsonObject{{"id",id},
                {"state",response["status"].toString() == "error" ? "error" : "done"},{"response",response}};
            publish_web_result();
        });
    });
}
void web_bridge::report_step(const QString& step,const QString& approval,qint64 pid)
{
    if(!web_dispatching) // e.g. a local agent's request in the GUI
        return;
    auto doc = google_api_wait("GET",google_doc_url());
    auto current = web_json(doc);
    if(current["from"].toString() != "dsi" || current["state"].toString() != "processing" || current["id"].toInteger() != web_last_id)
        return; // only this request's own "processing" message is rewritten
    QJsonObject message{{"id",web_last_id},{"state","processing"},{"step",step}};
    if(!approval.isEmpty())
        message["approval"] = approval;
    if(pid)
        message["pid"] = pid;
    write_google_doc(doc,message);
}
void web_bridge::publish_web_result()
{
    google_api("GET",google_doc_url(),{},[this,file = google_file_id](QJsonObject doc)
    {
        if(file != google_file_id)
            return;
        if(doc.isEmpty()) // without the current body the write would append instead of replace
            return web_timer.start(5000);
        auto current = web_json(doc);
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
QJsonObject web_bridge::google_api_wait(const QByteArray& verb,const QString& url,const QJsonObject& body)
{
    QJsonObject result;
    bool done = false;
    QEventLoop loop;
    google_api(verb,url,body,[&](QJsonObject reply){result = reply;done = true;loop.quit();});
    if(!done) // google_api() may already have finished
        loop.exec();
    return result;
}
QJsonObject web_bridge::web_file(const std::vector<std::string>& cmd,const ai_approval& approve,QString& error)
{
    auto name = QString::fromStdString(cmd[0]);
    if(cmd.size() != 2 || cmd[1].empty())
        return error = "usage: "+name+(name == "web_upload" ? " <local file>" : name == "web_download" ? " <drive_id>" : " <file name>"),QJsonObject();
    if(google_file_id.isEmpty() || google_folder_id.isEmpty())
        return error = name+" is available only in the active Web session",QJsonObject();
    auto param = QString::fromStdString(cmd[1]);
    if(name == "web_download_prepare")
    {
        // an empty Drive file created by DSI Studio, so drive.file can read it after the agent replaces its bytes
        auto file_name = QFileInfo(param).fileName(); // no local or Drive path
        if(file_name.isEmpty())
            return error = "invalid file name: "+param,QJsonObject();
        auto id = google_api_wait("POST","https://www.googleapis.com/drive/v3/files?fields=id",
                                  {{"name",web_session_id+"_in_"+file_name},{"mimeType","application/octet-stream"},{"parents",QJsonArray{google_folder_id}},
                                   {"appProperties",QJsonObject{{"dsi_session",web_session_id},{"dsi_transfer","inbound"},{"dsi_name",file_name}}}})["id"].toString();
        if(id.isEmpty())
            return error = "cannot create the Drive upload slot: "+google_error,QJsonObject();
        return {{"name",file_name},{"drive_id",id},{"url","https://drive.google.com/file/d/"+id+"/view"}};
    }
    bool done = false;
    QEventLoop loop;
    if(name == "web_download")
    {
        auto file_url = "https://www.googleapis.com/drive/v3/files/"+QString::fromLatin1(QUrl::toPercentEncoding(param));
        auto meta = google_api_wait("GET",file_url+"?fields=mimeType,parents,appProperties,size");
        if(meta.isEmpty())
            return error = "cannot read Drive file "+param+": "+google_error,QJsonObject();
        auto properties = meta["appProperties"].toObject();
        auto file_name = QFileInfo(properties["dsi_name"].toString()).fileName();
        // only a slot this Web session prepared, never an arbitrary file of the user
        if(!meta["parents"].toArray().contains(google_folder_id) || properties["dsi_session"].toString() != web_session_id ||
           properties["dsi_transfer"].toString() != "inbound" || file_name.isEmpty())
            return error = "not an upload slot of this Web session; use web_download_prepare and replace the content of its drive_id",QJsonObject();
        if(meta["mimeType"].toString().startsWith("application/vnd.google-apps."))
            return error = "Google Docs, Sheets, and Slides files are not supported; upload raw file content",QJsonObject();
        if(meta["size"].toString() == "0") // the untouched slot from web_download_prepare
            return error = "the upload slot is still empty; replace the content of drive_id "+param+" first",QJsonObject();
        auto dir = ai_project_dir+"/web_files/"+web_session_id;
        auto path = dir+"/"+file_name;
        if(QFileInfo::exists(path))
            return error = "local file already exists: "+path,QJsonObject();
        QSaveFile file(path); // committed only after a complete download
        if(!QDir().mkpath(dir) || !file.open(QIODevice::WriteOnly))
            return error = "cannot write "+path,QJsonObject();
        auto finish = [&](QString failure)
        {
            error = failure;
            done = true;
            loop.quit();
        };
        with_google_token([&](QString token)
        {
            if(token.isEmpty())
                return finish("Google Drive download failed: "+google_error);
            QNetworkRequest request{QUrl(file_url+"?alt=media")};
            request.setRawHeader("Authorization","Bearer "+token.toUtf8());
            auto* reply = web_manager.get(request); // streamed to disk, never held in memory; the 30 s timeout counts idle time
            connect(reply,&QNetworkReply::readyRead,this,[&,reply]{file.write(reply->readAll());});
            connect(reply,&QNetworkReply::finished,this,[&,reply]
            {
                reply->deleteLater();
                file.write(reply->readAll());
                finish(reply->error() == QNetworkReply::NoError ? QString() : "Google Drive download failed: "+reply->errorString());
            });
        });
        if(!done) // with_google_token() may already have finished
            loop.exec();
        if(error.isEmpty() && !file.commit())
            error = "cannot write "+path;
        if(!error.isEmpty())
            return {};
        google_api("PATCH",file_url,{{"trashed",true}},[](QJsonObject){}); // best effort: the local file is already complete
        return {{"name",file_name},{"path",QFileInfo(path).absoluteFilePath()},{"mime_type",meta["mimeType"]},
                {"size",QFileInfo(path).size()},{"drive_id",param}};
    }
    QFileInfo file_info(param); // relative to the working directory (set_work_dir)
    QFile file(file_info.absoluteFilePath());
    if(!file_info.exists())
        error = "file does not exist: "+param;
    else if(!file_info.isFile())
        error = "not a regular file: "+param;
    else if(!file.open(QIODevice::ReadOnly))
        error = "cannot open file: "+param;
    // reading a local file, like a read-only shell command, so --approval=read_only allows it
    else if(!approve("AI File Upload Request","The AI agent wants to upload this file ("+QString::number(file.size())+
                     " bytes) to your Google Drive for the Web agent to read:\n\n"+file_info.absoluteFilePath(),true))
        error = "user declined to upload this file";
    if(!error.isEmpty())
        return {};
    auto mime = QMimeDatabase().mimeTypeForFile(file_info).name(); // application/octet-stream when unknown
    // one resumable upload for every size: the file is streamed from disk, never held in memory
    QJsonObject uploaded;
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
    auto id = uploaded["id"].toString();
    if(error.isEmpty() && id.isEmpty()) // an HTTP success without a Location header or without the file's metadata
        error = "Google Drive upload failed: no file ID returned";
    if(!error.isEmpty())
        return {};
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
int ai(tipl::program_option<tipl::out>& po) // --action=ai: a headless Web session served until Ctrl+C; no window is created
{
    // the terminal shows the chat history (agent messages, commands, errors, approvals); other output only with --debug
    std::ostream chat(std::cout.rdbuf());
    auto* cerr_buf = std::cerr.rdbuf();
    struct : std::streambuf {int_type overflow(int_type c) override {return traits_type::not_eof(c);}} quiet;
    web_bridge web;
    web.stop_when_idle = false;
    web.status = [](session_status,const QString& message){tipl::out() << message.toStdString();};
    web.warn = [&](const QString& text){chat << "\nerror: " << text.toStdString() << std::endl;QCoreApplication::exit(1);}; // std::cerr is silenced after start
    std::string approval = po.has("approval") ? po.get("approval") : std::string("manual");
    if(approval != "manual" && approval != "read_only" && approval != "auto")
        return tipl::error() << "--approval must be manual, read_only, or auto",1;
    web.dispatch = [&](const QJsonObject& request)
    {
        if(auto text = request["chat"].toString().trimmed();!text.isEmpty())
            chat << "\nAgent: " << text.toStdString() << std::endl;
        auto approve = [&](const QString& title,const QString& text,bool read_only) // --approval decides, otherwise the terminal user; no terminal declines
        {
            if(approval == "auto" || (approval == "read_only" && read_only))
                return true;
            web.report_step("waiting_approval",text);
            chat << "\n" << title.toStdString() << "\n" << text.toStdString() << "\nApprove? [y/N/a = approve all from now on] " << std::flush;
            std::string answer;
            if(!std::getline(std::cin,answer) || (answer != "y" && answer != "Y" && answer != "a" && answer != "A"))
                return false;
            if(answer == "a" || answer == "A")
                approval = "auto";
            web.report_step("running");
            return true;
        };
        std::vector<std::vector<std::string> > cmds;
        QJsonArray results;
        if(auto command = request["command"];!command.isUndefined() && !command.isNull() && !ai_commands(command,cmds))
            results.append(QJsonObject{{"status","error"},{"error","invalid cmd text"}});
        for(const auto& cmd : cmds)
        {
            chat << ">";
            for(auto param : cmd) // first line of each parameter, shortened
                chat << " " << ((param = param.substr(0,param.find('\n'))).size() > 100 ? param.substr(0,97)+"..." : param);
            chat << std::endl;
            std::string error;
            QString upload_error;
            QJsonObject attachment;
            std::ostringstream out;
            auto* cout_buf = std::cout.rdbuf(out.rdbuf()); // the command's output is returned to the agent
            bool ok = true;
            if(cmd[0] == "run_shell")
                ok = ai_run_shell(cmd,approve,error,[&](qint64 pid){web.report_step("running",{},pid);});
            else if(cmd[0] == "write_text_file")
                ok = ai_write_text_file(cmd,approve,error);
            else if(cmd[0] == "web_upload" || cmd[0] == "web_download_prepare" || cmd[0] == "web_download")
                ok = !(attachment = web.web_file(cmd,approve,upload_error)).isEmpty();
            else if(cmd[0] == "set_work_dir") // the current directory for relative paths
            {
                if(!(ok = cmd.size() == 2 && QFileInfo(QString::fromStdString(cmd[1])).isDir() && QDir::setCurrent(QString::fromStdString(cmd[1]))))
                    error = "usage: set_work_dir <existing directory>";
                else
                    std::cout << QDir::currentPath().toStdString() << std::endl;
            }
            else if(cmd[0] == "set_title") // saved like a GUI chat's title, and names the Doc
            {
                auto* info = ai_info::create(web.web_session_id,"Web","Web");
                if(!(ok = cmd.size() == 2 && info->save_title(QString::fromStdString(cmd[1]))))
                    error = "usage: set_title <title>";
                else
                    web.set_web_title(info->project_titles);
            }
            else
                ok = false,error = "not available in headless mode (--action=ai); use run_shell, write_text_file, web_upload, web_download_prepare, web_download, set_work_dir, or set_title, and dispatch long work with run_shell \"sbatch ...\"";
            std::cout.rdbuf(cout_buf);
            std::cout << out.str() << std::flush;
            if(!upload_error.isEmpty())
                error = upload_error.toStdString();
            QJsonObject result{{"cmd",QString::fromStdString(cmd[0])},{"status",ok ? "success" : "error"}};
            if(!out.str().empty())
                result["output"] = QString::fromStdString(out.str());
            if(!attachment.isEmpty())
                result["attachment"] = attachment;
            if(!ok)
            {
                result["error"] = QString::fromStdString(error)+". Read DSI Studio Manuals and retry.";
                chat << "  error: " << error << std::endl;
            }
            results.append(result);
            if(!ok)
                break;
        }
        return QJsonObject{{"status",results.isEmpty() ? "success" : results.last().toObject()["status"].toString()},{"result",results}};
    };
    if(web.google_refresh_token.isEmpty() && !web.sign_in_device())
        return 1;
    auto start = [&](QString file,QString session)
    {
        web.start_web(file,session);
        chat << "\nDSI Studio Web AI session ready.\n\nPaste this prompt into a web AI agent:\n\n" << web.prompt().toStdString()
             << " This session is headless: DSI Studio has no window, so use only its headless commands. Approval: " << approval << "." // tells the agent the mode before its first request
             << "\n\nDSI Studio is running. Press Ctrl+C to stop." << std::endl;
        if(!po.has("debug")) // from now on only the chat history
            std::cout.rdbuf(&quiet),std::cerr.rdbuf(&quiet);
    };
    // --session=-1 (default) starts a new chat; --session=N resumes the N-th chat listed below (0 = most recent),
    // --session=<codename> or <uuid> that chat; --remove_session=N, <codename>, or <uuid> removes that chat instead
    QList<QJsonObject> sessions;
    QStringList items;
    auto session_arg = QString::fromStdString(po.has("remove_session") ? po.get("remove_session") : po.get("session","-1"));
    bool by_number = false;
    int choice = session_arg.toInt(&by_number),matches = 0;
    if(!by_number) // a codename or UUID not listed is reported below
        choice = INT_MAX;
    for(const auto& each : web.list_web_sessions())
    {
        auto file = each.toObject();
        auto name = file["name"].toString();
        auto props = file["appProperties"].toObject();
        if(!is_valid_session_id(props["dsi_session"].toString()))
            continue;
        if(!by_number && (props["dsi_code"].toString() == session_arg || props["dsi_session"].toString() == session_arg))
        {
            choice = int(sessions.size());
            ++matches;
        }
        sessions << file;
        auto title = name.section(web_title_separator,0,0);
        if(title == name) // never titled through set_title on any machine: the local title, if any
            title = web.settings.value("ai/title/"+props["dsi_session"].toString(),"(untitled)").toString();
        items << QString("%1  %2  %3").arg(props["dsi_code"].toString(),-16).arg(title.left(30),-30)
                 .arg(QDateTime::fromString(file["modifiedTime"].toString(),Qt::ISODate).toLocalTime().toString("yyyy-MM-dd hh:mm"));
    }
    chat << "\nDSI Studio AI chats (--session)\n\n";
    for(int i = 0;i < items.size();++i)
        chat << i << ". " << items[i].toStdString() << "\n";
    auto option_name = po.has("remove_session") ? "--remove_session=" : "--session=";
    if(matches > 1)
        return tipl::error() << option_name << session_arg.toStdString() << " matches multiple chats; use 0 to " << int(sessions.size())-1 << " or the session UUID",1;
    if(sessions.isEmpty() && choice >= 0)
        return tipl::error() << option_name << session_arg.toStdString() << ": no chats are listed; use -1 for a new chat",1;
    if(choice >= int(sessions.size()))
        return tipl::error() << option_name << session_arg.toStdString() << " is not listed; use 0 to " << int(sessions.size())-1 << ", a listed codename/UUID, or -1 for a new chat",1;
    if(po.has("remove_session"))
    {
        if(choice < 0)
            return tipl::error() << "no listed chat to remove",1;
        auto props = sessions[choice]["appProperties"].toObject();
        if(!web.remove_web_session(sessions[choice]["id"].toString(),props["dsi_session"].toString()))
            return tipl::error() << "cannot move the chat's Google Doc to the trash",1;
        chat << "removed " << props["dsi_code"].toString().toStdString() << std::endl;
        return 0;
    }
    if(choice < 0)
        web.create_web_session(start);
    else
    {
        auto file = sessions[choice];
        if(QDateTime::fromString(file["modifiedTime"].toString(),Qt::ISODate).secsTo(QDateTime::currentDateTimeUtc()) < 60)
            chat << "Warning: this chat changed in the last minute; make sure no other DSI Studio is serving it." << std::endl;
        start(file["id"].toString(),file["appProperties"].toObject()["dsi_session"].toString());
    }
    auto result = QCoreApplication::exec();
    std::cout.rdbuf(chat.rdbuf());
    std::cerr.rdbuf(cerr_buf);
    return result;
}
