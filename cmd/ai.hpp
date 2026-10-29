#ifndef CMD_AI_HPP
#define CMD_AI_HPP

#include <QByteArray>
#include <QDateTime>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QMap>
#include <QNetworkAccessManager>
#include <QObject>
#include <QSettings>
#include <QStringList>
#include <QTimer>
#include <QUrl>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

class QListWidgetItem;
class QProcess;
class QProcessEnvironment;
class QSettings;

enum class session_status {New,Thinking,WaitingUser,Completed,Failed}; // declaration order is not meaningful -- ai_info::is_running() classifies by name, not ordinal comparison
// New: no confirmed backend id yet (never launched, or a launch/reconnect awaiting the agent's own confirmation);
//   the only status that means "start a session, do not resume".
// Thinking: the session is established and the agent is preparing a response.
// WaitingUser: the agent finished its response and is waiting for the user's next message.
// Completed / Failed: the process ended normally / abnormally after the session was established; both stay resumable.
// A chat that fails before it is established returns to New: it has no real id to preserve.

struct ai_info{
    QString sessions,agent_name,provider,project_titles;
    QProcess* processes = nullptr;
    QList<QJsonObject> projects;
    QListWidgetItem* project_items = nullptr;
    QJsonObject model_settings; // "model"/"info": local Codex/Claude model choice; "google_file_id": session Doc, Web chats only
    quint64 log_position = quint64(-1);
    QString current_window = "main"; // persists across requests until changed by "set_window"
    session_status status = session_status::New; // see session_status -- this field is the only source of truth for whether this session has a real, established backend identity, and (via the sidebar dot's color) for whether the last run had trouble
    QString status_message;
    static ai_info* find(const QString&);
    static ai_info* create(QString,QString,QString = {}); // session, provider, optional display agent name
    static QString history_file(const QString&);
    static QString config_file(const QString&); // agent/model/Web-channel metadata: separate from history_file so it can be rewritten cheaply without touching the chat transcript
    void save_config() const;
    bool save_title(QString);
    QJsonObject record_history(QJsonObject); // returns the recorded entry (with "time" filled in), not the caller's pre-call copy -- written unconditionally, regardless of status
    QJsonObject record_reply(const QString&,const QString&); // returns the recorded entry so callers can pass it on to show_ai_project() for blink/visibility handling
    QString title() const {return project_titles.isEmpty() ? (agent_name.isEmpty() ? sessions : agent_name+"@"+sessions) : project_titles;}
    // "waiting on the agent" -- Thinking always is; New only while a launch/reconnect is actually in flight
    // (processes set). An untouched, never-launched New chat has neither and must not count as running
    bool is_running() const {return status == session_status::Thinking || (status == session_status::New && processes);}
    QString details() const;
};

// session registry: defined in cmd/ai.cpp alongside ai_info's own member implementations; every chat,
// local or web, is one entry here, keyed by its own ai_info::sessions
extern std::unordered_map<QString,ai_info> ai_infos;
extern QString ai_project_dir; // defined and created (mkpath) in main.cpp, before any window exists

bool is_valid_session_id(const QString&); // true iff the string is exactly a UUID (no braces) -- every id accepted as "the" resumable session identity (pipe requests, Web sessions, Codex's self-reported thread_id) must satisfy this or be rejected outright, not silently tolerated
QString session_status_text(session_status); // human-readable label shared by the sidebar dot, details, and bottom status line
QList<ai_info*> load_ai_projects(); // restores saved chats from ai_project_dir into ai_infos, oldest first; sessions already in ai_infos are skipped, never reloaded over
ai_info* assign_ai_session(const QString& from,const QString& to); // renames an existing session's key/files/title in place (e.g. Codex's placeholder id -> its real thread_id); a no-op lookup if from == to; nullptr (nothing changed) if to already exists or from is missing
QUrl agent_install_url(const QString& provider); // shared by the sidebar's Install button and a launch that finds the CLI missing, so the two can't drift apart
QByteArray json_line(const QJsonObject&); // one compact JSON message per line, the framing every agent protocol and the history file use
QPair<QUrl,bool> ai_ollama_url(QString host = QSettings().value("ai/ollama_host","localhost").toString(),int port = QSettings().value("ai/ollama_port",11434).toInt()); // ("ai/ollama_host"+"ai/ollama_port" as a URL, whether a host is actually configured) -- the bool distinguishes "empty/default" from "genuinely set to something that parses to the same URL"

// local CLI agents: discovery and noninteractive readiness probes (no login, no model turn, no widgets)
enum class ai_agent_status
{
    Unknown = 0,
    Checking,
    NotInstalled,
    SignInRequired,
    Ready,
    Error
};
extern const QStringList local_agents; // every local CLI agent, in settings order
QString find_agent_executable(const QString& provider); // PATH first, then the provider's known install location; empty if not found
ai_agent_status check_agent_status(const QString& provider,const QString& executable,QString& info); // blocking; info: the credential description when Ready
QProcessEnvironment agent_environment(const QString& provider);
void start_process(QProcess& process,const QString& executable,QStringList args);
void kill_process_tree(QProcess* process);
QByteArray jsonrpc(const QString& id,const QString& method,const QJsonObject& params = {});
QByteArray muse_initialize();
QByteArray grok_initialize();
QString muse_uuid_v7(); // a time-ordered UUID, as Muse expects for session and command ids
QByteArray muse_command(const QString& id,const QString& method,QJsonObject params);
// how a caller presents a local agent's turn; sign_in and signed_out are optional
struct agent_hooks
{
    std::function<void(ai_info&)> status;                   // info.status and status_message changed
    std::function<void(ai_info&,const QJsonObject&)> shown; // refresh the chat; entry: the entry just recorded, if any
    std::function<bool(const QString&)> sign_in;            // before a launch: true when the provider may start
    std::function<void(const QString&)> signed_out;         // the provider rejected its credential
    std::function<void(const QString&)> log;                // debug line
};
// sends text to info's local agent, launching or resuming its process when needed (shared by the GUI and headless);
// false, with the failure recorded, when the launch cannot start
bool start_agent(ai_info& info,const QString& text,const QString& executable,const QString& work_dir,QObject* owner,const agent_hooks& hooks);
QJsonObject ai_agent_record(const QString& provider,const QString& scope,const QString& executable,ai_agent_status status,
                            const QString& info,const QStringList& models,const QDateTime& checked); // one get_agent_settings entry; scope: "gui" or "headless"
// system commands shared by MainWindow's command center and headless --action=ai: get_agent_settings [agent|ready],
// new_chat <agent> [model] [title], send_chat <session_id> <message>, get_chat [-N | session_id [all|-N|N|N-|N-M]]; false if cmd
// is none of them. caller: the requesting chat; agent_records: readiness for the given agents on this host; send: deliver text to a chat's agent
bool ai_system_command(const std::vector<std::string>& cmd,const ai_info& caller,const std::function<QJsonArray(const QStringList&)>& agent_records,
                       const std::function<bool(ai_info&,const QString&)>& send,QString& output,QString& error);

// shared by MainWindow's command center and headless --action=ai
using ai_approval = std::function<bool(const QString& title,const QString& request,bool read_only)>; // true approves; read_only: a shell command that cannot change anything
extern QMap<QString,QString> shell_tasks; // running asynchronous run_shell (curl) commands, by id
bool ai_commands(const QJsonValue& command,std::vector<std::vector<std::string> >& cmds); // a request's command or batch as argument lists; false on an empty cmd
bool ai_run_cli(const std::vector<std::string>& cmd,std::string& error); // GUI only: headless --action=ai dispatches long work instead of running it in the AI process
bool ai_write_text_file(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error);
bool ai_run_shell(const std::vector<std::string>& cmd,const ai_approval& approve,std::string& error,
                  const std::function<void(qint64)>& started = {}); // started: the PID of a synchronous command, so a Web agent can find it if DSI Studio stops

// the Web channel, without GUI: Google OAuth tokens (drive.file only; never logged or written to chat history) and one session Doc
// (in the "DSI Studio AI" folder) used as a single-slot JSON mailbox, read directly for 3 minutes after activity, then on a Drive
// modifiedTime change. AIAgent owns one for GUI Web chats; --action=ai runs a local one
class web_bridge : public QObject
{
public:
    QSettings settings;
    QNetworkAccessManager web_manager; // Google OAuth and API calls
    QString google_access_token,google_refresh_token = settings.value("ai/google_refresh_token").toString(); // saved, so a restart stays signed in
    QDateTime google_token_expiry;
    bool google_device_auth = settings.value("ai/google_device_auth").toBool(); // the saved token came from the device client (headless sign-in)
    QTimer web_timer;
    QString google_folder_id = settings.value("ai/google_folder_id").toString();
    QString google_error; // the last Google failure, shown in the Web status; never holds a token
    QString google_file_id,web_session_id,web_code; // the connected Doc (empty when stopped), session, and codename (e.g. neon-raven)
    qint64 web_last_id = 0;
    QElapsedTimer web_idle; // since the last request or result: selects Docs or Drive polling and, if stop_when_idle, stops polling after 1 hour
    bool stop_when_idle = true;
    QString web_modified_time; // Drive modifiedTime of the last body read that needed no action
    QJsonObject web_pending_result; // staged until its reply is confirmed; retried, never re-executed
    bool web_dispatching = false; // a Web request is running: report_step() may rewrite its "processing" message
    std::function<QJsonObject(const QJsonObject& request)> dispatch; // runs one agent request and returns its response
    std::function<void(session_status,const QString&)> status = [](session_status,const QString&){};
    std::function<void(const QString&)> warn;
    web_bridge(void);
    bool sign_in_device(void); // headless sign-in; true once a token is held
    void google_token_post(QList<QPair<QString,QString>> form,std::function<void(QString error)> done); // token endpoint (code exchange or refresh); stores the tokens; only invalid_grant (revoked) clears the saved refresh token
    void with_google_token(std::function<void(QString token)> call); // the access token, refreshed first when near expiry; empty when signed out
    void google_api(const QByteArray& verb,const QString& url,const QJsonObject& body,std::function<void(QJsonObject)> done); // one Drive/Docs call; an empty object means failure
    void write_google_doc(const QJsonObject& doc,QJsonObject message,std::function<void(bool)> done = {}); // replaces the body with one JSON message; doc (from documents.get) guards the revision
    void create_web_session(std::function<void(QString file,QString session)> done); // resolves the folder when needed, then creates a session Doc
    void start_web(const QString& file,const QString& session); // starts (or resumes) polling this Doc after its last message
    QJsonArray list_web_sessions(void); // the recent session Docs in the DSI Studio AI folder: id, name ("DSI Studio <codename>", or "<title> <middle dot> DSI Studio <codename>" after set_title), modifiedTime, appProperties (dsi_session, dsi_code)
    bool remove_web_session(const QString& file_id,const QString& session); // trashes the session Doc (if any) and removes its local history, config, and title; false if the Doc was not trashed
    void set_web_title(const QString& title); // renames the connected Doc to "<title> <middle dot> DSI Studio <codename>"
    void stop_web(const QString& message = "Web stopped."); // stops polling; the Doc and session stay
    QString prompt(void) const; // the connection prompt an agent is given
    QString google_doc_url(void) const; // documents.get for the connected Doc, trimmed to the mailbox fields
    void poll_web(const QString& modified_time = QString()); // modified_time: the Drive change that triggered this read
    void publish_web_result(void);
    void report_step(const QString& step,const QString& approval = {},qint64 pid = 0); // the running Web request's "processing" message gains step (waiting_approval/running) and the shell PID; waits for the write
    QJsonObject google_api_wait(const QByteArray& verb,const QString& url,const QJsonObject& body = {}); // google_api() that waits for its reply
    QJsonObject web_file(const std::vector<std::string>& cmd,const ai_approval& approve,QString& error); // web_upload, web_download_prepare, web_download: the attachment, or empty on error
};

#endif
