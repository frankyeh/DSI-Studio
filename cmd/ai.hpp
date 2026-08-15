#ifndef CMD_AI_HPP
#define CMD_AI_HPP

#include <QByteArray>
#include <QJsonObject>
#include <QList>
#include <QStringList>
#include <QUrl>

#include <unordered_map>

class QComboBox;
class QLabel;
class QListWidgetItem;
class QNetworkAccessManager;
class QNetworkReply;
class QNetworkRequest;
class QProcess;
class QSettings;
class QWidget;

enum class ai_provider {Unknown = -1,Infer = -2,Codex = 0,Claude = 1,ChatGPT = 2,AgentServer = 3}; // ChatGPT/AgentServer: never index AIAgent::agent_entries (sized for Codex/Claude only). AgentServer: created by an external agent's request over the local pipe/socket server -- a log/routing record, never backed by a local subprocess, so it can't send a live chat or change its model. Unknown: genuinely unrecognized/invalid, always a hard failure. Infer: derive it from the agent name via identify_provider() -- these two are never interchangeable, so ai_info::create() takes them as one required argument instead of one meaning silently standing in for the other
enum class session_status {New,Thinking,WaitingUser,Completed,Failed}; // declaration order is not meaningful -- ai_info::is_running() classifies by name, not ordinal comparison
// New: chat created, no launch ever attempted yet, OR a launch/reconnection is currently in flight (the OS
//   process started, waiting for the agent's own protocol confirmation: Codex "thread.started", Claude
//   stream-json "system"/"init") -- the only value that means "no confirmed real id yet, use --session-id,
//   not --resume". Animated the same as Thinking (both mean "waiting on the agent"), since a fresh chat is
//   just as much "nothing to show yet" as one actively connecting.
// Thinking: the session is established and the agent is preparing a response.
// WaitingUser: the agent finished its response and is waiting for the user's next message.
// Completed: the process ended normally after having been Thinking/WaitingUser -- the session id remains resumable.
// Failed: the process ended abnormally after having been Thinking/WaitingUser -- functionally the same as Completed
//   (still resumable with --resume), it only tells the user something went wrong on the last run.
// A chat that fails before ever reaching Thinking/WaitingUser (FailedToStart, or a crash while still
// connecting) never becomes Failed -- it has no real id to preserve, so it reverts all the way back to New.

struct ai_info{
    QString sessions,agent_name,project_titles;
    ai_provider provider = ai_provider::Unknown;
    QProcess* processes = nullptr;
    QList<QJsonObject> projects;
    QStringList prompts;
    QListWidgetItem* project_items = nullptr;
    QJsonObject model_settings; // "model"/"info": local Codex/Claude model choice; "github_issue_url": bound issue, web (ChatGPT) sessions only
    quint64 log_position = quint64(-1);
    QString current_window = "main"; // persists across requests until changed by "set_window"
    session_status status = session_status::New; // see session_status -- this field is the only source of truth for whether this session has a real, established backend identity, and (via the sidebar dot's color) for whether the last run had trouble
    QString status_message;
    // the most recent (or in-flight) local launch attempt -- meaningful only while prepare_ai()/configure_*()
    // are actively using it; an idle chat just carries the last attempt's resolved values, always fully
    // overwritten before the next launch reads them. Kept on ai_info itself (not a separate parameter) so
    // configure_codex()/configure_claude() can read it straight off the chat, and so a not-yet-renamed old
    // placeholder's launch data stays reachable via ai_info::find() alone
    QString launch_name,launch_executable,launch_model;
    QUrl launch_model_url;
    static ai_provider identify_provider(const QString&);
    static ai_info* find(const QString&);
    static ai_info* create(QString,QString,ai_provider); // the one constructor for the whole registry; pass ai_provider::Infer to derive it from a trusted agent name (Codex/Claude/ChatGPT/...), or a known value directly (AgentServer, a persisted reload) -- never a default, every call site states its intent
    static QString history_file(const QString&);
    static QString config_file(const QString&); // agent/model/github-channel metadata: separate from history_file so it can be rewritten cheaply without touching the chat transcript
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

bool is_valid_session_id(const QString&); // true iff the string is exactly a UUID (no braces) -- every id accepted as "the" resumable session identity (pipe requests, GitHub issue sessions, Codex's self-reported thread_id) must satisfy this or be rejected outright, not silently tolerated
QString session_status_text(session_status); // human-readable label shared by the sidebar dot, details, and bottom status line
ai_info* assign_ai_session(const QString& from,const QString& to); // renames an existing session's key/files/title in place (e.g. Codex's placeholder id -> its real thread_id); a no-op lookup if from == to
QUrl agent_install_url(ai_provider provider); // shared by the sidebar's Install button and a launch that finds the CLI missing, so the two can't drift apart
void stop_blink(QWidget* row); // stops a sidebar row's attention-getting blink animation and clears its stylesheet
void update_status_dot(QLabel* dot,session_status status,bool pulse); // presentational: sets a sidebar/status dot's color and pulse animation for the given status
QString ai_dialog_style(); // shared stylesheet for the GitHub-setup/new-chat dialogs
bool github_permanent_failure(int http_status); // true iff retrying this GitHub HTTP status can't ever succeed (bad token/permissions/resource)
int github_retry_delay(QNetworkReply* reply,const QByteArray& data); // wait time in ms if GitHub signals rate limiting (429, or 403 meaning the same), else 0
QByteArray github_blocking(QNetworkAccessManager& manager,const QNetworkRequest& request,
                            const char* verb,const QByteArray& body,bool& ok,QString& error); // blocking GET/POST/PATCH: connect_github_issue() is one-shot and user-initiated, so a short local event loop keeps its bool/error interface synchronous without added state
QByteArray claude_input(const QString& text); // wraps text in Claude's stream-json stdin message format
QByteArray codex_turn_start(const QString& id,const QString& thread_id,const QString& text); // wraps text as a Codex app-server "turn/start" request on an idle thread
QByteArray codex_turn_steer(const QString& thread_id,const QString& turn_id,const QString& text); // wraps text as a Codex app-server "turn/steer" request into the thread's currently active turn
QPair<QUrl,bool> ai_ollama_url(const QSettings& settings); // ("ai/ollama_host"+"ai/ollama_port" as a URL, whether a host is actually configured) -- the bool distinguishes "empty/default" from "genuinely set to something that parses to the same URL"
QString model_combo_key(const QComboBox& model); // strips the " (Ollama@host)" suffix off an Ollama model's display text; "default" is a UI label only -- its data value is empty, the one universal representation of "no explicit choice"
void set_model_selector(QComboBox& model,const QJsonObject& profiles,
                        QString selected = {},QString fallback = {},
                        QJsonObject selected_info = {}); // populates model with "default"+profiles' native/Ollama entries, grouped and sorted, selecting selected (or falling back to fallback) -- selected_info backs an unrecognized selected value so it still shows up as a real entry

#endif
