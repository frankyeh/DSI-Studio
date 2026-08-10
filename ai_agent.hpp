#ifndef AI_AGENT_HPP
#define AI_AGENT_HPP

#include <QByteArray>
#include <QJsonObject>
#include <QList>
#include <QMainWindow>
#include <QNetworkAccessManager>
#include <QSettings>
#include <QStringList>
#include <QTimer>
#include <QUrl>

#include <array>
#include <unordered_map>

class MainWindow;
class QListWidgetItem;
class QMenu;
class QNetworkReply;
class QNetworkRequest;
class QProcess;
class QShowEvent;
class QCloseEvent;

namespace Ui {
class AIAgent;
}

enum class ai_provider {Unknown = -1,Infer = -2,Codex = 0,Claude = 1,ChatGPT = 2,AgentServer = 3}; // ChatGPT/AgentServer: never index AIAgent::agent_entries (sized for Codex/Claude only). AgentServer: created by an external agent's request over the local pipe/socket server -- a log/routing record, never backed by a local subprocess, so it can't send a live chat or change its model. Unknown: genuinely unrecognized/invalid, always a hard failure. Infer: derive it from the agent name via identify_provider() -- these two are never interchangeable, so ai_info::create() takes them as one required argument instead of one meaning silently standing in for the other
enum class ai_input {User,Pending};
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
bool is_valid_session_id(const QString&); // true iff the string is exactly a UUID (no braces) -- every id accepted as "the" resumable session identity (pipe requests, GitHub issue sessions, Codex's self-reported thread_id) must satisfy this or be rejected outright, not silently tolerated
QString session_status_text(session_status); // human-readable label shared by the sidebar dot, details, and bottom status line

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

// one entry per ai_provider (Codex/Claude): resolved executable path (empty if not found) and the discovered model profiles (name -> info)
struct ai_agent_entry
{
    QString executable;
    QJsonObject profiles;
};

class AIAgent : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    ai_info* dispatching_info = nullptr; // set around dispatch_cmd() in ai_request(); lets open_fib/open_src/open_image report their new window via update_current_window()
    Ui::AIAgent* ui;
    QSettings settings;
    QMenu* ai_project_menu = nullptr;
    QTimer* ai_status_timer = nullptr;
    int ai_debug_level = 0; // "ai/debug" setting: 0 = disabled, 1 = enabled (truncated), 2 = enabled (complete); read from QSettings in the constructor, kept in sync by AI Settings' own setValue+assign
    void ai_log(QString text);

    // app-wide default agent/model: only consulted for a chat that doesn't exist yet (New Chat's pre-fill, and
    // "Change Agent/Model" with nothing selected) -- an existing chat's own ai_info::model_settings is always
    // authoritative for that chat once created, never reconciled against these
    std::array<ai_agent_entry,2> agent_entries; // indexed by ai_provider
    int current_agent_index = 0;
    QString current_model_name; // empty is the one internal representation of "no explicit choice" (see model_combo_key()); never the literal word "default"
    QJsonObject current_model_info;
    void update_agent_status_label();
    void try_set_current_model(const QString& name); // name is always written as-is, even if unknown to profiles (the model combo is editable, so a typed name is meaningful, not a mistake); writes the app-wide default above, not any chat's own model
    void set_chat_model(ai_info& info,const QString& name) const; // same resolution as try_set_current_model(), but writes directly into this chat's own model_settings and persists it

    // GitHub issue channel: the issue body carries the next request; one pinned comment (marked "dsi_session_result":true) carries the result
    QNetworkAccessManager github_manager;
    QTimer github_timer;
    QUrl github_issue_api,github_result_api;
    QByteArray github_etag;
    QString github_token; // snapshot taken at connect time so a mid-session Settings change cannot swap the identity underneath a poll
    qint64 github_last_id = 0;
    QJsonObject github_pending_result; // staged until its PATCH is confirmed; retried, never re-executed
    quint64 github_connection_id = 0; // bumped on connect/disconnect; rejects callbacks from a superseded connection even to the same URL
    QString web_agent_session_id; // the actual chat this GitHub connection belongs to, independent of sidebar selection; survives Stop/Resume, cleared only on a fresh (non-resume) start

    QNetworkRequest github_request(const QUrl&) const;
    bool connect_github_issue(const QString&,QString& error);
    void disconnect_github_issue();
    // shared QNetworkReply::finished preamble for the poll/publish channels: consumes reply (deleteLater, reads
    // body into data), checks for a superseded connection / rate limiting / a permanent auth failure. Returns
    // false if the caller should stop (reply already handled).
    bool handle_github_reply(QNetworkReply* reply,quint64 connection_id,int& status,QByteArray& data);
    void poll_github_issue();
    void publish_github_result(QJsonObject);
    void send_pending_result();
    ai_info* selected_info() const; // ai_info bound to the sidebar's current chat, or null if none is selected
    bool github_connected(const ai_info&) const; // true iff this specific chat is the one the live GitHub issue channel is currently bound to (web_agent_session_id + a non-empty github_issue_api) -- a chat can be ChatGPT-provider without being the connection's current owner (e.g. a different/older web chat)
    enum class send_action {Disabled,Send,StopLocal,StopWeb,ResumeWeb};
    send_action current_send_action() const; // single source of truth for what the Send button means right now, including whether it's clickable at all -- update_send_button() only turns this into a label/enabled state, on_ai_send_message_clicked() only executes it
    void update_send_button(); // reflects Send / Stop / Resume / disabled, purely from current_send_action() and whether a chat is selected
    bool try_connect_github_issue(const QString& url); // connect_github_issue() plus the shared success/failure UI feedback; always targets web_agent_session_id, which the caller guarantees already refers to a real chat
    bool setup_github_token();
    void new_chat_dialog(bool resume); // shared by New Chat and Resume; resume locks the mode and disables the local agent/model panel
    ai_info* create_new_chat(const QString& agent); // drops any abandoned empty placeholder first, then creates+selects a fresh chat (status New, a bare uuid) for the given agent name ("Codex"/"Claude"/"ChatGPT(Web)"); for web, this exists even before a connection is attempted, so a failed connection is just this chat's own Error state rather than needing separate anonymous-session tracking
    bool run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                              int& agent_index,QString& value); // value: model name for a local agent, issue URL for ChatGPT (web) -- mutually exclusive, caller checks agent_index == ai_provider::ChatGPT
        // builds the Local/Web picker shared by new_chat_dialog() and on_ai_agent_status_clicked(); returns false if cancelled

    void add_ai_history(ai_info&,const QString&,const QString&);
    void add_ai_reply(ai_info&,const QString&,const QString&);
    bool run_agent_login(ai_provider provider);
    QString agent_login_info(ai_provider provider); // "" means not signed in (or executable missing); otherwise a short human-readable account/plan summary straight from the CLI's own status query -- never cached, so an external login/logout is always reflected
    void refresh_login_buttons(); // shows/hides ai_codex_login/ai_claude_login (bottom of the chat list) based on agent_login_info()
    void set_ai_status(const QString&,session_status,QString); // always updates/logs the session; updates the bottom label only when this chat is selected
    void update_ai_status(const ai_info&,bool = false); // presentation only; pulse toggles a running status dot on a real status update
    void show_ai_project(ai_info&,QJsonObject = {}); // sidebar row: create/update it, blink if the update is for a background chat, select it if nothing else was selected -- renders the chat transcript itself (show_ai_history()) only when this chat is the one currently selected
    void show_ai_history(ai_info&,QJsonObject added_entry); // markdown->HTML transcript rendering: a full rebuild, or just appending added_entry when that alone is enough
    void update_agent_models(int,const QStringList&,bool);
    void refresh_ollama_models();
    void refresh_codex_models();
    void start_ai(ai_info&,const QString&,ai_input);
    QStringList configure_codex(const ai_info&,const QString&); // reads info.sessions/info.status/info.launch_* as of the call -- synchronous only, never captured into the process's own async handlers (Codex can still rename/rekey the session)
    QStringList configure_claude(const ai_info&,const QString&); // same contract as configure_codex
    void prepare_ai(ai_info&,const QString&,ai_input); // populates info.launch_* and, on success, info.processes -- check info.processes to see whether it succeeded; the process callbacks advance info.status

public:
    explicit AIAgent(MainWindow*);
    ~AIAgent();
    void ai_request(const QByteArray& request,QByteArray& reply); // entry point for the local-socket AI protocol: resolves/creates the session, hands everything to MainWindow's CMD command center, then refreshes the sidebar
    void update_current_window(QWidget*); // called where open_fib/open_src/open_image create their window; no-op unless an AI dispatch is in progress

protected:
    void showEvent(QShowEvent*) override;
    void closeEvent(QCloseEvent*) override;

private slots:
    void on_ai_quick_settings_clicked();
    void on_ai_new_chat_clicked();
    void on_ai_send_message_clicked();
    void on_ai_agent_status_clicked();
};

#endif
