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
enum class session_status {New,Resume}; // New: no real backend-assigned id yet (still DSI Studio's own placeholder uuid); Resume: session already has a real, established id to continue
bool is_valid_session_id(const QString&); // true iff the string is exactly a UUID (no braces) -- every id accepted as "the" resumable session identity (pipe requests, GitHub issue sessions, Codex's self-reported thread_id) must satisfy this or be rejected outright, not silently tolerated

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
    bool has_error = false; // true once a run fails, cleared on the next run; sidebar dot: green (running), red (has_error), gray (otherwise)
    session_status status = session_status::New; // New until the first launch actually starts (Claude: flips in place; Codex: flips together with the placeholder->real-id rename, since Codex assigns its own id); sessions carry no prefix/marker of their own -- this field is the only source of truth
    static ai_provider identify_provider(const QString&);
    static ai_info* find(const QString&);
    static ai_info* create(QString,QString,ai_provider); // the one constructor for the whole registry; pass ai_provider::Infer to derive it from a trusted agent name (Codex/Claude/ChatGPT/...), or a known value directly (AgentServer, a persisted reload) -- never a default, every call site states its intent
    static QString history_file(const QString&);
    static QString config_file(const QString&); // agent/model/github-channel metadata: separate from history_file so it can be rewritten cheaply without touching the chat transcript
    void save_config() const;
    bool save_title(QString);
    QJsonObject record_history(QJsonObject); // returns the recorded entry (with "time" and, for the first entry, "agent"/"model_settings" filled in), not the caller's pre-call copy
    QJsonObject record_reply(const QString&,const QString&); // returns the recorded entry so callers can pass it on to show_ai_project() for blink/visibility handling
    QJsonObject record_request(const QString& command_name,QWidget* target = nullptr); // "window" is derived from current_window; "title" (from target's window title) is included only when target is given and current_window isn't main
    QString title() const {return project_titles.isEmpty() ? (agent_name.isEmpty() ? sessions : agent_name+"@"+sessions) : project_titles;}
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

struct ai_launch;
class AIAgent : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    ai_info* dispatching_info = nullptr; // set around dispatch_cmd() in ai_request(); lets open_fib/open_src/open_image report their new window via update_current_window()
    Ui::AIAgent* ui;
    QSettings settings;
    QString ai_status_activity;
    QMenu* ai_project_menu = nullptr;
    QTimer* ai_status_timer = nullptr;
    int ai_debug_level = 0; // "ai/debug" setting: 0 = disabled, 1 = enabled (truncated), 2 = enabled (complete); read from QSettings in the constructor, kept in sync by AI Settings' own setValue+assign
    void ai_log(QString text);

    // app-wide default agent/model: only consulted for a chat that doesn't exist yet (New Chat's pre-fill, and
    // "Change Agent/Model" with nothing selected) -- an existing chat's own ai_info::model_settings is always
    // authoritative for that chat once created, never reconciled against these
    std::array<ai_agent_entry,2> agent_entries; // indexed by ai_provider
    int current_agent_index = 0;
    QString current_model_name = "default";
    QJsonObject current_model_info;
    void update_agent_status_label();
    void try_set_current_model(const QString& name); // no-op if name is unknown, matching non-editable QComboBox::setCurrentText; writes the app-wide default above, not any chat's own model
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
    enum class send_action {Disabled,Send,StopLocal,StopWeb,ResumeWeb};
    send_action current_send_action() const; // single source of truth for what the Send button means right now, including whether it's clickable at all -- update_send_button() only turns this into a label/enabled state, on_ai_send_message_clicked() only executes it
    void update_send_button(); // reflects Send / Stop / Resume / disabled, purely from current_send_action() and whether a chat is selected
    bool is_status_target(const QString& session) const; // true if session is the currently selected chat (or, if none is, the still-anonymous chat being set up) -- gates set_ai_status() calls from a background process so a chat the user isn't looking at can't hijack the status label
    bool try_connect_github_issue(const QString& url); // connect_github_issue() plus the shared success/failure UI feedback; always targets web_agent_session_id, which the caller guarantees already refers to a real chat
    void new_chat_dialog(bool resume); // shared by New Chat and Resume; resume locks the mode and disables the local agent/model panel
    ai_info* create_new_chat(const QString& agent); // drops any abandoned empty placeholder first, then creates+selects a fresh chat (status New, a bare uuid) for the given agent name ("Codex"/"Claude"/"ChatGPT(Web)"); for web, this exists even before a connection is attempted, so a failed connection is just this chat's own Error state rather than needing separate anonymous-session tracking
    bool run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                              int& agent_index,QString& value); // value: model name for a local agent, issue URL for ChatGPT (web) -- mutually exclusive, caller checks agent_index == ai_provider::ChatGPT
        // builds the Local/Web picker shared by new_chat_dialog() and on_ai_agent_status_clicked(); returns false if cancelled

    void add_ai_history(ai_info&,const QString&,const QString&);
    void add_ai_reply(ai_info&,const QString&,const QString&);
    bool run_agent_login(ai_provider provider);
    bool agent_logged_in(ai_provider provider);
    void set_ai_status(QString = {},bool = false);
    void show_ai_project(ai_info&,QJsonObject = {});
    void update_agent_models(int,const QStringList&,bool);
    void refresh_ollama_models();
    void refresh_codex_models();
    void start_ai(QString,const QString&,ai_input);
    QStringList configure_codex(const ai_launch&,QString,const QString&,session_status);
    QStringList configure_claude(const ai_launch&,QString,const QString&,session_status);
    ai_launch prepare_ai(ai_provider,const QJsonObject& model_setting,QString&,const QString&,ai_input); // model_setting: resolved by the caller (the chat's own info.model_settings, or the app-wide default if no chat exists yet) -- prepare_ai no longer re-resolves or reconciles it. session is never empty on entry (always a placeholder or a real established id); its ai_info's status may flip from New to Resume in place (Claude: the same uuid is pre-declared via --session-id, so no rename is needed)

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
