#ifndef AI_AGENT_HPP
#define AI_AGENT_HPP

#include <QByteArray>
#include <QDateTime>
#include <QElapsedTimer>
#include <QHash>
#include <QJsonObject>
#include <QMainWindow>
#include <QNetworkAccessManager>
#include <QSettings>
#include <QStringList>
#include <QTimer>
#include <functional>


class MainWindow;
class QMenu;
class QProcess;
class QShowEvent;
class QCloseEvent;
struct ai_info; // full definition: cmd/ai.hpp -- every use here is by pointer/reference, so a forward declaration is enough; .cpp files that need member access include cmd/ai.hpp directly
enum class session_status; // full definition: cmd/ai.hpp

namespace Ui {
class AIAgent;
}

enum class ai_agent_status
{
    Unknown = 0,
    Checking,
    NotInstalled,
    SignInRequired,
    Ready,
    Error
};

// one entry per local agent name: executable, discovered model profiles, and cached status
struct ai_agent_entry
{
    QString executable;
    QJsonObject profiles;
    QString status_info;
    ai_agent_status status = ai_agent_status::Unknown;
    quint64 status_check_id = 0;
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
    int ai_debug_level = 0; // "ai/debug": 0 = disabled, 1 = truncated, 2 = complete
    void ai_log(QString text);
    QJsonObject next_json_line(QProcess*); // reads, logs and parses one available stdout line; callers loop while(process->canReadLine())

    // app-wide default agent/model for the next New Chat; an existing chat uses its own model_settings
    QHash<QString,ai_agent_entry> agent_entries;
    QString current_agent = "Codex";
    QString current_model_name; // empty is the one internal representation of "no explicit choice" (see model_combo_key()); never the literal word "default"
    QJsonObject current_model_info;
    void update_agent_status_label();

    QNetworkAccessManager web_manager; // Google OAuth and API calls

    // Google OAuth (desktop, PKCE + loopback, drive.file only): tokens are never logged or written to chat history
    QString google_access_token,google_refresh_token = settings.value("ai/google_refresh_token").toString(); // saved, so a restart stays signed in
    QDateTime google_token_expiry;
    bool sign_in_google(); // system-browser sign-in; true once a token is held
    void google_token_post(QList<QPair<QString,QString>> form,std::function<void(QString error)> done); // token endpoint (code exchange or refresh); stores the tokens; a failure clears them
    void with_google_token(std::function<void(QString token)> call); // the access token, refreshed first when near expiry; empty when signed out
    // Web channel: each Web chat's raw JSON file (in the "DSI Studio AI" folder) is a single-slot mailbox, polled directly
    QTimer web_timer;
    QString google_folder_id = settings.value("ai/google_folder_id").toString();
    QString google_file_id,web_session_id; // the connected chat's Doc (empty when stopped) and session
    qint64 web_last_id = 0;
    QElapsedTimer web_idle; // polling stops after 3 minutes without a request
    QJsonObject web_pending_result; // staged until its reply is confirmed; retried, never re-executed
    void google_api(const QByteArray& verb,const QString& url,const QJsonObject& body,std::function<void(QJsonObject)> done); // one Drive/Docs call; an empty object means failure
    void write_web_file(const QString& file,const QJsonObject& message,std::function<void(bool)> done); // replaces the session file's content
    void create_web_session(); // signs in and resolves the folder when needed, then creates the Doc and the chat
    void start_web(ai_info&); // starts (or resumes) polling this chat's Doc
    void stop_web(const QString& message = "Web stopped."); // stops polling; the Doc and session stay
    void poll_web();
    void publish_web_result();
    ai_info* selected_info() const; // ai_info bound to the sidebar's current chat, or null if none is selected
    bool web_connected(const ai_info&) const; // true iff this chat's Doc is the one being polled
    enum class send_action {Disabled,Send,Stop,Resume}; // local agents use persistent stdin/stdout processes; send-vs-queue is only internal startup timing
    send_action current_send_action() const; // single source of truth for what the Send button means right now, including whether it's clickable at all -- update_send_button() only turns this into a label/enabled state, on_ai_send_message_clicked() only executes it
    void update_send_button(); // reflects Send / Stop / Resume / disabled, purely from current_send_action() and whether a chat is selected
    void create_new_chat(const QString& provider);
    bool run_new_chat_dialog(const QString& title,const QString& accept_text); // the agent/model picker for New Chat and Change Agent/Model: sets current_agent/model on accept; Web starts its own session and returns false

    void add_ai_history(ai_info&,const QString&,const QString&);
    void add_ai_reply(ai_info&,const QString&,const QString&);
    void add_ai_reply(QProcess*,const QString& chat,const QString& reasoning);
    bool run_agent_login(const QString& provider);
    void finish_agent_turn(QProcess* process,QString error = {},bool cancelled = false); // one turn-end rule: idle (WaitingUser) or Failed with an error entry
    ai_info* establish_agent_session(QProcess* process,QString new_session = {}); // the one session-identity rule for every local agent; nullptr when the launch must stop
    void set_ai_status(const QString&,session_status,QString); // always updates/logs the session; updates the bottom label only when this chat is selected
    void update_ai_status(const ai_info&,bool = false); // presentation only; pulse toggles a running status dot on a real status update
    void show_ai_project(ai_info&,QJsonObject = {}); // sidebar row: create/update it, blink if the update is for a background chat, select it if nothing else was selected -- renders the chat transcript itself (show_ai_history()) only when this chat is the one currently selected
    void show_ai_history(ai_info&,QJsonObject added_entry); // markdown->HTML transcript rendering: a full rebuild, or just appending added_entry when that alone is enough
    void update_agent_models(const QString&,const QStringList&,bool);
    void refresh_agent_executables(); // re-runs local-agent executable discovery into agent_entries[agent].executable
    void refresh_agent_status(const QString& provider = {});
    void refresh_agent_models(const QString& provider); // one model-list probe per local agent
    void refresh_ollama_models();
    void start_ai(ai_info&,const QString&);
    QStringList configure_codex(const ai_info&,const QString&); // reads info.sessions/info.status/info.launch_* as of the call -- synchronous only, never captured into the process's own async handlers (Codex can still rename/rekey the session)
    QStringList configure_claude(const ai_info&,const QString&);
    QStringList configure_muse(const ai_info&,const QString&);
    QStringList configure_antigravity(const ai_info&,const QString&);
    QStringList configure_grok(const ai_info&,const QString&);
    QString prepare_ai(ai_info&); // populates info.launch_* and, on success, info.processes; returns the resolved executable, empty on failure; the process callbacks advance info.status

public:
    explicit AIAgent(MainWindow*);
    ~AIAgent();
    void ai_request(const QByteArray& request,QByteArray& reply); // entry point for the local-socket AI protocol: resolves/creates the session, hands everything to MainWindow's CMD command center, then refreshes the sidebar
    void update_current_window(QWidget*); // called where open_fib/open_src/open_image create their window; no-op unless an AI dispatch is in progress

signals:
    void agent_status_changed(const QString& provider);
    void agent_models_changed(const QString& provider);

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
