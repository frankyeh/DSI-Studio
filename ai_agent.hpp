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
#include "cmd/ai.hpp"
struct ai_info; // full definition: cmd/ai.hpp -- every use here is by pointer/reference, so a forward declaration is enough; .cpp files that need member access include cmd/ai.hpp directly
enum class session_status; // full definition: cmd/ai.hpp

namespace Ui {
class AIAgent;
}

// one entry per local agent name: executable, discovered model profiles, and cached status
struct ai_agent_entry
{
    QString executable;
    QJsonObject profiles;
    QString status_info;
    ai_agent_status status = ai_agent_status::Unknown;
    QDateTime status_time; // when status was last determined
    quint64 status_check_id = 0;
};

class AIAgent : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    ai_info* dispatching_info = nullptr; // set around dispatch_cmd() in ai_request(); lets open_fib/open_src/open_image report their new window via update_current_window()
    QString ai_chat_context; // the dispatching request's chat text, shown in approve()'s confirmation dialog
    bool ai_auto_approve = false; // the dispatching chat's approval mode, snapshotted at request start
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
    void update_model_list(); // the selected local chat's model dropdown; hidden for Web, AgentServer, or no chat

    web_bridge web; // the Web channel: Google tokens and the connected chat's session Doc
    bool sign_in_google(); // system-browser sign-in (PKCE + loopback); true once a token is held
    void start_web(ai_info&); // starts (or resumes) polling this chat's Doc and posts the connection prompt to its history
    ai_info* selected_info() const; // ai_info bound to the sidebar's current chat, or null if none is selected
    bool web_connected(const ai_info&) const; // true iff this chat's Doc is the one being polled
    enum class send_action {Disabled,Send,Stop,Resume}; // local agents use persistent stdin/stdout processes; send-vs-queue is only internal startup timing
    send_action current_send_action() const; // single source of truth for what the Send button means right now, including whether it's clickable at all -- update_send_button() only turns this into a label/enabled state, on_ai_send_message_clicked() only executes it
    void update_send_button(); // reflects Send / Stop / Resume / disabled, purely from current_send_action() and whether a chat is selected

    void add_ai_history(ai_info&,const QString&,const QString&);
    bool run_agent_login(const QString& provider);
    void set_ai_status(const QString&,session_status,QString); // always updates/logs the session; updates the bottom label only when this chat is selected
    void update_ai_status(const ai_info&,bool = false); // presentation only; pulse toggles a running status dot on a real status update
    void show_ai_project(ai_info&,QJsonObject = {}); // sidebar row: create/update it, blink if the update is for a background chat, select it if nothing else was selected -- renders the chat transcript itself only when this chat is the one currently selected
    void update_agent_models(const QString&,const QStringList&,bool);
    void refresh_agent_executables(); // re-runs local-agent executable discovery into agent_entries[agent].executable
    void refresh_agent_status(const QString& provider = {});
    void refresh_agent_models(const QString& provider); // one model-list probe per local agent
    void refresh_ollama_models();

public:
    void web_step(const QString& step,const QString& approval = {},qint64 pid = 0){web.report_step(step,approval,pid);} // see web_bridge::report_step
    void web_title(const ai_info& info,const QString& title){if(web_connected(info)) web.set_web_title(title);} // set_title also names this chat's Doc
    explicit AIAgent(MainWindow*);
    ~AIAgent();
    void ai_request(const QByteArray& request,QByteArray& reply); // entry point for the local-socket AI protocol: resolves/creates the session, hands everything to MainWindow's CMD command center, then refreshes the sidebar
    void update_current_window(QWidget*); // called where open_fib/open_src/open_image create their window; no-op unless an AI dispatch is in progress
    bool approve(const QString& title,const QString& request); // the local user's confirmation of an AI file write, shell command, or upload
    QJsonObject web_file(const ai_info& info,const std::vector<std::string>& cmd,QString& error); // web_upload, web_download_prepare, web_download for this chat's Web session
    QJsonArray agent_settings(const QStringList& providers) const; // cached readiness for get_agent_settings; never probes
    bool start_ai(ai_info&,const QString& text,const QString& work_dir,bool interactive); // the Send button (interactive) and send_chat: true when the message was recorded and sent or the launch began

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
};

#endif
