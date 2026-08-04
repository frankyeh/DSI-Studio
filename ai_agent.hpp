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

class MainWindow;
class QListWidgetItem;
class QMenu;
class QNetworkRequest;
class QProcess;
class QShowEvent;
class QCloseEvent;

namespace Ui {
class AIAgent;
}

enum class ai_provider {Unknown = -1,Codex = 0,Claude = 1,ChatGPT = 2}; // ChatGPT: web agent via GitHub issue, not a local CLI; never indexes AIAgent::agent_entries (sized for Codex/Claude only)
enum class ai_input {User,Pending};

struct ai_info{
    QString sessions,agent_name,project_titles;
    ai_provider provider = ai_provider::Unknown;
    QProcess* processes = nullptr;
    QList<QJsonObject> projects;
    QStringList prompts;
    QListWidgetItem* project_items = nullptr;
    QJsonObject model_settings;
    quint64 log_position = quint64(-1);
    QString current_window = "main"; // persists across requests until changed by "set_window"
    bool has_error = false; // true once a run fails, cleared on the next run; sidebar dot: green (running), red (has_error), gray (otherwise)
    static ai_provider identify_provider(const QString&);
    static ai_info* find(const QString&);
    static ai_info* create(QString,QString);
    static QString history_file(const QString&);
    bool save_title(QString);
    QJsonObject record_history(QJsonObject); // returns the recorded entry (with "time" and, for the first entry, "agent"/"model_settings" filled in), not the caller's pre-call copy
    QJsonObject record_reply(const QString&,const QString&); // returns the recorded entry so callers can pass it on to show_ai_project() for blink/visibility handling
    QString title() const {return project_titles.isEmpty() ? (agent_name.isEmpty() ? sessions : agent_name+"@"+sessions) : project_titles;}
    QString details() const;
};

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
    Ui::AIAgent* ui;
    QSettings settings;
    QString ai_status_activity;
    QMenu* ai_project_menu = nullptr;
    QTimer* ai_status_timer = nullptr;
    int active_ai_processes = 0;

    // current agent/model selection, shown via update_agent_status_label() instead of the visible combo boxes this used to be
    std::array<ai_agent_entry,2> agent_entries; // indexed by ai_provider
    int current_agent_index = 0;
    QString current_model_name = "default";
    QJsonObject current_model_info;
    void update_agent_status_label();
    void try_set_current_model(const QString& name); // no-op if name is unknown, matching non-editable QComboBox::setCurrentText

    // GitHub issue channel: the issue body carries the next request; one pinned comment (marked "dsi_session_result":true) carries the result
    QNetworkAccessManager github_manager;
    QTimer github_timer;
    QUrl github_issue_api,github_result_api;
    QByteArray github_etag;
    QString github_token; // snapshot taken at connect time so a mid-session Settings change cannot swap the identity underneath a poll
    qint64 github_last_id = 0;
    QJsonObject github_pending_result; // staged until its PATCH is confirmed; retried, never re-executed
    quint64 github_connection_id = 0; // bumped on connect/disconnect; rejects callbacks from a superseded connection even to the same URL
    bool web_agent_active_session = false; // true from New Chat (web agent) until New Chat starts a local one
    QString github_last_issue_url; // remembered so Resume can default to it
    QString web_agent_session_id; // the actual chat this GitHub connection belongs to, independent of sidebar selection; survives Stop/Resume, cleared only on a fresh (non-resume) start

    QNetworkRequest github_request(const QUrl&) const;
    bool connect_github_issue(const QString&,QString& error);
    void disconnect_github_issue();
    void poll_github_issue();
    void publish_github_result(QJsonObject);
    void send_pending_result();
    void update_send_button(); // reflects Send / Stop / Resume depending on web_agent_active_session
    bool is_status_target(const QString& session) const; // true if session is the currently selected chat (or, if none is, the still-anonymous chat being set up) -- gates set_ai_status() calls from a background process so a chat the user isn't looking at can't hijack the status label
    bool try_connect_github_issue(const QString& url,bool resume); // connect_github_issue() plus the shared success/failure UI feedback
    void new_chat_dialog(bool resume); // shared by New Chat and Resume; resume locks the mode and disables the local agent/model panel
    bool run_new_chat_dialog(bool resume,const QString& title,const QString& accept_text,
                              bool& web,int& agent_index,QString& model_name,QString& issue_url);
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
    QStringList configure_codex(const ai_launch&,QString,const QString&);
    QStringList configure_claude(const ai_launch&,QString,const QString&,bool);
    ai_launch prepare_ai(ai_provider,QString&,const QString&,ai_input);

public:
    explicit AIAgent(MainWindow*);
    ~AIAgent();
    void ai_request(const QByteArray& request,QByteArray& reply); // entry point for the local-socket AI protocol: resolves/creates the session, hands everything to MainWindow's CMD command center, then refreshes the sidebar
    void refresh_ai_info(ai_info& info)
    {show_ai_project(info);set_ai_status("Agent request completed.",true);}

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
