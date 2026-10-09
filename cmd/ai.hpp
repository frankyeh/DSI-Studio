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
class QProcess;
class QSettings;
class QWidget;

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
    // the latest local launch attempt, set by prepare_ai() and read by configure_*()
    QString launch_name,launch_model;
    QUrl launch_model_url;
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
ai_info* assign_ai_session(const QString& from,const QString& to); // renames an existing session's key/files/title in place (e.g. Codex's placeholder id -> its real thread_id); a no-op lookup if from == to; nullptr (nothing changed) if to already exists or from is missing
QUrl agent_install_url(const QString& provider); // shared by the sidebar's Install button and a launch that finds the CLI missing, so the two can't drift apart
void stop_blink(QWidget* row); // stops a sidebar row's attention-getting blink animation and clears its stylesheet
void update_status_dot(QLabel* dot,session_status status,bool pulse); // presentational: sets a sidebar/status dot's color and pulse animation for the given status
QString ai_dialog_style(); // shared stylesheet for the new-chat/settings dialogs
QByteArray json_line(const QJsonObject&); // one compact JSON message per line, the framing every agent protocol and the history file use
QPair<QUrl,bool> ai_ollama_url(const QSettings& settings); // ("ai/ollama_host"+"ai/ollama_port" as a URL, whether a host is actually configured) -- the bool distinguishes "empty/default" from "genuinely set to something that parses to the same URL"
QString model_combo_key(const QComboBox& model); // strips the " (Ollama@host)" suffix off an Ollama model's display text; "default" is a UI label only -- its data value is empty, the one universal representation of "no explicit choice"
void set_model_selector(QComboBox& model,const QJsonObject& profiles,
                        QString selected = {},QString fallback = {},
                        QJsonObject selected_info = {}); // populates model with "default"+profiles' native/Ollama entries, grouped and sorted, selecting selected (or falling back to fallback) -- selected_info backs an unrecognized selected value so it still shows up as a real entry

#endif
