#ifndef AI_AGENT_HPP
#define AI_AGENT_HPP

#include <QByteArray>
#include <QJsonArray>
#include <QJsonObject>
#include <QMainWindow>
#include <QMap>
#include <QSettings>
#include <QStringList>

class MainWindow;
class QListWidgetItem;
class QMenu;
class QProcess;
class QShowEvent;
class QTimer;

namespace Ui {
class AIAgent;
}

enum class ai_provider {Unknown = -1,Codex = 0,Claude = 1};
enum class ai_model_provider {Native,Ollama};
enum class ai_input {User,Pending};

struct ai_info{
    QString sessions,agent_name,project_titles;
    ai_provider provider = ai_provider::Unknown;
    QProcess* processes = nullptr;
    QJsonArray projects;
    QStringList prompts;
    QListWidgetItem* project_items = nullptr;
    QJsonObject model_settings;
    static ai_provider identify_provider(const QString&);
    static ai_info* find(const QString&);
    static ai_info* create(QString,QString);
    static QString history_file(const QString&);
    static bool save_title(ai_info&,QString);
    static void record_history(ai_info&,QJsonObject);
    static bool save_history(const ai_info&);
    QString title() const {return project_titles.isEmpty() ? (agent_name.isEmpty() ? sessions : agent_name+"@"+sessions) : project_titles;}
    QString details() const;
};
void ai_command(ai_info&,const QByteArray&,QByteArray&);

struct ai_launch;
class AIAgent : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    Ui::AIAgent* ui;
    QSettings settings;
    QString ai_status_activity,ai_status_waiting;
    QMenu* ai_project_menu = nullptr;
    QTimer* ai_status_timer = nullptr;
    int active_ai_processes = 0,ai_status_delay = 0,ai_status_dots = 0;

    void add_ai_history(ai_info&,const QString&,const QString&);
    void add_ai_reply(ai_info&,const QString&,const QString&);
    void set_ai_status(QString = {},bool = false);
    void show_ai_project(ai_info&,QJsonObject = {});
    void stop_ai_blink();
    void refresh_ollama_models();
    void refresh_codex_models(const QString&);
    void start_ai(QString,const QString&,ai_input);
    void start_codex(QString,const QString&,ai_input);
    void start_claude(QString,const QString&,ai_input);
    ai_launch prepare_ai(ai_provider,QString,const QString&,ai_input);
    void run_ai(const ai_launch&,QStringList);

public:
    explicit AIAgent(MainWindow*);
    ~AIAgent();
    void refresh_ai_info(ai_info&);

protected:
    void showEvent(QShowEvent*) override;

private slots:
    void on_ai_quick_settings_clicked();
    void on_ai_new_chat_clicked();
    void on_ai_send_message_clicked();
};

#endif
