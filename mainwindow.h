#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>
#include <memory>
#include <QListWidgetItem>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <string>


namespace Ui {
    class MainWindow;
}
class group_connectometry_analysis;
class AIAgent;
struct ai_info;
class QByteArray;
class QJsonObject;
class FiberDataHub;
enum class command_source;

QString command_window_id(QWidget* window); // "main"/"trackingXXXX"/"reconXXXX"/"imageXXXX"/"connectometryXXXX", or empty if not an AI-addressable window
QString command_window_type(const QString& id); // "main"/"tracking"/"recon"/"image"/"connectometry" derived from an id string, as produced by command_window_id

class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum { MaxRecentFiles = 50 };
    void updateRecentList(void);
    void addRecent(QString,const char*);
    QSettings settings;
    QString ai_chat_context; // the current AI request's accompanying chat text, if any; set/restored around dispatch_cmd() so run_shell's confirmation dialog can explain purpose -- used only within this class

public:
    QNetworkAccessManager manager;
    QSharedPointer<QNetworkReply> get(QUrl url);
    QString username,news,host_name,address;
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent(QCloseEvent *event) override;
    Ui::MainWindow *ui;
    void addFib(QString Filename);
    void addSrc(QString Filename);
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void openFile(QStringList file_name);
    std::string error_msg;
    bool command(const std::vector<std::string>&);
    bool command(const std::vector<std::string>&,command_source);
    QJsonObject dispatch_cmd(ai_info&,const QJsonObject& request); // the AI agent's single command center: every request (set_title/log/set_window/list_window/voice/run_shell/GUI commands) runs through this command array, dispatched against a target window (main/tracking/recon/image) when one applies
    AIAgent* ai_agent; // created in the constructor; callers (e.g. the local-socket server in main.cpp) use ai_agent->ai_request(...) directly
    void report_and_target_window(QWidget*); // logs "<type> window created, id: ..." and makes it the AI session's new target; call whenever a window is created
public:
    void open_DWI(QStringList files);
    bool loadFib(QString Filename);
    void loadNii(QStringList Filename);
    bool loadSrc(QStringList filenames);
    bool open_template(QString name);
    void add_work_dir(QString dir);
    QString fiber_data_hub_url(void) const { return info.value(4); }
    QString work_dir(void) const;
private:
    QStringList info;
    FiberDataHub* fiber_data_hub = nullptr;
    void login(void);
private slots:
    void open_fib_at(int,int);
    void open_src_at(int,int);
    void on_workDir_currentTextChanged(const QString &arg1);

    void on_linear_reg_clicked();
    void on_styles_activated(int index);
    void on_recentFib_cellClicked(int row, int column);
    void on_open_selected_fib_clicked();
    void on_template_list_itemDoubleClicked(QListWidgetItem *item);
    void on_open_selected_src_clicked();
    void on_recentSrc_cellClicked(int row, int column);
};

#endif // MAINWINDOW_H
