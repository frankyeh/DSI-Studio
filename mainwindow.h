#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>
#include <memory>
#include <QListWidgetItem>
#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonArray>
#include <QJsonObject>
#include <QMenu>
#include <QMap>
#include <string>
#include <unordered_map>


namespace Ui {
    class MainWindow;
}
class group_connectometry_analysis;
class FiberDataHub;
class QProcess;
struct ai_launch;

enum class ai_provider {Unknown = -1,Codex = 0,Claude = 1};
enum class ai_model_provider {Native,Ollama};
enum class ai_input {User,Pending};
struct ai_info{
    QString agent_name,work_dirs,project_titles;
    ai_provider provider = ai_provider::Unknown;
    QProcess* processes = nullptr;
    QJsonArray projects,prompts;
    QListWidgetItem* project_items = nullptr;
    QJsonObject model_settings;
    static ai_provider identify_provider(const QString&);
    QString title(const QString& session) const {return project_titles.isEmpty() ? (agent_name.isEmpty() ? session : agent_name+"@"+session) : project_titles;} QString details(const QString&) const;
    void update(const QString&,const QString&); void set_provider(ai_provider,const QString&); void set_process(QProcess*);
};
extern std::unordered_map<QString,ai_info> ai_infos;
extern QMap<QString,quint64> ai_log_positions;
void ai_log(QString);

class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum { MaxRecentFiles = 50 };
    void updateRecentList(void);
    QSettings settings;

public:
    QString ai_project_dir;
    QMenu* ai_project_menu = nullptr;
    void add_ai_history(const QString&,QJsonObject);
    void add_ai_history(const QString&,const QString&,const QString&);
    bool save_ai_entry(const QString&,const QJsonObject&);
    bool set_ai_title(const QString&,QString);
    void show_ai_project(const QString&);
    void show_ai_project(const QString&,QJsonObject);
    void stop_ai_blink();
    void refresh_ollama_models();
    void refresh_codex_models(const QString&);
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
    bool command(const std::vector<std::string>& cmd);
public:
    void open_DWI(QStringList files);
    bool loadFib(QString Filename);
    void loadNii(QStringList Filename);
    void loadSrc(QStringList filenames);
    void open_template(QString name);
    void add_work_dir(QString dir);
    bool load_db(std::shared_ptr<group_connectometry_analysis>& database,QString& file_name);
    QString fiber_data_hub_url(void) const { return info.value(4); }
    QString work_dir(void) const;
private:
    QStringList info;
    FiberDataHub* fiber_data_hub = nullptr;
    void login(void);
    void start_ai(QString,const QString&,ai_input);
    void start_codex(QString,const QString&,ai_input);
    void start_claude(QString,const QString&,ai_input);
    ai_launch prepare_ai(ai_provider,QString,const QString&,ai_input);
    void run_ai(const ai_launch&,QStringList);
private slots:

    void on_ai_quick_settings_clicked();
    void on_ai_new_chat_clicked();
    void on_ai_send_message_clicked();
    void on_averagefib_clicked();
    void on_vbc_clicked();
    void on_RenameDICOMDir_clicked();
    void on_RenameDICOM_clicked();
    void openRecentFibFile();
    void openRecentSrcFile();
    void open_fib_at(int,int);
    void open_src_at(int,int);
    void on_batch_reconstruction_clicked();
    void on_workDir_currentTextChanged(const QString &arg1);

    void on_linear_reg_clicked();
    void on_SRC_qc_clicked();
    void on_parse_network_measures_clicked();
    void on_nii2src_bids_clicked();
    void on_nii2src_sf_clicked();
    void on_dicom2nii_clicked();
    void on_styles_activated(int index);
    void on_clear_settings_clicked();
    void on_recentFib_cellClicked(int row, int column);
    void on_open_selected_fib_clicked();
    void on_template_list_itemDoubleClicked(QListWidgetItem *item);
    void on_open_selected_src_clicked();
    void on_recentSrc_cellClicked(int row, int column);
    void on_OpenDWI_NIFTI_clicked();
    void on_OpenDWI_DICOM_clicked();
    void on_OpenDWI_2dseq_clicked();
    void on_NII_qc_clicked();
    void on_FIB_qc_clicked();
};

#endif // MAINWINDOW_H
