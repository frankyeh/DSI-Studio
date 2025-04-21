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

namespace Ui {
    class MainWindow;
}
class group_connectometry_analysis;
class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum { MaxRecentFiles = 50 };
    void updateRecentList(void);
    QSettings settings;
    std::map<QString,QString> notes;
public:
    std::vector<QString> github_tsv_link;
    int github_api_rate_limit = 60;
    void update_rate_limit(QSharedPointer<QNetworkReply>);
public:
    bool fetch_github = false;
    QNetworkAccessManager manager;
    std::map<QString,QJsonArray> tags,assets;
    QSharedPointer<QNetworkReply> get(QUrl url);
    QString fnValue,adrValue;
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
public:
    void open_DWI(QStringList files);
    void batch_create_src(const std::vector<std::string>& dwi_nii_files,const std::string& output_dir);
    void loadFib(QString Filename);
    void loadNii(QStringList Filename);
    void loadSrc(QStringList filenames);
    void open_template(QString name);
    void add_work_dir(QString dir);
    bool load_db(std::shared_ptr<group_connectometry_analysis>& database,QString& file_name);
    void loadTags(QUrl url,QString repo,QJsonArray array,int per_page);
    void loadFiles(void);
    void login_with_param(QStringList param);
private slots:
    void login();
    void on_averagefib_clicked();
    void on_vbc_clicked();
    void on_RenameDICOMDir_clicked();
    void on_browseDir_clicked();
    void on_FiberTracking_clicked();
    void on_Reconstruction_clicked();
    void on_RenameDICOM_clicked();
    void openRecentFibFile();
    void openRecentSrcFile();
    void open_fib_at(int,int);
    void open_src_at(int,int);
    void on_batch_reconstruction_clicked();
    void on_view_image_clicked();
    void on_workDir_currentTextChanged(const QString &arg1);


    void on_open_db_clicked();
    void on_group_connectometry_clicked();

    void on_linear_reg_clicked();
    void on_nonlinear_reg_clicked();
    void on_SRC_qc_clicked();
    void on_parse_network_measures_clicked();
    void on_auto_track_clicked();
    void on_nii2src_bids_clicked();
    void on_nii2src_sf_clicked();
    void on_dicom2nii_clicked();
    void on_clear_src_history_clicked();
    void on_clear_fib_history_clicked();
    void on_xnat_download_clicked();
    void on_styles_activated(int index);
    void on_clear_settings_clicked();
    void on_console_clicked();
    void on_T1WFiberTracking_clicked();
    void on_TemplateFiberTracking_clicked();
    void on_recentFib_cellClicked(int row, int column);
    void on_open_selected_fib_clicked();
    void on_template_list_itemDoubleClicked(QListWidgetItem *item);
    void on_open_selected_src_clicked();
    void on_recentSrc_cellClicked(int row, int column);
    void on_OpenDWI_NIFTI_clicked();
    void on_OpenDWI_DICOM_clicked();
    void on_OpenDWI_2dseq_clicked();
    void on_load_tags_clicked();
    void on_github_tags_itemSelectionChanged();
    void on_tabWidget_currentChanged(int index);
    void on_browseDownloadDir_clicked();
    void on_github_release_files_itemSelectionChanged();
    void on_github_select_all_clicked();
    void on_github_download_clicked();
    void on_github_select_matching_clicked();
    void on_github_release_note_currentChanged(int index);
    void on_github_repo_currentIndexChanged(int index);
    void on_github_open_file_clicked();
    void on_NII_qc_clicked();
    void on_FIB_qc_clicked();
};

#endif // MAINWINDOW_H
