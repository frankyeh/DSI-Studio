#ifndef FIBER_DATA_HUB_HPP
#define FIBER_DATA_HUB_HPP

#include <QJsonArray>
#include <QMainWindow>
#include <QSettings>
#include <QSharedPointer>
#include <map>
#include <string>
#include <vector>

class MainWindow;
class QNetworkReply;
class QUrl;

namespace Ui {
class FiberDataHub;
}

class FiberDataHub : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    Ui::FiberDataHub* ui;
    QSettings settings;
    std::vector<QString> github_tsv_link;
    int github_api_rate_limit = 60;
    QString cur_tag;
    bool fetch_github = false;
    std::map<QString,QString> notes,dates;
    std::map<QString,QJsonArray> tags,assets;
    void initialize(void);
    void loadTags(QUrl,QString,QJsonArray,int);
    void loadFiles(void);
    void update_rate_limit(QSharedPointer<QNetworkReply>);

public:
    explicit FiberDataHub(MainWindow* parent);
    ~FiberDataHub();
    std::string error_msg;
    bool command(const std::vector<std::string>& cmd);

private slots:
    void on_load_tags_clicked();
    void on_github_tags_itemSelectionChanged();
    void on_browseDownloadDir_clicked();
    void on_github_release_files_itemSelectionChanged();
    void on_github_select_all_clicked();
    void on_github_download_clicked();
    void on_github_select_matching_clicked();
    void on_github_release_note_currentChanged(int index);
    void on_github_repo_currentIndexChanged(int index);
    void on_github_open_file_clicked();
};

#endif
