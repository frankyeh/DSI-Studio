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
class QByteArray;
class FiberDataHub;
class QLocalSocket;

class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum { MaxRecentFiles = 50 };
    void updateRecentList(void);
    QSettings settings;

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
    void ai_command(QLocalSocket*,const QByteArray&);
public:
    void open_DWI(QStringList files);
    bool loadFib(QString Filename);
    void loadNii(QStringList Filename);
    void loadSrc(QStringList filenames);
    bool open_template(QString name);
    void add_work_dir(QString dir);
    QString fiber_data_hub_url(void) const { return info.value(4); }
    QString work_dir(void) const;
private:
    QStringList info;
    AIAgent* ai_agent = nullptr;
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
