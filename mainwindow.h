#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum { MaxRecentFiles = 50 };
    void updateRecentList(void);
    QSettings settings;
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    Ui::MainWindow *ui;
    void addFib(QString Filename);
    void addSrc(QString Filename);
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);
private:
    void loadFib(QString Filename);
    void loadSrc(QStringList filenames);
    void add_work_dir(QString dir);
private slots:
    void on_averagefib_clicked();
    void on_vbc_clicked();
    void on_RenameDICOMDir_clicked();
    void on_simulateMRI_clicked();
    void on_browseDir_clicked();
    void on_FiberTracking_clicked();
    void on_Reconstruction_clicked();
    void on_OpenDICOM_clicked();
    void on_RenameDICOM_clicked();
    void openRecentFibFile();
    void openRecentSrcFile();
    void open_fib_at(int,int);
    void open_src_at(int,int);
    void on_batch_src_clicked();
    void on_batch_reconstruction_clicked();
    void on_view_image_clicked();
    void on_connectometry_clicked();
    void on_workDir_currentTextChanged(const QString &arg1);
    void on_bruker_browser_clicked();
};

#endif // MAINWINDOW_H
