#ifndef AUTO_TRACK_H
#define AUTO_TRACK_H
#include <future>
#include <QMainWindow>
#include <QProgressBar>
#include <QTimer>
#include "fib_data.hpp"
namespace Ui {
class auto_track;
}

class auto_track : public QMainWindow
{
    Q_OBJECT

public:
    explicit auto_track(QWidget *parent = nullptr);
    ~auto_track();
public:
    void update_list(void);
public:
    std::shared_ptr<std::future<void> > thread;
    int prog = 0;
    QStringList file_list;
    std::shared_ptr<QTimer> timer;
private slots:
    void on_open_clicked();

    void on_delete_2_clicked();

    void on_open_dir_clicked();

    void check_status();

    void on_delete_all_clicked();

    void on_run_clicked();

    void on_tractography_atlas_currentIndexChanged(int index);

private:
    Ui::auto_track *ui;
    QProgressBar* progress_bar;
};

#endif // AUTO_TRACK_H
