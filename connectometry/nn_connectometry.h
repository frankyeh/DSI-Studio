#ifndef NN_CONNECTOMETRY_H
#define NN_CONNECTOMETRY_H

#include <QDialog>
#include <QGraphicsScene>
#include "group_connectometry_analysis.h"
#include "nn_connectometry_analysis.h"
#include <tipl/tipl.hpp>
#include <QTimer>

namespace Ui {
class nn_connectometry;
}

class nn_connectometry : public QDialog
{
    Q_OBJECT

public:
    bool gui = true;
    QString work_dir;
    nn_connectometry_analysis nna;
    std::ostringstream out;

public:
    QTimer* timer = 0;
    QString log_text;
public:
    QGraphicsScene network_scene,layer_scene;
    QImage network_I,layer_I;

    explicit nn_connectometry(QWidget *parent,std::shared_ptr<fib_data> handle,QString db_file_name_,bool gui_);

    ~nn_connectometry();

private slots:
    void on_open_mr_files_clicked();

    void on_run_clicked();

    void on_stop_clicked();

    void update_network(void);

    void on_view_tab_currentChanged(int index);

    void on_reset_clicked();

    void on_foi_currentIndexChanged(int index);

    void on_regress_all_clicked();

private:
    Ui::nn_connectometry *ui;
};

#endif // NN_CONNECTOMETRY_H
