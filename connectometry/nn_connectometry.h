#ifndef NN_CONNECTOMETRY_H
#define NN_CONNECTOMETRY_H

#include <QDialog>
#include <QGraphicsScene>
#include "vbc/vbc_database.h"
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
public:
    std::string nn_text;
    tipl::ml::trainer t;
    tipl::ml::network nn;
    tipl::ml::network_data<float,char> fp_data,fp_data_test;
    QTimer* timer = 0;
    bool terminated;
    std::future<void> future;

public:
    std::shared_ptr<vbc_database> vbc;
    std::vector<double> X;
public:
    QString log_text;
    QGraphicsScene network_scene,layer_scene;
    QImage network_I,layer_I;
    std::vector<tipl::pixel_index<3> > fp_index;

    explicit nn_connectometry(QWidget *parent,std::shared_ptr<vbc_database> vbc_ptr,QString db_file_name_,bool gui_);

    ~nn_connectometry();

private slots:
    void on_open_mr_files_clicked();

    void on_run_clicked();

    void on_stop_clicked();

    void update_network(void);

    void on_view_tab_currentChanged(int index);

    void on_reset_clicked();

private:
    Ui::nn_connectometry *ui;
};

#endif // NN_CONNECTOMETRY_H
