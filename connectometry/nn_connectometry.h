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
    tipl::image<int,3> fp_mask;
    std::vector<std::vector<int> > fp_mapping;
    float fp_threshold = 0.0f;
public:
    tipl::ml::trainer t;
    tipl::ml::network nn;
    tipl::ml::network_data<float,float> fp_data;
    std::vector<tipl::ml::network_data_proxy<float,float> > train_data;
    std::vector<tipl::ml::network_data_proxy<float,float> > test_data;
    QTimer* timer = 0;
    bool terminated;
    std::future<void> future;
    void init_partially_connected_layer(tipl::ml::network& nn);
public:
    std::vector<int> source_location;
    std::vector<float> selected_label;
    float selected_label_max,selected_label_min,selected_label_mean,selected_label_scale;
public:
    std::vector<unsigned int> train_seq;
    std::vector<float> train_result;
    std::vector<unsigned int> test_seq;
    std::vector<float> test_result;

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

    void on_foi_currentIndexChanged(int index);

    void on_otsu_valueChanged(double arg1);

private:
    Ui::nn_connectometry *ui;
};

#endif // NN_CONNECTOMETRY_H
