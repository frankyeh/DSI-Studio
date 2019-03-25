#ifndef NN_CONNECTOMETRY_H
#define NN_CONNECTOMETRY_H

#include <QDialog>
#include <QGraphicsScene>
#include <QtCharts/QtCharts>
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
private:
    QChart* chart1;
    QChartView* chart1_view;
    QChart* chart2;
    QChartView* chart2_view;
    QChart* chart3;
    QChartView* chart3_view;
    QChart* chart4;
    QChartView* chart4_view;

    int cur_fold = 0;
    QLineSeries *s2 = 0;
    QLineSeries *s3 = 0;
    QLineSeries *s4 = 0;

public:
    bool gui = true;
    QString work_dir;
    nn_connectometry_analysis nna;

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

    void on_nn_regression_toggled(bool checked);

private:
    Ui::nn_connectometry *ui;
};

#endif // NN_CONNECTOMETRY_H
