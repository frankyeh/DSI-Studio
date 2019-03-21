#ifndef TRACT_REPORT_HPP
#define TRACT_REPORT_HPP

#include <QDialog>
#include <QtCharts/QtCharts>
class tracking_window;
namespace Ui {
class tract_report;
}

class tract_report : public QDialog
{
    Q_OBJECT
private:
    QChart* report_chart;
    QChartView* report_chart_view;
public:
    tracking_window* cur_tracking_window;
    explicit tract_report(QWidget *parent = 0);
    ~tract_report();
public slots:
    void on_refresh_report_clicked();

    void on_save_report_clicked();


    void on_save_image_clicked();

private slots:

private:
    Ui::tract_report *ui;
};

#endif // TRACT_REPORT_HPP
