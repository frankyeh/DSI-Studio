#ifndef TRACT_REPORT_HPP
#define TRACT_REPORT_HPP

#include <QDialog>
class tracking_window;
namespace Ui {
class tract_report;
}

class tract_report : public QDialog
{
    Q_OBJECT
    
public:
    tracking_window* cur_tracking_window;
    explicit tract_report(QWidget *parent = 0);
    ~tract_report();
    void copyToClipboard(void);
public slots:
    void on_refresh_report_clicked();

    void on_save_report_clicked();


    void on_save_image_clicked();

private slots:
    void on_max_y_valueChanged(double arg1);

    void on_min_y_valueChanged(double arg1);

private:
    Ui::tract_report *ui;
};

#endif // TRACT_REPORT_HPP
