#ifndef INDIVIDUAL_CONNECTOMETRY_HPP
#define INDIVIDUAL_CONNECTOMETRY_HPP

#include <QDialog>

namespace Ui {
class individual_connectometry;
}
class tracking_window;
class individual_connectometry : public QDialog
{
    Q_OBJECT
    QString subject_file,subject2_file;
    tracking_window& cur_tracking_window;
public:
    explicit individual_connectometry(QWidget *parent,tracking_window& cur_tracking_window);
    ~individual_connectometry();

private slots:

    void on_load_subject_clicked();

    void on_open_template_clicked();

    void on_open_subject2_clicked();

    void on_inv_inv_toggled(bool checked);


    void on_Close_clicked();

    void on_inv_template_toggled(bool checked);

    void on_inv_db_toggled(bool checked);

    void on_compare_clicked();

private:
    Ui::individual_connectometry *ui;
};

#endif // INDIVIDUAL_CONNECTOMETRY_HPP
