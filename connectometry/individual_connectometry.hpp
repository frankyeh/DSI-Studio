#ifndef INDIVIDUAL_CONNECTOMETRY_HPP
#define INDIVIDUAL_CONNECTOMETRY_HPP

#include <QDialog>
namespace Ui {
class individual_connectometry;
}
class individual_connectometry : public QDialog
{
    Q_OBJECT

public:
    explicit individual_connectometry(QWidget *parent);
    ~individual_connectometry();

private slots:


    void on_open_subject2_clicked();

    void on_Close_clicked();

    void on_compare_clicked();

    void on_load_baseline_clicked();

    void on_pushButton_clicked();

    void on_open_template_clicked();

private:
    Ui::individual_connectometry *ui;
};

#endif // INDIVIDUAL_CONNECTOMETRY_HPP
