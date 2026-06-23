#ifndef CREATEDB_H
#define CREATEDB_H

#include <QDialog>
#include <memory>
class fib_data;
namespace Ui {
    class CreateDBDialog;
}

class CreateDBDialog : public QDialog
{
    Q_OBJECT
private:
    bool create_db;
    QString sample_fib;
public:
    QStringList group;
    unsigned int dir_length;
    float template_reso = 1.0f;
    unsigned int template_id = 0;
    explicit CreateDBDialog(QWidget *parent,bool create_db_);
    ~CreateDBDialog();
private:
    Ui::CreateDBDialog *ui;
    void update_list(void);
    void load_data(void);
    void update_output_file_name(void);
private slots:
    void on_save_list1_clicked();
    void on_open_list1_clicked();
    void on_movedown_clicked();
    void on_moveup_clicked();
    void on_group1delete_clicked();
    void on_group1open_clicked();
    void on_close_clicked();
    void on_open_dir1_clicked();
    void on_select_output_file_clicked();
    void on_create_data_base_clicked();
    void on_sort_clicked();

};

#endif // VBCDIALOG_H
