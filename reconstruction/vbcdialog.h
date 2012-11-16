#ifndef VBCDIALOG_H
#define VBCDIALOG_H

#include <QDialog>
#include <memory>


namespace Ui {
    class VBCDialog;
}

class vbc_database;
class VBCDialog : public QDialog
{
    Q_OBJECT

public:
    QString work_dir;
    QStringList group;
    bool data_loaded;
    explicit VBCDialog(QWidget *parent,QString file_name,vbc_database* data_);
    ~VBCDialog();
    std::auto_ptr<vbc_database> data;
private:
    Ui::VBCDialog *ui;
    void update_list(void);
    void load_data(void);

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
};

#endif // VBCDIALOG_H
