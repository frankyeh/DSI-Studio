#ifndef VBCDIALOG_H
#define VBCDIALOG_H

#include <QDialog>
#include <memory>


namespace Ui {
    class VBCDialog;
}

class VBCDialog : public QDialog
{
    Q_OBJECT
private:
    bool create_db;
public:
    QStringList group;
    explicit VBCDialog(QWidget *parent,bool create_db_);
    ~VBCDialog();
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
    void on_open_skeleton_clicked();
    void on_sort_clicked();
};

#endif // VBCDIALOG_H
