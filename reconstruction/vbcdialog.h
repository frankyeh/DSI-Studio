#ifndef VBCDIALOG_H
#define VBCDIALOG_H

#include <QDialog>
#include <QTimer>
#include <memory>


namespace Ui {
    class VBCDialog;
}

class vbc;
class VBCDialog : public QDialog
{
    Q_OBJECT

public:
    QString work_dir;
    QStringList group1,group2;
    bool data_loaded;
    explicit VBCDialog(QWidget *parent,QString workDir);
    ~VBCDialog();
    std::auto_ptr<vbc> vbc_instance;
private:
    Ui::VBCDialog *ui;
    void update_list(void);
    void load_data(void);
private:
    std::auto_ptr<QTimer> timer;
private slots:
    void show_distribution();
    void on_load_subject_data_clicked();
    void on_save_list2_clicked();
    void on_open_list2_clicked();
    void on_save_list1_clicked();
    void on_open_list1_clicked();
    void on_movedown_clicked();
    void on_moveup_clicked();
    void on_group2delete_clicked();
    void on_group1delete_clicked();
    void on_group2open_clicked();
    void on_group1open_clicked();
    void on_vbc_group_toggled(bool checked);
    void on_vbc_single_toggled(bool checked);
    void on_vbc_trend_toggled(bool checked);
    void on_close_clicked();
    void on_open_dir1_clicked();
    void on_open_dir2_clicked();
    void on_open_template_clicked();
    void on_save_mapping_clicked();
    void on_run_null_clicked();
};

#endif // VBCDIALOG_H
