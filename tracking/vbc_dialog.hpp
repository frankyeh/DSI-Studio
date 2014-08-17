#ifndef VBC_DIALOG_HPP
#define VBC_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QTimer>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
namespace Ui {
class vbc_dialog;
}

class tracking_window;
class FibData;
class vbc_dialog : public QDialog
{
    Q_OBJECT
private:
    QGraphicsScene vbc_scene;
    QImage vbc_slice_image;
    unsigned int vbc_slice_pos;
    fib_data cur_subject_fib;
    void show_dis_table(void);
public:
    QString work_dir;
    std::vector<std::string> file_names;
public:
    std::auto_ptr<vbc_database> vbc;
    stat_model mr;
    std::vector<std::vector<float> > individual_data;
    std::auto_ptr<QTimer> timer;

    explicit vbc_dialog(QWidget *parent,vbc_database* vbc_ptr,QString work_dir_);
    ~vbc_dialog();
    bool eventFilter(QObject *obj, QEvent *event);
private slots:


    void on_subject_list_itemSelectionChanged();

    void on_save_vbc_dist_clicked();

    void show_report();

    void show_fdr_report();

    void on_save_fdr_dist_clicked();

    void on_open_files_clicked();

    void on_open_mr_files_clicked();

    void on_view_mr_result_clicked();

    void on_rb_individual_analysis_clicked();

    void on_rb_group_difference_clicked();

    void on_rb_multiple_regression_clicked();

    void on_rb_paired_difference_clicked();

    void on_run_clicked();

public slots:
    void calculate_FDR(void);
private:
    Ui::vbc_dialog *ui;
};

#endif // VBC_DIALOG_HPP
