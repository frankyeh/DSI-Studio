#ifndef VBC_DIALOG_HPP
#define VBC_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QTimer>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
#include "atlas.hpp"
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
    std::auto_ptr<fib_data> result_fib;
    void show_dis_table(void);
public:
    QString work_dir;
    std::vector<std::string> file_names,saved_file_name;
public:
    std::auto_ptr<vbc_database> vbc;
    std::auto_ptr<stat_model> model;
    std::vector<std::vector<float> > individual_data;
    std::auto_ptr<QTimer> timer;
    QString report;
    atlas study_region;
    QString study_region_file_name;
    explicit vbc_dialog(QWidget *parent,vbc_database* vbc_ptr,QString work_dir_);
    ~vbc_dialog();
    bool eventFilter(QObject *obj, QEvent *event);
private slots:


    void on_subject_list_itemSelectionChanged();

    void show_report();

    void show_fdr_report();

    void on_open_files_clicked();

    void on_open_mr_files_clicked();

    void on_rb_individual_analysis_clicked();

    void on_rb_group_difference_clicked();

    void on_rb_multiple_regression_clicked();

    void on_rb_paired_difference_clicked();

    void on_run_clicked();

    void on_save_name_list_clicked();

    void on_show_result_clicked();

    void on_roi_whole_brain_toggled(bool checked);

    void on_roi_file_toggled(bool checked);

    void on_roi_atlas_toggled(bool checked);

    void on_atlas_box_currentIndexChanged(int index);

    void on_remove_subject_clicked();

    void on_remove_sel_subject_clicked();

    void on_toolBox_currentChanged(int index);

    void on_x_pos_valueChanged(int arg1);

    void on_y_pos_valueChanged(int arg1);

    void on_z_pos_valueChanged(int arg1);

    void on_scatter_valueChanged(int arg1);

    void on_save_report_clicked();

    void on_remove_subject2_clicked();

    void on_save_R2_clicked();

public slots:
    void calculate_FDR(void);
    void setup_threshold(void);
private:
    Ui::vbc_dialog *ui;
};

#endif // VBC_DIALOG_HPP
