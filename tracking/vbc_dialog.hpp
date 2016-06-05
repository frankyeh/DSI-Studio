#ifndef VBC_DIALOG_HPP
#define VBC_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QItemDelegate>
#include <QTimer>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
#include "atlas.hpp"
namespace Ui {
class vbc_dialog;
}


class ROIViewDelegate : public QItemDelegate
 {
     Q_OBJECT

 public:
    ROIViewDelegate(QObject *parent)
         : QItemDelegate(parent)
     {
     }

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                                const QModelIndex &index) const;
          void setEditorData(QWidget *editor, const QModelIndex &index) const;
          void setModelData(QWidget *editor, QAbstractItemModel *model,
                            const QModelIndex &index) const;
private slots:
    void emitCommitData();
 };

class tracking_window;
class fib_data;
class vbc_dialog : public QDialog
{
    Q_OBJECT
private:
    QGraphicsScene fp_dif_scene, fp_scene;
    image::color_image fp_dif_map;
    QImage fp_dif_image,fp_image;
    image::color_image fp_image_buf;
    image::color_map_rgb color_map;
    image::color_bar color_bar;
    std::vector<float> fp_matrix;
    float fp_max_value;
    image::basic_image<char,3> fp_mask;
private:
    QGraphicsScene vbc_scene;
    QImage vbc_slice_image;
    unsigned int vbc_slice_pos;
    std::auto_ptr<connectometry_result> result_fib;
    void show_dis_table(void);
    void add_new_roi(QString name,QString source,std::vector<image::vector<3,short> >& new_roi);
public:
    bool gui;
    QString work_dir,db_file_name;
    std::vector<std::string> file_names,saved_file_name;
public:
    std::vector<std::vector<image::vector<3,short> > > roi_list;
public:
    std::auto_ptr<vbc_database> vbc;
    std::auto_ptr<stat_model> model;
    std::vector<std::vector<float> > individual_data;
    std::auto_ptr<QTimer> timer;
    QString report;
    void setup_model(stat_model& model);

    explicit vbc_dialog(QWidget *parent,vbc_database* vbc_ptr,QString db_file_name_,bool gui_);
    ~vbc_dialog();
    bool eventFilter(QObject *obj, QEvent *event);
    void update_subject_list();
public:
    bool load_demographic_file(QString filename);
public slots:


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

    void on_remove_subject_clicked();

    void on_toolBox_currentChanged(int index);

    void on_x_pos_valueChanged(int arg1);

    void on_y_pos_valueChanged(int arg1);

    void on_z_pos_valueChanged(int arg1);

    void on_scatter_valueChanged(int arg1);

    void on_save_report_clicked();

    void on_save_R2_clicked();

    void on_save_vector_clicked();

    void on_show_advanced_clicked();

    void on_missing_data_checked_toggled(bool checked);


    void on_suggest_threshold_clicked();
public slots:
    void calculate_FDR(void);
public:
    Ui::vbc_dialog *ui;
private slots:
    void on_load_roi_from_atlas_clicked();
    void on_clear_all_roi_clicked();
    void on_load_roi_from_file_clicked();
    void on_calculate_dif_clicked();
    void on_fp_zoom_valueChanged(double arg1);
    void on_subject_view_tabBarClicked(int index);
    void on_save_dif_clicked();
    void on_add_db_clicked();
    void on_view_x_toggled(bool checked);
    void on_load_fp_mask_clicked();
    void on_save_fp_mask_clicked();
    void on_rb_percentage_clicked();
    void on_rb_percentile_clicked();
    void on_rb_t_stat_clicked();
    void on_rb_beta_clicked();
    void on_rb_mean_dif_clicked();
};

#endif // VBC_DIALOG_HPP
