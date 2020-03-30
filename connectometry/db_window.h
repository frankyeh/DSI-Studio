#ifndef DB_WINDOW_H
#define DB_WINDOW_H
#include "connectometry/group_connectometry_analysis.h"
#include <QMainWindow>
#include <QGraphicsScene>

namespace Ui {
class db_window;
}

class db_window : public QMainWindow
{
    Q_OBJECT
    std::shared_ptr<group_connectometry_analysis> vbc;
private:
    QGraphicsScene fp_dif_scene, fp_scene;
    tipl::color_image fp_dif_map;
    QImage fp_dif_image,fp_image;
    tipl::color_image fp_image_buf;
    tipl::color_map_rgb color_map;
    tipl::color_bar color_bar;
    std::vector<float> fp_matrix;
    float fp_max_value;
    tipl::image<char,3> fp_mask;
private:
    QGraphicsScene vbc_scene;
    QImage vbc_slice_image;
    unsigned int vbc_slice_pos;
private:
    void update_db(void);
    void update_subject_list(void);
public:
    explicit db_window(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc);
    ~db_window();
    void closeEvent(QCloseEvent *event);
    bool eventFilter(QObject *obj, QEvent *event);
private slots:

    void on_subject_list_itemSelectionChanged();

    void on_actionSave_Subject_Name_as_triggered();
    void on_action_Save_R2_values_as_triggered();
    void on_actionSave_fingerprints_triggered();
    void on_actionSave_pair_wise_difference_as_triggered();
    void on_view_x_toggled(bool checked);
    void on_actionLoad_mask_triggered();

    void on_actionSave_mask_triggered();
    void on_calculate_dif_clicked();
    void on_fp_zoom_valueChanged(double arg1);

    void on_delete_subject_clicked();

    void on_actionCalculate_change_triggered();

    void on_actionSave_DB_as_triggered();

    void on_subject_view_currentChanged(int index);

    void on_move_down_clicked();

    void on_move_up_clicked();

    void on_actionAdd_DB_triggered();

    void on_actionSelect_Subjects_triggered();

    void on_actionCurrent_Subject_triggered();

    void on_actionAll_Subjects_triggered();

private:
    Ui::db_window *ui;
};

#endif // CONNECTOMETRY_DB_H
