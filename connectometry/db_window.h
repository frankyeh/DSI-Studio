#ifndef DB_WINDOW_H
#define DB_WINDOW_H
#include "vbc/vbc_database.h"
#include <QMainWindow>
#include <QGraphicsScene>

namespace Ui {
class db_window;
}

class db_window : public QMainWindow
{
    Q_OBJECT
    std::shared_ptr<vbc_database> vbc;
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
private:
    void update_db(void);
    void update_subject_list(void);
public:
    explicit db_window(QWidget *parent,std::shared_ptr<vbc_database> vbc);
    ~db_window();

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

private:
    Ui::db_window *ui;
};

#endif // CONNECTOMETRY_DB_H
