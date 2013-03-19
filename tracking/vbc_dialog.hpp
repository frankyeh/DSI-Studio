#ifndef VBC_DIALOG_HPP
#define VBC_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
namespace Ui {
class vbc_dialog;
}

class tracking_window;
class ODFModel;
class vbc_dialog : public QDialog
{
    Q_OBJECT
public://vbc
    QGraphicsScene vbc_scene;
    QImage vbc_slice_image;
    unsigned int vbc_slice_pos;
    fib_data cur_subject_fib;
    void show_report(const std::vector<std::vector<float> >& vbc_data);
public:
    tracking_window* cur_tracking_window;
    ODFModel* handle;
    explicit vbc_dialog(QWidget *parent,ODFModel* handle);
    ~vbc_dialog();
    void show_info_at(const image::vector<3,float>& pos);
private slots:
    void on_cal_null_trend_clicked();

    void on_cal_trend_clicked();

    void on_cal_lesser_tracts_clicked();

    void on_cal_group_dist_clicked();

    void on_show_null_distribution_clicked();

    void on_vbc_dist_update_clicked();

    void on_subject_list_itemSelectionChanged();

    void on_save_vbc_dist_clicked();

    void on_open_subject_clicked();

private:
    Ui::vbc_dialog *ui;
};

#endif // VBC_DIALOG_HPP
