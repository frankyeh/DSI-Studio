#ifndef CONNECTIVITY_MATRIX_DIALOG_H
#define CONNECTIVITY_MATRIX_DIALOG_H
#include <QDialog>
#include <QGraphicsScene>
#include "image/image.hpp"
#include "libs/tracking/tract_model.hpp"

class TractModel;
namespace Ui {
class connectivity_matrix_dialog;
}

class tracking_window;

class connectivity_matrix_dialog : public QDialog
{
    Q_OBJECT
    image::color_image cm;
    QImage view_image;
    QGraphicsScene scene;
    std::vector<std::string> region_name;
public:
    std::vector<std::vector<connectivity_info> > matrix;
    std::vector<unsigned int> connectivity_count;
    std::vector<float> tract_median_length;
    std::vector<float> tract_mean_length;
public:
    tracking_window* cur_tracking_window;
    explicit connectivity_matrix_dialog(tracking_window *parent);
    ~connectivity_matrix_dialog();

    void mouse_move(QMouseEvent *mouseEvent);
    bool is_graphic_view(QObject *) const;
private slots:

    void matrix_to_image(void);
    void on_recalculate_clicked();

    void on_zoom_valueChanged(double arg1);

    void on_log_toggled(bool checked);

    void on_save_as_clicked();

    void on_norm_toggled(bool checked);

private:
    Ui::connectivity_matrix_dialog *ui;
};

#endif // CONNECTIVITY_MATRIX_DIALOG_H
