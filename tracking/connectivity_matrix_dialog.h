#ifndef CONNECTIVITY_MATRIX_DIALOG_H
#define CONNECTIVITY_MATRIX_DIALOG_H
#include <QDialog>
#include <QGraphicsScene>
#include "libs/tracking/tract_model.hpp"

class TractModel;
namespace Ui {
class connectivity_matrix_dialog;
}

class tracking_window;

class connectivity_matrix_dialog : public QDialog
{
    Q_OBJECT
    tipl::color_image cm;
    QImage view_image;
    QGraphicsScene scene;
    QString method;
public:
    ConnectivityMatrix data;
public:
    tracking_window* cur_tracking_window;
    explicit connectivity_matrix_dialog(tracking_window *parent, QString method);
    ~connectivity_matrix_dialog();

    void mouse_move(QMouseEvent *mouseEvent);
    bool is_graphic_view(QObject *) const;
private slots:

    void on_recalculate_clicked();

    void on_zoom_valueChanged(double arg1);



    void on_save_matrix_clicked();

    void on_save_network_property_clicked();

    void on_save_connectogram_clicked();

    void on_copy_to_clipboard_clicked();

private:
    Ui::connectivity_matrix_dialog *ui;
};

#endif // CONNECTIVITY_MATRIX_DIALOG_H
