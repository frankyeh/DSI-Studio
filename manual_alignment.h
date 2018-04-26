#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H
#include <future>
#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "tipl/tipl.hpp"
#include "fib_data.hpp"
namespace Ui {
class manual_alignment;
}


class manual_alignment : public QDialog
{
    Q_OBJECT
public:
    tipl::image<float,3> from_original;
    tipl::image<float,3> from,to,warped_from;
    tipl::affine_transform<double> arg,b_upper,b_lower;
    tipl::vector<3> from_vs,to_vs;
    QGraphicsScene scene[3];
    tipl::color_image buffer[3];
    QImage slice_image[3];
private:
    tipl::thread thread;
private:

    void load_param(void);
public:
    tipl::transformation_matrix<double> T,iT;
public:
    QTimer* timer;
    explicit manual_alignment(QWidget *parent,
                              tipl::image<float,3> from_,
                              const tipl::vector<3>& from_vs,
                              tipl::image<float,3> to_,
                              const tipl::vector<3>& to_vs,
                              tipl::reg::reg_type reg_type,
                              tipl::reg::cost_type cost_function);
    ~manual_alignment();
    void connect_arg_update();
    void disconnect_arg_update();
private slots:
    void slice_pos_moved();
    void param_changed();
    void check_reg();
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();



    void on_switch_view_clicked();

    void on_save_warpped_clicked();

    void on_reg_type_currentIndexChanged(int index);
public slots:
    void on_rerun_clicked();
private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H
