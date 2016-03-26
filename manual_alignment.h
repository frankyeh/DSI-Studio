#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H
#include <future>
#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "image/image.hpp"
#include "fib_data.hpp"
namespace Ui {
class manual_alignment;
}


class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    image::affine_transform<double> b_upper,b_lower;
    image::vector<3> from_vs,to_vs;
    QGraphicsScene scene[3];
    image::color_image buffer[3];
    QImage slice_image[3];
private:
    std::shared_ptr<std::future<void> > reg_thread;
    bool terminated;
    void clear_thread(void)
    {
        if(reg_thread.get())
        {
            terminated = 1;
            reg_thread->wait();
            reg_thread.reset();
        }
    }
private:

    void load_param(void);
public:
    image::reg::normalization<double> data;
    image::reg::reg_cost_type cost_function;
    image::reg::reg_type reg_type;
public:
    QTimer* timer;
    explicit manual_alignment(QWidget *parent,
                              image::basic_image<float,3> from_,
                              const image::vector<3>& from_vs,
                              image::basic_image<float,3> to_,
                              const image::vector<3>& to_vs,
                              image::reg::reg_type reg_type,
                              image::reg::reg_cost_type cost_function);
    ~manual_alignment();
    void connect_arg_update();
    void disconnect_arg_update();
private slots:
    void slice_pos_moved();
    void param_changed();
    void check_reg();
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();

    void on_rerun_clicked();

    void on_switch_view_clicked();

private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H
