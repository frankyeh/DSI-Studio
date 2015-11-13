#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H

#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "image/image.hpp"
#include <boost/thread/thread.hpp>

namespace Ui {
class manual_alignment;
}

struct reg_data{
    reg_data(const image::geometry<3>& geo,int reg_type_,unsigned int factor = 1):
        bnorm_data(geo,image::geometry<3>(7*factor,9*factor,7*factor)),reg_type(reg_type_)
    {
        terminated = false;
        progress = 0;
        bn_progress = 0;
    }
    image::reg::bfnorm_mapping<float,3> bnorm_data;
    image::affine_transform<float> arg;
    image::transformation_matrix<float> T;
    image::transformation_matrix<float> iT;
    int reg_type;
    int cost_function;
    unsigned char terminated;
    unsigned char progress;
    unsigned int bn_progress;

};
void run_reg(const image::basic_image<float,3>& from,
             const image::vector<3>& from_vs,
             const image::basic_image<float,3>& to,
             const image::vector<3>& to_vs,
             reg_data& data,
             unsigned int thread_count,
             unsigned int cost_function = 0);

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    image::affine_transform<float> b_upper,b_lower;
    image::vector<3> from_vs,to_vs;
    QGraphicsScene scene[3];
    image::color_image buffer[3];
    QImage slice_image[3];
private:
    std::auto_ptr<boost::thread> reg_thread;
    void load_param(void);
public:
    reg_data data;
    QTimer* timer;
    explicit manual_alignment(QWidget *parent,
                              image::basic_image<float,3> from_,
                              const image::vector<3>& from_vs,
                              image::basic_image<float,3> to_,
                              const image::vector<3>& to_vs,
                              int reg_type, int cost_function = 0);
    ~manual_alignment();
    void connect_arg_update();
    void disconnect_arg_update();
    void update_affine(void);
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
