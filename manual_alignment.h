#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H

#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "image/image.hpp"
#include "libs/coreg/linear.hpp"
#include <boost/thread/thread.hpp>

namespace Ui {
class manual_alignment;
}

struct reg_data{
    reg_data(const image::geometry<3>& geo,
                   const image::affine_transform<3,float>& arg_,
                   int reg_type_):bnorm_data(geo,image::geometry<3>(7,9,7)),arg(arg_),reg_type(reg_type_)
    {
        terminated = false;
        progress = 0;
    }

    int reg_type;
    image::reg::bfnorm_mapping<double,3> bnorm_data;
    unsigned char terminated;
    unsigned char progress;
    image::affine_transform<3,float> arg;
};

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    QGraphicsScene scene[3];
    image::color_image buffer[3];
    QImage slice_image[3];
private:
    std::auto_ptr<boost::thread> reg_thread;
    void load_param(void);
public:
    reg_data data;
    QTimer* timer;
    image::transformation_matrix<3,float> T;
    image::transformation_matrix<3,float> iT;
    bool need_update_affine_matrix;
    explicit manual_alignment(QWidget *parent,
        image::basic_image<float,3> from_,
        image::basic_image<float,3> to_,
        const image::affine_transform<3,float>& arg,
                              int reg_type = image::reg::affine);
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

private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H
