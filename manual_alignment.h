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
    }
    image::reg::bfnorm_mapping<float,3> bnorm_data;
    image::affine_transform<3,float> arg;
    int reg_type;
    unsigned char terminated;
    unsigned char progress;

};

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    image::vector<3> vs;
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
    explicit manual_alignment(QWidget *parent,
        image::basic_image<float,3> from_,
        image::basic_image<float,3> to_,
        const image::vector<3>& vs,int reg_type = image::reg::affine);
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
