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

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    image::affine_transform<3,float> arg;
    QGraphicsScene scene[3];
    image::basic_image<image::rgb_color> buffer[3];
    QImage slice_image[3];
private:
    unsigned char thread_terminated;
    std::auto_ptr<boost::thread> reg_thread;
    float w;
    void load_param(void);
public:
    QTimer* timer;
    image::transformation_matrix<3,float> T;
    image::transformation_matrix<3,float> iT;
    explicit manual_alignment(QWidget *parent,
        const image::basic_image<float,3>& from_,
        const image::basic_image<float,3>& to_,
        const image::affine_transform<3,float>& arg);
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

private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H
