#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H

#include <QDialog>
#include <QGraphicsScene>
#include "image/image.hpp"

namespace Ui {
class manual_alignment;
}

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    image::basic_image<float,3> from,to,warped_from;
    image::affine_transform<3,double> arg;
    QGraphicsScene scene;
    image::basic_image<image::rgb_color> buffer;
    QImage slice_image;
public:
    explicit manual_alignment(QWidget *parent,
        const image::basic_image<float,3>& from_,
        const image::basic_image<float,3>& to_,
        const image::affine_transform<3,double>& arg);
    ~manual_alignment();
    
private slots:
    void on_slice_pos_sliderMoved(int position);

private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H
