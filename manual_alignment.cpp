#include "manual_alignment.h"
#include "ui_manual_alignment.h"

manual_alignment::manual_alignment(QWidget *parent,
                                   const image::basic_image<float,3>& from_,
                                   const image::basic_image<float,3>& to_,
                                   const image::affine_transform<3,double>& arg_) :
    QDialog(parent),
    ui(new Ui::manual_alignment),
    from(from_),to(to_),arg(arg_)
{
    image::normalize(to,1.0);
    image::normalize(from,1.0);
    ui->setupUi(this);
    ui->graphicsView->setScene(&scene);
    // translocation
    ui->tx->setMaximum(from.geometry()[0]/2);
    ui->tx->setMinimum(-from.geometry()[0]/2);
    ui->tx->setValue(arg.translocation[0]);
    ui->ty->setMaximum(from.geometry()[1]/2);
    ui->ty->setMinimum(-from.geometry()[1]/2);
    ui->ty->setValue(arg.translocation[1]);
    ui->tz->setMaximum(from.geometry()[2]/2);
    ui->tz->setMinimum(-from.geometry()[2]/2);
    ui->tz->setValue(arg.translocation[2]);
    // rotation
    ui->rx->setMaximum(3.14159265358979323846*0.2);
    ui->rx->setMinimum(-3.14159265358979323846*0.2);
    ui->rx->setValue(arg.rotation[0]);
    ui->ry->setMaximum(3.14159265358979323846*0.2);
    ui->ry->setMinimum(-3.14159265358979323846*0.2);
    ui->ry->setValue(arg.rotation[1]);
    ui->rz->setMaximum(3.14159265358979323846*0.2);
    ui->rz->setMinimum(-3.14159265358979323846*0.2);
    ui->rz->setValue(arg.rotation[2]);
    //scaling
    ui->sx->setMaximum(arg.scaling[0]*2.0);
    ui->sx->setMinimum(arg.scaling[0]/2.0);
    ui->sx->setValue(arg.scaling[0]);
    ui->sy->setMaximum(arg.scaling[1]*2.0);
    ui->sy->setMinimum(arg.scaling[1]/2.0);
    ui->sy->setValue(arg.scaling[1]);
    ui->sz->setMaximum(arg.scaling[2]*2.0);
    ui->sz->setMinimum(arg.scaling[2]/2.0);
    ui->sz->setValue(arg.scaling[2]);
    //tilting
    ui->xy->setMaximum(1);
    ui->xy->setMinimum(-1);
    ui->xy->setValue(arg.affine[0]);
    ui->xz->setMaximum(1);
    ui->xz->setMinimum(-1);
    ui->xz->setValue(arg.affine[1]);
    ui->yz->setMaximum(1);
    ui->yz->setMinimum(-1);
    ui->yz->setValue(arg.affine[2]);

    update_image();

    ui->slice_pos->setMaximum(to.geometry()[2]-1);
    ui->slice_pos->setMinimum(0);
    ui->slice_pos->setValue(to.geometry()[2] >> 1);


}
void manual_alignment::update_image(void)
{
    warped_from.resize(to.geometry());
    image::transformation_matrix<3,double> affine = arg;

    image::reg::shift_to_center(from.geometry(),to.geometry(),affine);
    affine.inverse();
    image::resample(from,warped_from,affine);

    warped_from += to;
    image::normalize(warped_from,1.0);

}

manual_alignment::~manual_alignment()
{
    delete ui;
}

void manual_alignment::on_slice_pos_sliderMoved(int position)
{

    buffer.resize(image::geometry<2>(warped_from.width(),warped_from.height()));
    const float* slice_image_ptr = &*warped_from.begin() + buffer.size()* position;
    for (unsigned int index = 0; index < buffer.size(); ++index)
    {
        float value = slice_image_ptr[index];
        value*=255.0;
        buffer[index] = image::rgb_color(value,value,value);
    }

    double ratio = std::max(1.0,
        std::min((double)ui->graphicsView->width()/(double)warped_from.width(),
                 (double)ui->graphicsView->height()/(double)warped_from.height()));
    scene.setSceneRect(0, 0, warped_from.width()*ratio,warped_from.height()*ratio);
    slice_image = QImage((unsigned char*)&*buffer.begin(),warped_from.width(),warped_from.height(),QImage::Format_RGB32).
                    scaled(warped_from.width()*ratio,warped_from.height()*ratio);
    scene.clear();
    scene.addRect(0, 0, warped_from.width()*ratio,warped_from.height()*ratio,QPen(),slice_image);

}
