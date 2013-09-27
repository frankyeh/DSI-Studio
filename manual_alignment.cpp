#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"

typedef image::reg::mutual_information cost_func;
void run_reg(const image::basic_image<float,3>& from,
             const image::basic_image<float,3>& to,
             image::affine_transform<3,float>* arg_min,
             unsigned char* terminated)
{
    image::reg::linear<boost::thread>(from,to,*arg_min,image::reg::affine,cost_func(),2,*terminated);
}
manual_alignment::manual_alignment(QWidget *parent,
                                   image::basic_image<float,3> from_,
                                   image::basic_image<float,3> to_,
                                   const image::affine_transform<3,float>& arg_) :
    QDialog(parent),ui(new Ui::manual_alignment),arg(arg_)
{
    from.swap(from_);
    to.swap(to_);
    image::filter::gaussian(from);
    from -= image::segmentation::otsu_threshold(from);
    image::lower_threshold(from,0.0);

    image::normalize(from,1.0);
    image::normalize(to,1.0);

    ui->setupUi(this);
    ui->sag_view->setScene(&scene[0]);
    ui->cor_view->setScene(&scene[1]);
    ui->axi_view->setScene(&scene[2]);

    load_param();
    update_image();

    ui->sag_slice_pos->setMaximum(to.geometry()[0]-1);
    ui->sag_slice_pos->setMinimum(0);
    ui->sag_slice_pos->setValue(to.geometry()[0] >> 1);
    ui->cor_slice_pos->setMaximum(to.geometry()[1]-1);
    ui->cor_slice_pos->setMinimum(0);
    ui->cor_slice_pos->setValue(to.geometry()[1] >> 1);
    ui->axi_slice_pos->setMaximum(to.geometry()[2]-1);
    ui->axi_slice_pos->setMinimum(0);
    ui->axi_slice_pos->setValue(to.geometry()[2] >> 1);
    timer = new QTimer(this);
    timer->setInterval(200);

    connect_arg_update();
    connect(ui->sag_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->cor_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->axi_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

    w = 0.0;

    thread_terminated = 0;
    reg_thread.reset(new boost::thread(run_reg,from,to,&arg,&thread_terminated));

}
void manual_alignment::connect_arg_update()
{
    connect(ui->tx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->ty,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->tz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->sx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->sy,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->sz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->rx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->ry,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->rz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->xy,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->xz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    connect(ui->yz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
}

void manual_alignment::disconnect_arg_update()
{
    disconnect(ui->tx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->ty,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->tz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->sx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->sy,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->sz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->rx,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->ry,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->rz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->xy,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->xz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
    disconnect(ui->yz,SIGNAL(valueChanged(double)),this,SLOT(param_changed()));
}

manual_alignment::~manual_alignment()
{
    timer->stop();
    if(reg_thread.get())
    {
        timer->stop();
        thread_terminated = 1;
        reg_thread->join();
    }
    delete ui;
}
void manual_alignment::load_param(void)
{
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

}
void manual_alignment::update_affine(void)
{
    T = image::transformation_matrix<3,float>(arg,from.geometry(),to.geometry());
    iT = T;
    iT.inverse();
}

void manual_alignment::update_image(void)
{
    update_affine();
    warped_from.clear();
    warped_from.resize(to.geometry());
    image::resample(from,warped_from,iT);
}
void manual_alignment::param_changed()
{
    arg.translocation[0] = ui->tx->value();
    arg.translocation[1] = ui->ty->value();
    arg.translocation[2] = ui->tz->value();

    arg.rotation[0] = ui->rx->value();
    arg.rotation[1] = ui->ry->value();
    arg.rotation[2] = ui->rz->value();

    arg.scaling[0] = ui->sx->value();
    arg.scaling[1] = ui->sy->value();
    arg.scaling[2] = ui->sz->value();

    arg.affine[0] = ui->xy->value();
    arg.affine[1] = ui->xz->value();
    arg.affine[2] = ui->yz->value();

    update_image();
    slice_pos_moved();
}



void manual_alignment::slice_pos_moved()
{
    int slice_pos[3];
    slice_pos[0] = ui->sag_slice_pos->value();
    slice_pos[1] = ui->cor_slice_pos->value();
    slice_pos[2] = ui->axi_slice_pos->value();
    double ratio =
        std::min((double)(ui->axi_view->width()-10)/(double)warped_from.width(),
                 (double)(ui->axi_view->height()-10)/(double)warped_from.height());
    float w1 = w;
    float w2 = 1.0-w1;
    w1*= 255.0;
    w2 *= 255.0;
    for(unsigned char dim = 0;dim < 3;++dim)
    {
        image::basic_image<float,2> slice,slice2;
        image::reslicing(warped_from,slice,dim,slice_pos[dim]);
        image::reslicing(to,slice2,dim,slice_pos[dim]);
        buffer[dim].resize(slice.geometry());
        for (unsigned int index = 0; index < slice.size(); ++index)
        {
            float value = slice[index]*w2+slice2[index]*w1;
            buffer[dim][index] = image::rgb_color(value,value,value);
        }
        scene[dim].setSceneRect(0, 0, buffer[dim].width()*ratio,buffer[dim].height()*ratio);
        slice_image[dim] = QImage((unsigned char*)&*buffer[dim].begin(),buffer[dim].width(),buffer[dim].height(),QImage::Format_RGB32).
                        scaled(buffer[dim].width()*ratio,buffer[dim].height()*ratio);
        if(dim != 2)
            slice_image[dim] = slice_image[dim].mirrored();
        scene[dim].clear();
        scene[dim].addRect(0, 0, buffer[dim].width()*ratio,buffer[dim].height()*ratio,QPen(),slice_image[dim]);
    }
}

void manual_alignment::check_reg()
{
    if(w > 0.95 && reg_thread.get())
    {
        disconnect_arg_update();
        ui->tx->setValue(arg.translocation[0]);
        ui->ty->setValue(arg.translocation[1]);
        ui->tz->setValue(arg.translocation[2]);
        ui->rx->setValue(arg.rotation[0]);
        ui->ry->setValue(arg.rotation[1]);
        ui->rz->setValue(arg.rotation[2]);
        ui->sx->setValue(arg.scaling[0]);
        ui->sy->setValue(arg.scaling[1]);
        ui->sz->setValue(arg.scaling[2]);
        ui->xy->setValue(arg.affine[0]);
        ui->xz->setValue(arg.affine[1]);
        ui->yz->setValue(arg.affine[2]);
        connect_arg_update();
        update_image();
    }
    w += 0.1;
    if(w > 1.05)
        w = 0.0;
    slice_pos_moved();
}



void manual_alignment::on_buttonBox_accepted()
{
    timer->stop();
    update_image(); // to update the affine matrix
}

void manual_alignment::on_buttonBox_rejected()
{
    timer->stop();
}
