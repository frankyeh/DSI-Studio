#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"
#include "fa_template.hpp"

typedef image::reg::mutual_information cost_func;


void run_reg(image::basic_image<float,3>& from,
             image::basic_image<float,3>& to,
             image::vector<3> vs,
             reg_data& data,
             unsigned int thread_count)
{
    data.arg.scaling[0] = vs[0];
    data.arg.scaling[1] = vs[1];
    data.arg.scaling[2] = vs[2];
    image::reg::align_center(from,to,data.arg);

    image::filter::gaussian(from);
    from -= image::segmentation::otsu_threshold(from);
    image::lower_threshold(from,0.0);
    image::normalize(from,1.0);
    image::normalize(to,1.0);

    data.progress = 0;
    image::reg::linear(from,to,data.arg,data.reg_type,cost_func(),data.terminated);
    if(data.terminated)
        return;
    image::transformation_matrix<3,float> affine(data.arg,from.geometry(),to.geometry());
    affine.inverse();
    data.progress = 1;
    image::basic_image<float,3> new_from(to.geometry());
    image::resample(from,new_from,affine,image::linear);
    if(thread_count == 1)
        image::reg::bfnorm(new_from,to,data.bnorm_data,data.terminated);
    else
        multi_thread_reg(data.bnorm_data,new_from,to,thread_count,data.terminated);
    if(!(data.terminated))
        data.progress = 2;
}

manual_alignment::manual_alignment(QWidget *parent,
                                   image::basic_image<float,3> from_,
                                   image::basic_image<float,3> to_,const image::vector<3>& vs_,int reg_type_) :
    QDialog(parent),ui(new Ui::manual_alignment),data(to_.geometry(),reg_type_),vs(vs_)
{
    from.swap(from_);
    to.swap(to_);
    reg_thread.reset(new boost::thread(run_reg,boost::ref(from),boost::ref(to),vs,boost::ref(data),1));
    ui->setupUi(this);
    if(reg_type_ == image::reg::rigid_body)
    {
        ui->scaling_group->hide();
        ui->tilting_group->hide();
    }

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
    timer->setInterval(1000);

    connect_arg_update();
    connect(ui->sag_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->cor_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->axi_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->blend_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

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
        data.terminated = 1;
        reg_thread->join();
    }
    delete ui;
}
void manual_alignment::load_param(void)
{
    // translocation
    ui->tx->setMaximum(from.geometry()[0]/2);
    ui->tx->setMinimum(-from.geometry()[0]/2);
    ui->tx->setValue(data.arg.translocation[0]);
    ui->ty->setMaximum(from.geometry()[1]/2);
    ui->ty->setMinimum(-from.geometry()[1]/2);
    ui->ty->setValue(data.arg.translocation[1]);
    ui->tz->setMaximum(from.geometry()[2]/2);
    ui->tz->setMinimum(-from.geometry()[2]/2);
    ui->tz->setValue(data.arg.translocation[2]);
    // rotation
    ui->rx->setMaximum(3.14159265358979323846*0.2);
    ui->rx->setMinimum(-3.14159265358979323846*0.2);
    ui->rx->setValue(data.arg.rotation[0]);
    ui->ry->setMaximum(3.14159265358979323846*0.2);
    ui->ry->setMinimum(-3.14159265358979323846*0.2);
    ui->ry->setValue(data.arg.rotation[1]);
    ui->rz->setMaximum(3.14159265358979323846*0.2);
    ui->rz->setMinimum(-3.14159265358979323846*0.2);
    ui->rz->setValue(data.arg.rotation[2]);
    //scaling
    ui->sx->setMaximum(data.arg.scaling[0]*2.0);
    ui->sx->setMinimum(data.arg.scaling[0]/2.0);
    ui->sx->setValue(data.arg.scaling[0]);
    ui->sy->setMaximum(data.arg.scaling[1]*2.0);
    ui->sy->setMinimum(data.arg.scaling[1]/2.0);
    ui->sy->setValue(data.arg.scaling[1]);
    ui->sz->setMaximum(data.arg.scaling[2]*2.0);
    ui->sz->setMinimum(data.arg.scaling[2]/2.0);
    ui->sz->setValue(data.arg.scaling[2]);
    //tilting
    ui->xy->setMaximum(1);
    ui->xy->setMinimum(-1);
    ui->xy->setValue(data.arg.affine[0]);
    ui->xz->setMaximum(1);
    ui->xz->setMinimum(-1);
    ui->xz->setValue(data.arg.affine[1]);
    ui->yz->setMaximum(1);
    ui->yz->setMinimum(-1);
    ui->yz->setValue(data.arg.affine[2]);

}
void manual_alignment::update_affine(void)
{
    T = image::transformation_matrix<3,float>(data.arg,from.geometry(),to.geometry());
    iT = T;
    iT.inverse();
}

void manual_alignment::update_image(void)
{
    update_affine();
    warped_from.clear();
    warped_from.resize(to.geometry());
    image::resample(from,warped_from,iT,image::linear);
}
void manual_alignment::param_changed()
{
    data.arg.translocation[0] = ui->tx->value();
    data.arg.translocation[1] = ui->ty->value();
    data.arg.translocation[2] = ui->tz->value();

    data.arg.rotation[0] = ui->rx->value();
    data.arg.rotation[1] = ui->ry->value();
    data.arg.rotation[2] = ui->rz->value();

    data.arg.scaling[0] = ui->sx->value();
    data.arg.scaling[1] = ui->sy->value();
    data.arg.scaling[2] = ui->sz->value();

    data.arg.affine[0] = ui->xy->value();
    data.arg.affine[1] = ui->xz->value();
    data.arg.affine[2] = ui->yz->value();

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
    float w1 = ui->blend_pos->value()/10.0;
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
    if(reg_thread.get())
    {
        disconnect_arg_update();
        ui->tx->setValue(data.arg.translocation[0]);
        ui->ty->setValue(data.arg.translocation[1]);
        ui->tz->setValue(data.arg.translocation[2]);
        ui->rx->setValue(data.arg.rotation[0]);
        ui->ry->setValue(data.arg.rotation[1]);
        ui->rz->setValue(data.arg.rotation[2]);
        ui->sx->setValue(data.arg.scaling[0]);
        ui->sy->setValue(data.arg.scaling[1]);
        ui->sz->setValue(data.arg.scaling[2]);
        ui->xy->setValue(data.arg.affine[0]);
        ui->xz->setValue(data.arg.affine[1]);
        ui->yz->setValue(data.arg.affine[2]);
        connect_arg_update();
        update_image();
    }
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

void manual_alignment::on_rerun_clicked()
{
    if(reg_thread.get())
    {
        data.terminated = 1;
        reg_thread->join();
    }
    reg_thread.reset(new boost::thread(run_reg,boost::ref(from),boost::ref(to),vs,boost::ref(data),1));

}
