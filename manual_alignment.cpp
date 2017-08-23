#include <QFileDialog>
#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"
#include "fa_template.hpp"



void show_view(QGraphicsScene& scene,QImage I);

manual_alignment::manual_alignment(QWidget *parent,
                                   image::basic_image<float,3> from_,
                                   const image::vector<3>& from_vs_,
                                   image::basic_image<float,3> to_,
                                   const image::vector<3>& to_vs_,
                                   image::reg::reg_type reg_type,
                                   image::reg::cost_type cost_function) :
    QDialog(parent),ui(new Ui::manual_alignment),from_vs(from_vs_),to_vs(to_vs_),timer(0)
{
    from_original = from_;
    from.swap(from_);
    to.swap(to_);
    image::normalize(from,1.0);
    image::normalize(to,1.0);
    image::reg::get_bound(from,arg,b_upper,b_lower,reg_type);


    ui->setupUi(this);
    ui->reg_type->setCurrentIndex(reg_type == image::reg::rigid_body? 0: 1);
    ui->cost_type->setCurrentIndex(cost_function == image::reg::mutual_info ? 1 : 0);


    ui->sag_view->setScene(&scene[0]);
    ui->cor_view->setScene(&scene[1]);
    ui->axi_view->setScene(&scene[2]);



    load_param();

    ui->sag_slice_pos->setMaximum(to.geometry()[0]-1);
    ui->sag_slice_pos->setMinimum(0);
    ui->sag_slice_pos->setValue(to.geometry()[0] >> 1);
    ui->cor_slice_pos->setMaximum(to.geometry()[1]-1);
    ui->cor_slice_pos->setMinimum(0);
    ui->cor_slice_pos->setValue(to.geometry()[1] >> 1);
    ui->axi_slice_pos->setMaximum(to.geometry()[2]-1);
    ui->axi_slice_pos->setMinimum(0);
    ui->axi_slice_pos->setValue(to.geometry()[2] >> 1);



    connect_arg_update();
    connect(ui->sag_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->cor_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->axi_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->blend_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));

    timer = new QTimer(this);
    timer->stop();
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

    update_image();
    slice_pos_moved();

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
    if(timer)
        timer->stop();
    delete ui;
}
void manual_alignment::load_param(void)
{


    // translocation
    ui->tx->setMaximum(b_upper.translocation[0]);
    ui->tx->setMinimum(b_lower.translocation[0]);
    ui->tx->setValue(arg.translocation[0]);
    ui->ty->setMaximum(b_upper.translocation[1]);
    ui->ty->setMinimum(b_lower.translocation[1]);
    ui->ty->setValue(arg.translocation[1]);
    ui->tz->setMaximum(b_upper.translocation[2]);
    ui->tz->setMinimum(b_lower.translocation[2]);
    ui->tz->setValue(arg.translocation[2]);
    // rotation
    ui->rx->setMaximum(3.14159265358);
    ui->rx->setMinimum(-3.14159265358);
    ui->rx->setValue(arg.rotation[0]);
    ui->ry->setMaximum(3.14159265358);
    ui->ry->setMinimum(-3.14159265358);
    ui->ry->setValue(arg.rotation[1]);
    ui->rz->setMaximum(3.14159265358);
    ui->rz->setMinimum(-3.14159265358);
    ui->rz->setValue(arg.rotation[2]);
    //scaling
    ui->sx->setMaximum(b_upper.scaling[0]);
    ui->sx->setMinimum(b_lower.scaling[0]);
    ui->sx->setValue(arg.scaling[0]);
    ui->sy->setMaximum(b_upper.scaling[1]);
    ui->sy->setMinimum(b_lower.scaling[1]);
    ui->sy->setValue(arg.scaling[1]);
    ui->sz->setMaximum(b_upper.scaling[2]);
    ui->sz->setMinimum(b_lower.scaling[2]);
    ui->sz->setValue(arg.scaling[2]);
    //tilting
    ui->xy->setMaximum(b_upper.affine[0]);
    ui->xy->setMinimum(b_lower.affine[0]);
    ui->xy->setValue(arg.affine[0]);
    ui->xz->setMaximum(b_upper.affine[1]);
    ui->xz->setMinimum(b_lower.affine[1]);
    ui->xz->setValue(arg.affine[1]);
    ui->yz->setMaximum(b_upper.affine[2]);
    ui->yz->setMinimum(b_lower.affine[2]);
    ui->yz->setValue(arg.affine[2]);

}

void manual_alignment::update_image(void)
{
    T = image::transformation_matrix<double>(arg,from.geometry(),from_vs,to.geometry(),to_vs);
    iT = T;
    iT.inverse();
    warped_from.clear();
    warped_from.resize(to.geometry());
    image::resample(from,warped_from,iT,image::linear);
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
    if(warped_from.empty() || to.empty())
        return;
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
        slice_image[dim] = QImage((unsigned char*)&*buffer[dim].begin(),buffer[dim].width(),buffer[dim].height(),QImage::Format_RGB32).
                        scaled(buffer[dim].width()*ratio,buffer[dim].height()*ratio);
        if(dim != 2)
            slice_image[dim] = slice_image[dim].mirrored();
        show_view(scene[dim],slice_image[dim]);
    }
}

void manual_alignment::check_reg()
{
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
    slice_pos_moved();
}



void manual_alignment::on_buttonBox_accepted()
{
    if(timer)
        timer->stop();
    update_image(); // to update the affine matrix
}

void manual_alignment::on_buttonBox_rejected()
{
    thread.terminated = true;
    if(timer)
        timer->stop();
}

void manual_alignment::on_rerun_clicked()
{
    auto cost = ui->cost_type->currentIndex() == 0 ? image::reg::corr : image::reg::mutual_info;
    auto reg = ui->reg_type->currentIndex() == 0 ? image::reg::rigid_body : image::reg::affine;
    thread.run([this,cost,reg]()
    {
        if(cost == image::reg::mutual_info)
        {
            image::reg::linear_mr(from,from_vs,to,to_vs,arg,reg,image::reg::mutual_information(),thread.terminated,0.1);
            image::reg::linear_mr(from,from_vs,to,to_vs,arg,reg,image::reg::mutual_information(),thread.terminated,0.01);
        }
        else
        {
            image::reg::linear_mr(from,from_vs,to,to_vs,arg,reg,image::reg::mt_correlation<image::basic_image<float,3>,
                           image::transformation_matrix<double> >(0),thread.terminated,0.1);
            image::reg::linear_mr(from,from_vs,to,to_vs,arg,reg,image::reg::mt_correlation<image::basic_image<float,3>,
                           image::transformation_matrix<double> >(0),thread.terminated,0.01);
        }

    });
    if(timer)
        timer->start();

}

void manual_alignment::on_switch_view_clicked()
{
    ui->blend_pos->setValue(ui->blend_pos->value() > ui->blend_pos->maximum()/2 ? 0:ui->blend_pos->maximum());
}

void manual_alignment::on_save_warpped_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Warpping Image","","Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    image::basic_image<float,3> I(to.geometry());
    image::resample(from_original,I,iT,image::cubic);
    gz_nifti nii;
    nii.set_voxel_size(to_vs.begin());
    image::flip_xy(I);
    nii << I;
    nii.save_to_file(filename.toStdString().c_str());
}

void manual_alignment::on_reg_type_currentIndexChanged(int index)
{
    image::reg::get_bound(from,arg,b_upper,b_lower,
                          ui->reg_type->currentIndex() == 0 ? image::reg::rigid_body : image::reg::affine);
    if(index == 0) // rigid body
    {
        ui->scaling_group->hide();
        ui->tilting_group->hide();
    }
    else
    {
        ui->scaling_group->show();
        ui->tilting_group->show();
    }
}
