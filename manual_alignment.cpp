#include <QFileDialog>
#include <QMessageBox>
#include "reg.hpp"
#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"

tipl::vector<3> adjust_to_vs(const tipl::image<3,float>& from,
               const tipl::vector<3>& from_vs,
               const tipl::image<3,float>& to,
               const tipl::vector<3>& to_vs)
{
    auto from_otsu = tipl::segmentation::otsu_threshold(from)*0.6f;
    auto to_otsu = tipl::segmentation::otsu_threshold(to)*0.6f;
    tipl::vector<3> from_min,from_max,to_min,to_max;
    tipl::bounding_box(from,from_min,from_max,from_otsu);
    tipl::bounding_box(to,to_min,to_max,to_otsu);
    from_max -= from_min;
    to_max -= to_min;
    tipl::vector<3> new_vs(to_vs);
    float rx = (to_max[0] > 0.0f) ? from_max[0]*from_vs[0]/(to_max[0]*to_vs[0]) : 1.0f;
    float ry = (to_max[1] > 0.0f) ? from_max[1]*from_vs[1]/(to_max[1]*to_vs[1]) : 1.0f;

    new_vs[0] *= rx;
    new_vs[1] *= ry;
    new_vs[2] *= (rx+ry)*0.5f; // z direction bounding box is largely affected by slice number, thus use rx and ry
    return new_vs;
}

manual_alignment::manual_alignment(QWidget *parent,
                                   tipl::image<3> from_,
                                   const tipl::vector<3>& from_vs_,
                                   tipl::image<3> to_,
                                   const tipl::vector<3>& to_vs_,
                                   tipl::reg::reg_type reg_type,
                                   tipl::reg::cost_type cost_function) :
    QDialog(parent),from_vs(from_vs_),to_vs(to_vs_),timer(nullptr),ui(new Ui::manual_alignment),warp_image_thread([&](void){warp_image();})
{
    from_original = from_;
    from.swap(from_);
    to.swap(to_);

    while(from.size() < to.size()/8)
    {
        tipl::image<3> new_to;
        tipl::downsample_with_padding(to,new_to);
        to.swap(new_to);
        to_vs *= 2.0f;
        to_downsample *= 0.5f;
        tipl::out() << "downsampling template image by 2 dim=" << to.shape() << std::endl;
    }
    while(to.size() < from.size()/8)
    {
        tipl::image<3> new_from;
        tipl::downsample_with_padding(from,new_from);
        from.swap(new_from);
        from_vs *= 2.0f;
        from_downsample *= 2.0f;
        tipl::out() << "downsampling subject image by 2 dim=" << from.shape() << std::endl;
    }

    warped_from.resize(to.shape());


    tipl::normalize(from);
    tipl::normalize(to);
    ui->setupUi(this);
    ui->options->hide();
    ui->menuBar->hide();
    ui->reg_translocation->setChecked(reg_type & tipl::reg::translocation);
    ui->reg_rotation->setChecked(reg_type & tipl::reg::rotation);
    ui->reg_scaling->setChecked(reg_type & tipl::reg::scaling);
    ui->reg_tilt->setChecked(reg_type & tipl::reg::tilt);

    ui->cost_type->setCurrentIndex(cost_function == tipl::reg::mutual_info ? 1 : 0);

    ui->sag_view->setScene(&scene[0]);
    ui->cor_view->setScene(&scene[1]);
    ui->axi_view->setScene(&scene[2]);




    ui->sag_slice_pos->setMaximum(to.width()-1);
    ui->sag_slice_pos->setMinimum(0);
    ui->sag_slice_pos->setValue(to.width() >> 1);
    ui->cor_slice_pos->setMaximum(to.height()-1);
    ui->cor_slice_pos->setMinimum(0);
    ui->cor_slice_pos->setValue(to.height() >> 1);
    ui->axi_slice_pos->setMaximum(to.depth()-1);
    ui->axi_slice_pos->setMinimum(0);
    ui->axi_slice_pos->setValue(to.depth() >> 1);

    load_param();

    connect_arg_update();
    connect(ui->sag_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->cor_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->axi_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->zoom,SIGNAL(valueChanged(double)),this,SLOT(slice_pos_moved()));
    connect(ui->contrast,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->blend_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));

    timer = new QTimer(this);
    timer->stop();
    timer->setInterval(500);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

    ui->zoom->setValue(400.0/std::max(from.width(),to.width()));
    update_image();
    slice_pos_moved();

}

void manual_alignment::warp_image(void)
{
    while(!free_thread)
    {
        if(image_need_update && warped_from.shape() == to.shape())
        {
            warp_image_ready = false;
            tipl::resample_mt(from,warped_from,iT);
            image_need_update = false;
            warp_image_ready = true;
        }
        std::this_thread::yield();
    }
}

void manual_alignment::add_images(std::shared_ptr<fib_data> handle)
{
    for(size_t i = 0;i < handle->view_item.size();++i)
        if(handle->view_item[i].name != "color")
        {
            tipl::transformation_matrix<float> T(handle->view_item[i].T);
            add_image(handle->view_item[i].name,
                      handle->view_item[i].get_image(),T);
        }
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
    free_thread = true;
    if(warp_image_thread.joinable())
        warp_image_thread.join();
    if(timer)
        timer->stop();
    delete ui;

}


void manual_alignment::load_param(void)
{
    tipl::affine_transform<float> b_upper,b_lower;
    tipl::reg::get_bound(from,to,from_vs,to_vs,arg,b_upper,b_lower,tipl::reg::affine,tipl::reg::large_bound);

    // translocation
    disconnect_arg_update();
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
    ui->rx->setMaximum(3.14159265358*1.2);
    ui->rx->setMinimum(-3.14159265358*1.2);
    ui->rx->setValue(arg.rotation[0]);
    ui->ry->setMaximum(3.14159265358*1.2);
    ui->ry->setMinimum(-3.14159265358*1.2);
    ui->ry->setValue(arg.rotation[1]);
    ui->rz->setMaximum(3.14159265358*1.2);
    ui->rz->setMinimum(-3.14159265358*1.2);
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
    ui->xy->setMaximum(double(b_upper.affine[0]));
    ui->xy->setMinimum(double(b_lower.affine[0]));
    ui->xy->setValue(double(arg.affine[0]));
    ui->xz->setMaximum(double(b_upper.affine[1]));
    ui->xz->setMinimum(double(b_lower.affine[1]));
    ui->xz->setValue(double(arg.affine[1]));
    ui->yz->setMaximum(double(b_upper.affine[2]));
    ui->yz->setMinimum(double(b_lower.affine[2]));
    ui->yz->setValue(double(arg.affine[2]));
    connect_arg_update();
}

void manual_alignment::update_image(void)
{
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    iT = T;
    iT.inverse();
    image_need_update = true;
}

tipl::transformation_matrix<float> manual_alignment::get_iT(void)
{
    update_image();
    tipl::transformation_matrix<float> result = iT;
    if(to_downsample != 1.0f)
        tipl::multiply_constant(result.sr,result.sr+9,to_downsample);
    if(from_downsample != 1.0f)
        tipl::multiply_constant(result.data,result.data+12,from_downsample);
    tipl::out() << "iT:" << std::endl;
    tipl::out() << result << std::endl;
    return result;
}
void manual_alignment::param_changed()
{
    arg.translocation[0] = float(ui->tx->value());
    arg.translocation[1] = float(ui->ty->value());
    arg.translocation[2] = float(ui->tz->value());

    arg.rotation[0] = float(ui->rx->value());
    arg.rotation[1] = float(ui->ry->value());
    arg.rotation[2] = float(ui->rz->value());

    arg.scaling[0] = float(ui->sx->value());
    arg.scaling[1] = float(ui->sy->value());
    arg.scaling[2] = float(ui->sz->value());

    arg.affine[0] = float(ui->xy->value());
    arg.affine[1] = float(ui->xz->value());
    arg.affine[2] = float(ui->yz->value());

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
    double ratio = ui->zoom->value();
    float w1 = ui->blend_pos->value()/10.0;
    float w2 = 1.0-w1;
    w1 *= 255.0;
    w2 *= 255.0;
    w1 *= 1.0+(ui->contrast->value()*0.2f);
    w2 *= 1.0+(ui->contrast->value()*0.2f);
    for(unsigned char dim = 0;dim < 3;++dim)
    {
        tipl::image<2,float> slice,slice2;
        tipl::volume2slice_scaled(warped_from,slice,dim,slice_pos[dim],ratio);
        tipl::volume2slice_scaled(to,slice2,dim,slice_pos[dim],ratio);

        tipl::color_image buffer(slice.shape());
        for (unsigned int index = 0; index < slice.size(); ++index)
        {
            float value = slice[index]*w2+slice2[index]*w1;
            value = std::min<float>(255,value);
            buffer[index] = tipl::rgb(value,value,value);
        }
        QImage slice_image;
        slice_image << buffer;
        slice_image = slice_image.mirrored(false,dim != 2);
        QPainter painter(&slice_image);
        tipl::qt::draw_ruler(painter,to.shape(),nifti_srow,
                        dim,dim,dim != 2,ratio,true);
        scene[dim] << slice_image;
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

    if(!thread.running)
    {
        ui->rerun->setText("Run registration");
        ui->refine->setText("Refine");
        ui->rerun->setEnabled(true);
        ui->refine->setEnabled(true);
        timer->stop();
    }

}



void manual_alignment::on_buttonBox_accepted()
{
    if(timer)
        timer->stop();
    update_image(); // to update the affine matrix
    accept();
}

void manual_alignment::on_buttonBox_rejected()
{
    thread.terminated = true;
    if(timer)
        timer->stop();
    reject();
}


void manual_alignment::on_rerun_clicked()
{
    if(thread.running)
    {
        thread.clear();
        check_reg();
        return;
    }

    auto cost = ui->cost_type->currentIndex() == 0 ? tipl::reg::corr : tipl::reg::mutual_info;
    int reg_type = 0;
    if(ui->reg_translocation->isChecked())
        reg_type += int(tipl::reg::translocation);
    if(ui->reg_rotation->isChecked())
        reg_type += int(tipl::reg::rotation);
    if(ui->reg_scaling->isChecked())
        reg_type += int(tipl::reg::scaling);
    if(ui->reg_tilt->isChecked())
        reg_type += int(tipl::reg::tilt);

    load_param();

    thread.run([this,cost,reg_type]()
    {
        if(cost == tipl::reg::mutual_info)
            linear_with_mi(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated);
        else
            linear_with_cc(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated);
        thread.running = false;
    });
    ui->rerun->setText("Stop");
    ui->refine->setEnabled(false);
    if(timer)
        timer->start();

}


void manual_alignment::on_refine_clicked()
{
    if(thread.running)
    {
        thread.clear();
        check_reg();
        return;
    }

    auto cost = ui->cost_type->currentIndex() == 0 ? tipl::reg::corr : tipl::reg::mutual_info;
    int reg_type = 0;
    if(ui->reg_translocation->isChecked())
        reg_type += int(tipl::reg::translocation);
    if(ui->reg_rotation->isChecked())
        reg_type += int(tipl::reg::rotation);
    if(ui->reg_scaling->isChecked())
        reg_type += int(tipl::reg::scaling);
    if(ui->reg_tilt->isChecked())
        reg_type += int(tipl::reg::tilt);

    load_param();

    thread.run([this,cost,reg_type]()
    {
        if(cost == tipl::reg::mutual_info)
            linear_with_mi_refine(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated);
        else
            linear_with_cc(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated);
        thread.running = false;
    });
    ui->refine->setText("Stop");
    ui->rerun->setEnabled(false);
    if(timer)
        timer->start();
}


void manual_alignment::on_switch_view_clicked()
{
    ui->blend_pos->setValue(ui->blend_pos->value() > ui->blend_pos->maximum()/2 ? 0:ui->blend_pos->maximum());
}

void manual_alignment::on_actionSave_Warpped_Image_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Warping Image","","Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::image<3> I(to.shape());
    if(tipl::is_label_image(from_original))
        tipl::resample_mt<tipl::interpolation::nearest>(from_original,I,iT);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(from_original,I,iT);
    tipl::io::gz_nifti::save_to_file(filename.toStdString().c_str(),I,to_vs,nifti_srow);
}

void manual_alignment::on_advance_options_clicked()
{
    if(ui->options->isVisible())
        ui->options->hide();
    else
        ui->options->show();
}

void manual_alignment::on_files_clicked()
{
    ui->menu_File->popup(QCursor::pos());
}

bool save_transform(const char* file_name,const tipl::affine_transform<float>& argmin);
void manual_alignment::on_actionSave_Transformation_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Linear Registration","linear_reg.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!(std::ofstream(filename.toStdString().c_str()) << arg))
        QMessageBox::critical(this,"ERROR","Cannot save file.");
}

void manual_alignment::on_actionLoad_Transformation_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Linear Registration","linear_reg.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(thread.running)
    {
        thread.terminated = true;
        thread.wait();
    }
    if(!(std::ifstream(filename.toStdString().c_str()) >> arg))
    {
        QMessageBox::critical(this,"ERROR","Invalid linear registration file.");
        return;
    }
    update_image();
}

void manual_alignment::on_actionApply_Transformation_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open NIFTI image","","Images (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;


    QString to_filename = QFileDialog::getSaveFileName(
            this,"Save Warping Image",filename+".wp.nii.gz","Images (*.nii *nii.gz);;All files (*)" );
    if(to_filename.isEmpty())
        return;

    tipl::image<3> from;
    tipl::vector<3> vs;
    if(!tipl::io::gz_nifti::load_from_file(filename.toStdString().c_str(),from,vs))
    {
        QMessageBox::critical(this,"ERROR","Cannot read the file");
        return;
    }
    if(from.shape() != from_original.shape())
    {
        QMessageBox::critical(this,"ERROR","The NIFTI file has different dimension");
        return;
    }

    tipl::image<3> I(to.shape());
    if(tipl::is_label_image(from))
        tipl::resample_mt<tipl::interpolation::nearest>(from,I,iT);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(from,I,iT);
    tipl::io::gz_nifti::save_to_file(to_filename.toStdString().c_str(),I,to_vs,nifti_srow);



}


void manual_alignment::on_actionSmooth_Signals_triggered()
{
    tipl::filter::gaussian(from);
    tipl::filter::gaussian(to);
    update_image();
}


void manual_alignment::on_actionSobel_triggered()
{
    tipl::filter::sobel(from);
    tipl::filter::sobel(to);
    update_image();
}


void manual_alignment::on_pushButton_clicked()
{
    ui->menu_Edit->popup(QCursor::pos());
}


