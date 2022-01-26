#include <QFileDialog>
#include <QMessageBox>
#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"

void show_view(QGraphicsScene& scene,QImage I);
bool is_label_image(const tipl::image<3>& I)
{
    for(size_t i = 0;i < I.size();++i)
        if(std::floor(I[i]) < I[i])
            return false;
    return true;
}
manual_alignment::manual_alignment(QWidget *parent,
                                   tipl::image<3> from_,
                                   const tipl::vector<3>& from_vs_,
                                   tipl::image<3> to_,
                                   const tipl::vector<3>& to_vs_,
                                   tipl::reg::reg_type reg_type,
                                   tipl::reg::cost_type cost_function) :
    QDialog(parent),from_vs(from_vs_),to_vs(to_vs_),timer(nullptr),ui(new Ui::manual_alignment)
{
    from_original = from_;
    from.swap(from_);
    to.swap(to_);

    while(tipl::minimum(to_vs) < tipl::minimum(from_vs)/2.0f)
    {
        tipl::image<3> new_to;
        tipl::downsample_with_padding(to,new_to);
        to.swap(new_to);
        to_vs *= 2.0f;
        to_downsample *= 0.5f;
        std::cout << "downsampling template image by 2 dim=" << to.shape() << std::endl;
    }
    while(tipl::minimum(from_vs) < tipl::minimum(to_vs)/2.0f)
    {
        tipl::image<3> new_from;
        tipl::downsample_with_padding(from,new_from);
        from.swap(new_from);
        from_vs *= 2.0f;
        from_downsample *= 2.0f;
        std::cout << "downsampling subject image by 2 dim=" << from.shape() << std::endl;
    }

    tipl::normalize(from,1.0);
    tipl::normalize(to,1.0);
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
    connect(ui->blend_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));

    timer = new QTimer(this);
    timer->stop();
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

    ui->zoom->setValue(400.0/std::max(from.width(),to.width()));
    update_image();
    slice_pos_moved();

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
    if(timer)
        timer->stop();
    delete ui;
}
void manual_alignment::load_param(void)
{
    tipl::affine_transform<float> b_upper,b_lower;
    tipl::reg::get_bound(from,to,arg,b_upper,b_lower,tipl::reg::affine,tipl::reg::large_bound);

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
    warped_from.clear();
    warped_from.resize(to.shape());
    tipl::resample(from,warped_from,iT);
}

void manual_alignment::set_arg(const tipl::affine_transform<float>& arg_min,
                               tipl::transformation_matrix<float> iT)
{
    if(to_downsample != 1.0f || from_downsample != 1.0f)
    {
        if(to_downsample != 1.0f)
            tipl::multiply_constant(iT.sr,iT.sr+9,1.0f/to_downsample);
        if(from_downsample != 1.0f)
            tipl::multiply_constant(iT.data,iT.data+12,1.0f/from_downsample);
        iT.inverse();
        iT.to_affine_transform(arg,from.shape(),from_vs,to.shape(),to_vs);
    }
    else
        arg = arg_min;
    check_reg();
}
tipl::transformation_matrix<float> manual_alignment::get_iT(void)
{
    update_image();
    tipl::transformation_matrix<float> result = iT;
    if(to_downsample != 1.0f)
        tipl::multiply_constant(result.sr,result.sr+9,to_downsample);
    if(from_downsample != 1.0f)
        tipl::multiply_constant(result.data,result.data+12,from_downsample);
    std::cout << "iT:" << std::endl;
    std::cout << result << std::endl;
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
    w1*= 255.0;
    w2 *= 255.0;
    for(unsigned char dim = 0;dim < 3;++dim)
    {
        tipl::image<2,float> slice,slice2;
        tipl::volume2slice(warped_from,slice,dim,slice_pos[dim]);
        tipl::volume2slice(to,slice2,dim,slice_pos[dim]);
        buffer[dim].resize(slice.shape());
        for (unsigned int index = 0; index < slice.size(); ++index)
        {
            float value = slice[index]*w2+slice2[index]*w1;
            buffer[dim][index] = tipl::rgb(value,value,value);
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
        {
            tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated,0.01,true,tipl::reg::large_bound);
            tipl::reg::linear<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated,0.001,true,tipl::reg::large_bound);
        }
        else
        {
            tipl::reg::linear<tipl::reg::correlation>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated,0.01,true,tipl::reg::large_bound);
            tipl::reg::linear<tipl::reg::correlation>(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),thread.terminated,0.001,true,tipl::reg::large_bound);
        }
    });
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
            this,"Save Warpping Image","","Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::image<3> I(to.shape());
    if(is_label_image(from_original))
        tipl::resample_mt<tipl::interpolation::nearest>(from_original,I,iT);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(from_original,I,iT);
    gz_nifti::save_to_file(filename.toStdString().c_str(),I,to_vs,nifti_srow);
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

bool save_transform(const char* file_name,const tipl::matrix<4,4>& T,
                    const tipl::affine_transform<float>& argmin);

void manual_alignment::on_actionSave_Transformation_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Mapping Matrix","mapping.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    tipl::matrix<4,4> T_;
    T.save_to_transform(T_.begin());
    if(!save_transform(filename.toStdString().c_str(),T_,arg))
        QMessageBox::critical(this,"ERROR","Cannot save mapping file.");
}

bool load_transform(const char* file_name,tipl::affine_transform<float>& arg_min);
void manual_alignment::on_actionLoad_Transformation_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Mapping Matrix",".mapping.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    thread.terminated = true;
    thread.wait();
    if(!load_transform(filename.toStdString().c_str(),arg))
    {
        QMessageBox::critical(this,"ERROR","Invalid mapping file.");
        return;
    }
    update_image();
}
