#include <QFileDialog>
#include <QMessageBox>
#include "reg.hpp"
#include "manual_alignment.h"
#include "ui_manual_alignment.h"
#include "tracking/tracking_window.h"



void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
manual_alignment::manual_alignment(QWidget *parent,
                                   tipl::image<3,unsigned char>&& from_,
                                   tipl::image<3,unsigned char>&& from2_,
                                   const tipl::vector<3>& from_vs_,
                                   tipl::image<3,unsigned char>&& to_,
                                   tipl::image<3,unsigned char>&& to2_,
                                   const tipl::vector<3>& to_vs_,
                                   tipl::reg::reg_type reg_type,
                                   tipl::reg::cost_type cost_function) :
    QDialog(parent),from_vs(from_vs_),to_vs(to_vs_),timer(nullptr),ui(new Ui::manual_alignment)
{
    tipl::out() << "manual alignment";
    tipl::out() << "from dim: " << from_.shape();
    tipl::out() << "from vs: " << from_vs;
    tipl::out() << "to dim: " << to_.shape();
    tipl::out() << "to vs: " << to_vs;

    from.swap(from_);
    to.swap(to_);
    from2.swap(from2_);
    to2.swap(to2_);
    from_original = from;

    initial_LPS_nifti_srow(to_T,to.shape(),to_vs);
    initial_LPS_nifti_srow(from_T,from.shape(),from_vs);
    while(from.size() < to.size()/8)
    {
        tipl::downsample_with_padding(to);
        if(!to2.empty())
            tipl::downsample_with_padding(to2);
        to_vs *= 2.0f;
        to_downsample *= 0.5f;
        tipl::out() << "downsampling template image by 2 dim=" << to.shape() << std::endl;
    }
    while(to.size() < from.size()/8)
    {
        tipl::downsample_with_padding(from);
        if(!from2.empty())
            tipl::downsample_with_padding(from2);
        from_vs *= 2.0f;
        from_downsample *= 2.0f;
        tipl::out() << "downsampling subject image by 2 dim=" << from.shape() << std::endl;
    }

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

    connect(ui->sag_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->cor_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->axi_slice_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->zoom,SIGNAL(valueChanged(double)),this,SLOT(slice_pos_moved()));
    connect(ui->contrast,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->blend_pos,SIGNAL(valueChanged(int)),this,SLOT(slice_pos_moved()));
    connect(ui->grid,SIGNAL(clicked(bool)),this,SLOT(slice_pos_moved()));

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

    connect_arg_update();

    timer = new QTimer(this);
    timer->stop();
    timer->setInterval(500);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_reg()));

    ui->zoom->setValue(400.0/std::max(from.width(),to.width()));
    update_image();
    slice_pos_moved();

}

void manual_alignment::add_images(std::shared_ptr<fib_data> handle)
{
    for(const auto& each : handle->slices)
    {
        if(each->optional())
            continue;
        add_image(each->name,each->get_image(),tipl::transformation_matrix<float>(each->T));
    }
}

void manual_alignment::connect_arg_update()
{
    ui->tx->blockSignals(false);
    ui->ty->blockSignals(false);
    ui->tz->blockSignals(false);
    ui->sx->blockSignals(false);
    ui->sy->blockSignals(false);
    ui->sz->blockSignals(false);
    ui->rx->blockSignals(false);
    ui->ry->blockSignals(false);
    ui->rz->blockSignals(false);
    ui->xy->blockSignals(false);
    ui->xz->blockSignals(false);
    ui->yz->blockSignals(false);

}

void manual_alignment::disconnect_arg_update()
{
    ui->tx->blockSignals(true);
    ui->ty->blockSignals(true);
    ui->tz->blockSignals(true);
    ui->sx->blockSignals(true);
    ui->sy->blockSignals(true);
    ui->sz->blockSignals(true);
    ui->rx->blockSignals(true);
    ui->ry->blockSignals(true);
    ui->rz->blockSignals(true);
    ui->xy->blockSignals(true);
    ui->xz->blockSignals(true);
    ui->yz->blockSignals(true);
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
    disconnect_arg_update();
    auto max_x = std::max<float>(float(from.width())*from_vs[0],float(to.width())*to_vs[0]);
    auto max_y = std::max<float>(float(from.height())*from_vs[1],float(to.height())*to_vs[1]);
    auto max_z = std::max<float>(float(from.depth())*from_vs[2],float(to.depth())*to_vs[2]);
    ui->tx->setMaximum(max_x*0.5f);
    ui->tx->setMinimum(-max_x*0.5f);
    ui->tx->setValue(arg.translocation[0]);
    ui->ty->setMaximum(max_y*0.5f);
    ui->ty->setMinimum(-max_y*0.5f);
    ui->ty->setValue(arg.translocation[1]);
    ui->tz->setMaximum(max_z*0.5f);
    ui->tz->setMinimum(-max_z*0.5f);
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
    ui->sx->setMaximum(5.0);
    ui->sx->setMinimum(0.2);
    ui->sx->setValue(arg.scaling[0]);
    ui->sy->setMaximum(5.0);
    ui->sy->setMinimum(0.2);
    ui->sy->setValue(arg.scaling[1]);
    ui->sz->setMaximum(5.0);
    ui->sz->setMinimum(0.2);
    ui->sz->setValue(arg.scaling[2]);
    //tilting
    ui->xy->setMaximum(0.5);
    ui->xy->setMinimum(-0.5);
    ui->xy->setValue(double(arg.affine[0]));
    ui->xz->setMaximum(0.5);
    ui->xz->setMinimum(-0.5);
    ui->xz->setValue(double(arg.affine[1]));
    ui->yz->setMaximum(0.5);
    ui->yz->setMinimum(-0.5);
    ui->yz->setValue(double(arg.affine[2]));
    connect_arg_update();
}

void manual_alignment::update_image(void)
{
    T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
    iT = T;
    iT.inverse();
}

tipl::transformation_matrix<float> manual_alignment::get_iT(void)
{
    update_image();
    tipl::transformation_matrix<float> result = iT;
    if(to_downsample != 1.0f)
        tipl::multiply_constant(result.sr,result.sr+9,to_downsample);
    if(from_downsample != 1.0f)
        tipl::multiply_constant(result.data(),result.data()+12,from_downsample);
    tipl::out() << "iT: " << std::endl;
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

template<int dim,typename value_type>
struct warped_image : public tipl::shape<dim>{
    tipl::const_pointer_image<dim,value_type> I;
    tipl::transformation_matrix<float,dim> trans;
    template<typename T,typename U,typename V>
    warped_image(const T& s,const U& I_,const V& trans_):tipl::shape<dim>(s)
    {
        I = I_;
        trans = trans_;
    }
    value_type at(tipl::vector<dim> xyz) const
    {
        trans(xyz);
        tipl::vector<dim,int> pos(xyz+0.5f);
        if(I.shape().is_valid(pos))
            return I.at(pos);
        return 0;
    }
    const auto& shape(void) const{return *this;}
};

void manual_alignment::slice_pos_moved()
{
    if(to.empty())
        return;
    int slice_pos[3] = {ui->sag_slice_pos->value(),ui->cor_slice_pos->value(),ui->axi_slice_pos->value()};
    double ratio = ui->zoom->value();
    float w1 = ui->blend_pos->value()/10.0;
    float w2 = 1.0-w1;
    w1 *= 1.0+(ui->contrast->value()*0.2f);
    w2 *= 1.0+(ui->contrast->value()*0.2f);
    for(unsigned char dim = 0;dim < 3;++dim)
    {
        tipl::image<2,unsigned char> slice,slice2;
        tipl::volume2slice_scaled(warped_image<3,unsigned char>(to.shape(),from,iT),slice,dim,slice_pos[dim],ratio);
        tipl::volume2slice_scaled(to,slice2,dim,slice_pos[dim],ratio);

        tipl::color_image buffer(slice.shape());
        for (unsigned int index = 0; index < slice.size(); ++index)
        {
            float value = std::min<float>(255.0f,slice[index]*w2+slice2[index]*w1);
            buffer[index] = tipl::rgb(value,value,value);
        }
        QImage slice_image;
        slice_image << buffer;
        slice_image = slice_image.mirrored(false,dim != 2);
        QPainter painter(&slice_image);
        if(ui->grid->checkState() != Qt::Unchecked)
            tipl::qt::draw_ruler(painter,to.shape(),to_T,
                        dim,dim,dim != 2,ratio,ui->grid->checkState() == Qt::Checked);
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
        tipl::reg::linear<tipl::out>(tipl::reg::make_list(from,from2),from_vs,
                                     tipl::reg::make_list(to,to2),to_vs,arg,
                                     tipl::reg::reg_type(reg_type),thread.terminated,tipl::reg::reg_bound,cost);
        auto trans = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
        trans.inverse();
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
        tipl::reg::linear_refine<tipl::out>(tipl::reg::make_list(from,from2),from_vs,
                                            tipl::reg::make_list(to,to2),to_vs,
                                            arg,tipl::reg::reg_type(reg_type),thread.terminated,cost);

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

void manual_alignment::on_actionSave_Warped_Image_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Warping Image",QDir::currentPath(),"Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::image<3> I(to.shape());
    if(tipl::is_label_image(from_original))
        tipl::resample<tipl::interpolation::majority>(from_original,I,iT);
    else
        tipl::resample<tipl::interpolation::cubic>(from_original,I,iT);
    if(tipl::io::gz_nifti(filename.toStdString(),std::ios::out) << to_vs << to_T << I)
        QMessageBox::information(this,QApplication::applicationName(),"file saved");
    else
        QMessageBox::critical(this,"ERROR","cannot save file.");
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
void manual_alignment::on_actionSave_Transformation_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Linear Registration","linear_reg.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!(std::ofstream(filename.toStdString()) << arg))
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
        thread.join();
    }
    if(!(std::ifstream(filename.toStdString()) >> arg))
    {
        QMessageBox::critical(this,"ERROR","Invalid linear registration file.");
        return;
    }
    update_image();
}

void manual_alignment::on_actionApply_Transformation_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open NIFTI image",QDir::currentPath(),"Images (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;


    QString to_filename = QFileDialog::getSaveFileName(
            this,"Save Warping Image",filename+".wp.nii.gz","Images (*.nii *nii.gz);;All files (*)" );
    if(to_filename.isEmpty())
        return;

    tipl::image<3> from(from_original.shape());
    if(!tipl::io::gz_nifti(filename.toStdString(),std::ios::in).to_space(from,from_T))
    {
        QMessageBox::critical(this,"ERROR","Cannot read the file");
        return;
    }


    tipl::image<3> I(to.shape());
    if(tipl::is_label_image(from))
        tipl::resample<tipl::interpolation::majority>(from,I,iT);
    else
        tipl::resample<tipl::interpolation::cubic>(from,I,iT);

    if(tipl::io::gz_nifti(to_filename.toStdString(),std::ios::out) << to_vs << to_T << I)
        QMessageBox::information(this,QApplication::applicationName(),"file saved");
    else
        QMessageBox::critical(this,"ERROR","cannot save file.");
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


