#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include "reg.hpp"
#include "regtoolbox.h"
#include "ui_regtoolbox.h"
#include "basic_voxel.hpp"
#include "console.h"
#include "view_image.h"
extern bool has_cuda;
RegToolBox::RegToolBox(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RegToolBox)
{
    ui->setupUi(this);
    ui->options->hide();
    ui->OpenSubject2->setVisible(false);
    ui->OpenTemplate2->setVisible(false);
    ui->It_view->setScene(&It_scene);
    ui->I_view->setScene(&I_scene);
    connect(ui->rb_switch, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->rb_blend, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->show_warp, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->dis_spacing, SIGNAL(currentIndexChanged(int)), this, SLOT(show_image()));
    connect(ui->zoom_template, SIGNAL(valueChanged(double)), this, SLOT(show_image()));
    connect(ui->zoom_subject, SIGNAL(valueChanged(double)), this, SLOT(show_image()));
    connect(ui->template_slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->subject_slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->min1, SIGNAL(valueChanged(double)), this, SLOT(change_contrast()));
    connect(ui->min2, SIGNAL(valueChanged(double)), this, SLOT(change_contrast()));
    connect(ui->max1, SIGNAL(valueChanged(double)), this, SLOT(change_contrast()));
    connect(ui->max2, SIGNAL(valueChanged(double)), this, SLOT(change_contrast()));

    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(on_timer()));
    timer->setInterval(1500);

    QMovie *movie = new QMovie(":/icons/icons/ajax-loader.gif");
    ui->running_label->setMovie(movie);
    ui->stop->hide();

    if(!has_cuda)
        ui->use_cuda->hide();

    v2c_I.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    v2c_It.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    change_contrast();
}

RegToolBox::~RegToolBox()
{
    thread.clear();
    delete ui;
}

void RegToolBox::clear(void)
{
    thread.clear();
    ui->run_reg->setText("run");
}
void RegToolBox::setup_slice_pos(bool subject)
{
    if(!subject)
    {
        if(!reg.It[0].empty())
        {
            int range = int(reg.It[0].shape()[template_cur_view]);
            ui->template_slice_pos->setMaximum(range-1);
            ui->template_slice_pos->setValue(range/2);
            ui->template_slice_pos->show();
        }
        else
        {
            ui->template_slice_pos->setMaximum(0);
            ui->template_slice_pos->setValue(0);
            ui->template_slice_pos->hide();
        }
    }
    if(subject)
    {
        if(!reg.I[0].empty())
        {
            int range = int(reg.I[0].shape()[subject_cur_view]);
            ui->subject_slice_pos->setMaximum(range-1);
            ui->subject_slice_pos->setValue(range/2);
            ui->subject_slice_pos->show();
        }
        else
        {
            ui->subject_slice_pos->setMaximum(0);
            ui->subject_slice_pos->setValue(0);
            ui->subject_slice_pos->hide();
        }
    }
}
void RegToolBox::change_contrast()
{
    v2c_I.set_range(float(ui->min1->value()),float(ui->max1->value()));
    v2c_It.set_range(float(ui->min2->value()),float(ui->max2->value()));
    show_image();
}
void RegToolBox::on_OpenTemplate_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    clear();

    if(filename.endsWith("nii.gz") || filename.endsWith("nii"))
    {
        if(!reg.load_template(0,filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
            return;
        }
        reg_2d.clear();
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg.It[0].width()));
    }
    else
    {
        if(!reg_2d.load_template(0,filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
            return;
        }
        reg.clear();
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg_2d.It[0].width()));
    }

    setup_slice_pos(false);
    show_image();
    ui->template_filename->setText(QFileInfo(filename).baseName());
    ui->template_filename->setToolTip(filename);


    std::string new_file_name;
    if(!reg.I[1].empty() && tipl::match_files(ui->subject_filename->toolTip().toStdString(),
                         subject2_name,filename.toStdString(),new_file_name) &&
           QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,QApplication::applicationName(),QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_template2(new_file_name);
    }

}
void RegToolBox::load_template2(const std::string& filename)
{
    if(!reg.load_template(1,filename.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    template2_name = filename;
    show_image();
}
void RegToolBox::on_OpenTemplate2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_template2(filename.toStdString());
}

void RegToolBox::on_OpenSubject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;2D Pictures (*.png *.bmp *.jpg *.tif);;All files (*)" );
    if(filename.isEmpty())
        return;
    clear();
    if(filename.endsWith("nii.gz") || filename.endsWith("nii"))
    {
        if(!reg.load_subject(0,filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
            return;
        }
        reg_2d.clear();
        ui->zoom_subject->setValue(width()*0.2f/(1.0f+reg.I[0].width()));
    }
    else
    {
        if(!reg_2d.load_subject(0,filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
            return;
        }
        reg.clear();
        ui->zoom_subject->setValue(width()*0.2f/(1.0f+reg_2d.I[0].width()));
    }

    setup_slice_pos(true);
    show_image();
    ui->subject_filename->setText(QFileInfo(filename).baseName());
    ui->subject_filename->setToolTip(filename);

    std::string new_file_name;
    if(!reg.It[1].empty() && tipl::match_files(ui->template_filename->toolTip().toStdString(),
                         template2_name,filename.toStdString(),new_file_name) &&
           QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,QApplication::applicationName(),QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_subject2(new_file_name);
    }

}
void RegToolBox::load_subject2(const std::string& file_name)
{
    if(!reg.load_subject(1,file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    subject2_name = file_name;
    std::string new_file_name;
    if(reg.It.size() == 1 && !reg.It.empty() &&
       tipl::match_files(ui->subject_filename->toolTip().toStdString(),
                          subject2_name,ui->template_filename->toolTip().toStdString(),new_file_name) &&
       QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,QApplication::applicationName(),QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_template2(new_file_name);
    }
    show_image();
}

void RegToolBox::on_OpenSubject2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_subject2(filename.toStdString());
}


template<int dim>
struct image_fascade{
    static constexpr int dimension = dim;
    typedef float value_type;
    const tipl::image<dimension>& I;
    const tipl::image<dimension>& It;
    const tipl::image<dimension,tipl::vector<dimension> >& t2f_dis;
    tipl::transformation_matrix<float,dimension> T;
    image_fascade(const tipl::image<dimension>& I_,
                  const tipl::image<dimension>& It_,
                  const tipl::image<dimension,tipl::vector<dimension> >& t2f_dis_,
                  const tipl::transformation_matrix<float,dimension>& T_):I(I_),It(It_),t2f_dis(t2f_dis_),T(T_){;}

    float at(const tipl::vector<dimension,int> xyz) const
    {
        if(!It.shape().is_valid(xyz))
            return 0.0f;
        tipl::vector<dimension> pos(xyz);
        if(!t2f_dis.empty() && t2f_dis.shape() == It.shape())
            pos += t2f_dis.at(xyz);
        T(pos);
        if(!t2f_dis.empty() && t2f_dis.shape() != It.shape() && t2f_dis.shape() == I.shape())
            pos += tipl::estimate(t2f_dis,pos);
        return tipl::estimate(I,pos);
    }
    auto width(void) const{return It.width();}
    auto height(void) const{return It.height();}
    auto depth(void) const{return It.depth();}
    const auto& shape(void) const{return It.shape();}
    bool empty(void) const{return It.empty();}
};

template<typename T,typename U>
inline void show_slice_at(QGraphicsScene& scene,const T& source1,const U& source2,
                          const T& source3,const U& source4,
                   const tipl::value_to_color<float>& v2c_1,
                   const tipl::value_to_color<float>& v2c_2,
                   int slice_pos,float ratio,uint8_t cur_view,uint8_t style)
{
    auto I1 = v2c_1[tipl::volume2slice_scaled(source1,cur_view,slice_pos,ratio)];
    auto I2 = v2c_2[tipl::volume2slice_scaled(source2,cur_view,slice_pos,ratio)];
    auto I3 = v2c_1[tipl::volume2slice_scaled(source3,cur_view,slice_pos,ratio)];
    auto I4 = v2c_2[tipl::volume2slice_scaled(source4,cur_view,slice_pos,ratio)];
    switch(style)
    {
    case 0:
        break;
    case 1:
        I2 = I1;
        I4 = I3;
        break;
    case 2:
        {
            tipl::par_for(tipl::begin_index(I1.shape()),tipl::end_index(I1.shape()),
                [&](const tipl::pixel_index<2>& index)
                {
                    int x = index[0] >> 6;
                    int y = index[1] >> 6;
                    I2[index.index()] = ((x&1) ^ (y&1)) ? I1[index.index()] : I2[index.index()];
                });
            tipl::par_for(tipl::begin_index(I3.shape()),tipl::end_index(I1.shape()),
                [&](const tipl::pixel_index<2>& index)
                {
                    int x = index[0] >> 6;
                    int y = index[1] >> 6;
                    I4[index.index()] = ((x&1) ^ (y&1)) ? I3[index.index()] : I4[index.index()];
                });
        }
        break;
    case 3:
        {
            for(size_t i = 0;i < I1.size();++i)
            {
                I2[i][0] >>= 1;
                I2[i][1] >>= 1;
                I2[i][2] >>= 1;
                I2[i][0] += I1[i][0] >> 1;
                I2[i][1] += I1[i][1] >> 1;
                I2[i][2] += I1[i][2] >> 1;
            }
            for(size_t i = 0;i < I3.size();++i)
            {
                I4[i][0] >>= 1;
                I4[i][1] >>= 1;
                I4[i][2] >>= 1;
                I4[i][0] += I3[i][0] >> 1;
                I4[i][1] += I3[i][1] >> 1;
                I4[i][2] += I3[i][2] >> 1;
            }
        }
        break;
    }

    if(cur_view != 2)
    {
        tipl::flip_y(I1);
        tipl::flip_y(I2);
        tipl::flip_y(I3);
        tipl::flip_y(I4);

    }
    tipl::color_image buffer(tipl::shape<2>(I1.width()+I2.width(),I1.height()+I3.height()));
    tipl::draw(I1,buffer,tipl::vector<2,int>());
    tipl::draw(I2,buffer,tipl::vector<2,int>(I1.width(),0));
    if(!I3.empty() && !I4.empty())
    {
        tipl::draw(I3,buffer,tipl::vector<2,int>(0,I1.height()));
        tipl::draw(I4,buffer,tipl::vector<2,int>(I1.width(),I1.height()));
    }
    scene << (QImage() << buffer);
}


void RegToolBox::show_image(void)
{
    // paint the template side
    if(!reg.It.empty())
        show_slice_at(It_scene,
                      reg.It[0],
                      image_fascade<3>(reg.I[0],
                                       reg.It[0],reg.t2f_dis,reg.T()),
                      reg.It[1],
                      image_fascade<3>(reg.I[1],
                                       reg.It[1],reg.t2f_dis,reg.T()),
                      v2c_It,v2c_I,ui->template_slice_pos->value(),
                      ui->zoom_template->value(),template_cur_view,blend_style());
    // paint the subject side
    if(!reg_2d.I.empty() && !reg_2d.I[0].empty())
    {
        auto invT = reg_2d.T();
        invT.inverse();
        auto It2d = reg_2d.It[0];
        if(It2d.empty() && !reg.It.empty() && !reg.It[0].empty())
        {
            It2d = tipl::volume2slice(reg.It[0],template_cur_view,ui->template_slice_pos->value());
            invT.identity();
            invT[0] = float(It2d.width())/float(reg_2d.I[0].width());
            invT[3] = float(It2d.height())/float(reg_2d.I[0].height());
            if(template_cur_view != 2)
                tipl::flip_y(It2d);
        }
        show_slice_at(I_scene,
                      reg_2d.I[0],
                      image_fascade<2>(It2d,reg_2d.I[0],reg_2d.f2t_dis,invT),
                      reg_2d.I[1],
                      image_fascade<2>(It2d,reg_2d.I[1],reg_2d.f2t_dis,invT),
                      v2c_I,v2c_It,0,ui->zoom_subject->value(),2,blend_style());
    }
    if(!reg.I.empty())
    {
        auto invT = reg.T();
        invT.inverse();
        show_slice_at(I_scene,
                      reg.I[0],
                      image_fascade<3>(reg.It[0],
                                       reg.I[0],reg.f2t_dis,invT),
                      reg.I[1],
                      image_fascade<3>(reg.It[1],
                                       reg.I[1],reg.f2t_dis,invT),
                      v2c_I,v2c_It,ui->subject_slice_pos->value(),ui->zoom_subject->value(),
                      subject_cur_view,blend_style());
    }

    // Show subject image on the left
    /*
    if(!reg.I.empty())
    {
        const auto& I_to_show = reg.show_subject_warped(false);
        int pos = std::min(I_to_show.depth()-1,I_to_show.depth()*ui->slice_pos->value()/ui->slice_pos->maximum());
        tipl::color_image cJ(v2c_I[tipl::volume2slice_scaled(I_to_show,cur_view,pos,ratio)]);
        QImage warp_image;
        warp_image << cJ;

        if(ui->show_warp->isChecked() && ui->dis_spacing->currentIndex() && !reg.t2f_dis.empty())
        {
            float sub_ratio = float(reg.t2f_dis.width())/float(I_to_show.width());
            QPainter paint(&warp_image);
            paint.setBrush(Qt::NoBrush);
            paint.setPen(Qt::red);
            tipl::image<2,tipl::vector<3> > dis_slice;
            tipl::volume2slice(reg.t2f_dis,dis_slice,cur_view,pos*sub_ratio);

            int cur_dis = 1 << (ui->dis_spacing->currentIndex()-1);
            sub_ratio = ratio/sub_ratio;
            for(int x = 0;x < dis_slice.width();x += cur_dis)
            {
                for(int y = 1,index = x;
                        y < dis_slice.height()-1;++y,index += dis_slice.width())
                {
                    auto vfrom = dis_slice[index];
                    auto vto = dis_slice[index+dis_slice.width()];
                    vfrom[0] += x;
                    vfrom[1] += y-1;
                    vto[0] += x;
                    vto[1] += y;
                    paint.drawLine(vfrom[0]*sub_ratio,vfrom[1]*sub_ratio,
                                   vto[0]*sub_ratio,vto[1]*sub_ratio);
                }
            }

            for(int y = 0;y < dis_slice.height();y += cur_dis)
            {
                for(int x = 1,index = y*dis_slice.width();
                        x < dis_slice.width()-1;++x,++index)
                {
                    auto vfrom = dis_slice[index];
                    auto vto = dis_slice[index+1];
                    vfrom[0] += x-1;
                    vfrom[1] += y;
                    vto[0] += x;
                    vto[1] += y;
                    paint.drawLine(vfrom[0]*sub_ratio,vfrom[1]*sub_ratio,
                                   vto[0]*sub_ratio,vto[1]*sub_ratio);
                }
            }
        }
        if(cur_view != 2)
            warp_image = warp_image.mirrored(false,true);
        I_scene << warp_image;
    }*/
}
extern console_stream console;

void RegToolBox::on_timer()
{
    console.show_output();
    on_switch_view_clicked();
    if(old_arg != reg.arg)
    {
        show_image();
        old_arg = reg.arg;
    }
    if(!thread.running)
    {
        timer->stop();
        ui->running_label->movie()->stop();
        ui->running_label->hide();
        ui->stop->hide();
        ui->run_reg->show();
        ui->run_reg->setText("re-run");
        flash = false;
        tipl::out() << "registration completed";
    }
}

void RegToolBox::on_run_reg_clicked()
{
    // 2d to 3D registration
    clear();
    if(!reg_2d.I[0].empty() && reg_2d.It[0].empty() && !reg.It[0].empty())
    {
        auto shape_2d = tipl::space2slice<tipl::vector<2> >(template_cur_view,reg.It[0].shape());
        reg_2d.Itvs = reg_2d.Ivs = tipl::space2slice<tipl::vector<2> >(template_cur_view,reg.Itvs);
        reg_2d.Ivs[0] *= shape_2d[0]/float(reg_2d.I[0].shape()[0]);
        reg_2d.Ivs[1] *= shape_2d[1]/float(reg_2d.I[0].shape()[1]);

        reg_2d.It[0] = tipl::volume2slice(reg.It[0],template_cur_view,ui->template_slice_pos->value());
        if(template_cur_view != 2)
            tipl::flip_y(reg_2d.It[0]);
    }
    if(!reg.data_ready() && !reg_2d.data_ready())
    {
        QMessageBox::critical(this,"ERROR","Please load image first");
        return;
    }



    auto run_reg = [this](auto& reg)
    {
        reg.param.resolution = ui->resolution->value();
        reg.param.min_dimension = uint32_t(ui->min_reso->value());
        reg.param.smoothing = float(ui->smoothing->value());
        reg.param.speed = float(ui->speed->value());
        reg.bound = ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound;
        reg.use_cuda = ui->use_cuda->isChecked();
        if(ui->cost_fun->currentIndex() == 2)
            reg.skip_linear();
        else
            reg.linear_reg(tipl::reg::affine,ui->cost_fun->currentIndex() == 0 ? tipl::reg::mutual_info : tipl::reg::corr,thread.terminated);
        reg.nonlinear_reg(thread.terminated);
    };
    if(reg_2d.data_ready())
        thread.run([this,run_reg](void){run_reg(reg_2d);});
    else
        thread.run([this,run_reg](void){run_reg(reg);});

    ui->running_label->movie()->start();
    ui->running_label->show();
    timer->start();
    ui->stop->show();
    ui->run_reg->hide();
}
bool load_nifti_file(std::string file_name_cmd,
                     tipl::image<3>& data,
                     tipl::vector<3>& vs,
                     tipl::matrix<4,4>& trans,
                     bool& is_mni);

void RegToolBox::on_actionApply_Warping_triggered()
{
    QStringList from = QFileDialog::getOpenFileNames(
            this,"Open Subject Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(from.isEmpty())
        return;
    if(from.size() == 1)
    {
        QString to = QFileDialog::getSaveFileName(
                this,"Save Transformed Image",from[0],
                "Images (*.nii *nii.gz);;All files (*)" );
        if(to.isEmpty())
            return;
        std::string error;
        if(!reg.apply_warping(from[0].toStdString().c_str(),
                              to.toStdString().c_str()))
            QMessageBox::critical(this,"ERROR",error.c_str());
        else
            QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
    else
    {
        tipl::progress prog("save files");
        for(int i = 0;prog(i,from.size());++i)
        {
            std::string error;
            if(!reg.apply_warping(from[i].toStdString().c_str(),
                          (from[i]+".wp.nii.gz").toStdString().c_str()))

            {
                QMessageBox::critical(this,"ERROR",error.c_str());
                return;
            }
        }
        QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
}

void RegToolBox::on_stop_clicked()
{
    timer->stop();
    ui->running_label->movie()->stop();
    ui->running_label->hide();
    ui->stop->hide();
    ui->run_reg->show();
    thread.clear();
    show_image();
}

void RegToolBox::on_actionSave_Warping_triggered()
{
    if(reg.to2from.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Mapping",QDir::currentPath(),
            "Images (*map.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!reg.save_warping(filename.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
}


void RegToolBox::on_show_option_clicked()
{
    ui->options->show();
    ui->show_option->hide();
}

void RegToolBox::on_axial_view_clicked()
{
    subject_cur_view = 2;
    setup_slice_pos(true);
    show_image();
}


void RegToolBox::on_coronal_view_clicked()
{
    subject_cur_view = 1;
    setup_slice_pos(true);
    show_image();
}

void RegToolBox::on_sag_view_clicked()
{
    subject_cur_view = 0;
    setup_slice_pos(true);
    show_image();
}

void RegToolBox::on_sag_view_2_clicked()
{
    template_cur_view = 0;
    setup_slice_pos(false);
    show_image();
}


void RegToolBox::on_coronal_view_2_clicked()
{
    template_cur_view = 1;
    setup_slice_pos(false);
    show_image();
}


void RegToolBox::on_axial_view_2_clicked()
{
    template_cur_view = 2;
    setup_slice_pos(false);
    show_image();
}

void RegToolBox::on_switch_view_clicked()
{
    ui->rb_switch->setChecked(true);
    flash = !flash;
    show_image();
}


void RegToolBox::on_actionDual_Modality_triggered()
{
    ui->OpenSubject2->setVisible(true);
    ui->OpenTemplate2->setVisible(true);
}

uint8_t RegToolBox::blend_style(void)
{
    uint8_t style = 0;
    if(ui->rb_switch->isChecked() && flash)
        style = 1;
    if(ui->rb_blend->isChecked())
        style = 3;
    return style;
}



void RegToolBox::on_actionSubject_Image_triggered()
{
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if(!reg_2d.I[0].empty())
    {
        dialog->cur_image->I_float32.resize(reg_2d.I[0].shape().expand(1));
        std::copy(reg_2d.I[0].begin(),reg_2d.I[0].end(),dialog->cur_image->I_float32.begin());
        dialog->cur_image->shape = reg_2d.I[0].shape().expand(1);
        dialog->cur_image->vs = tipl::vector<3>(reg_2d.Ivs[0],reg_2d.Ivs[1],reg_2d.Ivs[1]);
        dialog->cur_image->T.identity();
        dialog->cur_image->T[0] = reg_2d.Ivs[0];
        dialog->cur_image->T[5] = reg_2d.Ivs[1];
        dialog->cur_image->T[10] = reg_2d.Ivs[1];
    }
    else
    {
        dialog->cur_image->I_float32 = reg.I[0];
        dialog->cur_image->shape = reg.I[0].shape();
        dialog->cur_image->vs = reg.Ivs;
        dialog->cur_image->T = reg.IR;
    }
    dialog->cur_image->pixel_type = variant_image::float32;
    dialog->init_image();
    dialog->show();
}


void RegToolBox::on_actionTemplate_Image_triggered()
{
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if(!reg_2d.It.empty())
    {
        dialog->cur_image->I_float32.resize(reg_2d.It[0].shape().expand(1));
        std::copy(reg_2d.It[0].begin(),reg_2d.It[0].end(),dialog->cur_image->I_float32.begin());
        dialog->cur_image->shape = reg_2d.It[0].shape().expand(1);
        dialog->cur_image->vs = tipl::vector<3>(reg_2d.Itvs[0],reg_2d.Itvs[1],reg_2d.Itvs[1]);
        dialog->cur_image->T.identity();
        dialog->cur_image->T[0] = reg_2d.Itvs[0];
        dialog->cur_image->T[5] = reg_2d.Itvs[1];
        dialog->cur_image->T[10] = reg_2d.Itvs[1];
    }
    else
    {
        dialog->cur_image->I_float32 = reg.It[0];
        dialog->cur_image->shape = reg.It[0].shape();
        dialog->cur_image->vs = reg.Itvs;
        dialog->cur_image->T = reg.ItR;
    }
    dialog->cur_image->pixel_type = variant_image::float32;
    dialog->regtool_subject = false;
    dialog->init_image();
    dialog->show();
}


