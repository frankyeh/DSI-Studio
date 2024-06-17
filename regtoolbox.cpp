#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include "reg.hpp"
#include "regtoolbox.h"
#include "ui_regtoolbox.h"
#include "basic_voxel.hpp"
#include "console.h"
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
    ui->It_mix_view->setScene(&It_mix_scene);
    ui->I_view->setScene(&I_scene);
    connect(ui->rb_mosaic, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->rb_switch, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->rb_blend, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->show_warp, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->dis_spacing, SIGNAL(currentIndexChanged(int)), this, SLOT(show_image()));
    connect(ui->mosaic_size, SIGNAL(valueChanged(int)), this, SLOT(show_image()));
    connect(ui->slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->zoom, SIGNAL(valueChanged(double)), this, SLOT(show_image()));
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
    reg_done = false;
    reg.clear();
    ui->run_reg->setText("run");
}
void RegToolBox::setup_slice_pos(void)
{
    if(!reg.It.empty())
    {
        int range = int(reg.It.shape()[cur_view]);
        ui->slice_pos->setMaximum(range-1);
        ui->slice_pos->setValue(range/2);
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
            this,"Open Template Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    if(!reg.load_template(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    setup_slice_pos();
    clear();
    show_image();
    ui->template_filename->setText(QFileInfo(filename).baseName());
    ui->template_filename->setToolTip(filename);
    ui->cost_fun->setCurrentIndex(reg.It.shape() == reg.I.shape() ? 2:0);

    std::string new_file_name;
    if(!reg.I2.empty() && tipl::match_files(ui->subject_filename->toolTip().toStdString(),
                         subject2_name,filename.toStdString(),new_file_name) &&
           QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,"DSI Studio",QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_template2(new_file_name);
    }
}
void RegToolBox::load_template2(const std::string& filename)
{
    if(!reg.load_template2(filename.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    template2_name = filename;
}
void RegToolBox::on_OpenTemplate2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_template2(filename.toStdString());
}

void RegToolBox::on_OpenSubject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    if(!reg.load_subject(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    clear();
    show_image();
    ui->subject_filename->setText(QFileInfo(filename).baseName());
    ui->subject_filename->setToolTip(filename);
    ui->cost_fun->setCurrentIndex(reg.It.shape() == reg.I.shape() ? 2:0);

    std::string new_file_name;
    if(!reg.It2.empty() && tipl::match_files(ui->template_filename->toolTip().toStdString(),
                         template2_name,filename.toStdString(),new_file_name) &&
           QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,"DSI Studio",QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_subject2(new_file_name);
    }
}
void RegToolBox::load_subject2(const std::string& file_name)
{
    if(!reg.load_subject2(file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    subject2_name = file_name;
    std::string new_file_name;
    if(reg.It2.empty() && !reg.It.empty() &&
       tipl::match_files(ui->subject_filename->toolTip().toStdString(),
                          subject2_name,ui->template_filename->toolTip().toStdString(),new_file_name) &&
       QFileInfo(new_file_name.c_str()).exists())
    {
        if(QMessageBox::question(this,"DSI Studio",QString("load ") + new_file_name.c_str() + "?\n",
                    QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_template2(new_file_name);
    }
}

void RegToolBox::on_OpenSubject2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_subject2(filename.toStdString());
}



struct image_fascade{
    typedef float value_type;
    const tipl::image<3>& I;
    const tipl::image<3>& It;
    const tipl::image<3,tipl::vector<3> >& t2f_dis;
    tipl::transformation_matrix<float> T;
    image_fascade(const tipl::image<3>& I_,
                  const tipl::image<3>& It_,
                  const tipl::image<3,tipl::vector<3> >& t2f_dis_,
                  const tipl::transformation_matrix<float>& T_):I(I_),It(It_),t2f_dis(t2f_dis_),T(T_){;}

    float at(float x,float y, float z) const
    {
        if(!It.shape().is_valid(x,y,z))
            return 0.0f;
        tipl::vector<3> pos;
        if(!t2f_dis.empty())
        {
            if(!tipl::estimate(t2f_dis,tipl::vector<3>(x,y,z),pos))
                return 0.0f;
        }
        pos[0] += x;
        pos[1] += y;
        pos[2] += z;
        T(pos);
        return tipl::estimate(I,pos);
    }
    auto width(void) const{return It.width();}
    auto height(void) const{return It.height();}
    auto depth(void) const{return It.depth();}
    const auto& shape(void) const{return It.shape();}
    bool empty(void) const{return It.empty();}
};

template<typename T>
void show_slice_at(QGraphicsScene& scene,const T& source,
                   const tipl::value_to_color<float>& v2c,
                   int slice_pos,float ratio,uint8_t cur_view)
{
    scene << (QImage() << v2c[tipl::volume2slice_scaled(source,cur_view,slice_pos,ratio)]).mirrored(false,(cur_view != 2));
}

template<typename T,typename U>
void show_mosaic_slice_at(QGraphicsScene& scene,
                          const T& source1,const U& source2,
                          const tipl::value_to_color<float>& v2c1,
                          const tipl::value_to_color<float>& v2c2,
                          size_t slice_pos,float ratio,
                          uint8_t cur_view,unsigned int mosaic_size)
{
    if(source1.empty() || source2.empty())
        return;
    tipl::color_image buf1(v2c1[tipl::volume2slice_scaled(source1,cur_view,slice_pos,ratio)]),
                      buf2(v2c2[tipl::volume2slice_scaled(source2,cur_view,slice_pos,ratio)]),
                      buf;
    buf.resize(buf1.shape());
    tipl::par_for(tipl::begin_index(buf1.shape()),tipl::end_index(buf1.shape()),
        [&](const tipl::pixel_index<2>& index)
        {
            int x = index[0] >> mosaic_size;
            int y = index[1] >> mosaic_size;
            buf[index.index()] = ((x&1) ^ (y&1)) ? buf1[index.index()] : buf2[index.index()];
        });
    scene << (QImage() << buf).mirrored(false,(cur_view != 2));
}

template<typename T,typename U>
void show_blend_slice_at(QGraphicsScene& scene,
                         const T& source1,const U& source2,
                         const tipl::value_to_color<float>& v2c1,
                         const tipl::value_to_color<float>& v2c2,
                         size_t slice_pos,float ratio,
                         uint8_t cur_view)
{
    if(source1.empty() || source2.empty())
        return;
    tipl::color_image buf(v2c1[tipl::volume2slice_scaled(source1,cur_view,slice_pos,ratio)]),
                      buf2(v2c2[tipl::volume2slice_scaled(source2,cur_view,slice_pos,ratio)]);
    for(size_t i = 0;i < buf.size();++i)
    {
        buf[i][0] |= buf2[i][0];
        buf[i][1] |= buf2[i][1];
        buf[i][2] |= buf2[i][2];
    }
    scene << (QImage() << buf).mirrored(false,(cur_view != 2));
}
void RegToolBox::show_image(void)
{
    float ratio = ui->zoom->value();
    if(!reg.It.empty())
    {
        image_fascade I_to_show(reg.show_subject(ui->show_second->isChecked()),
                                reg.It,reg.t2f_dis,reg.T());
        const auto& It_to_show = reg.show_template(ui->show_second->isChecked());
        // show template image on the right
        show_slice_at(It_scene,It_to_show,v2c_It,ui->slice_pos->value(),ratio,cur_view);

        // show image in the middle
        if(ui->rb_mosaic->isChecked())
            show_mosaic_slice_at(It_mix_scene,I_to_show,It_to_show,v2c_I,v2c_It,
                                 ui->slice_pos->value(),ratio,cur_view,ui->mosaic_size->value());
        if(ui->rb_switch->isChecked())
        {
            if(flash)
                show_slice_at(It_mix_scene,I_to_show,v2c_I,ui->slice_pos->value(),ratio,cur_view);
            else
                show_slice_at(It_mix_scene,It_to_show,v2c_It,ui->slice_pos->value(),ratio,cur_view);
        }
        if(ui->rb_blend->isChecked())
            show_blend_slice_at(It_mix_scene,I_to_show,It_to_show,v2c_I,v2c_It,ui->slice_pos->value(),ratio,cur_view);
    }


    // Show subject image on the left
    if(!reg.I.empty())
    {
        const auto& I_to_show = reg.show_subject_warped(ui->show_second->isChecked());
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
    }
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
    if(reg_done)
    {
        tipl::out() << "registration completed";
        timer->stop();
        ui->running_label->movie()->stop();
        ui->running_label->hide();
        ui->stop->hide();
        ui->run_reg->show();
        ui->run_reg->setText("re-run");
    }
}

void RegToolBox::on_run_reg_clicked()
{
    if(!reg.data_ready())
    {
        QMessageBox::critical(this,"ERROR","Please load image first");
        return;
    }
    clear();
    thread.terminated = false;

    thread.run([this]()
    {
        // adjust Ivs for affine
        reg.bound = ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound;
        reg.linear_reg(tipl::reg::affine,
                       ui->cost_fun->currentIndex(),
                       thread.terminated);

        reg.param.resolution = ui->resolution->value();
        reg.param.min_dimension = uint32_t(ui->min_reso->value());
        reg.param.smoothing = float(ui->smoothing->value());
        reg.param.speed = float(ui->speed->value());

        reg.nonlinear_reg(thread.terminated,ui->use_cuda->isChecked());
        reg_done = true;

    });

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
            this,"Open Subject Image","",
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
            QMessageBox::information(this,"DSI Studio","Saved");
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
        QMessageBox::information(this,"DSI Studio","Saved");
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

void RegToolBox::on_actionMatch_Intensity_triggered()
{
    if(reg.I.shape() == reg.It.shape())
    {
        tipl::homogenize(reg.I,reg.It);
        show_image();
    }
}

void RegToolBox::on_actionRemove_Background_triggered()
{
    if(!reg.I.empty())
    {
        reg.I -= tipl::segmentation::otsu_threshold(reg.I);
        tipl::lower_threshold(reg.I,0.0);
        tipl::normalize(reg.I);
        show_image();
    }
}



void RegToolBox::on_actionSave_Warping_triggered()
{
    if(reg.to2from.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Mapping","",
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
    cur_view = 2;
    setup_slice_pos();
    show_image();
}


void RegToolBox::on_coronal_view_clicked()
{
    cur_view = 1;
    setup_slice_pos();
    show_image();
}

void RegToolBox::on_sag_view_clicked()
{
    cur_view = 0;
    setup_slice_pos();
    show_image();
}


void RegToolBox::on_actionSmooth_Subject_triggered()
{
    if(!reg.I.empty())
    {
        tipl::filter::gaussian(reg.I);
        tipl::normalize(reg.I);
    }
    if(!reg.I2.empty())
    {
        tipl::filter::gaussian(reg.I2);
        tipl::normalize(reg.I2);
    }
    clear();
    show_image();
}

void RegToolBox::on_actionSave_Transformed_Image_triggered()
{
    if(reg.JJ.empty())
        return;
    QString to = QFileDialog::getSaveFileName(
            this,"Save Transformed Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(to.isEmpty())
        return;
    if(!reg.save_transformed_image(to.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
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

