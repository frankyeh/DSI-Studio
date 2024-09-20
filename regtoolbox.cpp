#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
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
    ui->It_view->setScene(&It_scene);
    ui->I_view->setScene(&I_scene);
    connect(ui->rb_switch, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->rb_blend, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->zoom_template, SIGNAL(valueChanged(double)), this, SLOT(show_image()));
    connect(ui->zoom_subject, SIGNAL(valueChanged(double)), this, SLOT(show_image()));
    connect(ui->slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));

    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(on_timer()));
    timer->setInterval(1500);

    QMovie *movie = new QMovie(":/icons/icons/ajax-loader.gif");
    ui->running_label->setMovie(movie);
    ui->stop->hide();

    if(!has_cuda)
        ui->use_cuda->hide();

    setAcceptDrops(true);
    I_scene.installEventFilter(this);
    It_scene.installEventFilter(this);
}

RegToolBox::~RegToolBox()
{
    thread.clear();
    delete ui;
}

void RegToolBox::clear_thread(void)
{
    thread.clear();
    ui->run_reg->setText("run");
}
void RegToolBox::setup_slice_pos(void)
{
    if(reg.It.empty() || reg.I.empty())
        return;
    int range = std::max<int>(reg.Its[cur_view],reg.Is[cur_view]);
    if(range == ui->slice_pos->maximum()+1)
        return;
    ui->slice_pos->blockSignals(true);
    float pos_ratio = float(ui->slice_pos->value())/ui->slice_pos->maximum();
    ui->slice_pos->setMaximum(range-1);
    ui->slice_pos->setValue(pos_ratio*(range-1));
    ui->slice_pos->blockSignals(false);
    show_image();
}
void RegToolBox::on_OpenTemplate_clicked()
{
    if(template_names.size() >= reg.max_modality)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_template(filename.toStdString());
    show_image();
}
void RegToolBox::on_ClearTemplate_clicked()
{
    template_names.clear();
    reg.clear_reg();
    reg.It.clear();
    reg.It.resize(reg.max_modality);
    It_scene.clear();
}
void RegToolBox::load_template(const std::string& file_name)
{
    clear_thread();
    if(!reg.load_template(template_names.size(),file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    if(template_names.empty())
    {
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg.It[0].width()));
        setup_slice_pos();
    }
    template_names.push_back(file_name);
    auto_fill();
}

extern std::vector<std::string> fa_template_list,iso_template_list;
void RegToolBox::on_OpenSubject_clicked()
{
    if(subject_names.size() >= reg.max_modality)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    load_subject(filename.toStdString());
    if(filename.contains("qa"))
    {
        auto iso_file_name = QString(filename).replace("qa","iso");
        if(iso_file_name != filename && QFileInfo(iso_file_name).exists() &&
           QMessageBox::question(this,QApplication::applicationName(),QString("load iso from ") + iso_file_name + "?",
           QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
                load_subject(iso_file_name.toStdString());

        if(reg.It[0].empty() &&
           QMessageBox::question(this,QApplication::applicationName(),"load QA/ISO templates?",
           QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            {
                load_template(fa_template_list[0]);
                load_template(iso_template_list[0]);
            }
    }
    show_image();
}

void RegToolBox::on_ClearSubject_clicked()
{
    subject_names.clear();
    reg.clear_reg();
    reg.I.clear();
    reg.I.resize(reg.max_modality);
    I_scene.clear();
}

void RegToolBox::load_subject(const std::string& file_name)
{
    clear_thread();
    if(!reg.load_subject(subject_names.size(),file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    if(subject_names.empty())
    {
        ui->zoom_subject->setValue(width()*0.2f/(1.0f+reg.I[0].width()));
        setup_slice_pos();
    }
    subject_names.push_back(file_name);
    auto_fill();
}

void RegToolBox::auto_fill(void)
{
    if(template_names.size() == subject_names.size())
        return;
    std::string new_file_name;
    if(template_names.size() > subject_names.size())
    {
        if(!subject_names.empty() &&
           tipl::match_files(template_names[0],template_names[subject_names.size()],subject_names.front(),new_file_name) &&
           std::filesystem::exists(new_file_name) &&
           QMessageBox::question(this,QApplication::applicationName(),QString("load subject ") + new_file_name.c_str() + "?\n",
                        QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_subject(new_file_name);
    }
    else
    {

        if(!template_names.empty() &&
           tipl::match_files(subject_names[0],subject_names[template_names.size()],template_names.front(),new_file_name) &&
           std::filesystem::exists(new_file_name) &&
           QMessageBox::question(this,QApplication::applicationName(),QString("load template ") + new_file_name.c_str() + "?\n",
                        QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_template(new_file_name);
    }
}

void RegToolBox::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void RegToolBox::dropEvent(QDropEvent *event)
{
    event->acceptProposedAction();
    for(auto each : static_cast<QDropEvent *>(event)->mimeData()->urls())
    if(event->position().toPoint().x() < width()/2)
        load_subject(each.toLocalFile().toStdString());
    else
        load_template(each.toLocalFile().toStdString());
    show_image();
}
bool RegToolBox::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::GraphicsSceneMousePress && !reg.to2from.empty())
    {
        auto pos = static_cast<QGraphicsSceneMouseEvent *>(event)->scenePos();
        if(obj == &I_scene && subject_view_size[0])
        {
            auto x = float(pos.x()-subject_view_border)/subject_view_size[0];
            auto y = float(pos.y()-subject_view_border)/subject_view_size[1];
            if(x > 0.5f && int(y) < template_names.size()) // click on the right half
            {
                QString filename = QFileDialog::getSaveFileName(
                        this,("Save warped " + QFileInfo(template_names[y].c_str()).fileName()),template_names[y].c_str(),
                        "Images (*.nii *nii.gz);;All files (*)" );
                if(filename.isEmpty())
                    return true;
                if(!reg.apply_inv_warping(template_names[y].c_str(),filename.toStdString().c_str()))
                    QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
                else
                    QMessageBox::information(this,QApplication::applicationName(),"Saved");
            }
            return true;
        }
        if(obj == &It_scene && template_view_size[0])
        {
            auto x = float(pos.x()-template_view_border)/template_view_size[0];
            auto y = float(pos.y()-template_view_border)/template_view_size[1];
            if(x > 0.5f && int(y) < subject_names.size()) // click on the right half
            {
                QString filename = QFileDialog::getSaveFileName(
                        this,("Save warped " + QFileInfo(subject_names[y].c_str()).fileName()),subject_names[y].c_str(),
                        "Images (*.nii *nii.gz);;All files (*)" );
                if(filename.isEmpty())
                    return true;
                if(!reg.apply_warping(subject_names[y].c_str(),filename.toStdString().c_str()))
                    QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
                else
                    QMessageBox::information(this,QApplication::applicationName(),"Saved");
            }
        }
        return true;
    }
    return QObject::eventFilter(obj, event);
}

template<int dim>
struct image_fascade{
    static constexpr int dimension = dim;
    typedef float value_type;
    const tipl::image<dimension>& I;
    tipl::shape<dimension> It_shape;
    const tipl::image<dimension,tipl::vector<dimension> >& mapping;
    tipl::transformation_matrix<float,dimension> T;
    image_fascade(const tipl::image<dimension>& I_,
                  tipl::shape<dimension> It_shape_,
                  const tipl::image<dimension,tipl::vector<dimension> >& mapping_,
                  const tipl::transformation_matrix<float,dimension>& T_):I(I_),It_shape(It_shape_),mapping(mapping_),T(T_){;}

    float at(const tipl::vector<dimension,int>& xyz) const
    {
        if(!It_shape.is_valid(xyz))
            return 0.0f;
        tipl::vector<dimension> pos(xyz);
        if(!mapping.empty() && mapping.shape().is_valid(xyz))
            return tipl::estimate(I,mapping.at(xyz));
        T(pos);
        return tipl::estimate(I,pos);
    }
    auto width(void) const{return It_shape.width();}
    auto height(void) const{return It_shape.height();}
    auto depth(void) const{return It_shape.depth();}
    const auto& shape(void) const{return It_shape;}
    bool empty(void) const{return It_shape.empty();}
};

template<typename T,typename U>
inline auto show_slice_at(const T& source1,const U& source2,
                   int slice_pos,float ratio,uint8_t cur_view,uint8_t style)
{
    tipl::grayscale_image I1,I2;
    tipl::par_for(2,[&](size_t i)
    {
        if(i == 0)
            I1 = tipl::volume2slice_scaled(source1,cur_view,slice_pos,ratio);
        if(i == 1)
            I2 = tipl::volume2slice_scaled(source2,cur_view,slice_pos,ratio);
    },2);
    tipl::shape<2> shape(std::max<int>(I1.width(),I2.width()),std::max<int>(I1.height(),I2.height()));
    if(!shape.width())
        return tipl::grayscale_image();
    switch(style)
    {
    case 0:
        break;
    case 1:
        I2 = I1;
        break;
    case 2:
        {
            if(I2.empty())
                I2.resize(shape);
            if(!I1.empty())
            for(tipl::pixel_index<2> index(shape);index < shape.size();++index)
            {
                int x = index[0] >> 6;
                int y = index[1] >> 6;
                I2[index.index()] = ((x&1) ^ (y&1)) ? I1[index.index()] : I2[index.index()];
            }
        }
        break;
    case 3:
        {
            for(size_t i = 0;i < I1.size() && i < I2.size();++i)
            {
                I2[i] >>= 1;
                I2[i] += I1[i] >> 1;
            }
        }
        break;
    }
    if(cur_view != 2)
    {
        tipl::flip_y(I1);
        tipl::flip_y(I2);
    }
    tipl::grayscale_image buffer(tipl::shape<2>(2*shape.width(),shape.height()));
    if(!I1.empty())
        tipl::draw(I1,buffer,tipl::vector<2,int>());
    if(!I2.empty())
        tipl::draw(I2,buffer,tipl::vector<2,int>(std::max<int>(I1.width(),I2.width()),0));
    return buffer;
}


void RegToolBox::show_image(void)
{
    // paint the template side
    size_t row_count = std::max<int>(template_names.size(),subject_names.size());
    if(!row_count)
        return;
    tipl::grayscale_image subject_image,template_image;
    std::mutex subject_mutex,template_mutex;
    tipl::par_for(row_count*2,[&](size_t id)
    {
        if(id < row_count)
        {
            if(!reg.It[0].empty())
            {
                auto I = show_slice_at(
                          reg.It[id],image_fascade<3>(reg.I[id],reg.Its,reg.to2from,reg.T()),
                          float(ui->slice_pos->value())/ui->slice_pos->maximum()*(reg.Its[cur_view]-1),
                          ui->zoom_template->value(),cur_view,blend_style());
                {
                    std::lock_guard<std::mutex> lock(template_mutex);
                    if(template_image.empty())
                    {
                        template_view_size = I.shape();
                        template_image.resize(tipl::shape<2>(I.width(),I.height()*row_count));
                    }
                }
                tipl::draw(I,template_image,tipl::vector<2,int>(0,id*I.height()));
            }
        }
        else
        {
            if(!reg.I[0].empty())
            {
                id -= row_count;
                auto invT = reg.T();
                invT.inverse();
                auto I = show_slice_at(
                        reg.I[id],image_fascade<3>(reg.It[id],reg.Is,reg.from2to,invT),
                        float(ui->slice_pos->value())/ui->slice_pos->maximum()*(reg.Is[cur_view]-1),
                        ui->zoom_subject->value(),
                        cur_view,blend_style());
                {
                    std::lock_guard<std::mutex> lock(subject_mutex);
                    if(subject_image.empty())
                    {
                        subject_view_size = I.shape();
                        subject_image.resize(tipl::shape<2>(I.width(),I.height()*row_count));
                    }
                }
                tipl::draw(I,subject_image,tipl::vector<2,int>(0,id*I.height()));
            }
        }
    },row_count*2);

    I_scene.clear();
    It_scene.clear();

    auto add_text = [&](QGraphicsScene& scene,const std::vector<std::string>& list,
                        int template_view_border,int width,int height,bool left)
    {
        std::vector<QGraphicsTextItem*> names;
        for(auto each : list)
        {
            names.push_back(scene.addText(QFileInfo(each.c_str()).fileName()));
            names.back()->setRotation(270);
            names.back()->setPos(left ? 0 : template_view_border + width,
                                 template_view_border + height*(float(names.size()-0.5f)/float(row_count)) + names.back()->boundingRect().width()/2);
        }
    };
    if(!template_names.empty())
    {
        It_scene.blockSignals(true);
        auto top_text1 = It_scene.addText("template");
        auto top_text2 = It_scene.addText("subject->template");
        template_view_border = top_text1->boundingRect().height();
        It_scene.addPixmap(QPixmap::fromImage((QImage() << (tipl::color_image(template_image)))))->setPos(template_view_border, template_view_border);
        top_text1->setPos(template_view_border + template_image.width() * 0.25 - top_text1->boundingRect().width()/2,0);
        top_text2->setPos(template_view_border + template_image.width() * 0.75 - top_text2->boundingRect().width()/2,0);
        add_text(It_scene,template_names,template_view_border,template_image.width(),template_image.height(),true);
        if(blend_style() != 1)
            add_text(It_scene,subject_names,template_view_border,template_image.width(),template_image.height(),false);
        It_scene.blockSignals(false);
        It_scene.setSceneRect(0, 0, template_view_border+template_view_border+template_image.width(),template_view_border + template_image.height());
    }
    if(!subject_names.empty())
    {
        I_scene.blockSignals(true);
        auto top_text1 = I_scene.addText("subject");
        auto top_text2 = I_scene.addText("template->subject");
        subject_view_border = top_text1->boundingRect().height();
        I_scene.addPixmap(QPixmap::fromImage((QImage() << (tipl::color_image() = subject_image))))->setPos(subject_view_border, subject_view_border);
        top_text1->setPos(subject_view_border + subject_image.width() * 0.25 - top_text1->boundingRect().width()/2,0);
        top_text2->setPos(subject_view_border + subject_image.width() * 0.75 - top_text2->boundingRect().width()/2,0);
        add_text(I_scene,subject_names,subject_view_border,subject_image.width(),subject_image.height(),true);
        if(blend_style() != 1)
            add_text(I_scene,template_names,subject_view_border,subject_image.width(),subject_image.height(),false);
        I_scene.blockSignals(false);
        I_scene.setSceneRect(0, 0, subject_view_border+subject_view_border+subject_image.width(),subject_view_border + subject_image.height());
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
    if(old_arg != reg.arg)
        old_arg = reg.arg;
    if(!thread.running)
    {
        timer->stop();
        ui->running_label->movie()->stop();
        ui->running_label->hide();
        ui->stop->hide();
        ui->run_reg->show();
        ui->run_reg->setText("re-run");
        flash = true;
        tipl::out() << "registration completed";
    }
    on_switch_view_clicked();
}

void RegToolBox::on_run_reg_clicked()
{
    clear_thread();
    if(!reg.data_ready())
    {
        QMessageBox::critical(this,"ERROR","Please load image first");
        return;
    }

    reg.param.smoothing = float(ui->smoothing->value());
    reg.param.speed = float(ui->speed->value());
    reg.bound = ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound;
    reg.cost_type = ui->cost_fun->currentIndex() == 0 ? tipl::reg::corr : tipl::reg::mutual_info;
    reg.use_cuda = ui->use_cuda->isChecked();
    reg.skip_linear = ui->skip_linear->isChecked();
    reg.skip_nonlinear = ui->skip_nonlinear->isChecked();

    {
        reg.match_resolution(false);
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg.It[0].width()));
        thread.run([this](void){
            reg.linear_reg(thread.terminated);
            reg.nonlinear_reg(thread.terminated);});
    }

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
            "Images (*.mz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!reg.save_warping(filename.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
}

void RegToolBox::on_actionOpen_Mapping_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Mapping",QDir::currentPath(),
            "Images (*.mz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!reg.load_warping(filename.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
    show_image();
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

void RegToolBox::on_switch_view_clicked()
{
    ui->rb_switch->setChecked(true);
    flash = !flash;
    show_image();
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
    dialog->cur_image->I_float32 = reg.I[0];
    dialog->cur_image->shape = reg.Is;
    dialog->cur_image->vs = reg.Ivs;
    dialog->cur_image->T = reg.IR;
    dialog->cur_image->pixel_type = variant_image::float32;
    dialog->init_image();
    dialog->show();
}


void RegToolBox::on_actionTemplate_Image_triggered()
{
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->cur_image->I_float32 = reg.It[0];
    dialog->cur_image->shape = reg.Its;
    dialog->cur_image->vs = reg.Itvs;
    dialog->cur_image->T = reg.ItR;
    dialog->cur_image->pixel_type = variant_image::float32;
    dialog->regtool_subject = false;
    dialog->init_image();
    dialog->show();
}






void RegToolBox::on_actionApply_Subject_To_Template_Warping_triggered()
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
        if(!reg.apply_warping(from[0].toStdString().c_str(),to.toStdString().c_str()))
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        else
            QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
    else
    {
        tipl::progress prog("save files");
        for(int i = 0;prog(i,from.size());++i)
        {
            if(!reg.apply_warping(from[i].toStdString().c_str(),(from[i]+".wp.nii.gz").toStdString().c_str()))
            {
                QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
                return;
            }
        }
        QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
}


void RegToolBox::on_actionApply_Template_To_Subject_Warping_triggered()
{
    QStringList to = QFileDialog::getOpenFileNames(
            this,"Open Template Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(to.isEmpty())
        return;
    if(to.size() == 1)
    {
        QString from = QFileDialog::getSaveFileName(
                this,"Save Transformed Image",to[0],
                "Images (*.nii *nii.gz);;All files (*)" );
        if(from.isEmpty())
            return;
        if(!reg.apply_inv_warping(to[0].toStdString().c_str(),from.toStdString().c_str()))
            QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        else
            QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
    else
    {
        tipl::progress prog("save files");
        for(int i = 0;prog(i,to.size());++i)
        {
            if(!reg.apply_inv_warping(to[i].toStdString().c_str(),(to[i]+".wp.nii.gz").toStdString().c_str()))
            {
                QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
                return;
            }
        }
        QMessageBox::information(this,QApplication::applicationName(),"Saved");
    }
}


