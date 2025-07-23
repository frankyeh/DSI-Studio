#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include <QInputDialog>
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
    ui->It_view->setScene(&scene[1]);
    ui->I_view->setScene(&scene[0]);
    connect(ui->rb_switch, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->rb_blend, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->anchor, SIGNAL(clicked()), this, SLOT(show_image()));
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
    scene[0].installEventFilter(this);
    scene[1].installEventFilter(this);
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
    if(file_names[1].size() >= reg.max_modality)
        return;
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Template Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filenames.isEmpty())
        return;
    for(auto filename : filenames)
        load_template(filename.toStdString());
    show_image();
}
void RegToolBox::on_ClearTemplate_clicked()
{
    file_names[1].clear();
    reg.clear_reg();
    reg.It.clear();
    reg.It.resize(reg.max_modality);
    scene[1].clear();
}
void RegToolBox::load_template(const std::string& file_name)
{
    clear_thread();
    if(!reg.load_template(file_names[1].size(),file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    if(file_names[1].empty())
    {
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg.It[0].width()));
        setup_slice_pos();
    }
    while(!reg.It[file_names[1].size()].empty())
        file_names[1].push_back(file_name);
    auto_fill();
}

extern std::vector<std::string> fa_template_list,iso_template_list;
void RegToolBox::on_OpenSubject_clicked()
{
    if(file_names[0].size() >= reg.max_modality)
        return;
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Subject Image",QDir::currentPath(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filenames.isEmpty())
        return;
    for(auto filename : filenames)
    {
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
    }
    show_image();
}

void RegToolBox::on_ClearSubject_clicked()
{
    file_names[0].clear();
    reg.clear_reg();
    reg.I.clear();
    reg.I.resize(reg.max_modality);
    scene[0].clear();
}

void RegToolBox::load_subject(const std::string& file_name)
{
    clear_thread();
    if(!reg.load_subject(file_names[0].size(),file_name.c_str()))
    {
        QMessageBox::critical(this,"ERROR",reg.error_msg.c_str());
        return;
    }
    if(file_names[0].empty())
    {
        ui->zoom_subject->setValue(width()*0.2f/(1.0f+reg.I[0].width()));
        setup_slice_pos();
    }
    while(!reg.I[file_names[0].size()].empty())
        file_names[0].push_back(file_name);
    auto_fill();
}

void RegToolBox::auto_fill(void)
{
    if(file_names[1].size() == file_names[0].size())
        return;
    std::string new_file_name;
    if(file_names[1].size() > file_names[0].size())
    {
        if(!file_names[0].empty() &&
           tipl::match_files(file_names[1].front(),file_names[1][file_names[0].size()],
                             file_names[0].front(),new_file_name) &&
           std::filesystem::exists(new_file_name) &&
           QMessageBox::question(this,QApplication::applicationName(),QString("load subject ") + new_file_name.c_str() + "?\n",
                        QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::Yes)
            load_subject(new_file_name);
    }
    else
    {

        if(!file_names[1].empty() &&
           tipl::match_files(file_names[0].front(),file_names[0][file_names[1].size()],
                             file_names[1].front(),new_file_name) &&
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

void add_anchor(RegToolBox* host,
                tipl::vector<3> p,tipl::shape<3> shape,
                std::vector<tipl::vector<3> >& anchor)
{
    if(shape.is_valid(p))
    {
        bool remove_point = false;
        for(auto& each : anchor)
            if((each-p).length() < 8)
            {
                each = anchor.back();
                anchor.pop_back();
                remove_point = true;
                QMessageBox::information(host,QApplication::applicationName(),"anchor removed");
            }
        if(!remove_point)
        {
            QMessageBox::information(host,QApplication::applicationName(),"anchor added");
            anchor.push_back(p);
        }
    }
}

template<bool direction>
void save_warp(RegToolBox* host,
               dual_reg& reg,
               int y,
               const std::vector<std::string>& names)
{
    QString filename = QFileDialog::getSaveFileName(
            host,("Save warped " + QFileInfo(names[y].c_str()).fileName()),names[y].c_str(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!reg.apply_warping<direction>(names[y].c_str(),filename.toStdString().c_str()))
        QMessageBox::critical(host,"ERROR",reg.error_msg.c_str());
    else
        QMessageBox::information(host,QApplication::applicationName(),"Saved");
    return;
}
bool RegToolBox::eventFilter(QObject *obj, QEvent *event)
{
    auto mouse_event = dynamic_cast<QGraphicsSceneMouseEvent*>(event);
    if(!mouse_event)
        return QObject::eventFilter(obj, event);
    auto pos = mouse_event->scenePos();
    for(size_t st : {0,1})
    {
        auto shape = st ? reg.Its : reg.Is;
        if(obj == &scene[st] && view_size[st][0])
        {
            auto x = float(pos.x()-view_border[st])/view_size[st][0];
            auto y = float(pos.y()-view_border[st])/view_size[st][1];
            if(!reg.to2from.empty() && event->type() == QEvent::GraphicsSceneMousePress && x > 1.0f && int(y) < file_names[1-st].size()) // click on the right half
            {
                if(st)
                    save_warp<true>(this,reg,y,file_names[1-st]);
                else
                    save_warp<false>(this,reg,y,file_names[1-st]);
            }

            float zoom = st ? ui->zoom_template->value() : ui->zoom_subject->value();
            tipl::vector<3> p = tipl::slice2space<tipl::vector<3> >(cur_view,
                                 float(int(pos.x()-view_border[st])%view_size[st][0])/zoom,
                                 float(int(pos.y()-view_border[st])%view_size[st][1])/zoom,
                                 float(ui->slice_pos->value())/ui->slice_pos->maximum()*(shape[cur_view]-1));
            if(cur_view != 2)
                p[2] = shape[2]-p[2];

            ui->statusBar->showMessage(QString("(%1,%2,%3)").arg(p[0]).arg(p[1]).arg(p[2]));

            if(event->type() == QEvent::GraphicsSceneMousePress && x < 1.0f && x > 0.0f && ui->anchor->isChecked())
            {
                add_anchor(this,p,shape,reg.anchor[st]);
                show_image();
                return true;
            }
        }
    }
    return QObject::eventFilter(obj, event);
}


template<int dim,typename vtype>
struct nonlinear_warped_image : public tipl::shape<dim>{
    typedef vtype value_type;
    tipl::const_pointer_image<dim,vtype> I;
    tipl::const_pointer_image<dim,tipl::vector<dim> > mapping;
    tipl::transformation_matrix<float,dim> trans;
    template<typename T,typename U,typename V,typename W>
    nonlinear_warped_image(const T& s,const U& I_,const V& mapping_,const W& trans_):tipl::shape<dim>(s)
    {
        I = I_;
        mapping = mapping_;
        trans = trans_;
    }
    value_type at(tipl::vector<dim> xyz) const
    {
        if(!mapping.empty() && mapping.shape().is_valid(xyz))
            xyz = mapping.at(xyz);
        else
            trans(xyz);
        tipl::vector<dim,int> pos(xyz+0.5f);
        if(I.shape().is_valid(pos))
            return I.at(pos);
        return 0;
    }
    const auto& shape(void) const{return *this;}
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
    size_t row_count = std::max<int>(file_names[1].size(),file_names[0].size());
    if(!row_count)
        return;
    tipl::grayscale_image view_image[2];
    std::mutex mutex[2];
    float slice_pos = float(ui->slice_pos->value())/ui->slice_pos->maximum();
    double zoom[2] = {ui->zoom_subject->value(),ui->zoom_template->value()};
    tipl::par_for(row_count,[&](size_t id)
    {
        for(size_t st : {0,1})
        {
            tipl::grayscale_image I;
            const auto& reg_I = st ? reg.It:reg.I;
            const auto& reg_I2 = st ? reg.I:reg.It;
            const auto& mapping = st ? reg.to2from:reg.from2to;
            auto reg_Is = st ? reg.Its:reg.Is;
            auto T = st ? reg.T() : reg.invT();
            if(!reg_I[0].empty())
            {
                if(!thread.running && id < J[1-st].size())
                    I = show_slice_at(reg_I[id],J[1-st][id],slice_pos*(reg_Is[cur_view]-1),zoom[st],cur_view,blend_style());
                else
                    I = show_slice_at(reg_I[id],nonlinear_warped_image<3,unsigned char>(reg_Is,reg_I2[id],mapping,T),
                                       slice_pos*(reg_Is[cur_view]-1),zoom[st],cur_view,blend_style());
                {
                    std::lock_guard<std::mutex> lock(mutex[st]);
                    if(view_image[st].empty())
                    {
                        view_size[st] = I.shape();
                        view_size[st][0] /= 2;
                        view_image[st].resize(tipl::shape<2>(I.width(),I.height()*row_count));
                    }
                }
                tipl::draw(I,view_image[st],tipl::vector<2,int>(0,id*I.height()));
            }
        }
    },row_count*2);


    auto add_text = [&](QGraphicsScene& scene,const std::vector<std::string>& list,
                        int view_border,int width,int height,bool left)
    {
        std::vector<QGraphicsTextItem*> names;
        for(auto each : list)
        {
            names.push_back(scene.addText(QFileInfo(each.c_str()).fileName()));
            names.back()->setRotation(270);
            names.back()->setPos(left ? 0 : view_border + width,
                                 view_border + height*(float(names.size()-0.5f)/float(row_count)) + names.back()->boundingRect().width()/2);
        }
    };

    std::string caption[2] = {"subject","template"};
    for(size_t st : {0,1})
    {
        scene[st].clear();
        if(!file_names[st].empty())
        {
            QImage color_image;
            color_image << tipl::color_image(view_image[st]);
            if(!reg.anchor[st].empty() && ui->anchor->isChecked())
            {
                QPainter painter(&color_image);
                QPen pen(Qt::red);
                pen.setWidth(1);
                painter.setPen(pen);
                QFont font("Arial", 10);
                painter.setFont(font);
                auto reg_Is = st ? reg.Its:reg.Is;
                int slice_location = slice_pos*(reg_Is[cur_view]-1);

                for(size_t i = 0;i < reg.anchor[st].size();++i)
                {
                    auto pos = tipl::space2slice<tipl::vector<3> >(cur_view,reg.anchor[st][i]);
                    auto dis = std::fabs(pos[2]-slice_location);
                    if(dis > 5.0f)
                        continue;
                    dis = 5.0f-dis;
                    auto x = zoom[st]*(pos[0]);
                    auto y = zoom[st]*((cur_view == 2) ? pos[1] : reg_Is[2]-pos[1]);
                    painter.drawLine(x - dis, y, x + dis, y);
                    painter.drawLine(x, y - dis, x, y + dis);
                    painter.drawText(x + dis + 2, y + 3, QString::number(i));
                }
            }
            scene[st].blockSignals(true);
            auto top_text1 = scene[st].addText(caption[st].c_str());
            auto top_text2 = scene[st].addText((caption[1-st]+"->"+caption[st]).c_str());
            view_border[st] = top_text1->boundingRect().height();
            scene[st].addPixmap(QPixmap::fromImage(color_image))->setPos(view_border[st], view_border[st]);
            top_text1->setPos(view_border[st] + view_image[st].width() * 0.25 - top_text1->boundingRect().width()/2,0);
            top_text2->setPos(view_border[st] + view_image[st].width() * 0.75 - top_text2->boundingRect().width()/2,0);
            add_text(scene[st],file_names[st],view_border[st],view_image[st].width(),view_image[st].height(),true);
            if(blend_style() != 1)
                add_text(scene[st],file_names[1-st],view_border[st],view_image[st].width(),view_image[st].height(),false);
            scene[st].blockSignals(false);
            scene[st].setSceneRect(0, 0, view_border[st]+view_border[st]+view_image[st].width(),view_border[st] + view_image[st].height());
        }
    }


}
extern console_stream console;

void RegToolBox::on_timer()
{
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
    on_rb_switch_clicked();
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
    reg.param.resolution = ui->resolution->value();
    reg.param.gradient_type = ui->gradient->text().toStdString();
    reg.bound = ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound;
    reg.cost_type = ui->cost_fun->currentIndex() == 0 ? tipl::reg::corr : tipl::reg::mutual_info;
    reg.use_cuda = ui->use_cuda->isChecked();
    reg.skip_linear = ui->skip_linear->isChecked();
    reg.skip_nonlinear = ui->skip_nonlinear->isChecked();

    {
        reg.match_resolution(false);
        ui->zoom_template->setValue(width()*0.2f/(1.0f+reg.It[0].width()));
        J[0].clear();
        J[1].clear();
        reg.to2from.clear();
        reg.from2to.clear();
        thread.run([this](void){
            reg.linear_reg(thread.terminated);
            reg.nonlinear_reg(thread.terminated);
            size_t modality_count = 0;
            for(size_t i = 0;i < reg.I.size();++i)
                if(!reg.I[i].empty() && !reg.It[i].empty())
                    modality_count = i+1;
            J[0].swap(reg.J);
            J[0].resize(modality_count);
            J[1].resize(modality_count);
            tipl::par_for(modality_count,[&](size_t i)
            {
                J[1][i] = tipl::compose_mapping(reg.It[i],reg.from2to);
            },modality_count);

        });
    }

    ui->running_label->movie()->start();
    ui->running_label->show();
    timer->start();
    ui->stop->show();
    ui->run_reg->hide();

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

void RegToolBox::on_rb_switch_clicked()
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





template <bool subjectToTemplate>
void applyWarping(dual_reg& reg)
{
    QString filter = "Images (*.nii *nii.gz);;Tracts (*tt.gz);;All files (*)";
    QStringList file_list = QFileDialog::getOpenFileNames(nullptr,
                                                          subjectToTemplate ? "Open Subject Image" : "Open Template Image",
                                                          QDir::currentPath(), filter);
    if (file_list.isEmpty())
        return;
    if (file_list.size() == 1)
    {
        QString saveFileName = QFileDialog::getSaveFileName(nullptr, "Save Transformed Image", file_list[0], filter);
        if (saveFileName.isEmpty())
            return;
        if (!reg.apply_warping<subjectToTemplate>(file_list[0].toStdString().c_str(), saveFileName.toStdString().c_str()))
            goto error;
    }
    else
    {
        tipl::progress prog("save files");
        for (int i = 0; prog(i, file_list.size()); ++i)
        {
            if (!reg.apply_warping<subjectToTemplate>(file_list[i].toStdString().c_str(), (file_list[i] + ".wp.nii.gz").toStdString().c_str()))
                goto error;
        }
    }
    QMessageBox::information(nullptr, QApplication::applicationName(), "Saved");
    return;
    error:
    QMessageBox::critical(nullptr, "ERROR", reg.error_msg.c_str());
}

void RegToolBox::on_actionApply_Subject_To_Template_Warping_triggered()
{
    applyWarping<true>(reg);
}

void RegToolBox::on_actionApply_Template_To_Subject_Warping_triggered()
{
    applyWarping<false>(reg);
}


void RegToolBox::on_actionSet_Template_Size_triggered()
{
    bool okay = false;
    auto text = QInputDialog::getText(this,QApplication::applicationName(),"input template dimension",QLineEdit::Normal,
                                               QString::number(reg.Its[0])+" "+QString::number(reg.Its[1])+" "+QString::number(reg.Its[2]),&okay);
    if(!okay)
        return;
    int w(0),h(0),d(0);
    std::istringstream(text.toStdString()) >> w >> h >> d;
    if(w*h*d)
        reg.to_It_space(tipl::shape<3>(w,h,d));
    setup_slice_pos();
    show_image();
}

void RegToolBox::on_actionSet_Subject_Dimension_triggered()
{
    bool okay = false;
    auto text = QInputDialog::getText(this,QApplication::applicationName(),"input subject dimension",QLineEdit::Normal,
                                               QString::number(reg.Is[0])+" "+QString::number(reg.Is[1])+" "+QString::number(reg.Is[2]),&okay);
    if(!okay)
        return;
    int w(0),h(0),d(0);
    std::istringstream(text.toStdString()) >> w >> h >> d;
    if(w*h*d)
        reg.to_I_space(tipl::shape<3>(w,h,d));
    setup_slice_pos();
    show_image();

}


void RegToolBox::on_actionSave_Subject_Images_triggered()
{
    if(file_names[0].empty())
        return;
    QString from = QFileDialog::getSaveFileName(
            this,"Save Subject Images",file_names[0].front().c_str(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(from.isEmpty())
        return;
    reg.save_subject(from.toStdString());
}


void RegToolBox::on_actionSave_Template_Images_triggered()
{
    if(file_names[1].empty())
        return;
    QString from = QFileDialog::getSaveFileName(
            this,"Save Template Images",file_names[1].front().c_str(),
            "Images (*.nii *nii.gz);;All files (*)" );
    if(from.isEmpty())
        return;
    reg.save_template(from.toStdString());
}
void RegToolBox::on_anchor_toggled(bool checked)
{
    ui->I_view->setCursor(checked ? Qt::CrossCursor:Qt::ArrowCursor);
    ui->It_view->setCursor(checked ? Qt::CrossCursor:Qt::ArrowCursor);
}




