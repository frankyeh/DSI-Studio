#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include "regtoolbox.h"
#include "ui_regtoolbox.h"
#include "libs/gzip_interface.hpp"
#include "basic_voxel.hpp"

#include "mapping/fa_template.hpp"
extern fa_template fa_template_imp;


void show_view(QGraphicsScene& scene,QImage I);
RegToolBox::RegToolBox(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RegToolBox)
{
    ui->setupUi(this);
    ui->It_view->setScene(&It_scene);
    ui->I_view->setScene(&I_scene);
    connect(ui->slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->main_zoom, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->show_warp, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->dis_map, SIGNAL(clicked()), this, SLOT(show_image()));

    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(on_timer()));
    timer->setInterval(2000);

    QMovie *movie = new QMovie(":/data/icons/ajax-loader.gif");
    ui->running_label->setMovie(movie);
    ui->running_label->hide();
    ui->stop->hide();

}

RegToolBox::~RegToolBox()
{
    delete ui;
}

void RegToolBox::clear(void)
{
    thread.clear();
    reg_done = false;
    J.clear();
    J_view.clear();
    J_view2.clear();
    dis.clear();
    arg.clear();
    ui->run_reg->setText("run");
}

void RegToolBox::on_OpenTemplate_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::information(this,"Error","Invalid file format");
        return;
    }
    nifti.toLPS(It);
    nifti.get_image_transformation(ItR);
    It *= 1.0f/image::mean(It);
    nifti.get_voxel_size(Itvs.begin());
    ui->slice_pos->setMaximum(It.depth()-1);
    ui->slice_pos->setValue(It.depth()/2);
    clear();
    show_image();
}

void RegToolBox::on_OpenSubject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::information(this,"Error","Invalid file format");
        return;
    }
    nifti.toLPS(I);
    I *= 1.0f/image::mean(I);
    nifti.get_voxel_size(Ivs.begin());
    clear();
    show_image();
}


void show_slice_at(QGraphicsScene& scene,image::basic_image<float,2>& tmp,image::color_image& buf,float ratio)
{
    image::normalize(tmp,255.0);
    image::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);
    buf = tmp;
    show_view(scene,QImage((unsigned char*)&*buf.begin(),buf.width(),buf.height(),QImage::Format_RGB32).
              scaled(buf.width()*ratio,buf.height()*ratio));
}


void show_slice_at(QGraphicsScene& scene,const image::basic_image<float,3>& source,image::color_image& buf,int slice_pos,float ratio)
{
    image::basic_image<float,2> tmp;
    image::reslicing(source,tmp,2,slice_pos);
    show_slice_at(scene,tmp,buf,ratio);
}

void RegToolBox::show_image(void)
{
    float ratio = ui->main_zoom->value()/10.0;
    if(!It.empty())
    {
        if(ui->show_warp->isChecked())
        {
            if(!J_view2.empty())
                show_slice_at(It_scene,J_view2,cIt,ui->slice_pos->value(),ratio);
            else
                if(!J_view.empty())
                    show_slice_at(It_scene,J_view,cIt,ui->slice_pos->value(),ratio);
                else
                    show_slice_at(It_scene,It,cIt,ui->slice_pos->value(),ratio);
        }
        else
            show_slice_at(It_scene,It,cIt,ui->slice_pos->value(),ratio);
    }
    if(!J_view2.empty())
    {
        if(ui->dis_map->isChecked())
        {
            image::basic_image<float,2> J_view_slice(J_view.slice_at(ui->slice_pos->value()));
            image::normalize(J_view_slice,255.0);
            image::upper_lower_threshold(J_view_slice.begin(),J_view_slice.end(),J_view_slice.begin(),0.0f,255.0f);
            cJ = J_view_slice;
            QImage qcJ = QImage((unsigned char*)&*cJ.begin(),cJ.width(),cJ.height(),QImage::Format_RGB32).
                          scaled(cJ.width()*ratio,cJ.height()*ratio);

            QPainter paint(&qcJ);
            paint.setBrush(Qt::NoBrush);
            paint.setPen(Qt::red);
            auto dis_slice = dis_view.slice_at(ui->slice_pos->value());
            int cur_dis = 1 << (ui->dis_spacing->currentIndex()-1);
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
                    paint.drawLine(vfrom[0]*ratio,vfrom[1]*ratio,
                                   vto[0]*ratio,vto[1]*ratio);
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
                    paint.drawLine(vfrom[0]*ratio,vfrom[1]*ratio,
                                   vto[0]*ratio,vto[1]*ratio);
                }
            }
            show_view(I_scene,qcJ);
        }
        else
        show_slice_at(I_scene,J_view2,cJ,ui->slice_pos->value(),ratio);
    }
    else
    if(!J_view.empty())
        show_slice_at(I_scene,J_view,cJ,ui->slice_pos->value(),ratio);
    else
        if(!I.empty())
            show_slice_at(I_scene,I,cI,std::min(I.depth()-1,I.depth()*ui->slice_pos->value()/ui->slice_pos->maximum()),ratio);

    /*
    if(J_view2.empty())
        return;
    {

    }
    */
}

void RegToolBox::on_timer()
{
    {
        if(J.empty()) // linear registration
        {
            J_view.resize(It.geometry());
            image::resample_mt(I,J_view,image::transformation_matrix<double>(arg,It.geometry(),Itvs,I.geometry(),Ivs),image::linear);
            show_image();
        }
        else // nonlinear
        {
            if(!dis.empty())
            {
                dis_view = dis;
                J_view = J;
                std::vector<image::geometry<3> > geo_stack;
                while(J_view.width() > dis_view.width())
                {
                    geo_stack.push_back(J_view.geometry());
                    image::downsample_with_padding(J_view,J_view);
                }
                if(J_view.geometry() != dis_view.geometry())
                    return;
                image::compose_displacement(J_view,dis_view,J_view2);
                while(!geo_stack.empty())
                {
                    image::upsample_with_padding(J_view,J_view,geo_stack.back());
                    image::upsample_with_padding(J_view2,J_view2,geo_stack.back());
                    image::upsample_with_padding(dis_view,dis_view,geo_stack.back());
                    dis_view *= 2.0f;
                    geo_stack.pop_back();
                }
                show_image();
            }
        }
        if(reg_done)
        {
            timer->stop();
            ui->running_label->movie()->stop();
            ui->running_label->hide();
            ui->stop->hide();
            ui->run_reg->show();
            ui->run_reg->setText("re-run");
        }
    }
}

void RegToolBox::linear_reg(image::reg::reg_type reg_type)
{
    status = "linear registration";
    image::reg::linear_mr(It,Itvs,I,Ivs,arg,reg_type,image::reg::mutual_information(),thread.terminated);
    status += "..first round done";
    image::reg::linear_mr(It,Itvs,I,Ivs,arg,reg_type,image::reg::mutual_information(),thread.terminated);
    status += "..second round done";
    image::basic_image<float,3> J2(It.geometry());
    image::resample_mt(I,J2,image::transformation_matrix<double>(arg,It.geometry(),Itvs,I.geometry(),Ivs),image::cubic);
    J.swap(J2);
}

void RegToolBox::nonlinear_reg(int method)
{
    status = "nonlinear registration";
    if(method == 1)
    {
        image::reg::cdm(It,J,dis,thread.terminated,2.0f,ui->smoothness->value());
    }
    if(method == 0)
    {
        int order = ui->order->value();
        bnorm_data.reset(new image::reg::bfnorm_mapping<float,3>(It.geometry(),image::geometry<3>(7*order,9*order,7*order)));
        image::reg::bfnorm(*bnorm_data.get(),It,J,thread.terminated,std::thread::hardware_concurrency());
        image::basic_image<image::vector<3>,3> dis2(It.geometry());
        status += "..output mapping";
        dis2.for_each_mt([&](image::vector<3>& v,image::pixel_index<3>& index){
            bnorm_data->get_displacement(index,v);
        });
        dis2.swap(dis);
    }
    image::compose_displacement(J,dis,JJ);
}

void RegToolBox::on_run_reg_clicked()
{
    clear();
    thread.terminated = false;
    reg_type = ui->reg_type->currentIndex();
    switch(reg_type)
    {
    case 0: // rigid body
        thread.run([this]()
        {
            linear_reg(image::reg::rigid_body);
            reg_done = true;
            status = "registration done";
        });
        break;
    case 1: // linear + nonlinear
        thread.run([this]()
        {
            linear_reg(image::reg::affine);
            nonlinear_reg(ui->reg_method->currentIndex());
            reg_done = true;
            status = "registration done";
        });
        break;
    case 2: // nonlinear only
        thread.run([this]()
        {
            if(I.geometry() != It.geometry())
                linear_reg(image::reg::affine);
            else
                J = I;
            nonlinear_reg(ui->reg_method->currentIndex());
            reg_done = true;
            status = "registration done";
        });
        break;
    }

    ui->running_label->movie()->start();
    ui->running_label->show();
    timer->start();
    ui->stop->show();
    ui->run_reg->hide();
}

void RegToolBox::on_action_Save_Warpped_Image_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Warpped Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!JJ.empty())
    {
        gz_nifti nii;
        nii.set_voxel_size(Itvs);
        nii.set_image_transformation(ItR);
        nii << JJ;
        nii.save_to_file(filename.toStdString().c_str());
        return;
    }
    if(!J.empty())
    {
        gz_nifti nii;
        nii.set_voxel_size(Itvs);
        nii.set_image_transformation(ItR);
        nii << J;
        nii.save_to_file(filename.toStdString().c_str());

    }

}


void RegToolBox::on_reg_type_currentIndexChanged(int index)
{
    if(index)
        ui->nonlinear_widget->show();
    else
        ui->nonlinear_widget->hide();

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

void RegToolBox::on_reg_method_currentIndexChanged(int index)
{
    if(index) // diffe
    {
        ui->order_widget->hide();
        ui->smoothness_widget->show();
    }
    else
    {
        ui->order_widget->show();
        ui->smoothness_widget->hide();
    }
}

void RegToolBox::on_actionRemove_Skull_triggered()
{
    if(!It.empty())
    {
        image::vector<3> from(fa_template_imp.shift);
        from[0] -= (int)fa_template_imp.I.width()+ItR[3]-(int)It.width();
        from[1] -= (int)fa_template_imp.I.height()+ItR[7]-(int)It.height();
        from[2] -= ItR[11];

        It.for_each_mt([&](float& v,const image::pixel_index<3>& pos){
           image::vector<3> p(pos);
           p -= from;
           p.round();
           if(fa_template_imp.mask.geometry().is_valid(p) && fa_template_imp.mask.at(p[0],p[1],p[2]))
               return;
           v = 0.0f;
        });
        if(!J.empty())
        {

            if(!dis.empty())
            {
                image::transformation_matrix<double> T(arg,It.geometry(),Itvs,I.geometry(),Ivs);
                T.inverse();
                image::basic_image<image::vector<3>,3>  inv_dis;
                image::invert_displacement(dis,inv_dis);
                I.for_each_mt([&](float& v,const image::pixel_index<3>& pos)
                {
                    image::vector<3> p(pos),p2;
                    T(p);
                    image::estimate(inv_dis,p,p2);
                    p += p2;
                    p.round();
                    if(It.geometry().is_valid(p) && It.at(p[0],p[1],p[2]) != 0)
                        return;
                    v = 0.0f;
                });
            }
        }

    }

    show_image();
}
