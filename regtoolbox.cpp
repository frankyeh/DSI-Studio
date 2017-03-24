#include <QMessageBox>
#include <QFileDialog>
#include "regtoolbox.h"
#include "ui_regtoolbox.h"
#include "libs/gzip_interface.hpp"
#include "basic_voxel.hpp"
void show_view(QGraphicsScene& scene,QImage I);
RegToolBox::RegToolBox(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RegToolBox)
{
    ui->setupUi(this);
    ui->It_view->setScene(&It_scene);
    ui->I_view->setScene(&I_scene);
    ui->main_view->setScene(&main_scene);
    connect(ui->slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->template_view, SIGNAL(clicked()), this, SLOT(show_image()));
    connect(ui->main_zoom, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(on_timer()));
    timer->setInterval(2500);

}

RegToolBox::~RegToolBox()
{
    delete ui;
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
    image::normalize(It,1.0);
    nifti.get_voxel_size(Itvs.begin());
    ui->slice_pos->setMaximum(It.depth()-1);
    ui->slice_pos->setValue(It.depth()/2);
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
    image::normalize(I,1.0);
    nifti.get_voxel_size(Ivs.begin());
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
        show_slice_at(It_scene,It,cIt,ui->slice_pos->value(),1);
    if(!I.empty())
        show_slice_at(I_scene,I,cI,std::min(I.depth()-1,I.depth()*ui->slice_pos->value()/It.depth()),1);

    if(J_view2.empty())
    {
        if(ui->template_view->isChecked() && !It.empty())
            show_slice_at(main_scene,It,cJ,ui->slice_pos->value(),ratio);
        else
        if(!J_view.empty())
            show_slice_at(main_scene,J_view,cJ,ui->slice_pos->value(),ratio);
        else
            if(!I.empty())
                show_slice_at(main_scene,I,cJ,std::min(I.depth()-1,I.depth()*ui->slice_pos->value()/It.depth()),ratio);
    }
    else
    {
        image::basic_image<float,2> J_view_slice(image::geometry<2>(J_view.width() << 1,J_view.height()));
        image::draw(J_view.slice_at(ui->slice_pos->value()),J_view_slice,image::vector<2,int>(0,0));
        if(ui->template_view->isChecked())
            image::draw(It.slice_at(ui->slice_pos->value()),J_view_slice,image::vector<2,int>(J_view.width(),0));
        else
            image::draw(J_view2.slice_at(ui->slice_pos->value()),J_view_slice,image::vector<2,int>(J_view.width(),0));

        image::normalize(J_view_slice,255.0);
        image::upper_lower_threshold(J_view_slice.begin(),J_view_slice.end(),J_view_slice.begin(),0.0f,255.0f);
        cJ = J_view_slice;
        QImage qcJ = QImage((unsigned char*)&*cJ.begin(),cJ.width(),cJ.height(),QImage::Format_RGB32).
                      scaled(cJ.width()*ratio,cJ.height()*ratio);

        QPainter paint(&qcJ);
        paint.setBrush(Qt::NoBrush);
        paint.setPen(Qt::red);
        auto& dis_slice = dis_view.slice_at(ui->slice_pos->value());
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

        show_view(main_scene,qcJ);
    }
}

void RegToolBox::on_timer()
{
    if(!linear_done)
    {
        J_view.resize(It.geometry());
        image::resample_mt(I,J_view,image::transformation_matrix<double>(linear_reg,It.geometry(),Itvs,I.geometry(),Ivs),image::linear);
        show_image();
    }
    if(!dis.empty())
    {
        dis_view = dis;
        J_view = J;
        std::vector<image::geometry<3> > geo_stack;
        while(J_view.width() > dis_view.width())
        {
            geo_stack.push_back(J_view.geometry());
            image::reg::cdm_downsample(J_view,J_view);
        }
        if(J_view.geometry() == dis_view.geometry())
            image::compose_displacement(J_view,dis_view,J_view2);
        while(!geo_stack.empty())
        {
            image::reg::cdm_upsample(J_view,J_view,geo_stack.back());
            image::reg::cdm_upsample(J_view2,J_view2,geo_stack.back());
            image::reg::cdm_upsample(dis_view,dis_view,geo_stack.back());
            geo_stack.pop_back();
        }
        image::normalize(J_view,1.0);
        image::normalize(J_view2,1.0);
        show_image();
    }
    if(reg_done)
    {
        QMessageBox::information(this,"Registration","Complete");
        timer->stop();
    }
}

void RegToolBox::on_run_reg_clicked()
{

    //image::reg::get_bound(It,I,linear_reg.get_arg(),b_upper,b_lower,reg_type_);
    running_type = ui->reg_type->currentIndex();
    thread.run([this]()
    {
        float speed = ui->speed->value();
        float resolution = ui->resolution->value();
        linear_done = false;
        reg_done = false;
        linear_reg.clear();
        dis.clear();
        J.clear();
        JJ.clear();

        image::basic_image<float,3> J2(It.geometry());
        if(It.geometry() != I.geometry() || running_type == 0)
        {
            image::reg::reg_type linear_type[2] = { image::reg::rigid_scaling,image::reg::affine};
            if(ui->cost_func->currentIndex() == 1) // MI
            {
                image::reg::linear(It,Itvs,I,Ivs,linear_reg,linear_type[ui->linear_type->currentIndex()],image::reg::mutual_information(),thread.terminated);
                image::reg::linear(It,Itvs,I,Ivs,linear_reg,linear_type[ui->linear_type->currentIndex()],image::reg::mutual_information(),thread.terminated);
            }
            if(ui->cost_func->currentIndex() == 0) // correlation
            {
                image::reg::linear(It,Itvs,I,Ivs,linear_reg,linear_type[ui->linear_type->currentIndex()],image::reg::mt_correlation<image::basic_image<float,3>,
                               image::transformation_matrix<double> >(0),thread.terminated);
                image::reg::linear(It,Itvs,I,Ivs,linear_reg,linear_type[ui->linear_type->currentIndex()],image::reg::mt_correlation<image::basic_image<float,3>,
                               image::transformation_matrix<double> >(0),thread.terminated);
            }
            image::resample_mt(I,J2,
                               image::transformation_matrix<double>(linear_reg,It.geometry(),Itvs,I.geometry(),Ivs),image::cubic);
        }
        else
            J2 = I;
        linear_done = true;

        std::pair<double,double> r = image::linear_regression(J2.begin(),J2.end(),It.begin());
        for(unsigned int index = 0;index < J2.size();++index)
            J2[index] = std::max<float>(0,J2[index]*r.first+r.second);

        J = J2;

        if(running_type > 0) // nonlinear
        {
            image::reg::cdm(It,J2,dis,speed,thread.terminated,resolution);
            image::compose_displacement(J2,dis,JJ);
        }
        reg_done = true;
    });
    timer->start();

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
