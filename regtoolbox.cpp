#include <QMessageBox>
#include <QMovie>
#include <QFileDialog>
#include "regtoolbox.h"
#include "ui_regtoolbox.h"
#include "libs/gzip_interface.hpp"
#include "basic_voxel.hpp"
bool is_label_image(const tipl::image<float,3>& I);

void show_view(QGraphicsScene& scene,QImage I);
RegToolBox::RegToolBox(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RegToolBox)
{
    ui->setupUi(this);
    ui->options->hide();
    ui->It_view->setScene(&It_scene);
    ui->It_mix_view->setScene(&It_mix_scene);
    ui->I_view->setScene(&I_scene);
    connect(ui->slice_pos, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->contrast, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));
    connect(ui->main_zoom, SIGNAL(sliderMoved(int)), this, SLOT(show_image()));

    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(on_timer()));
    timer->setInterval(2000);

    flash_timer.reset(new QTimer());
    connect(flash_timer.get(), SIGNAL(timeout()), this, SLOT(flash_image()));
    flash_timer->setInterval(1000);
    flash_timer->start();
    QMovie *movie = new QMovie(":/data/icons/ajax-loader.gif");
    ui->running_label->setMovie(movie);
    ui->running_label->hide();
    ui->stop->hide();

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
    J.clear();
    J_view.clear();
    J_view2.clear();
    dis.clear();
    arg.clear();
    ui->run_reg->setText("run");
}
void RegToolBox::setup_slice_pos(void)
{
    if(!It.empty())
    {
        int range = int(It.geometry()[cur_view]);
        ui->slice_pos->setMaximum(range-1);
        ui->slice_pos->setValue(range/2);
    }
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
    //tipl::swap_xy(It);
    nifti.get_image_transformation(ItR);
    tipl::normalize(It,1.0f);
    nifti.get_voxel_size(Itvs);
    setup_slice_pos();
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
    ui->edge->setChecked(is_label_image(I));
    tipl::normalize(I,1.0f);
    nifti.get_voxel_size(Ivs);
    clear();
    show_image();
}


void RegToolBox::on_OpenSubject2_clicked()
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
    nifti.toLPS(I2);
    I2 *= 1.0f/tipl::mean(I2);
}

void RegToolBox::on_OpenTemplate2_clicked()
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
    nifti.toLPS(It2);
    It2 *= 1.0f/tipl::mean(It2);
}


void image2rgb(tipl::image<float,2>& tmp,tipl::color_image& buf,float contrast)
{
    tmp *= contrast;
    tipl::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);
    buf = tmp;
}


void show_slice_at(QGraphicsScene& scene,tipl::image<float,2>& tmp,tipl::color_image& buf,float ratio,float contrast,uint8_t cur_view)
{
    image2rgb(tmp,buf,contrast);
    QImage I(reinterpret_cast<unsigned char*>(&*buf.begin()),buf.width(),buf.height(),QImage::Format_RGB32);
    if(cur_view != 2)
        I = I.mirrored(false,true);
    show_view(scene,I.scaled(int(buf.width()*ratio),int(buf.height()*ratio)));
}


void show_slice_at(QGraphicsScene& scene,const tipl::image<float,3>& source,
                   tipl::color_image& buf,size_t slice_pos,float ratio,float contrast,uint8_t cur_view)
{
    tipl::image<float,2> tmp;
    tipl::volume2slice(source,tmp,cur_view,slice_pos);
    show_slice_at(scene,tmp,buf,ratio,contrast,cur_view);
}
void show_mosaic_slice_at(QGraphicsScene& scene,
                          const tipl::image<float,3>& source1,
                          const tipl::image<float,3>& source2,
                          tipl::color_image& buf,size_t slice_pos,float ratio,float contrast,bool mosaic,uint8_t cur_view)
{
    if(!mosaic)
    {
        show_slice_at(scene,source1,buf,slice_pos,ratio,contrast,cur_view);
        return;
    }
    tipl::image<float,2> tmp1,tmp2,tmp;
    tipl::volume2slice(source1,tmp1,cur_view,slice_pos);
    tipl::volume2slice(source2,tmp2,cur_view,slice_pos);
    if(tmp1.geometry() != tmp2.geometry())
        return;
    tmp.resize(tmp1.geometry());
    tmp.for_each([&](float& v,tipl::pixel_index<2>& index)
    {
        if(!(index[0] & 31) || !(index[1] & 31))
        {
            v = 0;
            return;
        }
        int x = index[0] >> 5;
        int y = index[1] >> 5;
        v = (x&1 ^ y&1) ? tmp1[index.index()] : tmp2[index.index()];
    });
    show_slice_at(scene,tmp,buf,ratio,contrast,cur_view);
}
void RegToolBox::flash_image()
{
    flash = !flash;
    if(ui->rb_flash->isChecked())
        show_image();
}
void RegToolBox::show_image(void)
{
    float ratio = ui->main_zoom->value()/10.0;
    float contrast = ui->contrast->value()*20;
    if(!It.empty())
    {
        const auto& I_show = (ui->show_second->isChecked() && It2.geometry() == It.geometry() ? It2 : It);

        if(ui->rb_mosaic->isChecked())
        {
            if(!J_view2.empty())
                show_mosaic_slice_at(It_mix_scene,J_view2,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,true,cur_view);
            else
                if(!J_view.empty())
                    show_mosaic_slice_at(It_mix_scene,J_view,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,true,cur_view);
        }
        if(ui->rb_flash->isChecked())
        {
            if(flash)
            {
                if(!J_view2.empty())
                    show_mosaic_slice_at(It_mix_scene,J_view2,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,false,cur_view);
                else
                    if(!J_view.empty())
                        show_mosaic_slice_at(It_mix_scene,J_view,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,false,cur_view);
                    else
                        show_slice_at(It_mix_scene,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,cur_view);
            }
            else
            {
                show_slice_at(It_mix_scene,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,cur_view);
            }
        }
        if(ui->rb_show_warp->isChecked())
        {
            if(ui->dis_spacing->currentIndex())
            {
                tipl::image<float,2> J_view_slice;
                tipl::volume2slice(J_view,J_view_slice,cur_view,ui->slice_pos->value());
                image2rgb(J_view_slice,cJ,contrast);
                QImage qcJ = QImage((unsigned char*)&*cJ.begin(),cJ.width(),cJ.height(),QImage::Format_RGB32).
                              scaled(cJ.width()*ratio,cJ.height()*ratio);

                QPainter paint(&qcJ);
                paint.setBrush(Qt::NoBrush);
                paint.setPen(Qt::red);
                tipl::image<tipl::vector<3>,2> dis_slice;
                tipl::volume2slice(dis_view,dis_slice,cur_view,ui->slice_pos->value());
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
                if(cur_view != 2)
                    qcJ = qcJ.mirrored(false,true);
                show_view(It_mix_scene,qcJ);
            }
            else
            {
                if(!J_view2.empty())
                    show_mosaic_slice_at(It_mix_scene,J_view2,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,false,cur_view);
                else
                    if(!J_view.empty())
                        show_mosaic_slice_at(It_mix_scene,J_view,I_show,cIt_mix,ui->slice_pos->value(),ratio,contrast,false,cur_view);
            }
        }
        show_slice_at(It_scene,I_show,cIt,ui->slice_pos->value(),ratio,contrast,cur_view);
    }


    if(!I.empty())
        show_slice_at(I_scene,I,cI,std::min(I.depth()-1,I.depth()*ui->slice_pos->value()/ui->slice_pos->maximum()),ratio,contrast,cur_view);

}

void RegToolBox::on_timer()
{
    {
        if(J.empty()) // linear registration
        {
            J_view.resize(It.geometry());
            tipl::resample_mt((ui->show_second->isChecked() && I2.geometry() == I.geometry() ? I2 : I),
                              J_view,tipl::transformation_matrix<double>(arg,It.geometry(),Itvs,I.geometry(),Ivs),tipl::linear);
            show_image();
        }
        else // nonlinear
        {
            if(!dis.empty())
            {
                dis_view = dis;
                J_view = (ui->show_second->isChecked() && J2.geometry() == J.geometry() ? J2 : J);
                std::vector<tipl::geometry<3> > geo_stack;
                while(J_view.width() > dis_view.width())
                {
                    geo_stack.push_back(J_view.geometry());
                    tipl::downsample_with_padding(J_view,J_view);
                }
                if(J_view.geometry() != dis_view.geometry())
                    return;
                tipl::compose_displacement(J_view,dis_view,J_view2);
                while(!geo_stack.empty())
                {
                    tipl::upsample_with_padding(J_view,J_view,geo_stack.back());
                    tipl::upsample_with_padding(J_view2,J_view2,geo_stack.back());
                    tipl::upsample_with_padding(dis_view,dis_view,geo_stack.back());
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

void RegToolBox::linear_reg(tipl::reg::reg_type reg_type)
{
    status = "linear registration";

    tipl::reg::two_way_linear_mr(It,Itvs,I,Ivs,T,reg_type,tipl::reg::mutual_information(),thread.terminated,
                                  std::thread::hardware_concurrency(),&arg,tipl::reg::large_bound);
    tipl::image<float,3> J_(It.geometry());
    tipl::resample_mt(I,J_,T,tipl::cubic);
    float r2 = tipl::correlation(J_.begin(),J_.end(),It.begin());
    std::cout << "linear:" << r2 << std::endl;
    J.swap(J_);
    J_view = J;

    if(I2.geometry() == I.geometry())
    {
        tipl::image<float,3> J2_(It.geometry());
        tipl::resample_mt(I2,J2_,T,tipl::cubic);
        J2.swap(J2_);
    }

}
double phase_estimate(const tipl::image<float,3>& It,
            const tipl::image<float,3>& Is,
            tipl::image<tipl::vector<3>,3>& d,// displacement field
            bool& terminated,
            float resolution = 2.0,
            float cdm_smoothness = 0.3f,
            unsigned int steps = 30);

void RegToolBox::nonlinear_reg(void)
{
    status = "nonlinear registration";
    {
        //phase_estimate(It,J,dis,thread.terminated,ui->resolution->value(),ui->smoothness->value(),60);
        if(ui->edge->isChecked())
        {
            tipl::image<float,3> sIt(It),sJ(J);
            tipl::filter::sobel(sIt);
            tipl::filter::sobel(sJ);
            tipl::filter::mean(sIt);
            tipl::filter::mean(sJ);
            tipl::reg::cdm(sIt,sJ,dis,thread.terminated,ui->resolution->value(),ui->smoothness->value(),ui->steps->value());
        }
        else
        {
            tipl::reg::cdm_pre(It,It2,J,J2);
            if(It2.geometry() == It.geometry() && J2.geometry() == J.geometry())
                tipl::reg::cdm2(It,It2,J,J2,dis,thread.terminated,ui->resolution->value(),ui->smoothness->value(),ui->steps->value());
            else
                tipl::reg::cdm(It,J,dis,thread.terminated,ui->resolution->value(),ui->smoothness->value(),ui->steps->value());
        }
    }
    tipl::compose_displacement(J,dis,JJ);
    std::cout << "nonlinear:" << tipl::correlation(JJ.begin(),JJ.end(),It.begin()) << std::endl;
}

void RegToolBox::on_run_reg_clicked()
{
    if(I.empty() || It.empty())
    {
        QMessageBox::information(this,"Error","Please load image first",0);
        return;
    }
    clear();
    thread.terminated = false;

    thread.run([this]()
    {
        // adjust Ivs for affine
        Ivs *= std::sqrt((It.plane_size()*Itvs[0]*Itvs[1])/
                    (I.plane_size()*Ivs[0]*Ivs[1]));
        linear_reg(tipl::reg::affine);
        /*
        // This skip affine registration
        else
        {
            J = I;
            if(I2.geometry() == I.geometry())
                J2 = I2;
        }
        */

        nonlinear_reg();
        reg_done = true;
        status = "registration done";
    });

    ui->running_label->movie()->start();
    ui->running_label->show();
    timer->start();
    ui->stop->show();
    ui->run_reg->hide();
}

bool apply_warping(const char* from,
                   const char* to,
                   tipl::geometry<3> from_geo,
                   tipl::geometry<3> to_geo,
                   tipl::image<tipl::vector<3>,3>& dis,
                   tipl::vector<3> Itvs,
                   tipl::matrix<4,4,float>& ItR,
                   tipl::transformation_matrix<double>& T,
                   std::string& error)
{
    gz_nifti nifti;
    if(!nifti.load_from_file(from))
    {
        error = "Invalid file format";
        return false;
    }
    tipl::image<float,3> I3;
    nifti.toLPS(I3);
    if(I3.geometry() != from_geo)
    {
        error = "The warping image does not match subject image dimension";
        return false;
    }
    tipl::image<float,3> J3(to_geo);
    tipl::compose_displacement(I3,J3,T,dis,is_label_image(I3) ? tipl::nearest : tipl::cubic);
    gz_nifti::save_to_file(to,J3,Itvs,ItR);
    return true;
}
void RegToolBox::on_actionApply_Warpping_triggered()
{
    QString from = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(from.isEmpty())
        return;
    QString to = QFileDialog::getSaveFileName(
            this,"Save Warpped Image",from,
            "Images (*.nii *nii.gz);;All files (*)" );
    if(to.isEmpty())
        return;
    std::string error;
    if(!apply_warping(from.toStdString().c_str(),
                      to.toStdString().c_str(),
                      I.geometry(),
                      It.geometry(),
                      dis,Itvs,ItR,T,error))
        QMessageBox::information(this,"Error",error.c_str());
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
    if(I.geometry() == It.geometry())
    {
        tipl::homogenize(I,It);
        show_image();
    }
}

void RegToolBox::on_actionRemove_Background_triggered()
{
    if(!I.empty())
    {
        I -= tipl::segmentation::otsu_threshold(I);
        tipl::lower_threshold(I,0.0);
        tipl::normalize(I,1.0);
        show_image();
    }
}



void RegToolBox::on_actionSave_Warpping_triggered()
{
    if(dis.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Warpping","",
            "Images (*.map.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    tipl::image<tipl::vector<3>,3> mapping(dis);
    tipl::displacement_to_mapping(mapping,T);
    gz_mat_write out(filename.toStdString().c_str());
    if(out)
        out.write("mapping",&mapping[0][0],3,mapping.size());
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
