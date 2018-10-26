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
    It *= 1.0f/tipl::mean(It);
    nifti.get_voxel_size(Itvs);
    ui->slice_pos->setMaximum(It.depth()-1);
    ui->slice_pos->setValue(It.depth()/2);
    clear();
    show_image();
    if(It.geometry() == I.geometry())
        ui->reg_type->setCurrentIndex(2);
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
    I *= 1.0f/tipl::mean(I);
    nifti.get_voxel_size(Ivs);
    clear();
    show_image();
    if(It.geometry() == I.geometry())
        ui->reg_type->setCurrentIndex(2);
}


void show_slice_at(QGraphicsScene& scene,tipl::image<float,2>& tmp,tipl::color_image& buf,float ratio)
{
    tipl::normalize(tmp,255.0);
    tipl::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);
    buf = tmp;
    show_view(scene,QImage((unsigned char*)&*buf.begin(),buf.width(),buf.height(),QImage::Format_RGB32).
              scaled(buf.width()*ratio,buf.height()*ratio));
}


void show_slice_at(QGraphicsScene& scene,const tipl::image<float,3>& source,tipl::color_image& buf,int slice_pos,float ratio)
{
    tipl::image<float,2> tmp;
    tipl::volume2slice(source,tmp,2,slice_pos);
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
            tipl::image<float,2> J_view_slice(J_view.slice_at(ui->slice_pos->value()));
            tipl::normalize(J_view_slice,255.0);
            tipl::upper_lower_threshold(J_view_slice.begin(),J_view_slice.end(),J_view_slice.begin(),0.0f,255.0f);
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
            tipl::resample_mt(I,J_view,tipl::transformation_matrix<double>(arg,It.geometry(),Itvs,I.geometry(),Ivs),tipl::linear);
            show_image();
        }
        else // nonlinear
        {
            if(!dis.empty())
            {
                dis_view = dis;
                J_view = J;
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
    tipl::transformation_matrix<double> T;
    tipl::reg::two_way_linear_mr(It,Itvs,I,Ivs,T,reg_type,tipl::reg::mutual_information(),thread.terminated,
                                  std::thread::hardware_concurrency(),&arg);
    tipl::image<float,3> J2(It.geometry());
    tipl::resample_mt(I,J2,T,tipl::cubic);
    float r2 = tipl::correlation(J2.begin(),J2.end(),It.begin());
    std::cout << "linear:" << r2 << std::endl;
    J.swap(J2);
    J_view = J;

}

void RegToolBox::nonlinear_reg(int method)
{
    status = "nonlinear registration";
    if(method == 1)
    {
        tipl::reg::cdm(It,J,dis,thread.terminated,2.0f,ui->smoothness->value());
    }
    if(method == 0)
    {
        int order = ui->order->value();
        bnorm_data.reset(new tipl::reg::bfnorm_mapping<float,3>(It.geometry(),tipl::geometry<3>(7*order,9*order,7*order)));
        tipl::reg::bfnorm(*bnorm_data.get(),It,J,thread.terminated,std::thread::hardware_concurrency());
        tipl::image<tipl::vector<3>,3> dis2(It.geometry());
        status += "..output mapping";
        dis2.for_each_mt([&](tipl::vector<3>& v,tipl::pixel_index<3>& index){
            bnorm_data->get_displacement(index,v);
        });
        dis2.swap(dis);
    }
    tipl::compose_displacement(J,dis,JJ);
    std::cout << "nonlinear:" << tipl::correlation(JJ.begin(),JJ.end(),It.begin()) << std::endl;
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
            linear_reg(tipl::reg::rigid_body);
            reg_done = true;
            status = "registration done";
        });
        break;
    case 1: // linear + nonlinear
        thread.run([this]()
        {
            linear_reg(tipl::reg::affine);
            nonlinear_reg(ui->reg_method->currentIndex());
            reg_done = true;
            status = "registration done";
        });
        break;
    case 2: // nonlinear only
        thread.run([this]()
        {
            if(I.geometry() != It.geometry())
                linear_reg(tipl::reg::affine);
            else
                J = I;
            nonlinear_reg(ui->reg_method->currentIndex());
            reg_done = true;
            status = "registration done";
        });
        break;
    case 3: // nonlinear only
        thread.run([this]()
        {
            linear_reg(tipl::reg::affine);
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
        nii.set_LPS_transformation(ItR,JJ.geometry());
        tipl::flip_xy(JJ);
        nii << JJ;
        nii.save_to_file(filename.toStdString().c_str());
        return;
    }
    if(!J.empty())
    {
        gz_nifti nii;
        nii.set_voxel_size(Itvs);
        nii.set_LPS_transformation(ItR,J.geometry());
        tipl::flip_xy(J);
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
        tipl::vector<3> from(fa_template_imp.shift);
        from[0] -= (int)fa_template_imp.I.width()+ItR[3]-(int)It.width();
        from[1] -= (int)fa_template_imp.I.height()+ItR[7]-(int)It.height();
        from[2] -= ItR[11];

        It.for_each_mt([&](float& v,const tipl::pixel_index<3>& pos){
           tipl::vector<3> p(pos);
           p -= from;
           p.round();
           if(fa_template_imp.I.geometry().is_valid(p) && fa_template_imp.I.at(p[0],p[1],p[2]) > 0)
               return;
           v = 0.0f;
        });
    }

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
/*

void phase_distortion_matrix(const std::vector<float>& location,std::vector<float>& m)
{
    int n = d.size();
    m.clear();
    m.resize(n*n);
    for(int i = 0,pos = 0;i < n;++i)
        for(int j = 0;j < n;++j,++pos)
        {
            float dis = std::fabs((float)i-location[j]);
            if(dis > 1.0f)
                continue;
            m[pos] += 1.0-dis;
        }
}

void get_location_vector(const std::vector<float>& d,std::vector<float>& location,bool ap)
{
    location.resize(n);
    if(ap)
    for(int i = 0;i < n;++i)
        location[i] = d[i] + (float)i;
    else
        for(int i = 0;i < n;++i)
            location[i] = d[i] - (float)i;
}





template<class pixel_type,unsigned int dimension,class terminate_type>
double topup(const tipl::image<pixel_type,dimension>& It,
            const tipl::image<pixel_type,dimension>& Is,
            tipl::image<pixel_type,dimension>& I,
            tipl::image<float,dimension>& d,// displacement field
            terminate_type& terminated,
            unsigned int steps = 30)
{
    tipl::geometry<dimension> geo = It.geometry();
    d.resize(geo);
    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > 32)
    {
        //downsampling
        image<pixel_type,dimension> rIs,rIt,I;
        downsample_with_padding(It,rIt);
        downsample_with_padding(Is,rIs);
        float r = topup(rIt,rIs,I,d,terminated,steps);
        upsample_with_padding(d,d,geo);
        d *= 2.0f;
    }
    tipl::image<pixel_type,dimension> Js;// transformed I
    tipl::image<float,dimension> new_d(d.geometry());// new displacements

    for (unsigned int index = 0;index < steps && !terminated;++index)
    {
        tipl::compose_displacement(Is,d,Js);
        r = tipl::correlation(Js.begin(),Js.end(),It.begin());
        if(r <= prev_r)
        {
            new_d.swap(d);
            break;
        }
        // dJ(cJ-I)
        tipl::gradient_sobel(Js,new_d);
        Js.for_each_mt([&](pixel_type&,tipl::pixel_index<dimension>& index){
            if(It[index.index()] == 0.0 || It.geometry().is_edge(index))
            {
                new_d[index.index()] = vtor_type();
                return;
            }
            std::vector<pixel_type> Itv,Jv;
            tipl::get_window(index,It,window_size,Itv);
            tipl::get_window(index,Js,window_size,Jv);
            double a,b,r2;
            tipl::linear_regression(Jv.begin(),Jv.end(),Itv.begin(),a,b,r2);
            if(a <= 0.0f)
                new_d[index.index()] = vtor_type();
            else
                new_d[index.index()] *= (Js[index.index()]*a+b-It[index.index()]);
        });
        // solving the poisson equation using Jacobi method
        image<vtor_type,dimension> solve_d(new_d);
        tipl::multiply_constant_mt(solve_d,-inv_d2);
        for(int iter = 0;iter < window_size*2 & !terminated;++iter)
        {
            image<vtor_type,dimension> new_solve_d(new_d.geometry());
            tipl::par_for(solve_d.size(),[&](int pos)
            {
                for(int d = 0;d < dimension;++d)
                {
                    int p1 = pos-shift[d];
                    int p2 = pos+shift[d];
                    if(p1 >= 0)
                       new_solve_d[pos] += solve_d[p1];
                    if(p2 < solve_d.size())
                       new_solve_d[pos] += solve_d[p2];
                }
                new_solve_d[pos] -= new_d[pos];
                new_solve_d[pos] *= inv_d2;
            });
            solve_d.swap(new_solve_d);
        }
        tipl::minus_constant_mt(solve_d,solve_d[0]);

        new_d = solve_d;
        if(theta == 0.0f)
        {
            tipl::par_for(new_d.size(),[&](int i)
            {
               float l = new_d[i].length();
               if(l > theta)
                   theta = l;
            });
        }
        tipl::multiply_constant_mt(new_d,0.5f/theta);
        tipl::add(new_d,d);

        image<vtor_type,dimension> new_ds(new_d);
        tipl::filter::gaussian2(new_ds);
        tipl::par_for(new_d.size(),[&](int i){
           new_ds[i] *= cdm_smoothness;
           new_d[i] *= cdm_smoothness2;
           new_d[i] += new_ds[i];
        });
        new_d.swap(d);
    }
    return r;
}
*/
void RegToolBox::on_actionTOPUP_triggered()
{
    /*
    clear();
    thread.terminated = false;
    reg_type = ui->reg_type->currentIndex();
    thread.run([this]()
    {
        cdm_y(It,I,dis,thread.terminated,1.0f,ui->smoothness->value());
        reg_done = true;
        status = "registration done";
    });

    ui->running_label->movie()->start();
    ui->running_label->show();
    timer->start();
    ui->stop->show();
    ui->run_reg->hide();
    */
}
