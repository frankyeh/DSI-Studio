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
    timer->setInterval(500);

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
    J.clear();
    JJ.clear();
    J2.clear();

    t2f_dis.clear();
    to2from.clear();
    f2t_dis.clear();
    from2to.clear();
    arg.clear();
    ui->run_reg->setText("run");
}
void RegToolBox::setup_slice_pos(void)
{
    if(!It.empty())
    {
        int range = int(It.shape()[cur_view]);
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

    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::critical(this,"ERROR","Invalid file format");
        return;
    }
    nifti.toLPS(It);
    nifti.get_image_transformation(ItR);
    tipl::normalize(It);
    nifti.get_voxel_size(Itvs);
    It_is_mni = nifti.is_mni();
    setup_slice_pos();
    clear();
    show_image();
    ui->template_filename->setText(QFileInfo(filename).baseName());

    ui->cost_fun->setCurrentIndex(It.shape() == I.shape() ? 2:0);
}

void RegToolBox::on_OpenSubject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::critical(this,"ERROR","Invalid file format");
        return;
    }
    nifti.toLPS(I);
    nifti.get_image_transformation(IR);
    tipl::normalize(I);
    nifti.get_voxel_size(Ivs);
    clear();
    show_image();
    ui->subject_filename->setText(QFileInfo(filename).baseName());
    ui->cost_fun->setCurrentIndex(It.shape() == I.shape() ? 2:0);
}


void RegToolBox::on_OpenSubject2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Subject Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::critical(this,"ERROR","Invalid file format");
        return;
    }
    nifti.toLPS(I2);
    tipl::normalize(I2);
}

void RegToolBox::on_OpenTemplate2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Template Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;

    tipl::io::gz_nifti nifti;
    if(!nifti.load_from_file(filename.toStdString()))
    {
        QMessageBox::critical(this,"ERROR","Invalid file format");
        return;
    }
    nifti.toLPS(It2);
    tipl::normalize(It2);
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
    if(!It.empty())
    {
        image_fascade I_to_show(I,It,t2f_dis,tipl::transformation_matrix<float>(arg,It.shape(),Itvs,I.shape(),Ivs));
        const auto& It_to_show = (ui->show_second->isChecked() && It2.shape() == It.shape() ? It2 : It);
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
    if(!I.empty())
    {
        const auto& I_to_show = (J.empty() ? I:J);
        int pos = std::min(I_to_show.depth()-1,I_to_show.depth()*ui->slice_pos->value()/ui->slice_pos->maximum());
        tipl::color_image cJ(v2c_I[tipl::volume2slice_scaled(I_to_show,cur_view,pos,ratio)]);
        QImage warp_image;
        warp_image << cJ;

        if(ui->show_warp->isChecked() && ui->dis_spacing->currentIndex() && !t2f_dis.empty())
        {
            float sub_ratio = float(t2f_dis.width())/float(I_to_show.width());
            QPainter paint(&warp_image);
            paint.setBrush(Qt::NoBrush);
            paint.setPen(Qt::red);
            tipl::image<2,tipl::vector<3> > dis_slice;
            tipl::volume2slice(t2f_dis,dis_slice,cur_view,pos*sub_ratio);

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
    if(old_arg != arg)
    {
        tipl::out() << arg;
        show_image();
        old_arg = arg;
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

void RegToolBox::linear_reg(tipl::reg::reg_type reg_type,int cost_type)
{
    status = "linear registration";
    tipl::image<3> J_(It.shape());
    if(cost_type == 2) // skip nonlinear registration
    {
        if(I.shape() == It.shape())
            J_ = I;
        else
            tipl::draw(I,J_,tipl::vector<3,int>(0,0,0));

        if(I2.shape() == I.shape())
        {
            tipl::image<3> J2_(It.shape());
            if(I.shape() == It.shape())
                J2_ = I2;
            else
                tipl::draw(I2,J2_,tipl::vector<3,int>(0,0,0));
            J2.swap(J2_);
        }
        arg.clear();
        T = tipl::transformation_matrix<float>(arg,It.shape(),Itvs,I.shape(),Ivs);
    }
    else
    {
        if(cost_type == 0)// mutual information
            linear_with_mi(It,Itvs,I,Ivs,arg,reg_type,thread.terminated,ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound);
        else
        if(cost_type == 1)// correlation
            linear_with_cc(It,Itvs,I,Ivs,arg,reg_type,thread.terminated,ui->large_deform->isChecked() ? tipl::reg::large_bound : tipl::reg::reg_bound);
        tipl::out() << "linear registration completed" << std::endl;
        T = tipl::transformation_matrix<float>(arg,It.shape(),Itvs,I.shape(),Ivs);
        tipl::resample<tipl::interpolation::cubic>(I,J_,T);
        if(I2.shape() == I.shape())
        {
            tipl::image<3> J2_(It.shape());
            tipl::resample<tipl::interpolation::cubic>(I2,J2_,T);
            tipl::normalize(J2);
            J2.swap(J2_);
        }

    }
    auto r = tipl::correlation(J_.begin(),J_.end(),It.begin());
    tipl::out() << "linear: " << r << std::endl;
    J.swap(J_);
}

void edge_for_cdm(tipl::image<3>& sIt,
                  tipl::image<3>& sJ,
                  tipl::image<3>& sIt2,
                  tipl::image<3>& sJ2)
{
    tipl::filter::sobel(sIt);
    tipl::filter::sobel(sJ);
    tipl::filter::mean(sIt);
    tipl::filter::mean(sJ);
    if(!sIt2.empty())
    {
        tipl::filter::sobel(sIt2);
        tipl::filter::mean(sIt2);
    }
    if(!sJ2.empty())
    {
        tipl::filter::sobel(sJ2);
        tipl::filter::mean(sJ2);
    }
}


void RegToolBox::nonlinear_reg(void)
{
    status = "nonlinear registration";
    tipl::out() << "begin nonlinear registration" << std::endl;
    {
        tipl::reg::cdm_param param;
        param.resolution = ui->resolution->value();
        param.min_dimension = uint32_t(ui->min_reso->value());
        param.smoothing = float(ui->smoothing->value());
        param.speed = float(ui->speed->value());
        if(ui->edge->isChecked())
        {
            tipl::image<3> sIt(It),sJ(J),sIt2(It2),sJ2(J2);
            edge_for_cdm(sIt,sJ,sIt2,sJ2);
            cdm_common(sIt,sIt2,sJ,sJ2,t2f_dis,f2t_dis,thread.terminated,param,ui->use_cuda->isChecked());
        }
        else
            cdm_common(It,It2,J,J2,t2f_dis,f2t_dis,thread.terminated,param,ui->use_cuda->isChecked());
    }
    tipl::out() << "nonlinear registration completed.";
    // calculate inverted to2from
    {
        from2to.resize(I.shape());
        tipl::inv_displacement_to_mapping(f2t_dis,from2to,T);
        tipl::displacement_to_mapping(t2f_dis,to2from,T);
    }
}



void RegToolBox::on_run_reg_clicked()
{
    if(I.empty() || It.empty())
    {
        QMessageBox::critical(this,"ERROR","Please load image first");
        return;
    }
    clear();
    thread.terminated = false;

    thread.run([this]()
    {
        // adjust Ivs for affine
        linear_reg(tipl::reg::affine,ui->cost_fun->currentIndex());
        /*
        // This skip affine registration
        else
        {
            J = I;
            if(I2.shape() == I.shape())
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
bool load_nifti_file(std::string file_name_cmd,
                     tipl::image<3>& data,
                     tipl::vector<3>& vs,
                     tipl::matrix<4,4>& trans,
                     bool& is_mni);
bool apply_warping(const char* from,
                   const char* to,
                   const tipl::shape<3>& I_shape,
                   const tipl::matrix<4,4>& IR,
                   tipl::image<3,tipl::vector<3> >& to2from,
                   tipl::vector<3> Itvs,
                   const tipl::matrix<4,4>& ItR,
                   bool It_is_mni,
                   std::string& error)
{
    tipl::out() << "apply warping to " << from << std::endl;

    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(from))
    {
        error = nii.error_msg;
        return false;
    }
    if(nii.dim(4) > 1)
    {
        // check data range
        std::vector<tipl::image<3> > I(nii.dim(4));
        for(unsigned int index = 0;index < nii.dim(4);++index)
        {
            if(!nii.toLPS(I[index]))
            {
                error = "failed to parse 4D NIFTI file";
                return false;
            }
            std::replace_if(I[index].begin(),I[index].end(),[](float v){return std::isnan(v) || std::isinf(v) || v < 0.0f;},0.0f);
        }
        if(I_shape != I[0].shape())
        {
            error = std::filesystem::path(from).filename().string();
            error += " has an image size or srow matrix from that of the original --from image.";
            return false;
        }
        bool is_label = tipl::is_label_image(I[0]);
        tipl::out() << (is_label ? "processed as labels using nearest assignment" : "processed as values using interpolation") << std::endl;
        tipl::image<4> J(to2from.shape().expand(nii.dim(4)));
        for(size_t i = 0;i < nii.dim(4);++i)
        {
            auto J_slice = J.slice_at(i);
            if(is_label)
                tipl::compose_mapping<tipl::interpolation::nearest>(I[i],to2from,J_slice);
            else
                tipl::compose_mapping<tipl::interpolation::cubic>(I[i],to2from,J_slice);
        }
        if(!tipl::io::gz_nifti::save_to_file(to,J,Itvs,ItR,It_is_mni))
        {
            error = "cannot write to file ";
            error += to;
            return false;
        }
        return true;
    }

    tipl::image<3> I3;
    if(!nii.toLPS(I3))
    {
        error = nii.error_msg;
        return false;
    }
    bool is_label = tipl::is_label_image(I3);
    tipl::out() << (is_label ? "processed as labels using nearest assignment" : "processed as values using interpolation") << std::endl;

    if(I_shape != I3.shape())
    {
        error = std::filesystem::path(from).filename().string();
        error += " has an image size or srow matrix from that of the original --from image.";
        return false;
    }

    tipl::image<3> J3;
    if(is_label)
        tipl::compose_mapping<tipl::interpolation::nearest>(I3,to2from,J3);
    else
        tipl::compose_mapping<tipl::interpolation::cubic>(I3,to2from,J3);
    tipl::out() << "save as to " << to;
    if(!tipl::io::gz_nifti::save_to_file(to,J3,Itvs,ItR,It_is_mni))
    {
        error = "cannot write to file ";
        error += to;
        return false;
    }
    return true;
}
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
        if(!apply_warping(from[0].toStdString().c_str(),
                          to.toStdString().c_str(),
                          I.shape(),IR,to2from,Itvs,ItR,It_is_mni,error))
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
            if(!apply_warping(from[i].toStdString().c_str(),
                          (from[i]+".wp.nii.gz").toStdString().c_str(),
                          I.shape(),IR,to2from,Itvs,ItR,It_is_mni,error))

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
    if(I.shape() == It.shape())
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
        tipl::normalize(I);
        show_image();
    }
}



void RegToolBox::on_actionSave_Warping_triggered()
{
    if(to2from.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,"Save Mapping","",
            "Images (*map.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    tipl::io::gz_mat_write out(filename.toStdString().c_str());
    if(!out)
    {
        QMessageBox::critical(this,"ERROR","Cannot write to file");
        return;
    }
    out.write("to2from",&to2from[0][0],3,to2from.size());
    out.write("to_dim",to2from.shape());
    out.write("to_vs",Itvs);
    out.write("to_trans",ItR);

    out.write("from2to",&from2to[0][0],3,from2to.size());
    out.write("from_dim",from2to.shape());
    out.write("from_vs",Ivs);
    out.write("from_trans",IR);
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
    if(!I.empty())
    {
        tipl::filter::gaussian(I);
        tipl::normalize(I);
    }
    if(!I2.empty())
    {
        tipl::filter::gaussian(I2);
        tipl::normalize(I2);
    }
    clear();
    show_image();
}

void RegToolBox::on_actionSave_Transformed_Image_triggered()
{
    if(JJ.empty())
        return;
    QString to = QFileDialog::getSaveFileName(
            this,"Save Transformed Image","",
            "Images (*.nii *nii.gz);;All files (*)" );
    if(to.isEmpty())
        return;
    tipl::io::gz_nifti::save_to_file(to.toStdString().c_str(),JJ,Itvs,ItR,It_is_mni);

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

