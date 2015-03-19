#include <QTimer>
#include <QMessageBox>
#include "motion_dialog.hpp"
#include "ui_motion_dialog.h"
#include "dicom_parser.h"
#include "libs/prog_interface_static_link.h"

void linear_reg(const image::basic_image<short,3>& from,
                const image::basic_image<short,3>& to,
                image::affine_transform<3,double>& arg,
                bool& terminated,
                unsigned int& finished)
{
    image::reg::linear(from,to,arg,image::reg::rigid_body,
                       image::reg::square_error(),terminated);
    ++finished;
}

void motion_detection(boost::thread_group& threads,
                      boost::ptr_vector<DwiHeader>& dwi_files,
                      std::vector<unsigned int>& b0_index,
                      std::vector<image::affine_transform<3,double> >& arg,
                      bool& terminated,
                      unsigned int& finished)
{
    b0_index.clear();
    for(unsigned int index = 0;index < dwi_files.size();++index)
    {
        if(dwi_files[index].get_bvalue() < 100)
            b0_index.push_back(index);
    }
    arg.clear();
    arg.resize(b0_index.size());
    for(unsigned int index = 1;index < b0_index.size();++index)
        threads.add_thread(new boost::thread(&linear_reg,
                                             boost::ref(dwi_files[b0_index[0]].image),
                                             boost::ref(dwi_files[b0_index[index]].image),
                                             boost::ref(arg[index]),
                                             boost::ref(terminated),
                                             boost::ref(finished)));
}

void motion_correction(boost::ptr_vector<DwiHeader>& dwi_files,
                       const std::vector<unsigned int>& b0_index,
                       const std::vector<image::affine_transform<3,double> >& arg)
{
    image::geometry<3> geo(dwi_files[0].image.geometry());
    unsigned int b1 = 0,b2 = 1;
    for(unsigned int i = 0;i < dwi_files.size();++i)
    {
        if(i == b0_index.front())// the first b0 is the reference
            continue;
        if(i == b0_index[b2])
        {
            b1 = std::min<unsigned int>(b1+1,b0_index.size()-1);
            b2 = std::min<unsigned int>(b2+1,b0_index.size()-1);
        }
        double dis = b0_index[b2]-b0_index[b1];
        double w1 = b0_index[b2]-i;
        double w2 = i - b0_index[b1];
        if(dis == 0.0)
        {
            w1 = 1.0;
            w2 = 0.0;
        }
        else
        {
            w1 /= dis;
            w2 /= dis;
        }
        image::affine_transform<3,double> interpolated_arg;
        for(unsigned int i = 0;i < 6;++i)
            interpolated_arg.translocation[i] =
                    arg[b1].translocation[i]*w1 +
                    arg[b2].translocation[i]*w2;

        image::transformation_matrix<3,double> T =
                image::transformation_matrix<3,double>(interpolated_arg,geo,geo);

        image::basic_image<short,3> new_image(geo);
        image::resample(dwi_files[i].image,new_image,T,image::linear);
        dwi_files[i].image.swap(new_image);

        // rotate b-table
        float iT[9];
        image::matrix::inverse(T.scaling_rotation,iT,image::dim<3,3>());
        image::vector<3> tmp;
        image::vector_rotation(dwi_files[i].bvec.begin(),tmp.begin(),iT,image::vdim<3>());
        tmp.normalize();
        dwi_files[i].bvec = tmp;
    }
}


motion_dialog::motion_dialog(QWidget *parent,boost::ptr_vector<DwiHeader>& dwi_files_) :
    QDialog(parent),dicom_gui(*(dicom_parser*)parent),dwi_files(dwi_files_),terminated(false),finished(0),
    ui(new Ui::motion_dialog)
{
    ui->setupUi(this);

    motion_detection(threads,dwi_files,b0_index,arg,terminated,finished);

    ui->progressBar->setMaximum(arg.size());// there is only arg.size()-1 calculating thread
    ui->progressBar->setValue(0);
    timer.reset(new QTimer(this));
    timer->setInterval(1000);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(show_progress()));
    timer->start();
    connect(ui->legend1, SIGNAL(stateChanged(int)), this, SLOT(show_progress()));
    connect(ui->legend2, SIGNAL(stateChanged(int)), this, SLOT(show_progress()));
    ui->correction->setVisible(false);
}

motion_dialog::~motion_dialog()
{
    terminated = true;
    threads.join_all();
    delete ui;
}

void motion_dialog::show_progress(void)
{
    ui->progressBar->setValue(finished);

    ui->translocation->clearGraphs();
    ui->rotation->clearGraphs();
    char legends[6][40] = {"translocation x","translocation y","translocation z","rotation x","rotation y","rotation z"};

    QPen pen;
    QColor color[3];
    color[0] = QColor(20,20,100,200);
    color[1] = QColor(100,20,20,200);
    color[2] = QColor(20,100,20,200);
    QVector<double> index(arg.size());
    for(unsigned int j = 0;j < arg.size();++j)
        index[j] = j;
    std::vector<QVector<double> > data(6);

    for(unsigned int i = 0;i < 6;++i)
    {
        data[i].resize(arg.size());
        for(unsigned int j = 0;j < arg.size();++j)
        {
            index[j] = j;
            data[i][j] = arg[j].translocation[i];
            if(i >= 3)
                data[i][j] *= 180.0/3.14159265358979323846;
            else
                data[i][j] *= dwi_files.front().voxel_size[i];
        }
        if(i < 3)
        {
            ui->translocation->addGraph();
            pen.setColor(color[i]);
            ui->translocation->graph()->setLineStyle(QCPGraph::lsLine);
            ui->translocation->graph()->setPen(pen);
            ui->translocation->graph()->setData(index, data[i]);
            ui->translocation->graph()->setName(legends[i]);
        }
        else
        {
            ui->rotation->addGraph();
            pen.setColor(color[i-3]);
            ui->rotation->graph()->setLineStyle(QCPGraph::lsLine);
            ui->rotation->graph()->setPen(pen);
            ui->rotation->graph()->setData(index, data[i]);
            ui->rotation->graph()->setName(legends[i]);
        }
    }

    QFont legendFont = font();  // start out with MainWindow's font..
    legendFont.setPointSize(9); // and make a bit smaller for legend

    ui->translocation->xAxis->setLabel("time");
    ui->translocation->xAxis->setAutoTickStep(false);
    ui->translocation->xAxis->setTickStep(5);
    ui->translocation->yAxis->setLabel("mm");
    ui->translocation->xAxis->setRange(0,arg.size()-1);
    ui->translocation->yAxis->setRange(-5.0,5.0);
    ui->translocation->legend->setVisible(ui->legend1->isChecked());
    ui->translocation->legend->setFont(legendFont);
    ui->translocation->legend->setPositionStyle(QCPLegend::psRight);
    ui->translocation->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->translocation->replot();

    ui->rotation->xAxis->setLabel("time");
    ui->rotation->xAxis->setAutoTickStep(false);
    ui->rotation->xAxis->setTickStep(5);
    ui->rotation->yAxis->setLabel("degress");
    ui->rotation->xAxis->setRange(0,arg.size()-1);
    ui->rotation->yAxis->setRange(-5.0,5.0);
    ui->rotation->legend->setVisible(ui->legend2->isChecked());
    ui->rotation->legend->setFont(legendFont);
    ui->rotation->legend->setPositionStyle(QCPLegend::psRight);
    ui->rotation->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->rotation->replot();

    {
        std::vector<double> m(arg.size()),r(arg.size());
        for(unsigned int i = 0;i < arg.size();++i)
        {
            image::vector<3> t(arg[i].translocation);
            t[0] *= dwi_files.front().voxel_size[0];
            t[1] *= dwi_files.front().voxel_size[1];
            t[2] *= dwi_files.front().voxel_size[2];
            m[i] = t.length();
            r[i] = image::vector<3>(arg[i].rotation).length()*180.0/3.14159265358979323846;
        }
        ui->label->setText(QString("averaged translocation:%1 mm rotation:%2 degrees").
                           arg(image::mean(m.begin(),m.end())).
                           arg(image::mean(r.begin(),r.end())));
    }

    if(finished == arg.size()-1)
    {
        finished = arg.size();
        timer->stop();
        ui->correction->setVisible(true);
        ui->progressBar->setVisible(false);
        ui->progress_label->setVisible(false);
    }
}

void motion_dialog::on_correction_clicked()
{
    begin_prog("correcting");
    check_prog(0,2);
    motion_correction(dwi_files,b0_index,arg);
    check_prog(1,2);
    dicom_gui.update_b_table();
    check_prog(2,2);
    QMessageBox::information(this,"Motion correction","Done",0);
    close();
}
