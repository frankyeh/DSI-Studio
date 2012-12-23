#include <QMessageBox>
#include <QFileDialog>
#include "simulation.h"
#include "ui_simulation.h"
#include "libs/dsi_interface_static_link.h"
#include "mix_gaussian_model.hpp"
#include "racian_noise.hpp"
#include "layout.hpp"


boost::mt19937 RacianNoise::generator(static_cast<unsigned> (std::time(0)));
boost::normal_distribution<float> RacianNoise::normal;
boost::uniform_real<float> RacianNoise::uniform(0.0,1.0);
boost::variate_generator<boost::mt19937&,
boost::normal_distribution<float> > RacianNoise::gen_normal(RacianNoise::generator,RacianNoise::normal);
boost::variate_generator<boost::mt19937&,
boost::uniform_real<float> > RacianNoise::gen_uniform(RacianNoise::generator,RacianNoise::uniform);
std::string error_msg;


Simulation::Simulation(QWidget *parent,QString dir_) :
    QDialog(parent),dir(dir_),
    ui(new Ui::Simulation)
{
    ui->setupUi(this);
    connect(ui->SNR,SIGNAL(valueChanged(int)),this,SLOT(update_file_name()));
    connect(ui->FA,SIGNAL(valueChanged(double)),this,SLOT(update_file_name()));
    connect(ui->MD,SIGNAL(valueChanged(double)),this,SLOT(update_file_name()));
    connect(ui->Trial,SIGNAL(valueChanged(int)),this,SLOT(update_file_name()));

}

Simulation::~Simulation()
{
    delete ui;
}



void Simulation::on_pushButton_clicked()
{
    ui->Btable->setText(QFileDialog::getOpenFileName(
                                this,
                                "Open Images files",
                                dir,
                                "Image files (*.txt);;All files (*.*)" ));
    update_file_name();
}

void Simulation::update_file_name()
{
    std::ostringstream out;
    out << QFileInfo(ui->Btable->text()).absolutePath().toLocal8Bit().begin() << "/"
        << QFileInfo(ui->Btable->text()).baseName().toLocal8Bit().begin() <<
           "_snr" << (int)ui->SNR->value() <<
           "_fa" << (int)(ui->FA->value()*10.0) <<
           "_dif" << ui->MD->value() <<
           "_n" << (int)ui->Trial->value() << ".src";
    ui->output->setText(out.str().c_str());
}

void Simulation::on_generate_clicked()
{
    if(!QFileInfo(ui->Btable->text()).exists())
    {
        QMessageBox::information(this,"Error","Cannot find the b-table",0);
        return;
    }
    unsigned int odf_fold = 8;
    Layout layout(ui->SNR->value(),ui->MD->value(),odf_fold);
    if (!layout.load_b_table(ui->Btable->text().toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error","Cannot open b-table file",0);
        return;
    }
    std::vector<float> angle;
    if(!ui->CrossingAngle->text().isEmpty())
    {
        std::string crossing_angle_iteration_str(ui->CrossingAngle->text().toLocal8Bit().begin());
        std::istringstream tmp(crossing_angle_iteration_str);
        std::copy(std::istream_iterator<float>(tmp),
                  std::istream_iterator<float>(),std::back_inserter(angle));
    }

    layout.createLayout(ui->output->text().toLocal8Bit().begin(),
                        ui->FA->value(),angle,
                        ui->Trial->value(),
                        50,5);

}
