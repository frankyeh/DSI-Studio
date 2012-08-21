#include "simulation.h"
#include "ui_simulation.h"
#include "libs/dsi_interface_static_link.h"
#include <QMessageBox>
#include <QFileDialog>

Simulation::Simulation(QWidget *parent,QString dir_) :
    QDialog(parent),dir(dir_),
    ui(new Ui::Simulation)
{
    ui->setupUi(this);
}

Simulation::~Simulation()
{
    delete ui;
}

void Simulation::on_buttonBox_accepted()
{
    if(!QFileInfo(ui->Btable->text()).exists())
    {
        QMessageBox::information(this,"Error","Cannot find the b-table",0);
        return;
    }
    generate_simulation(
            ui->Btable->text().toLocal8Bit().begin(),
            ui->SNR->value(),
            ui->MD->value(),
            8,
            ui->FA->text().toLocal8Bit().begin(),
            ui->CrossingAngle->text().toLocal8Bit().begin(),
            ui->Trial->value());
}

void Simulation::on_pushButton_clicked()
{
    ui->Btable->setText(QFileDialog::getOpenFileName(
                                this,
                                "Open Images files",
                                dir,
                                "Image files (*.txt);;All files (*.*)" ));
}
