#include <QFileInfo>
#include <QMessageBox>
#include <QFileDialog>
#include <QSettings>
#include "ui_tracking_window.h"
#include "individual_connectometry.hpp"
#include "ui_individual_connectometry.h"
#include "tracking/tracking_window.h"
#include "fib_data.hpp"

extern std::string fib_template_file_name_2mm;
individual_connectometry::individual_connectometry(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::individual_connectometry)
{
    ui->setupUi(this);
    ui->norm_method->hide();
    ui->Template->setText(fib_template_file_name_2mm.c_str());
    resize(width(),0);
    QSettings settings;
    switch(settings.value("individual_connectometry_norm").toInt())
    {
    case 0:
        ui->norm0->setChecked(true);
        break;
    case 1:
        ui->norm1->setChecked(true);
        break;
    case 2:
        ui->norm2->setChecked(true);
        break;
    case 3:
        ui->norm3->setChecked(true);
        break;

    }
}

individual_connectometry::~individual_connectometry()
{
    delete ui;
}


void individual_connectometry::on_open_template_clicked()
{
    QString subject_file = QFileDialog::getOpenFileName(
                           this,
                           "Open template Fib files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject_file.isEmpty())
        return;
    ui->Template->setText(subject_file);
}


void individual_connectometry::on_load_baseline_clicked()
{
    QString subject_file = QFileDialog::getOpenFileName(
                           this,
                           "Open baseline Fib files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject_file.isEmpty())
        return;
    ui->File1->setText(subject_file);

}

void individual_connectometry::on_open_subject2_clicked()
{
    QString subject2_file = QFileDialog::getOpenFileName(
                           this,
                           "Open subject Fib files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject2_file.isEmpty())
        return;
    ui->File2->setText(subject2_file);
}


void individual_connectometry::on_Close_clicked()
{
    accept();
}


void individual_connectometry::on_compare_clicked()
{
    if(!QFileInfo(ui->File1->text()).exists())
    {
        QMessageBox::information(this,"error","The baseline file does not exist.",0);
        return;

    }
    if(!QFileInfo(ui->File2->text()).exists())
    {
        QMessageBox::information(this,"error","The study file does not exist.",0);
        return;
    }
    if(!QFileInfo(ui->Template->text()).exists())
    {
        QMessageBox::information(this,"error","The template file does not exist.",0);
        return;
    }

    begin_prog("reading",0);

    std::shared_ptr<fib_data> baseline(std::make_shared<fib_data>());
    if (!baseline->load_from_file(ui->File1->text().toStdString().c_str()))
    {
        QMessageBox::information(this,"error",baseline->error_msg.c_str(),0);
        check_prog(0,0);
        return;
    }
    if(!baseline->is_qsdr)
    {
        QMessageBox::information(this,"error","Please open a QSDR reconstructed FIB file. Please see online document for details.",0);
        check_prog(0,0);
        return;
    }
    bool two_subjects = false;
    if(baseline->has_odfs())
    {
        two_subjects = true;
        baseline = std::make_shared<fib_data>();
        if(!baseline->load_from_file(ui->Template->text().toStdString().c_str()))
        {
            QMessageBox::information(this,"error",baseline->error_msg.c_str(),0);
            check_prog(0,0);
            return;
        }
    }


    tracking_window* new_mdi = new tracking_window(this,baseline);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(ui->File1->text());
    new_mdi->showNormal();

    unsigned char normalization;
    if(ui->norm0->isChecked())
        normalization = 0;
    if(ui->norm1->isChecked())
        normalization = 1;
    if(ui->norm2->isChecked())
        normalization = 2;
    if(ui->norm3->isChecked())
        normalization = 3;
    QSettings settings;
    settings.setValue("individual_connectometry_norm",(int)normalization);

    begin_prog("reading",0);
    if(two_subjects)
    {
        if(!new_mdi->cnt_result.individual_vs_individual(new_mdi->handle,
            ui->File1->text().toStdString().c_str(),ui->File2->text().toStdString().c_str(),normalization))
            goto error;
        else
            goto run;
    }

    if(baseline->db.has_db())
    {
        if(!new_mdi->cnt_result.individual_vs_db(new_mdi->handle,ui->File2->text().toStdString().c_str()))
            goto error;
        else
            goto run;
    }

    // versus template
    {
        if(normalization == 0)
            normalization = 1;
        if(!new_mdi->cnt_result.individual_vs_atlas(new_mdi->handle,ui->File2->text().toStdString().c_str(),normalization))
            goto error;
        else
            goto run;
    }

    {
        run:
        baseline->report = baseline->db.report + new_mdi->cnt_result.report;
        new_mdi->initialize_tracking_index(0);
        new_mdi->scene.show_slice();
        check_prog(0,0);
        QDir::setCurrent(QFileInfo(ui->File1->text()).absolutePath());
        return;
    }

    {
        error:
        QMessageBox::information(this,"Error",new_mdi->cnt_result.error_msg.c_str());
        new_mdi->close();
        delete new_mdi;
        check_prog(0,0);
        return;
    }

}


void individual_connectometry::on_pushButton_clicked()
{
    if(ui->norm_method->isVisible())
        ui->norm_method->hide();
    else
        ui->norm_method->show();
    resize(width(),0);
}
