#include <QFileInfo>
#include <QMessageBox>
#include <QFileDialog>
#include <QSettings>
#include "ui_tracking_window.h"
#include "individual_connectometry.hpp"
#include "ui_individual_connectometry.h"
#include "tracking/tracking_window.h"

individual_connectometry::individual_connectometry(QWidget *parent,tracking_window& cur_tracking_window_) :
    QDialog(parent),cur_tracking_window(cur_tracking_window_),
    ui(new Ui::individual_connectometry)
{
    ui->setupUi(this);
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
    on_inv_inv_toggled(true);
}

individual_connectometry::~individual_connectometry()
{
    delete ui;
}


void individual_connectometry::on_load_subject_clicked()
{
    subject_file = QFileDialog::getOpenFileName(
                           this,
                           "Open subject Fib files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject_file.isEmpty())
        return;
    ui->subject_file_label->setText(QFileInfo(subject_file).fileName());
}

void individual_connectometry::on_open_template_clicked()
{
    subject_file = QFileDialog::getOpenFileName(
                           this,
                           "Open template files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject_file.isEmpty())
        return;
    ui->subject_file_label->setText(QFileInfo(subject_file).fileName());
    if(ui->inv_inv)
        ui->compare->setEnabled(QFileInfo(subject_file).exists() && QFileInfo(subject2_file).exists());
}

void individual_connectometry::on_open_subject2_clicked()
{
    subject2_file = QFileDialog::getOpenFileName(
                           this,
                           "Open subject Fib files",
                           "",
                           "Fib files (*fib.gz);;All files (*)");
    if (subject2_file.isEmpty())
        return;
    ui->subject2_file_label->setText(QFileInfo(subject2_file).fileName());
    if(ui->inv_inv)
        ui->compare->setEnabled(QFileInfo(subject_file).exists() && QFileInfo(subject2_file).exists());
    if(ui->inv_db)
        ui->compare->setEnabled(QFileInfo(subject2_file).exists());
}

void individual_connectometry::on_inv_inv_toggled(bool checked)
{
    if(checked)
    {
        ui->norm_method->setEnabled(true);
        ui->widget->setVisible(true);
        ui->widget_2->setVisible(true);
        ui->subject_file->setText("Subject baseline (pre-treatment) FIB file:");
        ui->subject2_file->setText("Subject current (post-treatment) FIB file:");
        subject_file = "";
        subject2_file = "";
        ui->compare->setEnabled(false);
    }
}


void individual_connectometry::on_inv_template_toggled(bool checked)
{
    if(checked)
    {
        ui->norm_method->setEnabled(true);
        ui->widget->setVisible(false);
        ui->widget_2->setVisible(true);
        ui->subject2_file->setText("Subject current FIB file:");
        subject2_file = "";
        ui->compare->setEnabled(false);
    }
}

void individual_connectometry::on_inv_db_toggled(bool checked)
{
    if(checked)
    {
        ui->norm_method->setEnabled(false);
        ui->widget->setVisible(false);
        ui->widget_2->setVisible(true);
        ui->subject2_file->setText("Subject current FIB file:");
        subject2_file = "";
        ui->compare->setEnabled(false);
    }
}

void individual_connectometry::on_Close_clicked()
{
    accept();
}


void individual_connectometry::on_compare_clicked()
{
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

    begin_prog("comparing");
    if(ui->inv_db->isChecked())
    {
        if(!cur_tracking_window.cnt_result.individual_vs_db(cur_tracking_window.handle,subject2_file.toLocal8Bit().begin()))
            QMessageBox::information(this,"Error",cur_tracking_window.cnt_result.error_msg.c_str());
    }

    if(ui->inv_template->isChecked())
    {
        if(!cur_tracking_window.cnt_result.individual_vs_atlas(cur_tracking_window.handle,subject2_file.toLocal8Bit().begin(),normalization))
            QMessageBox::information(this,"Error",cur_tracking_window.cnt_result.error_msg.c_str());
    }

    if(ui->inv_inv->isChecked())
    {
        if(!cur_tracking_window.cnt_result.individual_vs_individual(cur_tracking_window.handle,
                                                                    subject_file.toLocal8Bit().begin(),subject2_file.toLocal8Bit().begin(),normalization))
            QMessageBox::information(this,"Error",cur_tracking_window.cnt_result.error_msg.c_str());
    }
    cur_tracking_window.initialize_tracking_index(cur_tracking_window.handle->dir.index_data.size()-1);
    cur_tracking_window.scene.show_slice();
    check_prog(0,0);
    hide();
    accept();

}
