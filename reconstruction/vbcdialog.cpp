#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include <fstream>
#include "vbcdialog.h"
#include "ui_vbcdialog.h"
#include "libs/vbc/vbc.hpp"
#include "prog_interface_static_link.h"


VBCDialog::VBCDialog(QWidget *parent,QString workDir) :
    QDialog(parent),
    ui(new Ui::VBCDialog),
    work_dir(workDir),
    vbc_instance(new vbc())
{
    ui->setupUi(this);
    ui->group1list->setModel(new QStringListModel);
    ui->group2list->setModel(new QStringListModel);
    ui->group1list->setSelectionModel(new QItemSelectionModel(ui->group1list->model()));
    ui->group2list->setSelectionModel(new QItemSelectionModel(ui->group2list->model()));
    timer.reset(new QTimer(this));
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(show_distribution()));

    ui->save_mapping->setEnabled(false);
    ui->load_subject_data->setEnabled(false);
    ui->run_null->setEnabled(false);
}

VBCDialog::~VBCDialog()
{
    delete ui;
}

void VBCDialog::on_close_clicked()
{
    timer->stop();
    vbc_instance.reset(0);
    close();
}


void VBCDialog::on_vbc_trend_toggled(bool checked)
{
    if(checked)
    {
        ui->group2_widget->hide();
        ui->group1_label->setText("List subjects in incremental order of the study variable");
        ui->movedown->show();
        ui->moveup->show();
    }
}

void VBCDialog::on_vbc_single_toggled(bool checked)
{
    if(checked)
    {
        ui->group2_widget->show();
        ui->group1_label->setText("Study subject");
        ui->group2_label->setText("Matched normal population");
        ui->movedown->hide();
        ui->moveup->hide();
    }
}

void VBCDialog::on_vbc_group_toggled(bool checked)
{
    if(checked)
    {
        ui->group2_widget->show();
        ui->group1_label->setText("Group1");
        ui->group2_label->setText("Group2");
        ui->movedown->hide();
        ui->moveup->hide();
    }
}

void VBCDialog::update_list(void)
{
    {
        QStringList filenames;
        for(unsigned int index = 0;index < group1.size();++index)
            filenames << QFileInfo(group1[index]).baseName();
        ((QStringListModel*)ui->group1list->model())->setStringList(filenames);
    }
    {
        QStringList filenames;
        for(unsigned int index = 0;index < group2.size();++index)
            filenames << QFileInfo(group2[index]).baseName();
        ((QStringListModel*)ui->group2list->model())->setStringList(filenames);
    }
}

void VBCDialog::on_group1open_clicked()
{
    if(ui->vbc_single->isChecked())
    {
        QString filename = QFileDialog::getOpenFileName(
                                     this,
                                     "Open Fib files",
                                     work_dir,
                                     "Fib files (*.fib.gz);;All files (*.*)" );
        if(filename.isEmpty())
            return;
        group1.clear();
        group1 << filename;
    }
    else
    {
        QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open Fib files",
                                     work_dir,
                                     "Fib files (*.fib.gz);;All files (*.*)" );
        if (filenames.isEmpty())
            return;
        group1 << filenames;
    }
    update_list();
}

void VBCDialog::on_group2open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                 this,
                                 "Open Fib files",
                                 work_dir,
                                 "Fib files (*.fib.gz);;All files (*.*)" );
    if (filenames.isEmpty())
        return;
    group2 << filenames;
    update_list();
}
void VBCDialog::on_group1delete_clicked()
{
    QModelIndexList indexes = ui->group1list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    group1.erase(group1.begin()+indexes.first().row());
    update_list();
}

void VBCDialog::on_group2delete_clicked()
{
    QModelIndexList indexes = ui->group2list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    group2.erase(group2.begin()+indexes.first().row());
    update_list();
}

void VBCDialog::on_moveup_clicked()
{
    QModelIndexList indexes = ui->group1list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == 0)
        return;
    group1.swap(indexes.first().row(),indexes.first().row()-1);
    update_list();
    ui->group1list->selectionModel()->select(ui->group1list->model()->index(indexes.first().row()-1,0),
                                             QItemSelectionModel::Select);
}

void VBCDialog::on_movedown_clicked()
{
    QModelIndexList indexes = ui->group1list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == group1.size()-1)
        return;
    group1.swap(indexes.first().row(),indexes.first().row()+1);
    update_list();
    ui->group1list->selectionModel()->select(ui->group1list->model()->index(indexes.first().row()+1,0),
                                             QItemSelectionModel::Select);
}

void VBCDialog::on_open_list1_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                 this,
                                 "Open text file",
                                 work_dir,
                                 "Text files (*.txt);;All files (*.*)" );
    if(filename.isEmpty())
        return;
    group1.clear();
    std::string line;
    std::ifstream in(filename.toLocal8Bit().begin());
    while(std::getline(in,line))
        group1 << line.c_str();
    update_list();
}

void VBCDialog::on_save_list1_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Open text file",
                                 work_dir,
                                 "Text files (*.txt);;All files (*.*)" );
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toLocal8Bit().begin());
    for(int index = 0;index < group1.size();++index)
        out << group1[index].toLocal8Bit().begin() <<  std::endl;
}

void VBCDialog::on_open_list2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                 this,
                                 "Open text file",
                                 work_dir,
                                 "Text files (*.txt);;All files (*.*)" );
    if(filename.isEmpty())
        return;
    group2.clear();
    std::string line;
    std::ifstream in(filename.toLocal8Bit().begin());
    while(std::getline(in,line))
        group2 << line.c_str();
    update_list();
}

void VBCDialog::on_save_list2_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Open text file",
                                 work_dir,
                                 "Text files (*.txt);;All files (*.*)" );
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toLocal8Bit().begin());
    for(int index = 0;index < group2.size();++index)
        out << group2[index].toLocal8Bit().begin() <<  std::endl;
}

void VBCDialog::on_load_subject_data_clicked()
{
    //instance.permutation_test(output_dir,num_files1,p_value_threshold))
    begin_prog("loading");
    QStringList all_list;
    all_list << group1 << group2;
    if(all_list.empty())
        return;
    std::vector<std::string> name_list(all_list.count());
    for (unsigned int index = 0;index < all_list.count();++index)
        name_list[index] = all_list[index].toLocal8Bit().begin();

    const char* msg =
            vbc_instance->load_subject_data(name_list,group1.count());

    if(msg)
    {
        QMessageBox::information(this,"error",msg,0);
        return;
    }
    ui->ODF_label->setText("subject data loaded");
    ui->save_mapping->setEnabled(true);
    ui->run_null->setEnabled(true);
    return;

}

void VBCDialog::show_distribution(void)
{
    static unsigned int cur_prog = 0;
    if(vbc_instance->cur_prog != cur_prog)
    {
        unsigned int max_x = 0;
        for(unsigned int index = 0;index < vbc_instance->length_dist.size();++index)
            if(vbc_instance->length_dist[index])
                max_x = index;
        if(max_x == 0)
            return;
        ui->report_widget->clearGraphs();
        QVector<double> x(max_x),y(max_x);
        double max_y = 0.0;
        for(unsigned int j = 0;j < max_x;++j)
        {
            x[j] = j;
            y[j] = vbc_instance->length_dist[j];
            max_y = std::max(max_y,y[j]);
        }
        ui->report_widget->addGraph();
        QPen pen;
        pen.setColor(QColor(20,20,100,200));
        ui->report_widget->graph(0)->setLineStyle(QCPGraph::lsLine);
        ui->report_widget->graph(0)->setPen(pen);
        ui->report_widget->graph(0)->setData(x, y);

        ui->report_widget->xAxis->setRange(0,max_x);
        ui->report_widget->yAxis->setRange(0,max_y);
        ui->report_widget->replot();

        cur_prog = vbc_instance->cur_prog;

        // calculate the cut-off tract_length

    }

    if(vbc_instance->cur_prog == vbc_instance->total_prog)
        timer->stop();
    else
        ui->progress->setText(QString("Progress %1/%2").arg(vbc_instance->cur_prog).arg(vbc_instance->total_prog));

}

QStringList search_files(QString dir,QString filter);
void VBCDialog::on_open_dir1_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                work_dir);
    if(dir.isEmpty())
        return;
    group1 << search_files(dir,"*.fib.gz");
    update_list();
}

void VBCDialog::on_open_dir2_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                work_dir);
    if(dir.isEmpty())
        return;
    group2 << search_files(dir,"*.fib.gz");
    update_list();
}

void VBCDialog::on_open_template_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                 this,
                                 "Open template file",
                                 work_dir,
                                 "Fib files (*.fib.gz);;All files (*.*)" );
    if(filename.isEmpty())
        return;
    if(!vbc_instance->load_fiber_template(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Cannot open template file",0);
        return;
    }
    else
    {
        ui->template_label->setText("template loaded");
        ui->qa_threshold->setValue(0.6*image::segmentation::otsu_threshold(
            image::basic_image<float, 3,image::const_pointer_memory<float> >(
                                           &*(vbc_instance->fa[0].begin()),vbc_instance->dim)));
        ui->load_subject_data->setEnabled(true);

    }
}

void VBCDialog::on_save_mapping_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Save file",
                                 work_dir,
                                 "FIB file (*.fib);;All files (*.*)");
    if(filename.isEmpty())
        return;
    vbc_instance->output_greater_lesser_mapping(
                filename.toLocal8Bit().begin(),ui->qa_threshold->value());
    if(!timer->isActive() && !vbc_instance->length_dist.empty())
    {
        std::cout << "output tracts" << std::endl;
        QString f1 = filename,f2 = filename;
        f1 += ".greater.trk";
        f2 += ".lesser.trk";
        if(!vbc_instance->fdr_tracking(f1.toLocal8Bit().begin(),
                                   ui->qa_threshold->value(),
                                   ui->t_threshold->value(),0.05,true))
            QMessageBox::information(this,"Notification","No tracts in greater mapping",0);
        if(!vbc_instance->fdr_tracking(f2.toLocal8Bit().begin(),
                                   ui->qa_threshold->value(),
                                   ui->t_threshold->value(),0.05,false))
            QMessageBox::information(this,"Notification","No tracts in lesser mapping",0);


    }
}

void VBCDialog::on_run_null_clicked()
{
    vbc_instance->calculate_null(ui->thread_count->value(),
                                        ui->permutation_num->value(),
                                        ui->qa_threshold->value(),
                                        ui->t_threshold->value());
    timer->start(2000);
}
