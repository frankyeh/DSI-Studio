#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include <fstream>
#include "vbcdialog.h"
#include "ui_vbcdialog.h"
#include "libs/vbc/vbc.hpp"
#include "prog_interface_static_link.h"


VBCDialog::VBCDialog(QWidget *parent,QString workDir,vbc* vbc_instance_) :
    QDialog(parent),
    ui(new Ui::VBCDialog),
    work_dir(workDir),
    vbc_instance(vbc_instance_)
{
    ui->setupUi(this);
    ui->group1list->setModel(new QStringListModel);
    ui->group2list->setModel(new QStringListModel);
    ui->group1list->setSelectionModel(new QItemSelectionModel(ui->group1list->model()));
    ui->group2list->setSelectionModel(new QItemSelectionModel(ui->group2list->model()));
    ui->mapping->setText(workDir + "/mapping.fib.gz");
    ui->cluster_group->hide();
    timer.reset(new QTimer(this));
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(show_distribution()));
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
    ui->subject_data_group->hide();
    ui->method_group->hide();
    ui->load_subject_data->hide();
    ui->cluster_group->show();

    begin_prog("loading");
    QStringList all_list;
    all_list << group1 << group2;
    std::vector<std::string> name_list(all_list.count());
    for (unsigned int index = 0;index < all_list.count();++index)
        name_list[index] = all_list[index].toLocal8Bit().begin();

    const char* msg =
            vbc_instance->load_subject_data(name_list,group1.count(),ui->qa_threshold->value());

    if(msg)
    {
        QMessageBox::information(this,"error",msg,0);
        ui->subject_data_group->show();
        ui->method_group->show();
        ui->load_subject_data->show();
        ui->cluster_group->hide();
        return;
    }
    vbc_instance->calculate_mapping(ui->mapping->text().toLocal8Bit().begin(),
                                    ui->p_value_threshold->value());
    return;
    vbc_instance->calculate_permutation(ui->thread_count->value(),
                                        ui->permutation_num->value(),
                                        ui->p_value_threshold->value());
    timer->start(2000);
}

void VBCDialog::show_distribution(void)
{
    static unsigned int cur_prog = 0;
    if(vbc_instance->cur_prog != cur_prog)
    {
        unsigned int resolution = 20;
        std::vector<unsigned int> hist1,hist2;
        std::vector<unsigned int> max_cluster_size(vbc_instance->get_max_cluster_size());
        std::vector<float> max_statistics(vbc_instance->get_max_statistics());
        float min_x1 = 0;
        float max_x1 = *std::max_element(max_cluster_size.begin(),max_cluster_size.end());
        float min_x2 = *std::min_element(max_statistics.begin(),max_statistics.end());
        float max_x2 = *std::max_element(max_statistics.begin(),max_statistics.end());

        {
            if(max_cluster_size.empty() || max_statistics.empty())
                return;
            image::histogram(max_cluster_size,hist1,0,max_x1,resolution);
            image::histogram(max_statistics,hist2,min_x2,max_x2,resolution);
            if(hist1.size() != resolution || hist2.size() != resolution)
                return;
        }
        ui->report_widget1->clearGraphs();
        QVector<double> x1(resolution),x2(resolution),y1(resolution),y2(resolution);
        double max_y1 = 0.0,max_y2 = 0.0;
        for(unsigned int j = 0;j < resolution;++j)
        {
            x1[j] = (max_x1-min_x1)*(float)j/(float)(resolution-1)+min_x1;
            x2[j] = (max_x2-min_x2)*(float)j/(float)(resolution-1)+min_x2;
            y1[j] = hist1[j];
            y2[j] = hist2[j];
            max_y1 = std::max(max_y1,y1[j]);
            max_y2 = std::max(max_y2,y2[j]);
        }
        ui->report_widget1->addGraph();
        ui->report_widget2->addGraph();
        QPen pen;
        pen.setColor(QColor(20,20,100,200));
        ui->report_widget1->graph(0)->setLineStyle(QCPGraph::lsLine);
        ui->report_widget1->graph(0)->setPen(pen);
        ui->report_widget1->graph(0)->setData(x1, y1);
        pen.setColor(QColor(20,100,20,200));
        ui->report_widget2->graph(0)->setLineStyle(QCPGraph::lsLine);
        ui->report_widget2->graph(0)->setPen(pen);
        ui->report_widget2->graph(0)->setData(x2, y2);

        ui->report_widget1->xAxis->setRange(min_x1,max_x1);
        ui->report_widget1->yAxis->setRange(0,max_y1);
        ui->report_widget1->replot();

        ui->report_widget2->xAxis->setRange(min_x2,max_x2);
        ui->report_widget2->yAxis->setRange(0,max_y2);
        ui->report_widget2->replot();

        cur_prog = vbc_instance->cur_prog;
    }

    if(vbc_instance->cur_prog == vbc_instance->total_prog)
    {
        // saving mapping
        vbc_instance->calculate_mapping(ui->mapping->text().toLocal8Bit().begin(),
                                        ui->p_value_threshold->value());
        timer->stop();
        QMessageBox::information(this,"Done","mapping saved",0);
    }
    else
        ui->progress->setText(QString("Progress %1/%2").arg(vbc_instance->cur_prog).arg(vbc_instance->total_prog));

}


void VBCDialog::on_open_mapping_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Save file",
                                 ui->mapping->text(),
                                 "FIB file (*.fib);;All files (*.*)");
    if(filename.isEmpty())
        return;
    ui->mapping->setText(filename);
}
QStringList search_files(QString dir,QString filter);
void VBCDialog::on_open_dir1_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->mapping->text());
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
                                ui->mapping->text());
    if(dir.isEmpty())
        return;
    group2 << search_files(dir,"*.fib.gz");
    update_list();
}
