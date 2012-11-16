#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include <fstream>
#include "vbcdialog.h"
#include "ui_vbcdialog.h"
#include "libs/vbc/vbc_database.h"
#include "prog_interface_static_link.h"


VBCDialog::VBCDialog(QWidget *parent,QString file_name,vbc_database* data_) :
    QDialog(parent),
    ui(new Ui::VBCDialog),
    data(data_)
{
    ui->setupUi(this);
    ui->group_list->setModel(new QStringListModel);
    ui->group_list->setSelectionModel(new QItemSelectionModel(ui->group_list->model()));

    work_dir = QFileInfo(file_name).absolutePath();
    ui->output_file_name->setText(file_name + ".db.fib.gz");
}

VBCDialog::~VBCDialog()
{
    delete ui;
}

void VBCDialog::on_close_clicked()
{
    close();
}


void VBCDialog::update_list(void)
{
    QStringList filenames;
    for(unsigned int index = 0;index < group.size();++index)
        filenames << QFileInfo(group[index]).baseName();
    ((QStringListModel*)ui->group_list->model())->setStringList(filenames);
}

void VBCDialog::on_group1open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open Fib files",
                                     work_dir,
                                     "Fib files (*.fib.gz);;All files (*.*)" );
    if (filenames.isEmpty())
        return;
    group << filenames;
    update_list();
}

void VBCDialog::on_group1delete_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    group.erase(group.begin()+indexes.first().row());
    update_list();
}


void VBCDialog::on_moveup_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == 0)
        return;
    group.swap(indexes.first().row(),indexes.first().row()-1);
    update_list();
    ui->group_list->selectionModel()->select(ui->group_list->model()->index(indexes.first().row()-1,0),
                                             QItemSelectionModel::Select);
}

void VBCDialog::on_movedown_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == group.size()-1)
        return;
    group.swap(indexes.first().row(),indexes.first().row()+1);
    update_list();
    ui->group_list->selectionModel()->select(ui->group_list->model()->index(indexes.first().row()+1,0),
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
    group.clear();
    std::string line;
    std::ifstream in(filename.toLocal8Bit().begin());
    while(std::getline(in,line))
        group << line.c_str();
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
    for(int index = 0;index < group.size();++index)
        out << group[index].toLocal8Bit().begin() <<  std::endl;
}


 /*
void VBCDialog::show_distribution(void)
{

    static unsigned int cur_prog = 0;
    if(data.->cur_prog != cur_prog)
    {
        unsigned int max_x = 0;
        for(unsigned int index = 0;index < data.->length_dist.size();++index)
            if(data.->length_dist[index])
                max_x = index;
        if(max_x == 0)
            return;
        ui->report_widget->clearGraphs();
        QVector<double> x(max_x),y(max_x);
        double max_y = 0.0;
        for(unsigned int j = 0;j < max_x;++j)
        {
            x[j] = j;
            y[j] = data.->length_dist[j];
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

        cur_prog = data.->cur_prog;

        // calculate the cut-off tract_length

    }

    if(data.->cur_prog == data.->total_prog)
        timer->stop();
    else
        ui->progress->setText(QString("Progress %1/%2").arg(data.->cur_prog).arg(data.->total_prog));

}
*/
QStringList search_files(QString dir,QString filter);
void VBCDialog::on_open_dir1_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                work_dir);
    if(dir.isEmpty())
        return;
    group << search_files(dir,"*.fib.gz");
    update_list();
}

void VBCDialog::on_select_output_file_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Save file",
                                 work_dir,
                                 "FIB file (*.fib);;All files (*.*)");
    if(filename.isEmpty())
        return;
    ui->output_file_name->setText(filename);
}

void VBCDialog::on_create_data_base_clicked()
{
    //instance.permutation_test(output_dir,num_files1,p_value_threshold))
    begin_prog("loading");
    if(group.empty())
        return;
    std::vector<std::string> name_list(group.count());
    for (unsigned int index = 0;index < group.count();++index)
        name_list[index] = group[index].toLocal8Bit().begin();
    if(!data->load_subject_files(name_list))
    {
        QMessageBox::information(this,"error",data->error_msg.c_str(),0);
        return;
    }
    data->save_subject_data(ui->output_file_name->text().toLocal8Bit().begin());
}
