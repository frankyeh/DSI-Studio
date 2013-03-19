#include <QMessageBox>
#include <QFileDialog>
#include "vbc_dialog.hpp"
#include "ui_vbc_dialog.h"
#include "ui_tracking_window.h"
#include "tracking_window.h"
#include "tract/tracttablewidget.h"

vbc_dialog::vbc_dialog(QWidget *parent,ODFModel* handle_) :
    QDialog(parent),
    cur_tracking_window((tracking_window*)parent),
    handle(handle_),
    ui(new Ui::vbc_dialog)
{
    ui->setupUi(this);
    ui->vbc_view->setScene(&vbc_scene);
    ui->subject_list->setColumnCount(3);
    ui->subject_list->setColumnWidth(0,300);
    ui->subject_list->setColumnWidth(1,50);
    ui->subject_list->setColumnWidth(2,50);
    ui->subject_list->setHorizontalHeaderLabels(
                QStringList() << "name" << "value" << "R2");
    ui->subject_list->setRowCount(handle->vbc->subject_count());
    for(unsigned int index = 0;index < handle->vbc->subject_count();++index)
    {
        ui->subject_list->setItem(index,0, new QTableWidgetItem(QString(handle->vbc->subject_name(index).c_str())));
        ui->subject_list->setItem(index,1, new QTableWidgetItem(QString::number(0)));
        ui->subject_list->setItem(index,2, new QTableWidgetItem(QString::number(handle->vbc->subject_R2(index))));
    }
    ui->subject_list->selectRow(0);
}

vbc_dialog::~vbc_dialog()
{
    delete ui;
}

void vbc_dialog::show_info_at(const image::vector<3,float>& pos)
{
    // show image
    if(vbc_slice_pos != cur_tracking_window->ui->AxiSlider->value())
       on_subject_list_itemSelectionChanged();

    // show data
    std::vector<float> vbc_data;
    handle->vbc->get_data_at(
            image::pixel_index<3>(std::floor(pos[0] + 0.5), std::floor(pos[1] + 0.5), std::floor(pos[2] + 0.5),
                                  handle->fib_data.dim).index(),0,vbc_data);
    if(!vbc_data.empty())
    {
        for(unsigned int index = 0;index < handle->vbc->subject_count();++index)
            ui->subject_list->item(index,1)->setText(QString::number(vbc_data[index]));

        vbc_data.erase(std::remove(vbc_data.begin(),vbc_data.end(),0.0),vbc_data.end());
        if(!vbc_data.empty())
        {
            float max_y = *std::max_element(vbc_data.begin(),vbc_data.end());
            std::vector<unsigned int> hist;
            image::histogram(vbc_data,hist,0,max_y,20);
            QVector<double> x(hist.size()+1),y(hist.size()+1);
            unsigned int max_hist = 0;
            for(unsigned int j = 0;j < hist.size();++j)
            {
                x[j] = max_y*(float)j/(float)hist.size();
                y[j] = hist[j];
                max_hist = std::max<unsigned int>(max_hist,hist[j]);
            }
            x.back() = max_y*(hist.size()+1)/hist.size();
            y.back() = 0;
            ui->vbc_report->clearGraphs();
            ui->vbc_report->addGraph();
            QPen pen;
            pen.setColor(QColor(20,20,100,200));
            ui->vbc_report->graph(0)->setLineStyle(QCPGraph::lsLine);
            ui->vbc_report->graph(0)->setPen(pen);
            ui->vbc_report->graph(0)->setData(x, y);

            ui->vbc_report->xAxis->setRange(0,x.back());
            ui->vbc_report->yAxis->setRange(0,max_hist);
            ui->vbc_report->replot();
            }
    }
}

void vbc_dialog::show_report(const std::vector<std::vector<float> >& vbc_data)
{

    unsigned int x_size = 0;
    for(unsigned int i = 0;i < vbc_data.size();++i)
        x_size = std::max<unsigned int>(x_size,vbc_data[i].size());
    if(x_size == 0)
        return;
    QVector<double> x(x_size);
    std::vector<QVector<double> > y(vbc_data.size());
    int min_x = -1;
    unsigned int max_x = 40;
    float max_y = 0.4;
    for(unsigned int i = 0;i < vbc_data.size();++i)
        y[i].resize(x_size);
    for(unsigned int j = 0;j < x_size;++j)
    {
        x[j] = (float)j;
        for(unsigned int i = 0; i < vbc_data.size(); ++i)
            if(j < vbc_data[i].size())
            {
                y[i][j] = vbc_data[i][j];
                if(min_x == -1 && vbc_data[i][j] > 0)
                    min_x = x[j];
            }
    }
    ui->null_dist->clearGraphs();
    QPen pen;

    QColor color[4];
    color[0] = QColor(20,20,100,200);
    color[1] = QColor(100,20,20,200);
    color[2] = QColor(20,100,20,200);
    color[3] = QColor(20,100,100,200);
    char legend[4][60] = {"subject greater","subject lesser","null greater","null lesser"};
    for(unsigned int i = 0; i < vbc_data.size(); ++i)
    {
        ui->null_dist->addGraph();
        pen.setColor(color[i]);
        ui->null_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->null_dist->graph()->setPen(pen);
        ui->null_dist->graph()->setData(x, y[i]);
        ui->null_dist->graph()->setName(QString(legend[i]));
    }

    ui->null_dist->xAxis->setRange(min_x,max_x);
    ui->null_dist->yAxis->setRange(0,max_y);
    ui->null_dist->legend->setVisible(true);
    QFont legendFont = font();  // start out with MainWindow's font..
    legendFont.setPointSize(9); // and make a bit smaller for legend
    ui->null_dist->legend->setFont(legendFont);
    ui->null_dist->legend->setPositionStyle(QCPLegend::psRight);
    ui->null_dist->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->null_dist->replot();


    ui->dist_table->setColumnCount(5);
    ui->dist_table->setColumnWidth(0,50);
    ui->dist_table->setColumnWidth(1,150);
    ui->dist_table->setColumnWidth(2,150);
    ui->dist_table->setColumnWidth(3,150);
    ui->dist_table->setColumnWidth(4,150);
    ui->dist_table->setHorizontalHeaderLabels(
                QStringList() << "span" << "pdf(x)" << "cdf(x)" << "pdf(x)" << "cdf(x)");


    ui->dist_table->setRowCount(100);
    float sum[2] = {0.0,0.0};
    for(unsigned int index = 0;index < 100;++index)
    {
        ui->dist_table->setItem(index,0, new QTableWidgetItem(QString::number(index + 1)));
        ui->dist_table->setItem(index,1, new QTableWidgetItem(QString::number(vbc_data[0][index+1])));
        ui->dist_table->setItem(index,2, new QTableWidgetItem(QString::number(sum[0] += vbc_data[0][index+1])));
        ui->dist_table->setItem(index,3, new QTableWidgetItem(QString::number(vbc_data[1][index+1])));
        ui->dist_table->setItem(index,4, new QTableWidgetItem(QString::number(sum[1] += vbc_data[1][index+1])));
    }
    ui->dist_table->selectRow(0);
}


void vbc_dialog::on_cal_null_trend_clicked()
{
    if(!handle->has_vbc())
        return;
    std::vector<std::vector<float> > vbc_data(2);
    handle->vbc->calculate_null_trend_distribution(ui->vbc_threshold->value(),vbc_data[0],vbc_data[1]);
    show_report(vbc_data);
}

void vbc_dialog::on_cal_trend_clicked()
{
    if(!handle->has_vbc())
        return;
    std::vector<unsigned int> permu(handle->vbc->subject_count());
    for(unsigned int index = 0;index < permu.size();++index)
        permu[index] = index;
    handle->vbc->tend_analysis(permu,cur_subject_fib);
    cur_subject_fib.add_greater_lesser_mapping_for_tracking(handle);
    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
    {
        cur_tracking_window->ui->tracking_index->addItem("greater mapping");
        cur_tracking_window->ui->tracking_index->addItem("lesser mapping");
    }
}

void vbc_dialog::on_cal_lesser_tracts_clicked()
{
    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
        return;
    std::vector<std::vector<float> > tracts;
    std::vector<float> fdr;
    begin_prog("calculating");
    handle->vbc->calculate_subject_fdr(ui->vbc_threshold->value(),cur_subject_fib,tracts,fdr);
    if(tracts.empty())
    {
        QMessageBox::information(this,"result","no significant lesser span",0);
        return;
    }

    for(float fdr_upper = 0.1,fdr_lower = 0.0;
        fdr_upper < 1.0;fdr_upper += 0.1,fdr_lower += 0.1)
    {
        std::vector<std::vector<float> > selected_tracts;
        std::vector<image::rgb_color> color;
        for(unsigned int index = 0;index < fdr.size();++index)
        {
            if(fdr[index] >= fdr_lower && fdr[index] < fdr_upper)
            {
                selected_tracts.push_back(std::vector<float>());
                selected_tracts.back().swap(tracts[index]);
                color.push_back(image::rgb_color(230,fdr[index]*230,fdr[index]*230));
            }
        }
        cur_tracking_window->tractWidget->addNewTracts(QString("FDR ") + QString::number(fdr_lower) + " to " + QString::number(fdr_upper));
        cur_tracking_window->tractWidget->tract_models.back()->add_tracts(selected_tracts);
        for(unsigned int index = 0;index < color.size();++index)
            cur_tracking_window->tractWidget->tract_models.back()->set_tract_color(index,color[index]);
        cur_tracking_window->tractWidget->item(cur_tracking_window->tractWidget->tract_models.size()-1,1)->
            setText(QString::number(cur_tracking_window->tractWidget->tract_models.back()->get_visible_track_count()));
    }

    //cur_tracking_window->renderWidget->setData("tract_color_style",1);//manual assigned
    //cur_tracking_window->glWidget->makeTracts();
    //cur_tracking_window->glWidget->updateGL();
}

void vbc_dialog::on_cal_group_dist_clicked()
{
    QStringList filename = QFileDialog::getOpenFileNames(
                                this,
                "Select subject fib file for analysis",QString(),"Fib files (*.fib.gz *.fib);;All files (*.*)" );
    if (filename.isEmpty())
        return;

    std::vector<std::string> file_names;
    for(unsigned int index = 0;index < filename.size();++index)
        file_names.push_back(filename[index].toLocal8Bit().begin());
    std::vector<std::vector<float> > vbc_data(2);
    if(!handle->vbc->calculate_group_distribution(ui->vbc_threshold->value(),

                                                  file_names,vbc_data[0],vbc_data[1]))
    {
        QMessageBox::information(this,"error",handle->vbc->error_msg.c_str(),0);
        return;
    }
    show_report(vbc_data);
}

/*

void tracking_window::on_actionPair_comparison_triggered()
{
    if(!handle->has_vbc())
        return;
    QString filename1 = QFileDialog::getOpenFileName(
                                this,
                                "Select subject fib file for analysis",
                                absolute_path,
                                "Fib files (*.fib.gz *.fib);;All files (*.*)" );
    if (filename1.isEmpty())
        return;
    QString filename2 = QFileDialog::getOpenFileName(
                                this,
                                "Select subject fib file for analysis",
                                absolute_path,
                                "Fib files (*.fib.gz *.fib);;All files (*.*)" );
    if (filename2.isEmpty())
        return;


    begin_prog("load data");
    if(!handle->vbc->single_subject_paired_analysis(
                                                    filename1.toLocal8Bit().begin(),
                                                    filename2.toLocal8Bit().begin()))
    {
        check_prog(1,1);
        QMessageBox::information(this,"error",handle->vbc->error_msg.c_str(),0);
        return;
    }
    check_prog(1,1);
    if(ui->tracking_index->findText("lesser mapping") == -1)
    {
        ui->tracking_index->addItem("greater mapping");
        ui->tracking_index->addItem("lesser mapping");
    }
}
    */

void vbc_dialog::on_show_null_distribution_clicked()
{
    std::vector<std::vector<float> > vbc_data(2);
    handle->vbc->calculate_null_distribution(ui->vbc_threshold->value(),vbc_data[0],vbc_data[1]);
    show_report(vbc_data);
}

void vbc_dialog::on_vbc_dist_update_clicked()
{

    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
        return;

    std::vector<std::vector<float> > vbc_data(2);
    handle->vbc->calculate_subject_distribution(ui->vbc_threshold->value(),cur_subject_fib,vbc_data[0],vbc_data[1]);
    show_report(vbc_data);

}

void vbc_dialog::on_subject_list_itemSelectionChanged()
{
    image::basic_image<float,2> slice;
    handle->vbc->get_subject_slice(ui->subject_list->currentRow(),ui->AxiSlider->value(),slice);
    image::normalize(slice);
    image::color_image color_slice(slice.geometry());
    std::copy(slice.begin(),slice.end(),color_slice.begin());
    QImage qimage((unsigned char*)&*color_slice.begin(),color_slice.width(),color_slice.height(),QImage::Format_RGB32);
    vbc_slice_image = qimage.scaled(color_slice.width()*cur_tracking_window->ui->zoom->value(),color_slice.height()*cur_tracking_window->ui->zoom->value());
    vbc_scene.clear();
    vbc_scene.setSceneRect(0, 0, vbc_slice_image.width(),vbc_slice_image.height());
    vbc_scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    vbc_scene.addRect(0, 0, vbc_slice_image.width(),vbc_slice_image.height(),QPen(),vbc_slice_image);
    vbc_slice_pos = ui->AxiSlider->value();
}

void vbc_dialog::on_save_vbc_dist_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                "report.txt",
                "Report file (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toLocal8Bit().begin());
    if(!out)
    {
        QMessageBox::information(this,"Error","Cannot write to file",0);
        return;
    }

    std::vector<QCPDataMap::const_iterator> iterators(ui->null_dist->graphCount());
    for(int row = 0;;++row)
    {
        bool has_output = false;
        for(int index = 0;index < ui->null_dist->graphCount();++index)
        {
            if(row == 0)
            {
                out << ui->null_dist->graph(index)->name().toLocal8Bit().begin() << "\t\t";
                has_output = true;
                continue;
            }
            if(row == 1)
            {
                out << "x\ty\t";
                iterators[index] = ui->null_dist->graph(index)->data()->begin();
                has_output = true;
                continue;
            }
            if(iterators[index] != ui->null_dist->graph(index)->data()->end())
            {
                out << iterators[index]->key << "\t" << iterators[index]->value << "\t";
                ++iterators[index];
                has_output = true;
            }
            else
                out << "\t\t";
        }
        out << std::endl;
        if(!has_output)
            break;
    }
}

void vbc_dialog::on_open_subject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                this,
                                "Select subject fib file for analysis",
                                QString(),
                                "Fib files (*.fib.gz *.fib);;All files (*.*)" );
    if (filename.isEmpty())
        return;

    begin_prog("load data");
    if(!handle->vbc->single_subject_analysis(filename.toLocal8Bit().begin(),cur_subject_fib))
    {
        check_prog(1,1);
        QMessageBox::information(this,"error",handle->vbc->error_msg.c_str(),0);
        return;
    }
    check_prog(1,1);

    cur_subject_fib.add_greater_lesser_mapping_for_tracking(handle);
    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
    {
        cur_tracking_window->ui->tracking_index->addItem("greater mapping");
        cur_tracking_window->ui->tracking_index->addItem("lesser mapping");
    }
}
