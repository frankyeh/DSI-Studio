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

    connect(ui->Individual,SIGNAL(toggled(bool)),this,SLOT(on_toggled(bool)));
    connect(ui->Trend,SIGNAL(toggled(bool)),this,SLOT(on_toggled(bool)));

    // dist report
    connect(ui->line_width,SIGNAL(valueChanged(int)),this,SLOT(show_report()));
    connect(ui->span_from,SIGNAL(valueChanged(int)),this,SLOT(show_report()));
    connect(ui->span_to,SIGNAL(valueChanged(int)),this,SLOT(show_report()));
    connect(ui->max_prob,SIGNAL(valueChanged(double)),this,SLOT(show_report()));
    connect(ui->show_null_greater,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_null_lesser,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_greater,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_lesser,SIGNAL(toggled(bool)),this,SLOT(show_report()));

    connect(ui->line_width_2,SIGNAL(valueChanged(int)),this,SLOT(show_fdr_report()));
    connect(ui->span_from_2,SIGNAL(valueChanged(int)),this,SLOT(show_fdr_report()));
    connect(ui->span_to_2,SIGNAL(valueChanged(int)),this,SLOT(show_fdr_report()));
    connect(ui->max_prob_2,SIGNAL(valueChanged(double)),this,SLOT(show_fdr_report()));
    connect(ui->show_greater_2,SIGNAL(toggled(bool)),this,SLOT(show_fdr_report()));
    connect(ui->show_lesser_2,SIGNAL(toggled(bool)),this,SLOT(show_fdr_report()));

    ui->individual_study->hide();
    on_toggled(true);
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

void vbc_dialog::show_fdr_report()
{
    ui->fdr_dist->clearGraphs();
    if(fdr.size() != 2)
        return;
    std::vector<std::vector<float> > vbc_data;
    char legends[4][60] = {"greater","lesser"};
    std::vector<const char*> legend;

    if(ui->show_greater_2->isChecked())
    {
        vbc_data.push_back(fdr[0]);
        legend.push_back(legends[0]);
    }
    if(ui->show_lesser_2->isChecked())
    {
        vbc_data.push_back(fdr[1]);
        legend.push_back(legends[1]);
    }


    QPen pen;
    QColor color[4];
    color[0] = QColor(20,20,100,200);
    color[1] = QColor(100,20,20,200);
    for(unsigned int i = 0; i < vbc_data.size(); ++i)
    {
        QVector<double> x(vbc_data[i].size());
        QVector<double> y(vbc_data[i].size());
        for(unsigned int j = 0;j < vbc_data[i].size();++j)
        {
            x[j] = (float)j;
            y[j] = vbc_data[i][j];
        }
        ui->fdr_dist->addGraph();
        pen.setColor(color[i]);
        pen.setWidth(ui->line_width_2->value());
        ui->fdr_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->fdr_dist->graph()->setPen(pen);
        ui->fdr_dist->graph()->setData(x, y);
        ui->fdr_dist->graph()->setName(QString(legend[i]));
    }

    ui->fdr_dist->xAxis->setRange(ui->span_from_2->value(),ui->span_to_2->value());
    ui->fdr_dist->yAxis->setRange(0,ui->max_prob_2->value());
    ui->fdr_dist->legend->setVisible(true);
    QFont legendFont = font();  // start out with MainWindow's font..
    legendFont.setPointSize(9); // and make a bit smaller for legend
    ui->fdr_dist->legend->setFont(legendFont);
    ui->fdr_dist->legend->setPositionStyle(QCPLegend::psRight);
    ui->fdr_dist->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->fdr_dist->replot();

}

void vbc_dialog::show_report()
{
    ui->null_dist->clearGraphs();
    if(dist.empty())
        return;
    std::vector<std::vector<float> > vbc_data;
    char legends[4][60] = {"greater","lesser","null greater","null lesser"};
    std::vector<const char*> legend;

    if(ui->show_null_greater->isChecked())
    {
        vbc_data.push_back(dist[0]);
        legend.push_back(legends[0]);
    }
    if(ui->show_null_lesser->isChecked())
    {
        vbc_data.push_back(dist[1]);
        legend.push_back(legends[1]);
    }
    if(ui->show_greater->isChecked())
    {
        vbc_data.push_back(dist[2]);
        legend.push_back(legends[2]);
    }
    if(ui->show_lesser->isChecked())
    {
        vbc_data.push_back(dist[3]);
        legend.push_back(legends[3]);
    }

    if(vbc_data.empty())
        return;

    unsigned int x_size = 0;
    for(unsigned int i = 0;i < vbc_data.size();++i)
        x_size = std::max<unsigned int>(x_size,vbc_data[i].size());
    if(x_size == 0)
        return;
    QVector<double> x(x_size);
    std::vector<QVector<double> > y(vbc_data.size());
    for(unsigned int i = 0;i < vbc_data.size();++i)
        y[i].resize(x_size);
    for(unsigned int j = 0;j < x_size;++j)
    {
        x[j] = (float)j;
        for(unsigned int i = 0; i < vbc_data.size(); ++i)
            if(j < vbc_data[i].size())
                y[i][j] = vbc_data[i][j];
    }
    QPen pen;
    QColor color[4];
    color[0] = QColor(20,20,100,200);
    color[1] = QColor(100,20,20,200);
    color[2] = QColor(20,100,20,200);
    color[3] = QColor(20,100,100,200);
    for(unsigned int i = 0; i < vbc_data.size(); ++i)
    {
        ui->null_dist->addGraph();
        pen.setColor(color[i]);
        pen.setWidth(ui->line_width->value());
        ui->null_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->null_dist->graph()->setPen(pen);
        ui->null_dist->graph()->setData(x, y[i]);
        ui->null_dist->graph()->setName(QString(legend[i]));
    }

    ui->null_dist->xAxis->setRange(ui->span_from->value(),ui->span_to->value());
    ui->null_dist->yAxis->setRange(0,ui->max_prob->value());
    ui->null_dist->legend->setVisible(true);
    QFont legendFont = font();  // start out with MainWindow's font..
    legendFont.setPointSize(9); // and make a bit smaller for legend
    ui->null_dist->legend->setFont(legendFont);
    ui->null_dist->legend->setPositionStyle(QCPLegend::psRight);
    ui->null_dist->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->null_dist->replot();
}

void vbc_dialog::show_dis_table(void)
{
    if(dist.size() != 4)
        return;
    ui->dist_table->setColumnCount(9);
    ui->dist_table->setColumnWidth(0,50);
    ui->dist_table->setColumnWidth(1,150);
    ui->dist_table->setColumnWidth(2,150);
    ui->dist_table->setColumnWidth(3,150);
    ui->dist_table->setColumnWidth(4,150);
    ui->dist_table->setColumnWidth(5,150);
    ui->dist_table->setColumnWidth(6,150);
    ui->dist_table->setColumnWidth(7,150);
    ui->dist_table->setColumnWidth(8,150);
    ui->dist_table->setHorizontalHeaderLabels(
                QStringList() << "span" << "null greater pdf" << "null greater cdf" <<
                                            "null lesser pdf" << "null lesser cdf" <<
                                            "greater pdf" << "greater cdf" <<
                                            "lesser pdf" << "lesser cdf");

    ui->dist_table->setRowCount(100);
    std::vector<float> sum(4);
    for(unsigned int index = 0;index < 100;++index)
    {
        ui->dist_table->setItem(index,0, new QTableWidgetItem(QString::number(index + 1)));
        ui->dist_table->setItem(index,1, new QTableWidgetItem(QString::number(dist[0][index+1])));
        ui->dist_table->setItem(index,2, new QTableWidgetItem(QString::number(sum[0] += dist[0][index+1])));
        ui->dist_table->setItem(index,3, new QTableWidgetItem(QString::number(dist[1][index+1])));
        ui->dist_table->setItem(index,4, new QTableWidgetItem(QString::number(sum[1] += dist[1][index+1])));
        ui->dist_table->setItem(index,5, new QTableWidgetItem(QString::number(dist[2][index+1])));
        ui->dist_table->setItem(index,6, new QTableWidgetItem(QString::number(sum[2] += dist[2][index+1])));
        ui->dist_table->setItem(index,7, new QTableWidgetItem(QString::number(dist[3][index+1])));
        ui->dist_table->setItem(index,8, new QTableWidgetItem(QString::number(sum[3] += dist[3][index+1])));
    }
    ui->dist_table->selectRow(0);
}

void vbc_dialog::show_fdr_table(void)
{
    if(fdr.size() != 2)
        return;
    ui->fdr_table->setColumnCount(3);
    ui->fdr_table->setColumnWidth(0,50);
    ui->fdr_table->setColumnWidth(1,150);
    ui->fdr_table->setColumnWidth(2,150);
    ui->fdr_table->setHorizontalHeaderLabels(
                QStringList() << "span" << "FDR greater" << "FDR lesser");

    ui->fdr_table->setRowCount(100);
    for(unsigned int index = 0;index < 100;++index)
    {
        ui->fdr_table->setItem(index,0, new QTableWidgetItem(QString::number(index + 1)));
        ui->fdr_table->setItem(index,1,
                               new QTableWidgetItem(index + 1 < fdr[0].size() ? QString::number(fdr[0][index+1]):QString()));
        ui->fdr_table->setItem(index,2,
                               new QTableWidgetItem(index + 1 < fdr[1].size() ? QString::number(fdr[1][index+1]):QString()));
    }
    ui->fdr_table->selectRow(0);
}

void vbc_dialog::on_cal_lesser_tracts_clicked()
{
    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
        return;
    std::vector<std::vector<float> > tracts;
    std::vector<float> fdr;
    begin_prog("calculating");
    handle->vbc->calculate_subject_fdr(1.0-ui->percentile_rank->value(),cur_subject_fib,tracts,fdr);
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

void vbc_dialog::on_cal_FDR_clicked()
{
    dist.clear();
    dist.resize(4);

    if(ui->Individual->isChecked())
    {
        QStringList filename = QFileDialog::getOpenFileNames(
                                    this,
                    "Select subject fib file for analysis",
                    cur_tracking_window->absolute_path,"Fib files (*.fib.gz *.fib);;All files (*.*)" );
        if (filename.isEmpty())
            return;

        std::vector<std::string> file_names;
        for(unsigned int index = 0;index < filename.size();++index)
            file_names.push_back(filename[index].toLocal8Bit().begin());
        if(!handle->vbc->calculate_group_distribution(1.0-ui->percentile_rank->value(),
                                                      file_names,dist[2],dist[3]))
        {
            QMessageBox::information(this,"error",handle->vbc->error_msg.c_str(),0);
            return;
        }
        handle->vbc->calculate_null_distribution(1.0-ui->percentile_rank->value(),dist[0],dist[1]);
    }
    else
    if(ui->Trend->isChecked())
    {
        QString filename = QFileDialog::getOpenFileName(
                                    this,
                                "Select a value text file for analysis",
                                cur_tracking_window->absolute_path,"Text files (*.txt);;All files (*.*)" );
        if (filename.isEmpty())
            return;

        std::ifstream in(filename.toLocal8Bit().begin());
        std::vector<float> data;
        std::copy(std::istream_iterator<float>(in),
                  std::istream_iterator<float>(),std::back_inserter(data));

        if(data.size() != handle->vbc->subject_count())
        {
            QMessageBox::information(this,"error","The number of data does not mactch the subject count",0);
            return;
        }
        handle->vbc->tend_analysis(data,cur_subject_fib);
        handle->vbc->calculate_subject_distribution(1.0-ui->percentile_rank->value(),cur_subject_fib,dist[2],dist[3]);
        cur_subject_fib.add_greater_lesser_mapping_for_tracking(handle);
        if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
        {
            cur_tracking_window->ui->tracking_index->addItem("greater mapping");
            cur_tracking_window->ui->tracking_index->addItem("lesser mapping");
        }
        handle->vbc->calculate_null_trend_distribution(handle->vbc->get_trend_std(data),1.0-ui->percentile_rank->value(),dist[0],dist[1]);
    }    
    else
        return;

    fdr.clear();
    fdr.resize(2);
    std::vector<double> sum(4);
    for(unsigned int index = 0;index < dist[0].size();++index)
    {
        for(unsigned int j = 0;j < 4;++j)
            sum[j] += dist[j][index];
        if(sum[2] < 1.0)
            fdr[0].push_back((1.0-sum[0])/(1.0-sum[2]));
        if(sum[3] < 1.0)
            fdr[1].push_back((1.0-sum[1])/(1.0-sum[3]));
    }

    show_report();
    show_dis_table();
    show_fdr_report();
    show_fdr_table();
    if(ui->Individual->isChecked())
        ui->individual_study->show();
    else
        ui->individual_study->hide();
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


void vbc_dialog::on_vbc_dist_update_clicked()
{
    /*
    if(cur_tracking_window->ui->tracking_index->findText("lesser mapping") == -1)
        return;

    std::vector<std::vector<float> > vbc_data(2);
    handle->vbc->calculate_subject_distribution(1.0-ui->percentile_rank->value(),cur_subject_fib,vbc_data[0],vbc_data[1]);
    show_report(vbc_data);
    */
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
                cur_tracking_window->absolute_path + "/dist_report.txt",
                "Report file (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    ui->null_dist->saveTxt(filename);
}

void vbc_dialog::on_save_fdr_dist_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                cur_tracking_window->absolute_path + "/fdr_report.txt",
                "Report file (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    ui->fdr_dist->saveTxt(filename);
}

void vbc_dialog::on_open_subject_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                this,
                                "Select subject fib file for analysis",
                                cur_tracking_window->absolute_path,
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

void vbc_dialog::on_toggled(bool checked)
{
    if(ui->Trend->isChecked())
    {
        ui->open_instruction->setText("Open the value text file");
    }

    if(ui->Individual->isChecked())
    {
        ui->open_instruction->setText("Open all subjects data");
    }
}


