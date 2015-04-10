#include <QClipboard>
#include <QFileDialog>
#include <QMessageBox>
#include "tract_report.hpp"
#include "ui_tract_report.h"
#include "ui_tracking_window.h"
#include "tracking_window.h"
#include "tract/tracttablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "libs/tracking/tract_model.hpp"
tract_report::tract_report(QWidget *parent) :
    QDialog(parent),
    cur_tracking_window((tracking_window*)parent),
    ui(new Ui::tract_report)
{
    ui->setupUi(this);
    std::vector<std::string> index_list;
    cur_tracking_window->handle->get_index_list(index_list);
    for (unsigned int index = 0; index < index_list.size(); ++index)
        ui->report_index->addItem(index_list[index].c_str());

    // report
    {
        connect(ui->report_index,SIGNAL(currentIndexChanged(int)),this,SLOT(on_refresh_report_clicked()));
        connect(ui->profile_dir,SIGNAL(currentIndexChanged(int)),this,SLOT(on_refresh_report_clicked()));
        connect(ui->linewidth,SIGNAL(valueChanged(int)),this,SLOT(on_refresh_report_clicked()));
        connect(ui->report_bandwidth,SIGNAL(valueChanged(double)),this,SLOT(on_refresh_report_clicked()));
        connect(ui->report_legend,SIGNAL(clicked()),this,SLOT(on_refresh_report_clicked()));
    }
}

tract_report::~tract_report()
{
    delete ui;
}
void tract_report::copyToClipboard(void)
{
    QImage image;
    ui->report_widget->saveImage(image);
    QApplication::clipboard()->setImage(image);
}

void tract_report::on_refresh_report_clicked()
{
    if(cur_tracking_window->tractWidget->tract_models.size() > 1 &&
       cur_tracking_window->tractWidget->tract_models[0]->get_tract_color(0) ==
       cur_tracking_window->tractWidget->tract_models[1]->get_tract_color(0))
        cur_tracking_window->tractWidget->assign_colors();

    ui->report_widget->clearGraphs();

    double max_y = 0.0,min_x = 0.0,max_x = 0;
    if(ui->profile_dir->currentIndex() <= 2)
        min_x = cur_tracking_window->slice.geometry[ui->profile_dir->currentIndex()];
    for(unsigned int index = 0;index < cur_tracking_window->tractWidget->tract_models.size();++index)
    {
        if(cur_tracking_window->tractWidget->item(index,0)->checkState() != Qt::Checked)
            continue;
        std::vector<float> values,data_profile;
        cur_tracking_window->tractWidget->tract_models[index]->get_report(
                    ui->profile_dir->currentIndex(),
                    ui->report_bandwidth->value(),
                    ui->report_index->currentText().toLocal8Bit().begin(),
                    values,data_profile);
        if(data_profile.empty())
            continue;

        for(unsigned int i = 0;i < data_profile.size();++i)
            if(data_profile[i] > 0.0)
            {
                max_y = std::max<float>(max_y,data_profile[i]);
                max_x = std::max<float>(max_x,values[i]);
                min_x = std::min<float>(min_x,values[i]);
            }

        QVector<double> x(data_profile.size()),y(data_profile.size());
        std::copy(values.begin(),values.end(),x.begin());
        std::copy(data_profile.begin(),data_profile.end(),y.begin());

        ui->report_widget->addGraph();
        QPen pen;
        image::rgb_color color = cur_tracking_window->tractWidget->tract_models[index]->get_tract_color(0);
        pen.setColor(QColor(color.r,color.g,color.b,200));
        pen.setWidth(ui->linewidth->value());
        ui->report_widget->graph()->setLineStyle(QCPGraph::lsLine);
        ui->report_widget->graph()->setPen(pen);
        ui->report_widget->graph()->setData(x, y);
        ui->report_widget->graph()->setName(cur_tracking_window->tractWidget->item(index,0)->text());
        // give the axes some labels:
        //customPlot->xAxis->setLabel("x");
        //customPlot->yAxis->setLabel("y");
        // set axes ranges, so we see all data:


    }
    ui->report_widget->xAxis->setRange(min_x,max_x);
    ui->report_widget->yAxis->setRange(ui->report_index->currentIndex() ? 0 :
            (*cur_tracking_window)["fa_threshold"].toFloat(), max_y);
    if(ui->report_legend->checkState() == Qt::Checked)
    {
        ui->report_widget->legend->setVisible(true);
        QFont legendFont = font();  // start out with MainWindow's font..
        legendFont.setPointSize(9); // and make a bit smaller for legend
        ui->report_widget->legend->setFont(legendFont);
        ui->report_widget->legend->setPositionStyle(QCPLegend::psRight);
        ui->report_widget->legend->setBrush(QBrush(QColor(255,255,255,230)));
    }
    else
        ui->report_widget->legend->setVisible(false);

    ui->report_widget->replot();
}

void tract_report::on_save_report_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                cur_tracking_window->absolute_path + "/report.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toLocal8Bit().begin());
    if(!out)
    {
        QMessageBox::information(this,"Error","Cannot write to file",0);
        return;
    }
    ui->report_widget->saveTxt(filename);
}


void tract_report::on_save_image_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                cur_tracking_window->absolute_path + "/report.jpg",
                "JPEC file (*.jpg);;BMP file (*.bmp);;PDF file (*.pdf);;PNG file (*.png);;All files (*)");
    if(QFileInfo(filename).completeSuffix().toLower() == "jpg")
        ui->report_widget->saveJpg(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "bmp")
        ui->report_widget->saveBmp(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "png")
        ui->report_widget->savePng(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "pdf")
        ui->report_widget->savePdf(filename,true);
}
