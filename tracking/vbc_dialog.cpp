#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <boost/math/distributions/students_t.hpp>
#include "vbc_dialog.hpp"
#include "ui_vbc_dialog.h"
#include "ui_tracking_window.h"
#include "tracking_window.h"
#include "tract/tracttablewidget.h"

vbc_dialog::vbc_dialog(QWidget *parent,vbc_database* vbc_ptr,QString work_dir_) :
    QDialog(parent),vbc(vbc_ptr),work_dir(work_dir_),
    ui(new Ui::vbc_dialog)
{
    ui->setupUi(this);
    ui->vbc_view->setScene(&vbc_scene);
    ui->individual_list->setModel(new QStringListModel);
    ui->individual_list->setSelectionModel(new QItemSelectionModel(ui->individual_list->model()));
    ui->subject_list->setColumnCount(3);
    ui->subject_list->setColumnWidth(0,300);
    ui->subject_list->setColumnWidth(1,50);
    ui->subject_list->setColumnWidth(2,50);
    ui->subject_list->setHorizontalHeaderLabels(
                QStringList() << "name" << "value" << "R2");

    ui->dist_table->setColumnCount(7);
    ui->dist_table->setColumnWidth(0,100);
    ui->dist_table->setColumnWidth(1,100);
    ui->dist_table->setColumnWidth(2,100);
    ui->dist_table->setColumnWidth(3,100);
    ui->dist_table->setColumnWidth(4,100);
    ui->dist_table->setColumnWidth(5,100);
    ui->dist_table->setColumnWidth(6,100);

    ui->dist_table->setHorizontalHeaderLabels(
                QStringList() << "length (mm)" << "FDR greater" << "FDR lesser"
                                               << "null greater pdf" << "null lesser pdf"
                                               << "greater pdf" << "lesser pdf");

    bool check_quality = false;
    ui->subject_list->setRowCount(vbc->subject_count());
    for(unsigned int index = 0;index < vbc->subject_count();++index)
    {
        ui->subject_list->setItem(index,0, new QTableWidgetItem(QString(vbc->subject_name(index).c_str())));
        ui->subject_list->setItem(index,1, new QTableWidgetItem(QString::number(0)));
        ui->subject_list->setItem(index,2, new QTableWidgetItem(QString::number(vbc->subject_R2(index))));
        if(vbc->subject_R2(index) < 0.5)
            check_quality = true;
    }
    ui->AxiSlider->setMaximum(vbc->handle->dim[2]-1);
    ui->AxiSlider->setMinimum(0);
    ui->AxiSlider->setValue(vbc->handle->dim[2] >> 1);

    // dist report
    connect(ui->span_to,SIGNAL(valueChanged(int)),this,SLOT(show_report()));
    connect(ui->span_to,SIGNAL(valueChanged(int)),this,SLOT(show_fdr_report()));
    connect(ui->show_null_greater,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_null_lesser,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_greater,SIGNAL(toggled(bool)),this,SLOT(show_report()));
    connect(ui->show_lesser,SIGNAL(toggled(bool)),this,SLOT(show_report()));

    connect(ui->show_greater_2,SIGNAL(toggled(bool)),this,SLOT(show_fdr_report()));
    connect(ui->show_lesser_2,SIGNAL(toggled(bool)),this,SLOT(show_fdr_report()));


    connect(ui->AxiSlider,SIGNAL(valueChanged(int)),this,SLOT(on_subject_list_itemSelectionChanged()));
    connect(ui->zoom,SIGNAL(valueChanged(double)),this,SLOT(on_subject_list_itemSelectionChanged()));

    ui->subject_list->selectRow(0);
    ui->toolBox->setCurrentIndex(1);
    ui->foi_widget->hide();
    ui->advanced_options_box->hide();
    on_rb_multiple_regression_clicked();
    qApp->installEventFilter(this);

    if(check_quality)
        QMessageBox::information(this,"Warning","The connectometry database contains low goodness-of-fit data. You may need to verify the data quality.");
}

vbc_dialog::~vbc_dialog()
{
    qApp->removeEventFilter(this);

    delete ui;
}

bool vbc_dialog::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->vbc_view)
        return false;
    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->vbc_view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    image::vector<3,float> pos;
    pos[0] =  ((float)point.x()) / ui->zoom->value() - 0.5;
    pos[1] =  ((float)point.y()) / ui->zoom->value() - 0.5;
    pos[2] = ui->AxiSlider->value();
    if(!vbc->handle->dim.is_valid(pos))
        return true;
    ui->coordinate->setText(QString("(%1,%2,%3)").arg(pos[0]).arg(pos[1]).arg(pos[2]));

    // show data
    std::vector<float> vbc_data;
    vbc->get_data_at(
            image::pixel_index<3>(std::floor(pos[0] + 0.5), std::floor(pos[1] + 0.5), std::floor(pos[2] + 0.5),
                                  vbc->handle->dim).index(),0,vbc_data);
    if(!vbc_data.empty())
    {
        for(unsigned int index = 0;index < vbc->subject_count();++index)
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
    return true;
}

void vbc_dialog::show_fdr_report()
{
    ui->fdr_dist->clearGraphs();
    std::vector<std::vector<float> > vbc_data;
    char legends[4][60] = {"greater","lesser"};
    std::vector<const char*> legend;

    if(ui->show_greater_2->isChecked())
    {
        vbc_data.push_back(vbc->fdr_greater);
        legend.push_back(legends[0]);
    }
    if(ui->show_lesser_2->isChecked())
    {
        vbc_data.push_back(vbc->fdr_lesser);
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
        ui->fdr_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->fdr_dist->graph()->setPen(pen);
        ui->fdr_dist->graph()->setData(x, y);
        ui->fdr_dist->graph()->setName(QString(legend[i]));
    }
    ui->fdr_dist->xAxis->setLabel("mm");
    ui->fdr_dist->yAxis->setLabel("FDR");
    ui->fdr_dist->xAxis->setRange(2,ui->span_to->value());
    ui->fdr_dist->yAxis->setRange(0,1.0);
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
    if(vbc->subject_greater_null.empty())
        return;
    ui->null_dist->clearGraphs();
    std::vector<std::vector<unsigned int> > vbc_data;
    char legends[4][60] = {"null greater","null lesser","greater","lesser"};
    std::vector<const char*> legend;

    if(ui->show_null_greater->isChecked())
    {
        vbc_data.push_back(vbc->subject_greater_null);
        legend.push_back(legends[0]);
    }
    if(ui->show_null_lesser->isChecked())
    {
        vbc_data.push_back(vbc->subject_lesser_null);
        legend.push_back(legends[1]);
    }
    if(ui->show_greater->isChecked())
    {
        vbc_data.push_back(vbc->subject_greater);
        legend.push_back(legends[2]);
    }
    if(ui->show_lesser->isChecked())
    {
        vbc_data.push_back(vbc->subject_lesser);
        legend.push_back(legends[3]);
    }

    // normalize
    float max_y1 = *std::max_element(vbc->subject_greater_null.begin(),vbc->subject_greater_null.end());
    float max_y2 = *std::max_element(vbc->subject_lesser_null.begin(),vbc->subject_lesser_null.end());
    float max_y3 = *std::max_element(vbc->subject_greater.begin(),vbc->subject_greater.end());
    float max_y4 = *std::max_element(vbc->subject_lesser.begin(),vbc->subject_lesser.end());


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
        ui->null_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->null_dist->graph()->setPen(pen);
        ui->null_dist->graph()->setData(x, y[i]);
        ui->null_dist->graph()->setName(QString(legend[i]));
    }

    ui->null_dist->xAxis->setLabel("mm");
    ui->null_dist->yAxis->setLabel("count");
    ui->null_dist->xAxis->setRange(4,ui->span_to->value());
    ui->null_dist->yAxis->setRange(0,std::max<float>(std::max<float>(max_y1,max_y2),std::max<float>(max_y3,max_y4))*1.1);
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
    ui->dist_table->setRowCount(100);
    for(unsigned int index = 0;index < vbc->fdr_greater.size()-1;++index)
    {
        ui->dist_table->setItem(index,0, new QTableWidgetItem(QString::number(index + 1)));
        ui->dist_table->setItem(index,1, new QTableWidgetItem(QString::number(vbc->fdr_greater[index+1])));
        ui->dist_table->setItem(index,2, new QTableWidgetItem(QString::number(vbc->fdr_lesser[index+1])));
        ui->dist_table->setItem(index,3, new QTableWidgetItem(QString::number(vbc->subject_greater_null[index+1])));
        ui->dist_table->setItem(index,4, new QTableWidgetItem(QString::number(vbc->subject_lesser_null[index+1])));
        ui->dist_table->setItem(index,5, new QTableWidgetItem(QString::number(vbc->subject_greater[index+1])));
        ui->dist_table->setItem(index,6, new QTableWidgetItem(QString::number(vbc->subject_lesser[index+1])));
    }
    ui->dist_table->selectRow(0);
}

void vbc_dialog::on_subject_list_itemSelectionChanged()
{
    image::basic_image<float,2> slice;
    vbc->get_subject_slice(ui->subject_list->currentRow(),ui->AxiSlider->value(),slice);
    image::normalize(slice);
    image::color_image color_slice(slice.geometry());
    std::copy(slice.begin(),slice.end(),color_slice.begin());
    QImage qimage((unsigned char*)&*color_slice.begin(),color_slice.width(),color_slice.height(),QImage::Format_RGB32);
    vbc_slice_image = qimage.scaled(color_slice.width()*ui->zoom->value(),color_slice.height()*ui->zoom->value());
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
                work_dir + "/dist_report.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    ui->null_dist->saveTxt(filename);
}

void vbc_dialog::on_save_fdr_dist_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                work_dir + "/fdr_report.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    ui->fdr_dist->saveTxt(filename);
}



void vbc_dialog::on_open_files_clicked()
{
    QStringList file_name = QFileDialog::getOpenFileNames(
                                this,
                "Select subject fib file for analysis",
                work_dir,"Fib files (*.fib.gz);;All files (*)" );
    if (file_name.isEmpty())
        return;
    QStringList filenames;
    file_names.clear();
    for(unsigned int index = 0;index < file_name.size();++index)
    {
        filenames << QFileInfo(file_name[index]).baseName();
        file_names.push_back(file_name[index].toLocal8Bit().begin());
    }
    ((QStringListModel*)ui->individual_list->model())->setStringList(filenames);

    if(!vbc->read_subject_data(file_names,individual_data))
    {
        QMessageBox::information(this,"error",vbc->error_msg.c_str(),0);
        return;
    }
    ui->run->setEnabled(true);
}


void vbc_dialog::on_open_mr_files_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                work_dir,
                "Text file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    file_names.clear();
    file_names.push_back(filename.toLocal8Bit().begin());

    mr.clear();

    ui->subject_demo->clear();
    if(ui->rb_multiple_regression->isChecked())
    {
        mr.type = 1;
        std::ifstream in(filename.toLocal8Bit().begin());
        std::string line;
        std::vector<std::string> titles;
        // read title
        {
            std::getline(in,line);
            std::istringstream str(line);
            std::copy(std::istream_iterator<std::string>(str),
                      std::istream_iterator<std::string>(),std::back_inserter(titles));
            mr.feature_count = titles.size()+1; // additional one for intercept
        }
        for(unsigned int index = 0;index < vbc->subject_count() && std::getline(in,line);++index)
        {
            std::istringstream str(line);
            std::vector<double> values;
            std::copy(std::istream_iterator<double>(str),
                      std::istream_iterator<double>(),std::back_inserter(values));
            if(values.size() != titles.size())
            {
                QMessageBox::information(this,"Error",QString("Cannot parse:") + line.c_str(),0);
                return;
            }
            mr.X.push_back(1); // for the intercep
            for(unsigned int j = 0;j < values.size();++j)
                mr.X.push_back(values[j]);
        }
        if(mr.X.size()/mr.feature_count != vbc->subject_count())
            QMessageBox::information(this,"Warning","Subject number mismatch");
        QStringList t;
        t << "Subject ID";
        for(unsigned int index = 0;index < titles.size();++index)
        {
            std::replace(titles[index].begin(),titles[index].end(),'/','_');
            std::replace(titles[index].begin(),titles[index].end(),'\\','_');
            t << titles[index].c_str();
        }
        ui->foi->clear();
        ui->foi->addItems(t);
        ui->foi->removeItem(0);
        ui->foi->setCurrentIndex(ui->foi->count()-1);
        ui->foi_widget->show();
        ui->subject_demo->setColumnCount(titles.size()+1);
        ui->subject_demo->setHorizontalHeaderLabels(t);
    }
    if(ui->rb_group_difference->isChecked())
    {
        mr.type = 0;
        std::ifstream in(filename.toLocal8Bit().begin());
        std::copy(std::istream_iterator<int>(in),
                  std::istream_iterator<int>(),std::back_inserter(mr.label));
        if(mr.label.size() != vbc->subject_count())
            QMessageBox::information(this,"Warning","Subject number mismatch");
        ui->subject_demo->setColumnCount(2);
        ui->subject_demo->setHorizontalHeaderLabels(QStringList() << "Subject ID" << "Group ID");
    }


    ui->subject_demo->setRowCount(vbc->subject_count());
    for(unsigned int row = 0,index = 0;row < ui->subject_demo->rowCount();++row)
    {
        ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->subject_name(row).c_str())));
        if(ui->rb_multiple_regression->isChecked())
        {
            ++index;// skip intercep
            for(unsigned int col = 1;col < ui->subject_demo->columnCount();++col,++index)
                ui->subject_demo->setItem(row,col,new QTableWidgetItem(QString::number(mr.X[index])));
        }
        if(ui->rb_group_difference->isChecked())
            ui->subject_demo->setItem(row,1,new QTableWidgetItem(QString::number(mr.label[row])));
    }
    if(!mr.pre_process())
    {
        QMessageBox::information(this,"Error","Invalid subjet information for statistical analysis",0);
        ui->run->setEnabled(false);
        return;
    }
    ui->run->setEnabled(true);
}

void vbc_dialog::on_view_mr_result_clicked()
{
    /*
    begin_prog("loading");
    mr.study_feature = ui->foi->currentIndex()+1;
    vbc->calculate_spm(mr,cur_subject_fib,mr.subject_index);
    std::auto_ptr<FibData> new_data(new FibData);
    *(new_data.get()) = *(vbc->handle);
    std::ostringstream out;
    out << " Diffusion MRI connectometry was conducted to study the effect of "
        << " on diffusion ODF. The multiple regression includes";
    new_data->report += out.str();
    cur_subject_fib.add_greater_lesser_mapping_for_tracking(new_data.get());
    tracking_window* new_mdi = new tracking_window(this,new_data.release());
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->absolute_path = work_dir;
    new_mdi->setWindowTitle(QString("Connectometry mapping"));
    new_mdi->showNormal();
    check_prog(0,0);
    */
}

void vbc_dialog::on_rb_individual_analysis_clicked()
{
    ui->individual_demo->show();
    ui->multiple_regression_demo->hide();
}

void vbc_dialog::on_rb_group_difference_clicked()
{
    ui->individual_demo->hide();
    ui->multiple_regression_demo->show();
    ui->regression_feature->hide();
}

void vbc_dialog::on_rb_multiple_regression_clicked()
{
    ui->individual_demo->hide();
    ui->multiple_regression_demo->show();
    ui->regression_feature->show();
}

void vbc_dialog::on_rb_paired_difference_clicked()
{
    ui->individual_demo->hide();
    ui->multiple_regression_demo->show();
    ui->regression_feature->show();
}

void vbc_dialog::calculate_FDR(void)
{
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();
    QString report;
    if(!vbc->handle->report.empty())
        report = vbc->handle->report.c_str();
    if(!vbc->report.empty())
        report += vbc->report.c_str();

    if(ui->rb_individual_analysis->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed no tracks with significant increase in anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy, and the FDR was " << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed no tracks with significant decrease in anisotropy.";
        else
            out << " The analysis results showed tracks with decreased anisotropy, and the FDR was " << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }
    if(ui->rb_multiple_regression->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed that there is no tracks with significantly increased anisotropy due to " << ui->foi->currentText().toLocal8Bit().begin() << ".";
        else
            out << " The analysis results showed tracks with increased anisotropy due to "
                << ui->foi->currentText().toLocal8Bit().begin()
                << ", and the FDR was " << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed that there is no tracks with significantly decreased anisotropy due to " << ui->foi->currentText().toLocal8Bit().begin() << ".";
        else
            out << " The analysis results showed tracks with decreased anisotropy due to "
                << ui->foi->currentText().toLocal8Bit().begin()
                << ", and the FDR was " << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }
    if(ui->rb_group_difference->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed that there is no tracks in group 0 with significantly increased anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy in group 0, and the FDR was " << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] >= 0.2)
            out << " The analysis results showed that there is no tracks in group 1 with significantly increased anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy in group 1, and the FDR was " << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }


    ui->textBrowser->setText(report);

    if(vbc->total_count >= vbc->permutation_count)
    {
        timer->stop();
        vbc->save_tracks_files();

        {
            std::ofstream out((vbc->trk_file_names[0]+".report.txt").c_str());
            out << report.toLocal8Bit().begin() << std::endl;
        }

        QMessageBox::information(this,"Finished","Trk files saved.",0);
        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset(0);
    }
    else
        ui->progressBar->setValue(100*vbc->total_count/vbc->permutation_count);
}
void vbc_dialog::on_run_clicked()
{
    if(ui->run->text() == "Stop")
    {
        vbc->clear_thread();
        timer->stop();
        timer.reset(0);
        ui->progressBar->setValue(0);
        ui->run->setText("Run");
        return;
    }
    ui->run->setText("Stop");
    ui->span_to->setValue(ui->length_threshold->value()*2);
    std::ostringstream out;
    vbc->permutation_count = ui->mr_permutation->value();
    vbc->length_threshold = ui->length_threshold->value();
    vbc->pruning = ui->pruning->value();
    vbc->trk_file_names = file_names;

    if(ui->rb_individual_analysis->isChecked())
    {
        vbc->tracking_threshold = 1.0-ui->percentile->value();
        vbc->individual_data = individual_data;

        out << "\nDiffusion MRI connectometry was conducted to identify affected pathway in "
            << vbc->individual_data.size() << " study patients.";
        out << " The diffusion data of the patients were compared with "
            << vbc->subject_count() << " normal subjects, and percentile rank was calculated for each fiber direction.";
        out << " A percentile threshold of " << ui->percentile->value() << " was used to select fiber orientations with deviant condition.";
    }
    if(ui->rb_group_difference->isChecked())
    {
        boost::math::students_t::students_t_distribution dist(vbc->subject_count()-2);
        vbc->tracking_threshold = boost::math::quantile(boost::math::complement(dist,ui->percentile->value()));
        vbc->individual_data.clear();
        vbc->model = mr;
        out << "\nDiffusion MRI connectometry was conducted to compare group differences."
            << " A p-value threshold of " << ui->percentile->value() << " was used to select fiber directions with substantial difference in anisotropy.";
        vbc->trk_file_names[0] += ".group";
    }
    if(ui->rb_multiple_regression->isChecked())
    {
        boost::math::students_t::students_t_distribution dist(vbc->subject_count()-mr.feature_count-1);
        vbc->tracking_threshold = boost::math::quantile(boost::math::complement(dist,ui->percentile->value()));
        mr.study_feature = ui->foi->currentIndex()+1;
        vbc->individual_data.clear();
        vbc->model = mr;
        out << "\nDiffusion MRI connectometry was conducted using a multiple regression model considering ";
        for(unsigned int index = 0;index < (int)ui->foi->count()-1;++index)
            out << ui->foi->itemText(index).toLower().toLocal8Bit().begin() << ", ";
        out << "and " << ui->foi->itemText(ui->foi->count()-1).toLower().toLocal8Bit().begin() << ".";
        out << " A p-value threshold of " << ui->percentile->value()
            << " was used to select fiber directions correlated with "
            << ui->foi->currentText().toLower().toLocal8Bit().begin() << ".";
        vbc->trk_file_names[0] += ".";
        vbc->trk_file_names[0] += ui->foi->currentText().toLower().toLocal8Bit().begin();
    }
    out << " A deterministic fiber tracking algorithm was conducted to connect these fiber orientations, and a length threshold of "
        << ui->length_threshold->value() << " mm was used to select tracks.";
    out << " The false discovery rate was calculated using a total of " << ui->mr_permutation->value() << " randomized permutations.";

    vbc->report = out.str().c_str();
    vbc->run_permutation(ui->multithread->value());
    timer.reset(new QTimer(this));
    timer->setInterval(1000);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(calculate_FDR()));
    timer->start();
}

void vbc_dialog::on_save_name_list_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save name list",
                work_dir + "/name.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toLocal8Bit().begin());
    for(unsigned int index = 0;index < vbc->subject_count();++index)
        out << vbc->subject_name(index) << std::endl;
}

void vbc_dialog::on_advanced_options_clicked()
{
    if(ui->advanced_options_box->isHidden())
        ui->advanced_options_box->show();
    else
        ui->advanced_options_box->hide();
}
