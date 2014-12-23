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
extern std::vector<atlas> atlas_list;

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

    ui->subject_list->setRowCount(vbc->subject_count());
    std::string check_quality,bad_r2;
    for(unsigned int index = 0;index < vbc->subject_count();++index)
    {
        ui->subject_list->setItem(index,0, new QTableWidgetItem(QString(vbc->subject_name(index).c_str())));
        ui->subject_list->setItem(index,1, new QTableWidgetItem(QString::number(0)));
        ui->subject_list->setItem(index,2, new QTableWidgetItem(QString::number(vbc->subject_R2(index))));
        if(vbc->subject_R2(index) < 0.3)
        {
            if(check_quality.empty())
                check_quality = "Low R2 value found in subject(s):";
            std::ostringstream out;
            out << " #" << index+1 << " " << vbc->subject_name(index);
            check_quality += out.str();
        }
        if(vbc->subject_R2(index) != vbc->subject_R2(index))
        {
            if(bad_r2.empty())
                bad_r2 = "Invalid data found in subject(s):";
            std::ostringstream out;
            out << " #" << index+1 << " " << vbc->subject_name(index);
            bad_r2 += out.str();
        }

    }
    ui->AxiSlider->setMaximum(vbc->handle->dim[2]-1);
    ui->AxiSlider->setMinimum(0);
    ui->AxiSlider->setValue(vbc->handle->dim[2] >> 1);
    ui->x_pos->setMaximum(vbc->handle->dim[0]-1);
    ui->y_pos->setMaximum(vbc->handle->dim[1]-1);
    ui->z_pos->setMaximum(vbc->handle->dim[2]-1);

    for(int index = 0; index < atlas_list.size(); ++index)
        ui->atlas_box->addItem(atlas_list[index].name.c_str());

    // dist report
    connect(ui->span_to,SIGNAL(valueChanged(int)),this,SLOT(show_report()));
    connect(ui->span_to,SIGNAL(valueChanged(int)),this,SLOT(show_fdr_report()));
    connect(ui->view_legend,SIGNAL(toggled(bool)),this,SLOT(show_report()));
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
    ui->show_result->hide();
    ui->ROI_widget->hide();
    on_rb_multiple_regression_clicked();
    qApp->installEventFilter(this);

    if(!check_quality.empty())
        QMessageBox::information(this,"Warning",check_quality.c_str());
    if(!bad_r2.empty())
        QMessageBox::information(this,"Warning",bad_r2.c_str());

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

    ui->x_pos->setValue(std::floor(pos[0] + 0.5));
    ui->y_pos->setValue(std::floor(pos[1] + 0.5));
    ui->z_pos->setValue(std::floor(pos[2] + 0.5));


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
    ui->fdr_dist->xAxis->setGrid(false);
    ui->fdr_dist->yAxis->setGrid(false);
    ui->fdr_dist->xAxis2->setVisible(true);
    ui->fdr_dist->xAxis2->setTicks(false);
    ui->fdr_dist->xAxis2->setTickLabels(false);
    ui->fdr_dist->yAxis2->setVisible(true);
    ui->fdr_dist->yAxis2->setTicks(false);
    ui->fdr_dist->yAxis2->setTickLabels(false);
    ui->fdr_dist->legend->setVisible(ui->view_legend->isChecked());
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
    ui->null_dist->xAxis->setGrid(false);
    ui->null_dist->yAxis->setGrid(false);
    ui->null_dist->xAxis2->setVisible(true);
    ui->null_dist->xAxis2->setTicks(false);
    ui->null_dist->xAxis2->setTickLabels(false);
    ui->null_dist->yAxis2->setVisible(true);
    ui->null_dist->yAxis2->setTicks(false);
    ui->null_dist->yAxis2->setTickLabels(false);
    ui->null_dist->legend->setVisible(ui->view_legend->isChecked());
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

    std::vector<std::vector<float> > new_individual_data;
    if(!vbc->read_subject_data(file_names,new_individual_data))
    {
        QMessageBox::information(this,"error",vbc->error_msg.c_str(),0);
        return;
    }
    individual_data.swap(new_individual_data);
    if(ui->rb_individual_analysis->isChecked())
    {
        mr.type = 2;
        ((QStringListModel*)ui->individual_list->model())->setStringList(filenames);
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

    std::vector<std::string> items;
    std::ifstream in(filename.toLocal8Bit().begin());
    std::copy(std::istream_iterator<std::string>(in),
              std::istream_iterator<std::string>(),std::back_inserter(items));


    mr.clear();
    if(ui->rb_multiple_regression->isChecked())
    {
        unsigned int feature_count = items.size()/(vbc->subject_count()+1);
        if(feature_count*(vbc->subject_count()+1) != items.size())
        {
            QMessageBox::information(this,"Warning",QString("Subject number mismatch."));
            return;
        }
        std::vector<double> X;
        for(unsigned int i = 0,index = 0;i < vbc->subject_count();++i)
        {
            bool ok = false;
            X.push_back(1); // for the intercep
            for(unsigned int j = 0;j < feature_count;++j,++index)
            {
                X.push_back(QString(items[index+feature_count].c_str()).toDouble(&ok));
                if(!ok)
                {
                    QMessageBox::information(this,"Error",QString("Cannot parse '") +
                                             QString(items[index+feature_count].c_str()) +
                                             QString("' at subject%1 feature%2.").arg(i+1).arg(j+1),0);
                    return;
                }
            }
        }
        mr.type = 1;
        mr.X = X;
        mr.feature_count = feature_count+1; // additional one for intercept
        QStringList t;
        t << "Subject ID";
        for(unsigned int index = 0;index < feature_count;++index)
        {
            std::replace(items[index].begin(),items[index].end(),'/','_');
            std::replace(items[index].begin(),items[index].end(),'\\','_');
            t << items[index].c_str();
        }
        ui->foi->clear();
        ui->foi->addItems(t);
        ui->foi->removeItem(0);
        ui->foi->setCurrentIndex(ui->foi->count()-1);
        ui->foi_widget->show();
        ui->subject_demo->clear();
        ui->subject_demo->setColumnCount(feature_count+1);
        ui->subject_demo->setHorizontalHeaderLabels(t);
        ui->subject_demo->setRowCount(vbc->subject_count());
        for(unsigned int row = 0,index = 0;row < ui->subject_demo->rowCount();++row)
        {
            ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->subject_name(row).c_str())));
            ++index;// skip intercep
            for(unsigned int col = 1;col < ui->subject_demo->columnCount();++col,++index)
                ui->subject_demo->setItem(row,col,new QTableWidgetItem(QString::number(mr.X[index])));
        }
    }
    if(ui->rb_group_difference->isChecked())
    {
        if(vbc->subject_count() != items.size() &&
           vbc->subject_count()+1 != items.size())
        {
            QMessageBox::information(this,"Warning",
                                     QString("Subject number mismatch. text file=%1 database=%2").
                                     arg(items.size()).arg(vbc->subject_count()));
            return;
        }
        if(vbc->subject_count()+1 == items.size())
            items.erase(items.begin());

        std::vector<int> label;
        for(unsigned int i = 0;i < vbc->subject_count();++i)
        {
            bool ok = false;
            label.push_back(QString(items[i].c_str()).toInt(&ok));
            if(!ok)
            {
                QMessageBox::information(this,"Error",QString("Cannot parse ") +
                                             QString(items[i].c_str()) +
                                             QString(" at subject%1").arg(i+1),0);
                return;
            }
        }

        mr.type = 0;
        mr.label = label;
        ui->subject_demo->clear();
        ui->subject_demo->setColumnCount(2);
        ui->subject_demo->setHorizontalHeaderLabels(QStringList() << "Subject ID" << "Group ID");
        ui->subject_demo->setRowCount(vbc->subject_count());
        for(unsigned int row = 0;row < ui->subject_demo->rowCount();++row)
        {
            ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->subject_name(row).c_str())));
            ui->subject_demo->setItem(row,1,new QTableWidgetItem(QString::number(mr.label[row])));
        }
    }
    if(ui->rb_paired_difference->isChecked())
    {
        if(vbc->subject_count() != items.size() &&
           vbc->subject_count()+1 != items.size())
        {
            QMessageBox::information(this,"Warning",
                                     QString("Subject number mismatch. text file=%1 database=%2").
                                     arg(items.size()).arg(vbc->subject_count()));
            return;
        }
        if(vbc->subject_count()+1 == items.size())
            items.erase(items.begin());

        std::vector<int> label;
        for(unsigned int i = 0;i < vbc->subject_count();++i)
        {
            bool ok = false;
            label.push_back(QString(items[i].c_str()).toInt(&ok));
            if(!ok)
            {
                QMessageBox::information(this,"Error",QString("Cannot parse ") +
                                             QString(items[i].c_str()) +
                                             QString(" at subject%1").arg(i+1),0);
                return;
            }
        }

        mr.type = 3;
        mr.pre.clear();
        mr.post.clear();
        for(unsigned int i = 0;i < label.size() && i < vbc->subject_count();++i)
            if(label[i] > 0)
            {
                for(unsigned int j = 0;j < label.size() && j < vbc->subject_count();++j)
                    if(label[j] == -label[i])
                    {
                        mr.pre.push_back(i);
                        mr.post.push_back(j);
                    }
            }
        ui->subject_demo->clear();
        ui->subject_demo->setColumnCount(2);
        ui->subject_demo->setHorizontalHeaderLabels(QStringList() << "Subject ID" << "Matched ID");
        ui->subject_demo->setRowCount(mr.pre.size());
        for(unsigned int row = 0;row < ui->subject_demo->rowCount();++row)
        {
            ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->subject_name(mr.pre[row]).c_str())));
            ui->subject_demo->setItem(row,1,new QTableWidgetItem(QString(vbc->subject_name(mr.post[row]).c_str())));
        }
    }



    if(!mr.pre_process())
    {
        QMessageBox::information(this,"Error","Invalid subjet information for statistical analysis",0);
        ui->run->setEnabled(false);
        return;
    }
    ui->run->setEnabled(true);
}

void vbc_dialog::on_rb_individual_analysis_clicked()
{
    ui->individual_demo->show();
    ui->individual_list->show();

    ui->multiple_regression_demo->hide();
    ui->subject_demo->hide();

    ui->percentile->show();
    ui->t_threshold->hide();
    ui->percentage_dif->hide();
    ui->percentage_label->show();
    ui->threshold_label->setText("Percentile");
    ui->explaination->setText("25~50%:physiological difference, 5~25%:psychiatric diseases, 0~5%: neurological diseases");
}

void vbc_dialog::on_rb_group_difference_clicked()
{
    ui->individual_demo->hide();
    ui->individual_list->hide();
    ui->multiple_regression_demo->show();
    ui->subject_demo->show();
    ui->regression_feature->hide();
    ui->percentile->hide();
    ui->t_threshold->hide();
    ui->percentage_dif->show();
    ui->percentage_label->show();
    ui->threshold_label->setText("Percentage difference");
    ui->explaination->setText("0~10%:physiological difference, 10~40%:psychiatric diseases, 40%~100%: neurological diseases");
}

void vbc_dialog::on_rb_multiple_regression_clicked()
{
    ui->individual_demo->hide();
    ui->individual_list->hide();

    ui->multiple_regression_demo->show();
    ui->subject_demo->show();

    ui->regression_feature->show();
    ui->percentile->hide();
    ui->t_threshold->show();
    ui->percentage_dif->hide();
    ui->percentage_label->hide();
    ui->threshold_label->setText("T threshold");
    ui->explaination->setText("<1.5:physiological difference, 1.5~2.5:psychiatric diseases, >2.5: neurological diseases");
}

void vbc_dialog::on_rb_paired_difference_clicked()
{
    ui->individual_demo->hide();
    ui->individual_list->hide();
    ui->multiple_regression_demo->show();
    ui->subject_demo->show();
    ui->regression_feature->hide();
    ui->percentile->hide();
    ui->t_threshold->hide();
    ui->percentage_dif->show();
    ui->percentage_label->show();
    ui->threshold_label->setText("Percentage difference");
    ui->explaination->setText("0~10%:physiological difference, 10~40%:psychiatric diseases, 40%~100%: neurological diseases");
}

void vbc_dialog::calculate_FDR(void)
{
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();
    report.clear();
    if(!vbc->handle->report.empty())
        report = vbc->handle->report.c_str();
    if(!vbc->report.empty())
        report += vbc->report.c_str();
    if(ui->rb_individual_analysis->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] > 0.5)
            out << " The analysis results showed no track with significant increase in anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy with an FDR of "
                << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] > 0.5)
            out << " The analysis results showed no track with significant decrease in anisotropy.";
        else
            out << " The analysis results showed tracks with decreased anisotropy with an FDR of "
                << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }
    if(ui->rb_multiple_regression->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] > 0.5)
            out << " The analysis results showed that there is no track with significantly increased anisotropy related to " << ui->foi->currentText().toLocal8Bit().begin() << ".";
        else
            out << " The analysis results showed tracks with increased anisotropy related to "
                << ui->foi->currentText().toLocal8Bit().begin() << " with an FDR of "
                << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] > 0.5)
            out << " The analysis results showed that there is no track with significantly decreased anisotropy related to " << ui->foi->currentText().toLocal8Bit().begin() << ".";
        else
            out << " The analysis results showed tracks with decreased anisotropy related to "
                << ui->foi->currentText().toLocal8Bit().begin() << " with an FDR of "
                << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }
    if(ui->rb_group_difference->isChecked() || ui->rb_paired_difference->isChecked())
    {
        std::ostringstream out;
        if(vbc->fdr_greater[vbc->length_threshold] > 0.5)
            out << " The analysis results showed that there is no track in group 0 with significantly increased anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy in group 0 with an FDR of "
                << vbc->fdr_greater[vbc->length_threshold] << ".";

        if(vbc->fdr_lesser[vbc->length_threshold] > 0.5)
            out << " The analysis results showed that there is no track in group 1 with significantly increased anisotropy.";
        else
            out << " The analysis results showed tracks with increased anisotropy in group 1 with an FDR of "
                << vbc->fdr_lesser[vbc->length_threshold] << ".";
        report += out.str().c_str();
    }

    ui->textBrowser->setText(report);

    if(vbc->total_count >= vbc->permutation_count)
    {
        timer->stop();
        // save trk files
        vbc->save_tracks_files(saved_file_name);
        // save report in text
        std::ofstream out((vbc->trk_file_names[0]+".report.txt").c_str());
        out << report.toLocal8Bit().begin() << std::endl;
        // save pdf plot and value txt
        ui->fdr_dist->savePdf((vbc->trk_file_names[0]+".fdr.pdf").c_str(),true,300,300);
        ui->null_dist->savePdf((vbc->trk_file_names[0]+".dist.pdf").c_str(),true,300,300);
        ui->fdr_dist->saveTxt((vbc->trk_file_names[0]+".fdr_value.txt").c_str());
        ui->null_dist->saveTxt((vbc->trk_file_names[0]+".dist_value.txt").c_str());

        QMessageBox::information(this,"Finished","Trk files saved.",0);
        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset(0);
        ui->show_result->show();
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
    ui->show_result->hide();
    ui->run->setText("Stop");
    ui->span_to->setValue(80);
    std::ostringstream out;
    vbc->permutation_count = ui->mr_permutation->value();
    vbc->length_threshold = ui->length_threshold->value();
    vbc->pruning = ui->pruning->value();
    vbc->trk_file_names = file_names;
    vbc->model = mr;
    vbc->individual_data.clear();



    if(ui->rb_individual_analysis->isChecked())
    {
        vbc->tracking_threshold = 1.0-(float)ui->percentile->value()*0.01;
        vbc->individual_data = individual_data;
        out << "\nDiffusion MRI connectometry (Yeh et al. Neuroimage Clin 2, 912, 2013) was conducted to identify affected pathway in "
            << vbc->individual_data.size() << " study patients.";
        out << " The diffusion data of the patients were compared with "
            << vbc->subject_count() << " normal subjects, and percentile rank was calculated for each fiber direction.";
        out << " A percentile rank threshold of " << ui->percentile->value() << "% was used to select fiber orientations with deviant condition.";
        for(unsigned int index = 0;index < vbc->trk_file_names.size();++index)
        {
            vbc->trk_file_names[index] += ".ind.p";
            vbc->trk_file_names[index] += QString::number(ui->percentile->value()).toLocal8Bit().begin();
        }
    }
    if(ui->rb_group_difference->isChecked())
    {
        vbc->tracking_threshold = (float)ui->percentage_dif->value()*0.01;
        out << "\nDiffusion MRI connectometry (Yeh et al. Neuroimage Clin 2, 912, 2013) was conducted to compare group differences."
            << " The group difference was quantified using percentage measurement (i.e. 2*(d1-d2)/(d1+d2) x %), where d1 and d2 are the group averages of the spin distribution function (SDF)."
            << " A threshold of " << ui->percentage_dif->value() << "% difference was used to select fiber directions with substantial difference in anisotropy.";
        vbc->trk_file_names[0] += ".group.p";
        vbc->trk_file_names[0] += QString::number(ui->percentage_dif->value()).toLocal8Bit().begin();

    }
    if(ui->rb_paired_difference->isChecked())
    {
        vbc->tracking_threshold = (float)ui->percentage_dif->value()*0.01;
        out << "\nDiffusion MRI connectometry (Yeh et al. Neuroimage Clin 2, 912, 2013) was conducted to compare paired group differences."
            << " A threshold of " << ui->percentage_dif->value() << "% difference was used to select fiber directions with substantial difference in anisotropy.";
        vbc->trk_file_names[0] += ".paired.p";
        vbc->trk_file_names[0] += QString::number(ui->percentage_dif->value()).toLocal8Bit().begin();
    }
    if(ui->rb_multiple_regression->isChecked())
    {
        vbc->tracking_threshold = ui->t_threshold->value();
        vbc->model.study_feature = ui->foi->currentIndex()+1;
        out << "\nDiffusion MRI connectometry (Yeh et al. Neuroimage Clin 2, 912, 2013) was conducted using a multiple regression model considering ";
        for(unsigned int index = 0;index < (int)ui->foi->count()-1;++index)
            out << ui->foi->itemText(index).toLower().toLocal8Bit().begin() << ", ";
        out << "and " << ui->foi->itemText(ui->foi->count()-1).toLower().toLocal8Bit().begin() << ".";
        out << " A t-score threshold of " << ui->t_threshold->value()
            << " was used to select fiber directions correlated with "
            << ui->foi->currentText().toLower().toLocal8Bit().begin() << ".";
        vbc->trk_file_names[0] += ".";
        vbc->trk_file_names[0] += ui->foi->currentText().toLower().toLocal8Bit().begin();
        vbc->trk_file_names[0] += ".t";
        vbc->trk_file_names[0] += QString::number(ui->t_threshold->value()).toLocal8Bit().begin();
    }

    out << " A deterministic fiber tracking algorithm (Yeh, et al. PLoS ONE 8(11): e80713) was conducted to connect these fiber directions";

    // load region
    vbc->roi.clear();
    if(ui->roi_file->isChecked() || ui->roi_atlas->isChecked())
    {
        unsigned short label = ui->atlas_region_box->currentIndex();
        image::geometry<3> geo = vbc->handle->dim;
        for (image::pixel_index<3>index; index.is_valid(geo); index.next(geo))
        {
            image::vector<3> pos((const unsigned int*)(index.begin()));
            image::vector<3> mni;
            image::vector_transformation(pos.begin(),mni.begin(), vbc->handle->trans_to_mni,image::vdim<3>());
            if(ui->roi_file->isChecked() && !study_region.is_labeled_as(mni,label))
                continue;
            if(ui->roi_atlas->isChecked() &&
                    !atlas_list[ui->atlas_box->currentIndex()].is_labeled_as(mni, label))
                continue;
            vbc->roi.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
        }
        vbc->roi_type = ui->region_type->currentIndex();
        out << " using ";
        if(ui->roi_atlas->isChecked())
            out << ui->atlas_region_box->currentText().toLocal8Bit().begin() << " ("
                << ui->atlas_box->currentText().toLocal8Bit().begin() << " atlas)";
        else
            out << study_region_file_name.toLocal8Bit().begin();
        out << " as the " << ui->region_type->currentText().toLocal8Bit().begin() << ".";
    }
    else
        out << " in whole brain regions.";
    out << " Tracks with length greater than " <<
            ui->length_threshold->value() << " mm were collected.";
    out << " To estimate the false discovery rate, a total of "
        << ui->mr_permutation->value()
        << " randomized permutations were applied to the group label to obtain the null distribution of the track length.";

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

void vbc_dialog::on_show_result_clicked()
{
    std::auto_ptr<FibData> new_data(new FibData);
    *(new_data.get()) = *(vbc->handle);
    if(!report.isEmpty())
    {
        std::ostringstream out;
        out << report.toLocal8Bit().begin();
        new_data->report += out.str();
    }
    if(vbc->model.type != 2) // not individual
    {
        result_fib.reset(new fib_data);
        std::vector<unsigned int> permu(vbc->subject_count());
        for(unsigned int index = 0;index < permu.size();++index)
            permu[index] = index;
        vbc->calculate_spm(vbc->model,*result_fib.get(),permu);
        for(unsigned int index = 0;index < new_data->dim.size();++index)
        {
            if(result_fib->lesser[0][index] < vbc->tracking_threshold)
                result_fib->lesser[0][index] = 0;
            if(result_fib->greater[0][index] < vbc->tracking_threshold)
                result_fib->greater[0][index] = 0;
        }
        new_data->view_item.push_back(ViewItem());
        new_data->view_item.back().name = "lesser";
        new_data->view_item.back().image_data = image::make_image(new_data->dim,result_fib->lesser_ptr[0]);
        new_data->view_item.back().is_overlay = true;
        new_data->view_item.back().set_scale(result_fib->lesser_ptr[0],
                                             result_fib->lesser_ptr[0]+new_data->dim.size());
        new_data->view_item.push_back(ViewItem());
        new_data->view_item.back().name = "greater";
        new_data->view_item.back().image_data = image::make_image(new_data->dim,result_fib->greater_ptr[0]);
        new_data->view_item.back().is_overlay = true;
        new_data->view_item.back().set_scale(result_fib->greater_ptr[0],
                                             result_fib->greater_ptr[0]+new_data->dim.size());

    }
    tracking_window* current_tracking_window = new tracking_window(this,new_data.release());
    current_tracking_window->setAttribute(Qt::WA_DeleteOnClose);
    current_tracking_window->absolute_path = work_dir;
    current_tracking_window->setWindowTitle(QString("Connectometry mapping"));
    current_tracking_window->showNormal();
    current_tracking_window->tractWidget->delete_all_tract();
    QStringList filenames;
    for(unsigned int index = 0;index < saved_file_name.size();++index)
        filenames << saved_file_name[index].c_str();
    current_tracking_window->tractWidget->load_tracts(filenames);

}

void vbc_dialog::on_roi_whole_brain_toggled(bool checked)
{
    if(checked)
        ui->ROI_widget->hide();
}

void vbc_dialog::on_roi_file_toggled(bool checked)
{
    if(checked)
    {
        QString openfilename = QFileDialog::getOpenFileName(
                    this,
                    "Load ROI from file",
                    work_dir + "/roi.nii.gz",
                    "Report file (*.txt *.nii *.nii.gz);;All files (*)");
        if(openfilename.isEmpty())
        {
            ui->roi_whole_brain->setChecked(true);
            return;
        }
        if(!study_region.load_from_file(openfilename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"Error","Invalid nifti file format",0);
            ui->roi_whole_brain->setChecked(true);
            return;
        }
        study_region_file_name = QFileInfo(openfilename).completeBaseName();
        ui->roi_file->setText(QString("Assigned by file:") + study_region_file_name);
        ui->ROI_widget->show();
        ui->atlas_box->hide();
        ui->atlas_region_box->clear();
        for (unsigned int index = 0; index < study_region.get_list().size(); ++index)
            ui->atlas_region_box->addItem(study_region.get_list()[index].c_str());
        ui->atlas_region_box->show();
    }
    else
        ui->roi_file->setText(QString("Assigned by file"));
}

void vbc_dialog::on_atlas_box_currentIndexChanged(int i)
{
    ui->atlas_region_box->clear();
    for (unsigned int index = 0; index < atlas_list[i].get_list().size(); ++index)
        ui->atlas_region_box->addItem(atlas_list[i].get_list()[index].c_str());
}

void vbc_dialog::on_roi_atlas_toggled(bool checked)
{
    if(checked)
    {
        ui->ROI_widget->show();
        ui->atlas_box->show();
        ui->atlas_region_box->show();
    }
}



void vbc_dialog::on_remove_subject_clicked()
{
    if(ui->rb_group_difference->isChecked() || ui->rb_multiple_regression->isChecked())
    if(ui->subject_demo->currentRow() >= 0 && vbc->subject_count() > 1)
    {
        unsigned int index = ui->subject_demo->currentRow();
        vbc->remove_subject(index);
        mr.remove_subject(index);
        ui->subject_demo->removeRow(index);
        ui->subject_list->removeRow(index);
    }
}

void vbc_dialog::on_remove_sel_subject_clicked()
{
    if(ui->subject_list->currentRow() >= 0 && vbc->subject_count() > 1)
    {
        unsigned int index = ui->subject_list->currentRow();
        ui->subject_list->removeRow(index);
        vbc->remove_subject(index);
        if(ui->subject_demo->rowCount() > 1 &&
          (ui->rb_group_difference->isChecked() || ui->rb_multiple_regression->isChecked()))
        {
            ui->subject_demo->removeRow(index);
            mr.remove_subject(index);
        }
    }
}

void vbc_dialog::on_toolBox_currentChanged(int index)
{
    if(index > 1 && !ui->run->isEnabled())
    {
        QMessageBox::information(this,"Missing information","Please provide patient information in STEP1 before going to STEP2 and 3",0);
        ui->toolBox->setCurrentIndex(1);
    }
}

void vbc_dialog::on_x_pos_valueChanged(int arg1)
{
    // show data
    std::vector<float> vbc_data;
    vbc->get_data_at(
            image::pixel_index<3>(ui->x_pos->value(),
                                  ui->y_pos->value(),
                                  ui->z_pos->value(),
                                  vbc->handle->dim).index(),0,vbc_data);
    if(vbc_data.empty())
        return;
    if(ui->run->isEnabled() && ui->rb_multiple_regression->isChecked())
    {
        QVector<double> variables(vbc->subject_count());
        for(unsigned int i = 0;i < vbc->subject_count();++i)
            variables[i] = mr.X[i*mr.feature_count+ui->foi->currentIndex()+1];

        QVector<double> y(vbc->subject_count());
        std::copy(vbc_data.begin(),vbc_data.end(),y.begin());

        ui->vbc_report->clearGraphs();
        ui->vbc_report->addGraph();
        ui->vbc_report->graph(0)->setLineStyle(QCPGraph::lsNone);
        ui->vbc_report->graph(0)->setScatterStyle(QCP::ScatterStyle(ui->scatter->value()));
        ui->vbc_report->graph(0)->setData(variables, y);
        float min_x = *std::min_element(variables.begin(),variables.end());
        float max_x = *std::max_element(variables.begin(),variables.end());
        float min_y = *std::min_element(vbc_data.begin(),vbc_data.end());
        float max_y = *std::max_element(vbc_data.begin(),vbc_data.end());

        ui->vbc_report->xAxis->setRange(min_x-(max_x-min_x)*0.1,
                                        max_x+(max_x-min_x)*0.1);
        ui->vbc_report->xAxis->setLabel(ui->foi->currentText());
        ui->vbc_report->yAxis->setRange(min_y-(max_y-min_y)*0.1,
                                        max_y+(max_y-min_y)*0.1);
        ui->vbc_report->yAxis->setLabel("QA");

    }
    else
    {
        for(unsigned int index = 0;index < vbc->subject_count();++index)
            ui->subject_list->item(index,1)->setText(QString::number(vbc_data[index]));

        vbc_data.erase(std::remove(vbc_data.begin(),vbc_data.end(),0.0),vbc_data.end());
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
    }

    ui->vbc_report->xAxis2->setVisible(true);
    ui->vbc_report->xAxis2->setTicks(false);
    ui->vbc_report->xAxis2->setTickLabels(false);
    ui->vbc_report->yAxis2->setVisible(true);
    ui->vbc_report->yAxis2->setTicks(false);
    ui->vbc_report->yAxis2->setTickLabels(false);

    ui->vbc_report->xAxis->setGrid(false);
    ui->vbc_report->yAxis->setGrid(false);
    ui->vbc_report->replot();

}

void vbc_dialog::on_y_pos_valueChanged(int arg1)
{
    on_x_pos_valueChanged(0);
}

void vbc_dialog::on_z_pos_valueChanged(int arg1)
{
    on_x_pos_valueChanged(0);
}

void vbc_dialog::on_scatter_valueChanged(int arg1)
{
    on_x_pos_valueChanged(0);
}

void vbc_dialog::on_save_report_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                work_dir + "/report.jpg",
                "JPEC file (*.jpg);;BMP file (*.bmp);;PDF file (*.pdf);;PNG file (*.png);;All files (*)");
    if(QFileInfo(filename).completeSuffix().toLower() == "jpg")
        ui->vbc_report->saveJpg(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "bmp")
        ui->vbc_report->saveBmp(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "png")
        ui->vbc_report->savePng(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "pdf")
        ui->vbc_report->savePdf(filename,true,300,300);
}
