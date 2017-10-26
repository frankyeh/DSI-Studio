#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include "group_connectometry.hpp"
#include "ui_group_connectometry.h"
#include "ui_tracking_window.h"
#include "tracking/tracking_window.h"
#include "tracking/tract/tracttablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "tracking/atlasdialog.h"
extern std::vector<atlas> atlas_list;

QWidget *ROIViewDelegate::createEditor(QWidget *parent,
                                     const QStyleOptionViewItem &option,
                                     const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        QComboBox *comboBox = new QComboBox(parent);
        comboBox->addItem("ROI");
        comboBox->addItem("ROA");
        comboBox->addItem("End");
        comboBox->addItem("Seed");
        comboBox->addItem("Terminative");
        connect(comboBox, SIGNAL(activated(int)), this, SLOT(emitCommitData()));
        return comboBox;
    }
    else
        return QItemDelegate::createEditor(parent,option,index);

}

void ROIViewDelegate::setEditorData(QWidget *editor,
                                  const QModelIndex &index) const
{

    if (index.column() == 2)
        ((QComboBox*)editor)->setCurrentIndex(index.model()->data(index).toString().toInt());
    else
        return QItemDelegate::setEditorData(editor,index);
}

void ROIViewDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                 const QModelIndex &index) const
{
    if (index.column() == 2)
        model->setData(index,QString::number(((QComboBox*)editor)->currentIndex()));
    else
        QItemDelegate::setModelData(editor,model,index);
}

void ROIViewDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}


group_connectometry::group_connectometry(QWidget *parent,std::shared_ptr<vbc_database> vbc_,QString db_file_name_,bool gui_) :
    QDialog(parent),vbc(vbc_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
    ui(new Ui::group_connectometry)
{

    ui->setupUi(this);
    setMouseTracking(true);
    ui->roi_table->setItemDelegate(new ROIViewDelegate(ui->roi_table));
    ui->roi_table->setAlternatingRowColors(true);


    ui->multithread->setValue(std::thread::hardware_concurrency());
    ui->advanced_options->hide();


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

    ui->foi_widget->hide();
    on_roi_whole_brain_toggled(true);

    // CHECK R2
    std::string check_quality;
    for(unsigned int index = 0;index < vbc->handle->db.num_subjects;++index)
    {
        if(vbc->handle->db.R2[index] < 0.5)
        {
            if(check_quality.empty())
                check_quality = "Poor image quality found found in subject(s):";
            std::ostringstream out;
            out << " #" << index+1 << " " << vbc->handle->db.subject_names[index];
            check_quality += out.str();
        }
    }
    if(!check_quality.empty())
        QMessageBox::information(this,"Warning",check_quality.c_str());

    ui->subject_demo->clear();
    ui->subject_demo->setColumnCount(1);
    ui->subject_demo->setHorizontalHeaderLabels(QStringList("Subject ID"));
    ui->subject_demo->setRowCount(vbc->handle->db.num_subjects);
    for(unsigned int row = 0;row < ui->subject_demo->rowCount();++row)
        ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->handle->db.subject_names[row].c_str())));
}

group_connectometry::~group_connectometry()
{
    qApp->removeEventFilter(this);
    delete ui;
}

void group_connectometry::show_fdr_report()
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
    color[0] = QColor(20,20,255,255);
    color[1] = QColor(255,20,20,255);
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
        pen.setWidth(2);
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
    legendFont.setPointSize(8); // and make a bit smaller for legend
    ui->fdr_dist->legend->setFont(legendFont);
    ui->fdr_dist->legend->setPositionStyle(QCPLegend::psTopRight);
    ui->fdr_dist->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->fdr_dist->replot();

}

void group_connectometry::show_report()
{
    if(vbc->subject_greater_null.empty())
        return;
    ui->null_dist->clearGraphs();
    std::vector<std::vector<unsigned int> > vbc_data;
    char legends[4][60] = {"permuted greater","permuted lesser","nonpermuted greater","nonpermuted lesser"};
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
        x_size = std::max<unsigned int>(x_size,vbc_data[i].size()-1);
    if(x_size == 0)
        return;
    QVector<double> x(x_size);
    std::vector<QVector<double> > y(vbc_data.size());
    for(unsigned int i = 0;i < vbc_data.size();++i)
        y[i].resize(x_size);

    // tracks length is at least 2 mm, so skip length < 2
    for(unsigned int j = 2;j < x_size && j < vbc_data[0].size();++j)
    {
        x[j-2] = (float)j;
        for(unsigned int i = 0; i < vbc_data.size(); ++i)
            y[i][j-2] = vbc_data[i][j];
    }

    QPen pen;
    QColor color[4];
    color[0] = QColor(20,20,255,255);
    color[1] = QColor(255,20,20,255);
    color[2] = QColor(20,255,20,255);
    color[3] = QColor(20,255,255,255);
    for(unsigned int i = 0; i < vbc_data.size(); ++i)
    {
        ui->null_dist->addGraph();
        pen.setColor(color[i]);
        pen.setWidth(2);
        ui->null_dist->graph()->setLineStyle(QCPGraph::lsLine);
        ui->null_dist->graph()->setPen(pen);
        ui->null_dist->graph()->setData(x, y[i]);
        ui->null_dist->graph()->setName(QString(legend[i]));
    }

    ui->null_dist->xAxis->setLabel("mm");
    ui->null_dist->yAxis->setLabel("count");
    ui->null_dist->xAxis->setRange(0,ui->span_to->value());
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
    legendFont.setPointSize(8); // and make a bit smaller for legend
    ui->null_dist->legend->setFont(legendFont);
    ui->null_dist->legend->setPositionStyle(QCPLegend::psTopRight);
    ui->null_dist->legend->setBrush(QBrush(QColor(255,255,255,230)));
    ui->null_dist->replot();
}

void group_connectometry::show_dis_table(void)
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

void group_connectometry::on_open_mr_files_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                work_dir,
                "Text or CSV file (*.txt *.csv);;All files (*)");
    if(filename.isEmpty())
        return;
    load_demographic_file(filename);
}

bool group_connectometry::load_demographic_file(QString filename)
{
    model.reset(new stat_model);
    model->init(vbc->handle->db.num_subjects);
    file_names.clear();
    file_names.push_back(filename.toLocal8Bit().begin());

    int col_count = 0;
    std::vector<std::string> titles;
    std::vector<std::string> items;
    // read demographic file
    {
        int row_count = 0,last_item_size = 0;
        bool is_csv = (QFileInfo(filename).completeSuffix() == "csv");
        std::ifstream in(filename.toLocal8Bit().begin());
        if(!in)
        {
            if(gui)
                QMessageBox::information(this,"Error",QString("Cannot find the demographic file."));
            else
                std::cout << "cannot find the demographic file at" << filename.toLocal8Bit().begin() << std::endl;
            return false;
        }

        std::string line;
        while(std::getline(in,line))
        {
            if(is_csv)
            {
                QString str(line.c_str());
                QStringList values = str.split(',');
                for(int i = 0;i < values.size();++i)
                    items.push_back(values[i].toStdString());
            }
            else
            {
                std::istringstream in2(line);
                std::copy(std::istream_iterator<std::string>(in2),
                          std::istream_iterator<std::string>(),std::back_inserter(items));
            }
            if(items.size() == last_item_size)
                break;
            ++row_count;
            if(col_count == 0)
                col_count = items.size();
            else
                if(items.size()-last_item_size != col_count)
                {
                    QString error_msg = QString("Row number %1 has %2 fields, which is different from the column size %3.").arg(row_count).arg(items.size()-last_item_size).arg(col_count);
                    if(!gui)
                        std::cout << error_msg.toStdString() << std::endl;
                    items.resize(col_count+last_item_size);
                }
            last_item_size = items.size();
        }
        if(row_count == 1)
            col_count = items.size()/(vbc->handle->db.num_subjects+1);

        if(items.size()/col_count < 2)
        {
            if(gui)
                QMessageBox::information(0,"Error","Invalid demographic file format",0);
            else
                std::cout << "Invalid demographic file format" << std::endl;
            return false;
        }
        // check subject count for command line
        if(items.size()/col_count != vbc->handle->db.num_subjects+1 && !gui) // +1 for title
        {
            QString error_msg = QString("Subject number mismatch. The demographic file has %1 subjects, but the database has %2 subjects.").arg(row_count-1).arg(vbc->handle->db.num_subjects);
            std::cout << error_msg.toStdString() << std::endl;
            return false;
        }
    }
    // first line moved to title vector
    titles.insert(titles.end(),items.begin(),items.begin()+col_count);
    items.erase(items.begin(),items.begin()+col_count);

    // get age and sex from subject name
    if((QString(vbc->handle->db.subject_names[0].c_str()).contains("_M") ||
        QString(vbc->handle->db.subject_names[0].c_str()).contains("_F")) &&
        QString(vbc->handle->db.subject_names[0].c_str()).contains("Y_") && gui &&
        QMessageBox::information(this,"Connectomtetry analysis",
                                 "Pull age and sex (1 = male, 0 = female) information from connectometry db?",
                                 QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
        {
            titles.insert(titles.begin(),"Sex");
            titles.insert(titles.begin(),"Age");
            std::vector<std::string> new_items;

            for(unsigned int index = 0;index < vbc->handle->db.num_subjects;++index)
            {
                QString name = vbc->handle->db.subject_names[index].c_str();
                int pos = name.indexOf("Y_")-2;
                if(pos <= 0)
                    continue;
                new_items.push_back(name.mid(pos,2).toStdString());
                if(name.contains("_M"))
                     new_items.push_back("1");
                else
                    if(name.contains("_F"))
                        new_items.push_back("0");
                    else
                        new_items.push_back("");
                for(int i = 0;i < col_count;++i)
                {
                    int item_pos = index*col_count+i;
                    if(item_pos < items.size())
                        new_items.push_back(items[item_pos]);
                    else
                        new_items.push_back("");
                }
            }
            col_count += 2;
            items.swap(new_items);
        }

    // fill up regression values
    {
        std::vector<int> feature_location;
        for(int i = 0;i < col_count;++i)
        {
            bool okay = true;
            QString(items[i].c_str()).toDouble(&okay);
            if(okay)
                feature_location.push_back(i);
        }

        std::vector<double> X;
        for(unsigned int i = 0;i < vbc->handle->db.num_subjects;++i)
        {
            bool ok = false;
            X.push_back(1); // for the intercep
            for(unsigned int j = 0;j < feature_location.size();++j)
            {
                int item_pos = i*col_count + feature_location[j];
                if(item_pos >= items.size())
                {
                    X.push_back(ui->missing_value->value());
                    continue;
                }
                double value = QString(items[item_pos].c_str()).toDouble(&ok);
                if(!ok)
                {
                    if(!gui)
                    {
                        QString text = QString("Cannot parse '")+
                        QString(items[item_pos].c_str())+
                        QString("' at subject%1 feature%2.").arg(i+1).arg(j+1);
                        std::cout << text.toStdString() << std::endl;
                        return false;
                    }
                    value = ui->missing_value->value();
                }
                X.push_back(value);
            }
        }

        model->type = 1;
        model->X = X;
        model->feature_count = feature_location.size()+1; // additional one for intercept
        QStringList t;
        for(unsigned int index = 0;index < feature_location.size();++index)
        {
            std::replace(titles[feature_location[index]].begin(),titles[feature_location[index]].end(),'/','_');
            std::replace(titles[feature_location[index]].begin(),titles[feature_location[index]].end(),'\\','_');
            t << titles[feature_location[index]].c_str();
        }
        ui->variable_list->clear();
        ui->variable_list->addItems(t);
        for(int i = 0;i < ui->variable_list->count();++i)
        {
            ui->variable_list->item(i)->setFlags(ui->variable_list->item(i)->flags() | Qt::ItemIsUserCheckable); // set checkable flag
            ui->variable_list->item(i)->setCheckState(Qt::Checked);
        }

        ui->foi->clear();
        ui->foi->addItems(t);
        ui->foi->setCurrentIndex(ui->foi->count()-1);
        ui->foi_widget->show();
        ui->missing_data_checked->setChecked(std::find(X.begin(),X.end(),ui->missing_value->value()) != X.end());
    }

    // show demographic GUI
    {
        std::vector<int> col_has_value(col_count);
        for(int i = 0;i < col_count;++i)
        {
            bool okay = true;
            QString(items[i].c_str()).toDouble(&okay);
            col_has_value[i] = okay ? 1:0;
        }
        QStringList t2;
        t2 << "Subject";
        for(int i = 0;i < titles.size();++i)
            t2 << titles[i].c_str();
        ui->subject_demo->clear();
        ui->subject_demo->setColumnCount(t2.size());
        ui->subject_demo->setHorizontalHeaderLabels(t2);
        ui->subject_demo->setRowCount(vbc->handle->db.num_subjects);
        for(unsigned int row = 0,index = 0;row < ui->subject_demo->rowCount();++row)
        {
            ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(vbc->handle->db.subject_names[row].c_str())));
            ++index;// skip intercep
            for(unsigned int col = 0;col < col_has_value.size();++col)
            {
                if(col_has_value[col])
                {
                    ui->subject_demo->setItem(row,col+1,new QTableWidgetItem(QString::number(model->X[index])));
                    ++index;
                    continue;
                }
                int item_pos = row*col_count+col;
                if(item_pos < items.size())
                    ui->subject_demo->setItem(row,col+1,new QTableWidgetItem(QString(items[item_pos].c_str())));
                else
                    ui->subject_demo->setItem(row,col+1,new QTableWidgetItem(QString()));
            }
        }
    }
    return true;
}

bool group_connectometry::setup_model(stat_model& m)
{
    m = *(model.get());
    m.study_feature = ui->foi->currentIndex()+1; // this index is after selection
    std::vector<char> sel(ui->variable_list->count()+1);
    sel[0] = 1; // intercept
    for(int i = 1;i < sel.size();++i)
        if(ui->variable_list->item(i-1)->checkState() == Qt::Checked)
            sel[i] = 1;
    m.select_variables(sel);
    m.threshold_type = stat_model::t;
    if(ui->missing_data_checked->isChecked())
        m.remove_missing_data(ui->missing_value->value());
    return m.pre_process();
}


void group_connectometry::calculate_FDR(void)
{
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();

    if(vbc->progress == 100)
    {
        vbc->wait();// make sure that all threads done
        timer->stop();
    }
    report.clear();
    if(!vbc->handle->report.empty())
        report = vbc->handle->report.c_str();
    if(!vbc->report.empty())
        report += vbc->report.c_str();

    if(vbc->progress == 100)
    {
        if(ui->output_track_data->isChecked())
            vbc->save_tracks_files(); // this also give track recognition
    }

    std::ostringstream out;
    out << " The connectometry analysis identified "
        << (vbc->fdr_greater[vbc->length_threshold]>0.5 || !vbc->has_greater_result ? "no track": vbc->greater_tracks_result.c_str())
        << " with increased connectivity related to "
        << ui->foi->currentText().toLocal8Bit().begin() << " (FDR="
        << vbc->fdr_greater[vbc->length_threshold] << ") "
        << "and "
        << (vbc->fdr_lesser[vbc->length_threshold]>0.5 || !vbc->has_lesser_result ? "no track": vbc->lesser_tracks_result.c_str())
        << " with decreased connectivity related to "
        << ui->foi->currentText().toLocal8Bit().begin() << " (FDR="
        << vbc->fdr_lesser[vbc->length_threshold] << ").";
    report += out.str().c_str();


    report += " The analysis was conducted using DSI Studio (http://dsi-studio.labsolver.org).";
    ui->textBrowser->setText(report);

    if(vbc->progress == 100)
    {


        // save report in text
        if(ui->output_report->isChecked())
        {
            std::ofstream out((vbc->trk_file_names[0]+".report.txt").c_str());
            out << report.toLocal8Bit().begin() << std::endl;
        }
        if(ui->output_dist->isChecked())
        {
            ui->show_null_greater->setChecked(true);
            ui->show_greater->setChecked(true);
            ui->show_null_lesser->setChecked(false);
            ui->show_lesser->setChecked(false);
            ui->null_dist->saveBmp((vbc->trk_file_names[0]+".greater.dist.bmp").c_str(),300,300,3);

            ui->show_null_greater->setChecked(false);
            ui->show_greater->setChecked(false);
            ui->show_null_lesser->setChecked(true);
            ui->show_lesser->setChecked(true);
            ui->null_dist->saveBmp((vbc->trk_file_names[0]+".lesser.dist.bmp").c_str(),300,300,3);
            ui->null_dist->saveTxt((vbc->trk_file_names[0]+".dist_value.txt").c_str());
        }

        if(ui->output_fdr->isChecked())
        {
            ui->show_greater_2->setChecked(true);
            ui->show_lesser_2->setChecked(false);
            ui->fdr_dist->saveBmp((vbc->trk_file_names[0]+".greater.fdr.bmp").c_str(),300,300,3);

            ui->show_greater_2->setChecked(false);
            ui->show_lesser_2->setChecked(true);
            ui->fdr_dist->saveBmp((vbc->trk_file_names[0]+".lesser.fdr.bmp").c_str(),300,300,3);

            ui->fdr_dist->saveTxt((vbc->trk_file_names[0]+".fdr_value.txt").c_str());
        }

        // restore all checked status
        ui->show_null_greater->setChecked(true);
        ui->show_greater->setChecked(true);
        ui->show_greater_2->setChecked(true);



        if(ui->output_track_image->isChecked())
        {
            std::shared_ptr<fib_data> new_data(new fib_data);
            *(new_data.get()) = *(vbc->handle);
            tracking_window* new_mdi = new tracking_window(0,new_data);
            new_mdi->setWindowTitle(vbc->trk_file_names[0].c_str());
            new_mdi->show();
            new_mdi->resize(1024,800);
            new_mdi->command("set_zoom","0.9");
            new_mdi->command("set_param","show_slice","0");
            new_mdi->command("set_param","show_region","0");
            new_mdi->command("set_param","show_surface","1");
            new_mdi->command("set_param","surface_alpha","0.1");
            new_mdi->command("add_surface");
            new_mdi->tractWidget->addNewTracts("greater");
            new_mdi->tractWidget->tract_models[0]->add(*vbc->greater_tracks[0].get());
            new_mdi->command("update_track");
            new_mdi->command("save_h3view_image",(vbc->trk_file_names[0]+".positive.jpg").c_str());
            new_mdi->command("delete_all_tract");
            new_mdi->tractWidget->addNewTracts("lesser");
            new_mdi->tractWidget->tract_models[0]->add(*vbc->lesser_tracks[0].get());
            new_mdi->command("update_track");
            new_mdi->command("save_h3view_image",(vbc->trk_file_names[0]+".negative.jpg").c_str());
            new_mdi->close();
        }
        if(gui)
        {
            if(vbc->has_greater_result || vbc->has_lesser_result)
                QMessageBox::information(this,"Finished","Trk files saved.",0);
            else
                QMessageBox::information(this,"Finished","No significant finding.",0);
        }
        else
        {
            if(vbc->has_greater_result || vbc->has_lesser_result)
                std::cout << "trk files saved" << std::endl;
            else
                std::cout << "no significant finding" << std::endl;
        }
        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset(0);
    }
    else
        ui->progressBar->setValue(vbc->progress);
}
void group_connectometry::on_run_clicked()
{
    if(ui->run->text() == "Stop")
    {
        vbc->clear();
        timer->stop();
        timer.reset(0);
        ui->progressBar->setValue(0);
        ui->run->setText("Run");
        return;
    }
    ui->run->setText("Stop");
    ui->span_to->setValue(80);
    vbc->seeding_density = ui->seed_density->value();
    vbc->trk_file_names = file_names;
    vbc->normalize_qa = ui->normalize_qa->isChecked();
    vbc->output_resampling = ui->output_resampling->isChecked();
    vbc->length_threshold = ui->length_threshold->value();
    vbc->track_trimming = ui->track_trimming->value();
    vbc->individual_data.clear();
    vbc->tracking_threshold = ui->threshold->value();

    vbc->model.reset(new stat_model);
    if(!setup_model(*vbc->model.get()))
    {
        QMessageBox::information(this,"Error","Cannot run the statistics. Subject number not enough?",0);
        return;
    }


    std::ostringstream out;
    std::string parameter_str;
    {
        std::ostringstream out;
        if(ui->normalize_qa->isChecked())
            out << ".nqa";
        char threshold_type[5][11] = {"percentage","t","beta","percentile","mean_dif"};
        out << ".length" << ui->length_threshold->value();
        out << ".s" << ui->seed_density->value();
        out << ".p" << ui->permutation_count->value();
        out << "." << threshold_type[vbc->model->threshold_type];
        out << "." << ui->threshold->value();

        parameter_str = out.str();
    }

    out << "\nDiffusion MRI connectometry (Yeh et al. NeuroImage 125 (2016): 162-171) was used to study the effect of "
        << ui->foi->currentText().toStdString()
        << ". A multiple regression model was used to consider ";
    for(unsigned int index = 0;index < ui->foi->count();++index)
    {
        if(index && ui->foi->count() > 2)
            out << ",";
        out << " ";
        if(ui->foi->count() >= 2 && index+1 == ui->foi->count())
            out << "and ";
        out << ui->foi->itemText(index).toStdString();
    }
    out << " in a total of "
        << vbc->model->subject_index.size()
        << " subjects. ";

    vbc->trk_file_names[0] += parameter_str;
    vbc->trk_file_names[0] += ".mr.";
    vbc->trk_file_names[0] += ui->foi->currentText().toLower().toLocal8Bit().begin();


    if(ui->normalize_qa->isChecked())
        out << " The SDF was normalized.";
    out << " A T-score threshold of " << ui->threshold->value();
    out << " was assigned to select local connectomes, and the local connectomes were tracked using a deterministic fiber tracking algorithm (Yeh et al. PLoS ONE 8(11): e80713, 2013).";

    // load region
    if(!ui->roi_whole_brain->isChecked() && !roi_list.empty())
    {
        out << " The tracking algorithm used ";
        const char roi_type_name[5][20] = {"region of interst","region of avoidance","ending region","seeding region","terminating region"};
        const char roi_type_name2[5][5] = {"roi","roa","end","seed"};
        vbc->roi_list = roi_list;
        vbc->roi_r_list = roi_r_list;
        vbc->roi_type.resize(roi_list.size());
        for(unsigned int index = 0;index < roi_list.size();++index)
        {
            if(index && roi_list.size() > 2)
                out << ",";
            out << " ";
            if(roi_list.size() >= 2 && index+1 == roi_list.size())
                out << "and ";
            out << ui->roi_table->item(index,0)->text().toStdString() << " as the " << roi_type_name[ui->roi_table->item(index,2)->text().toInt()];
            QString name = ui->roi_table->item(index,0)->text();
            name = name.replace('?','_');
            name = name.replace(':','_');
            name = name.replace('/','_');
            name = name.replace('\\','_');
            vbc->roi_type[index] = ui->roi_table->item(index,2)->text().toInt();
            for(unsigned int index = 0;index < vbc->trk_file_names.size();++index)
            {
                vbc->trk_file_names[index] += ".";
                vbc->trk_file_names[index] += roi_type_name2[vbc->roi_type.front()];
                vbc->trk_file_names[index] += ".";
                vbc->trk_file_names[index] += name.toStdString();
            }
        }
        out << ".";
    }
    else
    {
        vbc->roi_list.clear();
        vbc->roi_r_list.clear();
        vbc->roi_type.clear();
    }

    if(vbc->track_trimming)
        out << " Track trimming was conducted with " << vbc->track_trimming << " iterations.";

    if(vbc->output_resampling)
        out << " All tracks generated from bootstrap resampling were included.";

    out << " A length threshold of " << ui->length_threshold->value() << " mm was used to select tracks.";
    out << " The seeding density was " <<
            ui->seed_density->value() << " seed(s) per mm3.";

    out << " To estimate the false discovery rate, a total of "
        << ui->permutation_count->value()
        << " randomized permutations were applied to the group label to obtain the null distribution of the track length.";

    vbc->report = out.str().c_str();
    vbc->run_permutation(ui->multithread->value(),ui->permutation_count->value());
    timer.reset(new QTimer(this));
    timer->setInterval(1000);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(calculate_FDR()));
    timer->start();
}

void group_connectometry::on_show_result_clicked()
{
    if((vbc->greater_tracks.empty()))\
    {
        QMessageBox::information(this,"Error","Run connectometry analysis to get the results");
        return;
    }

    std::shared_ptr<fib_data> new_data(new fib_data);
    *(new_data.get()) = *(vbc->handle);
    if(!report.isEmpty())
    {
        std::ostringstream out;
        out << report.toLocal8Bit().begin();
        new_data->report += out.str();
    }
    stat_model cur_model;
    if(!setup_model(cur_model))
    {
        QMessageBox::information(this,"Error","Cannot run the statistics. Subject number not enough?",0);
        return;
    }

    {
        char threshold_type[5][11] = {"percentage","t","beta","percentile","mean_dif"};
        result_fib.reset(new connectometry_result);
        stat_model info;
        info.resample(cur_model,false,false);
        vbc->calculate_spm(*result_fib.get(),info,vbc->normalize_qa);
        new_data->view_item.push_back(item());
        new_data->view_item.back().name = threshold_type[cur_model.threshold_type];
        new_data->view_item.back().name += "-";
        new_data->view_item.back().image_data = image::make_image(result_fib->lesser_ptr[0],new_data->dim);
        new_data->view_item.back().set_scale(result_fib->lesser_ptr[0],
                                             result_fib->lesser_ptr[0]+new_data->dim.size());
        new_data->view_item.push_back(item());
        new_data->view_item.back().name = threshold_type[cur_model.threshold_type];
        new_data->view_item.back().name += "+";
        new_data->view_item.back().image_data = image::make_image(result_fib->greater_ptr[0],new_data->dim);
        new_data->view_item.back().set_scale(result_fib->greater_ptr[0],
                                             result_fib->greater_ptr[0]+new_data->dim.size());

    }
    tracking_window* current_tracking_window = new tracking_window(this,new_data);
    current_tracking_window->setAttribute(Qt::WA_DeleteOnClose);
    current_tracking_window->setWindowTitle(vbc->trk_file_names[0].c_str());
    current_tracking_window->showNormal();
    current_tracking_window->tractWidget->delete_all_tract();
    current_tracking_window->tractWidget->addNewTracts("Positive Correlation");
    current_tracking_window->tractWidget->addNewTracts("Negative Correlation");

    current_tracking_window->tractWidget->tract_models[0]->add(*(vbc->greater_tracks[0].get()));
    current_tracking_window->tractWidget->tract_models[1]->add(*(vbc->lesser_tracks[0].get()));
    current_tracking_window->command("update_track");
    current_tracking_window->command("set_param","show_slice","0");
    current_tracking_window->command("set_param","show_region","0");
    current_tracking_window->command("set_param","show_surface","1");
    current_tracking_window->command("set_param","surface_alpha","0.1");
    current_tracking_window->command("add_surface");

}

void group_connectometry::on_roi_whole_brain_toggled(bool checked)
{
    ui->roi_table->setEnabled(!checked);
    ui->load_roi_from_atlas->setEnabled(!checked);
    ui->clear_all_roi->setEnabled(!checked);
    ui->load_roi_from_file->setEnabled(!checked);
}

void group_connectometry::on_show_advanced_clicked()
{
    if(ui->advanced_options->isVisible())
        ui->advanced_options->hide();
    else
        ui->advanced_options->show();
}

void group_connectometry::on_missing_data_checked_toggled(bool checked)
{
    ui->missing_value->setEnabled(checked);
}

void group_connectometry::add_new_roi(QString name,QString source,
                                      const std::vector<image::vector<3,short> >& new_roi,
                                      float r)
{
    ui->roi_table->setRowCount(ui->roi_table->rowCount()+1);
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,0,new QTableWidgetItem(name));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,1,new QTableWidgetItem(source));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,2,new QTableWidgetItem(QString::number(0)));
    ui->roi_table->openPersistentEditor(ui->roi_table->item(ui->roi_table->rowCount()-1,2));
    roi_list.push_back(new_roi);
    roi_r_list.push_back(r);
}

void group_connectometry::on_load_roi_from_atlas_clicked()
{
    std::auto_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this));
    if(atlas_dialog->exec() == QDialog::Accepted)
    {
        for(unsigned int i = 0;i < atlas_dialog->roi_list.size();++i)
        {
            std::vector<image::vector<3,short> > new_roi;
            float r;
            vbc->handle->get_atlas_roi(atlas_dialog->atlas_index,atlas_dialog->roi_list[i],new_roi,r);
            if(!new_roi.empty())
                add_new_roi(atlas_dialog->roi_name[i].c_str(),atlas_dialog->atlas_name.c_str(),new_roi,r);
        }
    }
}

void group_connectometry::on_clear_all_roi_clicked()
{
    roi_list.clear();
    roi_r_list.clear();
    ui->roi_table->setRowCount(0);
}

void group_connectometry::on_load_roi_from_file_clicked()
{
    QString file = QFileDialog::getOpenFileName(
                                this,
                                "Load ROI from file",
                                work_dir + "/roi.nii.gz",
                                "Report file (*.txt *.nii *nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    image::basic_image<short,3> I;
    image::matrix<4,4,float> transform;
    gz_nifti nii;
    if(!nii.load_from_file(file.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid nifti file format",0);
        return;
    }
    nii >> I;
    transform.identity();
    nii.get_image_transformation(transform.begin());
    transform.inv();
    std::vector<image::vector<3,short> > new_roi;
    for (image::pixel_index<3> index(vbc->handle->dim);index < vbc->handle->dim.size();++index)
    {
        image::vector<3> pos;
        vbc->handle->subject2mni(index,pos);
        pos.to(transform);
        pos.round();
        if(!I.geometry().is_valid(pos) || I.at(pos[0],pos[1],pos[2]) == 0)
            continue;
        new_roi.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
    }
    if(new_roi.empty())
    {
        QMessageBox::information(this,"Error","The nifti contain no voxel with value greater than 0.",0);
        return;
    }
    add_new_roi(QFileInfo(file).baseName(),"Local File",new_roi,1.0);
}








void group_connectometry::on_variable_list_clicked(const QModelIndex &)
{
    ui->foi->clear();
    for(int i =0;i < ui->variable_list->count();++i)
        if(ui->variable_list->item(i)->checkState() == Qt::Checked)
            ui->foi->addItem(ui->variable_list->item(i)->text());
    if(ui->foi->count() == 0)
    {
        QMessageBox::information(this,"Error","At least one variable needs to be selected");
        ui->variable_list->item(0)->setCheckState(Qt::Checked);
        ui->foi->addItem(ui->variable_list->item(0)->text());
    }
    ui->foi->setCurrentIndex(ui->foi->count()-1);
}
