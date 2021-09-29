#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include "tracking/region/Regions.h"
#include "group_connectometry.hpp"
#include "ui_group_connectometry.h"
#include "ui_tracking_window.h"
#include "tracking/tracking_window.h"
#include "tracking/tract/tracttablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "tracking/atlasdialog.h"
#include "tracking/roi.hpp"
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
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


group_connectometry::group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_,QString db_file_name_) :
    QDialog(parent),
    null_pos_chart(new QChart),null_neg_chart(new QChart),
    null_pos_chart_view(new QChartView(null_pos_chart)),null_neg_chart_view(new QChartView(null_neg_chart)),
    fdr_chart(new QChart),fdr_chart_view(new QChartView(fdr_chart)),
    db_file_name(db_file_name_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),
    vbc(vbc_),db(vbc->handle->db),
    ui(new Ui::group_connectometry)
{

    ui->setupUi(this);
    ui->chart_widget_layout->addWidget(null_pos_chart_view);
    ui->chart_widget_layout->addWidget(null_neg_chart_view);
    ui->chart_widget_layout->addWidget(fdr_chart_view);
    null_pos_chart->setMargins(QMargins(0,0,0,0));
    null_pos_chart->setBackgroundRoundness(0);
    null_neg_chart->setMargins(QMargins(0,0,0,0));
    null_neg_chart->setBackgroundRoundness(0);
    fdr_chart->setMargins(QMargins(0,0,0,0));
    fdr_chart->setBackgroundRoundness(0);

    setMouseTracking(true);
    ui->roi_table->setItemDelegate(new ROIViewDelegate(ui->roi_table));
    ui->roi_table->setAlternatingRowColors(true);


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
    ui->foi_widget->hide();
    on_roi_whole_brain_toggled(true);

    // CHECK R2
    std::string check_quality;
    for(unsigned int index = 0;index < db.num_subjects;++index)
    {
        if(db.R2[index] < 0.5)
        {
            if(check_quality.empty())
                check_quality = "Poor image quality found found in subject(s):";
            std::ostringstream out;
            out << " #" << index+1 << " " << db.subject_names[index];
            check_quality += out.str();
        }
    }
    // setup normalize QA
    if(db.index_name == "qa")
        ui->normalize_qa->setChecked(true);
    else
    {
        ui->normalize_qa->setChecked(false);
        ui->normalize_qa->setVisible(false);
    }

    if(!check_quality.empty())
    {
        if(has_gui)
            QMessageBox::information(this,"Warning",check_quality.c_str());
        else
            std::cout << check_quality << std::endl;
    }
    ui->subject_demo->clear();
    ui->subject_demo->setColumnCount(1);
    ui->subject_demo->setHorizontalHeaderLabels(QStringList("Subject ID"));
    ui->subject_demo->setRowCount(db.num_subjects);
    for(unsigned int row = 0;row < ui->subject_demo->rowCount();++row)
        ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(db.subject_names[row].c_str())));

    ui->advanced_options->setCurrentIndex(0);
}

group_connectometry::~group_connectometry()
{
    qApp->removeEventFilter(this);
    delete ui;
}

template<typename data_type>
QLineSeries* get_line_series(const data_type& data, const char* name)
{
    QLineSeries* series = new QLineSeries;
    series->setName(name);
    auto max_size = data.size();
    while(max_size > 0 && (data[max_size-1] == 0.0f || data[max_size-1] == 1.0f))
        --max_size;
    ++max_size;
    for(size_t i = 0;i < max_size;++i)
        series->append(i,double(data[i]));
    return series;
}
void group_connectometry::show_fdr_report()
{
    if(vbc->fdr_pos_corr.empty())
        return;
    fdr_chart->removeAllSeries();
    fdr_chart->addSeries(get_line_series(vbc->fdr_pos_corr,"positive correlation"));
    fdr_chart->addSeries(get_line_series(vbc->fdr_neg_corr,"negative correlation"));
    fdr_chart->createDefaultAxes();
    fdr_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    fdr_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    fdr_chart->axes(Qt::Vertical).back()->setTitleText("FDR");
    fdr_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    fdr_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    fdr_chart->axes(Qt::Vertical).back()->setRange(0,1);

    fdr_chart->setTitle("FDR versus Track Length");
    ((QValueAxis*)fdr_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)fdr_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
}



void group_connectometry::show_report()
{
    if(vbc->subject_pos_corr_null.empty())
        return;

    null_pos_chart->removeAllSeries();
    null_neg_chart->removeAllSeries();
    null_pos_chart->addSeries(get_line_series(vbc->subject_pos_corr_null,"permuted positive correlation"));
    null_pos_chart->addSeries(get_line_series(vbc->subject_pos_corr,"nonpermuted positive correlation"));
    null_neg_chart->addSeries(get_line_series(vbc->subject_neg_corr_null,"permuted negative correlation"));
    null_neg_chart->addSeries(get_line_series(vbc->subject_neg_corr,"nonpermuted negative correlation"));
    null_pos_chart->createDefaultAxes();
    null_pos_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    null_pos_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    null_pos_chart->axes(Qt::Vertical).back()->setTitleText("Count");
    null_pos_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    null_pos_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    null_pos_chart->setTitle("Track count versus length (positive correlation)");
    null_neg_chart->createDefaultAxes();
    null_neg_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    null_neg_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    null_neg_chart->axes(Qt::Vertical).back()->setTitleText("Count");
    null_neg_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    null_neg_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    null_neg_chart->setTitle("Track count versus length (negative correlation)");
    ((QValueAxis*)null_pos_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)null_pos_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
    ((QValueAxis*)null_neg_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)null_neg_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
}

void group_connectometry::show_dis_table(void)
{
    ui->dist_table->setRowCount(100);
    for(unsigned int index = vbc->length_threshold_voxels;index < vbc->fdr_pos_corr.size()-1;++index)
    {
        int row = int(index-vbc->length_threshold_voxels);
        ui->dist_table->setItem(row,0,new QTableWidgetItem(QString::number(index)));
        ui->dist_table->setItem(row,1, new QTableWidgetItem(QString::number(double(vbc->fdr_pos_corr[index]))));
        ui->dist_table->setItem(row,2, new QTableWidgetItem(QString::number(double(vbc->fdr_neg_corr[index]))));
        ui->dist_table->setItem(row,3, new QTableWidgetItem(QString::number(vbc->subject_pos_corr_null[index])));
        ui->dist_table->setItem(row,4, new QTableWidgetItem(QString::number(vbc->subject_neg_corr_null[index])));
        ui->dist_table->setItem(row,5, new QTableWidgetItem(QString::number(vbc->subject_pos_corr[index])));
        ui->dist_table->setItem(row,6, new QTableWidgetItem(QString::number(vbc->subject_neg_corr[index])));
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
    std::string error_msg;
    if(!load_demographic_file(filename,error_msg))
        QMessageBox::information(this,"Error",error_msg.c_str(),0);
}


void fill_demo_table(const connectometry_db& db,
                     QTableWidget* table)
{
    QStringList t2;
    t2 << "Subject";
    for(size_t i = 0;i < db.titles.size();++i)
        t2 << db.titles[i].c_str();
    table->clear();
    table->setColumnCount(t2.size());
    table->setHorizontalHeaderLabels(t2);
    table->setRowCount(int(db.num_subjects));
    for(size_t row = 0;row < db.num_subjects;++row)
    {
        table->setItem(int(row),0,new QTableWidgetItem(QString(db.subject_names[row].c_str())));
        for(size_t col = 0;col < db.titles.size();++col)
        {
            auto item_pos = size_t(row)*db.titles.size()+col;
            if(item_pos < db.items.size())
                table->setItem(int(row),int(col)+1,new QTableWidgetItem(QString(db.items[item_pos].c_str())));
            else
                table->setItem(int(row),int(col)+1,new QTableWidgetItem(QString()));
        }
    }
}

bool group_connectometry::load_demographic_file(QString filename,std::string& error_msg)
{
    // read demographic file
    if(!db.parse_demo(filename.toStdString()))
    {
        error_msg = db.error_msg;
        return false;
    }
    demo_file_name = filename.toStdString();
    model.reset(new stat_model);
    model->read_demo(db);
    // fill up regression values
    {
        QStringList t;
        for(size_t i = 0; i < db.feature_titles.size();++i)
            t << db.feature_titles[i].c_str();
        ui->variable_list->clear();
        ui->variable_list->addItems(t);
        for(int i = 0;i < ui->variable_list->count();++i)
        {
            ui->variable_list->item(i)->setFlags(ui->variable_list->item(i)->flags() | Qt::ItemIsUserCheckable); // set checkable flag
            ui->variable_list->item(i)->setCheckState(i == 0 ? Qt::Checked : Qt::Unchecked);
            ui->variable_list->setItemAlignment(Qt::AlignLeft);
        }
        ui->cohort_index->clear();
        ui->cohort_index->addItems(t);
        ui->cohort_index->setCurrentIndex(0);
        ui->foi_widget->show();
    }
    on_variable_list_clicked(QModelIndex());
    fill_demo_table(db,ui->subject_demo);
    return true;
}

void group_connectometry::calculate_FDR(void)
{
    if(vbc->progress == 100)
    {
        if(timer.get())
            timer->stop();
    }

    ui->progressBar->setValue(vbc->progress);
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();


    std::string output;
    vbc->generate_report(output);
    int pos = ui->textBrowser->verticalScrollBar()->value();
    ui->textBrowser->setHtml(output.c_str());
    ui->textBrowser->verticalScrollBar()->setValue(pos);


    if(vbc->progress < 100)
        return;

    // progress = 100
    {

        // output distribution image
        delete null_pos_chart_view;
        delete null_neg_chart_view;
        delete fdr_chart_view;
        null_pos_chart = new QChart;
        null_pos_chart_view = new QChartView(null_pos_chart);
        null_neg_chart = new QChart;
        null_neg_chart_view = new QChartView(null_neg_chart);
        fdr_chart = new QChart;
        fdr_chart_view = new QChartView(fdr_chart);
        null_pos_chart->setMargins(QMargins(0,0,0,0));
        null_pos_chart->setBackgroundRoundness(0);
        null_neg_chart->setMargins(QMargins(0,0,0,0));
        null_neg_chart->setBackgroundRoundness(0);
        fdr_chart->setMargins(QMargins(0,0,0,0));
        fdr_chart->setBackgroundRoundness(0);
        show_report();
        show_fdr_report();
        null_pos_chart_view->grab().save((vbc->output_file_name+".pos_corr.dist.jpg").c_str());
        null_neg_chart_view->grab().save((vbc->output_file_name+".neg_corr.dist.jpg").c_str());
        fdr_chart_view->grab().save((vbc->output_file_name+".fdr.jpg").c_str());
        ui->chart_widget_layout->addWidget(null_pos_chart_view);
        ui->chart_widget_layout->addWidget(null_neg_chart_view);
        ui->chart_widget_layout->addWidget(fdr_chart_view);


        if(vbc->pos_corr_track->get_visible_track_count() ||
           vbc->neg_corr_track->get_visible_track_count())
            QMessageBox::information(this,"Finished","Trk files saved.",0);
        else
            QMessageBox::information(this,"Finished","No significant finding.",0);

        // save report in text
        {
            std::ofstream out((vbc->output_file_name+".report.html").c_str());
            out << output << std::endl;
        }

        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset();
    }
}
void group_connectometry::on_run_clicked()
{
    if(ui->run->text() == "Stop")
    {
        vbc->clear();
        timer->stop();
        timer.reset();
        ui->progressBar->setValue(0);
        ui->run->setText("Run");
        return;
    }
    // longitudinal data wihtout loading demographics
    if(db.is_longitudinal && !model.get())
    {
        model.reset(new stat_model);
        model->read_demo(vbc->handle->db);
    }

    if(!model.get())
    {
        QMessageBox::information(this,"DSI Studio","Load demographic file first");
        return;
    }

    // check cohort text
    on_show_cohort_clicked();
    if(model->remove_list.empty()) // select cohort failed
        return;

    // setup parameters
    {
        vbc->normalize_qa = ui->normalize_qa->isChecked();
        vbc->no_tractogram = ui->no_tractogram->isChecked();
        vbc->foi_str = ui->foi->currentText().toStdString();
        vbc->length_threshold_voxels = uint32_t(ui->length_threshold->value());
        vbc->tip = uint32_t(ui->tip->value());
        if(ui->fdr_control->isChecked())
            vbc->fdr_threshold = float(ui->fdr_threshold->value());
        else
            vbc->fdr_threshold = 0.0f;

        vbc->tracking_threshold = float(ui->threshold->value());
        vbc->output_file_name = ui->output_name->text().toStdString();
    }

    // setup statistical model
    {
        vbc->model.reset(new stat_model);
        *(vbc->model.get()) = *(model.get());
        vbc->model->nonparametric = ui->nonparametric->isChecked();
        if(!vbc->model->select_feature(db,ui->foi->currentText().toStdString()))
        {
            QMessageBox::critical(this,"Error",vbc->model->error_msg.c_str());
            return;
        }
    }

    // setup roi
    {
        vbc->roi_mgr = std::make_shared<RoiMgr>(vbc->handle);
        if(ui->exclude_cb->isChecked())
            vbc->exclude_cerebellum();

        // apply ROI
        if(!ui->roi_whole_brain->isChecked())
        {
            std::vector<unsigned char> roi_type(roi_list.size());
            std::vector<std::string> roi_name(roi_list.size());
            for(unsigned int index = 0;index < roi_list.size();++index)
            {
                roi_type[index] = uint8_t(ui->roi_table->item(int(index),2)->text().toInt());
                roi_name[index] = ui->roi_table->item(int(index),0)->text().toStdString();
            }
            for(unsigned int index = 0;index < roi_list.size();++index)
                vbc->roi_mgr->setRegions(roi_list[index],1.0f,roi_type[index],roi_name[index].c_str());
        }

        // if no seed assigned, assign whole brain
        if(vbc->roi_mgr->seeds.empty())
            vbc->roi_mgr->setWholeBrainSeed(vbc->fiber_threshold);
    }

    vbc->run_permutation(std::thread::hardware_concurrency(),uint32_t(ui->permutation_count->value()));


    ui->run->setText("Stop");
    timer.reset(new QTimer(this));
    timer->setInterval(1000);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(calculate_FDR()));
    timer->start();
}

void group_connectometry::on_show_result_clicked()
{
    if(!vbc->model.get())
        return;
    std::shared_ptr<fib_data> new_data(new fib_data);
    *(new_data.get()) = *(vbc->handle);
    {
        result_fib.reset(new connectometry_result);
        stat_model info;
        info.resample(*(vbc->model.get()),false,false,0);
        vbc->calculate_spm(*result_fib.get(),info,vbc->normalize_qa);
        new_data->view_item.push_back(item("dec_t",result_fib->neg_corr_ptr[0],new_data->dim));
        new_data->view_item.push_back(item("inc_t",result_fib->pos_corr_ptr[0],new_data->dim));
    }
    tracking_window* current_tracking_window = new tracking_window(this,new_data);
    current_tracking_window->setAttribute(Qt::WA_DeleteOnClose);
    current_tracking_window->setWindowTitle(vbc->output_file_name.c_str());
    current_tracking_window->showNormal();
    current_tracking_window->tractWidget->delete_all_tract();
    current_tracking_window->tractWidget->addNewTracts("Positive Correlation");
    current_tracking_window->tractWidget->addNewTracts("Negative Correlation");

    current_tracking_window->tractWidget->tract_models[0]->add(*(vbc->pos_corr_track.get()));
    current_tracking_window->tractWidget->tract_models[1]->add(*(vbc->neg_corr_track.get()));

    current_tracking_window->command("set_zoom","0.8");
    current_tracking_window->command("set_param","show_surface","1");
    current_tracking_window->command("set_param","show_slice","0");
    current_tracking_window->command("set_param","show_region","0");
    current_tracking_window->command("set_param","bkg_color","16777215");
    current_tracking_window->command("set_param","surface_alpha","0.2");
    current_tracking_window->command("set_roi_view_index","wm");
    current_tracking_window->command("add_surface","","0.1");
    current_tracking_window->command("update_track");


}

void group_connectometry::on_roi_whole_brain_toggled(bool checked)
{
    ui->roi_table->setEnabled(!checked);
    ui->load_roi_from_atlas->setEnabled(!checked);
    ui->clear_all_roi->setEnabled(!checked);
    ui->load_roi_from_file->setEnabled(!checked);
}

void group_connectometry::add_new_roi(QString name,QString source,
                                      const std::vector<tipl::vector<3,short> >& new_roi,
                                      int type)
{
    ui->roi_table->setRowCount(ui->roi_table->rowCount()+1);
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,0,new QTableWidgetItem(name));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,1,new QTableWidgetItem(source));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,2,new QTableWidgetItem(QString::number(type)));
    ui->roi_table->openPersistentEditor(ui->roi_table->item(ui->roi_table->rowCount()-1,2));
    roi_list.push_back(new_roi);
}

void group_connectometry::on_load_roi_from_atlas_clicked()
{
    if(vbc->handle->atlas_list.empty())
        return;
    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,vbc->handle));
    if(atlas_dialog->exec() == QDialog::Accepted)
    {
        for(unsigned int i = 0;i < atlas_dialog->roi_list.size();++i)
        {
            ROIRegion roi(vbc->handle);
            if(!load_region(vbc->handle,roi,atlas_dialog->atlas_name + ":" + atlas_dialog->roi_name[i]))
                return;
            add_new_roi(atlas_dialog->roi_name[i].c_str(),atlas_dialog->atlas_name.c_str(),
                             roi.get_region_voxels_raw());
        }
    }
}

void group_connectometry::on_clear_all_roi_clicked()
{
    roi_list.clear();
    ui->roi_table->setRowCount(0);
}

void group_connectometry::on_load_roi_from_file_clicked()
{
    QString file = QFileDialog::getOpenFileName(
                                this,
                                "Load ROI from file",
                                work_dir + "/roi.nii.gz",
                                "Report file (*.nii *nii.gz);;Text files (*.txt);;All files (*)");
    if(file.isEmpty())
        return;
    tipl::image<float,3> I;
    tipl::matrix<4,4> transform;
    gz_nifti nii;
    if(!nii.load_from_file(file.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid nifti file format",0);
        return;
    }
    nii.toLPS(I);
    nii.get_image_transformation(transform);
    transform.inv();
    transform *= vbc->handle->trans_to_mni;
    std::vector<tipl::vector<3,short> > new_roi;
    for (tipl::pixel_index<3> index(vbc->handle->dim);index < vbc->handle->dim.size();++index)
    {
        tipl::vector<3> pos(index);
        pos.to(transform);
        pos.round();
        if(!I.shape().is_valid(pos) || I.at(pos[0],pos[1],pos[2]) == 0)
            continue;
        new_roi.push_back(tipl::vector<3,short>((const unsigned int*)index.begin()));
    }
    if(new_roi.empty())
    {
        QMessageBox::information(this,"Error","The nifti contain no voxel with value greater than 0.",0);
        return;
    }
    add_new_roi(QFileInfo(file).baseName(),"Local File",new_roi);
}

void group_connectometry::on_variable_list_clicked(const QModelIndex &)
{
    ui->foi->clear();
    for(int i =0;i < ui->variable_list->count();++i)
        if((db.feature_selected[uint32_t(i)] = (ui->variable_list->item(i)->checkState() == Qt::Checked)))
            ui->foi->addItem(ui->variable_list->item(i)->text());
    if(db.is_longitudinal)
        ui->foi->addItem(QString("Intercept"));
    if(ui->foi->count() != 0)
        ui->foi->setCurrentIndex(ui->foi->count()-1);
}

void group_connectometry::on_show_cohort_clicked()
{
    if(!model.get())
        return;
    if(!model->select_cohort(db,ui->select_text->text().toStdString()))
    {
        QMessageBox::critical(this,"Error",model->error_msg.c_str());
        return;
    }
    size_t selected_count = 0;
    ui->subject_demo->setUpdatesEnabled(false);
    for(size_t i = 0;i < model->remove_list.size();++i)
    {
        if(!model->remove_list[i])
            selected_count++;
        for(int j = 0;j < ui->subject_demo->columnCount();++j)
            ui->subject_demo->item(int(i),j)->setBackgroundColor(model->remove_list[i] ? Qt::white : QColor(255,255,200));
    }
    ui->subject_demo->setUpdatesEnabled(true);
    ui->cohort_report->setText(QString("n=%1").arg(selected_count));

}

void group_connectometry::on_fdr_control_toggled(bool checked)
{
    ui->fdr_threshold->setEnabled(checked);
    ui->fdr_label->setEnabled(checked);
}

void group_connectometry::on_apply_selection_clicked()
{
    QString new_text(ui->select_text->text());
    if(!new_text.isEmpty())
        new_text += ",";
    new_text += ui->cohort_index->currentText();
    new_text += (ui->cohort_operator->currentIndex() == 3 ? QString("/") : ui->cohort_operator->currentText());
    new_text += ui->cohort_value->text();
    ui->select_text->setText(new_text);
    on_show_cohort_clicked();
}

