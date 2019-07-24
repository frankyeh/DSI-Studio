#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <QScrollBar>
#include "tracking/region/Regions.h"
#include "group_connectometry.hpp"
#include "ui_group_connectometry.h"
#include "ui_tracking_window.h"
#include "tracking/tracking_window.h"
#include "tracking/tract/tracttablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "tracking/atlasdialog.h"
#include "tracking/roi.hpp"

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


group_connectometry::group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_,QString db_file_name_,bool gui_) :
    QDialog(parent),
    null_pos_chart(new QChart),null_pos_chart_view(new QChartView(null_pos_chart)),
    null_neg_chart(new QChart),null_neg_chart_view(new QChartView(null_neg_chart)),
    fdr_chart(new QChart),fdr_chart_view(new QChartView(fdr_chart)),
    db_file_name(db_file_name_),vbc(vbc_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
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

template<typename data_type>
QLineSeries* get_line_series(const data_type& data, const char* name)
{
    QLineSeries* series = new QLineSeries;
    series->setName(name);
    int max_size = data.size();
    while(max_size > 0 && (data[max_size-1] == 0 || data[max_size-1] == 1))
        --max_size;
    ++max_size;
    for(int i = 0;i < max_size;++i)
        series->append(i,data[i]);
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
    null_pos_chart->axes(Qt::Vertical).back()->setTitleText("Count");
    null_pos_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    null_pos_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    null_pos_chart->setTitle("Track count versus length (positive correlation)");
    null_neg_chart->createDefaultAxes();
    null_neg_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
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
    for(unsigned int index = 0;index < vbc->fdr_pos_corr.size()-1;++index)
    {
        ui->dist_table->setItem(index,0, new QTableWidgetItem(QString::number(index + 1)));
        ui->dist_table->setItem(index,1, new QTableWidgetItem(QString::number(vbc->fdr_pos_corr[index+1])));
        ui->dist_table->setItem(index,2, new QTableWidgetItem(QString::number(vbc->fdr_neg_corr[index+1])));
        ui->dist_table->setItem(index,3, new QTableWidgetItem(QString::number(vbc->subject_pos_corr_null[index+1])));
        ui->dist_table->setItem(index,4, new QTableWidgetItem(QString::number(vbc->subject_neg_corr_null[index+1])));
        ui->dist_table->setItem(index,5, new QTableWidgetItem(QString::number(vbc->subject_pos_corr[index+1])));
        ui->dist_table->setItem(index,6, new QTableWidgetItem(QString::number(vbc->subject_neg_corr[index+1])));
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
    std::vector<int> col_has_value(db.titles.size());
    for(int i = 0;i < db.titles.size();++i)
    {
        bool okay = true;
        QString(db.items[i].c_str()).toDouble(&okay);
        col_has_value[i] = okay ? 1:0;
    }
    QStringList t2;
    t2 << "Subject";
    for(int i = 0;i < db.titles.size();++i)
        t2 << db.titles[i].c_str();
    table->clear();
    table->setColumnCount(t2.size());
    table->setHorizontalHeaderLabels(t2);
    table->setRowCount(db.num_subjects);
    for(unsigned int row = 0,index = 0;row < table->rowCount();++row)
    {
        table->setItem(row,0,new QTableWidgetItem(QString(db.subject_names[row].c_str())));
        ++index;// skip intercep
        for(unsigned int col = 0;col < col_has_value.size();++col)
        {
            if(col_has_value[col])
            {
                table->setItem(row,col+1,new QTableWidgetItem(QString::number(db.X[index])));
                ++index;
                continue;
            }
            int item_pos = row*db.titles.size()+col;
            if(item_pos < db.items.size())
                table->setItem(row,col+1,new QTableWidgetItem(QString(db.items[item_pos].c_str())));
            else
                table->setItem(row,col+1,new QTableWidgetItem(QString()));
        }
    }
}

bool group_connectometry::load_demographic_file(QString filename,std::string& error_msg)
{
    auto& db = vbc->handle->db;
    // read demographic file
    if(!db.parse_demo(filename.toStdString(),ui->missing_value->value()))
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
        for(int i = 0; i < db.feature_titles.size();++i)
            t << db.feature_titles[i].c_str();
        ui->variable_list->clear();
        ui->variable_list->addItems(t);
        for(int i = 0;i < ui->variable_list->count();++i)
        {
            ui->variable_list->item(i)->setFlags(ui->variable_list->item(i)->flags() | Qt::ItemIsUserCheckable); // set checkable flag
            ui->variable_list->item(i)->setCheckState(i == 0 ? Qt::Checked : Qt::Unchecked);
        }
        ui->foi->clear();
        ui->foi->addItem(t[0]);
        ui->foi->setCurrentIndex(0);
        ui->foi_widget->show();
        ui->missing_data_checked->setChecked(std::find(model->X.begin(),model->X.end(),ui->missing_value->value()) != model->X.end());
    }

    fill_demo_table(db,ui->subject_demo);
    return true;
}

bool group_connectometry::setup_model(stat_model& m)
{
    if(ui->rb_regression->isChecked())
    {
        m = *(model.get());
        m.type = 1;
        m.study_feature = ui->foi->currentIndex()+1; // this index is after selection
        m.variables.clear();
        std::vector<char> sel(ui->variable_list->count()+1);
        sel[0] = 1; // intercept
        m.variables.push_back("intercept");
        for(int i = 1;i < sel.size();++i)
            if(ui->variable_list->item(i-1)->checkState() == Qt::Checked)
            {
                sel[i] = 1;
                m.variables.push_back(ui->variable_list->item(i-1)->text().toStdString());
            }
        m.select_variables(sel);
        m.threshold_type = stat_model::t;
        if(ui->missing_data_checked->isChecked())
            m.remove_missing_data(ui->missing_value->value());
    }
    if(ui->rb_longitudina_dif->isChecked())
    {
        m.type = 3;
        m.read_demo(vbc->handle->db);
        m.X.clear();
    }
    return m.pre_process();
}


void group_connectometry::calculate_FDR(void)
{
    ui->progressBar->setValue(vbc->progress);
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();

    if(vbc->progress == 100)
    {
        vbc->wait();// make sure that all threads done
        if(timer.get())
            timer->stop();

        vbc->save_tracks_files();
    }
    std::string output;
    vbc->generate_report(output);
    int pos = ui->textBrowser->verticalScrollBar()->value();
    ui->textBrowser->setHtml(output.c_str());
    ui->textBrowser->verticalScrollBar()->setValue(pos);


    if(vbc->progress < 100)
        return;

    // progress = 100
    {
        // output distribution values
        {
            std::ofstream out((vbc->output_file_name+".fdr_dist.values.txt").c_str());
            out << "voxel_dis\tfdr_pos_cor\tfdr_neg_corr\t#track_pos_corr_null\t#track_neg_corr_null\t#track_pos_corr\t#track_neg_corr" << std::endl;
            for(unsigned int index = 1;index < vbc->fdr_pos_corr.size()-1;++index)
            {
                out << index
                    << "\t" << vbc->fdr_pos_corr[index]
                    << "\t" << vbc->fdr_neg_corr[index]
                    << "\t" << vbc->subject_pos_corr_null[index]
                    << "\t" << vbc->subject_neg_corr_null[index]
                    << "\t" << vbc->subject_pos_corr[index]
                    << "\t" << vbc->subject_neg_corr[index] << std::endl;
            }
        }
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


        if(gui)
        {
            if(vbc->has_pos_corr_result || vbc->has_neg_corr_result)
                QMessageBox::information(this,"Finished","Trk files saved.",0);
            else
                QMessageBox::information(this,"Finished","No significant finding.",0);
        }
        else
        {
            if(vbc->has_pos_corr_result || vbc->has_neg_corr_result)
                std::cout << "trk files saved" << std::endl;
            else
                std::cout << "no significant finding" << std::endl;
        }

        // save report in text
        {
            std::ofstream out((vbc->output_file_name+".report.html").c_str());
            out << output << std::endl;
        }

        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset(0);
    }
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
    vbc->seed_count = ui->seed_count->value();
    vbc->normalize_qa = ui->normalize_qa->isChecked();
    vbc->output_resampling = ui->output_resampling->isChecked();
    vbc->foi_str = ui->foi->currentText().toStdString();
    if(ui->rb_fdr->isChecked())
    {
        vbc->fdr_threshold = ui->fdr_threshold->value();
        vbc->length_threshold = 10;
    }
    else
    {
        vbc->fdr_threshold = 0;
        vbc->length_threshold = ui->length_threshold->value();
    }
    vbc->track_trimming = ui->track_trimming->value();
    vbc->tracking_threshold = ui->threshold->value();
    vbc->model.reset(new stat_model);
    if(!setup_model(*vbc->model.get()))
    {
        if(gui)
            QMessageBox::information(this,"Error","Cannot run the statistics. Subject number not enough?",0);
        else
            std::cout << "setup model failed" << std::endl;
        return;
    }
    if(vbc->model->type == 1) // regression
        vbc->output_file_name = demo_file_name;
    if(vbc->model->type == 3) // longitudinal change
        vbc->output_file_name = db_file_name.toStdString();

    if(ui->roi_whole_brain->isChecked())
        roi_list.clear();

    {
        std::vector<int> roi_type(roi_list.size());
        std::vector<std::string> roi_name(roi_list.size());
        for(unsigned int index = 0;index < roi_list.size();++index)
        {
            roi_type[index] = ui->roi_table->item(index,2)->text().toInt();
            roi_name[index] = ui->roi_table->item(index,0)->text().toStdString();
        }
        // if no seed assigned, assign whole brain
        vbc->roi_mgr = std::make_shared<RoiMgr>();
        if(roi_list.empty() || std::find(roi_type.begin(),roi_type.end(),3) == roi_type.end())
        {
            std::vector<tipl::vector<3,short> > seed;
            for(tipl::pixel_index<3> index(vbc->handle->dim);index < vbc->handle->dim.size();++index)
                if(vbc->handle->dir.fa[0][index.index()] > vbc->fiber_threshold)
                    seed.push_back(tipl::vector<3,short>(index.x(),index.y(),index.z()));

            vbc->roi_mgr->setRegions(vbc->handle->dim,seed,1.0f,3/*seed*/,"whole brain",tipl::vector<3>());
        }


        for(unsigned int index = 0;index < roi_list.size();++index)
            vbc->roi_mgr->setRegions(vbc->handle->dim,roi_list[index],1.0f,roi_type[index],
                                               "user assigned region",vbc->handle->vs);

        // setup roi related report text
        vbc->roi_mgr_text.clear();
        if(!roi_list.empty())
        {
            std::ostringstream out;
            out << " The tracking algorithm used";
            const char roi_type_name[5][20] = {"region of interst","region of avoidance","ending region","seeding region","terminating region"};
            for(unsigned int index = 0;index < roi_list.size();++index)
            {
                if(index && roi_list.size() > 2)
                    out << ",";
                out << " ";
                if(roi_list.size() >= 2 && index+1 == roi_list.size())
                    out << "and ";
                out << roi_name[index] << " as the " << roi_type_name[roi_type[index]];
            }
            out << ".";
            vbc->roi_mgr_text = out.str();
        }

        // setup roi related output suffix
        vbc->output_roi_suffix.clear();
        if(!roi_list.empty())
        {
            const char roi_type_name2[5][5] = {"roi","roa","end","seed"};
            for(unsigned int index = 0;index < roi_list.size();++index)
            {
                vbc->output_roi_suffix += ".";
                vbc->output_roi_suffix += roi_type_name2[roi_type[index]];
                vbc->output_roi_suffix += ".";
                vbc->output_roi_suffix += roi_name[index];
            }
            std::replace(vbc->output_roi_suffix.begin(),vbc->output_roi_suffix.end(),':','_');
        }
    }

    vbc->run_permutation(ui->multithread->value(),ui->permutation_count->value());
    if(gui)
    {
        timer.reset(new QTimer(this));
        timer->setInterval(1000);
        connect(timer.get(), SIGNAL(timeout()), this, SLOT(calculate_FDR()));
        timer->start();
    }
}

void group_connectometry::on_show_result_clicked()
{
    std::shared_ptr<fib_data> new_data(new fib_data);
    *(new_data.get()) = *(vbc->handle);
    stat_model cur_model;
    if(!setup_model(cur_model))
    {
        QMessageBox::information(this,"Error","Cannot run the statistics. Subject number not enough?",0);
        return;
    }

    {
        result_fib.reset(new connectometry_result);
        stat_model info;
        info.resample(cur_model,false,false);
        vbc->calculate_spm(*result_fib.get(),info,vbc->normalize_qa);
        new_data->view_item.push_back(item());
        new_data->view_item.back().name = "dec_t";
        new_data->view_item.back().image_data = tipl::make_image(result_fib->neg_corr_ptr[0],new_data->dim);
        new_data->view_item.back().set_scale(result_fib->neg_corr_ptr[0],
                                             result_fib->neg_corr_ptr[0]+new_data->dim.size());
        new_data->view_item.push_back(item());
        new_data->view_item.back().name = "inc_t";
        new_data->view_item.back().image_data = tipl::make_image(result_fib->pos_corr_ptr[0],new_data->dim);
        new_data->view_item.back().set_scale(result_fib->pos_corr_ptr[0],
                                             result_fib->pos_corr_ptr[0]+new_data->dim.size());

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
    current_tracking_window->command("set_param","surface_alpha","0.1");
    current_tracking_window->command("set_roi_view_index","icbm_wm");
    current_tracking_window->command("add_surface");
    current_tracking_window->command("update_track");


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
bool load_region(std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text);
void group_connectometry::on_load_roi_from_atlas_clicked()
{
    if(!vbc->handle->load_atlas())
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
    tipl::matrix<4,4,float> transform;
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
        if(!I.geometry().is_valid(pos) || I.at(pos[0],pos[1],pos[2]) == 0)
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

void group_connectometry::on_rb_longitudina_dif_clicked()
{
    ui->gb_regression->setEnabled(false);
}

void group_connectometry::on_rb_regression_clicked()
{
    ui->gb_regression->setEnabled(true);
}
