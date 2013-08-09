#define NOMINMAX
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
#include <QSettings>
#include <QContextMenuEvent>
#include <QColorDialog>
#include <QInputDialog>
#include "tracttablewidget.h"
#include "tracking/tracking_window.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "opengl/glwidget.h"
#include "../region/regiontablewidget.h"
#include "ui_tracking_window.h"
#include "tracking_static_link.h"
#include "opengl/renderingtablewidget.h"
#include "libs/gzip_interface.hpp"


TractTableWidget::TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent) :
    QTableWidget(parent),cur_tracking_window(cur_tracking_window_),
    tract_serial(0)
{
    setColumnCount(4);
    setColumnWidth(0,75);
    setColumnWidth(1,50);
    setColumnWidth(2,50);
    setColumnWidth(3,50);


    QStringList header;
    header << "Name" << "Tracts" << "Deleted" << "Seeds";
    setHorizontalHeaderLabels(header);
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);

    QObject::connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_check_status(int,int)));

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(fetch_tracts()));
}

TractTableWidget::~TractTableWidget(void)
{
}

void TractTableWidget::contextMenuEvent ( QContextMenuEvent * event )
{
    if(event->reason() == QContextMenuEvent::Mouse)
        cur_tracking_window.ui->menuTracts->popup(event->globalPos());
}

void TractTableWidget::check_check_status(int row, int col)
{
    if(col != 0)
        return;
    if(item(row,0)->checkState() == Qt::Checked)
    {
        if(item(row,0)->data(Qt::ForegroundRole) == QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
            emit need_update();
        }
    }
    else
    {
        if(item(row,0)->data(Qt::ForegroundRole) != QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
            emit need_update();
        }
    }
}

void TractTableWidget::addNewTracts(QString tract_name)
{
    thread_data.push_back(0);
    tract_models.push_back(new TractModel(cur_tracking_window.handle));
    tract_models.back()->get_fib().threshold = cur_tracking_window.ui->fa_threshold->value();
    tract_models.back()->get_fib().cull_cos_angle = std::cos(cur_tracking_window.ui->turning_angle->value() * 3.1415926 / 180.0);

    setRowCount(tract_models.size());
    QTableWidgetItem *item0 = new QTableWidgetItem(tract_name);
    item0->setCheckState(Qt::Checked);
    setItem(tract_models.size()-1, 0, item0);
    for(unsigned int index = 1;index <= 3;++index)
    {
        QTableWidgetItem *item1 = new QTableWidgetItem(QString::number(0));
        item1->setFlags(item1->flags() & ~Qt::ItemIsEditable);
        setItem(tract_models.size()-1, index, item1);
    }

    setRowHeight(tract_models.size()-1,22);

    setCurrentCell(tract_models.size()-1,0);
}
void TractTableWidget::start_tracking(void)
{

    ++tract_serial;
    addNewTracts(QString("Tracts ")+QString::number(tract_serial));
    thread_data.back() = new ThreadData;
    cur_tracking_window.set_tracking_param(*thread_data.back());
    cur_tracking_window.regionWidget->setROIs(thread_data.back());
    thread_data.back()->run(tract_models.back()->get_fib(),
                            cur_tracking_window.ui->thread_count->currentIndex()+1,
                            cur_tracking_window.ui->track_count->value());
    timer->start(1000);

    // record
    {
        QSettings settings;
        settings.setValue("step_size",cur_tracking_window.ui->step_size->value());
        settings.setValue("turning_angle",cur_tracking_window.ui->turning_angle->value());
        settings.setValue("smoothing",cur_tracking_window.ui->smoothing->value());
        settings.setValue("min_length",cur_tracking_window.ui->min_length->value());
        settings.setValue("max_length",cur_tracking_window.ui->max_length->value());
        settings.setValue("tracking_method",cur_tracking_window.ui->tracking_method->currentIndex());
        settings.setValue("seed_plan",cur_tracking_window.ui->seed_plan->currentIndex());
        settings.setValue("initial_direction",cur_tracking_window.ui->initial_direction->currentIndex());
        settings.setValue("interpolation",cur_tracking_window.ui->interpolation->currentIndex());
        settings.setValue("tracking_plan",cur_tracking_window.ui->tracking_plan->currentIndex());
        settings.setValue("track_count",cur_tracking_window.ui->track_count->value());
        settings.setValue("thread_count",cur_tracking_window.ui->thread_count->currentIndex());
    }
}
void TractTableWidget::stop_tracking(void)
{
    timer->stop();
    for(unsigned int index = 0;index < thread_data.size();++index)
    {
        delete thread_data[index];
        thread_data[index] = 0;
    }
}

void TractTableWidget::fetch_tracts(void)
{
    bool has_tracts = false;
    bool has_thread = false;
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index])
        {
            has_thread = true;
            // 2 for seed number
            if(thread_data[index]->fetchTracks(tract_models[index]))
            {
                // 1 for tract number
                item(index,1)->setText(
                        QString::number(tract_models[index]->get_visible_track_count()));
                has_tracts = true;
            }
            item(index,3)->setText(
                QString::number(thread_data[index]->get_total_seed_count()));
            if(thread_data[index]->is_ended())
            {
                delete thread_data[index];
                thread_data[index] = 0;
            }
        }
    if(has_tracts)
        emit need_update();
    if(!has_thread)
        timer->stop();
}

void TractTableWidget::load_tracts(void)
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,
            "Load tracts as",
            cur_tracking_window.absolute_path,
            "Tract files (*.txt *.trk *.mat);;All files (*.*)");
    if(!filenames.size())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filenames[0]).absolutePath();
    for(unsigned int index = 0;index < filenames.size();++index)
    {
        QString filename = filenames[index];
        if(!filename.size())
            continue;
        std::string sfilename = filename.toLocal8Bit().begin();
        addNewTracts(QFileInfo(filename).baseName());
        tract_models.back()->load_from_file(&*sfilename.begin(),false);
        if(tract_models.back()->get_cluster_info().empty()) // not multiple cluster file
        {
            item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        }
        else
        {
            std::vector<unsigned int> labels;
            labels.swap(tract_models.back()->get_cluster_info());
            load_cluster_label(labels,QFileInfo(filename).baseName());
        }
    }
    emit need_update();
}

void TractTableWidget::save_all_tracts_as(void)
{
    if(tract_models.empty())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text().replace(':','_') + ".txt",
                "Tract files (*.txt *.trk *.mat);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    TractModel::save_all(&*sfilename.begin(),tract_models);
}


void TractTableWidget::set_color(void)
{
    if(tract_models.empty())
        return;
    QColor color = QColorDialog::getColor(Qt::red);
    if(!color.isValid())
        return;
    tract_models[currentRow()]->set_color(color.rgb());
    cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
    emit need_update();
}
extern QColor ROIColor[15];
void TractTableWidget::assign_colors(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
        tract_models[index]->set_color(ROIColor[index%16].rgb());
    cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
    emit need_update();
}
void TractTableWidget::load_cluster_label(const std::vector<unsigned int>& labels,QString Name)
{
    std::vector<std::vector<float> > tracts;
    tract_models[currentRow()]->release_tracts(tracts);
    delete_row(currentRow());
    unsigned int cluster_num = *std::max_element(labels.begin(),labels.end());
    for(unsigned int cluster_index = 0;cluster_index <= cluster_num;++cluster_index)
    {
        unsigned int fiber_num = std::count(labels.begin(),labels.end(),cluster_index);
        if(!fiber_num)
            continue;
        std::vector<std::vector<float> > add_tracts(fiber_num);
        for(unsigned int index = 0,i = 0;index < labels.size();++index)
            if(labels[index] == cluster_index)
            {
                add_tracts[i].swap(tracts[index]);
                ++i;
            }
        addNewTracts(Name+QString::number(cluster_index));
        tract_models.back()->add_tracts(add_tracts);
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
}

void TractTableWidget::open_cluster_label(void)
{
    if(tract_models.empty())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Load cluster label",
            cur_tracking_window.absolute_path,
            "Cluster label files (*.txt);;All files (*.*)");
    if(!filename.size())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::ifstream in(filename.toLocal8Bit().begin());
    std::vector<unsigned int> labels(tract_models[currentRow()]->get_visible_track_count());
    std::copy(std::istream_iterator<unsigned int>(in),
              std::istream_iterator<unsigned int>(),labels.begin());

    load_cluster_label(labels,"cluster");
    assign_colors();
}

void TractTableWidget::clustering(int method_id)
{
    if(tract_models.empty())
        return;
    float param[4] = {0};
    if(method_id)// k-means or EM
    {
        param[0] = QInputDialog::getInteger(this,
            "DSI Studio","Number of clusters:",5,2,100,1);
    }
    else
    {
        std::copy(cur_tracking_window.slice.geometry.begin(),
                  cur_tracking_window.slice.geometry.end(),param);
        param[3] = QInputDialog::getDouble(this,
            "DSI Studio","Clustering detail (mm):",1.0,0.2,50.0,2);
    }
    void* handle = tract_cluster_create(method_id,param);
    for(int index = 0;index < tract_models[currentRow()]->get_visible_track_count();++index)
    {
        if(tract_models[currentRow()]->get_tract_length(index))
            tract_cluster_add_tract(handle,
                &*(tract_models[currentRow()]->get_tract(index).begin()),
                tract_models[currentRow()]->get_tract_length(index));
    }
    tract_cluster_add_tract(handle,0,0);//run clustering

    unsigned int cluster_count = std::min(20,
        (int)tract_cluster_get_cluster_count(handle));
    std::vector<std::vector<float> > tracts;
    tract_models[currentRow()]->release_tracts(tracts);
    delete_row(currentRow());
    for(int index = 0;index < cluster_count;++index)
    {
        addNewTracts(QString("Cluster")+QString::number(index));
        unsigned int cluster_size;
        const unsigned int* data = tract_cluster_get_cluster(handle,index,cluster_size);
        std::vector<std::vector<float> > add_tracts(cluster_size);
        for(int i = 0;i < cluster_size;++i)
            add_tracts[i].swap(tracts[data[i]]);
        tract_models.back()->add_tracts(add_tracts);
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    tract_cluster_free(handle);
    assign_colors();
}

void TractTableWidget::save_tracts_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text().replace(':','_') + ".txt",
                 "Tract files (*.txt *.trk *.mat);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_to_file(&*sfilename.begin());
}

void TractTableWidget::save_end_point_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text().replace(':','_') + "endpoint.txt",
                "Tract files (*.txt *.mat);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_end_points(&*sfilename.begin());
}

void TractTableWidget::saveTransformedTracts(const float* transform)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + ".txt",
                 "Tract files (*.txt *.trk *.mat);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_transformed_tracts_to_file(&*sfilename.begin(),transform,false);
}



void TractTableWidget::saveTransformedEndpoints(const float* transform)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end_point as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + ".txt",
                "Tract files (*.txt *.mat);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_transformed_tracts_to_file(&*sfilename.begin(),transform,true);
}

void TractTableWidget::load_tracts_color(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Load tracts color",
            cur_tracking_window.absolute_path,
            "Color files (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->load_tracts_color_from_file(&*sfilename.begin());
    emit need_update();
}

void TractTableWidget::save_tracts_color_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts color as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + "_color.txt",
                "Color files (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_color_to_file(&*sfilename.begin());
}

void TractTableWidget::show_tracts_statistics(void)
{
    if(currentRow() >= tract_models.size())
        return;
    std::string result;
    {
        std::vector<std::string> titles;
        titles.push_back("number of tracts");
        titles.push_back("tract length mean(mm)");
        titles.push_back("tract length sd(mm)");
        titles.push_back("tracts volume (mm^3)");
        cur_tracking_window.handle->get_index_titles(titles);
        std::vector<std::vector<float> > data(tract_models.size());
        begin_prog("calculating");
        for(unsigned int index = 0;check_prog(index,tract_models.size());++index)
            tract_models[index]->get_quantitative_data(data[index]);
        if(prog_aborted())
            return;
        std::ostringstream out;
        out << "Tract Name\t";
        for(unsigned int index = 0;index < tract_models.size();++index)
            out << item(index,0)->text().toLocal8Bit().begin() << "\t";
        out << std::endl;
        for(unsigned int i = 0;i < titles.size();++i)
        {
            out << titles[i] << "\t";
            for(unsigned int j = 0;j < tract_models.size();++j)
            {
                if(i < data[j].size())
                    out << data[j][i];
                out << "\t";
            }
            out << std::endl;
        }
        result = out.str();
    }
    QMessageBox msgBox;
    msgBox.setText("Tract Statistics");
    msgBox.setInformativeText(result.c_str());
    msgBox.setStandardButtons(QMessageBox::Ok|QMessageBox::Save);
    msgBox.setDefaultButton(QMessageBox::Ok);
    QPushButton *copyButton = msgBox.addButton("Copy To Clipboard", QMessageBox::ActionRole);


    if(msgBox.exec() == QMessageBox::Save)
    {
        QString filename;
        filename = QFileDialog::getSaveFileName(
                    this,
                    "Save satistics as",
                    cur_tracking_window.absolute_path + +"/" + item(currentRow(),0)->text() + "_stat.txt",
                    "Text files (*.txt);;All files|(*.*)");
        if(filename.isEmpty())
            return;
        cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
        std::ofstream out(filename.toLocal8Bit().begin());
        out << result.c_str();
    }
    if (msgBox.clickedButton() == copyButton)
        QApplication::clipboard()->setText(result.c_str());
}

void TractTableWidget::save_fa_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save QA as",
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + "_qa.txt",
                "Text files (*.txt);;All files|(*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    if(!tract_models[currentRow()]->save_fa_to_file(filename.toLocal8Bit().begin()))
        QMessageBox::information(this,"error","fail to save information",0);
}

void TractTableWidget::save_tracts_data_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save as",
                cur_tracking_window.absolute_path +"/" +
                item(currentRow(),0)->text() + "_" + action->data().toString() + ".txt",
                "Text files (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
    if(!tract_models[currentRow()]->save_data_to_file(
                    filename.toLocal8Bit().begin(),
                    action->data().toString().toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error","fail to save information",0);
    }
}

void TractTableWidget::merge_all(void)
{
    std::vector<unsigned int> merge_list;
    for(int index = 0;index < tract_models.size();++index)
        if(item(index,0)->checkState() == Qt::Checked)
            merge_list.push_back(index);
    if(merge_list.size() <= 1)
        return;

    for(int index = merge_list.size()-1;index >= 1;--index)
    {
        tract_models[merge_list[0]]->add(*tract_models[merge_list[index]]);
        delete_row(merge_list[index]);
    }
    item(merge_list[0],1)->setText(QString::number(tract_models[merge_list[0]]->get_visible_track_count()));
    item(merge_list[0],2)->setText(QString::number(tract_models[merge_list[0]]->get_deleted_track_count()));
}

void TractTableWidget::delete_row(int row)
{
    if(row >= tract_models.size())
        return;
    delete thread_data[row];
    delete tract_models[row];
    thread_data.erase(thread_data.begin()+row);
    tract_models.erase(tract_models.begin()+row);
    removeRow(row);
}

void TractTableWidget::delete_tract(void)
{
    delete_row(currentRow());
    emit need_update();
}

void TractTableWidget::delete_all_tract(void)
{
    setRowCount(0);
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        delete thread_data[index];
        delete tract_models[index];
    }
    thread_data.clear();
    tract_models.clear();
    emit need_update();
}



void TractTableWidget::edit_tracts(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        switch(edit_option)
        {
        case 1:
        case 2:
            tract_models[index]->cull(
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
                             cur_tracking_window.glWidget->pos,
                             cur_tracking_window.glWidget->pos,edit_option == 2);
            break;
        case 3:
            tract_models[index]->cut(
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
                             cur_tracking_window.glWidget->pos,
                             cur_tracking_window.glWidget->pos);
            break;
        case 4:
            tract_models[index]->paint(
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
                             cur_tracking_window.glWidget->pos,
                             cur_tracking_window.glWidget->pos,QColorDialog::getColor(Qt::red).rgb());
            cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
            break;
        }
        item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
        item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
    }
    emit need_update();
}

void TractTableWidget::undo_tracts(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        tract_models[index]->undo();
        item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
        item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
    }
    emit need_update();
}

void TractTableWidget::redo_tracts(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        tract_models[index]->redo();
        item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
        item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
    }
    emit need_update();
}

void TractTableWidget::trim_tracts(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        tract_models[index]->trim();
        item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
        item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
    }
    emit need_update();
}

void TractTableWidget::export_tract_density(image::geometry<3>& dim,
                          image::vector<3,float> vs,
                          std::vector<float>& transformation,bool color,bool end_point)
{
    if(color)
    {
        QString filename = QFileDialog::getSaveFileName(
                this,
                "Save Images files",
                cur_tracking_window.absolute_path+"/" + item(currentRow(),0)->text(),
                "BMP files (*.bmp);;PNG files (*.png );;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
        if(filename.isEmpty())
            return;
        cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
        image::basic_image<image::rgb_color,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        QImage qimage((unsigned char*)&*tdi.begin(),
                      tdi.width(),tdi.height()*tdi.depth(),QImage::Format_RGB32);
        qimage.save(filename);
    }
    else
    {
        QString filename = QFileDialog::getSaveFileName(
                    this,
                    "Save as",
                    cur_tracking_window.absolute_path+"/" + item(currentRow(),0)->text(),
                    "NIFTI files (*.nii.gz *.nii);;MAT File (*.mat);;");
        if(filename.isEmpty())
            return;
        cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();
        image::basic_image<unsigned int,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        if(QFileInfo(filename).completeSuffix().toLower() == "nii" ||
                QFileInfo(filename).completeSuffix().toLower() == "nii.gz")
        {
            gz_nifti nii_header;
            image::flip_xy(tdi);
            nii_header << tdi;
            nii_header.set_voxel_size(vs.begin());
            nii_header.save_to_file(filename.toLocal8Bit().begin());
        }
        else
            if(QFileInfo(filename).completeSuffix().toLower() == "mat")
            {
                image::io::mat_write mat_header(filename.toLocal8Bit().begin());
                mat_header << tdi;
            }
    }
}

