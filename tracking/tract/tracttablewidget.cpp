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
    tract_models.push_back(new TractModel(cur_tracking_window.handle,
                                          cur_tracking_window.slice.geometry,
                                          cur_tracking_window.slice.voxel_size));

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
    float param[8];
    param[0] = cur_tracking_window.ui->step_size->value();
    param[1] = cur_tracking_window.ui->turning_angle->value() * 3.1415926 / 180.0;
    param[2] = cur_tracking_window.ui->turning_angle->value() * 3.1415926 / 180.0;
    param[3] = cur_tracking_window.ui->fa_threshold->value();
    param[4] = cur_tracking_window.ui->smoothing->value();
    param[5] = cur_tracking_window.ui->min_length->value();
    param[6] = cur_tracking_window.ui->max_length->value();
    unsigned char methods[5];
    methods[0] = cur_tracking_window.ui->tracking_method->currentIndex();
    methods[1] = cur_tracking_window.ui->initial_direction->currentIndex();
    methods[2] = cur_tracking_window.ui->interpolation->currentIndex();
    methods[3] = cur_tracking_window.ui->tracking_plan->currentIndex();
    methods[4] = cur_tracking_window.ui->seed_plan->currentIndex();
    thread_data.back() = ThreadData::new_thread(cur_tracking_window.handle,param,methods,
                             cur_tracking_window.ui->track_count->value());

    cur_tracking_window.regionWidget->setROIs(thread_data.back());
    thread_data.back()->run(cur_tracking_window.ui->thread_count->currentIndex()+1);
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
                QString::number(thread_data[index]->total_seedings));
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
            "Tract files (*.txt *.trk);;All files (*.*)");
    if(!filenames.size())
        return;
    for(unsigned int index = 0;index < filenames.size();++index)
    {
        QString filename = filenames[index];
        if(!filename.size())
        continue;
        addNewTracts(QFileInfo(filename).baseName());
        std::string sfilename = filename.toLocal8Bit().begin();
        tract_models.back()->load_from_file(&*sfilename.begin(),false);
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    emit need_update();
}

void TractTableWidget::save_all_tracts_as(void)
{
    if(tract_models.empty())
        return;

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
        tract_models.back()->set_color(ROIColor[index%16].rgb());
    cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
    emit need_update();
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
                cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + ".txt",
                "Tract files (*.txt);; Travis file (*.trk);;All files (*.*)");
    if(filename.isEmpty())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_to_file(&*sfilename.begin());
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
                "Tract files (*.txt);; Travis file (*.trk);;All files (*.*)");
    if(filename.isEmpty())
        return;
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
                "Tract files (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
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
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_color_to_file(&*sfilename.begin());
}

void TractTableWidget::showCurTractStatistics(float threshold,float cull_angle_cos)
{
    if(currentRow() >= tract_models.size())
        return;
    std::string result;
    cur_tracking_window.handle->
        get_quantitative_info(tract_models[currentRow()]->get_tracts(),
                              threshold,cull_angle_cos,result);
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
    float threshold = cur_tracking_window.ui->fa_threshold->value();
    float cull_angle_cos = std::cos(cur_tracking_window.ui->turning_angle->value() * 3.1415926 / 180.0);
    if(!tract_models[currentRow()]->save_fa_to_file(
            filename.toLocal8Bit().begin(),threshold,cull_angle_cos))
    {
        QMessageBox::information(this,"error","fail to save information",0);
    }
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
                cur_tracking_window.absolute_path+"/" + item(currentRow(),0)->text() + ".bmp",
                "PNG files (*.png );;BMP files (*.bmp);;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
        if(filename.isEmpty())
            return;
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
                    cur_tracking_window.absolute_path+"/" + item(currentRow(),0)->text() + ".nii",
                    "NIFTI files (*.nii);;MAT File (*.mat);;");
        if(filename.isEmpty())
            return;

        image::basic_image<unsigned int,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        if(QFileInfo(filename).completeSuffix().toLower() == "nii")
        {
            image::io::nifti nii_header;
            image::flip_xy(tdi);
            nii_header << tdi;
            nii_header.set_voxel_size(vs.begin());
            nii_header.save_to_file(filename.toLocal8Bit().begin());
        }
        else
            if(QFileInfo(filename).completeSuffix().toLower() == "mat")
            {
                image::io::mat mat_header;
                mat_header << tdi;
                mat_header.save_to_file(filename.toLocal8Bit().begin());
            }
    }
}

/*

void __fastcall TTractFrameImp::ActionButtonClick(TObject *Sender) {
        if (!active_tract_model)
                return;
        switch (ActionList->ItemIndex) {
                // Duplicate
        case 0: {
                        void* last_active_tract = active_tract_model;
                        NewTractExecute(0);
                        tract_model_copy(last_active_tract, active_tract_model);
                        TractList->Cells[3][active_tract_index] =
                                tract_model_get_visible_track_count(active_tract_model);
                        render_tract();
                } break;
                // mean length
        case 1:
                ShowMessage(AnsiString(tract_model_get_quantitative_info
                        (active_tract_model, 0)) + " (mm)");
                break;
                // volume
        case 2:
                ShowMessage(AnsiString(tract_model_get_quantitative_info
                        (active_tract_model, 1)) + " (mm3)");
                break;
                break;
                // add cortex
        case 3:
                {
                float track_threshold = 0.1;
                UnicodeString TrackThreshold("0.1");
                GET_INI(TrackThreshold);
                if (!InputQuery("DSI STUDIO", "Enter Track Threshold",
                        TrackThreshold) || !TryStrToFloat(TrackThreshold,
                        track_threshold) || !ColorDialog->Execute())
                        return;
                PUT_INI(TrackThreshold);

                std::auto_ptr<RegionModel>region(new RegionModel);
                if (!region->load(tract_model_get_density_map(active_tract_model),
                        TrackForm->slice.geometry,
                        track_threshold * ((double)tract_model_get_visible_track_count
                        (active_tract_model)))) {
                        ShowMessage("No Vsible Mesh");
                        return;
                }
                region->color = ColorDialog->Color;
                TrackForm->ObjectFrameImp->AddRegion(none_roi_id,
                                                TractList->Cells[2][active_tract_index]);
                TrackForm->ObjectFrameImp->regions[0].show_region = *(region.get());
                TrackForm->ObjectFrameImp->ShowRegionsExecute(0);
                break;
                }
        // endpoint


         */

        /*
         if (!SaveX3DDialog->Execute())
         return;
         std::ofstream out(SaveX3DDialog->FileName.c_str());
         out << "<X3D profile='Immersive'><Scene><Transform scale='0.2 0.2 0.2'>\
         <Transform translation='" <<
         -TrackForm->slice.center_point[0] << " " <<
         -TrackForm->slice.center_point[1] << " " <<
         -TrackForm->slice.center_point[2] << " " <<
         "'>" << std::endl;

         if (TractList->Checked[0]) // Tracks
         {
         begin_prog("exporting tracks");

         for (unsigned int i = 0;check_prog(i,tract_model_get_visible_track_count(tracts_handle));++i)
         {
         if (tract_model_get_tract_length(tracts_handle,i) < 6)
         continue;
         AnsiString ConnectionText,ColorText,PointText;
         image::vector<3,float> from(tract_model_get_tract(tracts_handle,i)),to,dis;
         for (unsigned int j = 0,index = 0;j < tract_model_get_tract_length(tracts_handle,i);j+=3,++index)
         {
         to = tract_model_get_tract(tracts_handle,i) + j;
         if (index)
         {
         dis = to;
         dis -= from;
         dis.normalize();
         ColorText += AnsiString().sprintf("%f %f %f ",
         std::fabs(dis[0]),std::fabs(dis[1]),std::fabs(dis[2]));
         ConnectionText += AnsiString().sprintf("%d %d -1 ",index-1,index);
         }
         PointText += AnsiString().sprintf("%f %f %f ",to[0],to[1],to[2]);
         from = to;
         }
         out << "<Shape>\
         <IndexedLineSet colorPerVertex='false' \
         coordIndex='"<< ConnectionText.c_str() << "'>\
         <Color color='"<< ColorText.c_str() <<"'/>\
         <Coordinate point='" << PointText.c_str()<<"'/>\
         </IndexedLineSet></Shape>" << std::endl;
         }

         }
         for (unsigned int index = 1;index < ModelList->Items->Count;++index)
         {
         if (!ModelList->Checked[index])
         continue;
         if (index < slices.size() && slices[index]) // Tracks
         {
         image::color_image image;
         slices[index]->getSlice(slice_view[index],slice_pos[index],image);
         win::SaveToJpeg(image,(ExtractFilePath(SaveX3DDialog->FileName)+
         AnsiString("track") + index + ".jpg").c_str());
         AnsiString PictureCoor;

         {
         int x,y,z;
         slices[index]->get3dPosition(slice_view[index],0,0,slice_pos[index],x,y,z);
         PictureCoor += AnsiString().sprintf("%.1f %.1f %.1f,",(float)x-0.5,(float)y-0.5,(float)z-0.5);
         slices[index]->get3dPosition(slice_view[index],image.width(),0,slice_pos[index],x,y,z);
         PictureCoor += AnsiString().sprintf("%.1f %.1f %.1f,",(float)x-0.5,(float)y-0.5,(float)z-0.5);
         slices[index]->get3dPosition(slice_view[index],image.width(),image.height(),slice_pos[index],x,y,z);
         PictureCoor += AnsiString().sprintf("%.1f %.1f %.1f,",(float)x-0.5,(float)y-0.5,(float)z-0.5);
         slices[index]->get3dPosition(slice_view[index],0,image.height(),slice_pos[index],x,y,z);
         PictureCoor += AnsiString().sprintf("%.1f %.1f %.1f",(float)x-0.5,(float)y-0.5,(float)z-0.5);
         }
         out << "<Shape><Appearance><Material emissiveColor='1 1 1' transparency='0.3'/>\
         <ImageTexture url='track" << index << ".jpg'/></Appearance><IndexedFaceSet creaseAngle='.8' coordIndex='0 1 2 3'>\
         <Coordinate point='" << PictureCoor.c_str() << "'/><TextureCoordinate point='0 1,1 1,1 0,0 0'/>\
         </IndexedFaceSet></Shape><Shape><Appearance><Material emissiveColor='1 1 1' transparency='0.3'/>\
         <ImageTexture url='track" << index << ".jpg'/></Appearance><IndexedFaceSet creaseAngle='.8' coordIndex='0 3 2 1'>\
         <Coordinate point='" << PictureCoor.c_str() << "'/><TextureCoordinate point='0 1,1 1,1 0,0 0'/>\
         </IndexedFaceSet></Shape>" << std::endl;
         }
         if (index < regions.size() && regions[index])
         {
         begin_prog("exporting surfaces");
         out << "<Shape><Appearance><Material diffuseColor='" <<
         ((float)(regions[index]->get_color().b))/255.0 << " " <<
         ((float)(regions[index]->get_color().g))/255.0 << " " <<
         ((float)(regions[index]->get_color().r))/255.0 <<
         "'/></Appearance><IndexedFaceSet creaseAngle='3.1415926' coordIndex='";
         for (unsigned int i = 0;check_prog(i,regions[index]->get()->tri_list.size());++i)
         {
         out << regions[index]->get()->tri_list[i][0] << " "
         << regions[index]->get()->tri_list[i][1] << " "
         << regions[index]->get()->tri_list[i][2] << " -1 ";
         }
         out << "'><Coordinate point='";
         begin_prog("exporting surfaces coordinates");
         for (unsigned int i = 0;check_prog(i,regions[index]->get()->point_list.size());++i)
         {
         out << regions[index]->get()->point_list[i][0] << " "
         << regions[index]->get()->point_list[i][1] << " "
         << regions[index]->get()->point_list[i][2] << " ";
         }
         out << "'/></IndexedFaceSet></Shape>";

         }
         }

         out << "</Transform></Transform></Scene></X3D>";
         */

        /*
         Debuger call("TX3DLineForm::CortexParcellation1Click");
         if (!max_cluster)
         {
         ShowMessage("Cluster the fiber tracts first.");
         return;
         }
         begin_prog("constructing");
         for (unsigned int cluster_index = 1;check_prog(cluster_index,max_cluster >> 1);++cluster_index)
         {
         std::vector<image::vector<3,short> > end_points;
         end_points.reserve(tract_model_get_visible_track_count(TractFrameImp->tracts_handle) << 1);
         for (unsigned int index = 0;index < tract_model_get_visible_track_count(TractFrameImp->tracts_handle);index += 2)
         {
         if (cluster_index != tract_model_get_tract_cluster(TractFrameImp->tracts_handle,index))
         continue;
         const float* tract = tract_model_get_tract(TractFrameImp->tracts_handle,index);
         unsigned int length = tract_model_get_tract_length(TractFrameImp->tracts_handle,index);
         end_points.push_back(image::vector<3,short>(
         std::floor(tract[0]+0.5),
         std::floor(tract[1]+0.5),
         std::floor(tract[2]+0.5)));
         end_points.push_back(image::vector<3,short>(
         std::floor(tract[length-3]+0.5),
         std::floor(tract[length-2]+0.5),
         std::floor(tract[length-1]+0.5)));
         }

         std::auto_ptr<RegionModel> region(new RegionModel);
         if (!region->load(end_points))
         break;
         unsigned int color = tract_model_get_cluster_color(TractFrameImp->tracts_handle,cluster_index);
         ObjectFrameImp->addRegion(*(region.get()),AnsiString("Cluster")+cluster_index,color,1.0,true);
         }
         call.Finish();
}
*/
