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
#include "opengl/renderingtablewidget.h"
#include "libs/gzip_interface.hpp"
#include "atlas.hpp"
#include "../color_bar_dialog.hpp"
#include "gzip_interface.hpp"

void show_info_dialog(const std::string& title,const std::string& result);

TractTableWidget::TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent) :
    QTableWidget(parent),cur_tracking_window(cur_tracking_window_)
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
    setAlternatingRowColors(true);
    setStyleSheet("QTableView {selection-background-color: #AAAAFF; selection-color: #000000;}");

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

void TractTableWidget::check_check_status(int, int col)
{
    if(col != 0)
        return;
    emit need_update();
}

void TractTableWidget::addNewTracts(QString tract_name,bool checked)
{
    thread_data.push_back(std::make_shared<ThreadData>());
    tract_models.push_back(std::make_shared<TractModel>(cur_tracking_window.handle));
    insertRow(tract_models.size()-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(tract_name);
    item0->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
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
void TractTableWidget::addConnectometryResults(std::vector<std::vector<std::vector<float> > >& greater,
                             std::vector<std::vector<std::vector<float> > >& lesser)
{
    for(unsigned int index = 0;index < lesser.size();++index)
    {
        if(lesser[index].empty())
            continue;
        int color = std::min<int>((index+1)*50,255);
        addNewTracts(QString("Lesser_") + QString::number((index+1)*10),false);
        tract_models.back()->add_tracts(lesser[index],tipl::rgb(255,255-color,255-color));
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    for(unsigned int index = 0;index < greater.size();++index)
    {
        if(greater[index].empty())
            continue;
        int color = std::min<int>((index+1)*50,255);
        addNewTracts(QString("Greater_") + QString::number((index+1)*10),false);
        tract_models.back()->add_tracts(greater[index],tipl::rgb(255-color,255-color,255));
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
}

void TractTableWidget::start_tracking(void)
{
    if(cur_tracking_window.ui->target->currentIndex() > 0)
        addNewTracts(cur_tracking_window.ui->target->currentText());
    else
        addNewTracts(cur_tracking_window.regionWidget->getROIname());
    cur_tracking_window.set_tracking_param(*thread_data.back());
    cur_tracking_window.regionWidget->setROIs(thread_data.back().get());
    thread_data.back()->run(tract_models.back()->get_fib(),
                            cur_tracking_window["thread_count"].toInt(),
                            false);
    tract_models.back()->report += thread_data.back()->report.str();
    tract_models.back()->parameter_id = thread_data.back()->param.get_code();
    show_report();
    timer->start(1000);
}

void TractTableWidget::show_report(void)
{
    if(currentRow() >= tract_models.size())
        return;
    cur_tracking_window.report(tract_models[currentRow()]->report.c_str());
}

void TractTableWidget::ppv_analysis(void)
{
    if(cur_tracking_window.handle->dir.dt_fa.empty()) // No dt setting
    {
        QMessageBox::information(this,"Error","No DT index set in the tracking parameter",0);
        return;
    }
    std::vector<float> p1(100),p2(100);
    {
        int i = 0;
        for(double dt_thresdhold = 0.05;dt_thresdhold <= 0.5;dt_thresdhold += 0.05)
        for(int length_threshold = 5;length_threshold <= 50;length_threshold += 5,++i)
        {
            p1[i] = length_threshold;
            p2[i] = dt_thresdhold;
        }
    }
    std::vector<int> tracks_count(100);
    ThreadData base_thread;
    cur_tracking_window.set_tracking_param(base_thread);
    cur_tracking_window.regionWidget->setROIs(&base_thread);
    tracking_data fib;
    fib.read(*cur_tracking_window.handle.get());

    bool terminated = false;
    begin_prog("PPV analysis");
    tipl::par_for2(100,[&](int i,int j){
        if(terminated)
            return;
        if(j == 0)
        {
            if(prog_aborted())
            {
                check_prog(0,0);
                terminated = true;
                return;
            }
            check_prog(i,100);
        }
        ThreadData new_thread;
        new_thread.param = base_thread.param;
        new_thread.param.min_length = p1[i];
        new_thread.param.dt_threshold = p2[i];
        new_thread.roi_mgr = base_thread.roi_mgr;
        new_thread.run(fib,1,true);

        TractModel trk(cur_tracking_window.handle);
        new_thread.fetchTracks(&trk);
        for(int k = 0;k < base_thread.param.tip_iteration;++k)
            trk.trim();
        tracks_count[i] = trk.get_visible_track_count();
    });
    check_prog(0,0);
    if(prog_aborted())
        return;


    std::ostringstream out;
    out << fib.dt_threshold_name << "\t";
    for(int length_threshold = 5;length_threshold <= 50;length_threshold += 5)
        out << length_threshold << " mm\t";
    out << std::endl;

    for(int dt_thresdhold = 5,i = 0;dt_thresdhold <= 50;dt_thresdhold += 5)
    {
        out << dt_thresdhold << "%\t";
        for(int length_threshold = 5;length_threshold <= 50;length_threshold += 5,++i)
            out << tracks_count[i] << "\t";
        out << std::endl;
    }

    show_info_dialog("PPV results",out.str());
}

void TractTableWidget::filter_by_roi(void)
{
    ThreadData track_thread;
    cur_tracking_window.set_tracking_param(track_thread);
    cur_tracking_window.regionWidget->setROIs(&track_thread);
    for(int index = 0;index < tract_models.size();++index)
    if(item(index,0)->checkState() == Qt::Checked)
    {
        tract_models[index]->filter_by_roi(track_thread.roi_mgr);
        item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
        item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
    }
    emit need_update();
}

void TractTableWidget::fetch_tracts(void)
{
    bool has_tracts = false;
    bool has_thread = false;
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index].get())
        {
            has_thread = true;
            // 2 for seed number
            if(thread_data[index]->fetchTracks(tract_models[index].get()))
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
                if(thread_data[index]->param.tip_iteration)
                {
                    for(int i = 0;i < thread_data[index]->param.tip_iteration;++i)
                        tract_models[index]->trim();
                    item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
                    item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
                }
                thread_data[index].reset();
            }
        }
    if(has_tracts)
        emit need_update();
    if(!has_thread)
        timer->stop();
}

void TractTableWidget::stop_tracking(void)
{
    timer->stop();
    for(unsigned int index = 0;index < thread_data.size();++index)
        thread_data[index].reset();
}
void TractTableWidget::load_tracts(QStringList filenames)
{
    if(filenames.empty())
        return;
    for(unsigned int index = 0;index < filenames.size();++index)
    {
        QString filename = filenames[index];
        if(!filename.size())
            continue;
        QString label = QFileInfo(filename).fileName();
        label.remove(".trk");
        label.remove(".gz");
        label.remove(".txt");
        std::string sfilename = filename.toStdString();
        addNewTracts(label);
        if(!tract_models.back()->load_from_file(&*sfilename.begin(),false))
        {
            QMessageBox::information(this,"Error",
                                     QString("Fail to load tracks from %1. \
                                Please check file access privelige or move file to other location.").arg(QFileInfo(filename).baseName()),0);
            continue;
        }
        if(tract_models.back()->get_cluster_info().empty()) // not multiple cluster file
        {
            item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        }
        else
        {
            std::vector<unsigned int> labels;
            labels.swap(tract_models.back()->get_cluster_info());
            load_cluster_label(labels);
            if(QFileInfo(filename+".txt").exists())
                load_tract_label(filename+".txt");
        }
    }
    emit need_update();
}

void TractTableWidget::load_tracts(void)
{
    load_tracts(QFileDialog::getOpenFileNames(
            this,"Load tracts as",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
            "Tract files (*.trk *trk.gz *.tck);;Text files (*.txt);;All files (*)"));
    show_report();
}
void TractTableWidget::load_tract_label(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load tracts as",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
                "Tract files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    load_tract_label(filename);
}
void TractTableWidget::load_tract_label(QString filename)
{
    std::ifstream in(filename.toStdString().c_str());
    std::string line;
    for(int i = 0;in >> line && i < rowCount();++i)
        item(i,0)->setText(line.c_str());
}

void TractTableWidget::check_all(void)
{
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
    emit need_update();
}

void TractTableWidget::uncheck_all(void)
{
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
    emit need_update();
}
QString TractTableWidget::output_format(void)
{
    switch(cur_tracking_window["track_format"].toInt())
    {
    case 0:
        return ".trk.gz";
    case 1:
        return ".trk";
    case 2:
        return ".txt";
    }
    return "";
}

void TractTableWidget::save_all_tracts_to_dir(void)
{
    if (tract_models.empty())
        return;
    QString dir = QFileDialog::getExistingDirectory(this,"Open directory","");
    if(dir.isEmpty())
        return;
    begin_prog("save files...");
    for(unsigned int index = 0;check_prog(index,rowCount());++index)
        if (item(index,0)->checkState() == Qt::Checked)
        {
            std::string filename = dir.toLocal8Bit().begin();
            filename  += "/";
            filename  += item(index,0)->text().toLocal8Bit().begin();
            filename  += output_format().toStdString();
            tract_models[index]->save_tracts_to_file(filename.c_str());
        }
}
void TractTableWidget::save_all_tracts_as(void)
{
    if(tract_models.empty())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_tracks",filename);
}

void TractTableWidget::set_color(void)
{
    if(tract_models.empty())
        return;
    QColor color = QColorDialog::getColor(Qt::red,(QWidget*)this,"Select color",QColorDialog::ShowAlphaChannel);
    if(!color.isValid())
        return;
    tract_models[currentRow()]->set_color(color.rgb());
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
}
void TractTableWidget::assign_colors(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        tipl::rgb c;
        c.from_hsl(((color_gen++)*1.1-std::floor((color_gen++)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        tract_models[index]->set_color(c.color);
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
}
void TractTableWidget::load_cluster_label(const std::vector<unsigned int>& labels,QStringList Names)
{
    std::string report = tract_models[currentRow()]->report;
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
        if(cluster_index < Names.size())
            addNewTracts(Names[cluster_index],false);
        else
            addNewTracts(QString("cluster")+QString::number(cluster_index),false);
        tract_models.back()->add_tracts(add_tracts);
        tract_models.back()->report = report;
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
}

void TractTableWidget::open_cluster_label(void)
{
    if(tract_models.empty())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load cluster label",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
            "Cluster label files (*.txt);;All files (*)");
    if(!filename.size())
        return;

    std::ifstream in(filename.toLocal8Bit().begin());
    std::vector<unsigned int> labels(tract_models[currentRow()]->get_visible_track_count());
    std::copy(std::istream_iterator<unsigned int>(in),
              std::istream_iterator<unsigned int>(),labels.begin());
    load_cluster_label(labels);
    assign_colors();
}

extern std::vector<std::string> tractography_name_list;
void TractTableWidget::auto_recognition(void)
{
    if(!cur_tracking_window.tractography_atlas.get() && cur_tracking_window.ui->enable_auto_track->isVisible())
        cur_tracking_window.on_enable_auto_track_clicked();
    if(!cur_tracking_window.tractography_atlas.get())
        return;
    std::vector<unsigned int> c;
    tract_models[currentRow()]->recognize(c,cur_tracking_window.tractography_atlas);
    QStringList Names;
    for(int i = 0;i < tractography_name_list.size();++i)
        Names << tractography_name_list[i].c_str();
    Names << "false tracks";
    load_cluster_label(c,Names);
    assign_colors();
}
void TractTableWidget::recognize_rename(void)
{
    if(!cur_tracking_window.tractography_atlas.get() && cur_tracking_window.ui->enable_auto_track->isVisible())
        cur_tracking_window.on_enable_auto_track_clicked();
    if(!cur_tracking_window.tractography_atlas.get())
    {
        QMessageBox::information(this,"Error","Recognition is only available with [Step T3a][Template]=HCP1021");
        return;
    }
    begin_prog("Recognize and rename");
    for(unsigned int index = 0;check_prog(index,tract_models.size());++index)
        if(item(index,0)->checkState() == Qt::Checked)
        {
            std::map<float,std::string,std::greater<float> > sorted_list;
            if(!tract_models[index]->recognize(sorted_list,cur_tracking_window.tractography_atlas,true))
                return;
            item(index,0)->setText(sorted_list.begin()->second.c_str());
        }
}

void TractTableWidget::clustering(int method_id)
{
    if(tract_models.empty())
        return;
    bool ok = false;
    int n = QInputDialog::getInt(this,
            "DSI Studio",
            "Assign the maximum number of groups",50,1,5000,10,&ok);
    if(!ok)
        return;
    ok = true;
    double detail = method_id ? 0.0 : QInputDialog::getDouble(this,
            "DSI Studio","Clustering detail (mm):",cur_tracking_window.handle->vs[0],0.2,50.0,2,&ok);
    if(!ok)
        return;
    tract_models[currentRow()]->run_clustering(method_id,n,detail);
    std::vector<unsigned int> c = tract_models[currentRow()]->get_cluster_info();
    load_cluster_label(c);
    assign_colors();
}

void TractTableWidget::save_tracts_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;TCK file (*.tck);;ROI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_to_file(&*sfilename.begin());
}

void TractTableWidget::save_tracts_in_native(void)
{
    if(currentRow() >= tract_models.size())
        return;
    if(!cur_tracking_window.handle->is_qsdr)
    {
        QMessageBox::information(this,"This function only works with QSDR reconstructed FIB files.",0);
        return;
    }
    if(cur_tracking_window.handle->native_position.empty())
    {
        QMessageBox::information(this,"No mapping information included. Please reconstruct QSDR with mapping checked in advanced option.",0);
        return;
    }

    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_in_native_space(&*sfilename.begin(),
            cur_tracking_window.handle->native_position);
}

void TractTableWidget::save_vrml_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + ".wrl",
                 "Tract files (*.wrl);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string surface_text;
    if(cur_tracking_window.glWidget->surface.get() && cur_tracking_window["show_surface"].toInt())
    {
        std::ostringstream out;
        QString Coordinate, CoordinateIndex;
        const auto& point_list = cur_tracking_window.glWidget->surface->get()->point_list;
        for(unsigned int index = 0;index < point_list.size();++index)
            Coordinate += QString("%1 %2 %3 ").arg(point_list[index][0]).arg(point_list[index][1]).arg(point_list[index][2]);
        const auto& tri_list = cur_tracking_window.glWidget->surface->get()->tri_list;
        for (unsigned int j = 0;j < tri_list.size();++j)
            CoordinateIndex += QString("%1 %2 %3 -1 ").arg(tri_list[j][0]).arg(tri_list[j][1]).arg(tri_list[j][2]);

        out << "Shape {" << std::endl;
        out << "appearance Appearance { " << std::endl;
        out << "material Material { " << std::endl;
        out << "ambientIntensity 0.0" << std::endl;
        out << "diffuseColor 0.6 0.6 0.6" << std::endl;
        out << "specularColor 0.1 0.1 0.1" << std::endl;
        out << "emissiveColor 0.0 0.0 0.0" << std::endl;
        out << "shininess 0.1" << std::endl;
        out << "transparency 0.85" << std::endl;
        out << "} }" << std::endl;
        out << "geometry IndexedFaceSet {" << std::endl;
        out << "creaseAngle 3.14" << std::endl;
        out << "solid TRUE" << std::endl;
        out << "coord Coordinate { point [" << Coordinate.toStdString() << " ] }" << std::endl;
        out << "coordIndex ["<< CoordinateIndex.toStdString() <<"] } }" << std::endl;
        surface_text = out.str();
    }

    std::string sfilename = filename.toLocal8Bit().begin();

    tract_models[currentRow()]->save_vrml(&*sfilename.begin(),
                                                cur_tracking_window["tract_style"].toInt(),
                                                cur_tracking_window["tract_color_style"].toInt(),
                                                cur_tracking_window["tube_diameter"].toFloat(),
                                                cur_tracking_window["tract_tube_detail"].toInt(),surface_text);
}

void TractTableWidget::save_end_point_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",item(currentRow(),0)->text().replace(':','_') + "endpoint.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;

    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_end_points(&*sfilename.begin());
}

void TractTableWidget::save_end_point_in_mni(void)
{
    if(currentRow() >= tract_models.size())
        return;
    if(!cur_tracking_window.can_map_to_mni())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",item(currentRow(),0)->text().replace(':','_') + "endpoint.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    std::vector<tipl::vector<3,float> > points;
    std::vector<float> buffer;
    tract_models[currentRow()]->get_end_points(points);
    for(unsigned int index = 0;index < points.size();++index)
    {
        cur_tracking_window.handle->subject2mni(points[index]);
        buffer.push_back(points[index][0]);
        buffer.push_back(points[index][1]);
        buffer.push_back(points[index][2]);
    }

    if (QFileInfo(filename).suffix().toLower() == "txt")
    {
        std::ofstream out(filename.toLocal8Bit().begin(),std::ios::out);
        if (!out)
            return;
        std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(out," "));
    }
    if (QFileInfo(filename).suffix().toLower() == "mat")
    {
        tipl::io::mat_write out(filename.toLocal8Bit().begin());
        if(!out)
            return;
        out.write("end_points",buffer,3);
    }
}

void paint_track_on_volume(tipl::image<unsigned char,3>& track_map,const std::vector<float>& tracks)
{
    for(int j = 0;j < tracks.size();j += 3)
    {
        tipl::pixel_index<3> p(std::round(tracks[j]),std::round(tracks[j+1]),std::round(tracks[j+2]),track_map.geometry());
        if(track_map.geometry().is_valid(p))
            track_map[p.index()] = 1;
        if(j)
        {
            for(float r = 0.2f;r < 1.0f;r += 0.2f)
            {
                tipl::pixel_index<3> p2(std::round(tracks[j]*r+tracks[j-3]*(1-r)),
                                         std::round(tracks[j+1]*r+tracks[j-2]*(1-r)),
                                         std::round(tracks[j+2]*r+tracks[j-1]*(1-r)),track_map.geometry());
                if(track_map.geometry().is_valid(p2))
                    track_map[p2.index()] = 1;
            }
        }
    }
}

void TractTableWidget::deep_learning_train(void)
{
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save network as","network.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    // save atlas as a nifti file
    if(cur_tracking_window.handle->is_qsdr) //QSDR condition
    {
        tipl::image<int,4> atlas(tipl::geometry<4>(
                cur_tracking_window.handle->dim[0],
                cur_tracking_window.handle->dim[1],
                cur_tracking_window.handle->dim[2],
                rowCount()));

        for(unsigned int index = 0;check_prog(index,rowCount());++index)
        {
            tipl::image<unsigned char,3> track_map(cur_tracking_window.handle->dim);
            for(unsigned int i = 0;i < tract_models[index]->get_tracts().size();++i)
                paint_track_on_volume(track_map,tract_models[index]->get_tracts()[i]);
            while(tipl::morphology::smoothing_fill(track_map))
                ;
            tipl::morphology::defragment(track_map);

            QString track_file_name = QFileInfo(filename).absolutePath() + "/" + item(index,0)->text() + ".nii.gz";
            gz_nifti nifti2;
            nifti2.set_voxel_size(cur_tracking_window.current_slice->voxel_size);
            nifti2.set_image_transformation(cur_tracking_window.handle->trans_to_mni);
            nifti2 << track_map;
            nifti2.save_to_file(track_file_name.toLocal8Bit().begin());

            std::copy(track_map.begin(),track_map.end(),atlas.begin() + index*track_map.size());

            if(index+1 == rowCount())
            {
                filename = QFileInfo(filename).absolutePath() + "/tracks.nii.gz";
                gz_nifti nifti;
                nifti.set_voxel_size(cur_tracking_window.current_slice->voxel_size);
                nifti.set_image_transformation(cur_tracking_window.handle->trans_to_mni);
                nifti << atlas;
                nifti.save_to_file(filename.toLocal8Bit().begin());
            }
        }
    }

}

void TractTableWidget::recog_tracks(void)
{
    if(currentRow() >= tract_models.size() || tract_models[currentRow()]->get_tracts().size() == 0)
        return;
    if(!cur_tracking_window.tractography_atlas.get() && cur_tracking_window.ui->enable_auto_track->isVisible())
        cur_tracking_window.on_enable_auto_track_clicked();
    if(!cur_tracking_window.tractography_atlas.get())
        return;
    std::map<float,std::string,std::greater<float> > sorted_list;
    if(!tract_models[currentRow()]->recognize(sorted_list,cur_tracking_window.tractography_atlas))
    {
        QMessageBox::information(this,"Error","Cannot recognize tracks.",0);
        return;
    }
    std::ostringstream out;
    auto beg = sorted_list.begin();
    for(int i = 0;i < sorted_list.size();++i,++beg)
        if(beg->first != 0.0f)
            out << beg->first*100.0f << "% " << beg->second << std::endl;
    show_info_dialog("Tract Recognition Result",out.str());
}
void smoothed_tracks(const std::vector<float>& track,std::vector<float>& smoothed);
void TractTableWidget::saveTransformedTracts(const float* transform)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",item(currentRow(),0)->text() + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!cur_tracking_window.can_map_to_mni())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    if(transform)
        tract_models[currentRow()]->save_transformed_tracts_to_file(&*sfilename.begin(),transform,false);
    else
    {
        std::vector<std::vector<float> > tract_data(tract_models[currentRow()]->get_tracts());
        begin_prog("converting coordinates");
        for(unsigned int i = 0;check_prog(i,tract_data.size());++i)
        {
            for(unsigned int j = 0;j < tract_data[i].size();j += 3)
            {
                tipl::vector<3> v(&(tract_data[i][j]));
                cur_tracking_window.handle->subject2mni(v);
                tract_data[i][j] = v[0];
                tract_data[i][j+1] = v[1];
                tract_data[i][j+2] = v[2];
            }
            if(!cur_tracking_window.handle->is_qsdr)
            {
                std::vector<float> smooth_track;
                smoothed_tracks(tract_data[i],smooth_track);
                tract_data[i].swap(smooth_track);
            }
        }
        if(!prog_aborted())
        {
            tract_models[currentRow()]->get_tracts().swap(tract_data);
            tract_models[currentRow()]->save_tracts_to_file(&*sfilename.begin());
            tract_models[currentRow()]->get_tracts().swap(tract_data);
        }
    }
}



void TractTableWidget::saveTransformedEndpoints(const float* transform)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end_point as",item(currentRow(),0)->text() + ".txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
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
            this,"Load tracts color",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
            "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    command("load_track_color",filename,"");
}

void TractTableWidget::load_tracts_value(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load tracts color",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
            "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ifstream in(filename.toStdString().c_str());
    if (!in)
        return;
    std::vector<float> values;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),
              std::back_inserter(values));
    if(tract_models[currentRow()]->get_visible_track_count() != values.size())
    {
        QMessageBox::information(this,"Inconsistent track number",
                                 QString("The text file has %1 values, but there are %2 tracks.").
                                 arg(values.size()).arg(tract_models[currentRow()]->get_visible_track_count()),0);
        return;
    }
    color_bar_dialog dialog(0);
    auto min_max = std::minmax_element(values.begin(),values.end());
    dialog.set_value(*min_max.first,*min_max.second);
    dialog.exec();
    std::vector<unsigned int> colors(values.size());
    for(int i = 0;i < values.size();++i)
        colors[i] = (unsigned int)dialog.get_rgb(values[i]);
    tract_models[currentRow()]->set_tract_color(colors);
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
}

void TractTableWidget::save_tracts_color_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts color as",item(currentRow(),0)->text() + "_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_color_to_file(&*sfilename.begin());
}

void get_track_statistics(const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          const std::vector<std::string>& track_name,
                          std::string& result)
{
    if(tract_models.empty())
        return;
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    tipl::par_for(tract_models.size(),[&](unsigned int index)
    {
        std::string tmp,line;
        tract_models[index]->get_quantitative_info(tmp);
        std::istringstream in(tmp);
        while(std::getline(in,line))
        {
            if(line.find("\t") == std::string::npos)
                continue;
            track_results[index].push_back(line);
        }
    });

    std::ostringstream out;
    out << "Tract Name\t";
    for(unsigned int index = 0;index < tract_models.size();++index)
        out << track_name[index] << "\t";
    out << std::endl;
    for(unsigned int index = 0;index < track_results[0].size();++index)
    {
        out << track_results[0][index];
        for(unsigned int i = 1;i < track_results.size();++i)
            out << track_results[i][index].substr(track_results[i][index].find("\t"));
        out << std::endl;
    }
    result = out.str();
}
std::vector<std::shared_ptr<TractModel> > TractTableWidget::get_checked_tracks(void) const
{
    std::vector<std::shared_ptr<TractModel> > active_tracks;
    for(unsigned int index = 0;index < tract_models.size();++index)
        if(item(index,0)->checkState() == Qt::Checked)
            active_tracks.push_back(tract_models[index]);
    return active_tracks;
}
std::vector<std::string> TractTableWidget::get_checked_tracks_name(void) const
{
    std::vector<std::string> track_name;
    for(unsigned int index = 0;index < tract_models.size();++index)
        if(item(index,0)->checkState() == Qt::Checked)
            track_name.push_back(item(index,0)->text().toStdString());
    return track_name;
}
void TractTableWidget::show_tracts_statistics(void)
{
    if(tract_models.empty())
        return;
    std::string result;
    get_track_statistics(get_checked_tracks(),get_checked_tracks_name(),result);
    if(!result.empty())
        show_info_dialog("Tract Statistics",result);

}

bool TractTableWidget::command(QString cmd,QString param,QString param2)
{
    if(cmd == "update_track")
    {
        for(int index = 0;index < tract_models.size();++index)
        {
            item(index,1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
            item(index,2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
        }
        emit need_update();
        return true;
    }
    if(cmd == "run_tracking")
    {
        start_tracking();
        while(timer->isActive())
            fetch_tracts();
        emit need_update();
        return true;
    }
    if(cmd == "cut_by_slice")
    {
        cut_by_slice(param.toInt(),param2.toInt());
        return true;
    }
    if(cmd == "delete_all_tract")
    {
        setRowCount(0);
        thread_data.clear();
        tract_models.clear();
        emit need_update();
        return true;
    }
    if(cmd == "save_tracks")
    {
        return TractModel::save_all(param.toStdString().c_str(),
                             get_checked_tracks(),get_checked_tracks_name());
    }
    if(cmd == "load_track_color")
    {
        int index = currentRow();
        if(!param2.isEmpty())
        {
            index = param2.toInt();
            if(index < 0 || index >= tract_models.size())
            {
                std::cout << "Invalid track index:" << param2.toStdString() << std::endl;
                return false;
            }
        }
        std::string sfilename = param.toStdString().c_str();
        if(!tract_models[index]->load_tracts_color_from_file(&*sfilename.begin()))
        {
            std::cout << "Cannot find or open " << sfilename << std::endl;
            return false;
        }
        cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        emit need_update();
        return true;
    }
    return false;
}

void TractTableWidget::save_tracts_data_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    QString filename = QFileDialog::getSaveFileName(
                this,"Save as",item(currentRow(),0)->text() + "_" + action->data().toString() + ".txt",
                "Text files (*.txt);;MATLAB file (*.mat);;TRK file (*.trk *.trk.gz);;All files (*)");
    if(filename.isEmpty())
        return;

    if(!tract_models[currentRow()]->save_data_to_file(
                    filename.toLocal8Bit().begin(),
                    action->data().toString().toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error","fail to save information",0);
    }
    else
        QMessageBox::information(this,"DSI Studio","file saved",0);
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
    thread_data.erase(thread_data.begin()+row);
    tract_models.erase(tract_models.begin()+row);
    removeRow(row);
}

void TractTableWidget::copy_track(void)
{
    unsigned int cur_row = currentRow();
    addNewTracts(item(cur_row,0)->text() + "copy");
    *(tract_models.back()) = *(tract_models[cur_row]);
    item(currentRow(),1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    emit need_update();
}

void TractTableWidget::separate_deleted_track(void)
{
    unsigned int cur_row = currentRow();
    addNewTracts(item(cur_row,0)->text(),false);
    std::vector<std::vector<float> > new_tracks = tract_models[cur_row]->get_deleted_tracts();
    if(new_tracks.empty())
        return;
    tract_models.back()->add_tracts(new_tracks);
    tract_models[cur_row]->clear_deleted();
    item(rowCount()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    item(rowCount()-1,2)->setText(QString::number(tract_models.back()->get_deleted_track_count()));
    item(cur_row,1)->setText(QString::number(tract_models[cur_row]->get_visible_track_count()));
    item(cur_row,2)->setText(QString::number(tract_models[cur_row]->get_deleted_track_count()));
    emit need_update();
}
void TractTableWidget::sort_track_by_name(void)
{
    std::vector<std::string> name_list;
    for(int i= 0;i < rowCount();++i)
        name_list.push_back(item(i,0)->text().toStdString());
    for(int i= 0;i < rowCount()-1;++i)
    {
        int j = std::min_element(name_list.begin()+i,name_list.end())-name_list.begin();
        if(i == j)
            continue;
        std::swap(name_list[i],name_list[j]);
        for(unsigned int col = 0;col <= 3;++col)
        {
            QString tmp = item(i,col)->text();
            item(i,col)->setText(item(j,col)->text());
            item(j,col)->setText(tmp);
        }
        Qt::CheckState checked = item(i,0)->checkState();
        item(i,0)->setCheckState(item(j,0)->checkState());
        item(j,0)->setCheckState(checked);
        std::swap(thread_data[i],thread_data[j]);
        std::swap(tract_models[i],tract_models[j]);
    }
}
void TractTableWidget::merge_track_by_name(void)
{
    for(int i= 0;i < rowCount()-1;++i)
        for(int j= i+1;j < rowCount()-1;)
        if(item(i,0)->text() == item(j,0)->text())
        {
            tract_models[i]->add(*tract_models[j]);
            delete_row(j);
            item(i,1)->setText(QString::number(tract_models[i]->get_visible_track_count()));
            item(i,2)->setText(QString::number(tract_models[i]->get_deleted_track_count()));
        }
    else
        ++j;
}

void TractTableWidget::move_up(void)
{
    if(currentRow() > 0)
    {
        for(unsigned int col = 0;col <= 3;++col)
        {
            QString tmp = item(currentRow(),col)->text();
            item(currentRow(),col)->setText(item(currentRow()-1,col)->text());
            item(currentRow()-1,col)->setText(tmp);
        }
        Qt::CheckState checked = item(currentRow(),0)->checkState();
        item(currentRow(),0)->setCheckState(item(currentRow()-1,0)->checkState());
        item(currentRow()-1,0)->setCheckState(checked);
        std::swap(thread_data[currentRow()],thread_data[currentRow()-1]);
        std::swap(tract_models[currentRow()],tract_models[currentRow()-1]);
        setCurrentCell(currentRow()-1,0);
    }
    emit need_update();
}

void TractTableWidget::move_down(void)
{
    if(currentRow()+1 < tract_models.size())
    {
        for(unsigned int col = 0;col <= 3;++col)
        {
            QString tmp = item(currentRow(),col)->text();
            item(currentRow(),col)->setText(item(currentRow()+1,col)->text());
            item(currentRow()+1,col)->setText(tmp);
        }
        Qt::CheckState checked = item(currentRow(),0)->checkState();
        item(currentRow(),0)->setCheckState(item(currentRow()+1,0)->checkState());
        item(currentRow()+1,0)->setCheckState(checked);
        std::swap(thread_data[currentRow()],thread_data[currentRow()+1]);
        std::swap(tract_models[currentRow()],tract_models[currentRow()+1]);
        setCurrentCell(currentRow()+1,0);
    }
    emit need_update();
}


void TractTableWidget::delete_tract(void)
{
    if(is_running())
    {
        QMessageBox::information(this,"Error","Please wait for the termination of data processing",0);
        return;
    }
    delete_row(currentRow());
    emit need_update();
}

void TractTableWidget::delete_all_tract(void)
{
    if(is_running())
    {
        QMessageBox::information(this,"Error","Please wait for the termination of data processing",0);
        return;
    }
    command("delete_all_tract");
}

void TractTableWidget::delete_repeated(void)
{
    float distance = 1.0;
    bool ok;
    distance = QInputDialog::getDouble(this,
        "DSI Studio","Distance threshold (voxels)", distance,0,10,1,&ok);
    if (!ok)
        return;
    begin_prog("deleting tracks");
    for(int i = 0;check_prog(i,tract_models.size());++i)
    {
        if(item(i,0)->checkState() == Qt::Checked)
            tract_models[i]->delete_repeated(distance);
        item(i,1)->setText(QString::number(tract_models[i]->get_visible_track_count()));
        item(i,2)->setText(QString::number(tract_models[i]->get_deleted_track_count()));
    }
    emit need_update();
}


void TractTableWidget::delete_by_length(void)
{
    begin_prog("filtering tracks");

    float threshold = 60;
    bool ok;
    threshold = QInputDialog::getDouble(this,
        "DSI Studio","Length threshold in mm:", threshold,0,500,1,&ok);
    if (!ok)
        return;

    for(int i = 0;check_prog(i,tract_models.size());++i)
    {
        if(item(i,0)->checkState() == Qt::Checked)
            tract_models[i]->delete_by_length(threshold);
        item(i,1)->setText(QString::number(tract_models[i]->get_visible_track_count()));
        item(i,2)->setText(QString::number(tract_models[i]->get_deleted_track_count()));
    }
    emit need_update();
}
void TractTableWidget::reconnect_track(void)
{
    int cur_row = currentRow();
    if(cur_row < 0 || item(cur_row,0)->checkState() != Qt::Checked)
        return;
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign maximum bridging distance (in voxels) and angles (degrees)",
                                           QLineEdit::Normal,"4 30",&ok);

    if(!ok)
        return;
    std::istringstream in(result.toStdString());
    float dis,angle;
    in >> dis >> angle;
    if(dis <= 2.0f || angle <= 0.0f)
        return;
    tract_models[uint32_t(cur_row)]->reconnect_track(dis,std::cos(angle*3.14159265358979323846f/180.0f));
    item(cur_row,1)->setText(QString::number(tract_models[cur_row]->get_visible_track_count()));
    emit need_update();
}
void TractTableWidget::edit_tracts(void)
{
    QRgb color;
    if(edit_option == paint)
        color = QColorDialog::getColor(Qt::red,this,"Select color",QColorDialog::ShowAlphaChannel).rgb();
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        switch(edit_option)
        {
        case select:
        case del:
            tract_models[index]->cull(
                             cur_tracking_window.glWidget->angular_selection ?
                             cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos,edit_option == del);
            break;
        case cut:
            tract_models[index]->cut(
                        cur_tracking_window.glWidget->angular_selection ?
                        cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos);
            break;
        case paint:
            tract_models[index]->paint(
                        cur_tracking_window.glWidget->angular_selection ?
                        cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos,color);
            cur_tracking_window.set_data("tract_color_style",1);//manual assigned
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
void TractTableWidget::cut_by_slice(unsigned char dim,bool greater)
{
    cur_tracking_window.ui->SliceModality->setCurrentIndex(0);
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        tract_models[index]->cut_by_slice(dim,cur_tracking_window.current_slice->slice_pos[dim],greater);
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

void TractTableWidget::export_tract_density(tipl::geometry<3>& dim,
                          tipl::vector<3,float> vs,
                          tipl::matrix<4,4,float>& transformation,bool color,bool end_point)
{
    if(color)
    {
        QString filename = QFileDialog::getSaveFileName(
                this,"Save Images files",item(currentRow(),0)->text()+".nii.gz",
                "Image files (*.png *.bmp *nii.gz *.nii *.jpg *.tif);;All files (*)");
        if(filename.isEmpty())
            return;

        tipl::image<tipl::rgb,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        tipl::image<tipl::rgb,2> mosaic;
        if(QFileInfo(filename).fileName().endsWith(".nii") || QFileInfo(filename).fileName().endsWith(".nii.gz"))
        {
            gz_nifti nii;
            tipl::flip_xy(tdi);
            nii << tdi;
            nii.set_voxel_size(vs);
            nii.save_to_file(filename.toStdString().c_str());
        }
        else
        {
            tipl::mosaic(tdi,mosaic,std::sqrt(tdi.depth()));
            QImage qimage((unsigned char*)&*mosaic.begin(),
                          mosaic.width(),mosaic.height(),QImage::Format_RGB32);
            qimage.save(filename);
        }
    }
    else
    {
        QString filename = QFileDialog::getSaveFileName(
                    this,"Save as",item(currentRow(),0)->text()+".nii.gz",
                    "NIFTI files (*nii.gz *.nii);;MAT File (*.mat);;");
        if(filename.isEmpty())
            return;
        if(QFileInfo(filename.toLower()).completeSuffix() != "mat")
            filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";

        tipl::image<unsigned int,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        if(QFileInfo(filename).completeSuffix().toLower() == "mat")
        {
            tipl::io::mat_write mat_header(filename.toLocal8Bit().begin());
            mat_header << tdi;
        }
        else
        {
            tipl::matrix<4,4,float> new_trans(transformation),trans(cur_tracking_window.handle->trans_to_mni);
            if(cur_tracking_window.handle->is_qsdr)
            {
                new_trans.inv();
                trans *= new_trans;
            }
            gz_nifti::save_to_file(filename.toStdString().c_str(),tdi,vs,trans);
        }
    }
}


