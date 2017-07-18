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

extern std::vector<atlas> atlas_list;

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
    thread_data.push_back(0);
    tract_models.push_back(new TractModel(cur_tracking_window.handle));

    setRowCount(tract_models.size());
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
        tract_models.back()->add_tracts(lesser[index],image::rgb_color(255,255-color,255-color));
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    for(unsigned int index = 0;index < greater.size();++index)
    {
        if(greater[index].empty())
            continue;
        int color = std::min<int>((index+1)*50,255);
        addNewTracts(QString("Greater_") + QString::number((index+1)*10),false);
        tract_models.back()->add_tracts(greater[index],image::rgb_color(255-color,255-color,255));
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
}

void TractTableWidget::start_tracking(void)
{

    ++tract_serial;
    addNewTracts(cur_tracking_window.regionWidget->getROIname());
    thread_data.back() = new ThreadData(cur_tracking_window["random_seed"].toInt());
    cur_tracking_window.set_tracking_param(*thread_data.back());
    cur_tracking_window.regionWidget->setROIs(thread_data.back());
    thread_data.back()->run(tract_models.back()->get_fib(),
                            cur_tracking_window["thread_count"].toInt(),
                            cur_tracking_window["track_count"].toInt());
    tract_models.back()->report += thread_data.back()->report.str();
    cur_tracking_window.report(tract_models.back()->report.c_str());
    timer->start(1000);
}

void TractTableWidget::filter_by_roi(void)
{
    ThreadData track_thread(cur_tracking_window["random_seed"].toInt());
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

void TractTableWidget::stop_tracking(void)
{
    timer->stop();
    for(unsigned int index = 0;index < thread_data.size();++index)
    {
        delete thread_data[index];
        thread_data[index] = 0;
    }

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
                                     QString("Fail to open the tracks file named %1. \
                                Please check file access privelige or move file to other location.").arg(QFileInfo(filename).baseName()),0);
            break;
        }
        if(tract_models.back()->get_cluster_info().empty()) // not multiple cluster file
        {
            item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
            image::rgb_color c;
            c.from_hsl(((color_gen++)*1.1-std::floor((color_gen++)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
            tract_models.back()->set_color(c.color);
        }
        else
        {
            std::vector<unsigned int> labels;
            labels.swap(tract_models.back()->get_cluster_info());
            load_cluster_label(labels,label);
        }
    }
    emit need_update();
}

void TractTableWidget::load_tracts(void)
{
    load_tracts(QFileDialog::getOpenFileNames(
            this,"Load tracts as",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
            "Tract files (*.txt *.trk *trk.gz *.tck);;All files (*)"));

}
void TractTableWidget::load_tract_label(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load tracts as",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),
                "Tract files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
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
    std::string sfilename = filename.toLocal8Bit().begin();
    TractModel::save_all(&*sfilename.begin(),tract_models);
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
        image::rgb_color c;
        c.from_hsl(((color_gen++)*1.1-std::floor((color_gen++)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        tract_models[index]->set_color(c.color);
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
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
        addNewTracts(Name+QString::number(cluster_index),false);
        tract_models.back()->add_tracts(add_tracts);
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

    load_cluster_label(labels,"cluster");
    assign_colors();
}

void TractTableWidget::clustering(int method_id)
{
    if(tract_models.empty())
        return;
    bool ok = false;
    int n = QInputDialog::getInt(this,
            "DSI Studio",
            "Assign the maximum number of groups",50,1,1000,10,&ok);
    if(!ok)
        return;
    ok = true;
    double detail = method_id ? 0.0 : QInputDialog::getDouble(this,
            "DSI Studio","Clustering detail (mm):",cur_tracking_window.handle->vs[0],0.2,50.0,2,&ok);
    if(!ok)
        return;
    tract_models[currentRow()]->run_clustering(method_id,n,detail);
    std::vector<unsigned int> c = tract_models[currentRow()]->get_cluster_info();
    load_cluster_label(c,"cluster");
    assign_colors();
}

void TractTableWidget::save_tracts_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;ROI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_to_file(&*sfilename.begin());
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
    std::vector<image::vector<3,float> > points;
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
        image::io::mat_write out(filename.toLocal8Bit().begin());
        if(!out)
            return;
        out.write("end_points",(const float*)&*buffer.begin(),3,buffer.size()/3);
    }
}


void TractTableWidget::save_profile(void)
{
    if(currentRow() >= tract_models.size())
        return;
    if(!cur_tracking_window.can_map_to_mni())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save profile as",item(currentRow(),0)->text().replace(':','_') + "profile.mat",
                "MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;

    gz_mat_write out(filename.toLocal8Bit().begin());
    std::vector<float> profile;
    begin_prog("converting coordinates");
    for(unsigned int i = 0;check_prog(i,tract_models[currentRow()]->get_tracts().size());++i)
    {
        cur_tracking_window.handle->get_profile(tract_models[currentRow()]->get_tracts()[i],profile);
        out.write(QString("image%1").arg(i).toLocal8Bit().begin(),&profile[0],1,profile.size());
    }
    //out.write("dimension",&*profile.geometry().begin(),1,3);
}

void TractTableWidget::deep_learning_train(void)
{
    if(cnn.is_running)
    {
        QMessageBox msgBox;
        msgBox.setText("Trainning result");
        msgBox.setDetailedText(cnn.msg.c_str());
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.setDefaultButton(QMessageBox::Ok);
        return;
    }
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save network as","network.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;


    cnn.clear();
    begin_prog("reading");
    for(unsigned int index = 0;check_prog(index,rowCount());++index)
        if (item(index,0)->checkState() == Qt::Checked)
        {
            cnn.add_label(item(index,0)->text().toStdString());
            for(unsigned int i = 0;i < tract_models[index]->get_tracts().size();++i)
                cnn.add_sample(cur_tracking_window.handle.get(),index,tract_models[index]->get_tracts()[i]);
        }

    filename = QFileInfo(filename).absolutePath() + "/tracks.txt";
    {
        std::ofstream out(filename.toStdString().c_str());
        std::copy(cnn.cnn_name.begin(),cnn.cnn_name.end(),std::ostream_iterator<std::string>(out,"\n"));
    }


    filename = QFileInfo(filename).absolutePath() + "/network_data.bin";
    cnn.cnn_data.input = image::geometry<3>(64,80,3);
    cnn.cnn_data.output = image::geometry<3>(1,1,rowCount());
    cnn.cnn_data.save_to_file(filename.toStdString().c_str());


    // save atlas as a nifti file
    if(cur_tracking_window.handle->is_qsdr) //QSDR condition
    {
        image::basic_image<int,4> atlas(image::geometry<4>(
                cur_tracking_window.handle->dim[0],
                cur_tracking_window.handle->dim[1],
                cur_tracking_window.handle->dim[2],
                rowCount()));

        for(unsigned int index = 0;check_prog(index,rowCount());++index)
        {
            image::basic_image<unsigned char,3> track_map(cur_tracking_window.handle->dim);
            for(unsigned int i = 0;i < tract_models[index]->get_tracts().size();++i)
            {
                const std::vector<float>& tracks = tract_models[index]->get_tracts()[i];
                for(int j = 0;j < tracks.size();j += 3)
                {
                    image::pixel_index<3> p(std::round(tracks[j]),std::round(tracks[j+1]),std::round(tracks[j+2]),track_map.geometry());
                    if(track_map.geometry().is_valid(p))
                        track_map[p.index()] = 1;
                    if(j)
                    {
                        for(float r = 0.2f;r < 1.0f;r += 0.2f)
                        {
                            image::pixel_index<3> p2(std::round(tracks[j]*r+tracks[j-3]*(1-r)),
                                                     std::round(tracks[j+1]*r+tracks[j-2]*(1-r)),
                                                     std::round(tracks[j+2]*r+tracks[j-1]*(1-r)),track_map.geometry());
                            if(track_map.geometry().is_valid(p2))
                                track_map[p2.index()] = 1;
                        }
                    }
                }
            }
            while(image::morphology::smoothing_fill(track_map))
                image::morphology::defragment(track_map);
            image::morphology::smoothing(track_map);
            image::morphology::defragment(track_map);


            QString track_file_name = QFileInfo(filename).absolutePath() + "/" + item(index,0)->text() + ".nii.gz";
            gz_nifti nifti2;
            nifti2.set_voxel_size(cur_tracking_window.current_slice->voxel_size.begin());
            nifti2.set_image_transformation(cur_tracking_window.handle->trans_to_mni.begin());
            nifti2 << track_map;
            nifti2.save_to_file(track_file_name.toLocal8Bit().begin());

            std::copy(track_map.begin(),track_map.end(),atlas.begin() + index*track_map.size());

            if(index+1 == rowCount())
            {
                filename = QFileInfo(filename).absolutePath() + "/tracks.nii.gz";
                gz_nifti nifti;
                nifti.set_voxel_size(cur_tracking_window.current_slice->voxel_size.begin());
                nifti.set_image_transformation(cur_tracking_window.handle->trans_to_mni.begin());
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
    if(!cur_tracking_window.can_map_to_mni())
        return;
    std::map<float,std::string,std::greater<float> > sorted_list;
    if(!tract_models[currentRow()]->recognize(sorted_list))
    {
        QMessageBox::information(this,"Error","Cannot recognize tracks.",0);
        return;
    }
    std::ostringstream out;
    auto beg = sorted_list.begin();
    for(int i = 0;i < 5;++i,++beg)
        out << beg->second << "\t" << beg->first << std::endl;
    cur_tracking_window.show_info_dialog("Tract Recognition Result",out.str());

    /*
    std::string result;
    tract_models[currentRow()]->recognize_report(result);
    QMessageBox::information(this,"Result",result.c_str(),0);*/
}

void TractTableWidget::saveTransformedTracts(const float* transform)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",item(currentRow(),0)->text() + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
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
                image::vector<3> v(&(tract_data[i][j]));
                cur_tracking_window.handle->subject2mni(v);
                tract_data[i][j] = v[0];
                tract_data[i][j+1] = v[1];
                tract_data[i][j+2] = v[2];
            }
            std::vector<float> smooth_track(tract_data[i]);
            for(unsigned int j = 0;j < tract_data[i].size();j += 3)
            {
                if(j > 2)
                {
                    smooth_track[j] += tract_data[i][j-3];
                    smooth_track[j+1] += tract_data[i][j-2];
                    smooth_track[j+2] += tract_data[i][j-1];
                }
                else
                {
                    smooth_track[j] += tract_data[i][j];
                    smooth_track[j+1] += tract_data[i][j+1];
                    smooth_track[j+2] += tract_data[i][j+2];
                }

                if(j+3 < smooth_track.size())
                {
                    smooth_track[j] += tract_data[i][j+3];
                    smooth_track[j+1] += tract_data[i][j+4];
                    smooth_track[j+2] += tract_data[i][j+5];
                }
                else
                {
                    smooth_track[j] += tract_data[i][j];
                    smooth_track[j+1] += tract_data[i][j+1];
                    smooth_track[j+2] += tract_data[i][j+2];
                }
            }
            image::multiply_constant(smooth_track,1.0/3.0);
            tract_data[i].swap(smooth_track);

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

    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->load_tracts_color_from_file(&*sfilename.begin());
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit need_update();
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

void get_track_statistics(const std::vector<TractModel*>& tract_models,
                          const std::vector<std::string>& track_name,
                          std::string& result)
{
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    image::par_for(tract_models.size(),[&](unsigned int index)
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

void TractTableWidget::show_tracts_statistics(void)
{
    if(tract_models.empty())
        return;
    std::string result;
    {
        std::vector<TractModel*> active_tracks;
        std::vector<std::string> track_name;
        for(unsigned int index = 0;index < tract_models.size();++index)
            if(item(index,0)->checkState() == Qt::Checked)
            {
                active_tracks.push_back(tract_models[index]);
                track_name.push_back(item(index,0)->text().toStdString());
            }
        get_track_statistics(active_tracks,track_name,result);
    }
    cur_tracking_window.show_info_dialog("Tract Statistics",result);

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
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            delete thread_data[index];
            delete tract_models[index];
        }
        thread_data.clear();
        tract_models.clear();
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
                "Text files (*.txt);;MATLAB file (*.mat);;All files (*)");
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
    delete thread_data[row];
    delete tract_models[row];
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
    if(currentRow())
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
        "DSI Studio","Distance threshold", distance,0,10,1,&ok);
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
void TractTableWidget::edit_tracts(void)
{
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
                             cur_tracking_window.glWidget->pos,QColorDialog::getColor(Qt::red,this,"Select color",QColorDialog::ShowAlphaChannel).rgb());
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

void TractTableWidget::export_tract_density(image::geometry<3>& dim,
                          image::vector<3,float> vs,
                          image::matrix<4,4,float>& transformation,bool color,bool end_point)
{
    if(color)
    {
        QString filename = QFileDialog::getSaveFileName(
                this,"Save Images files",item(currentRow(),0)->text(),
                "Image files (*.png *.bmp *.jpg *.tif);;All files (*)");
        if(filename.isEmpty())
            return;

        image::basic_image<image::rgb_color,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        image::basic_image<image::rgb_color,2> mosaic;
        image::mosaic(tdi,mosaic,std::sqrt(tdi.depth()));
        QImage qimage((unsigned char*)&*mosaic.begin(),
                      mosaic.width(),mosaic.height(),QImage::Format_RGB32);
        qimage.save(filename);
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

        image::basic_image<unsigned int,3> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
        {
            if(item(index,0)->checkState() != Qt::Checked)
                continue;
            tract_models[index]->get_density_map(tdi,transformation,end_point);
        }
        if(QFileInfo(filename).completeSuffix().toLower() == "mat")
        {
            image::io::mat_write mat_header(filename.toLocal8Bit().begin());
            mat_header << tdi;
        }
        else
        {
            gz_nifti nii_header;
            nii_header.set_voxel_size(vs.begin());
            if(cur_tracking_window.handle->is_qsdr)
            {
                image::matrix<4,4,float> new_trans(transformation),trans(cur_tracking_window.handle->trans_to_mni.begin());
                new_trans.inv();
                trans *= new_trans;
                nii_header.set_image_transformation(trans.begin());
            }
            else
                image::flip_xy(tdi);
            nii_header << tdi;
            nii_header.save_to_file(filename.toLocal8Bit().begin());
        }
    }
}


