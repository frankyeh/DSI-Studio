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
#include "tract_cluster.hpp"

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

void TractTableWidget::addNewTracts(QString tract_name,bool checked)
{
    thread_data.push_back(0);
    tract_models.push_back(new TractModel(cur_tracking_window.handle));
    tract_models.back()->get_fib().threshold = cur_tracking_window["fa_threshold"].toFloat();
    tract_models.back()->get_fib().cull_cos_angle =
            std::cos(cur_tracking_window["turning_angle"].toDouble() * 3.1415926 / 180.0);

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
    cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
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
    tract_models.back()->report = thread_data.back()->report.str();
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
        std::string sfilename = filename.toLocal8Bit().begin();
        addNewTracts(label,false);
        tract_models.back()->load_from_file(&*sfilename.begin(),false);
        if(tract_models.back()->get_cluster_info().empty()) // not multiple cluster file
        {
            item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
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
            this,"Load tracts as","",
            "Tract files (*.txt *.trk *trk.gz *.tck);;All files (*)"));

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
            this,"Load cluster label","",
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
    float param[4] = {0};
    if(method_id)// k-means or EM
    {
        param[0] = QInputDialog::getInt(this,"DSI Studio","Number of clusters:",5,2,100,1);
    }
    else
    {
        std::copy(cur_tracking_window.slice.geometry.begin(),
                  cur_tracking_window.slice.geometry.end(),param);
        param[3] = QInputDialog::getDouble(this,
            "DSI Studio","Clustering detail (mm):",4.0,0.2,50.0,2);
    }
    std::auto_ptr<BasicCluster> handle;
    switch (method_id)
    {
    case 0:
        handle.reset(new TractCluster(param));
        break;
    case 1:
        handle.reset(new FeatureBasedClutering<image::ml::k_means<double,unsigned char> >(param));
        break;
    case 2:
        handle.reset(new FeatureBasedClutering<image::ml::expectation_maximization<double,unsigned char> >(param));
        break;
    }

    for(int index = 0;index < tract_models[currentRow()]->get_visible_track_count();++index)
    {
        if(tract_models[currentRow()]->get_tract_length(index))
            handle->add_tract(
                &*(tract_models[currentRow()]->get_tract(index).begin()),
                tract_models[currentRow()]->get_tract_length(index));
    }
    handle->run_clustering();

    QMessageBox msgBox;
    msgBox.setText("Separate tracks? (Yes: separate tracks according to the cluster labels, No: paint the tracks only)");
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::Yes);
    int rec = msgBox.exec();
    if(rec == QMessageBox::Yes)
    {
        unsigned int cluster_count = method_id ? handle->get_cluster_count() : std::min<float>(handle->get_cluster_count(),50);
        std::vector<std::vector<float> > tracts;
        tract_models[currentRow()]->release_tracts(tracts);
        delete_row(currentRow());
        for(int index = 0;index < cluster_count;++index)
        {
            addNewTracts(QString("Cluster")+QString::number(index));
            unsigned int cluster_size;
            const unsigned int* data = handle->get_cluster(index,cluster_size);
            std::vector<std::vector<float> > add_tracts(cluster_size);
            for(int i = 0;i < cluster_size;++i)
                add_tracts[i].swap(tracts[data[i]]);
            tract_models.back()->add_tracts(add_tracts);
            item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        }
    }
    else
    {
        unsigned int cluster_count = handle->get_cluster_count();
        for(int index = 0;index < cluster_count;++index)
        {
            image::rgb_color color;
            color.from_hsi(3.14159265358979323846*2.0*index/cluster_count,1.0,1.0);
            unsigned int cluster_size = 0;
            const unsigned int* data = handle->get_cluster(index,cluster_size);
            for(int i = 0;i < cluster_size;++i)
                tract_models[currentRow()]->set_tract_color(data[i],color);
        }
        cur_tracking_window.renderWidget->setData("tract_color_style",1);//manual assigned
        emit need_update();
    }
}

void TractTableWidget::save_tracts_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.trk *trk.gz);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
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
    if(!cur_tracking_window.can_convert())
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
        cur_tracking_window.subject2mni(points[index]);
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
            for(unsigned int j = 0;j < tract_data[i].size();j += 3)
            {
                image::vector<3> v(&(tract_data[i][j]));
                cur_tracking_window.subject2mni(v);
                tract_data[i][j] = v[0];
                tract_data[i][j+1] = v[1];
                tract_data[i][j+2] = v[2];
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
            this,"Load tracts color","",
            "Color files (*.txt);;All files (*)");
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
                this,"Save tracts color as",item(currentRow(),0)->text() + "_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::string sfilename = filename.toLocal8Bit().begin();
    tract_models[currentRow()]->save_tracts_color_to_file(&*sfilename.begin());
}

void TractTableWidget::show_tracts_statistics(void)
{
    if(currentRow() >= tract_models.size())
        return;
    std::ostringstream out;
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    begin_prog("calculating");
    for(unsigned int index = 0;check_prog(index,tract_models.size());++index)
    {
        std::string tmp,line;
        tract_models[index]->get_quantitative_info(tmp);
        std::istringstream in(tmp);
        while(std::getline(in,line))
        {
            if(line.find("\t") == std::string::npos)
            {
                if(index == 0)
                    out << line;
                continue;
            }
            track_results[index].push_back(line);
        }
    }
    out << std::endl;
    out << "Tract Name\t";
    for(unsigned int index = 0;index < tract_models.size();++index)
        out << item(index,0)->text().toLocal8Bit().begin() << "\t";
    out << std::endl;
    for(unsigned int index = 0;index < track_results[0].size();++index)
    {
        out << track_results[0][index];
        for(unsigned int i = 1;i < track_results.size();++i)
            out << track_results[i][index].substr(track_results[i][index].find("\t"));
        out << std::endl;
    }
    cur_tracking_window.show_info_dialog("Tract Statistics",out.str());

}

void TractTableWidget::show_method(void)
{
    std::ostringstream out;
    out << cur_tracking_window.handle->report.c_str();
    if(currentRow() < tract_models.size())
        out << tract_models[currentRow()]->report.c_str() << std::endl;
    cur_tracking_window.show_info_dialog("Methods",out.str());
}

void TractTableWidget::save_fa_as(void)
{
    if(currentRow() >= tract_models.size())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save QA as",item(currentRow(),0)->text() + "_qa.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

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
                this,"Save as",item(currentRow(),0)->text() + "_" + action->data().toString() + ".txt",
                "Text files (*.txt);;All files (*)");
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

void TractTableWidget::copy_track(void)
{
    unsigned int cur_row = currentRow();
    addNewTracts(item(cur_row,0)->text() + "copy");
    *(tract_models.back()) = *(tract_models[cur_row]);
    item(currentRow(),1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    emit need_update();
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
                             cur_tracking_window.glWidget->angular_selection ?
                             cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
                             cur_tracking_window.glWidget->pos,edit_option == 2);
            break;
        case 3:
            tract_models[index]->cut(
                        cur_tracking_window.glWidget->angular_selection ?
                        cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
                             cur_tracking_window.glWidget->pos);
            break;
        case 4:
            tract_models[index]->paint(
                        cur_tracking_window.glWidget->angular_selection ?
                        cur_tracking_window["tract_sel_angle"].toFloat():0.0,
                             cur_tracking_window.glWidget->dir1,
                             cur_tracking_window.glWidget->dir2,
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
void TractTableWidget::cut_by_slice(unsigned char dim,bool greater)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        if(item(index,0)->checkState() != Qt::Checked)
            continue;
        tract_models[index]->cut_by_slice(dim,cur_tracking_window.slice.slice_pos[cur_tracking_window.slice.cur_dim],greater);
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
        QImage qimage((unsigned char*)&*tdi.begin(),
                      tdi.width(),tdi.height()*tdi.depth(),QImage::Format_RGB32);
        qimage.save(filename);
    }
    else
    {
        QString filename = QFileDialog::getSaveFileName(
                    this,"Save as",item(currentRow(),0)->text()+".nii.gz",
                    "NIFTI files (*nii.gz *.nii);;MAT File (*.mat);;");
        if(filename.isEmpty())
            return;
#ifdef __APPLE__
// fix the Qt double extension bug here
if(QFileInfo(filename).completeSuffix().contains(".nii.gz"))
    filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
#endif

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
            nii_header.set_voxel_size(vs.begin());
            if(cur_tracking_window.is_qsdr) //QSDR condition
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
        else
            if(QFileInfo(filename).completeSuffix().toLower() == "mat")
            {
                image::io::mat_write mat_header(filename.toLocal8Bit().begin());
                mat_header << tdi;
            }
    }
}


