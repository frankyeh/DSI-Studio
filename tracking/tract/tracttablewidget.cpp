#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
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
#include "atlas.hpp"


TractTableWidget::TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent) :
    QTableWidget(parent),cur_tracking_window(cur_tracking_window_)
{
    setColumnCount(4);
    setColumnWidth(0,200);
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
    timer_update = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(fetch_tracts()));
    connect(timer_update, SIGNAL(timeout()), this, SLOT(show_tracking_progress()));

    update_color_map();
}

TractTableWidget::~TractTableWidget(void)
{
}

void TractTableWidget::update_color_map(void)
{
    if(cur_tracking_window["tract_color_map"].toInt()) // color map from file
    {
        QString filename = QCoreApplication::applicationDirPath()+"/color_map/"+
                cur_tracking_window.renderWidget->getListValue("tract_color_map")+".txt";
        color_map.load_from_file(filename.toStdString().c_str());
        color_map_rgb.load_from_file(filename.toStdString().c_str());
        cur_tracking_window.glWidget->tract_color_bar.load_from_file(filename.toStdString().c_str());
    }
    else
    {
        tipl::rgb from_color(uint32_t(cur_tracking_window["tract_color_min"].toUInt()));
        tipl::rgb to_color(uint32_t(cur_tracking_window["tract_color_max"].toUInt()));
        cur_tracking_window.glWidget->tract_color_bar.two_color(from_color,to_color);
        std::swap(from_color.r,from_color.b);
        std::swap(to_color.r,to_color.b);
        color_map.two_color(from_color,to_color);
        color_map_rgb.two_color(from_color,to_color);
    }
    cur_tracking_window.glWidget->tract_color_bar_pos = {10,10};
}
void TractTableWidget::contextMenuEvent(QContextMenuEvent * event )
{
    if(event->reason() == QContextMenuEvent::Mouse)
        cur_tracking_window.ui->menuTracts->popup(event->globalPos());
}

void TractTableWidget::check_check_status(int row, int col)
{
    if(col != 0)
        return;
    setCurrentCell(row,col);
    if (item(row,0)->checkState() == Qt::Checked)
    {
        if (item(row,0)->data(Qt::ForegroundRole) == QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
            emit show_tracts();
        }
    }
    else
    {
        if (item(row,0)->data(Qt::ForegroundRole) != QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
            emit show_tracts();
        }
    }
}



void TractTableWidget::draw_tracts(unsigned char dim,int pos,
                                   QImage& scaled_image,float display_ratio)
{
    auto selected_tracts = get_checked_tracks();
    auto selected_tracts_rendering = get_checked_tracks_rendering();
    if(selected_tracts.empty() || selected_tracts.size() != selected_tracts_rendering.size())
        return;
    uint32_t max_count = uint32_t(cur_tracking_window["roi_track_count"].toInt());
    auto tract_color_style = cur_tracking_window["tract_color_style"].toInt();
    unsigned int thread_count = tipl::max_thread_count;
    std::vector<std::vector<std::vector<tipl::vector<2,float> > > > lines_threaded(thread_count);
    std::vector<std::vector<std::vector<unsigned int> > > colors_threaded(thread_count);
    tipl::matrix<4,4>* pt = (cur_tracking_window.current_slice->is_diffusion_space ? nullptr : &(cur_tracking_window.current_slice->to_slice));
    max_count /= selected_tracts.size();
    tipl::par_for<tipl::sequential_with_id>(selected_tracts.size(),[&](unsigned int index,unsigned int thread)
    {
        if(cur_tracking_window.slice_need_update)
            return;
        auto lock = selected_tracts_rendering[index]->start_reading(false/* no wait*/);
        if(!lock.get())
            return;
        selected_tracts[index]->get_in_slice_tracts(dim,pos,pt,lines_threaded[thread],colors_threaded[thread],max_count,tract_color_style,
                            selected_tracts_rendering[index]->about_to_write);
    });
    if(cur_tracking_window.slice_need_update)
        return;
    std::vector<std::vector<tipl::vector<2,float> > > lines;
    std::vector<std::vector<unsigned int> > colors;
    tipl::aggregate_results(std::move(lines_threaded),lines);
    tipl::aggregate_results(std::move(colors_threaded),colors);
    struct draw_point_class{
        int height;
        int width;
        std::vector<QRgb*> I;
        draw_point_class(QImage& scaled_image):I(uint32_t(scaled_image.height()))
        {
            for (int y = 0; y < scaled_image.height(); ++y)
                I[uint32_t(y)] = reinterpret_cast<QRgb*>(scaled_image.scanLine(y));
            height = scaled_image.height();
            width = scaled_image.width();
        }
        inline void operator()(int x,int y,unsigned int color)
        {
            if(y < 0 || x < 0 || y >= height || x >= width)
                return;
            I[uint32_t(y)][uint32_t(x)] = color;
        }
    } draw_point(scaled_image);


    auto draw_line = [&](int x,int y,int x1,int y1,unsigned int color)
    {
        tipl::draw_line(x,y,x1,y1,[&](int xx,int yy)
        {
            draw_point(xx,yy,color);
        });
    };

    tipl::adaptive_par_for(lines.size(),[&](unsigned int i)
    {
        auto& line = lines[i];
        auto& color = colors[i];
        tipl::add_constant(line,0.5f);
        tipl::multiply_constant(line,display_ratio);
        for(size_t j = 1;j < line.size();++j)
            draw_line(int(line[j-1][0]),int(line[j-1][1]),int(line[j][0]),int(line[j][1]),color[j]);
    });
}

void TractTableWidget::addNewTracts(QString tract_name,bool checked)
{
    auto new_model = std::make_shared<TractModel>(cur_tracking_window.handle);
    new_model->name = tract_name.toStdString();
    addNewTracts(new_model,checked);
}
void TractTableWidget::addNewTracts(std::shared_ptr<TractModel> new_tract,bool checked)
{
    thread_data.push_back(nullptr);
    tract_rendering.push_back(std::make_shared<TractRender>());
    tract_models.push_back(new_tract);
    insertRow(tract_models.size()-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(QString(new_tract->name.c_str()));
    item0->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
    item0->setData(Qt::ForegroundRole,checked ? QBrush(Qt::black) : QBrush(Qt::gray));
    setItem(tract_models.size()-1, 0, item0);
    for(unsigned int index = 1;index <= 3;++index)
    {
        QTableWidgetItem *item1 = new QTableWidgetItem(index == 1 && new_tract->get_visible_track_count() ?
                                                       QString::number(new_tract->get_visible_track_count()) : QString());
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
        tract_rendering.back()->need_update = true;
    }
    for(unsigned int index = 0;index < greater.size();++index)
    {
        if(greater[index].empty())
            continue;
        int color = std::min<int>((index+1)*50,255);
        addNewTracts(QString("Greater_") + QString::number((index+1)*10),false);
        tract_models.back()->add_tracts(greater[index],tipl::rgb(255-color,255-color,255));
        item(tract_models.size()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        tract_rendering.back()->need_update = true;
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit show_tracts();
}
void TractTableWidget::load_built_in_atlas(const std::string& tract_name)
{
    if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }

    if(tract_name.empty()) // load all
    {
        for(const auto& each : cur_tracking_window.handle->tractography_name_list)
            load_built_in_atlas(each);
        return;
    }
    auto track_ids = cur_tracking_window.handle->get_track_ids(tract_name);
    if(track_ids.empty())
    {
        QMessageBox::critical(this,"ERROR",QString("cannot find a matched tract for ") + tract_name.c_str());
        return;
    }

    auto track_atlas = cur_tracking_window.handle->track_atlas;
    addNewTracts(tract_name.c_str());

    tract_rendering.back()->need_update = true;
    const auto& atlas_tract = track_atlas->get_tracts();
    const auto& atlas_cluster = track_atlas->tract_cluster;
    std::vector<std::vector<float> > new_tracts;
    for(size_t i = 0;i < atlas_cluster.size();++i)
        if(std::find(track_ids.begin(),track_ids.end(),atlas_cluster[i]) != track_ids.end())
            new_tracts.push_back(atlas_tract[i]);

    auto lock = tract_rendering.back()->start_writing();
    tract_models.back()->add_tracts(new_tracts);
    tract_models.back()->report = tract_name + (cur_tracking_window.handle->is_mni ?
            " was shown from a population-based tractography atlas (Yeh, Nat Commun 13(1), 4933, 2022).":
            " was mapped by nonlinearly warping a population-based tractography atlas (Yeh, Nat Commun 13(1), 4933, 2022) to the native space.");

    item(int(tract_models.size()-1),1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    item(int(tract_models.size()-1),2)->setText(QString::number(tract_models.back()->get_deleted_track_count()));

    show_report();
}
void TractTableWidget::load_built_in_atlas(void)
{
    load_built_in_atlas("");
}
void TractTableWidget::start_tracking(void)
{

    QString tract_name = cur_tracking_window.regionWidget->getROIname();
    std::vector<std::string> tracking_param({std::string("run_tracking"),
                                             tract_name.toStdString(),
                                             cur_tracking_window.get_parameter_id()});
    if(cur_tracking_window["dt_index1"].toInt() || cur_tracking_window["dt_index2"].toInt())
    {
        if(!command({std::string("set_dt_index"),
                     cur_tracking_window.dt_list[cur_tracking_window["dt_index1"].toInt()].toStdString() + '&' +
                     cur_tracking_window.dt_list[cur_tracking_window["dt_index2"].toInt()].toStdString(),
                     std::to_string(cur_tracking_window.renderWidget->getData("dt_threshold_type").toInt())}))
        {
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
            return;
        }
    }
    else
    if(cur_tracking_window.ui->tract_target_0->currentIndex() > 0) // auto track
    {
        tract_name = cur_tracking_window.ui->tract_target_1->currentText();
        if(cur_tracking_window.ui->tract_target_2->isVisible() &&
           cur_tracking_window.ui->tract_target_2->currentText() != "All")
        {
            tract_name += "_";
            tract_name += cur_tracking_window.ui->tract_target_2->currentText();
        }
        if(!cur_tracking_window.handle->trackable)
        {
            load_built_in_atlas(tract_name.toStdString());
            return;
        }
        tracking_param[1] = tract_name.toStdString();
        tracking_param.push_back(std::to_string(cur_tracking_window["tolerance"].toFloat()));
    }
    command(tracking_param);
}

void TractTableWidget::show_report(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    cur_tracking_window.report(tract_models[uint32_t(currentRow())]->report.c_str());
}

void TractTableWidget::filter_by_roi(void)
{
    ThreadData track_thread(cur_tracking_window.handle);
    track_thread.param.set_code(cur_tracking_window.get_parameter_id());
    cur_tracking_window.regionWidget->setROIs(&track_thread);
    for_each_bundle("filter by roi",[&](unsigned int index)
    {
        return tract_models[index]->filter_by_roi(track_thread.roi_mgr);
    });
}
void TractTableWidget::show_tracking_progress(void)
{
    bool has_thread = false;
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index].get())
        {
            item(int(index),1)->setText(QString::number(thread_data[index]->get_total_tract_count()));
            if(thread_data[index]->get_total_seed_count())
                item(int(index),3)->setText(QString::number(thread_data[index]->get_total_seed_count()));
            else
                item(int(index),3)->setText("initiating");
            has_thread = true;
        }
    if(!has_thread)
        timer_update->stop();
}

void TractTableWidget::fetch_tracts(void)
{
    bool has_tracts = false;
    bool has_thread = false;
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index].get())
        {    
            {
                auto lock = tract_rendering[index]->start_writing(false);
                if(lock.get())
                {
                    has_tracts = thread_data[index]->fetchTracks(tract_models[index].get());
                    tract_rendering[index]->need_update = true;
                }
            }
            if(thread_data[index]->is_ended())
            {
                // used in debugging autotrack
                {
                    auto regions = cur_tracking_window.regionWidget->regions;
                    if(regions.size() >= 3 && cur_tracking_window.regionWidget->item(int(0),0)->text() == "debug")
                        {
                            regions[0]->region = thread_data[index]->roi_mgr->atlas_seed;
                            regions[0]->modified = true;
                            regions[1]->region = thread_data[index]->roi_mgr->atlas_limiting;
                            regions[1]->modified = true;
                            regions[2]->region = thread_data[index]->roi_mgr->atlas_not_end;
                            regions[2]->modified = true;
                            if(regions.size() >= 4)
                            {
                                regions[3]->region = thread_data[index]->roi_mgr->atlas_roi;
                                regions[3]->modified = true;
                            }
                        }
                }
                tract_rendering[index]->need_update = true;
                auto lock = tract_rendering[index]->start_writing();
                has_tracts |= thread_data[index]->fetchTracks(tract_models[index].get()); // clear both front and back buffer
                has_tracts |= thread_data[index]->fetchTracks(tract_models[index].get()); // clear both front and back buffer
                tract_models[index]->trim(thread_data[index]->param.tip_iteration);
                item(int(index),1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
                item(int(index),2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
                item(int(index),3)->setText(QString::number(thread_data[index]->get_total_seed_count()));
                thread_data[index].reset();
            }
            else
                has_thread = true;
        }

    if(has_tracts)
        emit show_tracts();
    if(!has_thread)
    {
        timer->stop();
        cur_tracking_window.history.has_other_thread = false;
    }
}

void TractTableWidget::stop_tracking(void)
{
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index].get())
            thread_data[index]->end_thread();
}

void TractTableWidget::load_tract_label(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load Tracts Label",QFileInfo(cur_tracking_window.work_path).absolutePath(),
                "Tract files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    load_tract_label(filename);
}
void TractTableWidget::load_tract_label(QString filename)
{
    std::ifstream in(filename.toStdString().c_str());
    std::vector<std::string> name((std::istream_iterator<std::string>(in)),(std::istream_iterator<std::string>()));
    for(int i = 0;i < rowCount() && i < name.size();++i)
        item(rowCount()-1-i,0)->setText(name[name.size()-1-i].c_str());
}

void TractTableWidget::check_all(void)
{
    for(int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
}

void TractTableWidget::uncheck_all(void)
{
    for(int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
}
QString TractTableWidget::output_format(void)
{
    switch(cur_tracking_window["track_format"].toInt())
    {
    case 0:
        return ".tt.gz";
    case 1:
        return ".trk.gz";
    case 2:
        return ".txt";
    }
    return "";
}

void TractTableWidget::set_color(void)
{
    if(tract_models.empty() || currentRow() == -1 ||
       tract_models[uint32_t(currentRow())]->get_visible_track_count() == 0)
        return;
    QColor color = QColorDialog::getColor(tract_models[uint32_t(currentRow())]->get_tract_color(0),(QWidget*)this,"Select color",QColorDialog::ShowAlphaChannel);
    if(!color.isValid())
        return;
    tract_models[uint32_t(currentRow())]->set_color(color.rgb());
    tract_rendering[uint32_t(currentRow())]->need_update = true;
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit show_tracts();
}

void TractTableWidget::assign_colors(void)
{
    for(unsigned int index = 0;index < tract_models.size();++index)
    {
        tipl::rgb c = tipl::rgb::generate(index);
        auto lock = tract_rendering[index]->start_writing();
        tract_models[index]->set_color(c.color);
        tract_rendering[index]->need_update = true;
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit show_tracts();
}
void TractTableWidget::load_cluster_label(const std::vector<unsigned int>& labels,QStringList Names)
{
    auto cur_row = uint32_t(currentRow());
    std::vector<std::vector<float> > tracts;
    tract_models[cur_row]->release_tracts(tracts);
    tract_rendering[cur_row]->need_update = true;
    delete_row(currentRow());
    unsigned int cluster_count = uint32_t(Names.empty() ? int(1+tipl::max_value(labels)):int(Names.count()));
    tipl::progress prog("loading clusters");
    for(unsigned int cluster_index = 0;prog(cluster_index,cluster_count);++cluster_index)
    {
        unsigned int fiber_num = uint32_t(std::count(labels.begin(),labels.end(),cluster_index));
        if(!fiber_num)
            continue;
        std::vector<std::vector<float> > add_tracts(fiber_num);
        for(unsigned int index = 0,i = 0;index < labels.size();++index)
            if(labels[index] == cluster_index)
            {
                add_tracts[i].swap(tracts[index]);
                ++i;
            }
        if(int(cluster_index) < Names.size())
            addNewTracts(Names[int(cluster_index)],true);
        else
            addNewTracts(QString("cluster")+QString::number(cluster_index),true);
        tract_models.back()->add_tracts(add_tracts);
        tract_models.back()->report = tract_models[cur_row]->report;
        tract_models.back()->geo = tract_models[cur_row]->geo;
        tract_models.back()->vs = tract_models[cur_row]->vs;
        tract_models.back()->trans_to_mni = tract_models[cur_row]->trans_to_mni;
        item(int(tract_models.size())-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    }
    emit show_tracts();
}

void TractTableWidget::open_cluster_label(void)
{
    if(tract_models.empty())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load cluster label",QFileInfo(cur_tracking_window.work_path).absolutePath(),
            "Cluster label files (*.txt);;All files (*)");
    if(!filename.size())
        return;

    std::ifstream in(filename.toStdString().c_str());
    std::vector<unsigned int> labels(tract_models[uint32_t(currentRow())]->get_visible_track_count());
    std::copy(std::istream_iterator<unsigned int>(in),
              std::istream_iterator<unsigned int>(),labels.begin());
    load_cluster_label(labels);
    assign_colors();
}

void TractTableWidget::recog_tracks(void)
{
    if(currentRow() >= int(tract_models.size()) || tract_models[uint32_t(currentRow())]->get_tracts().size() == 0)
        return;
    if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }
    std::multimap<float,std::string,std::greater<float> > sorted_list;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        if(!cur_tracking_window.handle->recognize_and_sort(tract_models[uint32_t(currentRow())],sorted_list))
        {
            QMessageBox::critical(this,"ERROR","Cannot recognize tracks.");
            return;
        }
    }
    std::ostringstream out;
    auto beg = sorted_list.begin();
    for(size_t i = 0;i < sorted_list.size();++i,++beg)
        if(beg->first != 0.0f)
            out << beg->first*100.0f << "% " << beg->second << std::endl;
    show_info_dialog("Tract Recognition Result",out.str());
}


void TractTableWidget::recognize_and_cluster(void)
{
    std::vector<std::string> labels;
    std::vector<unsigned int> new_c;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        if(!cur_tracking_window.handle->recognize(tract_models[uint32_t(currentRow())],new_c,labels))
        {
            QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
            return;
        }
    }
    QStringList Names;
    for(const auto& str : labels)
        Names << str.c_str();
    load_cluster_label(new_c,Names);
}

void TractTableWidget::recognize_rename(void)
{
    if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }
    tipl::progress prog("Recognize and rename");
    for(unsigned int index = 0;prog(index,tract_models.size());++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
        {
            std::multimap<float,std::string,std::greater<float> > sorted_list;
            auto lock = tract_rendering[index]->start_reading();
            if(!cur_tracking_window.handle->recognize_and_sort(tract_models[index],sorted_list))
                return;
            item(int(index),0)->setText(sorted_list.begin()->second.c_str());
        }
}

void TractTableWidget::clustering(int method_id)
{
    if(tract_models.empty())
        return;
    bool ok = false;
    int n = QInputDialog::getInt(this,
            QApplication::applicationName(),
            "Assign the maximum number of groups",50,1,5000,10,&ok);
    if(!ok)
        return;
    ok = true;
    double detail = method_id ? 0.0 : QInputDialog::getDouble(this,
            QApplication::applicationName(),"Clustering detail (mm):",cur_tracking_window.handle->vs[0],0.2,50.0,2,&ok);
    if(!ok)
        return;
    std::vector<unsigned int> c;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        tract_models[uint32_t(currentRow())]->run_clustering(method_id,n,detail);
        c = tract_models[uint32_t(currentRow())]->tract_cluster;
    }
    load_cluster_label(c);
    assign_colors();
}


void TractTableWidget::save_all_tracts_end_point_as(void)
{
    auto selected_tracts = get_checked_tracks();
    if(selected_tracts.empty())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save end points as",
                QString::fromStdString(cur_tracking_window.history.file_stem()) + "_endpoint.nii.gz",
                "NIFTI files (*nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    bool ok;
    float dis = float(QInputDialog::getDouble(this,
        QApplication::applicationName(),"Assign end segment length in voxel distance:",3.0,0.0,10.0,1,&ok));
    if (!ok)
        return;

    auto locks = start_reading_checked_tracks();
    TractModel::export_end_pdi(filename.toStdString().c_str(),selected_tracts,dis);
}
void TractTableWidget::save_end_point_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",
                QString::fromStdString(cur_tracking_window.history.file_stem()) + "_endpoint.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    tract_models[uint32_t(currentRow())]->save_end_points(filename.toStdString().c_str());
}

void TractTableWidget::save_end_point_in_mni(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0)
        return;
    if(!cur_tracking_window.handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",
                QString::fromStdString(cur_tracking_window.history.file_stem()) + "_endpoint.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;

    std::vector<tipl::vector<3,short> > points1,points2;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        tract_models[size_t(currentRow())]->to_end_point_voxels(points1,points2);
    }
    points1.insert(points1.end(),points2.begin(),points2.end());

    std::vector<tipl::vector<3> > points(points1.begin(),points1.end());
    std::vector<float> buffer;
    for(unsigned int index = 0;index < points.size();++index)
    {
        cur_tracking_window.handle->sub2mni(points[index]);
        buffer.push_back(points1[index][0]);
        buffer.push_back(points1[index][1]);
        buffer.push_back(points1[index][2]);
    }

    if (QFileInfo(filename).suffix().toLower() == "txt")
    {
        std::ofstream out(filename.toStdString().c_str(),std::ios::out);
        if (!out)
            return;
        std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(out," "));
    }
    if (QFileInfo(filename).suffix().toLower() == "mat")
    {
        tipl::io::mat_write out(filename.toStdString().c_str());
        if(!out)
            return;
        out.write("end_points",buffer,3);
    }
}


void TractTableWidget::save_transformed_tracts(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",
                QString::fromStdString(cur_tracking_window.history.file_stem()) + "_" + cur_tracking_window.current_slice->get_name().c_str() + output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
    if(!slice)
    {
        QMessageBox::critical(this,"ERROR","Current slice is in the DWI space. Please use regular tract saving function");
        return;
    }
    if(slice->running)
    {
        QMessageBox::critical(this,"ERROR","Please wait until registration is complete");
        return;
    }
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_transformed_tracts_to_file(filename.toStdString().c_str(),slice->dim,slice->vs,slice->trans_to_mni,slice->to_slice,false))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR","File not saved. Please check write permission");
}


void TractTableWidget::save_transformed_endpoints(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end_point as",QString::fromStdString(cur_tracking_window.history.file_stem()) + "_" +
                cur_tracking_window.current_slice->get_name().c_str() + "_endpoints.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
    if(!slice)
    {
        QMessageBox::critical(this,"ERROR","Current slice is in the DWI space. Please use regular tract saving function");
        return;
    }
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_transformed_tracts_to_file(filename.toStdString().c_str(),slice->dim,slice->vs,slice->trans_to_mni,slice->to_slice,true))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR","File not saved. Please check write permission");
}
extern std::vector<std::string> fa_template_list;
void TractTableWidget::save_tracts_in_template(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    if(!cur_tracking_window.handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }

    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",QString::fromStdString(cur_tracking_window.history.file_stem()) + " " +
                QFileInfo(fa_template_list[cur_tracking_window.handle->template_id].c_str()).baseName() +
                output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_in_template_space(cur_tracking_window.handle,filename.toStdString().c_str()))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR","File not saved. Please check write permission");
}

void TractTableWidget::save_tracts_in_mni(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;

    if(!cur_tracking_window.handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }

    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",QString::fromStdString(cur_tracking_window.history.file_stem()) + "_mni" + output_format(),
                "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_in_template_space(cur_tracking_window.handle,filename.toStdString().c_str(),true))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR","File not saved. Please check write permission");
}


void TractTableWidget::save_tracts_color_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts color as",QString::fromStdString(cur_tracking_window.history.file_stem()) + "_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::string sfilename = filename.toStdString().c_str();
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    tract_models[uint32_t(currentRow())]->save_tracts_color_to_file(&*sfilename.begin());
}

void TractTableWidget::cell_changed(int row, int column)
{
    if(row >= 0 && row < tract_models.size())
        tract_models[row]->name = item(int(row),0)->text().toStdString();
    if(cur_tracking_window["show_track_label"].toInt())
        cur_tracking_window.glWidget->update();

}
void get_track_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          std::string& result)
{
    if(tract_models.empty())
        return;
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    {
        tipl::progress p("for each tract");
        for(size_t index = 0;p(index,tract_models.size());++index)
        {
            std::string tmp,line;
            tract_models[index]->get_quantitative_info(handle,tmp);
            std::istringstream in(tmp);
            while(std::getline(in,line))
            {
                if(line.find("\t") == std::string::npos)
                    continue;
                track_results[index].push_back(line);
            }
        }
        if(p.aborted())
            return;
    }
    std::vector<std::string> metrics_name;
    for(unsigned int j = 0;j < track_results[0].size();++j)
        metrics_name.push_back(track_results[0][j].substr(0,track_results[0][j].find("\t")));

    std::ostringstream out;
    out << "Tract Name\t";
    for(unsigned int index = 0;index < tract_models.size();++index)
        out << tract_models[index]->name << "\t";
    out << std::endl;
    for(unsigned int index = 0;index < metrics_name.size();++index)
    {
        out << metrics_name[index];
        for(unsigned int i = 0;i < track_results.size();++i)
            if(index < track_results[i].size())
                out << track_results[i][index].substr(track_results[i][index].find("\t"));
            else
                out << "\t";
        out << std::endl;
    }
    result = out.str();
}
std::vector<std::shared_ptr<TractModel> > TractTableWidget::get_checked_tracks(void)
{
    std::vector<std::shared_ptr<TractModel> > active_tracks;
    for(unsigned int index = 0;index < tract_models.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            active_tracks.push_back(tract_models[index]);
    return active_tracks;
}
std::vector<std::shared_ptr<TractRender> > TractTableWidget::get_checked_tracks_rendering(void)
{
    std::vector<std::shared_ptr<TractRender> > active_tracks_rendering;
    for(unsigned int index = 0;index < tract_rendering.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            active_tracks_rendering.push_back(tract_rendering[index]);
    return active_tracks_rendering;
}

std::vector<std::shared_ptr<TractRender::end_reading> > TractTableWidget::start_reading_checked_tracks(void)
{
    std::vector<std::shared_ptr<TractRender::end_reading> > locks;
    for(unsigned int index = 0;index < tract_rendering.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            locks.push_back(tract_rendering[index]->start_reading());
    return locks;
}
std::vector<std::shared_ptr<TractRender::end_writing> > TractTableWidget::start_writing_checked_tracks(void)
{
    std::vector<std::shared_ptr<TractRender::end_writing> > locks;
    for(unsigned int index = 0;index < tract_rendering.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            locks.push_back(tract_rendering[index]->start_writing());
    return locks;
}
void TractTableWidget::show_tracts_statistics(void)
{
    if(tract_models.empty())
        return;
    std::string result;
    {
        tipl::progress p("calculate tract statistics",true);
        get_track_statistics(cur_tracking_window.handle,get_checked_tracks(),result);
    }
    if(!result.empty())
        show_info_dialog("Tract Statistics",result);

}
void TractTableWidget::need_update_all(void)
{
    for(auto& t:tract_rendering)
        t->need_update = true;
}
bool TractTableWidget::render_tracts(GLWidget* glwidget,std::chrono::high_resolution_clock::time_point end_time)
{
    auto tracks = get_checked_tracks();
    auto renders = get_checked_tracks_rendering();

    std::vector<size_t> update_list;
    for(unsigned int index = 0;index < renders.size();++index)
        if(renders[index]->need_update)
            update_list.push_back(index);

    if(!update_list.empty())
    {
        TractRenderShader shader(cur_tracking_window);
        tipl::par_for(update_list.size(),[&](size_t index)
        {
            renders[update_list[index]]->prepare_update(cur_tracking_window,tracks[update_list[index]],shader);
        },update_list.size());
    }

    for(size_t index = 0;index < TractRender::data_block_count;++index)
    {
        for(auto each : renders)
            if(!each->render_tracts(index,glwidget,end_time))
                return false;
    }
    return true;
}
bool TractTableWidget::render_tracts(GLWidget* glwidget)
{
    if(!render_tracts(glwidget,std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(render_time)))
    {
        render_time *= 2;
        emit show_tracts();
        return false;
    }
    render_time = 200;
    return true;
}

bool TractTableWidget::command(std::vector<std::string> cmd)
{
    auto run = cur_tracking_window.history.record(error_msg,cmd);
    if(cmd.size() < 3)
        cmd.resize(3);

    auto get_cur_row = [&](std::string& cmd_text,int cur_row)->bool
    {
        if (tract_models.empty())
        {
            error_msg = "no tract available";
            return false;
        }
        bool okay = true;
        if(cmd_text.empty())
            cmd_text = std::to_string(cur_row);
        else
            cur_row = QString::fromStdString(cmd_text).toInt(&okay);
        if (cur_row >= tract_models.size() || !okay)
        {
            error_msg = "invalid tract index: " + cmd_text;
            return false;
        }
        return true;
    };

    if(cmd[0] == "delete_branch")
    {
        for_each_bundle(cmd[0].c_str(), [&](unsigned int index){return tract_models[index]->delete_branch();});
        return true;
    }
    if(cmd[0] == "undo_tract")
    {
        for_each_bundle(cmd[0].c_str(),[&](unsigned int index){return tract_models[index]->undo();});
        return true;
    }
    if(cmd[0] == "redo_tract")
    {
        for_each_bundle(cmd[0].c_str(),[&](unsigned int index){return tract_models[index]->redo();});
        return true;
    }
    if(cmd[0] == "trim_tract")
    {
        for_each_bundle(cmd[0].c_str(),[&](unsigned int index){return tract_models[index]->trim();});
        return true;
    }

    if(cmd[0] == "cut_tract_end_portion")
    {
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        for_current_bundle([&](void){tract_models[cur_row]->cut_end_portion(0.25f,0.75f);});
        return true;
    }
    if(tipl::begins_with(cmd[0],"flip_tract_"))
    {
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        for_current_bundle([&](void){tract_models[cur_row]->flip(cmd[0].back()-'x');});
        return true;
    }
    if(tipl::begins_with(cmd[0],"cut_tract_by_"))
    {
        bool other_side = cmd[0].back() == '2';
        if(other_side)
            cmd[0].pop_back();
        cut_by_slice(cmd[0].back()-'x',!other_side);
        return true;
    }

    if(cmd[0] == "set_dt_index")
    {
        auto pos = cmd[1].find('&');
        if(pos == std::string::npos)
            return run->failed("invalid dt index");
        if(!cur_tracking_window.handle->set_dt_index(std::make_pair(cmd[1].substr(0, pos), cmd[1].substr(pos + 1)),QString(cmd[2].c_str()).toInt()))
            return run->failed(cur_tracking_window.handle->error_msg);
        // turn off auto_tracks
        cur_tracking_window.ui->tract_target_0->setCurrentIndex(0);
        return true;
    }
    if(cmd[0] == "run_tracking")
    {
        // cmd[1]: tract name
        // cmd[2]: id
        // cmd[3]: tolerance distance
        if(!cur_tracking_window.handle->trackable)
            return run->failed("the data are not trackable");
        auto new_thread = std::make_shared<ThreadData>(cur_tracking_window.handle);
        if(!new_thread->param.set_code(cmd[2]))
            return run->failed("invalid parameter id");
        if(cmd.size() == 4) // has auto track
        {
            new_thread->roi_mgr->use_auto_track = true;
            new_thread->roi_mgr->tract_name = cmd[1];
            new_thread->roi_mgr->tolerance_dis_in_icbm152_mm = QString(cmd[3].c_str()).toFloat();
        }
        addNewTracts(cmd[1].c_str());
        thread_data.back() = new_thread;
        cur_tracking_window.regionWidget->setROIs(new_thread.get());
        tipl::progress prog("initiating fiber tracking");
        new_thread->run(cur_tracking_window.ui->thread_count->value(),false);
        tract_models.back()->report = cur_tracking_window.handle->report;
        tract_models.back()->report += thread_data.back()->report.str();
        show_report();
        timer->start(500);
        timer_update->start(100);
        cur_tracking_window.history.has_other_thread = true;
        cur_tracking_window.history.default_stem2 = cmd[1];
        return true;
    }
    if(cmd[0] == "open_tract" || cmd[0] == "open_mni_tract")
    {
        // cmd[1] : file name
        // cmd[2] : empty() = show or otherwise no show
        if(!cmd[1].empty())
        {
            bool is_mni_space = (cmd[0] == "open_mni_tract");
            if(is_mni_space && !cur_tracking_window.handle->map_to_mni())
                return run->failed(cur_tracking_window.handle->error_msg);

            auto models = TractModel::load_from_file(cmd[1].c_str(),cur_tracking_window.handle,is_mni_space);
            if(models.empty())
                return run->failed("cannot load tracks from " + cmd[1]);
            for(auto& each : models)
                if(each.get())
                    addNewTracts(each,cmd[2].empty());

            return true;
        }
        // allow for selecting multiple files
        auto file_list = QFileDialog::getOpenFileNames(this,QString::fromStdString(cmd[0]),
                         QString::fromStdString(cur_tracking_window.history.file_stem()) + output_format(),
                         "Tract files (*tt.gz *.trk *trk.gz *.tck);;Text files (*.txt);;All files (*)");
        if(file_list.isEmpty())
            return run->canceled();
        // allow sub command to be recorded
        --cur_tracking_window.history.current_recording_instance;
        for(auto each : file_list)
            if(!command({cmd[0],each.toStdString(),file_list.size() <= 7 ? "" : "0"}))
                break;
        ++cur_tracking_window.history.current_recording_instance;
        if(!error_msg.empty())
            return false;
        return run->canceled();
    }
    if(cmd[0] == "save_tract")
    {
        // cmd[1] : file name to be saved
        // cmd[2] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row) ||
           !cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name + output_format().toStdString()))
            return run->canceled();

        tipl::progress prog_(cmd[0]);
        auto lock = tract_rendering[cur_row]->start_reading();
        if(!tract_models[cur_row]->save_tracts_to_file(cmd[1].c_str()))
            return run->failed("cannot write to file at " + cmd[1]);
        return true;
    }
    if(cmd[0] == "save_all_tracts_to_folder")
    {
        // cmd[1] : directory output
        if(tract_models.empty() || !cur_tracking_window.history.get_dir(this,cmd[1]))
            return run->canceled();
        tipl::progress prog_("saving files");
        auto selected_tracts = get_checked_tracks();
        auto selected_tracts_rendering = get_checked_tracks_rendering();
        for(size_t index = 0;index < selected_tracts.size();++index)
        {
            auto filename = cmd[1] + "/" + selected_tracts[index]->name + output_format().toStdString();
            auto lock = selected_tracts_rendering[index]->start_reading();
            if(!selected_tracts[index]->save_tracts_to_file(filename.c_str()))
                return run->failed("cannot save file due to permission error" + filename);
        }
        return true;
    }
    if(cmd[0] == "save_all_tracts")
    {
        if(tract_models.empty() || !cur_tracking_window.history.get_filename(this,cmd[1],output_format().toStdString()))
            return run->canceled();

        auto locks = start_reading_checked_tracks();
        if(!TractModel::save_all(cmd[1].c_str(),get_checked_tracks()))
            return run->failed("cannot save file to " + cmd[1]);
        return true;
    }
    if(cmd[0] == "update_track")
    {
        for(int index = 0;index < int(tract_models.size());++index)
        {
            item(int(index),1)->setText(QString::number(tract_models[index]->get_visible_track_count()));
            item(int(index),2)->setText(QString::number(tract_models[index]->get_deleted_track_count()));
        }
        for(auto& t:tract_rendering)
            t->need_update = true;
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "cut_by_slice")
    {
        cut_by_slice(QString(cmd[1].c_str()).toInt(),QString(cmd[2].c_str()).toInt());
        return true;
    }
    if(cmd[0] == "delete_all_tracts")
    {
        if(tipl::progress::is_running())
            return run->failed("please wait for the termination of data processing");
        setRowCount(0);
        while(!tract_rendering.empty())
        {
            tract_rendering.pop_back();
            thread_data.pop_back();
            tract_models.pop_back();
        }
        emit show_tracts();
        return true;
    }

    if(cmd[0] == "load_tract_color" || cmd[0] == "load_tract_values")
    {
        // cmd[1] : file name
        // cmd[2] : current tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row) || !cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name))
            return run->canceled();
        if(cmd[0] == "load_tract_color")
        {
            auto lock = tract_rendering[cur_row]->start_reading();
            if(!tract_models[cur_row]->load_tracts_color_from_file(cmd[1].c_str()))
                return run->failed("cannot find or open " + cmd[1]);
            tract_rendering[cur_row]->need_update = true;
            cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        }
        else
        {
            std::ifstream in(cmd[1]);
            if(!in)
                return run->failed("cannot find or open " + cmd[1]);
            std::vector<float> values;
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),
                      std::back_inserter(values));
            if(tract_models[cur_row]->get_visible_track_count() != values.size())
                return run->failed("the number of values " + std::to_string(values.size()) +
                                   " does not match current tract count " +
                                   std::to_string(tract_models[cur_row]->get_visible_track_count()));
            auto lock = tract_rendering[cur_row]->start_reading();
            tract_models[cur_row]->loaded_values.swap(values);
            tract_rendering[cur_row]->need_update = true;
            cur_tracking_window.set_data("tract_color_style",6);//loaded values
        }
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "load_cluster_color" || cmd[0] == "load_cluster_values" || cmd[0] == "save_cluster_color")
    {
        if(tract_models.empty() || !cur_tracking_window.history.get_filename(this,
                                            cmd[1],cur_tracking_window.history.default_stem))
            return run->canceled();

        if(cmd[0] == "save_cluster_color")
        {
            std::ofstream out(cmd[1]);
            if(!out)
                return run->failed("cannot write to " + cmd[1]);
            for(unsigned int index = 0;index < tract_models.size();++index)
                if(item(int(index),0)->checkState() == Qt::Checked)
                {
                    auto lock = tract_rendering[index]->start_reading(true);
                    if(tract_models[index]->get_visible_track_count())
                    {
                        tipl::rgb color = tract_models[index]->get_tract_color(0);
                        out << int(color.r) << " " << int(color.g) << " " << int(color.b) << std::endl;
                    }
                    else
                        out << "0 0 0" << std::endl;
                }
            return true;
        }
        std::ifstream in(cmd[1]);
        if(!in)
            return run->failed("cannot find or open " + cmd[1]);
        if(cmd[0] == "load_cluster_color")
        {
            for(unsigned int index = 0;index < tract_models.size() && in;++index)
                if(item(int(index),0)->checkState() == Qt::Checked)
                {
                    int r(0),g(0),b(0);
                    in >> r >> g >> b;
                    auto lock = tract_rendering[index]->start_writing();
                    tract_models[index]->set_color(tipl::rgb(r,g,b));
                    tract_rendering[index]->need_update = true;
                }
            cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        }
        else
        {
            std::vector<float> values;
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),
                      std::back_inserter(values));
            auto checked_track = get_checked_tracks();
            if(checked_track.size() != values.size())
                return run->failed("the number of values " + std::to_string(values.size()) +
                                   " does not match bundle count " + std::to_string(checked_track.size()));
            tipl::out() << "assign values to each bundle";
            for(unsigned int index = 0,pos = 0;index < tract_models.size() && pos < values.size();++index)
                if(item(int(index),0)->checkState() == Qt::Checked)
                {
                    tract_models[index]->loaded_value = values[pos];
                    tract_rendering[index]->need_update = true;
                    ++pos;
                }
            cur_tracking_window.set_data("tract_color_style",6);//loaded values
        }
        emit show_tracts();
        return true;
    }
    return run->not_processed();
}

void TractTableWidget::save_tracts_data_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    QString filename = QFileDialog::getSaveFileName(
                this,"Save as",item(currentRow(),0)->text() + "_" + action->data().toString() + ".txt",
                "Text files (*.txt);;MATLAB file (*.mat);;TRK file (*.trk *.trk.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(!tract_models[uint32_t(currentRow())]->save_data_to_file(
                    cur_tracking_window.handle,filename.toStdString().c_str(),
                    action->data().toString().toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR","fail to save information");
    }
    else
        QMessageBox::information(this,QApplication::applicationName(),"file saved");
}


void TractTableWidget::merge_all(void)
{
    std::vector<unsigned int> merge_list;
    for(int index = 0;index < tract_models.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            merge_list.push_back(index);
    if(merge_list.size() <= 1)
        return;
    {
        auto lock1 = tract_rendering[merge_list[0]]->start_writing();
        for(int index = merge_list.size()-1;index >= 1;--index)
        {
            {
                auto lock2 = tract_rendering[merge_list[index]]->start_reading();
                tract_models[merge_list[0]]->add(*tract_models[merge_list[index]]);
            }
            delete_row(merge_list[index]);
        }
        tract_rendering[merge_list[0]]->need_update = true;
    }
    item(merge_list[0],1)->setText(QString::number(tract_models[merge_list[0]]->get_visible_track_count()));
    item(merge_list[0],2)->setText(QString::number(tract_models[merge_list[0]]->get_deleted_track_count()));
    emit show_tracts();
}

void TractTableWidget::delete_row(int row)
{
    if(row >= tract_models.size())
        return;
    tract_rendering.erase(tract_rendering.begin()+row);
    thread_data.erase(thread_data.begin()+row);
    tract_models.erase(tract_models.begin()+row);
    removeRow(row);
    emit show_tracts();
}

void TractTableWidget::copy_track(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    uint32_t old_row = uint32_t(currentRow());
    addNewTracts(item(currentRow(),0)->text() + "_copy");
    *(tract_models.back()) = *(tract_models[old_row]);
    item(currentRow(),1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    emit show_tracts();
}

void TractTableWidget::separate_deleted_track(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    std::vector<std::vector<float> > new_tracks;
    new_tracks.swap(tract_models[uint32_t(currentRow())]->get_deleted_tracts());
    if(new_tracks.empty())
        return;
    // clean the deleted tracks
    tract_models[uint32_t(currentRow())]->clear_deleted();
    item(currentRow(),1)->setText(QString::number(tract_models[uint32_t(currentRow())]->get_visible_track_count()));
    item(currentRow(),2)->setText(QString::number(tract_models[uint32_t(currentRow())]->get_deleted_track_count()));
    // add deleted tracks to a new entry
    addNewTracts(item(currentRow(),0)->text(),false);
    tract_models.back()->add_tracts(new_tracks);
    tract_models.back()->report = tract_models[uint32_t(currentRow())]->report;
    item(rowCount()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
    item(rowCount()-1,2)->setText(QString::number(tract_models.back()->get_deleted_track_count()));
    emit show_tracts();
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
        std::swap(tract_rendering[i],tract_rendering[j]);
    }
    emit show_tracts();
}
void TractTableWidget::merge_track_by_name(void)
{
    for(int i= 0;i < rowCount()-1;++i)
    {
        auto lock1 = tract_rendering[i]->start_writing();
        for(int j= i+1;j < rowCount()-1;)
            if(item(i,0)->text() == item(j,0)->text())
            {
                {
                    auto lock2 = tract_rendering[j]->start_reading();
                    tract_models[i]->add(*tract_models[j]);
                }
                tract_rendering[i]->need_update = true;
                delete_row(j);
                item(i,1)->setText(QString::number(tract_models[i]->get_visible_track_count()));
                item(i,2)->setText(QString::number(tract_models[i]->get_deleted_track_count()));
            }
        else
            ++j;
    }
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
        std::swap(thread_data[uint32_t(currentRow())],thread_data[currentRow()-1]);
        std::swap(tract_models[uint32_t(currentRow())],tract_models[currentRow()-1]);
        std::swap(tract_rendering[uint32_t(currentRow())],tract_rendering[currentRow()-1]);
        setCurrentCell(currentRow()-1,0);
    }
    emit show_tracts();
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
        std::swap(thread_data[uint32_t(currentRow())],thread_data[currentRow()+1]);
        std::swap(tract_models[uint32_t(currentRow())],tract_models[currentRow()+1]);
        std::swap(tract_rendering[uint32_t(currentRow())],tract_rendering[currentRow()+1]);
        setCurrentCell(currentRow()+1,0);
    }
    emit show_tracts();
}


void TractTableWidget::delete_tract(void)
{
    if(tipl::progress::is_running())
    {
        QMessageBox::critical(this,"ERROR","Please wait for the termination of data processing");
        return;
    }
    delete_row(currentRow());
    emit show_tracts();
}
void TractTableWidget::delete_repeated(void)
{
    float distance = 1.0;
    bool ok;
    distance = QInputDialog::getDouble(this,
        QApplication::applicationName(),"Distance threshold (voxels)", distance,0,500,1,&ok);
    if (!ok)
        return;
    for_each_bundle("delete repeated",[&](unsigned int index)
    {
        return tract_models[index]->delete_repeated(distance);
    });
}

void TractTableWidget::resample_step_size(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    float new_step = 0.5f;
    bool ok;
    new_step = float(QInputDialog::getDouble(this,
        QApplication::applicationName(),"New step size (voxels)",double(new_step),0.0,5.0,1,&ok));
    if (!ok)
        return;
    for_each_bundle("resample tracks",[&](unsigned int index)
    {
        tract_models[index]->resample(new_step);
        return true;
    });
}

void TractTableWidget::delete_by_length(void)
{
    tipl::progress prog_("filtering tracks");

    float threshold = 60;
    bool ok;
    threshold = QInputDialog::getDouble(this,
        QApplication::applicationName(),"Length threshold in mm:", threshold,0,500,1,&ok);
    if (!ok)
        return;
    for_each_bundle("delete by length",[&](unsigned int index)
    {
        return tract_models[index]->delete_by_length(threshold);
    });
}
void TractTableWidget::reconnect_track(void)
{
    for_current_bundle([&](void)
    {
        bool ok;
        QString result = QInputDialog::getText(this,QApplication::applicationName(),"Assign maximum bridging distance (in voxels) and angles (degrees)",
                                               QLineEdit::Normal,"4 30",&ok);

        if(!ok)
            return;
        std::istringstream in(result.toStdString());
        float dis,angle;
        in >> dis >> angle;
        if(dis <= 2.0f || angle <= 0.0f)
            return;
        tract_models[currentRow()]->reconnect_track(dis,std::cos(angle*3.14159265358979323846f/180.0f));
    });
}
void TractTableWidget::edit_tracts(void)
{
    QRgb color = 0;
    if(edit_option == paint)
        color = QColorDialog::getColor(Qt::red,this,"Select color",QColorDialog::ShowAlphaChannel).rgb();
    auto angle = cur_tracking_window.glWidget->angular_selection ? cur_tracking_window["tract_sel_angle"].toFloat():0.0;
    for_each_bundle("editing tracts",[&](unsigned int index)
    {
        switch(edit_option)
        {
        case select:
        case del:
            return tract_models[index]->cull(angle,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos,edit_option == del);
            break;
        case cut:
            return tract_models[index]->cut(angle,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos);
            break;
        case paint:
            return tract_models[index]->paint(angle,
                             cur_tracking_window.glWidget->dirs,
                             cur_tracking_window.glWidget->pos,color);
            break;
        default:
            ;
        }
        return false;
    },true);
    if(edit_option == paint)
        cur_tracking_window.set_data("tract_color_style",1);//manual assigned

}

void TractTableWidget::cut_by_slice(unsigned char dim,bool greater)
{
    for_each_bundle("cut by slice",[&](unsigned int index)
    {
        tract_models[index]->cut_by_slice(dim,cur_tracking_window.current_slice->slice_pos[dim],greater,
            (cur_tracking_window.current_slice->is_diffusion_space ? nullptr:&cur_tracking_window.current_slice->to_slice));
        return true;
    });
}

void TractTableWidget::export_tract_density(tipl::shape<3> dim,
                                            tipl::vector<3,float> vs,
                                            const tipl::matrix<4,4>& trans_to_mni,
                                            const tipl::matrix<4,4>& T,
                                            bool color,bool end_point)
{
    QString filename;
    if(color)
    {
        filename = QFileDialog::getSaveFileName(
                this,"Save Images files",item(currentRow(),0)->text()+".nii.gz",
                "Image files (*.png *.bmp *nii.gz *.nii *.jpg *.tif);;All files (*)");
        if(filename.isEmpty())
            return;
    }
    else
    {
        filename = QFileDialog::getSaveFileName(
                    this,"Save as",item(currentRow(),0)->text()+".nii.gz",
                    "NIFTI files (*nii.gz *.nii);;MAT File (*.mat);;");
        if(filename.isEmpty())
            return;
    }
    if(TractModel::export_tdi(filename.toStdString().c_str(),get_checked_tracks(),dim,vs,trans_to_mni,T,color,end_point))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR","Failed to save file");
}


