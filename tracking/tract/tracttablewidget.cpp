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
            if(!command({"load_tract_atlas",tract_name.toStdString()}))
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
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
                item(int(index),0)->setCheckState(Qt::Checked);
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

void TractTableWidget::cell_changed(int row, int column)
{
    if(row >= 0 && row < tract_models.size())
        tract_models[row]->name = item(int(row),0)->text().toStdString();
    if(cur_tracking_window["show_track_label"].toInt())
        cur_tracking_window.glWidget->update();

}
void get_track_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          std::string& result);
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

    auto get_cur_row = [&](std::string& cmd_text,int& cur_row)->bool
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
        // cmd [1] : slice number
        bool other_side = cmd[0].back() == '2';
        if(other_side)
            cmd[0].pop_back();
        auto dim = cmd[0].back()-'x';
        unsigned int slice_pos = run->from_cmd(1,cur_tracking_window.current_slice->slice_pos[dim]);
        for_each_bundle(cmd[0].c_str(),[&](unsigned int index)
        {
            tract_models[index]->cut_by_slice(dim,slice_pos,!other_side,
                (cur_tracking_window.current_slice->is_diffusion_space ? nullptr:&cur_tracking_window.current_slice->to_slice));
            return true;
        });
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
        addNewTracts(cmd[1].c_str(),false);
        thread_data.back() = new_thread;
        cur_tracking_window.regionWidget->setROIs(new_thread->roi_mgr);
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
    if(cmd[0] == "open_tract_name")
    {
        // cmd[1] : file name
        if(!cur_tracking_window.history.get_filename(this,cmd[1]))
            return run->canceled();
        std::ifstream in(cmd[1].c_str());
        std::vector<std::string> name((std::istream_iterator<std::string>(in)),(std::istream_iterator<std::string>()));
        for(int i = 0;i < rowCount() && i < name.size();++i)
            item(rowCount()-1-i,0)->setText(name[name.size()-1-i].c_str());
        return true;
    }
    if(cmd[0] == "load_tract_atlas")
    {
        // cmd[1] : name of tract or all (empty)
        auto load_tract_atlas = [&](const std::string& tract_name)->bool
        {
            auto track_ids = cur_tracking_window.handle->get_track_ids(tract_name);
            if(track_ids.empty())
                return run->failed("cannot find a matched tract for " + tract_name);
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
            return true;
        };

        if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
            return run->failed(cur_tracking_window.handle->error_msg);

        if(cmd[1].empty()) // load all
        {
            for(const auto& each : cur_tracking_window.handle->tractography_name_list)
                load_tract_atlas(each);
            return true;
        }
        else
            return load_tract_atlas(cmd[1]);
    }
    if(cmd[0] == "save_tract" || cmd[0] == "save_mni_tract" || cmd[0] == "save_template_tract" || cmd[0] == "save_slice_tract")
    {
        std::string post_fix;
        if(cmd[0] == "save_slice_tract")
            post_fix = "_"+cur_tracking_window.current_slice->get_name();
        if(cmd[0] == "save_mni_tract")
            post_fix = "_mni";
        if(cmd[0] == "save_template_tract")
            post_fix = "_template";
        // cmd[1] : file name to be saved
        // cmd[2] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name + post_fix + output_format().toStdString()))
            return run->canceled();

        tipl::progress prog(cmd[0]);
        auto lock = tract_rendering[cur_row]->start_reading();
        if(cmd[0] == "save_slice_tract")
        {
            if(!tract_models[cur_row]->save_transformed_tract(cmd[1].c_str(),
                    cur_tracking_window.current_slice->dim,
                    cur_tracking_window.current_slice->vs,
                    cur_tracking_window.current_slice->trans_to_mni,
                    cur_tracking_window.current_slice->to_slice,false/*not endpoint*/))
                return run->failed("cannot write to file at " + cmd[1]);
        }
        else
        if(cmd[0] == "save_template_tract" || cmd[0] == "save_mni_tract")
        {
            if(!cur_tracking_window.handle->map_to_mni())
                return run->failed(cur_tracking_window.handle->error_msg);
            if(!tract_models[cur_row]->save_tracts_in_template_space(cur_tracking_window.handle,cmd[1].c_str(),cmd[0] == "save_mni_tract"))
                return run->failed("cannot write to file at " + cmd[1]);
        }
        else
            if(!tract_models[cur_row]->save_tracts_to_file(cmd[1].c_str()))
                return run->failed("cannot write to file at " + cmd[1]);
        return true;
    }
    if(cmd[0] == "save_tract_endpoint" || cmd[0] == "save_slice_tract_endpoint" || cmd[0] == "save_mni_tract_endpoint")
    {
        // cmd[1] : file name to be saved
        // cmd[2] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name))
            return run->canceled();
        auto lock = tract_rendering[cur_row]->start_reading();

        if(cmd[0] == "save_slice_tract_endpoint")
        {
            if(!tract_models[cur_row]->save_transformed_tract(cmd[1].c_str(),
                    cur_tracking_window.current_slice->dim,
                    cur_tracking_window.current_slice->vs,
                    cur_tracking_window.current_slice->trans_to_mni,
                    cur_tracking_window.current_slice->to_slice,true))
                return run->failed("cannot write to file at " + cmd[1]);
        }
        else
        if(cmd[0] == "save_mni_tract_endpoint")
        {
            if(!cur_tracking_window.handle->map_to_mni())
                return run->failed(cur_tracking_window.handle->error_msg);
            std::vector<tipl::vector<3,short> > points1,points2;
            tract_models[cur_row]->to_end_point_voxels(points1,points2);
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

            if (tipl::ends_with(cmd[1],".txt"))
            {
                std::ofstream out(cmd[1].c_str(),std::ios::out);
                if (!out)
                    return run->failed("cannot write to file at " + cmd[1]);
                std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(out," "));
            }
            if (tipl::ends_with(cmd[1],".mat"))
            {
                tipl::io::mat_write out(cmd[1].c_str());
                if(!out)
                    return run->failed("cannot write to file at " + cmd[1]);
                out.write("end_points",buffer,3);
            }
        }
            if(!tract_models[cur_row]->save_end_points(cmd[1].c_str()))
                return run->failed("cannot write to file at " + cmd[1]);
        return true;
    }

    if(cmd[0] == "save_tract_values")
    {
        // cmd[1] : file name to be saved
        // cmd[2] : tract index
        // cmd[3] : metrics name
        if(cmd.size() <= 3)
        {
            cmd.resize(4);
            if(cmd[3].empty() && (cmd[3] = cur_tracking_window.get_action_data().toStdString()).empty())
                return run->canceled();
        }
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name + "_" + cmd[3]))
            return run->canceled();
        auto lock = tract_rendering[cur_row]->start_reading();
        if(!tract_models[cur_row]->save_data_to_file(cur_tracking_window.handle,cmd[1].c_str(),cmd[3].c_str()))
            return run->failed("fail to save " + cmd[3] + " from " + tract_models[cur_row]->name);
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
    if(cmd[0] == "tract_to_region")
    {
        // cmd[1] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        std::vector<tipl::vector<3,short> > points;
        tract_models[cur_row]->to_voxel(points,
            cur_tracking_window.current_slice->is_diffusion_space ? tipl::matrix<4,4>(tipl::identity_matrix()) :
            cur_tracking_window.current_slice->to_slice);
        cur_tracking_window.regionWidget->add_region(item(cur_row,0)->text());
        cur_tracking_window.regionWidget->regions.back()->add_points(std::move(points));
        cur_tracking_window.slice_need_update = true;
        cur_tracking_window.glWidget->update();
        return true;
    }
    if(cmd[0] == "endpoint_to_region")
    {
        // cmd[1] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        std::vector<tipl::vector<3,short> > points1,points2;
        tract_models[cur_row]->to_end_point_voxels(points1,points2,
                    cur_tracking_window.current_slice->is_diffusion_space ?
                        tipl::matrix<4,4>(tipl::identity_matrix()) :
                        cur_tracking_window.current_slice->to_slice);

        cur_tracking_window.regionWidget->begin_update();
        cur_tracking_window.regionWidget->add_region(item(cur_row,0)->text()+QString(" endpoints1"));
        cur_tracking_window.regionWidget->regions.back()->add_points(std::move(points1));
        cur_tracking_window.regionWidget->add_region(item(cur_row,0)->text()+QString(" endpoints2"));
        cur_tracking_window.regionWidget->regions.back()->add_points(std::move(points2));
        cur_tracking_window.regionWidget->end_update();
        cur_tracking_window.slice_need_update = true;
        cur_tracking_window.glWidget->update();
        return true;
    }
    if(cmd[0] == "update_tract")
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
    if(cmd[0] == "filter_tract")
    {
        std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(cur_tracking_window.handle));
        cur_tracking_window.regionWidget->setROIs(roi_mgr);
        for_each_bundle("filter by roi",[&](unsigned int index)
        {
            return tract_models[index]->filter_by_roi(roi_mgr);
        });
        return true;
    }
    if(cmd[0] == "copy_tract")
    {
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        addNewTracts(item(cur_row,0)->text() + "_copy");
        *(tract_models.back()) = *(tract_models[cur_row]);
        item(rowCount()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "delete_tract")
    {
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        delete_row(cur_row);
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "delete_all_tracts")
    {
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
    if(cmd[0] == "load_tract_color" || cmd[0] == "load_tract_values" || cmd[0] == "save_tract_color")
    {
        // cmd[1] : file name
        // cmd[2] : current tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],tract_models[cur_row]->name))
            return run->canceled();

        if(cmd[0] == "save_tract_color")
        {
            auto lock = tract_rendering[cur_row]->start_reading();
            tract_models[cur_row]->save_tracts_color_to_file(cmd[1].c_str());
            return true;
        }

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
    if(cmd[0] == "select_cluster_color")
    {
        // cmd[1] : region index
        // cmd[2] : rgb color
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        QColor color;
        if(cmd[2].empty())
        {
            color = QColorDialog::getColor(tract_models[cur_row]->get_tract_color(0),
                                        this,QString::fromStdString(cmd[0]),QColorDialog::ShowAlphaChannel);
            if(!color.isValid())
                return run->canceled();
            cmd[2] = std::to_string(color.rgb());
        }
        else
            color = QColor::fromRgba(QString::fromStdString(cmd[2]).toLongLong());
        tract_models[cur_row]->set_color(color.rgb());
        tract_rendering[cur_row]->need_update = true;
        cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "color_all_cluster")
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
        return true;
    }

    if(tipl::contains(cmd[0],"cluster_tract"))
    {
        // cmd[1] : tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;

        std::vector<unsigned int> labels;
        std::vector<std::string> names;
        {
            auto lock = tract_rendering[cur_row]->start_reading();
            if(cmd[0] == "cluster_tract_by_label")
            {
                // cmd[2] : file_name
                if(!cur_tracking_window.history.get_filename(this,cmd[2],tract_models[cur_row]->name))
                    return run->canceled();
                std::ifstream in(cmd[2].c_str());
                if(!in)
                    return run->failed("cannot read tract label " + cmd[2]);
                std::copy(std::istream_iterator<unsigned int>(in),
                          std::istream_iterator<unsigned int>(),std::back_inserter(labels));
                labels.resize(tract_models[cur_row]->get_visible_track_count());
            }
            else
            if(cmd[0] == "recognize_and_cluster_tract")
            {
                if(!cur_tracking_window.handle->recognize(tract_models[cur_row],labels,names))
                    return run->failed(cur_tracking_window.handle->error_msg);
            }
            else
            {
                // cmd[2] : method_id, number of cluster, details
                int method_id = 0,n = 10;
                double detail = 0.0f;
                if(cmd[0] == "cluster_tract_by_hy")
                    method_id = 0;
                if(cmd[0] == "cluster_tract_by_km")
                    method_id = 1;
                if(cmd[0] == "cluster_tract_by_em")
                    method_id = 2;
                if(cmd[2].empty())
                {
                    bool ok = false;
                    n = QInputDialog::getInt(this,QApplication::applicationName(),"assign the maximum number of groups",50,1,5000,10,&ok);
                    if(!ok)
                        return run->canceled();
                    if(method_id == 0)
                        detail = QInputDialog::getDouble(this,
                            QApplication::applicationName(),"clustering detail (mm):",cur_tracking_window.handle->vs[0],0.2,50.0,2,&ok);
                    if(!ok)
                        return run->canceled();
                    cmd[2] = std::to_string(n) + " " + std::to_string(detail);
                }
                else
                {
                    std::istringstream in(cmd[2]);
                    in >> n >> detail;
                }
                tract_models[cur_row]->run_clustering(method_id,n,detail);
                labels = tract_models[cur_row]->tract_cluster;
            }
        }

        std::vector<std::vector<float> > tracts;
        tract_models[cur_row]->release_tracts(tracts);
        tract_rendering[cur_row]->need_update = true;
        delete_row(cur_row);
        unsigned int cluster_count = uint32_t(names.empty() ? int(1+tipl::max_value(labels)):int(names.size()));
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
            if(int(cluster_index) < names.size())
                addNewTracts(QString::fromStdString(names[int(cluster_index)]),true);
            else
                addNewTracts(QString("cluster")+QString::number(cluster_index),true);
            tract_models.back()->add_tracts(add_tracts);
            tract_models.back()->report = tract_models[cur_row]->report;
            tract_models.back()->geo = tract_models[cur_row]->geo;
            tract_models.back()->vs = tract_models[cur_row]->vs;
            tract_models.back()->trans_to_mni = tract_models[cur_row]->trans_to_mni;
            item(int(tract_models.size())-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        }
        if(cmd[0] == "cluster_tract_by_hy")
        {
            std::vector<unsigned int> track_to_delete(tract_models.back()->get_visible_track_count());
            std::iota(track_to_delete.begin(), track_to_delete.end(), 0);
            tract_models.back()->delete_tracts(track_to_delete);
            tract_models.back()->name = "others";
            item(int(tract_models.size())-1,0)->setText("others");
            item(int(tract_models.size())-1,1)->setText("0");
            item(int(tract_models.size())-1,2)->setText(QString::number(track_to_delete.size()));
        }
        return command({"color_all_cluster"});
    }

    if(cmd[0] == "delete_repeated_tract")
    {
        // cmd[1] : distance
        float distance = 1.0;
        if(cmd[1].empty())
        {
            bool ok;
            distance = QInputDialog::getDouble(this,QApplication::applicationName(),
                                               "Distance threshold (voxels)", distance,0,500,1,&ok);
            if (!ok)
                return run->canceled();
        }
        else
            distance = QString::fromStdString(cmd[1]).toFloat();

        for_each_bundle(cmd[0].c_str(),[&](unsigned int index)
        {
            return tract_models[index]->delete_repeated(distance);
        });
        emit show_tracts();
        return true;
    }

    if(cmd[0] == "resample_tract")
    {

        // cmd[1] : new_step
        float new_step = 0.5;
        if(cmd[1].empty())
        {
            bool ok;
            new_step = float(QInputDialog::getDouble(this,QApplication::applicationName(),
                                                     "New step size (voxels)",double(new_step),0.0,5.0,1,&ok));
            if (!ok)
                return run->canceled();
        }
        else
            new_step = QString::fromStdString(cmd[1]).toFloat();

        for_each_bundle(cmd[0].c_str(),[&](unsigned int index)
        {
            tract_models[index]->resample(new_step);
            return true;
        });
        emit show_tracts();
        return true;
    }

    if(cmd[0] == "delete_tract_by_length")
    {
        // cmd[1] : new_step
        float threshold = 0.5;
        if(cmd[1].empty())
        {
            bool ok;
            threshold = QInputDialog::getDouble(this,
                    QApplication::applicationName(),"Length threshold in mm:", threshold,0,500,1,&ok);
            if (!ok)
                return run->canceled();
        }
        else
            threshold = QString::fromStdString(cmd[1]).toFloat();

        for_each_bundle(cmd[0].c_str(),[&](unsigned int index)
        {
            return tract_models[index]->delete_by_length(threshold);
        });
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "separate_deleted_tract")
    {
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        std::vector<std::vector<float> > new_tracks;
        new_tracks.swap(tract_models[cur_row]->get_deleted_tracts());
        if(new_tracks.empty())
            return run->canceled();
        // clean the deleted tracks
        tract_models[cur_row]->clear_deleted();
        item(cur_row,1)->setText(QString::number(tract_models[cur_row]->get_visible_track_count()));
        item(cur_row,2)->setText(QString::number(tract_models[cur_row]->get_deleted_track_count()));
        // add deleted tracks to a new entry
        addNewTracts(item(cur_row,0)->text(),false);
        tract_models.back()->add_tracts(new_tracks);
        tract_models.back()->report = tract_models[cur_row]->report;
        item(rowCount()-1,1)->setText(QString::number(tract_models.back()->get_visible_track_count()));
        item(rowCount()-1,2)->setText(QString::number(tract_models.back()->get_deleted_track_count()));
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "reconnect_tract")
    {
        // cmd[1] : tract id
        // cmd[2] = maximum bridging distance (in voxels) and angles
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;

        if(cmd[2].empty())
        {
            bool ok;
            cmd[2] = QInputDialog::getText(this,QApplication::applicationName(),"Assign maximum bridging distance (in voxels) and angles (degrees)",
                                                           QLineEdit::Normal,"4 30",&ok).toStdString();
            if (!ok)
                return run->canceled();
        }

        float dis(0.0f),angle(0.0f);
        std::istringstream(cmd[2]) >> dis >> angle;
        if(dis <= 2.0f || angle <= 0.0f)
            return run->failed("invalid distance and angles" + cmd[2]);
        tract_models[cur_row]->reconnect_tract(dis,std::cos(angle*3.14159265358979323846f/180.0f));
        return true;
    }

    if(cmd[0] == "recognize_tract")
    {
        // cmd[1] : current tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
            return run->failed(cur_tracking_window.handle->error_msg);
        std::multimap<float,std::string,std::greater<float> > sorted_list;
        {
            auto lock = tract_rendering[cur_row]->start_reading();
            if(!cur_tracking_window.handle->recognize_and_sort(tract_models[cur_row],sorted_list))
                return run->failed("cannot recognize tracks.");
        }
        std::ostringstream out;
        auto beg = sorted_list.begin();
        for(size_t i = 0;i < sorted_list.size();++i,++beg)
            if(beg->first != 0.0f)
                out << beg->first*100.0f << "% " << beg->second << std::endl;
        show_info_dialog("Tract Recognition Result",out.str(),cur_tracking_window.history.file_stem() + "_" +
                                            tract_models[cur_row]->name.c_str() + ".txt");
        return true;
    }
    if(cmd[0] == "recognize_and_rename_tract")
    {
        if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
            return run->failed(cur_tracking_window.handle->error_msg);
        tipl::progress prog("Recognize and rename");
        for(unsigned int index = 0;prog(index,tract_models.size());++index)
            if(item(int(index),0)->checkState() == Qt::Checked && tract_models[index]->get_visible_track_count())
            {
                std::multimap<float,std::string,std::greater<float> > sorted_list;
                auto lock = tract_rendering[index]->start_reading();
                if(!cur_tracking_window.handle->recognize_and_sort(tract_models[index],sorted_list))
                    return run->failed(cur_tracking_window.handle->error_msg);
                item(int(index),0)->setText(sorted_list.begin()->second.c_str());
            }
        return true;
    }
    if(cmd[0] == "merge_all_tracts")
    {
        std::vector<unsigned int> merge_list;
        for(int index = 0;index < tract_models.size();++index)
            if(item(int(index),0)->checkState() == Qt::Checked)
                merge_list.push_back(index);
        if(merge_list.size() <= 1)
            return run->canceled();
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
        return true;
    }
    if(cmd[0] == "merge_tract_by_name")
    {
        if(tract_models.empty())
            return run->canceled();
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
        emit show_tracts();
        return true;
    }
    if(cmd[0] == "sort_tract_by_name")
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
        return true;
    }
    if(cmd[0] == "save_tdi" || cmd[0] == "save_tdi2")
    {
        // cmd[1] : file name
        // cmd[2] : current tract index
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(
                    this,QString::fromStdString(cmd[0]),item(currentRow(),0)->text()+"_tdi.nii.gz",
                    "NIFTI files (*nii.gz *.nii);;All Files (*)").toStdString()).empty())
            return run->canceled();
        if(cmd[0] == "save_tdi2")
        {
            float ratio = 2.0f;
            tipl::matrix<4,4> tr((tipl::identity_matrix())),
                              inv_tr((tipl::identity_matrix())),
                              trans_to_mni(cur_tracking_window.handle->trans_to_mni);
            tr[0] = tr[5] = tr[10] = ratio;
            inv_tr[0] = inv_tr[5] = inv_tr[10] = 1.0f/ratio;
            trans_to_mni *= inv_tr;
            if(!TractModel::export_tdi(cmd[1].c_str(),
                        {tract_models[cur_row]},
                         cur_tracking_window.handle->dim*ratio,
                         cur_tracking_window.handle->vs/float(ratio),
                         trans_to_mni,tr,false,false))
                return run->failed("cannot save image to " + cmd[1]);
        }
        else
            if(!TractModel::export_tdi(cmd[1].c_str(),
                    {tract_models[cur_row]},
                     cur_tracking_window.current_slice->dim,
                     cur_tracking_window.current_slice->vs,
                     cur_tracking_window.current_slice->trans_to_mni,
                     cur_tracking_window.current_slice->to_slice,false,false))
            return run->failed("cannot save image to " + cmd[1]);
        return true;
    }
    if(cmd[0] == "check_uncheck_all_tract")
    {
        if(tract_models.empty())
            return run->canceled();
        // cmd[1] : all or none
        bool all = true;
        if(cmd[1].empty())
            cmd[1] = (all = (item(0,0)->checkState() != Qt::Checked)) ? "1":"0";
        else
            all = (cmd[1] == "1");
        for(int row = 0;row < rowCount();++row)
        {
            item(row,0)->setCheckState(all ? Qt::Checked : Qt::Unchecked);
            item(row,0)->setData(Qt::ForegroundRole,QBrush(all ? Qt::black : Qt::gray));
        }
        return true;
    }
    if(cmd[0] == "show_tract_statistics")
    {
        if(tract_models.empty())
            return run->canceled();
        std::string result;
        tipl::progress p("calculate tract statistics",true);
        get_track_statistics(cur_tracking_window.handle,get_checked_tracks(),result);
        show_info_dialog("Tract Statistics",result);
        return true;
    }
    return run->not_processed();
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



