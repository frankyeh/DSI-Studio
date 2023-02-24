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
#include "libs/gzip_interface.hpp"
#include "atlas.hpp"
#include "../color_bar_dialog.hpp"
#include "gzip_interface.hpp"


void show_info_dialog(const std::string& title,const std::string& result);

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
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<std::vector<std::vector<tipl::vector<2,float> > > > lines_threaded(thread_count);
    std::vector<std::vector<std::vector<unsigned int> > > colors_threaded(thread_count);
    tipl::matrix<4,4>* pt = (cur_tracking_window.current_slice->is_diffusion_space ? nullptr : &(cur_tracking_window.current_slice->invT));
    max_count /= selected_tracts.size();
    tipl::par_for(selected_tracts.size(),[&](unsigned int index,unsigned int thread)
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

    tipl::par_for(lines.size(),[&](unsigned int i)
    {
        auto& line = lines[i];
        auto& color = colors[i];
        tipl::add_constant(line,0.5f);
        tipl::multiply_constant(line,display_ratio);
        if(line.size() >= 2)
        {
            for(size_t j = 1;j < line.size();++j)
            {
                draw_line(int(line[j-1][0]),int(line[j-1][1]),int(line[j][0]),int(line[j][1]),color[j]);
            }
        }
    });
}

void TractTableWidget::addNewTracts(QString tract_name,bool checked)
{
    thread_data.push_back(nullptr);
    tract_rendering.push_back(std::make_shared<TractRender>());
    tract_models.push_back(std::make_shared<TractModel>(cur_tracking_window.handle));
    insertRow(tract_models.size()-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(tract_name);
    item0->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
    item0->setData(Qt::ForegroundRole,checked ? QBrush(Qt::black) : QBrush(Qt::gray));
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
    if(!cur_tracking_window.handle->set_dt_index(
            cur_tracking_window.get_dt_index_pair(),
            cur_tracking_window.renderWidget->getData("dt_threshold_type").toInt()))
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }

    QString tract_name = cur_tracking_window.regionWidget->getROIname();

    // if running differential tracking
    if(!cur_tracking_window.handle->dir.dt_fa.empty())
    {
        cur_tracking_window.ui->target->setCurrentIndex(0);
        tract_name = QString(cur_tracking_window.handle->dir.dt_threshold_name.c_str())+"_"+
                     QString::number(cur_tracking_window["dt_threshold"].toDouble());
    }
    // if running autotrack
    if(cur_tracking_window.ui->target->currentIndex() > 0) // auto track
        tract_name = cur_tracking_window.ui->target->currentText();

    addNewTracts(tract_name);
    thread_data.back() = std::make_shared<ThreadData>(cur_tracking_window.handle);
    cur_tracking_window.set_tracking_param(*thread_data.back());
    cur_tracking_window.regionWidget->setROIs(thread_data.back().get());
    thread_data.back()->run(cur_tracking_window.ui->thread_count->value(),false);
    tract_models.back()->report = cur_tracking_window.handle->report + thread_data.back()->report.str();
    show_report();
    timer->start(500);
    timer_update->start(100);
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
    cur_tracking_window.set_tracking_param(track_thread);
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
            item(int(index),3)->setText(QString::number(thread_data[index]->get_total_seed_count()));
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
                    if(regions.size() >= 2 && cur_tracking_window.regionWidget->item(int(0),0)->text() == "debug")
                        {
                            regions[0]->region = thread_data[index]->roi_mgr->atlas_seed;
                            regions[0]->modified = true;
                            regions[1]->region = thread_data[index]->roi_mgr->atlas_roa;
                            regions[1]->modified = true;
                        }
                }
                tract_rendering[index]->need_update = true;
                auto lock = tract_rendering[index]->start_writing();
                has_tracts |= thread_data[index]->fetchTracks(tract_models[index].get()); // clear both front and back buffer
                has_tracts |= thread_data[index]->fetchTracks(tract_models[index].get()); // clear both front and back buffer
                thread_data[index]->apply_tip(tract_models[index].get());
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
        timer->stop();
}

void TractTableWidget::stop_tracking(void)
{
    for(unsigned int index = 0;index < thread_data.size();++index)
        if(thread_data[index].get())
            thread_data[index]->end_thread();
}
void TractTableWidget::load_tracts(QStringList filenames)
{
    if(filenames.empty())
        return;
    progress p("load tracts");
    for(unsigned int index = 0;progress::at(index,filenames.size());++index)
    {
        QString filename = filenames[index];
        if(!filename.size())
            continue;
        QString label = QFileInfo(filename).fileName();
        label.remove(".tt.gz");
        label.remove(".trk.gz");
        label.remove(".txt");
        int pos = label.indexOf(".fib.gz");
        if(pos != -1)
            label = label.right(label.length()-pos-8);
        std::string sfilename = filename.toStdString();
        addNewTracts(label,filenames.size() == 1);
        if(!tract_models.back()->load_from_file(&*sfilename.begin(),false))
        {
            QMessageBox::critical(this,"ERROR",QString("Cannot load tracks from %1").arg(QFileInfo(filename).baseName()));
            continue;
        }
        if(tract_models.back()->trans_to_mni[0] != 0.0f &&
           tract_models.back()->trans_to_mni != cur_tracking_window.handle->trans_to_mni)
        {
            show_progress() << "tractography is from a different space" << std::endl;
            show_progress() << "host space=" << std::endl;
            show_progress() << cur_tracking_window.handle->trans_to_mni << std::endl;
            show_progress() << "tractography space= " << std::endl;
            show_progress() << tract_models.back()->trans_to_mni << std::endl;
            show_progress() << "apply transformation to tracts" << std::endl;
            tipl::matrix<4,4> T = tipl::from_space(tract_models.back()->trans_to_mni).
                                    to(cur_tracking_window.handle->trans_to_mni);
            auto& loaded_tract_data = tract_models.back()->get_tracts();
            tipl::par_for(loaded_tract_data.size(),[&](size_t index)
            {
                auto& tract = loaded_tract_data[index];
                for(size_t i = 0;i < tract.size();i += 3)
                {
                    tipl::vector<3> p(&tract[i]);
                    p.to(T);
                    tract[i] = p[0];
                    tract[i+1] = p[1];
                    tract[i+2] = p[2];
                }
            });
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
}

void TractTableWidget::load_tracts(void)
{
    load_tracts(QFileDialog::getOpenFileNames(
            this,"Load tracts as",QFileInfo(cur_tracking_window.work_path).absolutePath(),
            "Tract files (*tt.gz *.trk *trk.gz *.tck);;Text files (*.txt);;All files (*)"));
    show_report();
}
void TractTableWidget::load_tract_label(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load tracts as",QFileInfo(cur_tracking_window.work_path).absolutePath(),
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
    progress p("rendering tracts",true);
    for(int row = 0;progress::at(row,rowCount());++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
        QApplication::processEvents();
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

void TractTableWidget::save_all_tracts_to_dir(void)
{
    if (tract_models.empty())
        return;
    QString dir = QFileDialog::getExistingDirectory(this,"Open directory","");
    if(dir.isEmpty())
        return;
    if(!command("save_all_tracts_to_dir",dir))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
    else
        QMessageBox::information(this,"DSI Studio","file saved");
}
void TractTableWidget::save_all_tracts_as(void)
{
    if(tract_models.empty())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;NIFTI File (*nii.gz);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!command("save_tracks",filename))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
    else
        QMessageBox::information(this,"DSI Studio","file saved");
}

void TractTableWidget::set_color(void)
{
    if(tract_models.empty())
        return;
    QColor color = QColorDialog::getColor(Qt::red,(QWidget*)this,"Select color",QColorDialog::ShowAlphaChannel);
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
        tipl::rgb c;
        c.from_hsl((color_gen*1.1-std::floor(color_gen*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        color_gen++;
        auto lock = tract_rendering[index]->start_writing();
        tract_models[index]->set_color(c.color);
        tract_rendering[index]->need_update = true;
    }
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit show_tracts();
}
void TractTableWidget::load_cluster_label(const std::vector<unsigned int>& labels,QStringList Names)
{
    std::string report = tract_models[uint32_t(currentRow())]->report;
    std::vector<std::vector<float> > tracts;
    tract_models[uint32_t(currentRow())]->release_tracts(tracts);
    tract_rendering[uint32_t(currentRow())]->need_update = true;
    delete_row(currentRow());
    unsigned int cluster_count = uint32_t(Names.empty() ? int(1+tipl::max_value(labels)):int(Names.count()));
    for(unsigned int cluster_index = 0;cluster_index < cluster_count;++cluster_index)
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
            addNewTracts(Names[int(cluster_index)],false);
        else
            addNewTracts(QString("cluster")+QString::number(cluster_index),false);
        tract_models.back()->add_tracts(add_tracts);
        tract_models.back()->report = report;
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

    std::ifstream in(filename.toLocal8Bit().begin());
    std::vector<unsigned int> labels(tract_models[uint32_t(currentRow())]->get_visible_track_count());
    std::copy(std::istream_iterator<unsigned int>(in),
              std::istream_iterator<unsigned int>(),labels.begin());
    load_cluster_label(labels);
    assign_colors();
}
void TractTableWidget::open_cluster_color(void)
{
    if(tract_models.empty())
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load cluster color",QFileInfo(cur_tracking_window.work_path).absolutePath(),
            "RGB Value Text(*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!command("load_cluster_color",filename,""))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}
void TractTableWidget::save_cluster_color(void)
{
    if(tract_models.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,"Save cluster color",QFileInfo(cur_tracking_window.work_path).absolutePath(),
            "RGB Value Text(*.txt);;All files (*)");
    if(!filename.size())
        return;
    std::ofstream out(filename.toStdString());
    if(!out)
    {
        QMessageBox::critical(this,"ERROR","Cannot save file");
        return;
    }
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
    QMessageBox::information(this,"DSI Studio","saved");
}



void TractTableWidget::recog_tracks(void)
{
    if(currentRow() >= int(tract_models.size()) || tract_models[uint32_t(currentRow())]->get_tracts().size() == 0)
        return;
    if(!cur_tracking_window.handle->load_track_atlas())
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


void TractTableWidget::auto_recognition(void)
{
    if(!cur_tracking_window.handle->load_track_atlas())
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }
    std::vector<unsigned int> c,new_c,count;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        cur_tracking_window.handle->recognize(tract_models[uint32_t(currentRow())],c,count);
    }
    std::multimap<unsigned int,unsigned int,std::greater<unsigned int> > tract_list;
    for(unsigned int i = 0;i < count.size();++i)
        if(count[i])
            tract_list.insert(std::make_pair(count[i],i));

    QStringList Names;
    unsigned int index = 0;
    new_c.resize(c.size());
    for(auto p : tract_list)
    {
        for(size_t j = 0;j < c.size();++j)
            if(c[j] == p.second)
                new_c[j] = index;
        Names << cur_tracking_window.handle->tractography_name_list[p.second].c_str();
        ++index;
    }
    load_cluster_label(new_c,Names);
}

void TractTableWidget::recognize_rename(void)
{
    if(!cur_tracking_window.handle->load_track_atlas())
    {
        QMessageBox::critical(this,"ERROR",cur_tracking_window.handle->error_msg.c_str());
        return;
    }
    progress prog_("Recognize and rename");
    for(unsigned int index = 0;progress::at(index,tract_models.size());++index)
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
            "DSI Studio",
            "Assign the maximum number of groups",50,1,5000,10,&ok);
    if(!ok)
        return;
    ok = true;
    double detail = method_id ? 0.0 : QInputDialog::getDouble(this,
            "DSI Studio","Clustering detail (mm):",cur_tracking_window.handle->vs[0],0.2,50.0,2,&ok);
    if(!ok)
        return;
    std::vector<unsigned int> c;
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
        tract_models[uint32_t(currentRow())]->run_clustering(method_id,n,detail);
        c = tract_models[uint32_t(currentRow())]->get_cluster_info();
    }
    load_cluster_label(c);
    assign_colors();
}

void TractTableWidget::save_tracts_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;TCK file (*.tck);;ROI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string sfilename = filename.toLocal8Bit().begin();
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_to_file(&*sfilename.begin()))
        QMessageBox::information(this,"DSI Studio","file saved");
    else
        QMessageBox::critical(this,"Error","Cannot write to file. Please check write permission.");
}

void TractTableWidget::save_tracts_in_native(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0)
        return;
    if(!cur_tracking_window.handle->is_mni)
    {
        QMessageBox::critical(this,"ERROR","This function only works with QSDR reconstructed FIB files.");
        return;
    }
    if(cur_tracking_window.handle->get_native_position().empty())
    {
        QMessageBox::critical(this,"ERROR","No mapping information included. Please reconstruct QSDR with 'mapping' included in the output.");
        return;
    }

    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_in_native_space(cur_tracking_window.handle,filename.toStdString().c_str()))
        QMessageBox::information(this,"DSI Studio","file saved");
    else
        QMessageBox::critical(this,"Error","Cannot write to file. Please check write permission.");
}

void TractTableWidget::save_vrml_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",item(currentRow(),0)->text().replace(':','_') + ".obj",
                 "3D files (*.obj);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string surface_text;
    std::string sfilename = filename.toLocal8Bit().begin();
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    tract_models[uint32_t(currentRow())]->save_vrml(&*sfilename.begin(),
                                                cur_tracking_window["tract_style"].toInt(),
                                                cur_tracking_window["tract_color_style"].toInt(),
                                                cur_tracking_window["tube_diameter"].toFloat(),
                                                cur_tracking_window["tract_tube_detail"].toInt(),surface_text);
}
void TractTableWidget::save_all_tracts_end_point_as(void)
{
    auto selected_tracts = get_checked_tracks();
    if(selected_tracts.empty())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",item(currentRow(),0)->text().replace(':','_') + "endpoint.nii.gz",
                "NIFTI files (*nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    bool ok;
    float dis = float(QInputDialog::getDouble(this,
        "DSI Studio","Assign end segment length in voxel distance:",3.0,0.0,10.0,1,&ok));
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
                "Save end points as",item(currentRow(),0)->text().replace(':','_') + "endpoint.txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    tract_models[uint32_t(currentRow())]->save_end_points(filename.toStdString().c_str());
}

void TractTableWidget::save_end_point_in_mni(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() < 0 || !cur_tracking_window.map_to_mni())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end points as",item(currentRow(),0)->text().replace(':','_') + "endpoint.txt",
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


void TractTableWidget::save_transformed_tracts(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",item(currentRow(),0)->text() + "_in_" +
                cur_tracking_window.current_slice->get_name().c_str()+output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
    if(!slice)
    {
        QMessageBox::critical(this,"Error","Current slice is in the DWI space. Please use regular tract saving function");
        return;
    }

    slice->update_transform();
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_transformed_tracts_to_file(filename.toStdString().c_str(),slice->dim,slice->vs,slice->invT,false))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"Error","File not saved. Please check write permission");
}


void TractTableWidget::save_transformed_endpoints(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save end_point as",item(currentRow(),0)->text() + "_endpoints_in_" +
                cur_tracking_window.current_slice->get_name().c_str()+
                ".txt",
                "Tract files (*.txt);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
    if(!slice)
    {
        QMessageBox::critical(this,"Error","Current slice is in the DWI space. Please use regular tract saving function");
        return;
    }
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_transformed_tracts_to_file(filename.toStdString().c_str(),slice->dim,slice->vs,slice->invT,true))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"Error","File not saved. Please check write permission");
}
extern std::vector<std::string> fa_template_list;
void TractTableWidget::save_tracts_in_template(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1 || !cur_tracking_window.map_to_mni())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",item(currentRow(),0)->text() + "_in_" +
                QFileInfo(fa_template_list[cur_tracking_window.handle->template_id].c_str()).baseName() +
                output_format(),
                 "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;Text File (*.txt);;MAT files (*.mat);;NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_in_template_space(cur_tracking_window.handle,filename.toStdString().c_str()))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"Error","File not saved. Please check write permission");
}

void TractTableWidget::save_tracts_in_mni(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1 || !cur_tracking_window.map_to_mni())
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save tracts as",item(currentRow(),0)->text() + "_in_mni" + output_format(),
                 "NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->save_tracts_in_template_space(cur_tracking_window.handle,filename.toStdString().c_str()))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"Error","File not saved. Please check write permission");
}

void TractTableWidget::load_tracts_color(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load tracts color",QFileInfo(cur_tracking_window.work_path).absolutePath(),
            "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!command("load_track_color",filename,""))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());

}

void TractTableWidget::load_tracts_value(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Load tracts color",QFileInfo(cur_tracking_window.work_path).absolutePath(),
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
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    if(tract_models[uint32_t(currentRow())]->get_visible_track_count() != values.size())
    {
        QMessageBox::information(this,"Inconsistent track number",
                                 QString("The text file has %1 values, but there are %2 tracks.").
                                 arg(values.size()).arg(tract_models[uint32_t(currentRow())]->get_visible_track_count()),0);
        return;
    }
    color_bar_dialog dialog(nullptr);
    auto min_max = std::minmax_element(values.begin(),values.end());
    dialog.set_value(*min_max.first,*min_max.second);
    dialog.exec();
    std::vector<unsigned int> colors(values.size());
    for(int i = 0;i < values.size();++i)
        colors[i] = (unsigned int)dialog.get_rgb(values[i]);
    tract_models[uint32_t(currentRow())]->set_tract_color(colors);
    tract_rendering[uint32_t(currentRow())]->need_update = true;
    cur_tracking_window.set_data("tract_color_style",1);//manual assigned
    emit show_tracts();
}

void TractTableWidget::save_tracts_color_as(void)
{
    if(currentRow() >= int(tract_models.size()) || currentRow() == -1)
        return;
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts color as",item(currentRow(),0)->text() + "_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::string sfilename = filename.toLocal8Bit().begin();
    auto lock = tract_rendering[uint32_t(currentRow())]->start_reading();
    tract_models[uint32_t(currentRow())]->save_tracts_color_to_file(&*sfilename.begin());
}

void get_track_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          const std::vector<std::string>& track_name,
                          std::string& result)
{
    if(tract_models.empty())
        return;
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    for(size_t index = 0;index < tract_models.size();++index)
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
std::vector<std::string> TractTableWidget::get_checked_tracks_name(void) const
{
    std::vector<std::string> track_name;
    for(unsigned int index = 0;index < tract_models.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked)
            track_name.push_back(item(int(index),0)->text().toStdString());
    return track_name;
}
void TractTableWidget::show_tracts_statistics(void)
{
    if(tract_models.empty())
        return;
    std::string result;
    get_track_statistics(cur_tracking_window.handle,get_checked_tracks(),get_checked_tracks_name(),result);
    if(!result.empty())
        show_info_dialog("Tract Statistics",result);

}
void TractTableWidget::need_update_all(void)
{
    for(auto& t:tract_rendering)
        t->need_update = true;
}
void TractTableWidget::render_tracts(GLWidget* glwidget)
{
    for(unsigned int index = 0;index < tract_rendering.size();++index)
        if(item(int(index),0)->checkState() == Qt::Checked &&
           tract_models[index]->get_visible_track_count())
        tract_rendering[index]->render_tracts(tract_models[index],glwidget,cur_tracking_window);
}

bool TractTableWidget::command(QString cmd,QString param,QString param2)
{
    if(cmd == "save_all_tracts_to_dir")
    {
        progress prog_("save files");
        auto selected_tracts = get_checked_tracks();
        auto selected_tracts_rendering = get_checked_tracks_rendering();
        for(size_t index = 0;index < selected_tracts.size();++index)
        {
            std::string filename = param.toStdString();
            filename += "/";
            filename += item(int(index),0)->text().toStdString();
            filename += output_format().toStdString();
            auto lock = selected_tracts_rendering[index]->start_reading();
            if(!selected_tracts[index]->save_tracts_to_file(filename.c_str()))
            {
                error_msg = "cannot save file to ";
                error_msg = filename;
                return false;
            }
        }
        return true;
    }
    if(cmd == "update_track")
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
    if(cmd == "run_tracking")
    {
        start_tracking();
        while(timer->isActive())
            fetch_tracts();
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
        while(!tract_rendering.empty())
        {
            tract_rendering.pop_back();
            thread_data.pop_back();
            tract_models.pop_back();
        }
        emit show_tracts();
        return true;
    }
    if(cmd == "save_tracks")
    {
        auto locks = start_reading_checked_tracks();
        if(!TractModel::save_all(param.toStdString().c_str(),
                             get_checked_tracks(),get_checked_tracks_name()))
        {
            error_msg = "cannot save file to ";
            error_msg = param.toStdString();
            return false;
        }
        return true;
    }
    if(cmd == "load_track_color")
    {
        int index = currentRow();
        if(!param2.isEmpty())
        {
            index = param2.toInt();
            if(index < 0 || index >= tract_models.size())
            {
                error_msg = "invalid track index: ";
                error_msg += param2.toStdString();
                return false;
            }
        }
        auto lock = tract_rendering[index]->start_reading();
        if(!tract_models[index]->load_tracts_color_from_file(param.toStdString().c_str()))
        {
            error_msg = "cannot find or open ";
            error_msg += param.toStdString();
            return false;
        }
        tract_rendering[index]->need_update = true;
        cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        emit show_tracts();
        return true;
    }
    if(cmd == "load_cluster_color")
    {
        std::ifstream in(param.toStdString());
        if(!in)
        {
            error_msg = "cannot find or open ";
            error_msg += param.toStdString();

            return false;
        }
        progress p("rendering tracts");
        for(unsigned int index = 0;progress::at(index,tract_models.size()) && in;++index)
            if(item(int(index),0)->checkState() == Qt::Checked)
            {
                int r(0),g(0),b(0);
                in >> r >> g >> b;
                auto lock = tract_rendering[index]->start_writing();
                tract_models[index]->set_color(tipl::rgb(r,g,b));
                tract_rendering[index]->need_update = true;
            }
        cur_tracking_window.set_data("tract_color_style",1);//manual assigned
        emit show_tracts();
    }
    return false;
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
                    cur_tracking_window.handle,filename.toLocal8Bit().begin(),
                    action->data().toString().toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error","fail to save information");
    }
    else
        QMessageBox::information(this,"DSI Studio","file saved");
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
    if(progress::level())
    {
        QMessageBox::critical(this,"ERROR","Please wait for the termination of data processing");
        return;
    }
    delete_row(currentRow());
    emit show_tracts();
}

void TractTableWidget::delete_all_tract(void)
{
    if(progress::level())
    {
        QMessageBox::critical(this,"ERROR","Please wait for the termination of data processing");
        return;
    }
    command("delete_all_tract");
}

void TractTableWidget::delete_repeated(void)
{
    float distance = 1.0;
    bool ok;
    distance = QInputDialog::getDouble(this,
        "DSI Studio","Distance threshold (voxels)", distance,0,500,1,&ok);
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
        "DSI Studio","New step size (voxels)",double(new_step),0.0,5.0,1,&ok));
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
    progress prog_("filtering tracks");

    float threshold = 60;
    bool ok;
    threshold = QInputDialog::getDouble(this,
        "DSI Studio","Length threshold in mm:", threshold,0,500,1,&ok);
    if (!ok)
        return;
    for_each_bundle("delete by length",[&](unsigned int index)
    {
        return tract_models[index]->delete_by_length(threshold);
    });
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
    {
        auto lock = tract_rendering[uint32_t(currentRow())]->start_writing();
        tract_models[uint32_t(cur_row)]->reconnect_track(dis,std::cos(angle*3.14159265358979323846f/180.0f));
        tract_rendering[uint32_t(cur_row)]->need_update = true;
    }
    item(cur_row,1)->setText(QString::number(tract_models[uint32_t(cur_row)]->get_visible_track_count()));
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

void TractTableWidget::cut_by_slice(unsigned char dim,bool greater)
{
    for_each_bundle("cut by slice",[&](unsigned int index)
    {
        tract_models[index]->cut_by_slice(dim,cur_tracking_window.current_slice->slice_pos[dim],greater,
            (cur_tracking_window.current_slice->is_diffusion_space ? nullptr:&cur_tracking_window.current_slice->invT));
        return true;
    });
}

void TractTableWidget::export_tract_density(tipl::shape<3> dim,
                                            tipl::vector<3,float> vs,
                                            tipl::matrix<4,4> transformation,bool color,bool end_point)
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
    if(TractModel::export_tdi(filename.toStdString().c_str(),get_checked_tracks(),dim,vs,transformation,color,end_point))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"ERROR","Failed to save file");
}


