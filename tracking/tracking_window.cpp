#include <utility>
#include <filesystem>

#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <QCompleter>
#include <QSplitter>
#include <QSettings>
#include <QClipboard>
#include <QShortcut>
#include <QApplication>
#include <QMouseEvent>
#include <QMessageBox>
#include <QListWidget>

#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "console.h"
#include "region/regiontablewidget.h"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
#include "mapping/atlas.hpp"
#include "view_image.h"

#include "devicetablewidget.h"
#include "fib_data.hpp"
#include "regtoolbox.h"
#include "connectivity_matrix_dialog.h"

extern std::vector<std::string> fa_template_list;
extern std::vector<tracking_window*> tracking_windows;
extern size_t auto_track_pos[7];
extern unsigned char auto_track_rgb[6][3];               // projection

QByteArray default_geo,default_state;


void populate_templates(QComboBox* combo,size_t index)
{
    combo->clear();
    if(!fa_template_list.empty())
    {
        for(size_t i = 0;i < fa_template_list.size();++i)
            combo->addItem(QFileInfo(fa_template_list[i].c_str()).baseName());
        combo->setCurrentIndex(int(index));
    }
}

QVariant tracking_window::operator[](QString name) const
{
    return renderWidget->getData(name);
}
void tracking_window::set_data(QString name, QVariant value)
{
    renderWidget->setData(name,value);
}
void tracking_window::set_memorize_parameters(bool memorize)
{
    renderWidget->setMemorizeParameters(memorize);
}
std::shared_ptr<command_history::surrogate> command_history::record(std::string& error_msg_,
                                  std::vector<std::string>& cmd)
{
    error_msg_.clear();
    if(!cmd.empty())
        current_cmd = cmd[0];
    return std::make_shared<surrogate>(*this,cmd,error_msg_);
}

void command_history::add_record(const std::string& output)
{
    commands.push_back(output);
    if(is_loading(output))
    {
        auto p = std::filesystem::path(tipl::split(output,',')[1]);
        default_parent_path = p.parent_path().string();
        default_stem2 = p.stem().stem().string();
        std::replace(default_stem2.begin(),default_stem2.end(),'.','_');
        if(default_stem.empty())
            default_stem = default_stem2;
    }
}
std::string command_history::file_stem(bool extended) const
{
    if(tipl::contains(current_cmd,"to_folder"))
        return default_parent_path;
    auto result = (std::filesystem::path(default_parent_path)/default_stem).string();
    if(extended && default_stem2 != default_stem && !tipl::contains(current_cmd,"_all"))
        result += "_" + default_stem2;
    return result;
}
bool command_history::get_dir(QWidget* parent,std::string& cmd)
{
    return !cmd.empty() || !(cmd = QFileDialog::getExistingDirectory(parent,QString::fromStdString(current_cmd),
                                            QString::fromStdString(default_parent_path)).toStdString()).empty();
}


bool command_history::get_filename(QWidget* parent,std::string& filename,const std::string& post_fix)
{
    if(!filename.empty())
        return true;
    QString filter,default_file_name(QString::fromStdString(
                                post_fix.empty() ? file_stem() :
                                (std::filesystem::path(default_parent_path)/
                                    (post_fix[0] == '.' ? default_stem + post_fix : post_fix)).string()));
    if(tipl::ends_with(current_cmd,"_tracts") || tipl::ends_with(current_cmd,"_tract"))
        filter = "Tract files (*.tt.gz *tt.gz *trk.gz *.trk);;MAT files (*.mat)";
    else
    if(tipl::ends_with(current_cmd,"_regions") || tipl::ends_with(current_cmd,"_region") ||
       tipl::ends_with(current_cmd,"_volume"))
            filter = "NIFTI file(*nii.gz *.nii);;MAT file (*.mat)";
    else
        default_file_name += "_" + QString::fromStdString(current_cmd).split('_').back() + ".txt";
    if(!filter.isEmpty())
        filter += ";;";
    filter += "Text files (*.txt);;All files (*)";
    if(tipl::begins_with(current_cmd,"save_"))
        return !(filename = QFileDialog::getSaveFileName(
                    parent,QString::fromStdString(current_cmd),default_file_name,filter).toStdString()).empty();
    else
        return !(filename = QFileDialog::getOpenFileName(
                    parent,QString::fromStdString(current_cmd),default_file_name,filter).toStdString()).empty();
}

bool command_history::is_loading(const std::string& cmd)
{
    return cmd.find(',') < cmd.find('.') && cmd.find('.') != std::string::npos &&
               (tipl::begins_with(cmd,"load_") || tipl::begins_with(cmd,"open_"));
}
bool command_history::is_saving(const std::string& cmd)
{
    return cmd.find(',') < cmd.find('.') && cmd.find('.') != std::string::npos &&  tipl::begins_with(cmd,"save_");
}
bool command_history::run(tracking_window *parent,const std::vector<std::string>& cmd,bool apply_to_others)
{
    QStringList file_list;
    std::string original_file;
    int loading_index = -1;

    auto first_load = std::find_if(cmd.begin(),cmd.end(),[](const auto& cmd_line){return is_loading(cmd_line);});
    if(first_load != cmd.end() && apply_to_others)
    {
        loading_index = first_load-cmd.end();
        auto param = tipl::split(*first_load,',');
        original_file = param[1];
        QMessageBox::information(parent,QApplication::applicationName(),
            "select files for '" + QString::fromStdString(param[0]).replace('_',' ') + "' step");
        file_list = QFileDialog::getOpenFileNames(parent,"files for " + QString::fromStdString(param[0]),original_file.c_str(),
            QString("files (*%1);;all files (*)").arg(tipl::complete_suffix(original_file).c_str()));
        if(file_list.isEmpty())
            return false;
    }

    tipl::progress p("running commands",true);
    running_commands = true;
    size_t total_steps = std::max<int>(1,file_list.size())*cmd.size();
    for(size_t k = 0,steps = 0;k < std::max<int>(1,file_list.size()) && !p.aborted();++k)
    {
        tipl::progress p("processing " + (k < file_list.size() ? file_list[k].toStdString() : std::string()));
        tracking_window *backup_parent = nullptr;
        for(int j = 0;j < cmd.size() && !p.aborted();++j,++steps)
        {
            auto param = tipl::split(cmd[j],',');

            if(k < file_list.size())
            {
                if(loading_index == j)
                    param[1] = file_list[k].toStdString();
                else
                if(is_loading(cmd[j]) || is_saving(cmd[j]))
                {
                    std::string new_file_name;
                    if(tipl::match_files(original_file,param[1],file_list[k].toStdString(),new_file_name))
                    {
                        if(is_loading(cmd[j]) && !std::filesystem::exists(new_file_name))
                        {
                            tipl::warning() << "cannot find " << new_file_name;
                            auto search_path = std::filesystem::path(new_file_name).parent_path().string();
                            auto wildcard = "*" + tipl::complete_suffix(new_file_name);
                            auto files =  tipl::search_files(search_path,wildcard);
                            tipl::warning() << "searching for " << wildcard << " at " << search_path << " ..." << files.size() << " file(s) found";
                            auto name_cannot_find = std::filesystem::path(new_file_name).filename().string();
                            for (const auto& each : files)
                            {
                                auto name = QFileInfo(each.c_str()).fileName().toStdString();
                                if(tipl::contains(name_cannot_find,name) ||
                                   tipl::contains(name,name_cannot_find))
                                {
                                    tipl::warning() << "found and will use (need to check!): " << each;
                                    new_file_name = each;
                                    break;
                                }
                            }
                            if(!std::filesystem::exists(new_file_name))
                                tipl::warning() << "cannot identify one to replace";
                        }
                        if(is_loading(cmd[j]) && !std::filesystem::exists(new_file_name))
                        {
                            if(QMessageBox::question(parent, QApplication::applicationName(),
                                            QString("cannot find %1 at %2.\nCancel?\n").
                                                   arg(QFileInfo(new_file_name.c_str()).fileName()).
                                                   arg(QFileInfo(new_file_name.c_str()).absolutePath()),
                                            QMessageBox::No | QMessageBox::Yes) == QMessageBox::Yes)

                                file_list.clear();
                            break;
                        }
                        param[1] = new_file_name;
                    }
                }
            }
            tipl::out() << "run " << tipl::merge(param,',');
            if(!parent->command(param))
            {
                QMessageBox::critical(parent,"ERROR",(param.front() + " failed due to " + parent->error_msg).c_str());
                running_commands = false;
                return false;
            }
            while (p(steps,total_steps) && parent->history.has_other_thread)
            {
                QCoreApplication::processEvents();
                QThread::msleep(100);
            }
            if(tipl::begins_with(cmd[j],"open_fib"))
            {
                backup_parent = parent;
                parent = tracking_windows.back();
            }
        }
        if(backup_parent)
        {
            delete parent;
            parent = backup_parent;
        }
    }
    running_commands = false;
    return !p.aborted();
}

tracking_window::tracking_window(QWidget *parent,std::shared_ptr<fib_data> new_handle) :
        QMainWindow(parent),ui(new Ui::tracking_window),scene(*this),handle(new_handle),work_path(QFileInfo(new_handle->fib_file_name.c_str()).absolutePath()+"/")
{

    setAcceptDrops(true);
    tipl::progress prog("initializing tracking GUI");


    ui->setupUi(this);
    ui->thread_count->setValue(tipl::max_thread_count >> 1);

    // setup GUI
    {
        {
            tipl::out() << "create GUI objects" << std::endl;
            scene.statusbar = ui->statusbar;
            ui->regionDockWidget->setMinimumWidth(0);
            ui->ROIdockWidget->setMinimumWidth(0);
            ui->renderingLayout->addWidget(renderWidget = new RenderingTableWidget(*this,ui->renderingWidgetHolder));
            ui->glLayout->addWidget(glWidget = new GLWidget(*this,renderWidget));
            ui->verticalLayout_3->addWidget(regionWidget = new RegionTableWidget(*this,ui->regionDockWidget));
            ui->track_verticalLayout->addWidget(tractWidget = new TractTableWidget(*this,ui->TractWidgetHolder));
            ui->deviceLayout->addWidget(deviceWidget = new DeviceTableWidget(*this,ui->TractWidgetHolder));
            ui->graphicsView->setScene(&scene);
            ui->graphicsView->setCursor(Qt::CrossCursor);
            ui->DeviceDockWidget->hide();
        }
        {
            ui->TractWidgetHolder->show();
            ui->renderingWidgetHolder->show();
            ui->ROIdockWidget->show();
            ui->regionDockWidget->show();
        }
        {
            ui->zoom->setValue((*this)["roi_zoom"].toFloat());
            ui->roi_draw_edge->setChecked((*this)["roi_draw_edge"].toBool());
            ui->roi_track->setChecked((*this)["roi_track"].toBool());
            ui->roi_label->setChecked((*this)["roi_label"].toBool());
            ui->roi_position->setChecked((*this)["roi_position"].toBool());
            ui->roi_ruler->setChecked((*this)["roi_ruler"].toBool());
            ui->roi_fiber->setChecked((*this)["roi_fiber"].toBool());
            if(handle->dim[0] > 80)
                ui->zoom_3d->setValue(80.0/(float)std::max<int>(std::max<int>(handle->dim[0],handle->dim[1]),handle->dim[2]));
        }
        // Enabled/disable GUIs
        {
            if(!handle->trackable)
            {
                ui->perform_tracking->hide();
                ui->stop_tracking->hide();
                ui->enable_auto_tract->setText("Enable Tractography...");
            }            
        }
        {
            tipl::out() << "prepare template and atlases" << std::endl;
            populate_templates(ui->template_box,handle->template_id);
        }

        {
            tipl::out() << "initialize slices" << std::endl;

            for (auto each : handle->slices)
                addSlices(std::make_shared<SliceModel>(handle,each));

            if(handle->is_mni)
            {
                addSlices("t1w_template",handle->t1w_template_file_name);
                addSlices("t2w_template",handle->t2w_template_file_name);
                addSlices("wm_template",handle->wm_template_file_name);
            }

            if(!handle->other_images.empty())
                for(const auto& each : tipl::split(handle->other_images,','))
                    addSlices(std::filesystem::path(each).stem().stem().string(),each);

            current_slice = slices[0];
            glWidget->slice_texture.resize(slices.size());
        }
        {
            // update options: fa threshold
            {
                QStringList tracking_index_list;
                for(size_t index = 0;index < handle->dir.index_name.size();++index)
                    if(handle->dir.index_name[index].find("dec_") != 0 &&
                       handle->dir.index_name[index].find("inc_") != 0)
                        tracking_index_list.push_back(handle->dir.index_name[index].c_str());
                renderWidget->setList("tracking_index",tracking_index_list);
            }
            // update options: color map
            {
                QStringList color_map_list("none");
                for(auto each : QDir(QCoreApplication::applicationDirPath()+"/color_map/").
                                    entryList(QStringList("*.txt"),QDir::Files|QDir::NoSymLinks))
                    color_map_list << QFileInfo(each).baseName();
                renderWidget->setList("tract_color_map",color_map_list);
                renderWidget->setList("region_color_map",color_map_list);
            }

            if(handle->is_histology || handle->dim[0] > 1024)
            {
                set_data("fa_threshold",0.1f);
                set_data("step_size",handle->vs[0]*2.0f);
                set_data("turning_angle",15);
                set_data("tube_diameter",1.0f);
                set_data("track_count",50000);
            }
            else
            {
                set_data("fa_threshold",0.0f);
                set_data("step_size",0.0f);
                set_data("turning_angle",0.0f);
                set_data("tube_diameter",0.15f);                
            }
            set_data("min_length",handle->min_length());
            set_data("max_length",handle->max_length());
            set_data("tolerance",float(handle->min_length())*0.8f);

        }


        report(handle->report.c_str());

        // provide automatic tractography
        {
            ui->tract_target_0->hide();
            ui->tract_target_1->hide();
            ui->tract_target_2->hide();
        }

    }

    tipl::out() << "connect signal and slots " << std::endl;
    // opengl
    {
        connect(ui->zoom_3d,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double zoom){command({"set_zoom",std::to_string(zoom)});});


        connect(ui->max_color_gl,&QColorToolButton::clicked,this,[this](){run_command("set_slice_contrast");});
        connect(ui->min_color_gl,&QColorToolButton::clicked,this,[this](){run_command("set_slice_contrast");});

        connect(ui->min_value_gl,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double){
            ui->min_slider->setValue(int((ui->min_value_gl->value()-ui->min_value_gl->minimum())*double(ui->min_slider->maximum())/(ui->min_value_gl->maximum()-ui->min_value_gl->minimum())));
            run_command("set_slice_contrast");});
        connect(ui->max_value_gl,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double){
            ui->max_slider->setValue(int((ui->max_value_gl->value()-ui->max_value_gl->minimum())*double(ui->max_slider->maximum())/(ui->max_value_gl->maximum()-ui->max_value_gl->minimum())));
            run_command("set_slice_contrast");});
        connect(ui->min_slider,&QSlider::sliderMoved,this,[this](void){
            ui->min_value_gl->setValue(ui->min_value_gl->minimum()+(ui->min_value_gl->maximum()-ui->min_value_gl->minimum())*
                              double(ui->min_slider->value())/double(ui->min_slider->maximum()));});
        connect(ui->max_slider,&QSlider::sliderMoved,this,[this](void){
            ui->max_value_gl->setValue(ui->max_value_gl->minimum()+(ui->max_value_gl->maximum()-ui->max_value_gl->minimum())*
                              double(ui->max_slider->value())/double(ui->max_slider->maximum()));});

        connect(ui->actionInsert_Sagittal_Picture,SIGNAL(triggered()),this,SLOT(insertPicture()));
        connect(ui->actionInsert_Coronal_Pictures,SIGNAL(triggered()),this,SLOT(insertPicture()));
        connect(ui->actionInsert_Axial_Pictures,SIGNAL(triggered()),this,SLOT(insertPicture()));

    }
    // scene view
    {

        connect(&scene,&slice_view_scene::need_update,this,[this](void){slice_need_update = true;});
        connect(&scene,SIGNAL(need_update()),glWidget,SLOT(update()));


        connect(ui->actionAxial_View,&QAction::triggered,this,[this](void){if(ui->glAxiView->isChecked()){glWidget->set_view(cur_dim);glWidget->update();}else ui->glAxiView->setChecked(true);});
        connect(ui->actionCoronal_View,&QAction::triggered,this,[this](void){if(ui->glCorView->isChecked()){glWidget->set_view(cur_dim);glWidget->update();}else ui->glCorView->setChecked(true);});
        connect(ui->actionSagittal_view,&QAction::triggered,this,[this](void){if(ui->glSagView->isChecked()){glWidget->set_view(cur_dim);glWidget->update();}else ui->glSagView->setChecked(true);});

        connect(ui->glSagView,qOverload<bool>(&QPushButton::toggled),this,[this](bool checked){if(checked)cur_dim = 0;});
        connect(ui->glCorView,qOverload<bool>(&QPushButton::toggled),this,[this](bool checked){if(checked)cur_dim = 1;});
        connect(ui->glAxiView,qOverload<bool>(&QPushButton::toggled),this,[this](bool checked){if(checked)cur_dim = 2;});

        auto slice_view_toggled = [this](bool checked){
        if(checked)
            {
                auto keep_dim = cur_dim;
                cur_dim = 3; // disable updates
                ui->SlicePos->setRange(0,current_slice->dim[keep_dim]-1);
                cur_dim = keep_dim;
                ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
                glWidget->set_view(cur_dim);
                glWidget->update();
                glWidget->setFocus();
                slice_need_update = true;
                set_data("roi_layout",0);}
        };

        connect(ui->glSagView,&QPushButton::toggled,this,slice_view_toggled);
        connect(ui->glCorView,&QPushButton::toggled,this,slice_view_toggled);
        connect(ui->glAxiView,&QPushButton::toggled,this,slice_view_toggled);

        connect(ui->show_3view,qOverload<bool>(&QToolButton::toggled),this,[this](bool checked){if(checked){set_data("roi_layout",1);glWidget->update();slice_need_update = true;}});
        connect(ui->show_mosaic,&QToolButton::clicked,this,[this](void){set_data("roi_layout",std::min<int>(7,std::max<int>(2,(*this)["roi_layout"].toInt()+1)));glWidget->update();slice_need_update = true;});


        connect(ui->tool0,&QPushButton::pressed,this,[this](void){scene.sel_mode = 0;scene.setFocus();});
        connect(ui->tool1,&QPushButton::pressed,this,[this](void){scene.sel_mode = 1;scene.setFocus();});
        connect(ui->tool2,&QPushButton::pressed,this,[this](void){scene.sel_mode = 2;scene.setFocus();});
        connect(ui->tool3,&QPushButton::pressed,this,[this](void){scene.sel_mode = 3;scene.setFocus();});
        connect(ui->tool4,&QPushButton::pressed,this,[this](void){scene.sel_mode = 4;scene.setFocus();});
        connect(ui->tool5,&QPushButton::pressed,this,[this](void){scene.sel_mode = 5;scene.setFocus();});
        connect(ui->tool6,&QPushButton::pressed,this,[this](void){scene.sel_mode = 6;slice_need_update = true;scene.setFocus();});
        connect(ui->zoom,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double arg1){if(float(arg1) == (*this)["roi_zoom"].toFloat())return;set_data("roi_zoom",arg1);slice_need_update = true;});


        auto roi_show_toggled = [this](bool checked)
        {
            auto button = qobject_cast<QToolButton*>(sender());
            auto name = button->objectName().toStdString();
            button->setChecked(checked);
            if(button->isChecked() ^ (*this)[name.c_str()].toBool())
            {
                if(name == "roi_ruler" && checked)
                    scene.show_grid = !scene.show_grid;
                set_data(name.c_str(),button->isChecked());
            }
            slice_need_update = true;
        };
        connect(ui->roi_fiber,&QToolButton::toggled,this,roi_show_toggled);
        connect(ui->roi_track,&QToolButton::toggled,this,roi_show_toggled);
        connect(ui->roi_ruler,&QToolButton::toggled,this,roi_show_toggled);
        connect(ui->roi_position,&QToolButton::toggled,this,roi_show_toggled);
        connect(ui->roi_label,&QToolButton::toggled,this,roi_show_toggled);
        connect(ui->roi_draw_edge,&QToolButton::toggled,this,roi_show_toggled);

        connect(new QShortcut(QKeySequence(Qt::Key_F1),this),&QShortcut::activated,this,[this](void){ui->roi_fiber->setChecked(!ui->roi_fiber->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F2),this),&QShortcut::activated,this,[this](void){ui->roi_track->setChecked(!ui->roi_track->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F3),this),&QShortcut::activated,this,[this](void){ui->roi_ruler->setChecked(!ui->roi_ruler->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F4),this),&QShortcut::activated,this,[this](void){ui->roi_position->setChecked(!ui->roi_position->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F5),this),&QShortcut::activated,this,[this](void){ui->roi_label->setChecked(!ui->roi_label->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F6),this),&QShortcut::activated,this,[this](void){ui->roi_draw_edge->setChecked(!ui->roi_draw_edge->isChecked());});
    }

    // regions
    {

        connect(regionWidget,&RegionTableWidget::need_update,this,[this](void){slice_need_update = true;});
        connect(regionWidget,&RegionTableWidget::itemSelectionChanged,this,[this](void){slice_need_update = true;});
        connect(regionWidget,SIGNAL(need_update()),glWidget,SLOT(update()));
        // actions
        connect(ui->actionUndo_Edit,SIGNAL(triggered()),regionWidget,SLOT(undo()));
        connect(ui->actionRedo_Edit,SIGNAL(triggered()),regionWidget,SLOT(redo()));
        connect(ui->region_up,SIGNAL(clicked()),regionWidget,SLOT(move_up()));
        connect(ui->region_down,SIGNAL(clicked()),regionWidget,SLOT(move_down()));
    }
    // View
    {

        connect(ui->actionSingle,&QAction::triggered, this,[this](void){glWidget->view_mode = GLWidget::view_mode_type::single;glWidget->update();});
        connect(ui->actionDouble,&QAction::triggered, this,[this](void){
            glWidget->view_mode = GLWidget::view_mode_type::two;
            glWidget->transformation_matrix2 = glWidget->transformation_matrix;
            glWidget->rotation_matrix2 = glWidget->rotation_matrix;
            glWidget->update();});
        connect(ui->actionStereoscopic,&QAction::triggered, this,[this](void){glWidget->view_mode = GLWidget::view_mode_type::stereo;glWidget->update();});
        connect(ui->action3D_Screen,&QAction::triggered,this,[this](void){QApplication::clipboard()->setImage(tipl::qt::get_bounding_box(glWidget->grab_image()));});
        connect(ui->action3D_Screen_Left_Right_Views,&QAction::triggered,this,[this](void){QApplication::clipboard()->setImage(glWidget->getLRView());});
        connect(ui->action3D_Screen_3_Views,&QAction::triggered,this,[this](void){QApplication::clipboard()->setImage(glWidget->get3View(0));});
        connect(ui->action3D_Screen_3_Views_Horizontal,&QAction::triggered,this,[this](void){QApplication::clipboard()->setImage(glWidget->get3View(1));});
        connect(ui->action3D_Screen_3_Views_Vertical,&QAction::triggered,this,[this](void){QApplication::clipboard()->setImage(glWidget->get3View(2));});
        connect(ui->action3D_Screen_Each_Tract,&QAction::triggered,this,[this](void)
        {
            bool ok = true;
            int col_count = QInputDialog::getInt(this,QApplication::applicationName(),"Column Count",5,1,50,1&ok);
            if(!ok)
                return;
            glWidget->copyToClipboardEach(tractWidget,uint32_t(col_count));
        });
        connect(ui->action3D_Screen_Each_Region,&QAction::triggered,this,[this](void)
        {
            bool ok = true;
            int col_count = QInputDialog::getInt(this,QApplication::applicationName(),"Column Count",5,1,50,1&ok);
            if(!ok)
                return;
            glWidget->copyToClipboardEach(regionWidget,uint32_t(col_count));
        });

        connect(ui->actionRecord_Video,SIGNAL(triggered()),glWidget,SLOT(record_video()));
        connect(ui->actionROI,&QAction::triggered,this,[this](void){scene.copyClipBoard();});


    }
    // Device
    {
        connect(deviceWidget,SIGNAL(need_update()),glWidget,SLOT(update()));
        connect(ui->actionNewDevice,SIGNAL(triggered()),deviceWidget,SLOT(newDevice()));

        connect(ui->actionOpenDevice,SIGNAL(triggered()),deviceWidget,SLOT(load_device()));
        connect(ui->actionSaveDevice,SIGNAL(triggered()),deviceWidget,SLOT(save_device()));
        connect(ui->actionSave_All_Device,SIGNAL(triggered()),deviceWidget,SLOT(save_all_devices()));

        connect(ui->actionDeleteDevice,SIGNAL(triggered()),deviceWidget,SLOT(delete_device()));
        connect(ui->actionDeleteAllDevices,SIGNAL(triggered()),deviceWidget,SLOT(delete_all_devices()));

        connect(ui->actionCopy_Device,SIGNAL(triggered()),deviceWidget,SLOT(copy_device()));
        connect(ui->actionCheck_All_Devices,SIGNAL(triggered()),deviceWidget,SLOT(check_all()));
        connect(ui->actionUncheck_All_Devices,SIGNAL(triggered()),deviceWidget,SLOT(uncheck_all()));

        connect(ui->actionAssign_Colors_For_Devices,SIGNAL(triggered()),deviceWidget,SLOT(assign_colors()));

        connect(ui->actionDetect_Electrodes,SIGNAL(triggered()),deviceWidget,SLOT(detect_electrodes()));

        connect(ui->actionLeads_to_ROI,SIGNAL(triggered()),deviceWidget,SLOT(lead_to_roi()));


    }
    // Tracts
    {
        connect(ui->perform_tracking,SIGNAL(clicked()),tractWidget,SLOT(start_tracking()));
        connect(ui->stop_tracking,SIGNAL(clicked()),tractWidget,SLOT(stop_tracking()));
        connect(tractWidget,&TractTableWidget::show_tracts,this,[this](void)
        {
            slice_need_update = true;
            regionWidget->tract_map_id = 0;
            glWidget->update();
        });
        connect(tractWidget,SIGNAL(cellChanged(int,int)),tractWidget,SLOT(cell_changed(int,int))); //update label
        connect(tractWidget,SIGNAL(itemSelectionChanged()),tractWidget,SLOT(show_report()));
        connect(glWidget,SIGNAL(edited()),tractWidget,SLOT(edit_tracts()));
        connect(glWidget,SIGNAL(region_edited()),glWidget,SLOT(update()));
        connect(glWidget,&GLWidget::region_edited,this,[this](void){slice_need_update = true;});

        connect(ui->track_up,SIGNAL(clicked()),tractWidget,SLOT(move_up()));
        connect(ui->track_down,SIGNAL(clicked()),tractWidget,SLOT(move_down()));


        connect(ui->actionSelect_Tracts,&QAction::triggered,this,[this](void){glWidget->setCursor(Qt::CrossCursor);glWidget->editing_option = GLWidget::selecting;tractWidget->edit_option = TractTableWidget::select;});
        connect(ui->actionDelete_Tracts,&QAction::triggered,this,[this](void){glWidget->setCursor(Qt::CrossCursor);glWidget->editing_option = GLWidget::selecting;tractWidget->edit_option = TractTableWidget::del;});
        connect(ui->actionCut_Tracts,&QAction::triggered,   this,[this](void){glWidget->setCursor(Qt::CrossCursor);glWidget->editing_option = GLWidget::selecting;tractWidget->edit_option = TractTableWidget::cut;});
        connect(ui->actionPaint_Tracts,&QAction::triggered, this,[this](void){glWidget->setCursor(Qt::CrossCursor);glWidget->editing_option = GLWidget::selecting;tractWidget->edit_option = TractTableWidget::paint;});
        connect(ui->actionMove_Objects,&QAction::triggered, this,[this](void){glWidget->setCursor(Qt::CrossCursor);glWidget->editing_option = GLWidget::moving;});

    } 
    {
        connect(ui->actionRestore_window_layout,&QAction::triggered, this,[this](void){restoreGeometry(default_geo);restoreState(default_state);});
    }
    // Option
    {
        connect(ui->actionRestore_Settings,&QAction::triggered, this,[this](void){command({"restore_rendering"});});
        connect(ui->actionRestore_Tracking_Settings,&QAction::triggered, this,[this](void){command({"restore_tracking"});});
        connect(ui->reset_rendering,&QPushButton::clicked, this,[this](void)
        {
            command({"restore_tracking"});
            command({"restore_rendering"});
        });

    }

    {
        connect(ui->SliceModality,qOverload<int>(&QComboBox::currentIndexChanged),this,[this](int index)
        {
            if(!command({"set_slice"}))
            {
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
                if(index)
                    ui->SliceModality->setCurrentIndex(0);
            }
        });
    }
    {
        connect(new QShortcut(QKeySequence(tr("Q", "X+")),this),&QShortcut::activated,this,[this](void){ui->glSagSlider->setValue(ui->glSagSlider->value()+1);});
        connect(new QShortcut(QKeySequence(tr("A", "X+")),this),&QShortcut::activated,this,[this](void){ui->glSagSlider->setValue(ui->glSagSlider->value()-1);});
        connect(new QShortcut(QKeySequence(tr("W", "X+")),this),&QShortcut::activated,this,[this](void){ui->glCorSlider->setValue(ui->glCorSlider->value()+1);});
        connect(new QShortcut(QKeySequence(tr("S", "X+")),this),&QShortcut::activated,this,[this](void){ui->glCorSlider->setValue(ui->glCorSlider->value()-1);});
        connect(new QShortcut(QKeySequence(tr("E", "X+")),this),&QShortcut::activated,this,[this](void){ui->glAxiSlider->setValue(ui->glAxiSlider->value()+1);});
        connect(new QShortcut(QKeySequence(tr("D", "X+")),this),&QShortcut::activated,this,[this](void){ui->glAxiSlider->setValue(ui->glAxiSlider->value()-1);});

    }

    {
        foreach (QAction* action, findChildren<QAction*>())
            if(action->toolTip().startsWith("run "))
                connect(action,&QAction::triggered,this,[this,action](){run_command(action->toolTip().toStdString().substr(4));});
        foreach (QCheckBox* cb, findChildren<QCheckBox*>())
            if(cb->toolTip().startsWith("run "))
                connect(cb,qOverload<int>(&QCheckBox::stateChanged),this,[this,cb](int){run_command(cb->toolTip().toStdString().substr(4));});
        foreach (QPushButton* pb, findChildren<QPushButton*>())
            if(pb->toolTip().startsWith("run "))
                connect(pb,&QPushButton::pressed,this,[this,pb](){run_command(pb->toolTip().toStdString().substr(4));});
        foreach (QSlider* s, findChildren<QSlider*>())
            if(s->toolTip().startsWith("run "))
                connect(s,qOverload<int>(&QSlider::valueChanged),this,[this,s](int){run_command(s->toolTip().toStdString().substr(4));});

        updateSlicesMenu();
    }
    qApp->installEventFilter(this);
    // now begin visualization
    tipl::out() << "begin visualization" << std::endl;
    {
        ui->SliceModality->blockSignals(false);
        glWidget->no_update = false;
        scene.no_update = false;
        no_update = false;
        command({"set_slice"});
        ui->glAxiView->setChecked(true);
        if((*this)["orientation_convention"].toInt() == 1)
            glWidget->set_view(2);
    }
    tipl::out() << "GUI initialization complete" << std::endl;


    history.commands.clear();
    std::vector<std::string> cmd({"open_fib",handle->fib_file_name.c_str()});
    history.record(error_msg,cmd);
}

void tracking_window::closeEvent(QCloseEvent *event)
{
    for(size_t index = 0;index < tractWidget->tract_models.size();++index)
        if(!tractWidget->tract_models[index]->saved)
        {
            if (QMessageBox::question( this, QApplication::applicationName(),
                "Tractography not saved. Close?\n",QMessageBox::No | QMessageBox::Yes,QMessageBox::No) == QMessageBox::No)
            {
                event->ignore();
                return;
            }
            break;
        }

    QMainWindow::closeEvent(event);
    // clean up texture here when makeCurrent is still working
    glWidget->clean_up();

}
tracking_window::~tracking_window()
{
    for(size_t index = 0;index < tracking_windows.size();++index)
        if(tracking_windows[index] == this)
        {
            tracking_windows[index] = tracking_windows.back();
            tracking_windows.pop_back();
            break;
        }
    tractWidget->stop_tracking();
    tractWidget->command({"delete_all_tracts"});
    tractWidget->command({"delete_all_regions"});
    qApp->removeEventFilter(this);
    QSettings settings;
    settings.setValue("rendering_quality",ui->rendering_efficiency->currentIndex());
    delete ui;
    //tipl::out() << __FUNCTION__ << " " << __FILE__ << std::endl;
}
void tracking_window::report(QString string)
{
    ui->text_report->setText(string);
}
extern console_stream console;
bool tracking_window::eventFilter(QObject *obj, QEvent *event)
{
    bool has_info = false;
    tipl::vector<3> pos;
    // update slice here
    if(slice_need_update)
    {
        slice_need_update = false;
        scene.show_slice();
    }
    if(scene.complete_view_ready)
        scene.show_complete_slice();

    if (event->type() == QEvent::MouseMove && obj->isWidgetType())
    {
        if (obj == glWidget &&
            glWidget->editing_option == GLWidget::moving &&
                    (ui->glSagCheck->isChecked() ||
                     ui->glCorCheck->isChecked() ||
                     ui->glAxiCheck->isChecked()))
            has_info = glWidget->get_mouse_pos(static_cast<QMouseEvent*>(event)->pos(),pos);
        else
            if (obj->parent() == ui->graphicsView)
            {
                QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
                QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
                has_info = scene.to_3d_space(point.x(),point.y(),pos);
            }

        // for connectivity matrix
        if(connectivity_matrix.get() && connectivity_matrix->is_graphic_view(obj->parent()))
            connectivity_matrix->mouse_move(static_cast<QMouseEvent*>(event));

    }
    if(!has_info)
        return false;

    QString status = QString("LPS=(%1,%2,%3)")
            .arg(std::round(pos[0]*10.0)/10.0)
            .arg(std::round(pos[1]*10.0)/10.0)
            .arg(std::round(pos[2]*10.0)/10.0);

    if((handle->template_id == handle->matched_template_id && handle->is_mni && !handle->template_I.empty()) || !handle->s2t.empty())
    {
        tipl::vector<3,float> mni(pos);
        handle->sub2mni(mni);
        status += QString(" MNI=(%1,%2,%3)")
                .arg(std::round(mni[0]*10.0)/10.0)
                .arg(std::round(mni[1]*10.0)/10.0)
                .arg(std::round(mni[2]*10.0)/10.0);
    }
    else
        status += QString(" MNI=(click [atlas]...)");

    if(!current_slice->is_diffusion_space)
    {
        pos.to(current_slice->to_dif);
        status += QString(" %4=(%5,%6,%7)")
                .arg(ui->SliceModality->currentText())
                .arg(std::round(pos[0]*10.0)/10.0)
                .arg(std::round(pos[1]*10.0)/10.0)
                .arg(std::round(pos[2]*10.0)/10.0);
    }

    std::vector<float> data;
    pos.round();
    handle->get_voxel_information(pos[0],pos[1],pos[2], data);
    size_t data_index = 0;
    for(const auto& each : handle->slices)
        if(each->image_ready() && data_index < data.size())
        {
            status += QString(" %1=%2").arg(each->name.c_str()).arg(data[data_index]);
            ++data_index;
        }
    ui->statusbar->showMessage(status);
    return false;
}

float tracking_window::get_scene_zoom(std::shared_ptr<SliceModel> slice)
{
    float display_ratio = (*this)["roi_zoom"].toFloat();
    if(!current_slice->is_diffusion_space)
        display_ratio *= slice->vs[0]/handle->vs[0];
    display_ratio = std::min<float>(display_ratio,4096.0/handle->dim[0]);
    return display_ratio;
}


void tracking_window::move_slice_to(tipl::vector<3,float> slice_position)
{
    slice_position.round();
    for(int i = 0;i < 3; ++i)
    {
        if(slice_position[i] < 0)
            slice_position[i] = 0;
        if(slice_position[i] >= current_slice->dim[i]-1)
            slice_position[i] = 0;
    }
    current_slice->slice_pos = slice_position;

    ui->glSagSlider->setValue(slice_position[0]);
    ui->glCorSlider->setValue(slice_position[1]);
    ui->glAxiSlider->setValue(slice_position[2]);
    ui->glSagBox->setValue(slice_position[0]);
    ui->glCorBox->setValue(slice_position[1]);
    ui->glAxiBox->setValue(slice_position[2]);

    ui->SlicePos->setRange(0,current_slice->dim[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);

    glWidget->update_slice();
    slice_need_update = true;
}

void tracking_window::on_tracking_index_currentIndexChanged(int index)
{
    if(index < 0)
            return;
    handle->dir.set_tracking_index(index);
    float max_value = tipl::max_value(handle->dir.fa[0],handle->dir.fa[0]+handle->dim.size());
    renderWidget->setMinMax("fa_threshold",0.0,max_value*1.1f,max_value/50.0f);
    if(renderWidget->getData("fa_threshold").toFloat() != 0.0f)
        set_data("fa_threshold",
                 renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu);
    slice_need_update = true;
}

void tracking_window::keyPressEvent ( QKeyEvent * event )
{
    switch(event->key())
    {
        case Qt::Key_Left:
            glWidget->move_by(-1,0);
            break;
        case Qt::Key_Right:
            glWidget->move_by(1,0);
            break;
        case Qt::Key_Up:
            glWidget->move_by(0,-1);
            break;
        case Qt::Key_Down:
            glWidget->move_by(0,1);
            break;
        default:
            QWidget::keyPressEvent(event);
            return;
    }
    event->accept();
}

void tracking_window::set_roi_zoom(float zoom)
{
    ui->zoom->setValue(zoom);
}

void tracking_window::on_actionAuto_Rotate_triggered(bool checked)
{
    if(checked)
        {
            glWidget->time = std::chrono::high_resolution_clock::now();
            glWidget->last_time = 0;
            glWidget->rotate_timer.reset(new QTimer());
            glWidget->rotate_timer->setInterval(1);
            connect(glWidget->rotate_timer.get(), SIGNAL(timeout()), glWidget, SLOT(rotate()));
            glWidget->rotate_timer->start();
        }
    else
        glWidget->rotate_timer->stop();
}

QString tracking_window::get_save_file_name(QString title,QString file_name,QString file_type)
{
    return QFileDialog::getSaveFileName(this,title,file_name,file_type);
}

void tracking_window::on_rendering_efficiency_currentIndexChanged(int index)
{
    if(!renderWidget)
        return;
    switch(index)
    {
    case 0:
        set_data("line_smooth",0);
        set_data("point_smooth",0);

        set_data("tract_style",0);
        set_data("tract_visible_tract",10000);
        set_data("tract_tube_detail",0);

        break;
    case 1:
        set_data("line_smooth",0);
        set_data("point_smooth",0);

        set_data("tract_style",1);
        set_data("tract_visible_tract",25000);
        set_data("tract_tube_detail",1);
        break;
    case 2:
        set_data("line_smooth",1);
        set_data("point_smooth",1);

        set_data("tract_style",1);
        set_data("tract_visible_tract",100000);
        set_data("tract_tube_detail",3);
        break;
    }
    glWidget->update();
}




void tracking_window::updateSlicesMenu(void)
{
    auto new_item = [&](const std::string& each,const char* action)
    {
        QAction* Item = new QAction(this);
        Item->setText(QString("%1...").arg(each.c_str()));
        Item->setData(QString(each.c_str()));
        Item->setToolTip(action);
        Item->setVisible(true);
        connect(Item,&QAction::triggered,this,[this,Item](){run_command(Item->toolTip().toStdString().substr(4));});
        return Item;
    };
    ui->menuSave->clear();
    ui->menuE_xport->clear();
    // save along tract indices
    auto metrics_list = handle->get_index_list();
    for (const auto& each : metrics_list)
        ui->menuSave->addAction(new_item(each,"run save_tract_values"));

    // export slices
    metrics_list.push_back("fiber");
    if(handle->has_odfs())
        metrics_list.push_back("odfs");
    for (const auto& each : metrics_list)
        ui->menuE_xport->addAction(new_item(each,"run save_slice_image"));

    if(std::find(metrics_list.begin(),metrics_list.end(),"iso") != metrics_list.end())
    {
        ui->menuE_xport->addSeparator();
        ui->menuE_xport->addAction(new_item("Correct Bias Field","run correct_bias_field"));
    }

    // update options: color map
    {
        QStringList tract_index_list;
        for(auto each : handle->get_index_list())
            tract_index_list << each.c_str();
        renderWidget->setList("tract_color_metrics",tract_index_list);
        tract_index_list << "current tract";
        renderWidget->setList("region_color_metrics",tract_index_list);
    }
    // update dt metric menu
    dt_list.clear();
    dt_list << "zero";
    for (const auto& each : handle->get_index_list())
        dt_list << each.c_str();
    renderWidget->setList("dt_index1",dt_list);
    renderWidget->setList("dt_index2",dt_list);
}




void tracking_window::on_track_style_currentIndexChanged(int index)
{
    switch(index)
    {
        case 0: //Tube 1
            set_data("tract_style",1);
            set_data("bkg_color",-1);
            set_data("tract_alpha",1);
            set_data("tube_diameter",0.2);
            set_data("tract_light_option",1);
            set_data("tract_light_dir",2);
            set_data("tract_light_shading",3);
            set_data("tract_light_diffuse",6);
            set_data("tract_light_ambient",0);
            set_data("tract_light_specular",0);
            set_data("tract_specular",0);
            set_data("tract_shininess",0);
            set_data("tract_emission",0);
            set_data("tract_bend1",4);
            set_data("tract_bend2",5);
            set_data("tract_shader",7);
            break;
        case 1: //Tube 2
            set_data("tract_style",1);
            set_data("bkg_color",0);
            set_data("tract_alpha",1);
            set_data("tube_diameter",0.2);
            set_data("tract_light_option",1);
            set_data("tract_light_dir",2);
            set_data("tract_light_shading",2);
            set_data("tract_light_diffuse",4);
            set_data("tract_light_ambient",0);
            set_data("tract_light_specular",0);
            set_data("tract_specular",0);
            set_data("tract_shininess",0);
            set_data("tract_emission",0);
            set_data("tract_bend1",1);
            set_data("tract_bend2",5);
            set_data("tract_shader",8);
            break;
        case 2:
            set_data("tract_style",0);
            set_data("tract_line_width",2.0f);
            set_data("bkg_color",-1);
            set_data("tract_alpha",1);
            set_data("tract_bend1",4);
            set_data("tract_bend2",5);
            set_data("tract_shader",7);
            break;
    }
    glWidget->update();
}

void tracking_window::on_SlicePos_valueChanged(int value)
{
    if(cur_dim ==0)
    {
        if(ui->glSagSlider->value() != value)
            ui->glSagSlider->setValue(value);
    }
    if(cur_dim ==1)
    {
        if(ui->glCorSlider->value() != value)
            ui->glCorSlider->setValue(value);
    }
    if(cur_dim ==2)
    {
        if(ui->glAxiSlider->value() != value)
            ui->glAxiSlider->setValue(value);
    }
}

float tracking_window::get_fa_threshold(void)
{
    float threshold = renderWidget->getData("fa_threshold").toFloat();
    if(threshold == 0.0f)
        threshold = renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu;
    return threshold;
}

void tracking_window::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void tracking_window::dropEvent(QDropEvent *event)
{
    event->acceptProposedAction();
    QList<QUrl> droppedUrls = event->mimeData()->urls();
    QStringList tracts,regions,slices;
    for(auto each : droppedUrls)
    {
        auto file_name = each.toLocalFile();
        if(file_name.endsWith("tt.gz"))
            tracts << file_name;
        if(file_name.endsWith("nii.gz") || file_name.endsWith("nii"))
        {
            tipl::io::gz_nifti nii;
            if(nii.load_from_file(file_name.toStdString()))
            {
                tipl::image<3> I;
                nii >> I;
                if(tipl::is_label_image(I))
                    regions << file_name;
                else
                    command({"add_slice",file_name.toStdString()});
            }
        }
    }
    for(auto each : tracts)
        if(!command({"open_tract",each.toStdString()}))
        {
            if(!error_msg.empty())
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
            return;
        }
    for(auto each : regions)
        if(!command({"load_region",each.toStdString()}))
        {
            if(!error_msg.empty())
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
            return;
        }
}




void tracking_window::on_actionEdit_Slices_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get())
        return;
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->cur_image->I_float32 = slice->get_source();
    dialog->cur_image->shape = slice->dim;
    dialog->cur_image->vs = slice->vs;
    dialog->cur_image->T = slice->trans_to_mni;
    dialog->cur_image->pixel_type = variant_image::float32;
    dialog->slice = slice;
    dialog->init_image();
    dialog->show();
}

void tracking_window::on_actionCommand_History_triggered(){
    if(!command_dialog){
        command_dialog = new QDialog(this);
        command_dialog->setWindowTitle("Command History");
        command_dialog->resize(1024,600);
        auto layout = new QVBoxLayout(command_dialog);
        layout->addWidget(new QLabel("Select the commands to repeat", command_dialog));
        auto tableWidget = new QTableWidget(command_dialog);
        tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
        tableWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
        layout->addWidget(tableWidget);
        auto populate_table = [=](const std::vector<std::string>& lines){
            int cols = 4;
            tableWidget->clear();
            tableWidget->setRowCount(lines.size());
            tableWidget->setColumnCount(cols);
            QStringList headers{"Action"};
            for(int i = 1; i < cols; ++i) headers << QString("Parameter %1").arg(i);
            tableWidget->setHorizontalHeaderLabels(headers);
            for(int r = 0; r < lines.size(); ++r){
                auto items = QString::fromStdString(lines[r]).split(',');
                for(int c = 0; c < cols; ++c)
                    tableWidget->setItem(r, c, new QTableWidgetItem(c < items.size() ? items[c] : QString()));
            }
            tableWidget->resizeColumnsToContents();
        };
        auto get_sel = [=](){
            std::vector<std::string> sel;
            for(auto idx : tableWidget->selectionModel()->selectedRows()){
                QStringList items;
                for(int c = 0; c < tableWidget->columnCount(); ++c)
                    items << (tableWidget->item(idx.row(), c) ? tableWidget->item(idx.row(), c)->text() : "");
                auto line = items.join(',').toStdString();
                while(!line.empty() && line.back() == ',')
                    line.pop_back();
                sel.push_back(line);
            }
            return sel;
        };
        QStringList btns{"&Open...", "&Save...", "Reload...", "Repeat", "Apply to Others...", "Delete", "Close"};
        std::vector<QPushButton*> buttons;
        for(auto &b : btns) buttons.push_back(new QPushButton(b, command_dialog));
        auto openB = buttons[0], saveB = buttons[1], reloadB = buttons[2],
             repeatB = buttons[3], applyB = buttons[4], deleteB = buttons[5], closeB = buttons[6];
        reloadB->setObjectName("reloadButton");
        repeatB->setDisabled(true); deleteB->setDisabled(true); applyB->setDisabled(true);
        auto btnLayout = new QHBoxLayout;
        for(auto b : buttons) btnLayout->addWidget(b);
        layout->addLayout(btnLayout);
        connect(tableWidget, &QTableWidget::itemSelectionChanged, [=](){
            bool has = !tableWidget->selectionModel()->selectedRows().isEmpty();
            repeatB->setEnabled(has); deleteB->setEnabled(has); applyB->setEnabled(has);
        });
        connect(openB, &QPushButton::clicked, [=](){
            QString fn = QFileDialog::getOpenFileName(this, "Open text files",
                            QString::fromStdString(history.file_stem(false))+"_commands.csv",
                            "CSV files (*.csv);;All files (*)");
            if(!fn.isEmpty())
                populate_table(tipl::read_text_file(fn.toStdString()));

        });
        connect(saveB, &QPushButton::clicked, [=](){
            QString fn = QFileDialog::getSaveFileName(this, "Save text file",
                            QString::fromStdString(history.file_stem(false))+"_commands.csv",
                            "CSV files (*.csv);;All files (*)");
            if(!fn.isEmpty()){
                std::ofstream file(fn.toStdString());
                for(int r = 0; r < tableWidget->rowCount(); ++r){
                    QStringList items;
                    for(int c = 0; c < tableWidget->columnCount(); ++c)
                        items << (tableWidget->item(r, c) ? tableWidget->item(r, c)->text() : "");
                    file << items.join(',').toStdString() << '\n';
                }
                QMessageBox::information(this, QApplication::applicationName(), "File saved.");
            }
        });
        connect(reloadB, &QPushButton::clicked, [=](){ populate_table(history.commands); });
        connect(repeatB, &QPushButton::clicked, [=](){
            if(history.run(this, get_sel(), false))
                QMessageBox::information(this, QApplication::applicationName(), "Execution completed");
        });
        connect(applyB, &QPushButton::clicked, [=](){
            renderWidget->saveParameters();
            if(history.run(this, get_sel(), true))
                QMessageBox::information(this, QApplication::applicationName(), "Execution completed");
        });
        connect(deleteB, &QPushButton::clicked, [=](){
            auto rows = tableWidget->selectionModel()->selectedRows();
            std::vector<int> dels;
            for(auto idx : rows) dels.push_back(idx.row());
            std::sort(dels.rbegin(), dels.rend());
            for(auto r : dels) tableWidget->removeRow(r);
        });
        connect(closeB, &QPushButton::clicked, command_dialog, &QDialog::hide);
        populate_table(history.commands);
    }
    else {
        if(auto reloadB = command_dialog->findChild<QPushButton*>("reloadButton"))
            reloadB->click();
    }
    command_dialog->show();
}


void tracking_window::on_actionOpen_FIB_Directory_triggered()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(std::filesystem::path(handle->fib_file_name).parent_path().string().c_str()));
}

