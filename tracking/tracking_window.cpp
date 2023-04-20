#include <utility>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <QCompleter>
#include <QSplitter>
#include <QSettings>
#include <QClipboard>
#include <QShortcut>
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
#include "region/regiontablewidget.h"
#include "devicetablewidget.h"
#include <QApplication>
#include <QMouseEvent>
#include <QMessageBox>
#include "fib_data.hpp"
#include "manual_alignment.h"
#include "tract_report.hpp"
#include "color_bar_dialog.hpp"
#include "connectivity_matrix_dialog.h"
#include "connectometry/group_connectometry_analysis.h"
#include "mapping/atlas.hpp"
#include "tracking/atlasdialog.h"
#include "libs/tracking/tracking_thread.hpp"
#include "regtoolbox.h"
#include "fib_data.hpp"

#include <filesystem>

extern std::vector<std::string> fa_template_list,track_atlas_file_list;
extern std::vector<tracking_window*> tracking_windows;
extern size_t auto_track_pos[7];
extern unsigned char auto_track_rgb[6][3];               // projection

static QByteArray default_geo,default_state;

void show_info_dialog(const std::string& title,const std::string& result)
{
    QMessageBox msgBox;
    msgBox.setText(title.c_str());
    msgBox.setDetailedText(result.c_str());
    msgBox.setStandardButtons(QMessageBox::Ok|QMessageBox::Save);
    msgBox.setDefaultButton(QMessageBox::Ok);
    QPushButton *copyButton = msgBox.addButton("Copy To Clipboard", QMessageBox::ActionRole);
    if(msgBox.exec() == QMessageBox::Save)
    {
        QString filename;
        filename = QFileDialog::getSaveFileName(0,"Save as","report.txt","Text files (*.txt);;All files|(*)");
        if(filename.isEmpty())
            return;
        std::ofstream out(filename.toStdString().c_str());
        out << result.c_str();
    }
    if (msgBox.clickedButton() == copyButton)
        QApplication::clipboard()->setText(result.c_str());
}

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

tracking_window::tracking_window(QWidget *parent,std::shared_ptr<fib_data> new_handle) :
        QMainWindow(parent),ui(new Ui::tracking_window),scene(*this),handle(new_handle),work_path(QFileInfo(new_handle->fib_file_name.c_str()).absolutePath()+"/")
{
    tipl::progress prog("initializing tracking GUI");
    tipl::out() << "initiate image/slices" << std::endl;
    fib_data& fib = *new_handle;
    for (unsigned int index = 0;index < fib.view_item.size(); ++index)
        slices.push_back(std::make_shared<SliceModel>(handle.get(),index));
    current_slice = slices[0];

    ui->setupUi(this);
    ui->thread_count->setValue(std::thread::hardware_concurrency() >> 1);

    // setup GUI
    {
        {
            tipl::out() << "create GUI objects" << std::endl;
            scene.statusbar = ui->statusbar;
            setGeometry(10,10,800,600);
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

            color_bar.reset(new color_bar_dialog(this)); // need to initiate after glwidget for tract rendering
        }
        {
            tipl::out() << "recall previous settings" << std::endl;
            QSettings settings;
            if(!default_geo.size())
                default_geo = saveGeometry();
            if(!default_state.size())
                default_state = saveState();
            ui->TractWidgetHolder->show();
            ui->renderingWidgetHolder->show();
            ui->ROIdockWidget->show();
            ui->regionDockWidget->show();
        }
        {
            ui->zoom->setValue((*this)["roi_zoom"].toFloat());
            ui->show_edge->setChecked((*this)["roi_draw_edge"].toBool());
            ui->show_track->setChecked((*this)["roi_track"].toBool());
            ui->show_3view->setChecked((*this)["roi_layout"].toBool());
            ui->show_r->setChecked((*this)["roi_label"].toBool());
            ui->show_position->setChecked((*this)["roi_position"].toBool());
            ui->show_ruler->setChecked((*this)["roi_ruler"].toBool());
            ui->show_fiber->setChecked((*this)["roi_fiber"].toBool());
            if(handle->dim[0] > 80)
                ui->zoom_3d->setValue(80.0/(float)std::max<int>(std::max<int>(handle->dim[0],handle->dim[1]),handle->dim[2]));
        }
        // Enabled/disable GUIs
        {
            if(!handle->trackable)
            {
                ui->perform_tracking->hide();
                ui->stop_tracking->hide();
                ui->show_fiber->setChecked(false);
                ui->show_fiber->hide();
                ui->enable_auto_track->hide();
            }            
        }
        tipl::out() << "initialize slices" << std::endl;
        {
            ui->SliceModality->clear();
            for (unsigned int index = 0;index < fib.view_item.size(); ++index)
                ui->SliceModality->addItem(fib.view_item[index].name.c_str());
            updateSlicesMenu();
        }
        tipl::out() << "prepare template and atlases" << std::endl;
        {
            populate_templates(ui->template_box,handle->template_id);
            if(handle->is_mni)
            {
                if(std::filesystem::exists(handle->t1w_template_file_name.c_str()))
                    addSlices(QStringList() << QString(handle->t1w_template_file_name.c_str()),"t1w",true);
                if(std::filesystem::exists(handle->wm_template_file_name.c_str()))
                    addSlices(QStringList() << QString(handle->wm_template_file_name.c_str()),"wm",true);
            }
            // setup fa threshold
            {
                QStringList tracking_index_list;
                for(size_t index = 0;index < handle->dir.index_name.size();++index)
                    if(handle->dir.index_name[index].find("dec_") != 0 &&
                       handle->dir.index_name[index].find("inc_") != 0)
                        tracking_index_list.push_back(handle->dir.index_name[index].c_str());
                renderWidget->setList("tracking_index",tracking_index_list);
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
            set_data("autotrack_tolerance",float(handle->min_length())*0.8f);

        }


        report(handle->report.c_str());

        // provide automatic tractography
        {
            ui->target->setVisible(false);
            ui->target_label->setVisible(false);
        }

    }

    tipl::out() << "connect signal and slots " << std::endl;
    // opengl
    {
        connect(ui->glSagSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glCorSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glAxiSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glSagCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(update()));
        connect(ui->glCorCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(update()));
        connect(ui->glAxiCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(update()));

        connect(ui->max_color_gl,SIGNAL(clicked()),this,SLOT(change_contrast()));
        connect(ui->min_color_gl,SIGNAL(clicked()),this,SLOT(change_contrast()));

        connect(ui->actionSave_Screen,SIGNAL(triggered()),glWidget,SLOT(catchScreen()));
        connect(ui->actionSave_3D_screen_in_high_resolution,SIGNAL(triggered()),glWidget,SLOT(catchScreen2()));
        connect(ui->actionLoad_Camera,SIGNAL(triggered()),glWidget,SLOT(loadCamera()));
        connect(ui->actionSave_Camera,SIGNAL(triggered()),glWidget,SLOT(saveCamera()));
        connect(ui->actionSave_Rotation_Images,SIGNAL(triggered()),glWidget,SLOT(saveRotationSeries()));
        connect(ui->actionSave_3D_screen_in_3_views,SIGNAL(triggered()),glWidget,SLOT(save3ViewImage()));
        connect(ui->action3D_Screen,SIGNAL(triggered()),glWidget,SLOT(copyToClipboard()));
        connect(ui->action3D_Screen_Each_Tract,SIGNAL(triggered()),glWidget,SLOT(copyToClipboardEachTract()));
        connect(ui->action3D_Screen_Each_Region,SIGNAL(triggered()),glWidget,SLOT(copyToClipboardEachRegion()));
        connect(ui->actionRecord_Video,SIGNAL(triggered()),glWidget,SLOT(record_video()));


        connect(ui->reset_rendering,SIGNAL(clicked()),this,SLOT(on_actionRestore_Settings_triggered()));
        connect(ui->reset_rendering,SIGNAL(clicked()),this,SLOT(on_actionRestore_Tracking_Settings_triggered()));


        connect(ui->actionFull,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionRight,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionLeft,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionLower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionUpper,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionAnterior,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionPosterior,SIGNAL(triggered()),this,SLOT(stripSkull()));

        connect(ui->actionRight_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionLeft_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionAnterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionPosterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionRight_Anterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionLeft_Anterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionRight_Posterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));
        connect(ui->actionLeft_Posterior_Lower,SIGNAL(triggered()),this,SLOT(stripSkull()));


        connect(ui->actionFull,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionRight,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLeft,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionUpper,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionAnterior,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionPosterior,SIGNAL(triggered()),glWidget,SLOT(addSurface()));

        connect(ui->actionRight_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLeft_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionAnterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionPosterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionRight_Anterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLeft_Anterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionRight_Posterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLeft_Posterior_Lower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));

        connect(ui->actionInsert_T1_T2,SIGNAL(triggered()),this,SLOT(on_addSlices_clicked()));

    }
    // scene view
    {

        connect(&scene,SIGNAL(need_update()),this,SLOT(update_scene_slice()));
        connect(&scene,SIGNAL(need_update()),glWidget,SLOT(update()));

        connect(ui->actionAxial_View,SIGNAL(triggered()),this,SLOT(on_glAxiView_clicked()));
        connect(ui->actionCoronal_View,SIGNAL(triggered()),this,SLOT(on_glCorView_clicked()));
        connect(ui->actionSagittal_view,SIGNAL(triggered()),this,SLOT(on_glSagView_clicked()));


        connect(ui->actionSave_ROI_Screen,SIGNAL(triggered()),&scene,SLOT(catch_screen()));

    }

    // regions
    {

        connect(regionWidget,SIGNAL(need_update()),this,SLOT(update_scene_slice()));
        connect(regionWidget,SIGNAL(itemSelectionChanged()),this,SLOT(update_scene_slice()));
        connect(regionWidget,SIGNAL(need_update()),glWidget,SLOT(update()));

        connect(ui->actionNewRegion,SIGNAL(triggered()),regionWidget,SLOT(new_region()));
        connect(ui->actionOpenRegion,SIGNAL(triggered()),regionWidget,SLOT(load_region()));
        connect(ui->actionOpen_MNI_Region,SIGNAL(triggered()),regionWidget,SLOT(load_mni_region()));
        connect(ui->actionSaveRegionAs,SIGNAL(triggered()),regionWidget,SLOT(save_region()));
        connect(ui->actionSave_All_Regions_As,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions()));
        connect(ui->actionSave_All_Regions_As_Multiple_Files,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions_to_dir()));
        connect(ui->actionSave_All_Regions_as_4D_NIFTI,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions_to_4dnifti()));
        connect(ui->actionSave_Voxel_Data_As,SIGNAL(triggered()),regionWidget,SLOT(save_region_info()));
        connect(ui->actionDeleteRegion,SIGNAL(triggered()),regionWidget,SLOT(delete_region()));
        connect(ui->actionDeleteRegionAll,SIGNAL(triggered()),regionWidget,SLOT(delete_all_region()));

        connect(ui->actionCopy_Region,SIGNAL(triggered()),regionWidget,SLOT(copy_region()));

        // actions
        connect(ui->actionUndo_Edit,SIGNAL(triggered()),regionWidget,SLOT(undo()));
        connect(ui->actionRedo_Edit,SIGNAL(triggered()),regionWidget,SLOT(redo()));
        connect(ui->actionShift_X,SIGNAL(triggered()),regionWidget,SLOT(action_shiftx()));
        connect(ui->actionShift_X_2,SIGNAL(triggered()),regionWidget,SLOT(action_shiftnx()));
        connect(ui->actionShift_Y,SIGNAL(triggered()),regionWidget,SLOT(action_shifty()));
        connect(ui->actionShift_Y_2,SIGNAL(triggered()),regionWidget,SLOT(action_shiftny()));
        connect(ui->actionShift_Z,SIGNAL(triggered()),regionWidget,SLOT(action_shiftz()));
        connect(ui->actionShift_Z_2,SIGNAL(triggered()),regionWidget,SLOT(action_shiftnz()));

        connect(ui->actionFlip_X,SIGNAL(triggered()),regionWidget,SLOT(action_flipx()));
        connect(ui->actionFlip_Y,SIGNAL(triggered()),regionWidget,SLOT(action_flipy()));
        connect(ui->actionFlip_Z,SIGNAL(triggered()),regionWidget,SLOT(action_flipz()));

        connect(ui->actionThreshold,SIGNAL(triggered()),regionWidget,SLOT(action_threshold()));
        connect(ui->actionThreshold_2,SIGNAL(triggered()),regionWidget,SLOT(action_threshold_current()));

        connect(ui->actionSmoothing,SIGNAL(triggered()),regionWidget,SLOT(action_smoothing()));
        connect(ui->actionErosion,SIGNAL(triggered()),regionWidget,SLOT(action_erosion()));
        connect(ui->actionDilation,SIGNAL(triggered()),regionWidget,SLOT(action_dilation()));
        connect(ui->actionOpening,SIGNAL(triggered()),regionWidget,SLOT(action_opening()));
        connect(ui->actionClosing,SIGNAL(triggered()),regionWidget,SLOT(action_closing()));
        connect(ui->actionNegate,SIGNAL(triggered()),regionWidget,SLOT(action_negate()));
        connect(ui->actionDefragment,SIGNAL(triggered()),regionWidget,SLOT(action_defragment()));
        connect(ui->actionDilation_by_voxel,SIGNAL(triggered()),regionWidget,SLOT(action_dilation_by_voxel()));

        connect(ui->actionSeparate,SIGNAL(triggered()),regionWidget,SLOT(action_separate()));
        connect(ui->actionA_B,SIGNAL(triggered()),regionWidget,SLOT(action_A_B()));
        connect(ui->actionB_A,SIGNAL(triggered()),regionWidget,SLOT(action_B_A()));
        connect(ui->actionAB,SIGNAL(triggered()),regionWidget,SLOT(action_AB()));
        connect(ui->actionAll_To_First,SIGNAL(triggered()),regionWidget,SLOT(action_B2A()));
        connect(ui->actionBy_Name,SIGNAL(triggered()),regionWidget,SLOT(action_sort_name()));
        connect(ui->actionBy_Size,SIGNAL(triggered()),regionWidget,SLOT(action_sort_size()));
        connect(ui->actionBy_X,SIGNAL(triggered()),regionWidget,SLOT(action_sort_x()));
        connect(ui->actionBy_Y,SIGNAL(triggered()),regionWidget,SLOT(action_sort_y()));
        connect(ui->actionBy_Z,SIGNAL(triggered()),regionWidget,SLOT(action_sort_z()));
        connect(ui->actionMove_Slices_To_Current_Region,SIGNAL(triggered()),regionWidget,SLOT(move_slice_to_current_region()));

        connect(ui->actionMerge_All_2,SIGNAL(triggered()),regionWidget,SLOT(merge_all()));

        connect(ui->actionCheck_all_regions,SIGNAL(triggered()),regionWidget,SLOT(check_all()));
        connect(ui->actionUnckech_all_regions,SIGNAL(triggered()),regionWidget,SLOT(uncheck_all()));

        connect(ui->actionWhole_brain_seeding,SIGNAL(triggered()),regionWidget,SLOT(whole_brain()));
        connect(ui->actionLoad_Region_Color,SIGNAL(triggered()),regionWidget,SLOT(load_region_color()));
        connect(ui->actionSave_Region_Color,SIGNAL(triggered()),regionWidget,SLOT(save_region_color()));
        connect(ui->actionRegion_statistics,SIGNAL(triggered()),regionWidget,SLOT(show_statistics()));

        connect(ui->region_up,SIGNAL(clicked()),regionWidget,SLOT(move_up()));
        connect(ui->region_down,SIGNAL(clicked()),regionWidget,SLOT(move_down()));
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



    }
    // tracts
    {
        connect(ui->perform_tracking,SIGNAL(clicked()),tractWidget,SLOT(start_tracking()));
        connect(ui->stop_tracking,SIGNAL(clicked()),tractWidget,SLOT(stop_tracking()));
        connect(tractWidget,SIGNAL(show_tracts()),glWidget,SLOT(update()));
        connect(tractWidget,SIGNAL(show_tracts()),this,SLOT(update_scene_slice()));
        connect(tractWidget,SIGNAL(cellChanged(int,int)),glWidget,SLOT(update())); //update label
        connect(tractWidget,SIGNAL(itemSelectionChanged()),tractWidget,SLOT(show_report()));
        connect(glWidget,SIGNAL(edited()),tractWidget,SLOT(edit_tracts()));
        connect(glWidget,SIGNAL(region_edited()),glWidget,SLOT(update()));
        connect(glWidget,SIGNAL(region_edited()),this,SLOT(update_scene_slice()));

        connect(ui->actionFilter_by_ROI,SIGNAL(triggered()),tractWidget,SLOT(filter_by_roi()));

        connect(ui->actionOpenTract,SIGNAL(triggered()),tractWidget,SLOT(load_tracts()));
        connect(ui->actionOpen_Tracts_Label,SIGNAL(triggered()),tractWidget,SLOT(load_tract_label()));
        connect(ui->actionMerge_All,SIGNAL(triggered()),tractWidget,SLOT(merge_all()));
        connect(ui->actionMerge_Tracts_by_Name,SIGNAL(triggered()),tractWidget,SLOT(merge_track_by_name()));
        connect(ui->actionCopyTrack,SIGNAL(triggered()),tractWidget,SLOT(copy_track()));
        connect(ui->actionSort_Tracts_By_Names,SIGNAL(triggered()),tractWidget,SLOT(sort_track_by_name()));


        connect(ui->actionFlip_X_2,SIGNAL(triggered()),tractWidget,SLOT(flipx()));
        connect(ui->actionFlip_Y_2,SIGNAL(triggered()),tractWidget,SLOT(flipy()));
        connect(ui->actionFlip_Z_2,SIGNAL(triggered()),tractWidget,SLOT(flipz()));


        connect(ui->actionCheck_all_tracts,SIGNAL(triggered()),tractWidget,SLOT(check_all()));
        connect(ui->actionUncheck_all_tracts,SIGNAL(triggered()),tractWidget,SLOT(uncheck_all()));


        connect(ui->actionDeleteTract,SIGNAL(triggered()),tractWidget,SLOT(delete_tract()));
        connect(ui->actionDeleteTractAll,SIGNAL(triggered()),tractWidget,SLOT(delete_all_tract()));
        connect(ui->actionDelete_By_Length,SIGNAL(triggered()),tractWidget,SLOT(delete_by_length()));
        connect(ui->actionDelete_Branches,SIGNAL(triggered()),tractWidget,SLOT(delete_branches()));


        connect(ui->actionRemove_Repeated_Tracks,SIGNAL(triggered()),tractWidget,SLOT(delete_repeated()));
        connect(ui->actionSeparate_Deleted,SIGNAL(triggered()),tractWidget,SLOT(separate_deleted_track()));
        connect(ui->actionReconnect_Tracts,SIGNAL(triggered()),tractWidget,SLOT(reconnect_track()));
        connect(ui->actionResample_Step_Size,SIGNAL(triggered()),tractWidget,SLOT(resample_step_size()));

        connect(ui->actionOpen_Colors,SIGNAL(triggered()),tractWidget,SLOT(load_tracts_color()));
        connect(ui->actionOpen_Tract_Property,SIGNAL(triggered()),tractWidget,SLOT(load_tracts_value()));
        connect(ui->actionSave_Tracts_Colors_As,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_color_as()));
        connect(ui->actionAssign_Colors_For_Each,SIGNAL(triggered()),tractWidget,SLOT(assign_colors()));
        connect(ui->actionOpen_Cluster_Colors,SIGNAL(triggered()),tractWidget,SLOT(open_cluster_color()));
        connect(ui->actionSave_Cluster_Colors,SIGNAL(triggered()),tractWidget,SLOT(save_cluster_color()));

        connect(ui->actionUndo,SIGNAL(triggered()),tractWidget,SLOT(undo_tracts()));
        connect(ui->actionRedo,SIGNAL(triggered()),tractWidget,SLOT(redo_tracts()));
        connect(ui->actionTrim,SIGNAL(triggered()),tractWidget,SLOT(trim_tracts()));

        connect(ui->actionSet_Color,SIGNAL(triggered()),tractWidget,SLOT(set_color()));

        connect(ui->actionK_means_Clustering,SIGNAL(triggered()),tractWidget,SLOT(clustering_kmeans()));
        connect(ui->actionEM_Clustering,SIGNAL(triggered()),tractWidget,SLOT(clustering_EM()));
        connect(ui->actionHierarchical,SIGNAL(triggered()),tractWidget,SLOT(clustering_hie()));
        connect(ui->actionOpen_Cluster_Labels,SIGNAL(triggered()),tractWidget,SLOT(open_cluster_label()));
        connect(ui->actionRecognize_Clustering,SIGNAL(triggered()),tractWidget,SLOT(recognize_and_cluster()));
        connect(ui->actionRecognize_and_Rename,SIGNAL(triggered()),tractWidget,SLOT(recognize_rename()));


        //setup menu
        connect(ui->actionSaveTractAs,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_as()));
        connect(ui->actionSave_VMRL,SIGNAL(triggered()),tractWidget,SLOT(save_vrml_as()));
        connect(ui->actionSave_All_Tracts_As,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_as()));
        connect(ui->actionSave_All_Tracts_As_Multiple_Files,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_to_dir()));

        connect(ui->actionSave_End_Points_As,SIGNAL(triggered()),tractWidget,SLOT(save_end_point_as()));
        connect(ui->actionSave_End_Points_All_Tracts_As,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_end_point_as()));
        connect(ui->actionSave_Endpoints_in_Current_Mapping,SIGNAL(triggered()),tractWidget,SLOT(save_transformed_endpoints()));

        connect(ui->actionSave_Tracts_in_Template_Space,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_in_template()));
        connect(ui->actionSave_Tracts_in_Current_Mapping,SIGNAL(triggered()),tractWidget,SLOT(save_transformed_tracts()));
        connect(ui->actionSave_Tracts_In_Native_Space,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_in_native()));

        connect(ui->actionSave_Tract_in_MNI_Coordinates,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_in_mni()));
        connect(ui->actionSave_Endpoints_in_MNI_Coordinates,SIGNAL(triggered()),tractWidget,SLOT(save_end_point_in_mni()));

        connect(ui->actionStatistics,SIGNAL(triggered()),tractWidget,SLOT(show_tracts_statistics()));
        connect(ui->actionRecognize_Current_Tract,SIGNAL(triggered()),tractWidget,SLOT(recog_tracks()));

        connect(ui->track_up,SIGNAL(clicked()),tractWidget,SLOT(move_up()));
        connect(ui->track_down,SIGNAL(clicked()),tractWidget,SLOT(move_down()));



    } 

    {
        auto* ShortcutQ = new QShortcut(QKeySequence(tr("Q", "X+")),this);
        auto* ShortcutA = new QShortcut(QKeySequence(tr("A", "X+")),this);
        auto* ShortcutW = new QShortcut(QKeySequence(tr("W", "X+")),this);
        auto* ShortcutS = new QShortcut(QKeySequence(tr("S", "X+")),this);
        auto* ShortcutE = new QShortcut(QKeySequence(tr("E", "X+")),this);
        auto* ShortcutD = new QShortcut(QKeySequence(tr("D", "X+")),this);
        connect(ShortcutQ,SIGNAL(activated()),this,SLOT(Move_Slice_X()));
        connect(ShortcutA,SIGNAL(activated()),this,SLOT(Move_Slice_X2()));
        connect(ShortcutW,SIGNAL(activated()),this,SLOT(Move_Slice_Y()));
        connect(ShortcutS,SIGNAL(activated()),this,SLOT(Move_Slice_Y2()));
        connect(ShortcutE,SIGNAL(activated()),this,SLOT(Move_Slice_Z()));
        connect(ShortcutD,SIGNAL(activated()),this,SLOT(Move_Slice_Z2()));

    }

    qApp->installEventFilter(this);
    // now begin visualization
    tipl::out() << "begin visualization" << std::endl;
    {
        glWidget->no_update = false;
        scene.no_show = false;
        on_glAxiView_clicked();
        if((*this)["orientation_convention"].toInt() == 1)
            glWidget->set_view(2);
        ui->SliceModality->setCurrentIndex(0);
    }
    tipl::out() << "GUI initialization complete" << std::endl;
}

void tracking_window::closeEvent(QCloseEvent *event)
{
    for(size_t index = 0;index < tractWidget->tract_models.size();++index)
        if(!tractWidget->tract_models[index]->saved)
        {
            if (QMessageBox::question( this, "DSI Studio",
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
            tracking_windows[index] = 0;
            break;
        }
    tractWidget->stop_tracking();
    tractWidget->command("delete_all_tract");
    regionWidget->delete_all_region();
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

bool tracking_window::command(QString cmd,QString param,QString param2)
{
    tipl::out() << "run " << cmd.toStdString() << " " << param.toStdString() << " " << param2.toStdString() << std::endl;
    if(glWidget->command(cmd,param,param2) ||
       scene.command(cmd,param,param2) ||
       tractWidget->command(cmd,param,param2) ||
       regionWidget->command(cmd,param,param2))
        return true;
    if(!tractWidget->error_msg.empty())
    {
        error_msg = tractWidget->error_msg;
        return false;
    }
    if(cmd == "presentation_mode")
    {
        ui->ROIdockWidget->hide();
        if(!regionWidget->rowCount())
            ui->regionDockWidget->hide();
        return true;
    }
    if(cmd == "save_workspace")
    {
        if(!QDir(param).exists())
            return true;
        if(tractWidget->rowCount())
        {
            if(!QDir(param+"/tracts").exists() && !QDir().mkdir(param+"/tracts"))
                return true;
            tractWidget->command("save_all_tracts_to_dir",param+"/tracts");
        }
        if(regionWidget->rowCount())
        {
            if(!QDir(param+"/regions").exists() && !QDir().mkdir(param+"/regions"))
                return true;
            regionWidget->command("save_all_regions_to_dir",param+"/regions");
        }
        if(deviceWidget->rowCount())
        {
            if(!QDir(param+"/devices").exists() && !QDir().mkdir(param+"/devices"))
                return true;
            deviceWidget->command("save_all_devices",param+"/devices/device.dv.csv");
        }
        CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
        if(reg_slice)
        {
            if(!QDir(param+"/slices").exists() && !QDir().mkdir(param+"/slices"))
            {
                error_msg = "cannot create slices directory at ";
                error_msg += (param+"/slices").toStdString();
                return false;
            }
            auto I = reg_slice->source_images;
            tipl::normalize_upper_lower(I);
            tipl::image<3,unsigned char> II(I.shape());
            std::copy(I.begin(),I.end(),II.begin());
            tipl::io::gz_nifti::save_to_file((param+"/slices/" + ui->SliceModality->currentText() + ".nii.gz").toStdString().c_str(),
                                   II,reg_slice->vs,reg_slice->trans,reg_slice->is_mni);
        }

        QDir::setCurrent(param);
        command("save_setting",param + "/setting.ini");
        command("save_camera",param + "/camera.txt");

        std::ofstream out((param + "/command.txt").toStdString().c_str());
        if(ui->glSagCheck->checkState())
            out << "move_slice 0 " << current_slice->slice_pos[0] << std::endl;
        else
            out << "slice_off 0" << std::endl;
        if(ui->glCorCheck->checkState())
            out << "move_slice 1 " << current_slice->slice_pos[1] << std::endl;
        else
            out << "slice_off 1" << std::endl;
        if(ui->glAxiCheck->checkState())
            out << "move_slice 2 " << current_slice->slice_pos[2] << std::endl;
        else
            out << "slice_off 2" << std::endl;

        return true;

    }
    if(cmd == "load_workspace")
    {
        tipl::progress prog("loading data");
        if(QDir(param+"/tracts").exists())
        {
            if(tractWidget->rowCount())
                tractWidget->delete_all_tract();
            QDir::setCurrent(param+"/tracts");
            QStringList tract_list = QDir().entryList(QStringList("*tt.gz"),QDir::Files|QDir::NoSymLinks);
            if(tract_list.size())
                tractWidget->load_tracts(tract_list);
        }

        prog(1,5);

        if(QDir(param+"/regions").exists())
        {
            if(regionWidget->rowCount())
                regionWidget->delete_all_region();
            QDir::setCurrent(param+"/regions");
            QStringList region_list = QDir().entryList(QStringList("*nii.gz"),QDir::Files|QDir::NoSymLinks);
            for(int i = 0;i < region_list.size();++i)
                regionWidget->command("load_region",region_list[i]);
        }

        prog(2,5);

        if(QDir(param+"/devices").exists())
        {
            if(deviceWidget->rowCount())
                deviceWidget->delete_all_devices();
            QDir::setCurrent(param+"/devices");
            QStringList device_list = QDir().entryList(QStringList("*dv.csv"),QDir::Files|QDir::NoSymLinks);
            for(int i = 0;i < device_list.size();++i)
                deviceWidget->load_device(device_list);
        }

        prog(3,5);

        if(QDir(param+"/slices").exists())
        {
            QDir::setCurrent(param+"/slices");
            QStringList slice_list = QDir().entryList(QStringList("*nii.gz"),QDir::Files|QDir::NoSymLinks);
            for(int i = 0;i < slice_list.size();++i)
                addSlices(QStringList(slice_list[i]),QFileInfo(slice_list[0]).baseName(),true);
        }

        prog(4,5);

        QDir::setCurrent(param);
        command("load_setting",param + "/setting.ini");
        command("load_camera",param + "/camera.txt");

        std::ifstream in((param + "/command.txt").toStdString().c_str());
        std::string line;
        while(std::getline(in,line))
        {
            std::istringstream in2(line);
            std::string cmd_,param_,param2_;
            in2 >> cmd_ >> param_ >> param2_;
            command(cmd_.c_str(),param_.empty() ? "":param_.c_str(),param2_.empty() ? "":param2_.c_str());
        }

        std::string readme;
        if(QFileInfo(param+"/README").exists())
        {
            std::ifstream in((param+"/README").toStdString().c_str());
            readme = std::string((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        }
        report((readme + "\r\nMethods\r\n" + handle->report).c_str());
        return true;
    }
    if(cmd == "load_setting")
    {
        QString filename = param;
        if(!QFileInfo(filename).exists())
        {
            error_msg = "cannot open file ";
            error_msg += param.toStdString();
            return false;
        }
        {
            QSettings s(filename, QSettings::IniFormat);
            QStringList param_list = renderWidget->treemodel->getParamList();
            for(int index = 0;index < param_list.size();++index)
                if(s.contains(param_list[index]))
                    set_data(param_list[index],s.value(param_list[index]));
            glWidget->update();
        }
        return true;
    }
    if(cmd == "save_setting")
    {
        QString filename = param;
        QSettings s(filename, QSettings::IniFormat);
        QStringList param_list = renderWidget->treemodel->getParamList();
        for(int index = 0;index < param_list.size();++index)
            s.setValue(param_list[index],renderWidget->getData(param_list[index]));
        return true;
    }
    if(cmd == "save_rendering_setting")
    {
        QString filename = param;
        QSettings s(filename, QSettings::IniFormat);
        QStringList param_list = renderWidget->treemodel->get_param_list("ROI");
        param_list += renderWidget->treemodel->get_param_list("Rendering");
        param_list += renderWidget->treemodel->get_param_list("Slice");
        param_list += renderWidget->treemodel->get_param_list("Tract");
        param_list += renderWidget->treemodel->get_param_list("Region");
        param_list += renderWidget->treemodel->get_param_list("Surface");
        param_list += renderWidget->treemodel->get_param_list("Device");
        param_list += renderWidget->treemodel->get_param_list("Label");
        param_list += renderWidget->treemodel->get_param_list("ODF");
        for(int index = 0;index < param_list.size();++index)
            s.setValue(param_list[index],renderWidget->getData(param_list[index]));
        return true;
    }
    if(cmd == "load_rendering_setting")
    {
        QString filename = param;
        if(QFileInfo(filename).exists())
        {
            QSettings s(filename, QSettings::IniFormat);
            QStringList param_list = renderWidget->treemodel->get_param_list("ROI");
            param_list += renderWidget->treemodel->get_param_list("Rendering");
            param_list += renderWidget->treemodel->get_param_list("Slice");
            param_list += renderWidget->treemodel->get_param_list("Tract");
            param_list += renderWidget->treemodel->get_param_list("Region");
            param_list += renderWidget->treemodel->get_param_list("Surface");
            param_list += renderWidget->treemodel->get_param_list("Device");
            param_list += renderWidget->treemodel->get_param_list("Label");
            param_list += renderWidget->treemodel->get_param_list("ODF");
            for(int index = 0;index < param_list.size();++index)
                if(s.contains(param_list[index]))
                    set_data(param_list[index],s.value(param_list[index]));
        }
        return true;
    }
    if(cmd == "save_tracking_setting")
    {
        QString filename = param;
        QSettings s(filename, QSettings::IniFormat);
        QStringList param_list = renderWidget->treemodel->get_param_list("Tracking");
        param_list += renderWidget->treemodel->get_param_list("Tracking_dT");
        param_list += renderWidget->treemodel->get_param_list("Tracking_adv");
        for(int index = 0;index < param_list.size();++index)
            s.setValue(param_list[index],renderWidget->getData(param_list[index]));
        return true;
    }
    if(cmd == "load_tracking_setting")
    {
        QString filename = param;
        if(QFileInfo(filename).exists())
        {
            QSettings s(filename, QSettings::IniFormat);
            QStringList param_list = renderWidget->treemodel->get_param_list("Tracking");
            param_list += renderWidget->treemodel->get_param_list("Tracking_dT");
            param_list += renderWidget->treemodel->get_param_list("Tracking_adv");
            for(int index = 0;index < param_list.size();++index)
                if(s.contains(param_list[index]))
                    set_data(param_list[index],s.value(param_list[index]));
        }
        return true;
    }
    if(cmd == "restore_rendering")
    {
        renderWidget->setDefault("ROI");
        renderWidget->setDefault("Rendering");
        renderWidget->setDefault("show_slice");
        renderWidget->setDefault("show_tract");
        renderWidget->setDefault("show_region");
        renderWidget->setDefault("show_device");
        renderWidget->setDefault("show_surface");
        renderWidget->setDefault("show_label");
        renderWidget->setDefault("show_odf");
        glWidget->update();
        return true;
    }
    if(cmd == "set_roi_view")
    {
        if(param == "0")
            on_glSagView_clicked();
        if(param == "1")
            on_glCorView_clicked();
        if(param == "2")
            on_glAxiView_clicked();
        return true;
    }
    if(cmd == "set_roi_view_index")
    {
        bool okay = true;
        int index = param.toInt(&okay);
        if(okay)
        {
            ui->SliceModality->setCurrentIndex(index);
            return true;
        }
        index = ui->SliceModality->findText(param);
        if(index == -1)
        {
            error_msg = "cannot find index: ";
            error_msg += param.toStdString();
            return false;
        }
        ui->SliceModality->setCurrentIndex(index);
        return true;
    }
    if(cmd == "set_roi_view_contrast")
    {
        ui->min_value_gl->setValue(param.toDouble());
        ui->max_value_gl->setValue(param2.toDouble());
        change_contrast();
        return true;
    }
    if(cmd == "set_slice_color")
    {
        ui->min_color_gl->setColor(param.toUInt());
        ui->max_color_gl->setColor(param2.toUInt());
        change_contrast();
        return true;
    }
    if(cmd == "set_param")
    {
        set_data(param,param2);
        glWidget->update();
        slice_need_update = true;
        return true;
    }
    if(cmd == "tract_to_region")
    {
        on_actionTracts_to_seeds_triggered();
        return true;
    }
    if(cmd == "set_region_color")
    {
        if(regionWidget->regions.empty())
            return true;
        regionWidget->regions.back()->region_render.color = param.toInt();
        glWidget->update();
        slice_need_update = true;
        return true;
    }
    if(cmd == "add_slice")
    {
        if(!addSlices(QStringList() << param,param,true))
        {
            error_msg = "cannot add slice ";
            error_msg += param.toStdString();
            return false;
        }
        tipl::out() << "register image to the DWI space" << std::endl;
        CustomSliceModel* cur_slice = (CustomSliceModel*)slices.back().get();
        cur_slice->wait();
        cur_slice->update_transform();
        return true;
    }
    error_msg = "unknown command: ";
    error_msg += cmd.toStdString();
    return false;
}

void tracking_window::update_scene_slice(void)
{
    slice_need_update = true;
}
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

    if (event->type() == QEvent::MouseMove)
    {
        if (obj == glWidget && glWidget->editing_option == GLWidget::none &&
                (ui->glSagCheck->checkState() ||
                 ui->glCorCheck->checkState() ||
                 ui->glAxiCheck->checkState()))
        {
            has_info = glWidget->get_mouse_pos(static_cast<QMouseEvent*>(event),pos);
        }
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

    QString status = QString("pos=(%1,%2,%3)")
            .arg(std::round(pos[0]*10.0)/10.0)
            .arg(std::round(pos[1]*10.0)/10.0)
            .arg(std::round(pos[2]*10.0)/10.0);

    if(handle->template_id == handle->matched_template_id || !handle->s2t.empty())
    {
        tipl::vector<3,float> mni(pos);
        handle->sub2mni(mni);
        status += QString(" MNI=(%1,%2,%3)")
                .arg(std::round(mni[0]*10.0)/10.0)
                .arg(std::round(mni[1]*10.0)/10.0)
                .arg(std::round(mni[2]*10.0)/10.0);
    }

    if(!current_slice->is_diffusion_space)
    {
        pos.to(current_slice->T);
        status += QString(" %4=(%5,%6,%7)")
                .arg(ui->SliceModality->currentText())
                .arg(std::round(pos[0]*10.0)/10.0)
                .arg(std::round(pos[1]*10.0)/10.0)
                .arg(std::round(pos[2]*10.0)/10.0);
    }

    std::vector<float> data;
    pos.round();
    handle->get_voxel_information(pos[0],pos[1],pos[2], data);
    for(unsigned int index = 0,data_index = 0;index < handle->view_item.size() && data_index < data.size();++index)
        if(handle->view_item[index].name != "color" && handle->view_item[index].image_ready)
        {
            status += QString(" %1=%2").arg(handle->view_item[index].name.c_str()).arg(data[data_index]);
            ++data_index;
        }
    ui->statusbar->showMessage(status);
    return false;
}
std::pair<int,int> tracking_window::get_dt_index_pair(void)
{
    int metric_i = renderWidget->getData("dt_index1").toInt(); //0: none
    int metric_j = renderWidget->getData("dt_index2").toInt(); //0: none
    return std::make_pair(metric_i-1,metric_j-1);
}
void tracking_window::set_tracking_param(ThreadData& tracking_thread)
{
    tracking_thread.param.threshold = renderWidget->getData("fa_threshold").toFloat();
    tracking_thread.param.dt_threshold = renderWidget->getData("dt_threshold").toFloat();
    tracking_thread.param.cull_cos_angle = std::cos(renderWidget->getData("turning_angle").toDouble() * 3.14159265358979323846 / 180.0);
    tracking_thread.param.step_size = renderWidget->getData("step_size").toFloat();
    tracking_thread.param.smooth_fraction = renderWidget->getData("smoothing").toFloat();
    tracking_thread.param.min_length = renderWidget->getData("min_length").toFloat();
    tracking_thread.param.max_length = std::max<float>(tracking_thread.param.min_length,renderWidget->getData("max_length").toDouble());

    tracking_thread.param.tracking_method = renderWidget->getData("tracking_method").toInt();
    tracking_thread.param.stop_by_tract = renderWidget->getData("tracking_plan").toInt();
    tracking_thread.param.check_ending = renderWidget->getData("check_ending").toInt() && (renderWidget->getData("dt_index1").toInt() == 0);
    tracking_thread.param.termination_count = renderWidget->getData("track_count").toInt();
    tracking_thread.param.default_otsu = renderWidget->getData("otsu_threshold").toFloat();
    tracking_thread.param.tip_iteration =
            // only used in automatic fiber tracking
            (ui->target->currentIndex() > 0 ||
            // or differential tractography
            renderWidget->getData("dt_index1").toInt() > 0)
            ? renderWidget->getData("tip_iteration").toInt() : 0;

}
float tracking_window::get_scene_zoom(std::shared_ptr<SliceModel> slice)
{
    float display_ratio = (*this)["roi_zoom"].toFloat();
    if(!current_slice->is_diffusion_space)
        display_ratio *= slice->vs[0]/handle->vs[0];
    display_ratio = std::min<float>(display_ratio,4096.0/handle->dim[0]);
    return display_ratio;
}

void tracking_window::SliderValueChanged(void)
{
    if(!no_update && current_slice->set_slice_pos(
            ui->glSagSlider->value(),
            ui->glCorSlider->value(),
            ui->glAxiSlider->value()))
    {
        ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
        if((*this)["roi_layout"].toInt() < 2) // >2 is mosaic, there is no need to update
            slice_need_update = true;
        glWidget->update();
    }
}


void tracking_window::on_tool0_pressed()
{
    scene.sel_mode = 0;
    scene.setFocus();

}

void tracking_window::on_tool1_pressed()
{
    scene.sel_mode = 1;
    scene.setFocus();

}

void tracking_window::on_tool2_pressed()
{
    scene.sel_mode = 2;
    scene.setFocus();

}

void tracking_window::on_tool3_pressed()
{
    scene.sel_mode = 3;
    scene.setFocus();
}

void tracking_window::on_tool4_clicked()
{
    scene.sel_mode = 4;
    scene.setFocus();
}
void tracking_window::on_tool5_pressed()
{
    scene.sel_mode = 5;
    scene.setFocus();
}

void tracking_window::on_tool6_pressed()
{
    scene.sel_mode = 6;
    slice_need_update = true;
    scene.setFocus();

}

void tracking_window::on_actionSelect_Tracts_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = GLWidget::selecting;
    tractWidget->edit_option = TractTableWidget::select;

}

void tracking_window::on_actionDelete_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = GLWidget::selecting;
    tractWidget->edit_option = TractTableWidget::del;
}

void tracking_window::on_actionCut_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = GLWidget::selecting;
    tractWidget->edit_option = TractTableWidget::cut;
}


void tracking_window::on_actionPaint_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = GLWidget::selecting;
    tractWidget->edit_option = TractTableWidget::paint;
}

void tracking_window::on_actionMove_Objects_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = GLWidget::moving;
}


void tracking_window::on_glSagView_clicked()
{
    cur_dim = 0;
    ui->SlicePos->setRange(0,current_slice->dim[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(0);
    glWidget->update();
    glWidget->setFocus();

    slice_need_update = true;
}

void tracking_window::on_glCorView_clicked()
{
    cur_dim = 1;
    ui->SlicePos->setRange(0,current_slice->dim[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(1);
    glWidget->update();
    glWidget->setFocus();
    slice_need_update = true;
}

void tracking_window::on_glAxiView_clicked()
{
    cur_dim = 2;
    ui->SlicePos->setRange(0,current_slice->dim[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(2);
    glWidget->update();
    glWidget->setFocus();
    slice_need_update = true;

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
void tracking_window::change_contrast()
{
    if(no_update)
        return;
    current_slice->set_contrast_range(ui->min_value_gl->value(),ui->max_value_gl->value());
    current_slice->set_contrast_color(ui->min_color_gl->color().rgb(),ui->max_color_gl->color().rgb());
    slice_need_update = true;
    glWidget->update_slice();
}

void tracking_window::on_actionEndpoints_to_seeding_triggered()
{
    std::vector<tipl::vector<3,short> > points1,points2;
    if(tractWidget->tract_models.empty() || tractWidget->currentRow() < 0)
        return;

    tractWidget->tract_models[size_t(tractWidget->currentRow())]->
            to_end_point_voxels(points1,points2,
                current_slice->is_diffusion_space ? tipl::matrix<4,4>(tipl::identity_matrix()) :current_slice->invT);

    regionWidget->begin_update();
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints1"));


    regionWidget->regions.back()->add_points(std::move(points1));

    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints2"));
    regionWidget->regions.back()->add_points(std::move(points2));
    regionWidget->end_update();
    slice_need_update = true;
    glWidget->update();
}

void tracking_window::on_actionTracts_to_seeds_triggered()
{
    if(tractWidget->tract_models.empty()|| tractWidget->currentRow() < 0)
        return;
    std::vector<tipl::vector<3,short> > points;
    tractWidget->tract_models[tractWidget->currentRow()]->to_voxel(points,
        current_slice->is_diffusion_space ? tipl::matrix<4,4>(tipl::identity_matrix()) : current_slice->invT);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text());
    regionWidget->regions.back()->add_points(std::move(points));
    slice_need_update = true;
    glWidget->update();
}

bool ask_TDI_options(int& rec,int& rec2)
{
    QMessageBox msgBox;
    msgBox.setText("Export directional color ? (BMP format only)");
    msgBox.setInformativeText("If grayscale or NIFTI format is preferred, select No.");
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No| QMessageBox::Cancel);
    msgBox.setDefaultButton(QMessageBox::Yes);
    rec = msgBox.exec();
    if(rec == QMessageBox::Cancel)
        return false;
    msgBox.setText("Export whole tracts or end points?");
    msgBox.setInformativeText("Yes: whole tracts, No: end points ");
    rec2 = msgBox.exec();
    if(rec2 == QMessageBox::Cancel)
        return false;
    return true;

}
void tracking_window::on_actionTDI_Diffusion_Space_triggered()
{
    tipl::matrix<4,4> tr;
    tr.identity();
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(handle->dim,handle->vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionTDI_Subvoxel_Diffusion_Space_triggered()
{
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    bool ok;
    int ratio = QInputDialog::getInt(this,
            "DSI Studio",
            "Input super-resolution ratio (e.g. 2, 3, or 4):",2,2,8,1,&ok);
    if(!ok)
        return;
    tipl::matrix<4,4> tr;
    tr.identity();
    tr[0] = tr[5] = tr[10] = ratio;
    tractWidget->export_tract_density(handle->dim*ratio,
                                      handle->vs/float(ratio),
                                      tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}

void tracking_window::on_actionTDI_Import_Slice_Space_triggered()
{
    tipl::matrix<4,4> tr = current_slice->invT;
    tipl::shape<3> geo = current_slice->dim;
    tipl::vector<3,float> vs = current_slice->vs;
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(geo,vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionRestore_window_layout_triggered()
{
    restoreGeometry(default_geo);
    restoreState(default_state);
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

void tracking_window::on_deleteSlice_clicked()
{
    if(dynamic_cast<CustomSliceModel*>(current_slice.get()) == nullptr)
        return;
    if(current_slice->is_overlay)
        on_is_overlay_clicked();
    int index = ui->SliceModality->currentIndex();
    handle->view_item.erase(handle->view_item.begin()+index);
    slices.erase(slices.begin()+index);
    for(uint32_t i = uint32_t(index);i < slices.size();++i)
        slices[i]->view_id--;
    ui->SliceModality->removeItem(index);
    updateSlicesMenu();
}


bool tracking_window::map_to_mni(void)
{
    if(!handle->map_to_mni())
    {
        QMessageBox::critical(this,"Error",handle->error_msg.c_str());
        return false;
    }
    return true;
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
            goto next;
    }
    event->accept();
    return;
    next:
    if(event->key() >= Qt::Key_1 && event->key() <= Qt::Key_9)
    {
        QSettings settings;
        event->accept();
        int key_num =  event->key()-Qt::Key_1;
        char key_str[3] = "F1";
        key_str[1] += key_num;
        if(event->modifiers() & Qt::AltModifier)
        {
            std::ostringstream out;
            out << ui->glSagSlider->value() << " "
                << ui->glCorSlider->value() << " "
                << ui->glAxiSlider->value() << " ";
            std::copy(glWidget->transformation_matrix.begin(),glWidget->transformation_matrix.end(),std::ostream_iterator<float>(out," "));
            settings.setValue(key_str,QString(out.str().c_str()));
            QMessageBox::information(this,"DSI Studio","View position and slice location memorized");
        }
        else
        {
            QString value = settings.value(key_str,"").toString();
            if(value == "")
                return;
            std::istringstream in(value.toStdString().c_str());
            int sag,cor,axi;
            in >> sag >> cor >> axi;
            std::vector<float> tran((std::istream_iterator<float>(in)),(std::istream_iterator<float>()));
            if(tran.size() != 16)
                return;
            std::copy(tran.begin(),tran.begin()+16,glWidget->transformation_matrix.begin());
            ui->glSagSlider->setValue(sag);
            ui->glCorSlider->setValue(cor);
            ui->glAxiSlider->setValue(axi);
            glWidget->update();
        }
    }
    if(event->isAccepted())
        return;
    QWidget::keyPressEvent(event);

}



void tracking_window::on_actionTract_Analysis_Report_triggered()
{
    if(!tact_report_imp.get())
        tact_report_imp.reset(new tract_report(this));
    tact_report_imp->show();
    tact_report_imp->refresh_report();
}

void tracking_window::on_actionConnectivity_matrix_triggered()
{
    if(!tractWidget->tract_models.size())
    {
        QMessageBox::information(this,"DSI Studio","Run fiber tracking first");
        return;
    }
    std::ostringstream out;
    if(tractWidget->currentRow() < tractWidget->tract_models.size())
        out << tractWidget->tract_models[tractWidget->currentRow()]->report.c_str() << std::endl;
    connectivity_matrix.reset(new connectivity_matrix_dialog(this,out.str().c_str()));
    connectivity_matrix->show();
}



void tracking_window::on_zoom_3d_valueChanged(double)
{
    glWidget->command("set_zoom",QString::number(ui->zoom_3d->value()));
}


void tracking_window::restore_3D_window()
{
    ui->centralLayout->addWidget(ui->main_widget);
    gLdock = nullptr;
}

void tracking_window::on_actionFloat_3D_window_triggered()
{
    if(gLdock)
        gLdock->showMaximized();
    else
    {
        int w = ui->main_widget->width();
        int h = ui->main_widget->height();
        gLdock = new QGLDockWidget(this);
        gLdock->setWindowTitle(windowTitle());
        gLdock->setAllowedAreas(Qt::NoDockWidgetArea);
        gLdock->setWidget(ui->main_widget);
        gLdock->setFloating(true);
        gLdock->show();
        gLdock->resize(w,h+44);
        connect(gLdock,SIGNAL(closedSignal()),this,SLOT(restore_3D_window()));
        QMessageBox::information(this,"DSI Studio","Float 3D window again to maximize it");
    }
}

void tracking_window::on_actionSave_tracking_parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",QFileInfo(windowTitle()).baseName()+"_tracking.ini","Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    command("save_tracking_setting",filename);
}

void tracking_window::on_actionLoad_tracking_parameters_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open INI files",QFileInfo(work_path).absolutePath(),"Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    command("load_tracking_setting",filename);
}

void tracking_window::on_actionSave_Rendering_Parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",QFileInfo(windowTitle()).baseName()+"_rendering.ini","Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    command("save_rendering_setting",filename);
}

void tracking_window::on_actionLoad_Rendering_Parameters_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open INI files",QFileInfo(work_path).absolutePath(),"Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    command("load_rendering_setting",filename);
}

void tracking_window::on_addRegionFromAtlas_clicked()
{
    if(handle->atlas_list.empty())
    {
        QMessageBox::critical(this,"ERROR","no atlas data");
        raise();
        return;
    }
    if(!map_to_mni())
        return;
    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,handle));
    atlas_dialog->exec();
}


void tracking_window::on_actionRestore_Settings_triggered()
{
    command("restore_rendering");
}


void tracking_window::on_actionRestore_Tracking_Settings_triggered()
{
    renderWidget->setDefault("Tracking");
    renderWidget->setDefault("Tracking_dT");
    renderWidget->setDefault("Tracking_adv");
    on_tracking_index_currentIndexChanged((*this)["tracking_index"].toInt());
    glWidget->update();
}

void tracking_window::set_roi_zoom(float zoom)
{
    ui->zoom->setValue(zoom);
}

void tracking_window::on_actionQuality_Assessment_triggered()
{
    float threshold = renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu;
    std::pair<float,float> result = evaluate_fib(handle->dim,threshold,handle->dir.fa,
                                                 [&](int pos,char fib)
                                                 {return handle->dir.get_fib(pos,fib);});
    std::ostringstream out;
    out << "Fiber coherence index: " << result.first << std::endl;
    out << "Fiber incoherent index: " << result.second << std::endl;
    show_info_dialog("Quality assessment",out.str().c_str());
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

void tracking_window::on_action3D_Screen_3_Views_triggered()
{
    QImage all;
    glWidget->get3View(all,0);
    QApplication::clipboard()->setImage(all);
}

void tracking_window::on_action3D_Screen_3_Views_Horizontal_triggered()
{
    QImage all;
    glWidget->get3View(all,1);
    QApplication::clipboard()->setImage(all);
}

void tracking_window::on_action3D_Screen_3_Views_Vertical_triggered()
{
    QImage all;
    glWidget->get3View(all,2);
    QApplication::clipboard()->setImage(all);
}

void tracking_window::on_actionROI_triggered()
{
    scene.copyClipBoard();
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

void tracking_window::on_actionCut_X_triggered()
{
    tractWidget->cut_by_slice(0,true);
}

void tracking_window::on_actionCut_X_2_triggered()
{
    tractWidget->cut_by_slice(0,false);
}

void tracking_window::on_actionCut_Y_triggered()
{
    tractWidget->cut_by_slice(1,true);
}

void tracking_window::on_actionCut_Y_2_triggered()
{
    tractWidget->cut_by_slice(1,false);
}

void tracking_window::on_actionCut_Z_triggered()
{
    tractWidget->cut_by_slice(2,true);
}

void tracking_window::on_actionCut_Z_2_triggered()
{
    tractWidget->cut_by_slice(2,false);
}

void tracking_window::stripSkull()
{
    if(!ui->SliceModality->currentIndex() || !handle->is_human_data || handle->is_mni)
        return;
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice || !reg_slice->skull_removed_images.empty())
        return;

    tipl::io::gz_nifti in1,in2;
    tipl::image<3> It,Iw,J(reg_slice->get_source());
    if(!in1.load_from_file(handle->t1w_template_file_name.c_str()) || !in1.toLPS(It))
        return;
    if(!in2.load_from_file(handle->mask_template_file_name.c_str()) || !in2.toLPS(Iw))
        return;
    if(QMessageBox::information(this,"DSI Studio",
                                QString("Does %1 need to remove tissues outside the brain?").
                                arg(ui->SliceModality->currentText()),
                                QMessageBox::Yes|QMessageBox::No) == QMessageBox::No)
        return;

    tipl::vector<3> vs,vsJ(reg_slice->vs);
    in1.get_voxel_size(vs);

    tipl::downsampling(It);
    tipl::downsampling(It);
    tipl::downsampling(Iw);
    tipl::downsampling(Iw);
    vs *= 4.0f;



    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
            It,vs,J,vsJ,tipl::reg::affine,tipl::reg::cost_type::mutual_info));

    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    tipl::filter::mean(Iw);
    tipl::filter::mean(Iw);

    tipl::image<3> Iw_(reg_slice->source_images.shape());
    tipl::resample_mt(Iw,Iw_,manual->get_iT());

    reg_slice->skull_removed_images = reg_slice->source_images;
    reg_slice->skull_removed_images *= Iw_;
    slice_need_update = true;
}

void tracking_window::on_show_fiber_toggled(bool checked)
{
    ui->show_fiber->setChecked(checked);
    if(ui->show_fiber->isChecked() ^ (*this)["roi_fiber"].toBool())
        set_data("roi_fiber",ui->show_fiber->isChecked());
    slice_need_update = true;
}
void tracking_window::on_show_edge_toggled(bool checked)
{
    ui->show_edge->setChecked(checked);
    if(ui->show_edge->isChecked() ^ (*this)["roi_draw_edge"].toBool())
        set_data("roi_draw_edge",ui->show_edge->isChecked());
    slice_need_update = true;

}
void tracking_window::on_show_track_toggled(bool checked)
{
    ui->show_track->setChecked(checked);
    if(ui->show_track->isChecked() ^ (*this)["roi_track"].toBool())
        set_data("roi_track",ui->show_track->isChecked());
    slice_need_update = true;
}
void tracking_window::on_show_r_toggled(bool checked)
{
    ui->show_r->setChecked(checked);
    if(ui->show_r->isChecked() ^ (*this)["roi_label"].toBool())
        set_data("roi_label",ui->show_r->isChecked());
    slice_need_update = true;
}
void tracking_window::on_show_3view_toggled(bool checked)
{
    ui->show_3view->setChecked(checked);
    set_data("roi_layout",checked ? 1:0);
    if(checked)
        glWidget->update();
    slice_need_update = true;
}

void tracking_window::on_show_position_toggled(bool checked)
{
    ui->show_position->setChecked(checked);
    if(ui->show_position->isChecked() ^ (*this)["roi_position"].toBool())
        set_data("roi_position",ui->show_position->isChecked());
    slice_need_update = true;
}
void tracking_window::on_show_ruler_toggled(bool checked)
{
    ui->show_ruler->setChecked(checked);
    if(ui->show_ruler->isChecked() ^ (*this)["roi_ruler"].toBool())
    {
        if(ui->show_ruler->isChecked())
            scene.show_grid = !scene.show_grid;
        set_data("roi_ruler",ui->show_ruler->isChecked());
    }
    slice_need_update = true;
}

void tracking_window::on_actionAdjust_Mapping_triggered()
{
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice || !ui->SliceModality->currentIndex())
    {
        QMessageBox::critical(this,"ERROR","In the region window to the left, select the inserted slides to adjust mapping");
        return;
    }
    reg_slice->terminate();
    tipl::image<3> iso_fa;
    handle->get_iso_fa(iso_fa);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
        reg_slice->get_source(),reg_slice->vs,
        iso_fa,slices[0]->vs,
        tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));

    {
        reg_slice->update_transform();
        manual->arg = reg_slice->arg_min;
        manual->check_reg();
    }

    if(manual->exec() != QDialog::Accepted)
        return;

    reg_slice->arg_min = manual->arg;
    reg_slice->update_transform();
    reg_slice->is_diffusion_space = false;
    glWidget->update();
}

void tracking_window::on_actionSave_mapping_triggered()
{
    if(!ui->SliceModality->currentIndex())
        return;
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice)
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Linear Registration",QString(reg_slice->source_file_name.c_str())+".linear_reg.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!reg_slice->save_mapping(filename.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR","Cannot save mapping file.");
}

void tracking_window::on_actionLoad_mapping_triggered()
{
    if(!ui->SliceModality->currentIndex())
        return;
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Linear Registration",QString(reg_slice->source_file_name.c_str())+".linear_reg.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    reg_slice->terminate();
    if(!reg_slice->load_mapping(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR","Invalid linear registration file.");
        return;
    }
    glWidget->update();
}

void tracking_window::updateSlicesMenu(void)
{
    fib_data& fib = *handle;
    std::vector<std::string> index_list;
    fib.get_index_list(index_list);
    // save along track index
    ui->menuSave->clear();
    for (int index = 0; index < index_list.size(); ++index)
        {
            std::string& name = index_list[index];
            QAction* Item = new QAction(this);
            Item->setText(QString("Save %1...").arg(name.c_str()));
            Item->setData(QString(name.c_str()));
            Item->setVisible(true);
            connect(Item, SIGNAL(triggered()),tractWidget, SLOT(save_tracts_data_as()));
            ui->menuSave->addAction(Item);
        }
    // export index mapping
    ui->menuE_xport->clear();
    for (unsigned int index = 0;index < fib.view_item.size(); ++index)
    {
        std::string name = fib.view_item[index].name;
        QAction* Item = new QAction(this);
        Item->setText(QString("Save %1...").arg(name.c_str()));
        Item->setData(QString(name.c_str()));
        Item->setVisible(true);
        connect(Item, SIGNAL(triggered()),&scene, SLOT(save_slice_as()));
        ui->menuE_xport->addAction(Item);
    }

    // export fiber directions
    {
        QAction* Item = new QAction(this);
        Item->setText(QString("Save fiber directions..."));
        Item->setData(QString("fiber"));
        Item->setVisible(true);
        connect(Item, SIGNAL(triggered()),&scene, SLOT(save_slice_as()));
        ui->menuE_xport->addAction(Item);

        if(handle->has_odfs())
        {
            QAction* Item = new QAction(this);
            Item->setText(QString("Save ODFs..."));
            Item->setData(QString("odfs"));
            Item->setVisible(true);
            connect(Item, SIGNAL(triggered()),&scene, SLOT(save_slice_as()));
            ui->menuE_xport->addAction(Item);
        }
    }

    // update along track color dialog
    color_bar->update_slice_indices();

    // update dt metric menu
    QStringList dt_list;
    dt_list << "zero";
    for (auto& item : handle->view_item)
        dt_list << item.name.c_str();
    renderWidget->setList("dt_index1",dt_list);
    renderWidget->setList("dt_index2",dt_list);
}

void tracking_window::on_actionInsert_MNI_images_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open MNI Image",QFileInfo(work_path).absolutePath(),
                "Image files (*.hdr *.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty() || !map_to_mni())
        return;

    CustomSliceModel* reg_slice_ptr = nullptr;
    std::shared_ptr<SliceModel> new_slice(reg_slice_ptr = new CustomSliceModel(handle.get()));
    if(!reg_slice_ptr->load_slices(filename.toStdString(),true/* mni-space image*/))
    {
        QMessageBox::critical(this,"DSI Studio",reg_slice_ptr->error_msg.c_str());
        return;
    }
    slices.push_back(new_slice);
    ui->SliceModality->addItem(reg_slice_ptr->name.c_str());
    updateSlicesMenu();
    ui->SliceModality->setCurrentIndex(int(handle->view_item.size())-1);
    set_data("show_slice",Qt::Checked);
    ui->glSagCheck->setChecked(true);
    ui->glCorCheck->setChecked(true);
    ui->glAxiCheck->setChecked(true);
    glWidget->update();
}
bool tracking_window::addSlices(QStringList filenames,QString name,bool cmd)
{
    std::vector<std::string> files(uint32_t(filenames.size()));
    for (int index = 0; index < filenames.size(); ++index)
        files[size_t(index)] = filenames[index].toStdString();
    CustomSliceModel* reg_slice_ptr = nullptr;
    std::shared_ptr<SliceModel> new_slice(reg_slice_ptr = new CustomSliceModel(handle.get()));
    if(!reg_slice_ptr->load_slices(files))
    {
        if(!cmd)
            QMessageBox::critical(this,"ERROR",reg_slice_ptr->error_msg.c_str());
        else
            tipl::out() << "ERROR:" << reg_slice_ptr->error_msg << std::endl;
        return false;
    }
    slices.push_back(new_slice);
    ui->SliceModality->addItem(name);
    updateSlicesMenu();
    if(!timer2.get() && reg_slice_ptr->running)
    {
        timer2.reset(new QTimer());
        timer2->setInterval(1000);
        connect(timer2.get(), SIGNAL(timeout()), this, SLOT(check_reg()));
        timer2->start();
        check_reg();
    }
    ui->SliceModality->setCurrentIndex(int(handle->view_item.size())-1);
    if(!cmd)
    {
        set_data("show_slice",Qt::Checked);
        ui->glSagCheck->setChecked(true);
        ui->glCorCheck->setChecked(true);
        ui->glAxiCheck->setChecked(true);
        glWidget->update();
    }
    return true;
}

void tracking_window::check_reg(void)
{
    bool all_ended = true;
    for(unsigned int index = 0;index < slices.size();++index)
    {
        CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(slices[index].get());
        if(reg_slice && reg_slice->running)
        {
            all_ended = false;
            reg_slice->update_transform();
        }
    }
    slice_need_update = true;
    if(all_ended)
        timer2.reset();
    else
        glWidget->update();
}
void tracking_window::on_actionLoad_Color_Map_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
                "Load color map",QCoreApplication::applicationDirPath()+"/color_map/",
                "Text files (*.txt);;All files|(*)");
    if(filename.isEmpty())
        return;
    tipl::color_map_rgb new_color_map;
    if(!new_color_map.load_from_file(filename.toStdString().c_str()))
    {
          QMessageBox::critical(this,"ERROR","Invalid color map format");
          return;
    }
    handle->view_item[current_slice->view_id].v2c.set_color_map(new_color_map);
    slice_need_update = true;
    glWidget->update_slice();
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

void tracking_window::on_addSlices_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
        this,"Open Images files",QFileInfo(work_path).absolutePath(),
                "Image files (*.dcm *.hdr *.nii *nii.gz *db.fib.gz 2dseq);;Histology (*.jpg *.tif);;All files (*)" );
    if( filenames.isEmpty())
        return;
    if(QFileInfo(filenames[0]).completeSuffix() == "dcm" && filenames.size() == 1)
    {
        QDir directory = QFileInfo(filenames[0]).absoluteDir();
        QStringList file_list = directory.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        if(file_list.size() > filenames.size())
        {
            QString msg =
              QString("There are %1 DICOM files in the directory. Select all?").arg(file_list.size());
            int result = QMessageBox::information(this,"Input images",msg,
                                     QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::Yes)
            {
                filenames = file_list;
                for(int index = 0;index < filenames.size();++index)
                    filenames[index] = directory.absolutePath() + "/" + filenames[index];
            }
        }
    }
    if(filenames[0].endsWith(".nii.gz"))
    {
        for(int i = 0;i < filenames.size();++i)
            addSlices(QStringList() << filenames[i],QFileInfo(filenames[i]).baseName(),false);
    }
    else
        addSlices(filenames,QFileInfo(filenames[0]).baseName(),false);
}

void tracking_window::on_actionSingle_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::single;
    glWidget->update();
}

void tracking_window::on_actionDouble_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::two;
    glWidget->transformation_matrix2 = glWidget->transformation_matrix;
    glWidget->rotation_matrix2 = glWidget->rotation_matrix;
    glWidget->update();
}

void tracking_window::on_actionStereoscopic_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::stereo;
    glWidget->update();
}


void tracking_window::on_is_overlay_clicked()
{
    current_slice->is_overlay = (ui->is_overlay->isChecked());
    if(current_slice->is_overlay)
        overlay_slices.push_back(current_slice);
    else
        overlay_slices.erase(std::remove(overlay_slices.begin(),overlay_slices.end(),current_slice),overlay_slices.end());

}



void tracking_window::on_actionOpen_Connectivity_Matrix_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open Connectivity Matrices files",QFileInfo(work_path).absolutePath(),
                "Connectivity file (*.mat *.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(filename.endsWith(".mat"))
    {
        tipl::io::mat_read in;
        if(!in.load_from_file(filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",in.error_msg.c_str());
            return;
        }
        unsigned int row,col;
        const float* buf = nullptr;
        if(!in.read("connectivity",row,col,buf))
        {
            QMessageBox::information(this,"error","Cannot find a matrix named connectivity");
            return;
        }
        if(row != col)
        {
            QMessageBox::information(this,"error","The connectivity matrix should be a square matrix");
            return;
        }
        tipl::image<2,float> connectivity(tipl::shape<2>(row,col));
        std::copy(buf,buf+row*col,connectivity.begin());
        glWidget->connectivity = std::move(connectivity);

        if(in.has("atlas") && in.read<std::string>("atlas") != "roi")
        {
            std::string atlas = in.read<std::string>("atlas");
            for(size_t i = 0;i < handle->atlas_list.size();++i)
                if(atlas == handle->atlas_list[i]->name)
                {
                    regionWidget->delete_all_region();
                    regionWidget->begin_update();
                    for(size_t j = 0;j < handle->atlas_list[i]->get_list().size();++j)
                        regionWidget->add_region_from_atlas(handle->atlas_list[i],uint32_t(j));
                    regionWidget->end_update();
                    set_data("region_graph",1);
                    glWidget->update();
                    break;
                }
        }
    }
    if(regionWidget->regions.empty())
    {
        QMessageBox::information(this,"error","Please load the regions first for visualization");
        return;
    }
    if(filename.endsWith(".txt"))
    {
        std::vector<float> buf;
        std::ifstream in(filename.toStdString().c_str());
        while(in)
        {
            std::string v;
            in >> v;
            if(v.empty())
                break;
            std::istringstream ss(v);
            buf.push_back(0.0f);
            ss >> buf.back();
        }
        size_t dim = size_t(std::sqrt(buf.size()));
        if(dim*dim != buf.size())
        {
            QMessageBox::information(this,"error",
            QString("There are %1 values in the file. The matrix in the text file is not a square matrix.").arg(buf.size()));
            return;
        }
        glWidget->connectivity.resize(tipl::shape<2>(dim,dim));
        std::copy(buf.begin(),buf.end(),glWidget->connectivity.begin());
    }

    if(int(regionWidget->regions.size()) != glWidget->connectivity.width())
    {
        QMessageBox::information(this,"error",
            QString("The connectiviti matrix is %1-by-%2, but there are %3 regions. Please make sure the sizes are matched.").
                arg(glWidget->connectivity.width()).
                arg(glWidget->connectivity.height()).
                arg(regionWidget->regions.size()));
        return;
    }
    glWidget->max_connectivity = tipl::max_value(glWidget->connectivity);
    set_data("region_graph",1);
    regionWidget->check_all();
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

void tracking_window::on_actionKeep_Current_Slice_triggered()
{
    glWidget->keep_slice = true;
    glWidget->update();
    QMessageBox::information(this,"DSI Studio","Current viewing slice will reamin in the 3D window");
}
void tracking_window::on_enable_auto_track_clicked()
{
    if(!handle->load_track_atlas())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return;
    }
    ui->enable_auto_track->setVisible(false);
    ui->target->setVisible(true);
    ui->target_label->setVisible(true);
    if(ui->target->count() == 0)
    {
        ui->target->clear();
        ui->target->addItem("All");
        for(size_t i = 0;i < handle->tractography_name_list.size();++i)
            ui->target->addItem(handle->tractography_name_list[i].c_str());
        ui->target->setCurrentIndex(0);
    }
    raise();
}

float tracking_window::get_fa_threshold(void)
{
    float threshold = renderWidget->getData("fa_threshold").toFloat();
    if(threshold == 0.0f)
        threshold = renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu;
    return threshold;
}

void tracking_window::on_actionFIB_protocol_triggered()
{
    std::istringstream in(handle->steps);
    std::ostringstream out;
    std::string line;
    for(int i = 1;std::getline(in,line);++i)
    {
        if(line.find('=') != std::string::npos)
            line = std::string("Set ") + line;
        else
        if(std::count(line.begin(),line.end(),']') >= 3)
            line = std::string("At the top menu, select ") + line;
        else
            line = std::string("Click ") + line;
        out << "(" << i << ") " << line << std::endl;
    }
    show_info_dialog("FIB",out.str());
}

void tracking_window::on_SliceModality_currentIndexChanged(int index)
{
    if(index == -1 || !current_slice.get())
        return;

    no_update = true;

    tipl::vector<3,float> slice_position(current_slice->slice_pos);
    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->T);
    current_slice = slices[size_t(index)];

    if(!handle->view_item[current_slice->view_id].image_ready)
        current_slice->get_source();

    ui->is_overlay->setChecked(current_slice->is_overlay);
    ui->glSagSlider->setRange(0,int(current_slice->dim[0]-1));
    ui->glCorSlider->setRange(0,int(current_slice->dim[1]-1));
    ui->glAxiSlider->setRange(0,int(current_slice->dim[2]-1));
    ui->glSagBox->setRange(0,int(current_slice->dim[0]-1));
    ui->glCorBox->setRange(0,int(current_slice->dim[1]-1));
    ui->glAxiBox->setRange(0,int(current_slice->dim[2]-1));

    std::pair<float,float> range = current_slice->get_value_range();
    std::pair<float,float> contrast_range = current_slice->get_contrast_range();
    std::pair<unsigned int,unsigned int> contrast_color = current_slice->get_contrast_color();
    float r = range.second-range.first;
    float step = r/20.0f;
    ui->min_value_gl->setMinimum(double(range.first-r));
    ui->min_value_gl->setMaximum(double(range.second));
    ui->min_value_gl->setSingleStep(double(step));
    ui->min_color_gl->setColor(contrast_color.first);

    ui->max_value_gl->setMinimum(double(range.first));
    ui->max_value_gl->setMaximum(double(range.second+r));
    ui->max_value_gl->setSingleStep(double(step));
    ui->max_color_gl->setColor(contrast_color.second);

    ui->min_value_gl->setValue(double(contrast_range.first));
    ui->max_value_gl->setValue(double(contrast_range.second));

    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->invT);
    move_slice_to(slice_position);

    no_update = false;
    change_contrast();
}



void tracking_window::on_actionSave_T1W_T2W_images_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice)
        return;
    QString filename = QFileDialog::getSaveFileName(
        this,"Save T1W/T2W Image",QFileInfo(work_path).absolutePath()+"//"+slice->name.c_str()+"_modified.nii.gz","Image files (*nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    tipl::io::gz_nifti::save_to_file(filename.toStdString().c_str(),slice->source_images,slice->vs,slice->trans,slice->is_mni);
}

void tracking_window::on_actionMark_Region_on_T1W_T2W_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty())
        return;
    bool ok = true;
    double ratio = QInputDialog::getDouble(this,"DSI Studio",
            "Aissgn intensity (ratio to the maximum, e.g., 1.2 = 1.2*max)",1.0,0.0,10.0,1,&ok);
    if(!ok)
        return;
    auto current_region = regionWidget->regions[uint32_t(regionWidget->currentRow())];
    float mark_value = slice->get_value_range().second*float(ratio);
    tipl::image<3,unsigned char> mask;
    current_region->SaveToBuffer(mask);
    if(current_region->to_diffusion_space != slice->T)
    {
        tipl::image<3,unsigned char> new_mask(slice->dim);
        tipl::resample_mt<tipl::interpolation::nearest>(mask,new_mask,
            tipl::transformation_matrix<float>(tipl::from_space(slice->T).to(current_region->to_diffusion_space)));
        mask.swap(new_mask);
    }

    for(size_t i = 0;i < mask.size();++i)
        if(mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update = true;
    glWidget->update();
}

void paint_track_on_volume(tipl::image<3,unsigned char>& track_map,const std::vector<std::vector<float> >& all_tracts,SliceModel* slice)
{
    tipl::par_for(all_tracts.size(),[&](unsigned int i)
    {
        auto tracks = all_tracts[i];
        for(size_t k = 0;k < tracks.size();k +=3)
        {
            tipl::vector<3> p(&tracks[0] + k);
            p.to(slice->invT);
            tracks[k] = p[0];
            tracks[k+1] = p[1];
            tracks[k+2] = p[2];
        }
        for(size_t j = 0;j < tracks.size();j += 3)
        {
            tipl::pixel_index<3> p(std::round(tracks[j]),std::round(tracks[j+1]),std::round(tracks[j+2]),track_map.shape());
            if(track_map.shape().is_valid(p))
                track_map[p.index()] = 1;
            if(j)
            {
                for(float r = 0.2f;r < 1.0f;r += 0.2f)
                {
                    tipl::pixel_index<3> p2(std::round(tracks[j]*r+tracks[j-3]*(1-r)),
                                             std::round(tracks[j+1]*r+tracks[j-2]*(1-r)),
                                             std::round(tracks[j+2]*r+tracks[j-1]*(1-r)),track_map.shape());
                    if(track_map.shape().is_valid(p2))
                        track_map[p2.index()] = 1;
                }
            }
        }
    });
}

void tracking_window::on_actionMark_Tracts_on_T1W_T2W_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty() || tractWidget->tract_models.empty())
        return;
    bool ok = true;
    double ratio = QInputDialog::getDouble(this,"DSI Studio",
            "Aissgn intensity (ratio to the maximum, e.g., 1.2 = 1.2*max)",1.0,0.0,10.0,1,&ok);
    if(!ok)
        return;
    tipl::image<3,unsigned char> t_mask(slice->source_images.shape());
    for(auto checked_tracks : tractWidget->get_checked_tracks())
        paint_track_on_volume(t_mask,checked_tracks->get_tracts(),slice);
    float mark_value = slice->get_value_range().second*float(ratio);
    for(size_t i = 0;i < t_mask.size();++i)
        if(t_mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update = true;
    glWidget->update();
}

void tracking_window::on_actionSave_Slices_to_DICOM_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty())
        return;

    QMessageBox::information(this,"DSI Studio","Please assign the original DICOM files");
    QStringList files = QFileDialog::getOpenFileNames(this,"Assign DICOM files",
                                                      QFileInfo(work_path).absolutePath(),"DICOM files (*.dcm);;All files (*)");
    if(files.isEmpty())
        return;

    std::vector<std::string> file_list;
    for(int i = 0;i < files.size();++i)
        file_list.push_back(files[i].toStdString());
    tipl::io::dicom_volume volume;
    if(!volume.load_from_files(file_list))
    {
        QMessageBox::critical(this,"ERROR",volume.error_msg.c_str());
        return;
    }

    {
        tipl::image<3> I;
        volume >> I;
        if(I.shape() != slice->source_images.shape())
        {
            QMessageBox::critical(this,"ERROR","Selected DICOM files does not match the original slices. Please check if missing any files.");
            return;
        }
    }

    uint8_t new_dim_order[3];
    uint8_t new_flip[3];
    for(uint8_t i = 0;i < 3; ++i)
    {
        new_dim_order[uint8_t(volume.dim_order[i])] = i;
        new_flip[uint8_t(volume.dim_order[i])] = uint8_t(volume.flip[i]);
    }
    tipl::image<3> out;
    tipl::reorder(slice->source_images,out,new_dim_order,new_flip);

    tipl::io::dicom header;
    tipl::vector<3> row_orientation;
    if(!header.load_from_file(files[0].toStdString().c_str()) ||
       !header.get_image_row_orientation(row_orientation.begin()))
    {
        QMessageBox::information(this,"DSI Studio","Invalid DICOM files");
        return;
    }
    auto I = slice->source_images;
    bool is_axial = row_orientation[0] > row_orientation[1];
    size_t read_size = is_axial ? I.plane_size():size_t(I.height()*I.depth());

    tipl::progress prog("Writing data");
    for(int i = 0,pos = 0;prog(i,files.size());++i,pos += read_size)
    {
        std::vector<char> buf;
        {
            std::ifstream in(files[i].toStdString().c_str(),std::ios::binary);
            in.seekg(0,in.end);
            buf.resize(size_t(in.tellg()));
            in.seekg(0,in.beg);
            if(read_size*sizeof(short) > buf.size())
            {
                QMessageBox::critical(this,"ERROR","Compressed DICOM is not supported. Please convert DICOM to uncompressed format.");
                return;
            }
            if(!in.read(&buf[0],int64_t(buf.size())))
            {
                QMessageBox::critical(this,"ERROR","Read DICOM failed");
                return;
            }
        }
        std::copy(out.begin()+pos,out.begin()+pos+int(read_size),
                  reinterpret_cast<short*>(&*(buf.end()-int(read_size*sizeof(short)))));

        QFileInfo info(files[i]);
        QString output_name = info.path() + "/mod_" + info.completeBaseName() + ".dcm";
        std::ofstream out(output_name.toStdString().c_str(),std::ios::binary);
        if(!out)
        {
            QMessageBox::critical(this,"ERROR","Cannot output DICOM. Please check disk space or output permission.");
            return;
        }
        out.write(&buf[0],int64_t(buf.size()));
    }
}

void tracking_window::on_zoom_valueChanged(double arg1)
{
    if(float(arg1) == (*this)["roi_zoom"].toFloat())
        return;
    set_data("roi_zoom",arg1);
    slice_need_update = true;
}

void tracking_window::Move_Slice_X()
{
    ui->glSagSlider->setValue(ui->glSagSlider->value()+1);
}

void tracking_window::Move_Slice_X2()
{
    ui->glSagSlider->setValue(ui->glSagSlider->value()-1);
}

void tracking_window::Move_Slice_Y()
{
    ui->glCorSlider->setValue(ui->glCorSlider->value()+1);
}

void tracking_window::Move_Slice_Y2()
{
    ui->glCorSlider->setValue(ui->glCorSlider->value()-1);
}

void tracking_window::Move_Slice_Z()
{
    ui->glAxiSlider->setValue(ui->glAxiSlider->value()+1);
}

void tracking_window::Move_Slice_Z2()
{
    ui->glAxiSlider->setValue(ui->glAxiSlider->value()-1);
}

void tracking_window::on_actionLoad_Parameter_ID_triggered()
{
    QString id = QInputDialog::getText(this,"DSI Studio","Please assign parameter ID");
    if(id.isEmpty())
        return;
    TrackingParam param;
    param.set_code(id.toStdString());
    set_data("fa_threshold",float(param.threshold));
    set_data("dt_threshold",float(param.dt_threshold));
    set_data("turning_angle",float(std::acos(param.cull_cos_angle)*180.0f/3.14159265358979323846f));
    set_data("step_size",float(param.step_size));
    set_data("smoothing",float(param.smooth_fraction));
    set_data("min_length",float(param.min_length));
    set_data("max_length",float(param.max_length));

    set_data("tracking_method",int(param.tracking_method));
    set_data("tracking_plan",int(param.stop_by_tract));
    set_data("check_ending",int(param.check_ending));
    set_data("track_count",int(param.termination_count));

    set_data("otsu_threshold",float(param.default_otsu));
    set_data("tip_iteration",int(param.tip_iteration));

}


void tracking_window::on_actionLoad_Presentation_triggered()
{
    QString dir = QFileDialog::getExistingDirectory(this,"Open from directory",QFileInfo(windowTitle()).filePath());
    if(dir.isEmpty())
        return;
    command("load_workspace",dir);

}

void tracking_window::on_actionSave_Presentation_triggered()
{
    QString dir = QFileDialog::getExistingDirectory(this,"Save to directory",QFileInfo(windowTitle()).filePath());
    if(dir.isEmpty())
        return;
    command("save_workspace",dir);
    QMessageBox::information(this,"DSI Studio","File saved");
}

void tracking_window::on_actionZoom_In_triggered()
{
    ui->zoom_3d->setValue(ui->zoom_3d->value()+0.1);
}

void tracking_window::on_actionZoom_Out_triggered()
{
    ui->zoom_3d->setValue(ui->zoom_3d->value()-0.1);
}

void tracking_window::on_min_value_gl_valueChanged(double)
{
    ui->min_slider->setValue(int((ui->min_value_gl->value()-ui->min_value_gl->minimum())*double(ui->min_slider->maximum())/
                             (ui->min_value_gl->maximum()-ui->min_value_gl->minimum())));
    change_contrast();
}

void tracking_window::on_min_slider_sliderMoved(int)
{
    ui->min_value_gl->setValue(ui->min_value_gl->minimum()+(ui->min_value_gl->maximum()-ui->min_value_gl->minimum())*
                      double(ui->min_slider->value())/double(ui->min_slider->maximum()));
}

void tracking_window::on_max_value_gl_valueChanged(double)
{
    ui->max_slider->setValue(int((ui->max_value_gl->value()-ui->max_value_gl->minimum())*double(ui->max_slider->maximum())/
                             (ui->max_value_gl->maximum()-ui->max_value_gl->minimum())));
    change_contrast();
}

void tracking_window::on_max_slider_sliderMoved(int)
{
    ui->max_value_gl->setValue(ui->max_value_gl->minimum()+(ui->max_value_gl->maximum()-ui->max_value_gl->minimum())*
                      double(ui->max_slider->value())/double(ui->max_slider->maximum()));
}


void tracking_window::on_actionInsert_Axial_Pictures_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open Picture",QFileInfo(work_path).absolutePath(),"Pictures (*.jpg *.bmp *.png);;All files (*)" );
    if(filename.isEmpty())
        return;
    QStringList filenames;
    filenames << filename;
    if(!addSlices(filenames,QFileInfo(filename).baseName(),false))
        return;
    CustomSliceModel* reg_slice_ptr = dynamic_cast<CustomSliceModel*>(slices.back().get());
    if(!reg_slice_ptr)
        return;
    reg_slice_ptr->arg_min.rotation[1] = 3.1415926f;
    reg_slice_ptr->update_transform();
    QMessageBox::information(this,"DSI Studio","Press Ctrl+A and then hold LEFT/RIGHT button to MOVE/RESIZE slice close to the target before using [Slices][Adjust Mapping]");
    slice_need_update = true;
}

void tracking_window::on_actionInsert_Coronal_Pictures_triggered()
{
    size_t slice_size = slices.size();
    on_actionInsert_Axial_Pictures_triggered();
    if(slice_size == slices.size())
        return;
    CustomSliceModel* reg_slice_ptr = dynamic_cast<CustomSliceModel*>(slices.back().get());
    if(!reg_slice_ptr)
        return;
    reg_slice_ptr->arg_min.rotation[1] = 0.0f;
    reg_slice_ptr->update_transform();
    tipl::flip_y(reg_slice_ptr->picture);
    tipl::flip_y(reg_slice_ptr->source_images);
    tipl::swap_yz(reg_slice_ptr->source_images);
    std::swap(reg_slice_ptr->vs[1],reg_slice_ptr->vs[2]);
    handle->view_item.back().set_image(reg_slice_ptr->source_images.alias());
    reg_slice_ptr->update_image();
    ui->SliceModality->setCurrentIndex(0);
    ui->SliceModality->setCurrentIndex(int(handle->view_item.size())-1);
    on_glCorView_clicked();
}



void tracking_window::on_actionInsert_Sagittal_Picture_triggered()
{
    size_t slice_size = slices.size();
    on_actionInsert_Axial_Pictures_triggered();
    if(slice_size == slices.size())
        return;
    CustomSliceModel* reg_slice_ptr = dynamic_cast<CustomSliceModel*>(slices.back().get());
    if(!reg_slice_ptr)
        return;
    reg_slice_ptr->arg_min.rotation[1] = 0.0f;
    reg_slice_ptr->update_transform();
    tipl::flip_y(reg_slice_ptr->picture);
    tipl::flip_y(reg_slice_ptr->source_images);
    tipl::swap_xy(reg_slice_ptr->source_images);
    tipl::swap_xz(reg_slice_ptr->source_images);
    std::swap(reg_slice_ptr->vs[0],reg_slice_ptr->vs[2]);
    handle->view_item.back().set_image(reg_slice_ptr->source_images.alias());
    reg_slice_ptr->update_image();
    ui->SliceModality->setCurrentIndex(0);
    ui->SliceModality->setCurrentIndex(int(handle->view_item.size())-1);
    on_glSagView_clicked();
}

void tracking_window::on_template_box_currentIndexChanged(int index)
{
    if(index < 0 || index >= int(fa_template_list.size()))
        return;
    handle->set_template_id(size_t(index));
    ui->addRegionFromAtlas->setVisible(!handle->atlas_list.empty());
    ui->enable_auto_track->setVisible(std::filesystem::exists(track_atlas_file_list[uint32_t(index)]));
    ui->target->setCurrentIndex(0);
    ui->target->setVisible(false);
    ui->target_label->setVisible(false);
}

void tracking_window::on_actionManual_Atlas_Alignment_triggered()
{
    if(!handle->load_template())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return ;
    }
    tipl::image<3> iso_fa;
    handle->get_iso_fa(iso_fa);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
        iso_fa,handle->vs,
        handle->template_I2.empty() ? handle->template_I: handle->template_I2,handle->template_vs,
        tipl::reg::affine,tipl::reg::cost_type::mutual_info));
    if(manual->exec() != QDialog::Accepted)
        return;
    handle->manual_template_T = manual->get_iT();
    handle->has_manual_atlas = true;


    std::string output_file_name(handle->fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[handle->template_id].c_str()).
                        baseName().toLower().toStdString();
    output_file_name += ".map.gz";
    if(handle->s2t.empty() && std::filesystem::exists(output_file_name))
    {
        handle->s2t.clear();
        handle->t2s.clear();
        std::filesystem::remove(output_file_name);
    }

    if(!map_to_mni())
        return;
    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,handle));
    atlas_dialog->exec();
}

