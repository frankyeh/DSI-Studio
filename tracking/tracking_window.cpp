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

#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "console.h"
#include "region/regiontablewidget.h"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
#include "mapping/atlas.hpp"

#include "devicetablewidget.h"
#include "fib_data.hpp"
#include "color_bar_dialog.hpp"
#include "regtoolbox.h"
#include "connectivity_matrix_dialog.h"

extern std::vector<std::string> fa_template_list;
extern std::vector<tracking_window*> tracking_windows;
extern size_t auto_track_pos[7];
extern unsigned char auto_track_rgb[6][3];               // projection

static QByteArray default_geo,default_state;


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
                if(std::filesystem::exists(handle->t2w_template_file_name.c_str()))
                    addSlices(QStringList() << QString(handle->t2w_template_file_name.c_str()),"t2w",true);
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
            ui->tract_target_0->hide();
            ui->tract_target_1->hide();
            ui->tract_target_2->hide();
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
        connect(ui->actionNew_Region_From_MNI_Coordinate,SIGNAL(triggered()),regionWidget,SLOT(new_region_from_mni_coordinate()));
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
        connect(ui->actionAll_To_First2,SIGNAL(triggered()),regionWidget,SLOT(action_B2A2()));
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

        connect(ui->actionLeads_to_ROI,SIGNAL(triggered()),deviceWidget,SLOT(lead_to_roi()));


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
        connect(ui->actionOpen_MNI_space_Tracts,SIGNAL(triggered()),tractWidget,SLOT(load_mni_tracts()));
        connect(ui->actionLoad_Built_In_Atlas,SIGNAL(triggered()),tractWidget,SLOT(load_built_in_atlas()));
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
        connect(ui->actionCut_end_portion,SIGNAL(triggered()),tractWidget,SLOT(cut_end_portion()));


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
        connect(new QShortcut(QKeySequence(tr("Q", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_X()));
        connect(new QShortcut(QKeySequence(tr("A", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_X2()));
        connect(new QShortcut(QKeySequence(tr("W", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_Y()));
        connect(new QShortcut(QKeySequence(tr("S", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_Y2()));
        connect(new QShortcut(QKeySequence(tr("E", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_Z()));
        connect(new QShortcut(QKeySequence(tr("D", "X+")),this),SIGNAL(activated()),this,SLOT(Move_Slice_Z2()));
        connect(new QShortcut(QKeySequence(Qt::Key_F1),this),&QShortcut::activated,this,[this](void){on_show_fiber_toggled(!ui->show_fiber->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F2),this),&QShortcut::activated,this,[this](void){on_show_track_toggled(!ui->show_track->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F3),this),&QShortcut::activated,this,[this](void){on_show_ruler_toggled(!ui->show_ruler->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F4),this),&QShortcut::activated,this,[this](void){on_show_position_toggled(!ui->show_position->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F5),this),&QShortcut::activated,this,[this](void){on_show_r_toggled(!ui->show_r->isChecked());});
        connect(new QShortcut(QKeySequence(Qt::Key_F6),this),&QShortcut::activated,this,[this](void){on_show_edge_toggled(!ui->show_edge->isChecked());});
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



void tracking_window::update_scene_slice(void)
{
    slice_need_update = true;
}
extern console_stream console;
bool tracking_window::eventFilter(QObject *obj, QEvent *event)
{
    console.show_output();
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

    QString status = QString("LPS=(%1,%2,%3)")
            .arg(std::round(pos[0]*10.0)/10.0)
            .arg(std::round(pos[1]*10.0)/10.0)
            .arg(std::round(pos[2]*10.0)/10.0);

    if((handle->template_id == handle->matched_template_id && handle->is_mni) || !handle->s2t.empty())
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
    ui->show_3view->setChecked(false);
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
    ui->show_3view->setChecked(false);
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
    ui->show_3view->setChecked(false);
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


float tracking_window::get_fa_threshold(void)
{
    float threshold = renderWidget->getData("fa_threshold").toFloat();
    if(threshold == 0.0f)
        threshold = renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu;
    return threshold;
}


void tracking_window::on_SliceModality_currentIndexChanged(int index)
{
    if(index == -1 || !current_slice.get())
        return;

    no_update = true;

    tipl::vector<3,float> slice_position(current_slice->slice_pos);
    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->to_dif);
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

    ui->draw_threshold->setValue(0.0);
    ui->draw_threshold->setMaximum(range.second);
    ui->draw_threshold->setSingleStep(range.second/50.0);

    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->to_slice);
    move_slice_to(slice_position);

    no_update = false;
    change_contrast();
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

void tracking_window::on_actionLoad_MNI_mapping_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                       this,"Open MNI mapping",QFileInfo(work_path).absolutePath(),
                       "Mapping file(*map.gz);;NIFTI file(*nii.gz *.nii);;All file types (*)" );
    if (filename.isEmpty())
        return;
    {
        tipl::progress prog("loading mapping",true);
        if(!handle->load_mapping(filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
            return;
        }
    }
    QMessageBox::information(this,"DSI Studio","mapping loaded");
}


void tracking_window::on_actionSave_MNI_mapping_triggered()
{
    std::string output_file_name(handle->fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[handle->template_id].c_str()).baseName().toLower().toStdString();
    output_file_name += ".map.gz";

    QString filename = QFileDialog::getSaveFileName(
                       this,"Save MNI mapping",output_file_name.c_str(),
                       "Mapping file(*map.gz);;All file types (*)" );
    if (filename.isEmpty())
        return;
    {
        tipl::progress prog("saving mapping",true);
        if(!handle->save_mapping(filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
            return;
        }
    }
    QMessageBox::information(this,"DSI Studio","mapping saved");
}

