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

extern std::string t1w_template_file_name,wm_template_file_name;
extern std::vector<std::string> fa_template_list;
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
        std::ofstream out(filename.toLocal8Bit().begin());
        out << result.c_str();
    }
    if (msgBox.clickedButton() == copyButton)
        QApplication::clipboard()->setText(result.c_str());
}

void populate_templates(QComboBox* combo)
{
    if(!fa_template_list.empty())
    {
        for(int index = 0;index < fa_template_list.size();++index)
            combo->addItem(QFileInfo(fa_template_list[index].c_str()).baseName());
        combo->setCurrentIndex(0);
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
        QMainWindow(parent),ui(new Ui::tracking_window),handle(new_handle),scene(*this)

{
    fib_data& fib = *new_handle;
    scene.no_show = true;
    for (unsigned int index = 0;index < fib.view_item.size(); ++index)
        slices.push_back(std::make_shared<SliceModel>(handle.get(),index));
    current_slice = slices[0];

    ui->setupUi(this);

    // setup GUI
    {
        // create objects
        {
            setGeometry(10,10,800,600);
            ui->regionDockWidget->setMinimumWidth(0);
            ui->ROIdockWidget->setMinimumWidth(0);
            ui->renderingLayout->addWidget(renderWidget = new RenderingTableWidget(*this,ui->renderingWidgetHolder));
            ui->glLayout->addWidget(glWidget = new GLWidget(renderWidget->getData("anti_aliasing").toInt(),*this,renderWidget));
            ui->verticalLayout_3->addWidget(regionWidget = new RegionTableWidget(*this,ui->regionDockWidget));
            ui->track_verticalLayout->addWidget(tractWidget = new TractTableWidget(*this,ui->TractWidgetHolder));
            ui->graphicsView->setScene(&scene);
            ui->graphicsView->setCursor(Qt::CrossCursor);
            scene.statusbar = ui->statusbar;
            color_bar.reset(new color_bar_dialog(this));

        }
        // recall the setting
        {
            QSettings settings;
            if(!default_geo.size())
                default_geo = saveGeometry();
            if(!default_state.size())
                default_state = saveState();
            ui->rendering_efficiency->setCurrentIndex(settings.value("rendering_quality",1).toInt());
            ui->TractWidgetHolder->show();
            ui->renderingWidgetHolder->show();
            ui->ROIdockWidget->show();
            ui->regionDockWidget->show();
        }
        // update GUI values
        {
            ui->zoom->setValue((*this)["roi_zoom"].toFloat());
            ui->show_edge->setChecked((*this)["roi_edge"].toBool());
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
            if(!handle->is_qsdr)
                ui->actionManual_Registration->setEnabled(false);
            if(!handle->trackable)
            {
                ui->perform_tracking->hide();
                ui->show_fiber->setChecked(false);
                ui->show_fiber->hide();
                ui->enable_auto_track->hide();
            }
        }
        // Initialize slices
        {
            ui->SliceModality->clear();
            for (unsigned int index = 0;index < fib.view_item.size(); ++index)
                ui->SliceModality->addItem(fib.view_item[index].name.c_str());
            ui->SliceModality->setCurrentIndex(-1);
            updateSlicesMenu();
        }
        // Handle template and atlases
        {
            if(handle->is_qsdr)
                ui->actionOpen_MNI_Region->setVisible(false);
            if(handle->is_qsdr && handle->is_human_data)
            {
                if(QFileInfo(QString(t1w_template_file_name.c_str())).exists())
                    addSlices(QStringList() << QString(t1w_template_file_name.c_str()),"icbm_t1w",false,false);
                if(QFileInfo(QString(wm_template_file_name.c_str())).exists())
                    addSlices(QStringList() << QString(wm_template_file_name.c_str()),"icbm_wm",false,false);
            }
            populate_templates(ui->template_box);
            ui->template_box->setCurrentIndex(handle->template_id);
            if(handle->is_qsdr)
                handle->load_template();
            set_data("min_length",handle->vs[0]*15.0f);
            set_data("max_length",handle->vs[0]*150.0f);
        }

        // setup fa threshold
        initialize_tracking_index(0);

        report(handle->report.c_str());

        // provide automatic tractography
        {
            ui->target->setVisible(false);
            ui->target_label->setVisible(false);
        }
    }

    // opengl
    {
        connect(ui->glSagSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glCorSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glAxiSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->glSagCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->glCorCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->glAxiCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));

        connect(ui->min_value_gl,SIGNAL(valueChanged(double)),this,SLOT(change_contrast()));
        connect(ui->max_value_gl,SIGNAL(valueChanged(double)),this,SLOT(change_contrast()));
        connect(ui->max_color_gl,SIGNAL(clicked()),this,SLOT(change_contrast()));
        connect(ui->min_color_gl,SIGNAL(clicked()),this,SLOT(change_contrast()));

        connect(ui->actionSave_Screen,SIGNAL(triggered()),glWidget,SLOT(catchScreen()));
        connect(ui->actionSave_3D_screen_in_high_resolution,SIGNAL(triggered()),glWidget,SLOT(catchScreen2()));
        connect(ui->actionLoad_Camera,SIGNAL(triggered()),glWidget,SLOT(loadCamera()));
        connect(ui->actionSave_Camera,SIGNAL(triggered()),glWidget,SLOT(saveCamera()));
        connect(ui->actionSave_Rotation_Images,SIGNAL(triggered()),glWidget,SLOT(saveRotationSeries()));
        connect(ui->actionSave_3D_screen_in_3_views,SIGNAL(triggered()),glWidget,SLOT(save3ViewImage()));
        connect(ui->action3D_Screen,SIGNAL(triggered()),glWidget,SLOT(copyToClipboard()));
        connect(ui->action3D_Screen_Each_Tract,SIGNAL(triggered()),glWidget,SLOT(copyToClipboardEach()));
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


        connect(ui->actionFull,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionRight,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLeft,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionLower,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionUpper,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionAnterior,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->actionPosterior,SIGNAL(triggered()),glWidget,SLOT(addSurface()));


        connect(ui->actionInsert_T1_T2,SIGNAL(triggered()),this,SLOT(on_addSlices_clicked()));

    }
    // scene view
    {

        connect(&scene,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(&scene,SIGNAL(need_update()),glWidget,SLOT(updateGL()));

        connect(ui->actionAxial_View,SIGNAL(triggered()),this,SLOT(on_glAxiView_clicked()));
        connect(ui->actionCoronal_View,SIGNAL(triggered()),this,SLOT(on_glCorView_clicked()));
        connect(ui->actionSagittal_view,SIGNAL(triggered()),this,SLOT(on_glSagView_clicked()));


        connect(ui->actionSave_ROI_Screen,SIGNAL(triggered()),&scene,SLOT(catch_screen()));

    }

    // regions
    {

        connect(regionWidget,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(regionWidget,SIGNAL(itemSelectionChanged()),&scene,SLOT(show_slice()));
        connect(regionWidget,SIGNAL(need_update()),glWidget,SLOT(updateGL()));

        connect(ui->actionNewRegion,SIGNAL(triggered()),regionWidget,SLOT(new_region()));
        connect(ui->actionNew_Super_Resolution_Region,SIGNAL(triggered()),regionWidget,SLOT(new_high_resolution_region()));
        connect(ui->actionOpenRegion,SIGNAL(triggered()),regionWidget,SLOT(load_region()));
        connect(ui->actionOpen_MNI_Region,SIGNAL(triggered()),regionWidget,SLOT(load_mni_region()));
        connect(ui->actionSaveRegionAs,SIGNAL(triggered()),regionWidget,SLOT(save_region()));
        connect(ui->actionSave_All_Regions_As,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions()));
        connect(ui->actionSave_All_Regions_As_Multiple_Files,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions_to_dir()));
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

        connect(ui->actionSeparate,SIGNAL(triggered()),regionWidget,SLOT(action_separate()));
        connect(ui->actionA_B,SIGNAL(triggered()),regionWidget,SLOT(action_A_B()));
        connect(ui->actionB_A,SIGNAL(triggered()),regionWidget,SLOT(action_B_A()));
        connect(ui->actionAB,SIGNAL(triggered()),regionWidget,SLOT(action_AB()));
        connect(ui->actionBy_Name,SIGNAL(triggered()),regionWidget,SLOT(action_sort_name()));
        connect(ui->actionBy_Size,SIGNAL(triggered()),regionWidget,SLOT(action_sort_size()));
        connect(ui->actionBy_X,SIGNAL(triggered()),regionWidget,SLOT(action_sort_x()));
        connect(ui->actionBy_Y,SIGNAL(triggered()),regionWidget,SLOT(action_sort_y()));
        connect(ui->actionBy_Z,SIGNAL(triggered()),regionWidget,SLOT(action_sort_z()));
        connect(ui->actionMove_Slices_To_Current_Region,SIGNAL(triggered()),regionWidget,SLOT(move_slice_to_current_region()));





        connect(ui->actionSet_Opacity,SIGNAL(triggered()),regionWidget,SLOT(action_set_opa()));

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
    // tracts
    {
        connect(ui->perform_tracking,SIGNAL(clicked()),tractWidget,SLOT(start_tracking()));
        connect(ui->stop_tracking,SIGNAL(clicked()),tractWidget,SLOT(stop_tracking()));

        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(makeTracts()));
        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(updateGL()));
        connect(tractWidget,SIGNAL(cellChanged(int,int)),glWidget,SLOT(updateGL())); //update label
        connect(tractWidget,SIGNAL(itemSelectionChanged()),tractWidget,SLOT(show_report()));
        connect(glWidget,SIGNAL(edited()),tractWidget,SLOT(edit_tracts()));
        connect(glWidget,SIGNAL(region_edited()),glWidget,SLOT(updateGL()));
        connect(glWidget,SIGNAL(region_edited()),&scene,SLOT(show_slice()));


        connect(ui->actionFilter_by_ROI,SIGNAL(triggered()),tractWidget,SLOT(filter_by_roi()));

        connect(ui->actionOpenTract,SIGNAL(triggered()),tractWidget,SLOT(load_tracts()));
        connect(ui->actionOpen_Tracts_Label,SIGNAL(triggered()),tractWidget,SLOT(load_tract_label()));
        connect(ui->actionMerge_All,SIGNAL(triggered()),tractWidget,SLOT(merge_all()));
        connect(ui->actionMerge_Tracts_by_Name,SIGNAL(triggered()),tractWidget,SLOT(merge_track_by_name()));
        connect(ui->actionCopyTrack,SIGNAL(triggered()),tractWidget,SLOT(copy_track()));
        connect(ui->actionSort_Tracts_By_Names,SIGNAL(triggered()),tractWidget,SLOT(sort_track_by_name()));



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

        connect(ui->actionUndo,SIGNAL(triggered()),tractWidget,SLOT(undo_tracts()));
        connect(ui->actionRedo,SIGNAL(triggered()),tractWidget,SLOT(redo_tracts()));
        connect(ui->actionTrim,SIGNAL(triggered()),tractWidget,SLOT(trim_tracts()));

        connect(ui->actionSet_Color,SIGNAL(triggered()),tractWidget,SLOT(set_color()));

        connect(ui->actionK_means_Clustering,SIGNAL(triggered()),tractWidget,SLOT(clustering_kmeans()));
        connect(ui->actionEM_Clustering,SIGNAL(triggered()),tractWidget,SLOT(clustering_EM()));
        connect(ui->actionHierarchical,SIGNAL(triggered()),tractWidget,SLOT(clustering_hie()));
        connect(ui->actionOpen_Cluster_Labels,SIGNAL(triggered()),tractWidget,SLOT(open_cluster_label()));
        connect(ui->actionRecognize_Clustering,SIGNAL(triggered()),tractWidget,SLOT(auto_recognition()));
        connect(ui->actionRecognize_and_Rename,SIGNAL(triggered()),tractWidget,SLOT(recognize_rename()));


        //setup menu
        connect(ui->actionSaveTractAs,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_as()));
        connect(ui->actionSave_VMRL,SIGNAL(triggered()),tractWidget,SLOT(save_vrml_as()));
        connect(ui->actionSave_All_Tracts_As,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_as()));
        connect(ui->actionSave_All_Tracts_As_Multiple_Files,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_to_dir()));
        connect(ui->actionSave_End_Points_As,SIGNAL(triggered()),tractWidget,SLOT(save_end_point_as()));
        connect(ui->actionSave_Enpoints_In_MNI_Space,SIGNAL(triggered()),tractWidget,SLOT(save_end_point_in_mni()));
        connect(ui->actionSave_Tracts_In_Native_Space,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_in_native()));
        connect(ui->actionDeep_Learning_Train,SIGNAL(triggered()),tractWidget,SLOT(deep_learning_train()));
        connect(ui->actionStatistics,SIGNAL(triggered()),tractWidget,SLOT(show_tracts_statistics()));
        connect(ui->actionRecognize_Current_Tract,SIGNAL(triggered()),tractWidget,SLOT(recog_tracks()));

        connect(ui->track_up,SIGNAL(clicked()),tractWidget,SLOT(move_up()));
        connect(ui->track_down,SIGNAL(clicked()),tractWidget,SLOT(move_down()));

        connect(ui->actionPPV_analysis,SIGNAL(triggered()),tractWidget,SLOT(ppv_analysis()));



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
    {
        scene.no_show = false;
        ui->SliceModality->setCurrentIndex(0);
        on_glAxiView_clicked();
        if((*this)["orientation_convention"].toInt() == 1)
            glWidget->set_view(2);

    }

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
    tractWidget->delete_all_tract();
    qApp->removeEventFilter(this);
    QSettings settings;
    settings.setValue("rendering_quality",ui->rendering_efficiency->currentIndex());
    delete ui;
    //std::cout << __FUNCTION__ << " " << __FILE__ << std::endl;
}
void tracking_window::report(QString string)
{
    ui->text_report->setText(string);
}
bool tracking_window::command(QString cmd,QString param,QString param2)
{
    if(glWidget->command(cmd,param,param2) ||
       scene.command(cmd,param,param2) ||
       tractWidget->command(cmd,param,param2) ||
       regionWidget->command(cmd,param,param2))
        return true;
    if(cmd == "restore_rendering")
    {
        renderWidget->setDefault("ROI");
        renderWidget->setDefault("Rendering");
        renderWidget->setDefault("show_slice");
        renderWidget->setDefault("show_tract");
        renderWidget->setDefault("show_region");
        renderWidget->setDefault("show_surface");
        renderWidget->setDefault("show_label");
        renderWidget->setDefault("show_odf");
        glWidget->updateGL();
        scene.show_slice();
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
            std::cout << "cannot find index:" << param.toStdString() << std::endl;
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
    if(cmd == "set_param")
    {
        set_data(param,param2);
        glWidget->updateGL();
        scene.show_slice();
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
        regionWidget->regions.back()->show_region.color = param.toInt();
        glWidget->updateGL();
        scene.show_slice();
        return true;
    }
    if(cmd == "add_slice")
    {
        if(!addSlices(QStringList() << param,param,renderWidget->getData("slice_smoothing").toBool(),true))
            return true;
        std::cout << "register image to the DWI space" << std::endl;
        CustomSliceModel* cur_slice = (CustomSliceModel*)slices.back().get();
        if(cur_slice->thread.get())
            cur_slice->thread->wait();
        cur_slice->update();

        return true;
    }
    return false;
}

void tracking_window::initialize_tracking_index(int p)
{
    QStringList tracking_index_list,dt_list;
    dt_list << "none";
    for(int index = 0;index < handle->dir.index_name.size();++index)
        if(handle->dir.index_name[index].find("dec_") != 0 &&
           handle->dir.index_name[index].find("inc_") != 0)
            tracking_index_list.push_back(handle->dir.index_name[index].c_str());
    for(int index = 0;index < handle->dir.dt_index_name.size();++index)
        dt_list.push_back(handle->dir.dt_index_name[index].c_str());

    renderWidget->setList("tracking_index",tracking_index_list);
    renderWidget->setList("dt_index",dt_list);
    set_data("tracking_index",p);
    set_data("dt_index",0);
    on_tracking_index_currentIndexChanged(p);
    scene.center();
}
bool tracking_window::eventFilter(QObject *obj, QEvent *event)
{
    bool has_info = false;
    tipl::vector<3,float> pos;
    if (event->type() == QEvent::MouseMove)
    {
        if (obj == glWidget)
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

    QString status;

    if(!current_slice->is_diffusion_space)
    {
        tipl::vector<3,float> pos_dwi(pos);
        pos_dwi.to(current_slice->T);
        status = QString("(%1,%2,%3) %4(%5,%6,%7)").arg(std::round(pos_dwi[0]*10.0)/10.0)
                .arg(std::round(pos_dwi[1]*10.0)/10.0)
                .arg(std::round(pos_dwi[2]*10.0)/10.0)
                .arg(ui->SliceModality->currentText())
                .arg(std::round(pos[0]*10.0)/10.0)
                .arg(std::round(pos[1]*10.0)/10.0)
                .arg(std::round(pos[2]*10.0)/10.0);
        pos = pos_dwi;
    }
    else
    {
        status = QString("(%1,%2,%3) ").arg(std::round(pos[0]*10.0)/10.0)
                .arg(std::round(pos[1]*10.0)/10.0)
                .arg(std::round(pos[2]*10.0)/10.0);
    }

    if(!handle->need_normalization || !handle->mni_position.empty())
    {
        tipl::vector<3,float> mni(pos);
        handle->subject2mni(mni);
        status += QString("MNI(%1,%2,%3) ")
                .arg(std::round(mni[0]*10.0)/10.0)
                .arg(std::round(mni[1]*10.0)/10.0)
                .arg(std::round(mni[2]*10.0)/10.0);
    }
    status += " ";
    std::vector<float> data;
    pos.round();
    handle->get_voxel_information(pos[0],pos[1],pos[2], data);
    for(unsigned int index = 0,data_index = 0;index < handle->view_item.size() && data_index < data.size();++index)
        if(handle->view_item[index].name != "color")
        {
            status += handle->view_item[index].name.c_str();
            status += QString("=%1 ").arg(data[data_index]);
            ++data_index;
        }
    ui->statusbar->showMessage(status);
    return false;
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
    tracking_thread.param.initial_direction = renderWidget->getData("initial_direction").toInt();
    tracking_thread.param.interpolation_strategy = renderWidget->getData("interpolation").toInt();
    tracking_thread.param.stop_by_tract = renderWidget->getData("tracking_plan").toInt();
    tracking_thread.param.center_seed = renderWidget->getData("seed_plan").toInt();
    tracking_thread.param.random_seed = renderWidget->getData("random_seed").toInt();
    tracking_thread.param.check_ending = renderWidget->getData("check_ending").toInt();
    tracking_thread.param.termination_count = renderWidget->getData("track_count").toInt();
    tracking_thread.param.default_otsu = renderWidget->getData("otsu_threshold").toFloat();
    tracking_thread.param.tip_iteration =
            // only used in automatic fiber tracking
            ui->target->currentIndex() > 0 ? renderWidget->getData("auto_tip").toInt() : 0;
}
float tracking_window::get_scene_zoom(void)
{
    float display_ratio = (*this)["roi_zoom"].toFloat();
    if(!current_slice->is_diffusion_space)
        display_ratio *= current_slice->voxel_size[0]/handle->vs[0];
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
        scene.show_slice();
        glWidget->updateGL();
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
    scene.show_slice();
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
    ui->SlicePos->setRange(0,current_slice->geometry[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(0);
    glWidget->updateGL();
    glWidget->setFocus();

    scene.show_slice();
}

void tracking_window::on_glCorView_clicked()
{
    cur_dim = 1;
    ui->SlicePos->setRange(0,current_slice->geometry[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(1);
    glWidget->updateGL();
    glWidget->setFocus();
    scene.show_slice();
}

void tracking_window::on_glAxiView_clicked()
{
    cur_dim = 2;
    ui->SlicePos->setRange(0,current_slice->geometry[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);
    glWidget->set_view(2);
    glWidget->updateGL();
    glWidget->setFocus();
    scene.show_slice();

}

void tracking_window::move_slice_to(tipl::vector<3,float> slice_position)
{
    slice_position.round();
    for(int i = 0;i < 3; ++i)
    {
        if(slice_position[i] < 0)
            slice_position[i] = 0;
        if(slice_position[i] >= current_slice->geometry[i]-1)
            slice_position[i] = 0;
    }
    current_slice->slice_pos = slice_position;

    ui->glSagSlider->setValue(slice_position[0]);
    ui->glCorSlider->setValue(slice_position[1]);
    ui->glAxiSlider->setValue(slice_position[2]);
    ui->glSagBox->setValue(slice_position[0]);
    ui->glCorBox->setValue(slice_position[1]);
    ui->glAxiBox->setValue(slice_position[2]);

    ui->SlicePos->setRange(0,current_slice->geometry[cur_dim]-1);
    ui->SlicePos->setValue(current_slice->slice_pos[cur_dim]);

    glWidget->slice_pos[0] = glWidget->slice_pos[1] = glWidget->slice_pos[2] = -1;
    glWidget->updateGL();
    scene.show_slice();
}
void tracking_window::change_contrast()
{
    if(no_update)
        return;
    current_slice->set_contrast_range(ui->min_value_gl->value(),ui->max_value_gl->value());
    current_slice->set_contrast_color(ui->min_color_gl->color().rgb(),ui->max_color_gl->color().rgb());
    glWidget->slice_pos[0] = glWidget->slice_pos[1] = glWidget->slice_pos[2] = -1;
    glWidget->updateGL();
    scene.show_slice();
}

void tracking_window::on_actionEndpoints_to_seeding_triggered()
{
    std::vector<tipl::vector<3,short> > points1,points2;
    if(tractWidget->tract_models.empty() || tractWidget->currentRow() < 0)
        return;
    const float resolution_ratio = 2.0f;
    tractWidget->tract_models[size_t(tractWidget->currentRow())]->to_end_point_voxels(points1,points2,resolution_ratio);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints1"),roi_id);
    regionWidget->regions.back()->resolution_ratio = resolution_ratio;
    regionWidget->regions.back()->add_points(points1,false,resolution_ratio);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints2"),roi_id);
    regionWidget->regions.back()->resolution_ratio = resolution_ratio;
    regionWidget->regions.back()->add_points(points2,false,resolution_ratio);
    scene.show_slice();
    glWidget->updateGL();
}

void tracking_window::on_actionTracts_to_seeds_triggered()
{
    if(tractWidget->tract_models.empty())
        return;
    std::vector<tipl::vector<3,short> > points;
    tractWidget->tract_models[tractWidget->currentRow()]->to_voxel(points,2.0f);
    if(points.size() < 2)
        return;
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text(),roi_id);
    regionWidget->regions.back()->resolution_ratio = 2.0;
    regionWidget->add_points(points,false,false,2.0);
    scene.show_slice();
    glWidget->updateGL();
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
    tipl::matrix<4,4,float> tr;
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
    tipl::matrix<4,4,float> tr;
    tr.identity();
    tr[0] = tr[5] = tr[10] = ratio;
    tipl::geometry<3> new_geo(handle->dim[0]*ratio,handle->dim[1]*ratio,handle->dim[2]*ratio);
    tipl::vector<3,float> new_vs(handle->vs);
    new_vs /= (float)ratio;
    tractWidget->export_tract_density(new_geo,new_vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}

void tracking_window::on_actionTDI_Import_Slice_Space_triggered()
{
    tipl::matrix<4,4,float> tr = current_slice->invT;
    tipl::geometry<3> geo = current_slice->geometry;
    tipl::vector<3,float> vs = current_slice->voxel_size;
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(geo,vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionSave_Tracts_in_Current_Mapping_triggered()
{
    tractWidget->saveTransformedTracts(&*current_slice->invT.begin());
}
void tracking_window::on_actionSave_Endpoints_in_Current_Mapping_triggered()
{
    tractWidget->saveTransformedEndpoints(&*current_slice->invT.begin());
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
    float max_value = *std::max_element(handle->dir.fa[0],handle->dir.fa[0]+handle->dim.size());
    renderWidget->setMinMax("fa_threshold",0.0,max_value*1.1,max_value/50.0);
    if(renderWidget->getData("fa_threshold").toFloat() != 0.0)
        set_data("fa_threshold",
                 renderWidget->getData("otsu_threshold").toFloat()*
                 tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim)));
    scene.show_slice();
}

void tracking_window::on_dt_index_currentIndexChanged(int index)
{
    handle->dir.set_dt_index(index-1); // skip the first "none" item
    scene.show_slice();
}



void tracking_window::on_deleteSlice_clicked()
{
    if(dynamic_cast<CustomSliceModel*>(current_slice.get()) == 0)
        return;
    int index = ui->SliceModality->currentIndex();
    handle->view_item.erase(handle->view_item.begin()+index);
    slices.erase(slices.begin()+index);
    ui->SliceModality->setCurrentIndex(0);
    ui->SliceModality->removeItem(index);
    updateSlicesMenu();
}


bool tracking_window::can_map_to_mni(void)
{
    return handle->can_map_to_mni();
}

void tracking_window::on_actionSave_Tracts_in_MNI_space_triggered()
{
    if(!can_map_to_mni())
    {
        QMessageBox::information(this,"Error",handle->error_msg.c_str(),0);
        return;
    }
    if(handle->is_qsdr)
        tractWidget->saveTransformedTracts(&*(handle->trans_to_mni.begin()));
    else
        tractWidget->saveTransformedTracts(nullptr);
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
            QMessageBox::information(this,"DSI Studio","View position and slice location memorized",0);
        }
        else
        {
            QString value = settings.value(key_str,"").toString();
            if(value == "")
                return;
            std::istringstream in(value.toLocal8Bit().begin());
            int sag,cor,axi;
            in >> sag >> cor >> axi;
            std::vector<float> tran((std::istream_iterator<float>(in)),(std::istream_iterator<float>()));
            if(tran.size() != 16)
                return;
            std::copy(tran.begin(),tran.begin()+16,glWidget->transformation_matrix.begin());
            ui->glSagSlider->setValue(sag);
            ui->glCorSlider->setValue(cor);
            ui->glAxiSlider->setValue(axi);
            glWidget->updateGL();
        }
    }
    if(event->isAccepted())
        return;
    QWidget::keyPressEvent(event);

}



void tracking_window::on_actionManual_Registration_triggered()
{
    if(!handle->load_template())
    {
        QMessageBox::information(this,"Error","No template image loaded.",0);
        return;
    }
    tipl::image<float,3> from = current_slice->get_source();
    tipl::filter::gaussian(from);
    from -= tipl::segmentation::otsu_threshold(from);
    tipl::lower_threshold(from,0.0);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                   from,handle->vs,
                                   handle->template_I,handle->template_vs,
                                   tipl::reg::affine,tipl::reg::cost_type::corr));

    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
}


void tracking_window::on_actionTract_Analysis_Report_triggered()
{
    set_data("tract_color_style",2);// local directional
    if(!tact_report_imp.get())
        tact_report_imp.reset(new tract_report(this));
    tact_report_imp->show();
    tact_report_imp->refresh_report();
}

void tracking_window::on_actionConnectivity_matrix_triggered()
{
    if(!tractWidget->tract_models.size())
    {
        QMessageBox::information(this,"DSI Studio","Run fiber tracking first",0);
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
        QMessageBox::information(this,"DSI Studio","Float 3D window again to maximize it",0);
    }
}

void tracking_window::on_actionSave_tracking_parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",QFileInfo(windowTitle()).baseName()+".ini","Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    QSettings s(filename, QSettings::IniFormat);
    QStringList param_list = renderWidget->getChildren("Tracking");
    for(unsigned int index = 0;index < param_list.size();++index)
        s.setValue(param_list[index],renderWidget->getData(param_list[index]));

}

void tracking_window::on_actionLoad_tracking_parameters_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open INI files",QFileInfo(windowTitle()).absolutePath(),"Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    QSettings s(filename, QSettings::IniFormat);
    QStringList param_list = renderWidget->getChildren("Tracking");
    for(unsigned int index = 0;index < param_list.size();++index)
        if(s.contains(param_list[index]))
            set_data(param_list[index],s.value(param_list[index]));
}

void tracking_window::on_actionSave_Rendering_Parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",QFileInfo(windowTitle()).baseName()+".ini","Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    QSettings s(filename, QSettings::IniFormat);
    QStringList param_list;
    param_list += renderWidget->getChildren("Rendering");
    param_list += renderWidget->getChildren("Slice");
    param_list += renderWidget->getChildren("Tract");
    param_list += renderWidget->getChildren("Region");
    param_list += renderWidget->getChildren("Surface");
    param_list += renderWidget->getChildren("ODF");
    for(unsigned int index = 0;index < param_list.size();++index)
        s.setValue(param_list[index],renderWidget->getData(param_list[index]));
}

void tracking_window::on_actionLoad_Rendering_Parameters_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open INI files",QFileInfo(windowTitle()).absolutePath(),"Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    QSettings s(filename, QSettings::IniFormat);
    QStringList param_list;
    param_list += renderWidget->getChildren("Rendering");
    param_list += renderWidget->getChildren("Slice");
    param_list += renderWidget->getChildren("Tract");
    param_list += renderWidget->getChildren("Region");
    param_list += renderWidget->getChildren("Surface");
    param_list += renderWidget->getChildren("ODF");
    for(unsigned int index = 0;index < param_list.size();++index)
        if(s.contains(param_list[index]))
            set_data(param_list[index],s.value(param_list[index]));
    glWidget->updateGL();
}

void tracking_window::on_addRegionFromAtlas_clicked()
{
    if(handle->atlas_list.empty() || !handle->can_map_to_mni())
    {
        QMessageBox::information(this,"Error",handle->error_msg.c_str());
        raise();
        return;
    }
    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,handle));
    if(atlas_dialog->exec() == QDialog::Accepted)
    {
        if(!handle->atlas_list[atlas_dialog->atlas_index]->load_from_file())
        {
            QMessageBox::information(this,"Error",handle->atlas_list[atlas_dialog->atlas_index]->error_msg.c_str());
            return;
        }
        begin_prog("adding regions");
        regionWidget->begin_update();
        for(unsigned int i = 0;check_prog(i,atlas_dialog->roi_list.size());++i)
        {
            regionWidget->add_region_from_atlas(handle->atlas_list[atlas_dialog->atlas_index],atlas_dialog->roi_list[i]);
            raise();
        }
        regionWidget->end_update();
        glWidget->updateGL();
        scene.show_slice();
    }
    }


void tracking_window::on_actionRestore_Settings_triggered()
{
    command("restore_rendering");
}


void tracking_window::on_actionRestore_Tracking_Settings_triggered()
{
    renderWidget->setDefault("Tracking");
    on_tracking_index_currentIndexChanged((*this)["tracking_index"].toInt());
    glWidget->updateGL();
}

void tracking_window::set_roi_zoom(float zoom)
{
    ui->zoom->setValue(zoom);
}

void tracking_window::on_actionQuality_Assessment_triggered()
{
    float threshold = renderWidget->getData("otsu_threshold").toFloat()*
                tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    std::pair<float,float> result = evaluate_fib(handle->dim,threshold,handle->dir.fa,
                                                 [&](int pos,char fib)
                                                 {return tipl::vector<3>(handle->dir.get_dir(pos,fib));});
    std::ostringstream out;
    out << "Fiber coherence index: " << result.first << std::endl;
    out << "Fiber discoherent index: " << result.second << std::endl;
    show_info_dialog("Quality assessment",out.str().c_str());
}

#include "tessellated_icosahedron.hpp"
void tracking_window::on_actionImprove_Quality_triggered()
{
    tracking_data fib;
    fib.read(*handle);
    float threshold = renderWidget->getData("otsu_threshold").toFloat()*
                tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
    if(!fib.dir.empty())
        return;
    for(float cos_angle = 0.99f;check_prog(1000-cos_angle*1000,1000-866);cos_angle -= 0.005f)
    {
        std::vector<std::vector<float> > new_fa(handle->dir.num_fiber);
        std::vector<std::vector<short> > new_index(handle->dir.num_fiber);
        unsigned int size = handle->dim.size();
        for(unsigned int i = 0 ;i < new_fa.size();++i)
        {
            new_fa[i].resize(size);
            new_index[i].resize(size);
            std::copy(handle->dir.fa[i],handle->dir.fa[i]+size,new_fa[i].begin());
            std::copy(handle->dir.findex[i],handle->dir.findex[i]+size,new_index[i].begin());
        }

        auto I = tipl::make_image(handle->dir.fa[0],handle->dim);
        I.for_each_mt([&](float value,tipl::pixel_index<3> index)
        {
            if(value < threshold)
                return;
            std::vector<tipl::pixel_index<3> > neighbors;
            tipl::get_neighbors(index,handle->dim,neighbors);

            std::vector<tipl::vector<3> > dis(neighbors.size());
            std::vector<tipl::vector<3> > fib_dir(neighbors.size());
            std::vector<float> fib_fa(neighbors.size());


            for(unsigned char i = 0;i < neighbors.size();++i)
            {
                dis[i] = neighbors[i];
                dis[i] -= tipl::vector<3>(index);
                dis[i].normalize();
                unsigned char fib_order,reverse;
                if(fib.get_nearest_dir_fib(neighbors[i].index(),dis[i],fib_order,reverse,threshold,cos_angle,0))
                {
                    fib_dir[i] = handle->dir.get_dir(neighbors[i].index(),fib_order);
                    if(reverse)
                        fib_dir[i] = -fib_dir[i];
                    fib_fa[i] = handle->dir.fa[fib_order][neighbors[i].index()];
                }
            }


            for(unsigned char i = 0;i < neighbors.size();++i)
            if(fib_fa[i] > threshold)
            {
                for(unsigned char j = i+1;j < neighbors.size();++j)
                if(fib_fa[j] > threshold)
                {
                    float angle = fib_dir[i]*fib_dir[j];
                    if(angle > -cos_angle) // select opposite side
                        continue;
                    tipl::vector<3> predict_dir(fib_dir[i]);
                    if(angle > 0)
                        predict_dir += fib_dir[j];
                    else
                        predict_dir -= fib_dir[j];
                    predict_dir.normalize();
                    unsigned char fib_order,reverse;
                    bool has_match = false;
                    if(fib.get_nearest_dir_fib(index.index(),predict_dir,fib_order,reverse,threshold,cos_angle,0))
                    {
                        if(reverse)
                            predict_dir -= tipl::vector<3>(handle->dir.get_dir(index.index(),fib_order));
                        else
                            predict_dir += tipl::vector<3>(handle->dir.get_dir(index.index(),fib_order));
                        predict_dir.normalize();
                        has_match = true;
                    }
                    short dir_index = 0;
                    float max_value = 0.0;
                    for (unsigned int k = 0; k < handle->dir.half_odf_size; ++k)
                    {
                        float value = std::abs(predict_dir*handle->dir.odf_table[k]);
                        if (value > max_value)
                        {
                            max_value = value;
                            dir_index = k;
                        }
                    }

                    if(has_match)
                        new_index[fib_order][index.index()] = dir_index;
                    else
                    {
                        float add_fa = (fib_fa[i]+fib_fa[j])*0.5;
                        for(unsigned char m = 0;m < new_fa.size();++m)
                        if(add_fa > new_fa[m][index.index()])
                        {
                            std::swap(add_fa,new_fa[m][index.index()]);
                            std::swap(dir_index,new_index[m][index.index()]);
                        }
                    }
                }
            }

        });
        for(unsigned int i = 0 ;i < new_fa.size();++i)
        {
            std::copy(new_fa[i].begin(),new_fa[i].begin()+size,(float*)handle->dir.fa[i]);
            std::copy(new_index[i].begin(),new_index[i].begin()+size,(short*)handle->dir.findex[i]);
        }
    }
    scene.show_slice();
}

void tracking_window::on_actionAuto_Rotate_triggered(bool checked)
{
    if(checked)
        {
            glWidget->time.start();
            glWidget->last_time = glWidget->time.elapsed();
            timer.reset(new QTimer());
            timer->setInterval(1);
            connect(timer.get(), SIGNAL(timeout()), glWidget, SLOT(rotate()));
            timer->start();
        }
    else
        timer->stop();
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
    glWidget->updateGL();
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
extern std::string t1w_template_file_name,t1w_mask_template_file_name;
void tracking_window::stripSkull()
{
    if(!ui->SliceModality->currentIndex() || !handle->is_human_data || handle->is_qsdr)
        return;
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice || !reg_slice->skull_removed_images.empty())
        return;
    gz_nifti in1,in2;
    tipl::image<float,3> It,Iw,J(reg_slice->get_source());
    if(!in1.load_from_file(t1w_template_file_name.c_str()) || !in1.toLPS(It))
        return;
    if(!in2.load_from_file(t1w_mask_template_file_name.c_str()) || !in2.toLPS(Iw))
        return;
    tipl::vector<3> vs,vsJ(reg_slice->voxel_size);
    in1.get_voxel_size(vs);

    tipl::downsampling(It);
    tipl::downsampling(It);
    tipl::downsampling(Iw);
    tipl::downsampling(Iw);
    tipl::downsampling(J);
    tipl::downsampling(J);
    vs *= 4.0f;
    vsJ *= 4.0f;

    QMessageBox::information(this,"DSI Studio","Please align brain images to visualize isosurface.");
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
            It,vs,J,vsJ,tipl::reg::affine,tipl::reg::cost_type::mutual_info));

    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
    auto T = tipl::transformation_matrix<double>(manual->arg,
        It.geometry(),vs,J.geometry(),vsJ);

    tipl::multiply_constant(T.data,T.data+12,4.0f);
    tipl::transformation_matrix<double> iT = T;
    iT.inverse();

    tipl::filter::mean(Iw);
    tipl::filter::mean(Iw);

    tipl::image<float,3> Iw_(reg_slice->source_images.geometry());
    tipl::resample_mt(Iw,Iw_,iT,tipl::linear);

    reg_slice->skull_removed_images = reg_slice->source_images;
    reg_slice->skull_removed_images *= Iw_;
    scene.show_slice();
}

void tracking_window::on_show_fiber_toggled(bool checked)
{
    ui->show_fiber->setChecked(checked);
    if(ui->show_fiber->isChecked() ^ (*this)["roi_fiber"].toBool())
        set_data("roi_fiber",ui->show_fiber->isChecked());
    scene.show_slice();
}
void tracking_window::on_show_edge_toggled(bool checked)
{
    ui->show_edge->setChecked(checked);
    if(ui->show_edge->isChecked() ^ (*this)["roi_edge"].toBool())
        set_data("roi_edge",ui->show_edge->isChecked());
    scene.show_slice();

}
void tracking_window::on_show_r_toggled(bool checked)
{
    ui->show_r->setChecked(checked);
    if(ui->show_r->isChecked() ^ (*this)["roi_label"].toBool())
        set_data("roi_label",ui->show_r->isChecked());
    scene.show_slice();
}
void tracking_window::on_show_3view_toggled(bool checked)
{
    ui->show_3view->setChecked(checked);
    set_data("roi_layout",checked ? 1:0);
    if(checked)
        glWidget->updateGL();
    scene.show_slice();
}

void tracking_window::on_show_position_toggled(bool checked)
{
    ui->show_position->setChecked(checked);
    if(ui->show_position->isChecked() ^ (*this)["roi_position"].toBool())
        set_data("roi_position",ui->show_position->isChecked());
    scene.show_slice();
}
void tracking_window::on_show_ruler_toggled(bool checked)
{
    ui->show_ruler->setChecked(checked);
    if(ui->show_ruler->isChecked() ^ (*this)["roi_ruler"].toBool())
        set_data("roi_ruler",ui->show_ruler->isChecked());
    scene.show_slice();
}



void tracking_window::on_actionAdjust_Mapping_triggered()
{
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice || !ui->SliceModality->currentIndex())
    {
        QMessageBox::information(this,"Error","In the region window to the left, select the inserted slides to adjust mapping");
        return;
    }
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
        slices[0]->get_source(),slices[0]->voxel_size,
        reg_slice->get_source(),reg_slice->voxel_size,
            tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
    reg_slice->terminate();
    reg_slice->arg_min = manual->arg;
    reg_slice->update();
    reg_slice->is_diffusion_space = false;
    glWidget->updateGL();
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
            "Save Mapping Matrix",QFileInfo(windowTitle()).completeBaseName()+".mapping.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toLocal8Bit().begin());

    for(int row = 0,index = 0;row < 4;++row)
    {
        for(int col = 0;col < 4;++col,++index)
            out << reg_slice->T[index] << " ";
        out << std::endl;
    }
}

void tracking_window::on_actionLoad_mapping_triggered()
{
    if(!ui->SliceModality->currentIndex())
        return;
    CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!reg_slice)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Mapping Matrix",QFileInfo(windowTitle()).absolutePath(),"Text files (*.txt);;All files (*)");
    std::ifstream in(filename.toLocal8Bit().begin());
    if(filename.isEmpty() || !in)
        return;
    reg_slice->terminate();
    std::vector<float> data;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),std::back_inserter(data));
    data.resize(16);
    data[15] = 1.0;
    reg_slice->T = data;
    reg_slice->invT = data;
    reg_slice->invT.inv();
    glWidget->updateGL();
}

void tracking_window::on_actionInsert_MNI_images_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open MNI Image",QFileInfo(windowTitle()).absolutePath(),"Image files (*.hdr *.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty() || !can_map_to_mni() || handle->get_mni_mapping().empty())
        return;

    gz_nifti reader;
    if(!reader.load_from_file(filename.toStdString()))
    {
        QMessageBox::information(this,"Error","Cannot open the nifti file",0);
        return;
    }
    tipl::image<float,3> I,J(handle->mni_position.geometry());
    tipl::matrix<4,4,float> T;
    reader.toLPS(I);
    reader.get_image_transformation(T);
    T.inv();
    J.for_each_mt([&](float& v,const tipl::pixel_index<3>& pos)
    {
        tipl::vector<3> mni(handle->mni_position[pos.index()]);
        mni.to(T);
        tipl::estimate(I,mni,v);
    });
    QString name = QFileInfo(filename).baseName();
    std::shared_ptr<SliceModel> new_slice(new CustomSliceModel(handle.get()));
    CustomSliceModel* reg_slice_ptr = dynamic_cast<CustomSliceModel*>(new_slice.get());
    reg_slice_ptr->source_images.swap(J);
    reg_slice_ptr->T.identity();
    reg_slice_ptr->invT.identity();
    reg_slice_ptr->is_diffusion_space = true;
    reg_slice_ptr->initialize();
    slices.push_back(new_slice);
    ui->SliceModality->addItem(name);
    ui->SliceModality->setCurrentIndex(handle->view_item.size()-1);
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
    }

    // update along track color dialog
    color_bar->update_slice_indices();
}

bool tracking_window::addSlices(QStringList filenames,QString name,bool correct_intensity,bool cmd)
{
    std::vector<std::string> files(filenames.size());
    for (unsigned int index = 0; index < filenames.size(); ++index)
            files[index] = filenames[index].toLocal8Bit().begin();
    std::shared_ptr<SliceModel> new_slice(new CustomSliceModel(handle.get()));
    CustomSliceModel* reg_slice_ptr = dynamic_cast<CustomSliceModel*>(new_slice.get());
    if(!reg_slice_ptr)
        return false;
    if(!reg_slice_ptr->initialize(files,correct_intensity))
    {
        if(!cmd)
            QMessageBox::information(this,"DSI Studio",reg_slice_ptr->error_msg.c_str(),0);
        else
            std::cout << reg_slice_ptr->error_msg << std::endl;
        return false;
    }
    slices.push_back(new_slice);
    ui->SliceModality->addItem(name);

    if(!cmd && !timer2.get())
    {
        timer2.reset(new QTimer());
        timer2->setInterval(200);
        connect(timer2.get(), SIGNAL(timeout()), this, SLOT(check_reg()));
        timer2->start();
        check_reg();
    }
    ui->SliceModality->setCurrentIndex(handle->view_item.size()-1);
    updateSlicesMenu();
    return true;
}

void tracking_window::check_reg(void)
{
    bool all_ended = true;
    for(unsigned int index = 0;index < slices.size();++index)
    {
        CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(slices[index].get());
        if(reg_slice && !reg_slice->ended)
        {
            all_ended = false;
            reg_slice->update();
        }
    }
    scene.show_slice();
    if(all_ended)
        timer2.reset(0);
    else
        glWidget->updateGL();
}
void tracking_window::on_actionLoad_Color_Map_triggered()
{
    QMessageBox::information(this,"Load color map","Please assign a text file of RGB numbers as the colormap.");
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
                "Load color map","color_map.txt",
                "Text files (*.txt);;All files|(*)");
    if(filename.isEmpty())
        return;
    tipl::color_map_rgb new_color_map;
    if(!new_color_map.load_from_file(filename.toStdString().c_str()))
    {
          QMessageBox::information(this,"Error","Invalid color map format");
          return;
    }
    current_slice->v2c.set_color_map(new_color_map);
    glWidget->slice_pos[0] = glWidget->slice_pos[1] = glWidget->slice_pos[2] = -1;
    glWidget->updateGL();
    scene.show_slice();
}

void tracking_window::on_track_style_currentIndexChanged(int index)
{
    switch(index)
    {
        case 0: //Tube 1
            set_data("tract_style",1);
            set_data("bkg_color",-1);
            set_data("tract_alpha",1);
            set_data("tract_alpha_style",1);
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
            set_data("tract_alpha_style",1);
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
            set_data("tract_alpha_style",0);
            set_data("tract_bend1",4);
            set_data("tract_bend2",5);
            set_data("tract_shader",7);
            break;
        case 3:
            set_data("tract_style",0);
            set_data("tract_line_width",1.0f);
            set_data("bkg_color",0);
            set_data("tract_alpha",0.2);
            set_data("tract_alpha_style",0);
            set_data("tract_bend1",4);
            set_data("tract_bend2",1);
            set_data("tract_shader",0);
            break;
    }
    glWidget->update();
}

void tracking_window::on_addSlices_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
        this,"Open Images files",QFileInfo(windowTitle()).absolutePath(),"Image files (*.dcm *.hdr *.nii *nii.gz *.jpg *.bmp 2dseq);;All files (*)" );
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
    addSlices(filenames,QFileInfo(filenames[0]).baseName(),renderWidget->getData("slice_smoothing").toBool(),false);
}

void tracking_window::on_actionSingle_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::single;
    glWidget->updateGL();
}

void tracking_window::on_actionDouble_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::two;
    glWidget->transformation_matrix2 = glWidget->transformation_matrix;
    glWidget->rotation_matrix2 = glWidget->rotation_matrix;
    glWidget->updateGL();
}

void tracking_window::on_actionStereoscopic_triggered()
{
    glWidget->view_mode = GLWidget::view_mode_type::stereo;
    glWidget->updateGL();
}


void tracking_window::on_is_overlay_clicked()
{
    current_slice->is_overlay = (ui->is_overlay->isChecked());
    if(current_slice->is_overlay)
        overlay_slices.push_back(current_slice);
    else
    {
        for(size_t index = 0;index < overlay_slices.size();++index)
            if(current_slice == overlay_slices[index])
                overlay_slices.erase(overlay_slices.begin()+int64_t(index));
    }
}



void tracking_window::on_actionOpen_Connectivity_Matrix_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
        this,"Open Connectivity Matrices files",QFileInfo(windowTitle()).absolutePath(),
                "Connectivity files (*.mat);;All files (*)" );
    if( filenames.isEmpty())
        return;
    tipl::image<float,2> connectivity;
    std::string atlas;
    for(int i = 0;i < filenames.size();++i)
    {
        tipl::io::mat_read in;
        if(!in.load_from_file(filenames[i].toStdString().c_str()))
        {
            QMessageBox::information(this,"Error",QString("Failed to load file:")+filenames[i],0);
            return;
        }
        if(!in.has("atlas"))
        {
            QMessageBox::information(this,"Error",QString("Cannot find atlas matrix in file:")+filenames[i],0);
            return;
        }
        if(i == 0)
            atlas = in.read_string("atlas");
        else
        {
            if(atlas != in.read_string("atlas"))
            {
                QMessageBox::information(this,"Error",QString("Inconsistent atlas setting in file:")+filenames[i],0);
                return;
            }
        }
        unsigned int row,col;
        const float* buf = nullptr;
        if(!in.read("connectivity",row,col,buf))
        {
            QMessageBox::information(this,"Error",QString("Cannot find connectivity matrix in file:")+filenames[i],0);
            return;
        }
        if(i == 0)
        {
            connectivity.resize(tipl::geometry<2>(row,col));
            std::copy(buf,buf+row*col,connectivity.begin());
        }
        else
        {
            if(row != connectivity.width() || col != connectivity.height())
            {
                QMessageBox::information(this,"Error",QString("Inconsistent matrix size in file:")+filenames[i],0);
                return;
            }
            tipl::add(connectivity.begin(),connectivity.end(),buf);
        }
    }
    tipl::multiply_constant(connectivity,1.0f/tipl::maximum(connectivity));
    for(size_t i = 0;i < connectivity.size();++i)
        if(connectivity[i] < 0.05f)
            connectivity[i] = 0.0f;
    glWidget->connectivity = std::move(connectivity);
    if(atlas != "roi")
    {
        for(size_t i = 0;i < handle->atlas_list.size();++i)
            if(atlas == handle->atlas_list[i]->name)
            {
                regionWidget->delete_all_region();
                regionWidget->begin_update();
                for(size_t j = 0;j < handle->atlas_list[i]->get_list().size();++j)
                    regionWidget->add_region_from_atlas(handle->atlas_list[i],uint32_t(j));
                regionWidget->end_update();
                return;
            }
        QMessageBox::information(this,"Error",QString("Cannot find ")+atlas.c_str()+
                    " atlas in DSI Studio. Please update DSI Studio package or check the atlas folder",0);

    }
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
    glWidget->updateGL();
    QMessageBox::information(this,"DSI Studio","Current viewing slice will reamin in the 3D window");
}
void tracking_window::on_enable_auto_track_clicked()
{
    if(!handle->load_track_atlas())
    {
        QMessageBox::information(this,"Error",handle->error_msg.c_str());
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
        threshold = renderWidget->getData("otsu_threshold").toFloat()
                        *tipl::segmentation::otsu_threshold(tipl::make_image(handle->dir.fa[0],handle->dim));
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

void tracking_window::on_template_box_activated(int index)
{
    handle->set_template_id(index);
    ui->enable_auto_track->setVisible(true);
    ui->target->setCurrentIndex(0);
    ui->target->setVisible(false);
    ui->target_label->setVisible(false);
}
void tracking_window::on_SliceModality_currentIndexChanged(int index)
{
    if(index == -1 || !current_slice.get())
        return;
    no_update = true;
    tipl::vector<3,float> slice_position(current_slice->slice_pos);
    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->T);
    current_slice = slices[index];


    ui->is_overlay->setChecked(current_slice->is_overlay);
    ui->glSagSlider->setRange(0,current_slice->geometry[0]-1);
    ui->glCorSlider->setRange(0,current_slice->geometry[1]-1);
    ui->glAxiSlider->setRange(0,current_slice->geometry[2]-1);
    ui->glSagBox->setRange(0,current_slice->geometry[0]-1);
    ui->glCorBox->setRange(0,current_slice->geometry[1]-1);
    ui->glAxiBox->setRange(0,current_slice->geometry[2]-1);


    std::pair<float,float> range = current_slice->get_value_range();
    std::pair<float,float> contrast_range = current_slice->get_contrast_range();
    std::pair<unsigned int,unsigned int> contrast_color = current_slice->get_contrast_color();
    float r = range.second-range.first;
    if(r == 0.0)
        r = 1;
    float step = r/20.0;
    ui->min_value_gl->setMinimum(range.first-r);
    ui->min_value_gl->setMaximum(range.second+r);
    ui->min_value_gl->setSingleStep(step);
    ui->min_color_gl->setColor(contrast_color.first);

    ui->max_value_gl->setMinimum(range.first-r);
    ui->max_value_gl->setMaximum(range.second+r);
    ui->max_value_gl->setSingleStep(step);
    ui->max_color_gl->setColor(contrast_color.second);

    ui->min_value_gl->setValue(contrast_range.first);
    ui->max_value_gl->setValue(contrast_range.second);

    if(!current_slice->is_diffusion_space)
        slice_position.to(current_slice->invT);

    move_slice_to(slice_position);
    no_update = false;
}



void tracking_window::on_actionSave_T1W_T2W_images_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice)
        return;
    QString filename = QFileDialog::getSaveFileName(
        this,"Save T1W/T2W Image",QFileInfo(windowTitle()).absolutePath()+"//"+slice->name.c_str()+"_modified.nii.gz","Image files (*nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    auto I = slice->source_images;
    gz_nifti::save_to_file(filename.toStdString().c_str(),I,slice->voxel_size,slice->trans);
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
    tipl::image<unsigned char, 3> mask;
    current_region->SaveToBuffer(mask);
    float resolution_ratio = current_region->resolution_ratio;
    tipl::matrix<4,4,float> T(slice->T);
    tipl::multiply_constant(&T[0],&T[0]+12,resolution_ratio);
    tipl::image<unsigned char, 3> t_mask(slice->source_images.geometry());
    tipl::resample(mask,t_mask,T,tipl::nearest);
    for(size_t i = 0;i < t_mask.size();++i)
        if(t_mask[i])
            slice->source_images[i] = mark_value;
    scene.show_slice();
    glWidget->updateGL();
}

void paint_track_on_volume(tipl::image<unsigned char,3>& track_map,const std::vector<float>& tracks);
void tracking_window::on_actionMark_Tracts_on_T1W_T2W_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty())
        return;
    if(tractWidget->tract_models.empty())
        return;
    bool ok = true;
    double ratio = QInputDialog::getDouble(this,"DSI Studio",
            "Aissgn intensity (ratio to the maximum, e.g., 1.2 = 1.2*max)",1.0,0.0,10.0,1,&ok);
    if(!ok)
        return;
    tipl::image<unsigned char, 3> t_mask(slice->source_images.geometry());
    auto checked_tracks = tractWidget->get_checked_tracks();
    tipl::par_for(checked_tracks.size(),[&](size_t i){
        for(size_t j = 0;j < checked_tracks[i]->get_visible_track_count();++j)
        {
            std::vector<float> tracks = checked_tracks[i]->get_tract(j);
            for(size_t k = 0;k < tracks.size();k +=3)
            {
                tipl::vector<3> p(&tracks[0] + k);
                p.to(slice->invT);
                tracks[k] = p[0];
                tracks[k+1] = p[1];
                tracks[k+2] = p[2];
            }
            paint_track_on_volume(t_mask,tracks);
        }
    });
    float mark_value = slice->get_value_range().second*float(ratio);
    for(size_t i = 0;i < t_mask.size();++i)
        if(t_mask[i])
            slice->source_images[i] = mark_value;
    scene.show_slice();
    glWidget->updateGL();
}


void tracking_window::on_actionApply_Operation_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty())
        return;
    QString cmd = QInputDialog::getText(this,"Apply Operation","Please specify command");
    if(cmd == "upsample2")
    {
        slice->terminate();
        slice->voxel_size *= 0.5;
        tipl::matrix<4,4,float> nT;
        nT.identity();
        nT[0] = nT[5] = nT[10] = 0.5f;
        slice->T = slice->T*nT;
        slice->invT = tipl::inverse(slice->T);
        tipl::image<float,3> J(tipl::geometry<3>(slice->source_images.width()*2,slice->source_images.height()*2,slice->source_images.depth()*2));
        tipl::resample_mt(slice->source_images,J,nT,tipl::cubic);
        slice->source_images.swap(J);
        slice->initialize();
        on_SliceModality_currentIndexChanged(ui->SliceModality->currentIndex());
        return;
    }
    if(cmd == "smoothing")
    {
        tipl::filter::mean(slice->source_images);
        on_SliceModality_currentIndexChanged(ui->SliceModality->currentIndex());
        return;
    }
}

void tracking_window::on_actionSave_Slices_to_DICOM_triggered()
{
    CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
    if(!slice || slice->source_images.empty())
        return;

    QMessageBox::information(this,"DSI Studio","Please assign the original DICOM files");
    QStringList files = QFileDialog::getOpenFileNames(this,"Assign DICOM files",
                                                      QFileInfo(windowTitle()).absolutePath(),"DICOM files (*.dcm);;All files (*)");
    if(files.isEmpty())
        return;
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
    int failcode= 0;
    std::vector<char> slice_matched(size_t(is_axial ? I.depth():I.width()));
    begin_prog("Writing data");
    tipl::par_for2(files.size(),[&](int i,int id)
    {
        if(id == 0)
            check_prog(i,files.size());
        std::vector<char> buf;
        std::ifstream in(files[i].toStdString().c_str(),std::ios::binary);
        in.seekg(0,in.end);
        buf.resize(size_t(in.tellg()));
        in.seekg(0,in.beg);
        if(read_size*sizeof(short) > buf.size())
        {
            failcode = 1;
            return;
        }
        if(!in.read(&buf[0],uint64_t(buf.size())))
        {
            failcode = 4;
            return;
        }
        std::vector<short> image_buf(read_size);
        std::copy(buf.end()-read_size*sizeof(short),buf.end(),(char*)&image_buf[0]);
        for(size_t j = 0;j < slice_matched.size();++j)
        {
            if(slice_matched[j])
                continue;
            tipl::image<short,2> slice;
            tipl::volume2slice(I, slice, is_axial ? 2 : 0, j);
            if(!is_axial)
                tipl::flip_y(slice);
            size_t mismatched_count = 0;
            size_t check_count = slice.size()/3;
            size_t max_mismatch_count = check_count/2;
            for(size_t k = 0;k < check_count && mismatched_count < max_mismatch_count;++k)
                if(image_buf[k+check_count] != slice[k+check_count])
                    ++mismatched_count;
            if(mismatched_count < max_mismatch_count) // found a match
            {
                std::copy(slice.begin(),slice.end(),(short*)&(char&)*(buf.end()-read_size*sizeof(short)));
                QFileInfo info(files[i]);
                QString output_name = info.path() + "/mod_" + info.completeBaseName() + ".dcm";
                std::ofstream out(output_name.toStdString().c_str(),std::ios::binary);
                if(!out)
                {
                    failcode = 3;
                    return;
                }
                out.write(&buf[0],int64_t(buf.size()));
                slice_matched[j] = 1;
                return;
            }
        }
        failcode = 2;
    });
    check_prog(0,0);
    if(failcode == 0)
        QMessageBox::information(this,"DSI Studio","File saved");
    if(failcode == 1)
        QMessageBox::information(this,"DSI Studio","Compressed DICOM is not supported. Please convert DICOM to uncompressed format.");
    if(failcode == 2)
        QMessageBox::information(this,"DSI Studio","Subject DICOM mismatch");
    if(failcode == 3)
        QMessageBox::information(this,"DSI Studio","Cannot output DICOM");
    if(failcode == 4)
        QMessageBox::information(this,"DSI Studio","Read DICOM failed");

}

void tracking_window::on_zoom_valueChanged(double arg1)
{
    if(float(arg1) == (*this)["roi_zoom"].toFloat())
        return;
    set_data("roi_zoom",arg1);
    scene.center();
    scene.show_slice();
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
    set_data("initial_direction",int(param.initial_direction));
    set_data("interpolation",int(param.interpolation_strategy));
    set_data("tracking_plan",int(param.stop_by_tract));
    set_data("seed_plan",int(param.center_seed));
    set_data("random_seed",int(param.random_seed));
    set_data("check_ending",int(param.check_ending));
    set_data("track_count",int(param.termination_count));

    set_data("otsu_threshold",float(param.default_otsu));
    set_data("auto_tip",int(param.tip_iteration));

}

