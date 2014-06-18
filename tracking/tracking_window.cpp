#include <utility>
#include <QFileDialog>
#include <QSplitter>
#include <QSettings>
#include <QClipboard>
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "tracking_static_link.h"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
#include "region/regiontablewidget.h"
#include <QApplication>
#include <QScrollBar>
#include <QMouseEvent>
#include <QMessageBox>
#include "tracking_model.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "manual_alignment.h"
#include "tract_report.hpp"
#include "color_bar_dialog.hpp"
#include "connectivity_matrix_dialog.h"
#include "mapping/atlas.hpp"
#include "mapping/fa_template.hpp"
#include "tracking/atlasdialog.h"

extern std::vector<atlas> atlas_list;
extern fa_template fa_template_imp;
QByteArray default_geo,default_state;

void tracking_window::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
}

QVariant tracking_window::operator[](QString name) const
{
    return renderWidget->getData(name);
}

tracking_window::tracking_window(QWidget *parent,ODFModel* new_handle,bool handle_release_) :
        QMainWindow(parent),handle(new_handle),handle_release(handle_release_),
        ui(new Ui::tracking_window),scene(*this,new_handle),slice(new_handle),gLdock(0),atlas_dialog(0)

{

    ODFModel* odf_model = (ODFModel*)handle;
    FibData& fib_data = odf_model->fib_data;

    odf_size = fib_data.fib.odf_table.size();
    odf_face_size = fib_data.fib.odf_faces.size();
    has_odfs = fib_data.fib.has_odfs() ? 1:0;
    // check whether first index is "fa0"
    is_dti = (fib_data.view_item[0].name[0] == 'f');

    ui->setupUi(this);
    {
        setGeometry(10,10,800,600);

        ui->regionDockWidget->setMinimumWidth(0);
        ui->dockWidget->setMinimumWidth(0);
        ui->renderingLayout->addWidget(renderWidget = new RenderingTableWidget(*this,ui->renderingWidgetHolder));
        ui->main_layout->insertWidget(1,glWidget = new GLWidget(renderWidget->getData("anti_aliasing").toInt(),*this,renderWidget));
        ui->verticalLayout_3->addWidget(regionWidget = new RegionTableWidget(*this,ui->regionDockWidget));
        ui->tractverticalLayout->addWidget(tractWidget = new TractTableWidget(*this,ui->TractWidgetHolder));
        ui->graphicsView->setScene(&scene);
        ui->graphicsView->setCursor(Qt::CrossCursor);
        scene.statusbar = ui->statusbar;
        color_bar.reset(new color_bar_dialog(this));
    }



    // setup sliders
    {
        slice_no_update = true;
        ui->SagSlider->setRange(0,slice.geometry[0]-1);
        ui->CorSlider->setRange(0,slice.geometry[1]-1);
        ui->AxiSlider->setRange(0,slice.geometry[2]-1);
        ui->SagSlider->setValue(slice.slice_pos[0]);
        ui->CorSlider->setValue(slice.slice_pos[1]);
        ui->AxiSlider->setValue(slice.slice_pos[2]);

        ui->glSagBox->setRange(0,slice.geometry[0]-1);
        ui->glCorBox->setRange(0,slice.geometry[1]-1);
        ui->glAxiBox->setRange(0,slice.geometry[2]-1);
        ui->glSagBox->setValue(slice.slice_pos[0]);
        ui->glCorBox->setValue(slice.slice_pos[1]);
        ui->glAxiBox->setValue(slice.slice_pos[2]);
        slice_no_update = false;
        on_SliceModality_currentIndexChanged(0);

        for (unsigned int index = 0;index < fib_data.view_item.size(); ++index)
        {
            ui->sliceViewBox->addItem(fib_data.view_item[index].name.c_str());
            if(fib_data.view_item[index].is_overlay)
                ui->overlay->addItem(fib_data.view_item[index].name.c_str());
        }
        ui->sliceViewBox->setCurrentIndex(0);
        ui->overlay->setCurrentIndex(0);
        if(ui->overlay->count() == 1)
           ui->overlay->hide();
    }

    is_qsdr = !handle->fib_data.trans_to_mni.empty();

    // setup atlas
    if(!fa_template_imp.I.empty() &&
        handle->fib_data.dim[0]*handle->fib_data.vs[0] > 100 &&
        handle->fib_data.dim[1]*handle->fib_data.vs[1] > 120 &&
        handle->fib_data.dim[2]*handle->fib_data.vs[2] > 50 && !is_qsdr)
    {
        mi3_arg.scaling[0] = slice.voxel_size[0] / std::fabs(fa_template_imp.tran[0]);
        mi3_arg.scaling[1] = slice.voxel_size[1] / std::fabs(fa_template_imp.tran[5]);
        mi3_arg.scaling[2] = slice.voxel_size[2] / std::fabs(fa_template_imp.tran[10]);
        image::reg::align_center(slice.source_images,fa_template_imp.I,mi3_arg);
        mi3.reset(new manual_alignment(this,slice.source_images,fa_template_imp.I,mi3_arg));
    }
    else
        ui->actionManual_Registration->setEnabled(false);
    ui->actionConnectometry->setEnabled(handle->fib_data.fib.has_odfs() && is_qsdr);



    {
        if(is_dti)
            ui->actionQuantitative_anisotropy_QA->setText("Save FA...");
        for (int index = fib_data.other_mapping_index; index < fib_data.view_item.size(); ++index)
            {
                std::string& name = fib_data.view_item[index].name;
                QAction* Item = new QAction(this);
                Item->setText(QString("Save %1...").arg(name.c_str()));
                Item->setData(QString(name.c_str()));
                Item->setVisible(true);
                connect(Item, SIGNAL(triggered()),tractWidget, SLOT(save_tracts_data_as()));
                ui->menuSave->addAction(Item);
            }
    }

    // opengl
    {
        connect(ui->glSagSlider,SIGNAL(valueChanged(int)),this,SLOT(glSliderValueChanged()));
        connect(ui->glCorSlider,SIGNAL(valueChanged(int)),this,SLOT(glSliderValueChanged()));
        connect(ui->glAxiSlider,SIGNAL(valueChanged(int)),this,SLOT(glSliderValueChanged()));

        connect(ui->glSagCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->glCorCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->glAxiCheck,SIGNAL(stateChanged(int)),glWidget,SLOT(updateGL()));

        connect(ui->glSagView,SIGNAL(clicked()),this,SLOT(on_SagView_clicked()));
        connect(ui->glCorView,SIGNAL(clicked()),this,SLOT(on_CorView_clicked()));
        connect(ui->glAxiView,SIGNAL(clicked()),this,SLOT(on_AxiView_clicked()));

        connect(ui->addSlices,SIGNAL(clicked()),this,SLOT(on_actionInsert_T1_T2_triggered()));
        connect(ui->actionAdd_surface,SIGNAL(triggered()),glWidget,SLOT(addSurface()));
        connect(ui->SliceModality,SIGNAL(currentIndexChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->actionSave_Screen,SIGNAL(triggered()),glWidget,SLOT(catchScreen()));
        connect(ui->actionSave_3D_screen_in_high_resolution,SIGNAL(triggered()),glWidget,SLOT(catchScreen2()));
        connect(ui->actionLoad_Camera,SIGNAL(triggered()),glWidget,SLOT(loadCamera()));
        connect(ui->actionSave_Camera,SIGNAL(triggered()),glWidget,SLOT(saveCamera()));
        connect(ui->actionLoad_mapping,SIGNAL(triggered()),glWidget,SLOT(loadMapping()));
        connect(ui->actionSave_mapping,SIGNAL(triggered()),glWidget,SLOT(saveMapping()));
        connect(ui->actionSave_Rotation_Images,SIGNAL(triggered()),glWidget,SLOT(saveRotationSeries()));
        connect(ui->actionSave_Left_Right_3D_Image,SIGNAL(triggered()),glWidget,SLOT(saveLeftRight3DImage()));
    }
    // scene view
    {
        connect(ui->SagSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->CorSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->AxiSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));


        connect(&scene,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(&scene,SIGNAL(need_update()),glWidget,SLOT(updateGL()));

        connect(ui->actionAxial_View,SIGNAL(triggered()),this,SLOT(on_AxiView_clicked()));
        connect(ui->actionCoronal_View,SIGNAL(triggered()),this,SLOT(on_CorView_clicked()));
        connect(ui->actionSagittal_view,SIGNAL(triggered()),this,SLOT(on_SagView_clicked()));


        connect(ui->actionSave_ROI_Screen,SIGNAL(triggered()),&scene,SLOT(catch_screen()));

        connect(ui->actionSave_Anisotrpy_Map_as,SIGNAL(triggered()),&scene,SLOT(save_slice_as()));


        connect(ui->overlay,SIGNAL(currentIndexChanged(int)),this,SLOT(on_sliceViewBox_currentIndexChanged(int)));

    }

    // regions
    {

        connect(regionWidget,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(regionWidget,SIGNAL(need_update()),glWidget,SLOT(updateGL()));
        connect(ui->whole_brain,SIGNAL(clicked()),regionWidget,SLOT(whole_brain()));

        connect(ui->actionNewRegion,SIGNAL(triggered()),regionWidget,SLOT(new_region()));
        connect(ui->actionOpenRegion,SIGNAL(triggered()),regionWidget,SLOT(load_region()));
        connect(ui->actionSaveRegionAs,SIGNAL(triggered()),regionWidget,SLOT(save_region()));
        connect(ui->actionSave_All_Regions_As,SIGNAL(triggered()),regionWidget,SLOT(save_all_regions()));
        connect(ui->actionSave_Voxel_Data_As,SIGNAL(triggered()),regionWidget,SLOT(save_region_info()));
        connect(ui->actionDeleteRegion,SIGNAL(triggered()),regionWidget,SLOT(delete_region()));
        connect(ui->actionDeleteRegionAll,SIGNAL(triggered()),regionWidget,SLOT(delete_all_region()));

        connect(ui->actionCopy_Region,SIGNAL(triggered()),regionWidget,SLOT(copy_region()));

        // actions
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



        connect(ui->actionSmoothing,SIGNAL(triggered()),regionWidget,SLOT(action_smoothing()));
        connect(ui->actionErosion,SIGNAL(triggered()),regionWidget,SLOT(action_erosion()));
        connect(ui->actionDilation,SIGNAL(triggered()),regionWidget,SLOT(action_dilation()));
        connect(ui->actionNegate,SIGNAL(triggered()),regionWidget,SLOT(action_negate()));
        connect(ui->actionDefragment,SIGNAL(triggered()),regionWidget,SLOT(action_defragment()));

        connect(ui->actionCheck_all_regions,SIGNAL(triggered()),regionWidget,SLOT(check_all()));
        connect(ui->actionUnckech_all_regions,SIGNAL(triggered()),regionWidget,SLOT(uncheck_all()));

        connect(ui->actionWhole_brain_seeding,SIGNAL(triggered()),regionWidget,SLOT(whole_brain()));
        connect(ui->actionRegion_statistics,SIGNAL(triggered()),regionWidget,SLOT(show_statistics()));

        connect(ui->region_up,SIGNAL(clicked()),regionWidget,SLOT(move_up()));
        connect(ui->region_down,SIGNAL(clicked()),regionWidget,SLOT(move_down()));
    }
    // tracts
    {
        connect(ui->perform_tracking,SIGNAL(clicked()),tractWidget,SLOT(start_tracking()));

        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(makeTracts()));
        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(updateGL()));

        connect(glWidget,SIGNAL(edited()),tractWidget,SLOT(edit_tracts()));
        connect(glWidget,SIGNAL(region_edited()),glWidget,SLOT(updateGL()));
        connect(glWidget,SIGNAL(region_edited()),&scene,SLOT(show_slice()));

        connect(ui->actionOpenTract,SIGNAL(triggered()),tractWidget,SLOT(load_tracts()));
        connect(ui->actionMerge_All,SIGNAL(triggered()),tractWidget,SLOT(merge_all()));
        connect(ui->actionCopyTrack,SIGNAL(triggered()),tractWidget,SLOT(copy_track()));
        connect(ui->actionDeleteTract,SIGNAL(triggered()),tractWidget,SLOT(delete_tract()));
        connect(ui->actionDeleteTractAll,SIGNAL(triggered()),tractWidget,SLOT(delete_all_tract()));

        connect(ui->actionCheck_all_tracts,SIGNAL(triggered()),tractWidget,SLOT(check_all()));
        connect(ui->actionUncheck_all_tracts,SIGNAL(triggered()),tractWidget,SLOT(uncheck_all()));


        connect(ui->actionOpen_Colors,SIGNAL(triggered()),tractWidget,SLOT(load_tracts_color()));
        connect(ui->actionSave_Tracts_Colors_As,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_color_as()));

        connect(ui->actionUndo,SIGNAL(triggered()),tractWidget,SLOT(undo_tracts()));
        connect(ui->actionRedo,SIGNAL(triggered()),tractWidget,SLOT(redo_tracts()));
        connect(ui->actionTrim,SIGNAL(triggered()),tractWidget,SLOT(trim_tracts()));

        connect(ui->actionSet_Color,SIGNAL(triggered()),tractWidget,SLOT(set_color()));

        connect(ui->actionK_means,SIGNAL(triggered()),tractWidget,SLOT(clustering_kmeans()));
        connect(ui->actionEM,SIGNAL(triggered()),tractWidget,SLOT(clustering_EM()));
        connect(ui->actionHierarchical,SIGNAL(triggered()),tractWidget,SLOT(clustering_hie()));
        connect(ui->actionOpen_Cluster_Labels,SIGNAL(triggered()),tractWidget,SLOT(open_cluster_label()));

        //setup menu
        connect(ui->actionSaveTractAs,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_as()));
        connect(ui->actionSave_All_Tracts_As,SIGNAL(triggered()),tractWidget,SLOT(save_all_tracts_as()));
        connect(ui->actionQuantitative_anisotropy_QA,SIGNAL(triggered()),tractWidget,SLOT(save_fa_as()));
        connect(ui->actionSave_End_Points_As,SIGNAL(triggered()),tractWidget,SLOT(save_end_point_as()));
        connect(ui->actionStatistics,SIGNAL(triggered()),tractWidget,SLOT(show_tracts_statistics()));

        connect(ui->track_up,SIGNAL(clicked()),tractWidget,SLOT(move_up()));
        connect(ui->track_down,SIGNAL(clicked()),tractWidget,SLOT(move_down()));

    }




    // recall the setting
    {

        QSettings settings;
        if(!default_geo.size())
            default_geo = saveGeometry();
        if(!default_state.size())
            default_state = saveState();
        restoreGeometry(settings.value("geometry").toByteArray());
        restoreState(settings.value("state").toByteArray());
        ui->glSagCheck->setChecked(settings.value("SagSlice",1).toBool());
        ui->glCorCheck->setChecked(settings.value("CorSlice",1).toBool());
        ui->glAxiCheck->setChecked(settings.value("AxiSlice",1).toBool());
        ui->RenderingQualityBox->setCurrentIndex(settings.value("RenderingQuality",1).toInt());
    }

    {
        scene.center();
        slice_no_update = false;
        copy_target = 0;
    }

    // setup fa threshold
    {
        QStringList tracking_index_list;
        for(int index = 0;index < fib_data.fib.index_name.size();++index)
            tracking_index_list.push_back(fib_data.fib.index_name[index].c_str());

        renderWidget->setList("tracking_index",tracking_index_list);
        renderWidget->setData("tracking_index",0);
        renderWidget->setData("step_size",fib_data.vs[0]/2.0);
    }

    on_glAxiView_clicked();
    if(renderWidget->getData("orientation_convention").toInt() == 1)
        on_glAxiView_clicked();
    qApp->installEventFilter(this);
}

tracking_window::~tracking_window()
{
    qApp->removeEventFilter(this);
    QSettings settings;
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    tractWidget->delete_all_tract();
    delete ui;
    if(handle_release)
        delete handle;
    handle = 0;
    //std::cout << __FUNCTION__ << " " << __FILE__ << std::endl;
}

void tracking_window::subject2mni(image::vector<3>& pos)
{
    if(mi3.get())
    {
        mi3->T(pos);
        if(mi3->progress >= 1)
        {
            image::vector<3> mni;
            mi3->bnorm_data(pos,mni);
            pos = mni;
        }
        fa_template_imp.to_mni(pos);
    }
    else
    if(!handle->fib_data.trans_to_mni.empty())
    {
        image::vector<3> mni;
        image::vector_transformation(pos.begin(),mni.begin(), handle->fib_data.trans_to_mni,image::vdim<3>());
        pos = mni;
    }
}
bool tracking_window::eventFilter(QObject *obj, QEvent *event)
{
    bool has_info = false;
    image::vector<3,float> pos;
    if (event->type() == QEvent::MouseMove)
    {
        if (obj == glWidget)
        {
            has_info = glWidget->get_mouse_pos(static_cast<QMouseEvent*>(event),pos);
            copy_target = 0;
        }
        if (obj->parent() == ui->graphicsView)
        {
            QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
            QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
            has_info = scene.get_location(point.x(),point.y(),pos);
            copy_target = 1;
        }
        // for connectivity matrix
        if(connectivity_matrix.get() && connectivity_matrix->is_graphic_view(obj->parent()))
            connectivity_matrix->mouse_move(static_cast<QMouseEvent*>(event));
    }
    if(!has_info)
        return false;

    QString status;
    status = QString("(%1,%2,%3) ").arg(std::floor(pos[0]*10.0+0.5)/10.0)
            .arg(std::floor(pos[1]*10.0+0.5)/10.0)
            .arg(std::floor(pos[2]*10.0+0.5)/10.0);
    // show atlas position
    if(mi3.get() && mi3->need_update_affine_matrix)
    {
        mi3->update_affine();
        handle->fib_data.trans_to_mni.resize(16);
        image::create_affine_transformation_matrix(mi3->T.get(),mi3->T.get() + 9,handle->fib_data.trans_to_mni.begin(),image::vdim<3>());
        fa_template_imp.add_transformation(handle->fib_data.trans_to_mni);
        if(mi3->progress >= 1)
            mi3->need_update_affine_matrix = false;
    }

    if(!handle->fib_data.trans_to_mni.empty())
    {
        image::vector<3,float> mni(pos);
        subject2mni(mni);
        status += QString("MNI(%1,%2,%3) ")
                .arg(std::floor(mni[0]*10.0+0.5)/10.0)
                .arg(std::floor(mni[1]*10.0+0.5)/10.0)
                .arg(std::floor(mni[2]*10.0+0.5)/10.0);
        unsigned int atlas_index = atlas_dialog ? atlas_dialog->index() : 0;
        if(!atlas_list.empty())
        {
            status += atlas_list[atlas_index].name.c_str();
            status += ":";
            status += atlas_list[atlas_index].get_label_name_at(mni).c_str();
        }
    }
    status += " ";
    std::vector<float> data;
    handle->get_voxel_information(std::floor(pos[0] + 0.5), std::floor(pos[1] + 0.5), std::floor(pos[2] + 0.5), data);
    for(unsigned int index = 0,data_index = 0;index < handle->fib_data.view_item.size() && data_index < data.size();++index)
        if(handle->fib_data.view_item[index].name != "color")
        {
            status += handle->fib_data.view_item[index].name.c_str();
            status += QString("=%1 ").arg(data[data_index]);
            ++data_index;
        }
    ui->statusbar->showMessage(status);


    return false;
}

void tracking_window::set_tracking_param(ThreadData& tracking_thread)
{
    tracking_thread.param.step_size = renderWidget->getData("step_size").toDouble();
    tracking_thread.param.smooth_fraction = renderWidget->getData("smoothing").toDouble();
    tracking_thread.param.min_points_count3 = 3.0*renderWidget->getData("min_length").toDouble()/renderWidget->getData("step_size").toDouble();
    if(tracking_thread.param.min_points_count3 < 6)
        tracking_thread.param.min_points_count3 = 6;
    tracking_thread.param.max_points_count3 =
            std::max<unsigned int>(tracking_thread.param.min_points_count3,
                                   3.0*renderWidget->getData("max_length").toDouble()/renderWidget->getData("step_size").toDouble());

    tracking_thread.tracking_method = renderWidget->getData("tracking_method").toInt();
    tracking_thread.initial_direction = renderWidget->getData("initial_direction").toInt();
    tracking_thread.interpolation_strategy = renderWidget->getData("interpolation").toInt();
    tracking_thread.stop_by_tract = renderWidget->getData("tracking_plan").toInt();
    tracking_thread.center_seed = renderWidget->getData("seed_plan").toInt();
}

void tracking_window::SliderValueChanged(void)
{
    if(!slice_no_update && slice.set_slice_pos(
            ui->SagSlider->value(),
            ui->CorSlider->value(),
            ui->AxiSlider->value()))
    {
        if(renderWidget->getData("roi_layout").toInt() <= 1)
            scene.show_slice();
        if(glWidget->current_visible_slide == 0)
            glWidget->updateGL();
    }



}
void tracking_window::glSliderValueChanged(void)
{
    if(!glWidget->current_visible_slide)
        return;
    SliceModel& cur_slice =
                glWidget->other_slices[glWidget->current_visible_slide-1];
    if(!slice_no_update && cur_slice.set_slice_pos(
                ui->glSagSlider->value(),
                ui->glCorSlider->value(),
                ui->glAxiSlider->value()))
            glWidget->updateGL();

}


void tracking_window::on_AxiView_clicked()
{
    slice.cur_dim = 2;
    if(renderWidget->getData("roi_layout").toInt() == 0)
        scene.show_slice();
    scene.setFocus();
}

void tracking_window::on_CorView_clicked()
{
    slice.cur_dim = 1;
    if(renderWidget->getData("roi_layout").toInt() == 0)
        scene.show_slice();
    scene.setFocus();
}

void tracking_window::on_SagView_clicked()
{
    slice.cur_dim = 0;
    if(renderWidget->getData("roi_layout").toInt() == 0)
        scene.show_slice();
    scene.setFocus();
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

void tracking_window::on_sliceViewBox_currentIndexChanged(int index)
{
    ui->actionSave_Anisotrpy_Map_as->setText(QString("Save ") +
                                             ui->sliceViewBox->currentText()+" as...");
    slice.set_view_name(ui->sliceViewBox->currentText().toLocal8Bit().begin(),
                        ui->overlay->currentText().toLocal8Bit().begin());
    float range = handle->get_value_range(ui->sliceViewBox->currentText().toLocal8Bit().begin());
    if(range != 0.0)
    {
        ui->offset_value->setMaximum(range);
        ui->offset_value->setMinimum(-range);
        ui->offset_value->setSingleStep(range/50.0);
        ui->contrast_value->setMaximum(range*11.0);
        ui->contrast_value->setMinimum(range/11.0);
        ui->contrast_value->setSingleStep(range/50.0);
        ui->contrast_value->setValue(range);
        ui->offset_value->setValue(0.0);

        if(glWidget->current_visible_slide == 0) // Show diffusion
        {
            ui->gl_offset_value->setMaximum(range);
            ui->gl_offset_value->setMinimum(-range);
            ui->gl_offset_value->setSingleStep(range/50.0);
            ui->gl_contrast_value->setMaximum(range*11.0);
            ui->gl_contrast_value->setMinimum(range/11.0);
            ui->gl_contrast_value->setSingleStep(range/50.0);
            ui->gl_contrast_value->setValue(range);
            ui->gl_offset_value->setValue(0.0);
        }
    }
}

void tracking_window::on_actionSelect_Tracts_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = 1;
    tractWidget->edit_option = 1;

}

void tracking_window::on_actionDelete_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = 1;
    tractWidget->edit_option = 2;
}

void tracking_window::on_actionCut_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = 1;
    tractWidget->edit_option = 3;
}


void tracking_window::on_actionPaint_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = 1;
    tractWidget->edit_option = 4;
}

void tracking_window::on_actionMove_Object_triggered()
{
    glWidget->setCursor(Qt::CrossCursor);
    glWidget->editing_option = 2;
}


void tracking_window::on_glSagView_clicked()
{
    glWidget->set_view(0);
    glWidget->updateGL();
    glWidget->setFocus();
}

void tracking_window::on_glCorView_clicked()
{
    glWidget->set_view(1);
    glWidget->updateGL();
    glWidget->setFocus();
}

void tracking_window::on_glAxiView_clicked()
{
    glWidget->set_view(2);
    glWidget->updateGL();
    glWidget->setFocus();
}

void tracking_window::on_SliceModality_currentIndexChanged(int index)
{
    glWidget->current_visible_slide = index;
    slice_no_update = true;

    {
        float range;
        if(index)
            range = glWidget->other_slices[glWidget->current_visible_slide-1].get_value_range();
        else
            range =  handle->get_value_range(ui->sliceViewBox->currentText().toLocal8Bit().begin());
        ui->gl_offset_value->setMaximum(range);
        ui->gl_offset_value->setMinimum(-range);
        ui->gl_offset_value->setSingleStep(range/50.0);
        ui->gl_contrast_value->setMaximum(range*11.0);
        ui->gl_contrast_value->setMinimum(range/11.0);
        ui->gl_contrast_value->setSingleStep(range/50.0);
        ui->gl_contrast_value->setValue(range);
        ui->gl_offset_value->setValue(0.0);
    }

    if(index)
    {
        disconnect(ui->sliceViewBox,SIGNAL(currentIndexChanged(int)),glWidget,SLOT(updateGL()));
        disconnect(ui->glSagSlider,SIGNAL(valueChanged(int)),ui->SagSlider,SLOT(setValue(int)));
        disconnect(ui->glCorSlider,SIGNAL(valueChanged(int)),ui->CorSlider,SLOT(setValue(int)));
        disconnect(ui->glAxiSlider,SIGNAL(valueChanged(int)),ui->AxiSlider,SLOT(setValue(int)));
        disconnect(ui->SagSlider,SIGNAL(valueChanged(int)),ui->glSagSlider,SLOT(setValue(int)));
        disconnect(ui->CorSlider,SIGNAL(valueChanged(int)),ui->glCorSlider,SLOT(setValue(int)));
        disconnect(ui->AxiSlider,SIGNAL(valueChanged(int)),ui->glAxiSlider,SLOT(setValue(int)));

        SliceModel& cur_slice =
                glWidget->other_slices[glWidget->current_visible_slide-1];

        ui->glSagSlider->setRange(0,cur_slice.geometry[0]-1);
        ui->glCorSlider->setRange(0,cur_slice.geometry[1]-1);
        ui->glAxiSlider->setRange(0,cur_slice.geometry[2]-1);
        ui->glSagBox->setRange(0,cur_slice.geometry[0]-1);
        ui->glCorBox->setRange(0,cur_slice.geometry[1]-1);
        ui->glAxiBox->setRange(0,cur_slice.geometry[2]-1);
        ui->glSagSlider->setValue(cur_slice.slice_pos[0]);
        ui->glCorSlider->setValue(cur_slice.slice_pos[1]);
        ui->glAxiSlider->setValue(cur_slice.slice_pos[2]);
        ui->glSagBox->setValue(ui->glSagSlider->value());
        ui->glCorBox->setValue(ui->glCorSlider->value());
        ui->glAxiBox->setValue(ui->glAxiSlider->value());
    }
    else
    //diffusion
    {
        ui->glSagSlider->setRange(0,slice.geometry[0]-1);
        ui->glCorSlider->setRange(0,slice.geometry[1]-1);
        ui->glAxiSlider->setRange(0,slice.geometry[2]-1);
        ui->glSagBox->setRange(0,slice.geometry[0]-1);
        ui->glCorBox->setRange(0,slice.geometry[1]-1);
        ui->glAxiBox->setRange(0,slice.geometry[2]-1);

        ui->glSagSlider->setValue(ui->SagSlider->value());
        ui->glCorSlider->setValue(ui->CorSlider->value());
        ui->glAxiSlider->setValue(ui->AxiSlider->value());
        ui->glSagBox->setValue(ui->glSagSlider->value());
        ui->glCorBox->setValue(ui->glCorSlider->value());
        ui->glAxiBox->setValue(ui->glAxiSlider->value());

        connect(ui->sliceViewBox,SIGNAL(currentIndexChanged(int)),glWidget,SLOT(updateGL()));
        connect(ui->glSagSlider,SIGNAL(valueChanged(int)),ui->SagSlider,SLOT(setValue(int)));
        connect(ui->glCorSlider,SIGNAL(valueChanged(int)),ui->CorSlider,SLOT(setValue(int)));
        connect(ui->glAxiSlider,SIGNAL(valueChanged(int)),ui->AxiSlider,SLOT(setValue(int)));
        connect(ui->SagSlider,SIGNAL(valueChanged(int)),ui->glSagSlider,SLOT(setValue(int)));
        connect(ui->CorSlider,SIGNAL(valueChanged(int)),ui->glCorSlider,SLOT(setValue(int)));
        connect(ui->AxiSlider,SIGNAL(valueChanged(int)),ui->glAxiSlider,SLOT(setValue(int)));

        std::fill(slice.texture_need_update,
                  slice.texture_need_update+3,1);
    }
    slice_no_update = false;

}

void tracking_window::on_actionEndpoints_to_seeding_triggered()
{
    std::vector<image::vector<3,short> >points;

    if(tractWidget->tract_models.empty())
        return;
    tractWidget->tract_models[tractWidget->currentRow()]->get_end_points(points);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" end points"),roi_id);
    regionWidget->add_points(points,false);
    scene.show_slice();
    glWidget->updateGL();
}

void tracking_window::on_actionTracts_to_seeds_triggered()
{
    std::vector<image::vector<3,short> >points;
    if(tractWidget->tract_models.empty())
        return;
    tractWidget->tract_models[tractWidget->currentRow()]->get_tract_points(points);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text(),roi_id);
    regionWidget->add_points(points,false);
    scene.show_slice();
    glWidget->updateGL();
}

void tracking_window::on_actionInsert_T1_T2_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
        this,
        "Open Images files",get_path("t1_path"),
        "Image files (*.dcm *.hdr *.nii *.nii.gz 2dseq);;All files (*)" );
    if( filenames.isEmpty() || !glWidget->addSlices(filenames))
        return;
    add_path("t1_path",filenames[0]);
    ui->SliceModality->addItem(QFileInfo(filenames[0]).baseName());
    ui->SliceModality->setCurrentIndex(glWidget->other_slices.size());
    ui->sliceViewBox->addItem(QFileInfo(filenames[0]).baseName().toLocal8Bit().begin());
    handle->fib_data.view_item.push_back(handle->fib_data.view_item[0]);
    handle->fib_data.view_item.back().name = QFileInfo(filenames[0]).baseName().toLocal8Bit().begin();
    handle->fib_data.view_item.back().is_overlay = false;
    handle->fib_data.view_item.back().image_data = image::make_image(glWidget->roi_image.back().geometry(),
                                                                     glWidget->roi_image_buf.back());
    handle->fib_data.view_item.back().set_scale(
                glWidget->other_slices.back().source_images.begin(),
                glWidget->other_slices.back().source_images.end());
    ui->sliceViewBox->setCurrentIndex(ui->sliceViewBox->count()-1);

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
    std::vector<float> tr(16);
    tr[0] = tr[5] = tr[10] = tr[15] = 1.0;
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(slice.geometry,slice.voxel_size,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionTDI_Subvoxel_Diffusion_Space_triggered()
{
    std::vector<float> tr(16);
    tr[0] = tr[5] = tr[10] = tr[15] = 4.0;
    image::geometry<3> new_geo(slice.geometry[0]*4,slice.geometry[1]*4,slice.geometry[2]*4);
    image::vector<3,float> new_vs(slice.voxel_size);
    new_vs /= 4.0;
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(new_geo,new_vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}

void tracking_window::on_actionTDI_Import_Slice_Space_triggered()
{
    std::vector<float> tr(16);
    image::geometry<3> geo;
    image::vector<3,float> vs;
    glWidget->get_current_slice_transformation(geo,vs,tr);
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(geo,vs,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionSave_Tracts_in_Current_Mapping_triggered()
{
    std::vector<float> tr(16);
    image::geometry<3> geo;
    image::vector<3,float> vs;
    glWidget->get_current_slice_transformation(geo,vs,tr);
    tractWidget->saveTransformedTracts(&*tr.begin());
}
void tracking_window::on_actionSave_Endpoints_in_Current_Mapping_triggered()
{

    std::vector<float> tr(16);
    image::geometry<3> geo;
    image::vector<3,float> vs;
    glWidget->get_current_slice_transformation(geo,vs,tr);
    tractWidget->saveTransformedEndpoints(&*tr.begin());
}

void tracking_window::on_RenderingQualityBox_currentIndexChanged(int index)
{
    if(slice_no_update)
        return;
    QSettings settings;
    settings.setValue("RenderingQuality",index);
    switch(index)
    {
    case 0:
        renderWidget->setData("anti_aliasing",0);
        renderWidget->setData("tract_tube_detail",0);
        renderWidget->setData("tract_visible_tract",5000);
        break;
    case 1:
        renderWidget->setData("anti_aliasing",0);
        renderWidget->setData("tract_tube_detail",1);
        renderWidget->setData("tract_visible_tract",25000);
        break;
    case 2:
        renderWidget->setData("anti_aliasing",1);
        renderWidget->setData("tract_tube_detail",3);
        renderWidget->setData("tract_visible_tract",100000);
        break;
    }
    glWidget->updateGL();
}

void tracking_window::on_actionCopy_to_clipboard_triggered()
{
    switch(copy_target)
    {
    case 0:
        glWidget->copyToClipboard();
        return;
    case 1:
        scene.copyClipBoard();
        return;
    case 2:
        if(tact_report_imp.get())
            tact_report_imp->copyToClipboard();
        return;
    }
}


void tracking_window::on_actionRestore_window_layout_triggered()
{
    restoreGeometry(default_geo);
    restoreState(default_state);
}



void tracking_window::on_tracking_index_currentIndexChanged(int index)
{
    handle->fib_data.fib.set_tracking_index(index);
    if(renderWidget->getData("tracking_index").toString().contains("greater") ||
       renderWidget->getData("tracking_index").toString().contains("lesser")) // connectometry
    {
        renderWidget->setMinMax("fa_threshold",0.5,1.0,0.01);
        renderWidget->setData("fa_threshold",0.75);
    }
    else
    {
        float max_value = *std::max_element(handle->fib_data.fib.fa[0],handle->fib_data.fib.fa[0]+handle->fib_data.fib.dim.size());
        renderWidget->setMinMax("fa_threshold",0.0,max_value*1.1,max_value/50.0);
        renderWidget->setData("fa_threshold",0.6*image::segmentation::otsu_threshold(image::make_image(handle->fib_data.fib.dim,
                                                                                                       handle->fib_data.fib.fa[0])));
    }
}


void tracking_window::on_deleteSlice_clicked()
{
    if(ui->SliceModality->currentIndex() == 0)
        return;
    int index = ui->SliceModality->currentIndex();
    unsigned int view_item_index = handle->fib_data.view_item.size()-glWidget->mi3s.size()+index-1;
    if(ui->sliceViewBox->currentIndex() == view_item_index)
        ui->sliceViewBox->setCurrentIndex(0);
    ui->sliceViewBox->removeItem(view_item_index);
    handle->fib_data.view_item.erase(handle->fib_data.view_item.begin()+view_item_index);
    ui->SliceModality->setCurrentIndex(0);
    glWidget->delete_slice(index-1);
    ui->SliceModality->removeItem(index);


}


void tracking_window::on_actionSave_Tracts_in_MNI_space_triggered()
{
    if(handle->fib_data.trans_to_mni.empty())
        return;
    tractWidget->saveTransformedTracts(&*(handle->fib_data.trans_to_mni.begin()));
}


void tracking_window::on_offset_sliderMoved(int position)
{
    ui->offset_value->setValue((float)position*ui->offset_value->maximum()/100.0);
}
void tracking_window::on_gl_offset_sliderMoved(int position)
{
    ui->gl_offset_value->setValue((float)position*ui->gl_offset_value->maximum()/100.0);
}

void tracking_window::on_contrast_sliderMoved(int position)
{
    ui->contrast_value->setValue((position >= 0) ? ui->offset_value->maximum()/(1+(float)position/10.0) :
                                                   ui->offset_value->maximum()*(1-(float)position/10.0));
}
void tracking_window::on_gl_contrast_sliderMoved(int position)
{
    ui->gl_contrast_value->setValue((position >= 0) ? ui->gl_offset_value->maximum()/(1+(float)position/10.0) :
                                                      ui->gl_offset_value->maximum()*(1-(float)position/10.0));
}

void tracking_window::on_offset_value_valueChanged(double arg1)
{
    ui->offset->setValue(arg1*100.0/ui->offset_value->maximum());
    scene.show_slice();
}

void tracking_window::on_gl_offset_value_valueChanged(double arg1)
{
    ui->gl_offset->setValue(arg1*100.0/ui->gl_offset_value->maximum());
    glWidget->updateGL();
}

void tracking_window::on_contrast_value_valueChanged(double arg1)
{
    ui->contrast->setValue((arg1 >= ui->offset_value->maximum()) ?
                           10-arg1*10.0/ui->offset_value->maximum():
                           ui->offset_value->maximum()*10.0/arg1-10);
    scene.show_slice();
}

void tracking_window::on_gl_contrast_value_valueChanged(double arg1)
{
    ui->gl_contrast->setValue((arg1 >= ui->gl_offset_value->maximum()) ?
                            10-arg1*10.0/ui->gl_offset_value->maximum():
                            ui->gl_offset_value->maximum()*10.0/arg1-10);
    glWidget->updateGL();
}




void tracking_window::keyPressEvent ( QKeyEvent * event )
{
    if(copy_target == 0) // glWidget
    {
        switch(event->key())
        {
        case Qt::Key_Q:
            ui->glSagSlider->setValue(ui->glSagSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_A:
            ui->glSagSlider->setValue(ui->glSagSlider->value()-1);
            event->accept();
            break;
        case Qt::Key_W:
            ui->glCorSlider->setValue(ui->glCorSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_S:
            ui->glCorSlider->setValue(ui->glCorSlider->value()-1);
            event->accept();
            break;
        case Qt::Key_E:
            ui->glAxiSlider->setValue(ui->glAxiSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_D:
            ui->glAxiSlider->setValue(ui->glAxiSlider->value()-1);
            event->accept();
            break;
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

            /*
        case Qt::Key_Z:
            event->accept();
            glWidget->set_view(0);
            glWidget->updateGL();
            break;
        case Qt::Key_X:
            event->accept();
            glWidget->set_view(1);
            glWidget->updateGL();
            break;
        case Qt::Key_C:
            event->accept();
            glWidget->set_view(2);
            glWidget->updateGL();
            break;*/
        }
    }
    else
    {
        switch(event->key())
        {
        case Qt::Key_Q:
            ui->SagSlider->setValue(ui->SagSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_A:
            ui->SagSlider->setValue(ui->SagSlider->value()-1);
            event->accept();
            break;
        case Qt::Key_W:
            ui->CorSlider->setValue(ui->CorSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_S:
            ui->CorSlider->setValue(ui->CorSlider->value()-1);
            event->accept();
            break;
        case Qt::Key_E:
            ui->AxiSlider->setValue(ui->AxiSlider->value()+1);
            event->accept();
            break;
        case Qt::Key_D:
            ui->AxiSlider->setValue(ui->AxiSlider->value()-1);
            event->accept();
            break;
            /*
        case Qt::Key_Z:
            on_SagView_clicked();
            event->accept();
            break;
        case Qt::Key_X:
            on_CorView_clicked();
            event->accept();
            break;
        case Qt::Key_C:
            on_AxiView_clicked();
            event->accept();
            break;*/
        }

    }


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
            std::copy(glWidget->transformation_matrix,glWidget->transformation_matrix+16,std::ostream_iterator<float>(out," "));
            settings.setValue(key_str,QString(out.str().c_str()));
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
            std::copy(tran.begin(),tran.begin()+16,glWidget->transformation_matrix);
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
    if(mi3.get())
    {
        mi3->timer->start();
        mi3->show();
    }
}


void tracking_window::on_actionTract_Analysis_Report_triggered()
{
    if(!tact_report_imp.get())
        tact_report_imp.reset(new tract_report(this));
    tact_report_imp->show();
    tact_report_imp->on_refresh_report_clicked();
}

void tracking_window::on_actionConnectivity_matrix_triggered()
{
    if(tractWidget->tract_models.size() == 0)
    {
        QMessageBox::information(this,"DSI Studio","Run fiber tracking first",0);
        return;
    }
    connectivity_matrix.reset(new connectivity_matrix_dialog(this));
    connectivity_matrix->show();
}


void tracking_window::on_zoom_3d_valueChanged(double arg1)
{
    if(std::fabs(ui->zoom_3d->value() - glWidget->current_scale) > 0.02)
    {
        glWidget->scale_by(ui->zoom_3d->value()/glWidget->current_scale);
        glWidget->current_scale = ui->zoom_3d->value();
    }
}
QString tracking_window::get_path(const std::string& id)
{
    std::map<std::string,QString>::const_iterator iter = path_map.find(id);
    if(iter == path_map.end())
        return absolute_path;
    return iter->second;
}
void tracking_window::add_path(const std::string& id,QString filename)
{
    path_map[id] = QFileInfo(filename).absolutePath();
}

void tracking_window::on_actionConnectometry_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Database files",
                           absolute_path,
                           "Database files (*.db.fib.gz);;All files (*)");
    if (filename.isEmpty())
        return;

    std::auto_ptr<vbc_database> database(new vbc_database);
    database.reset(new vbc_database);
    if(!database->load_database(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid database format",0);
        return;
    }
    if(!database->single_subject_analysis(this->windowTitle().toLocal8Bit().begin(),0.95,database->handle->connectometry))
    {
        QMessageBox::information(this,"error",database->error_msg.c_str(),0);
        return;
    }
    std::vector<std::vector<std::vector<float> > > greater,lesser;
    database->calculate_individual_affected_tracks(database->handle->connectometry,0.95,greater,lesser);
    tractWidget->addConnectometryResults(greater,lesser);
    /*
    tracking_window* new_mdi = new tracking_window((QWidget*)(this->parent()),database->handle.release());
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->absolute_path = absolute_path;
    new_mdi->setWindowTitle(this->windowTitle() + " : connectometry mapping");
    new_mdi->showNormal();*/
}


void tracking_window::restore_3D_window()
{
    ui->centralLayout->addWidget(ui->main_widget);
    delete gLdock;
    gLdock = 0;
}

void tracking_window::on_actionFloat_3D_window_triggered()
{
    if(!gLdock)
    {
        gLdock = new QGLDockWidget(this);
        int w = ui->main_widget->width();
        gLdock->setWindowTitle("3D Window");
        gLdock->setAllowedAreas(Qt::NoDockWidgetArea);
        gLdock->setWidget(ui->main_widget);
        gLdock->setFloating(true);
        gLdock->show();
        gLdock->resize(w,w);
        connect(gLdock,SIGNAL(closedSignal()),this,SLOT(restore_3D_window()));
    }
    else
        restore_3D_window();
}

void tracking_window::on_actionSave_tracking_parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",
                           absolute_path,
                           "Setting file (*.ini);;All files (*)");
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
                           this,
                           "Open INI files",
                           absolute_path,
                           "Setting file (*.ini);;All files (*)");
    if (filename.isEmpty())
        return;
    QSettings s(filename, QSettings::IniFormat);
    QStringList param_list = renderWidget->getChildren("Tracking");
    for(unsigned int index = 0;index < param_list.size();++index)
        if(s.contains(param_list[index]))
            renderWidget->setData(param_list[index],s.value(param_list[index]));
}


void tracking_window::on_tbDefaultParam_clicked()
{
    renderWidget->setDefault();
    renderWidget->setData("tracking_index",0);
    on_tracking_index_currentIndexChanged(0);
    scene.show_slice();
    glWidget->updateGL();
}

void tracking_window::on_actionSave_Rendering_Parameters_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save INI files",
                           absolute_path,
                           "Setting file (*.ini);;All files (*)");
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
                           this,
                           "Open INI files",
                           absolute_path,
                           "Setting file (*.ini);;All files (*)");
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
            renderWidget->setData(param_list[index],s.value(param_list[index]));
    glWidget->updateGL();
}

void tracking_window::on_addRegionFromAtlas_clicked()
{
    if(!atlas_dialog)
    {
        atlas_dialog = new AtlasDialog(this);
        connect(atlas_dialog,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(atlas_dialog,SIGNAL(need_update()),glWidget,SLOT(updateGL()));
    }
    atlas_dialog->show();
}
