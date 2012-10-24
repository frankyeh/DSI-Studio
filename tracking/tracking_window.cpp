#include <QFileDialog>
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
#include <QGraphicsTextItem>
#include "tracking_model.hpp"
#include "libs/tracking/tracking_model.hpp"


#include "mapping/atlas.hpp"
#include "mapping/fa_template.hpp"

extern std::vector<atlas> atlas_list;
extern fa_template fa_template_imp;
QByteArray default_geo,default_state;

void tracking_window::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
}

tracking_window::tracking_window(QWidget *parent,ODFModel* new_handle) :
        QMainWindow(parent),handle(new_handle),
        ui(new Ui::tracking_window),scene(*this,new_handle),slice(new_handle)
{


    ODFModel* odf_model = (ODFModel*)handle;
    FibData& fib_data = odf_model->fib_data;

    odf_size = fib_data.fib.odf_table.size();
    odf_face_size = fib_data.fib.odf_faces.size();
    has_odfs = fib_data.fib.has_odfs() ? 1:0;
    if(fib_data.trans)
    {
        trans_to_mni.resize(16);
        trans_to_mni[15] = 1.0;
        std::copy(fib_data.trans,fib_data.trans+12,trans_to_mni.begin());
        // this is 1-based transformation, need to change to 0-based and flip xy

        // spm_affine = [1 0 0 -1                   [1 0 0 1
        //               0 1 0 -1                    0 1 0 1
        //               0 0 1 -1   * my_affine *    0 0 1 1
        //               0 0 0 1]                    0 0 0 1]
        trans_to_mni[3] = std::accumulate(trans_to_mni.begin(),trans_to_mni.begin()+3,trans_to_mni[3])-1.0;
        trans_to_mni[7] = std::accumulate(trans_to_mni.begin()+4,trans_to_mni.begin()+4+3,trans_to_mni[7])-1.0;
        trans_to_mni[11] = std::accumulate(trans_to_mni.begin()+8,trans_to_mni.begin()+8+3,trans_to_mni[11])-1.0;



    }
    for (unsigned int index = 0;index < fib_data.view_item.size();++index)
        view_name.push_back(fib_data.view_item[index].name);
    is_dti = (view_name[0][0] == 'f');

    ui->setupUi(this);
    {
        setGeometry(10,10,800,600);
        ui->regionDockWidget->setMinimumWidth(0);
        ui->dockWidget->setMinimumWidth(0);
        ui->dockWidget_3->setMinimumWidth(0);
        ui->renderingLayout->addWidget(renderWidget = new RenderingTableWidget(*this,ui->renderingWidgetHolder,has_odfs));
        ui->centralLayout->insertWidget(1,glWidget = new GLWidget(renderWidget->getData("anti_aliasing").toInt(),
                                                                  *this,renderWidget,ui->centralwidget));
        ui->verticalLayout_3->addWidget(regionWidget = new RegionTableWidget(*this,ui->regionDockWidget));
        ui->tractverticalLayout->addWidget(tractWidget = new TractTableWidget(*this,ui->TractWidgetHolder));
        ui->color_bar->hide();
        ui->dockWidget_report->hide();
        ui->graphicsView->setScene(&scene);
        ui->color_bar_view->setScene(&color_bar);
        ui->graphicsView->setCursor(Qt::CrossCursor);
        scene.statusbar = ui->statusbar;
    }


    // setup fa threshold
    {
        for(int index = 0;index < fib_data.fib.index_name.size();++index)
            ui->tracking_index->addItem((fib_data.fib.index_name[index]+" threshold").c_str());
        ui->tracking_index->setCurrentIndex(0);
        ui->step_size->setValue(fib_data.vs[0]/2.0);
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

        for (unsigned int index = 0; index < view_name.size(); ++index)
            ui->sliceViewBox->addItem(view_name[index].c_str());
        ui->sliceViewBox->setCurrentIndex(0);

        for (unsigned int index = 0;index < fib_data.view_item.size();++index)
            if(fib_data.view_item[index].is_overlay)
                ui->overlay->addItem(view_name[index].c_str());
        ui->overlay->setCurrentIndex(0);
        if(ui->overlay->count() == 1)
           ui->overlay->hide();
    }

    // setup atlas
    if(!fa_template_imp.I.empty() && fib_data.vs[0] > 0.5)
    {
        mi3.reset(new lm3_type);
        mi3->from = slice.source_images;
        mi3->to = fa_template_imp.I;
        mi3->arg_min.scaling[0] = slice.voxel_size[0] / std::fabs(fa_template_imp.tran[0]);
        mi3->arg_min.scaling[1] = slice.voxel_size[1] / std::fabs(fa_template_imp.tran[5]);
        mi3->arg_min.scaling[2] = slice.voxel_size[2] / std::fabs(fa_template_imp.tran[10]);
        mi3->cost_function_id = 2;/* MutualInformation */;
        mi3->thread_argmin(image::reg::affine);
            for(int index = 0;index < atlas_list.size();++index)
                ui->atlasListBox->addItem(atlas_list[index].name.c_str());
    }


    {
        if(is_dti)
            ui->actionQuantitative_anisotropy_QA->setText("Save FA...");
        ui->report_index->addItem((is_dti) ? "fa":"qa");
        ui->tract_color_index->addItem((is_dti) ? "fa":"qa");
        for (int index = fib_data.other_mapping_index; index < view_name.size(); ++index)
            {
                QAction* Item = new QAction(this);
                Item->setText(QString("Save %1...").arg(view_name[index].c_str()));
                Item->setData(QString(view_name[index].c_str()));
                Item->setVisible(true);
                connect(Item, SIGNAL(triggered()),tractWidget, SLOT(save_tracts_data_as()));
                ui->menuSave->addAction(Item);
                ui->report_index->addItem(view_name[index].c_str());
                ui->tract_color_index->addItem(view_name[index].c_str());
            }
    }

    // opengl
    {
        connect(renderWidget->treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),
                glWidget,SLOT(updateGL()));
        connect(ui->tbDefaultParam,SIGNAL(clicked()),renderWidget,SLOT(setDefault()));
        connect(ui->tbDefaultParam,SIGNAL(clicked()),glWidget,SLOT(updateGL()));

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
        connect(ui->actionLoad_Camera,SIGNAL(triggered()),glWidget,SLOT(loadCamera()));
        connect(ui->actionSave_Camera,SIGNAL(triggered()),glWidget,SLOT(saveCamera()));
        connect(ui->actionLoad_mapping,SIGNAL(triggered()),glWidget,SLOT(loadMapping()));
        connect(ui->actionSave_mapping,SIGNAL(triggered()),glWidget,SLOT(saveMapping()));
        connect(ui->actionSave_Rotation_Images,SIGNAL(triggered()),glWidget,SLOT(saveRotationSeries()));

    }
    // scene view
    {
        connect(ui->SagSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->CorSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));
        connect(ui->AxiSlider,SIGNAL(valueChanged(int)),this,SLOT(SliderValueChanged()));


        connect(&scene,SIGNAL(need_update()),&scene,SLOT(show_slice()));
        connect(&scene,SIGNAL(need_update()),glWidget,SLOT(updateGL()));
        connect(ui->fa_threshold,SIGNAL(valueChanged(double)),&scene,SLOT(show_slice()));
        connect(ui->contrast,SIGNAL(valueChanged(int)),&scene,SLOT(show_slice()));
        connect(ui->offset,SIGNAL(valueChanged(int)),&scene,SLOT(show_slice()));
        connect(ui->show_fiber,SIGNAL(clicked()),&scene,SLOT(show_slice()));
        connect(ui->show_pos,SIGNAL(clicked()),&scene,SLOT(show_slice()));

        connect(ui->zoomIn,SIGNAL(clicked()),&scene,SLOT(zoom_in()));
        connect(ui->zoomOut,SIGNAL(clicked()),&scene,SLOT(zoom_out()));

        connect(ui->actionZoom_In,SIGNAL(triggered()),&scene,SLOT(zoom_in()));
        connect(ui->actionZoom_Out,SIGNAL(triggered()),&scene,SLOT(zoom_out()));

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

        connect(ui->view_style,SIGNAL(currentIndexChanged(int)),&scene,SLOT(show_slice()));

        //atlas
        connect(ui->addRegionFromAtlas,SIGNAL(clicked()),regionWidget,SLOT(add_atlas()));


        connect(ui->actionNewRegion,SIGNAL(triggered()),regionWidget,SLOT(new_region()));
        connect(ui->actionOpenRegion,SIGNAL(triggered()),regionWidget,SLOT(load_region()));
        connect(ui->actionSaveRegionAs,SIGNAL(triggered()),regionWidget,SLOT(save_region()));
        connect(ui->actionSave_Voxel_Data_As,SIGNAL(triggered()),regionWidget,SLOT(save_region_info()));
        connect(ui->actionDeleteRegion,SIGNAL(triggered()),regionWidget,SLOT(delete_region()));
        connect(ui->actionDeleteRegionAll,SIGNAL(triggered()),regionWidget,SLOT(delete_all_region()));


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

        connect(ui->actionWhole_brain_seeding,SIGNAL(triggered()),regionWidget,SLOT(whole_brain()));


    }
    // tracts
    {
        connect(ui->perform_tracking,SIGNAL(clicked()),tractWidget,SLOT(start_tracking()));
        connect(ui->stopTracking,SIGNAL(clicked()),tractWidget,SLOT(stop_tracking()));

        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(makeTracts()));
        connect(tractWidget,SIGNAL(need_update()),glWidget,SLOT(updateGL()));

        connect(glWidget,SIGNAL(edited()),tractWidget,SLOT(edit_tracts()));
        connect(glWidget,SIGNAL(region_edited()),glWidget,SLOT(updateGL()));
        connect(glWidget,SIGNAL(region_edited()),&scene,SLOT(show_slice()));

        connect(ui->actionOpenTract,SIGNAL(triggered()),tractWidget,SLOT(load_tracts()));
        connect(ui->actionMerge_All,SIGNAL(triggered()),tractWidget,SLOT(merge_all()));
        connect(ui->actionDeleteTract,SIGNAL(triggered()),tractWidget,SLOT(delete_tract()));
        connect(ui->actionDeleteTractAll,SIGNAL(triggered()),tractWidget,SLOT(delete_all_tract()));

        connect(ui->actionOpen_Colors,SIGNAL(triggered()),tractWidget,SLOT(load_tracts_color()));
        connect(ui->actionSave_Tracts_Colors_As,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_color_as()));

        connect(ui->actionUndo,SIGNAL(triggered()),tractWidget,SLOT(undo_tracts()));
        connect(ui->actionRedo,SIGNAL(triggered()),tractWidget,SLOT(redo_tracts()));
        connect(ui->actionTrim,SIGNAL(triggered()),tractWidget,SLOT(trim_tracts()));

        connect(ui->actionSet_Color,SIGNAL(triggered()),tractWidget,SLOT(set_color()));

        connect(ui->actionK_means,SIGNAL(triggered()),tractWidget,SLOT(clustering_kmeans()));
        connect(ui->actionEM,SIGNAL(triggered()),tractWidget,SLOT(clustering_EM()));
        connect(ui->actionHierarchical,SIGNAL(triggered()),tractWidget,SLOT(clustering_hie()));

        //setup menu
        connect(ui->actionSaveTractAs,SIGNAL(triggered()),tractWidget,SLOT(save_tracts_as()));
        connect(ui->actionQuantitative_anisotropy_QA,SIGNAL(triggered()),tractWidget,SLOT(save_fa_as()));
    }

    // report
    {
        connect(ui->report_index,SIGNAL(currentIndexChanged(int)),this,SLOT(on_refresh_report_clicked()));
        connect(ui->profile_dir,SIGNAL(currentIndexChanged(int)),this,SLOT(on_refresh_report_clicked()));
    }
    // color bar
    {
        connect(ui->color_bar_style,SIGNAL(currentIndexChanged(int)),this,SLOT(update_color_map()));
        connect(ui->color_from,SIGNAL(clicked()),this,SLOT(update_color_map()));
        connect(ui->color_to,SIGNAL(clicked()),this,SLOT(update_color_map()));
        connect(ui->tract_color_max_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
        connect(ui->tract_color_min_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
        connect(ui->update_rendering,SIGNAL(clicked()),glWidget,SLOT(makeTracts()));
        connect(ui->update_rendering,SIGNAL(clicked()),glWidget,SLOT(updateGL()));
        on_tract_color_index_currentIndexChanged(0);
    }


    {
        scene.show_slice();
        scene.center();
        slice_no_update = false;
        copy_target = 0;
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
        ui->turning_angle->setValue(settings.value("turning_angle",60).toDouble());
        ui->smoothing->setValue(settings.value("smoothing",0.0).toDouble());
        ui->min_length->setValue(settings.value("min_length",0.0).toDouble());
        ui->max_length->setValue(settings.value("max_length",500).toDouble());
        ui->tracking_method->setCurrentIndex(settings.value("tracking_method",0).toInt());
        ui->seed_plan->setCurrentIndex(settings.value("seed_plan",0).toInt());
        ui->initial_direction->setCurrentIndex(settings.value("initial_direction",1).toInt());
        ui->interpolation->setCurrentIndex(settings.value("interpolation",0).toInt());
        ui->tracking_plan->setCurrentIndex(settings.value("tracking_plan",0).toInt());
        ui->track_count->setValue(settings.value("track_count",2000).toInt());
        ui->thread_count->setCurrentIndex(settings.value("thread_count",0).toInt());

        ui->glSagCheck->setChecked(settings.value("SagSlice",1).toBool());
        ui->glCorCheck->setChecked(settings.value("CorSlice",1).toBool());
        ui->glAxiCheck->setChecked(settings.value("AxiSlice",1).toBool());
        ui->RenderingQualityBox->setCurrentIndex(settings.value("RenderingQuality",1).toInt());

        ui->color_from->setColor(settings.value("color_from",0x00FF1010).toInt());
        ui->color_to->setColor(settings.value("color_to",0x00FFFF10).toInt());
    }

    qApp->installEventFilter(this);
}

tracking_window::~tracking_window()
{

    QSettings settings;
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    settings.setValue("color_from",ui->color_from->color().rgba());
    settings.setValue("color_to",ui->color_to->color().rgba());
    tractWidget->delete_all_tract();
    delete ui;
    delete handle;
    handle = 0;
    //std::cout << __FUNCTION__ << " " << __FILE__ << std::endl;
}
void tracking_window::get_nifti_trans(std::vector<float>& trans)
{
    if(!trans_to_mni.empty())
        trans = trans_to_mni;
    else
        if(mi3.get())
        {
            trans.resize(16);
            image::create_affine_transformation_matrix(
                        mi3->get(),
                        mi3->get()+9,trans.begin(),image::vdim<3>());
            fa_template_imp.get_transformation(trans);
        }
}
void tracking_window::get_dicom_trans(std::vector<float>& trans)
{
    std::vector<float> flip_xy(16),t(16);
    flip_xy[0] = -1;
    flip_xy[3] = slice.geometry[0]-1;
    flip_xy[5] = -1;
    flip_xy[7] = slice.geometry[1]-1;
    flip_xy[10] = 1;
    flip_xy[15] = 1;
    get_nifti_trans(t);
    trans.resize(16);
    trans[15] = 1.0;
    math::matrix_product(t.begin(),flip_xy.begin(),trans.begin(),math::dim<3,4>(),math::dim<4,4>());
}

bool tracking_window::eventFilter(QObject *obj, QEvent *event)
{
  if (event->type() == QEvent::MouseMove &&
      obj->parent() && obj->parent()->objectName() == QString("graphicsView"))
  {

      QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
      QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
      float x,y,z;
      if(ui->view_style->currentIndex() == 0)// single slice
      {
          if(slice.cur_dim != 2)
              point.setY(scene.height() - point.y());
          if (!slice.get3dPosition(((float)point.x()) / scene.display_ratio,
                                   ((float)point.y()) / scene.display_ratio, x, y, z))
              return false;
      }
      else
      {
          x = ((float)point.x())*(float)scene.mosaic_size / scene.display_ratio;
          y = ((float)point.y())*(float)scene.mosaic_size / scene.display_ratio;
          z = std::floor(y/slice.geometry[1])*scene.mosaic_size + std::floor(x/slice.geometry[0]);
          x -= std::floor(x/slice.geometry[0])*slice.geometry[0];
          y -= std::floor(y/slice.geometry[1])*slice.geometry[1];
      }
      QString status;
      status = QString("(%1,%2,%3) ").arg((int)x).arg((int)y).arg((int)z);

      // show atlas position
      if(mi3.get() || !trans_to_mni.empty())
      {
          image::vector<3,float> cur_coordinate(x, y, z),mni_coordinate;
          if(!trans_to_mni.empty())
          {
              // flip xy
              mni_coordinate[0] = slice.geometry[0]-mni_coordinate[0]-1;
              mni_coordinate[1] = slice.geometry[1]-mni_coordinate[1]-1;
              image::vector_transformation(cur_coordinate.begin(),mni_coordinate.begin(), trans_to_mni,image::vdim<3>());
          }
          else
          {
              const float* m = mi3->get();
              image::vector_transformation(cur_coordinate.begin(),mni_coordinate.begin(), m,m + 9, image::vdim<3>());
              fa_template_imp.to_mni(mni_coordinate);
          }

          status += QString("MNI(%1,%2,%3) ").
                    arg(mni_coordinate[0]).
                        arg(mni_coordinate[1]).
                            arg(mni_coordinate[2]);

          if(!atlas_list.empty())
            status += atlas_list[ui->atlasListBox->currentIndex()].get_label_name_at(mni_coordinate).c_str();

      }
      status += " ";
      std::vector<float> data;
      handle->get_voxel_information(x, y, z, data);
      for(unsigned int index = 0,data_index = 0;index < view_name.size() && data_index < data.size();++index)
          if(view_name[index] != "color")
          {
              status += view_name[index].c_str();
              status += QString("=%1 ").arg(data[data_index]);
              ++data_index;
          }
      ui->statusbar->showMessage(status);
      copy_target = 1;
  }
  return false;
}


void tracking_window::SliderValueChanged(void)
{
    if(!slice_no_update && slice.set_slice_pos(
            ui->SagSlider->value(),
            ui->CorSlider->value(),
            ui->AxiSlider->value()))
    {
        if(ui->view_style->currentIndex() == 0)
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
    ui->statusbar->showMessage(
                QString("slice position (sagittal,coronal,axial)=(%1,%2,%3) ").
                arg(ui->glSagSlider->value()).
                arg(ui->glCorSlider->value()).
                arg(ui->glAxiSlider->value()));
    if(!slice_no_update && cur_slice.set_slice_pos(
                ui->glSagSlider->value(),
                ui->glCorSlider->value(),
                ui->glAxiSlider->value()))
            glWidget->updateGL();

}


void tracking_window::on_AxiView_clicked()
{
    slice.cur_dim = 2;
    if(ui->view_style->currentIndex() == 0)
        scene.show_slice();
}

void tracking_window::on_CorView_clicked()
{
    slice.cur_dim = 1;
    if(ui->view_style->currentIndex() == 0)
        scene.show_slice();
}

void tracking_window::on_SagView_clicked()
{
    slice.cur_dim = 0;
    if(ui->view_style->currentIndex() == 0)
        scene.show_slice();
}

void tracking_window::on_tool0_pressed()
{
    scene.sel_mode = 0;
}

void tracking_window::on_tool1_pressed()
{
    scene.sel_mode = 1;
}

void tracking_window::on_tool2_pressed()
{
    scene.sel_mode = 2;
}

void tracking_window::on_tool3_pressed()
{
    scene.sel_mode = 3;
}

void tracking_window::on_tool4_clicked()
{
    scene.sel_mode = 4;
}

void tracking_window::on_tool5_pressed()
{
    scene.sel_mode = 5;
}

void tracking_window::on_sliceViewBox_currentIndexChanged(int index)
{
    ui->actionSave_Anisotrpy_Map_as->setText(QString("Save ") +
                                             ui->sliceViewBox->currentText()+" as...");
    slice.set_view_name(ui->sliceViewBox->currentText().toLocal8Bit().begin(),
                        ui->overlay->currentText().toLocal8Bit().begin());
    scene.show_slice();
    glWidget->updateGL();
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
}

void tracking_window::on_glCorView_clicked()
{
    glWidget->set_view(1);
    glWidget->updateGL();
}

void tracking_window::on_glAxiView_clicked()
{
    glWidget->set_view(2);
    glWidget->updateGL();
}

void tracking_window::on_SliceModality_currentIndexChanged(int index)
{
    glWidget->current_visible_slide = index;
    slice_no_update = true;
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
        "Open Images files",absolute_path,
        "Image files (*.dcm *.hdr *.nii);;All files (*.*)" );
    if( filenames.isEmpty() || !glWidget->addSlices(filenames))
        return;
    ui->SliceModality->addItem(QFileInfo(filenames[0]).baseName());
    ui->SliceModality->setCurrentIndex(glWidget->other_slices.size());
}



void tracking_window::on_actionStatistics_triggered()
{
    tractWidget->showCurTractStatistics(
            ui->fa_threshold->value(),
            std::cos(ui->turning_angle->value() * 3.1415926 / 180.0));
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
        renderWidget->setData("tract_visible_tracts",0);
        break;
    case 1:
        renderWidget->setData("anti_aliasing",0);
        renderWidget->setData("tract_tube_detail",1);
        renderWidget->setData("tract_visible_tracts",2);
        break;
    case 2:
        renderWidget->setData("anti_aliasing",1);
        renderWidget->setData("tract_tube_detail",3);
        renderWidget->setData("tract_visible_tracts",4);
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
        {
            QImage image;
            ui->report_widget->saveImage(image);
            QApplication::clipboard()->setImage(image);
        }
        return;
    }
}

void tracking_window::on_atlasListBox_currentIndexChanged(int atlas_index)
{
    ui->atlasComboBox->clear();
    for (unsigned int index = 0; index < atlas_list[atlas_index].get_list().size(); ++index)
        ui->atlasComboBox->addItem(atlas_list[atlas_index].get_list()[index].c_str());
}

void tracking_window::on_refresh_report_clicked()
{
    if(tractWidget->tract_models.size() > 1 &&
       tractWidget->tract_models[0]->get_tract_color(0) ==
       tractWidget->tract_models[1]->get_tract_color(0))
        tractWidget->assign_colors();
    ui->dockWidget_report->show();
    ui->report_widget->clearGraphs();

    float threshold = ui->fa_threshold->value();
    float cull_angle_cos = std::cos(ui->turning_angle->value() * 3.1415926 / 180.0);
    unsigned int profile_dir = ui->profile_dir->currentIndex();
    int profile_on_length = 0;// 1 :along tract 2: mean value
    if(profile_dir > 2)
    {
        profile_on_length = ui->profile_dir->currentIndex()-2;
        profile_dir = 0;
    }
    double detail = profile_on_length ? 1.0 : 2.0;
    unsigned int profile_width = (slice.geometry[profile_dir]+1)*detail;
    double max_y = 0.0;
    double min_x = slice.geometry[profile_dir],max_x = 0;

    float band_width = ui->report_bandwidth->value();
    std::vector<float> weighting((int)(1.0+band_width*3.0));
    for(int index = 0;index < weighting.size();++index)
    {
        float x = index;
        weighting[index] = std::exp(-x*x/2.0/band_width/band_width);
    }
    for(unsigned int index = 0;index < tractWidget->tract_models.size();++index)
    {
        if(tractWidget->item(index,0)->checkState() != Qt::Checked)
            continue;
        const std::vector<std::vector<float> >& tracts =
                tractWidget->tract_models[index]->get_tracts();
        if(tracts.empty())
            continue;
        // along tract profile
        if(profile_on_length == 1)
            profile_width = tracts[0].size()/2.0;
        // mean value of each tract
        if(profile_on_length == 2)
            profile_width = tracts.size();

        std::vector<float> data_profile(profile_width);
        std::vector<float> data_profile_w(profile_width);
        {
            std::vector<std::vector<float> > data;
            if(ui->report_index->currentIndex())
                handle->get_tracts_data(tracts,ui->report_index->currentText().toLocal8Bit().begin(),data);
            else
                handle->get_tracts_fa(tracts,threshold,cull_angle_cos,data);

            if(profile_on_length == 2)// list the mean fa value of each tract
            {
                data_profile.resize(data.size());
                data_profile_w.resize(data.size());
                for(unsigned int index = 0;index < data_profile.size();++index)
                {
                    data_profile[index] = image::mean(data[index].begin(),data[index].end());
                    data_profile_w[index] = 1.0;
                }
            }
            else
                for(int i = 0;i < data.size();++i)
                    for(int j = 0;j < data[i].size();++j)
                    {
                        int pos = profile_on_length ?
                                  j*(int)profile_width/data[i].size() :
                                  std::floor(tracts[i][j + j + j + profile_dir]*detail+0.5);
                        if(pos < 0)
                            pos = 0;
                        if(pos >= profile_width)
                            pos = profile_width-1;

                        data_profile[pos] += data[i][j]*weighting[0];
                        data_profile_w[pos] += weighting[0];
                        for(int k = 1;k < weighting.size();++k)
                        {
                            if(pos > k)
                            {
                                data_profile[pos-k] += data[i][j]*weighting[k];
                                data_profile_w[pos-k] += weighting[k];
                            }
                            if(pos+k < data_profile.size())
                            {
                                data_profile[pos+k] += data[i][j]*weighting[k];
                                data_profile_w[pos+k] += weighting[k];
                            }
                        }
                    }
        }

        QVector<double> x(profile_width),y(profile_width);
        for(unsigned int j = 0;j < profile_width;++j)
        {
            x[j] = (double)j/detail;
            if(data_profile_w[j] + 1.0 != 1.0)
                y[j] = data_profile[j]/data_profile_w[j];
            else
                y[j]= 0.0;
            if(y[j] > max_y)
                max_y = y[j];
            if(y[j] > 0.0)
            {
                if(x[j] < min_x)
                    min_x = x[j];
                if(x[j] > max_x)
                    max_x = x[j];
            }
        }
        ui->report_widget->addGraph();
        QPen pen;
        image::rgb_color color = tractWidget->tract_models[index]->get_tract_color(0);
        pen.setColor(QColor(color.r,color.g,color.b,200));
        pen.setWidth(ui->linewidth->value());
        ui->report_widget->graph()->setLineStyle(QCPGraph::lsLine);
        ui->report_widget->graph()->setPen(pen);
        ui->report_widget->graph()->setData(x, y);
        ui->report_widget->graph()->setName(tractWidget->item(index,0)->text());
        // give the axes some labels:
        //customPlot->xAxis->setLabel("x");
        //customPlot->yAxis->setLabel("y");
        // set axes ranges, so we see all data:


    }
    ui->report_widget->xAxis->setRange(min_x,max_x);
    ui->report_widget->yAxis->setRange(ui->report_index->currentIndex() ? 0 : threshold, max_y);
    if(ui->report_legend->checkState() == Qt::Checked)
    {
        ui->report_widget->legend->setVisible(true);
        QFont legendFont = font();  // start out with MainWindow's font..
        legendFont.setPointSize(9); // and make a bit smaller for legend
        ui->report_widget->legend->setFont(legendFont);
        ui->report_widget->legend->setPositionStyle(QCPLegend::psRight);
        ui->report_widget->legend->setBrush(QBrush(QColor(255,255,255,230)));
    }
    else
        ui->report_widget->legend->setVisible(false);

    ui->report_widget->replot();
    copy_target = 2;
}

void tracking_window::on_actionRestore_window_layout_triggered()
{
    restoreGeometry(default_geo);
    restoreState(default_state);

}


unsigned char color_spectrum_value(unsigned char center, unsigned char value)
{
    unsigned char dif = center > value ? center-value:value-center;
    if(dif < 32)
        return 255;
    dif -= 32;
    if(dif >= 64)
        return 0;
    return 255-(dif << 2);
}

void tracking_window::update_color_map(void)
{
    color_map.resize(256);
    bar.resize(image::geometry<2>(20,256));

    if(ui->color_bar_style->currentIndex() == 0)
    {
        image::rgb_color from_color = ui->color_from->color().rgba();
        image::rgb_color to_color = ui->color_to->color().rgba();
        for(unsigned int index = 0;index < color_map.size();++index)
        {
            float findex = (float)index/255.0;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color_map[index][2-rgb_index] =
                        (float)to_color[rgb_index]*findex/255.0+
                        (float)from_color[rgb_index]*(1.0-findex)/255.0;
        }



        for(unsigned int index = 1;index < 255;++index)
        {
            float findex = (float)index/256.0;
            image::rgb_color color;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[rgb_index] = (float)from_color[rgb_index]*findex+(float)to_color[rgb_index]*(1.0-findex);
            std::fill(bar.begin()+index*20+1,bar.begin()+(index+1)*20-1,color);
        }
    }

    if(ui->color_bar_style->currentIndex() == 1)
    {
        for(unsigned int index = 0;index < color_map.size();++index)
        {
            color_map[index][0] = (float)color_spectrum_value(128+64,index)/255.0;
            color_map[index][1] = (float)color_spectrum_value(128,index)/255.0;
            color_map[index][2] = (float)color_spectrum_value(64,index)/255.0;
        }
        for(unsigned int index = 1;index < 255;++index)
        {
            image::rgb_color color;
            color.r = color_spectrum_value(64,index);
            color.g = color_spectrum_value(128,index);
            color.b = color_spectrum_value(128+64,index);
            std::fill(bar.begin()+index*20+1,bar.begin()+(index+1)*20-1,color);
        }
    }

    color_bar.clear();
    QGraphicsTextItem *max_text = color_bar.addText(QString::number(ui->tract_color_max_value->value()));
    QGraphicsTextItem *min_text = color_bar.addText(QString::number(ui->tract_color_min_value->value()));
    QGraphicsPixmapItem *map = color_bar.addPixmap(QPixmap::fromImage(
            QImage((unsigned char*)&*bar.begin(),bar.width(),bar.height(),QImage::Format_RGB32)));
    max_text->moveBy(10,-128-10);
    min_text->moveBy(10,128-10);
    map->moveBy(-10,-128);
    ui->color_bar_view->show();

}

void tracking_window::on_tract_color_index_currentIndexChanged(int index)
{
    unsigned int item_index = index ? index+handle->fib_data.other_mapping_index-1:0;
    float max_value = handle->fib_data.view_item[item_index].max_value;
    float min_value = handle->fib_data.view_item[item_index].min_value;
    float scale2 = std::pow(10.0,std::floor(2.0-std::log10(max_value)));
    float scale1 = std::pow(10.0,std::floor(1.0-std::log10(max_value)));
    float decimal = std::floor(2.0-std::log10(max_value));
    if(decimal < 1.0)
        decimal = 1.0;
    ui->tract_color_max_value->setDecimals(decimal);
    ui->tract_color_max_value->setMaximum(std::ceil(max_value*scale1)/scale1);
    ui->tract_color_max_value->setMinimum(std::floor(min_value*scale1)/scale1);
    ui->tract_color_max_value->setSingleStep(std::ceil(max_value*scale1)/scale1/50);
    ui->tract_color_max_value->setValue(std::ceil(max_value*scale2)/scale1);

    ui->tract_color_min_value->setDecimals(decimal);
    ui->tract_color_min_value->setMaximum(std::ceil(max_value*scale1)/scale1);
    ui->tract_color_min_value->setMinimum(std::floor(min_value*scale1)/scale1);
    ui->tract_color_min_value->setSingleStep(std::ceil(max_value*scale1)/scale1/50);
    ui->tract_color_min_value->setValue(std::floor(min_value*scale2)/scale1);
    update_color_map();
}

void tracking_window::on_save_report_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                absolute_path + "/report.txt",
                "Report file (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toLocal8Bit().begin());
    if(!out)
    {
        QMessageBox::information(this,"Error","Cannot write to file",0);
        return;
    }

    std::vector<QCPDataMap::const_iterator> iterators(ui->report_widget->graphCount());
    for(int row = 0;;++row)
    {
        bool has_output = false;
        for(int index = 0;index < ui->report_widget->graphCount();++index)
        {
            if(row == 0)
            {
                out << ui->report_widget->graph(index)->name().toLocal8Bit().begin() << "\t\t";
                has_output = true;
                continue;
            }
            if(row == 1)
            {
                out << "x\ty\t";
                iterators[index] = ui->report_widget->graph(index)->data()->begin();
                has_output = true;
                continue;
            }
            if(iterators[index] != ui->report_widget->graph(index)->data()->end())
            {
                out << iterators[index]->key << "\t" << iterators[index]->value << "\t";
                ++iterators[index];
                has_output = true;
            }
            else
                out << "\t\t";
        }
        out << std::endl;
        if(!has_output)
            break;
    }
}

void tracking_window::on_tracking_index_currentIndexChanged(int index)
{
    handle->fib_data.fib.set_tracking_index(index);
    float max_value = *std::max_element(handle->fib_data.fib.fa[0],handle->fib_data.fib.fa[0]+handle->fib_data.fib.dim.size());
    ui->fa_threshold->setRange(0.0,max_value*1.1);
    ui->fa_threshold->setValue(0.6*image::segmentation::otsu_threshold(
        image::basic_image<float, 3,image::const_pointer_memory<float> >(handle->fib_data.fib.fa[0],handle->fib_data.fib.dim)));
    ui->fa_threshold->setSingleStep(max_value/50.0);
}


void tracking_window::on_deleteSlice_clicked()
{
    if(ui->SliceModality->currentIndex() == 0)
        return;
    int index = ui->SliceModality->currentIndex();
    ui->SliceModality->setCurrentIndex(0);
    glWidget->delete_slice(index-1);
    ui->SliceModality->removeItem(index);
}



void tracking_window::on_actionSave_Report_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save report as",
                absolute_path + "/report.jpg",
                "JPEC file (*.jpg);;BMP file (*.bmp);;PDF file (*.pdf);;PNG file (*.png);;All files (*.*)");
    if(QFileInfo(filename).completeSuffix().toLower() == "jpg")
        ui->report_widget->saveJpg(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "bmp")
        ui->report_widget->saveBmp(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "png")
        ui->report_widget->savePng(filename);
    if(QFileInfo(filename).completeSuffix().toLower() == "pdf")
        ui->report_widget->savePdf(filename);

}

void tracking_window::on_actionSave_Tracts_in_MNI_space_triggered()
{
    std::vector<float> t(16);
    get_dicom_trans(t);
    tractWidget->saveTransformedTracts(&*t.begin());
}
