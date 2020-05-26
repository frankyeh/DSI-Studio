#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <QDockWidget>
#include <QTextBrowser>
#include <tipl/tipl.hpp>
#include <vector>
#include "SliceModel.h"
#include "slice_view_scene.h"
#include "tract/tracttablewidget.h"
#include "connectometry/group_connectometry_analysis.h"
class fib_data;
class RenderingTableWidget;
class RegionTableWidget;
namespace Ui {
    class tracking_window;
}

class GLWidget;
class tract_report;
class color_bar_dialog;
class connectivity_matrix_dialog;
class QGLDockWidget : public QDockWidget
{
    Q_OBJECT
public:
    explicit QGLDockWidget(QWidget *parent = nullptr, Qt::WindowFlags flags = 0):QDockWidget(parent,flags){;}
protected:
    void closeEvent(QCloseEvent *e)
    {
        QWidget::closeEvent(e);
        emit closedSignal();
    }
signals:
    void closedSignal();
};

class tracking_window : public QMainWindow
{
    Q_OBJECT
public:
    void closeEvent(QCloseEvent *event);
    void keyPressEvent ( QKeyEvent * event );

public:
    explicit tracking_window(QWidget *parent,std::shared_ptr<fib_data> handle);
    ~tracking_window();

    Ui::tracking_window *ui;
    ::GLWidget *glWidget = nullptr;
    QGLDockWidget* gLdock = nullptr;
    RegionTableWidget *regionWidget = nullptr;
    TractTableWidget *tractWidget = nullptr;
    RenderingTableWidget *renderWidget = nullptr;
public:
    slice_view_scene scene;
    float get_scene_zoom(void);
public:
    unsigned char cur_dim = 2;
    std::vector<std::shared_ptr<SliceModel> > overlay_slices;
public:
    connectometry_result cnt_result;
public:
    std::auto_ptr<QTimer> timer,timer2;
    void set_tracking_param(ThreadData& tracking_thread);
public:
    std::auto_ptr<tract_report> tact_report_imp;
    std::auto_ptr<color_bar_dialog> color_bar;
    std::auto_ptr<connectivity_matrix_dialog> connectivity_matrix;
public:
    std::shared_ptr<fib_data> handle;
    std::vector<std::shared_ptr<SliceModel> > slices;
    std::shared_ptr<SliceModel> current_slice;
    bool addSlices(QStringList filenames,QString name,bool correct_intensity,bool cmd);
    void updateSlicesMenu(void);
    float get_fa_threshold(void);
    bool no_update = false;
public:
    bool eventFilter(QObject *obj, QEvent *event);
    QVariant operator[](QString name)const;
    void set_data(QString name, QVariant value);
    void on_tracking_index_currentIndexChanged(int index);
    void on_dt_index_currentIndexChanged(int index);
    QString get_save_file_name(QString title,QString file_name,QString file_type);
    void initialize_tracking_index(int index);
    void report(QString string);
    void move_slice_to(tipl::vector<3,float> pos);
    bool can_map_to_mni(void);
    void set_roi_zoom(float zoom);
    bool command(QString cmd,QString param = "",QString param2 = "");
public slots:
    void restore_3D_window();
    void on_show_fiber_toggled(bool checked);
    void on_show_r_toggled(bool checked);
    void on_show_position_toggled(bool checked);
    void on_show_ruler_toggled(bool checked);
    void check_reg(void);
    void change_contrast();
    void on_enable_auto_track_clicked();
private slots:
    void on_actionRestore_window_layout_triggered();
    void on_actionSave_Tracts_in_Current_Mapping_triggered();
    void on_actionTDI_Import_Slice_Space_triggered();
    void on_actionTDI_Subvoxel_Diffusion_Space_triggered();
    void on_actionTDI_Diffusion_Space_triggered();
    void on_actionPaint_triggered();
    void on_actionTracts_to_seeds_triggered();
    void on_actionEndpoints_to_seeding_triggered();
    void on_tool4_clicked();
    void on_glAxiView_clicked();
    void on_glCorView_clicked();
    void on_glSagView_clicked();
    void on_actionCut_triggered();
    void on_actionDelete_triggered();
    void on_actionSelect_Tracts_triggered();
    void on_tool3_pressed();
    void on_tool2_pressed();
    void on_tool1_pressed();
    void on_tool0_pressed();
    void on_tool5_pressed();
    void on_tool6_pressed();
    void SliderValueChanged(void);

    void on_actionSave_Endpoints_in_Current_Mapping_triggered();
    void on_deleteSlice_clicked();
    void on_actionSave_Tracts_in_MNI_space_triggered();

    void on_actionManual_Registration_triggered();
    void on_actionTract_Analysis_Report_triggered();
    void on_actionConnectivity_matrix_triggered();
    void on_actionFloat_3D_window_triggered();
    void on_actionSave_tracking_parameters_triggered();
    void on_actionLoad_tracking_parameters_triggered();
    void on_actionSave_Rendering_Parameters_triggered();
    void on_actionLoad_Rendering_Parameters_triggered();
    void on_addRegionFromAtlas_clicked();
    void on_actionRestore_Settings_triggered();
    void on_actionQuality_Assessment_triggered();
    void on_actionAuto_Rotate_triggered(bool checked);
    void on_action3D_Screen_3_Views_triggered();
    void on_action3D_Screen_3_Views_Horizontal_triggered();
    void on_action3D_Screen_3_Views_Vertical_triggered();
    void on_actionROI_triggered();
    void on_rendering_efficiency_currentIndexChanged(int index);
    void on_actionCut_X_triggered();
    void on_actionCut_X_2_triggered();
    void on_actionCut_Y_triggered();
    void on_actionCut_Y_2_triggered();
    void on_actionCut_Z_triggered();
    void on_actionCut_Z_2_triggered();
    void on_actionStrip_skull_for_T1w_image_triggered();
    void on_actionImprove_Quality_triggered();
    void on_actionRestore_Tracking_Settings_triggered();
    void on_actionAdjust_Mapping_triggered();
    void on_actionSave_mapping_triggered();
    void on_actionLoad_mapping_triggered();
    void on_zoom_3d_valueChanged(double arg1);
    void on_actionLoad_Color_Map_triggered();
    void on_track_style_currentIndexChanged(int index);
    void on_addSlices_clicked();
    void on_actionSingle_triggered();
    void on_actionDouble_triggered();
    void on_actionStereoscopic_triggered();

    void on_is_overlay_clicked();
    void on_actionInsert_MNI_images_triggered();
    void on_actionOpen_Connectivity_Matrix_triggered();
    void on_SlicePos_valueChanged(int value);
    void on_actionKeep_Current_Slice_triggered();
    void on_show_3view_toggled(bool checked);
    void on_show_edge_toggled(bool checked);
    void on_actionFIB_protocol_triggered();
    void on_template_box_activated(int index);
    void on_SliceModality_currentIndexChanged(int index);
    void on_actionSave_T1W_T2W_images_triggered();
    void on_actionMark_Region_on_T1W_T2W_triggered();
    void on_actionMark_Tracts_on_T1W_T2W_triggered();
    void on_actionApply_Operation_triggered();
    void on_actionSave_Slices_to_DICOM_triggered();
    void on_zoom_valueChanged(double arg1);
    void Move_Slice_X();
    void Move_Slice_X2();
    void Move_Slice_Y();
    void Move_Slice_Y2();
    void Move_Slice_Z();
    void Move_Slice_Z2();
    void on_actionLoad_Parameter_ID_triggered();
    void on_actionMove_Objects_triggered();
};

#endif // TRACKING_WINDOW_H
