#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <QDockWidget>
#include <QTextBrowser>
#include "TIPL/tipl.hpp"
#include <vector>
#include "SliceModel.h"
#include "slice_view_scene.h"
#include "tract/tracttablewidget.h"
#include "connectometry/group_connectometry_analysis.h"
class fib_data;
class RenderingTableWidget;
class RegionTableWidget;
class DeviceTableWidget;
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
    explicit QGLDockWidget(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags()):QDockWidget(parent,flags){}
protected:
    void closeEvent(QCloseEvent *e) override
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
    explicit tracking_window(QWidget *parent,
                             std::shared_ptr<fib_data> handle);
    ~tracking_window();

    Ui::tracking_window *ui;
    ::GLWidget *glWidget = nullptr;
    QGLDockWidget* gLdock = nullptr;
    RegionTableWidget *regionWidget = nullptr;
    TractTableWidget *tractWidget = nullptr;
    RenderingTableWidget *renderWidget = nullptr;
    DeviceTableWidget *deviceWidget = nullptr;
public:
    slice_view_scene scene;
    bool slice_need_update = false;
    float get_scene_zoom(void){return get_scene_zoom(current_slice);}
    float get_scene_zoom(std::shared_ptr<SliceModel> slice);
public:
    unsigned char cur_dim = 2;
    std::vector<std::shared_ptr<SliceModel> > overlay_slices;
    bool slice_view_flip_x(unsigned char d) const {return d && (*this)["orientation_convention"].toInt();}
    bool slice_view_flip_y(unsigned char d) const {return d != 2;}
public:
    connectometry_result cnt_result;
public:
    std::shared_ptr<QTimer> timer2;
    std::pair<int,int> get_dt_index_pair(void);
    void set_tracking_param(ThreadData& tracking_thread);
public:
    std::shared_ptr<tract_report> tact_report_imp;
    std::shared_ptr<color_bar_dialog> color_bar;
    std::shared_ptr<connectivity_matrix_dialog> connectivity_matrix;
public:
    QString work_path;
    std::shared_ptr<fib_data> handle;
    std::vector<std::shared_ptr<SliceModel> > slices;
    std::shared_ptr<SliceModel> current_slice;
    bool addSlices(QStringList filenames,QString name,bool cmd);
    void updateSlicesMenu(void);
    float get_fa_threshold(void);
    bool no_update = false;
public:
    bool eventFilter(QObject *obj, QEvent *event) override;
    QVariant operator[](QString name)const;
    void set_data(QString name, QVariant value);
    void on_tracking_index_currentIndexChanged(int index);
    QString get_save_file_name(QString title,QString file_name,QString file_type);
    void report(QString string);
    void move_slice_to(tipl::vector<3,float> pos);
    bool map_to_mni(void);
    void set_roi_zoom(float zoom);
public:
    std::string error_msg;
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
    void update_scene_slice(void);
private slots:
    void on_actionRestore_window_layout_triggered();
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

    void on_deleteSlice_clicked();

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
    void stripSkull();
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
    void on_SliceModality_currentIndexChanged(int index);
    void on_actionSave_T1W_T2W_images_triggered();
    void on_actionMark_Region_on_T1W_T2W_triggered();
    void on_actionMark_Tracts_on_T1W_T2W_triggered();
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
    void on_actionLoad_Presentation_triggered();
    void on_actionSave_Presentation_triggered();
    void on_actionZoom_In_triggered();
    void on_actionZoom_Out_triggered();
    void on_min_value_gl_valueChanged(double arg1);
    void on_max_value_gl_valueChanged(double arg1);
    void on_min_slider_sliderMoved(int position);
    void on_max_slider_sliderMoved(int position);
    void on_actionInsert_Axial_Pictures_triggered();
    void on_actionInsert_Coronal_Pictures_triggered();
    void on_show_track_toggled(bool checked);
    void on_actionInsert_Sagittal_Picture_triggered();
    void on_template_box_currentIndexChanged(int index);
    void on_actionManual_Atlas_Alignment_triggered();
};

#endif // TRACKING_WINDOW_H
