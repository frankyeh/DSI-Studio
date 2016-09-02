#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <QDockWidget>
#include <QTextBrowser>
#include <image/image.hpp>
#include <vector>
#include "SliceModel.h"
#include "slice_view_scene.h"
#include "tract/tracttablewidget.h"
#include "vbc/vbc_database.h"
class fib_data;
class RenderingTableWidget;
class RegionTableWidget;

namespace Ui {
    class tracking_window;
}

class GLWidget;
class manual_alignment;
class tract_report;
class color_bar_dialog;
class connectivity_matrix_dialog;
class QGLDockWidget : public QDockWidget
{
    Q_OBJECT
public:
    explicit QGLDockWidget(QWidget *parent = 0, Qt::WindowFlags flags = 0):QDockWidget(parent,flags){;}
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
protected:
    void closeEvent(QCloseEvent *event);
    void keyPressEvent ( QKeyEvent * event );

public:
    explicit tracking_window(QWidget *parent,std::shared_ptr<fib_data> handle);
    ~tracking_window();

    Ui::tracking_window *ui;
    GLWidget *glWidget,*glWidget2;
    std::auto_ptr<QGLDockWidget> gLdock;
    RegionTableWidget *regionWidget;
    TractTableWidget *tractWidget;
    RenderingTableWidget *renderWidget;
    slice_view_scene scene;
public:
    image::value_to_color<float> v2c,v2c_gl;
public:
    connectometry_result cnt_result;
public:
    std::auto_ptr<QTimer> timer,timer2;
    unsigned int odf_size;
    unsigned int odf_face_size;
    void set_tracking_param(ThreadData& tracking_thread);
public:
    std::auto_ptr<tract_report> tact_report_imp;
    std::auto_ptr<color_bar_dialog> color_bar;
    std::auto_ptr<connectivity_matrix_dialog> connectivity_matrix;
public:
    std::shared_ptr<fib_data> handle;
    FibSliceModel slice;
    std::vector<std::shared_ptr<CustomSliceModel> > other_slices;
    bool slice_no_update;
    bool addSlices(QStringList filenames,bool correct_intensity,bool cmd);

public:
    bool eventFilter(QObject *obj, QEvent *event);
    QVariant operator[](QString name)const;
    void set_data(QString name, QVariant value);
    void on_tracking_index_currentIndexChanged(int index);
    void add_slice_name(QString name);
    void show_info_dialog(const std::string& title,const std::string& result);
    QString get_save_file_name(QString title,QString file_name,QString file_type);
    void float3dwindow(int w,int h);
    void initialize_tracking_index(int index);
    void report(QString string);
    bool command(QString cmd,QString param = "",QString param2 = "");
public slots:
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();
    void restore_3D_window();
    void on_show_fiber_toggled(bool checked);
    void on_show_r_toggled(bool checked);
    void on_show_position_toggled(bool checked);
    void check_reg(void);
    void update_gl(void);
private slots:
    void on_actionRestore_window_layout_triggered();
    void on_actionSave_Tracts_in_Current_Mapping_triggered();
    void on_actionTDI_Import_Slice_Space_triggered();
    void on_actionTDI_Subvoxel_Diffusion_Space_triggered();
    void on_actionTDI_Diffusion_Space_triggered();
    void on_actionInsert_T1_T2_triggered();
    void on_actionPaint_triggered();
    void on_actionTracts_to_seeds_triggered();
    void on_actionEndpoints_to_seeding_triggered();
    void on_tool4_clicked();
    void on_SliceModality_currentIndexChanged(int index);
    void on_glAxiView_clicked();
    void on_glCorView_clicked();
    void on_glSagView_clicked();
    void on_actionCut_triggered();
    void on_actionDelete_triggered();
    void on_actionSelect_Tracts_triggered();
    void on_sliceViewBox_currentIndexChanged(int index);
    void on_tool3_pressed();
    void on_tool2_pressed();
    void on_tool1_pressed();
    void on_tool0_pressed();

    void SliderValueChanged(void);
    void glSliderValueChanged(void);

    void on_actionSave_Endpoints_in_Current_Mapping_triggered();
    void on_deleteSlice_clicked();
    void on_tool5_pressed();
    void on_actionMove_Object_triggered();
    void on_actionSave_Tracts_in_MNI_space_triggered();
    void on_tool6_pressed();
    void on_actionManual_Registration_triggered();
    void on_actionTract_Analysis_Report_triggered();
    void on_actionConnectivity_matrix_triggered();
    void on_zoom_3d_valueChanged(double arg1);
    void on_actionFloat_3D_window_triggered();
    void on_actionSave_tracking_parameters_triggered();
    void on_actionLoad_tracking_parameters_triggered();
    void on_actionSave_Rendering_Parameters_triggered();
    void on_actionLoad_Rendering_Parameters_triggered();
    void on_addRegionFromAtlas_clicked();
    void on_actionRestore_Settings_triggered();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_actionQuality_Assessment_triggered();
    void on_actionAuto_Rotate_triggered(bool checked);
    void on_auto_rotate_toggled(bool checked);
    void on_action3D_Screen_triggered();
    void on_action3D_Screen_3_Views_triggered();
    void on_action3D_Screen_3_Views_Horizontal_triggered();
    void on_action3D_Screen_3_Views_Vertical_triggered();
    void on_actionROI_triggered();
    void on_actionTrack_Report_triggered();
    void on_rendering_efficiency_currentIndexChanged(int index);
    void on_load_color_map_clicked();
    void on_max_value_valueChanged(double arg1);
    void on_min_value_valueChanged(double arg1);
    void on_max_color_released();
    void on_min_color_released();
    void on_load_color_map_gl_released();
    void on_actionCut_X_triggered();
    void on_actionCut_X_2_triggered();
    void on_actionCut_Y_triggered();
    void on_actionCut_Y_2_triggered();
    void on_actionCut_Z_triggered();
    void on_actionCut_Z_2_triggered();
    void on_actionStrip_skull_for_T1w_image_triggered();
    void on_actionImprove_Quality_triggered();
    void on_actionRestore_Tracking_Settings_triggered();
    void add_roi_from_atlas();
    void on_actionIndividual_Connectometry_triggered();
    void on_actionAdjust_Mapping_triggered();
    void on_actionSave_mapping_triggered();
    void on_actionLoad_mapping_triggered();
    void on_glView_currentIndexChanged(int index);
};

#endif // TRACKING_WINDOW_H
