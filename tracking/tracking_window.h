#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <QDockWidget>
#include <QTextBrowser>
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
    std::vector<std::shared_ptr<SliceModel> > overlay_slices,stay_slices;
    bool slice_view_flip_x(unsigned char d) const {return d && (*this)["orientation_convention"].toInt();}
    bool slice_view_flip_y(unsigned char d) const {return d != 2;}
public:
    std::shared_ptr<tipl::ml3d::unet3d> unet;
    std::vector<std::string> unet_label_name;
    bool run_unet(void);
public:
    connectometry_result cnt_result;
public:
    std::shared_ptr<QTimer> timer2;
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
    bool addSlices(QStringList filenames,QString name,bool cmd,bool mni = false);
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
    void check_reg(void);
    void change_contrast();
    void on_enable_auto_track_clicked();
private slots:
    void on_actionTDI_Import_Slice_Space_triggered();
    void on_actionTDI_Subvoxel_Diffusion_Space_triggered();
    void on_actionTDI_Diffusion_Space_triggered();
    void on_actionTracts_to_seeds_triggered();
    void on_actionEndpoints_to_seeding_triggered();

    void SliderValueChanged(void);

    void on_deleteSlice_clicked();

    void on_actionTract_Analysis_Report_triggered();
    void on_actionConnectivity_matrix_triggered();
    void on_addRegionFromAtlas_clicked();
    void on_actionQuality_Assessment_triggered();
    void on_actionAuto_Rotate_triggered(bool checked);
    void on_rendering_efficiency_currentIndexChanged(int index);

    void stripSkull();
    void on_actionAdjust_Mapping_triggered();
    void on_actionSave_mapping_triggered();
    void on_actionLoad_mapping_triggered();
    void on_actionLoad_Color_Map_triggered();
    void on_track_style_currentIndexChanged(int index);
    void on_addSlices_clicked();

    void on_is_overlay_clicked();
    void on_actionInsert_MNI_images_triggered();
    void on_actionOpen_Connectivity_Matrix_triggered();
    void on_SlicePos_valueChanged(int value);


    void on_actionFIB_protocol_triggered();
    void on_SliceModality_currentIndexChanged(int index);
    void on_actionSave_T1W_T2W_images_triggered();
    void on_actionMark_Region_on_T1W_T2W_triggered();
    void on_actionMark_Tracts_on_T1W_T2W_triggered();
    void on_actionSave_Slices_to_DICOM_triggered();

    void on_actionLoad_Parameter_ID_triggered();

    void insertPicture();

    void on_template_box_currentIndexChanged(int index);
    void on_actionManual_Atlas_Alignment_triggered();

    void on_actionStrip_Skull_triggered();
    void on_actionSegment_Tissue_triggered();
    void on_tract_target_0_currentIndexChanged(int index);
    void on_tract_target_1_currentIndexChanged(int index);
    void on_actionLoad_MNI_mapping_triggered();
    void on_actionSave_MNI_mapping_triggered();


    void on_stay_clicked();
};

#endif // TRACKING_WINDOW_H
