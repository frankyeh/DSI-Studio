#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <image/image.hpp>
#include <vector>
#include "SliceModel.h"
#include "slice_view_scene.h"
#include "tract/tracttablewidget.h"
#include "libs/coreg/linear.hpp"
class ODFModel;
class RenderingTableWidget;
class RegionTableWidget;

namespace Ui {
    class tracking_window;
}

class GLWidget;

class tracking_window : public QMainWindow
{
    Q_OBJECT
protected:
    void closeEvent(QCloseEvent *event);
public:
    explicit tracking_window(QWidget *parent,ODFModel* handle);
    ~tracking_window();

    Ui::tracking_window *ui;
    GLWidget *glWidget;
    RegionTableWidget *regionWidget;
    TractTableWidget *tractWidget;
    RenderingTableWidget *renderWidget;
    slice_view_scene scene;

    // For clipboard
    unsigned int copy_target;

public:

    unsigned int odf_size;
    unsigned int odf_face_size;
    unsigned char has_odfs;
    std::vector<std::string> view_name;
    bool is_dti;

public:// color_bar
    image::color_image bar;
    QGraphicsScene color_bar;
    std::vector<image::vector<3,float> > color_map;

public:
    QString absolute_path;
    ODFModel* handle;
    FibSliceModel slice;
    std::auto_ptr<LinearMapping<image::basic_image<float,3,image::const_pointer_memory<float> >,image::affine_transform<3> > > mi3;
    std::vector<float> trans_to_mni;
    bool slice_no_update;
    bool eventFilter(QObject *obj, QEvent *event);
private slots:
    void on_tract_color_index_currentIndexChanged(int index);
    void on_actionRestore_window_layout_triggered();
    void on_refresh_report_clicked();
    void on_atlasListBox_currentIndexChanged(int index);
    void on_actionCopy_to_clipboard_triggered();
    void on_RenderingQualityBox_currentIndexChanged(int index);
    void on_actionSave_Tracts_in_Current_Mapping_triggered();
    void on_actionTDI_Import_Slice_Space_triggered();
    void on_actionTDI_Subvoxel_Diffusion_Space_triggered();
    void on_actionTDI_Diffusion_Space_triggered();
    void on_actionStatistics_triggered();
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
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();
    void SliderValueChanged(void);
    void glSliderValueChanged(void);
    void update_color_map(void);

    void on_save_report_clicked();
    void on_tracking_index_currentIndexChanged(int index);
    void on_actionSave_Endpoints_in_Current_Mapping_triggered();
    void on_deleteSlice_clicked();
    void on_tool5_pressed();
    void on_actionMove_Object_triggered();
};

#endif // TRACKING_WINDOW_H
