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
std::string show_info_dialog(const std::string& title,const std::string& result,const std::string& file_name_hint = "report.txt");
struct command_history{
private:
    static bool is_loading(const std::string& cmd);
    static bool is_saving(const std::string& cmd);
public:
    int current_recording_instance = 0;
    bool has_other_thread = false;
    std::string default_parent_path,default_stem,default_stem2,current_cmd;
    std::vector<std::string> commands;
    bool run(tracking_window *parent,const std::vector<std::string>& cmd,bool apply_to_others);
public:
    struct surrogate {
        command_history& owner;
        std::vector<std::string>& cmd;
        std::string& error_msg;
        surrogate(command_history& owner,
                  std::vector<std::string>& cmd_,
                  std::string& error_msg_) : owner(owner),cmd(cmd_),error_msg(error_msg_)
        {
            ++owner.current_recording_instance;
        }
        bool canceled(void)
        {
            error_msg = "canceled";
            return false;
        }
        bool failed(const std::string& msg)
        {
            error_msg = msg;
            return false;
        }
        bool not_processed(void)
        {
            error_msg = "not_processed";
            return false;
        }
        ~surrogate()
        {
            --owner.current_recording_instance;
            if(error_msg == "canceled" || error_msg == "not_processed")
                return;
            while(!cmd.empty() && cmd.back().empty())
                cmd.pop_back();
            std::string output(tipl::merge(cmd,','));
            tipl::out() << "--cmd=" << output;
            if(!error_msg.empty())
            {
                tipl::error() << error_msg;
                return;
            }
            owner.record(output);
        }
    };
    std::shared_ptr<surrogate> record(std::string& error_msg_,
                                      std::vector<std::string>& cmd);
    void record(const std::string& output);
    std::string file_stem(void) const;
    bool get_dir(QWidget* parent,std::string& cmd);
    bool get_filename(QWidget* parent,std::string& cmd,const std::string& post_fix = "");

};
class GLWidget;
class tract_report;
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
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent( QKeyEvent * event) override;

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

    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
public:
    command_history history;
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
    QStringList dt_list; // for dt_index1 dt_index2
public:
    std::shared_ptr<tipl::ml3d::unet3d> unet;
    std::vector<std::string> unet_label_name;
    bool run_unet(void);
public:
    connectometry_result cnt_result;
public:
    std::shared_ptr<QTimer> timer2;
    void start_reg(void);
    std::string get_parameter_id(void);
public:
    std::shared_ptr<tract_report> tact_report_imp;
    std::shared_ptr<connectivity_matrix_dialog> connectivity_matrix;
public:
    QString work_path;
    std::shared_ptr<fib_data> handle;
    std::vector<std::shared_ptr<SliceModel> > slices;
    std::shared_ptr<SliceModel> current_slice;
    bool addSlices(const std::string& name,const std::string& path);
    bool addSlices(std::shared_ptr<SliceModel> new_slice);
    bool openSlices(const std::string& filename,bool is_mni = false);
    void updateSlicesMenu(void);
    float get_fa_threshold(void);
    bool no_update = true;
public:
    bool eventFilter(QObject *obj, QEvent *event) override;
    QVariant operator[](QString name)const;
    void set_data(QString name, QVariant value);
    void on_tracking_index_currentIndexChanged(int index);
    QString get_save_file_name(QString title,QString file_name,QString file_type);
    void report(QString string);
    void move_slice_to(tipl::vector<3,float> pos);
    void set_roi_zoom(float zoom);
public:
    std::string error_msg;
    bool command(std::vector<std::string> cmd);
    QString get_action_data(void) const
    {
        QAction *action = qobject_cast<QAction *>(sender());
        return action ? action->data().toString() : QString();
    }
public slots:
    void check_reg(void);
    void change_contrast();
    void on_enable_auto_track_clicked();
private slots:
    void SliderValueChanged(void);

    void on_deleteSlice_clicked();

    void on_actionTract_Analysis_Report_triggered();
    void on_actionConnectivity_matrix_triggered();
    void on_addRegionFromAtlas_clicked();

    void on_actionAuto_Rotate_triggered(bool checked);
    void on_rendering_efficiency_currentIndexChanged(int index);

    void stripSkull();
    void on_actionAdjust_Mapping_triggered();
    void on_actionSave_mapping_triggered();
    void on_actionLoad_mapping_triggered();
    void on_actionLoad_Color_Map_triggered();
    void on_track_style_currentIndexChanged(int index);

    void on_is_overlay_clicked();
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


    void on_stay_clicked();
    void on_actionSave_3D_Model_triggered();
    void on_actionEdit_Slices_triggered();
    void on_alt_mapping_currentIndexChanged(int index);
    void on_directional_color_clicked();
    void on_actionCommand_History_triggered();

    void run_action();
};

#endif // TRACKING_WINDOW_H
