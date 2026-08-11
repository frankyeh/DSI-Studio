#ifndef TRACKING_WINDOW_H
#define TRACKING_WINDOW_H

#include <QMainWindow>
#include <QTreeWidget>
#include <QGraphicsScene>
#include <QDockWidget>
#include <QTextBrowser>
#include <vector>
#include <memory>
#include <map>
#include "SliceModel.h"
#include "slice_view_scene.h"
#include "tract/tracttablewidget.h"
#include "connectometry/group_connectometry_analysis.h"
#include "console.h"
class fib_data;
class RenderingTableWidget;
class RegionTableWidget;
class DeviceTableWidget;
namespace Ui {
    class tracking_window;
}
std::string show_info_dialog(const std::string& title,const std::string& result,const std::string& file_name_hint = "report.txt");
enum class command_source{User,AI,Internal};
QString command_window_id(QWidget*,const char*);
std::string command_record(const QString& window_id,
                           const std::vector<std::string>&,command_source);
struct command_history{
private:
    static bool is_loading(const std::string& cmd);
    static bool is_saving(const std::string& cmd);
public:
    command_history(QWidget* window):window(window){}
    QWidget* window;
    command_source source = command_source::User;
    std::string ai_forwarding_cmd; // non-empty for the duration of command(cmd,source): cmd[0] of the AI-dispatched command being forwarded here, so surrogate/report() can tell "I am that command" from "I am a side effect of it"
    int current_recording_instance = 0;
    bool has_other_thread = false,running_commands = false,replacing = false;
    std::string default_parent_path,default_stem,default_stem2,current_cmd,
                pending_command,pending_report;
    std::vector<std::string> commands;
    bool run(tracking_window *parent,const std::vector<std::string>& cmd,char type);
    bool run(tracking_window *parent,const std::vector<std::string>& cmd,const std::string& path,std::string& error_msg); // AI-facing: path is a folder (searched using the recorded load step's extension) or an "&"-joined explicit file list, no dialog involved
private:
    bool run(tracking_window *parent,const std::vector<std::string>& cmd,QStringList file_list); // shared replay core; by value since it's mutated (cleared) internally to abort remaining iteration
public:
    struct surrogate {
        command_history& owner;
        std::vector<std::string>& cmd;
        std::string& error_msg;
        command_source source;
        surrogate(command_history& owner,
                  std::vector<std::string>& cmd_,
                  std::string& error_msg_) :
            owner(owner),cmd(cmd_),error_msg(error_msg_),
            // console.capture means some outer AI dispatch is in progress; demote to Internal unless this
            // command IS that dispatch's own forwarded command (owner.ai_forwarding_cmd), not just a side effect of it
            source(owner.current_recording_instance || owner.running_commands ||
                   (console.capture && (owner.ai_forwarding_cmd.empty() || cmd.empty() || cmd[0] != owner.ai_forwarding_cmd)) ?
                   command_source::Internal : owner.source)
        {
            ++owner.current_recording_instance;
        }
        template<typename value_type>
        value_type from_cmd(size_t index,value_type default_value)
        {
            if(index >= cmd.size())
                cmd.resize(index+1);
            if(cmd[index].empty())
                cmd[index] = std::to_string(default_value);
            else
            {
                if constexpr (std::is_integral_v<value_type>)
                    default_value = static_cast<value_type>(QString::fromStdString(cmd[index]).toInt());
                else
                if constexpr (std::is_floating_point_v<value_type>)
                    default_value = static_cast<value_type>(QString::fromStdString(cmd[index]).toDouble());
            }
            return default_value;
        }
        const std::string& from_cmd(size_t index,const std::string& default_value)
        {
            if(index >= cmd.size())
                cmd.resize(index+1);
            if(cmd[index].empty())
                return cmd[index] = default_value;
            return cmd[index];
        }
        bool canceled(void)
        {
            error_msg = "canceled";
            return false;
        }
        bool succeed(void)
        {
            error_msg.clear();
            return true;
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
            if(!error_msg.empty())
            {
                tipl::error() << error_msg;
                return;
            }
            if(!output.empty())
                owner.report(cmd,source);
            if(owner.current_recording_instance || output.empty() || owner.running_commands)
                return;
            owner.add_record(output);
        }
    };
    std::shared_ptr<surrogate> record(std::string& error_msg_,
                                      std::vector<std::string>& cmd);
    void add_record(const std::string& output);
    void report(const std::vector<std::string>&,command_source);
    std::string file_stem(bool extended = true) const;
    bool get_directory(QWidget* parent,std::string& cmd);
    bool get_filename(QWidget* parent,std::string& cmd,const std::string& post_fix = "");
    void overwrite(const std::string& cmd)
    {
        replacing = true;
        if(commands.empty())
            return;
        if(tipl::begins_with(commands.back(),cmd))
            commands.pop_back();
    }

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
    QDialog* command_dialog = nullptr;
public:
    slice_view_scene scene;
    slice_update_type slice_need_update = none;
    // grayscale caches from the last preview_screen capture, keyed by channel label, so a zoom
    // request can crop and re-render text art without regrabbing from OpenGL or the slice scene
    std::map<std::string,QImage> last_3d_preview,last_roi_preview;
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
    connectometry_result cnt_result;
public:
    std::shared_ptr<QTimer> timer2;
    void start_reg(void);
    std::string get_parameter_id(bool auto_track); // auto_track: whether TIP applies (automatic fiber tracking) -- caller decides, typically from ui->tract_target_0->currentIndex() > 0 for an interactive call, or true for the AI run_auto_track command (which never touches that combo)
public:
    std::shared_ptr<tract_report> tact_report_imp;
    std::shared_ptr<connectivity_matrix_dialog> connectivity_matrix;
public:
    QString work_path;
    std::shared_ptr<fib_data> handle;
    std::vector<std::shared_ptr<SliceModel> > slices;
    std::shared_ptr<SliceModel> current_slice;
    bool addSlices(const std::string& name,const std::filesystem::path& path);
    bool addSlices(std::shared_ptr<SliceModel> new_slice);
    QAction* addSubMenuItem(const std::string& each,const std::string& title,const char* action);
    void updateSlicesMenu(void);
    float get_fa_threshold(void);
public:
    bool eventFilter(QObject *obj, QEvent *event) override;
    QVariant operator[](QString name)const;
    void set_data(QString name, QVariant value);
    void set_memorize_parameters(bool memorize);
    void on_tracking_index_currentIndexChanged(int index);
    void report(QString string);
    void move_slice_to(tipl::vector<3,float> pos);
    void set_roi_zoom(float zoom);
public:
    std::string error_msg;
    bool command(std::vector<std::string> cmd);
    bool command(std::vector<std::string> cmd,command_source);
    QString get_action_data(void) const
    {
        QAction *action = qobject_cast<QAction *>(sender());
        return action ? action->data().toString() : QString();
    }
signals:
    void need_gl_update(); // queued to glWidget so paintGL isn't reentered from inside a mouse-event handler
public slots:
    void check_reg(void);
private slots:

    void on_actionTract_Analysis_Report_triggered();
    void on_actionConnectivity_matrix_triggered();
    void on_addRegionFromAtlas_clicked();

    void on_actionAuto_Rotate_triggered(bool checked);
    void on_rendering_efficiency_currentIndexChanged(int index);

    void on_actionAdjust_Mapping_triggered();
    void on_actionLoad_Color_Map_triggered();
    void on_track_style_currentIndexChanged(int index);

    void on_actionOpen_Connectivity_Matrix_triggered();
    void on_SlicePos_valueChanged(int value);


    void on_actionFIB_protocol_triggered();

    void on_actionMark_Region_on_T1W_T2W_triggered();
    void on_actionMark_Tracts_on_T1W_T2W_triggered();
    void on_actionSave_Slices_to_DICOM_triggered();

    void on_actionLoad_Parameter_ID_triggered();

    void insertPicture();

    void update_unet_models(void);
    void on_template_box_currentIndexChanged(int index);
    void on_actionManual_Atlas_Alignment_triggered();

    void on_tract_target_0_currentIndexChanged(int index);
    void on_tract_target_1_currentIndexChanged(int index);

    void on_actionSave_3D_Model_triggered();
    void on_actionEdit_Slices_triggered();
    void on_alt_mapping_currentIndexChanged(int index);

    void on_actionCommand_History_triggered();

    void run_command(const std::string& cmd);
    void on_actionOpen_FIB_Directory_triggered();
    void on_device_coordinate_currentIndexChanged(int index);
    void on_segmentButton_clicked();
};

#endif // TRACKING_WINDOW_H
