#ifndef GROUP_CNT_HPP
#define GROUP_CNT_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QItemDelegate>
#include <QTimer>
#include <QtCharts/QtCharts>
#include "group_connectometry_analysis.h"
#include "atlas.hpp"
namespace Ui {
class group_connectometry;
}


class ROIViewDelegate : public QItemDelegate
 {
     Q_OBJECT

 public:
    ROIViewDelegate(QObject *parent)
         : QItemDelegate(parent)
     {
     }

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                                const QModelIndex &index) const;
          void setEditorData(QWidget *editor, const QModelIndex &index) const;
          void setModelData(QWidget *editor, QAbstractItemModel *model,
                            const QModelIndex &index) const;
private slots:
    void emitCommitData();
 };

class tracking_window;
class fib_data;
enum class command_source;
class group_connectometry : public QDialog
{
    Q_OBJECT
private:
    QChart* null_pos_chart;
    QChart* null_neg_chart;
    QChartView* null_pos_chart_view;
    QChartView* null_neg_chart_view;
    QChart* fdr_chart;
    QChartView* fdr_chart_view;
private:
    std::shared_ptr<connectometry_result> result_fib;
    void show_dis_table(void);
public:
    QString db_file_name,work_dir;
public:
    std::vector<std::vector<tipl::vector<3,short> > > roi_list;
    void add_new_roi(QString name,QString source,const std::vector<tipl::vector<3,short> >& new_roi,int type = 0);

public:
    std::shared_ptr<group_connectometry_analysis> vbc;
    connectometry_db& db;
    std::shared_ptr<stat_model> model;
    std::shared_ptr<QTimer> timer;
    bool run_started = false; // false only before the very first "run" on this window; never reset by Stop
    bool run_completed = false; // true only once a run reaches 100% naturally; false again once a new run starts
    bool suppress_run_dialogs = false; // set from the starting command's source; true unless a local user clicked Run
    size_t selected_count = 0;
    explicit group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_ptr,QString db_file_name_);
    ~group_connectometry();

public:
    void load_demographics(void);
    std::string error_msg;
    bool command(std::vector<std::string> cmd,command_source source);
private:
    // makes the variable_list checkboxes and foi's dropdown match whichever features are currently
    // db.feature[i].selected; on_variable_list_clicked already wrote that state from the checkboxes
    // just before calling this, so re-applying it to the checkboxes here is a harmless no-op there
    void sync_variable_list(void);
public slots:

    void show_report();

    void show_fdr_report();

    void on_roi_whole_brain_toggled(bool checked);

public slots:
    void calculate_FDR(void);
    void on_variable_list_clicked(const QModelIndex &index);
public:
    Ui::group_connectometry *ui;
private slots:
    // shared by every button whose click just forwards its own name as a command()
    // (open_mr_files/run/show_result/load_roi_from_atlas/clear_all_roi/load_roi_from_file/show_cohort/apply_selection);
    // wired up explicitly in the constructor rather than via connectSlotsByName()'s on_<name>_<signal>
    // convention, since one shared slot can't match every button's individual name; deliberately
    // not named on_..._clicked, since connectSlotsByName() would otherwise still try (and fail) to
    // parse it as on_<widget:button_command>_clicked and warn on every construction
    void forward_button_command();
    void on_fdr_control_toggled(bool checked);
    void on_effect_size_valueChanged(double arg1);
    void on_threshold_valueChanged(double arg1);
    void on_index_name_currentIndexChanged(int index);
};

#endif // VBC_DIALOG_HPP
