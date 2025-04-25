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
    size_t selected_count = 0;
    explicit group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_ptr,QString db_file_name_);
    ~group_connectometry();

public:
    void load_demographics(void);
public slots:

    void show_report();

    void show_fdr_report();

    void on_open_mr_files_clicked();

    void on_run_clicked();

    void on_show_result_clicked();

    void on_roi_whole_brain_toggled(bool checked);

public slots:
    void calculate_FDR(void);
    void on_variable_list_clicked(const QModelIndex &index);
public:
    Ui::group_connectometry *ui;
private slots:
    void on_load_roi_from_atlas_clicked();
    void on_clear_all_roi_clicked();
    void on_load_roi_from_file_clicked();
    void on_show_cohort_clicked();
    void on_fdr_control_toggled(bool checked);
    void on_apply_selection_clicked();
    void on_threshold_valueChanged(double arg1);
};

#endif // VBC_DIALOG_HPP
