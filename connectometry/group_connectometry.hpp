#ifndef GROUP_CNT_HPP
#define GROUP_CNT_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QItemDelegate>
#include <QTimer>
#include <QtCharts/QtCharts>
#include "tipl/tipl.hpp"
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
    QChart* null_chart;
    QChart* fdr_chart;
    QChartView* null_chart_view;
    QChartView* fdr_chart_view;
private:
    std::auto_ptr<connectometry_result> result_fib;
    void show_dis_table(void);
public:
    bool gui = true;
    QString db_file_name,work_dir;
    std::string demo_file_name;
public:
    std::vector<std::vector<tipl::vector<3,short> > > roi_list;
    void add_new_roi(QString name,QString source,const std::vector<tipl::vector<3,short> >& new_roi,int type = 0);

public:
    std::shared_ptr<group_connectometry_analysis> vbc;
    std::auto_ptr<stat_model> model;
    std::auto_ptr<QTimer> timer;
    bool setup_model(stat_model& model);

    explicit group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_ptr,QString db_file_name_,bool gui_);
    ~group_connectometry();

public:
    bool load_demographic_file(QString filename,std::string& error_msg);
public slots:

    void show_report();

    void show_fdr_report();

    void on_open_mr_files_clicked();

    void on_run_clicked();

    void on_show_result_clicked();

    void on_roi_whole_brain_toggled(bool checked);

    void on_show_advanced_clicked();

    void on_missing_data_checked_toggled(bool checked);

public slots:
    void calculate_FDR(void);
    void on_variable_list_clicked(const QModelIndex &index);
public:
    Ui::group_connectometry *ui;
private slots:
    void on_load_roi_from_atlas_clicked();
    void on_clear_all_roi_clicked();
    void on_load_roi_from_file_clicked();
    void on_rb_longitudina_dif_clicked();
    void on_rb_regression_clicked();
};

#endif // VBC_DIALOG_HPP
