#ifndef GROUP_CNT_HPP
#define GROUP_CNT_HPP
#include <QDialog>
#include <QGraphicsScene>
#include <QItemDelegate>
#include <QTimer>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
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
    std::auto_ptr<connectometry_result> result_fib;
    void show_dis_table(void);
    void add_new_roi(QString name,QString source,std::vector<image::vector<3,short> >& new_roi);
public:
    bool gui;
    QString work_dir;
    std::vector<std::string> file_names;
public:
    std::vector<std::vector<image::vector<3,short> > > roi_list;
public:
    std::shared_ptr<vbc_database> vbc;
    std::auto_ptr<stat_model> model;
    std::vector<std::vector<float> > individual_data;
    std::auto_ptr<QTimer> timer;
    QString report;
    bool setup_model(stat_model& model);

    explicit group_connectometry(QWidget *parent,std::shared_ptr<vbc_database> vbc_ptr,QString db_file_name_,bool gui_);
    ~group_connectometry();

public:
    bool load_demographic_file(QString filename);
public slots:

    void show_report();

    void show_fdr_report();

    void on_open_mr_files_clicked();

    void on_run_clicked();

    void on_show_result_clicked();

    void on_roi_whole_brain_toggled(bool checked);

    void on_show_advanced_clicked();

    void on_missing_data_checked_toggled(bool checked);


    void on_suggest_threshold_clicked();
public slots:
    void calculate_FDR(void);
public:
    Ui::group_connectometry *ui;
private slots:
    void on_load_roi_from_atlas_clicked();
    void on_clear_all_roi_clicked();
    void on_load_roi_from_file_clicked();
    void on_rb_percentage_clicked();
    void on_rb_t_stat_clicked();
    void on_rb_beta_clicked();
    void on_variable_list_clicked(const QModelIndex &index);
};

#endif // VBC_DIALOG_HPP
