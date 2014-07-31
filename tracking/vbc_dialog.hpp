#ifndef VBC_DIALOG_HPP
#define VBC_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include "image/image.hpp"
#include "vbc/vbc_database.h"
namespace Ui {
class vbc_dialog;
}

class tracking_window;
class ODFModel;
class vbc_dialog : public QDialog
{
    Q_OBJECT
private:
    QGraphicsScene vbc_scene;
    QImage vbc_slice_image;
    unsigned int vbc_slice_pos;
    fib_data cur_subject_fib;
    std::vector<std::vector<unsigned int> > dist;
    std::vector<std::vector<float> > fdr;


    void show_dis_table(void);
    void show_fdr_table(void);
private:
    QStringList filename;
public:
    QString work_dir;
    std::auto_ptr<vbc_database> vbc;
    stat_model mr;
    void calculate_FDR(void);
    explicit vbc_dialog(QWidget *parent,vbc_database* vbc_ptr,QString work_dir_);
    ~vbc_dialog();
    bool eventFilter(QObject *obj, QEvent *event);
private slots:


    void on_subject_list_itemSelectionChanged();

    void on_save_vbc_dist_clicked();

    void show_report();

    void show_fdr_report();

    void on_save_fdr_dist_clicked();

    void on_open_files_clicked();

    void on_FDR_analysis_clicked();

    void on_buttonBox_accepted();

    void on_tabWidget_currentChanged(int index);

    void on_open_mr_files_clicked();

    void on_run_mr_analysis_clicked();

    void on_view_mr_result_clicked();

private:
    Ui::vbc_dialog *ui;
};

#endif // VBC_DIALOG_HPP
