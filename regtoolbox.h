#ifndef REGTOOLBOX_H
#define REGTOOLBOX_H

#include <QMainWindow>
#include <QTimer>
#include <QGraphicsScene>
#include <QProgressBar>
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include "reg.hpp"
namespace Ui {
class RegToolBox;
}


class RegToolBox : public QMainWindow
{
    Q_OBJECT

public:
    uint8_t cur_view = 2;
    dual_reg reg;
    std::vector<tipl::image<3,unsigned char> > J[2];
    std::vector<tipl::vector<3,int> > anchor[2];
public:
    tipl::transformation_matrix<float> T;
public:
    tipl::thread thread;
    std::shared_ptr<QTimer> timer;
    tipl::affine_transform<float> old_arg;
public:
    std::shared_ptr<tipl::reg::bfnorm_mapping<float,3> > bnorm_data;
    bool flash = false;
    void clear_thread(void);
private:
    void setup_slice_pos(void);
    uint8_t blend_style(void);
private:
    std::vector<std::string> file_names[2];
    void auto_fill(void);
    void load_subject(const std::string& file_name);
    void load_template(const std::string& file_name);
public:
    explicit RegToolBox(QWidget *parent = nullptr);
    ~RegToolBox();
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
public:
    int view_border[2] = {0,0};
    tipl::shape<2> view_size[2];
    bool eventFilter(QObject *obj, QEvent *event) override;
public slots:

    void show_image();
private slots:

    void on_OpenTemplate_clicked();

    void on_OpenSubject_clicked();


    void on_run_reg_clicked();
    void on_timer();

    void on_stop_clicked();

    void on_actionSave_Warping_triggered();

    void on_show_option_clicked();

    void on_axial_view_clicked();

    void on_coronal_view_clicked();

    void on_sag_view_clicked();

    void on_actionSubject_Image_triggered();

    void on_actionTemplate_Image_triggered();


    void on_ClearSubject_clicked();

    void on_ClearTemplate_clicked();

    void on_actionApply_Subject_To_Template_Warping_triggered();

    void on_actionApply_Template_To_Subject_Warping_triggered();

    void on_actionOpen_Mapping_triggered();

    void on_actionSet_Template_Size_triggered();

    void on_actionSave_Subject_Images_triggered();

    void on_actionSave_Template_Images_triggered();

    void on_anchor_toggled(bool checked);

    void on_rb_switch_clicked();

private:
    Ui::RegToolBox *ui;
    QGraphicsScene scene[2];
};

#endif // REGTOOLBOX_H
