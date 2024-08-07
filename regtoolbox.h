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
    uint8_t subject_cur_view = 2,template_cur_view = 2;
    dual_reg<2> reg_2d;
    dual_reg<3> reg;
public:
    tipl::transformation_matrix<float> T;
    tipl::value_to_color<float> v2c_I,v2c_It;
public:
    tipl::thread thread;
    std::shared_ptr<QTimer> timer;
    tipl::affine_transform<float> old_arg;
public:
    std::shared_ptr<tipl::reg::bfnorm_mapping<float,3> > bnorm_data;
    bool flash = false;
    void clear(void);
private:
    void setup_slice_pos(bool subject = true);
    uint8_t blend_style(void);
private:
    std::string template2_name,subject2_name;
    void load_subject2(const std::string& file_name);
    void load_template2(const std::string& file_name);
    void load_template(const std::string& file_name);
public:
    explicit RegToolBox(QWidget *parent = nullptr);
    ~RegToolBox();
public slots:
    void show_image();
private slots:

    void change_contrast();
    void on_OpenTemplate_clicked();

    void on_OpenSubject_clicked();


    void on_run_reg_clicked();
    void on_timer();




    void on_stop_clicked();

    void on_OpenSubject2_clicked();

    void on_OpenTemplate2_clicked();

    void on_actionApply_Warping_triggered();

    void on_actionSave_Warping_triggered();

    void on_show_option_clicked();

    void on_axial_view_clicked();

    void on_coronal_view_clicked();

    void on_sag_view_clicked();

    void on_switch_view_clicked();

    void on_actionDual_Modality_triggered();


    void on_actionSubject_Image_triggered();

    void on_actionTemplate_Image_triggered();

    void on_sag_view_2_clicked();

    void on_coronal_view_2_clicked();

    void on_axial_view_2_clicked();

private:
    Ui::RegToolBox *ui;
    QGraphicsScene It_scene,I_scene;
};

#endif // REGTOOLBOX_H
