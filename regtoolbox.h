#ifndef REGTOOLBOX_H
#define REGTOOLBOX_H

#include <QMainWindow>
#include <QTimer>
#include <QGraphicsScene>
#include <QProgressBar>
#include "zlib.h"
#include "TIPL/tipl.hpp"
namespace Ui {
class RegToolBox;
}


class RegToolBox : public QMainWindow
{
    Q_OBJECT

public:
    uint8_t cur_view = 2;
    tipl::image<3> It,I,J,JJ,I2,It2,J2;
    tipl::image<3,tipl::vector<3> > t2f_dis,to2from,f2t_dis,from2to;
    tipl::vector<3> Itvs,Ivs;
    tipl::matrix<4,4> ItR,IR;
    bool It_is_mni;
public:
    tipl::transformation_matrix<float> T;
    tipl::value_to_color<float> v2c_I,v2c_It;
public:
    tipl::affine_transform<float> arg,old_arg;
    tipl::thread thread;
    std::shared_ptr<QTimer> timer;
    std::string status;
public:
    std::shared_ptr<tipl::reg::bfnorm_mapping<float,3> > bnorm_data;
    bool reg_done;
    bool flash = false;
private:
    void clear(void);
    void setup_slice_pos(void);
    void linear_reg(tipl::reg::reg_type reg_type,int cost_type);
    void nonlinear_reg(void);

public:
    explicit RegToolBox(QWidget *parent = nullptr);
    ~RegToolBox();

private slots:

    void change_contrast();
    void on_OpenTemplate_clicked();

    void on_OpenSubject_clicked();


    void on_run_reg_clicked();
    void on_timer();

    void show_image();



    void on_stop_clicked();

    void on_actionMatch_Intensity_triggered();

    void on_actionRemove_Background_triggered();

    void on_OpenSubject2_clicked();

    void on_OpenTemplate2_clicked();

    void on_actionApply_Warping_triggered();

    void on_actionSave_Warping_triggered();

    void on_show_option_clicked();

    void on_axial_view_clicked();

    void on_coronal_view_clicked();

    void on_sag_view_clicked();

    void on_actionSmooth_Subject_triggered();

    void on_actionSave_Transformed_Image_triggered();

    void on_switch_view_clicked();

private:
    Ui::RegToolBox *ui;
    QGraphicsScene It_scene,I_scene,It_mix_scene;
};

#endif // REGTOOLBOX_H
