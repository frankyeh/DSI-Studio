#ifndef REGTOOLBOX_H
#define REGTOOLBOX_H

#include <QMainWindow>
#include <QTimer>
#include <QGraphicsScene>
#include <QProgressBar>
#include "image/image.hpp"
namespace Ui {
class RegToolBox;
}


class RegToolBox : public QMainWindow
{
    Q_OBJECT

public:
    image::basic_image<float,3> It,I,J,JJ;
    image::basic_image<image::vector<3>,3> dis;
    image::vector<3> Itvs,Ivs;
    float ItR[12];
public:
    image::basic_image<float,3> J_view,J_view2;
    image::basic_image<image::vector<3>,3> dis_view;
public:
    image::affine_transform<double> arg;
    image::thread thread;
    std::shared_ptr<QTimer> timer;
    std::string status;
public:
    std::shared_ptr<image::reg::bfnorm_mapping<float,3> > bnorm_data;

    int reg_type;
    bool reg_done;
private:
    void clear(void);
    void linear_reg(image::reg::reg_type reg_type);
    void nonlinear_reg(int method);

public:
    explicit RegToolBox(QWidget *parent = 0);
    ~RegToolBox();

private slots:
    void on_OpenTemplate_clicked();

    void on_OpenSubject_clicked();


    void on_run_reg_clicked();
    void on_timer();

    void show_image();
    void on_action_Save_Warpped_Image_triggered();


    void on_reg_type_currentIndexChanged(int index);

    void on_stop_clicked();

    void on_reg_method_currentIndexChanged(int index);

    void on_actionRemove_Skull_triggered();

private:
    Ui::RegToolBox *ui;
    QGraphicsScene It_scene,I_scene;
private:
    image::color_image cIt,cI,cJ;

};

#endif // REGTOOLBOX_H
