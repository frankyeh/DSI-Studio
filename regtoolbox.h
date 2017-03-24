#ifndef REGTOOLBOX_H
#define REGTOOLBOX_H

#include <QMainWindow>
#include <QTimer>
#include <QGraphicsScene>
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
    image::affine_transform<double> linear_reg;
    image::thread thread;
    std::shared_ptr<QTimer> timer;
    int running_type;
    bool reg_done,linear_done;
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

private:
    Ui::RegToolBox *ui;
    QGraphicsScene It_scene,I_scene,main_scene;
private:
    image::color_image cIt,cI,cJ;

};

#endif // REGTOOLBOX_H
