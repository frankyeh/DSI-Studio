#ifndef VIEW_IMAGE_H
#define VIEW_IMAGE_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <tipl/tipl.hpp>

namespace Ui {
class view_image;
}

class view_image : public QMainWindow
{
    Q_OBJECT
    
public:
    QString file_name;
    explicit view_image(QWidget *parent = nullptr);
    ~view_image();
    bool open(QStringList file_name);
    bool eventFilter(QObject *obj, QEvent *event);
private slots:
    void update_image(void);
    void init_image(void);
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();

    void on_actionResample_triggered();

    void on_action_Save_as_triggered();

    void on_actionMasking_triggered();

    void on_actionResize_triggered();

    void on_actionTranslocate_triggered();

    void on_actionTrim_triggered();

    void on_actionSet_Translocation_triggered();

    void on_actionLower_threshold_triggered();

    void on_actionLPS_RAS_swap_triggered();

    void on_actionSet_Transformation_triggered();

    void on_actionIntensity_shift_triggered();

    void on_actionIntensity_scale_triggered();

    void on_actionSave_as_Int8_triggered();

    void on_actionSave_as_Int16_triggered();

    void on_actionUpper_Threshold_triggered();

    void on_actionSmoothing_triggered();

    void on_actionTo_Edge_triggered();

    void on_actionNormalize_Intensity_triggered();

    void change_contrast();
    void on_min_slider_sliderMoved(int position);

    void on_min_valueChanged(double arg1);

    void on_max_slider_sliderMoved(int position);

    void on_max_valueChanged(double arg1);

    void on_AxiView_clicked();

    void on_CorView_clicked();

    void on_SagView_clicked();

    void on_slice_pos_valueChanged(int value);

private:
    Ui::view_image *ui;
    tipl::image<float,3> data;
    float min_value,max_value;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4,float> T;
    tipl::value_to_color<float> v2c;
private:
    bool no_update = true;
    unsigned char cur_dim = 2;
    int slice_pos[3];
    QGraphicsScene source;
    tipl::color_image buffer;
    QImage source_image;
    float max_source_value,source_ratio;

};

#endif // VIEW_IMAGE_H
