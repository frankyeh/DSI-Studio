#ifndef VIEW_IMAGE_H
#define VIEW_IMAGE_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <tipl/tipl.hpp>
#include "libs/gzip_interface.hpp"
namespace Ui {
class view_image;
}

class view_image : public QMainWindow
{
    Q_OBJECT
    
public:
    QString file_name;
    gz_nifti nifti;
    explicit view_image(QWidget *parent = nullptr);
    ~view_image();
    bool open(QStringList file_name);
    bool eventFilter(QObject *obj, QEvent *event);
    bool command(std::string cmd,std::string param1 = std::string(),std::string param2 = std::string());
private slots:
    void show_image(void);
    void init_image(void);
    void update_overlay_menu(void);
    void add_overlay(void);
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();

    void on_actionResample_triggered();

    void on_action_Save_as_triggered();

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

    void on_actionSobel_triggered();

    void on_actionMorphology_triggered();

    void on_actionMorphology_Thin_triggered();

    void on_actionMorphology_XY_triggered();

    void on_actionMorphology_XZ_triggered();

    void on_actionSave_triggered();

    void on_actionImageAddition_triggered();

    void on_actionImageMultiplication_triggered();

    void on_actionSignal_Smoothing_triggered();

    void on_dwi_volume_valueChanged(int value);

    void on_actionDownsample_by_2_triggered();

    void on_actionUpsample_by_2_triggered();

private:
    Ui::view_image *ui;
    tipl::image<3> data,overlay;
    float min_value,max_value;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4> T;
    tipl::value_to_color<float> v2c,overlay_v2c;
private:
    std::vector<tipl::image<3> > dwi_volume_buf;
    size_t cur_dwi_volume = 0;
private:// batch processing
    std::vector<tipl::image<3> > other_data;
    std::vector<std::string> other_file_name;
    std::vector<tipl::vector<3,float> > other_vs;
    std::vector<tipl::matrix<4,4> > other_T;
private:
    bool no_update = true;
    unsigned char cur_dim = 2;
    int slice_pos[3];
    QGraphicsScene source;
    tipl::color_image buffer;
    QImage source_image;
    float source_ratio;

};

#endif // VIEW_IMAGE_H
