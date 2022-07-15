#ifndef VIEW_IMAGE_H
#define VIEW_IMAGE_H

#include <QMainWindow>
#include <QGraphicsScene>
#include "TIPL/tipl.hpp"
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
    bool command(std::string cmd,std::string param1 = std::string());
private:
    bool has_flip_x(void);
    bool has_flip_y(void);
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

    void on_actionSet_Transformation_triggered();

    void on_actionSave_as_Int8_triggered();

    void on_actionSave_as_Int16_triggered();

    void change_contrast();

    void on_min_slider_sliderMoved(int position);

    void on_min_valueChanged(double arg1);

    void on_max_slider_sliderMoved(int position);

    void on_max_valueChanged(double arg1);

    void on_AxiView_clicked();

    void on_CorView_clicked();

    void on_SagView_clicked();

    void on_slice_pos_valueChanged(int value);

    void on_actionSave_triggered();

    void on_actionImageAddition_triggered();

    void on_actionImageMultiplication_triggered();

    void on_dwi_volume_valueChanged(int value);

    void on_actionMinus_Image_triggered();

    void run_action();

    void run_action2();

private:
    Ui::view_image *ui;
    tipl::image<3> data,overlay;
    bool is_mni = false;
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
    std::vector<bool> other_is_mni;
private:
    bool no_update = true;
    unsigned char cur_dim = 2;
    int slice_pos[3];
    QGraphicsScene source;
    tipl::color_image buffer;
    QImage source_image;
    float source_ratio;
    std::string error_msg;
};

#endif // VIEW_IMAGE_H
