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
    void update_other_images(void);
    bool has_flip_x(void);
    bool has_flip_y(void);
private slots:
    void show_image(bool update_others);
    void init_image(void);
    void update_overlay_menu(void);
    void set_overlay(void);
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();

    void on_actionResample_triggered();

    void on_action_Save_as_triggered();

    void on_actionResize_triggered();

    void on_actionTranslocate_triggered();

    void on_actionTrim_triggered();

    void on_actionSet_Translocation_triggered();

    void on_actionSet_Transformation_triggered();

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

    void on_type_currentIndexChanged(int index);

private:
    Ui::view_image *ui;
private:
    tipl::image<3,unsigned char,tipl::buffer_container> I_uint8;
    tipl::image<3,unsigned short,tipl::buffer_container> I_uint16;
    tipl::image<3,unsigned int,tipl::buffer_container> I_uint32;
    tipl::image<3,float,tipl::buffer_container> I_float32;
    enum {uint8 = 0,uint16 = 1,uint32 = 2,float32 = 3} data_type = uint8;
    tipl::shape<3> shape;
    bool is_mni = false;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4> T;
    template <typename T>
    void apply(T&& fun)
    {
        switch(data_type)
        {
            case uint8:fun(I_uint8);return;
            case uint16:fun(I_uint16);return;
            case uint32:fun(I_uint32);return;
            case float32:fun(I_float32);return;
        }
    }
private: //overlay
    std::vector<size_t> overlay_images;
    std::vector<bool> overlay_images_visible;
    size_t this_index = 0;
private:
    std::vector<std::vector<unsigned char> > dwi_volume_buf;
    size_t cur_dwi_volume = 0;
private:// batch processing
    /*
    std::vector<tipl::image<3> > other_data;
    std::vector<std::string> other_file_name;
    std::vector<tipl::vector<3,float> > other_vs;
    std::vector<tipl::matrix<4,4> > other_T;
    std::vector<bool> other_is_mni;
    */
private: // visualization
    bool no_update = true;
    tipl::value_to_color<float> v2c;
    unsigned char cur_dim = 2;
    int slice_pos[3];
    QGraphicsScene source;
    QImage source_image;
    float source_ratio;
    std::string error_msg;
};

#endif // VIEW_IMAGE_H
