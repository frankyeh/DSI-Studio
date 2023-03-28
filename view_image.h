#ifndef VIEW_IMAGE_H
#define VIEW_IMAGE_H

#include <QMainWindow>
#include <QGraphicsScene>
#include "zlib.h"
#include "TIPL/tipl.hpp"
namespace Ui {
class view_image;
}
class QTableWidget;
class TableKeyEventWatcher : public QObject {
    Q_OBJECT
    using QObject::QObject;
private:
    QTableWidget* table;
    bool eventFilter(QObject * receiver, QEvent * event) override;
public:
    TableKeyEventWatcher(QTableWidget* table);
    Q_SIGNAL void DeleteRowPressed(int row);
};


class view_image : public QMainWindow
{
    Q_OBJECT
    std::shared_ptr<TableKeyEventWatcher> table_event;
public:
    QString file_name,original_file_name;
    QStringList file_names;
    tipl::io::gz_nifti nifti;
    explicit view_image(QWidget *parent = nullptr);
    ~view_image();
    bool open(QStringList file_name);
    bool eventFilter(QObject *obj, QEvent *event);
public:
    std::vector<std::string> command_list;
    std::vector<std::string> param_list;
    bool command(std::string cmd,std::string param1 = std::string());
private:
    void update_other_images(void);
    bool has_flip_x(void);
    bool has_flip_y(void);
private slots:
    void DeleteRowPressed(int row);
    void show_info(QString info);
    void show_image(bool update_others);
    void init_image(void);
    void update_overlay_menu(void);
    void set_overlay(void);

    void on_action_Save_as_triggered();

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

    void on_dwi_volume_valueChanged(int value);

    void run_action();

    void run_action2();

    void on_type_currentIndexChanged(int index);

    void on_zoom_valueChanged(double arg1);

    void on_info_cellChanged(int row, int column);

    void on_info_cellDoubleClicked(int row, int column);

    void on_mat_images_currentIndexChanged(int index);

    void on_actionLoad_Image_to_4D_triggered();

private:
    Ui::view_image *ui;
private:
    tipl::image<3,unsigned char,tipl::buffer_container> I_uint8;
    tipl::image<3,unsigned short,tipl::buffer_container> I_uint16;
    tipl::image<3,unsigned int,tipl::buffer_container> I_uint32;
    tipl::image<3,float,tipl::buffer_container> I_float32;
    enum {uint8 = 0,uint16 = 1,uint32 = 2,float32 = 3} pixel_type = uint8;
    tipl::shape<3> shape;
    bool is_mni = false;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4> T;
    template <typename T>
    void apply(T&& fun)
    {
        switch(pixel_type)
        {
            case uint8:fun(I_uint8);return;
            case uint16:fun(I_uint16);return;
            case uint32:fun(I_uint32);return;
            case float32:fun(I_float32);return;
        }
    }
    void change_type(decltype(pixel_type));

private:
    tipl::io::gz_mat_read mat;
    void read_mat_info(void);
    bool read_mat_image(void);
    void write_mat_image(void);
    bool read_mat(void);
private: //overlay
    std::vector<size_t> overlay_images;
    std::vector<bool> overlay_images_visible;
    size_t this_index = 0;
private:
    std::vector<std::vector<unsigned char> > buf4d;
    size_t cur_4d_index = 0;
    void read_4d_at(size_t index);
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
    std::string error_msg;
};

#endif // VIEW_IMAGE_H
