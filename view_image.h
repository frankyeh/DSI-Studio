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

private:
    Ui::view_image *ui;
    tipl::image<float,3> data;
    tipl::vector<3,float> vs;
    tipl::matrix<4,4,float> T;
private:
    QGraphicsScene source;
    tipl::color_image buffer;
    QImage source_image;
    float max_source_value,source_ratio;

};

#endif // VIEW_IMAGE_H
