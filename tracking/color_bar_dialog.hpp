#ifndef COLOR_BAR_DIALOG_HPP
#define COLOR_BAR_DIALOG_HPP
#include "image/image.hpp"
#include <QDialog>
#include <QGraphicsScene>

namespace Ui {
class color_bar_dialog;
}

class tracking_window;

class color_bar_dialog : public QDialog
{
    Q_OBJECT
public:// color_bar
    image::color_map color_map;
    image::color_map_rgb color_map_rgb;
    image::color_bar bar;
    QGraphicsScene color_bar;
    float color_r,color_min;
public:
    tracking_window* cur_tracking_window;
    Ui::color_bar_dialog *ui;
    explicit color_bar_dialog(QWidget *parent = 0);
    ~color_bar_dialog();    
public:
    const image::vector<3,float>& get_color(float value) const
    {
        return color_map[std::floor(std::min(1.0f,(std::max<float>(value-color_min,0.0))/color_r)*255.0+0.49)];
    }
    const image::rgb_color& get_rgb(float value) const
    {
        return color_map_rgb[std::floor(std::min(1.0f,(std::max<float>(value-color_min,0.0))/color_r)*255.0+0.49)];
    }
    QString get_tract_color_name(void) const;
    void set_value(float min_value,float max_value);
public slots:
    void update_color_map(void);
    void on_tract_color_index_currentIndexChanged(int index);
};

#endif // COLOR_BAR_DIALOG_HPP
