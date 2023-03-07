#ifndef COLOR_BAR_DIALOG_HPP
#define COLOR_BAR_DIALOG_HPP
#include <QDialog>
#include <QGraphicsScene>
#include "zlib.h"
#include "TIPL/tipl.hpp"

namespace Ui {
class color_bar_dialog;
}

class tracking_window;

class color_bar_dialog : public QDialog
{
    Q_OBJECT
public:// color_bar
    tipl::color_map color_map;
    tipl::color_map_rgb color_map_rgb;
    tipl::color_bar bar;
    QGraphicsScene color_bar;
    double color_r,color_min;
public:
    tracking_window* cur_tracking_window;
    Ui::color_bar_dialog *ui;
    explicit color_bar_dialog(QWidget *parent = nullptr);
    ~color_bar_dialog();    
public:
    const tipl::vector<3,float>& get_color(double value) const
    {
        return color_map[uint32_t(std::floor(std::min(1.0,(std::max<double>(value-color_min,0.0))/color_r)*255.0+0.49))];
    }
    const tipl::rgb& get_rgb(double value) const
    {
        return color_map_rgb[uint32_t(std::floor(std::min(1.0,(std::max<double>(value-color_min,0.0))/color_r)*255.0+0.49))];
    }
    QString get_tract_color_name(void) const;
    void set_value(double min_value,double max_value);
public slots:
    void update_color_map(void);
    void update_slice_indices(void);
    void on_tract_color_index_currentIndexChanged(int index);
};

#endif // COLOR_BAR_DIALOG_HPP
