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
    image::color_bar bar;
    QGraphicsScene color_bar;
public:
    tracking_window* cur_tracking_window;
    Ui::color_bar_dialog *ui;
    explicit color_bar_dialog(QWidget *parent = 0);
    ~color_bar_dialog();    
public:
    float get_color_max_value(void) const;
    float get_color_min_value(void) const;
    unsigned int get_tract_color_index(void) const;

public slots:
    void update_color_map(void);
    void on_tract_color_index_currentIndexChanged(int index);
};

#endif // COLOR_BAR_DIALOG_HPP
