#ifndef VIEW_IMAGE_H
#define VIEW_IMAGE_H

#include <QDialog>
#include <QGraphicsScene>
#include <image/image.hpp>

namespace Ui {
class view_image;
}

class view_image : public QDialog
{
    Q_OBJECT
    
public:
    explicit view_image(QWidget *parent = 0);
    ~view_image();
    bool open(QString file_name);
    bool eventFilter(QObject *obj, QEvent *event);
private slots:
    void update_image(void);
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();

private:
    Ui::view_image *ui;
    image::basic_image<float,3> data;
private:
    QGraphicsScene source;
    image::color_image buffer;
    QImage source_image;
    float max_source_value,source_ratio;

};

#endif // VIEW_IMAGE_H
