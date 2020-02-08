#ifndef SLICE_VIEW_SCENE_H
#define SLICE_VIEW_SCENE_H
#include <QGraphicsScene>
#include <vector>
#include "tipl/tipl.hpp"
#include <QStatusBar>
#include <QEvent>
class fib_data;
class tracking_window;
class slice_view_scene : public QGraphicsScene
{
    Q_OBJECT
public:
    bool no_show = false;
    slice_view_scene(tracking_window& cur_tracking_window_):
            sel_mode(0),
            mid_down(false),
            mouse_down(false),
            cur_tracking_window(cur_tracking_window_),
            statusbar(0)
    {

    }
    unsigned char sel_mode;
    QStatusBar* statusbar;
private:
    tracking_window& cur_tracking_window;
    tipl::color_image slice_image,mosaic_image;
public:
    unsigned int mosaic_size;
    void adjust_xy_to_layout(float& X,float& Y);
    bool to_3d_space_single_slice(float x,float y,tipl::vector<3,float>& pos);
    bool to_3d_space(float x,float y,tipl::vector<3,float>& pos);
private:
    bool clicked_3d = false;
    bool click_on_3D(float x,float y);
    void send_event_to_3D(QEvent::Type type,
                          QGraphicsSceneMouseEvent * mouseEvent);
public:    // record the mouse press points
    std::vector<tipl::vector<3,float> >sel_coord;
    std::vector<tipl::vector<2,short> >sel_point;
    int cur_region;
    bool mouse_down;
    bool mid_down;
    bool move_slice = false;
    bool move_viewing_slice = false;
    int cX, cY;

    QImage view_image,annotated_image;
    void new_annotated_image(void);
    void show_ruler2(QPainter& painter);
    void show_ruler(QPainter& painter);
    void show_pos(QPainter& painter);
    void show_fiber(QPainter& painter);
    void get_view_image(QImage& new_view_image);
    void add_R_label(QPainter& painter);
    bool command(QString cmd,QString param = "",QString param2 = "");
    // update cursor info
protected:
    void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mouseMoveEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mouseReleaseEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void wheelEvent(QGraphicsSceneWheelEvent *wheelEvent);
public slots:
    void show_slice();
    void catch_screen();
    void copyClipBoard();
    void center();
    void save_slice_as();
signals:
    void need_update();

};

#endif // SLICE_VIEW_SCENE_H
