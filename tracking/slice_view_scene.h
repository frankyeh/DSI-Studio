#ifndef SLICE_VIEW_SCENE_H
#define SLICE_VIEW_SCENE_H
#include <QGraphicsScene>
#include <vector>
#include <QStatusBar>
#include <QEvent>
#include "zlib.h"
#include "TIPL/tipl.hpp"

class fib_data;
class tracking_window;
class SliceModel;
class slice_view_scene : public QGraphicsScene
{
    Q_OBJECT
public:
    bool no_show = true;
    bool free_thread = false;
    slice_view_scene(tracking_window& cur_tracking_window_):cur_tracking_window(cur_tracking_window_),paint_image_thread([&](void){paint_image();}){}
    ~slice_view_scene(void)
    {
        free_thread = true;
        if(paint_image_thread.joinable())
            paint_image_thread.join();
    }
    unsigned char sel_mode = 0;
    QStatusBar* statusbar = nullptr;
private:
    tracking_window& cur_tracking_window;
    QColor line_color = Qt::white;
public:
    unsigned int mosaic_column_count,mosaic_row_count;
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
    int cur_region = -1;
    bool mouse_down = false;
    bool mid_down = false;
    bool move_slice = false;
    bool move_viewing_slice = false;
    int cX, cY;
public:
    std::thread paint_image_thread;

    QImage view_image,complete_view_image,annotated_image;
    bool need_complete_view = false;
    bool complete_view_ready = false;
    int view1_h = 0; // for 3 views updating 3D
    int view1_w = 0; // for 3 views updating 3D
    void paint_image(QImage& I,bool simple);
    void paint_image(void);
public:
    bool show_grid = false;
    void new_annotated_image(void);
    void show_ruler2(QPainter& painter);
    void show_ruler(QPainter& painter,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim);
    void show_pos(QPainter& painter,std::shared_ptr<SliceModel> current_slice,const tipl::color_image& slice_image,unsigned char cur_dim);
    void show_fiber(QPainter& painter,std::shared_ptr<SliceModel> current_slice,const tipl::color_image& slice_image,unsigned char cur_dim);
    void get_view_image(QImage& new_view_image,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim,float display_ratio,bool simple);
    void add_R_label(QPainter& painter,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim);
    void manage_slice_orientation(QImage& slice,QImage& new_slice,unsigned char cur_dim);
    bool command(QString cmd,QString param = "",QString param2 = "");
    void update_3d(QImage captured);
    // update cursor info
protected:
    void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mouseMoveEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void mouseReleaseEvent ( QGraphicsSceneMouseEvent * mouseEvent );
    void wheelEvent(QGraphicsSceneWheelEvent *wheelEvent);
public slots:
    void show_slice();
    void show_complete_slice();
    void catch_screen();
    void copyClipBoard();
    void center();
    void save_slice_as();
signals:
    void need_update();

};

#endif // SLICE_VIEW_SCENE_H
