#include "slice_view_scene.h"
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QSettings>
#include <QFileDialog>
#include <QClipboard>
#include <QMessageBox>
#include <QMouseEvent>
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "region/regiontablewidget.h"
#include "fib_data.hpp"
#include "opengl/glwidget.h"

void slice_view_scene::show_ruler2(QPainter& paint)
{
    float zoom = cur_tracking_window.get_scene_zoom();
    if(sel_mode == 6 && sel_point.size() >= 2)
    {
        QPen pen;  // creates a default pen
        pen.setWidth(2);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setColor(Qt::white);
        paint.setPen(pen);

        short tX = sel_point[0][0];
        short tY = sel_point[0][1];
        short X = sel_point[1][0];
        short Y = sel_point[1][1];
        paint.drawLine(X,Y,tX,tY);
        tipl::vector<2,float> from(X,Y);
        tipl::vector<2,float> to(tX,tY);
        from -= to;
        float pixel_length = float(from.length());
        from /= zoom;
        from[0] *= cur_tracking_window.current_slice->vs[0];
        from[1] *= cur_tracking_window.current_slice->vs[1];
        from[2] *= cur_tracking_window.current_slice->vs[2];
        float length = float(from.length());
        float precision = float(std::pow(10.0,std::floor(std::log10(double(length)))-1));
        float tic_dis = float(std::pow(10.0,std::floor(std::log10(50.0*double(length/pixel_length)))));
        if(tic_dis*pixel_length/length < 10)
            tic_dis *= 5.0f;
        tipl::vector<2,float> tic_dir(Y-tY,tX-X);
        tic_dir.normalize();
        tic_dir *= 5.0f;
        for(float L = 0.0;L < length;L+=tic_dis)
        {
            tipl::vector<2,float> npos(tX+(X-tX)*L/length,
                                       tY+(Y-tY)*L/length);
            paint.drawLine(int(npos[0]),int(npos[1]),
                           int(npos[0]+tic_dir[0]),
                           int(npos[1]+tic_dir[1]));
            npos += tic_dir*3;
            paint.drawText(int(npos[0])-40,int(npos[1])-40,80,80,
                               Qt::AlignHCenter|Qt::AlignVCenter,
                               QString::number(double(L)));
        }
        paint.drawText(X-40,Y-40,80,80,
                       Qt::AlignHCenter|Qt::AlignVCenter,
                       QString::number(std::round(double(length)*10.0/double(precision))*double(precision)/10.0)+" mm");

    }
}




void slice_view_scene::show_ruler(QPainter& paint,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim,float tic_ratio)
{
    auto trans = cur_tracking_window.handle->trans_to_mni;
    if(cur_tracking_window.handle->is_mni)
    {
        if(!current_slice->is_diffusion_space)
            trans *= current_slice->to_dif;
    }
    else
        trans.identity();
    QPen pen;
    pen.setColor(line_color);
    paint.setPen(pen);
    paint.setFont(font());

    tipl::qt::draw_ruler(paint,current_slice->dim,trans,cur_dim,
               cur_tracking_window.slice_view_flip_x(cur_dim),
               cur_tracking_window.slice_view_flip_y(cur_dim),
               cur_tracking_window.get_scene_zoom(current_slice),show_grid,tic_ratio);

}
void slice_view_scene::show_fiber(QPainter& painter,std::shared_ptr<SliceModel> current_slice,const tipl::color_image& slice_image,unsigned char cur_dim)
{
    if((cur_tracking_window["dt_index1"].toInt() != 0 || cur_tracking_window["dt_index2"].toInt() != 0 )
            && cur_tracking_window.handle->dir.dt_fa.empty())
            return;

    float display_ratio = cur_tracking_window.get_scene_zoom(current_slice);
    float r = display_ratio * cur_tracking_window["roi_fiber_length"].toFloat();
    float pen_w = display_ratio * cur_tracking_window["roi_fiber_width"].toFloat();
    int steps = 1;
    if(!current_slice->is_diffusion_space)
    {
        steps = int(std::ceil(cur_tracking_window.handle->vs[0]/current_slice->vs[0]));
        r *= steps;
        pen_w *= steps;
    }
    if(r < 1.0f)
        return;

    auto dim = cur_tracking_window.handle->dim;
    auto& dir = cur_tracking_window.handle->dir;

    int roi_fiber = cur_tracking_window["roi_fiber"].toInt();
    float threshold = cur_tracking_window.get_fa_threshold();
    float threshold2 = cur_tracking_window.handle->dir.dt_fa.empty() ?
                        0.0f:cur_tracking_window["dt_threshold"].toFloat();

    if (threshold == 0.0f)
        threshold = 0.00000001f;
    unsigned char dir_x[3] = {1,0,0};
    unsigned char dir_y[3] = {2,2,1};

    if(cur_tracking_window["roi_fiber_antialiasing"].toInt())
        painter.setRenderHint(QPainter::Antialiasing, true);
    int fiber_color = cur_tracking_window["roi_fiber_color"].toInt();
    if(fiber_color)
    {
        QPen pen(QColor(fiber_color == 1 ? 255:0,fiber_color == 2 ? 255:0,fiber_color == 3 ? 255:0));
        pen.setWidthF(double(pen_w));
        painter.setPen(pen);
    }
    int max_fiber = int(dir.num_fiber-1);
    for (int y = 0; y < slice_image.height(); y += steps)
        for (int x = 0; x < slice_image.width(); x += steps)
            {
                auto v = current_slice->toDiffusionSpace(cur_dim,x, y);
                if(!dim.is_valid(v))
                    continue;
                tipl::pixel_index<3> pos(v[0],v[1],v[2],dim);
                if (pos.index() >= dim.size() || dir.fa[0][pos.index()] == 0.0f)
                    continue;
                for (int fiber = max_fiber; fiber >= 0; --fiber)
                    if(dir.fa[fiber][pos.index()] > threshold)
                    {
                        if(threshold2 != 0.0f && fiber < dir.dt_fa.size() &&
                                 dir.dt_fa[fiber][pos.index()] < threshold2)
                            continue;
                        if((roi_fiber == 2 && fiber != 0) ||
                           (roi_fiber == 3 && fiber != 1))
                            continue;
                        auto dir_ptr = dir.get_fib(pos.index(),uint8_t(fiber));
                        if(!fiber_color)
                        {
                            QPen pen(QColor(int(std::abs(dir_ptr[0]) * 255.0f),
                                            int(std::abs(dir_ptr[1]) * 255.0f),
                                            int(std::abs(dir_ptr[2]) * 255.0f)));
                            pen.setWidthF(double(pen_w));
                            painter.setPen(pen);
                        }
                        float dx,dy;
                        dx = dy = (dir.fa[fiber][pos.index()]-threshold) * r;
                        dx *= dir_ptr[dir_x[cur_dim]];
                        dy *= dir_ptr[dir_y[cur_dim]];
                        painter.drawLine(
                            int(display_ratio*(float(x) + 0.5f) - dx),
                            int(display_ratio*(float(y) + 0.5f) - dy),
                            int(display_ratio*(float(x) + 0.5f) + dx),
                            int(display_ratio*(float(y) + 0.5f) + dy));
                    }
            }
}
void slice_view_scene::show_pos(QPainter& painter,std::shared_ptr<SliceModel> current_slice,const tipl::color_image& slice_image,unsigned char cur_dim)
{
    int x_pos,y_pos;
    float display_ratio = cur_tracking_window.get_scene_zoom(current_slice);
    current_slice->get_other_slice_pos(cur_dim,x_pos, y_pos);
    x_pos = int((float(x_pos) + 0.5f)*display_ratio);
    y_pos = int((float(y_pos) + 0.5f)*display_ratio);
    auto pen = painter.pen();
    pen.setColor(QColor(255,move_slice?255:0,move_slice ? 255:0,0x70));
    pen.setWidth(display_ratio);
    painter.setPen(pen);
    painter.drawLine(x_pos,0,x_pos,int(slice_image.height()*display_ratio));
    painter.drawLine(0,y_pos,int(slice_image.width()*display_ratio),y_pos);
}

void slice_view_scene::manage_slice_orientation(QImage& slice,QImage& new_slice,unsigned char cur_dim)
{
    new_slice = slice.mirrored(cur_tracking_window.slice_view_flip_x(cur_dim),cur_tracking_window.slice_view_flip_y(cur_dim));
}
QImage slice_view_scene::get_view_image(std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim,int pos,float display_ratio,bool simple)
{
    QImage new_view_image;
    tipl::color_image slice_image;
    current_slice->get_slice(slice_image,cur_dim,pos,cur_tracking_window.overlay_slices);
    if(slice_image.empty())
        return new_view_image;

    QImage scaled_image;
    {
        QImage slice_qimage;
        if(!simple)
        {
            tipl::color_image high_reso_slice_image;
            current_slice->get_high_reso_slice(high_reso_slice_image,cur_dim,pos,cur_tracking_window.overlay_slices);
            slice_qimage << high_reso_slice_image;
        }
        else
            slice_qimage << slice_image;

        scaled_image = slice_qimage.scaled(int(slice_image.width()*display_ratio),
                                        int(slice_image.height()*display_ratio));
    }

    if(!simple)
    {
        QImage region_image;
        cur_tracking_window.regionWidget->draw_region(current_slice->to_dif,cur_dim,pos,slice_image.shape(),display_ratio,region_image);
        if(!region_image.isNull())
        {
            QPainter painter(&scaled_image);
            painter.setCompositionMode(QPainter::CompositionMode_SourceAtop);
            painter.drawImage(0,0,region_image);
        }

        if(cur_tracking_window["roi_track"].toInt())
            cur_tracking_window.tractWidget->draw_tracts(cur_dim,pos,scaled_image,display_ratio);
    }



    if(cur_tracking_window["roi_layout"].toInt() <= 1) // not mosaic
    {
        QPainter painter(&scaled_image);
        if(!simple && cur_tracking_window["roi_fiber"].toInt() && cur_tracking_window.handle->trackable)
            show_fiber(painter,current_slice,slice_image,cur_dim);
        if(cur_tracking_window["roi_position"].toInt())
            show_pos(painter,current_slice,slice_image,cur_dim);
    }

    manage_slice_orientation(scaled_image,new_view_image,cur_dim);

    if(cur_tracking_window["roi_layout"].toInt() <= 1) // not mosaic
    {

        float grey = slice_image[0].r;
        grey += slice_image[0].g;
        grey += slice_image[0].b;
        grey /= 3.0f;
        line_color = grey < 128 ? Qt::white : Qt::black;

        QPainter painter2(&new_view_image);
        if(cur_tracking_window["roi_ruler"].toInt())
            show_ruler(painter2,current_slice,cur_dim,float(cur_tracking_window["roi_tic"].toFloat()*0.5f));
        if(cur_tracking_window["roi_label"].toInt())
            add_R_label(painter2,current_slice,cur_dim);
    }
    return new_view_image;
}

void slice_view_scene::add_R_label(QPainter& painter,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim)
{
    if(cur_dim)
    {
        painter.setPen(QPen(line_color));
        QFont f = font();  // start out with MainWindow's font..
        f.setPointSize(cur_tracking_window.get_scene_zoom(current_slice)+10); // and make a bit smaller for legend
        painter.setFont(f);
        painter.drawText(5,5,painter.window().width()-10,painter.window().height()-10,
                         cur_tracking_window["orientation_convention"].toInt() ? Qt::AlignTop|Qt::AlignRight: Qt::AlignTop|Qt::AlignLeft,"R");
    }
}

bool slice_view_scene::to_3d_space_single_slice(float x,float y,tipl::vector<3,float>& pos)
{
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    if(cur_tracking_window.slice_view_flip_x(cur_tracking_window.cur_dim))
        x = (cur_tracking_window.cur_dim ? geo[0]:geo[1])-x;
    if(cur_tracking_window.slice_view_flip_y(cur_tracking_window.cur_dim))
        y = geo[2] - y;
    pos = cur_tracking_window.current_slice->to3DSpace<tipl::vector<3,float> >(cur_tracking_window.cur_dim,x - 0.5f,y - 0.5f);
    return cur_tracking_window.current_slice->dim.is_valid(pos);
}

bool slice_view_scene::to_3d_space(float x,float y,tipl::vector<3,float>& pos)
{
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    float display_ratio = cur_tracking_window.get_scene_zoom();
    x /= display_ratio;
    y /= display_ratio;
    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
        return to_3d_space_single_slice(x,y,pos);
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slice
    {

        if(mouse_down)
        {
            if(cur_tracking_window["orientation_convention"].toInt())
            {
                if(cur_tracking_window.cur_dim == 0)
                    x -= geo[0];
            }
            else
                if(cur_tracking_window.cur_dim)
                    x -= geo[1];
            if(cur_tracking_window.cur_dim == 2)
                y -= geo[2];
        }
        else
        {
            unsigned char new_dim = 0;
            if(cur_tracking_window["orientation_convention"].toInt())
            {
                if(x > geo[0])
                {
                    x -= geo[0];
                    if(y < geo[2])
                        new_dim = 0;
                    else
                        return false;
                }
                else
                {
                    if(y < geo[2])
                        new_dim = 1;
                    else
                    {
                        new_dim = 2;
                        y -= geo[2];
                    }
                }
            }
            else
            {
                if(x < geo[1])
                {
                    if(y < geo[2])
                        new_dim = 0;
                    else
                        return false;
                }
                else
                {
                    x -= geo[1];
                    if(y < geo[2])
                        new_dim = 1;
                    else
                    {
                        new_dim = 2;
                        y -= geo[2];
                    }
                }
            }
            cur_tracking_window.cur_dim = new_dim;
        }

        return to_3d_space_single_slice(x,y,pos);
    }
    // mosaic
    /*
    if(cur_tracking_window["orientation_convention"].toInt())
        return false;
    pos[0] = x*float(mosaic_column_count);
    pos[1] = y*float(mosaic_column_count);
    pos[2] = std::floor(pos[1]/geo[1])*mosaic_column_count +
             std::floor(pos[0]/geo[0]);
    pos[0] -= std::floor(pos[0]/geo[0])*geo[0];
    pos[1] -= std::floor(pos[1]/geo[1])*geo[1];
    */
    return false;
    //return geo.is_valid(pos);
}

void slice_view_scene::update_3d(QImage captured)
{
    if(cur_tracking_window["roi_layout"].toInt() != 1)// 3 slices
        return;
    int view3_h = view_image.height()-view1_h;
    QImage I = captured.scaledToWidth(view1_w);
    int dif = (I.height()-view3_h)/2;
    QImage view4 = I.copy(QRect(0, dif, view1_w, dif+view3_h));
    QPainter painter(&view_image);
    painter.setPen(Qt::black);
    painter.setBrush(Qt::black);
    if(cur_tracking_window["orientation_convention"].toInt())
    {
        painter.drawRect(view_image.width()-view1_w,view1_h,view4.width(),view4.height()); // this avoid alpha corrupts the output
        painter.drawImage(view_image.width()-view1_w,view1_h,view4);
    }
    else
    {
        painter.drawRect(0,view1_h,view4.width(),view4.height()); // this avoid alpha corrupts the output
        painter.drawImage(0,view1_h,view4);
    }
    *this << view_image;
}

void slice_view_scene::show_slice(void)
{
    if(no_update)
        return;
    need_complete_view = true;
    paint_image(view_image,true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(complete_view_ready)
        show_complete_slice();
    else
        *this << view_image;
}

void slice_view_scene::show_complete_slice(void)
{
    complete_view_ready = false;
    if(no_update || need_complete_view)
        return;
    view_image = complete_view_image;
    *this << view_image;
}


void slice_view_scene::paint_image(void)
{
    while(!free_thread)
    {
        if(!no_update && need_complete_view)
        {
            complete_view_ready = false;
            need_complete_view = false;
            if(need_complete_view)
                continue;
            paint_image(complete_view_image,false);
            if(need_complete_view)
                continue;
            complete_view_ready = true;
        }
        while(!need_complete_view && !free_thread)
            std::this_thread::yield();
    }
}

void slice_view_scene::paint_image(QImage& out,bool simple)
{
    QImage I;
    auto current_slice = cur_tracking_window.current_slice;
    unsigned char cur_dim = cur_tracking_window.cur_dim;
    float display_ratio = cur_tracking_window.get_scene_zoom(current_slice);
    simple |= cur_tracking_window.slice_need_update;
    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
        I = get_view_image(current_slice,cur_dim,current_slice->slice_pos[cur_dim],display_ratio,simple);
    else
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slices
    {
        auto view1 = get_view_image(current_slice,0,current_slice->slice_pos[0],display_ratio,simple);
        auto view2 = get_view_image(current_slice,1,current_slice->slice_pos[1],display_ratio,simple);
        auto view3 = get_view_image(current_slice,2,current_slice->slice_pos[2],display_ratio,simple);
        I = QImage(QSize(view1.width()+view2.width(),view1.height()+view3.height()),QImage::Format_RGB32);
        view1_h = view1.height();
        view1_w = view1.width();
        QPainter painter(&I);
        painter.fillRect(0,0,I.width(),I.height(),QColor(0,0,0));


        if(cur_tracking_window["orientation_convention"].toInt())
        {
            painter.drawImage(view2.width(),0,view1);
            painter.drawImage(0,0,view2);
            painter.drawImage(0,view2.height(),view3);
        }
        else
        {
            painter.drawImage(0,0,view1);
            painter.drawImage(view1.width(),0,view2);
            painter.drawImage(view1.width(),view1.height(),view3);
        }
        QPen pen(QColor(255,255,255));
        pen.setWidthF(std::max(1.0,double(display_ratio)/4.0));
        painter.setPen(pen);
        painter.drawLine(cur_tracking_window["orientation_convention"].toInt() ? view2.width() : view1.width(),0,
                         cur_tracking_window["orientation_convention"].toInt() ? view2.width() : view1.width(),I.height());
        painter.drawLine(0,view1.height(),I.width(),view1.height());
    }
    else
    // mosaic
    {
        auto dim = current_slice->dim;

        unsigned int skip = uint32_t(std::pow(2,cur_tracking_window["roi_layout"].toInt()-2));
        unsigned int skip_row = uint32_t(cur_tracking_window["roi_mosaic_skip_row"].toInt());
        mosaic_column_count = cur_tracking_window["roi_mosaic_column"].toInt() ?
                uint32_t(cur_tracking_window["roi_mosaic_column"].toInt()):
                std::max<uint32_t>(1,uint32_t(std::ceil(
                                   std::sqrt(float(current_slice->dim[cur_dim]) / skip))));
        mosaic_row_count = std::max<uint32_t>(1,uint32_t(std::ceil(float(dim[cur_dim]/skip)/float(mosaic_column_count))));
        if(mosaic_row_count >= skip_row+skip_row+2)
            mosaic_row_count -= skip_row+skip_row;
        else
            skip_row = 0;
        float scale = display_ratio/float(mosaic_column_count);
        unsigned char dim_order[3][2]= {{1,2},{0,2},{0,1}};

        I = QImage(QSize(
                                int(dim[dim_order[uint8_t(cur_dim)][0]]*scale*mosaic_column_count),
                                int(dim[dim_order[uint8_t(cur_dim)][1]]*scale*mosaic_row_count)),QImage::Format_RGB32);
        QPainter painter(&I);
        tipl::shape<2> mosaic_tile_geo;
        {
            unsigned int skip_slices = skip_row*mosaic_column_count;
            for(unsigned int z = 0,slice_pos = skip-1;slice_pos < dim[cur_dim]-skip_slices;++z,slice_pos += skip)
            {
                if(z < skip_slices)
                    continue;
                auto view = get_view_image(current_slice,cur_dim,slice_pos,scale,simple);
                if(z == skip_slices)
                    painter.fillRect(0,0,I.width(),I.height(),view.pixel(0,0));
                painter.drawImage(QPoint(scale*int(dim[dim_order[uint8_t(cur_dim)][0]]*((z-skip_slices)%mosaic_column_count)),
                                         scale*int(dim[dim_order[uint8_t(cur_dim)][1]]*((z-skip_slices)/mosaic_column_count))), view);
            }
        }

        if(cur_tracking_window["roi_label"].toInt()) // not sagittal view
            add_R_label(painter,current_slice,cur_dim);
    }
    out = I;
}

void slice_view_scene::copyClipBoard()
{
    if(cur_tracking_window["roi_layout"].toInt() != 0)// mosaic
    {
        QApplication::clipboard()->setImage(view_image);
        return;
    }
    QImage output = view_image;
    QApplication::clipboard()->setImage(output);
}

void slice_view_scene::wheelEvent(QGraphicsSceneWheelEvent *wheelEvent)
{
    if(views().size() == 0)
        return;
    auto* vb = views()[0]->verticalScrollBar();
    bool no_scroll = wheelEvent->modifiers() & Qt::ControlModifier;
    if(vb->isVisible() && !no_scroll)
    {
        if((wheelEvent->delta() < 0 && vb->maximum() != vb->value()) ||
            (wheelEvent->delta() > 0 && vb->value() > 0))
            return; //let default wheel event handle it by change the vertical scroll
    }
    tipl::vector<3,float> pos;
    // 3 view condition, transfer event to 3D window
    if(click_on_3D(float(wheelEvent->scenePos().x()),float(wheelEvent->scenePos().y())))
    {
        QWheelEvent we(wheelEvent->pos(),wheelEvent->screenPos(),QPoint(),QPoint(0,wheelEvent->delta()),
                       wheelEvent->buttons(),wheelEvent->modifiers(),Qt::NoScrollPhase,false);
        cur_tracking_window.glWidget->wheelEvent(&we);
        return;
    }

    // zoom in or out
    cur_tracking_window.set_roi_zoom(cur_tracking_window["roi_zoom"].toFloat()+ ((wheelEvent->delta() > 0) ? 0.5f:-0.5f));
    wheelEvent->accept();
}
void slice_view_scene::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(clicked_3d)
    {
        send_event_to_3D(QEvent::MouseButtonDblClick,mouseEvent);
        return;
    }

    if (mouseEvent->button() == Qt::MiddleButton)
        return;
    if(sel_mode == 4)
    {
        mousePressEvent(mouseEvent);
        sel_mode = 1;
        mouseReleaseEvent(mouseEvent);
        sel_mode = 4;
        sel_point.clear();
        sel_coord.clear();
    }
    else
    {
        if(cur_tracking_window["roi_layout"].toInt() > 1)// single slice or 3 slices
            return;
        float Y = mouseEvent->scenePos().y();
        float X = mouseEvent->scenePos().x();
        tipl::vector<3,float> pos;
        if(!to_3d_space(X,Y,pos))
            return;
        pos.round();
        cur_tracking_window.ui->glSagSlider->setValue(pos[0]);
        cur_tracking_window.ui->glCorSlider->setValue(pos[1]);
        cur_tracking_window.ui->glAxiSlider->setValue(pos[2]);
    }
}

void slice_view_scene::adjust_xy_to_layout(float& X,float& Y)
{
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    float display_ratio = cur_tracking_window.get_scene_zoom();
    if(cur_tracking_window["roi_layout"].toInt() == 1)
    {
        if(cur_tracking_window["orientation_convention"].toInt())
        {
            if(cur_tracking_window.cur_dim == 0)
                X -= geo[0]*display_ratio;
        }
        else
            if(cur_tracking_window.cur_dim >= 1)
                X -= geo[1]*display_ratio;
        if(cur_tracking_window.cur_dim == 2)
            Y -= geo[2]*display_ratio;
    }
}
void slice_view_scene::send_event_to_3D(QEvent::Type type,
                                        QGraphicsSceneMouseEvent * mouseEvent)
{
    float y = mouseEvent->scenePos().y();
    float x = mouseEvent->scenePos().x();
    float display_ratio = cur_tracking_window.get_scene_zoom();
    auto dim = cur_tracking_window.current_slice->dim;
    y -= dim[2]*display_ratio;
    if(cur_tracking_window["orientation_convention"].toInt())
        x -= dim[1]*display_ratio;

    float view_scale = cur_tracking_window.glWidget->width()/(dim[1]*display_ratio);
    x *= view_scale;
    y *= view_scale;
    y += (cur_tracking_window.glWidget->height()-dim[1]*display_ratio*view_scale)*0.5;

    if(type == QEvent::MouseMove)
    {
        QMouseEvent me(QEvent::MouseMove, QPointF(x,y), QPointF(x,y), QPointF(x,y),
            mouseEvent->button(), mouseEvent->buttons(), mouseEvent->modifiers(), Qt::MouseEventNotSynthesized);
        cur_tracking_window.glWidget->mouseMoveEvent(&me);
    }
    if(type == QEvent::MouseButtonPress)
    {
        QMouseEvent me(QEvent::MouseButtonPress, QPointF(x,y), QPointF(x,y), QPointF(x,y),
            mouseEvent->button(), mouseEvent->buttons(), mouseEvent->modifiers(), Qt::MouseEventNotSynthesized);
        cur_tracking_window.glWidget->mousePressEvent(&me);
    }

    if(type == QEvent::MouseButtonRelease)
    {
        QMouseEvent me(QEvent::MouseButtonRelease, QPointF(x,y), QPointF(x,y), QPointF(x,y),
            mouseEvent->button(), mouseEvent->buttons(), mouseEvent->modifiers(), Qt::MouseEventNotSynthesized);
        cur_tracking_window.glWidget->mouseReleaseEvent(&me);
    }

    if(type == QEvent::MouseButtonDblClick)
    {
        QMouseEvent me(QEvent::MouseButtonDblClick, QPointF(x,y), QPointF(x,y), QPointF(x,y),
            mouseEvent->button(), mouseEvent->buttons(), mouseEvent->modifiers(), Qt::MouseEventNotSynthesized);
        cur_tracking_window.glWidget->mouseDoubleClickEvent(&me);
    }

}
bool slice_view_scene::click_on_3D(float x,float y)
{
    float display_ratio = cur_tracking_window.get_scene_zoom();
    x /= display_ratio;
    y /= display_ratio;
    auto dim = cur_tracking_window.current_slice->dim;
    return cur_tracking_window["roi_layout"].toInt() == 1 &&
           y > dim[2] &&
           ((x < dim[1]) ^ cur_tracking_window["orientation_convention"].toInt());
}
void slice_view_scene::mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window["roi_layout"].toInt() > 1)// mosaic
    {
        QMessageBox::critical(0,"ERROR","Switch to regular view to edit ROI. (Right side under Options, change [Region Window][Slice Layout] to Single Slice) ");
        return;
    }

    tipl::vector<3,float> pos;
    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();

    if(click_on_3D(X,Y))
    {
        clicked_3d = true;
        send_event_to_3D(QEvent::MouseButtonPress,mouseEvent);
        return;
    }
    if(!to_3d_space(X,Y,pos))
        return;
    adjust_xy_to_layout(X,Y);
    if(mouseEvent->button() == Qt::MiddleButton || sel_mode == 5)// move object
    {
        auto slice = cur_tracking_window.current_slice;
        tipl::vector<3,float> p(pos);
        if(!slice->is_diffusion_space)
            p.to(slice->to_dif);
        move_viewing_slice = false;
        // select slice focus?
        float display_ratio = cur_tracking_window.get_scene_zoom();
        if(cur_tracking_window["roi_position"].toInt() &&
           std::abs(p[0]-slice->slice_pos[0]) < 10.0f/display_ratio &&
           std::abs(p[1]-slice->slice_pos[1]) < 10.0f/display_ratio &&
           std::abs(p[2]-slice->slice_pos[2]) < 10.0f/display_ratio)
        {
            move_slice = true;
            show_slice();
        }
        else
        // select region
        {
            move_slice = false;
            bool find_region = false;
            size_t found_index = 0;
            tipl::adaptive_par_for(cur_tracking_window.regionWidget->regions.size(),[&](size_t index)
            {
                if(find_region || cur_tracking_window.regionWidget->item(int(index),0)->checkState() != Qt::Checked)
                    return;

                if(cur_tracking_window.regionWidget->regions[index]->has_point(p) && !find_region)
                {
                    find_region = true;
                    found_index = index;
                }
            });
            if(find_region)
            {
                cur_tracking_window.regionWidget->selectRow(found_index);
                cur_tracking_window.regionWidget->regions[found_index]->add_points(std::vector<tipl::vector<3,short> >());
            }
            else
                move_viewing_slice = true;
        }
        mid_down = true;
    }
    else
    {
        if(cur_tracking_window.regionWidget->regions.empty())
        {
            cur_tracking_window.regionWidget->new_region();
            cur_region = -1;
        }

        if(cur_tracking_window.regionWidget->currentRow() != cur_region)
        {
            cur_region = cur_tracking_window.regionWidget->currentRow();
            sel_point.clear();
            sel_coord.clear();
        }
    }

    if(sel_mode != 4)
    {
        sel_point.clear();
        sel_coord.clear();
        sel_point.push_back(tipl::vector<2,short>(X, Y));
        sel_coord.push_back(pos);
    }

    switch (sel_mode)
    {
    case 0:
    case 2:
    case 3:
    case 4:
    case 6:
        sel_coord.push_back(pos);
        sel_point.push_back(tipl::vector<2,short>(X, Y));
        break;
    default:
        break;
    }
    mouse_down = true;
}
void slice_view_scene::new_annotated_image(void)
{
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    float display_ratio = cur_tracking_window.get_scene_zoom();
    if(cur_tracking_window["roi_layout"].toInt() == 0)
        annotated_image = view_image;
    else
    {
        annotated_image = view_image.copy(
                        cur_tracking_window["orientation_convention"].toInt() ?
                            (cur_tracking_window.cur_dim != 0 ? 0:geo[0]*display_ratio):
                            (cur_tracking_window.cur_dim == 0 ? 0:geo[1]*display_ratio),
                        cur_tracking_window.cur_dim != 2 ? 0:geo[2]*display_ratio,
                        cur_tracking_window.cur_dim == 0 ? geo[1]*display_ratio:geo[0]*display_ratio,
                        cur_tracking_window.cur_dim != 2 ? geo[2]*display_ratio:geo[1]*display_ratio);
    }
}

void slice_view_scene::mouseMoveEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(clicked_3d)
    {
        send_event_to_3D(QEvent::MouseMove,mouseEvent);
        return;
    }
    if(cur_tracking_window["roi_layout"].toInt() > 1)// single slice
        return;
    if (!mouse_down)
        return;
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();
    float display_ratio = cur_tracking_window.get_scene_zoom();


    cX = X;
    cY = Y;
    tipl::vector<3,float> pos;
    if (!mouse_down || sel_mode == 4)
        return;
    to_3d_space(cX,cY,pos);
    adjust_xy_to_layout(X,Y);

    if(mid_down) // move object
    {
        auto slice = cur_tracking_window.current_slice;
        if(move_slice)
        {
            tipl::vector<3,float> p(pos);
            if(!slice->is_diffusion_space)
                p.to(slice->to_dif);
            cur_tracking_window.ui->glSagBox->setValue(p[0]);
            cur_tracking_window.ui->glCorBox->setValue(p[1]);
            cur_tracking_window.ui->glAxiBox->setValue(p[2]);
            return;
        }
        else
        if(move_viewing_slice && !sel_point.empty())
        {
            if(sel_point.back()[0] < X)
                cur_tracking_window.ui->SlicePos->setValue(cur_tracking_window.ui->SlicePos->value()+1);
            else
                cur_tracking_window.ui->SlicePos->setValue(cur_tracking_window.ui->SlicePos->value()-1);
            sel_point.back() = tipl::vector<2,short>(X, Y);
            return;
        }
        else
        if(!cur_tracking_window.regionWidget->regions.empty() &&
                !sel_coord.empty() && pos != sel_coord.back())
            {
                auto cur_region = cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()];
                tipl::vector<3,float> p1(pos),p2(sel_coord.back());

                if(!slice->is_diffusion_space)
                {
                    p1.to(slice->to_dif);
                    p2.to(slice->to_dif);
                }
                p1 -= p2;
                p1.round();
                if(p1.length() != 0)
                {
                    if(!cur_region->is_diffusion_space)
                    {
                        tipl::matrix<4,4> iT(tipl::inverse(cur_region->to_diffusion_space));
                        p2 = p1;
                        p2.rotate(tipl::transformation_matrix<float>(iT).sr);
                        cur_region->shift(p2);
                    }
                    else
                        cur_region->shift(p1);

                    p1.to(slice->to_slice);
                    tipl::vector<3> zero;
                    zero.to(slice->to_slice);
                    sel_coord.back() += p1-zero;
                    emit need_update();
                }
            }

        return;
    }

    new_annotated_image();
    if(sel_coord.empty() || sel_point.empty())
        return;

    if(sel_mode == 6) // ruler
    {
        sel_coord.back() = pos;
        sel_point.back() = tipl::vector<2,short>(X, Y);
        QPainter paint(&annotated_image);
        show_ruler2(paint);
    }

    QPainter paint(&annotated_image);
    paint.setPen(cur_tracking_window.regionWidget->currentRowColor());
    paint.setBrush(Qt::NoBrush);
    switch (sel_mode)
    {
    case 0: // draw rectangle
        paint.drawRect(X, Y, sel_point.front()[0]-X,sel_point.front()[1]-Y);
        sel_coord.back() = pos;
        sel_point.back() = tipl::vector<2,short>(X, Y);
        break;
    case 1: //free hand
        sel_coord.push_back(pos);
        sel_point.push_back(tipl::vector<2,short>(X, Y));

        for (unsigned int index = 1; index < sel_point.size(); ++index)
            paint.drawLine(sel_point[index-1][0], sel_point[index-1][1],sel_point[index][0], sel_point[index][1]);
        break;
    case 2:
    {
        int dx = X - sel_point.front()[0];
        int dy = Y - sel_point.front()[1];
        int dis = std::sqrt((double)(dx * dx + dy * dy));
        paint.drawEllipse(QPoint(sel_point.front()[0],sel_point.front()[1]),dis,dis);
        sel_coord.back() = pos;
        sel_point.back() = tipl::vector<2,short>(X, Y);
    }
    break;
    case 3:
    {
        int dx = X - sel_point.front()[0];
        int dy = Y - sel_point.front()[1];
        int dis = std::sqrt((double)(dx * dx + dy * dy));
        paint.drawRect(sel_point.front()[0] - dis,
                        sel_point.front()[1] - dis, dis*2, dis*2);
        sel_coord.back() = pos;
        sel_point.back() = tipl::vector<2,short>(X, Y);
    }
    break;
    default:
        break;
    }

    clear();
    setItemIndexMethod(QGraphicsScene::NoIndex);

    if(cur_tracking_window["roi_layout"].toInt() == 1)
    {
        QImage temp = view_image;
        QPainter paint(&temp);
        paint.drawImage(cur_tracking_window["orientation_convention"].toInt()?
                        (cur_tracking_window.cur_dim != 0 ? 0:geo[0]*display_ratio):
                        (cur_tracking_window.cur_dim == 0 ? 0:geo[1]*display_ratio),
                        cur_tracking_window.cur_dim != 2 ? 0:geo[2]*display_ratio,annotated_image);
        addPixmap(tipl::qt::image2pixelmap(temp));
    }
    else
        addPixmap(tipl::qt::image2pixelmap(annotated_image));

}


void slice_view_scene::mouseReleaseEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(clicked_3d)
    {
        clicked_3d = false;
        send_event_to_3D(QEvent::MouseButtonRelease,mouseEvent);
        return;
    }
    if (!mouse_down)
        return;
    mouse_down = false;
    if(mid_down)
    {
        sel_point.clear();
        sel_coord.clear();
        mid_down = false;
        if(move_slice)
        {
            move_slice = false;
            need_update();
        }
        return;
    }
    auto& cur_slice = cur_tracking_window.current_slice;
    tipl::shape<3> geo(cur_slice->dim);
    float display_ratio = cur_tracking_window.get_scene_zoom();
    auto regionWidget = cur_tracking_window.regionWidget;
    if (regionWidget->currentRow() < 0 || regionWidget->currentRow() >= int(regionWidget->regions.size()))
    {
        need_update();
        return;
    }
    if(regionWidget->item(regionWidget->currentRow(),0)->checkState() != Qt::Checked)
        regionWidget->check_row(regionWidget->currentRow(),true);

    std::vector<tipl::vector<3,float> >points;
    switch (sel_mode)
    {
    case 0: // rectangle
    {
        if (sel_coord.size() < 2)
            return;
        tipl::vector<3,float> min_coord, max_coord;
        for (unsigned int i = 0; i < 3; ++i)
            if (sel_coord[0][i] > sel_coord[1][i])
            {
                max_coord[i] = sel_coord[0][i];
                min_coord[i] = sel_coord[1][i];
            }
            else
            {
                max_coord[i] = sel_coord[1][i];
                min_coord[i] = sel_coord[0][i];
            }
        float dis = 1.0f/display_ratio;
        for (float z = min_coord[2]; z <= max_coord[2]; z += dis)
            for (float y = min_coord[1]; y <= max_coord[1]; y += dis)
                for (float x = min_coord[0]; x <= max_coord[0]; x += dis)
                    if (cur_slice->dim.is_valid(x, y, z))
                        points.push_back(tipl::vector<3,float>(x, y, z));
    }
    break;
    case 1: //free hand
    {
        if (sel_coord.size() <= 2)
        {
            points.push_back(sel_coord.front());
            break;
        }
        {
            QImage bitmap(annotated_image.width(),annotated_image.height(),QImage::Format_Mono);
            QPainter paint(&bitmap);
            paint.setBrush(Qt::black);
            paint.drawRect(0,0,bitmap.width(),bitmap.height());
            paint.setBrush(Qt::white);
            std::vector<QPoint> qpoints(sel_point.size());
            for(unsigned int index = 0;index < sel_point.size();++index)
                qpoints[index] = QPoint(sel_point[index][0],sel_point[index][1]);
            paint.drawPolygon(&*qpoints.begin(),qpoints.size() - 1);
            tipl::shape<2> geo2(bitmap.width(),bitmap.height());
            for (tipl::pixel_index<2>index(geo2); index < geo2.size();++index)
            {
                tipl::vector<3,float> pos;
                if (QColor(bitmap.pixel(index.x(),index.y())).red() < 64
                    || !to_3d_space_single_slice(float(index.x())/display_ratio,float(index.y())/display_ratio,pos))
                    continue;
                points.push_back(pos);
            }
            sel_point.clear();
            sel_coord.clear();
        }
    }
    break;
    case 2:
    {
        if (sel_coord.size() < 2)
            return;
        float dx = sel_coord[1][0] - sel_coord[0][0];
        float dy = sel_coord[1][1] - sel_coord[0][1];
        float dz = sel_coord[1][2] - sel_coord[0][2];
        float distance2 = (dx * dx + dy * dy + dz * dz);
        float dis = std::sqrt((double)distance2);
        float interval = 1.0f/(std::min<float>(16.0,display_ratio));
        for (float z = -dis; z <= dis; z += interval)
        {
            float zz = z*z;
            for (float y = -dis; y <= dis; y += interval)
            {
                float yy = y*y;
                for (float x = -dis; x <= dis; x += interval)
                    if (x*x + yy + zz <= distance2)
                        points.push_back(tipl::vector<3,float>(sel_coord[0][0] + x,
                                            sel_coord[0][1] + y, sel_coord[0][2] + z));
            }
        }
    }
    break;
    case 3:
    {
        if (sel_coord.size() < 2)
            return;
        float dx = sel_coord[1][0] - sel_coord[0][0];
        float dy = sel_coord[1][1] - sel_coord[0][1];
        float dz = sel_coord[1][2] - sel_coord[0][2];
        float distance2 = (dx * dx + dy * dy + dz * dz);
        float dis = std::sqrt((double)distance2);
        float interval = 1.0f/(std::min<float>(16.0,display_ratio));
        for (float z = -dis; z <= dis; z += interval)
            for (float y = -dis; y <= dis; y += interval)
                for (float x = -dis; x <= dis; x += interval)
                    points.push_back(tipl::vector<3,float>(sel_coord[0][0] + x,
                                                        sel_coord[0][1] + y, sel_coord[0][2] + z));

    }
    break;
    case 4:
    {
        new_annotated_image();
        QPainter paint(&annotated_image);
        paint.setPen(cur_tracking_window.regionWidget->currentRowColor());
        paint.setBrush(Qt::NoBrush);

        for(int index = 1;index < sel_point.size();++index)
            paint.drawLine(sel_point[index-1][0],
                           sel_point[index-1][1],
                           sel_point[index][0],
                           sel_point[index][1]);

        clear();
        setItemIndexMethod(QGraphicsScene::NoIndex);
        if(cur_tracking_window["roi_layout"].toInt() == 1)
        {
            QImage temp = view_image;
            QPainter paint(&temp);
            paint.drawImage(cur_tracking_window["orientation_convention"].toInt() ?
                                (cur_tracking_window.cur_dim != 0 ? 0:geo[0]*display_ratio) :
                                (cur_tracking_window.cur_dim == 0 ? 0:geo[1]*display_ratio),
                            cur_tracking_window.cur_dim != 2 ? 0:geo[2]*display_ratio,annotated_image);

            addPixmap(tipl::qt::image2pixelmap(temp));
        }
        else
            addPixmap(tipl::qt::image2pixelmap(annotated_image));
        return;
    }
    case 6:
    {
        auto slice = dynamic_cast<CustomSliceModel*>(cur_slice.get());
        if(slice && !slice->picture.empty())
        {
            tipl::vector<2> from = sel_point.front();
            tipl::vector<2> to = sel_point.back();
            from /= display_ratio;
            to /= display_ratio;
            if(cur_tracking_window.cur_dim == 0)
            {
                from[1] = float(slice->dim.depth()) - from[1];
                to[1] = float(slice->dim.depth()) - to[1];
            }
            tipl::out() << "warping picture: " << from << " to " << to;
            slice->warp_picture(from,to);
        }
        need_update();
        return;
    }
    }


    std::vector<tipl::vector<3,short> > points_int16(points.size());
    std::copy(points.begin(),points.end(),points_int16.begin());


    // apply thresholded edits
    if(cur_tracking_window.ui->draw_threshold->value() != 0.0f)
    {
        float threshold = cur_tracking_window.ui->draw_threshold->value();
        bool upper = cur_tracking_window.ui->draw_threshold_direction->currentIndex() == 0;
        tipl::const_pointer_image<3,float> I = cur_slice->get_source();
        for(size_t i = 0;i < points_int16.size();)
        {
            if(I.shape().is_valid(points_int16[i]))
            {
                auto cur_value = I.at(points_int16[i]);
                if ((upper ^ (cur_value >= threshold)))
                {
                    points_int16[i] = points_int16.back();
                    points_int16.pop_back();
                    continue;
                }
            }
            ++i;
        }
    }
    auto& cur_region = regionWidget->regions[regionWidget->currentRow()];
    if(cur_slice->dim != cur_region->dim ||
       cur_slice->to_dif == cur_region->to_diffusion_space)
    {
        ROIRegion draw_region(cur_slice->dim,tipl::vector<3>());
        draw_region.to_diffusion_space = cur_slice->to_dif;
        draw_region.region.swap(points_int16);
        points_int16 = std::move(draw_region.to_space(cur_region->dim,cur_region->to_diffusion_space));
    }
    cur_region->add_points(std::move(points_int16),
                           mouseEvent->button() == Qt::RightButton || mouseEvent->modifiers() & Qt::ShiftModifier);

    need_update();
}

void slice_view_scene::center()
{
    QList<QGraphicsView*> views = this->views();
    for(int index = 0;index < views.size();++index)
        views[index]->centerOn(view_image.width()/2,view_image.height()/2);
}



