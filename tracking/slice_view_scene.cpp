#include "tipl/tipl.hpp"
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
#include "libs/gzip_interface.hpp"
#include "libs/tracking/fib_data.hpp"
#include "opengl/glwidget.h"
#include <QScrollBar>

QPixmap fromImage(const QImage &I)
{
    #ifdef WIN32
        return QPixmap::fromImage(I);
    #else
        //For Mac, the endian system is BGRA and all QImage needs to be converted.
        return QPixmap::fromImage(I.convertToFormat(QImage::Format_ARGB32));
    #endif
}

void show_view(QGraphicsScene& scene,QImage I)
{
    scene.setSceneRect(0, 0, I.width(),I.height());
    scene.clear();
    scene.addPixmap(fromImage(I));
}
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
void slice_view_scene::show_ruler(QPainter& paint,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim)
{
    int roi_index = cur_tracking_window["roi_label"].toInt();
    float zoom = cur_tracking_window.get_scene_zoom(current_slice);
    float zoom_2 = zoom/2;
    int tic_dis = 5;
    while(zoom*float(tic_dis) < 20.0f)
        tic_dis *= 2;
    int tic_length = int(zoom*float(tic_dis));

    QPen pen;  // creates a default pen
    pen.setWidth(int(zoom_2));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setColor(line_color);
    paint.setPen(pen);

    QFont f = font();
    f.setPointSize(tic_length/4);
    paint.setFont(f);
    // for qsdr
    bool is_qsdr = cur_tracking_window.handle->is_qsdr;
    auto trans = cur_tracking_window.handle->trans_to_mni;
    if(!current_slice->is_diffusion_space)
        trans *= current_slice->T;
    tipl::vector<3,int> qsdr_origin;
    tipl::vector<3> qsdr_scale(trans[0],trans[5],trans[10]);
    tipl::vector<3> qsdr_shift(trans[3],trans[7],trans[11]);
    if(is_qsdr)
    {
        qsdr_origin[0] = int(std::round(-trans[3]/trans[0]));
        qsdr_origin[1] = int(std::round(-trans[7]/trans[5]));
        qsdr_origin[2] = int(std::round(-trans[11]/trans[10]));
    }
    // horizontal direction
    int space_x = 0;
    {
        bool flip_x = cur_tracking_window.slice_view_flip_x(cur_dim);
        uint8_t dim = (cur_dim == 0 ? 1:0);
        int Y = paint.window().height()-tic_length;
        int pad_x =  (is_qsdr ? qsdr_origin[dim]%tic_dis:0);
        if(pad_x == 0)
            pad_x = tic_dis;
        int length = (current_slice->dim[dim]-pad_x-pad_x)
                        /tic_dis*tic_dis;
        space_x = int(float(pad_x)*zoom+zoom_2);
        for(int tic = 0;tic <= length;tic += tic_dis)
        {
            int X = int(float(pad_x+tic)*zoom+zoom_2);
            if(flip_x)
                X = paint.window().width()-X;
            if(tic+tic_dis <= length)
                paint.drawLine(X,Y,X+(flip_x ? -tic_length:tic_length),Y);
            paint.drawLine(X,Y,X,Y+int(zoom));

            float axis_label = tic+pad_x;
            if(is_qsdr)
                axis_label = axis_label*qsdr_scale[dim]+qsdr_shift[dim];
            paint.drawText(X-40,Y+tic_length/2-40,80,80,
                               Qt::AlignHCenter|Qt::AlignVCenter,
                               QString::number(double(std::round(axis_label*100.0f)/100.0f)));
        }
    }
    {
        bool flip_y = cur_dim != 2;
        uint8_t dim = (cur_dim == 2 ? 1:2);
        int X = space_x;
        int pad_y = (is_qsdr ? int(qsdr_origin[dim]%tic_dis):0);
        if(pad_y == 0)
            pad_y = tic_dis;
        if(flip_y && pad_y < tic_dis)
            pad_y += tic_dis;
        int length = (current_slice->dim[dim]-pad_y-tic_dis)
                        /tic_dis*tic_dis;
        for(int tic = 0;tic <= length;tic += tic_dis)
        {
            int Y = int(float(pad_y+tic)*zoom+zoom_2);
            if(flip_y)
                Y = paint.window().height()-Y;
            else
                if(tic == 0 && roi_index)
                    continue;
            if(tic+tic_dis <= length)
                paint.drawLine(X,Y,X,Y+(flip_y ? -tic_length: tic_length));
            paint.drawLine(X,Y,X-int(zoom),Y);
            float axis_label = tic+pad_y;
            if(is_qsdr)
                axis_label = axis_label*qsdr_scale[dim]+qsdr_shift[dim];
            paint.drawText(2,Y-40,X-int(zoom)-2,80,
                               Qt::AlignRight|Qt::AlignVCenter,
                               QString::number(double(std::round(axis_label*100.0f)/100.0f)));
        }
    }
}
void slice_view_scene::show_fiber(QPainter& painter,std::shared_ptr<SliceModel> current_slice,const tipl::color_image& slice_image,unsigned char cur_dim)
{
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
    float threshold2 = cur_tracking_window["dt_index"].toInt() ? cur_tracking_window["dt_threshold"].toFloat() : 0.0f;
    if (threshold == 0.0f)
        threshold = 0.00000001f;
    int X(0),Y(0),Z(0);
    unsigned char dir_x[3] = {1,0,0};
    unsigned char dir_y[3] = {2,2,1};

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
                current_slice->toDiffusionSpace(cur_dim,x, y, X, Y, Z);
                if(!dim.is_valid(X,Y,Z))
                    continue;
                tipl::pixel_index<3> pos(X,Y,Z,dim);
                if (pos.index() >= dim.size() || dir.fa[0][pos.index()] == 0.0f)
                    continue;
                for (int fiber = max_fiber; fiber >= 0; --fiber)
                    if(dir.fa[fiber][pos.index()] > threshold)
                    {
                        if(threshold2 != 0.0f && dir.dt_fa[fiber][pos.index()] < threshold2)
                            continue;
                        if((roi_fiber == 2 && fiber != 0) ||
                           (roi_fiber == 3 && fiber != 1))
                            continue;
                        const float* dir_ptr = dir.get_fib(pos.index(),uint8_t(fiber));
                        if(!fiber_color)
                        {
                            QPen pen(QColor(int(std::abs(dir_ptr[0]) * 255.0f),
                                            int(std::abs(dir_ptr[1]) * 255.0f),
                                            int(std::abs(dir_ptr[2]) * 255.0f)));
                            pen.setWidthF(double(pen_w));
                            painter.setPen(pen);
                        }
                        float dx = r * dir_ptr[dir_x[cur_dim]] + 0.5f;
                        float dy = r * dir_ptr[dir_y[cur_dim]] + 0.5f;
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
    painter.setPen(QColor(255,move_slice?255:0,move_slice ? 255:0));
    painter.drawLine(x_pos,0,x_pos,std::max<int>(0,y_pos-20));
    painter.drawLine(x_pos,std::min<int>(y_pos+20,int(slice_image.height()*display_ratio)),x_pos,int(slice_image.height()*display_ratio));
    painter.drawLine(0,y_pos,std::max<int>(0,x_pos-20),y_pos);
    painter.drawLine(std::min<int>(x_pos+20,int(slice_image.width()*display_ratio)),y_pos,int(slice_image.width()*display_ratio),y_pos);
}

void slice_view_scene::manage_slice_orientation(QImage& slice,QImage& new_slice,unsigned char cur_dim)
{
    bool flip_x = cur_tracking_window.slice_view_flip_x(cur_dim);
    bool flip_y = cur_tracking_window.slice_view_flip_y(cur_dim);
    new_slice = (!flip_x && !flip_y ? slice : slice.mirrored(flip_x,flip_y));
}
void slice_view_scene::get_view_image(QImage& new_view_image,std::shared_ptr<SliceModel> current_slice,unsigned char cur_dim,float display_ratio,bool simple)
{
    tipl::color_image slice_image;
    current_slice->get_slice(slice_image,cur_dim,cur_tracking_window.overlay_slices);
    if(slice_image.empty())
        return;

    QImage scaled_image;
    {
        QImage slice_qimage;
        tipl::color_image high_reso_slice_image;
        if(!simple && current_slice->handle && current_slice->handle->has_high_reso)
        {
            current_slice->get_high_reso_slice(high_reso_slice_image,cur_dim);
            slice_qimage = QImage(reinterpret_cast<const unsigned char*>(&*high_reso_slice_image.begin()),
                                  high_reso_slice_image.width(),high_reso_slice_image.height(),QImage::Format_RGB32);

        }
        else
            slice_qimage = QImage(reinterpret_cast<const unsigned char*>(&*slice_image.begin()),
                          slice_image.width(),slice_image.height(),QImage::Format_RGB32);

        scaled_image = slice_qimage.scaled(int(slice_image.width()*display_ratio),
                                        int(slice_image.height()*display_ratio));
        scaled_image.detach();
    }

    if(!simple)
    {
        QImage region_image;
        cur_tracking_window.regionWidget->draw_region(current_slice,
                                                  cur_dim,
                                                  cur_tracking_window["roi_edge_width"].toInt(),
                                                  cur_tracking_window["roi_edge"].toInt(),
                                                  slice_image,display_ratio,region_image);
        if(!region_image.isNull())
        {
            QPainter painter(&scaled_image);
            painter.setCompositionMode(QPainter::CompositionMode_Plus);
            painter.drawImage(0,0,region_image);
        }

        if(cur_tracking_window["roi_track"].toInt())
            cur_tracking_window.tractWidget->draw_tracts(cur_dim,
                                                     current_slice->slice_pos[cur_dim],
                                                     scaled_image,display_ratio,uint32_t(cur_tracking_window["roi_track_count"].toInt()));
    }



    if(cur_tracking_window["roi_layout"].toInt() <= 1) // not mosaic
    {
        QPainter painter(&scaled_image);
        if(!simple && cur_tracking_window["roi_fiber"].toInt())
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
            show_ruler(painter2,current_slice,cur_dim);
        if(cur_tracking_window["roi_label"].toInt())
            add_R_label(painter2,current_slice,cur_dim);
    }
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
bool slice_view_scene::command(QString cmd,QString param,QString param2)
{
    if(cmd == "save_roi_image")
    {
        show_slice();
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).absolutePath() + "/" +
                    QFileInfo(cur_tracking_window.windowTitle()).baseName()+"_"+
                    QString(cur_tracking_window.handle->view_item[cur_tracking_window.ui->SliceModality->currentIndex()].name.c_str())+"_"+
                    QString(cur_tracking_window["roi_layout"].toString())+
                    ".jpg";
        QImage output = view_image;
        if(cur_tracking_window["roi_layout"].toInt() > 2 && !param2.isEmpty()) //mosaic
        {
            int cut_row = param2.toInt();
            output = output.copy(QRect(0,cut_row,output.width(),output.height()-cut_row-cut_row));
        }
        output.save(param);
        return true;
    }
    if(cmd == "save_mapping")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).baseName()+"_"+param2+".nii.gz";
        return cur_tracking_window.handle->save_mapping(param2.toStdString(),param.toStdString(),cur_tracking_window.current_slice->v2c);
    }
    return false;
}

bool slice_view_scene::to_3d_space_single_slice(float x,float y,tipl::vector<3,float>& pos)
{
    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    if(cur_tracking_window.slice_view_flip_x(cur_tracking_window.cur_dim))
        x = (cur_tracking_window.cur_dim ? geo[0]:geo[1])-x;
    if(cur_tracking_window.slice_view_flip_y(cur_tracking_window.cur_dim))
        y = geo[2] - y;
    return cur_tracking_window.current_slice->to3DSpace(cur_tracking_window.cur_dim,x - 0.5f,y - 0.5f,pos[0], pos[1], pos[2]);
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
    if(cur_tracking_window["orientation_convention"].toInt())
        painter.drawImage(view_image.width()-view1_w,view1_h,view4);
    else
        painter.drawImage(0,view1_h,view4);
    show_view(*this,view_image);
}

void slice_view_scene::show_slice(void)
{
    if(no_show)
        return;
    need_complete_view = true;
    paint_image(view_image,true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(complete_view_ready)
        show_complete_slice();
    else
        show_view(*this,view_image);
}

void slice_view_scene::show_complete_slice(void)
{
    complete_view_ready = false;
    if(no_show || need_complete_view)
        return;
    view_image = complete_view_image;
    show_view(*this,view_image);
}


void slice_view_scene::paint_image(void)
{
    while(!free_thread)
    {
        if(!no_show && need_complete_view)
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
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void slice_view_scene::paint_image(QImage& out,bool simple)
{
    QImage I;
    auto current_slice = cur_tracking_window.current_slice;
    unsigned char cur_dim = cur_tracking_window.cur_dim;
    float display_ratio = cur_tracking_window.get_scene_zoom(current_slice);

    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
        get_view_image(I,current_slice,cur_dim,display_ratio,simple);
    else
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slices
    {
        QImage view1,view2,view3;
        get_view_image(view1,current_slice,0,display_ratio,simple);
        get_view_image(view2,current_slice,1,display_ratio,simple);
        get_view_image(view3,current_slice,2,display_ratio,simple);
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
        mosaic_row_count = uint32_t(std::ceil(float(dim[cur_dim]/skip)/float(mosaic_column_count)));
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
            int old_z = current_slice->slice_pos[cur_dim];
            unsigned int skip_slices = skip_row*mosaic_column_count;
            for(unsigned int z = 0,slice_pos = skip-1;slice_pos < dim[cur_dim]-skip_slices;++z,slice_pos += skip)
            {
                if(z < skip_slices)
                    continue;
                QImage view;
                current_slice->slice_pos[cur_dim] = int(slice_pos);
                get_view_image(view,current_slice,cur_dim,scale,simple);
                if(z == skip_slices)
                    painter.fillRect(0,0,I.width(),I.height(),view.pixel(0,0));
                int x = int(dim[dim_order[uint8_t(cur_dim)][0]]*((z-skip_slices)%mosaic_column_count));
                int y = int(dim[dim_order[uint8_t(cur_dim)][1]]*((z-skip_slices)/mosaic_column_count));
                x *= scale;
                y *= scale;
                painter.drawImage(QPoint(x,y), view);
            }
            current_slice->slice_pos[cur_dim] = old_z;
        }

        if(cur_tracking_window["roi_label"].toInt()) // not sagittal view
            add_R_label(painter,current_slice,cur_dim);
    }
    out = I;
}



void slice_view_scene::save_slice_as()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    QString filename = QFileDialog::getSaveFileName(
                0,
                "Save as",
                QFileInfo(cur_tracking_window.windowTitle()).baseName()+"_"+
                action->data().toString()+".nii.gz",
                "NIFTI files (*nii.gz *.nii);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_mapping",filename,action->data().toString());
}

void slice_view_scene::catch_screen()
{
    auto* region = cur_tracking_window.regionWidget;
    QString filename = QFileDialog::getSaveFileName(
                0,"Save Images files",
                region->currentRow() >= 0 ?
                    region->item(region->currentRow(),0)->text()+".png" :
                    QFileInfo(cur_tracking_window.windowTitle()).baseName()+"_"+
                    QString(cur_tracking_window.handle->view_item[cur_tracking_window.ui->SliceModality->currentIndex()].name.c_str())+".jpg",
                    "Image files (*.png *.bmp *.jpg);;All files (*)");
        if(filename.isEmpty())
            return;
    command("save_roi_image",filename,cur_tracking_window["roi_layout"].toString());
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
    bool vb_visible = views()[0]->verticalScrollBar()->isVisible();
    bool hb_visible = views()[0]->horizontalScrollBar()->isVisible();
    auto* vb = views()[0]->verticalScrollBar();
    auto* hb = views()[0]->horizontalScrollBar();
    if(vb_visible)
    {
        if((wheelEvent->delta() < 0 && vb->maximum() != vb->value()) ||
            (wheelEvent->delta() > 0 && vb->value() > 0))
            return;
    }
    tipl::vector<3,float> pos;
    float Y = wheelEvent->scenePos().y();
    float X = wheelEvent->scenePos().x();
    if(click_on_3D(X,Y))
    {
        QWheelEvent we(wheelEvent->pos(),wheelEvent->delta(),wheelEvent->buttons(),wheelEvent->modifiers());
        cur_tracking_window.glWidget->wheelEvent(&we);
    }

    float hb_ratio = 0.0f;
    if(hb_visible)
        hb_ratio = float(hb->value())/(hb->maximum());
    if(wheelEvent->delta() < 0)
    {
        cur_tracking_window.set_roi_zoom(cur_tracking_window["roi_zoom"].toFloat()-0.5f);
        if(views().size() > 0 && views()[0]->verticalScrollBar()->isVisible())
            views()[0]->verticalScrollBar()->setValue(views()[0]->verticalScrollBar()->maximum());
    }
    else
    {
        cur_tracking_window.set_roi_zoom(cur_tracking_window["roi_zoom"].toFloat()+0.5f);
        if(views().size() > 0 && views()[0]->verticalScrollBar()->isVisible())
            views()[0]->verticalScrollBar()->setValue(0);
    }
    if(hb_visible)
        hb->setValue(hb->maximum()*hb_ratio);
}
void slice_view_scene::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(clicked_3d)
    {
        send_event_to_3D(QEvent::MouseButtonDblClick,mouseEvent);
        return;
    }

    if (mouseEvent->button() == Qt::MidButton)
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
        QMouseEvent me(QEvent::MouseMove, QPointF(x,y),
            mouseEvent->button(),mouseEvent->buttons(),mouseEvent->modifiers());
        cur_tracking_window.glWidget->mouseMoveEvent(&me);
    }
    if(type == QEvent::MouseButtonPress)
    {
        QMouseEvent me(QEvent::MouseButtonPress, QPointF(x,y),
            mouseEvent->button(),mouseEvent->buttons(),mouseEvent->modifiers());
        cur_tracking_window.glWidget->mousePressEvent(&me);
    }

    if(type == QEvent::MouseButtonRelease)
    {
        QMouseEvent me(QEvent::MouseButtonRelease, QPointF(x,y),
            mouseEvent->button(),mouseEvent->buttons(),mouseEvent->modifiers());
        cur_tracking_window.glWidget->mouseReleaseEvent(&me);

    }
    if(type == QEvent::MouseButtonDblClick)
    {
        QMouseEvent me(QEvent::MouseButtonDblClick, QPointF(x,y),
            mouseEvent->button(),mouseEvent->buttons(),mouseEvent->modifiers());
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
        QMessageBox::information(0,"Error","Switch to regular view to edit ROI. (Right side under Options, change [Region Window][Slice Layout] to Single Slice) ");
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
    if(mouseEvent->button() == Qt::MidButton || sel_mode == 5)// move object
    {
        auto slice = cur_tracking_window.current_slice;
        tipl::vector<3,float> p(pos);
        if(!slice->is_diffusion_space)
            p.to(slice->T);
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
            for(unsigned int index = 0;index <
                cur_tracking_window.regionWidget->regions.size();++index)
                if(cur_tracking_window.regionWidget->item(index,0)->checkState() == Qt::Checked &&
                   cur_tracking_window.regionWidget->regions[index]->has_point(p))
                {
                    find_region = true;
                    cur_tracking_window.regionWidget->selectRow(index);
                    std::vector<tipl::vector<3,float> > dummy;
                    cur_tracking_window.regionWidget->add_points(dummy,true,false); // create a undo point
                    break;
                }
            if(!find_region)
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
                p.to(slice->T);
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
                tipl::vector<3,float> p1(pos),p2(sel_coord.back());
                if(!slice->is_diffusion_space)
                {
                    p1.to(slice->T);
                    p2.to(slice->T);
                }
                p1 -= p2;
                p1.round();
                if(p1.length() != 0)
                {
                    cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()]->shift(p1);
                    p1.to(slice->invT);
                    tipl::vector<3> zero;
                    zero.to(slice->invT);
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
        addPixmap(fromImage(temp));
    }
    else
        addPixmap(fromImage(annotated_image));

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

    tipl::shape<3> geo(cur_tracking_window.current_slice->dim);
    float display_ratio = cur_tracking_window.get_scene_zoom();


    std::vector<tipl::vector<3,float> >points;
    switch (sel_mode)
    {
    case 0: // rectangle
    {
        if (sel_coord.size() < 2)
            return;
        tipl::vector<3,float> min_cood, max_cood;
        for (unsigned int i = 0; i < 3; ++i)
            if (sel_coord[0][i] > sel_coord[1][i])
            {
                max_cood[i] = sel_coord[0][i];
                min_cood[i] = sel_coord[1][i];
            }
            else
            {
                max_cood[i] = sel_coord[1][i];
                min_cood[i] = sel_coord[0][i];
            }
        float dis = 1.0f/display_ratio;
        for (float z = min_cood[2]; z <= max_cood[2]; z += dis)
            for (float y = min_cood[1]; y <= max_cood[1]; y += dis)
                for (float x = min_cood[0]; x <= max_cood[0]; x += dis)
                    if (cur_tracking_window.current_slice->dim.is_valid(x, y, z))
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
                    || !to_3d_space_single_slice((float)index.x()/display_ratio,(float)index.y()/display_ratio,pos))
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

            addPixmap(fromImage(temp));
        }
        else
            addPixmap(fromImage(annotated_image));
        return;
    }
    case 6:
    {
        return;
    }
    }

    float resolution = display_ratio;
    if(!cur_tracking_window.current_slice->is_diffusion_space)
    {
        // resolution difference between DWI and current slices;
        resolution = std::min<float>(16.0f,display_ratio*std::floor(cur_tracking_window.handle->vs[0]/cur_tracking_window.current_slice->vs[0]));
        tipl::par_for(points.size(),[&](int index)
        {
            points[index].to(cur_tracking_window.current_slice->T);
            points[index] *= resolution;
        });
    }
    else
    {
        tipl::par_for(points.size(),[&](int index)
        {
            points[index] *= resolution;
        });
    }

    cur_tracking_window.regionWidget->add_points(points,mouseEvent->button() == Qt::RightButton,false,resolution);
    need_update();
}

void slice_view_scene::center()
{
    QList<QGraphicsView*> views = this->views();
    for(int index = 0;index < views.size();++index)
        views[index]->centerOn(view_image.width()/2,view_image.height()/2);
}



