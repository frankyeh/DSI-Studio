#include "tipl/tipl.hpp"
#include "slice_view_scene.h"
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QSettings>
#include <QFileDialog>
#include <QClipboard>
#include <QMessageBox>
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "region/regiontablewidget.h"
#include "libs/gzip_interface.hpp"
#include "libs/tracking/fib_data.hpp"

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

void slice_view_scene::show_ruler(QPainter& paint)
{
    if(sel_mode != 6 || sel_point.size() < 2)
        return;
    QPen pen;  // creates a default pen
    pen.setWidth(2);
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setColor(Qt::white);
    paint.setPen(pen);
    for(unsigned int index = 1;index < sel_point.size();index += 2)
    {
        float tX = sel_point[index-1][0];
        float tY = sel_point[index-1][1];
        float X = sel_point[index][0];
        float Y = sel_point[index][1];
        paint.drawLine(X, Y, tX, tY);
        tipl::vector<2,float> from(X,Y);
        tipl::vector<2,float> to(tX,tY);
        from -= to;
        float pixel_length = from.length();
        from /= cur_tracking_window.get_scene_zoom();
        from[0] *= cur_tracking_window.current_slice->voxel_size[0];
        from[1] *= cur_tracking_window.current_slice->voxel_size[1];
        from[2] *= cur_tracking_window.current_slice->voxel_size[2];
        float length = from.length();
        float precision = std::pow(10.0,std::floor(std::log10((double)length))-1);
        float tic_dis = std::pow(10.0,std::floor(std::log10((double)50.0*length/pixel_length)));
        if(tic_dis*pixel_length/length < 10)
            tic_dis *= 5.0;

        tipl::vector<2,float> tic_dir(Y-tY,tX-X);
        tic_dir.normalize();
        tic_dir *= 5.0;
        for(double L = 0.0;1;L+=tic_dis)
        {
            if(L+tic_dis > length)
                L = length;
            tipl::vector<2,float> npos(tX,tY);
            npos[0] += ((float)X-npos[0])*L/length;
            npos[1] += ((float)Y-npos[1])*L/length;
            paint.drawLine(npos[0],npos[1],npos[0]+tic_dir[0],npos[1]+tic_dir[1]);
            npos += tic_dir;
            npos += tic_dir;
            npos += tic_dir;
            if(L < length)
                paint.drawText(npos[0]-40,npos[1]-40,80,80,
                               Qt::AlignHCenter|Qt::AlignVCenter,
                               QString::number(L));
            else
            {
                paint.drawText(npos[0]-40,npos[1]-40,80,80,
                               Qt::AlignHCenter|Qt::AlignVCenter,
                               QString::number(std::round(L*10.0/precision)*precision/10.0)+" mm");
                break;
            }

        }
    }
}
void slice_view_scene::show_fiber(QPainter& painter)
{
    int roi_fiber = cur_tracking_window["roi_fiber"].toInt();
    float threshold = cur_tracking_window.get_fa_threshold();
    float threshold2 = cur_tracking_window["dt_index"].toInt() ? cur_tracking_window["dt_threshold"].toFloat() : 0.0f;
    if (threshold == 0.0f)
        threshold = 0.00000001f;
    int X,Y,Z;
    float display_ratio = cur_tracking_window.get_scene_zoom();
    const char dir_x[3] = {1,0,0};
    const char dir_y[3] = {2,2,1};

    int fiber_color = cur_tracking_window["roi_fiber_color"].toInt();
    float pen_w = display_ratio * cur_tracking_window["roi_fiber_width"].toFloat();
    float r = display_ratio * cur_tracking_window["roi_fiber_length"].toFloat();
    if(fiber_color)
    {
        QPen pen(QColor(fiber_color == 1 ? 255:0,fiber_color == 2 ? 255:0,fiber_color == 3 ? 255:0));
        pen.setWidthF(pen_w);
        painter.setPen(pen);
    }
    const fib_data& fib = *(cur_tracking_window.handle);
    char max_fiber = fib.dir.num_fiber-1;
    int steps = 1;
    if(!cur_tracking_window.current_slice->is_diffusion_space)
    {
        steps = std::ceil(cur_tracking_window.handle->vs[0]/cur_tracking_window.current_slice->voxel_size[0]);
        r *= steps;
        pen_w *= steps;
    }
    for (int y = 0; y < slice_image.height(); y += steps)
        for (int x = 0; x < slice_image.width(); x += steps)
            {
                cur_tracking_window.current_slice->toDiffusionSpace(cur_tracking_window.cur_dim,x, y, X, Y, Z);
                if(!cur_tracking_window.handle->dim.is_valid(X,Y,Z))
                    continue;
                tipl::pixel_index<3> pos(X,Y,Z,fib.dim);
                if (pos.index() >= fib.dim.size() || fib.dir.get_fa(pos.index(),0) == 0.0)
                    continue;
                for (char fiber = max_fiber; fiber >= 0; --fiber)
                    if(fib.dir.get_fa(pos.index(),fiber) > threshold)
                    {
                        if(threshold2 != 0.0f && fib.dir.get_dt_fa(pos.index(),fiber) < threshold2)
                            continue;
                        if((roi_fiber == 2 && fiber != 0) ||
                           (roi_fiber == 3 && fiber != 1))
                            continue;
                        const float* dir_ptr = fib.dir.get_dir(pos.index(),fiber);
                        if(!fiber_color)
                        {
                            QPen pen(QColor(std::abs(dir_ptr[0]) * 255.0,std::abs(dir_ptr[1]) * 255.0, std::abs(dir_ptr[2]) * 255.0));
                            pen.setWidthF(pen_w);
                            painter.setPen(pen);
                        }
                        float dx = r * dir_ptr[dir_x[cur_tracking_window.cur_dim]] + 0.5;
                        float dy = r * dir_ptr[dir_y[cur_tracking_window.cur_dim]] + 0.5;
                        painter.drawLine(
                            display_ratio*((float)x + 0.5) - dx,
                            display_ratio*((float)y + 0.5) - dy,
                            display_ratio*((float)x + 0.5) + dx,
                            display_ratio*((float)y + 0.5) + dy);
                    }
            }
}
void slice_view_scene::show_pos(QPainter& painter)
{
    int x_pos,y_pos;
    float display_ratio = cur_tracking_window.get_scene_zoom();
    cur_tracking_window.current_slice->get_other_slice_pos(cur_tracking_window.cur_dim,x_pos, y_pos);
    x_pos = ((double)x_pos + 0.5)*display_ratio;
    y_pos = ((double)y_pos + 0.5)*display_ratio;
    painter.setPen(QColor(255,0,0));
    painter.drawLine(x_pos,0,x_pos,std::max<int>(0,y_pos-20));
    painter.drawLine(x_pos,std::min<int>(y_pos+20,slice_image.height()*display_ratio),x_pos,slice_image.height()*display_ratio);
    painter.drawLine(0,y_pos,std::max<int>(0,x_pos-20),y_pos);
    painter.drawLine(std::min<int>(x_pos+20,slice_image.width()*display_ratio),y_pos,slice_image.width()*display_ratio,y_pos);
}

void slice_view_scene::get_view_image(QImage& new_view_image)
{
    float display_ratio = cur_tracking_window.get_scene_zoom();
    cur_tracking_window.current_slice->get_slice(slice_image,
                                                 cur_tracking_window.cur_dim,
                                                 cur_tracking_window.overlay_slices);
    // draw region colors on the image
    tipl::color_image slice_image_with_region(slice_image);
    if(!cur_tracking_window["roi_edge"].toInt())
        cur_tracking_window.regionWidget->draw_region(slice_image_with_region);
    QImage qimage((unsigned char*)&*slice_image_with_region.begin(),slice_image_with_region.width(),slice_image_with_region.height(),QImage::Format_RGB32);
    // make sure that qimage get a hard copy
    qimage.setPixel(0,0,qimage.pixel(0,0));
    QImage scaled_image = qimage.scaled(slice_image.width()*display_ratio,slice_image.height()*display_ratio);

    cur_tracking_window.regionWidget->draw_edge(qimage,scaled_image,
                            cur_tracking_window["roi_edge"].toInt());

    QPainter painter(&scaled_image);
    if(cur_tracking_window["roi_fiber"].toInt())
        show_fiber(painter);
    if(cur_tracking_window["roi_position"].toInt())
        show_pos(painter);



    bool flip_x = false;
    bool flip_y = false;
    if(cur_tracking_window.cur_dim != 2)
        flip_y = true;
    if(cur_tracking_window["orientation_convention"].toInt())
        flip_x = true;

    new_view_image = (!flip_x && !flip_y ? scaled_image : scaled_image.mirrored(flip_x,flip_y));

    if(cur_tracking_window.cur_dim && cur_tracking_window["roi_label"].toInt())
    {
        QPainter painter(&new_view_image);
        painter.setPen(QPen(QColor(255,255,255)));
        QFont f = font();  // start out with MainWindow's font..
        f.setPointSize(cur_tracking_window.get_scene_zoom()+10); // and make a bit smaller for legend
        painter.setFont(f);
        painter.drawText(5,5,new_view_image.width()-10,new_view_image.height()-10,
                         cur_tracking_window["orientation_convention"].toInt() ? Qt::AlignTop|Qt::AlignRight: Qt::AlignTop|Qt::AlignLeft,"R");
    }
}
bool slice_view_scene::command(QString cmd,QString param,QString param2)
{
    if(cmd == "save_roi_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).absolutePath() + "/" +
                    QFileInfo(cur_tracking_window.windowTitle()).baseName()+"_"+
                    QString(cur_tracking_window.handle->view_item[cur_tracking_window.ui->SliceModality->currentIndex()].name.c_str())+"_"+
                    QString(cur_tracking_window["roi_layout"].toString())+
                    ".jpg";
        if(param2 != "0")// mosaic
        {
            view_image.save(param);
            return true;
        }
        QImage output = view_image;
        QPainter paint(&output);
        show_ruler(paint);
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
    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
    if(cur_tracking_window["orientation_convention"].toInt())
        x = (cur_tracking_window.cur_dim ? geo[0]:geo[1])-x;
    if(cur_tracking_window.cur_dim != 2)
        y = geo[2] - y;
    return cur_tracking_window.current_slice->to3DSpace(cur_tracking_window.cur_dim,x - 0.5f,y - 0.5f,pos[0], pos[1], pos[2]);
}

bool slice_view_scene::to_3d_space(float x,float y,tipl::vector<3,float>& pos)
{
    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
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
    if(cur_tracking_window["orientation_convention"].toInt())
        return false;
    pos[0] = x*(float)mosaic_size;
    pos[1] = y*(float)mosaic_size;
    pos[2] = std::floor(pos[1]/geo[1])*mosaic_size +
             std::floor(pos[0]/geo[0]);
    pos[0] -= std::floor(pos[0]/geo[0])*geo[0];
    pos[1] -= std::floor(pos[1]/geo[1])*geo[1];
    return geo.is_valid(pos);
}

void slice_view_scene::show_slice(void)
{
    if(no_show)
        return;
    float display_ratio = cur_tracking_window.get_scene_zoom();

    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
        get_view_image(view_image);
    else
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slices
    {
        unsigned char old_dim = cur_tracking_window.cur_dim;
        QImage view1,view2,view3;
        cur_tracking_window.cur_dim = 0;
        get_view_image(view1);
        cur_tracking_window.cur_dim = 1;
        get_view_image(view2);
        cur_tracking_window.cur_dim = 2;
        get_view_image(view3);
        cur_tracking_window.cur_dim = old_dim;
        view_image = QImage(QSize(view1.width()+view2.width(),view1.height()+view3.height()),QImage::Format_RGB32);
        QPainter painter(&view_image);
        painter.fillRect(0,0,view_image.width(),view_image.height(),QColor(0,0,0));
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
        pen.setWidthF(std::max(1.0,display_ratio/4.0));
        painter.setPen(pen);
        painter.drawLine(cur_tracking_window["orientation_convention"].toInt() ? view2.width() : view1.width(),0,
                         cur_tracking_window["orientation_convention"].toInt() ? view2.width() : view1.width(),view_image.height());
        painter.drawLine(0,view1.height(),view_image.width(),view1.height());
    }
    else
    {
        unsigned int skip = cur_tracking_window["roi_layout"].toInt()-1;
        mosaic_size = std::max((int)1,(int)std::ceil(std::sqrt((float)(cur_tracking_window.current_slice->geometry[2] / skip))));


        {
            auto geometry = cur_tracking_window.current_slice->geometry;
            unsigned slice_num = geometry[2] / skip;
            mosaic_image = std::move(tipl::color_image(tipl::geometry<2>(geometry[0]*mosaic_size,
                                                  geometry[1]*(std::ceil((float)slice_num/(float)mosaic_size)))));
            int old_z = cur_tracking_window.current_slice->slice_pos[2];
            for(unsigned int z = 0;z < slice_num;++z)
            {
                cur_tracking_window.current_slice->slice_pos[2] = z*skip;
                tipl::color_image slice_image;
                cur_tracking_window.current_slice->get_slice(slice_image,2,cur_tracking_window.overlay_slices);

                cur_tracking_window.regionWidget->draw_region(slice_image);
                tipl::vector<2,int> pos(geometry[0]*(z%mosaic_size),
                                         geometry[1]*(z/mosaic_size));
                tipl::draw(slice_image,mosaic_image,pos);
            }
            cur_tracking_window.current_slice->slice_pos[2] = old_z;
        }

        QImage qimage((unsigned char*)&*mosaic_image.begin(),mosaic_image.width(),mosaic_image.height(),QImage::Format_RGB32);
        view_image = qimage.scaled(mosaic_image.width()*display_ratio/(float)mosaic_size,mosaic_image.height()*display_ratio/(float)mosaic_size);
        if(cur_tracking_window["orientation_convention"].toInt())
        {
            QImage I = view_image;
            view_image = I.mirrored(true,false);
        }
    }
    show_view(*this,view_image);
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
    QPainter paint(&output);
    show_ruler(paint);
    QApplication::clipboard()->setImage(output);
}


void slice_view_scene::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
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
    if (mouseEvent->button() == Qt::RightButton) // move to slice
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
    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
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

void slice_view_scene::mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window["roi_layout"].toInt() > 1)// mosaic
    {
        QMessageBox::information(0,"Error","Switch to regular view to edit ROI. (Right side under Options, change [Region Window][Slice Layout] to Single Slice) ");
        return;
    }
    if(mouseEvent->button() == Qt::MidButton)
    {
        mid_down = true;
        return;
    }
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

    tipl::vector<3,float> pos;
    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();
    if(!to_3d_space(X,Y,pos))
        return;
    adjust_xy_to_layout(X,Y);
    if(sel_mode == 5)// move object
    {
        bool find_region = false;
        tipl::vector<3,float> p(pos);
        if(!cur_tracking_window.current_slice->is_diffusion_space)
            p.to(cur_tracking_window.current_slice->T);
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
            return;
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
    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
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
    if(cur_tracking_window["roi_layout"].toInt() > 1)// single slice
        return;

    if (!mouse_down && !mid_down)
        return;
    if(mid_down)
    {
        return;
    }
    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
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

    if(sel_mode == 5 && !cur_tracking_window.regionWidget->regions.empty()) // move object
    {
        if(!sel_coord.empty() && pos != sel_coord.back())
        {
            tipl::vector<3,float> p1(pos),p2(sel_coord.back());
            if(!cur_tracking_window.current_slice->is_diffusion_space)
            {
                p1.to(cur_tracking_window.current_slice->T);
                p2.to(cur_tracking_window.current_slice->T);
            }
            p1 -= p2;
            p1.round();
            if(p1.length() != 0)
            {
                cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()]->shift(p1);
                p1.to(cur_tracking_window.current_slice->invT);
                tipl::vector<3> zero;
                zero.to(cur_tracking_window.current_slice->invT);
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
        show_ruler(paint);
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
    if (!mouse_down && !mid_down)
        return;
    mouse_down = false;
    if(mid_down || sel_mode == 5)
    {
        sel_point.clear();
        sel_coord.clear();
        mid_down = false;
        return;
    }

    tipl::geometry<3> geo(cur_tracking_window.current_slice->geometry);
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
                    if (cur_tracking_window.current_slice->geometry.is_valid(x, y, z))
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
            tipl::geometry<2> geo2(bitmap.width(),bitmap.height());
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
            for (float y = -dis; y <= dis; y += interval)
                for (float x = -dis; x <= dis; x += interval)
                    if (cur_tracking_window.current_slice->geometry.is_valid(sel_coord[0][0] + x,
                                                 sel_coord[0][1] + y, sel_coord[0][2] + z) && x*x +
                            y*y + z*z <= distance2)
                        points.push_back(tipl::vector<3,float>(sel_coord[0][0] + x,
                                            sel_coord[0][1] + y, sel_coord[0][2] + z));
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
        resolution = std::min<float>(16.0f,display_ratio*std::floor(cur_tracking_window.handle->vs[0]/cur_tracking_window.current_slice->voxel_size[0]));
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

    cur_tracking_window.regionWidget->add_points(points,mouseEvent->button() == Qt::RightButton,cur_tracking_window.ui->all_edit->isChecked(),resolution);
    need_update();
}

void slice_view_scene::center()
{
    QList<QGraphicsView*> views = this->views();
    for(int index = 0;index < views.size();++index)
        views[index]->centerOn(view_image.width()/2,view_image.height()/2);
}



