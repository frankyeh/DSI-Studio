#include "image/image.hpp"
#include "slice_view_scene.h"
#include "tracking_static_link.h"
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QFileDialog>
#include <QClipboard>
#include <QMessageBox>
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "region/regiontablewidget.h"
#include "libs/gzip_interface.hpp"

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
        image::vector<2,float> from(X,Y);
        image::vector<2,float> to(tX,tY);
        from -= to;
        float pixel_length = from.length();
        from /= cur_tracking_window["roi_zoom"].toInt();
        from[0] *= cur_tracking_window.handle->fib_data.vs[0];
        from[1] *= cur_tracking_window.handle->fib_data.vs[1];
        from[2] *= cur_tracking_window.handle->fib_data.vs[2];
        float length = from.length();
        float precision = std::pow(10.0,std::floor(std::log10((double)length))-1);
        float tic_dis = std::pow(10.0,std::floor(std::log10((double)50.0*length/pixel_length)));
        if(tic_dis*pixel_length/length < 10)
            tic_dis *= 5.0;

        image::vector<2,float> tic_dir(Y-tY,tX-X);
        tic_dir.normalize();
        tic_dir *= 5.0;
        for(double L = 0.0;1;L+=tic_dis)
        {
            if(L+tic_dis > length)
                L = length;
            image::vector<2,float> npos(tX,tY);
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
                               QString::number(std::floor(L*10.0/precision+0.5)*precision/10.0)+" mm");
                break;
            }

        }
    }
}
void slice_view_scene::show_fiber(QPainter& painter)
{
    float threshold = cur_tracking_window["fa_threshold"].toFloat();
    if (threshold == 0.0)
        threshold = 0.00000001;
    int X,Y,Z;
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();
    float r = display_ratio /  3.0;
    float pen_w = display_ratio /  5.0;
    const char dir_x[3] = {1,0,0};
    const char dir_y[3] = {2,2,1};

    const FibData& fib_data = handle->fib_data;
    for (unsigned int y = 0; y < slice_image.height(); ++y)
        for (unsigned int x = 0; x < slice_image.width(); ++x)
            if (cur_tracking_window.slice.get3dPosition(x, y, X, Y, Z))
            {
                image::pixel_index<3> pos(X,Y,Z,fib_data.dim);
                if (pos.index() >= fib_data.total_size || fib_data.fib.getFA(pos.index(),0) == 0.0)
                    continue;
                for (char fiber = 2; fiber >= 0; --fiber)
                    if(fib_data.fib.getFA(pos.index(),fiber) > threshold)
                    {
                        const float* dir_ptr = fib_data.fib.getDir(pos.index(),fiber);
                        QPen pen(QColor(std::abs(dir_ptr[0]) * 255.0,std::abs(dir_ptr[1]) * 255.0, std::abs(dir_ptr[2]) * 255.0));
                        pen.setWidthF(pen_w);
                        painter.setPen(pen);
                        float dx = r * dir_ptr[dir_x[cur_tracking_window.slice.cur_dim]] + 0.5;
                        float dy = r * dir_ptr[dir_y[cur_tracking_window.slice.cur_dim]] + 0.5;
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
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();
    cur_tracking_window.slice.get_other_slice_pos(x_pos, y_pos);
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
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();
    float contrast = cur_tracking_window.ui->contrast_value->value();
    float offset = cur_tracking_window.ui->offset_value->value();
    cur_tracking_window.slice.get_slice(slice_image,contrast,offset);
    QImage qimage((unsigned char*)&*slice_image.begin(),slice_image.width(),slice_image.height(),QImage::Format_RGB32);
    // draw region colors on the image
    cur_tracking_window.regionWidget->draw_region(qimage);
    QImage scaled_image = qimage.scaled(slice_image.width()*display_ratio,slice_image.height()*display_ratio);

    QPainter painter(&scaled_image);

    if(cur_tracking_window["roi_fiber"].toInt())
        show_fiber(painter);

    if(cur_tracking_window["roi_position"].toInt())
        show_pos(painter);


    bool flip_x = false;
    bool flip_y = false;
    if(cur_tracking_window.slice.cur_dim != 2)
        flip_y = true;
    if(cur_tracking_window["orientation_convention"].toInt())
        flip_x = true;

    new_view_image = (!flip_x && !flip_y ? scaled_image : scaled_image.mirrored(flip_x,flip_y));

    if(cur_tracking_window.slice.cur_dim && cur_tracking_window["roi_label"].toInt())
    {
        QPainter painter(&new_view_image);
        painter.setPen(QPen(QColor(255,255,255)));
        QFont f = font();  // start out with MainWindow's font..
        f.setPointSize(cur_tracking_window["roi_zoom"].toInt()+10); // and make a bit smaller for legend
        painter.setFont(f);
        painter.drawText(5,5,new_view_image.width()-10,new_view_image.height()-10,
                         cur_tracking_window["orientation_convention"].toInt() ? Qt::AlignTop|Qt::AlignRight: Qt::AlignTop|Qt::AlignLeft,"R");
    }
}
bool slice_view_scene::get_location(float x,float y,image::vector<3,float>& pos)
{
    image::geometry<3> geo(cur_tracking_window.slice.geometry);
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();
    x /= display_ratio;
    y /= display_ratio;
    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
    {
        if(cur_tracking_window["orientation_convention"].toInt())
            x = (cur_tracking_window.slice.cur_dim ? geo[0]:geo[1])-x;
        if(cur_tracking_window.slice.cur_dim != 2)
            y = geo[2] - y;
        return cur_tracking_window.slice.get3dPosition(x - 0.5,y - 0.5,pos[0], pos[1], pos[2]);
    }
    else
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slice
    {

        if(mouse_down)
        {
            if(cur_tracking_window["orientation_convention"].toInt())
            {
                if(cur_tracking_window.slice.cur_dim == 0)
                    x -= geo[0];
            }
            else
                if(cur_tracking_window.slice.cur_dim)
                    x -= geo[1];
            if(cur_tracking_window.slice.cur_dim == 2)
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
            cur_tracking_window.slice.cur_dim = new_dim;
        }

        if(cur_tracking_window["orientation_convention"].toInt())
            x = (cur_tracking_window.slice.cur_dim ? geo[0]:geo[1])-x;
        if(cur_tracking_window.slice.cur_dim != 2)
            y = geo[2] - y;
        return cur_tracking_window.slice.get3dPosition(x - 0.5,y - 0.5,pos[0], pos[1], pos[2]);
    }
    else
    {
        if(cur_tracking_window["orientation_convention"].toInt())
            return false;
        pos[0] = x*(float)mosaic_size;
        pos[1] = y*(float)mosaic_size;
        pos[2] = std::floor(pos[1]/geo[1])*mosaic_size +
                 std::floor(pos[0]/geo[0]);
        pos[0] -= std::floor(pos[0]/geo[0])*geo[0];
        pos[1] -= std::floor(pos[1]/geo[1])*geo[1];
    }
    return geo.is_valid(pos);
}

void slice_view_scene::show_slice(void)
{
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();

    if(cur_tracking_window["roi_layout"].toInt() == 0)// single slice
        get_view_image(view_image);
    else
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slices
    {
        unsigned char old_dim = cur_tracking_window.slice.cur_dim;
        QImage view1,view2,view3;
        cur_tracking_window.slice.cur_dim = 0;
        get_view_image(view1);
        cur_tracking_window.slice.cur_dim = 1;
        get_view_image(view2);
        cur_tracking_window.slice.cur_dim = 2;
        get_view_image(view3);
        cur_tracking_window.slice.cur_dim = old_dim;
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
        float contrast = cur_tracking_window.ui->contrast_value->value();
        float offset = cur_tracking_window.ui->offset_value->value();
        unsigned int skip = cur_tracking_window["roi_layout"].toInt()-2;
        mosaic_size = std::max((int)1,(int)std::ceil(std::sqrt((float)(cur_tracking_window.slice.geometry[2] >> skip))));
        cur_tracking_window.slice.get_mosaic(mosaic_image,mosaic_size,contrast,offset,skip);
        QImage qimage((unsigned char*)&*mosaic_image.begin(),mosaic_image.width(),mosaic_image.height(),QImage::Format_RGB32);
        cur_tracking_window.regionWidget->draw_mosaic_region(qimage,mosaic_size,skip);
        view_image = qimage.scaled(mosaic_image.width()*display_ratio/(float)mosaic_size,mosaic_image.height()*display_ratio/(float)mosaic_size);
        if(cur_tracking_window["orientation_convention"].toInt())
        {
            QImage I = view_image;
            view_image = I.mirrored(true,false);
        }
    }
    setSceneRect(0, 0, view_image.width(),view_image.height());
    clear();
    setItemIndexMethod(QGraphicsScene::NoIndex);
    addRect(0, 0, view_image.width(),view_image.height(),QPen(),view_image);
    // clear point buffer
    if(sel_mode != 5) // move object need the selection record
    {
        sel_point.clear();
        sel_coord.clear();
    }
}

void slice_view_scene::save_slice_as()
{
    if( cur_tracking_window.ui->sliceViewBox->currentText() == "color")
        return;
    QString filename = QFileDialog::getSaveFileName(
                0,
                "Save as",
                cur_tracking_window.get_path("slice") + "/" +
                cur_tracking_window.handle->fib_data.view_item[cur_tracking_window.ui->sliceViewBox->currentIndex()].name.c_str(),
                "NIFTI files (*.nii.gz);;MAT files (*.mat);;All files (*)");
    if(filename.isEmpty())
        return;
#ifdef __APPLE__
// fix the Qt double extension bug here
if(QFileInfo(filename).completeSuffix() == "nii.gz")
    filename = QFileInfo(filename).absolutePath() + QFileInfo(filename).baseName() + ".nii.gz";
#endif

    cur_tracking_window.add_path("region",filename);

    int index = cur_tracking_window.handle->get_name_index(
                cur_tracking_window.ui->sliceViewBox->currentText().toLocal8Bit().begin());
    if(index >= cur_tracking_window.handle->fib_data.view_item.size())
        return;

    if(QFileInfo(filename).completeSuffix().toLower() == "nii" ||
            QFileInfo(filename).completeSuffix().toLower() == "nii.gz")
    {
        image::basic_image<float,3> buf(cur_tracking_window.handle->fib_data.view_item[index].image_data);
        gz_nifti file;
        file.set_voxel_size(cur_tracking_window.slice.voxel_size.begin());
        if(cur_tracking_window.is_qsdr) //QSDR condition
        {
            file.set_image_transformation(cur_tracking_window.handle->fib_data.trans_to_mni.begin());
            file << cur_tracking_window.handle->fib_data.view_item[index].image_data;
        }
        else
        {
            image::flip_xy(buf);
            file << buf;
        }
        file.save_to_file(filename.toLocal8Bit().begin());

    }
    if(QFileInfo(filename).completeSuffix().toLower() == "mat")
    {
        image::io::mat_write file(filename.toLocal8Bit().begin());
        file << cur_tracking_window.handle->fib_data.view_item[index].image_data;
    }

}

void slice_view_scene::catch_screen()
{
    QString filename = QFileDialog::getSaveFileName(
            0,"Save Images files",
            cur_tracking_window.absolute_path,
            "PNG files (*.png);;BMP files (*.bmp);;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*)");
    if(filename.isEmpty())
        return;

    if(cur_tracking_window["roi_layout"].toInt() != 0)// mosaic
    {
        view_image.save(filename);
        return;
    }
    bool flip_x = false;
    bool flip_y = false;
    if(cur_tracking_window.slice.cur_dim != 2)
        flip_y = true;
    if(cur_tracking_window["orientation_convention"].toInt())
        flip_x = true;

    QImage output = view_image;
    QPainter paint(&view_image);
    show_ruler(paint);
    output.save(filename);
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
        image::vector<3,float> pos;
        if(!get_location(X,Y,pos))
            return;
        pos += 0.5;
        pos.floor();
        cur_tracking_window.ui->SagSlider->setValue(pos[0]);
        cur_tracking_window.ui->CorSlider->setValue(pos[1]);
        cur_tracking_window.ui->AxiSlider->setValue(pos[2]);
    }
}

void slice_view_scene::mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window["roi_layout"].toInt() > 1)// mosaic
        return;

    if(mouseEvent->button() == Qt::MidButton)
    {
        mid_down = true;
        return;
    }
    cur_tracking_window.copy_target = 1;
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

    int x, y, z;
    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();

    image::geometry<3> geo(cur_tracking_window.slice.geometry);


    {
        image::vector<3,float> pos;
        unsigned char old_dim = cur_tracking_window.slice.cur_dim;
        if(!get_location(X,Y,pos))
            return;
        x = std::floor(pos[0]+0.5);
        y = std::floor(pos[1]+0.5);
        z = std::floor(pos[2]+0.5);
        if(old_dim != cur_tracking_window.slice.cur_dim)
        {
            sel_point.clear();
            sel_coord.clear();
        }
    }

    if(cur_tracking_window["roi_layout"].toInt() == 1)
    {
        if(cur_tracking_window["orientation_convention"].toInt())
        {
            if(cur_tracking_window.slice.cur_dim == 0)
                X -= geo[0]*display_ratio;
        }
        else
            if(cur_tracking_window.slice.cur_dim >= 1)
                X -= geo[1]*display_ratio;
        if(cur_tracking_window.slice.cur_dim == 2)
            Y -= geo[2]*display_ratio;
    }

    if(sel_mode == 5)// move object
    {
        bool find_region = false;
        image::vector<3,short> cur_point(x, y, z);
        for(unsigned int index = 0;index <
            cur_tracking_window.regionWidget->regions.size();++index)
            if(cur_tracking_window.regionWidget->regions[index].has_point(cur_point))
            {
                find_region = true;
                cur_tracking_window.regionWidget->selectRow(index);
                break;
            }
        if(!find_region)
            return;
    }

    if(sel_mode != 4)
    {
        if(sel_mode != 6)
        {
            sel_point.clear();
            sel_coord.clear();
        }
        sel_point.push_back(image::vector<2,short>(X, Y));
        sel_coord.push_back(image::vector<3,short>(x, y, z));
    }

    switch (sel_mode)
    {
    case 0:
    case 2:
    case 3:
    case 4:
    case 6:
        sel_coord.push_back(image::vector<3,short>(x, y, z));
        sel_point.push_back(image::vector<2,short>(X, Y));
        break;
    default:
        break;
    }
    mouse_down = true;
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
    image::geometry<3> geo(cur_tracking_window.slice.geometry);
    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();


    cX = X;
    cY = Y;
    int x, y, z;
    if (!mouse_down || sel_mode == 4)
        return;

    {
        image::vector<3> pos;
        this->get_location(cX,cY,pos);
        x = std::floor(pos[0]+0.5);
        y = std::floor(pos[1]+0.5);
        z = std::floor(pos[2]+0.5);
    }

    if(cur_tracking_window["roi_layout"].toInt() == 1)
    {
        if(cur_tracking_window["orientation_convention"].toInt())
        {
            if(cur_tracking_window.slice.cur_dim == 0)
                X -= geo[0]*display_ratio;
        }
        else
            if(cur_tracking_window.slice.cur_dim >= 1)
                X -= geo[1]*display_ratio;
        if(cur_tracking_window.slice.cur_dim == 2)
            Y -= geo[2]*display_ratio;
    }

    if(sel_mode == 5 && !cur_tracking_window.regionWidget->regions.empty()) // move object
    {
        image::vector<3,short> cur_point(x, y, z);
        if(!sel_coord.empty() && cur_point != sel_coord.back())
        {
            cur_point -= sel_coord.back();
            cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()].shift(cur_point);
            sel_coord.back() += cur_point;
            emit need_update();
        }
        return;
    }

    if(cur_tracking_window["roi_layout"].toInt() == 0)
        annotated_image = view_image;
    else
    {
        annotated_image = view_image.copy(
                        cur_tracking_window["orientation_convention"].toInt() ?
                            (cur_tracking_window.slice.cur_dim != 0 ? 0:geo[0]*display_ratio):
                            (cur_tracking_window.slice.cur_dim == 0 ? 0:geo[1]*display_ratio),
                        cur_tracking_window.slice.cur_dim != 2 ? 0:geo[2]*display_ratio,
                        cur_tracking_window.slice.cur_dim == 0 ? geo[1]*display_ratio:geo[0]*display_ratio,
                        cur_tracking_window.slice.cur_dim != 2 ? geo[2]*display_ratio:geo[1]*display_ratio);
    }

    if(sel_mode == 6) // ruler
    {
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
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
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
        break;
    case 1: //free hand
        sel_coord.push_back(image::vector<3,short>(x, y, z));
        sel_point.push_back(image::vector<2,short>(X, Y));

        for (unsigned int index = 1; index < sel_point.size(); ++index)
            paint.drawLine(sel_point[index-1][0], sel_point[index-1][1],sel_point[index][0], sel_point[index][1]);
        break;
    case 2:
    {
        int dx = X - sel_point.front()[0];
        int dy = Y - sel_point.front()[1];
        int dis = std::sqrt((double)(dx * dx + dy * dy));
        paint.drawEllipse(QPoint(sel_point.front()[0],sel_point.front()[1]),dis,dis);
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
    }
    break;
    case 3:
    {
        int dx = X - sel_point.front()[0];
        int dy = Y - sel_point.front()[1];
        int dis = std::sqrt((double)(dx * dx + dy * dy));
        paint.drawRect(sel_point.front()[0] - dis,
                        sel_point.front()[1] - dis, dis*2, dis*2);
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
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
                        (cur_tracking_window.slice.cur_dim != 0 ? 0:geo[0]*display_ratio):
                        (cur_tracking_window.slice.cur_dim == 0 ? 0:geo[1]*display_ratio),
                        cur_tracking_window.slice.cur_dim != 2 ? 0:geo[2]*display_ratio,annotated_image);
        addRect(0, 0, temp.width(),temp.height(),QPen(),temp);
    }
    else
        addRect(0, 0, annotated_image.width(),annotated_image.height(),QPen(),annotated_image);

}


void slice_view_scene::mouseReleaseEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window["roi_layout"].toInt() > 1)
        return;

    if (!mouse_down && !mid_down)
        return;
    if(mid_down || sel_mode == 5)
    {
        mid_down = false;
        return;
    }
    mouse_down = false;
    image::geometry<3> geo(cur_tracking_window.slice.geometry);
    float display_ratio = cur_tracking_window["roi_zoom"].toInt();


    std::vector<image::vector<3,short> >points;
    switch (sel_mode)
    {
    case 0: // rectangle
    {
        image::vector<3,short>min_cood, max_cood;
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
        for (int z = min_cood[2]; z <= max_cood[2]; ++z)
            for (int y = min_cood[1]; y <= max_cood[1]; ++y)
                for (int x = min_cood[0]; x <= max_cood[0]; ++x)
                    if (cur_tracking_window.slice.geometry.is_valid(x, y, z))
                        points.push_back(image::vector<3,short>(x, y, z));
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
            QImage bitmap(cur_tracking_window.slice.cur_dim == 0 ? geo[1]:geo[0],
                          cur_tracking_window.slice.cur_dim != 2 ? geo[2]:geo[1],QImage::Format_Mono);
            QPainter paint(&bitmap);
            paint.setBrush(Qt::black);
            paint.drawRect(0,0,bitmap.width(),bitmap.height());
            paint.setBrush(Qt::white);
            std::vector<QPoint> qpoints(sel_point.size());
            for(unsigned int index = 0;index < sel_point.size();++index)
                qpoints[index] = QPoint(
                            !cur_tracking_window["orientation_convention"].toInt() ?
                                sel_point[index][0]/display_ratio:(cur_tracking_window.slice.cur_dim ? geo[0]:geo[1]) - sel_point[index][0]/display_ratio,
                            cur_tracking_window.slice.cur_dim == 2 ?
                                sel_point[index][1]/display_ratio:geo[2] - sel_point[index][1]/display_ratio);
            paint.drawPolygon(&*qpoints.begin(),qpoints.size() - 1);
            image::geometry<2> geo2(bitmap.width(),bitmap.height());
            int x, y, z;
            for (image::pixel_index<2>index; index.is_valid(geo2);index.next(geo2))
            {
                if (QColor(bitmap.pixel(index.x(),index.y())).red() < 64
                    || !cur_tracking_window.slice.get3dPosition(index.x(),index.y(),x, y, z))
                    continue;
                points.push_back(image::vector<3,short>(x, y, z));
            }
        }
    }
    break;
    case 2:
    {
        int dx = sel_coord[1][0] - sel_coord[0][0];
        int dy = sel_coord[1][1] - sel_coord[0][1];
        int dz = sel_coord[1][2] - sel_coord[0][2];
        int distance2 = (dx * dx + dy * dy + dz * dz);
        int dis = std::sqrt((double)distance2);
        for (int z = -dis; z <= dis; ++z)
            for (int y = -dis; y <= dis; ++y)
                for (int x = -dis; x <= dis; ++x)
                    if (cur_tracking_window.slice.geometry.is_valid(sel_coord[0][0] + x,
                                                 sel_coord[0][1] + y, sel_coord[0][2] + z) && x*x +
                            y*y + z*z <= distance2)
                        points.push_back(image::vector<3,short>(sel_coord[0][0] + x,
                                            sel_coord[0][1] + y, sel_coord[0][2] + z));
    }
    break;
    case 3:
    {
        int dx = sel_coord[1][0] - sel_coord[0][0];
        int dy = sel_coord[1][1] - sel_coord[0][1];
        int dz = sel_coord[1][2] - sel_coord[0][2];
        int distance2 = (dx * dx + dy * dy + dz * dz);
        int dis = std::sqrt((double)distance2);
        for (int z = -dis; z <= dis; ++z)
            for (int y = -dis; y <= dis; ++y)
                for (int x = -dis; x <= dis; ++x)
                    points.push_back(image::vector<3,short>(sel_coord[0][0] + x,
                                                        sel_coord[0][1] + y, sel_coord[0][2] + z));

    }
    break;
    case 4:
    {
        if(cur_tracking_window["roi_layout"].toInt() == 0)
            annotated_image = view_image;
        else
        {
            annotated_image = view_image.copy(
                        cur_tracking_window["orientation_convention"].toInt() ?
                            (cur_tracking_window.slice.cur_dim != 0 ? 0:geo[0]*display_ratio):
                            (cur_tracking_window.slice.cur_dim == 0 ? 0:geo[1]*display_ratio),
                            cur_tracking_window.slice.cur_dim != 2 ? 0:geo[2]*display_ratio,
                            cur_tracking_window.slice.cur_dim == 0 ? geo[1]*display_ratio:geo[0]*display_ratio,
                            cur_tracking_window.slice.cur_dim != 2 ? geo[2]*display_ratio:geo[1]*display_ratio);
        }
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
                                (cur_tracking_window.slice.cur_dim != 0 ? 0:geo[0]*display_ratio) :
                                (cur_tracking_window.slice.cur_dim == 0 ? 0:geo[1]*display_ratio),
                            cur_tracking_window.slice.cur_dim != 2 ? 0:geo[2]*display_ratio,annotated_image);
            addRect(0, 0, temp.width(),temp.height(),QPen(),temp);
        }
        else
            addRect(0, 0, annotated_image.width(),annotated_image.height(),QPen(),annotated_image);

        return;
    }
    case 6:
    {
        return;
    }
    }
    cur_tracking_window.regionWidget->add_points(points,mouseEvent->button() == Qt::RightButton);
    need_update();
}

void slice_view_scene::center()
{
    QList<QGraphicsView*> views = this->views();
    for(int index = 0;index < views.size();++index)
        views[index]->centerOn(view_image.width()/2,view_image.height()/2);
}



