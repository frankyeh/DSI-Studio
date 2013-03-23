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
        if(cur_tracking_window.slice.cur_dim != 2)
        {
            Y = view_image.height() - Y;
            tY = view_image.height() - tY;
        }
        paint.drawLine(X, Y, tX, tY);
        image::vector<2,float> from(X,Y);
        image::vector<2,float> to(tX,tY);
        from -= to;
        float pixel_length = from.length();
        from /= cur_tracking_window.ui->zoom->value();
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

void slice_view_scene::show_slice(void)
{
    float display_ratio = cur_tracking_window.ui->zoom->value();
    float contrast = cur_tracking_window.ui->contrast->value()/100.0;
    float offset = cur_tracking_window.ui->offset->value()/100.0;
    if(cur_tracking_window.ui->view_style->currentIndex() == 0)// single slice
    {
        cur_tracking_window.slice.get_slice(slice_image,contrast,offset);

        QImage qimage((unsigned char*)&*slice_image.begin(),slice_image.width(),slice_image.height(),QImage::Format_RGB32);
        // draw region colors on the image
        cur_tracking_window.regionWidget->draw_region(qimage);
        view_image = qimage.scaled(slice_image.width()*display_ratio,slice_image.height()*display_ratio);


        QPainter painter(&view_image);

        if(cur_tracking_window.ui->show_fiber->checkState() == Qt::Checked)
        {
            float fa[3];
            float dir[9];
            float threshold = cur_tracking_window.ui->fa_threshold->value();
            if (threshold == 0.0)
                threshold = 0.00000001;
            int X,Y,Z;
            float r = display_ratio /  3.0;
            float pen_w = display_ratio /  5.0;
            const char dir_x[3] = {1,0,0};
            const char dir_y[3] = {2,2,1};

            for (unsigned int y = 0; y < slice_image.height(); ++y)
                for (unsigned int x = 0; x < slice_image.width(); ++x)
                    if (cur_tracking_window.slice.get3dPosition(x, y, X, Y, Z))
                    {
                        if (!tracking_get_voxel_dir(handle, X, Y, Z,fa, dir))
                            continue;
                        for (char fiber = 2; fiber >= 0; --fiber)
                            if(fa[fiber] > threshold)
                            {
                                float* dir_ptr = dir + fiber + fiber + fiber;
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

        if(cur_tracking_window.ui->show_pos->checkState() == Qt::Checked)
        {
            int x_pos,y_pos;
            cur_tracking_window.slice.get_other_slice_pos(x_pos, y_pos);
            painter.setPen(QColor(255,0,0));
            painter.drawLine(((double)x_pos + 0.5)*display_ratio, 0,((double)x_pos + 0.5)*display_ratio,view_image.height());
            painter.drawLine(0, ((double)y_pos + 0.5)*display_ratio,view_image.width(),((double)y_pos + 0.5)*display_ratio);
        }


    }
    else
    {
        unsigned int skip = cur_tracking_window.ui->view_style->currentIndex()-1;
        mosaic_size = std::max((int)1,(int)std::ceil(std::sqrt((float)(cur_tracking_window.slice.geometry[2] >> skip))));
        cur_tracking_window.slice.get_mosaic(mosaic_image,mosaic_size,contrast,offset,skip);
        QImage qimage((unsigned char*)&*mosaic_image.begin(),mosaic_image.width(),mosaic_image.height(),QImage::Format_RGB32);
        cur_tracking_window.regionWidget->draw_mosaic_region(qimage,mosaic_size,skip);
        view_image = qimage.scaled(mosaic_image.width()*display_ratio/(float)mosaic_size,mosaic_image.height()*display_ratio/(float)mosaic_size);
    }
    setSceneRect(0, 0, view_image.width(),view_image.height());
    clear();
    setItemIndexMethod(QGraphicsScene::NoIndex);
    addRect(0, 0, view_image.width(),view_image.height(),QPen(),
            (cur_tracking_window.slice.cur_dim == 2 || cur_tracking_window.ui->view_style->currentIndex() != 0) ? view_image : view_image.mirrored());
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
                cur_tracking_window.absolute_path + "/" +
                cur_tracking_window.handle->fib_data.view_item[cur_tracking_window.ui->sliceViewBox->currentIndex()].name.c_str(),
                "NIFTI files (*.nii.gz *.nii);;MAT File (*.mat);;");
    if(filename.isEmpty())
        return;

    int index = cur_tracking_window.handle->get_name_index(
                cur_tracking_window.ui->sliceViewBox->currentText().toLocal8Bit().begin());
    if(index >= cur_tracking_window.handle->fib_data.view_item.size())
        return;

    if(QFileInfo(filename).completeSuffix().toLower() == "nii" ||
            QFileInfo(filename).completeSuffix().toLower() == "nii.gz")
    {
        gz_nifti file;
        file.set_voxel_size(cur_tracking_window.slice.voxel_size.begin());
        if(!cur_tracking_window.mi3.get() && !cur_tracking_window.handle->fib_data.trans_to_mni.empty()) //QSDR condition
        {
            file.set_image_transformation(cur_tracking_window.handle->fib_data.trans_to_mni.begin());
            file << cur_tracking_window.handle->fib_data.view_item[index].image_data;
            file.save_to_file(filename.toLocal8Bit().begin());
        }
        else
        {
            image::basic_image<float,3> buf(cur_tracking_window.handle->fib_data.view_item[index].image_data);
            image::flip_xy(buf);
            file << buf;
            file.save_to_file(filename.toLocal8Bit().begin());
        }
    }
    if(QFileInfo(filename).completeSuffix().toLower() == "mat")
    {
        image::io::mat file;
        file << cur_tracking_window.handle->fib_data.view_item[index].image_data;
        file.save_to_file(filename.toLocal8Bit().begin());
    }

}

void slice_view_scene::catch_screen()
{
    QString filename = QFileDialog::getSaveFileName(
            0,"Save Images files",
            cur_tracking_window.absolute_path,
            "PNG files (*.png);;BMP files (*.bmp);;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
    if(filename.isEmpty())
        return;

    if(cur_tracking_window.ui->view_style->currentIndex() != 0)// mosaic
    {
        view_image.save(filename);
        return;
    }
    QImage output = (cur_tracking_window.slice.cur_dim == 2) ? view_image:view_image.mirrored();
    QPainter paint(&output);
    show_ruler(paint);
    output.save(filename);
}

void slice_view_scene::copyClipBoard()
{
    if(cur_tracking_window.ui->view_style->currentIndex() != 0)// mosaic
    {
        QApplication::clipboard()->setImage(view_image);
        return;
    }
    QImage output = (cur_tracking_window.slice.cur_dim == 2) ? view_image:view_image.mirrored();
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
}

void slice_view_scene::mousePressEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window.ui->view_style->currentIndex() != 0)// single slice
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

    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();

    if (cur_tracking_window.slice.cur_dim != 2)
        Y = view_image.height() - Y;

    int x, y, z;

    cur_tracking_window.slice.get3dPosition(((float)X) / cur_tracking_window.ui->zoom->value(),
                                            ((float)Y) / cur_tracking_window.ui->zoom->value(), x, y, z);
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
    if(cur_tracking_window.ui->view_style->currentIndex() != 0)// single slice
        return;

    if (!mouse_down && !mid_down)
        return;
    if(mid_down)
    {
        return;
    }

    float Y = mouseEvent->scenePos().y();
    float X = mouseEvent->scenePos().x();
    if (cur_tracking_window.slice.cur_dim != 2)
        Y = view_image.height() - Y;
    cX = X;
    cY = Y;
    int x, y, z;
    if (!mouse_down || sel_mode == 4 ||
        !cur_tracking_window.slice.get3dPosition(((float)cX) / cur_tracking_window.ui->zoom->value(),
                            ((float)cY) / cur_tracking_window.ui->zoom->value(), x, y, z))
        return;

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

    QImage annotated_image = view_image;
    if(sel_mode == 6) // ruler
    {
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
        QImage temp = (cur_tracking_window.slice.cur_dim == 2) ? view_image:view_image.mirrored();
        QPainter paint(&temp);
        show_ruler(paint);
        annotated_image = (cur_tracking_window.slice.cur_dim == 2) ? temp:temp.mirrored();
    }



    QPainter paint(&annotated_image);
    paint.setPen(cur_tracking_window.regionWidget->currentRowColor());
    paint.setBrush(Qt::NoBrush);
    switch (sel_mode)
    {
    case 0:
        paint.drawRect(X, Y, sel_point.front()[0]-X,sel_point.front()[1]-Y);
        sel_coord.back() = image::vector<3,short>(x, y, z);
        sel_point.back() = image::vector<2,short>(X, Y);
        break;
    case 1:
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
    addRect(0, 0, annotated_image.width(),annotated_image.height(),QPen(),
            (cur_tracking_window.slice.cur_dim == 2) ? annotated_image : annotated_image.mirrored());

}


void slice_view_scene::mouseReleaseEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
    if(cur_tracking_window.ui->view_style->currentIndex() != 0)// single slice
        return;

    if (!mouse_down && !mid_down)
        return;
    if(mid_down || sel_mode == 5)
    {
        mid_down = false;
        return;
    }
    int Y = mouseEvent->scenePos().y();
    int X = mouseEvent->scenePos().x();
    if (cur_tracking_window.slice.cur_dim != 2)
        Y = view_image.height() - Y;
    mouse_down = false;
    /*
    {
            if (TrackForm->odf_model.get()) {
                    TrackForm->odf_model->loadFromData
                            (TrackForm->cur_tracking_window.slice.getOriginal3dPosition
                            (((float)X) / display_ratio, ((float)Y) / display_ratio));
                    TrackForm->odf_model->Show(0);
            }
            if (TrackForm->odf_slice_model.get())
                    TrackForm->odf_slice_model->SetPosition(((float)X) / display_ratio,
                    ((float)Y) / display_ratio);
    }*/

    std::vector<image::vector<3,short> >points;
    switch (sel_mode)
    {
    case 0:
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
    case 1:
    {
        if (sel_coord.size() <= 2)
        {
            points.push_back(sel_coord.front());
            break;
        }
        {
            QImage bitmap(slice_image.width(),slice_image.height(),QImage::Format_Mono);
            QPainter paint(&bitmap);
            paint.setBrush(Qt::black);
            paint.drawRect(0,0,slice_image.width(),slice_image.height());
            paint.setBrush(Qt::white);
            std::vector<QPoint> qpoints(sel_point.size());
            for(unsigned int index = 0;index < sel_point.size();++index)
                qpoints[index] = QPoint(
                            sel_point[index][0]/cur_tracking_window.ui->zoom->value(),
                            sel_point[index][1]/cur_tracking_window.ui->zoom->value());
            paint.drawPolygon(&*qpoints.begin(),qpoints.size() - 1);
            int x, y, z;
            for (image::pixel_index<2>index; index.valid(slice_image.geometry());
                    index.next(slice_image.geometry()))
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
        if (distance2 > 400)
            distance2 = 400;
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
        if (distance2 > 400)
            distance2 = 400;
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
        QImage annotated_image = view_image;
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
        addRect(0, 0, annotated_image.width(),annotated_image.height(),QPen(),
                (cur_tracking_window.slice.cur_dim == 2) ? annotated_image : annotated_image.mirrored());
        return;
    }
    case 6:
    {
        return;
    }
    }
    /*
    if(mouseEvent->button() != Qt::RightButton &&
       !cur_tracking_window.regionWidget->regions.empty() &&
       !cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()].empty() &&
       !cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()].has_points(points))
    {
        int result = QMessageBox::information(0,"DSI Studio","Draw a new region?",QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
        if(result == QMessageBox::Cancel)
            return;
        if(result == QMessageBox::Yes)
            cur_tracking_window.regionWidget->new_region();
    }
    */
    cur_tracking_window.regionWidget->add_points(points,mouseEvent->button() == Qt::RightButton);
    need_update();
}

void slice_view_scene::center()
{
    QList<QGraphicsView*> views = this->views();
    for(int index = 0;index < views.size();++index)
        views[index]->centerOn(view_image.width()/2,view_image.height()/2);
}



