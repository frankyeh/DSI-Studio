#include <QGraphicsTextItem>
#include <QMouseEvent>
#include <QFileDialog>
#include "connectivity_matrix_dialog.h"
#include "ui_connectivity_matrix_dialog.h"
#include "region/regiontablewidget.h"
#include "tract/tracttablewidget.h"
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
extern std::vector<atlas> atlas_list;
connectivity_matrix_dialog::connectivity_matrix_dialog(tracking_window *parent) :
    QDialog(parent),cur_tracking_window(parent),
    ui(new Ui::connectivity_matrix_dialog)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(&scene);
    // atlas

    ui->region_list->addItem("ROIs");
    for(int index = 0;index < atlas_list.size();++index)
        ui->region_list->addItem(atlas_list[index].name.c_str());
    if(cur_tracking_window->regionWidget->regions.size() > 1)
        ui->region_list->setCurrentIndex(0);
    else
        ui->region_list->setCurrentIndex(1);
    on_recalculate_clicked();

}

connectivity_matrix_dialog::~connectivity_matrix_dialog()
{
    delete ui;
}

bool connectivity_matrix_dialog::is_graphic_view(QObject *object) const
{
    return object == ui->graphicsView;
}
void connectivity_matrix_dialog::mouse_move(QMouseEvent *mouseEvent)
{
    QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    int x = std::floor(((float)point.x()) / ui->zoom->value() - 0.5);
    int y = std::floor(((float)point.y()) / ui->zoom->value() - 0.5);
    if(x >= 0 && y >= 0 && x < region_name.size() && y < region_name.size())
    {
        matrix_to_image();
        // line x
        for(unsigned int x_pos = 0,pos = y*matrix.size();x_pos < matrix.size();++x_pos,++pos)
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        // line y
        for(unsigned int y_pos = 0,pos = x;y_pos < matrix.size();++y_pos,pos += matrix.size())
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        on_zoom_valueChanged(0);
        QGraphicsTextItem *x_text = scene.addText(region_name[x].c_str());
        QGraphicsTextItem *y_text = scene.addText(region_name[y].c_str());
        x_text->moveBy(point.x()-x_text->boundingRect().width()/2,-x_text->boundingRect().height());
        y_text->rotate(270);
        y_text->moveBy(-y_text->boundingRect().height(),point.y()+y_text->boundingRect().width()/2);
    }

}


void connectivity_matrix_dialog::on_recalculate_clicked()
{
    if(cur_tracking_window->tractWidget->tract_models.size() == 0)
        return;
    typedef std::map<float,std::pair<std::vector<image::vector<3,short> >,std::string> > region_table_type;
    region_table_type region_table;
    image::geometry<3> geo = cur_tracking_window->slice.geometry;
    if(ui->region_list->currentIndex() == 0)
    {
        for(unsigned int index = 0;index < cur_tracking_window->regionWidget->regions.size();++index)
        {
            const std::vector<image::vector<3,short> >& cur_region =
                    cur_tracking_window->regionWidget->regions[index].get();
            image::vector<3,float> pos = std::accumulate(cur_region.begin(),cur_region.end(),image::vector<3,float>(0,0,0));
            pos /= cur_region.size();

            region_table[pos[0] > (geo[0] >> 1) ? pos[1]-geo[1]:geo[1]-pos[1]] = std::make_pair(cur_region,cur_tracking_window->regionWidget->item(index,0)->text().toLocal8Bit().begin());
        }
    }
    else  // from atlas
        if(!cur_tracking_window->handle->fib_data.trans_to_mni.empty())
        {
            std::vector<image::vector<3,float> > mni_position(geo.size());
            std::vector<image::vector<3,short> > subject_position(geo.size());
            for (image::pixel_index<3>index; index.valid(geo);index.next(geo))
            {
                image::vector<3,float> mni;
                image::vector<3,float>cur_coordinate((const unsigned int*)(index.begin()));
                image::vector_transformation(cur_coordinate.begin(),
                                             mni.begin(),
                                             cur_tracking_window->handle->fib_data.trans_to_mni.begin(),
                                             image::vdim<3>());
                mni_position[index.index()] = mni;
                subject_position[index.index()] = image::vector<3,short>((const unsigned int*)index.begin());
            }

            unsigned int atlas_index = ui->region_list->currentIndex()-1;
            for (unsigned int label = 0; label < atlas_list[atlas_index].get_list().size(); ++label)
            {
                std::vector<image::vector<3,short> > cur_region;
                image::vector<3,float> mni_avg_pos;
                float min_x = 200,max_x = -200;
                for(unsigned int pos = 0;pos < subject_position.size();++pos)
                    if (atlas_list[atlas_index].is_labeled_as(mni_position[pos], label))
                    {
                        cur_region.push_back(subject_position[pos]);
                        mni_avg_pos += mni_position[pos];
                        if(mni_position[pos][0] > max_x)
                           max_x = mni_position[pos][0];
                        if(mni_position[pos][0] < min_x)
                           min_x = mni_position[pos][0];
                    }
                if(cur_region.empty())
                    continue;
                mni_avg_pos /= cur_region.size();
                const std::vector<std::string>& region_names = atlas_list[atlas_index].get_list();
                float order;
                if(mni_avg_pos[0] > 0)
                    order = 500.0-mni_avg_pos[1];
                else
                    order = mni_avg_pos[1]-500.0;

                // is at middle?
                if((max_x-min_x)/8.0 > std::fabs(mni_avg_pos[0]))
                    order = mni_avg_pos[1];
                region_table[order] = std::make_pair(cur_region,region_names[label]);
            }
        }
        else
            return;

    if(region_table.size() == 0)
        return;

    std::vector<std::vector<image::vector<3,short> > > regions(region_table.size());
    region_name.resize(region_table.size());

    region_table_type::const_iterator iter = region_table.begin();
    region_table_type::const_iterator end = region_table.end();
    for(unsigned int index = 0;iter != end;++iter,++index)
    {
        regions[index] = iter->second.first;
        region_name[index] = iter->second.second;
    }



    cur_tracking_window->tractWidget->tract_models[cur_tracking_window->tractWidget->currentRow()]
            ->get_connectivity_matrix(regions,matrix);

    matrix_buf.resize(matrix.size()*matrix.size());
    for(unsigned int index = 0;index < matrix.size();++index)
        std::copy(matrix[index].begin(),matrix[index].end(),matrix_buf.begin() + index*matrix.size());


    matrix_to_image();
    on_zoom_valueChanged(0);
}

void connectivity_matrix_dialog::matrix_to_image(void)
{
    if(matrix.empty())
        return;
    cm.resize(image::geometry<2>(matrix.size(),matrix.size()));
    unsigned int max_value = *std::max_element(matrix_buf.begin(),matrix_buf.end());
    if(ui->log->isChecked())
        max_value = std::log(max_value + 1.0);
    for(unsigned int index = 0;index < matrix_buf.size();++index)
    {
        float value = matrix_buf[index];
        if(ui->log->isChecked())
            value = std::log(value + 1.0);
        value *= 255.9;
        value /= max_value;
        value = std::floor(value);
        cm[index] = image::rgb_color((unsigned char)value,(unsigned char)value,(unsigned char)value);
    }
}

void connectivity_matrix_dialog::on_zoom_valueChanged(double arg1)
{
    if(cm.empty())
        return;
    QImage qimage((unsigned char*)&*cm.begin(),cm.width(),cm.height(),QImage::Format_RGB32);
    view_image = qimage.scaled(cm.width()*ui->zoom->value(),cm.height()*ui->zoom->value());
    scene.setSceneRect(0, 0, view_image.width(),view_image.height());
    scene.clear();
    scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    scene.addRect(0, 0, view_image.width(),view_image.height(),QPen(),view_image);
}

void connectivity_matrix_dialog::on_log_toggled(bool checked)
{
    matrix_to_image();
    on_zoom_valueChanged(0);
}

void connectivity_matrix_dialog::on_save_as_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                0,
                "Save as",
                cur_tracking_window->absolute_path + "/" +
                cur_tracking_window->tractWidget->item(
                    cur_tracking_window->tractWidget->currentRow(),0)->text() + ".mat",
                "MAT File (*.mat);;Image Files (*.png *.tif *.bmp)");
    if(filename.isEmpty())
        return;

    if(QFileInfo(filename).suffix().toLower() == "mat")
    {
        image::io::mat mat_header;
        mat_header.add_matrix("connectivity",&*matrix_buf.begin(),matrix.size(),matrix.size());
        std::ostringstream out;
        std::copy(region_name.begin(),region_name.end(),std::ostream_iterator<std::string>(out,"\n"));
        std::string result(out.str());
        mat_header.add_matrix("name",result.c_str(),1,result.length());
        mat_header.save_to_file(filename.toLocal8Bit().begin());
    }
    else
    {
        matrix_to_image();
        QImage qimage((unsigned char*)&*cm.begin(),cm.width(),cm.height(),QImage::Format_RGB32);
        qimage.save(filename);
    }

}
