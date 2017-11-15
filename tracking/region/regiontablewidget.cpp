#include <regex>
#include <QFileDialog>
#include <QInputDialog>
#include <QContextMenuEvent>
#include <QMessageBox>
#include <QClipboard>
#include <QSettings>
#include <QTableWidgetItem>
#include "regiontablewidget.h"
#include "tracking/tracking_window.h"
#include "qcolorcombobox.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "mapping/fa_template.hpp"
#include "opengl/glwidget.h"
#include "libs/tracking/fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
extern std::vector<atlas> atlas_list;
extern fa_template fa_template_imp;


void split(std::vector<std::string>& list,const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    list = {first, last};
}

QWidget *ImageDelegate::createEditor(QWidget *parent,
                                     const QStyleOptionViewItem &option,
                                     const QModelIndex &index) const
{
    if (index.column() == 1)
    {
        QComboBox *comboBox = new QComboBox(parent);
        comboBox->addItem("ROI");
        comboBox->addItem("ROA");
        comboBox->addItem("End");
        comboBox->addItem("Seed");
        comboBox->addItem("Terminative");
        connect(comboBox, SIGNAL(activated(int)), this, SLOT(emitCommitData()));
        return comboBox;
    }
    else if (index.column() == 2)
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        return sd;
    }
    else
        return QItemDelegate::createEditor(parent,option,index);

}

void ImageDelegate::setEditorData(QWidget *editor,
                                  const QModelIndex &index) const
{

    if (index.column() == 1)
        ((QComboBox*)editor)->setCurrentIndex(index.model()->data(index).toString().toInt());
    else
        if (index.column() == 2)
        {
            image::rgb_color color((unsigned int)(index.data(Qt::UserRole).toInt()));
            ((QColorToolButton*)editor)->setColor(
                QColor(color.r,color.g,color.b));
        }
        else
            return QItemDelegate::setEditorData(editor,index);
}

void ImageDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                 const QModelIndex &index) const
{
    if (index.column() == 1)
        model->setData(index,QString::number(((QComboBox*)editor)->currentIndex()));
    else
        if (index.column() == 2)
            model->setData(index,(int)(((QColorToolButton*)editor)->color().rgba()),Qt::UserRole);
        else
            QItemDelegate::setModelData(editor,model,index);
}

void ImageDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}


RegionTableWidget::RegionTableWidget(tracking_window& cur_tracking_window_,QWidget *parent):
        QTableWidget(parent),cur_tracking_window(cur_tracking_window_)
{
    setColumnCount(3);
    setColumnWidth(0,140);
    setColumnWidth(1,60);
    setColumnWidth(2,40);

    QStringList header;
    header << "Name" << "Type" << "Color";
    setHorizontalHeaderLabels(header);
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setAlternatingRowColors(true);
    setStyleSheet("QTableView {selection-background-color: #AAAAFF; selection-color: #000000;}");

    setItemDelegate(new ImageDelegate(this));

    QObject::connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_check_status(int,int)));
    QObject::connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateRegions(QTableWidgetItem*)));

}



RegionTableWidget::~RegionTableWidget(void)
{
}

void RegionTableWidget::contextMenuEvent ( QContextMenuEvent * event )
{
    if (event->reason() == QContextMenuEvent::Mouse)
    {
        cur_tracking_window.ui->menuRegions->popup(event->globalPos());
    }
}

void RegionTableWidget::updateRegions(QTableWidgetItem* item)
{
    if (item->column() == 1)
        regions[item->row()]->regions_feature = item->text().toInt();
    else
        if (item->column() == 2)
        {
            regions[item->row()]->show_region.color = item->data(Qt::UserRole).toInt();
            emit need_update();
        }
}

QColor RegionTableWidget::currentRowColor(void)
{
    return (unsigned int)regions[currentRow()]->show_region.color;
}
void RegionTableWidget::add_region_from_atlas(unsigned int atlas,unsigned int label)
{
    std::vector<image::vector<3,short> > points;
    add_region(atlas_list[atlas].get_list()[label].c_str(),roi_id);
    float r;
    cur_tracking_window.handle->get_atlas_roi(atlas,label,points,r);
    regions.back()->resolution_ratio = r;
    regions.back()->add_points(points,false,r);

}

void RegionTableWidget::add_region(QString name,unsigned char feature,int color)
{
    if(color == 0x00FFFFFF || !color)
    {
        image::rgb_color c;
        c.from_hsl(((color_gen++)*1.1-std::floor((color_gen++)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        color = c.color;
    }
    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle->dim,cur_tracking_window.current_slice->voxel_size));
    regions.back()->show_region.color = color;
    regions.back()->regions_feature = feature;
    cur_tracking_window.scene.no_show = true;
    setRowCount(regions.size());

    QTableWidgetItem *item0 = new QTableWidgetItem(name);

    setItem(regions.size()-1, 0, item0);

    QTableWidgetItem *item1 = new QTableWidgetItem(QString::number((int)feature));
    QTableWidgetItem *item2 = new QTableWidgetItem();

    setItem(regions.size()-1, 1, item1);
    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    setItem(regions.size()-1, 2, item2);
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::UserRole,0xFF000000 | color);

    openPersistentEditor(item1);
    openPersistentEditor(item2);
    item0->setCheckState(Qt::Checked);

    setRowHeight(regions.size()-1,22);
    setCurrentCell(regions.size()-1,0);
    cur_tracking_window.scene.no_show = false;

}
void RegionTableWidget::check_check_status(int row, int col)
{
    if (col != 0)
        return;
    if (item(row,0)->checkState() == Qt::Checked)
    {
        if (item(row,0)->data(Qt::ForegroundRole) == QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
            emit need_update();
        }
    }
    else
    {
        if (item(row,0)->data(Qt::ForegroundRole) != QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
            emit need_update();
        }
    }
}

void RegionTableWidget::draw_region(image::color_image& I)
{
    auto current_slice = cur_tracking_window.current_slice;
    int slice_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];
    std::vector<std::shared_ptr<ROIRegion> > checked_regions;
    for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
    {
        if (item(roi_index,0)->checkState() != Qt::Checked)
            continue;
        checked_regions.push_back(regions[roi_index]);
    }
    if(checked_regions.empty())
        return;
    if(current_slice->is_diffusion_space)
    {
        for(unsigned int roi_index = 0;roi_index < checked_regions.size();++roi_index)
        {
            float r = checked_regions[roi_index]->resolution_ratio;
            unsigned int cur_color = checked_regions[roi_index]->show_region.color;
            image::par_for(checked_regions[roi_index]->size(),[&](unsigned int index)
            {
                image::vector<3,float> p(checked_regions[roi_index]->region[index]);
                if(r != 1.0)
                    p /= r;
                p.round();
                int X, Y, Z;
                image::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
                if (slice_pos != Z || X < 0 || Y < 0 || X >= I.width() || Y >= I.height())
                    return;
                unsigned int pos = X+Y*I.width();
                I[pos] = (unsigned int)I[pos] | cur_color;
            });
        }
    }
    else
    {
        //handle resolution_ratio = 1 all together
        {
            image::basic_image<unsigned int,3> buf(cur_tracking_window.handle->dim);
            for(unsigned int roi_index = 0;roi_index < checked_regions.size();++roi_index)
            {
                unsigned int cur_color = checked_regions[roi_index]->show_region.color;
                if(checked_regions[roi_index]->resolution_ratio == 1)
                image::par_for(regions[roi_index]->size(),[&](unsigned int index)
                {
                    image::pixel_index<3> pindex(regions[roi_index]->region[index][0],
                                                 regions[roi_index]->region[index][1],
                                                 regions[roi_index]->region[index][2],buf.geometry());
                    if(pindex.index() >= buf.size())
                        return;
                    buf[pindex.index()] = cur_color;
                });
            }
            for(int y = 0,index = 0;y < I.height();++y)
                for(int x = 0;x < I.width();++x,++index)
                {
                    int dx,dy,dz;
                    current_slice->toDiffusionSpace(cur_tracking_window.cur_dim,x,y,dx,dy,dz);
                    if(!buf.geometry().is_valid(dx,dy,dz))
                        continue;
                    image::pixel_index<3> pindex(dx,dy,dz,buf.geometry());
                    if(buf[pindex.index()] != 0)
                        I[index] = (unsigned int)I[index] | buf[pindex.index()];

                }
        }

        // now the most time consuming part with high resolution regions
        image::par_for(checked_regions.size(),[&](unsigned int roi_index)
        {
            if(checked_regions[roi_index]->resolution_ratio == 1)
                return;
            unsigned int cur_color = checked_regions[roi_index]->show_region.color;
            float r = checked_regions[roi_index]->resolution_ratio;
            image::geometry<3> geo(cur_tracking_window.handle->dim);
            geo[0] *= r;
            geo[1] *= r;
            geo[2] *= r;
            std::vector<std::vector<std::vector<unsigned int> > > buf(geo[0]);
            for(unsigned int index = 0;index < regions[roi_index]->size();++index)
            {
                const auto& p = regions[roi_index]->region[index];
                if(!geo.is_valid(p))
                    return;
                auto& x_pos = buf[p[0]];
                if(x_pos.empty())
                    x_pos.resize(geo[1]);
                auto& y_pos = x_pos[p[1]];
                if(y_pos.empty())
                    y_pos.resize(geo[2]);
                y_pos[p[2]] = cur_color;
            }
            for(int y = 0,index = 0;y < I.height();++y)
                for(int x = 0;x < I.width();++x,++index)
                {

                    image::vector<3,float> v;
                    image::slice2space(cur_tracking_window.cur_dim, x, y,
                                       slice_pos, v[0],v[1],v[2]);
                    v.to(current_slice->transform);
                    v *= r;
                    v.round();
                    int dx = v[0];
                    int dy = v[1];
                    int dz = v[2];
                    if(!geo.is_valid(dx,dy,dz) ||
                            buf[dx].empty() ||
                            buf[dx][dy].empty() ||
                            buf[dx][dy][dz] == 0)
                        continue;
                    I[index] = (unsigned int)I[index] | buf[dx][dy][dz];
                }
        });
    }
}

void RegionTableWidget::draw_edge(QImage&,QImage&)
{
    if(!cur_tracking_window.current_slice->is_diffusion_space)
        return;
    if(rowCount() == 0 || currentRow() == -1 || currentRow() >= regions.size() ||
       item(currentRow(),0)->checkState() != Qt::Checked)
        return;
    /*
    int X, Y, Z;
    image::basic_image<unsigned char,2> cur_image_mask;
    cur_image_mask.resize(image::geometry<2>(qimage.width(),qimage.height()));
    for (unsigned int index = 0;index < regions[currentRow()]->size();++index)
    {
        regions[currentRow()]->getSlicePosition(cur_tracking_window.cur_dim, index, X, Y, Z);
        if (cur_tracking_window.current_slice->slice_pos[cur_tracking_window.cur_dim] != Z ||
                X < 0 || Y < 0 || X >= qimage.width() || Y >= qimage.height())
            continue;
        cur_image_mask.at(X,Y) = 1;
    }

    float display_ratio = (float)scaled_image.width()/(float)qimage.width();
    unsigned int cur_color = ((unsigned int)regions[currentRow()]->show_region.color) ^ 0x00FFFFFF;
    QPainter paint(&scaled_image);
    paint.setBrush(Qt::NoBrush);
    QPen pen(QColor(cur_color), 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
    paint.setPen(pen);
    for(int y = 1,cur_index = qimage.width();y < qimage.height()-1;++y)
    for(int x = 0;x < qimage.width();++x,++cur_index)
    {
        if(x == 0 || x+1 >= qimage.width() || !cur_image_mask[cur_index])
            continue;
        float xd = x*display_ratio;
        float xd_1 = xd+display_ratio;
        float yd = y*display_ratio;
        float yd_1 = yd+display_ratio;
        if(!(cur_image_mask[cur_index-qimage.width()]))
            paint.drawLine(xd,yd,xd_1,yd);
        if(!(cur_image_mask[cur_index+qimage.width()]))
            paint.drawLine(xd,yd_1,xd_1,yd_1);
        if(!(cur_image_mask[cur_index-1]))
            paint.drawLine(xd,yd,xd,yd_1);
        if(!(cur_image_mask[cur_index+1]))
            paint.drawLine(xd_1,yd,xd_1,yd_1);
    }*/
}

void RegionTableWidget::draw_mosaic_region(QImage& qimage,unsigned int mosaic_size,unsigned int skip)
{
    image::geometry<3> geo = cur_tracking_window.handle->dim;
    unsigned int slice_number = geo[2] / skip;
    std::vector<int> shift_x(slice_number),shift_y(slice_number);
    for(unsigned int z = 0;z < slice_number;++z)
    {
        shift_x[z] = geo[0]*(z%mosaic_size);
        shift_y[z] = geo[1]*(z/mosaic_size);
    }

    for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
    {
        if (item(roi_index,0)->checkState() != Qt::Checked)
            continue;
        unsigned int cur_color = regions[roi_index]->show_region.color;
        for (unsigned int index = 0;index < regions[roi_index]->size();++index)
        {
            int X = regions[roi_index]->get()[index][0];
            int Y = regions[roi_index]->get()[index][1];
            int Z = regions[roi_index]->get()[index][2];
            if(Z != ((Z / skip) * skip))
                continue;
            X += shift_x[Z / skip];
            Y += shift_y[Z / skip];
            if(X < 0 || Y < 0 || X >= qimage.width() || Y >= qimage.height())
                continue;
            qimage.setPixel(X,Y,(unsigned int)qimage.pixel(X,Y) | cur_color);
        }
    }
}

void RegionTableWidget::new_region(void)
{
    add_region("New Region",roi_id);
    if(cur_tracking_window.current_slice->is_diffusion_space)
        regions.back()->resolution_ratio = 1;
    else
    {
        regions.back()->resolution_ratio =
            std::min<float>(4.0f,std::max<float>(1.0f,std::ceil(
            (float)cur_tracking_window.get_scene_zoom()*
            cur_tracking_window.handle->vs[0]/
            cur_tracking_window.current_slice->voxel_size[0])));
    }
}
void RegionTableWidget::new_high_resolution_region(void)
{
    bool ok;
    int ratio = QInputDialog::getInt(this,
            "DSI Studio",
            "Input resolution ratio (e.g. 2 for 2X, 8 for 8X",8,2,16,2,&ok);
    if(!ok)
        return;
    add_region("New High Resolution Region",roi_id);
    regions.back()->resolution_ratio = ratio;
}

void RegionTableWidget::copy_region(void)
{
    unsigned int cur_row = currentRow();
    add_region(item(cur_row,0)->text(),regions[cur_row]->regions_feature);
    unsigned int color = regions.back()->show_region.color.color;
    *regions.back() = *regions[cur_row];
    regions.back()->show_region.color.color = color;
}
void load_nii_label(const char* filename,std::map<short,std::string>& label_map)
{
    std::ifstream in(filename);
    if(in)
    {
        std::string line,txt;
        while(std::getline(in,line))
        {
            if(line.empty() || line[0] == '#')
                continue;
            std::istringstream read_line(line);
            short num = 0;
            read_line >> num >> txt;
            label_map[num] = txt;
        }
    }
}

bool RegionTableWidget::load_multiple_roi_nii(QString file_name)
{
    gz_nifti header;
    if (!header.load_from_file(file_name.toLocal8Bit().begin()))
        return false;

    image::basic_image<unsigned int, 3> from;
    {
        image::basic_image<float, 3> tmp;
        header.toLPS(tmp);
        image::add_constant(tmp,0.5);
        from = tmp;
    }

    std::vector<unsigned char> value_map(std::numeric_limits<unsigned short>::max());
    unsigned int max_value = 0;
    for (image::pixel_index<3>index(from.geometry());index < from.size();++index)
    {
        value_map[(unsigned short)from[index.index()]] = 1;
        max_value = std::max<unsigned short>(from[index.index()],max_value);
    }
    value_map.resize(max_value+1);

    unsigned short region_count = std::accumulate(value_map.begin(),value_map.end(),(unsigned short)0);
    bool multiple_roi = region_count > 2;


    std::map<short,std::string> label_map;
    if(multiple_roi)
    {
        QString base_name = QFileInfo(file_name).completeBaseName();
        if(QFileInfo(base_name).suffix().toLower() == "nii")
            base_name = QFileInfo(base_name).completeBaseName();
        QString label_file = QFileInfo(file_name).absolutePath()+"/"+base_name+".txt";

        if(QFileInfo(label_file).exists())
            load_nii_label(label_file.toLocal8Bit().begin(),label_map);
    }

    image::matrix<4,4,float> convert;
    bool has_transform = false;

    // searching for T1/T2 mappings
    for(unsigned int index = 0;index < cur_tracking_window.slices.size();++index)
    {
        CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.slices[index].get());
        if(slice && from.geometry() == slice->geometry)
        {
            convert = slice->invT;
            has_transform = true;
            break;
        }
    }

    if(from.geometry() != cur_tracking_window.handle->dim)
    {
        float r1 = (float)from.width()/(float)cur_tracking_window.handle->dim[0];
        float r2 = (float)from.height()/(float)cur_tracking_window.handle->dim[1];
        float r3 = (float)from.depth()/(float)cur_tracking_window.handle->dim[2];
        if(r1 == r2 && r1 == r3)
            has_transform = false;
        else
        if(cur_tracking_window.handle->is_qsdr && !has_transform)// use transformation information
        {
            // searching QSDR mappings
            image::basic_image<unsigned int, 3> new_from;
            for(unsigned int index = 0;index < cur_tracking_window.handle->view_item.size();++index)
                if(cur_tracking_window.handle->view_item[index].native_geo == from.geometry())
                {
                    new_from.resize(cur_tracking_window.handle->dim);
                    for(image::pixel_index<3> pos(new_from.geometry());pos < new_from.size();++pos)
                    {
                        image::vector<3> new_pos(cur_tracking_window.handle->view_item[index].mx[pos.index()],
                                                 cur_tracking_window.handle->view_item[index].my[pos.index()],
                                                 cur_tracking_window.handle->view_item[index].mz[pos.index()]);
                        new_pos.round();
                        new_from[pos.index()] = from.at(new_pos[0],new_pos[1],new_pos[2]);
                    }
                    break;
                }

            if(new_from.empty())
            {
                QMessageBox::information(this,"Warning","The nii file has different image dimension. Transformation will be applied to load the region",0);
                convert.identity();
                header.get_image_transformation(convert.begin());
                convert.inv();
                convert *= cur_tracking_window.handle->trans_to_mni;
                has_transform = true;
            }
            else
                new_from.swap(from);
        }
    }

    if(!multiple_roi)
    {
        ROIRegion region(cur_tracking_window.handle->dim,cur_tracking_window.current_slice->voxel_size);

        if(has_transform)
            region.LoadFromBuffer(from,convert);
        else
            region.LoadFromBuffer(from);

        int color = 0;
        int type = roi_id;

        try{
            std::vector<std::string> info;
            split(info,header.get_descrip(),";");
            for(unsigned int index = 0;index < info.size();++index)
            {
                std::vector<std::string> name_value;
                split(name_value,info[index],"=");
                if(name_value.size() != 2)
                    continue;
                if(name_value[0] == "color")
                    std::istringstream(name_value[1]) >> color;
                if(name_value[0] == "roi")
                    std::istringstream(name_value[1]) >> type;
            }
        }catch(...){}

        region.show_region.color = max_value;
        add_region(QFileInfo(file_name).baseName(),type,color);

        regions.back()->assign(region.get(),region.resolution_ratio);
        item(currentRow(),0)->setCheckState(Qt::Checked);
        item(currentRow(),0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
        return true;
    }
    begin_prog("loading ROIs");
    for(unsigned int value = 1;check_prog(value,value_map.size());++value)
        if(value_map[value])
        {
            image::basic_image<unsigned char,3> mask(from.geometry());
            for(unsigned int i = 0;i < mask.size();++i)
                if(from[i] == value)
                    mask[i] = 1;
            ROIRegion region(cur_tracking_window.handle->dim,cur_tracking_window.current_slice->voxel_size);
            if(has_transform)
                region.LoadFromBuffer(mask,convert);
            else
                region.LoadFromBuffer(mask);
            QString name = (label_map.find(value) == label_map.end() ?
                                QString("roi_") + QString::number(value):QString(label_map[value].c_str()));
            add_region(name,roi_id);
            regions.back()->assign(region.get(),region.resolution_ratio);
            item(currentRow(),0)->setCheckState(Qt::Unchecked);
            item(currentRow(),0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
        }
    return true;
}


void RegionTableWidget::load_region(void)
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,"Open region",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),"Region files (*.txt *.nii *.hdr *nii.gz *.mat);;All files (*)" );
    if (filenames.isEmpty())
        return;

    for (unsigned int index = 0;index < filenames.size();++index)
    {
        // check for multiple nii
        if((QFileInfo(filenames[index]).suffix() == "gz" ||
           QFileInfo(filenames[index]).suffix() == "nii") &&
                load_multiple_roi_nii(filenames[index]))
            continue;

        ROIRegion region(cur_tracking_window.handle->dim,cur_tracking_window.current_slice->voxel_size);
        if(!region.LoadFromFile(filenames[index].toLocal8Bit().begin(),
                cur_tracking_window.handle->is_qsdr ? cur_tracking_window.handle->trans_to_mni:std::vector<float>()))
        {
            QMessageBox::information(this,"error","Unknown file format",0);
            return;
        }
        add_region(QFileInfo(filenames[index]).baseName(),roi_id,region.show_region.color.color);
        regions.back()->assign(region.get(),region.resolution_ratio);

    }
    emit need_update();
}

void RegionTableWidget::merge_all(void)
{
    std::vector<unsigned int> merge_list;
    for(int index = 0;index < regions.size();++index)
        if(item(index,0)->checkState() == Qt::Checked)
            merge_list.push_back(index);
    if(merge_list.size() <= 1)
        return;

    for(int index = merge_list.size()-1;index >= 1;--index)
    {
        regions[merge_list[0]]->add(*regions[merge_list[index]]);
        regions.erase(regions.begin()+merge_list[index]);
        removeRow(merge_list[index]);
    }
    emit need_update();
}

void RegionTableWidget::check_all(void)
{
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
    emit need_update();
}

void RegionTableWidget::uncheck_all(void)
{
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
    emit need_update();
}

void RegionTableWidget::move_up(void)
{
    if(currentRow())
    {
        regions[currentRow()].swap(regions[currentRow()-1]);

        QString name = item(currentRow()-1,0)->text();
        item(currentRow()-1,0)->setText(item(currentRow(),0)->text());
        item(currentRow(),0)->setText(name);

        closePersistentEditor(item(currentRow()-1,1));
        closePersistentEditor(item(currentRow(),1));
        closePersistentEditor(item(currentRow()-1,2));
        closePersistentEditor(item(currentRow(),2));
        item(currentRow()-1,1)->setData(Qt::DisplayRole,regions[currentRow()-1]->regions_feature);
        item(currentRow(),1)->setData(Qt::DisplayRole,regions[currentRow()]->regions_feature);
        item(currentRow()-1,2)->setData(Qt::UserRole,regions[currentRow()-1]->show_region.color.color);
        item(currentRow(),2)->setData(Qt::UserRole,regions[currentRow()]->show_region.color.color);
        openPersistentEditor(item(currentRow()-1,1));
        openPersistentEditor(item(currentRow(),1));
        openPersistentEditor(item(currentRow()-1,2));
        openPersistentEditor(item(currentRow(),2));
        setCurrentCell(currentRow()-1,0);
    }
    emit need_update();
}

void RegionTableWidget::move_down(void)
{
    if(currentRow()+1 < regions.size())
    {
        regions[currentRow()].swap(regions[currentRow()+1]);

        QString name = item(currentRow()+1,0)->text();
        item(currentRow()+1,0)->setText(item(currentRow(),0)->text());
        item(currentRow(),0)->setText(name);

        closePersistentEditor(item(currentRow()+1,1));
        closePersistentEditor(item(currentRow(),1));
        closePersistentEditor(item(currentRow()+1,2));
        closePersistentEditor(item(currentRow(),2));
        item(currentRow()+1,1)->setData(Qt::DisplayRole,regions[currentRow()+1]->regions_feature);
        item(currentRow(),1)->setData(Qt::DisplayRole,regions[currentRow()]->regions_feature);
        item(currentRow()+1,2)->setData(Qt::UserRole,regions[currentRow()+1]->show_region.color.color);
        item(currentRow(),2)->setData(Qt::UserRole,regions[currentRow()]->show_region.color.color);
        openPersistentEditor(item(currentRow()+1,1));
        openPersistentEditor(item(currentRow(),1));
        openPersistentEditor(item(currentRow()+1,2));
        openPersistentEditor(item(currentRow(),2));
        setCurrentCell(currentRow()+1,0);
    }
    emit need_update();
}

void RegionTableWidget::save_region(void)
{
    if (currentRow() >= regions.size())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save region",item(currentRow(),0)->text() + output_format(),"NIFTI file(*nii.gz *.nii);;Text file(*.txt);;MAT file (*.mat);;All files(*)" );
    if (filename.isEmpty())
        return;
    if(QFileInfo(filename.toLower()).completeSuffix() != "mat" && QFileInfo(filename.toLower()).completeSuffix() != "txt")
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
    std::vector<float> no_trans;
    regions[currentRow()]->SaveToFile(filename.toLocal8Bit().begin(),
                                     cur_tracking_window.handle->is_qsdr ? cur_tracking_window.handle->trans_to_mni: no_trans);
    item(currentRow(),0)->setText(QFileInfo(filename).baseName());
}
QString RegionTableWidget::output_format(void)
{
    switch(cur_tracking_window["region_format"].toInt())
    {
    case 0:
        return ".nii.gz";
    case 1:
        return ".mat";
    case 2:
        return ".txt";
    }
    return "";
}

void RegionTableWidget::save_all_regions_to_dir(void)
{
    if (regions.empty())
        return;
    QString dir = QFileDialog::getExistingDirectory(this,"Open directory","");
    if(dir.isEmpty())
        return;
    begin_prog("save files...");
    for(unsigned int index = 0;check_prog(index,rowCount());++index)
        if (item(index,0)->checkState() == Qt::Checked) // either roi roa end or seed
        {
            std::vector<float> no_trans;
            std::string filename = dir.toLocal8Bit().begin();
            filename  += "/";
            filename  += item(index,0)->text().toLocal8Bit().begin();
            filename  += output_format().toStdString();
            regions[index]->SaveToFile(filename.c_str(),
                                         cur_tracking_window.handle->is_qsdr ? cur_tracking_window.handle->trans_to_mni: no_trans);
        }
}
void RegionTableWidget::save_all_regions(void)
{
    if (regions.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save region",item(currentRow(),0)->text() + output_format(),
                           "Region file(*nii.gz *.nii *.mat *.txt);;All file types (*)" );
    if (filename.isEmpty())
        return;
    QString base_name = QFileInfo(filename).completeBaseName();
    if(QFileInfo(base_name).suffix().toLower() == "nii")
        base_name = QFileInfo(base_name).completeBaseName();
    QString label_file = QFileInfo(filename).absolutePath()+"/"+base_name+".txt";
    std::ofstream out(label_file.toLocal8Bit().begin());
    image::geometry<3> geo = cur_tracking_window.handle->dim;
    image::basic_image<unsigned int, 3>mask(geo);
    for (unsigned int i = 0; i < regions.size(); ++i)
        if (item(i,0)->checkState() == Qt::Checked)
        {
            for (unsigned int j = 0; j < regions[i]->get().size(); ++j)
            {
                if (geo.is_valid(regions[i]->get()[j][0], regions[i]->get()[j][1],regions[i]->get()[j][2]))
                    mask[image::pixel_index<3>(regions[i]->get()[j][0],
                                           regions[i]->get()[j][1],
                                           regions[i]->get()[j][2], geo).index()] = i+1;

            }
            out << i+1
                << " " << item(i,0)->text().toStdString() << std::endl;
        }
    gz_nifti header;
    header.set_voxel_size(cur_tracking_window.current_slice->voxel_size);
    if(cur_tracking_window.handle->is_qsdr)
        header.set_LPS_transformation(
                    cur_tracking_window.handle->trans_to_mni.begin(),
                    mask.geometry());
    image::flip_xy(mask);
    header << mask;
    header.save_to_file(filename.toLocal8Bit().begin());


}

void RegionTableWidget::save_region_info(void)
{
    if (currentRow() >= regions.size())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save voxel information",item(currentRow(),0)->text() + "_info.txt",
                           "Text files (*.txt)" );
    if (filename.isEmpty())
        return;

    std::ofstream out(filename.toLocal8Bit().begin());
    out << "x\ty\tz";
    for(unsigned int index = 0;index < cur_tracking_window.handle->dir.num_fiber;++index)
            out << "\tdx" << index << "\tdy" << index << "\tdz" << index;

    for(unsigned int index = 0;index < cur_tracking_window.handle->view_item.size();++index)
        if(cur_tracking_window.handle->view_item[index].name != "color")
            out << "\t" << cur_tracking_window.handle->view_item[index].name;

    out << std::endl;
    for(int index = 0;index < regions[currentRow()]->get().size();++index)
    {
        std::vector<float> data;
        image::vector<3,short> point(regions[currentRow()]->get()[index]);
        cur_tracking_window.handle->get_voxel_info2(point[0],point[1],point[2],data);
        cur_tracking_window.handle->get_voxel_information(point[0],point[1],point[2],data);
        std::copy(point.begin(),point.end(),std::ostream_iterator<float>(out,"\t"));
        std::copy(data.begin(),data.end(),std::ostream_iterator<float>(out,"\t"));
        out << std::endl;
    }
}

void RegionTableWidget::delete_region(void)
{
    if (currentRow() >= regions.size())
        return;
    regions.erase(regions.begin()+currentRow());
    removeRow(currentRow());
    emit need_update();
}

void RegionTableWidget::delete_all_region(void)
{
    setRowCount(0);
    regions.clear();
    emit need_update();
}

void RegionTableWidget::whole_brain_points(std::vector<image::vector<3,short> >& points)
{
    image::geometry<3> geo = cur_tracking_window.handle->dim;
    float threshold = cur_tracking_window["fa_threshold"].toFloat();
    if(threshold == 0)
        threshold = 0.6*image::segmentation::otsu_threshold(image::make_image(cur_tracking_window.handle->dir.fa[0],cur_tracking_window.handle->dim));
    for (image::pixel_index<3>index(geo); index < geo.size();++index)
    {
        image::vector<3,short> pos(index);
        if(cur_tracking_window.handle->dir.fa[0][index.index()] > threshold)
            points.push_back(pos);
    }
}

void RegionTableWidget::whole_brain(void)
{
    std::vector<image::vector<3,short> > points;
    whole_brain_points(points);
    add_region("whole brain",seed_id);
    add_points(points,false,1.0);
    emit need_update();
}

void get_regions_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result)
{
    std::vector<std::string> titles;
    std::vector<std::vector<float> > data(regions.size());
    image::par_for(regions.size(),[&](unsigned int index){
        std::vector<std::string> dummy;
        regions[index]->get_quantitative_data(handle,(index == 0) ? titles : dummy,data[index]);
    });
    std::ostringstream out;
    out << "Name\t";
    for(unsigned int index = 0;index < regions.size();++index)
        out << region_name[index] << "\t";
    out << std::endl;
    for(unsigned int i = 0;i < titles.size();++i)
    {
        out << titles[i] << "\t";
        for(unsigned int j = 0;j < regions.size();++j)
        {
            if(i < data[j].size())
                out << data[j][i];
            out << "\t";
        }
        out << std::endl;
    }
    result = out.str();
}

void RegionTableWidget::show_statistics(void)
{
    if(currentRow() >= regions.size())
        return;
    std::string result;
    {
        std::vector<std::shared_ptr<ROIRegion> > active_regions;
        std::vector<std::string> region_name;
        for(unsigned int index = 0;index < regions.size();++index)
            if(item(index,0)->checkState() == Qt::Checked)
            {
                active_regions.push_back(regions[index]);
                region_name.push_back(item(index,0)->text().toStdString());
            }
        get_regions_statistics(cur_tracking_window.handle,active_regions,region_name,result);
    }
    QMessageBox msgBox;
    msgBox.setText("Region Statistics");
    msgBox.setDetailedText(result.c_str());
    msgBox.setStandardButtons(QMessageBox::Ok|QMessageBox::Save);
    msgBox.setDefaultButton(QMessageBox::Ok);
    QPushButton *copyButton = msgBox.addButton("Copy To Clipboard", QMessageBox::ActionRole);


    if(msgBox.exec() == QMessageBox::Save)
    {
        QString filename;
        filename = QFileDialog::getSaveFileName(
                    this,"Save satistics as",item(currentRow(),0)->text() + "_stat.txt",
                    "Text files (*.txt);;All files|(*)");
        if(filename.isEmpty())
            return;
        std::ofstream out(filename.toLocal8Bit().begin());
        out << result.c_str();
    }
    if (msgBox.clickedButton() == copyButton)
        QApplication::clipboard()->setText(result.c_str());
}

extern std::vector<float> mni_fa0_template_tran;
extern image::basic_image<float,3> mni_fa0_template;

bool RegionTableWidget::has_seeding(void)
{
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() &&
                item(index,0)->checkState() == Qt::Checked &&
                regions[index]->regions_feature == seed_id) // either roi roa end or seed
            return true;
    return false;
}
void RegionTableWidget::set_whole_brain(ThreadData* data)
{
    std::vector<image::vector<3,short> > points;
    whole_brain_points(points);
    data->roi_mgr.setRegions(cur_tracking_window.handle->dim,points,1.0,seed_id,"whole brain",image::vector<3>());
}

void RegionTableWidget::setROIs(ThreadData* data)
{
    // check if there is seeds
    if(!has_seeding())
        set_whole_brain(data);
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(index,0)->checkState() == Qt::Checked)
            data->roi_mgr.setRegions(cur_tracking_window.handle->dim,regions[index]->get(),
                                     regions[index]->resolution_ratio,
                             regions[index]->regions_feature,item(index,0)->text().toLocal8Bit().begin(),
                                     cur_tracking_window.handle->vs);
}

QString RegionTableWidget::getROIname(void)
{
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(index,0)->checkState() == Qt::Checked &&
             regions[index]->regions_feature == roi_id)
                return item(index,0)->text();
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(index,0)->checkState() == Qt::Checked &&
             regions[index]->regions_feature == seed_id)
                return item(index,0)->text();
    return "whole_brain";
}
void RegionTableWidget::undo(void)
{
    regions[currentRow()]->undo();
    emit need_update();
}
void RegionTableWidget::redo(void)
{
    regions[currentRow()]->redo();
    emit need_update();
}

void RegionTableWidget::do_action(QString action)
{
    if (regions.empty())
        return;
    unsigned total_region_size = regions.size();
    for (unsigned int k = 0;k < total_region_size;++k)
    if (!regions[k]->empty() && item(k,0)->checkState() == Qt::Checked)
    {
        ROIRegion& cur_region = *regions[k];
        cur_region.perform(action.toStdString());
        if(action == "thresholding")
        {
            image::basic_image<unsigned char, 3>mask;
            image::const_pointer_image<float,3> I = cur_tracking_window.current_slice->get_source();
            if(I.empty())
                return;
            mask.resize(I.geometry());
            auto m = std::minmax_element(I.begin(),I.end());
            bool ok;
            float threshold = QInputDialog::getDouble(this,
                "DSI Studio","Threshold:", image::segmentation::otsu_threshold(I),
                *m.first,
                *m.second,
                4, &ok);
            if(!ok)
                return;

            for(unsigned int i = 0;i < mask.size();++i)
                mask[i]  = I[i] > threshold ? 1:0;
            cur_region.LoadFromBuffer(mask);
        }
        if(action == "separate")
        {
            image::basic_image<unsigned char, 3>mask;
            cur_region.SaveToBuffer(mask, 1);
            QString name = item(currentRow(),0)->text();
            image::basic_image<unsigned int,3> labels;
            std::vector<std::vector<unsigned int> > r;
            image::morphology::connected_component_labeling(mask,labels,r);

            for(unsigned int j = 0,total_count = 0;j < r.size() && total_count < 10;++j)
                if(!r[j].empty())
                {
                    std::fill(mask.begin(),mask.end(),0);
                    for(unsigned int i = 0;i < r[j].size();++i)
                        mask[r[j][i]] = 1;
                    ROIRegion region(cur_tracking_window.handle->dim,cur_tracking_window.current_slice->voxel_size);
                    region.LoadFromBuffer(mask);
                    add_region(name + "_"+QString::number(total_count+1),
                               roi_id,region.show_region.color.color);
                    regions.back()->assign(region.get(),region.resolution_ratio);
                    ++total_count;
                }
        }
        }
    emit need_update();
}
