#include <regex>
#include <QFileDialog>
#include <QInputDialog>
#include <QContextMenuEvent>
#include <QMessageBox>
#include <QClipboard>
#include <QSettings>
#include <QTableWidgetItem>
#include <QTextStream>
#include "regiontablewidget.h"
#include "tracking/tracking_window.h"
#include "qcolorcombobox.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "opengl/glwidget.h"
#include "libs/tracking/fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"


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
            tipl::rgb color((unsigned int)(index.data(Qt::UserRole).toInt()));
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
void RegionTableWidget::add_region_from_atlas(std::shared_ptr<atlas> at,unsigned int label)
{
    float r = 1.0f;
    std::vector<tipl::vector<3,short> > points;
    cur_tracking_window.handle->get_atlas_roi(at,label,points,r);
    if(points.empty())
        return;
    add_region(at->get_list()[label].c_str(),roi_id);
    regions.back()->resolution_ratio = r;
    regions.back()->add_points(points,false,r);
}
void RegionTableWidget::begin_update(void)
{
    cur_tracking_window.scene.no_show = true;
    cur_tracking_window.disconnect(cur_tracking_window.regionWidget,SIGNAL(need_update()),cur_tracking_window.glWidget,SLOT(updateGL()));
    cur_tracking_window.disconnect(cur_tracking_window.regionWidget,SIGNAL(cellChanged(int,int)),cur_tracking_window.glWidget,SLOT(updateGL()));
}

void RegionTableWidget::end_update(void)
{
    cur_tracking_window.scene.no_show = false;
    cur_tracking_window.connect(cur_tracking_window.regionWidget,SIGNAL(need_update()),cur_tracking_window.glWidget,SLOT(updateGL()));
    cur_tracking_window.connect(cur_tracking_window.regionWidget,SIGNAL(cellChanged(int,int)),cur_tracking_window.glWidget,SLOT(updateGL()));
}

void RegionTableWidget::add_region(QString name,unsigned char feature,unsigned int color)
{
    if(color == 0x00FFFFFF || !color)
    {
        tipl::rgb c;
        c.from_hsl(((color_gen++)*1.1-std::floor((color_gen++)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        color = c.color;
    }
    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle));
    regions.back()->show_region.color = color;
    regions.back()->regions_feature = feature;

    insertRow(regions.size()-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(name);
    QTableWidgetItem *item1 = new QTableWidgetItem(QString::number((int)feature));
    QTableWidgetItem *item2 = new QTableWidgetItem();

    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item0->setCheckState(Qt::Checked);
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::UserRole,0xFF000000 | color);


    setItem(regions.size()-1, 0, item0);
    setItem(regions.size()-1, 1, item1);
    setItem(regions.size()-1, 2, item2);


    openPersistentEditor(item1);
    openPersistentEditor(item2);

    setRowHeight(regions.size()-1,22);
    setCurrentCell(regions.size()-1,0);



}
void RegionTableWidget::check_check_status(int row, int col)
{
    if (col != 0)
        return;
    setCurrentCell(row,col);
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

void RegionTableWidget::move_slice_to_current_region(void)
{
    if(currentRow() == -1)
        return;
    auto current_slice = cur_tracking_window.current_slice;
    auto current_region = regions[currentRow()];
    if(current_region->region.empty())
        return;
    tipl::vector<3,float> p(current_region->get_center());
    if(!current_slice->is_diffusion_space)
    {
        auto iT = current_slice->T;
        iT.inv();
        p.to(iT);
    }
    cur_tracking_window.move_slice_to(p);
}

void RegionTableWidget::draw_region(tipl::color_image& I)
{
    auto current_slice = cur_tracking_window.current_slice;
    int slice_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];
    auto checked_regions = get_checked_regions();
    if(checked_regions.empty())
        return;
    if(current_slice->is_diffusion_space)
    {
        for(unsigned int roi_index = 0;roi_index < checked_regions.size();++roi_index)
        {
            float r = checked_regions[roi_index]->resolution_ratio;
            unsigned int cur_color = checked_regions[roi_index]->show_region.color;
            tipl::par_for(checked_regions[roi_index]->size(),[&](unsigned int index)
            {
                tipl::vector<3,float> p(checked_regions[roi_index]->region[index]);
                if(r != 1.0)
                    p /= r;
                p.round();
                int X, Y, Z;
                tipl::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
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
            tipl::image<unsigned int,3> buf(cur_tracking_window.handle->dim);
            for(unsigned int roi_index = 0;roi_index < checked_regions.size();++roi_index)
            {
                unsigned int cur_color = checked_regions[roi_index]->show_region.color;
                if(checked_regions[roi_index]->resolution_ratio == 1)
                tipl::par_for(checked_regions[roi_index]->size(),[&](unsigned int index)
                {
                    tipl::pixel_index<3> pindex(checked_regions[roi_index]->region[index][0],
                                                 checked_regions[roi_index]->region[index][1],
                                                 checked_regions[roi_index]->region[index][2],buf.geometry());
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
                    tipl::pixel_index<3> pindex(dx,dy,dz,buf.geometry());
                    if(buf[pindex.index()] != 0)
                        I[index] = (unsigned int)I[index] | buf[pindex.index()];

                }
        }

        // now the most time consuming part with high resolution regions
        tipl::par_for(checked_regions.size(),[&](unsigned int roi_index)
        {
            if(checked_regions[roi_index]->resolution_ratio == 1)
                return;
            unsigned int cur_color = checked_regions[roi_index]->show_region.color;
            float r = checked_regions[roi_index]->resolution_ratio;
            tipl::geometry<3> geo(cur_tracking_window.handle->dim);
            geo[0] *= r;
            geo[1] *= r;
            geo[2] *= r;
            std::vector<std::vector<std::vector<unsigned int> > > buf(geo[0]);
            for(unsigned int index = 0;index < checked_regions[roi_index]->size();++index)
            {
                const auto& p = checked_regions[roi_index]->region[index];
                if(!geo.is_valid(p))
                    return;
                auto& x_pos = buf[p[0]];
                if(x_pos.empty())
                    x_pos.resize(geo[1]);
                auto& y_pos = x_pos[p[1]];
                if(y_pos.empty())
                    y_pos.resize(geo[2]);
                y_pos[p[2]] = cur_color;

                tipl::vector<3> v(p),v2;
                v /= r;
                v.to(current_slice->invT);
                tipl::space2slice(cur_tracking_window.cur_dim,v[0],v[1],v[2],v2[0],v2[1],v2[2]);
                v2.round();
                if(v2[2] == slice_pos && I.geometry().is_valid(v2))
                {
                    int pos = v2[0]+v2[1]*I.width();
                    I[pos] = (unsigned int)I[pos] | cur_color;
                }
            }

            for(int y = 0,index = 0;y < I.height();++y)
                for(int x = 0;x < I.width();++x,++index)
                {

                    tipl::vector<3,float> v;
                    tipl::slice2space(cur_tracking_window.cur_dim, x, y,
                                       slice_pos, v[0],v[1],v[2]);
                    v.to(current_slice->T);
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

void RegionTableWidget::draw_edge(QImage& qimage,QImage& scaled_image,bool draw_all)
{
    auto current_slice = cur_tracking_window.current_slice;
    // during region removal, there will be a call with invalid currentRow
    if(regions.empty() || currentRow() >= regions.size() || currentRow() == -1)
        return;
    std::vector<std::shared_ptr<ROIRegion> > checked_regions;
    int cur_roi_index = -1;
    if(draw_all)
    {
        checked_regions = get_checked_regions();
        if(currentRow() >= 0)
        for(int i = 0;i < checked_regions.size();++i)
            if(checked_regions[i] == regions[currentRow()])
            {
                cur_roi_index = i;
                break;
            }
    }
    else
    if(currentRow() >= 0)
    {
        if(item(currentRow(),0)->checkState() != Qt::Checked)
            return;
        checked_regions.push_back(regions[currentRow()]);
        cur_roi_index = 0;
    }
    if(checked_regions.empty())
        return;
    float display_ratio = (float)scaled_image.width()/(float)qimage.width();
    int slice_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];

    //if(display_ratio >= 1.0f)
    for (int roi_index = 0;roi_index < checked_regions.size();++roi_index)
    {
        tipl::image<unsigned char,2> cur_image_mask;
        cur_image_mask.resize(tipl::geometry<2>(qimage.width(),qimage.height()));

        float r = checked_regions[roi_index]->resolution_ratio;
        auto iT = current_slice->T;
        iT.inv();

        tipl::par_for(checked_regions[roi_index]->size(),[&](unsigned int index)
        {
            tipl::vector<3,float> p(checked_regions[roi_index]->region[index]);
            if(r != 1.0)
                p /= r;
            if(!current_slice->is_diffusion_space)
                p.to(iT);
            p.round();
            int X, Y, Z;
            tipl::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
            if (slice_pos != Z || X < 0 || Y < 0 || X >= cur_image_mask.width() || Y >= cur_image_mask.height())
                return;
            cur_image_mask.at(X,Y) = 1;
        });

        unsigned int cur_color = 0xFFFFFFFF;
        if(draw_all)
            cur_color = checked_regions[roi_index]->show_region.color;

        QPainter paint(&scaled_image);
        paint.setBrush(Qt::NoBrush);
        QPen pen(QColor(cur_color), cur_roi_index == roi_index ? display_ratio : display_ratio*0.5f, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
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
            "Input resolution ratio (e.g. 2 for 2X, 8 for 8X",8,2,64,2,&ok);
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
void load_nii_label(const char* filename,std::map<int,std::string>& label_map)
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
            int num = 0;
            read_line >> num >> txt;
            label_map[num] = txt;
        }
    }
}
void get_roi_label(QString file_name,std::map<int,std::string>& label_map,
                          std::map<int,tipl::rgb>& label_color,bool is_free_surfer,bool mute_cmd = true)
{
    label_map.clear();
    label_color.clear();
    QString base_name = QFileInfo(file_name).baseName();
    if(base_name.contains("aparc+aseg") || is_free_surfer) // FreeSurfer
    {
        if(!mute_cmd)
            std::cout << "Use freesurfer labels." << std::endl;
        QFile data(":/data/FreeSurferColorLUT.txt");
        if (data.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream in(&data);
            while (!in.atEnd())
            {
                QString line = in.readLine();
                if(line.isEmpty() || line[0] == '#')
                    continue;
                std::istringstream in(line.toStdString());
                int value,r,b,g;
                std::string name;
                in >> value >> name >> r >> g >> b;
                label_map[value] = name;
                label_color[value] = tipl::rgb(uint8_t(r),uint8_t(g),uint8_t(b));
            }
            return;
        }
    }
    QString label_file = QFileInfo(file_name).absolutePath()+"/"+base_name+".txt";
    if(QFileInfo(label_file).exists())
    {
        load_nii_label(label_file.toLocal8Bit().begin(),label_map);
        std::cout << "Load label file:" << label_file.toStdString() << std::endl;
        return;
    }
    if(!mute_cmd)
        std::cout << "No label file found. Use default ROI numbering." << std::endl;
}
bool is_label_image(const tipl::image<float,3>& I);
bool RegionTableWidget::load_multiple_roi_nii(QString file_name)
{
    gz_nifti header;
    if (!header.load_from_file(file_name.toLocal8Bit().begin()))
        return false;

    tipl::image<unsigned int, 3> from;
    {
        tipl::image<float, 3> tmp;
        header.toLPS(tmp);
        if(is_label_image(tmp))
            from = tmp;
        else
        {
            from.resize(tmp.geometry());
            for(size_t i = 0;i < from.size();++i)
                from[i] = (tmp[i] == 0.0f ? 0:1);
        }
    }

    std::vector<unsigned short> value_list;
    std::vector<unsigned short> value_map(std::numeric_limits<unsigned short>::max()+1);

    {
        unsigned short max_value = 0;
        for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
        {
            value_map[uint16_t(from[index.index()])] = 1;
            max_value = std::max<unsigned short>(uint16_t(from[index.index()]),max_value);
        }
        for(unsigned short value = 1;value <= max_value;++value)
            if(value_map[value])
            {
                value_map[value] = uint16_t(value_list.size());
                value_list.push_back(value);
            }
    }

    bool multiple_roi = value_list.size() > 1;


    std::map<int,std::string> label_map;
    std::map<int,tipl::rgb> label_color;

    std::string des(header.get_descrip());
    if(multiple_roi)
        get_roi_label(file_name,label_map,label_color,des.find("FreeSurfer") == 0);

    tipl::matrix<4,4,float> convert;
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
        float r1 = float(from.width())/float(cur_tracking_window.handle->dim[0]);
        float r2 = float(from.height())/float(cur_tracking_window.handle->dim[1]);
        float r3 = float(from.depth())/float(cur_tracking_window.handle->dim[2]);
        if(std::fabs(r1-r2) < 0.02f && std::fabs(r1-r3) < 0.02f)
            has_transform = false;
        else
        if(cur_tracking_window.handle->is_qsdr && !has_transform)// use transformation information
        {
            // searching QSDR mappings
            tipl::image<unsigned int, 3> new_from;
            for(unsigned int index = 0;index < cur_tracking_window.handle->view_item.size();++index)
                if(cur_tracking_window.handle->view_item[index].native_geo == from.geometry())
                {
                    new_from.resize(cur_tracking_window.handle->dim);
                    for(tipl::pixel_index<3> pos(new_from.geometry());pos < new_from.size();++pos)
                    {
                        tipl::vector<3> new_pos(cur_tracking_window.handle->view_item[index].mx[pos.index()],
                                                 cur_tracking_window.handle->view_item[index].my[pos.index()],
                                                 cur_tracking_window.handle->view_item[index].mz[pos.index()]);
                        new_pos.round();
                        new_from[pos.index()] = from.at(uint32_t(new_pos[0]),uint32_t(new_pos[1]),uint32_t(new_pos[2]));
                    }
                    break;
                }

            if(new_from.empty())
            {
                QMessageBox::information(this,"Warning","The nii file has different image dimension. Transformation will be applied to load the region",0);
                header.get_image_transformation(convert);
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
        ROIRegion region(cur_tracking_window.handle);

        if(has_transform)
            region.LoadFromBuffer(from,convert);
        else
            region.LoadFromBuffer(from);

        unsigned int color = 0;
        unsigned char type = roi_id;

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
        add_region(QFileInfo(file_name).baseName(),type,color);

        regions.back()->assign(region.get_region_voxels_raw(),region.resolution_ratio);
        item(currentRow(),0)->setCheckState(Qt::Checked);
        item(currentRow(),0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
        return true;
    }

    std::vector<std::vector<tipl::vector<3,short> > > region_points(value_list.size());
    if(has_transform)
    {
        tipl::geometry<3> geo = cur_tracking_window.handle->dim;
        for (tipl::pixel_index<3>index(geo);index < geo.size();++index)
        {
            tipl::vector<3> p(index.begin()); // point in subject space
            p.to(convert); // point in "from" space
            p += 0.5f;
            if (from.geometry().is_valid(p))
            {
                unsigned int value = from.at(uint32_t(p[0]),uint32_t(p[1]),uint32_t(p[2]));
                if(value)
                    region_points[value_map[value]].push_back(tipl::vector<3,short>(index.x(),index.y(),index.z()));
            }
        }
    }
    else
    {
        if(from.geometry() == cur_tracking_window.handle->dim)
        {
            for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
                if(from[index.index()])
                    region_points[value_map[from[index.index()]]].push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
        }
        else
        {
            float r = float(cur_tracking_window.handle->dim[0])/float(from.width());
            for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
                if(from[index.index()])
                    region_points[value_map[from[index.index()]]].
                            push_back(tipl::vector<3,short>(r*index.x(), r*index.y(),r*index.z()));
        }
    }
    begin_prog("loading ROIs");
    begin_update();
    for(uint32_t i = 0;check_prog(i,region_points.size());++i)
        if(!region_points[i].empty())
        {
            unsigned short value = value_list[i];
            QString name = (label_map.find(value) == label_map.end() ?
                 QFileInfo(file_name).baseName() + "_" + QString::number(value):QString(label_map[value].c_str()));
            add_region(name,roi_id,label_color.empty() ? 0x00FFFFFF : label_color[value].color);
            regions.back()->add_points(region_points[i],false,1.0f);
            item(currentRow(),0)->setCheckState(Qt::Unchecked);
            item(currentRow(),0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
        }
    end_update();
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
            QFileInfo(filenames[index]).suffix() == "nii" ||
            QFileInfo(filenames[index]).suffix() == "hdr") &&
                load_multiple_roi_nii(filenames[index]))
            continue;

        ROIRegion region(cur_tracking_window.handle);
        if(!region.LoadFromFile(filenames[index].toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"error","Unknown file format",0);
            return;
        }
        add_region(QFileInfo(filenames[index]).baseName(),roi_id,region.show_region.color.color);
        regions.back()->assign(region.get_region_voxels_raw(),region.resolution_ratio);

    }
    emit need_update();
}

void RegionTableWidget::load_mni_region(void)
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,"Open region",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),"NIFTI files (*.nii *nii.gz);;All files (*)" );
    if (filenames.isEmpty())
        return;

    if(!cur_tracking_window.can_map_to_mni())
    {
        QMessageBox::information(this,"Error","Atlas is not supported for the current image resolution.",0);
        return;
    }

    for (unsigned int index = 0;index < filenames.size();++index)
    {
        std::shared_ptr<atlas> a(new atlas);
        a->filename = filenames[index].toStdString();
        a->load_from_file();
        begin_prog("loading");
        begin_update();
        for(int j = 0;check_prog(j,a->get_list().size());++j)
            add_region_from_atlas(a,j);
        end_update();
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
    cur_tracking_window.glWidget->no_update = true;
    cur_tracking_window.scene.no_show = true;
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
    cur_tracking_window.scene.no_show = false;
    cur_tracking_window.glWidget->no_update = false;
    emit need_update();
}

void RegionTableWidget::uncheck_all(void)
{
    cur_tracking_window.glWidget->no_update = true;
    cur_tracking_window.scene.no_show = true;
    for(unsigned int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
    cur_tracking_window.scene.no_show = false;
    cur_tracking_window.glWidget->no_update = false;
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
    regions[currentRow()]->SaveToFile(filename.toLocal8Bit().begin());
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
            std::string filename = dir.toLocal8Bit().begin();
            filename  += "/";
            filename  += item(index,0)->text().toLocal8Bit().begin();
            filename  += output_format().toStdString();
            regions[index]->SaveToFile(filename.c_str());
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
    tipl::geometry<3> geo = cur_tracking_window.handle->dim;
    tipl::image<unsigned int, 3> mask(geo);
    for (unsigned int i = 0; i < regions.size(); ++i)
        if (item(i,0)->checkState() == Qt::Checked)
        {
            for (unsigned int j = 0; j < regions[i]->size(); ++j)
            {
                tipl::vector<3,short> p = regions[i]->get_region_voxel(j);
                if (geo.is_valid(p))
                    mask[tipl::pixel_index<3>(p[0],p[1],p[2], geo).index()] = i+1;

            }
            out << i+1 << " " << item(i,0)->text().toStdString() << std::endl;
        }
    gz_nifti header;
    header.set_voxel_size(cur_tracking_window.current_slice->voxel_size);
    if(cur_tracking_window.handle->is_qsdr)
        header.set_LPS_transformation(
                    cur_tracking_window.handle->trans_to_mni,
                    mask.geometry());
    tipl::flip_xy(mask);
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
    for(int index = 0;index < regions[currentRow()]->size();++index)
    {
        std::vector<float> data;
        tipl::vector<3,short> point = regions[currentRow()]->get_region_voxel(index);
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


void get_regions_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result)
{
    std::vector<std::string> titles;
    std::vector<std::vector<float> > data(regions.size());
    tipl::par_for(regions.size(),[&](unsigned int index){
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


void RegionTableWidget::whole_brain_points(std::vector<tipl::vector<3,short> >& points)
{
    tipl::geometry<3> geo = cur_tracking_window.handle->dim;
    float threshold = cur_tracking_window.get_fa_threshold();
    for (tipl::pixel_index<3>index(geo); index < geo.size();++index)
    {
        tipl::vector<3,short> pos(index);
        if(cur_tracking_window.handle->dir.fa[0][index.index()] > threshold)
            points.push_back(pos);
    }
}

void RegionTableWidget::whole_brain(void)
{
    std::vector<tipl::vector<3,short> > points;
    whole_brain_points(points);
    add_region("whole brain",seed_id);
    add_points(points,false,1.0);
    emit need_update();
}

void RegionTableWidget::setROIs(ThreadData* data)
{
    int roi_count = 0;
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(int(index),0)->checkState() == Qt::Checked
                && regions[index]->regions_feature == 0 /*ROI*/)
            ++roi_count;
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(int(index),0)->checkState() == Qt::Checked
                && !(regions[index]->regions_feature == 0 && roi_count > 5))
            data->roi_mgr->setRegions(cur_tracking_window.handle->dim,regions[index]->get_region_voxels_raw(),
                                     regions[index]->resolution_ratio,
                             regions[index]->regions_feature,item(int(index),0)->text().toLocal8Bit().begin(),
                                     cur_tracking_window.handle->vs);
    // auto track
    if(cur_tracking_window.ui->target->currentIndex() > 0 &&
       cur_tracking_window.tractography_atlas.get())
        data->roi_mgr->setAtlas(cur_tracking_window.tractography_atlas,
                                uint32_t(cur_tracking_window.ui->target->currentIndex()-1));

    data->roi_mgr->setWholeBrainSeed(cur_tracking_window.handle,cur_tracking_window.get_fa_threshold());
}

QString RegionTableWidget::getROIname(void)
{
    for (size_t index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(int(index),0)->checkState() == Qt::Checked &&
             regions[index]->regions_feature == roi_id)
                return item(int(index),0)->text();
    for (size_t index = 0;index < regions.size();++index)
        if (!regions[index]->empty() && item(int(index),0)->checkState() == Qt::Checked &&
             regions[index]->regions_feature == seed_id)
                return item(int(index),0)->text();
    return "whole_brain";
}
void RegionTableWidget::undo(void)
{
    if(currentRow() < 0)
        return;
    regions[size_t(currentRow())]->undo();
    emit need_update();
}
void RegionTableWidget::redo(void)
{
    if(currentRow() < 0)
        return;
    regions[size_t(currentRow())]->redo();
    emit need_update();
}

void RegionTableWidget::do_action(QString action)
{
    if(regions.empty() || currentRow() < 0)
        return;
    size_t roi_index = currentRow();

    {
        ROIRegion& cur_region = *regions[size_t(roi_index)];
        if(cur_tracking_window.ui->all_edit->isChecked())
            for_each_checked_region([&](std::shared_ptr<ROIRegion> region){region->perform(action.toStdString());});
        else
            cur_region.perform(action.toStdString());


        if(action == "A-B" || action == "B-A" || action == "A*B")
        {
            auto checked_regions = get_checked_regions();
            if(checked_regions.size() < 2)
                return;
            tipl::image<unsigned char, 3> A,B;
            checked_regions[0]->SaveToBuffer(A, 1);
            for(size_t r = 1;r < checked_regions.size();++r)
            {
                checked_regions[r]->SaveToBuffer(B, 1);
                if(action == "A-B")
                {
                    for(size_t i = 0;i < A.size();++i)
                        if(B[i])
                            A[i] = 0;
                    checked_regions[0]->LoadFromBuffer(A);
                }
                if(action == "B-A")
                {
                    for(size_t i = 0;i < A.size();++i)
                        if(A[i])
                            B[i] = 0;
                    checked_regions[r]->LoadFromBuffer(B);
                }
                if(action == "A*B")
                {
                    for(size_t i = 0;i < A.size();++i)
                        B[i] = (A[i] & B[i]);
                    checked_regions[r]->LoadFromBuffer(B);
                }
            }
        }
        if(action == "set_opacity")
        {
            bool ok;
            double threshold = QInputDialog::getDouble(this,
                "DSI Studio","Set opacity (between 0 and 1)",
                    double(cur_region.opacity < 0.0f ?
                    cur_tracking_window["region_alpha"].toFloat() : cur_region.opacity),
                    0.0,1.0,1,&ok);
            if(!ok)
                return;
            cur_region.opacity = float(threshold);
        }
        if(action == "threshold")
        {
            tipl::image<unsigned char, 3>mask;
            tipl::const_pointer_image<float,3> I = cur_tracking_window.current_slice->get_source();
            if(I.empty())
                return;
            mask.resize(I.geometry());
            double m = *std::max_element(I.begin(),I.end());
            bool ok;
            bool flip = false;
            float threshold = float(QInputDialog::getDouble(this,
                "DSI Studio","Threshold (assign negative value to get low pass):",
                double(tipl::segmentation::otsu_threshold(I)),-m,m,4, &ok));
            if(!ok)
                return;
            if(threshold < 0)
            {
                flip = true;
                threshold = -threshold;
            }

            for(unsigned int i = 0;i < mask.size();++i)
                mask[i]  = (I[i] > threshold ^ flip )? 1:0;

            if(cur_tracking_window.current_slice->is_diffusion_space)
                cur_region.LoadFromBuffer(mask);
            else
            {
                auto iT = cur_tracking_window.current_slice->T;
                iT.inv();
                cur_region.LoadFromBuffer(mask,iT);
            }

        }
        if(action == "threshold_current")
        {
            tipl::const_pointer_image<float,3> I = cur_tracking_window.current_slice->get_source();
            if(I.empty())
                return;
            double m = *std::max_element(I.begin(),I.end());
            bool ok;
            bool flip = false;
            float threshold = float(QInputDialog::getDouble(this,
                "DSI Studio","Threshold (assign negative value to get low pass):",
                double(tipl::segmentation::otsu_threshold(I)),-m,m,4, &ok));
            if(!ok)
                return;
            if(threshold < 0)
            {
                flip = true;
                threshold = -threshold;
            }
            const std::vector<tipl::vector<3,short> >& region = cur_region.get_region_voxels_raw();
            std::vector<tipl::vector<3,short> > new_region;
            if(cur_tracking_window.current_slice->is_diffusion_space)
            {
                if(cur_region.resolution_ratio != 1.0f)
                {
                    for(size_t i = 0;i < region.size();++i)
                    {
                        tipl::vector<3,float> pos(region[i]);
                        pos /= cur_region.resolution_ratio;
                        if(I.geometry().is_valid(pos[0],pos[1],pos[2]) &&
                            I.at(pos[0],pos[1],pos[2]) > threshold ^ flip)
                                   new_region.push_back(region[i]);
                    }
                }
                else
                for(size_t i = 0;i < region.size();++i)
                {
                    if(I.geometry().is_valid(region[i][0],region[i][1],region[i][2]) &&
                        I.at(region[i][0],region[i][1],region[i][2]) > threshold ^ flip)
                               new_region.push_back(region[i]);
                }
            }
            else
            {
                auto iT = cur_tracking_window.current_slice->T;
                iT.inv();
                for(size_t i = 0;i < region.size();++i)
                {
                    tipl::vector<3,float> pos(region[i]);
                    if(cur_region.resolution_ratio != 1.0f)
                        pos /= cur_region.resolution_ratio;
                    pos.to(iT);
                    if(I.geometry().is_valid(pos[0],pos[1],pos[2]) &&
                        I.at(pos[0],pos[1],pos[2]) > threshold ^ flip)
                               new_region.push_back(region[i]);
                }
            }
            cur_region.assign(new_region,cur_region.resolution_ratio);
        }
        if(action == "separate")
        {
            tipl::image<unsigned char, 3>mask;
            cur_region.SaveToBuffer(mask, 1);
            QString name = item(roi_index,0)->text();
            tipl::image<unsigned int,3> labels;
            std::vector<std::vector<unsigned int> > r;
            tipl::morphology::connected_component_labeling(mask,labels,r);

            for(unsigned int j = 0,total_count = 0;j < r.size() && total_count < 256;++j)
                if(!r[j].empty())
                {
                    std::fill(mask.begin(),mask.end(),0);
                    for(unsigned int i = 0;i < r[j].size();++i)
                        mask[r[j][i]] = 1;
                    ROIRegion region(cur_tracking_window.handle);
                    region.LoadFromBuffer(mask);
                    add_region(name + "_"+QString::number(total_count+1),
                               roi_id,region.show_region.color.color);
                    regions.back()->assign(region.get_region_voxels_raw(),region.resolution_ratio);
                    ++total_count;
                }
        }
        if(action.contains("sort_"))
        {
            // reverse when repeated
            bool negate = false;
            {
                static QString last_action;
                if(action == last_action)
                {
                    last_action  = "";
                    negate = true;
                }
            else
                last_action = action;
            }
            std::vector<unsigned int> arg;
            if(action == "sort_name")
            {
                arg = tipl::arg_sort(regions.size(),[&]
                (size_t lhs,size_t rhs)
                {
                    return negate ^ (item(lhs,0)->text() < item(rhs,0)->text());
                });
            }
            else
            {
                std::vector<std::vector<float> > data(regions.size());
                tipl::par_for(regions.size(),[&](unsigned int index){
                    std::vector<std::string> dummy;
                    regions[index]->get_quantitative_data(cur_tracking_window.handle,dummy,data[index]);
                });
                size_t comp_index = 0; // sort_size
                if(action == "sort_x")
                    comp_index = 2;
                if(action == "sort_y")
                    comp_index = 3;
                if(action == "sort_z")
                    comp_index = 4;

                arg = tipl::arg_sort(data,[negate,comp_index]
                    (const std::vector<float>& lhs,const std::vector<float>& rhs)
                    {
                        return negate ^ (lhs[comp_index] < rhs[comp_index]);
                    });
            }

            std::vector<std::shared_ptr<ROIRegion> > new_region(arg.size());
            std::vector<int> new_region_checked(arg.size());
            std::vector<std::string> new_region_names(arg.size());
            for(size_t i = 0;i < arg.size();++i)
            {
                new_region[i] = regions[arg[i]];
                new_region_checked[i] = item(arg[i],0)->checkState() == Qt::Checked ? 1:0;
                new_region_names[i] = item(arg[i],0)->text().toStdString();
            }
            regions.swap(new_region);
            for(size_t i = 0;i < arg.size();++i)
            {
                item(i,0)->setCheckState(new_region_checked[i] ? Qt::Checked : Qt::Unchecked);
                item(i,0)->setText(new_region_names[i].c_str());
                closePersistentEditor(item(i,1));
                closePersistentEditor(item(i,2));
                item(i,1)->setData(Qt::DisplayRole,regions[i]->regions_feature);
                item(i,2)->setData(Qt::UserRole,regions[i]->show_region.color.color);
                openPersistentEditor(item(i,1));
                openPersistentEditor(item(i,2));
            }
        }
    }
    emit need_update();
}
