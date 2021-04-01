#include <regex>
#include <QFileDialog>
#include <QInputDialog>
#include <QContextMenuEvent>
#include <QMessageBox>
#include <QClipboard>
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
        comboBox->addItem("NotEnd");
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
        dynamic_cast<QComboBox*>(editor)->setCurrentIndex(index.model()->data(index).toString().toInt());
    else
        if (index.column() == 2)
        {
            tipl::rgb color(uint32_t(index.data(Qt::UserRole).toInt()));
            dynamic_cast<QColorToolButton*>(editor)->setColor(
                QColor(color.r,color.g,color.b,color.a));
        }
        else
            return QItemDelegate::setEditorData(editor,index);
}

void ImageDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                 const QModelIndex &index) const
{
    if (index.column() == 1)
        model->setData(index,QString::number(dynamic_cast<QComboBox*>(editor)->currentIndex()));
    else
        if (index.column() == 2)
            model->setData(index,int((dynamic_cast<QColorToolButton*>(editor)->color().rgba())),Qt::UserRole);
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

    connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_check_status(int,int)));
    connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateRegions(QTableWidgetItem*)));
    setEditTriggers(QAbstractItemView::DoubleClicked|QAbstractItemView::EditKeyPressed);
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
        regions[uint32_t(item->row())]->regions_feature = uint8_t(item->text().toInt());
    else
        if (item->column() == 2)
        {
            regions[uint32_t(item->row())]->show_region.color = uint32_t(item->data(Qt::UserRole).toInt());
            emit need_update();
        }
}

QColor RegionTableWidget::currentRowColor(void)
{
    return uint32_t(regions[uint32_t(currentRow())]->show_region.color);
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
        ++color_gen;
        c.from_hsl(((color_gen)*1.1-std::floor((color_gen)*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        color = c.color;
    }
    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle.get()));
    regions.back()->show_region.color = color;
    regions.back()->regions_feature = feature;

    insertRow(int(regions.size())-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(name);
    QTableWidgetItem *item1 = new QTableWidgetItem(QString::number(int(feature)));
    QTableWidgetItem *item2 = new QTableWidgetItem();

    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item0->setCheckState(Qt::Checked);
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::UserRole,0xFF000000 | color);


    setItem(int(regions.size())-1, 0, item0);
    setItem(int(regions.size())-1, 1, item1);
    setItem(int(regions.size())-1, 2, item2);


    openPersistentEditor(item1);
    openPersistentEditor(item2);

    setRowHeight(int(regions.size())-1,22);
    setCurrentCell(int(regions.size())-1,0);

    if(cur_tracking_window.ui->target->count())
        cur_tracking_window.ui->target->setCurrentIndex(0);
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

bool RegionTableWidget::command(QString cmd,QString param,QString)
{
    if(cmd == "save_all_regions_to_dir")
    {
        begin_prog("save files...");
        for(int index = 0;check_prog(index,rowCount());++index)
            if (item(index,0)->checkState() == Qt::Checked) // either roi roa end or seed
            {
                std::string filename = param.toStdString();
                filename  += "/";
                filename  += item(index,0)->text().toStdString();
                filename  += output_format().toStdString();
                regions[size_t(index)]->SaveToFile(filename.c_str());
            }
        return true;
    }
    if(cmd == "detele_all_region")
    {
        delete_all_region();
        return true;
    }
    if(cmd == "load_region")
    {
        // check for multiple nii
        if(QFileInfo(param).suffix() == "gz" ||
            QFileInfo(param).suffix() == "nii" ||
            QFileInfo(param).suffix() == "hdr")
        {
            if(!load_multiple_roi_nii(param))
                QMessageBox::information(this,"error","Invalid Region File. If it is created from T1W/T2W, please insert the original T1W/T2W in [Slices][Insert]",0);
            emit need_update();
            return true;
        }
        ROIRegion region(cur_tracking_window.handle.get());
        if(!region.LoadFromFile(param.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"error","Unknown file format",0);
            return true;
        }
        add_region(QFileInfo(param).baseName(),roi_id,region.show_region.color.color);
        regions.back()->assign(region.get_region_voxels_raw(),region.resolution_ratio);
        emit need_update();
        return true;
    }
    return false;
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

template<typename func,typename func2>
void draw_region_voxels(tracking_window& cur_tracking_window,
                        int w,int h,
                        const std::vector<std::shared_ptr<ROIRegion> >& regions,
                        func draw_voxel,
                        func2 draw_edge)
{
    auto current_slice = cur_tracking_window.current_slice;
    int slice_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];
    std::vector<uint32_t> yw((size_t(h)));
    for(uint32_t y = 0;y < uint32_t(h);++y)
        yw[y] = y*uint32_t(w);

    for (uint32_t roi_index = 0;roi_index < regions.size();++roi_index)
    {
        auto color = regions[roi_index]->show_region.color;
        float r = regions[roi_index]->resolution_ratio;
        auto iT = current_slice->T;
        iT.inv();

        if(!current_slice->is_diffusion_space)
        {
            // apply resolution ratio to iT
            if(r != 1.0f)
                for(int i : {0,1,2,4,5,6,8,9,10})
                    iT[uint32_t(i)] /= r;
            auto rT = iT;
            rT[3] = rT[7] = rT[11] = 0.0f;
            uint32_t x_range = 0;
            uint32_t y_range = 0;
            uint32_t z_range = 0;
            tipl::vector<3,float> p;
            tipl::slice2space(cur_tracking_window.cur_dim,1.0f,0.0f,0.0f,p[0],p[1],p[2]);
            p.to(rT);
            x_range = uint32_t(std::ceil(p.length()));
            tipl::slice2space(cur_tracking_window.cur_dim,0.0f,1.0f,0.0f,p[0],p[1],p[2]);
            p.to(rT);
            y_range = uint32_t(std::ceil(p.length()));
            tipl::slice2space(cur_tracking_window.cur_dim,0.0f,0.0f,1.0f,p[0],p[1],p[2]);
            p.to(rT);
            z_range = uint32_t(std::ceil(p.length()));

            tipl::par_for(regions[roi_index]->size(),[&](unsigned int index)
            {
                tipl::vector<3,float> p(regions[roi_index]->region[index]);
                p.to(iT);
                float X,Y,Z;
                tipl::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
                if (std::fabs(float(slice_pos)-Z) >= z_range || X < 0.0f || Y < 0.0f ||
                    int(X) >= w || int(Y) >= h)
                    return;
                uint32_t iX = uint32_t(std::floor(X));
                uint32_t iY = uint32_t(std::floor(Y));
                uint32_t pos = yw[iY]+iX;
                for(uint32_t dy = 0;dy < y_range;++dy,pos += uint32_t(w))
                    for(uint32_t dx = 0,pos2 = pos;dx < x_range;++dx,++pos2)
                        draw_voxel(pos2,color);
            });
        }
        else
        {
            tipl::par_for(regions[roi_index]->size(),[&](unsigned int index)
            {
                int X, Y, Z;
                if(r != 1.0f)
                {
                    tipl::vector<3,float> p(regions[roi_index]->region[index]);
                    p /= r;
                    p.round();
                    tipl::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
                }
                else
                {
                    auto p = regions[roi_index]->region[index];
                    tipl::space2slice(cur_tracking_window.cur_dim,p[0],p[1],p[2],X,Y,Z);
                }
                if (slice_pos != Z || X < 0 || Y < 0 ||
                    X >= w || Y >= h)
                    return;
                draw_voxel(uint32_t(yw[uint32_t(Y)]+X),color);
            });
        }
        draw_edge(roi_index);
    }
}

void RegionTableWidget::draw_edge(int slice_width,int slice_height,QImage& scaled_image,bool draw_all)
{
    // during region removal, there will be a call with invalid currentRow
    if(regions.empty() || currentRow() >= int(regions.size()) || currentRow() == -1)
        return;
    std::vector<std::shared_ptr<ROIRegion> > checked_regions;
    int cur_roi_index = -1;
    if(draw_all)
    {
        checked_regions = get_checked_regions();
        if(currentRow() >= 0)
        for(unsigned int i = 0;i < checked_regions.size();++i)
            if(checked_regions[i] == regions[uint32_t(currentRow())])
            {
                cur_roi_index = int(i);
                break;
            }
    }
    else
    if(currentRow() >= 0)
    {
        if(item(currentRow(),0)->checkState() != Qt::Checked)
            return;
        checked_regions.push_back(regions[uint32_t(currentRow())]);
        cur_roi_index = 0;
    }
    if(checked_regions.empty())
        return;
    float display_ratio = cur_tracking_window.get_scene_zoom();

    tipl::image<unsigned char,2> cur_image_mask;
    cur_image_mask.resize(tipl::geometry<2>(uint32_t(slice_width),
                                            uint32_t(slice_height)));

    draw_region_voxels(cur_tracking_window,
                       slice_width,slice_height,
                        checked_regions,
    // draw_voxel
    [&cur_image_mask](uint32_t pos,uint32_t)
    {
        cur_image_mask[pos] = 1;
    },
    // draw_edge
    [&](uint32_t roi_index)
    {
        unsigned int cur_color = 0xFFFFFFFF;
        if(draw_all)
            cur_color = checked_regions[roi_index]->show_region.color;

        QPainter paint(&scaled_image);
        paint.setBrush(Qt::NoBrush);
        QPen pen(QColor(cur_color),
                 double(int(display_ratio)+1)*(cur_roi_index == int(roi_index) ? 0.25 : 0.2), Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
        paint.setPen(pen);
        for(int y = 1,cur_index = slice_width;y < slice_height-1;++y)
        for(int x = 0;x < slice_width;++x,++cur_index)
        {
            if(x == 0 || x+1 >= slice_width || !cur_image_mask[cur_index])
                continue;
            int xd = int(x*display_ratio);
            int xd_1 = int(xd+display_ratio);
            int yd = int(y*display_ratio);
            int yd_1 = int(yd+display_ratio);
            if(!(cur_image_mask[cur_index-slice_width]))
                paint.drawLine(xd,yd,xd_1,yd);
            if(!(cur_image_mask[cur_index+slice_width]))
                paint.drawLine(xd,yd_1,xd_1,yd_1);
            if(!(cur_image_mask[cur_index-1]))
                paint.drawLine(xd,yd,xd,yd_1);
            if(!(cur_image_mask[cur_index+1]))
                paint.drawLine(xd_1,yd,xd_1,yd_1);
        }
        std::fill(cur_image_mask.begin(),cur_image_mask.end(),0);
    }
    );


}
void RegionTableWidget::draw_region(tipl::color_image& I)
{
    auto current_slice = cur_tracking_window.current_slice;
    auto checked_regions = get_checked_regions();
    if(checked_regions.empty())
        return;

    draw_region_voxels(cur_tracking_window,
                       I.width(),I.height(),
                        checked_regions,
    // draw_voxel
    [&I](uint32_t pos,uint32_t color)
    {
        I[pos] = color;
    },
    // draw_edge
    [](uint32_t)
    {
    }
    );
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
                          std::map<int,tipl::rgb>& label_color,bool is_free_surfer,bool verbose)
{
    label_map.clear();
    label_color.clear();
    QString base_name = QFileInfo(file_name).baseName();
    if(verbose)
        std::cout << "looking for region label file" << std::endl;
    if(base_name.contains("aparc+aseg") || is_free_surfer) // FreeSurfer
    {
        if(verbose)
            std::cout << "using freesurfer labels." << std::endl;
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
        if(verbose)
            std::cout << "label file loaded:" << label_file.toStdString() << std::endl;
        return;
    }
    if(verbose)
        std::cout << "no label file found. Use default ROI numbering." << std::endl;
}
bool is_label_image(const tipl::image<float,3>& I);
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<std::pair<tipl::geometry<3>,tipl::matrix<4,4,float> > >& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names,
              bool verbose)
{
    gz_nifti header;
    if (!header.load_from_file(file_name.c_str()))
        return false;

    tipl::image<unsigned int, 3> from;
    {
        tipl::image<float, 3> tmp;
        header.toLPS(tmp);
        if(header.is_integer() || is_label_image(tmp))
            from = tmp;
        else
        {
            from.resize(tmp.geometry());
            for(size_t i = 0;i < from.size();++i)
                from[i] = (tmp[i] == 0.0f ? 0:1);
        }
    }

    if(verbose)
    {
        std::cout << "DWI dimension=" << handle->dim << std::endl;
        std::cout << "NIFTI dimension=" << from.geometry() << std::endl;
    }
    std::vector<unsigned short> value_list;
    std::vector<unsigned short> value_map(std::numeric_limits<unsigned short>::max()+1);

    {
        unsigned short max_value = 0;
        for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
        {
            if(from[index.index()] >= value_map.size())
                return false;
            value_map[from[index.index()]] = 1;
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

    if(verbose)
        std::cout << (multiple_roi ? "nifti loaded as multiple ROI file":"nifti loaded as single ROI file") << std::endl;

    std::map<int,std::string> label_map;
    std::map<int,tipl::rgb> label_color;

    std::string des(header.get_descrip());
    if(multiple_roi)
        get_roi_label(file_name.c_str(),label_map,label_color,des.find("FreeSurfer") == 0,verbose);

    tipl::matrix<4,4,float> convert;
    bool has_transform = false;
    bool scale_image = false;


    if(from.geometry() != handle->dim)
    {
        if(verbose)
            std::cout << "NIFTI file has a different dimension from DWI." << std::endl;

        if(handle->is_qsdr)
        {
            tipl::image<unsigned int, 3> new_from;
            if(!handle->native_position.empty())
            for(unsigned int index = 0;index < handle->view_item.size();++index)
                if(handle->view_item[index].native_geo == from.geometry())
                {
                    auto T = handle->view_item[index].native_trans;
                    new_from.resize(handle->dim);
                    for(size_t i = 0;i < new_from.size();++i)
                    {
                        auto pos = handle->native_position[i];
                        T(pos);
                        tipl::estimate(from,pos,new_from[i],tipl::nearest);
                    }
                    break;
                }
            if(!new_from.empty())
            {
                if(verbose)
                    std::cout << "applying QSDR warpping to the native space NIFTI file" << std::endl;
                new_from.swap(from);
                goto end;
            }
        }

        if(verbose)
            std::cout << "looking for a matched dimension from loaded t1wt2w." << std::endl;
        for(unsigned int index = 0;index < transform_lookup.size();++index)
            if(from.geometry() == transform_lookup[index].first)
            {
                if(verbose)
                    std::cout << "find a matched dimension from loaded t1wt2w." << std::endl;
                convert = transform_lookup[index].second;
                has_transform = true;
                goto end;
            }

        if(handle->is_qsdr)
        {
            if(verbose)
                std::cout << "loaded NIFTI file used as MNI space image." << std::endl;
            header.get_image_transformation(convert);
            convert.inv();
            convert *= handle->trans_to_mni;
            has_transform = true;
            goto end;
        }

        float r1 = float(from.width())/float(handle->dim[0]);
        float r2 = float(from.height())/float(handle->dim[1]);
        float r3 = float(from.depth())/float(handle->dim[2]);
        if(std::fabs(r1-r2) < 0.02f && std::fabs(r1-r3) < 0.02f)
        {
            has_transform = false;
            scale_image = true;
        }
        else
        {
            if(verbose)
            {
                std::cout << "ERROR: The ROI file has a different dimension from DWI, and there is no registration strategy." << std::endl;
                std::cout << "ERROR: use --t1t2 to load the original T1W/T2W for guiding the registration." << std::endl;
            }
            return false;
        }
    }
    end:
    // single region ROI
    if(!multiple_roi)
    {
        regions.push_back(std::make_shared<ROIRegion>(handle.get()));
        names.push_back(QFileInfo(file_name.c_str()).baseName().toStdString());
        if(has_transform)
            regions[0]->LoadFromBuffer(from,convert);
        else
            regions[0]->LoadFromBuffer(from);

        unsigned int color = 0;
        unsigned int type = roi_id;

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
        regions.back()->show_region.color = color;
        regions.back()->regions_feature = uint8_t(type);
        return true;
    }

    std::vector<std::vector<tipl::vector<3,short> > > region_points(value_list.size());
    if(from.geometry() == handle->dim)
    {
        for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
            if(from[index.index()])
                region_points[value_map[from[index.index()]]].push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
    }
    else
    {
        if(has_transform)
        {
            tipl::geometry<3> geo = handle->dim;
            for (tipl::pixel_index<3>index(geo);index < geo.size();++index)
            {
                tipl::vector<3> p(index.begin()); // point in subject space
                p.to(convert); // point in "from" space
                p.round();
                if (from.geometry().is_valid(p))
                {
                    unsigned int value = from.at(uint32_t(p[0]),uint32_t(p[1]),uint32_t(p[2]));
                    if(value)
                        region_points[value_map[value]].push_back(tipl::vector<3,short>(index.x(),index.y(),index.z()));
                }
            }
        }
        else
        if(scale_image)
        {
            float r = float(handle->dim[0])/float(from.width());
            if(r <= 1.0f)
            {
                for (tipl::pixel_index<3>index(from.geometry());index < from.size();++index)
                if(from[index.index()])
                    region_points[value_map[from[index.index()]]].
                            push_back(tipl::vector<3,short>(r*index.x(), r*index.y(),r*index.z()));
            }
        }
        else
            return false;
    }

    for(uint32_t i = 0;i < region_points.size();++i)
        {
            unsigned short value = value_list[i];
            QString name = (label_map.find(value) == label_map.end() ?
                 QFileInfo(file_name.c_str()).baseName() + "_" + QString::number(value):QString(label_map[value].c_str()));
            regions.push_back(std::make_shared<ROIRegion>(handle.get()));
            names.push_back(name.toStdString());
            regions.back()->show_region.color = label_color.empty() ? 0x00FFFFFF : label_color[value].color;
            if(!region_points[i].empty())
                regions.back()->add_points(region_points[i],false,1.0f);
        }
    if(verbose)
        std::cout << "a total of " << regions.size() << " regions are loaded." << std::endl;
    return !regions.empty();
}

bool RegionTableWidget::load_multiple_roi_nii(QString file_name)
{

    std::vector<std::pair<tipl::geometry<3>,tipl::matrix<4,4,float> > > transform_lookup;
    // searching for T1/T2 mappings
    for(unsigned int index = 0;index < cur_tracking_window.slices.size();++index)
    {
        CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.slices[index].get());
        if(slice)
            transform_lookup.push_back(std::make_pair(slice->geometry,slice->invT));
    }
    std::vector<std::shared_ptr<ROIRegion> > loaded_regions;
    std::vector<std::string> names;
    if(!load_nii(cur_tracking_window.handle,
                 file_name.toStdString(),
                 transform_lookup,
                 loaded_regions,
                 names,false))
        return false;
    begin_prog("loading ROIs");
    begin_update();
    for(uint32_t i = 0;check_prog(i,loaded_regions.size());++i)
        {
            add_region(names[i].c_str(),roi_id,loaded_regions[i]->show_region.color);
            unsigned int color = regions.back()->show_region.color;
            regions.back()->swap(*loaded_regions[i].get());
            regions.back()->show_region.color = color;
            item(currentRow(),0)->setCheckState(loaded_regions.size() == 1 ? Qt::Checked : Qt::Unchecked);
            item(currentRow(),0)->setData(Qt::ForegroundRole,
                loaded_regions.size() == 1 ? QBrush(Qt::black) : QBrush(Qt::gray));
        }
    end_update();
    return true;
}

void RegionTableWidget::load_region_color(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load region color",QFileInfo(cur_tracking_window.windowTitle()).absolutePath()+"/region_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::ifstream in(filename.toStdString().c_str());
    if (!in)
        return;
    std::vector<int> colors((std::istream_iterator<float>(in)),
                              (std::istream_iterator<float>()));
    for(size_t index = 0,pos = 0;index < regions.size() && pos+2 < colors.size();++index,pos+=3)
    {
        tipl::rgb c(std::min<int>(colors[pos],255),
                    std::min<int>(colors[pos+1],255),
                    std::min<int>(colors[pos+2],255));
        regions[index]->show_region.color = c;
        regions[index]->modified = true;
        item(int(index),2)->setData(Qt::UserRole,0xFF000000 | uint32_t(c));
    }
    emit need_update();
}
void RegionTableWidget::save_region_color(void)
{
    QString filename = QFileDialog::getSaveFileName(
                this,"Save region color",QFileInfo(cur_tracking_window.windowTitle()).absolutePath()+"/region_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toStdString().c_str());
    if (!out)
        return;
    for(size_t index = 0;index < regions.size();++index)
    {
        tipl::rgb c(regions[index]->show_region.color);
        out << int(c[2]) << " " << int(c[1]) << " " << int(c[0]) << std::endl;
    }
    QMessageBox::information(this,"DSI Studio","File saved");
}

void RegionTableWidget::load_region(void)
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,"Open region",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),"Region files (*.nii *.hdr *nii.gz *.mat);;Text files (*.txt);;All files (*)" );
    if (filenames.isEmpty())
        return;

    for (int index = 0;index < filenames.size();++index)
        command("load_region",filenames[index]);
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
        QMessageBox::information(this,"Error",cur_tracking_window.handle->error_msg.c_str(),0);
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
    for(int row = 0;row < rowCount();++row)
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
    for(int row = 0;row < rowCount();++row)
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
    if (regions.empty() || currentRow() >= regions.size())
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
    switch(cur_tracking_window["roi_format"].toInt())
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
    command("save_all_regions_to_dir",dir);
}
void RegionTableWidget::save_all_regions(void)
{
    if (regions.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save region",item(currentRow(),0)->text() + output_format(),
                           "Region file(*nii.gz *.nii *.mat);;Text file (*.txt);;All file types (*)" );
    if (filename.isEmpty())
        return;
    QString base_name = QFileInfo(filename).completeBaseName();
    if(QFileInfo(base_name).suffix().toLower() == "nii")
        base_name = QFileInfo(base_name).completeBaseName();
    QString label_file = QFileInfo(filename).absolutePath()+"/"+base_name+".txt";
    std::ofstream out(label_file.toLocal8Bit().begin());
    tipl::geometry<3> geo = cur_tracking_window.handle->dim;
    tipl::image<unsigned short, 3> mask(geo);
    for (unsigned int i = 0; i < regions.size(); ++i)
        if (item(int(i),0)->checkState() == Qt::Checked)
        {
            for (unsigned int j = 0; j < regions[i]->size(); ++j)
            {
                tipl::vector<3,short> p = regions[i]->get_region_voxel(j);
                if (geo.is_valid(p))
                    mask[tipl::pixel_index<3>(p[0],p[1],p[2], geo).index()] = uint16_t(i+1);

            }
            out << i+1 << " " << item(int(i),0)->text().toStdString() << std::endl;
        }
    if(regions.size() <= 255)
    {
        tipl::image<uint8_t, 3> i8mask(mask);
        gz_nifti::save_to_file(filename.toStdString().c_str(),i8mask,
                           cur_tracking_window.current_slice->voxel_size,
                           cur_tracking_window.handle->trans_to_mni);
    }
    else
    {
        gz_nifti::save_to_file(filename.toStdString().c_str(),mask,
                           cur_tracking_window.current_slice->voxel_size,
                           cur_tracking_window.handle->trans_to_mni);
    }
}

void RegionTableWidget::save_region_info(void)
{
    if (regions.empty())
        return;
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


void get_regions_statistics(const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            const std::vector<std::string>& region_name,
                            std::string& result)
{
    std::vector<std::string> titles;
    std::vector<std::vector<float> > data(regions.size());
    tipl::par_for(regions.size(),[&](unsigned int index){
        std::vector<std::string> dummy;
        regions[index]->get_quantitative_data((index == 0) ? titles : dummy,data[index]);
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
        get_regions_statistics(active_regions,region_name,result);
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
            data->roi_mgr->setRegions(regions[index]->get_region_voxels_raw(),
                                     regions[index]->resolution_ratio,
                             regions[index]->regions_feature,item(int(index),0)->text().toLocal8Bit().begin());
    // auto track
    if(cur_tracking_window.ui->target->currentIndex() > 0 &&
       cur_tracking_window.handle->track_atlas.get())
        data->roi_mgr->setAtlas(uint32_t(cur_tracking_window.ui->target->currentIndex()-1),
                                cur_tracking_window["autotrack_tolerance"].toFloat()/cur_tracking_window.handle->vs[0]);

    if(data->roi_mgr->seeds.empty())
        data->roi_mgr->setWholeBrainSeed(cur_tracking_window.get_fa_threshold());
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
        if(cur_tracking_window.ui->actionModify_All->isChecked())
            for_each_checked_region([&](std::shared_ptr<ROIRegion> region){region->perform(action.toStdString());});
        else
            cur_region.perform(action.toStdString());


        if(action == "A-B" || action == "B-A" || action == "A*B")
        {
            auto checked_regions = get_checked_regions();
            if(checked_regions.size() < 2)
                return;
            if(action == "A-B")
            {
                tipl::image<unsigned char, 3> A,B;
                checked_regions[0]->SaveToBuffer(A);
                for(size_t r = 1;r < checked_regions.size();++r)
                {
                    checked_regions[r]->SaveToBuffer(B,checked_regions[0]->resolution_ratio);
                    for(size_t i = 0;i < A.size();++i)
                        if(B[i])
                            A[i] = 0;
                }
                checked_regions[0]->LoadFromBuffer(A);
            }
            else
            {
                tipl::image<unsigned char, 3> A,B;
                for(size_t r = 1;r < checked_regions.size();++r)
                {
                    checked_regions[r]->SaveToBuffer(B);
                    if(B.geometry() != A.geometry())
                        checked_regions[0]->SaveToBuffer(A,checked_regions[r]->resolution_ratio);

                    if(action == "B-A")
                    {
                        for(size_t i = 0;i < B.size();++i)
                            if(A[i])
                                B[i] = 0;
                        checked_regions[r]->LoadFromBuffer(B);
                    }
                    if(action == "A*B")
                    {
                        for(size_t i = 0;i < B.size();++i)
                            B[i] = (A[i] & B[i]);
                        checked_regions[r]->LoadFromBuffer(B);
                    }
                }
            }

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
                mask[i]  = ((I[i] > threshold) ^ flip) ? 1:0;

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
                            ((I.at(pos[0],pos[1],pos[2]) > threshold) ^ flip))
                                   new_region.push_back(region[i]);
                    }
                }
                else
                for(size_t i = 0;i < region.size();++i)
                {
                    if(I.geometry().is_valid(region[i][0],region[i][1],region[i][2]) &&
                        ((I.at(region[i][0],region[i][1],region[i][2]) > threshold) ^ flip))
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
                        ((I.at(pos[0],pos[1],pos[2]) > threshold) ^ flip))
                               new_region.push_back(region[i]);
                }
            }
            cur_region.assign(new_region,cur_region.resolution_ratio);
        }
        if(action == "separate")
        {
            tipl::image<unsigned char, 3>mask;
            cur_region.SaveToBuffer(mask);
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
                    ROIRegion region(cur_tracking_window.handle.get());
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
                    regions[index]->get_quantitative_data(dummy,data[index]);
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
                new_region_checked[i] = item(int(arg[i]),0)->checkState() == Qt::Checked ? 1:0;
                new_region_names[i] = item(int(arg[i]),0)->text().toStdString();
            }
            regions.swap(new_region);
            for(int i = 0;i < int(arg.size());++i)
            {
                item(i,0)->setCheckState(new_region_checked[uint32_t(i)] ? Qt::Checked : Qt::Unchecked);
                item(i,0)->setText(new_region_names[uint32_t(i)].c_str());
                closePersistentEditor(item(i,1));
                closePersistentEditor(item(i,2));
                item(i,1)->setData(Qt::DisplayRole,regions[uint32_t(i)]->regions_feature);
                item(i,2)->setData(Qt::UserRole,regions[uint32_t(i)]->show_region.color.color);
                openPersistentEditor(item(i,1));
                openPersistentEditor(item(i,2));
            }
        }
    }
    emit need_update();
}
