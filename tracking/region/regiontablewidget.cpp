#include <QFileDialog>
#include <QContextMenuEvent>
#include <QMessageBox>
#include "regiontablewidget.h"
#include "tracking/tracking_window.h"
#include "tracking_static_link.h"
#include "qcolorcombobox.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "mapping/fa_template.hpp"
extern std::vector<atlas> atlas_list;
extern fa_template fa_template_imp;

QColor ROIColor[15] =
{
    Qt::red, Qt::green, Qt::blue, Qt::yellow, Qt::magenta, Qt::cyan,  Qt::gray,
    Qt::darkRed,Qt::darkGreen, Qt::darkBlue, Qt::darkYellow,  Qt::darkMagenta, Qt::darkCyan,
    Qt::darkGray, Qt::lightGray
};

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
                QColor(color.r,color.g,color.b,color.a));
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
        QTableWidget(parent),cur_tracking_window(cur_tracking_window_),regions_index(0)
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
    setItemDelegate(new ImageDelegate(this));

    QObject::connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_check_status(int,int)));
    QObject::connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateRegions(QTableWidgetItem*)));

    // for updating region
    timer = new QTimer(this);
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_update()));
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
        regions[item->row()].regions_feature = item->text().toInt();
    else
        if (item->column() == 2)
        {
            regions[item->row()].show_region.color = item->data(Qt::UserRole).toInt();
            emit need_update();
        }
}

QColor RegionTableWidget::currentRowColor(void)
{
    return (unsigned int)regions[currentRow()].show_region.color;
}

void RegionTableWidget::add_region(QString name,unsigned char feature)
{
    regions.push_back(new ROIRegion(cur_tracking_window.slice.geometry,cur_tracking_window.slice.voxel_size));

    setRowCount(regions.size());

    // the name item
    QTableWidgetItem *item0 = new QTableWidgetItem(name);

    setItem(regions.size()-1, 0, item0);

    QTableWidgetItem *item1 = new QTableWidgetItem(QString::number((int)feature));
    QTableWidgetItem *item2 = new QTableWidgetItem();

    setItem(regions.size()-1, 1, item1);
    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    setItem(regions.size()-1, 2, item2);
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::UserRole,(int)ROIColor[regions_index].rgb());
    setRowHeight(regions.size()-1,22);


    openPersistentEditor(item1);
    openPersistentEditor(item2);
    item0->setCheckState(Qt::Checked);

    setCurrentCell(regions.size()-1,0);
    if (feature != none_roi_id)
    {
        ++regions_index;
        if (regions_index >= 15)
            regions_index = 0;
    }
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

void RegionTableWidget::draw_region(QImage& qimage)
{
    int X, Y, Z;
    for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
    {
        if (item(roi_index,0)->checkState() != Qt::Checked)
            continue;
        unsigned int cur_color = regions[roi_index].show_region.color;
        for (unsigned int index = 0;index < regions[roi_index].size();++index)
        {
            regions[roi_index].getSlicePosition(&cur_tracking_window.slice, index, X, Y, Z);
            if (cur_tracking_window.slice.slice_pos[cur_tracking_window.slice.cur_dim] != Z ||
                    X < 0 || Y < 0 || X >= qimage.width() || Y >= qimage.height())
                continue;
            qimage.setPixel(X,Y,(unsigned int)qimage.pixel(X,Y) | cur_color);
        }
    }
}
void RegionTableWidget::draw_mosaic_region(QImage& qimage,unsigned int mosaic_size,unsigned int skip)
{
    int X, Y, Z;
    image::geometry<3> geo = cur_tracking_window.slice.geometry;
    unsigned int slice_number = geo[2] >> skip;
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
        unsigned int cur_color = regions[roi_index].show_region.color;
        for (unsigned int index = 0;index < regions[roi_index].size();++index)
        {
            regions[roi_index].getSlicePosition(&cur_tracking_window.slice, index, X, Y, Z);
            if(Z != ((Z >> skip) << skip))
                continue;
            X += shift_x[Z >> skip];
            Y += shift_y[Z >> skip];
            if(X >= qimage.width() || Y >= qimage.height())
                continue;
            qimage.setPixel(X,Y,(unsigned int)qimage.pixel(X,Y) | cur_color);
        }
    }
}

void RegionTableWidget::new_region(void)
{
    add_region("New Region",roi_id);
}

void RegionTableWidget::load_region(void)
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,
                                "Open region",
                                cur_tracking_window.absolute_path,
                                "Region files (*.txt *.nii *.hdr *.nii.gz *.mat);;All files (*.*)" );
    if (filenames.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filenames[0]).absolutePath();

    for (unsigned int index = 0;index < filenames.size();++index)
    {
        ROIRegion region(cur_tracking_window.slice.geometry,cur_tracking_window.slice.voxel_size);
        std::vector<float> trans;
        cur_tracking_window.get_dicom_trans(trans);
        if(!region.LoadFromFile(filenames[index].toLocal8Bit().begin(),trans))
        {
            QMessageBox::information(this,"error","Inconsistent geometry",0);
            return;
        }
        add_region(QFileInfo(filenames[index]).baseName(),roi_id);
        regions.back().assign(region.get());
    }
    emit need_update();
    timer->start();
}

void RegionTableWidget::save_region(void)
{
    if (currentRow() >= regions.size())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save region",
                           cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + ".txt",
                           "Text files (*.txt);;Nifti file(*.nii;*.nii.gz);;Maylab file (*.mat)" );
    if (filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();

    std::vector<float> trans;
    cur_tracking_window.get_nifti_trans(trans);
    regions[currentRow()].SaveToFile(filename.toLocal8Bit().begin(),trans);
    item(currentRow(),0)->setText(QFileInfo(filename).baseName());
}
void RegionTableWidget::save_region_info(void)
{
    if (currentRow() >= regions.size())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save voxel information",
                           cur_tracking_window.absolute_path + "/" + item(currentRow(),0)->text() + "_info.txt",
                           "Text files (*.txt)" );
    if (filename.isEmpty())
        return;
    cur_tracking_window.absolute_path = QFileInfo(filename).absolutePath();

    std::ofstream out(filename.toLocal8Bit().begin());
    out << "x\ty\tz";
    for(unsigned int index = 0;index < cur_tracking_window.handle->fib_data.fib.findex.size();++index)
            out << "\tdx" << index << "\tdy" << index << "\tdz" << index;

    for(unsigned int index = 0;index < cur_tracking_window.view_name.size();++index)
        if(cur_tracking_window.view_name[index] != "color")
            out << "\t" << cur_tracking_window.view_name[index];

    out << std::endl;
    for(int index = 0;index < regions[currentRow()].get().size();++index)
    {
        std::vector<float> data;
        image::vector<3,short> point(regions[currentRow()].get()[index]);
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
    this->removeRow(currentRow());
    emit need_update();
}

void RegionTableWidget::delete_all_region(void)
{
    setRowCount(0);
    regions.clear();
    regions_index = 0;
    emit need_update();
}

void RegionTableWidget::whole_brain_points(std::vector<image::vector<3,short> >& points)
{
    float fa[3];
    float dir[9];
    image::geometry<3> geo = cur_tracking_window.slice.geometry;
    float threshold = cur_tracking_window.ui->fa_threshold->value();
    for (image::pixel_index<3>index; index.valid(geo);index.next(geo))
    {
        image::vector<3,short> pos(index);
        if (!tracking_get_voxel_dir(cur_tracking_window.handle,pos[0],pos[1],pos[2],fa, dir))
            continue;
        if (fa[0] > threshold)
            points.push_back(pos);
    }
}

void RegionTableWidget::whole_brain(void)
{
    std::vector<image::vector<3,short> > points;
    whole_brain_points(points);
    add_region("whole brain",seed_id);
    add_points(points,false);
    emit need_update();
    timer->start();
}
extern std::vector<float> mni_fa0_template_tran;
extern image::basic_image<float,3> mni_fa0_template;

void RegionTableWidget::add_atlas(void)
{
    int atlas_index = cur_tracking_window.ui->atlasListBox->currentIndex();
    std::vector<image::vector<3,short> > points;
    unsigned short label = cur_tracking_window.ui->atlasComboBox->currentIndex();
    const float* m = cur_tracking_window.mi3->get();
    image::vector<3,float>mni_coordinate;
    image::geometry<3> geo = cur_tracking_window.slice.geometry;
    for (image::pixel_index<3>index; index.valid(geo);index.next(geo))
    {
        image::vector<3,float>cur_coordinate((const unsigned int*)(index.begin()));
        image::vector_transformation(cur_coordinate.begin(),
                                     mni_coordinate.begin(), m, m + 9, image::vdim<3>());
        fa_template_imp.to_mni(mni_coordinate);
        if (!atlas_list[atlas_index].is_labeled_as(mni_coordinate, label))
                continue;
        points.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
    }
    add_region(cur_tracking_window.ui->atlasComboBox->currentText(),seed_id);
    add_points(points,false);
    emit need_update();
    timer->start();
}

void RegionTableWidget::add_points(std::vector<image::vector<3,short> >& points,bool erase)
{
    if (currentRow() >= regions.size())
        return;
    regions[currentRow()].add_points(points,erase);
    item(currentRow(),0)->setCheckState(Qt::Checked);
    item(currentRow(),0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    timer->start();
}

bool RegionTableWidget::has_seeding(void)
{
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index].empty() &&
                item(index,0)->checkState() == Qt::Checked &&
                regions[index].regions_feature == seed_id) // either roi roa end or seed
            return true;
    return false;
}

void RegionTableWidget::setROIs(ThreadData* data)
{
    // check if there is seeds
    if(!has_seeding())
    {
        std::vector<image::vector<3,short> > points;
        whole_brain_points(points);
        data->setRegions(points,seed_id);
    }

    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index].empty() &&
                item(index,0)->checkState() == Qt::Checked &&
                regions[index].regions_feature <= 3) // either roi roa end or seed
            data->setRegions(regions[index].get(),
                             regions[index].regions_feature);
}

void RegionTableWidget::check_update(void)
{

    for(unsigned int index = 0;index < regions.size();++index)
        if(regions[index].has_background_thread())
        {
            emit need_update();
            return;
        }
    timer->stop();
}

void RegionTableWidget::do_action(int id)
{
    if (regions.empty())
        return;
    image::basic_image<unsigned char, 3>mask;
    ROIRegion& cur_region = regions[currentRow()];
    switch (id)
    {
    case 0: // Smoothing
        cur_region.SaveToBuffer(mask, 1);
        image::morphology::smoothing(mask);
        cur_region.LoadFromBuffer(mask);
        break;
    case 1: // Erosion
        cur_region.SaveToBuffer(mask, 1);
        image::morphology::erosion(mask);
        cur_region.LoadFromBuffer(mask);
        break;
    case 2: // Expansion
        cur_region.SaveToBuffer(mask, 1);
        image::morphology::dilation(mask);
        cur_region.LoadFromBuffer(mask);
        break;
    case 3: // Defragment
        cur_region.SaveToBuffer(mask, 1);
        image::morphology::defragment(mask);
        cur_region.LoadFromBuffer(mask);
        break;
    case 4: // Negate
        cur_region.SaveToBuffer(mask, 1);
        for (unsigned int index = 0; index < mask.size(); ++index)
            mask[index] = 1 - mask[index];
        cur_region.LoadFromBuffer(mask);
        break;
    case 5:
        cur_region.Flip(0);
        break;
    case 6:
        cur_region.Flip(1);
        break;
    case 7:
        cur_region.Flip(2);
        break;
    case 8: //
        {
            float threshold = cur_tracking_window.ui->fa_threshold->value();
            for(int index = 0;index < cur_region.size();)
            {
                image::vector<3,short> point(cur_region.get()[index]);
                if(threshold > ((const FibSliceModel&)cur_tracking_window.slice).source_images.at(point[0],point[1],point[2]))
                    cur_region.erase(index);
                else
                    ++index;
            }
        }
        break;
    case 9: // shift
        cur_region.shift(image::vector<3,short>(1, 0, 0));
        break;
    case 10: // shift
        cur_region.shift(image::vector<3,short>(-1, 0, 0));
        break;
    case 11: // shift
        cur_region.shift(image::vector<3,short>(0, 1, 0));
        break;
    case 12: // shift
        cur_region.shift(image::vector<3,short>(0, -1, 0));
        break;
    case 13: // shift
        cur_region.shift(image::vector<3,short>(0, 0, 1));
        break;
    case 14: // shift
        cur_region.shift(image::vector<3,short>(0, 0, -1));
        break;
    }
    emit need_update();
    timer->start();
}
