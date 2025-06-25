#include <regex>
#include <cmath>
#include <QFileDialog>
#include <QInputDialog>
#include <QContextMenuEvent>
#include <QMessageBox>
#include <QClipboard>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QHeaderView>
#include "regiontablewidget.h"
#include "tracking/tracking_window.h"
#include "tracking/devicetablewidget.h"
#include "qcolorcombobox.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
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
        comboBox->addItem("Limiting");
        comboBox->addItem("...");
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
    setColumnCount(4);
    setColumnWidth(0,140);
    setColumnWidth(1,60);
    setColumnWidth(2,40);
    setColumnWidth(3,200);
    horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);

    QStringList header;
    header << "Name" << "Type" << "Color" << "Dimension × Resolution (mm)";
    setHorizontalHeaderLabels(header);
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setAlternatingRowColors(true);

    setItemDelegate(new ImageDelegate(this));

    connect(this, &QTableWidget::itemChanged, this, [=](QTableWidgetItem* item)
    {
        if (item->column() == 0)
        {
            regions[uint32_t(item->row())]->name = item->text().toStdString();
            auto current = item->checkState();
            if (current != static_cast<Qt::CheckState>(item->data(Qt::UserRole+1).toInt()))
            {
                item->setData(Qt::UserRole+1, current);
                emit need_update();
            }
        }
        if (item->column() == 1)
            regions[uint32_t(item->row())]->regions_feature = uint8_t(item->text().toInt());
        if (item->column() == 2)
        {
            regions[uint32_t(item->row())]->region_render->color = uint32_t(item->data(Qt::UserRole).toInt());
            emit need_update();
        }
    });
    setEditTriggers(QAbstractItemView::DoubleClicked|QAbstractItemView::EditKeyPressed);


    update_color_map();
}


void RegionTableWidget::contextMenuEvent ( QContextMenuEvent * event )
{
    if (event->reason() == QContextMenuEvent::Mouse)
    {
        cur_tracking_window.ui->menuRegions->popup(event->globalPos());
    }
}

QColor RegionTableWidget::currentRowColor(void)
{
    return uint32_t(regions[uint32_t(currentRow())]->region_render->color);
}

void RegionTableWidget::add_merged_regions_from_atlas(std::shared_ptr<atlas> at,QString name,const std::vector<unsigned int>& roi_list)
{
    add_region(name);
    std::vector<tipl::vector<3,short> > all_points;
    for(auto label : roi_list)
    {
        std::vector<tipl::vector<3,short> > points;
        cur_tracking_window.handle->get_atlas_roi(at,label,regions.back()->dim,regions.back()->to_diffusion_space,points);
        all_points.insert(all_points.end(),points.begin(),points.end());
    }
    if(all_points.empty())
        return;
    regions.back()->add_points(std::move(all_points));
}
void RegionTableWidget::begin_update(void)
{
    cur_tracking_window.scene.no_update = true;
    cur_tracking_window.disconnect(cur_tracking_window.regionWidget,SIGNAL(need_update()),cur_tracking_window.glWidget,SLOT(update()));
    cur_tracking_window.disconnect(cur_tracking_window.regionWidget,SIGNAL(cellChanged(int,int)),cur_tracking_window.glWidget,SLOT(update()));
}

void RegionTableWidget::end_update(void)
{
    cur_tracking_window.scene.no_update = false;
    cur_tracking_window.connect(cur_tracking_window.regionWidget,SIGNAL(need_update()),cur_tracking_window.glWidget,SLOT(update()));
    cur_tracking_window.connect(cur_tracking_window.regionWidget,SIGNAL(cellChanged(int,int)),cur_tracking_window.glWidget,SLOT(update()));
}

void RegionTableWidget::add_row(int row,QString name)
{
    {
        uint32_t color = uint32_t(regions[row]->region_render->color);
        if(color == 0xFFFFFFFF || color == 0x00FFFFFF || !color)
            regions[row]->region_render->color = tipl::rgb::generate_hue(color_gen++);
    }
    if(regions[row]->region_render->color.a == 0)
        regions[row]->region_render->color.a = 255;

    auto handle = cur_tracking_window.handle;
    insertRow(row);
    QTableWidgetItem *item0 = new QTableWidgetItem(name);
    QTableWidgetItem *item1 = new QTableWidgetItem(QString::number(int(regions[row]->regions_feature)));
    QTableWidgetItem *item2 = new QTableWidgetItem();
    QTableWidgetItem *item3 = new QTableWidgetItem(QString("(%1,%2,%3)x(%4,%5,%6)")
                                                    .arg(regions[row]->dim[0])
                                                    .arg(regions[row]->dim[1])
                                                    .arg(regions[row]->dim[2])
                                                    .arg(regions[row]->vs[0])
                                                    .arg(regions[row]->vs[1])
                                                    .arg(regions[row]->vs[2]));
    item2->setData(Qt::UserRole,uint32_t(regions[row]->region_render->color));


    setItem(row, 0, item0);
    setItem(row, 1, item1);
    setItem(row, 2, item2);
    setItem(row, 3, item3);


    openPersistentEditor(item1);
    openPersistentEditor(item2);

    setRowHeight(row,22);
    setCurrentCell(row,0);

    if(cur_tracking_window.ui->tract_target_0->count())
        cur_tracking_window.ui->tract_target_0->setCurrentIndex(0);

    check_row(row,true);

}
void RegionTableWidget::add_high_reso_region(QString name,float reso,unsigned char feature,unsigned int color)
{
    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle));
    regions.back()->region_render->color = color;
    regions.back()->regions_feature = feature;
    regions.back()->vs /= reso;
    regions.back()->dim = tipl::shape<3>(regions.back()->dim[0]*reso,regions.back()->dim[1]*reso,regions.back()->dim[2]*reso);
    regions.back()->trans_to_mni[0] /= reso;
    regions.back()->trans_to_mni[5] /= reso;
    regions.back()->trans_to_mni[10] /= reso;
    regions.back()->is_diffusion_space = false;
    regions.back()->to_diffusion_space[0] /= reso;
    regions.back()->to_diffusion_space[5] /= reso;
    regions.back()->to_diffusion_space[10] /= reso;

    add_row(int(regions.size()-1),name);
}
void RegionTableWidget::add_region(QString name,unsigned char feature,unsigned int color)
{
    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle));
    regions.back()->region_render->color = color;
    regions.back()->regions_feature = feature;
    regions.back()->dim = cur_tracking_window.current_slice->dim;
    regions.back()->vs = cur_tracking_window.current_slice->vs;
    regions.back()->trans_to_mni = cur_tracking_window.current_slice->trans_to_mni;
    regions.back()->is_diffusion_space = cur_tracking_window.current_slice->is_diffusion_space;
    regions.back()->to_diffusion_space = cur_tracking_window.current_slice->to_dif;
    add_row(int(regions.size()-1),name);
}

void RegionTableWidget::update_color_map(void)
{
    if(cur_tracking_window["region_color_map"].toInt()) // color map from file
    {
        QString filename = QCoreApplication::applicationDirPath()+"/color_map/"+
                cur_tracking_window.renderWidget->getListValue("region_color_map")+".txt";
        color_map_rgb.load_from_file(filename.toStdString().c_str());
        cur_tracking_window.glWidget->region_color_bar.load_from_file(filename.toStdString().c_str());
    }
    else
    {
        tipl::rgb from_color(uint32_t(cur_tracking_window["region_color_min"].toUInt()));
        tipl::rgb to_color(uint32_t(cur_tracking_window["region_color_max"].toUInt()));
        cur_tracking_window.glWidget->region_color_bar.two_color(from_color,to_color);
        color_map_rgb.two_color(from_color,to_color);
    }
    cur_tracking_window.glWidget->region_color_bar_pos = {10,10};
}
tipl::rgb RegionTableWidget::get_region_rendering_color(size_t index)
{
    if(index >= regions.size())
        return tipl::rgb(0xFFFFFFFF);
    if(cur_tracking_window["region_color_style"].toInt() == 0) // assigned color
        return regions[index]->region_render->color;
    if(color_map_values.size() != regions.size())
        color_map_values.clear();
    color_map_values.resize(regions.size(),std::nanf(""));

    {
        auto metric_index = cur_tracking_window["region_color_metrics"].toInt();
        if(metric_index < cur_tracking_window.handle->slices.size() &&
           !cur_tracking_window.handle->slices[metric_index]->optional()) // sample slices values
        {
            if(std::isnan(color_map_values[index]))
            {
                float mean,max_v,min_v;
                regions[index]->get_quantitative_data(cur_tracking_window.handle->slices[metric_index],mean,max_v,min_v);
                color_map_values[index] = mean;
            }
        }
        else // compute tract-region interscept
        {
            int tract_index = cur_tracking_window.tractWidget->currentRow();
            if(tract_index >= 0 && tract_index < cur_tracking_window.tractWidget->tract_models.size())
            {
                auto tract = cur_tracking_window.tractWidget->tract_models[tract_index];
                if(tract->get_visible_track_count())
                {
                    size_t id = size_t(tract_index+1)*
                                ((tract->get_visible_track_count()+2) & 255)*
                                ((tract->get_tracts().back().size()+3) & 255);
                    if(id != tract_map_id)
                    {
                        tract_map_id = id;
                        Parcellation p(cur_tracking_window.handle);
                        p.load_from_regions(regions);
                        color_map_values = p.get_t2r_values(tract);
                    }
                }
            }
        }
    }
    if(std::isnan(color_map_values[index]))
        color_map_values[index] = 0;
    auto color_min = cur_tracking_window["region_color_min_value"].toFloat();
    auto color_r = cur_tracking_window["region_color_max_value"].toFloat()-color_min;
    auto c = color_map_rgb.value2color(color_map_values[index],color_min,color_r);
    c.a = 255;
    return c;
}
void get_regions_statistics(std::shared_ptr<fib_data> handle,const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            std::string& result);
void get_devices_statistics(std::shared_ptr<fib_data> handle,const std::vector<std::shared_ptr<Device> >& devices,
                            std::string& result);
void get_tract_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          std::string& result);
extern std::vector<std::vector<std::string> > atlas_file_name_list;
bool RegionTableWidget::command(std::vector<std::string> cmd)
{
    auto run = cur_tracking_window.history.record(error_msg,cmd);
    if(cmd.size() < 3)
        cmd.resize(3);

    auto get_cur_row = [&](std::string& cmd_text,int& cur_row)->bool
    {
        if (regions.empty())
        {
            error_msg = "no available region";
            return false;
        }
        bool okay = true;
        if(cmd_text.empty())
            cmd_text = std::to_string(cur_row);
        else
            cur_row = QString::fromStdString(cmd_text).toInt(&okay);
        if (cur_row >= regions.size() || !okay)
        {
            error_msg = "invalid region index: " + cmd_text;
            return false;
        }
        return run->succeed();
    };

    if(cmd[0] == "new_region")
    {
        add_region("new region");
        return run->succeed();
    }
    if(cmd[0] == "new_region_whole_brain_seed")
    {
        // cmd[1] : otsu ratio for threshold
        float otsu = run->from_cmd(1,cur_tracking_window["otsu_threshold"].toFloat());
        float threshold = otsu*cur_tracking_window.handle->dir.fa_otsu;
        auto cur_slice = cur_tracking_window.current_slice;
        tipl::image<3,unsigned char> mask(cur_slice->dim);
        auto fa_map = tipl::make_image(cur_tracking_window.handle->dir.fa[0],cur_tracking_window.handle->dim);
        if(cur_slice->is_diffusion_space)
            tipl::threshold(fa_map,mask,threshold);
        else
            tipl::adaptive_par_for(tipl::begin_index(mask.shape()),
                          tipl::end_index(mask.shape()),
                          [&](const tipl::pixel_index<3>& index)
            {
                tipl::vector<3> pos(index);
                pos.to(cur_slice->to_dif);
                if(tipl::estimate(fa_map,pos) > threshold)
                    mask[index.index()] = 1;
            });
        add_region("whole brain",seed_id);
        regions.back()->load_region_from_buffer(mask);
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "new_region_from_threshold")
    {
        // cmd[1] : threshold applied
        add_region("New Region");
        std::vector<std::string> cmd_proxy = {"region_action_threshold",
                                              std::to_string(regions.size()-1),
                                              cmd[1]};
        if(!do_action(cmd_proxy))
        {
            if(!error_msg.empty())
                return false;
            return run->canceled();
        }
        cmd[1] = cmd_proxy[2]; // record the threshold specified by users back to cmd[1]
        return run->succeed();
    }
    if(cmd[0] == "new_region_from_mni" || cmd[0] == "new_region_from_sphere")
    {
        if(cmd[1].empty() && (cmd[1] =
                QInputDialog::getText(this,QApplication::applicationName(),
                "Please specify the MNI Coordinate and radius of the region, separated by spaces (e.g. 0 -10 21 10)",
                                                    QLineEdit::Normal,"0 0 0 10").toStdString()).empty())
            return run->canceled();
        QStringList params = QString::fromStdString(cmd[1]).split(' ');
        if(params.size() == 3)
        {
            bool ok;
            int radius = QInputDialog::getInt(this,QApplication::applicationName(),"radius (voxels):",
                                                     5,1,100,1,&ok);
            if (!ok)
                return run->canceled();
            params.push_back(QString::number(radius));
            cmd[1] += " " + std::to_string(radius);
        }
        if(params.size() != 4)
            return run->failed("invalid numbers. please specify four numbers separated by spaces");

        add_region("New Region");
        tipl::vector<3> pos(params[0].toFloat(),params[1].toFloat(),params[2].toFloat());

        if(cmd[0] == "new_region_from_mni")
        {
            if(!cur_tracking_window.handle->map_to_mni())
                return run->failed("cannot map to MNI space: " + cur_tracking_window.handle->error_msg);
            cur_tracking_window.handle->mni2sub(pos);
            if(!regions.back()->is_diffusion_space)
            {
                auto T = regions.back()->to_diffusion_space;
                T.inv();
                pos.to(T);
            }
        }
        regions.back()->new_from_sphere(pos,params[3].toFloat());
        return run->succeed();
    }
    if(tipl::begins_with(cmd[0],"region_action_"))
    {
        // cmd[0] : action
        // cmd[1] : region index (default current row) separated by '&'
        // cmd[2] : additional parameters used by threshold and dilation by voxel
        if(cmd[1].empty())
        {
            if(cur_tracking_window.ui->actionModify_All->isChecked())
            {
                for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
                    if (item(roi_index,0)->checkState() == Qt::Checked)
                    {
                        if(!cmd[1].empty())
                            cmd[1] += '&';
                        cmd[1] += std::to_string(roi_index);
                    }
            }
            else
            {
                int cur_row = currentRow();
                if(!get_cur_row(cmd[1],cur_row))
                    return false;
            }
        }
        if(!do_action(cmd))
        {
            if(!error_msg.empty())
                return false;
            return run->canceled();
        }
        return run->succeed();
    }
    if(cmd[0] == "move_region")
    {
        // cmd[1] : target location in region space
        // cmd[2] : the region index (default: current selected one)
        if(cmd[1].empty())
            return run->failed("please specify location");
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        auto& cur_region = regions[cur_row];
        tipl::vector<3> pos;
        std::istringstream(cmd[1]) >> pos[0] >> pos[1] >> pos[2];

        if(cur_region->region.empty())
            return run->succeed();
        auto cm = std::accumulate(cur_region->region.begin(),cur_region->region.end(),tipl::vector<3>(0,0,0));
        cm /= cur_region->region.size();
        cur_region->shift(pos-cm);
        return run->succeed();
    }
    if(cmd[0] == "save_region")
    {
        // cmd[1] : file name to be saved
        // cmd[2] : the region index (default: current selected one)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],regions[cur_row]->name + output_format().toStdString()))
            return run->canceled();

        if(!tipl::ends_with(cmd[1],".mat") &&
           !tipl::ends_with(cmd[1],".txt") &&
           !tipl::ends_with(cmd[1],".nii") &&
           !tipl::ends_with(cmd[1],".nii.gz"))
            cmd[1] += ".nii.gz";
        if(!regions[cur_row]->save_region_to_file(cmd[1].c_str()))
            return run->failed("cannot save region to "+cmd[1]);
        return run->succeed();
    }
    if(cmd[0] == "save_4d_region")
    {
        auto checked_regions = get_checked_regions();
        if(checked_regions.empty())
            return run->failed("no checked region to save");
        if(!cur_tracking_window.history.get_filename(this,cmd[1],output_format().toStdString()))
            return run->canceled();

        auto dim = checked_regions[0]->dim;
        tipl::image<4,unsigned char> multiple_I(dim.expand(uint32_t(checked_regions.size())));
        tipl::progress prog("aggregating regions");
        size_t p = 0;
        tipl::par_for (checked_regions.size(),[&](unsigned int region_index)
        {
            if(prog.aborted())
                return;
            prog(p++,checked_regions.size());
            size_t offset = region_index*dim.size();
            for (auto& p : checked_regions[region_index]->to_space(
                             dim,checked_regions[0]->to_diffusion_space))
                if (dim.is_valid(p))
                    multiple_I[offset+tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()] = 1;
        });

        if(prog.aborted())
            return run->canceled();
        if(!tipl::io::gz_nifti::save_to_file(cmd[1],multiple_I,
                                  checked_regions[0]->vs,
                                  checked_regions[0]->trans_to_mni,
                                  cur_tracking_window.handle->is_mni))
            return run->failed("cannot save region to " + cmd[1]);
        save_checked_region_label_file(cmd[1].c_str(),0);  // 4d nifti index starts from 0
        return run->succeed();
    }
    if(cmd[0] == "save_all_regions")
    {
        auto checked_regions = get_checked_regions();
        if (checked_regions.empty())
            return run->failed("no checked region to save");
        if(!cur_tracking_window.history.get_filename(this,cmd[1],output_format().toStdString()))
            return run->canceled();
        tipl::shape<3> dim = checked_regions[0]->dim;
        tipl::image<3,unsigned short> mask(dim);
        tipl::par_for (checked_regions.size(),[&](unsigned int region_index)
        {
            auto region_id = uint16_t(region_index+1);
            for (const auto& p : checked_regions[region_index]->to_space(dim,checked_regions[0]->to_diffusion_space))
                if (dim.is_valid(p))
                {
                    auto pos = tipl::pixel_index<3>(p[0],p[1],p[2],dim).index();
                    if(mask[pos] < region_id)
                        mask[pos] = region_id;
                }
        });

        bool result = true;
        if(checked_regions.size() <= 255)
        {
            tipl::image<3,uint8_t> i8mask(mask);
            result = tipl::io::gz_nifti::save_to_file(cmd[1].c_str(),i8mask,
                               checked_regions[0]->vs,
                               checked_regions[0]->trans_to_mni,
                               cur_tracking_window.handle->is_mni);
        }
        else
        {
            result = tipl::io::gz_nifti::save_to_file(cmd[1].c_str(),mask,
                               checked_regions[0]->vs,
                               checked_regions[0]->trans_to_mni,
                               cur_tracking_window.handle->is_mni);
        }
        if(!result)
            return run->failed("cannot write to file " + cmd[1]);
        save_checked_region_label_file(cmd[1].c_str(),1); // 3d nifti index starts from 1
        return run->succeed();
    }


    if(cmd[0] == "save_all_regions_to_folder")
    {
        auto checked_regions = get_checked_regions();
        if(checked_regions.empty())
            return run->failed("no checked region to save");
        if(!cur_tracking_window.history.get_dir(this,cmd[1]))
            return run->canceled();
        tipl::progress prog(cmd[0]);
        for(auto each : checked_regions)
            if(!each->save_region_to_file((cmd[1] + "/" + each->name + output_format().toStdString()).c_str()))
                return run->failed("cannot save " + each->name + " to " + cmd[1]);
        return run->succeed();
    }
    if(cmd[0] == "save_region_info")
    {
        // cmd[1] : file name to be saved
        // cmd[2] : the region index (default: current selected one)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        if(!cur_tracking_window.history.get_filename(this,cmd[1],regions[cur_row]->name + output_format().toStdString()))
            return run->canceled();

        std::ofstream out(cmd[1].c_str());
        out << "x\ty\tz";
        for(unsigned int index = 0;index < cur_tracking_window.handle->dir.num_fiber;++index)
            out << "\tdx" << index << "\tdy" << index << "\tdz" << index;

        for(const auto& each : cur_tracking_window.handle->get_index_list())
            out << "\t" << each;
        for(const auto& each : cur_tracking_window.handle->slices)
            if(!each->optional())
                each->get_image();
        out << std::endl;
        for(auto& point : regions[cur_row]->to_space(cur_tracking_window.handle->dim))
        {
            std::vector<float> data;
            cur_tracking_window.handle->get_voxel_info2(point[0],point[1],point[2],data);
            cur_tracking_window.handle->get_voxel_information(point[0],point[1],point[2],data);
            std::copy(point.begin(),point.end(),std::ostream_iterator<float>(out,"\t"));
            std::copy(data.begin(),data.end(),std::ostream_iterator<float>(out,"\t"));
            out << std::endl;
        }
        return run->succeed();
    }
    if(cmd[0] == "open_region" || cmd[0] == "open_mni_region")
    {
        // cmd[1] : contain only single file name
        if(!cmd[1].empty())
        {
            if(cmd[0] == "open_mni_region" && !cur_tracking_window.handle->map_to_mni())
                return run->failed(cur_tracking_window.handle->error_msg);
            if(!load_multiple_roi_nii(cmd[1].c_str(),cmd[0] == "open_mni_region"))
                return run->failed(error_msg);
            emit need_update();
            return run->succeed();
        }
        // allow for selecting multiple files
        auto file_list = QFileDialog::getOpenFileNames(this,QString::fromStdString(cmd[0]),
            QString::fromStdString(cur_tracking_window.history.file_stem()) + ".nii.gz",
            "Region files (*.nii *.hdr *nii.gz *.mat);;Text files (*.txt);;All files (*)");
        if(file_list.isEmpty())
            return run->canceled();

        // allow sub command to be recorded
        --cur_tracking_window.history.current_recording_instance;
        for(auto each : file_list)
            if(!command({cmd[0],each.toStdString()}))
                break;
        ++cur_tracking_window.history.current_recording_instance;
        if(!error_msg.empty())
            return false;
        return run->canceled(); // no need to record on history
    }
    if(cmd[0] == "load_region_color")
    {
        // cmd[1] : file name
        if(!cur_tracking_window.history.get_filename(this,cmd[1]))
            return run->canceled();

        std::ifstream in(cmd[1].c_str());
        if (!in)
            return run->failed("cannot load file "+cmd[1]);
        std::vector<int> colors((std::istream_iterator<float>(in)),
                                  (std::istream_iterator<float>()));
        if(colors.size() == regions.size()*4) // RGBA
        {
            for(size_t index = 0,pos = 0;index < regions.size() && pos+2 < colors.size();++index,pos+=4)
            {
                tipl::rgb c(std::min<int>(colors[pos],255),
                            std::min<int>(colors[pos+1],255),
                            std::min<int>(colors[pos+2],255),
                            std::min<int>(colors[pos+3],255));
                regions[index]->region_render->color = c;
                regions[index]->modified = true;
                item(int(index),2)->setData(Qt::UserRole,uint32_t(c));
            }
        }
        else
        //RGB
        {
            for(size_t index = 0,pos = 0;index < regions.size() && pos+2 < colors.size();++index,pos+=3)
            {
                tipl::rgb c(std::min<int>(colors[pos],255),
                            std::min<int>(colors[pos+1],255),
                            std::min<int>(colors[pos+2],255),255);
                regions[index]->region_render->color = c;
                regions[index]->modified = true;
                item(int(index),2)->setData(Qt::UserRole,uint32_t(c));
            }
        }
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "save_region_color")
    {
        if(!cur_tracking_window.history.get_filename(this,cmd[1]))
            return run->canceled();

        std::ofstream out(cmd[1].c_str());
        if (!out)
            return run->failed("cannot save region to "+cmd[1]);
        for(size_t index = 0;index < regions.size();++index)
        {
            tipl::rgb c(regions[index]->region_render->color);
            out << int(c[2]) << " " << int(c[1]) << " " << int(c[0]) << " " << int(c[3]) << std::endl;
        }
        return run->succeed();
    }
    if(cmd[0] == "check_all_regions" || cmd[0] == "uncheck_all_regions")
    {
        bool checked = cmd[0] == "check_all_regions";
        cur_tracking_window.glWidget->no_update = true;
        cur_tracking_window.scene.no_update = true;
        for(int row = 0;row < rowCount();++row)
            check_row(row,checked);
        cur_tracking_window.scene.no_update = false;
        cur_tracking_window.glWidget->no_update = false;
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "copy_region")
    {
        // cmd[1] : region index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        unsigned int color = regions[cur_row]->region_render->color.color;
        regions.insert(regions.begin() + cur_row + 1,std::make_shared<ROIRegion>(cur_tracking_window.handle));
        *regions[cur_row + 1] = *regions[cur_row];
        regions[cur_row + 1]->region_render->color.color = color;
        add_row(int(cur_row+1),regions[cur_row]->name.c_str());
        return run->succeed();
    }
    if(cmd[0] == "add_region_from_atlas")
    {
        // cmd[1] : template id  (e.g. 0 for human) [space]
        //          atlas id     (e.g. 0 for the first atlas of human template) [space]
        //          region label , if not supply add all

        size_t template_id,atlas_id;
        std::string region_label;
        std::istringstream(cmd[1]) >> template_id >> atlas_id >> region_label;
        if(template_id >= atlas_file_name_list.size() || atlas_id >= atlas_file_name_list[template_id].size())
            return run->failed("invalid index " + cmd[1]);

        if(template_id != cur_tracking_window.handle->template_id)
            cur_tracking_window.handle->set_template_id(template_id);

        auto at = cur_tracking_window.handle->atlas_list[atlas_id];

        std::vector<std::vector<tipl::vector<3,short> > > points;
        std::vector<std::string> labels;

        if(!region_label.empty()) // has region labels
        {
            std::vector<size_t> label_list;
            for(auto each : tipl::split(region_label,'&'))
            {
                int label = 0;
                std::istringstream(each) >> label;
                if(label < 0 || label >= at->get_list().size())
                    return run->failed("invalid label id: " + each);
                std::vector<tipl::vector<3,short> > point0;
                if(!cur_tracking_window.handle->get_atlas_roi(at,label,
                        cur_tracking_window.current_slice->dim,
                        cur_tracking_window.current_slice->to_dif,point0))
                    return run->failed(cur_tracking_window.handle->error_msg);
                labels.push_back(at->get_list()[label]);
                points.push_back(std::move(point0));
            }
        }
        else
        if(!cur_tracking_window.handle->get_atlas_all_roi(at,
                    cur_tracking_window.current_slice->dim,
                    cur_tracking_window.current_slice->to_dif,points,labels))
            return run->failed(cur_tracking_window.handle->error_msg);


        tipl::progress prog("adding regions");
        begin_update();
        for(size_t i = 0;prog(i,points.size());++i)
        {
            add_region(labels[i].c_str());
            if(!points.empty())
                regions.back()->add_points(std::move(points[i]));
        }
        end_update();
        cur_tracking_window.glWidget->update();
        cur_tracking_window.slice_need_update = true;
        cur_tracking_window.raise();
        return run->succeed();
    }
    if(cmd[0] == "merge_regions")
    {
        // cmd[1] : region index to merge (default: use checked regions)
        std::vector<size_t> merge_list;
        if(cmd[1].empty())
        {
            for(size_t index = 0;index < regions.size();++index)
                if(item(int(index),0)->checkState() == Qt::Checked)
                {
                    merge_list.push_back(index);
                    if(!cmd[1].empty())
                        cmd[1] += '&';
                    cmd[1] += std::to_string(index);
                }
            if(merge_list.size() <= 1)
                return run->failed("select more than two regions to merge");
        }
        else
        {
            // get merge_list from cmd[1]
            for(auto each : QString::fromStdString(cmd[1]).split('&'))
            {
                merge_list.push_back(each.toInt());
                if(merge_list.back() >= regions.size())
                    return run->failed("invalid region index: " + each.toStdString());
            }
        }
        tipl::image<3,unsigned char> mask(regions[merge_list[0]]->dim);
        tipl::progress prog("merging regions",true);
        size_t p = 0;
        tipl::adaptive_par_for(merge_list.size(),[&](size_t index)
        {
            if(prog.aborted())
                return;
            prog(p++,merge_list.size());
            for(auto& p: regions[merge_list[index]]->to_space(
                                    regions[merge_list[0]]->dim,
                                    regions[merge_list[0]]->to_diffusion_space))
                if (mask.shape().is_valid(p))
                    mask.at(p) = 1;
        });
        if(prog.aborted())
            return run->canceled();
        regions[merge_list[0]]->load_region_from_buffer(mask);
        begin_update();
        for(int index = merge_list.size()-1;index >= 1;--index)
        {
            regions.erase(regions.begin()+merge_list[index]);
            removeRow(merge_list[index]);
        }
        end_update();
        emit need_update();
        return run->succeed();
    }

    if(cmd[0] == "delete_region")
    {
        // cmd[1] : region index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        regions.erase(regions.begin()+cur_row);
        removeRow(cur_row);
        emit need_update();
        return run->succeed();
    }

    if(cmd[0] == "delete_all_regions")
    {
        setRowCount(0);
        regions.clear();
        color_gen = 0;
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "move_slice_to_region")
    {
        // cmd[1] : region index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        auto current_slice = cur_tracking_window.current_slice;
        auto current_region = regions[cur_row];
        if(current_region->region.empty())
            return run->canceled();
        tipl::vector<3,float> p(current_region->get_center());
        if(!current_slice->is_diffusion_space)
            p.to(current_slice->to_slice);
        cur_tracking_window.move_slice_to(p);
        return run->succeed();
    }

    if(cmd[0] == "show_device_statistics" || cmd[0] == "save_device_statistics" ||
       cmd[0] == "show_region_statistics" || cmd[0] == "save_region_statistics" ||
       cmd[0] == "show_t2r" || cmd[0] == "save_t2r" ||
       cmd[0] == "show_tract_statistics" || cmd[0] == "save_tract_statistics" ||
       cmd[0] == "show_tract_recognition" || cmd[0] == "save_tract_recognition")
    {
        // cmd[1] : file name to save
        auto regions = get_checked_regions();
        auto tracts = cur_tracking_window.tractWidget->get_checked_tracks();
        auto devices = cur_tracking_window.deviceWidget->devices;

        std::string result,title,default_file(cur_tracking_window.history.file_stem(false/*basic stem*/));
        tipl::progress p(cmd[0],true);
        if(tipl::contains(cmd[0],"t2r"))
        {
            if(regions.empty())
                return run->failed("please add parcellation regions");
            if(tracts.empty())
                return run->failed("please specify tract(s)");
            Parcellation p(cur_tracking_window.handle);
            p.load_from_regions(regions);
            result = p.get_t2r(tracts);
            title = "Tract-To-Region Connectome";
            default_file += "_" + tracts.front()->name + "_t2r.txt";

        }
        if(tipl::contains(cmd[0],"tract"))
        {
            if(tracts.empty())
                return run->failed("please specify tract(s)");
            if(tipl::ends_with(cmd[0],"recognition"))
            {
                // cmd[2] : tract id for recognition
                int cur_row = run->from_cmd(2,cur_tracking_window.tractWidget->currentRow());
                if(!cur_tracking_window.handle->load_track_atlas(false/*asymmetric*/))
                    return run->failed(cur_tracking_window.handle->error_msg);

                auto lock = cur_tracking_window.tractWidget->tract_rendering[cur_row]->start_reading();
                auto sorted_list = cur_tracking_window.handle->recognize_and_sort(cur_tracking_window.tractWidget->tract_models[cur_row]);
                if(sorted_list.empty())
                    return run->failed("cannot recognize tracks.");
                std::ostringstream out;
                for(const auto& each : sorted_list)
                    if(each.first != 0.0f)
                        out << each.first*100.0f << "%\t" << each.second << std::endl;
                result = out.str();
                title = "Tract Recognition";
                default_file += "_tract_names.txt";
            }
            else
            {
                get_tract_statistics(cur_tracking_window.handle,tracts,result);
                title = "Tract Statistics";
                default_file += "_tract_stat.txt";
            }

        }
        if(tipl::contains(cmd[0],"region"))
        {
            if(regions.empty())
                return run->failed("please specify regions");
            get_regions_statistics(cur_tracking_window.handle,regions,result);

            // add t2r
            if(!tracts.empty())
            {

                Parcellation p(cur_tracking_window.handle);
                p.load_from_regions(regions);
                auto result2 = p.get_t2r(tracts);
                result2.erase(0, result2.find('\n') + 1);
                result += result2;
            }
            title = "Region Statistics";
            default_file += "_region_stat.txt";
        }
        if(tipl::contains(cmd[0],"device"))
        {
            if(devices.empty())
                return run->failed("please specify devices");
            get_devices_statistics(cur_tracking_window.handle,devices,result);
            title = "Device Statistics";
            default_file += "_device_stat.txt";
        }

        if(!cmd[1].empty())
        {
            tipl::out() << "save " << cmd[1];
            std::ofstream out(cmd[1]);
            if(!out)
                return run->failed("cannot write to " + cmd[1]);
            out << result;
        }
        else
        {
            cmd[1] = show_info_dialog(title,result,default_file);
            if(!cmd[1].empty())
            {
                // change show to save
                cmd[0][1] = 'a';cmd[0][2] = 'v';cmd[0][3] = 'e';
            }
        }
        return run->succeed();
    }


    return run->not_processed();
}


void RegionTableWidget::draw_region(const tipl::matrix<4,4>& current_slice_T,unsigned char dim,int slice_pos,
                                    const tipl::shape<2>& slice_image_shape,float display_ratio,QImage& scaled_image)
{

    // during region removal, there will be a call with invalid currentRow
    auto checked_regions = get_checked_regions();
    if(checked_regions.empty() || currentRow() >= int(regions.size()) || currentRow() == -1)
        return;

    std::vector<tipl::image<2,uint8_t> > region_masks(checked_regions.size());
    {
        int w = slice_image_shape.width();
        int h = slice_image_shape.height();
        std::vector<uint32_t> yw((size_t(h)));
        for(uint32_t y = 0;y < uint32_t(h);++y)
            yw[y] = y*uint32_t(w);
        tipl::adaptive_par_for(region_masks.size(),[&](uint32_t roi_index)
        {
            tipl::image<2,uint8_t> region_mask(slice_image_shape);
            if(current_slice_T != checked_regions[roi_index]->to_diffusion_space)
            {
                auto iT = tipl::from_space(checked_regions[roi_index]->to_diffusion_space).to(current_slice_T);
                tipl::transformation_matrix<float> m = iT;
                tipl::vector<3,uint32_t> range;
                for(int d = 0;d < 3;++d)
                {
                    auto p = tipl::slice2space<tipl::vector<3,float> >(dim,char(d == 0),char(d == 1),char(d == 2));
                    p.rotate(m);
                    range[d] = uint32_t(std::ceil(p.length()));
                }

                for(const auto& index : checked_regions[roi_index]->region)
                {
                    tipl::vector<3,float> p(index);
                    p.to(iT);
                    auto p2 = tipl::space2slice<tipl::vector<3,float> >(dim,p);
                    if (std::fabs(float(slice_pos)-p2[2]) >= range[2])
                        continue;
                    p2.round();
                    if(!slice_image_shape.is_valid(p2))
                        continue;
                    uint32_t pos = yw[uint32_t(p2[1])]+uint32_t(p2[0]);
                    for(uint32_t dy = 0;dy < range[1];++dy,pos += uint32_t(w))
                        for(uint32_t dx = 0,pos2 = pos;dx < range[0];++dx,++pos2)
                            region_mask[pos2] = 1;
                }
            }
            else
            {
                for(const auto& p : checked_regions[roi_index]->region)
                {
                    auto pos = tipl::space2slice<tipl::vector<3,int> > (dim,p);
                    if (slice_pos != pos[2] || !slice_image_shape.is_valid(pos))
                        continue;
                    region_mask[uint32_t(yw[uint32_t(pos[1])]+uint32_t(pos[0]))] = 1;
                }
            }
            region_masks[roi_index].swap(region_mask);
        });
    }

    int cur_roi_index = -1;
    if(currentRow() >= 0)
        for(unsigned int i = 0;i < checked_regions.size();++i)
            if(checked_regions[i] == regions[uint32_t(currentRow())])
            {
                cur_roi_index = int(i);
                break;
            }

    std::vector<tipl::rgb> colors;
    for(auto& region : checked_regions)
    {
        colors.push_back(region->region_render->color);
        colors.back().a = 255;
    }
    scaled_image = tipl::qt::draw_regions(region_masks,colors,
                    cur_tracking_window["roi_draw_edge"].toInt(),
                    cur_tracking_window["roi_edge_width"].toInt(),
                    cur_roi_index,display_ratio);

}
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<SliceModel*>& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::string& error_msg,
              bool is_mni);
bool RegionTableWidget::load_multiple_roi_nii(QString file_name,bool is_mni)
{
    QStringList files = file_name.split('&');
    std::vector<SliceModel*> transform_lookup;
    // searching for T1/T2 mappings
    for(unsigned int index = 0;index < cur_tracking_window.slices.size();++index)
    {
        auto slice = cur_tracking_window.slices[index];
        if(!slice->is_diffusion_space)
            transform_lookup.push_back(slice.get());
    }
    std::vector<std::vector<std::shared_ptr<ROIRegion> > > loaded_regions(files.size());

    {
        tipl::progress prog("reading region files");
        size_t p = 0;
        bool failed = false;
        tipl::adaptive_par_for(files.size(),[&](unsigned int i)
        {
            if(prog.aborted() || failed)
                return;
            prog(p++,files.size());
            if(files[i].endsWith("nii.gz") || files[i].endsWith("nii") || files[i].endsWith("hdr"))
            {
                if(!load_nii(cur_tracking_window.handle,
                         files[i].toStdString(),
                         transform_lookup,
                         loaded_regions[i],
                         error_msg,is_mni))
                {
                    failed = true;
                    return;
                }
            }
            else
            {
                std::shared_ptr<ROIRegion> region(new ROIRegion(cur_tracking_window.handle));
                if(!region->load_region_from_file(files[i].toStdString().c_str()))
                {
                    error_msg = "cannot read " + files[i].toStdString();
                    failed = true;
                    return;
                }
                region->name = QFileInfo(files[i]).completeBaseName().toStdString();
                loaded_regions[i].push_back(region);
            }
        });

        if(prog.aborted())
            return true;
        if(failed)
            return false;
    }

    tipl::aggregate_results(std::move(loaded_regions),loaded_regions[0]);

    {
        tipl::progress prog("loading ROIs");
        begin_update();
        for(uint32_t i = 0;prog(i,loaded_regions[0].size());++i)
            {
                regions.push_back(loaded_regions[0][i]);
                add_row(int(regions.size()-1),loaded_regions[0][i]->name.c_str());
                check_row(currentRow(),loaded_regions.size() == 1);
            }
        end_update();
    }
    return true;

}

void RegionTableWidget::check_row(size_t row,bool checked)
{
    item(row,0)->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
    item(row,0)->setData(Qt::UserRole+1,checked ? Qt::Unchecked : Qt::Checked);
}

void RegionTableWidget::move_up(void)
{
    if(currentRow())
    {
        regions[uint32_t(currentRow())].swap(regions[uint32_t(currentRow())-1]);
        begin_update();
        for(int i = 0;i < columnCount();++i)
        {
            QTableWidgetItem* item0 = takeItem(currentRow()-1,i);
            QTableWidgetItem* item1 = takeItem(currentRow(),i);
            setItem(currentRow()-1,i,item1);
            setItem(currentRow(),i,item0);
        }
        end_update();
        setCurrentCell(currentRow()-1,0);
    }

}

void RegionTableWidget::move_down(void)
{
    if(currentRow()+1 < int(regions.size()))
    {
        regions[uint32_t(currentRow())].swap(regions[uint32_t(currentRow())+1]);
        begin_update();
        for(int i = 0;i < columnCount();++i)
        {
            QTableWidgetItem* item0 = takeItem(currentRow()+1,i);
            QTableWidgetItem* item1 = takeItem(currentRow(),i);
            setItem(currentRow()+1,i,item1);
            setItem(currentRow(),i,item0);
        }
        end_update();
        setCurrentCell(currentRow()+1,0);
    }
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

void RegionTableWidget::save_checked_region_label_file(QString filename,int first_index)
{
    QString base_name = QFileInfo(filename).completeBaseName();
    if(base_name.endsWith(".nii"))
        base_name.chop(4);
    QString label_file = QFileInfo(filename).absolutePath()+"/"+base_name+".txt";
    std::ofstream out(label_file.toStdString().c_str());
    for (auto each : get_checked_regions())
    {
        out << first_index << " " << each->name << std::endl;
        ++first_index;
    }
}


bool RegionTableWidget::set_roi(const std::string& settings,std::shared_ptr<RoiMgr> roi)
{
    for (auto each : tipl::split(settings,'&'))
    {
        std::replace(each.begin(),each.end(),':',' ');
        size_t roi_index = 0,roi_type = 0;
        std::istringstream(each) >> roi_index >> roi_type;
        if(roi_index >= regions.size() || roi_type >= default_id)
        {
            error_msg = "invalid roi/roa/end region settings";
            return false;
        }
        roi->setRegions(regions[roi_index]->region,
                        regions[roi_index]->dim,
                        regions[roi_index]->to_diffusion_space,
                        roi_type,
                        regions[roi_index]->name.c_str());
    }
    return true;
}
std::string RegionTableWidget::get_roi_settings(void)
{
    std::string result;
    for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
    {
        if (item(roi_index,0)->checkState() != Qt::Checked ||
            regions[roi_index]->regions_feature == default_id)
            continue;
        if(!result.empty())
            result += '&';
        result += std::to_string(roi_index) + ":" + std::to_string(int(regions[roi_index]->regions_feature));
    }
    return result;
}

QString RegionTableWidget::getROIname(void)
{
    for (auto each : get_checked_regions())
        if (each->regions_feature == roi_id)
            return each->name.c_str();
    for (auto each : get_checked_regions())
        if (each->regions_feature == seed_id)
            return each->name.c_str();
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

bool RegionTableWidget::do_action(std::vector<std::string>& cmd)
{
    // cmd[0] : action
    // cmd[1] : region index (default current row)
    // cmd[2] : additional parameters
    auto checked_regions = get_checked_regions();


    QString action = cmd[0].substr(14).c_str();
    int roi_index = 0;
    std::vector<std::shared_ptr<ROIRegion> > region_to_be_processed;
    for(auto each : QString::fromStdString(cmd[1]).split('&'))
    {
        auto index = each.toInt();
        if(region_to_be_processed.empty())
            roi_index = index;
        if(index < regions.size())
            region_to_be_processed.push_back(regions[each.toInt()]);
    }


    std::vector<int> rows_to_be_updated;
    tipl::progress prog(action.toStdString().c_str(),true);
    {
        if(action == "1st_ex_all" || action == "all_ex_1st" || action == "all_inter_1st" || action == "all_to_1st" || action == "all_to_1st_2")
        {
            if(checked_regions.size() < 2)
                return false;
            auto base_dim = checked_regions[0]->dim;
            auto base_to_dif = checked_regions[0]->to_diffusion_space;
            std::vector<unsigned int> checked_row;
            for (unsigned int r = 0;r < regions.size();++r)
                if (item(r,0)->checkState() == Qt::Checked)
                    checked_row.push_back(r);

            tipl::image<3,unsigned char> A;
            tipl::image<3,uint16_t> A_labels;
            checked_regions[0]->save_region_to_buffer(A);
            if(action == "all_to_1st" || action == "all_to_1st_2")
                A_labels.resize(base_dim);

            {
                tipl::progress prog2("processing regions");
                size_t prog_count = 0;
                tipl::adaptive_par_for(checked_regions.size(),[&](size_t r)
                {
                    prog2(prog_count++,checked_regions.size());
                    if(r == 0)
                        return;
                    tipl::image<3,unsigned char> B;
                    checked_regions[r]->save_region_to_buffer(B,base_dim,base_to_dif);
                    if(action == "1st_ex_all")
                    {
                        tipl::masking(A,B);
                        return; // don't update B
                    }
                    if(action == "all_ex_1st")
                        tipl::masking(B,A);
                    if(action == "all_inter_1st")
                        for(size_t i = 0;i < B.size();++i)
                            B[i] = (A[i] & B[i]);
                    if(action == "all_to_1st" || action == "all_to_1st_2")
                        for(size_t i = 0;i < A.size();++i)
                            if(A[i] && B[i] && A_labels[i] < r)
                                A_labels[i] = uint16_t(r);

                    checked_regions[r]->vs = checked_regions[0]->vs;
                    checked_regions[r]->dim = base_dim;
                    checked_regions[r]->is_diffusion_space = checked_regions[0]->is_diffusion_space;
                    checked_regions[r]->to_diffusion_space = base_to_dif;
                    checked_regions[r]->trans_to_mni = checked_regions[0]->trans_to_mni;
                    checked_regions[r]->is_mni = checked_regions[0]->is_mni;
                    checked_regions[r]->region.clear();
                    checked_regions[r]->undo_backup.clear();
                    checked_regions[r]->redo_backup.clear();
                    checked_regions[r]->load_region_from_buffer(B);
                    rows_to_be_updated.push_back(checked_row[r]);
                });
            }
            if(action == "1st_ex_all")
                checked_regions[0]->load_region_from_buffer(A);


            if(action == "all_to_1st_2")
            {
                auto I = tipl::resample(cur_tracking_window.current_slice->get_source(),base_dim,
                                        tipl::from_space(cur_tracking_window.current_slice->to_dif).
                                        to(base_to_dif));
                tipl::image<3,unsigned char> edges(base_dim);
                tipl::adaptive_par_for(checked_regions.size(),[&](size_t r)
                {
                    if(r == 0)
                        return;
                    tipl::image<3,unsigned char> B;
                    checked_regions[r]->save_region_to_buffer(B);
                    tipl::morphology::edge(B);
                    for(size_t pos = 0;pos < B.size();++pos)
                        if(A[pos] && B[pos])
                            edges[pos] += 1;
                });

                tipl::adaptive_par_for(tipl::begin_index(base_dim),tipl::end_index(base_dim),
                                       [&](const tipl::pixel_index<3>& pos)
                {
                    if(edges[pos.index()] <= 1)
                        return;
                    std::vector<float> votes(checked_regions.size());
                    votes[A_labels[pos.index()]] += 4.0f;


                    // spatial voting, total vote: 32 x 0.5 = 16
                    tipl::for_each_connected_neighbors(pos,base_dim,[&](const tipl::pixel_index<3>& rhs_pos)
                    {
                        votes[A_labels[rhs_pos.index()]] += 0.5f;
                    });
                    tipl::for_each_neighbors(pos,base_dim,[&](const tipl::pixel_index<3>& rhs_pos)
                    {
                        votes[A_labels[rhs_pos.index()]] += 0.5f;
                    });

                    // value voting, total vote: 16
                    {
                        float pos_value = I[pos.index()];
                        std::vector<float> dif_values(16,std::numeric_limits<float>::max());
                        std::vector<size_t> regions(16);
                        tipl::for_each_neighbors(pos,base_dim,4,[&](const auto& rhs_pos)
                        {
                            float dif_value = std::fabs(pos_value-I[rhs_pos.index()]);
                            if(dif_value > dif_values.back())
                                return;
                            size_t ins_pos = dif_values.size()-1;
                            for(;ins_pos;--ins_pos)
                                if(dif_value > dif_values[ins_pos-1])
                                    break;
                            dif_values.insert(dif_values.begin()+ins_pos,dif_value);
                            regions.insert(regions.begin()+ins_pos,A_labels[rhs_pos.index()]);
                            dif_values.pop_back();
                            regions.pop_back();
                        });

                        for(auto r : regions)
                            if(r)
                                votes[r] += 1.0f;
                    }
                    A_labels[pos.index()] = std::max_element(votes.begin(),votes.end())-votes.begin();
                });
            }
            if(action == "all_to_1st")
            {
                // usde region growing to assign labels
                {
                    tipl::image<3,unsigned char> fillup_map(A);
                    for(size_t i = 0;i < fillup_map.size();++i)
                        if(fillup_map[i] && A_labels[i])
                            fillup_map[i] = 0;
                    size_t total_fillups = std::accumulate(fillup_map.begin(),fillup_map.end(),size_t(0));
                    std::vector<std::vector<tipl::pixel_index<3> > > region_front(checked_regions.size());
                    for(tipl::pixel_index<3> index(base_dim);index < base_dim.size();++index)
                        if(A_labels[index.index()])
                        {
                            bool is_front = false;
                            tipl::for_each_connected_neighbors(index,base_dim,[&](const auto& index2)
                            {
                                if(fillup_map[index2.index()])
                                    is_front = true;
                            });
                            if(is_front)
                                region_front[A_labels[index.index()]].push_back(index);
                        }
                    tipl::progress prog2("region growing",true);
                    for(size_t cur_fillups = 0;prog2(cur_fillups,total_fillups);)
                    {
                        size_t sum_front = 0;
                        for(size_t r = 1;r < region_front.size();++r)
                        {
                            std::vector<tipl::pixel_index<3> > new_front;
                            for(const auto& each : region_front[r])
                            {
                                tipl::for_each_connected_neighbors(each,base_dim,[&](const auto& index2)
                                {
                                    if(fillup_map[index2.index()])
                                    {
                                        new_front.push_back(index2);
                                        fillup_map[index2.index()] = 0;
                                        A_labels[index2.index()] = r;
                                    }
                                });
                            }
                            sum_front += new_front.capacity();
                            region_front[r].swap(new_front);
                        }
                        if(!sum_front)
                            break;
                    }
                    if(prog.aborted())
                        return false;
                }




                std::vector<size_t> need_fill_up;
                {
                    std::vector<std::vector<size_t> > need_fill_ups(tipl::max_thread_count);
                    tipl::par_for<tipl::sequential_with_id>(A.size(),[&](size_t index,int id)
                    {
                        if(A[index] && !A_labels[index])
                            need_fill_ups[id].push_back(index);
                    });
                    tipl::aggregate_results(std::move(need_fill_ups),need_fill_up);
                }
                {
                    size_t prog_count = 0;
                    tipl::progress prog2("assign labels",true);
                    tipl::par_for(need_fill_up.size(),[&](size_t i)
                    {
                        prog2(prog_count++, need_fill_up.size());
                        tipl::vector<3> pos(tipl::pixel_index<3>(need_fill_up[i], A.shape()));
                        size_t nearest_label = 0;
                        size_t max_distance = A_labels.size();
                        for(tipl::pixel_index<3> index2(A_labels.shape());index2 < A_labels.size();++index2)
                        {
                            if(!A_labels[index2.index()])
                                continue;
                            tipl::vector<3> pos2(index2);
                            if (std::abs(pos2[0] - pos[0]) < max_distance &&
                                std::abs(pos2[1] - pos[1]) < max_distance &&
                                std::abs(pos2[2] - pos[2]) < max_distance)
                            {
                                size_t length2 = (pos2 - pos).length2();
                                if(length2 < max_distance)
                                {
                                    max_distance = length2;
                                    nearest_label = A_labels[index2.index()];
                                }
                            }
                        }
                        A_labels[need_fill_up[i]] = nearest_label;
                    },tipl::max_thread_count);
                    if(prog2.aborted())
                        return false;
                }
            }

            if(action == "all_to_1st_2" || action == "all_to_1st")
            {
                tipl::progress prog("loading regions",true);
                size_t prog_count = 0;
                tipl::adaptive_par_for(checked_regions.size(),[&](size_t r)
                {
                    prog(prog_count++,checked_regions.size());
                    if(r == 0)
                        return;
                    tipl::image<3,unsigned char> B(base_dim);
                    for(size_t i = 0;i < A_labels.size();++i)
                        if(A_labels[i] == r)
                            B[i] = 1;
                    checked_regions[r]->load_region_from_buffer(B);
                });
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
                (int lhs,int rhs)
                {
                    auto lstr = regions[lhs]->name;
                    auto rstr = regions[rhs]->name;
                    return negate ^ (lstr.length() == rstr.length() ? lstr < rstr : lstr.length() < rstr.length());
                });
            }
            else
            {
                std::vector<float> data(regions.size());
                tipl::adaptive_par_for(regions.size(),[&](unsigned int index){
                    if(action == "sort_x")
                        data[index] = regions[index]->get_pos()[0];
                    if(action == "sort_y")
                        data[index] = regions[index]->get_pos()[1];
                    if(action == "sort_z")
                        data[index] = regions[index]->get_pos()[2];
                    if(action == "sort_size")
                        data[index] = regions[index]->get_volume();
                });

                arg = tipl::arg_sort(data,[negate](float lhs,float rhs){return negate ^ (lhs < rhs);});
            }

            std::vector<QTableWidgetItem*> items;
            std::vector<std::shared_ptr<ROIRegion> > new_region(arg.size());
            begin_update();
            size_t col_count = columnCount();
            for(size_t i = 0;i < arg.size();++i)
            {
                new_region[i] = regions[arg[i]];
                for(size_t j = 0;j < col_count;++j)
                    items.push_back(takeItem(arg[i],j));
            }
            regions.swap(new_region);
            for(size_t i = 0,pos = 0;i < arg.size();++i)
                for(size_t j = 0;j < col_count;++j,++pos)
                    setItem(i,j,items[pos]);
            end_update();
        }

        tipl::adaptive_par_for(region_to_be_processed.size(),[&](unsigned int i)
        {
            region_to_be_processed[i]->perform(action.toStdString());
        });


        if(action == "dilation_by_voxel")
        {
            int threshold = 10;
            if(cmd[2].empty())
            {
                bool ok;
                threshold = QInputDialog::getInt(this,QApplication::applicationName(),"Voxel distance",10,1,100,1,&ok);
                if(!ok)
                    return false;
                cmd[2] = std::to_string(threshold);
            }
            else
                threshold = QString::fromStdString(cmd[2]).toInt();
            size_t p = 0;
            for(auto& region : region_to_be_processed)
            {
                prog(p++,region_to_be_processed.size());
                tipl::image<3,unsigned char> mask;
                region->save_region_to_buffer(mask);
                tipl::morphology::dilation2(mask,threshold);
                region->load_region_from_buffer(mask);
            }
        }
        if(action == "threshold" || action == "threshold_current")
        {
            tipl::const_pointer_image<3,float> I = cur_tracking_window.current_slice->get_source();
            if(I.empty())
                return false;
            double m = tipl::max_value(I);
            bool flip = false;
            float threshold = 0.0f;
            if(cmd[2].empty())
            {
                bool ok;
                threshold = float(QInputDialog::getDouble(this,
                                  QApplication::applicationName(),"Threshold (assign negative value to get low pass):",
                                  double(tipl::segmentation::otsu_threshold(I)),-m,m,4, &ok));
                if(!ok)
                    return false;
                cmd[2] = std::to_string(threshold);
            }
            else
                threshold = QString::fromStdString(cmd[2]).toFloat();
            if(threshold < 0)
            {
                flip = true;
                threshold = -threshold;
            }
            size_t p = 0;
            for(auto& region : region_to_be_processed)
            {
                if(!prog(p++,region_to_be_processed.size()))
                    break;
                tipl::image<3,unsigned char> mask(I.shape());
                if(action == "threshold_current")
                {
                    bool need_trans = false;
                    tipl::matrix<4,4> trans = tipl::identity_matrix();
                    if(cur_tracking_window.current_slice->dim != region->dim ||
                       cur_tracking_window.current_slice->to_dif != region->to_diffusion_space)
                    {
                        need_trans = true;
                        trans = cur_tracking_window.current_slice->to_slice*region->to_diffusion_space;
                    }

                    region->save_region_to_buffer(mask);

                    tipl::adaptive_par_for(tipl::begin_index(mask.shape()),tipl::end_index(mask.shape()),
                                  [&](const tipl::pixel_index<3>& pos)
                    {
                        if(!mask[pos.index()])
                            return;
                        float value = 0.0f;
                        size_t i = pos.index();
                        if(need_trans)
                        {
                            tipl::vector<3> p(pos);
                            p.to(trans);
                            if(!tipl::estimate(I,p,value))
                                return;
                        }
                        else
                            value = I[i];
                        mask[i]  = ((value > threshold) ^ flip) ? 1:0;
                    });
                }
                else
                {
                    tipl::adaptive_par_for(mask.size(),[&](size_t i)
                    {
                        mask[i]  = ((I[i] > threshold) ^ flip) ? 1:0;
                    });
                }
                region->load_region_from_buffer(mask);
            }

        }
        if(action == "separate")
        {
            ROIRegion& cur_region = *regions[size_t(roi_index)];
            tipl::image<3,unsigned char>mask;
            cur_region.save_region_to_buffer(mask);
            QString name = item(roi_index,0)->text();
            tipl::image<3,unsigned int> labels;
            std::vector<std::vector<size_t> > r;
            tipl::morphology::connected_component_labeling(mask,labels,r);
            begin_update();
            for(unsigned int j = 0,total_count = 0;j < r.size() && total_count < 256;++j)
                if(!r[j].empty())
                {
                    mask = 0;
                    for(unsigned int i = 0;i < r[j].size();++i)
                        mask[r[j][i]] = 1;
                    regions.push_back(std::make_shared<ROIRegion>(cur_tracking_window.handle));
                    regions.back()->dim = cur_region.dim;
                    regions.back()->vs = cur_region.vs;
                    regions.back()->is_diffusion_space = cur_region.is_diffusion_space;
                    regions.back()->to_diffusion_space = cur_region.to_diffusion_space;
                    regions.back()->is_mni = cur_region.is_mni;
                    regions.back()->load_region_from_buffer(mask);
                    add_row(int(regions.size()-1),(name + "_"+QString::number(total_count+1)));
                    ++total_count;
                }
            end_update();
        }
    }

    for(int i : rows_to_be_updated)
    {
        closePersistentEditor(item(i,1));
        closePersistentEditor(item(i,2));
        item(i,1)->setData(Qt::DisplayRole,regions[uint32_t(i)]->regions_feature);
        item(i,2)->setData(Qt::UserRole,regions[uint32_t(i)]->region_render->color.color);
        item(i,3)->setText(QString("(%1,%2,%3)x(%4,%5,%6)")
                           .arg(regions[i]->dim[0])
                           .arg(regions[i]->dim[1])
                           .arg(regions[i]->dim[2])
                           .arg(regions[i]->vs[0])
                           .arg(regions[i]->vs[1])
                           .arg(regions[i]->vs[2]));
        openPersistentEditor(item(i,1));
        openPersistentEditor(item(i,2));
    }
    emit need_update();
    return true;
}
