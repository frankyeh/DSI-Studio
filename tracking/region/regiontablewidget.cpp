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
#include "qcolorcombobox.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "opengl/glwidget.h"
#include "opengl/renderingtablewidget.h"
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

void get_regions_statistics(std::shared_ptr<fib_data> handle,const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            std::string& result)
{
    std::vector<std::string> titles;
    std::vector<std::vector<float> > data(regions.size());
    tipl::progress p("for each region");
    for(size_t index = 0;index < regions.size();++index)
    {
        std::vector<std::string> dummy;
        regions[index]->get_quantitative_data(handle,(index == 0) ? titles : dummy,data[index]);
    }
    std::ostringstream out;
    out << "Name";
    for(auto each : regions)
        out << "\t" << each->name;
    out << std::endl;
    for(unsigned int i = 0;i < titles.size();++i)
    {
        out << titles[i];
        for(unsigned int j = 0;j < regions.size();++j)
        {
            out << "\t";
            if(i < data[j].size())
                out << data[j][i];
        }
        out << std::endl;
    }
    result = out.str();
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
    setStyleSheet("QTableView {selection-background-color: #AAAAFF; selection-color: #000000;}");

    setItemDelegate(new ImageDelegate(this));

    connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_check_status(int,int)));
    connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateRegions(QTableWidgetItem*)));
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

void RegionTableWidget::updateRegions(QTableWidgetItem* item)
{
    if (item->column() == 0)
        regions[uint32_t(item->row())]->name = item->text().toStdString();
    if (item->column() == 1)
        regions[uint32_t(item->row())]->regions_feature = uint8_t(item->text().toInt());
    if (item->column() == 2)
    {
        regions[uint32_t(item->row())]->region_render->color = uint32_t(item->data(Qt::UserRole).toInt());
        emit need_update();
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

    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
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
extern std::vector<std::vector<std::string> > atlas_file_name_list;
bool RegionTableWidget::command(std::vector<std::string> cmd)
{
    auto run = cur_tracking_window.history.record(error_msg,cmd);
    if(cmd.size() < 3)
        cmd.resize(3);

    auto get_cur_row = [&](std::string& cmd_text,int cur_row)->bool
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
        return true;
    };

    if(cmd[0] == "new_region")
    {
        add_region("New Region");
        return true;
    }
    if(cmd[0] == "new_region_from_threshold")
    {
        add_region("New Region");
        if(!do_action("threshold",cmd[1]))
        {
            if(!error_msg.empty())
                return false;
            return run->canceled();
        }
        return true;
    }
    if(cmd[0] == "new_region_from_mni")
    {
        if(cmd[1].empty() && (cmd[1] =
                QInputDialog::getText(this,QApplication::applicationName(),
                "Please specify the MNI Coordinate and radius of the region, separated by spaces (e.g. 0 -10 21 10)",
                                                    QLineEdit::Normal,"0 0 0 10").toStdString()).empty())
            return run->canceled();
        QStringList params = QString::fromStdString(cmd[1]).split(' ');
        if(params.size() != 4)
            return run->failed("invalid numbers. please specify four numbers separated by spaces");
        if(!cur_tracking_window.handle->map_to_mni())
            return run->failed("cannot map to MNI space: " + cur_tracking_window.handle->error_msg);
        add_region("New Region");
        regions.back()->new_from_mni_sphere(cur_tracking_window.handle,
                                            tipl::vector<3>(params[0].toFloat(),params[1].toFloat(),params[2].toFloat()),params[3].toFloat());
        return true;
    }
    if(cmd[0] == "save_region")
    {
        int cur_row = currentRow();
        // cmd[1] : file name to be saved
        // cmd[2] : the region index (default: current selected one)
        if(!get_cur_row(cmd[2],cur_row))
            return false;

        if(cmd[1].empty() && (cmd[1] =
                QFileDialog::getSaveFileName(
                this,"Save region",QString(regions[cur_row]->name.c_str()) + output_format(),
                "NIFTI file(*nii.gz *.nii);;Text file(*.txt);;MAT file (*.mat);;All files(*)" ).toStdString()).empty())
            return run->canceled();

        if(!tipl::ends_with(cmd[1],".mat") &&
           !tipl::ends_with(cmd[1],".txt") &&
           !tipl::ends_with(cmd[1],".nii") &&
           !tipl::ends_with(cmd[1],".nii.gz"))
            cmd[1] += ".nii.gz";
        if(!regions[cur_row]->save_region_to_file(cmd[1].c_str()))
            return run->failed("cannot save region to "+cmd[1]);
        return true;
    }
    if(cmd[0] == "save_4d_region")
    {
        auto checked_regions = get_checked_regions();
        if (checked_regions.empty())
            return run->failed("no checked region to save");
        if(cmd[1].empty() && (cmd[1] =
                QFileDialog::getSaveFileName(
                               this,"Save region",QString(checked_regions[0]->name.c_str()) + output_format(),
                               "Region file(*nii.gz *.nii);;All file types (*)" ).toStdString()).empty())
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
        return true;
    }
    if(cmd[0] == "save_all_regions")
    {
        auto checked_regions = get_checked_regions();
        if (checked_regions.empty())
            return run->failed("no checked region to save");
        if(cmd[1].empty() && (cmd[1] =
                        QFileDialog::getSaveFileName(
                        this,"Save region",QString(checked_regions[0]->name.c_str()) + output_format(),
                        "Region file(*nii.gz *.nii *.mat);;Text file (*.txt);;All file types (*)" ).toStdString()).empty())
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
        return true;
    }


    if(cmd[0] == "save_regions_to_folder")
    {
        auto checked_regions = get_checked_regions();
        if (checked_regions.empty())
            return run->failed("no checked region to save");
        if(cmd[1].empty() && (cmd[1] =
                QFileDialog::getExistingDirectory(this,"Open directory","").toStdString()).empty())
            return run->canceled();
        tipl::progress prog("saving files");
        for(auto each : checked_regions)
            if(!each->save_region_to_file((cmd[1] + "/" + each->name + output_format().toStdString()).c_str()))
                return run->failed("cannot save " + each->name + " to " + cmd[1]);
        return true;
    }
    if(cmd[0] == "open_region" || cmd[0] == "open_mni_region")
    {
        // cmd[1] : contain only single file name
        if(!cmd[1].empty())
        {
            if(cmd[0] == "open_mni_region" && !cur_tracking_window.handle->map_to_mni())
                return run->failed(cur_tracking_window.handle->error_msg);
            if(!load_multiple_roi_nii(cmd[1].c_str(),cmd[0] == "open_mni_region"))
                return false;
            emit need_update();
            return true;
        }
        // allow for selecting multiple files
        auto file_list = QFileDialog::getOpenFileNames(
            this,"Open Region(s)",QFileInfo(cur_tracking_window.work_path).absolutePath(),
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
        return true;
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
        return true;
    }
    if(cmd[0] == "add_region_from_atlas")
    {
        // cmd[1] : template id  (e.g. o for human)
        // cmd[2] : atlas id     (e.g. 0 for the first atlas of human template)
        // cmd[3] : region label , if not supply add all
        size_t template_id = QString::fromStdString(cmd[1]).toInt();
        if(template_id >= atlas_file_name_list.size())
            return run->failed("invalid template id: " + cmd[1]);
        if(template_id != cur_tracking_window.handle->template_id)
            cur_tracking_window.handle->set_template_id(template_id);
        size_t atlas_id = QString::fromStdString(cmd[2]).toInt();
        if(atlas_id >= atlas_file_name_list[template_id].size())
            return run->failed("invalid atlas id: " + cmd[2]);
        auto at = cur_tracking_window.handle->atlas_list[atlas_id];

        std::vector<std::vector<tipl::vector<3,short> > > points;
        std::vector<std::string> labels;

        if(cmd.size() > 3)
        {
            std::vector<size_t> label_list;
            for(auto each : QString::fromStdString(cmd[3]).split('&'))
            {
                int label = each.toInt();
                if(label < 0 || label >= at->get_list().size())
                    return run->failed("invalid label id: " + each.toStdString());
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
        return true;
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
        return true;
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
        return true;
    }

    if(cmd[0] == "delete_all_regions")
    {
        setRowCount(0);
        regions.clear();
        color_gen = 0;
        emit need_update();
        return true;
    }
    if(cmd[0] == "show_region_statistics" || cmd[0] == "show_t2r")
    {
        // cmd[1] : file name to save
        auto regions = get_checked_regions();
        if(regions.empty())
            return run->failed("please add parcellation regions");

        std::string result;

        if(cmd[0] == "show_t2r")
        {
            auto tracts = cur_tracking_window.tractWidget->get_checked_tracks();
            if(tracts.empty())
                return run->failed("please specify tract(s)");
            Parcellation p(cur_tracking_window.handle);
            p.load_from_regions(regions);
            result = p.get_t2r(tracts);
        }
        else
        {
            tipl::progress p("calculate region statistics",true);
            get_regions_statistics(cur_tracking_window.handle,regions,result);
        }

        if(!cmd[1].empty())
        {
            std::ofstream out(cmd[1]);
            if(!out)
                return run->failed("cannot write to " + cmd[1]);
            out << result;
        }
        else
        {
            if(cmd[0] == "show_t2r")
                cmd[1] = show_info_dialog("Tract-To-Region Connectome",result,cur_tracking_window.history.file_stem() + "_t2r.txt");
            else
                cmd[1] = show_info_dialog("Region Statistics",result,cur_tracking_window.history.file_stem() + "_stat.txt");
        }
        return true;
    }


    return run->not_processed();
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
        p.to(current_slice->to_slice);
    cur_tracking_window.move_slice_to(p);
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
void load_json_label(const char* filename,std::map<int,std::string>& label_map)
{
    std::ifstream in(filename);
    if(!in)
        return;
    std::string line,label;
    while(std::getline(in,line))
    {
        std::replace(line.begin(),line.end(),'\"',' ');
        std::replace(line.begin(),line.end(),'\t',' ');
        std::replace(line.begin(),line.end(),',',' ');
        line.erase(std::remove(line.begin(),line.end(),' '),line.end());
        if(line.find("arealabel") != std::string::npos)
        {
            label = line.substr(line.find(":")+1,std::string::npos);
            continue;
        }
        if(line.find("labelIndex") != std::string::npos)
        {
            line = line.substr(line.find(":")+1,std::string::npos);
            std::istringstream in3(line);
            int num = 0;
            in3 >> num;
            label_map[num] = label;
        }
    }
}

std::string get_label_file_name(const std::string& file_name);
void get_roi_label(QString file_name,std::map<int,std::string>& label_map,std::map<int,tipl::rgb>& label_color)
{
    label_map.clear();
    label_color.clear();
    QString base_name = QFileInfo(file_name).completeBaseName();
    if(base_name.endsWith(".nii"))
        base_name.chop(4);
    QString label_file = get_label_file_name(file_name.toStdString()).c_str();
    tipl::out() <<"looking for region label file " << label_file.toStdString() << std::endl;
    if(QFileInfo(label_file).exists())
    {
        load_nii_label(label_file.toStdString().c_str(),label_map);
        tipl::out() <<"label file loaded" << std::endl;
        return;
    }
    label_file = QFileInfo(file_name).absolutePath()+"/"+base_name+".json";
    if(QFileInfo(label_file).exists())
    {
        load_json_label(label_file.toStdString().c_str(),label_map);
        tipl::out() <<"json file loaded " << label_file.toStdString() << std::endl;
        return;
    }
    if(QFileInfo(file_name).fileName().contains("aparc") || QFileInfo(file_name).fileName().contains("aseg")) // FreeSurfer
    {
        tipl::out() <<"using freesurfer labels." << std::endl;
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
    tipl::out() <<"no label file found. Use default ROI numbering." << std::endl;
}
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<SliceModel*>& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::string& error_msg,
              bool is_mni)
{
    tipl::progress prog("opening file ",std::filesystem::path(file_name).stem().u8string().c_str());

    if(QFileInfo(file_name.c_str()).baseName().toLower().contains(".mni."))
    {
        tipl::out() << QFileInfo(file_name.c_str()).baseName().toStdString() <<
                     " has '.mni.' in the file name. It will be treated as mni space image" << std::endl;
        is_mni = true;
    }

    tipl::io::gz_nifti header;
    if (!header.load_from_file(file_name.c_str()))
    {
        error_msg = header.error_msg;
        return false;
    }
    bool is_4d = header.dim(4) > 1;
    tipl::image<3,unsigned int> from;
    std::string nifti_name = std::filesystem::path(file_name).stem().u8string();
    nifti_name = nifti_name.substr(0,nifti_name.find('.'));

    if(is_4d)
        from.resize(tipl::shape<3>(header.dim(1),header.dim(2),header.dim(3)));
    else
    {
        tipl::image<3> tmp;
        header.toLPS(tmp);
        if(std::filesystem::exists(get_label_file_name(file_name)) || tipl::is_label_image(tmp))
            from = tmp;
        else
        {
            tipl::out() << "The NIFTI file does not have a label text file. Using it as a mask.";
            from.resize(tmp.shape());
            for(size_t i = 0;i < from.size();++i)
                from[i] = (tmp[i] > 0.0f ? 1 : 0);
        }
    }

    tipl::vector<3> vs;
    tipl::matrix<4,4> trans_to_mni;
    header.get_image_transformation(trans_to_mni);
    header.get_voxel_size(vs);

    std::vector<unsigned short> value_list;
    std::vector<unsigned short> value_map(std::numeric_limits<unsigned short>::max()+1);

    if(is_4d)
    {
        value_list.resize(header.dim(4));
        for(unsigned int index = 0;index <value_list.size();++index)
        {
            value_list[index] = uint16_t(index);
            value_map[uint16_t(index)] = 0;
        }
    }
    else
    {
        unsigned short max_value = 0;
        for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
        {
            if(from[index.index()] >= value_map.size())
            {
                error_msg = "exceedingly large value found in the ROI file: ";
                error_msg += std::to_string(from[index.index()]);
                return false;
            }
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


    tipl::out() << nifti_name << (multiple_roi ? " loaded as multiple ROI file":" loaded as single ROI file") << std::endl;

    std::map<int,std::string> label_map;
    std::map<int,tipl::rgb> label_color;

    std::string des(header.get_descrip());
    if(multiple_roi)
        get_roi_label(file_name.c_str(),label_map,label_color);

    bool need_trans = false;
    tipl::matrix<4,4> to_diffusion_space = tipl::identity_matrix();

    tipl::out() << "FIB file size: " << handle->dim << " vs: " << handle->vs << (handle->is_mni ? " mni space": " not mni space") << std::endl;
    tipl::out() << nifti_name << " size: " << from.shape() << " vs: " << vs << (is_mni ? " mni space": " not mni space (if mni space, add '.mni.' in file name)") << std::endl;

    if(from.shape() != handle->dim)
    {
        tipl::out() << nifti_name << " has a different dimension from the FIB file. need transformation or warping." << std::endl;
        if(handle->is_mni)
        {
            if(!is_mni)
                tipl::out() << "assume " << nifti_name << " is in the mni space (likely wrong. need to check)." << std::endl;
            else
                tipl::out() << nifti_name << " is in the mni space." << std::endl;


            tipl::out() <<"applying " << nifti_name << "'s header srow matrix to align." << std::endl;
            to_diffusion_space = tipl::from_space(trans_to_mni).to(handle->trans_to_mni);
            need_trans = true;
            goto end;
        }
        else
        {
            if(is_mni)
            {
                tipl::out() << "warping " << nifti_name << " from the template space to the native space." << std::endl;
                if(!handle->mni2sub<tipl::interpolation::majority>(from,trans_to_mni))
                {
                    error_msg = handle->error_msg;
                    return false;
                }
                trans_to_mni = handle->trans_to_mni;
                goto end;
            }
            else
            for(unsigned int index = 0;index < transform_lookup.size();++index)
                if(from.shape() == transform_lookup[index]->dim)
                {
                    tipl::out() << "applying previous transformation." << std::endl;
                    tipl::out() << "tran_to_mni: " << std::endl;
                    tipl::out() << (trans_to_mni = transform_lookup[index]->trans_to_mni) << std::endl;
                    tipl::out() << "to_dif: " << std::endl;
                    tipl::out() << (to_diffusion_space = transform_lookup[index]->to_dif) << std::endl;
                    need_trans = true;
                    goto end;
                }
        }
        error_msg = "No strategy to align ";
        error_msg += nifti_name;
        error_msg += " with FIB. If ";
        error_msg += nifti_name;
        error_msg += " is in the MNI space, ";
        if(tipl::show_prog)
            error_msg += "open it using [Region][Open MNI Region]. If not, insert its reference T1W/T2W using [Slices][Insert T1WT2W] to guide the registration.";
        else
            error_msg += "specify mni in the file name (e.g. region_mni.nii.gz). If not, use --other_slices to load the reference T1W/T2W to guide the registration.";
        return false;
    }
    else
    {
        if(is_mni && !handle->is_mni)
            tipl::out() << "The '.mni.' in the filename is ignored, and " << nifti_name << " is treated as DWI regions because of identical image dimension. " << std::endl;
    }


    end:
    // single region ROI
    if(!multiple_roi)
    {
        regions.push_back(std::make_shared<ROIRegion>(handle));
        regions.back()->name = nifti_name;
        if(need_trans)
        {
            regions.back()->dim = from.shape();
            regions.back()->vs = vs;
            regions.back()->is_diffusion_space = false;
            regions.back()->to_diffusion_space = to_diffusion_space;
            regions.back()->trans_to_mni = trans_to_mni;
        }
        tipl::image<3,unsigned char> mask(from);
        regions.back()->load_region_from_buffer(mask);

        unsigned int color = 0x00FFFFFF;
        unsigned int type = default_id;

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
        regions.back()->region_render->color = color;
        regions.back()->regions_feature = uint8_t(type);
        return true;
    }

    std::vector<std::vector<tipl::vector<3,short> > > region_points(value_list.size());
    if(is_4d)
    {
        tipl::progress prog_("loading");
        for(size_t region_index = 0;prog(region_index,region_points.size());++region_index)
        {
            header.toLPS(from);
            for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
                if(from[index.index()])
                    region_points[region_index].push_back(index);
        }
    }
    else
    {
        for (tipl::pixel_index<3>index(from.shape());index < from.size();++index)
            if(from[index.index()])
                region_points[value_map[from[index.index()]]].push_back(index);
    }

    for(uint32_t i = 0;i < region_points.size();++i)
        {
            unsigned short value = value_list[i];
            regions.push_back(std::make_shared<ROIRegion>(handle));
            regions.back()->name = (label_map.find(value) == label_map.end() ?
                                    nifti_name + "_" + std::to_string(int(value)): label_map[value]);
            if(need_trans)
            {
                regions.back()->dim = from.shape();
                regions.back()->vs = vs;
                regions.back()->is_diffusion_space = false;
                regions.back()->to_diffusion_space = to_diffusion_space;
                regions.back()->trans_to_mni = trans_to_mni;
            }
            regions.back()->region_render->color = label_color.empty() ? 0x00FFFFFF : label_color[value].color;
            if(!region_points[i].empty())
                regions.back()->add_points(std::move(region_points[i]));
        }
    tipl::out() <<"a total of " << regions.size() << " regions are loaded." << std::endl;
    if(regions.empty())
    {
        error_msg = "empty region file";
        return false;
    }
    return true;
}

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

void RegionTableWidget::load_region_color(void)
{
    QString filename = QFileDialog::getOpenFileName(
                this,"Load region color",QFileInfo(cur_tracking_window.work_path).absolutePath()+"/region_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::ifstream in(filename.toStdString().c_str());
    if (!in)
        return;
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
}
void RegionTableWidget::save_region_color(void)
{
    QString filename = QFileDialog::getSaveFileName(
                this,"Save region color",QFileInfo(cur_tracking_window.work_path).absolutePath()+"/region_color.txt",
                "Color files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toStdString().c_str());
    if (!out)
        return;
    for(size_t index = 0;index < regions.size();++index)
    {
        tipl::rgb c(regions[index]->region_render->color);
        out << int(c[2]) << " " << int(c[1]) << " " << int(c[0]) << " " << int(c[3]) << std::endl;
    }
    QMessageBox::information(this,QApplication::applicationName(),"File saved");
}


void RegionTableWidget::check_row(size_t row,bool checked)
{
    if(checked)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
    else
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
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


void RegionTableWidget::save_region_info(void)
{
    if (regions.empty() || currentRow() >= regions.size())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save voxel information",QString(regions[currentRow()]->name.c_str()) + "_info.txt",
                           "Text files (*.txt)" );
    if (filename.isEmpty())
        return;

    std::ofstream out(filename.toStdString().c_str());
    out << "x\ty\tz";
    for(unsigned int index = 0;index < cur_tracking_window.handle->dir.num_fiber;++index)
            out << "\tdx" << index << "\tdy" << index << "\tdz" << index;

    for(const auto& each : cur_tracking_window.handle->get_index_list())
        out << "\t" << each;
    for(const auto& each : cur_tracking_window.handle->slices)
        if(!each->optional())
            each->get_image();
    out << std::endl;
    for(auto& point : regions[currentRow()]->to_space(cur_tracking_window.handle->dim))
    {
        std::vector<float> data;
        cur_tracking_window.handle->get_voxel_info2(point[0],point[1],point[2],data);
        cur_tracking_window.handle->get_voxel_information(point[0],point[1],point[2],data);
        std::copy(point.begin(),point.end(),std::ostream_iterator<float>(out,"\t"));
        std::copy(data.begin(),data.end(),std::ostream_iterator<float>(out,"\t"));
        out << std::endl;
    }
}



void RegionTableWidget::whole_brain(void)
{
    auto cur_slice = cur_tracking_window.current_slice;
    float threshold = cur_tracking_window.get_fa_threshold();
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
}

void RegionTableWidget::setROIs(ThreadData* data)
{
    for (unsigned int index = 0;index < regions.size();++index)
        if (!regions[index]->region.empty() && item(int(index),0)->checkState() == Qt::Checked
                && regions[index]->regions_feature != default_id)
            data->roi_mgr->setRegions(regions[index]->region,
                                      regions[index]->dim,
                                      regions[index]->to_diffusion_space,
                                      regions[index]->regions_feature,
                                      regions[index]->name.c_str());
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

bool RegionTableWidget::do_action(QString action,std::string& value)
{
    if(regions.empty() || currentRow() < 0)
        return false;
    tipl::progress prog(action.toStdString().c_str(),true);
    std::vector<int> rows_to_be_updated;
    size_t roi_index = currentRow();
    auto checked_regions = get_checked_regions();

    {
        if(action == "A-B" || action == "B-A" || action == "A*B" || action == "A<<B" || action == "A>>B")
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
            if(action == "A<<B" || action == "A>>B")
                A_labels.resize(base_dim);

            {
                tipl::out() << "processing regions";
                size_t prog_count = 0;
                tipl::adaptive_par_for(checked_regions.size(),[&](size_t r)
                {
                    prog(prog_count++,checked_regions.size());
                    if(r == 0)
                        return;
                    tipl::image<3,unsigned char> B;
                    checked_regions[r]->save_region_to_buffer(B,base_dim,base_to_dif);
                    if(action == "A-B")
                    {
                        tipl::masking(A,B);
                        return; // don't update B
                    }
                    if(action == "B-A")
                        tipl::masking(B,A);
                    if(action == "A*B")
                        for(size_t i = 0;i < B.size();++i)
                            B[i] = (A[i] & B[i]);
                    if(action == "A<<B" || action == "A>>B")
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
            if(action == "A-B")
                checked_regions[0]->load_region_from_buffer(A);


            if(action == "A>>B")
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
            if(action == "A<<B")
            {
                auto I = tipl::resample(cur_tracking_window.current_slice->get_source(),base_dim,
                                        tipl::from_space(cur_tracking_window.current_slice->to_dif).
                                        to(base_to_dif));
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
                    tipl::out() << "assign labels";
                    size_t prog_count = 0;
                    tipl::adaptive_par_for(need_fill_up.size(),[&](size_t i)
                    {
                        prog(prog_count++, need_fill_up.size());
                        tipl::pixel_index<3> index(need_fill_up[i], A.shape());
                        tipl::vector<3> pos(index);
                        std::vector<size_t> label_count(checked_regions.size()); // Store label frequency within radius
                        float search_radius = 2.0;

                        for (size_t r = 1; r < checked_regions.size(); ++r)
                            for (auto pos2 : checked_regions[r]->region)
                                if (std::abs(pos2[0] - pos[0]) < search_radius &&
                                    std::abs(pos2[1] - pos[1]) < search_radius &&
                                    std::abs(pos2[2] - pos[2]) < search_radius &&
                                    (pos2 - pos).length() <= search_radius)
                                    label_count[r]++;

                        size_t majority_label = 1;
                        int max_count = label_count[1];
                        for (size_t r = 2;r < label_count.size();++r)
                            if (label_count[r] > max_count)
                            {
                                majority_label = r;
                                max_count = label_count[r];
                            }

                        A_labels[index.index()] = majority_label;
                    });
                }
            }

            if(action == "A>>B" || action == "A<<B")
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


        std::vector<std::shared_ptr<ROIRegion> > region_to_be_processed;
        if(action == "threshold")
            region_to_be_processed.push_back(regions.back());
        else
        {
            if(cur_tracking_window.ui->actionModify_All->isChecked())
                region_to_be_processed = checked_regions;
            else
                region_to_be_processed.push_back(regions[size_t(roi_index)]);
        }

        tipl::adaptive_par_for(region_to_be_processed.size(),[&](unsigned int i)
        {
            region_to_be_processed[i]->perform(action.toStdString());
        });


        if(action == "dilation_by_voxel")
        {
            int threshold = 10;
            if(value.empty())
            {
                bool ok;
                threshold = QInputDialog::getInt(this,QApplication::applicationName(),"Voxel distance",10,1,100,1,&ok);
                if(!ok)
                    return false;
                value = std::to_string(threshold);
            }
            else
                threshold = QString::fromStdString(value).toInt();
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
            if(value.empty())
            {
                bool ok;
                threshold = float(QInputDialog::getDouble(this,
                                  QApplication::applicationName(),"Threshold (assign negative value to get low pass):",
                                  double(tipl::segmentation::otsu_threshold(I)),-m,m,4, &ok));
                if(!ok)
                    return false;
                value = std::to_string(threshold);
            }
            else
                threshold = QString::fromStdString(value).toFloat();
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
