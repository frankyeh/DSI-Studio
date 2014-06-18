#include <QStringListModel>
#include "atlasdialog.h"
#include "ui_atlasdialog.h"
#include "tracking_window.h"
#include "region/regiontablewidget.h"
#include "atlas.hpp"
extern std::vector<atlas> atlas_list;
AtlasDialog::AtlasDialog(tracking_window *parent) :
    QDialog(parent),
    ui(new Ui::AtlasDialog),
    cur_tracking_window(*parent)
{
    ui->setupUi(this);
    ui->region_list->setModel(new QStringListModel);
    ui->region_list->setSelectionModel(new QItemSelectionModel(ui->region_list->model()));
    for(int index = 0; index < atlas_list.size(); ++index)
        ui->atlasListBox->addItem(atlas_list[index].name.c_str());
    on_atlasListBox_currentIndexChanged(0);
}

AtlasDialog::~AtlasDialog()
{
    delete ui;
}
unsigned int AtlasDialog::index(void)
{
    return ui->atlasListBox->currentIndex();
}

void AtlasDialog::on_add_atlas_clicked()
{
    if(cur_tracking_window.handle->fib_data.trans_to_mni.empty())
        return;
    int atlas_index = ui->atlasListBox->currentIndex();
    QModelIndexList indexes = ui->region_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    for(unsigned int index = 0; index < indexes.size(); ++index)
    {
        std::vector<image::vector<3,short> > points;
        unsigned short label = indexes[index].row();
        image::geometry<3> geo = cur_tracking_window.slice.geometry;
        for (image::pixel_index<3>index; index.is_valid(geo); index.next(geo))
        {
            image::vector<3,float> mni((const unsigned int*)(index.begin()));
            cur_tracking_window.subject2mni(mni);
            if (!atlas_list[atlas_index].is_labeled_as(mni, label))
                continue;
            points.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
        }
        cur_tracking_window.regionWidget->add_region(
            ((QStringListModel*)ui->region_list->model())->stringList()[label],roi_id);
        cur_tracking_window.regionWidget->add_points(points,false);
    }
    emit need_update();
}

void AtlasDialog::on_atlasListBox_currentIndexChanged(int i)
{
    QStringList list;
    for (unsigned int index = 0; index < atlas_list[i].get_list().size(); ++index)
        list.push_back(atlas_list[i].get_list()[index].c_str());
    ((QStringListModel*)ui->region_list->model())->setStringList(list);
}

void AtlasDialog::on_pushButton_clicked()
{
    close();
}
