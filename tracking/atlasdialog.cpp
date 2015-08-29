#include <QStringListModel>
#include <QMessageBox>
#include "atlasdialog.h"
#include "ui_atlasdialog.h"
#include "region/regiontablewidget.h"
#include "atlas.hpp"
extern std::vector<atlas> atlas_list;
AtlasDialog::AtlasDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::AtlasDialog)
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

void AtlasDialog::on_add_atlas_clicked()
{
    atlas_index = ui->atlasListBox->currentIndex();
    QModelIndexList indexes = ui->region_list->selectionModel()->selectedRows();
    if(!indexes.count())
    {
        reject();
        return;
    }
    for(unsigned int index = 0; index < indexes.size(); ++index)
    {
        roi_list.push_back(indexes[index].row());
        roi_name.push_back(((QStringListModel*)ui->region_list->model())->stringList()[indexes[index].row()].toStdString());
    }
    accept();
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
    reject();
}
