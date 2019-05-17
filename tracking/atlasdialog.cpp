#include <QStringListModel>
#include <QMessageBox>
#include "atlasdialog.h"
#include "ui_atlasdialog.h"
#include "region/regiontablewidget.h"
#include "fib_data.hpp"
AtlasDialog::AtlasDialog(QWidget *parent,std::shared_ptr<fib_data> handle_) :
    QDialog(parent),
    ui(new Ui::AtlasDialog),
    handle(handle_)
{
    ui->setupUi(this);
    ui->region_list->setModel(new QStringListModel);
    ui->region_list->setSelectionModel(new QItemSelectionModel(ui->region_list->model()));
    for(int index = 0; index < handle->atlas_list.size(); ++index)
        ui->atlasListBox->addItem(handle->atlas_list[index]->name.c_str());
    on_atlasListBox_currentIndexChanged(0);

    QStringList items;
    for(int i = 0;i < handle->atlas_list.size();++i)
    {
        const std::vector<std::string>& label = handle->atlas_list[i]->get_list();
        for(auto str : label)
        items << QString(str.c_str()) + ":" + handle->atlas_list[i]->name.c_str();
    }
    ui->search_atlas->setList(items);
}

AtlasDialog::~AtlasDialog()
{
    delete ui;
}

void AtlasDialog::on_add_atlas_clicked()
{
    atlas_index = ui->atlasListBox->currentIndex();
    atlas_name = ui->atlasListBox->currentText().toStdString();
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
    for (unsigned int index = 0; index < handle->atlas_list[i]->get_list().size(); ++index)
        list.push_back(handle->atlas_list[i]->get_list()[index].c_str());
    ((QStringListModel*)ui->region_list->model())->setStringList(list);
}

void AtlasDialog::on_pushButton_clicked()
{
    reject();
}

void AtlasDialog::on_search_atlas_textChanged(const QString &arg1)
{
    QStringList name_value = ui->search_atlas->text().split(":");
    if(name_value.size() != 2)
        return;
    for(int i = 0;i < handle->atlas_list.size();++i)
        if(name_value[1].toStdString() == handle->atlas_list[i]->name)
        {
            ui->atlasListBox->setCurrentIndex(i);
            for(int j = 0;j < handle->atlas_list[i]->get_list().size();++j)
            if(handle->atlas_list[i]->get_list()[j] == name_value[0].toStdString())
            {
                ui->region_list->setCurrentIndex(ui->region_list->model()->index(j,0));
                ui->search_atlas->setText("");
                return;
            }
        }
}
