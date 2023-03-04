#include <QStringListModel>
#include <QMessageBox>
#include "atlasdialog.h"
#include "ui_atlasdialog.h"
#include "fib_data.hpp"
#include "tracking_window.h"
#include "region/regiontablewidget.h"
#include "opengl/glwidget.h"
AtlasDialog::AtlasDialog(QWidget *parent,std::shared_ptr<fib_data> handle_) :
    QDialog(parent),
    handle(handle_),
    ui(new Ui::AtlasDialog)
{
    ui->setupUi(this);
    auto* w = dynamic_cast<tracking_window*>(parent);
    if(!w)
        ui->add_all_regions->setVisible(false);
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
    atlas_index = uint32_t(ui->atlasListBox->currentIndex());
    atlas_name = ui->atlasListBox->currentText().toStdString();
    QModelIndexList indexes = ui->region_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;

    auto* w = dynamic_cast<tracking_window*>(parent());
    if(!w) // for connectometry atlas
    {
        for(int index = 0; index < indexes.size(); ++index)
        {
            roi_list.push_back(uint32_t(indexes[index].row()));
            roi_name.push_back(dynamic_cast<QStringListModel*>(ui->region_list->model())->stringList()[indexes[index].row()].toStdString());
        }
        accept();
        return;
    }


    if(!handle->atlas_list[atlas_index]->load_from_file())
    {
        QMessageBox::critical(this,"ERROR",handle->atlas_list[atlas_index]->error_msg.c_str());
        return;
    }
    tipl::progress prog("adding regions");
    w->regionWidget->begin_update();
    if(indexes.count() == ui->region_list->model()->rowCount()) // select all
        w->regionWidget->add_all_regions_from_atlas(handle->atlas_list[atlas_index]);
    else
    {
        for(unsigned int index = 0;prog(index,indexes.size()); ++index)
            w->regionWidget->add_region_from_atlas(handle->atlas_list[atlas_index],uint32_t(indexes[int(index)].row()));
    }
    w->regionWidget->end_update();
    w->glWidget->update();
    w->slice_need_update = true;
    w->raise();

    ui->region_list->clearSelection();
    ui->search_atlas->setText("");
    ui->search_atlas->setFocus();
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

void AtlasDialog::on_search_atlas_textChanged(const QString &)
{
    std::string name_value = ui->search_atlas->text().toStdString();
    size_t pos = name_value.find_last_of(':');
    if(pos == std::string::npos)
        return;
    std::string atlas_name = name_value.substr(pos+1);
    std::string region_name = name_value.substr(0,pos);
    for(size_t i = 0;i < handle->atlas_list.size();++i)
        if(atlas_name == handle->atlas_list[i]->name)
        {
            ui->atlasListBox->setCurrentIndex(int(i));
            for(size_t j = 0;j < handle->atlas_list[i]->get_list().size();++j)
            if(handle->atlas_list[i]->get_list()[j] == region_name)
            {
                ui->region_list->setCurrentIndex(ui->region_list->model()->index(int(j),0));
                ui->search_atlas->setText("");
                return;
            }
        }
}

void AtlasDialog::on_add_all_regions_clicked()
{
    ui->region_list->selectAll();
    ui->region_list->setFocus();
    ui->region_list->raise();
}
