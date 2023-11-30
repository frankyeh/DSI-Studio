#include <QStringListModel>
#include <QInputDialog>
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
    {
        ui->add_all_regions->setVisible(false);
        ui->merge_and_add->setVisible(false);
    }
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


void AtlasDialog::on_merge_and_add_clicked()
{
    QModelIndexList indexes = ui->region_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    atlas_index = uint32_t(ui->atlasListBox->currentIndex());
    atlas_name = ui->atlasListBox->currentText().toStdString();
    if(!handle->atlas_list[atlas_index]->load_from_file())
    {
        QMessageBox::critical(this,"ERROR",handle->atlas_list[atlas_index]->error_msg.c_str());
        return;
    }
    tipl::progress prog("adding regions");

    std::vector<unsigned int> roi_list;
    for(unsigned int index = 0;index < indexes.size(); ++index)
        roi_list.push_back(uint32_t(indexes[index].row()));

    auto lcp = [](const QString& str1, const QString& str2){
        int length = std::min(str1.length(), str2.length());
        for (int i = 0; i < length; ++i)
            if (str1[i] != str2[i])
                return str1.left(i);
        return str1.left(length);
    };

    auto name = lcp(handle->atlas_list[atlas_index]->get_list()[roi_list.front()].c_str(),
                    handle->atlas_list[atlas_index]->get_list()[roi_list.back()].c_str());
    if(name.isEmpty())
        name = handle->atlas_list[atlas_index]->get_list()[roi_list.front()].c_str();

    auto* w = dynamic_cast<tracking_window*>(parent());
    w->regionWidget->add_merged_regions_from_atlas(handle->atlas_list[atlas_index],name,roi_list);

    w->glWidget->update();
    w->slice_need_update = true;
    w->raise();

    ui->region_list->clearSelection();
    ui->search_atlas->setText("");
    ui->search_atlas->setFocus();
}

void AtlasDialog::on_add_atlas_clicked()
{
    QModelIndexList indexes = ui->region_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    atlas_index = uint32_t(ui->atlasListBox->currentIndex());
    atlas_name = ui->atlasListBox->currentText().toStdString();
    auto model = dynamic_cast<QStringListModel*>(ui->region_list->model());
    auto* w = dynamic_cast<tracking_window*>(parent());
    if(!w) // for connectometry atlas
    {
        for(int index = 0; index < indexes.size(); ++index)
        {
            roi_list.push_back(uint32_t(indexes[index].row()));
            roi_name.push_back(model->stringList()[indexes[index].row()].toStdString());
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

void AtlasDialog::on_select_clicked()
{
    bool ok;
    QString sel = QInputDialog::getText(this,"DSI Studio","Please specify the select text",QLineEdit::Normal,"",&ok);
    if(!ok)
        return;
    ui->region_list->clearSelection();
    auto model = dynamic_cast<QStringListModel*>(ui->region_list->model());
    for (int row = 0; row < model->rowCount(); ++row)
        if (model->data(model->index(row, 0), Qt::DisplayRole).toString().startsWith(sel))
            ui->region_list->selectionModel()->select(model->index(row, 0), QItemSelectionModel::Select);
}


