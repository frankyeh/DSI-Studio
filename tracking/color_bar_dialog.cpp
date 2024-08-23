#include <QSettings>
#include <QGraphicsTextItem>
#include <QDir>
#include "color_bar_dialog.hpp"
#include "ui_color_bar_dialog.h"
#include "tracking_window.h"
#include "opengl/glwidget.h"
#include "libs/tracking/fib_data.hpp"

color_bar_dialog::color_bar_dialog(QWidget *parent) :
    QDialog(parent),bar(20,256),
    cur_tracking_window(static_cast<tracking_window*>(parent)),
    ui(new Ui::color_bar_dialog)
{
    ui->setupUi(this);

    if(cur_tracking_window)
    {
        connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->tractWidget,SLOT(need_update_all()));
        connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->glWidget,SLOT(update()));
    }
    else
    {
        ui->tract_color_index->hide();
        ui->update_rendering->hide();
        ui->index_label->hide();
    }
    ui->color_bar_view->setScene(&color_bar);

    // populate color bar
    {
        QStringList name_list = QDir(QCoreApplication::applicationDirPath()+"/color_map/").
                                entryList(QStringList("*.txt"),QDir::Files|QDir::NoSymLinks);
        for(int i = 0;i < name_list.size();++i)
            ui->color_bar_style->addItem(QFileInfo(name_list[i]).baseName());
    }
    ui->color_bar_style->setCurrentText("jet");
    connect(ui->color_bar_style,SIGNAL(currentIndexChanged(int)),this,SLOT(update_color_map()));
    connect(ui->color_from,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->color_to,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->tract_color_max_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    connect(ui->tract_color_min_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    on_tract_color_index_currentIndexChanged(0);

    QSettings settings;
    ui->color_from->setColor(uint32_t(settings.value("color_from",0xFFFF1010).toInt()));
    ui->color_to->setColor(uint32_t(settings.value("color_to",0xFFFFFF10).toInt()));
}


color_bar_dialog::~color_bar_dialog()
{
    QSettings settings;
    settings.setValue("color_from",ui->color_from->color().rgb());
    settings.setValue("color_to",ui->color_to->color().rgb());
    delete ui;
}

void color_bar_dialog::update_slice_indices(void)
{
    if(cur_tracking_window)
    {
        ui->tract_color_index->clear();
        for (const auto& each : cur_tracking_window->handle->get_index_list())
            ui->tract_color_index->addItem(each.c_str());
    }
}

void color_bar_dialog::set_value(double min_value,double max_value)
{
    double decimal = std::floor(2.0-std::log10(max_value));
    double scale = std::pow(10.0,decimal);
    if(decimal < 1.0)
        decimal = 1.0;
    max_value = std::ceil(max_value*scale)/scale;
    min_value = std::floor(min_value*scale)/scale;

    ui->tract_color_max_value->setDecimals(int(decimal));
    ui->tract_color_max_value->setMaximum(max_value);
    ui->tract_color_max_value->setMinimum(min_value);
    ui->tract_color_max_value->setSingleStep((max_value-min_value)/50);
    ui->tract_color_max_value->setValue(max_value);

    ui->tract_color_min_value->setDecimals(int(decimal));
    ui->tract_color_min_value->setMaximum(max_value);
    ui->tract_color_min_value->setMinimum(min_value);
    ui->tract_color_min_value->setSingleStep((max_value-min_value)/50);
    ui->tract_color_min_value->setValue(min_value);
    update_color_map();
}

void color_bar_dialog::on_tract_color_index_currentIndexChanged(int index)
{
    if(!cur_tracking_window || index < 0)
        return;
    std::string index_name = ui->tract_color_index->currentText().toStdString();
    if(index_name.empty())
        return;
    size_t item_index = cur_tracking_window->handle->get_name_index(index_name);
    if(item_index == cur_tracking_window->handle->slices.size())
        return;
    cur_tracking_window->handle->slices[item_index]->get_minmax();
    set_value(double(cur_tracking_window->handle->slices[item_index]->min_value),
              double(cur_tracking_window->handle->slices[item_index]->max_value));
}




void color_bar_dialog::update_color_map(void)
{
    color_r = ui->tract_color_max_value->value()-ui->tract_color_min_value->value();
    if(color_r + 1.0 == 1.0)
        color_r = 1.0;
    color_min = ui->tract_color_min_value->value();


    if(ui->color_bar_style->currentIndex() == 0)
    {
        ui->color_from->show();
        ui->color_to->show();
        tipl::rgb from_color = ui->color_from->color().rgb();
        tipl::rgb to_color = ui->color_to->color().rgb();
        bar.two_color(from_color,to_color);
        std::swap(from_color.r,from_color.b);
        std::swap(to_color.r,to_color.b);
        color_map.two_color(from_color,to_color);
        color_map_rgb.two_color(from_color,to_color);
    }
    else
    {
        ui->color_from->hide();
        ui->color_to->hide();
        QString filename = QCoreApplication::applicationDirPath()+"/color_map/"+ui->color_bar_style->currentText()+".txt";
        color_map.load_from_file(filename.toStdString().c_str());
        color_map_rgb.load_from_file(filename.toStdString().c_str());
        bar.load_from_file(filename.toStdString().c_str());
    }

    color_bar.clear();
    QGraphicsTextItem *max_text = color_bar.addText(QString::number(ui->tract_color_max_value->value()));
    QGraphicsTextItem *min_text = color_bar.addText(QString::number(ui->tract_color_min_value->value()));
    QGraphicsPixmapItem *map = color_bar.addPixmap(tipl::qt::image2pixelmap(QImage() << *(tipl::color_image*)&bar));
    max_text->moveBy(10,-128-10);
    min_text->moveBy(10,128-10);
    map->moveBy(-10,-128);
    ui->color_bar_view->show();

}

QString color_bar_dialog::get_tract_color_name(void) const
{
    return ui->tract_color_index->currentText();
}
