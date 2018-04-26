#include <QSettings>
#include <QGraphicsTextItem>
#include "color_bar_dialog.hpp"
#include "ui_color_bar_dialog.h"
#include "tracking_window.h"
#include "opengl/glwidget.h"
#include "libs/tracking/fib_data.hpp"

color_bar_dialog::color_bar_dialog(QWidget *parent) :
    QDialog(parent),
    cur_tracking_window((tracking_window*)parent),bar(20,256),
    ui(new Ui::color_bar_dialog)
{
    ui->setupUi(this);
    std::vector<std::string> index_list;

    if(cur_tracking_window)
    {
        cur_tracking_window->handle->get_index_list(index_list);
        for (unsigned int index = 0; index < index_list.size(); ++index)
            ui->tract_color_index->addItem(index_list[index].c_str());
        connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->glWidget,SLOT(makeTracts()));
        connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->glWidget,SLOT(updateGL()));
    }
    else
    {
        ui->tract_color_index->hide();
        ui->update_rendering->hide();
        ui->index_label->hide();
    }
    ui->color_bar_view->setScene(&color_bar);
    ui->color_bar_style->setCurrentIndex(1);

    connect(ui->color_bar_style,SIGNAL(currentIndexChanged(int)),this,SLOT(update_color_map()));
    connect(ui->color_from,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->color_to,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->tract_color_max_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    connect(ui->tract_color_min_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    on_tract_color_index_currentIndexChanged(0);

    QSettings settings;
    ui->color_from->setColor(settings.value("color_from",0x00FF1010).toInt());
    ui->color_to->setColor(settings.value("color_to",0x00FFFF10).toInt());
}

color_bar_dialog::~color_bar_dialog()
{
    QSettings settings;
    settings.setValue("color_from",ui->color_from->color().rgb());
    settings.setValue("color_to",ui->color_to->color().rgb());
    delete ui;
}

void color_bar_dialog::set_value(float min_value,float max_value)
{
    float decimal = std::floor(2.0-std::log10(max_value));
    float scale = std::pow(10.0,(double)decimal);
    if(decimal < 1.0)
        decimal = 1.0;
    max_value = std::ceil(max_value*scale)/scale;
    min_value = std::floor(min_value*scale)/scale;

    ui->tract_color_max_value->setDecimals(decimal);
    ui->tract_color_max_value->setMaximum(max_value);
    ui->tract_color_max_value->setMinimum(min_value);
    ui->tract_color_max_value->setSingleStep((max_value-min_value)/50);
    ui->tract_color_max_value->setValue(max_value);

    ui->tract_color_min_value->setDecimals(decimal);
    ui->tract_color_min_value->setMaximum(max_value);
    ui->tract_color_min_value->setMinimum(min_value);
    ui->tract_color_min_value->setSingleStep((max_value-min_value)/50);
    ui->tract_color_min_value->setValue(min_value);
    update_color_map();
}

void color_bar_dialog::on_tract_color_index_currentIndexChanged(int)
{
    if(!cur_tracking_window)
        return;
    unsigned int item_index = cur_tracking_window->handle->get_name_index(ui->tract_color_index->currentText().toStdString());
    float max_value = cur_tracking_window->handle->view_item[item_index].max_value;
    float min_value = cur_tracking_window->handle->view_item[item_index].min_value;
    set_value(min_value,max_value);
}





void color_bar_dialog::update_color_map(void)
{
    color_r = ui->tract_color_max_value->value()-ui->tract_color_min_value->value();
    if(color_r + 1.0 == 1.0)
        color_r = 1.0;
    color_min = ui->tract_color_min_value->value();


    if(ui->color_bar_style->currentIndex() == 0)
    {
        tipl::rgb from_color = ui->color_from->color().rgb();
        tipl::rgb to_color = ui->color_to->color().rgb();
        bar.two_color(from_color,to_color);
        std::swap(from_color.r,from_color.b);
        std::swap(to_color.r,to_color.b);
        color_map.two_color(from_color,to_color);
        color_map_rgb.two_color(from_color,to_color);
    }

    if(ui->color_bar_style->currentIndex() == 1)
    {
        color_map.spectrum();
        color_map_rgb.spectrum();
        bar.spectrum();
    }

    color_bar.clear();
    QGraphicsTextItem *max_text = color_bar.addText(QString::number(ui->tract_color_max_value->value()));
    QGraphicsTextItem *min_text = color_bar.addText(QString::number(ui->tract_color_min_value->value()));
    QGraphicsPixmapItem *map = color_bar.addPixmap(QPixmap::fromImage(
            QImage((unsigned char*)&*bar.begin(),bar.width(),bar.height(),QImage::Format_RGB32)));
    max_text->moveBy(10,-128-10);
    min_text->moveBy(10,128-10);
    map->moveBy(-10,-128);
    ui->color_bar_view->show();

}

QString color_bar_dialog::get_tract_color_name(void) const
{
    return ui->tract_color_index->currentText();
}
