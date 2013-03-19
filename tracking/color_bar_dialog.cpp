#include <QSettings>
#include <QGraphicsTextItem>
#include "color_bar_dialog.hpp"
#include "ui_color_bar_dialog.h"
#include "tracking_window.h"
#include "opengl/glwidget.h"

color_bar_dialog::color_bar_dialog(QWidget *parent) :
    QDialog(parent),
    cur_tracking_window((tracking_window*)parent),
    ui(new Ui::color_bar_dialog)
{
    ui->setupUi(this);
    ODFModel* odf_model = (ODFModel*)cur_tracking_window->handle;
    FibData& fib_data = odf_model->fib_data;
    ui->tract_color_index->addItem((cur_tracking_window->is_dti) ? "fa":"qa");
    for (int index = fib_data.other_mapping_index; index < fib_data.view_item.size(); ++index)
        ui->tract_color_index->addItem(fib_data.view_item[index].name.c_str());
    ui->color_bar_view->setScene(&color_bar);
    ui->color_bar_style->setCurrentIndex(1);

    connect(ui->color_bar_style,SIGNAL(currentIndexChanged(int)),this,SLOT(update_color_map()));
    connect(ui->color_from,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->color_to,SIGNAL(clicked()),this,SLOT(update_color_map()));
    connect(ui->tract_color_max_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    connect(ui->tract_color_min_value,SIGNAL(valueChanged(double)),this,SLOT(update_color_map()));
    connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->glWidget,SLOT(makeTracts()));
    connect(ui->update_rendering,SIGNAL(clicked()),cur_tracking_window->glWidget,SLOT(updateGL()));
    on_tract_color_index_currentIndexChanged(0);

    QSettings settings;
    ui->color_from->setColor(settings.value("color_from",0x00FF1010).toInt());
    ui->color_to->setColor(settings.value("color_to",0x00FFFF10).toInt());
}

color_bar_dialog::~color_bar_dialog()
{
    QSettings settings;
    settings.setValue("color_from",ui->color_from->color().rgba());
    settings.setValue("color_to",ui->color_to->color().rgba());
    delete ui;
}


void color_bar_dialog::on_tract_color_index_currentIndexChanged(int index)
{
    unsigned int item_index = index ? index+cur_tracking_window->handle->fib_data.other_mapping_index-1:0;
    float max_value = cur_tracking_window->handle->fib_data.view_item[item_index].max_value;
    float min_value = cur_tracking_window->handle->fib_data.view_item[item_index].min_value;
    float scale2 = std::pow(10.0,std::floor(2.0-std::log10(max_value)));
    float scale1 = std::pow(10.0,std::floor(1.0-std::log10(max_value)));
    float decimal = std::floor(2.0-std::log10(max_value));
    if(decimal < 1.0)
        decimal = 1.0;
    ui->tract_color_max_value->setDecimals(decimal);
    ui->tract_color_max_value->setMaximum(std::ceil(max_value*scale1)/scale1);
    ui->tract_color_max_value->setMinimum(std::floor(min_value*scale1)/scale1);
    ui->tract_color_max_value->setSingleStep(std::ceil(max_value*scale1)/scale1/50);
    ui->tract_color_max_value->setValue(std::ceil(max_value*scale2)/scale1);

    ui->tract_color_min_value->setDecimals(decimal);
    ui->tract_color_min_value->setMaximum(std::ceil(max_value*scale1)/scale1);
    ui->tract_color_min_value->setMinimum(std::floor(min_value*scale1)/scale1);
    ui->tract_color_min_value->setSingleStep(std::ceil(max_value*scale1)/scale1/50);
    ui->tract_color_min_value->setValue(std::floor(min_value*scale2)/scale1);
    update_color_map();
}


unsigned char color_spectrum_value(unsigned char center, unsigned char value)
{
    unsigned char dif = center > value ? center-value:value-center;
    if(dif < 32)
        return 255;
    dif -= 32;
    if(dif >= 64)
        return 0;
    return 255-(dif << 2);
}


void color_bar_dialog::update_color_map(void)
{
    color_map.resize(256);
    bar.resize(image::geometry<2>(20,256));

    if(ui->color_bar_style->currentIndex() == 0)
    {
        image::rgb_color from_color = ui->color_from->color().rgba();
        image::rgb_color to_color = ui->color_to->color().rgba();
        for(unsigned int index = 0;index < color_map.size();++index)
        {
            float findex = (float)index/255.0;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color_map[index][2-rgb_index] =
                        (float)to_color[rgb_index]*findex/255.0+
                        (float)from_color[rgb_index]*(1.0-findex)/255.0;
        }



        for(unsigned int index = 1;index < 255;++index)
        {
            float findex = (float)index/256.0;
            image::rgb_color color;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[rgb_index] = (float)from_color[rgb_index]*findex+(float)to_color[rgb_index]*(1.0-findex);
            std::fill(bar.begin()+index*20+1,bar.begin()+(index+1)*20-1,color);
        }
    }

    if(ui->color_bar_style->currentIndex() == 1)
    {
        for(unsigned int index = 0;index < color_map.size();++index)
        {
            color_map[index][0] = (float)color_spectrum_value(128+64,index)/255.0;
            color_map[index][1] = (float)color_spectrum_value(128,index)/255.0;
            color_map[index][2] = (float)color_spectrum_value(64,index)/255.0;
        }
        for(unsigned int index = 1;index < 255;++index)
        {
            image::rgb_color color;
            color.r = color_spectrum_value(64,index);
            color.g = color_spectrum_value(128,index);
            color.b = color_spectrum_value(128+64,index);
            std::fill(bar.begin()+index*20+1,bar.begin()+(index+1)*20-1,color);
        }
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

float color_bar_dialog::get_color_max_value(void) const
{
    return ui->tract_color_max_value->value();
}
float color_bar_dialog::get_color_min_value(void) const
{
    return ui->tract_color_min_value->value();
}
unsigned int color_bar_dialog::get_tract_color_index(void) const
{
    return ui->tract_color_index->currentIndex();
}
