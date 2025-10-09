#include <QGraphicsTextItem>
#include <QMessageBox>
#include <QMouseEvent>
#include <QFileDialog>
#include <QClipboard>
#include "connectivity_matrix_dialog.h"
#include "ui_connectivity_matrix_dialog.h"
#include "region/regiontablewidget.h"
#include "tract/tracttablewidget.h"
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "libs/tracking/fib_data.hpp"
connectivity_matrix_dialog::connectivity_matrix_dialog(tracking_window *parent,QString method_) :
    QDialog(parent),method(method_),cur_tracking_window(parent),
    ui(new Ui::connectivity_matrix_dialog),data(cur_tracking_window->handle)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(&scene);

    for (const auto& each: cur_tracking_window->handle->get_index_list())
        ui->matrix_value->addItem(each.c_str());

    // atlas
    ui->region_list->addItem("ROIs");
    for(int index = 0;index < parent->handle->atlas_list.size();++index)
        ui->region_list->addItem(parent->handle->atlas_list[index]->name.c_str());
    if(ui->region_list->count() > 1)
        ui->region_list->setCurrentIndex(1);

    for(unsigned int index = 0;index < cur_tracking_window->regionWidget->regions.size();++index)
        if(cur_tracking_window->regionWidget->item(index,0)->checkState() == Qt::Checked)
        {
            ui->region_list->setCurrentIndex(0);
            break;
        }
    on_recalculate_clicked();

}

connectivity_matrix_dialog::~connectivity_matrix_dialog()
{
    delete ui;
}

bool connectivity_matrix_dialog::is_graphic_view(QObject *object) const
{
    return object == ui->graphicsView;
}
void connectivity_matrix_dialog::mouse_move(QMouseEvent *mouseEvent)
{
    QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    int x = std::floor(float(point.x()) / ui->zoom->value() - 0.5f);
    int y = std::floor(float(point.y()) / ui->zoom->value() - 0.5f);
    if(x >= 0 && y >= 0 && x < data.region_name.size() && y < data.region_name.size())
    {
        data.save_to_image(cm);
        // line x
        for(unsigned int x_pos = 0,pos = y*data.matrix_value.height();x_pos < data.matrix_value.width();++x_pos,++pos)
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        // line y
        for(unsigned int y_pos = 0,pos = x;y_pos < data.matrix_value.height();++y_pos,pos += data.matrix_value.width())
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        on_zoom_valueChanged(0);
        QGraphicsTextItem *x_text = scene.addText(data.region_name[x].c_str());
        QGraphicsTextItem *y_text = scene.addText(data.region_name[y].c_str());
        x_text->moveBy(point.x()-x_text->boundingRect().width()/2,-x_text->boundingRect().height());
        y_text->setRotation(270);
        y_text->moveBy(-y_text->boundingRect().height(),point.y()+y_text->boundingRect().width()/2);
        unsigned int index = x+y*data.matrix_value.width();
        if(index < data.matrix_value.size())
        {
            QGraphicsTextItem *value_text =
                    scene.addText(QString("%1:%2").
                                  arg(ui->matrix_value->currentText()).
                                  arg(data.matrix_value[index]));
            value_text->moveBy(0,-x_text->boundingRect().height()-value_text->boundingRect().height());
        }
    }

}

QString check_citation(QString str);
void connectivity_matrix_dialog::on_recalculate_clicked()
{
    if(cur_tracking_window->tractWidget->tract_models.size() == 0)
        return;
    tipl::progress prog("calculating connectivity matrix");
    cm.clear();
    if(ui->region_list->currentIndex() == 0)
        data.load_from_regions(cur_tracking_window->regionWidget->get_checked_regions(),"current regions");
    else
        data.load_from_atlas(ui->region_list->currentText().toStdString());

    TractModel tracks(cur_tracking_window->handle);
    for(int index = 0;index < cur_tracking_window->tractWidget->tract_models.size();++index)
        if(cur_tracking_window->tractWidget->item(index,0)->checkState() == Qt::Checked)
            tracks.add(*cur_tracking_window->tractWidget->tract_models[index]);
    if(!data.calculate(cur_tracking_window->handle,tracks,ui->matrix_value->currentText().toStdString().c_str(),
                       ui->end_only->currentIndex(),
                       ui->apply_threshold->isChecked() ? ui->network_threshold->value() : 0.0))
    {
        QMessageBox::critical(this,"ERROR",data.error_msg.c_str());
        return;
    }
    data.save_to_image(cm);
    on_zoom_valueChanged(0);
    QString out = QString("%1 %2 was used as the brain parcellation, and the connectivity matrix was calculated by using %3 of the connecting tracks.").
            arg(method).arg(ui->region_list->currentText()).arg(ui->matrix_value->currentText());
    out += " The connectivity matrix and graph theoretical analysis was conducted.";
    ui->report->setText(check_citation(out));

    std::string report;
    data.network_property(report);
    ui->network_measures->setText(report.c_str());
}


void connectivity_matrix_dialog::on_zoom_valueChanged(double)
{
    if(cm.empty())
        return;
    view_image << cm;
    view_image = view_image.scaled(cm.width()*ui->zoom->value(),cm.height()*ui->zoom->value());
    scene << view_image;
}


void connectivity_matrix_dialog::on_save_matrix_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,"Save as",cur_tracking_window->tractWidget->item(cur_tracking_window->tractWidget->currentRow(),0)->text() + "_" +
                ui->region_list->currentText() + ".mat",
                "MAT File (*.mat);;Image Files (*.png *.tif *.bmp)");
    if(filename.isEmpty())
        return;
    if(QFileInfo(filename).suffix().toLower() == "mat")
    {
        data.save_to_file(filename.toStdString().c_str());
    }
    else
    {
        data.save_to_image(cm);
        QImage qimage((unsigned char*)&*cm.begin(),cm.width(),cm.height(),QImage::Format_RGB32);
        qimage.save(filename);
    }
}

void connectivity_matrix_dialog::on_save_network_property_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,"Save as",cur_tracking_window->tractWidget->item(cur_tracking_window->tractWidget->currentRow(),0)->text() + "_" +
                ui->region_list->currentText() + "_net.txt",
                "Text File (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::string report;
    data.network_property(report);
    std::ofstream out(filename.toStdString().c_str());
    out << report;
}

void connectivity_matrix_dialog::on_save_connectogram_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,"Save as",cur_tracking_window->tractWidget->item(cur_tracking_window->tractWidget->currentRow(),0)->text() + "_" +
                ui->region_list->currentText() + "_cg.txt",
                "Text File (*.txt)");
    if(filename.isEmpty())
        return;
    data.save_to_connectogram(filename.toStdString().c_str());
}

void connectivity_matrix_dialog::on_copy_to_clipboard_clicked()
{
    std::string text;
    data.save_to_text(text);
    QApplication::clipboard()->setText(text.c_str());
    QMessageBox::information(this,QApplication::applicationName(),"Results copied to clipboard");
}
