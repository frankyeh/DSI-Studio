#include <QSplitter>
#include "reconstruction_window.h"
#include "ui_reconstruction_window.h"
#include "dsi_interface_static_link.h"
#include "ml/ml.hpp"
#include "image/image.hpp"
#include "mainwindow.h"
#include <QImage>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QSettings>
#include "prog_interface_static_link.h"
#include "tracking/region/Regions.h"
#include "libs/dsi/image_model.hpp"


void reconstruction_window::load_src(int index)
{
    begin_prog("load src");
    check_prog(index,filenames.size());
    handle = (ImageModel*)init_reconstruction(filenames[index].toLocal8Bit().begin());
    if (!handle)
    {
        QMessageBox::information(this,"error","Cannot load the .src file, please check the memory sufficiency",0);
        throw;
    }
    dim[0] = handle->voxel.matrix_width;
    dim[1] = handle->voxel.matrix_height;
    dim[2] = handle->voxel.slice_number;
    image.resize(dim);
    std::copy(handle->mask.begin(),handle->mask.end(),image.begin());
}

reconstruction_window::reconstruction_window(QStringList filenames_,QWidget *parent) :
        QMainWindow(parent),filenames(filenames_),
        ui(new Ui::reconstruction_window)
{
    load_src(0);
    image::segmentation::otsu(image,handle->mask);
    image::morphology::erosion(handle->mask);
    image::morphology::smoothing(handle->mask);
    image::morphology::smoothing(handle->mask);
    image::morphology::smoothing(handle->mask);
    image::morphology::defragment(handle->mask);
    image::morphology::dilation(handle->mask);
    image::morphology::dilation(handle->mask);
    image::morphology::smoothing(handle->mask);
    image::morphology::smoothing(handle->mask);
    image::morphology::smoothing(handle->mask);
    max_source_value = 0.0;

    ui->setupUi(this);
    QSplitter *splitter = new QSplitter(ui->source_page);
    ui->source_page_layout->addWidget(splitter);
    splitter->addWidget(ui->b_table);
    splitter->addWidget(ui->source_widget);
    ui->toolBox->setCurrentIndex(1);
    ui->graphicsView->setScene(&scene);
    ui->view_source->setScene(&source);
    ui->b_table->setColumnWidth(0,40);
    ui->b_table->setColumnWidth(1,60);
    ui->b_table->setColumnWidth(2,60);
    ui->b_table->setColumnWidth(3,60);
    ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");

    ui->SlicePos->setRange(0,dim[2]-1);
    ui->SlicePos->setValue((dim[2]-1) >> 1);
    ui->z_pos->setRange(0,dim[2]-1);
    ui->z_pos->setValue((dim[2]-1) >> 1);

    ui->x->setMaximum(dim[0]-1);
    ui->y->setMaximum(dim[1]-1);
    ui->z->setMaximum(dim[2]-1);

    absolute_path = QFileInfo(filenames[0]).absolutePath();
    source_ratio = std::max(1.0,500/(double)dim.height());

    load_b_table();

    if(filenames.size() == 1)
        ui->QSDRT->hide();
    switch(settings.value("rec_method_id",4).toInt())
    {
    case 0:
        ui->DSI->setChecked(true);
        on_DSI_toggled(true);
        break;
    case 1:
        ui->DTI->setChecked(true);
        on_DTI_toggled(true);
        break;
    case 3:
        ui->QBI->setChecked(true);
        on_QBI_toggled(true);
        break;
    case 7:
        ui->QDif->setChecked(true);
        on_QDif_toggled(true);
        break;
    case 8:
        if(filenames.size() != 1)
        {
            ui->QSDRT->setChecked(true);
            on_QSDRT_toggled(true);
            break;
        }
    default:
        ui->GQI->setChecked(true);
        on_GQI_toggled(true);
        break;
    }

    ui->ThreadCount->setCurrentIndex(settings.value("rec_thread_count",0).toInt());
    ui->RecordODF->setChecked(settings.value("rec_record_odf",0).toInt());
    ui->ODFSharpening->setCurrentIndex(settings.value("rec_odf_sharpening",0).toInt());
    ui->Decomposition->setCurrentIndex(settings.value("rec_odf_decomposition",0).toInt());
    ui->decom_m->setValue(settings.value("rec_decom_m",10).toInt());
    ui->HalfSphere->setChecked(settings.value("rec_half_sphere",0).toInt());
    ui->NumOfFibers->setValue(settings.value("rec_num_fiber",5).toInt());
    ui->ODFDef->setCurrentIndex(settings.value("rec_gqi_def",0).toInt());

    ui->diffusion_sampling->setValue(settings.value("rec_gqi_sampling",1.25).toDouble());
    ui->regularization_param->setValue(settings.value("rec_qbi_reg",0.006).toDouble());
    ui->hamming_filter->setValue(settings.value("rec_hamming_filter",17).toDouble());
    ui->SharpeningParam->setValue(settings.value("rec_deconvolution_param",3.0).toDouble());

    ui->mni_resolution->setValue(settings.value("rec_mni_resolution",2.0).toDouble());


    connect(ui->z_pos,SIGNAL(sliderMoved(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->contrast,SIGNAL(sliderMoved(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->brightness,SIGNAL(sliderMoved(int)),this,SLOT(on_b_table_itemSelectionChanged()));
}
void reconstruction_window::load_b_table(void)
{
    ui->b_table->clear();
    ui->b_table->setRowCount(handle->voxel.bvalues.size());
    for(unsigned int index = 0;index < handle->voxel.bvalues.size();++index)
    {
        ui->b_table->setItem(index,0, new QTableWidgetItem(QString::number(handle->voxel.bvalues[index])));
        ui->b_table->setItem(index,1, new QTableWidgetItem(QString::number(handle->voxel.bvectors[index][0])));
        ui->b_table->setItem(index,2, new QTableWidgetItem(QString::number(handle->voxel.bvectors[index][1])));
        ui->b_table->setItem(index,3, new QTableWidgetItem(QString::number(handle->voxel.bvectors[index][2])));
    }
    ui->b_table->selectRow(0);
}
void reconstruction_window::on_b_table_itemSelectionChanged()
{
    image::basic_image<float,2> tmp(image::geometry<2>(dim[0],dim[1]));
    unsigned int offset = ui->z_pos->value()*tmp.size();
    unsigned int b_index = ui->b_table->currentRow();
    std::copy(handle->dwi_data[b_index] + offset,
              handle->dwi_data[b_index] + offset + tmp.size(),tmp.begin());
    max_source_value = std::max<float>(max_source_value,*std::max_element(tmp.begin(),tmp.end()));
    if(max_source_value + 1.0 != 1.0)
        image::divide_constant(tmp.begin(),tmp.end(),max_source_value/255.0);

    float mean_value = image::mean(tmp.begin(),tmp.end());
    image::minus_constant(tmp.begin(),tmp.end(),mean_value);
    image::multiply_constant(tmp.begin(),tmp.end(),ui->contrast->value());
    image::add_constant(tmp.begin(),tmp.end(),mean_value+ui->brightness->value()*25.5);

    image::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);


    buffer_source.resize(image::geometry<2>(dim[0],dim[1]));
    std::copy(tmp.begin(),tmp.end(),buffer_source.begin());

    source.setSceneRect(0, 0, dim.width()*source_ratio,dim.height()*source_ratio);
    source_image = QImage((unsigned char*)&*buffer_source.begin(),dim.width(),dim.height(),QImage::Format_RGB32).
                    scaled(dim.width()*source_ratio,dim.height()*source_ratio);
    source.clear();
    source.addRect(0, 0, dim.width()*source_ratio,dim.height()*source_ratio,QPen(),source_image);
}


void reconstruction_window::resizeEvent ( QResizeEvent * event )
{
    QMainWindow::resizeEvent(event);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}
void reconstruction_window::showEvent ( QShowEvent * event )
{
    QMainWindow::showEvent(event);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);

}

reconstruction_window::~reconstruction_window()
{
    delete ui;
    ui = 0;
    if(handle)
    free_reconstruction((ImageModel*)handle);
    handle = 0;
}

void reconstruction_window::doReconstruction(unsigned char method_id,bool prompt)
{
    if(!handle)
        return;
    if (*std::max_element(handle->mask.begin(),handle->mask.end()) == 0)
    {
        QMessageBox::information(this,"error","Please select mask for reconstruction",0);
        return;
    }

    if (ui->ODFSharpening->currentIndex() > 0 && method_id != 1) // not DTI
    {
        params[2] = ui->SharpeningParam->value();
        settings.setValue("rec_deconvolution_param",params[2]);
    }
    if (ui->Decomposition->currentIndex() > 0 && method_id != 1) // not DTI
    {
        params[3] = ui->decompose_fraction->value();
        params[4] = ui->decom_m->value();
        settings.setValue("rec_decomposition_param",params[3]);
        settings.setValue("rec_decom_m",params[4]);
    }



    settings.setValue("rec_method_id",method_id);
    settings.setValue("rec_thread_count",ui->ThreadCount->currentIndex());
    settings.setValue("rec_record_odf",ui->RecordODF->isChecked() ? 1 : 0);
    settings.setValue("rec_odf_sharpening",ui->ODFSharpening->currentIndex());
    settings.setValue("rec_odf_decomposition",ui->Decomposition->currentIndex());
    settings.setValue("rec_half_sphere",ui->HalfSphere->isChecked() ? 1 : 0);
    settings.setValue("rec_num_fiber",ui->NumOfFibers->value());
    settings.setValue("rec_gqi_def",ui->ODFDef->currentIndex());


    begin_prog("reconstructing");
    int odf_order[4] = {4, 5, 6, 8};
    handle->thread_count = ui->ThreadCount->currentIndex() + 1;
    handle->voxel.ti.init(odf_order[ui->ODFDim->currentIndex()]);
    handle->voxel.need_odf = ui->RecordODF->isChecked() ? 1 : 0;
    handle->voxel.odf_deconvolusion = ui->ODFSharpening->currentIndex() >= 1 ? 1 : 0;
    handle->voxel.odf_decomposition = ui->Decomposition->currentIndex() >= 1 ? 1 : 0;
    handle->voxel.odf_xyz[0] = ui->x->value();
    handle->voxel.odf_xyz[1] = ui->y->value();
    handle->voxel.odf_xyz[2] = ui->z->value();
    handle->voxel.half_sphere = ui->HalfSphere->isChecked() ? 1 : 0;
    handle->voxel.max_fiber_number = ui->NumOfFibers->value();
    handle->voxel.r2_weighted = ui->ODFDef->currentIndex();

    const char* msg = (const char*)reconstruction(handle, method_id, params);
    if (!QFileInfo(msg).exists())
    {
        QMessageBox::information(this,"error",msg,0);
        return;
    }
    if(!prompt)
        return;

    QMessageBox::information(this,"DSI Studio","done!",0);
    ((MainWindow*)parent())->addFib(msg);
}

void reconstruction_window::on_SlicePos_sliderMoved(int position)
{
    if (!image.size())
        return;
    buffer.resize(image::geometry<2>(image.width(),image.height()));
    unsigned int offset = position*buffer.size();
    std::copy(image.begin() + offset,image.begin()+ offset + buffer.size(),buffer.begin());

    unsigned char* slice_image_ptr = &*image.begin() + buffer.size()* position;
    unsigned char* slice_mask = &*handle->mask.begin() + buffer.size()* position;
    for (unsigned int index = 0; index < buffer.size(); ++index)
    {
        unsigned char value = slice_image_ptr[index];
        if (slice_mask[index])
            buffer[index] = image::rgb_color(255, value, value);
        else
            buffer[index] = image::rgb_color(value, value, value);
    }

    double ratio = std::max(1.0,
        std::min((double)ui->graphicsView->width()/(double)image.width(),
                 (double)ui->graphicsView->height()/(double)image.height()));
    scene.setSceneRect(0, 0, image.width()*ratio,image.height()*ratio);
    slice_image = QImage((unsigned char*)&*buffer.begin(),image.width(),image.height(),QImage::Format_RGB32).
                    scaled(image.width()*ratio,image.height()*ratio);
    scene.clear();
    scene.addRect(0, 0, image.width()*ratio,image.height()*ratio,QPen(),slice_image);
}

void reconstruction_window::on_erosion_clicked()
{
    image::morphology::erosion(handle->mask);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::on_dilation_clicked()
{
    image::morphology::dilation(handle->mask);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::on_defragment_clicked()
{
    image::morphology::defragment(handle->mask);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::on_smoothing_clicked()
{
    image::morphology::smoothing(handle->mask);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::on_thresholding_clicked()
{
    bool ok;
    int threshold = QInputDialog::getInt(this,"DSI Studio","Please assign the threshold",
                                         (int)image::segmentation::otsu_threshold(image),
                                         (int)*std::min_element(image.begin(),image.end()),
                                         (int)*std::max_element(image.begin(),image.end()),1,&ok);
    if (!ok)
        return;
    image::threshold(image,handle->mask,threshold);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}

void reconstruction_window::on_load_mask_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open region",
            absolute_path,
            "Mask files (*.txt *.nii *.hdr);;All files (*.*)" );
    if(filename.isEmpty())
        return;
    ROIRegion region(image.geometry(),image::vector<3>(get_voxel_size((ImageModel*)handle)));
    std::vector<float> trans;
    region.LoadFromFile(filename.toLocal8Bit().begin(),trans);
    region.SaveToBuffer(handle->mask);
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}


void reconstruction_window::on_save_mask_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save region",
            absolute_path+"/mask.txt",
            "Text files (*.txt);;Nifti file(*.nii)" );
    if(filename.isEmpty())
        return;
    ROIRegion region(image.geometry(),image::vector<3>(get_voxel_size((ImageModel*)handle)));
    region.LoadFromBuffer(handle->mask);
    std::vector<float> trans;
    region.SaveToFile(filename.toLocal8Bit().begin(),trans);
}

void reconstruction_window::on_doDTI_clicked()
{
    for(int index = 0;index < filenames.size();++index)
    {
        if(index)
        {
            if(handle)
                free_reconstruction((ImageModel*)handle);
            handle = 0;
            begin_prog("load src");
            load_src(index);
        }
        std::fill(params,params+5,0.0);
        if(ui->DTI->isChecked())
            doReconstruction(1,index+1 == filenames.size());
        else
        if(ui->DSI->isChecked())
        {
            params[0] = ui->hamming_filter->value();
            settings.setValue("rec_hamming_filter",params[0]);
            doReconstruction(0,index+1 == filenames.size());
        }
        else
        if(ui->QBI->isChecked())
        {
            params[0] = ui->regularization_param->value();
            settings.setValue("rec_qbi_reg",params[0]);
            doReconstruction(3,index+1 == filenames.size());
        }
        else
        if(ui->GQI->isChecked() || ui->QDif->isChecked() || ui->QSDRT->isChecked())
        {
            params[0] = ui->diffusion_sampling->value();
            settings.setValue("rec_gqi_sampling",ui->diffusion_sampling->value());
            if(ui->QDif->isChecked() || ui->QSDRT->isChecked())
            {
                params[1] = ui->mni_resolution->value();
                settings.setValue("rec_mni_resolution",params[1]);
                if(ui->QSDRT->isChecked())
                {
                    handle->voxel.template_file_name = (QFileInfo(filenames[0]).absolutePath() + "/atlas").toLocal8Bit().begin();
                    handle->voxel.file_list.clear();
                    for(int index = 0;index < filenames.size();++index)
                        handle->voxel.file_list.push_back(filenames[index].toLocal8Bit().begin());
                    doReconstruction(8,true);
                    return;
                }
                else
                    doReconstruction(7,index+1 == filenames.size());

            }
            else
                doReconstruction(4,index+1 == filenames.size());
        }
        if(prog_aborted())
            break;
    }
}

void reconstruction_window::on_DTI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->OptionGroupBox->setVisible(!checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);
}

void reconstruction_window::on_DSI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->OptionGroupBox->setVisible(checked);
    ui->DSIOption_2->setVisible(checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);
}

void reconstruction_window::on_QBI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->OptionGroupBox->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(checked);
    ui->GQIOption_2->setVisible(!checked);
}

void reconstruction_window::on_GQI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->OptionGroupBox->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);
}

void reconstruction_window::on_QDif_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(checked);
    ui->OptionGroupBox->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);
}

void reconstruction_window::on_QSDRT_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(checked);
    ui->OptionGroupBox->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);
}

void reconstruction_window::on_ODFSharpening_currentIndexChanged(int index)
{
    if(ui->ODFSharpening->currentIndex() == 1 && ui->Decomposition->currentIndex() == 1)
        ui->Decomposition->setCurrentIndex(0);
    ui->xyz_widget->setVisible(ui->ODFSharpening->currentIndex() ||
                               ui->Decomposition->currentIndex());
}

void reconstruction_window::on_Decomposition_currentIndexChanged(int index)
{
    if(ui->ODFSharpening->currentIndex() == 1 && ui->Decomposition->currentIndex() == 1)
        ui->ODFSharpening->setCurrentIndex(0);
    ui->xyz_widget->setVisible(ui->ODFSharpening->currentIndex() ||
                               ui->Decomposition->currentIndex());

}

void reconstruction_window::on_remove_background_clicked()
{
    for(int index = 0;index < handle->mask.size();++index)
        if(handle->mask[index] == 0)
            image[index] = 0;

    for(int index = 0;index < handle->dwi_data.size();++index)
    {
        unsigned short* buf = (unsigned short*)handle->dwi_data[index];
        for(int i = 0;i < handle->mask.size();++i)
            if(handle->mask[i] == 0)
                buf[i] = 0;
    }
    on_SlicePos_sliderMoved(ui->SlicePos->value());
}


void reconstruction_window::on_zoom_in_clicked()
{
    source_ratio *= 1.1;
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_zoom_out_clicked()
{
    source_ratio *= 0.9;
    on_b_table_itemSelectionChanged();
}
