#include <QSplitter>
#include <QThread>
#include "reconstruction_window.h"
#include "ui_reconstruction_window.h"
#include "dsi_interface_static_link.h"
#include "mapping/fa_template.hpp"
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
#include "gzip_interface.hpp"
#include "manual_alignment.h"


bool reconstruction_window::load_src(int index)
{
    begin_prog("load src");
    check_prog(index,filenames.size());
    handle.reset(new ImageModel);
    if (!handle->load_from_file(filenames[index].toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error",QString("Cannot open ") +
            filenames[index] + " : " +handle->error_msg.c_str(),0);
        check_prog(0,0);
        return false;
    }

    update_image();
    return true;
}

void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);

reconstruction_window::reconstruction_window(QStringList filenames_,QWidget *parent) :
    QMainWindow(parent),filenames(filenames_),terminated(false),motion_correction_thread(0),
        ui(new Ui::reconstruction_window)
{

    ui->setupUi(this);
    if(!load_src(0))
        throw std::runtime_error("Cannot load src file");

    ui->toolBox->setCurrentIndex(1);
    ui->graphicsView->setScene(&scene);
    ui->view_source->setScene(&source);
    ui->b_table->setColumnWidth(0,60);
    ui->b_table->setColumnWidth(1,80);
    ui->b_table->setColumnWidth(2,80);
    ui->b_table->setColumnWidth(3,80);
    ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");
    ui->gqi_spectral->hide();

    update_dimension();

    absolute_path = QFileInfo(filenames[0]).absolutePath();


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
    default:
        ui->GQI->setChecked(true);
        on_GQI_toggled(true);
        break;
    }
    ui->AdvancedWidget->setVisible(false);
    ui->ThreadCount->setValue(settings.value("rec_thread_num",2).toInt());
    ui->NumOfFibers->setValue(settings.value("rec_num_fiber",5).toInt());
    ui->ODFDef->setCurrentIndex(settings.value("rec_gqi_def",0).toInt());
    ui->reg_method->setCurrentIndex(settings.value("rec_reg_method",0).toInt());
    ui->interpo_method->setCurrentIndex(settings.value("rec_interpo_method",2).toInt());


    ui->diffusion_sampling->setValue(settings.value("rec_gqi_sampling",1.25).toDouble());
    ui->regularization_param->setValue(settings.value("rec_qbi_reg",0.006).toDouble());
    ui->SHOrder->setValue(settings.value("rec_qbi_sh_order",8).toInt());
    ui->hamming_filter->setValue(settings.value("rec_hamming_filter",17).toDouble());

    ui->odf_sharpening->setCurrentIndex(settings.value("rec_odf_sharpening",0).toInt());
    ui->decon_param->setValue(settings.value("rec_deconvolution_param",3.0).toDouble());
    ui->decom_m->setValue(settings.value("rec_decom_m",10).toInt());

    ui->resolution->setCurrentIndex(settings.value("rec_resolution",2).toInt());

    ui->ODFDim->setCurrentIndex(settings.value("odf_order",3).toInt());

    ui->RecordODF->setChecked(settings.value("rec_record_odf",0).toInt());
    ui->output_jacobian->setChecked(settings.value("output_jacobian",0).toInt());
    ui->output_mapping->setChecked(settings.value("output_mapping",0).toInt());
    ui->rdi->setChecked(settings.value("output_rdi",1).toInt());
    ui->check_btable->setChecked(settings.value("check_btable",1).toInt());

    ui->report->setText(handle->voxel.report.c_str());

    max_source_value = *std::max_element(handle->dwi_data.back(),
                                         handle->dwi_data.back()+handle->voxel.dim.size());
    ui->brightness->setMaximum(max_source_value);
    ui->brightness->setMinimum(-max_source_value);
    ui->brightness->setSingleStep(max_source_value/50.0);
    ui->contrast->setMaximum(max_source_value*11.0);
    ui->contrast->setMinimum(max_source_value/11.0);
    ui->contrast->setSingleStep(max_source_value/50.0);
    ui->contrast->setValue(max_source_value);
    ui->brightness->setValue(0.0);


    on_odf_sharpening_currentIndexChanged(ui->odf_sharpening->currentIndex());
    connect(ui->z_pos,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->contrast,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->brightness,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));

    on_b_table_itemSelectionChanged();


    {
        std::vector<unsigned int> shell;
        calculate_shell(handle->voxel.bvalues,shell);
        ui->half_sphere->setChecked((shell.size() > 5) && (shell[1] - shell[0] <= 3));
        if(!ui->half_sphere->isChecked())
        {
            ui->scheme_balance->setChecked((shell.size() <= 5) && !shell.empty() &&
                handle->voxel.bvalues.size()-shell.back() < 100);
        }
    }
}
void reconstruction_window::update_dimension(void)
{
    ui->SlicePos->setRange(0,handle->voxel.dim[2]-1);
    ui->SlicePos->setValue((handle->voxel.dim[2]-1) >> 1);
    ui->z_pos->setRange(0,handle->voxel.dim[2]-1);
    ui->z_pos->setValue((handle->voxel.dim[2]-1) >> 1);
    ui->x->setMaximum(handle->voxel.dim[0]-1);
    ui->y->setMaximum(handle->voxel.dim[1]-1);
    ui->z->setMaximum(handle->voxel.dim[2]-1);
    source_ratio = std::max(1.0,500/(double)handle->voxel.dim.height());
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
    image::basic_image<float,2> tmp(image::geometry<2>(handle->voxel.dim[0],handle->voxel.dim[1]));
    unsigned int b_index = ui->b_table->currentRow();
    if(motion_args.empty())
        std::copy(handle->dwi_data[b_index] + ui->z_pos->value()*tmp.size(),
                  handle->dwi_data[b_index] + ui->z_pos->value()*tmp.size() + tmp.size(),tmp.begin());
    else
    {
        image::transformation_matrix<float> T(motion_args[b_index],handle->voxel.dim,handle->voxel.vs,handle->voxel.dim,handle->voxel.vs);
        for(int y = 0,index = 0;y < handle->voxel.dim[1];++y)
            for(int x = 0;x < handle->voxel.dim[0];++x,++index)
            {
                image::vector<3> pos(x,y,ui->z_pos->value()),to;
                T(pos,to);
                image::estimate(image::make_image(handle->voxel.dim,handle->dwi_data[b_index]),
                                to,tmp[index],image::cubic);
            }
    }
    tmp += ui->brightness->value();
    if( ui->contrast->value() != 0.0)
        tmp *= 255.99/ui->contrast->value();
    image::upper_lower_threshold(tmp,(float)0.0,(float)255.0);

    image::geometry<3> dim(handle->voxel.dim);
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
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::showEvent ( QShowEvent * event )
{
    QMainWindow::showEvent(event);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);

}

reconstruction_window::~reconstruction_window()
{
    if(motion_correction_thread.get())
    {
        terminated = true;
        motion_correction_thread->join();
        motion_correction_thread.reset(0);
    }
    delete ui;
}

void reconstruction_window::doReconstruction(unsigned char method_id,bool prompt)
{
    if(!handle.get())
        return;

    if(method_id == 7)
        std::fill(handle->mask.begin(),handle->mask.end(),1.0);
    else
    if (*std::max_element(handle->mask.begin(),handle->mask.end()) == 0)
    {
        QMessageBox::information(this,"error","Please select mask for reconstruction",0);
        return;
    }

    if (ui->odf_sharpening->currentIndex() == 1 && method_id != 1) // deconvolution
    {
        params[2] = ui->decon_param->value();
        settings.setValue("rec_deconvolution_param",params[2]);
    }
    if (ui->odf_sharpening->currentIndex() == 2 && method_id != 1) // decomposition
    {
        params[3] = ui->decom_fraction->value();
        params[4] = ui->decom_m->value();
        settings.setValue("rec_decomposition_param",params[3]);
        settings.setValue("rec_decom_m",params[4]);
    }

    settings.setValue("rec_method_id",method_id);
    settings.setValue("rec_thread_num",ui->ThreadCount->value());
    settings.setValue("rec_odf_sharpening",ui->odf_sharpening->currentIndex());
    settings.setValue("rec_num_fiber",ui->NumOfFibers->value());
    settings.setValue("rec_gqi_def",ui->ODFDef->currentIndex());
    settings.setValue("rec_reg_method",ui->reg_method->currentIndex());
    settings.setValue("rec_interpo_method",ui->interpo_method->currentIndex());

    settings.setValue("odf_order",ui->ODFDim->currentIndex());
    settings.setValue("rec_record_odf",ui->RecordODF->isChecked() ? 1 : 0);
    settings.setValue("output_jacobian",ui->output_jacobian->isChecked() ? 1 : 0);
    settings.setValue("output_mapping",ui->output_mapping->isChecked() ? 1 : 0);
    settings.setValue("output_rdi",(ui->rdi->isChecked() && method_id == 4) ? 1 : 0); // only for GQI
    settings.setValue("check_btable",ui->check_btable->isChecked() ? 1 : 0);

    begin_prog("reconstruction",true);
    int odf_order[8] = {4, 5, 6, 8, 10, 12, 16, 20};
    handle->voxel.ti.init(odf_order[ui->ODFDim->currentIndex()]);
    handle->voxel.odf_deconvolusion = ui->odf_sharpening->currentIndex() == 1 ? 1 : 0;
    handle->voxel.odf_decomposition = ui->odf_sharpening->currentIndex() == 2 ? 1 : 0;
    handle->voxel.odf_xyz[0] = ui->x->value();
    handle->voxel.odf_xyz[1] = ui->y->value();
    handle->voxel.odf_xyz[2] = ui->z->value();
    handle->voxel.max_fiber_number = ui->NumOfFibers->value();
    handle->voxel.r2_weighted = ui->ODFDef->currentIndex();
    handle->voxel.reg_method = ui->reg_method->currentIndex();
    handle->voxel.interpo_method = ui->interpo_method->currentIndex();

    handle->voxel.need_odf = ui->RecordODF->isChecked() ? 1 : 0;
    handle->voxel.output_jacobian = ui->output_jacobian->isChecked() ? 1 : 0;
    handle->voxel.output_mapping = ui->output_mapping->isChecked() ? 1 : 0;
    handle->voxel.output_rdi = ui->rdi->isChecked() ? 1 : 0;
    if(method_id == 7 || method_id == 4)
    {
        handle->voxel.half_sphere = ui->half_sphere->isChecked() ? 1:0;
        handle->voxel.scheme_balance = ui->scheme_balance->isChecked() ? 1:0;
    }
    else
    {
        handle->voxel.half_sphere = false;
        handle->voxel.scheme_balance = false;
    }

    const char* msg = (const char*)reconstruction(handle.get(), method_id,
                                                  params,ui->check_btable->isChecked(),
                                                  ui->ThreadCount->value());
    if (!QFileInfo(msg).exists())
    {
        QMessageBox::information(this,"error",msg,0);
        return;
    }
    if(!prompt)
        return;

    QMessageBox::information(this,"DSI Studio","FIB file created.",0);
    if(method_id == 6)
        ((MainWindow*)parent())->addSrc(msg);
    else
        ((MainWindow*)parent())->addFib(msg);
}


void reconstruction_window::on_erosion_clicked()
{
    image::morphology::erosion(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_dilation_clicked()
{
    image::morphology::dilation(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_defragment_clicked()
{
    image::morphology::defragment(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_smoothing_clicked()
{
    image::morphology::smoothing(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::on_negate_clicked()
{
    image::morphology::negate(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


void reconstruction_window::on_thresholding_clicked()
{
    bool ok;
    int threshold = QInputDialog::getInt(this,"DSI Studio","Please assign the threshold",
                                         (int)image::segmentation::otsu_threshold(dwi),
                                         (int)*std::min_element(dwi.begin(),dwi.end()),
                                         (int)*std::max_element(dwi.begin(),dwi.end()),1,&ok);
    if (!ok)
        return;
    image::threshold(dwi,handle->mask,threshold);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_load_mask_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open region",
            absolute_path,
            "Mask files (*.txt *.nii *nii.gz *.hdr);;All files (*)" );
    if(filename.isEmpty())
        return;
    ROIRegion region(dwi.geometry(),handle->voxel.vs);
    std::vector<float> trans;
    region.LoadFromFile(filename.toLocal8Bit().begin(),trans);
    region.SaveToBuffer(handle->mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


void reconstruction_window::on_save_mask_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save region",
            absolute_path+"/mask.txt",
            "Text files (*.txt);;Nifti file(*nii.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(QFileInfo(filename.toLower()).completeSuffix() != "txt")
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
    ROIRegion region(dwi.geometry(),handle->voxel.vs);
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
            begin_prog("load src");
            if(!load_src(index))
                return;
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
            params[1] = ui->SHOrder->value();
            settings.setValue("rec_qbi_reg",params[0]);
            settings.setValue("rec_qbi_sh_order",params[1]);
            doReconstruction(3,index+1 == filenames.size());
        }
        else
        if(ui->GQI->isChecked() || ui->QDif->isChecked())
        {
            params[0] = ui->diffusion_sampling->value();
            if(params[0] == 0.0)
                params[1] = ui->diffusion_time->value();
            settings.setValue("rec_gqi_sampling",ui->diffusion_sampling->value());
            if(ui->QDif->isChecked())
            {
                float res[4] = {0.5,1.0,2.0,3.0};
                params[1] = res[ui->resolution->currentIndex()];
                settings.setValue("rec_resolution",ui->resolution->currentIndex());
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
    ui->ODFSharpening->setVisible(!checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);

    ui->AdvancedOptions->setVisible(!checked);

    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->RecordODF->setVisible(!checked);
    ui->rdi->setVisible(!checked);


}

void reconstruction_window::on_DSI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->ODFSharpening->setVisible(checked);
    ui->DSIOption_2->setVisible(checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->RecordODF->setVisible(checked);
    ui->rdi->setVisible(!checked);

}

void reconstruction_window::on_QBI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->ODFSharpening->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(checked);
    ui->GQIOption_2->setVisible(!checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->RecordODF->setVisible(checked);
    ui->rdi->setVisible(!checked);

}

void reconstruction_window::on_GQI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);
    ui->ODFSharpening->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->RecordODF->setVisible(checked);

    ui->rdi->setVisible(checked);
    if(checked)
        ui->rdi->setChecked(true);
}

void reconstruction_window::on_QDif_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(checked);
    ui->ODFSharpening->setVisible(checked);
    ui->DSIOption_2->setVisible(!checked);
    ui->QBIOption_2->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_mapping->setVisible(checked);
    ui->output_jacobian->setVisible(checked);
    ui->RecordODF->setVisible(checked);
    ui->rdi->setVisible(!checked);


}


void reconstruction_window::on_remove_background_clicked()
{
    for(int index = 0;index < handle->mask.size();++index)
        if(handle->mask[index] == 0)
            dwi[index] = 0;

    for(int index = 0;index < handle->dwi_data.size();++index)
    {
        unsigned short* buf = (unsigned short*)handle->dwi_data[index];
        for(int i = 0;i < handle->mask.size();++i)
            if(handle->mask[i] == 0)
                buf[i] = 0;
    }
    on_SlicePos_valueChanged(ui->SlicePos->value());
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

extern fa_template fa_template_imp;
void reconstruction_window::on_manual_reg_clicked()
{
    std::auto_ptr<manual_alignment> manual(new manual_alignment(this,dwi,handle->voxel.vs,fa_template_imp.I,fa_template_imp.vs,image::reg::rigid_body));
    manual->timer->start();
    if(manual->exec() == QDialog::Accepted)
        handle->voxel.qsdr_trans = manual->data.T;
}

void reconstruction_window::on_odf_sharpening_currentIndexChanged(int index)
{
    ui->xyz_widget->setVisible(ui->odf_sharpening->currentIndex() > 0);
    ui->decom_panel->setVisible(ui->odf_sharpening->currentIndex() == 2);
    ui->decon_param->setVisible(ui->odf_sharpening->currentIndex() == 1);
    on_RFSelection_currentIndexChanged(0);
}

void reconstruction_window::on_RFSelection_currentIndexChanged(int index)
{
    ui->ODFSelection->setVisible(ui->RFSelection->currentIndex() > 0);
}

void reconstruction_window::on_AdvancedOptions_clicked()
{
    if(ui->AdvancedOptions->text() == "Advanced Options >>")
    {
        ui->AdvancedWidget->setVisible(true);
        ui->AdvancedOptions->setText("Advanced Options <<");
    }
    else
    {
        ui->AdvancedWidget->setVisible(false);
        ui->AdvancedOptions->setText("Advanced Options >>");
    }
}


void reconstruction_window::on_actionSave_4D_nifti_triggered()
{
    if(filenames.size() > 1)
    {
        for(int index = 0;check_prog(index,filenames.size());++index)
        {
            ImageModel model;
            if (!model.load_from_file(filenames[index].toLocal8Bit().begin()))
            {
                QMessageBox::information(this,"error",QString("Cannot open ") +
                    filenames[index] + " : " +handle->error_msg.c_str(),0);
                check_prog(0,0);
                return;
            }
            model.save_to_nii((filenames[index]+".nii.gz").toLocal8Bit().begin());
        }
        return;
    }
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save image as...",
                            filenames[0] + ".nii.gz",
                                "All files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_to_nii(filename.toLocal8Bit().begin());
}

void reconstruction_window::on_actionSave_b_table_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save b table as...",
                            QFileInfo(filenames[0]).absolutePath() + "/b_table.txt",
                                "Text files (*.txt)" );
    if ( filename.isEmpty() )
        return;
    handle->save_b_table(filename.toLocal8Bit().begin());
}

void reconstruction_window::on_actionSave_bvals_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save b table as...",
                                QFileInfo(filenames[0]).absolutePath() + "/bvals",
                                "Text files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_bval(filename.toLocal8Bit().begin());
}

void reconstruction_window::on_actionSave_bvecs_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save b table as...",
                                QFileInfo(filenames[0]).absolutePath() + "/bvecs",
                                "Text files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_bvec(filename.toLocal8Bit().begin());
}

void reconstruction_window::update_image(void)
{
    dwi.resize(handle->voxel.dim);
    for(unsigned int index = 0;index < dwi.size();++index)
        dwi[index] = handle->voxel.dwi_sum[index]*255.0;
    load_b_table();
}

void reconstruction_window::on_actionFlip_x_triggered()
{
    handle->flip(0);
    update_image();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_actionFlip_y_triggered()
{
    handle->flip(1);
    update_image();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_actionFlip_z_triggered()
{
    handle->flip(2);
    update_image();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_actionFlip_xy_triggered()
{
    begin_prog("rotating");
    handle->flip(3);
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::on_actionFlip_yz_triggered()
{
    begin_prog("rotating");
    handle->flip(4);
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::on_actionFlip_xz_triggered()
{
    begin_prog("rotating");
    handle->flip(5);
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


void reconstruction_window::on_actionRotate_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    image::basic_image<float,3> ref;
    image::vector<3> vs;
    gz_nifti in;
    if(!in.load_from_file(filename.toLocal8Bit().begin()) || !in.toLPS(ref))
    {
        QMessageBox::information(this,"Error","Not a valid nifti file",0);
        return;
    }
    in.get_voxel_size(vs.begin());
    std::auto_ptr<manual_alignment> manual(new manual_alignment(this,dwi,handle->voxel.vs,ref,vs,image::reg::rigid_body));
    manual->timer->start();
    if(manual->exec() != QDialog::Accepted)
        return;

    begin_prog("rotating");
    handle->rotate(ref.geometry(),manual->data.iT);
    handle->calculate_mask();
    handle->voxel.vs = vs;
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());

}


void reconstruction_window::on_delete_2_clicked()
{
    if(handle->dwi_data.size() == 1)
        return;
    unsigned int index = ui->b_table->currentRow();
    ui->b_table->removeRow(index);
    handle->dwi_data.erase(handle->dwi_data.begin()+index);
    handle->voxel.bvalues.erase(handle->voxel.bvalues.begin()+index);
    handle->voxel.bvectors.erase(handle->voxel.bvectors.begin()+index);
}

void reconstruction_window::on_actionTrim_image_triggered()
{
    begin_prog("trimming");
    handle->trim();
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_diffusion_sampling_valueChanged(double arg1)
{
    if(arg1 == 0.0)
        ui->gqi_spectral->show();
    else
        ui->gqi_spectral->hide();
}

void reconstruction_window::on_SlicePos_valueChanged(int position)
{
    if (!dwi.size())
        return;
    buffer.resize(image::geometry<2>(dwi.width(),dwi.height()));
    unsigned int offset = position*buffer.size();
    std::copy(dwi.begin() + offset,dwi.begin()+ offset + buffer.size(),buffer.begin());

    unsigned char* slice_image_ptr = &*dwi.begin() + buffer.size()* position;
    unsigned char* slice_mask = &*handle->mask.begin() + buffer.size()* position;
    for (unsigned int index = 0; index < buffer.size(); ++index)
    {
        unsigned char value = slice_image_ptr[index]*0.8;
        if (slice_mask[index])
            buffer[index] = image::rgb_color(255, value, value);
        else
            buffer[index] = image::rgb_color(value, value, value);
    }

    double ratio = std::max(1.0,
        std::min(((double)ui->graphicsView->width()-5)/(double)dwi.width(),
                 ((double)ui->graphicsView->height()-5)/(double)dwi.height()));
    scene.setSceneRect(0, 0, dwi.width()*ratio,dwi.height()*ratio);
    slice_image = QImage((unsigned char*)&*buffer.begin(),dwi.width(),dwi.height(),QImage::Format_RGB32).
                    scaled(dwi.width()*ratio,dwi.height()*ratio);
    scene.clear();
    scene.addRect(0, 0, dwi.width()*ratio,dwi.height()*ratio,QPen(),slice_image);
}

void rec_motion_correction_parallel(ImageModel* handle,
                                    std::vector<image::affine_transform<float> >& args,
                                    unsigned int total_thread,unsigned int id,unsigned int& progress,bool& terminated)
{
    for(unsigned int i = id;i < handle->voxel.bvalues.size() && !terminated;i += total_thread)
    {

        if(id == 0)
            progress = i*99/handle->voxel.bvalues.size();
        if(i == 0)
            continue;
        image::basic_image<float,3> I0,I1;
        I0 = image::make_image(handle->voxel.dim,handle->dwi_data[0]);
        I1 = image::make_image(handle->voxel.dim,handle->dwi_data[i]);
        image::filter::mean(I0);
        image::filter::mean(I1);
        image::filter::mean(I0);
        image::filter::mean(I1);
        image::filter::mean(I0);
        image::filter::mean(I1);
        image::filter::gradient_magnitude(I0);
        image::filter::gradient_magnitude(I1);
        image::normalize(I0);
        image::normalize(I1);


        image::affine_transform<float> upper,lower;
        upper.translocation[0] = 2;
        upper.translocation[1] = 2;
        upper.translocation[2] = 2;
        lower.translocation[0] = -2;
        lower.translocation[1] = -2;
        lower.translocation[2] = -2;
        upper.rotation[0] = 3.1415926*3.0/180.0;
        upper.rotation[1] = 3.1415926*3.0/180.0;
        upper.rotation[2] = 3.1415926*3.0/180.0;
        lower.rotation[0] = -3.1415926*3.0/180.0;
        lower.rotation[1] = -3.1415926*3.0/180.0;
        lower.rotation[2] = -3.1415926*3.0/180.0;
        upper.scaling[0] = 1.03;
        upper.scaling[1] = 1.03;
        upper.scaling[2] = 1.03;
        lower.scaling[0] = 0.96;
        lower.scaling[1] = 0.96;
        lower.scaling[2] = 0.96;
        upper.affine[0] = 0.04;
        upper.affine[1] = 0.04;
        upper.affine[2] = 0.04;
        lower.affine[0] = -0.04;
        lower.affine[1] = -0.04;
        lower.affine[2] = -0.04;
        image::normalize(I1);
        image::reg::fun_adoptor<image::basic_image<float,3>,
                                image::vector<3>,
                                image::affine_transform<float>,
                                image::affine_transform<float>,
                                image::reg::square_error> fun(I0,handle->voxel.vs,I1,handle->voxel.vs,args[i]);
        double optimal_value = fun(args[i][0]);
        image::optimization::graient_descent(args[i].begin(),args[i].end(),
                                             upper.begin(),lower.begin(),fun,optimal_value,terminated,0.05);
        //for(unsigned int index = 0;index < 12;++index)
        //    std::cout << args[i][index] << " ";
        //std::cout << std::endl;
    }
}

void rec_motion_correction(ImageModel* handle,unsigned int total_thread,
                           std::vector<image::affine_transform<float> >& args,
                           unsigned int& progress,
                           bool& terminated)
{
    args.resize(handle->voxel.bvalues.size());
    boost::thread_group threads;
    for(unsigned int i = 1;i < total_thread;++i)
        threads.add_thread(new boost::thread(&rec_motion_correction_parallel,
                                             handle,boost::ref(args),
                                             total_thread,i,boost::ref(progress),boost::ref(terminated)));
    rec_motion_correction_parallel(handle,args,total_thread,0,boost::ref(progress),terminated);
    threads.join_all();

    for(unsigned int i = 0;i < handle->voxel.bvalues.size();++i)
        handle->rotate_dwi(i,image::transformation_matrix<float>(args[i],handle->voxel.dim,handle->voxel.vs,handle->voxel.dim,handle->voxel.vs));
    args.clear();
    progress = 100;
}

void reconstruction_window::on_motion_correction_clicked()
{
    if(motion_correction_thread.get())
    {
        terminated = true;
        motion_correction_thread->join();
        ui->motion_correction->setText("Motion/Eddy Correction");
        timer.reset(0);
        ui->motion_correction_progress->setValue(0);
        motion_args.clear();
        motion_correction_thread.reset(0);
        return;
    }
    terminated = false;
    motion_correction_thread.reset(new boost::thread(&rec_motion_correction,
                                                     handle.get(),QThread::idealThreadCount(),
                                                     boost::ref(motion_args),boost::ref(progress),boost::ref(terminated)));
    timer.reset(new QTimer(this));
    timer->setInterval(1000);
    timer->start();
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(check_progress()));
    ui->motion_correction->setText("Stop");
}

void reconstruction_window::check_progress(void)
{
    ui->motion_correction_progress->setValue(progress);
    if(progress == 100)
    {
        on_motion_correction_clicked();
        handle->calculate_dwi_sum();
        handle->calculate_mask();
        update_image();
    }
}




void reconstruction_window::on_scheme_balance_toggled(bool checked)
{
    if(checked)
        ui->half_sphere->setChecked(false);
}



void reconstruction_window::on_half_sphere_toggled(bool checked)
{
    if(checked)
        ui->scheme_balance->setChecked(false);
}

void reconstruction_window::on_add_t1t2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    image::basic_image<float,3> ref;
    image::vector<3> vs;
    gz_nifti in;
    if(!in.load_from_file(filename.toLocal8Bit().begin()) || !in.toLPS(ref))
    {
        QMessageBox::information(this,"Error","Not a valid nifti file",0);
        return;
    }
    image::transformation_matrix<float> affine;
    bool has_registered = false;
    for(unsigned int index = 0;index < handle->voxel.other_image.size();++index)
        if(ref.dimension == handle->voxel.other_image[index].dimension)
        {
            int result = QMessageBox::information(this,"Adding T1W/T2W","Apply previous registration?",QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::Yes)
            {
                affine = handle->voxel.other_image_affine[index];
                has_registered = true;
            }
            break;
        }
    if(!has_registered)
    {
        in.get_voxel_size(vs.begin());
        std::auto_ptr<manual_alignment> manual(new manual_alignment(this,dwi,handle->voxel.vs,ref,vs,image::reg::rigid_body));
        manual->timer->start();
        if(manual->exec() != QDialog::Accepted)
            return;
        affine = manual->data.T;
    }
    handle->voxel.other_image.push_back(ref);
    handle->voxel.other_image_name.push_back(QFileInfo(filename).baseName().toLocal8Bit().begin());
    handle->voxel.other_image_affine.push_back(affine);
    ui->output_mapping->setChecked(true);
}

void reconstruction_window::on_actionManual_Rotation_triggered()
{
    std::auto_ptr<manual_alignment> manual(new manual_alignment(this,dwi,handle->voxel.vs,dwi,handle->voxel.vs,0));
    if(manual->exec() != QDialog::Accepted)
        return;
    begin_prog("rotating");
    handle->rotate(dwi.geometry(),manual->data.iT);
    handle->calculate_mask();
    update_image();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


