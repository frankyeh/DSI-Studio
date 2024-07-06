#include <filesystem>
#include <QSplitter>
#include <QThread>
#include <QImage>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QSettings>
#include "reconstruction_window.h"
#include "ui_reconstruction_window.h"
#include "reg.hpp"
#include "mainwindow.h"
#include "tracking/region/Regions.h"
#include "libs/dsi/image_model.hpp"
#include "manual_alignment.h"

void populate_templates(QComboBox* combo,size_t index);
void move_current_dir_to(const std::string& file_name);
bool reconstruction_window::load_src(int index)
{
    handle = std::make_shared<src_data>();
    if (!handle->load_from_file(filenames[index].toStdString().c_str()))
        return false;
    move_current_dir_to(filenames[index].toStdString());
    tipl::progress prog("initiate interface");
    existing_steps = handle->voxel.steps;
    if(handle->voxel.is_histology)
        return true;
    update_dimension();
    double m = double(tipl::max_value(tipl::make_image(handle->src_dwi_data[0],handle->voxel.dim)));
    ui->max_value->setMaximum(m*1.5);
    ui->max_value->setMinimum(0.0);
    ui->max_value->setSingleStep(m*0.05);
    ui->max_value->setValue(m*0.2f);
    ui->min_value->setMaximum(m*1.5);
    ui->min_value->setMinimum(0.0);
    ui->min_value->setSingleStep(m*0.05);
    ui->min_value->setValue(0.0);
    load_b_table();

    ui->align_slices->setVisible(false);
    return true;
}

extern std::vector<std::string> fa_template_list,iso_template_list;
reconstruction_window::reconstruction_window(QStringList filenames_,QWidget *parent) :
    QMainWindow(parent),filenames(filenames_),ui(new Ui::reconstruction_window)
{
    ui->setupUi(this);
    if(!load_src(0))
        throw std::runtime_error(handle->error_msg.c_str());
    setWindowTitle(filenames[0]);
    ui->ThreadCount->setMaximum(tipl::max_thread_count);
    ui->toolBox->setCurrentIndex(1);
    ui->graphicsView->setScene(&scene);
    ui->view_source->setScene(&source);
    ui->b_table->setColumnWidth(0,60);
    ui->b_table->setColumnWidth(1,80);
    ui->b_table->setColumnWidth(2,80);
    ui->b_table->setColumnWidth(3,80);
    ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");

    populate_templates(ui->primary_template,handle->voxel.template_id);
    if(ui->primary_template->currentIndex() == 0)
        ui->diffusion_sampling->setValue(1.25); // human studies
    else
        ui->diffusion_sampling->setValue(0.6);  // animal studies (likely ex-vivo)

    v2c.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));

    absolute_path = QFileInfo(filenames[0]).absolutePath();


    switch(settings.value("rec_method_id",4).toInt())
    {
    case 1:
        ui->DTI->setChecked(true);
        on_DTI_toggled(true);
        break;
    case 7:
        ui->QSDR->setChecked(true);
        on_QSDR_toggled(true);
        break;
    default:
        ui->GQI->setChecked(true);
        on_GQI_toggled(true);
        break;
    }

    ui->odf_resolving->setVisible(false);

    ui->AdvancedWidget->setVisible(false);
    ui->ThreadCount->setValue(settings.value("rec_thread_count",tipl::max_thread_count).toInt());

    ui->odf_resolving->setChecked(settings.value("odf_resolving",0).toInt());

    ui->RecordODF->setChecked(settings.value("rec_record_odf",0).toInt());

    ui->report->setText(handle->voxel.report.c_str());
    ui->dti_no_high_b->setChecked(handle->is_human_data());

    ui->method_group->setVisible(!handle->voxel.is_histology);
    ui->param_group->setVisible(!handle->voxel.is_histology);
    ui->hist_param_group->setVisible(handle->voxel.is_histology);

    ui->qsdr_reso->setValue(handle->voxel.vs[2]);

    if(handle->voxel.is_histology)
    {
        delete ui->menuCorrections;
        delete ui->menuB_table;
        delete ui->menuFile;
        ui->source_page->hide();
        ui->toolBox->removeItem(0);
        auto actions = ui->menuEdit->actions();
        for(int i = 4;i < actions.size();++i)
            actions[i]->setVisible(false);

        ui->hist_downsampling->setValue(std::ceil(std::log2(handle->voxel.hist_image.width()))-12);
    }


    connect(ui->z_pos,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->max_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->min_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));

    on_b_table_itemSelectionChanged();

    ui->mask_edit->setVisible(false);

}
void reconstruction_window::update_dimension(void)
{
    source_ratio = std::max(1.0,500/(double)handle->voxel.dim.height());
    if(ui->SlicePos->maximum() != handle->voxel.dim[2]-1)
    {
        ui->SlicePos->setRange(0,handle->voxel.dim[2]-1);
        ui->SlicePos->setValue((handle->voxel.dim[2]-1) >> 1);
    }
    if(ui->z_pos->maximum() != handle->voxel.dim[view_orientation]-1)
    {
        ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
        ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    }
}

void reconstruction_window::load_b_table(void)
{
    if(handle->src_bvalues.empty())
        return;
    ui->b_table->clear();
    ui->b_table->setRowCount(handle->src_bvalues.size());
    for(unsigned int index = 0;index < handle->src_bvalues.size();++index)
    {
        ui->b_table->setItem(index,0, new QTableWidgetItem(QString::number(handle->src_bvalues[index])));
        ui->b_table->setItem(index,1, new QTableWidgetItem(QString::number(handle->src_bvectors[index][0])));
        ui->b_table->setItem(index,2, new QTableWidgetItem(QString::number(handle->src_bvectors[index][1])));
        ui->b_table->setItem(index,3, new QTableWidgetItem(QString::number(handle->src_bvectors[index][2])));
    }
    ui->b_table->selectRow(0);
}

void reconstruction_window::on_b_table_itemSelectionChanged()
{
    if(handle->src_bvalues.empty())
        return;
    v2c.set_range(ui->min_value->value(),ui->max_value->value());

    QImage source_image;
    source_image << v2c[
                    tipl::volume2slice_scaled(
                       tipl::make_image(handle->src_dwi_data[ui->b_table->currentRow()],handle->voxel.dim),
                       view_orientation,ui->z_pos->value(),
                       source_ratio)];

    if(view_orientation != 2)
    {
        // show bad_slices
        if(bad_slice_analyzed)
        {
            std::vector<size_t> mark_slices;
            for(size_t index = 0;index < bad_slices.size();++index)
                if(bad_slices[index].first == ui->b_table->currentRow())
                    mark_slices.push_back(bad_slices[index].second);
            for(size_t index = 0;index < mark_slices.size();++index)
            {
                QPainter paint(&source_image);
                paint.setPen(Qt::red);
                paint.drawLine(0,source_ratio*mark_slices[index],
                               source_image.width(),source_ratio*mark_slices[index]);
            }
        }
        source_image = source_image.mirrored();
    }
    source << source_image;
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
    delete ui;
}

void reconstruction_window::Reconstruction(unsigned char method_id,bool prompt)
{
    if(!handle.get())
        return;

    if (tipl::max_value(handle->voxel.mask) == 0)
    {
        QMessageBox::critical(this,"ERROR","Please select mask for reconstruction");
        return;
    }

    settings.setValue("rec_method_id",method_id);
    settings.setValue("rec_thread_count",ui->ThreadCount->value());

    settings.setValue("odf_resolving",ui->odf_resolving->isChecked() ? 1 : 0);
    settings.setValue("rec_record_odf",ui->RecordODF->isChecked() ? 1 : 0);
    settings.setValue("other_output",ui->other_output->text());

    handle->voxel.method_id = method_id;
    handle->voxel.odf_resolving = ui->odf_resolving->isChecked();
    handle->voxel.output_odf = ui->RecordODF->isChecked();
    handle->voxel.dti_no_high_b = ui->dti_no_high_b->isChecked();
    handle->voxel.other_output = ui->other_output->text().toStdString();
    handle->voxel.thread_count = ui->ThreadCount->value();
    handle->voxel.template_id = ui->primary_template->currentIndex();

    if(handle->voxel.is_histology)
    {
        handle->voxel.vs[0] = handle->voxel.vs[1] = handle->voxel.vs[2] = float(ui->hist_resolution->value());
        handle->voxel.hist_downsampling = uint32_t(ui->hist_downsampling->value());
        handle->voxel.hist_raw_smoothing = uint32_t(ui->hist_raw_smoothing->value());
        handle->voxel.hist_tensor_smoothing = uint32_t(ui->hist_tensor_smoothing->value());
    }

    if(method_id == 7)
    {
        if(fa_template_list.empty())
        {
            QMessageBox::information(this,"error","Cannot find template files");
            return;
        }
        handle->voxel.qsdr_reso = ui->qsdr_reso->value();
    }


    auto dim_backup = handle->voxel.dim; // for QSDR
    auto vs = handle->voxel.vs; // for QSDR
    if (!handle->reconstruction())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return;
    }
    handle->voxel.dim = dim_backup;
    handle->voxel.vs = vs;
    if(method_id == 7) // QSDR
        handle->calculate_dwi_sum(true);
    if(!prompt)
        return;
    QMessageBox::information(this,QApplication::applicationName(),"FIB file created.");
    raise(); // for Mac
    QString filename = handle->file_name.c_str();
    filename += handle->get_file_ext().c_str();
    if(method_id == 6)
        ((MainWindow*)parent())->addSrc(filename);
    else
        ((MainWindow*)parent())->addFib(filename);
}




void reconstruction_window::on_save_mask_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save region",
            absolute_path+"/mask.nii.gz",
            "Nifti file(*nii.gz *.nii);;Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(QFileInfo(filename.toLower()).completeSuffix() != "txt")
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
    ROIRegion region(handle->dwi.shape(),handle->voxel.vs);
    region.load_region_from_buffer(handle->voxel.mask);
    region.save_region_to_file(filename.toStdString().c_str());
}

bool reconstruction_window::command(std::string cmd,std::string param)
{
    if(cmd == "[Step T2][Edit][Resample]" || cmd == "[Step T2][Edit][Align ACPC]")
    {
        bool ok;
        float nv = float(QInputDialog::getDouble(this,
            QApplication::applicationName(),"Assign output resolution in (mm):", double(handle->voxel.vs[0]),0.0,3.0,4, &ok));
        if (!ok || nv == 0.0f)
            return false;
        param = std::to_string(nv);
    }
    if(cmd == "[Step T2][File][Save 4D NIFTI]")
    {
        QString filename = QFileDialog::getSaveFileName(
                    this,"Save image as...",filenames[0] + ".nii.gz",
                                "NIFTI files (*nii.gz);;All files (*)" );
        if(filename.isEmpty())
            return false;
        param = filename.toStdString();
    }
    if(cmd == "[Step T2][File][Save Src File]")
    {
        QString filename = QFileDialog::getSaveFileName(
                this,"Save SRC file",filenames[0],
                        "SRC files (*src.gz);;All files (*)" );
        if(filename.isEmpty())
            return false;
        param = filename.toStdString();
    }
    if(tipl::contains_case_insensitive(cmd,"topup") && !std::filesystem::exists(handle->file_name+".corrected.nii.gz"))
    {
        QMessageBox::information(this,QApplication::applicationName(),"Please specify another NIFTI or SRC.GZ file with reversed phase encoding data");
        auto other_src = QFileDialog::getOpenFileName(
                    this,"Open SRC file",absolute_path,
                    "Images (*src.gz *.nii *nii.gz);;DICOM image (*.dcm);;All files (*)" );
        if(other_src.isEmpty())
            return false;
        param = other_src.toStdString();
    }

    if(tipl::contains_case_insensitive(cmd,"open"))
    {
        QString filename = QFileDialog::getOpenFileName(
            this,
            "Open region",
            absolute_path,
            "Mask files (*.nii *nii.gz *.hdr);;Text files (*.txt);;All files (*)" );
        if(filename.isEmpty())
            return false;
        param = filename.toStdString();
    }

    bool result = handle->command(cmd,param);
    if(!result)
    {
        if(!handle->error_msg.empty())
            QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
    }
    else
    {
        if(tipl::contains(cmd,"Corrections"))
            QMessageBox::information(this,QApplication::applicationName(),"correction result loaded");
        if(tipl::contains(cmd,"B-table"))
            QMessageBox::information(this,QApplication::applicationName(),cmd.find("Check") ? handle->error_msg.c_str() : "b-table updated");
    }
    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());


    if(filenames.size() > 1 && tipl::contains_case_insensitive(cmd,"save") &&
        QMessageBox::information(this,QApplication::applicationName(),"Apply to other SRC files?",
        QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel) == QMessageBox::Yes)
    {
        tipl::progress prog("apply to other SRC files");
        std::string steps(handle->voxel.steps.begin()+existing_steps.length(),handle->voxel.steps.end());
        steps += cmd;
        if(!param.empty())
        {
            steps += "=";
            steps += param;
            steps += "\n";
        }
        for(int index = 1;prog(index,filenames.size());++index)
        {
            src_data model;
            if (!model.load_from_file(filenames[index].toStdString().c_str()) ||
                !model.run_steps(handle->file_name,steps))
            {
                if(QMessageBox::information(this,QApplication::applicationName(),
                    QFileInfo(filenames[index]).fileName() + " : " + model.error_msg.c_str() + " Continue?",
                                QMessageBox::Yes|QMessageBox::No) == QMessageBox::No)
                    return false;
            }
        }
    }
    return result;
}
void reconstruction_window::on_doDTI_clicked()
{
    if(handle->voxel.vs[2] > handle->voxel.vs[0]*1.2f && handle->is_human_data() && !ui->QSDR->isChecked()) // non isotropic resolution
    {
        auto result = QMessageBox::information(this,QApplication::applicationName(),
            QString("The slice thickness is much larger than slice resolution. This is not ideal for fiber tracking. Resample slice thickness to 2mm isotropic resolution?"),
                QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
        if(result == QMessageBox::Cancel)
            return;
        if(result == QMessageBox::Yes)
            handle->align_acpc(2.0f);
    }
    std::string ref_file_name = handle->file_name;
    std::string ref_steps(handle->voxel.steps.begin()+existing_steps.length(),handle->voxel.steps.end());
    std::shared_ptr<src_data> ref_handle = handle;
    tipl::progress prog("process SRC files");
    for(int index = 0;prog(index,filenames.size());++index)
    {
        tipl::out() << "processing " << filenames[index].toStdString() << std::endl;
        if(index)
        {
            if(!load_src(index) || !handle->run_steps(ref_file_name,ref_steps))
            {
                if(!prog.aborted())
                    QMessageBox::critical(this,"ERROR",QFileInfo(filenames[index]).fileName() + " : " + handle->error_msg.c_str());
                break;
            }
        }
        if(ui->DTI->isChecked())
            Reconstruction(1,index+1 == filenames.size());
        else
        if(ui->GQI->isChecked() || ui->QSDR->isChecked())
        {
            handle->voxel.param[0] = float(ui->diffusion_sampling->value());
            settings.setValue("rec_gqi_sampling",ui->diffusion_sampling->value());
            if(ui->QSDR->isChecked())
                Reconstruction(7,index+1 == filenames.size());
            else
                Reconstruction(4,index+1 == filenames.size());
        }
    }
    handle = ref_handle;
}

void reconstruction_window::on_DTI_toggled(bool checked)
{
    ui->qsdr_options->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->RecordODF->setVisible(!checked);
    ui->qsdr_reso->setVisible(!checked);
    ui->qsdr_reso_label->setVisible(!checked);
    if(checked && (!ui->other_output->text().contains("rd") &&
                   !ui->other_output->text().contains("ad") &&
                   !ui->other_output->text().contains("md")))
        ui->other_output->setText("fa,rd,ad,md");

}


void reconstruction_window::on_GQI_toggled(bool checked)
{
    ui->qsdr_options->setVisible(!checked);


    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->RecordODF->setVisible(checked);

    ui->qsdr_reso->setVisible(!checked);
    ui->qsdr_reso_label->setVisible(!checked);
}

void reconstruction_window::on_QSDR_toggled(bool checked)
{
    ui->qsdr_options->setVisible(checked);
    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->RecordODF->setVisible(checked);

    ui->qsdr_reso->setVisible(checked);
    ui->qsdr_reso_label->setVisible(checked);

    ui->menuQSDR->setEnabled(checked);

}

void reconstruction_window::on_zoom_in_clicked()
{
    source_ratio *= 1.1f;
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_zoom_out_clicked()
{
    source_ratio *= 0.9f;
    on_b_table_itemSelectionChanged();
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

void reconstruction_window::on_actionSave_b0_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save image as...",
                            filenames[0] + ".b0.nii.gz",
                                "All files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_b0_to_nii(filename.toStdString().c_str());
}

void reconstruction_window::on_actionSave_DWI_sum_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save image as...",
                            filenames[0] + ".dwi_sum.nii.gz",
                                "All files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_dwi_sum_to_nii(filename.toStdString().c_str());
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
    handle->save_b_table(filename.toStdString().c_str());
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
    handle->save_bval(filename.toStdString().c_str());
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
    handle->save_bvec(filename.toStdString().c_str());
}


bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>&);
void reconstruction_window::on_actionRotate_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if( filenames.isEmpty())
        return;

    tipl::image<3> ref;
    tipl::vector<3> vs;
    tipl::matrix<4,4> t;
    if(!load_image_from_files(filenames,ref,vs,t))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                template_image_pre(tipl::image<3>(ref)),tipl::image<3,unsigned char>(),vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    tipl::progress prog_("rotating");
    tipl::image<3> ref2(ref);
    float m = tipl::median(ref2.begin(),ref2.end());
    tipl::multiply_constant(ref,0.5f/m);
    handle->rotate(ref.shape(),vs,manual->get_iT());
    handle->voxel.report += " The diffusion images were rotated and scaled to the space of ";
    handle->voxel.report += QFileInfo(filenames[0]).baseName().toStdString();
    handle->voxel.report += ". The b-table was also rotated accordingly.";
    ui->report->setText(handle->voxel.report.c_str());
    load_b_table();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());

}


void reconstruction_window::on_delete_2_clicked()
{
    if(handle->src_dwi_data.size() == 1)
        return;
    int index = ui->b_table->currentRow();
    if(index < 0)
        return;
    bad_slice_analyzed = false;
    ui->b_table->removeRow(index);
    handle->remove(uint32_t(index));

}

void reconstruction_window::on_remove_below_clicked()
{
    if(handle->src_dwi_data.size() == 1)
        return;
    int index = ui->b_table->currentRow();
    if(index <= 0)
        return;
    bad_slice_analyzed = false;
    while(ui->b_table->rowCount() > index)
    {
        ui->b_table->removeRow(index);
        handle->remove(uint32_t(index));
    }
}


void reconstruction_window::on_SlicePos_valueChanged(int position)
{
    tipl::color_image buffer;
    handle->draw_mask(buffer,position);
    double ratio =
        std::min(double(ui->graphicsView->width()-5)/double(buffer.width()),
                 double(ui->graphicsView->height()-5)/double(buffer.height()));
    QImage view;
    view << buffer;
    view = view.scaled(int(buffer.width()*ratio),int(buffer.height()*ratio));
    scene << view;
}

void reconstruction_window::on_actionAttach_Images_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    if(handle->add_other_image(QFileInfo(filename).baseName().toStdString(),filename.toStdString()))
        QMessageBox::information(this,QApplication::applicationName(),"File added");
    else
        QMessageBox::critical(this,"ERROR","Not a valid nifti file");

}

void reconstruction_window::on_actionT1W_based_QSDR_triggered()
{
    QString subject_image = QFileDialog::getOpenFileName(
            this,"Open Subject Image",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if(subject_image.isEmpty())
        return;
    QString template_image = QFileDialog::getOpenFileName(
            this,"Open Template Image",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if(template_image.isEmpty())
        return;
    if(handle->add_other_image("reg",subject_image.toStdString()))
    {
        handle->voxel.other_modality_template = template_image.toStdString();
        QMessageBox::information(this,QApplication::applicationName(),"Registration images added");
    }
    else
        QMessageBox::critical(this,"ERROR","Not a valid nifti file");

}

void reconstruction_window::on_actionPartial_FOV_triggered()
{
    QString values = QInputDialog::getText(this,QApplication::applicationName(),"Specify the range of MNI coordinates separated by spaces (minx miny minz maxx maxy maxz)",QLineEdit::Normal,
                                           QString("-36 -30 -20 36 30 24"));
    if(values.isEmpty())
        return;
    handle->command("[Step T2b(2)][Partial FOV]",values.toStdString());
}

void reconstruction_window::on_actionManual_Rotation_triggered()
{
    std::shared_ptr<manual_alignment> manual(
                new manual_alignment(this,subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                          subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
    if(manual->exec() != QDialog::Accepted)
        return;
    tipl::progress prog_("rotating");
    handle->rotate(handle->dwi.shape(),handle->voxel.vs,manual->get_iT());
    handle->voxel.report += " The diffusion images were manually rotated.";
    load_b_table();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


bool get_src(std::string filename,src_data& src2,std::string& error_msg)
{
    tipl::progress prog_("load ",filename.c_str());
    tipl::image<3,unsigned short> I;
    if(QString(filename.c_str()).endsWith(".dcm"))
    {
        tipl::io::dicom in;
        if(!in.load_from_file(filename.c_str()))
        {
            error_msg = "invalid dicom format";
            return false;
        }
        in >> I;
        src2.voxel.dim = I.shape();
        src2.src_dwi_data.push_back(&I[0]);
    }
    else
    if(QString(filename.c_str()).endsWith(".nii.gz") ||
       QString(filename.c_str()).endsWith(".nii"))
    {
        tipl::io::gz_nifti in;
        if(!in.load_from_file(filename.c_str()))
        {
            error_msg = "invalid NIFTI format";
            return false;
        }
        in.toLPS(I);
        src2.voxel.dim = I.shape();
        src2.src_dwi_data.push_back(&I[0]);
    }
    else
    {
        if (!src2.load_from_file(filename.c_str()))
        {
            error_msg = "cannot open ";
            error_msg += filename;
            error_msg += " : ";
            error_msg += src2.error_msg;
            return false;
        }
    }
    return true;
}

void reconstruction_window::on_actionCorrect_AP_PA_scans_triggered()
{
    QMessageBox::information(this,QApplication::applicationName(),"Please specify another SRC/DICOM/NIFTI file with an opposite phase encoding");
    QString filename = QFileDialog::getOpenFileName(
            this,"Open SRC file",absolute_path,
            "Images (*src.gz *.nii *nii.gz);;DICOM image (*.dcm);;All files (*)" );
    if( filename.isEmpty())
        return;

    if(!handle->distortion_correction(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error",handle->error_msg.c_str());
        return;
    }
    on_SlicePos_valueChanged(ui->SlicePos->value());
}



void reconstruction_window::on_actionEnable_TEST_features_triggered()
{
    ui->odf_resolving->setVisible(true);
    ui->align_slices->setVisible(true);
}

void reconstruction_window::on_actionImage_upsample_to_T1W_TESTING_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if( filenames.isEmpty())
        return;

    tipl::image<3> ref;
    tipl::vector<3> vs;
    tipl::matrix<4,4> t;
    if(!load_image_from_files(filenames,ref,vs,t))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                subject_image_pre(tipl::image<3>(ref)),tipl::image<3,unsigned char>(),vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
    tipl::progress prog_("rotating");
    handle->rotate(ref.shape(),vs,manual->get_iT(),tipl::image<3,tipl::vector<3> >());
    handle->voxel.report += " The diffusion images were rotated and scaled to the space of ";
    handle->voxel.report += QFileInfo(filenames[0]).baseName().toStdString();
    handle->voxel.report += ". The b-table was also rotated accordingly.";
    ui->report->setText(handle->voxel.report.c_str());

    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_SagView_clicked()
{
    view_orientation = 0;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_CorView_clicked()
{
    view_orientation = 1;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_AxiView_clicked()
{
    view_orientation = 2;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_show_bad_slice_clicked()
{
    if(!bad_slice_analyzed)
    {
        bad_slices = handle->get_bad_slices();
        bad_slice_analyzed = true;
        std::vector<char> is_bad(ui->b_table->rowCount());
        for(int i = 0;i < bad_slices.size();++i)
            if(bad_slices[i].first < is_bad.size())
                is_bad[bad_slices[i].first] = 1;

        for(int i = 0;i < ui->b_table->rowCount();++i)
            for(int j = 0;j < ui->b_table->columnCount();++j)
                ui->b_table->item(i, j)->setData(Qt::BackgroundRole,is_bad[i] ?  QColor (255,200,200): QColor (255,255,255));
    }
    if(bad_slices.size() == 0)
    {
        QMessageBox::information(this,QApplication::applicationName(),"No bad slice found in this data");
        return;
    }
    on_b_table_itemSelectionChanged();
    ui->bad_slice_label->setText(QString("A total %1 bad slices marked by red").arg(bad_slices.size()));

}

void reconstruction_window::on_align_slices_clicked()
{
    tipl::image<3> from(handle->voxel.dim);
    tipl::image<3> to(handle->voxel.dim);
    std::copy(handle->src_dwi_data[0],handle->src_dwi_data[0]+to.size(),to.begin());
    std::copy(handle->src_dwi_data[ui->b_table->currentRow()],
              handle->src_dwi_data[ui->b_table->currentRow()]+from.size(),from.begin());
    tipl::normalize(from);
    tipl::normalize(to);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                subject_image_pre(std::move(from)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                subject_image_pre(std::move(to)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    handle->rotate_one_dwi(ui->b_table->currentRow(),manual->get_iT());

    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());

}

void reconstruction_window::on_edit_mask_clicked()
{
    ui->edit_mask->setVisible(false);
    ui->mask_edit->setVisible(true);

}


void reconstruction_window::on_actionOverwrite_Voxel_Size_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,QApplication::applicationName(),"Assign voxel size in mm",
                                           QLineEdit::Normal,
                                           QString("%1 %2 %3").arg(double(handle->voxel.vs[0]))
                                                              .arg(double(handle->voxel.vs[1]))
                                                              .arg(double(handle->voxel.vs[2])),&ok);
    if(!ok)
        return;
    command("[Step T2][Edit][Overwrite Voxel Size]",result.toStdString());
    handle->get_report(handle->voxel.report);
    ui->report->setText(handle->voxel.report.c_str());
}

void match_template_resolution(tipl::image<3>& VG,
                               tipl::vector<3>& VGvs,
                               tipl::image<3>& VF,
                               tipl::vector<3>& VFvs)
{
    float ratio = float(VF.width())/float(VG.width());
    while(ratio < 0.5f)   // if subject resolution is substantially lower, downsample template
    {
        tipl::downsampling(VG);
        VGvs *= 2.0f;
        ratio *= 2.0f;
        tipl::out() << "ratio lower than 0.5, downsampling template to " << VGvs[0] << " mm resolution" << std::endl;
    }
    while(ratio > 2.5f)  // if subject resolution is higher, downsample it for registration
    {
        tipl::downsampling(VF);
        VFvs *= 2.0f;
        ratio /= 2.0f;
        tipl::out() << "ratio larger than 2.5, register using subject resolution of " << VFvs[0] << " mm resolution" << std::endl;
    }
}


void reconstruction_window::on_actionManual_Align_triggered()
{
    tipl::image<3> VG,VG2,VF(handle->dwi);
    tipl::vector<3> VGvs,VFvs(handle->voxel.vs);
    {
        tipl::io::gz_nifti read,read2;
        if(!read.load_from_file(fa_template_list[handle->voxel.template_id]))
        {
            QMessageBox::critical(this,"Error",QString("Cannot load template:"));
            return;
        }
        read.toLPS(VG);
        read.get_voxel_size(VGvs);
        if(read2.load_from_file(iso_template_list[handle->voxel.template_id]))
            read2.toLPS(VG2);
    }

    match_template_resolution(VG,VGvs,VF,VFvs);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                template_image_pre(VG),template_image_pre(VG2),VGvs,
                                                                subject_image_pre(VF),subject_image_pre(VF),VFvs,
                                                                tipl::reg::affine,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
    handle->voxel.qsdr_arg = manual->arg;
    handle->voxel.manual_alignment = true;
}

void reconstruction_window::on_mask_from_unet_clicked()
{
    handle->voxel.template_id = ui->primary_template->currentIndex();
    if(!handle->mask_from_unet())
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
    on_SlicePos_valueChanged(ui->SlicePos->value());
    raise();
}


