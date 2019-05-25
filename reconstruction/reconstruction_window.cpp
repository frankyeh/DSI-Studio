#include <QSplitter>
#include <QThread>
#include "reconstruction_window.h"
#include "ui_reconstruction_window.h"
#include "tipl/tipl.hpp"
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

extern std::vector<std::string> fa_template_list,iso_template_list;
void show_view(QGraphicsScene& scene,QImage I);
void populate_templates(QComboBox* combo);
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
    float m = (float)*std::max_element(handle->src_dwi_data[0],handle->src_dwi_data[0]+handle->voxel.dim.size());
    float otsu = tipl::segmentation::otsu_threshold(tipl::make_image(handle->src_dwi_data[0],handle->voxel.dim));
    ui->max_value->setMaximum(m*1.5f);
    ui->max_value->setMinimum(0.0f);
    ui->max_value->setSingleStep(m*0.05f);
    ui->max_value->setValue(otsu*3.0f);
    ui->min_value->setMaximum(m*1.5f);
    ui->min_value->setMinimum(0.0f);
    ui->min_value->setSingleStep(m*0.05f);
    ui->min_value->setValue(0.0f);
    load_b_table();
    return true;
}

void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_dsi(const std::vector<unsigned int>& shell);
bool is_multishell(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
extern std::vector<std::string> fa_template_list;
reconstruction_window::reconstruction_window(QStringList filenames_,QWidget *parent) :
    QMainWindow(parent),filenames(filenames_),ui(new Ui::reconstruction_window)
{
    ui->setupUi(this);
    if(!load_src(0))
        throw std::runtime_error("Cannot load src file");
    setWindowTitle(filenames[0]);
    ui->ThreadCount->setMaximum(std::thread::hardware_concurrency());
    ui->toolBox->setCurrentIndex(1);
    ui->graphicsView->setScene(&scene);
    ui->view_source->setScene(&source);
    ui->b_table->setColumnWidth(0,60);
    ui->b_table->setColumnWidth(1,80);
    ui->b_table->setColumnWidth(2,80);
    ui->b_table->setColumnWidth(3,80);
    ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");

    populate_templates(ui->primary_template);

    ui->primary_template->setCurrentIndex(0);

    v2c.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    update_dimension();

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

    ui->DT_Option->setVisible(false);
    ui->odf_resolving->setVisible(false);

    ui->AdvancedWidget->setVisible(false);
    ui->ThreadCount->setValue(settings.value("rec_thread_count",std::thread::hardware_concurrency()).toInt());
    ui->NumOfFibers->setValue(settings.value("rec_num_fiber",5).toInt());
    ui->ODFDef->setCurrentIndex(settings.value("rec_gqi_def",0).toInt());
    ui->diffusion_sampling->setValue(settings.value("rec_gqi_sampling",1.25).toDouble());
    ui->csf_calibration->setChecked(settings.value("csf_calibration",1).toInt());

    ui->odf_resolving->setChecked(settings.value("odf_resolving",0).toInt());
    ui->ODFDim->setCurrentIndex(settings.value("odf_order",3).toInt());

    ui->RecordODF->setChecked(settings.value("rec_record_odf",0).toInt());
    ui->output_jacobian->setChecked(settings.value("output_jacobian",0).toInt());
    ui->output_mapping->setChecked(settings.value("output_mapping",0).toInt());
    ui->output_diffusivity->setChecked(settings.value("output_diffusivity",1).toInt());
    ui->output_tensor->setChecked(settings.value("output_tensor",0).toInt());
    ui->output_helix_angle->setChecked(settings.value("output_helix_angle",0).toInt());
    ui->rdi->setChecked(settings.value("output_rdi",1).toInt());
    ui->check_btable->setChecked(settings.value("check_btable",1).toInt());

    ui->report->setText(handle->voxel.report.c_str());

    max_source_value = *std::max_element(handle->src_dwi_data.back(),
                                         handle->src_dwi_data.back()+handle->voxel.dim.size());



    connect(ui->z_pos,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->max_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->min_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));

    on_b_table_itemSelectionChanged();


    {
        ui->half_sphere->setChecked(handle->is_dsi_half_sphere());
        ui->scheme_balance->setChecked(handle->need_scheme_balance());
        if(handle->is_dsi())
            ui->scheme_balance->setEnabled(false);
        else
            ui->half_sphere->setEnabled(false);
    }

}
void reconstruction_window::update_dimension(void)
{
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
    source_ratio = std::max(1.0,500/(double)handle->voxel.dim.height());
}

void reconstruction_window::load_b_table(void)
{
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
void reconstruction_window::command(QString cmd)
{
    handle->command(cmd.toStdString(),"");
    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());
    steps.push_back(cmd.toStdString());
}
void reconstruction_window::on_b_table_itemSelectionChanged()
{
    v2c.set_range(ui->min_value->value(),ui->max_value->value());
    tipl::image<float,2> tmp;
    tipl::volume2slice(tipl::make_image(handle->src_dwi_data[ui->b_table->currentRow()],handle->voxel.dim),tmp,view_orientation,ui->z_pos->value());
    buffer_source.resize(tmp.geometry());
    for(int i = 0;i < tmp.size();++i)
        buffer_source[i] = v2c[tmp[i]];
    source_image = QImage((unsigned char*)&*buffer_source.begin(),tmp.width(),tmp.height(),QImage::Format_RGB32).
                    scaled(tmp.width()*source_ratio,tmp.height()*source_ratio);
    if(view_orientation != 2)
        source_image = source_image.mirrored();
    show_view(source,source_image);
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

void reconstruction_window::doReconstruction(unsigned char method_id,bool prompt)
{
    if(!handle.get())
        return;

    if (*std::max_element(handle->voxel.mask.begin(),handle->voxel.mask.end()) == 0)
    {
        QMessageBox::information(this,"error","Please select mask for reconstruction",0);
        return;
    }

    //QSDR
    if(method_id == 7)
    {
        if(fa_template_list.empty())
        {
            QMessageBox::information(this,"error","Cannot find template files",0);
            return;
        }
        handle->voxel.primary_template = fa_template_list[ui->primary_template->currentIndex()];
        handle->voxel.secondary_template = iso_template_list[ui->primary_template->currentIndex()];
    }

    settings.setValue("rec_method_id",method_id);
    settings.setValue("rec_thread_count",ui->ThreadCount->value());
    settings.setValue("rec_num_fiber",ui->NumOfFibers->value());
    settings.setValue("rec_gqi_def",ui->ODFDef->currentIndex());
    settings.setValue("csf_calibration",ui->csf_calibration->isChecked() ? 1 : 0);



    settings.setValue("odf_order",ui->ODFDim->currentIndex());

    settings.setValue("odf_resolving",ui->odf_resolving->isChecked() ? 1 : 0);
    settings.setValue("rec_record_odf",ui->RecordODF->isChecked() ? 1 : 0);
    settings.setValue("output_jacobian",ui->output_jacobian->isChecked() ? 1 : 0);
    settings.setValue("output_mapping",ui->output_mapping->isChecked() ? 1 : 0);
    settings.setValue("output_diffusivity",ui->output_diffusivity->isChecked() ? 1 : 0);
    settings.setValue("output_tensor",ui->output_tensor->isChecked() ? 1 : 0);
    settings.setValue("output_helix_angle",ui->output_helix_angle->isChecked() ? 1 : 0);

    settings.setValue("output_rdi",(ui->rdi->isChecked() && (method_id == 4 || method_id == 7)) ? 1 : 0); // only for GQI
    settings.setValue("check_btable",ui->check_btable->isChecked() ? 1 : 0);

    begin_prog("reconstruction",true);
    int odf_order[8] = {4, 5, 6, 8, 10, 12, 16, 20};
    handle->voxel.method_id = method_id;
    handle->voxel.ti.init(odf_order[ui->ODFDim->currentIndex()]);
    handle->voxel.odf_resolving = ui->odf_resolving->isChecked();
    handle->voxel.csf_calibration = (ui->csf_calibration->isVisible() && ui->csf_calibration->isChecked()) ? 1: 0;
    handle->voxel.max_fiber_number = ui->NumOfFibers->value();
    handle->voxel.r2_weighted = ui->ODFDef->currentIndex();
    handle->voxel.output_odf = ui->RecordODF->isChecked();
    handle->voxel.check_btable = ui->check_btable->isChecked();
    handle->voxel.output_jacobian = ui->output_jacobian->isChecked();
    handle->voxel.output_mapping = ui->output_mapping->isChecked();
    handle->voxel.output_diffusivity = ui->output_diffusivity->isChecked();
    handle->voxel.output_tensor = ui->output_tensor->isChecked();
    handle->voxel.output_helix_angle = ui->output_helix_angle->isChecked();

    handle->voxel.output_rdi = ui->rdi->isChecked();
    handle->voxel.thread_count = ui->ThreadCount->value();


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

    if(!handle->voxel.study_src_file_path.empty())
        handle->voxel.dt_deform = ui->dt_deform->isChecked();
    const char* msg = handle->reconstruction();
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

void reconstruction_window::on_load_mask_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open region",
            absolute_path,
            "Mask files (*.txt *.nii *nii.gz *.hdr);;All files (*)" );
    if(filename.isEmpty())
        return;
    ROIRegion region(std::make_shared<fib_data>(handle->dwi.geometry(),handle->voxel.vs));
    region.LoadFromFile(filename.toLocal8Bit().begin());
    region.SaveToBuffer(handle->voxel.mask);
    on_SlicePos_valueChanged(ui->SlicePos->value());
    handle->voxel.steps += std::string("[Step T2a][Open...]=") + QFileInfo(filename).fileName().toStdString()+"\n";
}


void reconstruction_window::on_save_mask_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save region",
            absolute_path+"/mask.txt",
            "Text files (*.txt);;Nifti file(*nii.gz *.nii);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(QFileInfo(filename.toLower()).completeSuffix() != "txt")
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
    ROIRegion region(std::make_shared<fib_data>(handle->dwi.geometry(),handle->voxel.vs));
    region.LoadFromBuffer(handle->voxel.mask);
    region.SaveToFile(filename.toLocal8Bit().begin());
}

void reconstruction_window::on_doDTI_clicked()
{
    std::vector<std::string> prior_steps(steps);
    for(int index = 0;index < filenames.size();++index)
    {
        if(index)
        {
            begin_prog("load src");
            if(!load_src(index))
                break;
            // apply the previous steps
            steps.clear();
            for(int j = 0;j < prior_steps.size();++j)
                command(prior_steps[j].c_str());
        }
        std::fill(handle->voxel.param.begin(),handle->voxel.param.end(),0.0);
        if(ui->DTI->isChecked())
            doReconstruction(1,index+1 == filenames.size());
        else
        if(ui->GQI->isChecked() || ui->QSDR->isChecked())
        {
            handle->voxel.param[0] = ui->diffusion_sampling->value();
            settings.setValue("rec_gqi_sampling",ui->diffusion_sampling->value());
            if(ui->QSDR->isChecked())
                doReconstruction(7,index+1 == filenames.size());
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
    ui->GQIOption_2->setVisible(!checked);

    ui->AdvancedOptions->setVisible(checked);
    ui->ODFOption->setVisible(!checked);
    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->output_tensor->setVisible(checked);
    ui->output_helix_angle->setVisible(checked);

    ui->output_diffusivity->setVisible(!checked);

    ui->RecordODF->setVisible(!checked);
    ui->rdi->setVisible(!checked);



}


void reconstruction_window::on_GQI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);


    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);
    ui->ODFOption->setVisible(checked);

    ui->output_mapping->setVisible(!checked);
    ui->output_jacobian->setVisible(!checked);
    ui->output_tensor->setVisible(!checked);
    ui->output_helix_angle->setVisible(!checked);
    ui->output_diffusivity->setVisible(checked);

    ui->RecordODF->setVisible(checked);

    ui->rdi->setVisible(checked);
    if(checked)
        ui->rdi->setChecked(true);
    ui->csf_calibration->setVisible(handle->is_human_data());
}
int match_template(float volume);
void reconstruction_window::on_QSDR_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(checked);
    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);
    ui->ODFOption->setVisible(checked);

    ui->output_mapping->setVisible(checked);
    ui->output_jacobian->setVisible(checked);
    ui->output_tensor->setVisible(!checked);
    ui->output_helix_angle->setVisible(!checked);

    ui->output_diffusivity->setVisible(checked);

    ui->RecordODF->setVisible(checked);
    ui->rdi->setVisible(checked);
    if(checked)
        ui->rdi->setChecked(true);

    ui->csf_calibration->setVisible(false);
    if(checked)
    {
        ui->primary_template->setCurrentIndex(match_template(
            handle->voxel.vs[0]*handle->voxel.vs[1]*handle->voxel.vs[2]*handle->voxel.dim.size()));
    }
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

void reconstruction_window::on_actionSave_b0_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                                this,
                                "Save image as...",
                            filenames[0] + ".b0.nii.gz",
                                "All files (*)" );
    if ( filename.isEmpty() )
        return;
    handle->save_b0_to_nii(filename.toLocal8Bit().begin());
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
    handle->save_dwi_sum_to_nii(filename.toLocal8Bit().begin());
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


bool load_image_from_files(QStringList filenames,tipl::image<float,3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4,float>&);
void reconstruction_window::on_actionRotate_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if( filenames.isEmpty())
        return;

    tipl::image<float,3> ref;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> t;
    if(!load_image_from_files(filenames,ref,vs,t))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                handle->dwi,handle->voxel.vs,ref,vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    begin_prog("rotating");
    tipl::image<float,3> ref2(ref);
    float m = tipl::median(ref2.begin(),ref2.end());
    tipl::multiply_constant_mt(ref,0.5f/m);
    handle->rotate(ref.geometry(),manual->iT);
    handle->voxel.vs = vs;
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
    unsigned int index = ui->b_table->currentRow();
    ui->b_table->removeRow(index);
    handle->remove(index);

}

void reconstruction_window::on_SlicePos_valueChanged(int position)
{
    handle->draw_mask(buffer,position);
    double ratio = std::max(1.0,
        std::min(((double)ui->graphicsView->width()-5)/(double)buffer.width(),
                 ((double)ui->graphicsView->height()-5)/(double)buffer.height()));
    slice_image = QImage((unsigned char*)&*buffer.begin(),buffer.width(),buffer.height(),QImage::Format_RGB32).
                    scaled(buffer.width()*ratio,buffer.height()*ratio);
    show_view(scene,slice_image);
}

void rec_motion_correction(ImageModel* handle)
{
    begin_prog("correcting motion...");
    tipl::par_for2(handle->src_bvalues.size(),[&](int i,int id)
    {
        if(i == 0 || prog_aborted() || handle->src_bvalues[i] > 1500)
            return;
        if(id == 0)
            check_prog(i*99/handle->src_bvalues.size(),100);
        tipl::affine_transform<double> arg;
        bool terminated = false;
        tipl::reg::linear_mr(tipl::make_image(handle->src_dwi_data[0],handle->voxel.dim),handle->voxel.vs,
                             tipl::make_image(handle->src_dwi_data[i],handle->voxel.dim),handle->voxel.vs,
                                  arg,tipl::reg::affine,tipl::reg::correlation(),terminated,0.0001);
        tipl::transformation_matrix<double> T(arg,handle->voxel.dim,handle->voxel.vs,handle->voxel.dim,handle->voxel.vs);
        handle->rotate_one_dwi(i,T);
    });
    check_prog(1,1);

}

void reconstruction_window::on_motion_correction_clicked()
{
    rec_motion_correction(handle.get());
    if(!prog_aborted())
    {
        handle->calculate_dwi_sum();
        handle->voxel.calculate_mask(handle->dwi_sum);
        load_b_table();
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

bool add_other_image(ImageModel* handle,QString name,QString filename,bool full_auto)
{
    tipl::image<float,3> ref;
    tipl::vector<3> vs;
    gz_nifti in;
    if(!in.load_from_file(filename.toLocal8Bit().begin()) || !in.toLPS(ref))
    {
        if(full_auto)
            std::cout << "Not a valid nifti file:" << filename.toStdString() << std::endl;
        else
            QMessageBox::information(0,"Error","Not a valid nifti file",0);
        return false;
    }
    tipl::transformation_matrix<double> affine;
    bool has_registered = false;
    for(unsigned int index = 0;index < handle->voxel.other_image.size();++index)
        if(ref.geometry() == handle->voxel.other_image[index].geometry())
        {
            affine = handle->voxel.other_image_affine[index];
            has_registered = true;
        }
    if(!has_registered && ref.geometry() != handle->voxel.dim)
    {
        in.get_voxel_size(vs);
        if(full_auto)
        {
            std::cout << "add " << filename.toStdString() << " as " << name.toStdString() << std::endl;
            tipl::image<float,3> from(handle->dwi_sum),to(ref);
            tipl::normalize(from,1.0);
            tipl::normalize(to,1.0);
            bool terminated = false;
            tipl::affine_transform<float> arg;
            tipl::reg::linear_mr(from,handle->voxel.vs,to,vs,arg,tipl::reg::rigid_body,tipl::reg::mutual_information(),terminated,0.1);
            tipl::reg::linear_mr(from,handle->voxel.vs,to,vs,arg,tipl::reg::rigid_body,tipl::reg::mutual_information(),terminated,0.01);
            affine = tipl::transformation_matrix<float>(arg,handle->voxel.dim,handle->voxel.vs,to.geometry(),vs);
        }
        else
        {
            std::shared_ptr<manual_alignment> manual(new manual_alignment(0,
                        handle->dwi_sum,handle->voxel.vs,ref,vs,tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
            manual->on_rerun_clicked();
            if(manual->exec() != QDialog::Accepted)
                return false;
            affine = manual->T;
        }

    }
    handle->voxel.other_image.push_back(ref);
    handle->voxel.other_image_name.push_back(name.toLocal8Bit().begin());
    handle->voxel.other_image_affine.push_back(affine);
    return true;
}

void reconstruction_window::on_add_t1t2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    add_other_image(handle.get(),QFileInfo(filename).baseName(),filename,false);
}

void reconstruction_window::on_actionManual_Rotation_triggered()
{
    std::shared_ptr<manual_alignment> manual(
                new manual_alignment(this,handle->dwi,handle->voxel.vs,handle->dwi,handle->voxel.vs,tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
    if(manual->exec() != QDialog::Accepted)
        return;
    begin_prog("rotating");
    handle->rotate(handle->dwi.geometry(),manual->iT);
    load_b_table();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}



void reconstruction_window::on_actionReplace_b0_by_T2W_image_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    tipl::image<float,3> ref;
    tipl::vector<3> vs;
    gz_nifti in;
    if(!in.load_from_file(filename.toLocal8Bit().begin()) || !in.toLPS(ref))
    {
        QMessageBox::information(this,"Error","Not a valid nifti file",0);
        return;
    }
    in.get_voxel_size(vs);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,handle->dwi,handle->voxel.vs,ref,vs,tipl::reg::rigid_body,tipl::reg::cost_type::corr));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    begin_prog("rotating");
    handle->rotate(ref.geometry(),manual->iT);
    handle->voxel.vs = vs;
    tipl::pointer_image<unsigned short,3> I = tipl::make_image((unsigned short*)handle->src_dwi_data[0],handle->voxel.dim);
    ref *= (float)(*std::max_element(I.begin(),I.end()))/(*std::max_element(ref.begin(),ref.end()));
    std::copy(ref.begin(),ref.end(),I.begin());
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_actionCorrect_AP_PA_scans_triggered()
{
    QMessageBox::information(this,"DSI Studio","Please assign another SRC file with phase encoding flipped",0);
    QString filename = QFileDialog::getOpenFileName(
            this,"Open SRC file",absolute_path,
            "Images (*src.gz);;All files (*)" );
    if( filename.isEmpty())
        return;

    begin_prog("load src");
    ImageModel src2;
    if (!src2.load_from_file(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error",QString("Cannot open ") +
           filename + " : " +src2.error_msg.c_str(),0);
        check_prog(0,0);
        return;
    }
    check_prog(0,0);
    if(handle->voxel.dim != src2.voxel.dim)
    {
        QMessageBox::information(this,"error","The image dimension is different.",0);
        return;
    }
    if(handle->src_dwi_data.size() != src2.src_dwi_data.size())
    {
        QMessageBox::information(this,"error","The DWI number is different.",0);
        return;
    }
    handle->distortion_correction(src2);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}



void reconstruction_window::on_actionEnable_TEST_features_triggered()
{
    ui->DT_Option->setVisible(true);
    ui->odf_resolving->setVisible(true);
}

void reconstruction_window::on_actionImage_upsample_to_T1W_TESTING_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if( filenames.isEmpty())
        return;

    tipl::image<float,3> ref;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> t;
    if(!load_image_from_files(filenames,ref,vs,t))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                handle->dwi,handle->voxel.vs,ref,vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    begin_prog("rotating");
    tipl::image<float,3> ref2(ref);
    float m = tipl::median(ref2.begin(),ref2.end());
    tipl::multiply_constant_mt(ref,0.5f/m);

    handle->rotate(ref.geometry(),manual->iT);
    handle->voxel.vs = vs;
    handle->voxel.report += " The diffusion images were rotated and scaled to the space of ";
    handle->voxel.report += QFileInfo(filenames[0]).baseName().toStdString();
    handle->voxel.report += ". The b-table was also rotated accordingly.";
    ui->report->setText(handle->voxel.report.c_str());

    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}


void reconstruction_window::on_open_ddi_study_src_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Study SRC file",absolute_path,
            "Images (*src.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    handle->voxel.study_src_file_path = filename.toStdString();
    ui->ddi_file->setText(QFileInfo(filename).baseName());
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

void reconstruction_window::on_actionResample_triggered()
{
    bool ok;
    float nv = float(QInputDialog::getDouble(this,
        "DSI Studio","Assign output resolution in (mm):", double(handle->voxel.vs[0]),0.0,3.0,4, &ok));
    if (!ok || nv == 0.0f)
        return;

    handle->command("[Step T2][Edit][Resample]",QString::number(nv).toStdString());
    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
