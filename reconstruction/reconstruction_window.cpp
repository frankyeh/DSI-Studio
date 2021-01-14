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
    prog_init p("load src");
    check_prog(index,filenames.size());
    handle.reset(new ImageModel);
    if (!handle->load_from_file(filenames[index].toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"error",QString("Cannot open ") +
            filenames[index] + " : " +handle->error_msg.c_str(),0);
        return false;
    }
    double m = double(*std::max_element(handle->src_dwi_data[0],handle->src_dwi_data[0]+handle->voxel.dim.size()));
    double otsu = double(tipl::segmentation::otsu_threshold(tipl::make_image(handle->src_dwi_data[0],handle->voxel.dim)));
    ui->max_value->setMaximum(m*1.5);
    ui->max_value->setMinimum(0.0);
    ui->max_value->setSingleStep(m*0.05);
    ui->max_value->setValue(otsu*3.0);
    ui->min_value->setMaximum(m*1.5);
    ui->min_value->setMinimum(0.0);
    ui->min_value->setSingleStep(m*0.05);
    ui->min_value->setValue(0.0);
    load_b_table();

    ui->align_slices->setVisible(false);
    return true;
}

void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool is_dsi(const std::vector<unsigned int>& shell);
bool is_multishell(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
size_t match_template(float volume);
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
    ui->primary_template->setCurrentIndex(int(match_template(
        handle->voxel.vs[0]*handle->voxel.vs[1]*handle->voxel.vs[2]*handle->voxel.dim.size())));

    if(ui->primary_template->currentIndex() == 0)
        ui->diffusion_sampling->setValue(1.25); // human studies
    else
        ui->diffusion_sampling->setValue(0.6);  // animal studies (likely ex-vivo)

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

    ui->odf_resolving->setVisible(false);

    ui->AdvancedWidget->setVisible(false);
    ui->ThreadCount->setValue(settings.value("rec_thread_count",std::thread::hardware_concurrency()).toInt());
    ui->csf_calibration->setChecked(settings.value("csf_calibration",1).toInt());

    ui->odf_resolving->setChecked(settings.value("odf_resolving",0).toInt());

    ui->RecordODF->setChecked(settings.value("rec_record_odf",0).toInt());
    ui->output_tensor->setChecked(settings.value("output_tensor",0).toInt());
    ui->output_helix_angle->setChecked(settings.value("output_helix_angle",0).toInt());
    ui->not_human_brain->setChecked(settings.value("not_human_brain",0).toInt());
    ui->check_btable->setChecked(settings.value("check_btable",1).toInt());
    if(handle->voxel.vs[2] > handle->voxel.vs[0]*2.0f || handle->voxel.vs[0] < 0.5f)
        ui->check_btable->setChecked(false);
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

    ui->mask_edit->setVisible(false);

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

void reconstruction_window::on_b_table_itemSelectionChanged()
{
    v2c.set_range(ui->min_value->value(),ui->max_value->value());
    tipl::image<float,2> tmp;
    tipl::volume2slice(tipl::make_image(handle->src_dwi_data[ui->b_table->currentRow()],handle->voxel.dim),tmp,view_orientation,ui->z_pos->value());
    buffer_source.resize(tmp.geometry());
    for(int i = 0;i < tmp.size();++i)
        buffer_source[i] = v2c[tmp[i]];

    // show bad_slices
    if(view_orientation != 2 && bad_slice_analzed)
    {
        std::vector<size_t> mark_slices;
        for(size_t index = 0;index < bad_slices.size();++index)
            if(bad_slices[index].first == ui->b_table->currentRow())
                mark_slices.push_back(bad_slices[index].second);
        for(size_t index = 0;index < mark_slices.size();++index)
        {
            for(size_t x = 0,pos = mark_slices[index]*buffer_source.width();x < buffer_source.width();++x,++pos)
                buffer_source[pos].r |= 64;
        }
    }
    if(view_orientation == 2 && bad_slice_analzed)
    {
        std::vector<size_t> mark_slices;
        for(size_t index = 0;index < bad_slices.size();++index)
            if(bad_slices[index].first == ui->b_table->currentRow() && ui->z_pos->value() == bad_slices[index].second)
            {
                for(size_t i = 0;i < buffer_source.size();++i)
                    buffer_source[i].r |= 64;
                break;
            }
    }

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
    settings.setValue("csf_calibration",ui->csf_calibration->isChecked() ? 1 : 0);

    settings.setValue("odf_resolving",ui->odf_resolving->isChecked() ? 1 : 0);
    settings.setValue("rec_record_odf",ui->RecordODF->isChecked() ? 1 : 0);
    settings.setValue("output_tensor",ui->output_tensor->isChecked() ? 1 : 0);
    settings.setValue("output_helix_angle",ui->output_helix_angle->isChecked() ? 1 : 0);
    settings.setValue("not_human_brain",ui->not_human_brain->isChecked() ? 1 : 0);
    settings.setValue("check_btable",ui->check_btable->isChecked() ? 1 : 0);

    begin_prog("reconstruction",true);
    handle->voxel.method_id = method_id;
    handle->voxel.ti.init(8);
    handle->voxel.odf_resolving = ui->odf_resolving->isChecked();
    handle->voxel.csf_calibration = (ui->csf_calibration->isVisible() && ui->csf_calibration->isChecked()) ? 1: 0;
    handle->voxel.output_odf = ui->RecordODF->isChecked();
    handle->voxel.not_human_brain = ui->not_human_brain->isChecked();
    handle->voxel.check_btable = ui->check_btable->isChecked();
    handle->voxel.output_tensor = ui->output_tensor->isChecked();
    handle->voxel.output_helix_angle = ui->output_helix_angle->isChecked();

    handle->voxel.output_rdi = (method_id == 4 || method_id == 7);
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

    auto dim_backup = handle->voxel.dim; // for QSDR
    auto vs = handle->voxel.vs; // for QSDR
    if (!handle->reconstruction())
    {
        QMessageBox::information(this,"ERROR",handle->error_msg.c_str());
        return;
    }
    handle->voxel.dim = dim_backup;
    handle->voxel.vs = vs;
    if(method_id == 7) // QSDR
        handle->calculate_dwi_sum(true);
    if(!prompt)
        return;
    QMessageBox::information(this,"DSI Studio","FIB file created.");
    raise(); // for Mac
    QString filename = handle->file_name.c_str();
    filename += handle->get_file_ext().c_str();
    if(method_id == 6)
        ((MainWindow*)parent())->addSrc(filename);
    else
        ((MainWindow*)parent())->addFib(filename);
}

void reconstruction_window::on_load_mask_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open region",
            absolute_path,
            "Mask files (*.nii *nii.gz *.hdr);;Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    fib_data fib(handle->dwi.geometry(),handle->voxel.vs);
    ROIRegion region(&fib);
    region.LoadFromFile(filename.toLocal8Bit().begin());
    region.SaveToBuffer(handle->voxel.mask,1.0f);
    on_SlicePos_valueChanged(ui->SlicePos->value());
    handle->voxel.steps += std::string("[Step T2a][Open...]=") + QFileInfo(filename).fileName().toStdString()+"\n";
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
    fib_data fib(handle->dwi.geometry(),handle->voxel.vs);
    ROIRegion region(&fib);
    region.LoadFromBuffer(handle->voxel.mask);
    region.SaveToFile(filename.toLocal8Bit().begin());
}
void reconstruction_window::on_actionFlip_bx_triggered()
{
    command("[Step T2][Edit][Change b-table:flip bx]");
    ui->check_btable->setChecked(false);
    QMessageBox::information(this,"DSI Studio","B-table flipped",0);
}
void reconstruction_window::on_actionFlip_by_triggered()
{
    command("[Step T2][Edit][Change b-table:flip by]");
    ui->check_btable->setChecked(false);
    QMessageBox::information(this,"DSI Studio","B-table flipped",0);
}
void reconstruction_window::on_actionFlip_bz_triggered()
{
    command("[Step T2][Edit][Change b-table:flip bz]");
    ui->check_btable->setChecked(false);
    QMessageBox::information(this,"DSI Studio","B-table flipped",0);
}

void reconstruction_window::command(std::string cmd,std::string param)
{
    handle->command(cmd,param);
    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::on_doDTI_clicked()
{

    for(int index = 0;index < filenames.size();++index)
    {
        if(index)
        {
            std::istringstream in(handle->voxel.steps);
            begin_prog("load src");
            if(!load_src(index))
                break;
            std::string step;
            std::getline(in,step); // ignore the first step [Step T2][Reconstruction]
            while(std::getline(in,step))
            {
                size_t pos = step.find('=');
                if(pos == std::string::npos)
                    command(step);
                else
                    command(step.substr(0,pos),step.substr(pos+1,step.size()-pos-1));
            }
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
    ui->output_tensor->setVisible(checked);
    ui->output_helix_angle->setVisible(checked);

    ui->RecordODF->setVisible(!checked);
    ui->DT_Option->setVisible(!checked);

}


void reconstruction_window::on_GQI_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(!checked);


    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_tensor->setVisible(!checked);
    ui->output_helix_angle->setVisible(!checked);

    ui->RecordODF->setVisible(checked);
    ui->csf_calibration->setVisible(handle->is_human_data());
    ui->DT_Option->setVisible(checked);
}

void reconstruction_window::on_QSDR_toggled(bool checked)
{
    ui->ResolutionBox->setVisible(checked);
    ui->GQIOption_2->setVisible(checked);

    ui->AdvancedOptions->setVisible(checked);

    ui->output_tensor->setVisible(!checked);
    ui->output_helix_angle->setVisible(!checked);

    ui->RecordODF->setVisible(checked);

    ui->csf_calibration->setVisible(false);

    ui->DT_Option->setVisible(!checked);
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
        prog_init p("loading");
        for(int index = 0;check_prog(index,filenames.size());++index)
        {
            ImageModel model;
            if (!model.load_from_file(filenames[index].toLocal8Bit().begin()))
            {
                QMessageBox::information(this,"error",QString("Cannot open ") +
                    filenames[index] + " : " +handle->error_msg.c_str(),0);
                return;
            }
            model.save_to_nii((filenames[index]+".nii.gz").toLocal8Bit().begin());
            model.save_bval((filenames[index]+".bval").toLocal8Bit().begin());
            model.save_bvec((filenames[index]+".bvec").toLocal8Bit().begin());
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
    QString basename = filename;
    basename.chop(7);
    handle->save_bval((basename+".bval").toLocal8Bit().begin());
    handle->save_bvec((basename+".bvec").toLocal8Bit().begin());
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
    int index = ui->b_table->currentRow();
    if(index <= 0)
        return;
    bad_slice_analzed = false;
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
    bad_slice_analzed = false;
    while(ui->b_table->rowCount() > index)
    {
        ui->b_table->removeRow(index);
        handle->remove(uint32_t(index));
    }
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
    int down_size = 0;
    tipl::image<float,3> from(handle->voxel.dim);
    std::copy(handle->src_dwi_data[0],handle->src_dwi_data[0]+from.size(),from.begin());
    for(int width = from.width();width > 128;width /= 2)
        down_size++;
    for(int i = 0;i < down_size;++i)
        tipl::downsampling(from);
    tipl::filter::sobel(from);
    tipl::normalize(from,1.0f);
    tipl::par_for2(handle->src_bvalues.size(),[&](unsigned int i,int id)
    {
        if(i == 0 || prog_aborted())
            return;
        if(id == 0)
            check_prog(i*99/handle->src_bvalues.size(),100);
        tipl::affine_transform<double> arg;
        bool terminated = false;

        tipl::image<float,3> to(handle->voxel.dim);
        std::copy(handle->src_dwi_data[i],
                  handle->src_dwi_data[i]+to.size(),to.begin());

        for(int i = 0;i < down_size;++i)
            tipl::downsampling(to);
        tipl::filter::sobel(to);
        tipl::normalize(to,1.0f);
        arg.translocation[0] = 0.05;
        tipl::reg::linear(from,handle->voxel.vs,to,handle->voxel.vs,
                                  arg,tipl::reg::affine,tipl::reg::correlation(),terminated,0.01,0,tipl::reg::narrow_bound);
        tipl::reg::linear(from,handle->voxel.vs,to,handle->voxel.vs,
                                  arg,tipl::reg::affine,tipl::reg::correlation(),terminated,0.001,0,tipl::reg::narrow_bound);
        for(int i = 0;i < down_size;++i)
        {
            arg.translocation[0] *= 2.0;
            arg.translocation[1] *= 2.0;
            arg.translocation[2] *= 2.0;
        }
        tipl::transformation_matrix<double> T(arg,handle->voxel.dim,handle->voxel.vs,
                                                  handle->voxel.dim,handle->voxel.vs);
        handle->rotate_one_dwi(i,T);
    });
    check_prog(1,1);

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

bool add_other_image(ImageModel* handle,QString name,QString filename)
{
    tipl::image<float,3> ref;
    tipl::vector<3> vs;
    gz_nifti in;
    if(!in.load_from_file(filename.toLocal8Bit().begin()) || !in.toLPS(ref))
    {
        std::cout << "not a valid nifti file:" << filename.toStdString() << std::endl;
        return false;
    }
    tipl::transformation_matrix<double> affine;
    bool has_registered = false;
    for(unsigned int index = 0;index < handle->voxel.other_image.size();++index)
        if(ref.geometry() == handle->voxel.other_image[index].geometry())
        {
            affine = handle->voxel.other_image_trans[index];
            has_registered = true;
        }
    if(!has_registered && ref.geometry() != handle->voxel.dim)
    {
        in.get_voxel_size(vs);
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
    handle->voxel.other_image.push_back(ref);
    handle->voxel.other_image_name.push_back(name.toLocal8Bit().begin());
    handle->voxel.other_image_trans.push_back(affine);
    return true;
}

void reconstruction_window::on_add_t1t2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Images files",absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    if(add_other_image(handle.get(),QFileInfo(filename).baseName(),filename))
        QMessageBox::information(this,"DSI Studio","File added");
    else
        QMessageBox::information(this,"Error","Not a valid nifti file");

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

bool get_src(std::string filename,ImageModel& src2,std::string& error_msg)
{
    prog_init p("load ",filename.c_str());
    tipl::image<unsigned short,3> I;
    if(QString(filename.c_str()).endsWith(".dcm"))
    {
        tipl::io::dicom in;
        if(!in.load_from_file(filename.c_str()))
        {
            error_msg = "invalid dicom format";
            return false;
        }
        in >> I;
        src2.voxel.dim = I.geometry();
        src2.src_dwi_data.push_back(&I[0]);
    }
    else
    if(QString(filename.c_str()).endsWith(".nii.gz") ||
       QString(filename.c_str()).endsWith(".nii"))
    {
        gz_nifti in;
        if(!in.load_from_file(filename.c_str()))
        {
            error_msg = "invalid NIFTI format";
            return false;
        }
        in.toLPS(I);
        src2.voxel.dim = I.geometry();
        src2.src_dwi_data.push_back(&I[0]);
    }
    else {
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
    QMessageBox::information(this,"DSI Studio","Please assign another SRC/DICOM/NIFTI file with an opposite phase encoding",0);
    QString filename = QFileDialog::getOpenFileName(
            this,"Open SRC file",absolute_path,
            "Images (*src.gz);;DICOM image (*.dcm);;NIFTI image (*.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;

    ImageModel src2;
    std::string msg;
    if(!get_src(filename.toStdString(),src2,msg))
    {
        QMessageBox::information(this,"Error",msg.c_str());
        return;
    }
    if(handle->voxel.dim != src2.voxel.dim)
    {
        QMessageBox::information(this,"Error","inconsistent appa image dimension");
        return;
    }
    handle->distortion_correction(src2);
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
    command("[Step T2b(2)][Advanced Options][Compare SRC]",filename.toStdString());
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
    command("[Step T2][Edit][Resample]",QString::number(nv).toStdString());
}

void reconstruction_window::on_actionSave_SRC_file_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save SRC file",filenames[0],
            "SRC files (*src.gz);;All files (*)" );
    if(filename.isEmpty())
        return;
    prog_init p("saving ",QFileInfo(filename).fileName().toStdString().c_str());
    handle->save_to_file(filename.toStdString().c_str());
}


void reconstruction_window::on_actionEddy_Motion_Correction_triggered()
{
    rec_motion_correction(handle.get());

    if(!prog_aborted())
    {
        handle->calculate_dwi_sum(true);
        load_b_table();
        on_SlicePos_valueChanged(ui->SlicePos->value());
    }
}

void reconstruction_window::on_show_bad_slice_clicked()
{
    if(!bad_slice_analzed)
    {
        bad_slices = handle->get_bad_slices();
        bad_slice_analzed = true;
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
        QMessageBox::information(this,"DSI Studio","No bad slice found in this data");
        return;
    }
    on_b_table_itemSelectionChanged();
    ui->bad_slice_label->setText(QString("A total %1 bad slices marked by red").arg(bad_slices.size()));

}

void reconstruction_window::on_align_slices_clicked()
{
    tipl::image<float,3> from(handle->voxel.dim);
    tipl::image<float,3> to(handle->voxel.dim);
    std::copy(handle->src_dwi_data[0],handle->src_dwi_data[0]+to.size(),to.begin());
    std::copy(handle->src_dwi_data[ui->b_table->currentRow()],
              handle->src_dwi_data[ui->b_table->currentRow()]+from.size(),from.begin());
    tipl::normalize(from,1.0f);
    tipl::normalize(to,1.0f);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                from,handle->voxel.vs,
                                                                to,handle->voxel.vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    handle->rotate_one_dwi(ui->b_table->currentRow(),manual->iT);

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
    QString result = QInputDialog::getText(this,"DSI Studio","Assign voxel size in mm",
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
