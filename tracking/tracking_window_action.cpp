#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QClipboard>
#include <QMessageBox>

#include "atlasdialog.h"
#include "tracking_window.h"
#include "opengl/renderingtablewidget.h"
#include "ui_tracking_window.h"
#include "region/regiontablewidget.h"
#include "opengl/glwidget.h"
#include "tract_report.hpp"
#include "connectivity_matrix_dialog.h"
#include "mapping/atlas.hpp"
#include "manual_alignment.h"
#include "devicetablewidget.h"
#include "libs/tracking/tracking_thread.hpp"
#include "reg.hpp"

extern std::vector<std::string> fa_template_list;
std::string show_info_dialog(const std::string& title, const std::string& result,const std::string& file_name_hint)
{
    std::string saved_file;
    QMessageBox msgBox;
    msgBox.setText(title.c_str());

    // Set a brief summary in the main message box text
    std::string summary = result.length() > 200 ? result.substr(0, 200) + "..." : result;
    msgBox.setInformativeText(summary.c_str());
    msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Save);
    msgBox.setDefaultButton(QMessageBox::Ok);

    QPushButton *copyButton = msgBox.addButton("Copy To Clipboard", QMessageBox::ActionRole);
    QPushButton *viewDetailsButton = msgBox.addButton("View Full Text", QMessageBox::ActionRole);

    if (msgBox.exec() == QMessageBox::Save) {
        QString filename = QFileDialog::getSaveFileName(nullptr, "Save as",
                           QString::fromStdString(file_name_hint), "Text files (*.txt);;All files (*)");
        if (!filename.isEmpty()) {
            std::ofstream out((saved_file = filename.toStdString()).c_str());
            out << result;
        }
    }

    if (msgBox.clickedButton() == copyButton) {
        QApplication::clipboard()->setText(QString::fromStdString(result));
    }

    if (msgBox.clickedButton() == viewDetailsButton) {
        // Show the full text in a separate scrollable dialog
        QDialog detailDialog;
        detailDialog.setWindowTitle("Detailed Text");
        detailDialog.resize(600, 400);

        QVBoxLayout layout(&detailDialog);
        QTextEdit *textEdit = new QTextEdit(&detailDialog);
        textEdit->setReadOnly(true);
        textEdit->setText(QString::fromStdString(result));
        layout.addWidget(textEdit);

        QPushButton closeButton("Close", &detailDialog);
        layout.addWidget(&closeButton);
        QObject::connect(&closeButton, &QPushButton::clicked, &detailDialog, &QDialog::accept);

        detailDialog.exec();
    }
    return saved_file;
}

void tracking_window::run_action(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action || !action->toolTip().startsWith("run "))
        return;
    if(!command({action->toolTip().toStdString().substr(4)})) // skip "run " part
    {
        if(!error_msg.empty() && error_msg != "canceled")
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
    }
    else
        if(action->toolTip().contains("save_"))
            QMessageBox::information(this,QApplication::applicationName(),"file saved");
}
extern std::vector<tracking_window*> tracking_windows;
bool tracking_window::command(std::vector<std::string> cmd)
{
    if(glWidget->command(cmd))
        return true;
    if(!glWidget->error_msg.empty() && glWidget->error_msg != "not_processed")
    {
        error_msg = glWidget->error_msg;
        return false;
    }
    if(tractWidget->command(cmd))
        return true;
    if(!tractWidget->error_msg.empty() && tractWidget->error_msg != "not_processed")
    {
        error_msg = tractWidget->error_msg;
        return false;
    }
    if(regionWidget->command(cmd))
        return true;
    if(!regionWidget->error_msg.empty() && regionWidget->error_msg != "not_processed")
    {
        error_msg = regionWidget->error_msg;
        return false;
    }

    auto run = history.record(error_msg,cmd);
    cmd.resize(3);
    if(cmd[0] == "open_fib")
    {
        std::shared_ptr<fib_data> new_handle(new fib_data);
        if(!new_handle->load_from_file(cmd[1]))
            return run->failed(new_handle->error_msg);
        tracking_windows.push_back(new tracking_window(parentWidget(),new_handle));
        tracking_windows.back()->setAttribute(Qt::WA_DeleteOnClose);
        tracking_windows.back()->setWindowTitle(cmd[1].c_str());
        tracking_windows.back()->showNormal();
        tracking_windows.back()->resize(size().width(),size().height());
        return true;
    }
    if(cmd[0] == "save_fib_as")
    {
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(this,"Save FIB file",
           windowTitle().replace(".fib.gz",".fz"),"FIB files (*.fz);;All files (*)").toStdString()).empty())
            return run->canceled();
        if(!handle->save_to_file(cmd[1]))
            return run->failed(handle->error_msg);

        return true;
    }
    if(cmd[0] == "open_mapping")
    {
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getOpenFileName(
                    this,"Open MNI mapping",QFileInfo(work_path).absolutePath(),
                    "Mapping file(*.mz);;All file types (*)" ).toStdString()).empty())
            return run->canceled();
        tipl::progress prog(cmd[0],true);
        if(!handle->load_template() || !handle->load_mapping(cmd[1]))
            return run->failed(handle->error_msg);
        return true;
    }
    if(cmd[0] == "save_roi_screen")
    {
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(
                    0,"Save Images files",
                    regionWidget->currentRow() >= 0 ?
                    regionWidget->item(regionWidget->currentRow(),0)->text()+".png" :
                    QFileInfo(windowTitle()).baseName()+"_"+ui->SliceModality->currentText()+".jpg",
                    "Image files (*.png *.bmp *.jpg);;All files (*)").toStdString()).empty())
            return run->canceled();

        slice_need_update = false; // turn off simple drawing
        scene.paint_image(scene.view_image,false);     
        if(!scene.view_image.save(cmd[1].c_str()))
            return run->failed("cannot save mapping to " + cmd[1]);
        return true;
    }
    if(cmd[0] == "save_slice")
    {
        if(cmd[2].empty())
        {
            QAction *action = qobject_cast<QAction *>(sender());
            if(!action)
                return run->canceled();

            cmd[2] = action->data().toString().toStdString();
        }
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(
                    this,"Save as",
                    QFileInfo(windowTitle()).baseName()+"_"+ QString::fromStdString(cmd[2])+".nii.gz",
                    "NIFTI files (*nii.gz *.nii);;MAT files (*.mat);;All files (*)").toStdString()).empty())
            return run->canceled();

        if(!handle->save_slice(cmd[2],cmd[1]))
            return run->failed(cmd[2] + " not found or cannot save it to " + cmd[1]);

        return true;
    }
    if(cmd[0] == "save_all_devices")
    {
        if (deviceWidget->devices.empty())
            return run->canceled();
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(
                               this,"Save all devices",deviceWidget->item(deviceWidget->currentRow(),0)->text() + ".dv.csv",
                               "CSV file(*dv.csv);;All files(*)").toStdString()).empty())
            return run->canceled();

        std::ofstream out(cmd[1]);
        for (size_t i = 0; i < deviceWidget->devices.size(); ++i)
            if (deviceWidget->item(int(i),0)->checkState() == Qt::Checked)
            {
                deviceWidget->devices[i]->name = deviceWidget->item(int(i),0)->text().toStdString();
                out << deviceWidget->devices[i]->to_str();
            }
        return true;
    }
    if(cmd[0] == "presentation_mode")
    {
        ui->ROIdockWidget->hide();
        if(!regionWidget->rowCount())
            ui->regionDockWidget->hide();
        return true;
    }
    if(cmd[0] == "save_workspace")
    {
        if(cmd[1].empty() && (cmd[1] =
            QFileDialog::getExistingDirectory(this,"Specify Workspace Directory",QFileInfo(windowTitle()).absolutePath()).toStdString()).empty())
            return run->canceled();

        std::filesystem::create_directory(cmd[1]);
        if (!std::filesystem::exists(cmd[1]) || !std::filesystem::is_directory(cmd[1]))
            return run->failed("cannot save workspace to " + cmd[1]);

        if(tractWidget->rowCount())
        {
            std::filesystem::create_directory(cmd[1]+"/tracts");
            tractWidget->command({"save_all_tracts_to_dir",cmd[1]+"/tracts"});
        }
        if(regionWidget->rowCount())
        {
            std::filesystem::create_directory(cmd[1]+"/regions");
            regionWidget->command({"save_all_regions_to_dir",cmd[1]+"/regions"});
        }
        if(deviceWidget->rowCount())
        {
            std::filesystem::create_directory(cmd[1]+"/devices");
            command({"save_all_devices",cmd[1]+"/devices/device.dv.csv"});
        }
        auto reg_slice = dynamic_cast<CustomSliceModel*>(current_slice.get());
        if(reg_slice)
        {
            std::filesystem::create_directory(cmd[1]+"/slices");
            auto I = reg_slice->source_images;
            tipl::normalize_upper_lower(I,255.99);
            tipl::image<3,unsigned char> II(I.shape());
            std::copy(I.begin(),I.end(),II.begin());
            tipl::io::gz_nifti::save_to_file((cmd[1]+"/slices/" + ui->SliceModality->currentText().toStdString() + ".nii.gz").c_str(),
                                   II,reg_slice->vs,reg_slice->trans_to_mni,reg_slice->is_mni);
            reg_slice->save_mapping((cmd[1]+"/slices/" + ui->SliceModality->currentText().toStdString() + ".linear_reg.txt").c_str());
        }

        command({"save_setting",cmd[1] + "/setting.ini"});
        command({"save_camera",cmd[1] + "/camera.txt"});

        std::ofstream out(cmd[1] + "/command.txt");
        if(ui->glSagCheck->checkState())
            out << "move_slice 0 " << current_slice->slice_pos[0] << std::endl;
        else
            out << "slice_off 0" << std::endl;
        if(ui->glCorCheck->checkState())
            out << "move_slice 1 " << current_slice->slice_pos[1] << std::endl;
        else
            out << "slice_off 1" << std::endl;
        if(ui->glAxiCheck->checkState())
            out << "move_slice 2 " << current_slice->slice_pos[2] << std::endl;
        else
            out << "slice_off 2" << std::endl;

        return true;

    }
    if(cmd[0] == "load_workspace")
    {
        if(cmd[1].empty() && (cmd[1] =
           QFileDialog::getExistingDirectory(this,"Specify Workspace Directory",QFileInfo(windowTitle()).absolutePath()).toStdString()).empty())
            return run->canceled();

        if(!std::filesystem::exists(cmd[1]))
            return run->failed(error_msg = "cannot load workspace from " + cmd[1]);

        tipl::progress prog("loading data");
        if(std::filesystem::exists(cmd[1]+"/tracts"))
        {
            if(tractWidget->rowCount())
                tractWidget->command({"delete_all_tracts"});;
            for(const auto& each : tipl::search_files(cmd[1]+"/tracts","*tt.gz"))
                tractWidget->command({"open_tract",each});
        }

        prog(1,5);

        if(std::filesystem::exists(cmd[1]+"/slices"))
        {
            for(const auto& each : tipl::search_files(cmd[1]+"/slices","*nii.gz"))
                if(openSlices(each))
                {
                    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
                    if(reg_slice.get())
                        reg_slice->load_mapping((cmd[1]+"/slices/" + ui->SliceModality->currentText().toStdString() + ".linear_reg.txt").c_str());
                }
        }

        prog(2,5);
        if(std::filesystem::exists(cmd[1]+"/devices"))
        {
            if(deviceWidget->rowCount())
                deviceWidget->delete_all_devices();
            for(const auto& each : tipl::search_files(cmd[1]+"/devices","*dv.csv"))
                deviceWidget->load_device(each.c_str());
        }

        prog(3,5);
        if(std::filesystem::exists(cmd[1]+"/regions"))
        {
            if(regionWidget->rowCount())
                regionWidget->command({"delete_all_regions"});
            for(const auto& each : tipl::search_files(cmd[1]+"/regions","*nii.gz"))
                regionWidget->command({"load_region",each});
        }

        prog(4,5);

        command({"load_setting",cmd[1] + "/setting.ini"});
        command({"load_camera",cmd[1] + "/camera.txt"});

        std::ifstream in(cmd[1] + "/command.txt");
        std::string line;
        while(std::getline(in,line))
        {
            std::istringstream in2(line);
            std::vector<std::string> cmd2;
            std::copy(std::istream_iterator<std::string>(in2),std::istream_iterator<std::string>(),std::back_inserter(cmd2));
            command(cmd2);
        }

        std::string readme;
        if(std::filesystem::exists(cmd[1]+"/README"))
        {
            std::ifstream in(cmd[1]+"/README");
            readme = std::string((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        }
        report((readme + "\r\nMethods\r\n" + handle->report).c_str());
        return true;
    }
    if(cmd[0] == "save_setting" || cmd[0] == "save_rendering_setting" || cmd[0] == "save_tracking_setting")
    {
        if(cmd[1].empty() && (cmd[1] =
            QFileDialog::getSaveFileName(this,"Save INI files",QFileInfo(windowTitle()).baseName()
                        +cmd[0].substr(5).c_str() + ".ini","Setting file (*.ini);;All files (*)").toStdString()).empty())
            return run->canceled();

        QSettings s(cmd[1].c_str(), QSettings::IniFormat);
        if(cmd[0] == "save_setting")
        {
            for(const auto& each : renderWidget->treemodel->getParamList())
                s.setValue(each,renderWidget->getData(each));
        }
        if(cmd[0] == "save_rendering_setting")
        {
            QStringList param_list = renderWidget->treemodel->get_param_list("ROI");
            param_list += renderWidget->treemodel->get_param_list("Rendering");
            param_list += renderWidget->treemodel->get_param_list("Slice");
            param_list += renderWidget->treemodel->get_param_list("Tract");
            param_list += renderWidget->treemodel->get_param_list("Region");
            param_list += renderWidget->treemodel->get_param_list("Surface");
            param_list += renderWidget->treemodel->get_param_list("Device");
            param_list += renderWidget->treemodel->get_param_list("Label");
            param_list += renderWidget->treemodel->get_param_list("ODF");
            for(int index = 0;index < param_list.size();++index)
                s.setValue(param_list[index],renderWidget->getData(param_list[index]));
        }
        if(cmd[0] == "save_tracking_setting")
        {
            QStringList param_list = renderWidget->treemodel->get_param_list("Tracking");
            param_list += renderWidget->treemodel->get_param_list("Tracking_dT");
            param_list += renderWidget->treemodel->get_param_list("Tracking_adv");
            for(int index = 0;index < param_list.size();++index)
                s.setValue(param_list[index],renderWidget->getData(param_list[index]));
            return true;
        }
        return true;
    }
    if(cmd[0] == "load_setting" || cmd[0] == "load_rendering_setting" || cmd[0] == "load_tracking_setting")
    {
        if(cmd[1].empty() && (cmd[1] =
            QFileDialog::getOpenFileName(this,"Open INI files",
            QFileInfo(work_path).absolutePath(),"Setting file (*.ini);;All files (*)").toStdString()).empty())
            return run->canceled();

        if(!std::filesystem::exists(cmd[1]))
            return run->failed(error_msg = "cannot find " + cmd[1]);
        QSettings s(cmd[1].c_str(), QSettings::IniFormat);
        if(cmd[0] == "load_setting")
        {
            for(const auto& each : renderWidget->treemodel->getParamList())
                if(s.contains(each))
                    set_data(each,s.value(each));
            glWidget->update();
        }
        if(cmd[0] == "load_rendering_setting")
        {
            QStringList param_list = renderWidget->treemodel->get_param_list("ROI");
            param_list += renderWidget->treemodel->get_param_list("Rendering");
            param_list += renderWidget->treemodel->get_param_list("Slice");
            param_list += renderWidget->treemodel->get_param_list("Tract");
            param_list += renderWidget->treemodel->get_param_list("Region");
            param_list += renderWidget->treemodel->get_param_list("Surface");
            param_list += renderWidget->treemodel->get_param_list("Device");
            param_list += renderWidget->treemodel->get_param_list("Label");
            param_list += renderWidget->treemodel->get_param_list("ODF");
            for(int index = 0;index < param_list.size();++index)
                if(s.contains(param_list[index]))
                    set_data(param_list[index],s.value(param_list[index]));
        }
        if(cmd[0] == "load_tracking_setting")
        {
            QStringList param_list = renderWidget->treemodel->get_param_list("Tracking");
            param_list += renderWidget->treemodel->get_param_list("Tracking_dT");
            param_list += renderWidget->treemodel->get_param_list("Tracking_adv");
            for(int index = 0;index < param_list.size();++index)
                if(s.contains(param_list[index]))
                    set_data(param_list[index],s.value(param_list[index]));
        }
        return true;
    }

    if(cmd[0] == "restore_rendering")
    {
        renderWidget->setDefault("ROI");
        renderWidget->setDefault("Rendering");
        renderWidget->setDefault("show_slice");
        renderWidget->setDefault("show_tract");
        renderWidget->setDefault("show_region");
        renderWidget->setDefault("show_device");
        renderWidget->setDefault("show_surface");
        renderWidget->setDefault("show_label");
        renderWidget->setDefault("show_odf");
        renderWidget->setDefault("Tract_color");
        renderWidget->setDefault("Region_color");
        renderWidget->setDefault("Region_graph");
        tractWidget->update_color_map();
        regionWidget->update_color_map();
        regionWidget->color_map_values.clear();
        tractWidget->need_update_all();
        slice_need_update = true;
        glWidget->update();
        return true;
    }
    if(cmd[0] == "restore_tracking")
    {
        renderWidget->setDefault("Tracking");
        renderWidget->setDefault("Tracking_dT");
        renderWidget->setDefault("Tracking_adv");
        on_tracking_index_currentIndexChanged((*this)["tracking_index"].toInt());
        return true;
    }
    if(cmd[0] == "enable_auto_track")
    {
        if(!handle->load_track_atlas(true/*symmetric*/))
            return run->failed(handle->error_msg);

        auto level0 = handle->get_tractography_level0();

        ui->enable_auto_track->setVisible(false);
        ui->tract_target_0->setVisible(true);

        ui->tract_target_0->clear();
        ui->tract_target_0->addItem("All");
        for(const auto& each: level0)
            ui->tract_target_0->addItem(each.c_str());
        ui->tract_target_0->setCurrentIndex(0);
        raise();
        // for adding atlas tract in t1w as fib
        ui->perform_tracking->show();
        return true;
    }
    // the following must has cmd[1]
    if(cmd[0] == "set_roi_view")
    {
        if(cmd[1] == "0")
            ui->glSagView->setChecked(true);
        if(cmd[1] == "1")
            ui->glCorView->setChecked(true);
        if(cmd[1] == "2")
            ui->glAxiView->setChecked(true);
        return true;
    }
    if(cmd[0] == "set_roi_view_index")
    {
        bool okay = true;
        int index = QString(cmd[1].c_str()).toInt(&okay);
        if(okay)
        {
            ui->SliceModality->setCurrentIndex(index);
            return true;
        }
        index = ui->SliceModality->findText(cmd[1].c_str());
        if(index == -1)
            return run->failed(error_msg = "cannot find index: " + cmd[1]);
        ui->SliceModality->setCurrentIndex(index);
        return true;
    }
    if(cmd[0] == "set_roi_view_contrast")
    {
        ui->min_value_gl->setValue(QString(cmd[1].c_str()).toDouble());
        ui->max_value_gl->setValue(QString(cmd[2].c_str()).toDouble());
        change_contrast();
        return true;
    }
    if(cmd[0] == "set_slice_color")
    {
        ui->min_color_gl->setColor(QString(cmd[1].c_str()).toUInt());
        ui->max_color_gl->setColor(QString(cmd[2].c_str()).toUInt());
        change_contrast();
        return true;
    }
    if(cmd[0] == "set_param")
    {
        set_data(cmd[1].c_str(),cmd[2].c_str());
        glWidget->update();
        slice_need_update = true;
        return true;
    }
    if(cmd[0] == "tract_to_region")
    {
        on_actionTracts_to_seeds_triggered();
        return true;
    }
    if(cmd[0] == "set_region_color")
    {
        if(regionWidget->regions.empty())
            return true;
        regionWidget->regions.back()->region_render->color = QString(cmd[1].c_str()).toInt();
        glWidget->update();
        slice_need_update = true;
        return true;
    }
    if(cmd[0] == "add_slice")
    {
        if(!openSlices(cmd[1]))
            return run->failed("cannot add slice " + cmd[1]);
        tipl::out() << "register image to the DWI space" << std::endl;
        auto cur_slice = std::dynamic_pointer_cast<CustomSliceModel>(slices.back());
        cur_slice->wait();
        return true;
    }
    return run->failed("unknown command: " + cmd[0]);
}

std::string tracking_window::get_parameter_id(void)
{
    TrackingParam param;
    param.threshold = renderWidget->getData("fa_threshold").toFloat();
    param.dt_threshold = renderWidget->getData("dt_threshold").toFloat();
    param.cull_cos_angle = std::cos(renderWidget->getData("turning_angle").toDouble() * 3.14159265358979323846 / 180.0);
    param.step_size = renderWidget->getData("step_size").toFloat();
    param.smooth_fraction = renderWidget->getData("smoothing").toFloat();
    param.min_length = renderWidget->getData("min_length").toFloat();
    param.max_length = std::max<float>(param.min_length,renderWidget->getData("max_length").toDouble());

    param.tracking_method = renderWidget->getData("tracking_method").toInt();
    param.check_ending = renderWidget->getData("check_ending").toInt() && (renderWidget->getData("dt_index1").toInt() == 0);
    param.max_seed_count = renderWidget->getData("max_seed_count").toInt();
    param.max_tract_count = renderWidget->getData("max_tract_count").toInt();
    param.track_voxel_ratio = renderWidget->getData("track_voxel_ratio").toInt();
    param.default_otsu = renderWidget->getData("otsu_threshold").toFloat();
    param.tip_iteration =
            // only used in automatic fiber tracking
            (ui->tract_target_0->currentIndex() > 0 ||
            // or differential tractography
            renderWidget->getData("dt_index1").toInt() > 0)
            ? renderWidget->getData("tip_iteration").toInt() : 0;
    return param.get_code();
}


void tracking_window::on_actionLoad_Parameter_ID_triggered()
{
    QString id = QInputDialog::getText(this,QApplication::applicationName(),"Please assign parameter ID");
    if(id.isEmpty())
        return;
    TrackingParam param;
    param.set_code(id.toStdString());
    set_data("fa_threshold",float(param.threshold));
    set_data("dt_threshold",float(param.dt_threshold));
    set_data("turning_angle",float(std::acos(param.cull_cos_angle)*180.0f/3.14159265358979323846f));
    set_data("step_size",float(param.step_size));
    set_data("smoothing",float(param.smooth_fraction));
    set_data("min_length",float(param.min_length));
    set_data("max_length",float(param.max_length));

    set_data("tracking_method",int(param.tracking_method));
    set_data("check_ending",int(param.check_ending));
    set_data("max_tract_count",int(param.max_tract_count));
    set_data("max_seed_count",int(param.max_seed_count));
    set_data("track_voxel_ratio",float(param.track_voxel_ratio));

    set_data("otsu_threshold",float(param.default_otsu));
    set_data("tip_iteration",int(param.tip_iteration));

}


void tracking_window::on_actionEndpoints_to_seeding_triggered()
{
    std::vector<tipl::vector<3,short> > points1,points2;
    if(tractWidget->tract_models.empty() || tractWidget->currentRow() < 0)
        return;

    tractWidget->tract_models[size_t(tractWidget->currentRow())]->
            to_end_point_voxels(points1,points2,
                current_slice->is_diffusion_space ? tipl::matrix<4,4>(tipl::identity_matrix()) :current_slice->to_slice);

    regionWidget->begin_update();
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints1"));


    regionWidget->regions.back()->add_points(std::move(points1));

    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text()+
            QString(" endpoints2"));
    regionWidget->regions.back()->add_points(std::move(points2));
    regionWidget->end_update();
    slice_need_update = true;
    glWidget->update();
}

void tracking_window::on_actionTracts_to_seeds_triggered()
{
    if(tractWidget->tract_models.empty()|| tractWidget->currentRow() < 0)
        return;
    std::vector<tipl::vector<3,short> > points;
    tractWidget->tract_models[tractWidget->currentRow()]->to_voxel(points,
        current_slice->is_diffusion_space ? tipl::matrix<4,4>(tipl::identity_matrix()) : current_slice->to_slice);
    regionWidget->add_region(
            tractWidget->item(tractWidget->currentRow(),0)->text());
    regionWidget->regions.back()->add_points(std::move(points));
    slice_need_update = true;
    glWidget->update();
}


void tracking_window::on_actionQuality_Assessment_triggered()
{
    float threshold = renderWidget->getData("otsu_threshold").toFloat()*handle->dir.fa_otsu;
    std::pair<float,float> result = evaluate_fib(handle->dim,threshold,handle->dir.fa,
                                                 [&](int pos,char fib)
                                                 {return handle->dir.get_fib(pos,fib);});
    std::ostringstream out;
    out << "Fiber coherence index: " << result.first << std::endl;
    out << "Fiber incoherent index: " << result.second << std::endl;
    show_info_dialog("Quality assessment",out.str().c_str());
}


bool ask_TDI_options(int& rec,int& rec2)
{
    QMessageBox msgBox;
    msgBox.setText("Export directional color ? (BMP format only)");
    msgBox.setInformativeText("If grayscale or NIFTI format is preferred, select No.");
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No| QMessageBox::Cancel);
    msgBox.setDefaultButton(QMessageBox::Yes);
    rec = msgBox.exec();
    if(rec == QMessageBox::Cancel)
        return false;
    msgBox.setText("Export whole tracts or end points?");
    msgBox.setInformativeText("Yes: whole tracts, No: end points ");
    rec2 = msgBox.exec();
    if(rec2 == QMessageBox::Cancel)
        return false;
    return true;

}
void tracking_window::on_actionTDI_Diffusion_Space_triggered()
{
    tipl::matrix<4,4> tr;
    tr.identity();
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(handle->dim,handle->vs,handle->trans_to_mni,tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}


void tracking_window::on_actionTDI_Subvoxel_Diffusion_Space_triggered()
{
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    bool ok;
    int ratio = QInputDialog::getInt(this,
            QApplication::applicationName(),
            "Input super-resolution ratio (e.g. 2, 3, or 4):",2,2,8,1,&ok);
    if(!ok)
        return;
    tipl::matrix<4,4> tr,inv_tr,trans_to_mni(handle->trans_to_mni);
    tr.identity();
    tr[0] = tr[5] = tr[10] = ratio;
    inv_tr.identity();
    inv_tr[0] = inv_tr[5] = inv_tr[10] = 1.0f/ratio;
    trans_to_mni *= inv_tr;
    tractWidget->export_tract_density(handle->dim*ratio,
                                      handle->vs/float(ratio),
                                      trans_to_mni,
                                      tr,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}

void tracking_window::on_actionTDI_Import_Slice_Space_triggered()
{
    int rec,rec2;
    if(!ask_TDI_options(rec,rec2))
        return;
    tractWidget->export_tract_density(current_slice->dim,current_slice->vs,current_slice->trans_to_mni,current_slice->to_slice,rec == QMessageBox::Yes,rec2 != QMessageBox::Yes);
}



void tracking_window::on_actionTract_Analysis_Report_triggered()
{
    if(!tact_report_imp.get())
        tact_report_imp.reset(new tract_report(this));
    tact_report_imp->show();
    tact_report_imp->refresh_report();
}

void tracking_window::on_actionConnectivity_matrix_triggered()
{
    if(!tractWidget->tract_models.size())
    {
        QMessageBox::information(this,QApplication::applicationName(),"Run fiber tracking first");
        return;
    }
    std::ostringstream out;
    if(tractWidget->currentRow() < tractWidget->tract_models.size())
        out << tractWidget->tract_models[tractWidget->currentRow()]->report.c_str() << std::endl;
    connectivity_matrix.reset(new connectivity_matrix_dialog(this,out.str().c_str()));
    connectivity_matrix->show();
}


void tracking_window::on_actionOpen_Connectivity_Matrix_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open Connectivity Matrices files",QFileInfo(work_path).absolutePath(),
                "Connectivity file (*.mat *.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(filename.endsWith(".mat"))
    {
        tipl::io::mat_read in;
        if(!in.load_from_file(filename.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",in.error_msg.c_str());
            return;
        }
        unsigned int row,col;
        const float* buf = nullptr;
        if(!in.read("connectivity",row,col,buf))
        {
            QMessageBox::critical(this,"ERROR","Cannot find a matrix named connectivity");
            return;
        }
        if(row != col)
        {
            QMessageBox::critical(this,"ERROR","The connectivity matrix should be a square matrix");
            return;
        }
        glWidget->connectivity.resize(tipl::shape<2>(row,col));
        std::copy(buf,buf+row*col,glWidget->connectivity.begin());



        if(in.has("atlas") && in.read<std::string>("atlas") != "roi")
        {
            std::string atlas = in.read<std::string>("atlas");
            for(size_t i = 0;i < handle->atlas_list.size();++i)
                if(atlas == handle->atlas_list[i]->name)
                {
                    if(handle->atlas_list[i]->get_list().size() != row)
                    {
                        QMessageBox::critical(this,"ERROR","The atlas of connectivity matrix does not match the parcellation number");
                        return;
                    }
                    command({"delete_all_regions"});
                    command({"add_region_from_atlas",std::to_string(handle->template_id),std::to_string(i)});
                    set_data("region_graph",1);
                    break;
                }
        }
    }
    if(regionWidget->regions.empty())
    {
        QMessageBox::critical(this,"ERROR","Please load the regions first for visualization");
        return;
    }
    if(filename.endsWith(".txt"))
    {
        std::vector<float> buf;
        std::ifstream in(filename.toStdString().c_str());
        while(in)
        {
            std::string v;
            in >> v;
            if(v.empty())
                break;
            std::istringstream ss(v);
            buf.push_back(0.0f);
            ss >> buf.back();
        }
        size_t dim = size_t(std::sqrt(buf.size()));
        if(dim*dim != buf.size())
        {
            QMessageBox::critical(this,"ERROR",
            QString("There are %1 values in the file. The matrix in the text file is not a square matrix.").arg(buf.size()));
            return;
        }
        glWidget->connectivity.resize(tipl::shape<2>(dim,dim));
        std::copy(buf.begin(),buf.end(),glWidget->connectivity.begin());
    }

    if(int(regionWidget->regions.size()) != glWidget->connectivity.width())
    {
        QMessageBox::critical(this,"ERROR",
            QString("The connectivity matrix is %1-by-%2, but there are %3 regions. Please make sure the sizes are matched.").
                arg(glWidget->connectivity.width()).
                arg(glWidget->connectivity.height()).
                arg(regionWidget->regions.size()));
        return;
    }
    for(size_t i = 0,pos = 0;i < glWidget->connectivity.height();++i)
    {
        std::string line;
        for(size_t j = 0;j < glWidget->connectivity.width();++j,++pos)
        {
            line += std::to_string(glWidget->connectivity[pos]);
            line += " ";
        }
        tipl::out() << line;
    }
    glWidget->pos_max_connectivity = tipl::max_value(glWidget->connectivity);
    glWidget->neg_max_connectivity = tipl::min_value(glWidget->connectivity);
    if(glWidget->pos_max_connectivity == 0.0f)
        glWidget->pos_max_connectivity = 1.0f;
    if(glWidget->neg_max_connectivity == 0.0f)
        glWidget->neg_max_connectivity = -1.0f;

    set_data("region_graph",1);
    command({"check_all_regions"});
}


void tracking_window::on_actionFIB_protocol_triggered()
{
    std::istringstream in(handle->steps);
    std::ostringstream out;
    std::string line;
    for(int i = 1;std::getline(in,line);++i)
    {
        if(line.find('=') != std::string::npos)
            line = std::string("Set ") + line;
        else
        if(std::count(line.begin(),line.end(),']') >= 3)
            line = std::string("At the top menu, select ") + line;
        else
            line = std::string("Click ") + line;
        out << "(" << i << ") " << line << std::endl;
    }
    show_info_dialog("FIB",out.str());
}


void tracking_window::check_reg(void)
{
    bool all_ended = true;
    for(auto each : slices)
    {
        auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(each);
        if(reg_slice.get())
        {
            if(reg_slice->running)
            {
                all_ended = false;
                reg_slice->update_transform();
            }
        }
    }
    slice_need_update = true;
    if(all_ended)
        timer2.reset();
    else
        glWidget->update();
}

bool tracking_window::addSlices(std::shared_ptr<SliceModel> new_slice)
{
    if(!new_slice.get())
        return false;
    slices.push_back(new_slice);
    glWidget->slice_texture.push_back(std::vector<std::shared_ptr<QOpenGLTexture> >());
    ui->SliceModality->addItem(new_slice->view->name.c_str());
    return true;
}
bool tracking_window::addSlices(const std::string& name,const std::string& path)
{
    if(!tipl::begins_with(path,"http") && !std::filesystem::exists(path))
        return false;
    return addSlices(std::dynamic_pointer_cast<SliceModel>(
                std::make_shared<CustomSliceModel>(handle,std::make_shared<slice_model>(name,path))));
}
void tracking_window::start_reg(void)
{
    timer2.reset(new QTimer());
    timer2->setInterval(500);
    connect(timer2.get(), SIGNAL(timeout()), this, SLOT(check_reg()));
    timer2->start();
}
bool tracking_window::openSlices(const std::string& filename,bool is_mni)
{
    auto files = tipl::split(filename,',');
    if(files.empty())
        return false;
    auto slice = std::make_shared<CustomSliceModel>(handle,std::make_shared<slice_model>(
                            std::filesystem::path(files[0]).stem().stem().string(),files[0]));
    if(files.size() > 1)
        slice->source_files = files;
    slice->is_mni = is_mni;
    if(!slice->load_slices())
    {
        QMessageBox::critical(this,"ERROR",slice->error_msg.c_str());
        tipl::error() << slice->error_msg << std::endl;
        return false;
    }
    addSlices(slice);
    ui->SliceModality->setCurrentIndex(ui->SliceModality->count()-1);
    if(slice->running)
        start_reg();
    updateSlicesMenu();
    set_data("show_slice",Qt::Checked);
    glWidget->update();
    return true;
}
void tracking_window::on_addSlices_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
        this,"Open Images files",QFileInfo(work_path).absolutePath(),
                "Image files (*.dcm *.hdr *.nii *nii.gz *db.fz *db.fib.gz 2dseq);;Histology (*.jpg *.tif);;All files (*)" );
    if( filenames.isEmpty())
        return;
    if(QFileInfo(filenames[0]).completeSuffix() == "dcm" && filenames.size() == 1)
    {
        QDir directory = QFileInfo(filenames[0]).absoluteDir();
        QStringList file_list = directory.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        if(file_list.size() > filenames.size())
        {
            QString msg =
              QString("There are %1 DICOM files in the directory. Select all?").arg(file_list.size());
            int result = QMessageBox::information(this,"Input images",msg,
                                     QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::Yes)
            {
                filenames = file_list;
                for(int index = 0;index < filenames.size();++index)
                    filenames[index] = directory.absolutePath() + "/" + filenames[index];
            }
        }
    }
    if(filenames[0].endsWith(".nii.gz"))
    {
        for(const auto& each : filenames)
            openSlices(each.toStdString());
    }
    else
        openSlices(filenames.join(",").toStdString());
}

void tracking_window::on_actionInsert_MNI_images_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this,"Open MNI Image",QFileInfo(work_path).absolutePath(),
                "Image files (*.hdr *.nii *nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    if(!handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return;
    }
    openSlices(filename.toStdString(),true);
}

void tracking_window::insertPicture()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    if(action->text().contains("Sagittal"))
        cur_dim = 0;
    if(action->text().contains("Coronal"))
        cur_dim = 1;
    if(action->text().contains("Axial"))
        cur_dim = 2;


    float location = 0;
    switch(cur_dim)
    {
        case 0:
            location = (float(ui->glSagSlider->value())-0.5f*ui->glSagSlider->maximum())*handle->vs[0];
            break;
        case 1:
            location = (float(ui->glCorSlider->value())-0.5f*ui->glCorSlider->maximum())*handle->vs[1];
            break;
        case 2:
            location = (float(ui->glAxiSlider->value())-0.5f*ui->glAxiSlider->maximum())*handle->vs[2];
            break;

    }
    QString filename = QFileDialog::getOpenFileName(
        this,"Open Picture",QFileInfo(work_path).absolutePath(),"Pictures (*.jpg *.tif *.bmp *.png);;All files (*)" );
    if(filename.isEmpty() || !openSlices(filename.toStdString()))
        return;
    auto reg_slice_ptr = std::dynamic_pointer_cast<CustomSliceModel>(slices.back());
    if(!reg_slice_ptr.get())
        return;

    glWidget->set_view(cur_dim);
    switch(cur_dim)
    {
        case 0:
            tipl::flip_y(reg_slice_ptr->picture);
            tipl::flip_y(reg_slice_ptr->high_reso_picture);
            tipl::flip_y(reg_slice_ptr->source_images);
            tipl::swap_xy(reg_slice_ptr->source_images);
            tipl::swap_xz(reg_slice_ptr->source_images);
            std::swap(reg_slice_ptr->vs[0],reg_slice_ptr->vs[2]);
            reg_slice_ptr->update_image();
            reg_slice_ptr->arg_min.rotation[1] = 0.0f;
            reg_slice_ptr->arg_min.translocation[0] = location;
            ui->glSagCheck->setChecked(true);
            ui->glCorCheck->setChecked(false);
            ui->glAxiCheck->setChecked(false);
            break;
        case 1:
            tipl::flip_y(reg_slice_ptr->picture);
            tipl::flip_y(reg_slice_ptr->high_reso_picture);
            tipl::flip_y(reg_slice_ptr->source_images);
            tipl::swap_yz(reg_slice_ptr->source_images);
            std::swap(reg_slice_ptr->vs[1],reg_slice_ptr->vs[2]);
            reg_slice_ptr->update_image();
            reg_slice_ptr->arg_min.rotation[1] = 0.0f;
            reg_slice_ptr->arg_min.translocation[1] = location;
            ui->glSagCheck->setChecked(false);
            ui->glCorCheck->setChecked(true);
            ui->glAxiCheck->setChecked(false);
            break;
        case 2:
            reg_slice_ptr->arg_min.rotation[1] = 3.1415926f;
            reg_slice_ptr->arg_min.translocation[2] = location;
            ui->glSagCheck->setChecked(false);
            ui->glCorCheck->setChecked(false);
            ui->glAxiCheck->setChecked(true);
            break;
    }
    handle->slices.back()->set_image(reg_slice_ptr->source_images.alias());

    reg_slice_ptr->is_diffusion_space = false;
    reg_slice_ptr->update_transform();

    slice_need_update = true;
    if(QMessageBox::Yes == QMessageBox::question(this,QApplication::applicationName(),"Apply registration?",QMessageBox::No | QMessageBox::Yes))
    {
        reg_slice_ptr->run_registration();
        start_reg();
    }
    else
        QMessageBox::information(this,QApplication::applicationName(),"Press Ctrl+A and then hold LEFT/RIGHT button to MOVE/RESIZE slice close to the target before using [Slices][Adjust Mapping]");

    ui->SliceModality->setCurrentIndex(int(handle->slices.size())-1);
    glWidget->update();

}


void tracking_window::on_deleteSlice_clicked()
{
    int index = ui->SliceModality->currentIndex();
    if(index <= 1)
        return;
    if(current_slice->is_overlay)
        on_is_overlay_clicked();
    if(current_slice->stay)
        on_stay_clicked();
    if(current_slice->directional_color)
        on_directional_color_clicked();
    slices.erase(slices.begin()+index);
    glWidget->slice_texture.erase(glWidget->slice_texture.begin()+index);
    ui->SliceModality->removeItem(index);    
    updateSlicesMenu();
}



void tracking_window::on_actionAdjust_Mapping_triggered()
{
    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!reg_slice.get())
    {
        QMessageBox::critical(this,"ERROR","In the region window to the left, select the inserted slides to adjust mapping");
        return;
    }
    reg_slice->terminate();
    auto iso_fa = handle->get_iso_fa();
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
        subject_image_pre(tipl::image<3>(reg_slice->get_source())),subject_image_pre(tipl::image<3>(reg_slice->get_source())),reg_slice->vs,
        subject_image_pre(tipl::image<3>(iso_fa.first)),subject_image_pre(tipl::image<3>(iso_fa.second)),slices[0]->vs,
        tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));

    {
        reg_slice->update_transform();
        manual->arg = reg_slice->arg_min;
        manual->check_reg();
    }

    if(manual->exec() != QDialog::Accepted)
        return;

    reg_slice->arg_min = manual->arg;
    reg_slice->update_transform();
    reg_slice->is_diffusion_space = false;
    glWidget->update();
}

bool tracking_window::run_unet(void)
{
    QMessageBox::information(this,QApplication::applicationName(),"Specify the UNet model");
    QString filename = QFileDialog::getOpenFileName(this,
                "Select model",QCoreApplication::applicationDirPath()+"/network/",
                "Text files (*.net.gz);;All files|(*)");
    if(filename.isEmpty())
        return false;
    tipl::progress p("processing",true);
    unet = tipl::ml3d::unet3d::load_model<tipl::io::gz_mat_read>(filename.toStdString().c_str());
    if(!unet.get())
    {
        QMessageBox::critical(this,"ERROR","Cannot read the model file");
        return false;
    }
    if(!unet->forward(current_slice->get_source(),current_slice->vs,p))
    {
        QMessageBox::critical(this,"ERROR","Cannot process image");
        return false;
    }
    filename.chop(6);
    filename += "txt";
    if(std::filesystem::exists(filename.toStdString()))
    {
        unet_label_name.clear();
        std::ifstream in(filename.toStdString());
        std::string line;
        while(std::getline(in,line))
            unet_label_name.push_back(line);
    }
    return true;
}


void tracking_window::on_actionStrip_Skull_triggered()
{
    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!reg_slice.get())
    {
        QMessageBox::critical(this,"ERROR","This funciton only applied to inserted images.");
        return;
    }
    if(!run_unet())
        return;
    reg_slice->source_images *= unet->sum;
    slice_need_update = true;
    glWidget->update_slice();
}

void tracking_window::on_actionSegment_Tissue_triggered()
{
    if(!run_unet())
        return;
    // soft_max
    {
        tipl::adaptive_par_for(current_slice->dim.size(),[&](size_t pos)
        {
            float m = 0.0f;
            for(size_t i = pos;i < unet->out.size();i += current_slice->dim.size())
                if(unet->out[i] > m)
                    m = unet->out[i];
            if(unet->sum[pos] <= 0.5f)
            {
                for(size_t i = pos;i < unet->out.size();i += current_slice->dim.size())
                    unet->out[i] = 0.0f;
                return;
            }
            for(size_t i = pos;i < unet->out.size();i += current_slice->dim.size())
                unet->out[i] = (unet->out[i] >= m ? 1.0f:0.0f);
        });

    }
    {
        // to 3d label
        tipl::image<3> I(current_slice->dim);
        tipl::adaptive_par_for(current_slice->dim.size(),[&](size_t pos)
        {
            for(size_t i = pos,label = 1;i < unet->out.size();i += current_slice->dim.size(),++label)
                if(unet->out[i])
                {
                    I[pos] = label;
                    return;
                }
        });
        std::vector<std::vector<tipl::vector<3,short> > > regions(unet->out_channels_);
        tipl::adaptive_par_for(unet->out_channels_,[&](size_t label)
        {
            for(tipl::pixel_index<3> p(current_slice->dim);p < current_slice->dim.size();++p)
            {
                if(I[p.index()] == label+1)
                    regions[label].push_back(p);
            }
        });
        regionWidget->begin_update();
        for(size_t i = 0;i < unet->out_channels_;++i)
        {
            tipl::rgb color = tipl::rgb(255,255,255);
            std::string name = i < unet_label_name.size() ? unet_label_name[i].c_str() : (std::string("tissue")+std::to_string(i+1)).c_str();
            if(name.find("White") != std::string::npos)
                color = tipl::rgb(255,255,255,18);
            if(name.find("Gray") != std::string::npos)
                color = tipl::rgb(190,190,190,36);
            if(name.find("Cortex") != std::string::npos)
                color = tipl::rgb(150,150,150,36);
            if(name.find("Basal") != std::string::npos)
                color = tipl::rgb(110,110,110,128);
            if(name.find("Others") != std::string::npos)
                color = tipl::rgb(205,233,255,6);
            if(name.find("Edema") != std::string::npos)
                color = tipl::rgb(155,162,255,100);
            if(name.find("Tumor") != std::string::npos)
                color = tipl::rgb(255,170,127,128);
            if(name.find("Necrosis") != std::string::npos)
                color = tipl::rgb(75,75,75,200);
            regionWidget->add_region(name.c_str(),default_id,color);
            if(!regions[i].empty())
                regionWidget->regions.back()->add_points(std::move(regions[i]));
        }
        regionWidget->end_update();
    }

    slice_need_update = true;
    glWidget->update_slice();
}



void tracking_window::on_actionSave_Slices_to_DICOM_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get() || slice->source_files.empty())
    {
        QMessageBox::critical(this,"ERROR","This function needs original DICOM files (loading them at the[Slices] menu)");
        return;
    }

    QMessageBox::information(this,QApplication::applicationName(),"Please assign the output directory");
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Assign output directory",slice->source_files[0].c_str());
    if(dir.isEmpty())
        return;
    tipl::io::dicom_volume volume;
    if(!volume.load_from_files(slice->source_files))
    {
        QMessageBox::critical(this,"ERROR","Failed to load the original DICOM files");
        return;
    }

    {
        tipl::image<3> I;
        volume >> I;
        if(I.shape() != slice->source_images.shape())
        {
            QMessageBox::critical(this,"ERROR","Selected DICOM files does not match the original slices. Please check if missing any files.");
            return;
        }
    }


    tipl::image<3> out;
    {
        uint8_t new_dim_order[3];
        uint8_t new_flip[3];
        for(uint8_t i = 0;i < 3; ++i)
        {
            new_dim_order[uint8_t(volume.dim_order[i])] = i;
            new_flip[uint8_t(volume.dim_order[i])] = uint8_t(volume.flip[i]);
        }
        tipl::reorder(slice->source_images,out,new_dim_order,new_flip);
    }

    size_t read_size = 0;
    {
        tipl::io::dicom header;
        if(!header.load_from_file(slice->source_files[0]))
        {
            QMessageBox::critical(this,"ERROR","Invalid DICOM files");
            return;
        }
        read_size = header.width()*header.height();
    }

    tipl::progress prog("output dicom",true);
    for(int i = 0,pos = 0;prog(i,slice->source_files.size());++i,pos += read_size)
    {
        std::vector<char> buf;
        {
            std::ifstream in(slice->source_files[i],std::ios::binary | std::ios::ate);
            if(!in)
            {
                QMessageBox::critical(this,"ERROR",QString("Failed to load the original DICOM files: ") + slice->source_files[i].c_str());
                return;
            }
            buf.resize(size_t(in.tellg()));
            in.seekg(0,in.beg);
            if(read_size*sizeof(short) > buf.size())
            {
                QMessageBox::critical(this,"ERROR","Compressed DICOM is not supported. Please convert DICOM to uncompressed format.");
                return;
            }
            if(!in.read(buf.data(),int64_t(buf.size())))
            {
                QMessageBox::critical(this,"ERROR","Read DICOM failed");
                return;
            }
        }
        std::copy(out.begin()+pos,out.begin()+pos+int(read_size),
                  reinterpret_cast<short*>(&*(buf.end()-int(read_size*sizeof(short)))));

        QString output_name = dir + "/mod_" + QFileInfo(slice->source_files[i].c_str()).completeBaseName() + ".dcm";

        if(i == 0 && QFileInfo(output_name).exists() &&
           QMessageBox::information(this,QApplication::applicationName(),"Previous modifications found. Overwrite?",
           QMessageBox::Yes|QMessageBox::Cancel) == QMessageBox::Cancel)
                return;

        std::ofstream out(output_name.toStdString().c_str(),std::ios::binary);
        if(!out)
        {
            QMessageBox::critical(this,"ERROR","Cannot output DICOM. Please check disk space or output permission.");
            return;
        }
        out.write(&buf[0],int64_t(buf.size()));
    }
    QMessageBox::information(this,QApplication::applicationName(),"File Saved");
}



void tracking_window::on_enable_auto_track_clicked()
{
    if(!command({"enable_auto_track"}))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}

void tracking_window::on_tract_target_0_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    ui->tract_target_1->setVisible(false);
    ui->tract_target_2->setVisible(false);
    ui->tract_target_1->clear();
    if(index == 0) //track all without atk
        return;
    auto level1 = handle->get_tractography_level1(ui->tract_target_0->currentText().toStdString());
    if(level1.empty())
        return;
    for(const auto& each: level1)
        ui->tract_target_1->addItem(each.c_str());
    ui->tract_target_1->setCurrentIndex(0);
    ui->tract_target_1->setVisible(true);}

void tracking_window::on_tract_target_1_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    ui->tract_target_2->setVisible(false);
    ui->tract_target_2->clear();
    auto level2 = handle->get_tractography_level2(ui->tract_target_0->currentText().toStdString(),ui->tract_target_1->currentText().toStdString());
    if(level2.empty())
        return;
    ui->tract_target_2->addItem("All");
    for(const auto& each: level2)
        ui->tract_target_2->addItem(each.c_str());
    ui->tract_target_2->setCurrentIndex(0);
    ui->tract_target_2->setVisible(true);
}



void tracking_window::on_addRegionFromAtlas_clicked()
{
    if(handle->atlas_list.empty())
    {
        QMessageBox::critical(this,"ERROR","no atlas data");
        raise();
        return;
    }
    if(!handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return;
    }
    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,handle));
    atlas_dialog->exec();
}

void tracking_window::on_actionManual_Atlas_Alignment_triggered()
{
    if(!handle->load_template())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return ;
    }
    auto iso_fa = handle->get_iso_fa();
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
        subject_image_pre(tipl::image<3>(iso_fa.first)),subject_image_pre(tipl::image<3>(iso_fa.second)),handle->vs,
        template_image_pre(tipl::image<3>(handle->template_I)),template_image_pre(tipl::image<3>(handle->template_I2)),handle->template_vs,
        tipl::reg::affine,tipl::reg::cost_type::mutual_info));
    if(manual->exec() != QDialog::Accepted)
        return;
    handle->manual_template_T = manual->arg;
    handle->has_manual_atlas = true;


    std::string output_file_name(handle->fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[handle->template_id].c_str()).
                        baseName().toLower().toStdString();
    output_file_name += ".mz";
    if(handle->s2t.empty() && std::filesystem::exists(output_file_name))
    {
        handle->s2t.clear();
        handle->t2s.clear();
        std::filesystem::remove(output_file_name);
    }

    if(!handle->map_to_mni())
    {
        QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
        return;
    }

    std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,handle));
    atlas_dialog->exec();
}




void tracking_window::on_template_box_currentIndexChanged(int index)
{
    if(index < 0 || index >= int(fa_template_list.size()))
        return;
    handle->set_template_id(size_t(index));
    ui->alt_mapping->clear();
    ui->alt_mapping->addItem("regular");
    ui->alt_mapping->setCurrentIndex(0);
    ui->alt_mapping->setVisible(handle->alternative_mapping.size() > 1);
    for(size_t i = 1;i < handle->alternative_mapping.size();++i)
    {
        auto name = tipl::split(std::filesystem::path(handle->alternative_mapping[i]).filename().string(),'.');
        ui->alt_mapping->addItem(name.size() > 1 ? name[1].c_str() : name[0].c_str());
    }

    ui->tract_target_0->setCurrentIndex(0);
    ui->tract_target_0->hide();
    ui->tract_target_1->hide();
    ui->tract_target_2->hide();
    ui->enable_auto_track->setVisible(true);
    ui->addRegionFromAtlas->setVisible(!handle->atlas_list.empty());

}

void tracking_window::on_alt_mapping_currentIndexChanged(int index)
{
    if(index >= 0 && index < handle->alternative_mapping.size())
    {
        handle->alternative_mapping_index = index;
        handle->s2t.clear();
        handle->t2s.clear();
    }

}



void tracking_window::stripSkull()
{
    if(!ui->SliceModality->currentIndex() || !handle->is_human_data || handle->is_mni)
        return;
    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!reg_slice.get() || !reg_slice->skull_removed_images.empty())
        return;

    tipl::io::gz_nifti in1,in2;
    tipl::image<3> It,Iw,J(reg_slice->get_source());
    if(!in1.load_from_file(handle->t1w_template_file_name.c_str()) || !in1.toLPS(It))
        return;
    if(!in2.load_from_file(handle->mask_template_file_name.c_str()) || !in2.toLPS(Iw))
        return;
    if(QMessageBox::information(this,QApplication::applicationName(),
                                QString("Does %1 need to remove tissues outside the brain?").
                                arg(ui->SliceModality->currentText()),
                                QMessageBox::Yes|QMessageBox::No) == QMessageBox::No)
        return;

    tipl::vector<3> vs,vsJ(reg_slice->vs);
    in1.get_voxel_size(vs);

    tipl::downsampling(It);
    tipl::downsampling(It);
    tipl::downsampling(Iw);
    tipl::downsampling(Iw);
    vs *= 4.0f;



    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
            template_image_pre(tipl::image<3>(It)),tipl::image<3,unsigned char>(),vs,
            subject_image_pre(tipl::image<3>(J)),tipl::image<3,unsigned char>(),vsJ,tipl::reg::affine,tipl::reg::cost_type::mutual_info));

    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    tipl::filter::mean(Iw);
    tipl::filter::mean(Iw);

    reg_slice->skull_removed_images = reg_slice->source_images;
    reg_slice->skull_removed_images *= tipl::resample(Iw,reg_slice->source_images.shape(),manual->get_iT());
    slice_need_update = true;
}


void paint_track_on_volume(tipl::image<3,unsigned char>& track_map,const std::vector<std::vector<float> >& all_tracts,
                           std::shared_ptr<SliceModel> slice)
{
    tipl::adaptive_par_for(all_tracts.size(),[&](unsigned int i)
    {
        auto tracks = all_tracts[i];
        for(size_t k = 0;k < tracks.size();k +=3)
        {
            tipl::vector<3> p(&tracks[0] + k);
            p.to(slice->to_slice);
            tracks[k] = p[0];
            tracks[k+1] = p[1];
            tracks[k+2] = p[2];
        }
        for(size_t j = 0;j < tracks.size();j += 3)
        {
            tipl::pixel_index<3> p(std::round(tracks[j]),std::round(tracks[j+1]),std::round(tracks[j+2]),track_map.shape());
            if(track_map.shape().is_valid(p))
                track_map[p.index()] = 1;
            if(j)
            {
                for(float r = 0.2f;r < 1.0f;r += 0.2f)
                {
                    tipl::pixel_index<3> p2(std::round(tracks[j]*r+tracks[j-3]*(1-r)),
                                             std::round(tracks[j+1]*r+tracks[j-2]*(1-r)),
                                             std::round(tracks[j+2]*r+tracks[j-1]*(1-r)),track_map.shape());
                    if(track_map.shape().is_valid(p2))
                        track_map[p2.index()] = 1;
                }
            }
        }
    });
}



void tracking_window::on_actionSave_T1W_T2W_images_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get())
        return;
    QString filename = QFileDialog::getSaveFileName(
        this,"Save T1W/T2W Image",QFileInfo(work_path).absolutePath()+"//"+slice->name.c_str()+"_modified.nii.gz","Image files (*nii.gz);;All files (*)" );
    if( filename.isEmpty())
        return;
    tipl::io::gz_nifti::save_to_file(filename.toStdString().c_str(),slice->source_images,slice->vs,slice->trans_to_mni,slice->is_mni);
}

void tracking_window::on_actionMark_Region_on_T1W_T2W_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get() || slice->source_images.empty())
        return;
    bool ok = true;
    double ratio = QInputDialog::getDouble(this,QApplication::applicationName(),
            "Assign intensity (ratio to the maximum, e.g., 1.2 = 1.2*max)",1.0,0.0,10.0,1,&ok);
    if(!ok)
        return;
    auto current_region = regionWidget->regions[uint32_t(regionWidget->currentRow())];
    float mark_value = slice->get_value_range().second*float(ratio);
    tipl::image<3,unsigned char> mask;
    current_region->save_region_to_buffer(mask);
    if(current_region->to_diffusion_space != slice->to_dif)
    {
        tipl::image<3,unsigned char> new_mask(slice->dim);
        tipl::resample<tipl::interpolation::majority>(mask,new_mask,
            tipl::transformation_matrix<float>(tipl::from_space(slice->to_dif).to(current_region->to_diffusion_space)));
        mask.swap(new_mask);
    }

    for(size_t i = 0;i < mask.size();++i)
        if(mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update = true;
    glWidget->update();
}


void tracking_window::on_actionMark_Tracts_on_T1W_T2W_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get() || slice->source_images.empty() || tractWidget->tract_models.empty())
        return;
    bool ok = true;
    double ratio = QInputDialog::getDouble(this,QApplication::applicationName(),
            "Assign intensity (ratio to the maximum, e.g., 1.2 = 1.2*max)",1.0,0.0,10.0,1,&ok);
    if(!ok)
        return;
    tipl::image<3,unsigned char> t_mask(slice->source_images.shape());
    for(auto checked_tracks : tractWidget->get_checked_tracks())
        paint_track_on_volume(t_mask,checked_tracks->get_tracts(),slice);
    float mark_value = slice->get_value_range().second*float(ratio);
    for(size_t i = 0;i < t_mask.size();++i)
        if(t_mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update = true;
    glWidget->update();
}


void tracking_window::on_actionSave_mapping_triggered()
{
    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!reg_slice.get())
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Linear Registration",QString(reg_slice->source_file_name.c_str())+".linear_reg.txt",
            "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    if(!reg_slice->save_mapping(filename.toStdString().c_str()))
        QMessageBox::critical(this,"ERROR","Cannot save mapping file.");
}

void tracking_window::on_actionLoad_mapping_triggered()
{
    auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!reg_slice)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,"Open Linear Registration",QString(reg_slice->source_file_name.c_str())+".linear_reg.txt",
                "Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    reg_slice->terminate();
    if(!reg_slice->load_mapping(filename.toStdString().c_str()))
    {
        QMessageBox::critical(this,"ERROR","Invalid linear registration file.");
        return;
    }
    glWidget->update();
}


void tracking_window::on_actionLoad_Color_Map_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
                "Load color map",QCoreApplication::applicationDirPath()+"/color_map/",
                "Text files (*.txt);;All files|(*)");
    if(filename.isEmpty())
        return;
    tipl::color_map_rgb new_color_map;
    if(!new_color_map.load_from_file(filename.toStdString().c_str()))
    {
          QMessageBox::critical(this,"ERROR","Invalid color map format");
          return;
    }
    current_slice->view->v2c.set_color_map(new_color_map);
    slice_need_update = true;
    glWidget->update_slice();
}




void tracking_window::on_actionSave_3D_Model_triggered()
{
    auto tracts = tractWidget->get_checked_tracks();
    auto regions = regionWidget->get_checked_regions();
    if(tracts.empty() && regions.empty())
    {
        QMessageBox::critical(this,"ERROR","No visible tract or region to export");
        return;
    }
    for(auto& each_tract : tracts)
        if(each_tract->get_visible_track_count() > 3000)
        {
            QMessageBox::critical(this,"ERROR","Too many tracts. Please reduce the each tract count to less than 3,000 using [Tract Misc][Delete Repeated Tracks]");
            return;
        }
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,"Save tracts as",QFileInfo(windowTitle()).baseName()+".model.obj","3D files (*.obj);;All files (*)");
    if(filename.isEmpty())
        return;
    tipl::progress prog("exporting models",true);
    size_t total_prog = 3 + tracts.size() + regions.size()+1;
    size_t cur_prog = 0;
    std::ofstream out(filename.toStdString()),mtl(filename.toStdString()+".mtl");
    out << "mtllib " << QFileInfo(filename).fileName().toStdString() << ".mtl" << std::endl;
    out << "g" << std::endl;
    unsigned int coordinate_count = 0;



    if ((*this)["show_slice"].toInt())
    {

        for(size_t dim = 0;dim < 3 && prog(cur_prog++,total_prog);++dim)
        {
            if(!current_slice->slice_visible[dim])
                continue;
            // output texture
            float slice_alpha = (*this)["slice_alpha"].toFloat();
            {
                tipl::color_image texture;
                current_slice->get_high_reso_slice(texture,dim,current_slice->slice_pos[dim],overlay_slices);
                QImage I;
                I << texture;
                mtl << "newmtl slice" << int(dim) << std::endl;
                mtl << "Ka 1.000 1.000 1.000" << std::endl;
                mtl << "Kd 1.000 1.000 1.000" << std::endl;
                mtl << "d " << slice_alpha << std::endl;
                mtl << "Tr " << 1.0f-slice_alpha << std::endl;
                mtl << "map_Kd " << QFileInfo(filename).fileName().toStdString() << ".slice" << int(dim) << ".jpg" << std::endl;
                I.save(filename+".slice"+std::to_string(int(dim)).c_str()+".jpg");
            }

            // output texture
            {
                const float vt[4][3] = {{0.0f,1.0f},{1.0f,1.0f},{0.0f,0.0f},{1.0f,0.0f}};
                std::vector<tipl::vector<3> > points;
                current_slice->get_slice_positions(dim,points);
                for(size_t i = 0;i < 4;++i)
                {
                    points[i][0] *= handle->vs[0];
                    points[i][1] *= handle->vs[1];
                    points[i][2] *= handle->vs[2];
                    points[i][1] = -points[i][1];
                    std::swap(points[i][1],points[i][2]);
                    out << "v " << points[i] << std::endl;
                    out << "vt " << vt[i][0] << " " << vt[i][1] << std::endl;
                }
                size_t j = coordinate_count;
                out << "usemtl slice" << int(dim) << std::endl;
                out << "f " << j+1 << "/" << j+1 << " " << j+2 << "/" << j+2 << " " << j+4 << "/" << j+4 << std::endl;
                out << "f " << j+3 << "/" << j+3 << " " << j+1 << "/" << j+1 << " " << j+4 << "/" << j+4 << std::endl;
                coordinate_count += 4;
            }
        }
    }




    auto push_mtl = [&](tipl::rgb color,float alpha,std::string name,size_t id)
    {
        mtl << "newmtl " << name << id << std::endl;
        mtl << "Ka " << float(color.r)/255.0f << " " << float(color.g)/255.0f << " " << float(color.b)/255.0f << std::endl;
        mtl << "Kd " << float(color.r)/255.0f << " " << float(color.g)/255.0f << " " << float(color.b)/255.0f << std::endl;
        mtl << "d " << alpha << std::endl;
        mtl << "Tr " << 1.0f-alpha << std::endl;
        out << "usemtl " << name << id << std::endl;
    };

    size_t tract_count = 0;
    for(auto& each_tract : tracts)
    {
        if(!prog(cur_prog++,total_prog))
            break;
        if(each_tract->get_tracts().empty())
            continue;
        push_mtl(each_tract->get_tract_color(0),(*this)["tract_alpha"].toFloat(),"tract",tract_count++);
        out << each_tract->get_obj(coordinate_count,1/*tube*/,(*this)["tube_diameter"].toFloat(),0/*coarse*/) << std::endl;
    }
    if(prog.aborted())
        return;
    size_t render_count = 0;
    float region_alpha = (*this)["region_alpha"].toFloat();
    for(auto& each_region : regions)
    {
        if(!prog(cur_prog++,total_prog))
            break;
        if(each_region->region_render->object->point_list.empty())
            continue;
        push_mtl(each_region->region_render->color,float(each_region->region_render->color.a)/255.0f*region_alpha,"region",render_count++);
        out << each_region->region_render->get_obj(coordinate_count,handle->vs) << std::endl;
    }
    if(prog.aborted())
        return;

    if (glWidget->surface.get() && (*this)["show_surface"].toInt())
    {
        push_mtl(glWidget->surface->color,(*this)["surface_alpha"].toFloat(),"surface",0);
        out << glWidget->surface->get_obj(coordinate_count,handle->vs) << std::endl;
    }
    if(prog.aborted())
        return;
    QMessageBox::information(this,QApplication::applicationName(),"File Saved");
}

