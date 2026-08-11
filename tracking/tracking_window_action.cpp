#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QSignalBlocker>
#include <QClipboard>
#include <QMessageBox>
#include <QComboBox>

#include "atlasdialog.h"
#include "tracking_window.h"
#include "mainwindow.h"
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
#include "cmd/img.hpp"


extern std::vector<std::vector<std::string> > unet_names,unet_http,unet_desc;
bool download_unet_model(tipl::ml3d::tissue_seg& unet,const std::string& name);
extern std::vector<std::string> template_name_list;
std::string show_info_dialog(const std::string& title,
                             const std::string& result,
                             const std::string& file_name_hint)
{
    std::string saved_file;
    QWidget* parent = QApplication::activeWindow();
    QDialog* dlg = new QDialog(parent);
    dlg->setWindowTitle(QString::fromStdString(title));
    dlg->setMinimumSize(600, 400);
    QVBoxLayout* mainLay = new QVBoxLayout(dlg);

    // Use a vertical splitter so the user can adjust the space between the summary and table.
    QSplitter* splitter = new QSplitter(Qt::Vertical, dlg);
    QTextEdit* txt = new QTextEdit(dlg);
    txt->setReadOnly(true);
    txt->setText(QString::fromStdString(result));
    splitter->addWidget(txt);
    QTableWidget* table = new QTableWidget(dlg);
    table->setVisible(false);
    splitter->addWidget(table);
    mainLay->addWidget(splitter);

    QHBoxLayout* btnLay = new QHBoxLayout;
    QPushButton* copyBtn  = new QPushButton("Copy to Clipboard", dlg);
    QPushButton* saveBtn  = new QPushButton("Save as...", dlg);
    QPushButton* tableBtn = new QPushButton("Show Table", dlg);
    QPushButton* closeBtn = new QPushButton("Close", dlg);
    btnLay->addWidget(copyBtn); btnLay->addWidget(saveBtn);
    btnLay->addWidget(tableBtn); btnLay->addWidget(closeBtn);
    mainLay->addLayout(btnLay);

    QObject::connect(copyBtn, &QPushButton::clicked, [result](){
        QApplication::clipboard()->setText(QString::fromStdString(result));
    });
    QObject::connect(saveBtn, &QPushButton::clicked, [dlg, file_name_hint, result, &saved_file](){
        QString fn = QFileDialog::getSaveFileName(dlg, "Save as",
                        QString::fromStdString(file_name_hint),
                        "Text files (*.txt);;All files (*)");
        if (!fn.isEmpty()){
            tipl::out() << "save " << fn.toStdString();
            tipl::write_text_file(fn.toStdString(),result,tipl::error());
            saved_file = fn.toStdString();
            QMessageBox::information(dlg,QApplication::applicationName(),"file saved");
        }
    });
    QObject::connect(tableBtn, &QPushButton::clicked, [table, tableBtn, result](){
        if (!table->isVisible()){
            QStringList lines = QString::fromStdString(result).split('\n', Qt::SkipEmptyParts);
            int r = lines.size(), c = 0;
            QList<QStringList> data;
            for (const QString &line : lines) {
                QStringList cols = line.split('\t');
                data.append(cols);
                c = std::max<int>(c, cols.size());
            }
            table->clear(); table->setRowCount(r); table->setColumnCount(c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < data[i].size(); ++j)
                    table->setItem(i, j, new QTableWidgetItem(data[i][j]));
            table->resizeColumnsToContents(); table->resizeRowsToContents();
            tableBtn->setText("Hide Table");
            table->setVisible(true);
        } else {
            table->setVisible(false);
            tableBtn->setText("Show Table");
        }
    });
    QObject::connect(closeBtn, &QPushButton::clicked, dlg, &QDialog::accept);

    dlg->show();
    QEventLoop loop;
    QObject::connect(dlg, &QDialog::finished, &loop, &QEventLoop::quit);
    loop.exec();
    dlg->deleteLater();
    return saved_file;
}



void tracking_window::run_command(const std::string& cmd)
{
    if(!command({cmd}))
    {
        if(!error_msg.empty() && error_msg != "canceled")
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
    }
    else
        if(tipl::begins_with(cmd,"save_"))
            QMessageBox::information(this,QApplication::applicationName(),"file saved");
}

extern std::vector<tracking_window*> tracking_windows;
extern std::vector<std::filesystem::path> iso_template_list;
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
    if(deviceWidget->command(cmd))
        return true;
    if(!deviceWidget->error_msg.empty() && deviceWidget->error_msg != "not_processed")
    {
        error_msg = deviceWidget->error_msg;
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
        if(auto* mw = qobject_cast<MainWindow*>(parentWidget()))
            mw->report_and_target_window(tracking_windows.back());
        tracking_windows.back()->setAttribute(Qt::WA_DeleteOnClose);
        tracking_windows.back()->setWindowTitle(cmd[1].c_str());
        tracking_windows.back()->showNormal();
        tracking_windows.back()->resize(size().width(),size().height());
        return run->succeed();
    }
    if(cmd[0] == "correct_bias_field")
    {
        if(handle->correct_bias_field())
            return run->succeed();
        return run->failed("cannot find iso");
    }
    if(cmd[0] == "save_fib_as")
    {
        if(cmd[1].empty() && (cmd[1] = tipl::qt::save_image_file(this,
           windowTitle().replace(".fib.gz",".fz"),"FIB files (*.fz);;All files (*)").toStdString()).empty())
            return run->canceled();
        if(!handle->save_to_file(cmd[1]))
            return run->failed(handle->error_msg);
        return run->succeed();
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
        return run->succeed();
    }
    if(cmd[0] == "list_atlas")
    {
        if(!cmd[1].empty())
        {
            size_t atlas_id = handle->atlas_list.size();
            for(size_t i = 0;i < handle->atlas_list.size();++i)
                if(handle->atlas_list[i]->name == cmd[1])
                {
                    atlas_id = i;
                    break;
                }
            if(atlas_id == handle->atlas_list.size())
            {
                std::istringstream in(cmd[1]);
                size_t index;
                if((in >> index) && in.eof() && index < handle->atlas_list.size())
                    atlas_id = index;
            }
            if(atlas_id == handle->atlas_list.size())
                return run->failed("atlas not found: "+cmd[1]+"; use list_atlas without an argument to list atlases for the current template");

            tipl::out() << "region\tname";
            auto& region_list = handle->atlas_list[atlas_id]->get_list();
            for(size_t r = 0;r < region_list.size();++r)
                tipl::out() << r << "\t" << region_list[r];
            return run->succeed();
        }

        tipl::out() << "template: " << handle->template_id; // atlas_list is always scoped to the current template
        tipl::out() << "atlas\tname\tregions";
        for(size_t i = 0;i < handle->atlas_list.size();++i)
            tipl::out() << i << "\t"
                        << handle->atlas_list[i]->name << "\t"
                        << handle->atlas_list[i]->get_list().size();
        return run->succeed();
    }
    if(cmd[0] == "list_slice")
    {
        tipl::out() << "index\tcurrent\tname\tstatus";
        for(int index = 0;index < ui->SliceModality->count();++index)
        {
            auto& slice = slices[index];
            auto custom = std::dynamic_pointer_cast<CustomSliceModel>(slice);
            auto status = !custom ? "ready" : custom->running ? "registering" :
                          tipl::begins_with(custom->source_file_name.u8string(),"http") ?
                          "available" : "ready";
            tipl::out() << index << "\t" << (index == ui->SliceModality->currentIndex()) << "\t"
                        << ui->SliceModality->itemText(index).toStdString() << "\t" << status;
        }
        return run->succeed();
    }
    if(cmd[0] == "set_slice")
    {
        size_t index = run->from_cmd(1,ui->SliceModality->currentIndex());
        if(index >= slices.size())
            return run->failed("invalid slice index " + cmd[1]);
        auto new_slice = slices[index];
        auto new_custom_slice = std::dynamic_pointer_cast<CustomSliceModel>(new_slice);


        if(!new_slice->view->image_ready())
        {
            if(new_custom_slice.get())
            {
                if(!new_custom_slice->load_slices())
                    return run->failed(new_custom_slice->error_msg);
                if(new_custom_slice->running)
                    start_reg();
            }
            else
                new_slice->get_source();
        }

        QSignalBlocker blocker(ui->SliceModality);
        ui->SliceModality->setCurrentIndex(int(index));


        setUpdatesEnabled(false);

        auto previous_slice = current_slice;
        auto previous_custom_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
        current_slice = new_slice;

        ui->is_overlay->setChecked(new_slice->is_overlay);
        ui->stay->setChecked(new_slice->stay);
        ui->directional_color->setChecked(new_slice->directional_color);
        ui->segmentButton->setText(QString("Segment ") + ui->SliceModality->currentText() + "...");

        if(!glWidget->slice_texture[index].empty())
        {
            ui->glSagCheck->setChecked(new_slice->slice_visible[0]);
            ui->glCorCheck->setChecked(new_slice->slice_visible[1]);
            ui->glAxiCheck->setChecked(new_slice->slice_visible[2]);
        }
        ui->glSagSlider->setRange(0,int(new_slice->dim[0]-1));
        ui->glCorSlider->setRange(0,int(new_slice->dim[1]-1));
        ui->glAxiSlider->setRange(0,int(new_slice->dim[2]-1));
        ui->glSagBox->setRange(0,int(new_slice->dim[0]-1));
        ui->glCorBox->setRange(0,int(new_slice->dim[1]-1));
        ui->glAxiBox->setRange(0,int(new_slice->dim[2]-1));

        // update contrast color
        {
            std::pair<unsigned int,unsigned int> contrast_color = new_slice->get_contrast_color();
            ui->min_color_gl->setColor(contrast_color.first);
            ui->max_color_gl->setColor(contrast_color.second);
        }

        // setting up ranges
        {
            std::pair<float,float> range = new_slice->get_value_range();
            float r = range.second-range.first;
            float step = r/20.0f;
            ui->min_value_gl->setMinimum(double(range.first-r*0.2f));
            ui->min_value_gl->setMaximum(double(range.second));
            ui->min_value_gl->setSingleStep(double(step));
            ui->max_value_gl->setMinimum(double(range.first));
            ui->max_value_gl->setMaximum(double(range.second+r*0.2f));
            ui->max_value_gl->setSingleStep(double(step));
            ui->draw_threshold->setValue(0.0);
            ui->draw_threshold->setMaximum(range.second);
            ui->draw_threshold->setSingleStep(range.second/50.0);
        }

        // setupping values
        {
            std::pair<float,float> contrast_range = new_slice->get_contrast_range();
            ui->min_value_gl->setValue(double(contrast_range.first));
            ui->max_value_gl->setValue(double(contrast_range.second));
            ui->min_slider->setValue(int((contrast_range.first-ui->min_value_gl->minimum())*double(ui->min_slider->maximum())/(ui->min_value_gl->maximum()-ui->min_value_gl->minimum())));
            ui->max_slider->setValue(int((contrast_range.second-ui->max_value_gl->minimum())*double(ui->max_slider->maximum())/(ui->max_value_gl->maximum()-ui->max_value_gl->minimum())));
        }

        if((previous_custom_slice.get() && previous_custom_slice->running) ||
           (new_custom_slice.get() && new_custom_slice->running))
            move_slice_to(new_slice->slice_pos);
        else
        {
            tipl::vector<3> slice_position(previous_slice->slice_pos);
            if(!previous_slice->is_diffusion_space)
                slice_position.to(previous_slice->to_dif);
            if(!new_slice->is_diffusion_space)
                slice_position.to(new_slice->to_slice);
            move_slice_to(slice_position);
        }
        update_unet_models();

        command({"set_slice_contrast"});

        setUpdatesEnabled(true);
        return run->succeed();
    }
    if(cmd[0] == "list_unet")
    {
        update_unet_models();
        tipl::out() << "index\tavailable\tmodel\tname\tdescription";
        const auto& actions = ui->menuSegment->actions();
        for(int i = 0;i < actions.size();++i)
            tipl::out() << i << "\t" << actions[i]->isEnabled() << "\t"
                        << actions[i]->data().toString().toStdString() << "\t"
                        << actions[i]->statusTip().toStdString() << "\t"
                        << actions[i]->whatsThis().toStdString();
        return run->succeed();
    }
    if(cmd[0] == "segment_brain")
    {
        if(cmd[1].empty() && (cmd[1] = get_action_data().toStdString()).empty())
            return run->canceled();
        if(!cmd[2].empty())
        {
            int index = ui->SliceModality->findText(cmd[2].c_str());
            if(index < 0)
            {
                bool okay;
                int value = QString::fromStdString(cmd[2]).toInt(&okay);
                if(okay && value >= 0 && value < ui->SliceModality->count())
                    index = value;
            }
            if(index < 0)
                return run->failed(
                    "cannot find slice: use an exact name or index returned by list_slice");
            int previous_index = ui->SliceModality->currentIndex();
            QSignalBlocker blocker(ui->SliceModality);
            ui->SliceModality->setCurrentIndex(index);
            if(!command({"set_slice",std::to_string(index)}))
            {
                ui->SliceModality->setCurrentIndex(previous_index);
                return false;
            }
        }
        auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
        if(reg_slice)
        {
            reg_slice->wait();
            check_reg();
        }
        if(!current_slice->view->image_ready() || (reg_slice && reg_slice->running))
            return run->failed("slice is not ready: " + current_slice->get_name());


        tipl::image<3> source_images(current_slice->get_source());
        tipl::progress prog(cmd[0],true);
        tipl::ml3d::tissue_seg unet;
        if(!download_unet_model(unet,cmd[1]))
            return run->failed(unet.error_msg);

        if(tipl::contains(unet.preproc,"bet") && !current_slice->is_diffusion_space)
        {
            auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
            if(reg_slice.get())
            {
                tipl::progress p("brain extraction",true);
                tipl::image<3,unsigned char> mask;
                if(!handle->get_template_mask(reg_slice->source_images.shape(),reg_slice->to_dif,mask))
                    return run->failed(handle->error_msg);
                tipl::image<3> maskJ(mask);
                tipl::filter::gaussian(maskJ);
                tipl::filter::gaussian(maskJ);
                source_images *= maskJ;
            }
        }

        {
            tipl::progress p("running segmentation inference",true);
            if(!unet.forward(std::move(source_images),current_slice->vs))
                return run->failed(unet.error_msg);
        }

        /**
        {
            auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
            if(!reg_slice.get())
                return run->failed("invalid slice modality");
            reg_slice->source_images *= tipl::ml3d::soft_mask(unet_label);
            slice_need_update = true;
            glWidget->update_slice();
        }
        */
        {
            tipl::progress p("creating segmentation regions",true);
            const auto& unet_label = unet.data.label;
            std::vector<std::vector<tipl::vector<3,short> > > regions(unet.data.cur_count-1);
            std::vector<size_t> count(regions.size());
            for(auto label : unet_label)
                if(label)
                    ++count[label-1];
            for(size_t label = 0;label < regions.size();++label)
                regions[label].reserve(count[label]);
            size_t sz = current_slice->dim.size();
            for(tipl::pixel_index<3> p(current_slice->dim);p < sz;++p)
                if(auto label = unet_label[p.index()])
                    regions[label-1].push_back(p);

            setUpdatesEnabled(false);
            for(size_t i = 0;prog(i,regions.size());++i)
            {
                std::string name = i < unet.labels.size() ? unet.labels[i] : "tissue" + std::to_string(i + 1);

                auto get_color = [](std::string n,size_t index)
                {
                    std::transform(n.begin(),n.end(),n.begin(),[](uchar c){return char(std::tolower(c));});

                    auto clamp = [](int v){return std::clamp(v,0,255);};
                    auto color = [&](std::array<int,4> c)
                    {
                        size_t h = std::hash<std::string>{}(n) + index*131;
                        return tipl::rgb(clamp(c[0]+int(h%17)-8),
                                         clamp(c[1]+int((h>>4)%17)-8),
                                         clamp(c[2]+int((h>>8)%17)-8),c[3]);
                    };
                    auto match = [&](std::string_view s)
                    {
                        for(size_t p = 0,q;p < s.size();p = q+1)
                        {
                            q = s.find(',',p);
                            if(q == s.npos)
                                q = s.size();
                            if(n.find(s.substr(p,q-p)) != n.npos)
                                return true;
                        }
                        return false;
                    };

                    for(auto [key,c] : {
                             std::pair{"white,wm",std::array{238,238,232,35}},
                             {"stem",{220,220,210,80}},
                             {"gray,gm,cortex",{180,175,175,55}},
                             {"thal",{110,125,140,228}},
                             {"hipp",{95,135,115,228}},
                             {"amyg",{125,105,130,228}},
                             {"caud",{130,145,105,228}},
                             {"put",{145,135,100,228}},
                             {"accu",{120,150,120,228}},
                             {"pal",{155,130,100,228}},
                             {"basal,sub",{150,135,105,228}},
                             {"vent",{95,135,165,70}},
                             {"csf",{120,160,180,70}},
                             {"edema",{125,145,170,128}},
                             {"tumor,nor",{185,115,105,200}},
                             {"vas",{185,50,50,128}},
                             {"necro",{85,75,70,200}},
                             {"other,head,skull,dura",{200,210,215,15}}
                         })
                        if(match(key))
                            return color(c);

                    return color({255,255,255,255});
                };

                tipl::rgb color = get_color(name,i);

                regionWidget->add_region(name.c_str(),default_id,color);
                if(!regions[i].empty())
                    regionWidget->regions.back()->add_points(std::move(regions[i]));
            }
            setUpdatesEnabled(true);

            slice_need_update |= region_updated;
            glWidget->update_slice();
        }
        return true;

    }
    if(cmd[0] == "enable_slice")
    {
        bool x = ui->glSagCheck->isChecked(),
             y = ui->glCorCheck->isChecked(),
             z = ui->glAxiCheck->isChecked();
        if(cmd[1].empty())
            cmd[1] = std::to_string(x?1:0) + " " + std::to_string(y?1:0) + " " + std::to_string(z?1:0);
        else
            std::istringstream(cmd[1]) >> x >> y >> z;
        ui->glSagCheck->setChecked(x);
        ui->glCorCheck->setChecked(y);
        ui->glAxiCheck->setChecked(z);
        glWidget->update();
        history.overwrite(cmd[0]);
        return run->succeed();
    }

    if(cmd[0] == "move_slice")
    {
        int x = ui->glSagSlider->value(),y = ui->glCorSlider->value(),z = ui->glAxiSlider->value();
        if(cmd[1].empty())
            cmd[1] = std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
        else
            std::istringstream(cmd[1]) >> x >> y >> z;
        move_slice_to(tipl::vector<3>(x,y,z));
        history.overwrite(cmd[0]);
        return run->succeed();
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

        slice_need_update = none; // turn off simple drawing
        scene.paint_image(scene.view_image,false);
        if(!scene.view_image.save(cmd[1].c_str()))
            return run->failed("cannot save mapping to " + cmd[1]);
        return run->succeed();
    }
    if(cmd[0] == "preview_screen")
    {
        if(cmd[1] != "roi" && cmd[1] != "3d")
            return run->failed("cmd[1] must be \"roi\" or \"3d\"");
        // serves a cached crop (zoom) or a fresh capture via `grab`, then prints it as text art
        auto preview_channel = [&](std::map<std::string,QImage>& cache,const std::string& label,auto&& grab)->bool
        {
            QImage gray;
            if(!cmd[2].empty()) // zoom: reuse the cached capture, no regrab
            {
                auto it = cache.find(label);
                if(it == cache.end())
                    return run->failed("no previous capture of channel \""+label+"\" to zoom into");
                gray = it->second;
            }
            else
                cache[label] = gray = grab();
            // measured on the full capture, not the crop below, whose corners may not be background
            double bg = (double(gray.constScanLine(0)[0])+double(gray.constScanLine(0)[gray.width()-1])+
                         double(gray.constScanLine(gray.height()-1)[0])+double(gray.constScanLine(gray.height()-1)[gray.width()-1]))/4.0;
            QImage region = gray;
            if(!cmd[2].empty())
            {
                std::istringstream in(cmd[2]);
                double x0,y0,x1,y1;
                if(!(in >> x0 >> y0 >> x1 >> y1))
                    return run->failed("invalid zoom rectangle, expected \"x0 y0 x1 y1\" in 0..1");
                region = gray.copy(int(x0*gray.width()),int(y0*gray.height()),
                                    int((x1-x0)*gray.width()),int((y1-y0)*gray.height()));
            }
            TextPreview preview(region.width(),region.height(),
                [&](int x,int y){ return double(region.constScanLine(y)[x]); },bg);
            tipl::out() << label << ":";
            tipl::out() << preview.render_art(16);
            tipl::out() << preview.render_occupancy();
            tipl::out() << preview.format_stats();
            return true;
        };

        if(cmd[1] == "3d")
        {
            glWidget->command({"get_camera"});
            static const char* channels[] = {"show_slice","show_region","show_tract","show_surface"};
            for(const char* flag : channels)
            {
                if(!(*this)[flag].toInt())
                    continue;
                std::string label = flag+5; // strip "show_"
                if(!preview_channel(last_3d_preview,label,[&](void)->QImage
                {
                    // isolate this one layer by temporarily toggling the others off
                    std::vector<std::pair<std::string,QVariant> > saved;
                    for(const char* other : channels)
                    {
                        saved.push_back({other,(*this)[other]});
                        set_data(other,int(other == std::string(flag)));
                    }
                    QImage gray = glWidget->grab_image().convertToFormat(QImage::Format_Grayscale8);
                    for(auto& kv : saved)
                        set_data(kv.first.c_str(),kv.second);
                    glWidget->update();
                    gray.save(QDir::tempPath()+"/dsi_preview_3d_"+QString::fromStdString(label)+".jpg","JPG");
                    return gray;
                }))
                    return false;
            }
            return run->succeed();
        }

        // roi: slice is the plain anatomy; region/tract are read as their own isolated layers from
        // slice_view_scene's cache (see overlay_cache), not re-rendered per channel -- only valid
        // for the single-slice layout, since those layers are cached per dimension, not per
        // multi-view composite
        tipl::out() << "R_side=" << ((*this)["orientation_convention"].toInt() ? "right" : "left");
        {
            static const char* dim_names[3] = {"sagittal","coronal","axial"};
            tipl::out() << "slice_info: " << ui->SliceModality->currentText().toStdString()
                        << " " << dim_names[cur_dim]
                        << " " << (current_slice->slice_pos[cur_dim]+1) << "/" << current_slice->dim[cur_dim];
        }
        if((*this)["roi_layout"].toInt() != 0)
        {
            struct channel_def{ const char* label; bool roi_track; bool simple; bool on; };
            const channel_def channels[] = {
                {"slice", false,true, true},
                {"region",false,false,!regionWidget->get_checked_regions().empty()},
                {"tract", true, false,ui->roi_track->isChecked()},
            };
            for(const auto& ch : channels)
            {
                if(!ch.on)
                    continue;
                if(!preview_channel(last_roi_preview,ch.label,[&](void)->QImage
                {
                    bool prior_roi_track = ui->roi_track->isChecked();
                    ui->roi_track->setChecked(ch.roi_track);
                    slice_need_update = none;
                    scene.paint_image(scene.view_image,ch.simple);
                    QImage gray = scene.view_image.convertToFormat(QImage::Format_Grayscale8);
                    ui->roi_track->setChecked(prior_roi_track);
                    slice_need_update |= position_updated;
                    gray.save(QDir::tempPath()+"/dsi_preview_roi_"+QString(ch.label)+".jpg","JPG");
                    return gray;
                }))
                    return false;
            }
            return run->succeed();
        }

        QImage base,region_layer,tract_layer;
        bool region_on = false,tract_on = false;
        if(cmd[2].empty()) // not zooming: refresh the caches and read them for this call
        {
            slice_need_update = none;
            QImage refreshed;
            scene.paint_image(refreshed,false); // side effect: brings overlay_cache[cur_dim] up to date
            slice_need_update |= position_updated;
            scene.paint_image(base,true); // plain slice, no overlays
            region_layer = scene.region_layer(cur_dim);
            tract_layer = scene.tract_layer(cur_dim);
            region_on = !region_layer.isNull();
            tract_on = ui->roi_track->isChecked() && !tract_layer.isNull();
        }
        else // zooming: only channels with an existing cached capture can be zoomed into
        {
            region_on = last_roi_preview.count("region") != 0;
            tract_on = last_roi_preview.count("tract") != 0;
        }

        if(!preview_channel(last_roi_preview,"slice",[&](void)->QImage
        {
            QImage gray = base.convertToFormat(QImage::Format_Grayscale8);
            gray.save(QDir::tempPath()+"/dsi_preview_roi_slice.jpg","JPG");
            return gray;
        }))
            return false;

        // shown as its own isolated layer (transparent -> black once grayscaled), not composited
        // onto the anatomical slice: a small region blob or thin tract line would otherwise be
        // visually swamped by the much larger, dominant brain-shape luminance in a coarse digit grid
        if(region_on && !preview_channel(last_roi_preview,"region",[&](void)->QImage
        {
            QImage gray = region_layer.convertToFormat(QImage::Format_Grayscale8);
            gray.save(QDir::tempPath()+"/dsi_preview_roi_region.jpg","JPG");
            return gray;
        }))
            return false;

        if(tract_on && !preview_channel(last_roi_preview,"tract",[&](void)->QImage
        {
            QImage gray = tract_layer.convertToFormat(QImage::Format_Grayscale8);
            gray.save(QDir::tempPath()+"/dsi_preview_roi_tract.jpg","JPG");
            return gray;
        }))
            return false;

        return run->succeed();
    }
    if(cmd[0] == "save_slice_image" || cmd[0] == "save_slice_mni_image")
    {
        if(cmd[2].empty() && (cmd[2] = get_action_data().toStdString()).empty())
            return run->canceled();
        if(cmd[1].empty() && (cmd[1] = tipl::qt::save_image_file(
                    this,QFileInfo(windowTitle()).baseName()+"_"+ QString::fromStdString(cmd[2])+".nii.gz",
                    "NIFTI files (*nii.gz *.nii);;MAT files (*.mat);;All files (*)").toStdString()).empty())
            return run->canceled();

        if(!handle->save_slice(cmd[2],cmd[1],cmd[0] == "save_slice_mni_image"))
            return run->failed(handle->error_msg);
        return run->succeed();
    }
    if(cmd[0] == "show_only_regions" || cmd[0] == "show_only_tracts")
    {
        bool is_region = cmd[0] == "show_only_regions";
        QTableWidget* table = is_region ? static_cast<QTableWidget*>(regionWidget) :
                                  static_cast<QTableWidget*>(tractWidget);
        auto index_list = QString::fromStdString(cmd[1]).split('&',Qt::SkipEmptyParts);
        if(index_list.empty())
            return run->failed(std::string("no ") +
                               (is_region ? "region" : "tract") + " index specified");

        std::vector<bool> shown(size_t(table->rowCount()));
        for(const auto& each : index_list)
        {
            bool ok;
            int row = each.toInt(&ok);
            if(!ok || row < 0 || row >= table->rowCount())
                return run->failed(std::string("invalid ") +
                                   (is_region ? "region" : "tract") +
                                   " index: " + each.toStdString());
            shown[size_t(row)] = true;
        }

        {
            QSignalBlocker blocker(table);
            for(int row = 0;row < table->rowCount();++row)
            {
                auto state = shown[size_t(row)] ? Qt::Checked : Qt::Unchecked;
                table->item(row,0)->setCheckState(state);
                table->item(row,0)->setData(Qt::UserRole+1,state);
            }
        }

        if(is_region)
            emit regionWidget->region_changed();
        else
            emit tractWidget->tract_changed();
        return run->succeed();
    }
    if(cmd[0] == "presentation_mode")
    {
        ui->ROIdockWidget->hide();
        if(!regionWidget->rowCount())
            ui->regionDockWidget->hide();
        return run->succeed();
    }
    if(cmd[0] == "save_workspace")
    {
        if(!history.get_directory(this,cmd[1]))
            return run->canceled();

        std::filesystem::create_directory(cmd[1]);
        if (!std::filesystem::exists(cmd[1]) || !std::filesystem::is_directory(cmd[1]))
            return run->failed("cannot save workspace to " + cmd[1]);

        if(tractWidget->rowCount())
        {
            std::filesystem::create_directory(cmd[1]+"/tracts");
            tractWidget->command({"save_all_tracts_to_folder",cmd[1]+"/tracts"});
        }
        if(regionWidget->rowCount())
        {
            std::filesystem::create_directory(cmd[1]+"/regions");
            regionWidget->command({"save_all_regions_to_folder",cmd[1]+"/regions"});
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
            tipl::io::gz_nifti(cmd[1]+"/slices/" + ui->SliceModality->currentText().toStdString() + ".nii.gz",std::ios::out) << reg_slice->bind(II);
            reg_slice->save_mapping((cmd[1]+"/slices/" + ui->SliceModality->currentText().toStdString() + ".linear_reg.txt").c_str());
        }

        command({"save_setting",cmd[1] + "/setting.ini"});
        command({"save_camera",cmd[1] + "/camera.txt"});

        std::ofstream out(cmd[1] + "/commands.csv");
        out << "move_slice" << "," << current_slice->slice_pos[0] << " " << current_slice->slice_pos[1] << " " << current_slice->slice_pos[2] << std::endl;
        out << "enable_slice" << "," << (ui->glSagCheck->isChecked()?1:0) << " " << (ui->glCorCheck->isChecked()?1:0) << " " << (ui->glAxiCheck->isChecked()?1:0) << std::endl;
        out << "set_zoom" << "," << ui->zoom_3d->value();
        return run->succeed();

    }
    if(cmd[0] == "load_workspace")
    {
        if(!history.get_directory(this,cmd[1]))
            return run->canceled();

        if(!std::filesystem::exists(cmd[1]))
            return run->failed(error_msg = "cannot load workspace from " + cmd[1]);

        tipl::progress prog("loading data");
        if(std::filesystem::exists(cmd[1]+"/tracts"))
        {
            if(tractWidget->rowCount())
                tractWidget->command({"delete_all_tracts"});;
            for(const auto& each : tipl::search_files(cmd[1]+"/tracts","*tt.gz"))
                tractWidget->command({"open_tract",each.u8string()});
        }

        prog(1,5);

        if(std::filesystem::exists(cmd[1]+"/slices"))
        {
            for(const auto& each : tipl::search_files(cmd[1]+"/slices","*nii.gz"))
                if(command({"add_slice",each.u8string()}))
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
                deviceWidget->command({"delete_all_devices"});
            for(const auto& each : tipl::search_files(cmd[1]+"/devices","*dv.csv"))
                deviceWidget->load_device(each);
        }

        prog(3,5);
        if(std::filesystem::exists(cmd[1]+"/regions"))
        {
            if(regionWidget->rowCount())
                regionWidget->command({"delete_all_regions"});
            for(const auto& each : tipl::search_files(cmd[1]+"/regions","*.nii.gz"))
                regionWidget->command({"open_region",each.u8string()});
        }

        prog(4,5);      

        for(const auto& line : tipl::read_text_file(cmd[1] + "/commands.csv"))
            command(tipl::split(line,','));

        command({"load_setting",cmd[1] + "/setting.ini"});
        command({"open_camera",cmd[1] + "/camera.txt"});


        std::string readme;
        if(std::filesystem::exists(cmd[1]+"/README"))
        {
            std::ifstream in(cmd[1]+"/README");
            readme = std::string((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        }
        report((readme + handle->report).c_str());
        return run->succeed();
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
        }
        return run->succeed();
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
        return run->succeed();
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
        slice_need_update |= position_updated;
        glWidget->update();
        return run->succeed();
    }
    if(cmd[0] == "restore_tracking")
    {
        renderWidget->setDefault("Tracking");
        renderWidget->setDefault("Tracking_dT");
        renderWidget->setDefault("Tracking_adv");
        on_tracking_index_currentIndexChanged((*this)["tracking_index"].toInt());
        set_data("min_length",handle->default_min_length());
        set_data("max_length",handle->default_max_length());
        set_data("track_voxel_ratio",handle->default_track_voxel_ratio());
        set_data("tolerance",handle->default_tolerance());
        return run->succeed();
    }
    if(cmd[0] == "enable_auto_tract")
    {
        if(!handle->load_track_atlas(true/*symmetric*/))
            return run->failed(handle->error_msg);

        auto level0 = handle->get_tractography_level0();

        ui->enable_auto_tract->setVisible(false);
        ui->tract_target_0->setVisible(true);

        ui->tract_target_0->clear();
        ui->tract_target_0->addItem("All");
        for(const auto& each: level0)
            ui->tract_target_0->addItem(each.c_str());
        ui->tract_target_0->setCurrentIndex(0);
        raise();
        // for adding atlas tract in t1w as fib
        ui->perform_tracking->show();
        return run->succeed();
    }
    if(cmd[0] == "list_auto_tract")
    {
        if(!handle->load_tractography_name_list())
            return run->failed(handle->error_msg);
        tipl::out() << "name";
        for(const auto& name : handle->tractography_name_list)
            tipl::out() << name;
        return run->succeed();
    }
    if(cmd[0] == "run_auto_track")
    {
        if(cmd[1].empty())
            return run->failed("please specify tract name");
        if(!handle->load_track_atlas(true))
            return run->failed(handle->error_msg);
        auto param = get_parameter_id(true); // auto_track: TIP must apply here regardless of tract_target_0's state, which this AI path never touches
        if(!cmd[2].empty())
            param += " " + cmd[2];
        if(!tractWidget->command({"run_tracking",cmd[1],param,
                                   std::to_string((*this)["tolerance"].toFloat())}))
            return run->failed(tractWidget->error_msg);
        return run->succeed();
    }
    if(cmd[0] == "run_dif_tracking")
    {
        // cmd[1]: new tract-bundle name
        // cmd[2]: optional region settings
        if(!(*this)["dt_index1"].toInt() && !(*this)["dt_index2"].toInt())
            return run->failed("dt_index1/dt_index2 not set");
        if(!tractWidget->command({"set_dt_index",
                     dt_list[(*this)["dt_index1"].toInt()].toStdString() + '&' +
                     dt_list[(*this)["dt_index2"].toInt()].toStdString(),
                     std::to_string(renderWidget->getData("dt_threshold_type").toInt())}))
            return run->failed(tractWidget->error_msg);
        if(!tractWidget->command({"run_tracking",cmd[1],cmd[2]}))
            return run->failed(tractWidget->error_msg);
        return run->succeed();
    }
    if(cmd[0] == "list_history")
    {
        tipl::out() << "index\tcommand";
        for(size_t i = 0;i < history.commands.size();++i)
            tipl::out() << i << "\t" << history.commands[i];
        return run->succeed();
    }
    if(cmd[0] == "run_command_history")
    {
        // cmd[1]: folder to search using the recorded load step's extension, or an "&"-joined explicit file list
        // cmd[2]: optional "from:to" 0-based inclusive index range into list_history; omit to use every recorded command
        if(cmd[1].empty())
            return run->failed("missing folder or file path");
        if(history.commands.empty())
            return run->failed("no recorded commands to replay");
        auto selected = history.commands;
        if(!cmd[2].empty())
        {
            auto range = tipl::split(cmd[2],':');
            bool okay = true;
            int from = QString(range[0].c_str()).toInt(&okay);
            int to = from;
            if(okay && range.size() > 1)
                to = QString(range[1].c_str()).toInt(&okay);
            if(!okay || from < 0 || to < from || size_t(to) >= history.commands.size())
                return run->failed("invalid command range: " + cmd[2]);
            selected = std::vector<std::string>(history.commands.begin()+from,history.commands.begin()+to+1);
        }
        std::string batch_error;
        if(!history.run(this,selected,cmd[1],batch_error))
            return run->failed(batch_error);
        return run->succeed();
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
        return run->succeed();
    }
    if(cmd[0] == "set_slice_by_name")
    {
        if(cmd[1].empty())
            return run->canceled();
        auto index = ui->SliceModality->findText(cmd[1].c_str());
        if(index == -1)
            return run->failed("cannot find index: " + cmd[1]);
        ui->SliceModality->setCurrentIndex(index);
        history.overwrite(cmd[0]);
        tipl::out() << "index\tname";
        tipl::out() << index << "\t" << ui->SliceModality->itemText(index).toStdString();
        return run->succeed();
    }
    if(cmd[0] == "set_slice_contrast")
    {
        // cmd[1] : min max values
        // cmd[2] : min max colors
        double min_value_gl(ui->min_value_gl->value()),max_value_gl(ui->max_value_gl->value());
        if(cmd[1].empty())
            cmd[1] = std::to_string(min_value_gl) + " " +
                     std::to_string(max_value_gl);
        else
            std::istringstream(cmd[1]) >> min_value_gl >> max_value_gl;

        unsigned int min_color_gl(ui->min_color_gl->color().rgb()),max_color_gl(ui->max_color_gl->color().rgb());
        if(cmd[2].empty())
            cmd[2] =std::to_string(min_color_gl) + " " +
                     std::to_string(max_color_gl);
        else
            std::istringstream(cmd[2]) >> min_color_gl >> max_color_gl;

        current_slice->set_contrast_range(min_value_gl,max_value_gl);
        current_slice->set_contrast_color(min_color_gl,max_color_gl);
        slice_need_update |= image_updated;
        glWidget->update_slice();
        history.overwrite(cmd[0]);
        return run->succeed();
    }
    if(cmd[0] == "set_slice_dir_color")
    {
        // cmd[1] = slice_index
        // cmd[2] = checked
        int slice_index= run->from_cmd(1,ui->SliceModality->currentIndex());
        if(slice_index < 0 || slice_index >= slices.size())
            return run->canceled();
        bool checked = run->from_cmd(2,ui->directional_color->isChecked()?1:0);
        if(slices[slice_index]->directional_color == checked)
            return run->canceled();
        slices[slice_index]->directional_color = checked;
        glWidget->update_slice();
        slice_need_update |= image_updated;
        history.overwrite(cmd[0]);
        return run->succeed();
    }

    if(cmd[0] == "set_slice_overlay")
    {
        // cmd[1] = slice_index
        // cmd[2] = checked
        int slice_index= run->from_cmd(1,ui->SliceModality->currentIndex());
        if(slice_index < 0 || slice_index >= slices.size())
            return run->canceled();
        bool checked = run->from_cmd(2,ui->is_overlay->isChecked()?1:0);
        if(slices[slice_index]->is_overlay == checked)
            return run->canceled();

        if((slices[slice_index]->is_overlay = checked))
            overlay_slices.push_back(slices[slice_index]);
        else
            overlay_slices.erase(std::remove(overlay_slices.begin(),overlay_slices.end(),slices[slice_index]),overlay_slices.end());

        glWidget->update_slice();
        slice_need_update |= image_updated;

        history.overwrite(cmd[0]);
        return run->succeed();
    }

    if(cmd[0] == "set_slice_stay")
    {
        // cmd[1] = slice_index
        // cmd[2] = checked
        int slice_index= run->from_cmd(1,ui->SliceModality->currentIndex());
        if(slice_index < 0 || slice_index >= slices.size())
            return run->canceled();
        bool checked = run->from_cmd(2,ui->stay->isChecked()?1:0);
        if(slices[slice_index]->stay == checked)
            return run->canceled();

        if((slices[slice_index]->stay = checked))
            stay_slices.push_back(slices[slice_index]);
        else
            stay_slices.erase(std::remove(stay_slices.begin(),stay_slices.end(),slices[slice_index]),stay_slices.end());

        glWidget->update_slice();
        slice_need_update |= image_updated;

        history.overwrite(cmd[0]);
        return run->succeed();
    }
    if(cmd[0] == "list_param")
    {
        struct domain_type
        {
            const char* name;
            const char* root_param;
            const char* groups[3];
        };
        static const domain_type domains[] =
            {
                {"tracking","",{"Tracking","Tracking_dT","Tracking_adv"}},
                {"region_window","",{"ROI"}},
                {"background_rendering","",{"Rendering"}},
                {"slice_rendering","show_slice",{"Slice"}},
                {"tract_rendering","show_tract",{"Tract","Tract_color"}},
                {"region_rendering","show_region",{"Region","Region_color","Region_graph"}},
                {"surface_rendering","show_surface",{"Surface"}},
                {"device_rendering","show_device",{"Device"}},
                {"label_rendering","show_label",{"Label"}},
                {"odf_rendering","show_odf",{"ODF"}}
            };

        auto get_params = [&](const domain_type& domain)
        {
            QStringList params;
            if(*domain.root_param)
                params << domain.root_param;
            for(auto group : domain.groups)
                if(group)
                    params += renderWidget->treemodel->get_param_list(group);
            return params;
        };

        auto print = [&](const domain_type& domain)
        {
            tipl::out() << std::string("[")+domain.name+"]";
            tipl::out() << "id\tvalue";
            for(const auto& id : get_params(domain))
                tipl::out() << id.toStdString() << "\t"
                            << renderWidget->getData(id).toString().toStdString();
        };

        auto id = QString::fromStdString(cmd[1]).trimmed().toLower();
        id.replace('-','_');

        if(id.isEmpty() || id == "all")
        {
            for(const auto& domain : domains)
                print(domain);
            return run->succeed();
        }

        for(const auto& domain : domains)
            if(id == domain.name)
            {
                print(domain);
                return run->succeed();
            }

        if(!renderWidget->treemodel->getParamList().contains(id))
            return run->failed(
                "invalid parameter or domain: "+cmd[1]+
                "; use list_param without an argument to list all domains");

        auto& item = (*renderWidget->treemodel)[id];
        tipl::out() << id.toStdString() << "\t" << item.getValue().toString().toStdString();
        if(auto* combo = qobject_cast<QComboBox*>(item.GUI)) // dropdown: content can change at runtime (e.g. loaded metrics)
        {
            QStringList options;
            for(int i = 0;i < combo->count();++i)
                options << QString("%1=%2").arg(i).arg(combo->itemText(i));
            tipl::out() << "options: " << options.join(", ").toStdString();
        }
        return run->succeed();
    }
    if(cmd[0] == "set_param" || cmd[0] == "set_params")
    {
        if(cmd[0] == "set_param")
        {
            set_data(cmd[1].c_str(),cmd[2].c_str());
            renderWidget->reveal(cmd[1].c_str()); // show which parameter this AI call just changed
        }
        else
            for(auto param : tipl::split(cmd[1],'&'))
            {
                auto pos = param.find('=');
                if(pos != std::string::npos)
                {
                    set_data(param.substr(0,pos).c_str(),
                             param.substr(pos+1).c_str());
                    renderWidget->reveal(param.substr(0,pos).c_str());
                }
            }
        glWidget->update();
        slice_need_update |= position_updated;
        return run->succeed();
    }
    if(cmd[0] == "set_region_name" || cmd[0] == "set_region_color" ||
        cmd[0] == "set_region_type")
    {
        bool ok;
        int row = QString::fromStdString(cmd[1]).toInt(&ok);
        if(!ok || row < 0 || row >= regionWidget->rowCount())
            return run->failed("invalid region index: " + cmd[1]);
        if(cmd[0] == "set_region_name")
        {
            if(cmd[2].empty())
                return run->failed("region name cannot be empty");
            regionWidget->item(row,0)->setText(QString::fromStdString(cmd[2]));
        }
        else if(cmd[0] == "set_region_type")
        {
            int type = QString::fromStdString(cmd[2]).toInt(&ok);
            if(!ok || type < 0 || type > 6)
                return run->failed("invalid region type: " + cmd[2]);
            regionWidget->item(row,1)->setText(QString::number(type));
        }
        else
        {
            uint color = QString::fromStdString(cmd[2]).toUInt(&ok);
            if(!ok)
                return run->failed("invalid region color: " + cmd[2]);
            regionWidget->item(row,2)->setData(Qt::UserRole,color);
        }
        return run->succeed();
    }
    if(cmd[0] == "add_slice" || cmd[0] == "add_mni_slice")
    {
        // cmd[1] : file name
        if(!cmd[1].empty())
        {
            if(cmd[0] == "add_mni_slice" && !handle->map_to_mni())
                return run->failed(handle->error_msg);
            auto slice = std::make_shared<CustomSliceModel>(handle,tipl::to_path(tipl::split(cmd[1],',')));
            slice->is_mni = (cmd[0] == "add_mni_slice");
            if(!slice->load_slices())
                return run->failed(error_msg = slice->error_msg);
            addSlices(slice);
            ui->SliceModality->setCurrentIndex(ui->SliceModality->count()-1);
            if(slice->running)
                start_reg();
            updateSlicesMenu();
            set_data("show_slice",Qt::Checked);
            glWidget->update();
            slice_need_update |= image_updated;
            return run->succeed();
        }

        auto filenames = tipl::qt::open_image_files(
            this,QFileInfo(work_path).absolutePath(),
                    "Image files (*.dcm *.hdr *.nii *nii.gz *db.fz *db.fib.gz *.dz 2dseq);;Histology (*.jpg *.tif);;All files (*)" );
        if(filenames.isEmpty())
            return run->canceled();

        if(filenames[0].endsWith(".dcm") && filenames.size() == 1)
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
                    return run->canceled();
                if(result == QMessageBox::Yes)
                {
                    filenames = file_list;
                    for(int index = 0;index < filenames.size();++index)
                        filenames[index] = directory.absolutePath() + "/" + filenames[index];
                }
            }
        }
        --history.current_recording_instance;
        if(filenames[0].endsWith(".nii.gz"))
        {
            for(const auto& each : filenames)
                command({cmd[0],each.toStdString()});

        }
        else
            command({cmd[0],filenames.join(',').toStdString()});
        ++history.current_recording_instance;
        return run->canceled();
    }
    if(cmd[0] == "skull_strip_slice")
    {
        auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(
                    slices[run->from_cmd(1,ui->SliceModality->currentIndex())]);
        if(!reg_slice.get())
            return run->canceled();
        tipl::image<3,unsigned char> mask;
        if(!handle->get_template_mask(reg_slice->source_images.shape(),
                                      reg_slice->to_dif,mask))
            return run->failed(handle->error_msg);

        tipl::image<3> maskJ(mask);
        tipl::filter::gaussian(maskJ);
        tipl::filter::gaussian(maskJ);
        reg_slice->source_images *= maskJ;
        slice_need_update |= image_updated;
        glWidget->update_slice();
        return run->succeed();
    }
    if(cmd[0] == "save_slice_mapping" || cmd[0] == "open_slice_mapping" || cmd[0] == "save_slice_volume")
    {
        // cmd[1] : file name
        // cmd[2] : slice index
        int slice_index = run->from_cmd(2,ui->SliceModality->currentIndex());
        if(slice_index < 0 || slice_index >= slices.size())
            return run->canceled();
        auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(slices[slice_index]);
        if(!reg_slice.get())
            return run->failed("cannot apply to built-in slices.");
        if(!history.get_filename(this,cmd[1],ui->SliceModality->currentText().toStdString()))
            return run->canceled();

        if(cmd[0] == "save_slice_volume")
        {
            if(!(tipl::io::gz_nifti(cmd[1],std::ios::out) << reg_slice->binded_image()))
                return run->failed("cannot save mapping to " + cmd[1]);
        }
        else
        if(cmd[0] == "save_slice_mapping")
        {
            if(!reg_slice->save_mapping(cmd[1]))
                return run->failed("cannot save mapping to " + cmd[1]);
        }
        else
        {
            reg_slice->terminate();
            if(!reg_slice->load_mapping(cmd[1]))
                return run->failed("invalid linear registration file " + cmd[1]);
        }
        return run->succeed();
    }
    if(cmd[0] == "delete_slice")
    {
        // cmd[1] : slice index
        int slice_index = run->from_cmd(1,ui->SliceModality->currentIndex());
        auto reg_slice = std::dynamic_pointer_cast<CustomSliceModel>(slices[slice_index]);
        if(!reg_slice.get())
            return run->failed("cannot delete built-in slices.");
        slices.erase(slices.begin()+slice_index);
        glWidget->slice_texture.erase(glWidget->slice_texture.begin()+slice_index);
        ui->SliceModality->removeItem(slice_index);
        updateSlicesMenu();
        return run->succeed();
    }
    if(tipl::begins_with(cmd[0],"add_surface"))
    {
        // cmd[1] : slice index
        // cmd[2] : threshold
        tipl::image<3> crop_image;
        float resolution_ratio = 1.0;
        auto slice_index = run->from_cmd(1,ui->SliceModality->currentIndex());
        if(slice_index >= slices.size())
            return run->failed("invalid slice index " + cmd[1]);
        auto this_slice = slices[slice_index];
        bool is_wm = (this_slice->get_name() == "wm_template");

        if(!std::dynamic_pointer_cast<CustomSliceModel>(this_slice).get())
        {
            // use ICBM152 wm as the surface
            tipl::matrix<4,4,float> trans;
            if(tipl::io::gz_nifti(handle->wm_template_file_name,std::ios::in) >> crop_image >> trans)
            {
                if(handle->mni2sub(crop_image,trans))
                    is_wm = true;
                else
                    crop_image.clear();
            }
        }

        if(crop_image.empty())
            crop_image = this_slice->get_source();

        float threshold = is_wm ? 25.0f : tipl::segmentation::otsu_threshold(crop_image)*1.25f;
        if(cmd[2].empty())
        {
            bool ok;
            threshold = float(QInputDialog::getDouble(this,QApplication::applicationName(),"Threshold:", double(threshold),
                    double(tipl::min_value(crop_image)),
                    double(tipl::max_value(crop_image)),
                    4, &ok));
            if (!ok)
                return run->canceled();
        }
        threshold = run->from_cmd(2,threshold);

        {
            glWidget->surface = std::make_shared<RegionRender>();
            {
                tipl::image<3,unsigned char> remain_part;
                if(tipl::contains(cmd[0],"left"))
                {
                    remain_part.resize(crop_image.shape());
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.width())
                    {
                        std::fill(remain_part.begin()+index+this_slice->slice_pos[0],
                                  remain_part.begin()+index+remain_part.width(),1);
                    }
                }
                if(tipl::contains(cmd[0],"right"))
                {
                    remain_part.resize(crop_image.shape());
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.width())
                    {
                        std::fill(remain_part.begin()+index,
                                  remain_part.begin()+index+this_slice->slice_pos[0],1);
                    }
                }
                if(tipl::contains(cmd[0],"upper"))
                {
                    remain_part.resize(crop_image.shape());
                    std::fill(remain_part.begin()+this_slice->slice_pos[2]*remain_part.plane_size(),
                              remain_part.end(),1);
                }
                if(tipl::contains(cmd[0],"lower"))
                {
                    remain_part.resize(crop_image.shape());
                    std::fill(remain_part.begin(),
                              remain_part.begin()+this_slice->slice_pos[2]*remain_part.plane_size(),1);
                }
                if(tipl::contains(cmd[0],"posterior"))
                {
                    remain_part.resize(crop_image.shape());
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.plane_size())
                    {
                        std::fill(remain_part.begin()+index+int64_t(this_slice->slice_pos[1])*remain_part.width(),
                                  remain_part.begin()+index+int64_t(remain_part.plane_size()),1);
                    }
                }
                if(tipl::contains(cmd[0],"anterior"))
                {
                    remain_part.resize(crop_image.shape());
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.plane_size())
                    {
                        std::fill(remain_part.begin()+index,
                                  remain_part.begin()+index+int64_t(this_slice->slice_pos[1])*remain_part.width(),1);
                    }
                }
                if(!remain_part.empty())
                    crop_image *= remain_part;
            }


            switch((*this)["surface_mesh_smoothed"].toInt())
            {
            case 1:
                tipl::filter::gaussian(crop_image);
                break;
            case 2:
                {
                    crop_image *= tipl::filter::gaussian(
                    tipl::image<3>(tipl::morphology::dndnco(crop_image > threshold)),2);
                }
                break;
            }
            if(!glWidget->surface->load(crop_image,threshold))
            {
                glWidget->surface.reset();
                return run->succeed();
            }
        }

        if(!this_slice->is_diffusion_space)
            glWidget->surface->transform_point_list(this_slice->to_dif);

        glWidget->update();
        return run->succeed();
    }

    return run->failed("unknown command: " + cmd[0]);
}
bool tracking_window::command(std::vector<std::string> cmd,
                              command_source source)
{
    struct restore_dispatch_state{
        command_history& history;
        command_source source;
        std::string ai_forwarding_cmd;
        ~restore_dispatch_state(){history.source = source;history.ai_forwarding_cmd = ai_forwarding_cmd;}
    } restore{history,history.source,history.ai_forwarding_cmd};
    history.source = source;
    history.ai_forwarding_cmd = cmd.empty() ? std::string() : cmd[0];
    return command(std::move(cmd));
}

std::string tracking_window::get_parameter_id(bool auto_track)
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
    param.track_voxel_ratio = renderWidget->getData("track_voxel_ratio").toFloat();
    param.default_otsu = renderWidget->getData("otsu_threshold").toFloat();
    param.tip_iteration =
            // only used in automatic fiber tracking (auto_track, decided by the caller)
            // or differential tractography
            (auto_track || renderWidget->getData("dt_index1").toInt() > 0)
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



void tracking_window::on_actionTract_Analysis_Report_triggered()
{
    if(tractWidget->tract_models.empty())
        return;
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
    connectivity_matrix.reset(new connectivity_matrix_dialog(this));
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
        if(!in.load_from_file(filename.toStdString()))
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
        std::copy_n(buf,row*col,glWidget->connectivity.begin());



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
                    command({"add_region_from_atlas",std::to_string(handle->template_id)+" "+std::to_string(i)});
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
        std::ifstream in(tipl::qt::to_path(filename));
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
    slice_need_update |= position_updated;
    if(all_ended)
    {
        timer2.reset();
        history.has_other_thread = false;
    }
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
bool tracking_window::addSlices(const std::string& name,const std::filesystem::path& path)
{
    if(!tipl::begins_with(path.u8string(),"http") && !std::filesystem::exists(path))
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
    history.has_other_thread = true;
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
    if(filename.isEmpty() || !command({"add_slice",filename.toStdString()}))
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

    slice_need_update |= position_updated;
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
        tipl::reg::subject_image_pre(tipl::image<3>(reg_slice->get_source())),tipl::reg::subject_image_pre(tipl::image<3>(reg_slice->get_source())),reg_slice->vs,
        tipl::reg::subject_image_pre(tipl::image<3>(iso_fa.first)),tipl::reg::subject_image_pre(tipl::image<3>(iso_fa.second)),handle->vs,
        tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
    manual->from_T = reg_slice->trans_to_mni;
    manual->to_T = handle->trans_to_mni;

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



void tracking_window::on_actionSave_Slices_to_DICOM_triggered()
{
    auto slice = std::dynamic_pointer_cast<CustomSliceModel>(current_slice);
    if(!slice.get() || slice->source_files.empty())
    {
        QMessageBox::critical(this,"ERROR","This function needs original DICOM files (loading them at the[Slices] menu)");
        return;
    }

    QMessageBox::information(this,QApplication::applicationName(),"Please assign the output directory");
    QString dir = QFileDialog::getExistingDirectory(this,"Assign output directory",tipl::qt::to_qstring(slice->source_files[0].parent_path()));
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
                QMessageBox::critical(this,"ERROR",QString("Failed to load the original DICOM files: ") + tipl::qt::to_qstring(slice->source_files[i]));
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
        std::copy_n(out.begin()+pos,read_size,reinterpret_cast<short*>(&*(buf.end()-int(read_size*sizeof(short)))));

        QString output_name = dir + "/mod_" + tipl::qt::to_qstring(slice->source_files[i].stem()) + ".dcm";

        if(i == 0 && QFileInfo(output_name).exists() &&
           QMessageBox::information(this,QApplication::applicationName(),"Previous modifications found. Overwrite?",
           QMessageBox::Yes|QMessageBox::Cancel) == QMessageBox::Cancel)
                return;

        std::ofstream out(tipl::qt::to_path(output_name),std::ios::binary);
        if(!out)
        {
            QMessageBox::critical(this,"ERROR","Cannot output DICOM. Please check disk space or output permission.");
            return;
        }
        out.write(&buf[0],int64_t(buf.size()));
    }
    QMessageBox::information(this,QApplication::applicationName(),"File Saved");
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
        tipl::reg::template_image_pre(tipl::image<3>(handle->template_I)),tipl::reg::template_image_pre(tipl::image<3>(handle->template_I2)),handle->template_vs,
        tipl::reg::subject_image_pre(tipl::image<3>(iso_fa.first)),tipl::reg::subject_image_pre(tipl::image<3>(iso_fa.second)),handle->vs,
        tipl::reg::affine,tipl::reg::cost_type::mutual_info));
    manual->from_T = manual->to_T = handle->trans_to_mni;
    if(manual->exec() != QDialog::Accepted)
        return;
    handle->manual_template_T = manual->arg;
    handle->has_manual_atlas = true;


    auto output_file_name = handle->fib_file_name;
    output_file_name += "." + template_name_list[handle->template_id] + ".mz";
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


void tracking_window::update_unet_models(void)
{
    if(!current_slice)
        return;
    bool is_t1 = tipl::contains_case_insensitive(current_slice->get_name(),{"t1","mpr"});
    bool is_t2 = tipl::contains_case_insensitive(current_slice->get_name(),{"t2","tse"});
    bool is_flair = tipl::contains_case_insensitive(current_slice->get_name(),{"flair","t2f"});
    for(auto& each : ui->menuSegment->actions())
    {
        if(each->data().toString().toLower().contains("t1") && !is_t1)
        {
            each->setText("(need T1w)" + each->statusTip());
            each->setEnabled(false);
        }
        else
            if(each->data().toString().toLower().contains("t2") && !is_t2)
            {
                each->setText("(need T2w)" + each->statusTip());
                each->setEnabled(false);
            }
            else
                if(each->data().toString().toLower().contains("flair") && !is_flair)
                {
                    each->setText("(need FLAIR)" + each->statusTip());
                    each->setEnabled(false);
                }
                else
                {
                    each->setText(each->statusTip());
                    each->setEnabled(true);
                }

    }
}

void tracking_window::on_template_box_currentIndexChanged(int set_template_id)
{
    if(set_template_id < 0 || set_template_id >= int(template_name_list.size()))
        return;
    handle->set_template_id(size_t(set_template_id));
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
    ui->enable_auto_tract->setVisible(true);
    ui->addRegionFromAtlas->setVisible(!handle->atlas_list.empty());

    ui->menuSegment->clear();
    for(size_t i = 0;i < unet_names[set_template_id].size();++i)
    {
        QAction* added_action;
        ui->menuSegment->addAction(added_action = addSubMenuItem(
            std::filesystem::path(unet_http[set_template_id][i]).stem().u8string(),unet_names[set_template_id][i],"run segment_brain"));
        added_action->setStatusTip(QString::fromStdString(unet_names[set_template_id][i]));
        added_action->setWhatsThis(QString::fromStdString(unet_desc[set_template_id][i]));
    }
    update_unet_models();

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


void paint_track_on_volume(tipl::image<3,unsigned char>& track_map,const std::vector<std::vector<float> >& all_tracts,
                           std::shared_ptr<SliceModel> slice)
{
    tipl::par_for(all_tracts.size(),[&](unsigned int i)
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
    auto mask = current_region->to_mask();
    if(current_region->to_diffusion_space != slice->to_dif)
    {
        tipl::image<3,unsigned char> new_mask(slice->dim);
        tipl::resample<tipl::interpolation::majority>(mask,new_mask,
            tipl::transformation_matrix<float>(tipl::from_space(slice->to_dif).to(current_region->to_diffusion_space)));
        mask.swap(new_mask);
    }

    for(size_t i = 0,sz = mask.size();i < sz;++i)
        if(mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update |= image_updated;
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
    for(size_t i = 0,sz = t_mask.size();i < sz;++i)
        if(t_mask[i])
            slice->source_images[i] = mark_value;
    slice_need_update |= image_updated;
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
    if(!new_color_map.load_from_file(filename.toStdString()))
    {
          QMessageBox::critical(this,"ERROR","Invalid color map format");
          return;
    }
    current_slice->view->v2c.set_color_map(new_color_map);
    slice_need_update |= image_updated;
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
    std::ofstream out(tipl::qt::to_path(filename)),mtl(tipl::qt::to_path(filename+".mtl"));
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

