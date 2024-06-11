#include <map>
#include <QTextStream>
#include <QInputDialog>
#include <QFileDialog>
#include "view_image.h"
#include "ui_view_image.h"
#include <QPlainTextEdit>
#include <QFileInfo>
#include <QMessageBox>
#include <QBuffer>
#include <QImageReader>
#include <filesystem>

std::vector<view_image*> opened_images;

bool resize_mat(tipl::io::gz_mat_read& mat_reader,const tipl::shape<3>& new_dim);
bool translocate_mat(tipl::io::gz_mat_read& mat_reader,const tipl::vector<3,int>& shift);
bool resample_mat(tipl::io::gz_mat_read& mat_reader,float resolution);

void view_image::read_4d_at(size_t new_index)
{
    if(new_index == cur_4d_index ||
       new_index >= nifti.dim(4))
        return;
    cur_image->apply([&](auto& I)
    {
        // give image buffer back
        if(!I.buf().empty())
            I.buf().swap(buf4d[cur_4d_index]);

        if(buf4d[new_index].empty())
        {
            I.resize(cur_image->shape);
            tipl::show_prog = false;
            nifti.select_volume(new_index);
            nifti.get_untouched_image(I);
            tipl::show_prog = true;
        }
        else
            I.buf().swap(buf4d[new_index]);
    });

    cur_4d_index = size_t(new_index);
}
void view_image::get_4d_buf(std::vector<unsigned char>& buf)
{
    size_t size_per_image = cur_image->buf_size();
    buf.resize(size_per_image*buf4d.size());
    cur_image->apply([&](auto& I)
    {
        for(size_t i = 0;i < buf4d.size();++i)
        {
            read_4d_at(i);
            std::memcpy(buf.data() + i*size_per_image,I.buf().data(),size_per_image);
        }
    });
}

bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param);
bool view_image::command(std::string cmd,std::string param1)
{
    if(cur_image->empty())
        return true;
    if(param1.empty())
        tipl::out() << cmd;
    else
        tipl::out() << cmd << ":" << param1;

    error_msg.clear();
    bool result = true;


    if(mat.size())
    {
        result = modify_fib(mat,cmd,param1);
        if(result)
            read_mat();
        goto end_command;
    }

    if((cmd == "normalize" || cmd == "normalize_otsu_median") && cur_image->pixel_type != variant_image::float32)
        command("change_type",std::to_string(int(variant_image::float32)));


    if(cmd == "reshape")
    {
        std::istringstream in(param1);
        tipl::shape<3> new_shape;
        int dim4 = 1;
        in >> new_shape[0] >> new_shape[1] >> new_shape[2] >> dim4;
        if(!buf4d.empty() || dim4 != 1)
        {
            std::vector<unsigned char> buf;
            if(!buf4d.empty() && dim4 != 1)
                get_4d_buf(buf);
            else
                cur_image->apply([&](auto& I)
                {
                    I.buf().swap(buf);
                });

            cur_image->shape = new_shape;
            buf.resize(cur_image->buf_size()*dim4);

            if(dim4 == 1)
                buf4d.clear();
            else
            {
                size_t size_per_image = cur_image->buf_size();
                buf4d.resize(dim4);
                for(size_t i = 0;i < dim4;++i)
                {
                    buf4d[i].resize(size_per_image);
                    std::memcpy(buf4d[i].data(),buf.data()+i*size_per_image,size_per_image);
                }
                cur_4d_index = 0;
                buf.swap(buf4d[0]);
            }

            cur_image->apply([&](auto& I)
            {
                I.buf().swap(buf);
                I.resize(cur_image->shape);
            });

            // clear up undo and redo
            undo_list.clear();
            redo_list.clear();
            goto end_command;
        }
    }
    if(!buf4d.empty())
    {
        auto old_4d_index = cur_4d_index;
        if(cmd == "save")
        {
            std::vector<unsigned char> buf;
            get_4d_buf(buf);
            cur_image->apply([&](auto& I)
            {
                tipl::io::gz_nifti nii;
                nii.set_image_transformation(cur_image->T,cur_image->is_mni);
                nii.set_voxel_size(cur_image->vs);
                nii << tipl::make_image(reinterpret_cast<decltype(&I[0])>(buf.data()),tipl::shape<4>(cur_image->shape[0],cur_image->shape[1],cur_image->shape[2],buf4d.size()));
                result = nii.save_to_file(param1.c_str());
            });
            goto end_command;
        }

        {
            auto old_shape = cur_image->shape;
            auto old_vs = cur_image->vs;
            auto old_T = cur_image->T;
            auto old_type = cur_image->pixel_type;
            auto old_4d_index = cur_4d_index;
            for(size_t i = 0;i < buf4d.size() && result;++i)
            {
                cur_image->shape = old_shape;
                cur_image->vs = old_vs;
                cur_image->T = old_T;
                cur_image->pixel_type = old_type;
                read_4d_at(i);
                result = cur_image->command(cmd,param1);
            }
        }

        read_4d_at(old_4d_index);
        goto end_command;
    }


    undo_list.push_back(std::make_shared<variant_image>(*cur_image.get()));
    cur_image->command(cmd,param1);

    end_command:

    if(!result)
    {
        tipl::out() << "ERROR: " << error_msg << std::endl;
        if(!undo_list.empty())
        {
            swap(cur_image,undo_list.back());
            undo_list.pop_back();
        }
        return false;
    }


    init_image();

    command_list.push_back(cmd);
    param_list.push_back(param1);

    redo_list.clear();
    redo_command_list.clear();
    redo_param_list.clear();

    if(cmd == "save" && !file_names.empty())
    {
        if(QMessageBox::question(nullptr,"DSI Studio","Applying processing to other images and save them?",
                                 QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::No)
        {
            file_names.clear();
            return true;
        }

        tipl::progress prog("apply to other images");
        int file_index = 0;
        for(;prog(file_index,file_names.size());++file_index)
        {
            auto file_name2 = file_names[file_index];
            tipl::out() << "processing " << file_name2.toStdString();

            std::shared_ptr<view_image> dialog(new view_image(parentWidget()));
            dialog->setAttribute(Qt::WA_DeleteOnClose);
            if(!dialog->open(QStringList() << file_name2))
            {
                QMessageBox::critical(this,"ERROR",QString("Cannot open ")+file_name2);
                break;
            }
            for(size_t i = 0;i < param_list.size();++i)
            {
                std::string param2 = param_list[i];
                if(command_list[i] == "save" ||
                   command_list[i].find("_image") != std::string::npos)
                {
                    if(tipl::match_files(original_file_name.toStdString(),param_list[i],
                                file_name2.toStdString(),param2))
                    {
                        tipl::out() << "matched path: " << std::filesystem::path(param2).parent_path().string() << std::endl;
                        tipl::out() << "matched file name: " << std::filesystem::path(param2).filename().string() << std::endl;
                    }
                }
                if(!dialog->command(command_list[i],param2))
                {
                    QMessageBox::critical(this,"ERROR",QString(dialog->error_msg.c_str()) + "\n"
                                          + command_list[i].c_str() + " at\n"
                                          + file_name2);
                    goto end;
                }
            }
        }
        if(prog.aborted())
            return false;
        end:
        if(file_index < file_names.size() && // The processed is aborted, or there is an error happened
           file_index && // Some files were processed without a problem. file_index=0 is current image, file_index = 1 is the first to-be processed image.
           original_file_name.toStdString() == param1) // those files were overwritten to original file
        {
            QMessageBox::critical(this,"ERROR","Some files were processed and overwritten. They will be ignored in the next analyses");
            #ifdef QT6_PATCH
            file_names.remove(0,file_index);
            #else
            for(int i = 0;i < file_index;++i)
                file_names.removeFirst();
            #endif
            // remove the last save command
            command_list.pop_back();
            param_list.pop_back();
            return true;
        }

        command_list.clear();
        param_list.clear();
    }
    return true;
}

bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>& trans)
{
    if(filenames.size() == 1 && filenames[0].toLower().contains("nii"))
    {
        tipl::io::gz_nifti in;
        if(!in.load_from_file(filenames[0].toStdString().c_str()) || !in.toLPS(ref))
        {
            QMessageBox::information(nullptr,"Error","Not a valid nifti file");
            return false;
        }
        in.get_voxel_size(vs);
        in.get_image_transformation(trans);
        return true;
    }
    else
        if(filenames.size() == 1 && filenames[0].contains("2dseq"))
        {
            tipl::io::bruker_2dseq seq;
            if(!seq.load_from_file(filenames[0].toStdString().c_str()))
            {
                QMessageBox::information(nullptr,"Error","Not a valid 2dseq file");
                return false;
            }
            seq.get_image().swap(ref);
            seq.get_voxel_size(vs);
            return true;
        }
    else
    {
        tipl::io::dicom_volume v;
        std::vector<std::string> file_list;
        for(int i = 0;i < filenames.size();++i)
            file_list.push_back(filenames[i].toStdString());
        v.load_from_files(file_list);
        v >> ref;
        v.get_voxel_size(vs);
        return true;
    }
}
TableKeyEventWatcher::TableKeyEventWatcher(QTableWidget* table_):table(table_)
{
    table_->installEventFilter(this);
}
bool TableKeyEventWatcher::eventFilter(QObject * receiver, QEvent * event)
{
    auto table = qobject_cast<QTableWidget*>(receiver);
    if (table && event->type() == QEvent::KeyPress)
    {
        auto keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Delete && keyEvent->modifiers() & Qt::ControlModifier)
            emit DeleteRowPressed(table->currentRow());
    }
    return false;
}


view_image::view_image(QWidget *parent) :
    QMainWindow(parent),
    cur_image(new variant_image),
    ui(new Ui::view_image)
{
    ui->setupUi(this);

    table_event.reset(new TableKeyEventWatcher(ui->info));
    connect(table_event.get(),SIGNAL(DeleteRowPressed(int)),this,SLOT(DeleteRowPressed(int)));

    ui->mat_images->hide();
    ui->info->setColumnWidth(0,120);
    ui->info->setColumnWidth(1,200);
    ui->info->setHorizontalHeaderLabels(QStringList() << "Header" << "Value");
    ui->view->setScene(&source);
    ui->max_color->setColor(0xFFFFFFFF);
    ui->min_color->setColor(0XFF000000);


    foreach (QAction* action, findChildren<QAction*>())
    {
        if(action->text().contains("&") || action->text().isEmpty())
            continue;
        if(action->text().contains("..."))
            connect(action, SIGNAL(triggered()),this, SLOT(run_action2()));
        else
            connect(action, SIGNAL(triggered()),this, SLOT(run_action()));
    }


    connect(ui->max_color,SIGNAL(clicked()),this,SLOT(change_contrast()));
    connect(ui->min_color,SIGNAL(clicked()),this,SLOT(change_contrast()));
    connect(ui->orientation,SIGNAL(currentIndexChanged(int)),this,SLOT(change_contrast()));
    connect(ui->axis_grid,SIGNAL(currentIndexChanged(int)),this,SLOT(change_contrast()));
    connect(ui->overlay_style,SIGNAL(currentIndexChanged(int)),this,SLOT(change_contrast()));
    connect(ui->menuOverlay, SIGNAL(aboutToShow()),this, SLOT(update_overlay_menu()));


    ui->tabWidget->setCurrentIndex(0);
    ui->overlay_style->setVisible(false);

    qApp->installEventFilter(this);
    this_index = opened_images.size();
    opened_images.push_back(this);
}


view_image::~view_image()
{
    opened_images[this_index] = nullptr;
    update_other_images();
    qApp->removeEventFilter(this);
    delete ui;
}
bool view_image::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::Wheel && obj->parent() == ui->view)
    {
        QWheelEvent* we = dynamic_cast<QWheelEvent*>(event);
        if(!we)
            return false;
        if(we->angleDelta().y() < 0)
            ui->zoom->setValue(ui->zoom->value()*0.8f);
        else
            ui->zoom->setValue(ui->zoom->value()*1.2f);
        event->accept();

        return true;
    }
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->view)
        return false;

    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    auto x = point.x();
    auto y = point.y();
    if(has_flip_x())
        x = source.width() - x;
    if(has_flip_y())
        y = source.height() - y;

    auto pos = tipl::slice2space<tipl::vector<3,float> > (cur_dim,
                      std::round(float(x) / ui->zoom->value()),
                      std::round(float(y) / ui->zoom->value()),ui->slice_pos->value());
    if(!cur_image->shape.is_valid(pos))
        return true;
    auto mni = pos;
    mni.to(cur_image->T);

    cur_image->apply([&](auto& data)
    {
        ui->info_label->setText(QString("(i,j,k)=(%1,%2,%3) (x,y,z)=(%4,%5,%6) value=%7").arg(pos[0]).arg(pos[1]).arg(pos[2])
                                                                        .arg(mni[0]).arg(mni[1]).arg(mni[2])
                                                                        .arg(float(data.at(pos))));
    });

    return true;
}



void view_image::read_mat_info(void)
{
    QString info;
    for(unsigned int index = 0;index < mat.size();++index)
    {
        std::string data;
        mat[index].get_info(data);
        info += data.c_str();
        info += "\n";
    }
    show_info(info);
}
void view_image::DeleteRowPressed(int row)
{
    if(ui->info->currentRow() == -1)
        return;
    if(mat.size())
        command("remove",std::to_string(row));
}
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
bool view_image::read_mat(void)
{
    if(!mat.read("dimension",cur_image->shape))
    {
        error_msg = "cannot find dimension matrix";
        return false;
    }
    bool has_data = true;
    ui->mat_images->clear();
    for(size_t i = 0;i < mat.size();++i)
        if(mat[i].get_cols()*mat[i].get_rows() == cur_image->shape.size())
            ui->mat_images->addItem(mat[i].get_name().c_str());
    if(!ui->mat_images->count())
    {
        error_msg = "cannot find images";
        return false;
    }

    mat.get_voxel_size(cur_image->vs);
    if(mat.has("trans"))
        mat.read("trans",cur_image->T);
    else
        initial_LPS_nifti_srow(cur_image->T,cur_image->shape,cur_image->vs);
    read_mat_info();
    ui->mat_images->setCurrentIndex(0);
    ui->mat_images->show();
    return true;
}

void view_image::on_actionLoad_Image_to_4D_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open image",original_file_name,"NIFTI files (*.nii *nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    tipl::image<3> new_image(cur_image->shape);
    if(!tipl::io::gz_nifti::load_to_space(filename.toStdString().c_str(),new_image,cur_image->T))
    {
        QMessageBox::critical(this,"Error","Invalid NIFTI file");
        return;
    }
    if(buf4d.empty())
        buf4d.push_back(std::vector<unsigned char>());
    buf4d.push_back(std::vector<unsigned char>());
    read_4d_at(buf4d.size()-1);
    cur_image->apply([&](auto& I)
    {
        I = new_image;
    });
    init_image();
}
void prepare_idx(const char* file_name,std::shared_ptr<tipl::io::gz_istream> in);
QImage read_qimage(QString filename,std::string& error);
bool view_image::open(QStringList file_names_)
{
    if(file_names_.empty())
        return false;
    no_update = true;


    file_names = file_names_;
    original_file_name = file_name = file_names[0];
    file_names.removeFirst();
    setWindowTitle(QFileInfo(file_name).fileName());


    std::string info;
    if(file_names.size() > 1 &&
       (QString(file_name).endsWith(".bmp") ||
        QString(file_name).endsWith(".png") ||
        QString(file_name).endsWith(".tif") ||
        QString(file_name).endsWith(".tiff")))
    {
        QImage in = read_qimage(file_name,error_msg);
        if(in.isNull())
            return false;
        cur_image->pixel_type = variant_image::int8;
        cur_image->shape[0] = in.width();
        cur_image->shape[1] = in.height();
        cur_image->shape[2] = uint32_t(file_names.size());
        cur_image->T.identity();
        cur_image->vs[0] = cur_image->vs[1] = cur_image->vs[2] = 1.0f;
        cur_image->I_int8.resize(cur_image->shape);
        buf4d.resize(3);
        for(size_t i = 1;i < 3;++i)
            buf4d[i].resize(cur_image->shape.size());
        tipl::progress prog("open image file ",std::filesystem::path(file_name.toStdString()).filename().string().c_str());
        for(size_t file_index = 0;prog(file_index,cur_image->shape[2]);++file_index)
        {
            QImage I = read_qimage(file_names[file_index],error_msg);
            if(I.isNull())
                return false;

            if(I.width() != cur_image->shape[0] || I.height() != cur_image->shape[1])
            {
                error_msg = "inconsistent image size : ";
                error_msg += file_names[file_index].toStdString();
                return false;
            }
            I = I.convertToFormat(QImage::Format_RGB32);
            auto ptr = reinterpret_cast<uint32_t*>(I.bits());
            for(size_t i = 0,pos = file_index*cur_image->shape.plane_size();i < cur_image->shape.plane_size();++i,++pos)
            {
                tipl::rgb rgb(ptr[i]);
                cur_image->I_int8[pos] = rgb.r;
                buf4d[1][pos] = rgb.g;
                buf4d[2][pos] = rgb.b;
            }
        }
        if(prog.aborted())
            return false;
        file_names.clear();
        cur_4d_index = 0;
    }
    else
        if(QString(file_name).endsWith(".mat") ||
           QString(file_name).endsWith("fib.gz") ||
           QString(file_name).endsWith("src.gz"))
        {
            tipl::progress prog("open file ",std::filesystem::path(file_name.toStdString()).filename().string().c_str());
            if(!mat.load_from_file(file_name.toStdString().c_str()))
            {
                error_msg = "invalid format";
                return false;
            }
            if(!read_mat())
                return false;

        }
        else
        if(!cur_image->load_from_file(file_name.toStdString().c_str(),info))
        {
            QMessageBox::critical(this,"ERROR",(error_msg = cur_image->error_msg).c_str());
            return false;
        }

    if(cur_image->error_msg == "4d image")
    {
        prepare_idx(file_name.toStdString().c_str(),nifti.input_stream);
        if(!nifti.load_from_file(file_name.toStdString().c_str()))
        {
            error_msg = nifti.error_msg;
            return false;
        }
        buf4d.resize(nifti.dim(4));
        nifti.input_stream->sample_access_point = true;
        ui->dwi_volume->setValue(0);
    }
    if(!info.empty())
        show_info(info.c_str());
    ui->type->setCurrentIndex(cur_image->pixel_type);
    ui->zoom->setValue(0.9f*width()/cur_image->shape.width());

    if(cur_image->shape.size())
        init_image();
    return cur_image->shape.size() || !!info.empty();
}

void view_image::init_image(void)
{
    no_update = true;
    float min_value = 0.0f;
    float max_value = 0.0f;
    cur_image->apply([&](auto& data)
    {
        auto minmax = tipl::minmax_value(data);
        min_value = minmax.first;
        max_value = minmax.second;
    });

    float range = max_value-min_value;
    QString dim_text = QString("%1,%2,%3").arg(cur_image->shape.width()).arg(cur_image->shape.height()).arg(cur_image->shape.depth());
    if(!buf4d.empty())
        dim_text += QString(",%1").arg(buf4d.size());
    ui->image_info->setText(QString("dim=(%1) vs=(%4,%5,%6) srow=[%7 %8 %9 %10][%11 %12 %13 %14][%15 %16 %17 %18] %19").
            arg(dim_text).
            arg(double(cur_image->vs[0])).arg(double(cur_image->vs[1])).arg(double(cur_image->vs[2])).
            arg(double(cur_image->T[0])).arg(double(cur_image->T[1])).arg(double(cur_image->T[2])).arg(double(cur_image->T[3])).
            arg(double(cur_image->T[4])).arg(double(cur_image->T[5])).arg(double(cur_image->T[6])).arg(double(cur_image->T[7])).
            arg(double(cur_image->T[8])).arg(double(cur_image->T[9])).arg(double(cur_image->T[10])).arg(double(cur_image->T[11])).arg(cur_image->is_mni?"mni":"native"));

    if(ui->min->maximum() != double(max_value) ||
       ui->max->minimum() != double(min_value))
    {
        ui->min->setRange(double(min_value),double(max_value));
        ui->max->setRange(double(min_value),double(max_value));
        ui->min->setSingleStep(double(range/20));
        ui->max->setSingleStep(double(range/20));
        ui->min->setValue(double(min_value));
        ui->max->setValue(double(max_value));
    }
    if(ui->slice_pos->maximum() != int(cur_image->shape[cur_dim]-1))
    {
        ui->slice_pos->setRange(0,cur_image->shape[cur_dim]-1);
        slice_pos[0] = cur_image->shape.width()/2;
        slice_pos[1] = cur_image->shape.height()/2;
        slice_pos[2] = cur_image->shape.depth()/2;
        ui->slice_pos->setValue(slice_pos[cur_dim]);
    }

    ui->actionSet_MNI->setStatusTip(cur_image->is_mni ? "1":"0");
    ui->actionRegrid->setStatusTip(QString("%1 %2 %3").arg(cur_image->vs[0]).arg(cur_image->vs[1]).arg(cur_image->vs[2]));
    ui->actionResize->setStatusTip(QString("%1 %2 %3").arg(cur_image->shape[0]).arg(cur_image->shape[1]).arg(cur_image->shape[2]));
    ui->actionReshape->setStatusTip(buf4d.empty() ? QString("%1 %2 %3").arg(cur_image->shape[0]).arg(cur_image->shape[1]).arg(cur_image->shape[2]) :
                                                    QString("%1 %2 %3 %4").arg(cur_image->shape[0]).arg(cur_image->shape[1]).arg(cur_image->shape[2]).arg(buf4d.size()));
    ui->actionSet_Translocation->setStatusTip(QString("%1 %2 %3").arg(cur_image->T[3]).arg(cur_image->T[7]).arg(cur_image->T[11]));

    std::string t_string;
    {
        std::ostringstream out;
        for(int i = 0;i < 16;++i)
            out << cur_image->T[i] << " ";
        t_string = out.str();
    }
    ui->actionSet_Transformation->setStatusTip(t_string.c_str());
    ui->actionTransform->setStatusTip(t_string.c_str());

    if(buf4d.empty())
    {
        ui->dwi_volume->hide();
        ui->dwi_label->hide();
    }
    else
    {
        ui->dwi_volume->setMaximum(buf4d.size()-1);
        ui->dwi_volume->show();
        ui->dwi_label->show();
    }
    ui->type->setCurrentIndex(cur_image->pixel_type);
    no_update = false;
    show_image(true);
}
void view_image::set_overlay(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    overlay_images_visible[action->data().toInt()] = action->isChecked();
    show_image(false);
}
void view_image::update_overlay_menu(void)
{
    {
        std::vector<size_t> new_overlay_images;
        std::vector<bool> new_overlay_images_visible;
        for(size_t i = 0;i < opened_images.size();++i)
            if(opened_images[i] && this_index != i &&
               opened_images[i]->cur_image->shape == cur_image->shape)
            {
                new_overlay_images.push_back(i);
                auto pos = std::find(overlay_images.begin(),overlay_images.end(),i);
                new_overlay_images_visible.push_back(pos != overlay_images.end() && overlay_images_visible[pos-overlay_images.begin()]);
            }
        overlay_images.swap(new_overlay_images);
        overlay_images_visible.swap(new_overlay_images_visible);
    }

    while(ui->menuOverlay->actions().size() > int(overlay_images.size()))
        ui->menuOverlay->removeAction(ui->menuOverlay->actions()[ui->menuOverlay->actions().size()-1]);
    for (size_t index = 0; index < overlay_images.size(); ++index)
    {
        if(index >= size_t(ui->menuOverlay->actions().size()))
        {
            QAction* Item = new QAction(this);
            Item->setVisible(true);
            Item->setData(int(index));
            Item->setCheckable(true);
            connect(Item, SIGNAL(triggered()),this, SLOT(set_overlay()));
            ui->menuOverlay->addAction(Item);
        }
        auto action = ui->menuOverlay->actions()[index];
        action->setText(opened_images[overlay_images[index]]->windowTitle());
        action->setChecked(overlay_images_visible[index]);
    }
    ui->overlay_style->setVisible(!overlay_images.empty());
}
bool view_image::has_flip_x(void)
{
    bool flip_x = false;
    if(ui->orientation->currentIndex())
    {
        flip_x = cur_dim;
        // this handles the "to LPS"
        if(cur_image->T[0] > 0)
        {
            if(cur_dim == 2)
                flip_x = !flip_x;
            if(cur_dim == 1)
                flip_x = !flip_x;
        }
        if(cur_image->T[5] > 0)
        {
            if(cur_dim == 0)
                flip_x = !flip_x;
        }
    }
    return flip_x;
}
bool view_image::has_flip_y(void)
{
    bool flip_y = false;
    if(ui->orientation->currentIndex())
    {
        flip_y = (cur_dim != 2);
        if(cur_image->T[5] > 0)
        {
            if(cur_dim == 2)
                flip_y = !flip_y;
        }
    }
    return flip_y;
}
void view_image::show_info(QString info)
{
    QStringList list = info.split("\n");
    ui->info->clear();
    ui->info->setRowCount(list.size());
    for(int row = 0;row < list.size();++row)
    {
        QString line = list[row];
        QStringList value_list = line.split("=");
        ui->info->setItem(row,0, new QTableWidgetItem(value_list[0]));
        if(value_list.size() > 1)
            ui->info->setItem(row,1, new QTableWidgetItem(value_list[1]));
    }
    ui->info->selectRow(0);
}
void view_image::show_image(bool update_others)
{
    if(cur_image->empty() || no_update)
        return;

    tipl::color_image buffer;
    cur_image->apply([&](auto& data)
    {
        v2c.convert(tipl::volume2slice_scaled(data,cur_dim, size_t(slice_pos[cur_dim]),ui->zoom->value()),buffer);
    });
    if(ui->overlay_style->currentIndex() == 1)
        std::fill(buffer.begin(),buffer.end(),tipl::rgb(0,0,0));
    // draw overlay
    for(size_t i = 0;i < overlay_images.size() && ui->overlay_style->currentIndex() <= 1;++i)
    if(overlay_images_visible[i] && opened_images[overlay_images[i]])
        opened_images[overlay_images[i]]->cur_image->apply([&](auto& data)
        {
            tipl::color_image buffer2(
                opened_images[overlay_images[i]]->v2c[tipl::volume2slice_scaled(data,cur_dim, size_t(slice_pos[cur_dim]),ui->zoom->value())]);
            for(size_t j = 0;j < buffer.size();++j)
                buffer[j] |= buffer2[j];
        });
    source_image << buffer;
    source_image = source_image.mirrored(has_flip_x(),has_flip_y());
    {
        QPainter paint(&source_image);

        QPen pen;
        pen.setColor(Qt::white);
        paint.setPen(pen);
        paint.setFont(font());

        tipl::qt::draw_ruler(paint,cur_image->shape,(ui->orientation->currentIndex()) ? cur_image->T : tipl::matrix<4,4>(tipl::identity_matrix()),cur_dim,
                        has_flip_x(),has_flip_y(),ui->zoom->value(),ui->axis_grid->currentIndex());
    }
    source << source_image;
    if(update_others)
        update_other_images();
}
void view_image::update_other_images(void)
{
    for(size_t i = 0;i < opened_images.size();++i)
    if(i != this_index && opened_images[i])
        for(size_t j = 0;j < opened_images[i]->overlay_images.size();++j)
            if(opened_images[i]->overlay_images_visible[j] &&
               opened_images[i]->overlay_images[j] == this_index)
                opened_images[i]->show_image(false);
}
void view_image::change_contrast()
{
    v2c.set_range(float(ui->min->value()),float(ui->max->value()));
    v2c.two_color(ui->min_color->color().rgb(),ui->max_color->color().rgb());
    show_image(true);
}

void view_image::on_actionSave_triggered()
{
    if(command("save",file_name.toStdString()))
        QMessageBox::information(this,"DSI Studio","Saved");
    else
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}

void view_image::on_action_Save_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save",file_name,
                            mat.size() ?
                            "FIB/SRC file(*fib.gz *src.gz);;All Files (*)":
                            "NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    file_name = filename;
    setWindowTitle(QFileInfo(file_name).fileName());
    on_actionSave_triggered();
}

void view_image::on_min_slider_sliderMoved(int)
{
    ui->min->setValue(ui->min->minimum()+(ui->min->maximum()-ui->min->minimum())*
                      double(ui->min_slider->value())/double(ui->min_slider->maximum()));
}

void view_image::on_min_valueChanged(double)
{
    ui->min_slider->setValue(int((ui->min->value()-ui->min->minimum())*double(ui->min_slider->maximum())/
                             (ui->min->maximum()-ui->min->minimum())));
    change_contrast();
}

void view_image::on_max_slider_sliderMoved(int)
{
    ui->max->setValue(ui->max->minimum()+(ui->max->maximum()-ui->max->minimum())*
                      double(ui->max_slider->value())/double(ui->max_slider->maximum()));
}

void view_image::on_max_valueChanged(double)
{
    ui->max_slider->setValue(int((ui->max->value()-ui->max->minimum())*double(ui->max_slider->maximum())/
                             (ui->max->maximum()-ui->max->minimum())));
    change_contrast();
}

void view_image::on_AxiView_clicked()
{
    no_update = true;
    cur_dim = 2;
    ui->slice_pos->setRange(0,cur_image->shape.depth()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_CorView_clicked()
{
    no_update = true;
    cur_dim = 1;
    ui->slice_pos->setRange(0,cur_image->shape.height()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_SagView_clicked()
{
    no_update = true;
    cur_dim = 0;
    ui->slice_pos->setRange(0,cur_image->shape.width()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_slice_pos_valueChanged(int value)
{
    if(cur_image->empty())
        return;
    slice_pos[cur_dim] = value;
    show_image(false);
}


void view_image::on_dwi_volume_valueChanged(int value)
{
    read_4d_at(value);
    ui->dwi_label->setText(QString("(%1/%2)").arg(value+1).arg(buf4d.size()));
    show_image(false);
}


void view_image::run_action()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    if(!command(action->text().toLower().replace(' ','_').toStdString()))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}


void view_image::run_action2()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if(!action)
        return;
    QString value;
    if(action->text().toLower().contains(" image..."))
        value = QFileDialog::getOpenFileName(
                               this,"Open other another image to apply",QFileInfo(file_name).absolutePath(),"NIFTI file(*nii.gz *.nii)" );
    else
    {
        bool ok;
        value = QInputDialog::getText(this,"DSI Studio",action->toolTip(),QLineEdit::Normal,action->statusTip(),&ok);
        if(!ok || value.isEmpty())
            return;
    }
    if(value.isEmpty())
        return;
    if(!command(action->text().remove("...").toLower().replace(' ','_').toStdString(),value.toStdString()))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}


void view_image::on_type_currentIndexChanged(int index)
{
    if(cur_image->empty() || no_update)
        return;
    command("change_type",std::to_string(index));
    init_image();
}


void view_image::on_zoom_valueChanged(double arg1)
{
    show_image(false);
}

void view_image::on_info_cellDoubleClicked(int row, int column)
{
    if(!mat.size() || row >= mat.size())
        return;
    if(column == 1 && mat[row].is_type<char>())
    {
        bool okay = false;
        auto text = QInputDialog::getMultiLineText(this,"DSI Studio","Input Content",
                                                   mat[row].get_data<char>(),&okay);
        if(!okay)
            return;
        mat[row].set_text(text.toStdString());
        read_mat_info();
        ui->info->selectRow(row);
    }
    if(column == 0)
    {
        bool okay = false;
        auto text = QInputDialog::getMultiLineText(this,"DSI Studio","Input New Name",ui->info->item(row,0)->text(),&okay);
        if(!okay)
            return;
        command("rename",std::to_string(row)+" "+text.toStdString());
    }
}


void view_image::on_mat_images_currentIndexChanged(int index)
{
    no_update = true;
    if(!cur_image->read_mat_image(ui->mat_images->currentText().toStdString(),mat))
        return;
    ui->type->setCurrentIndex(cur_image->pixel_type);
    init_image();
}


void view_image::on_actionUndo_triggered()
{
    if(mat.size() || !buf4d.empty())
    {
        QMessageBox::critical(this,"ERROR","Undo not supported for 4D images");
        return;
    }
    if(undo_list.empty())
        return;

    // push current to redo data
    redo_list.push_back(cur_image);

    // restore data from undo_list
    cur_image = undo_list.back();
    undo_list.pop_back();

    // handle commands
    redo_command_list.push_back(command_list.back());
    redo_param_list.push_back(param_list.back());
    command_list.pop_back();
    param_list.pop_back();
    init_image();

}


void view_image::on_actionRedo_triggered()
{
    if(redo_list.empty())
        return;
    // push current data to undo list
    undo_list.push_back(cur_image);

    // restore data from redo_list
    cur_image = redo_list.back();
    redo_list.pop_back();

    // handle commands
    command_list.push_back(redo_command_list.back());
    param_list.push_back(redo_param_list.back());
    redo_command_list.pop_back();
    redo_param_list.pop_back();
    init_image();
}

