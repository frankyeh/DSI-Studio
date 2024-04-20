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
std::map<std::string,std::string> dicom_dictionary;
std::vector<view_image*> opened_images;

bool resize_mat(tipl::io::gz_mat_read& mat_reader,const tipl::shape<3>& new_dim);
bool translocate_mat(tipl::io::gz_mat_read& mat_reader,const tipl::vector<3,int>& shift);
bool resample_mat(tipl::io::gz_mat_read& mat_reader,float resolution);

void view_image::change_type(decltype(pixel_type) new_type)
{
    std::vector<unsigned char> buf,empty_buf;
    buf.resize(shape.size()*pixelbit[new_type]);
    apply([&](auto& I)
    {
        switch(new_type)
        {
            case int8:     tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned char*>(&buf[0]));break;
            case int16:    tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned short*>(&buf[0]));break;
            case int32:    tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned int*>(&buf[0]));break;
            case float32:   tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<float*>(&buf[0]));break;
        }
        I.buf().swap(empty_buf);
    });

    pixel_type = new_type;
    apply([&](auto& I)
    {
        I.buf().swap(buf);
        I.resize(shape);
    });
}
void view_image::swap(std::shared_ptr<view_image_record> data)
{
    I_float32.swap(data->I_float32);
    I_int32.swap(data->I_int32);
    I_int16.swap(data->I_int16);
    I_int8.swap(data->I_int8);

    auto temp = pixel_type;
    pixel_type = decltype(pixel_type)(data->pixel_type);
    data->pixel_type = temp;

    shape.swap(data->shape);
    std::swap(is_mni,data->is_mni);
    std::swap(vs,data->vs);
    T.swap(data->T);
}
void view_image::assign(std::shared_ptr<view_image_record> data)
{
    I_float32 = data->I_float32;
    I_int32 = data->I_int32;
    I_int16 = data->I_int16;
    I_int8 = data->I_int8;
    pixel_type = decltype(pixel_type)(data->pixel_type);
    shape = data->shape;
    is_mni = data->is_mni;
    vs = data->vs;
    T = data->T;
}

void view_image::read_4d_at(size_t index)
{
    if(index == cur_4d_index)
        return;
    apply([&](auto& I)
    {
        // give image buffer back
        I.buf().swap(buf4d[cur_4d_index]);
        cur_4d_index = size_t(index);
        if(buf4d[cur_4d_index].empty())
        {
            I.resize(shape);
            if(index && index < nifti.dim(4))
            {
                tipl::show_prog = false;
                nifti.select_volume(cur_4d_index);
                nifti.get_untouched_image(I);
                tipl::show_prog = true;
            }
        }
        else
            I.buf().swap(buf4d[cur_4d_index]);
    });
}
void view_image::get_4d_buf(std::vector<unsigned char>& buf)
{
    buf.resize(shape.size()*buf4d.size()*pixelbit[pixel_type]);
    size_t size_per_image = shape.size()*pixelbit[pixel_type];
    apply([&](auto& I){I.buf().swap(buf4d[cur_4d_index]);});
    for(size_t i = 0;i < buf4d.size();++i)
        std::memcpy(buf.data() + i*size_per_image,buf4d[i].data(),size_per_image);
    apply([&](auto& I){I.buf().swap(buf4d[cur_4d_index]);});
}

bool img_command_int8(tipl::image<3,unsigned char,tipl::buffer_container>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,std::string param1,std::string& error_msg);
bool img_command_int16(tipl::image<3,unsigned short,tipl::buffer_container>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,std::string param1,std::string& error_msg);
bool img_command_int32(tipl::image<3,unsigned int,tipl::buffer_container>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,std::string param1,std::string& error_msg);
bool img_command_float32(tipl::image<3,float,tipl::buffer_container>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,std::string param1,std::string& error_msg);
template<typename image_type>
bool img_command(image_type& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,std::string param1,std::string& error_msg)
{
    if constexpr(std::is_same_v<image_type,tipl::image<3,unsigned char,tipl::buffer_container> >)
        return img_command_int8(data,vs,T,is_mni,cmd,param1,error_msg);
    if constexpr(std::is_same_v<image_type,tipl::image<3,unsigned short,tipl::buffer_container> >)
        return img_command_int16(data,vs,T,is_mni,cmd,param1,error_msg);
    if constexpr(std::is_same_v<image_type,tipl::image<3,unsigned int,tipl::buffer_container> >)
        return img_command_int32(data,vs,T,is_mni,cmd,param1,error_msg);
    if constexpr(std::is_same_v<image_type,tipl::image<3,float,tipl::buffer_container> >)
        return img_command_float32(data,vs,T,is_mni,cmd,param1,error_msg);
    throw;
}



bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param);
bool view_image::command(std::string cmd,std::string param1)
{
    if(!shape.size())
        return true;
    error_msg.clear();
    bool result = true;

    // add undo

    if(!mat.size() && buf4d.empty())
    {
        undo_list.push_back(std::make_shared<view_image_record>());
        swap(undo_list.back());
        assign(undo_list.back());
    }


    {
        tipl::progress prog(cmd.c_str());
        if(!param1.empty())
            tipl::out() << "param: " << param1;

        if(mat.size())
        {
            result = modify_fib(mat,cmd,param1);
            goto end_command;
        }

        // load all buffer for processing
        if(!buf4d.empty())
        {
            auto old_4d_index = cur_4d_index;
            for(size_t i = 0;i < buf4d.size();++i)
                read_4d_at(i);
            read_4d_at(old_4d_index);
        }

        if(cmd =="change_type")
        {
            auto new_type = decltype(pixel_type)(std::stoi(param1));
            if(pixel_type != new_type)
            {
                if(!buf4d.empty())
                {
                    // return image buffer to the 4d buffer
                    apply([&](auto& I){I.buf().swap(buf4d[cur_4d_index]);});

                    // change type for each 4d image
                    auto old_type = pixel_type;
                    for(size_t i = 0;i < buf4d.size() && result;++i)
                    {
                        pixel_type = old_type;
                        apply([&](auto& I)
                        {
                            I.buf().swap(buf4d[i]);
                        });
                        change_type(new_type);
                        apply([&](auto& I)
                        {
                            I.buf().swap(buf4d[i]);
                        });
                    }

                    apply([&](auto& I){I.buf().swap(buf4d[cur_4d_index]);});
                }
                else
                    change_type(new_type);
            }
            goto end_command;
        }

        if(cmd == "reshape")
        {
            std::istringstream in(param1);
            tipl::shape<3> new_shape;
            int dim4 = 1;
            in >> new_shape[0] >> new_shape[1] >> new_shape[2] >> dim4;

            std::vector<unsigned char> buf;
            if(!buf4d.empty())
                get_4d_buf(buf);
            else
                apply([&](auto& I)
                {
                    I.buf().swap(buf);
                });

            shape = new_shape;
            buf.resize(new_shape.size()*dim4*pixelbit[pixel_type]);


            if(dim4 == 1)
                buf4d.clear();
            else
            {
                size_t size_per_image = shape.size()*pixelbit[pixel_type];
                buf4d.resize(dim4);
                for(size_t i = 0;i < dim4;++i)
                {
                    buf4d[i].resize(size_per_image);
                    std::memcpy(buf4d[i].data(),buf.data()+i*size_per_image,size_per_image);
                }
                cur_4d_index = 0;
                buf.swap(buf4d[0]);
            }

            apply([&](auto& I)
            {
                I.buf().swap(buf);
                I.resize(shape);
            });
            goto end_command;
        }
        if(!buf4d.empty() && cmd == "save")
        {
            std::vector<unsigned char> buf;
            get_4d_buf(buf);

            apply([&](auto& I)
            {
                tipl::io::gz_nifti nii;
                nii.set_image_transformation(T,is_mni);
                nii.set_voxel_size(vs);
                nii << tipl::make_image(reinterpret_cast<decltype(&I[0])>(buf.data()),tipl::shape<4>(shape[0],shape[1],shape[2],buf4d.size()));
                result = nii.save_to_file(param1.c_str());
            });
            goto end_command;
        }

        {
            auto old_shape = shape;
            auto old_vs = vs;
            auto old_T = T;
            apply([&](auto& I)
            {
                result = img_command(I,vs,T,is_mni,cmd,param1,error_msg);
                shape = I.shape();
            });
            if(!buf4d.empty())
            {
                auto old_4d_index = cur_4d_index;
                vs = old_vs;
                T = old_T;
                for(size_t i = 0;i < buf4d.size() && result;++i)
                {
                    if(i == old_4d_index)
                        continue;
                    read_4d_at(i);
                    apply([&](auto& I)
                    {
                        result = img_command(I,vs,T,is_mni,cmd,param1,error_msg);
                    });
                }
                read_4d_at(old_4d_index);
            }
        }
    }

    end_command:

    tipl::out() << "result: " << (result ? "succeeded":"failed") << std::endl;
    if(!result)
    {
        tipl::out() << "ERROR: " << error_msg << std::endl;
        if(!undo_list.empty())
        {
            swap(undo_list.back());
            undo_list.pop_back();
        }
        return false;
    }


    if(mat.size())
        read_mat();
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

void save_idx(const char* file_name,std::shared_ptr<tipl::io::gz_istream> in);
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
    if(!shape.is_valid(pos))
        return true;
    auto mni = pos;
    mni.to(T);

    apply([&](auto& data)
    {
        ui->info_label->setText(QString("(i,j,k)=(%1,%2,%3) (x,y,z)=(%4,%5,%6) value=%7").arg(pos[0]).arg(pos[1]).arg(pos[2])
                                                                        .arg(mni[0]).arg(mni[1]).arg(mni[2])
                                                                        .arg(float(data.at(pos))));
    });

    return true;
}


bool get_compressed_image(tipl::io::dicom& dicom,tipl::image<2,short>& I)
{
    QByteArray array((char*)&*dicom.compressed_buf.begin(),dicom.buf_size);
    QBuffer qbuff(&array);
    QImageReader qimg;
    qimg.setDecideFormatFromContent(true);
    qimg.setDevice(&qbuff);
    QImage img;
    if(!qimg.read(&img))
    {
        std::cout << "ERROR: unsupported transfer syntax " << dicom.encoding;
        return false;
    }
    QImage buf = img.convertToFormat(QImage::Format_RGB32);
    I.resize(tipl::shape<2>(buf.width(),buf.height()));
    const uchar* ptr = buf.bits();
    for(int j = 0;j < I.size();++j,ptr += 4)
        I[j] = *ptr;
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
bool view_image::read_mat_image(void)
{
    auto cur_metric = ui->mat_images->currentText().toStdString();
    unsigned int row(0), col(0);
    if(!mat.get_col_row(cur_metric.c_str(),row,col) || row*col != shape.size())
        return false;
    if(mat.type_compatible<float>(cur_metric.c_str()))
    {
        I_float32.resize(shape);
        mat.read(cur_metric.c_str(),I_float32.begin(),I_float32.end());
        pixel_type = float32;
        ui->type->setCurrentIndex(pixel_type);
        return true;
    }
    if(mat.type_compatible<unsigned int>(cur_metric.c_str()))
    {
        I_int32.resize(shape);
        mat.read(cur_metric.c_str(),I_int32.begin(),I_int32.end());
        pixel_type = int32;
        ui->type->setCurrentIndex(pixel_type);
        return true;
    }
    if(mat.type_compatible<unsigned short>(cur_metric.c_str()))
    {
        I_int16.resize(shape);
        mat.read(cur_metric.c_str(),I_int16.begin(),I_int16.end());
        pixel_type = int16;
        ui->type->setCurrentIndex(pixel_type);
        return true;
    }
    if(mat.type_compatible<unsigned char>(cur_metric.c_str()))
    {
        I_int8.resize(shape);
        mat.read(cur_metric.c_str(),I_int8.begin(),I_int8.end());
        pixel_type = int8;
        ui->type->setCurrentIndex(pixel_type);
        return true;
    }
    if(mat.type_compatible<double>(cur_metric.c_str()))
    {
        I_float32.resize(shape);
        mat.read(cur_metric.c_str(),I_float32.begin(),I_float32.end());
        pixel_type = float32;
        ui->type->setCurrentIndex(pixel_type);
        return true;
    }
    return false;
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
    if(!mat.read("dimension",shape))
    {
        error_msg = "cannot find dimension matrix";
        return false;
    }
    bool has_data = true;
    ui->mat_images->clear();
    for(size_t i = 0;i < mat.size();++i)
        if(mat[i].get_cols()*mat[i].get_rows() == shape.size())
            ui->mat_images->addItem(mat[i].get_name().c_str());
    if(!ui->mat_images->count())
    {
        error_msg = "cannot find images";
        return false;
    }
    ui->mat_images->setCurrentIndex(0);

    mat.get_voxel_size(vs);
    if(mat.has("trans"))
        mat.read("trans",T);
    else
        initial_LPS_nifti_srow(T,shape,vs);
    read_mat_info();
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
    tipl::image<3> new_image(shape);
    if(!tipl::io::gz_nifti::load_to_space(filename.toStdString().c_str(),new_image,T))
    {
        QMessageBox::critical(this,"Error","Invalid NIFTI file");
        return;
    }
    if(buf4d.empty())
        buf4d.push_back(std::vector<unsigned char>());
    buf4d.push_back(std::vector<unsigned char>());
    read_4d_at(buf4d.size()-1);
    apply([&](auto& I)
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


    file_names = file_names_;
    original_file_name = file_name = file_names[0];
    file_names.removeFirst();

    tipl::io::dicom dicom;
    is_mni = false;
    T.identity();
    QString info;

    setWindowTitle(QFileInfo(file_name).fileName());
    tipl::progress prog("open image file ",std::filesystem::path(file_name.toStdString()).filename().string().c_str());
    if(file_names_.size() > 1 &&
       (QString(file_name).endsWith(".bmp") ||
        QString(file_name).endsWith(".png") ||
        QString(file_name).endsWith(".tif") ||
        QString(file_name).endsWith(".tiff")))
    {
        QImage in = read_qimage(file_name,error_msg);
        if(in.isNull())
            return false;
        pixel_type = int8;
        shape[0] = in.width();
        shape[1] = in.height();
        shape[2] = uint32_t(file_names.size());
        T.identity();
        vs[0] = vs[1] = vs[2] = 1.0f;
        I_int8.resize(shape);
        buf4d.resize(3);
        for(size_t i = 1;i < 3;++i)
            buf4d[i].resize(shape.size());

        for(size_t file_index = 0;prog(file_index,shape[2]);++file_index)
        {
            QImage I = read_qimage(file_names[file_index],error_msg);
            if(I.isNull())
                return false;

            if(I.width() != shape[0] || I.height() != shape[1])
            {
                error_msg = "inconsistent image size : ";
                error_msg += file_names[file_index].toStdString();
                return false;
            }
            I = I.convertToFormat(QImage::Format_RGB32);
            auto ptr = reinterpret_cast<uint32_t*>(I.bits());
            for(size_t i = 0,pos = file_index*shape.plane_size();i < shape.plane_size();++i,++pos)
            {
                tipl::rgb rgb(ptr[i]);
                I_int8[pos] = rgb.r;
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
    if(QString(file_name).endsWith(".nhdr"))
    {
        tipl::io::nrrd<tipl::progress> nrrd;
        if(!nrrd.load_from_file(file_name.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",nrrd.error_msg.c_str());
            return false;
        }

        shape = nrrd.size;
        pixel_type = float32;
        if(nrrd.values["type"] == "int" || nrrd.values["type"] == "unsigned int")
            pixel_type = int32;
        if(nrrd.values["type"] == "short" || nrrd.values["type"] == "unsigned short")
            pixel_type = int16;
        if(nrrd.values["type"] == "uchar")
            pixel_type = int8;

        apply([&](auto& data)
        {
            nrrd >> data;
        });

        if(!nrrd.error_msg.empty())
        {
            QMessageBox::critical(this,"ERROR",nrrd.error_msg.c_str());
            return false;
        }
        nrrd.get_voxel_size(vs);
        nrrd.get_image_transformation(T);

        info.clear();
        for(const auto& iter : nrrd.values)
            info += QString("%1=%2\n").arg(iter.first.c_str()).arg(iter.second.c_str());

    }
    else
    if(QString(file_name).endsWith(".nii.gz") || QString(file_name).endsWith(".nii"))
    {
        prepare_idx(file_name.toStdString().c_str(),nifti.input_stream);
        if(!nifti.load_from_file(file_name.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",nifti.error_msg.c_str());
            return false;
        }
        if(nifti.dim(4) > 1)
        {
            buf4d.resize(nifti.dim(4));
            nifti.input_stream->sample_access_point = true;
        }
        nifti.get_image_dimension(shape);
        switch (nifti.nif_header.datatype)
        {
        case 2://DT_UNSIGNED_CHAR 2
        case 256: // DT_INT8
            pixel_type = int8;
            break;
        case 4://DT_SIGNED_SHORT 4
        case 512: // DT_UINT16
            pixel_type = int16;
            break;
        case 8://DT_SIGNED_INT 8
        case 768: // DT_UINT32
        case 1024: // DT_INT64
        case 1280: // DT_UINT64
            pixel_type = int32;
            break;
        case 16://DT_FLOAT 16
        case 64://DT_DOUBLE 64
            pixel_type = float32;
            break;
        default:
            QMessageBox::critical(this,"ERROR","Unsupported pixel format");
            return false;
        }
        if(std::floor(nifti.nif_header.scl_inter) != nifti.nif_header.scl_inter || nifti.nif_header.scl_inter < 0.0f ||
           std::floor(nifti.nif_header.scl_slope) != nifti.nif_header.scl_slope)
            pixel_type = float32;

        bool succeed = true;
        apply([&](auto& data)
        {
            succeed = nifti.get_untouched_image(data,prog);
            if constexpr(!std::is_integral<typename std::remove_reference<decltype(*data.begin())>::type>::value)
            {
                tipl::par_for(data.size(),[&](size_t pos)
                {
                   if(std::isnan(data[pos]))
                       data[pos] = 0;
                });
            }
        });
        if(!succeed)
        {
            QMessageBox::critical(this,"ERROR",nifti.error_msg.c_str());
            return false;
        }
        if(nifti.dim(4) == 1)
            save_idx(file_name.toStdString().c_str(),nifti.input_stream);
        nifti.get_voxel_size(vs);
        nifti.get_image_transformation(T);
        is_mni = nifti.is_mni();
        std::ostringstream out;
        out << nifti;
        info = out.str().c_str();
    }
    else
        if(dicom.load_from_file(file_name.toStdString()))
        {
            pixel_type = int16;
            dicom.get_image_dimension(shape);
            apply([&](auto& data){dicom >> data;});

            if(dicom.is_compressed)
            {
                tipl::image<2,short> I;
                if(!get_compressed_image(dicom,I))
                {
                    QMessageBox::critical(this,"ERROR",QString("Unsupported transfer syntax ") + QString(dicom.encoding.c_str()));
                    return false;
                }
                if(I.size() == shape.size())
                    std::copy(I.begin(),I.end(),I_int16.begin());
            }
            dicom.get_voxel_size(vs);
            std::string info_;
            dicom >> info_;

            if(dicom_dictionary.empty())
            {
                QFile data(":/data/dicom_tag.txt");
                if (data.open(QIODevice::ReadOnly | QIODevice::Text))
                {
                    QTextStream in(&data);
                    while (!in.atEnd())
                    {
                        QStringList list = in.readLine().split('\t');
                        if(list.size() < 3)
                            continue;
                        std::string value = list[2].toStdString();
                        std::replace(value.begin(),value.end(),' ','_');
                        dicom_dictionary[list[0].toStdString()] = value;
                    }
                }
            }
            std::ostringstream out;
            std::istringstream in(info_);
            std::string line;
            while(std::getline(in,line))
            {

                for(size_t pos = 0;(pos = line.find('(',pos)) != std::string::npos;++pos)
                {
                    std::string tag = line.substr(pos,11);
                    if(tag.length() != 11)
                        continue;
                    std::string tag2 = tag;
                    tag2[3] = 'x';
                    tag2[4] = 'x';
                    auto iter = dicom_dictionary.find(tag);
                    if(iter == dicom_dictionary.end())
                        iter = dicom_dictionary.find(tag2);
                    if(iter != dicom_dictionary.end())
                        line.replace(pos,11,tag+iter->second);
                }
                out << line << std::endl;
            }
            info_ = out.str();
            info = info_.c_str();
        }
        else
            if(QString(file_name).endsWith(".mat") || QString(file_name).endsWith("fib.gz") || QString(file_name).endsWith("src.gz"))
            {
                if(!mat.load_from_file(file_name.toStdString().c_str()))
                {
                    error_msg = "invalid format";
                    return false;
                }
                pixel_type = float32;
                if(!read_mat())
                    return false;

            }
            else
            if(QString(file_name).endsWith("2dseq"))
            {
                tipl::io::bruker_2dseq seq;
                if(!seq.load_from_file(file_name.toStdString().c_str()))
                {
                    error_msg = "cannot parse file";
                    return false;
                }
                pixel_type = float32;
                shape = seq.get_image().shape();
                I_float32 = seq.get_image();
                seq.get_voxel_size(vs);
            }
            else
            {
                error_msg = "unsupported file format";
                return false;
            }
    if(!info.isEmpty())
        show_info(info);
    no_update = true;
    ui->type->setCurrentIndex(pixel_type);
    ui->zoom->setValue(0.9f*width()/shape.width());
    if(shape.size())
        init_image();  
    return shape.size() || !info.isEmpty();
}

void view_image::init_image(void)
{
    no_update = true;
    float min_value = 0.0f;
    float max_value = 0.0f;
    apply([&](auto& data)
    {
        auto minmax = tipl::minmax_value_mt(data);
        min_value = minmax.first;
        max_value = minmax.second;
    });
    float range = max_value-min_value;
    QString dim_text = QString("%1,%2,%3").arg(shape.width()).arg(shape.height()).arg(shape.depth());
    if(!buf4d.empty())
        dim_text += QString(",%1").arg(buf4d.size());
    ui->image_info->setText(QString("dim=(%1) vs=(%4,%5,%6) srow=[%7 %8 %9 %10][%11 %12 %13 %14][%15 %16 %17 %18] %19").
            arg(dim_text).
            arg(double(vs[0])).arg(double(vs[1])).arg(double(vs[2])).
            arg(double(T[0])).arg(double(T[1])).arg(double(T[2])).arg(double(T[3])).
            arg(double(T[4])).arg(double(T[5])).arg(double(T[6])).arg(double(T[7])).
            arg(double(T[8])).arg(double(T[9])).arg(double(T[10])).arg(double(T[11])).arg(is_mni?"mni":"native"));

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
    if(ui->slice_pos->maximum() != int(shape[cur_dim]-1))
    {
        ui->slice_pos->setRange(0,shape[cur_dim]-1);
        slice_pos[0] = shape.width()/2;
        slice_pos[1] = shape.height()/2;
        slice_pos[2] = shape.depth()/2;
        ui->slice_pos->setValue(slice_pos[cur_dim]);
    }

    ui->actionSet_MNI->setStatusTip(is_mni ? "1":"0");
    ui->actionRegrid->setStatusTip(QString("%1 %2 %3").arg(vs[0]).arg(vs[1]).arg(vs[2]));
    ui->actionResize->setStatusTip(QString("%1 %2 %3").arg(shape[0]).arg(shape[1]).arg(shape[2]));
    ui->actionReshape->setStatusTip(buf4d.empty() ? QString("%1 %2 %3").arg(shape[0]).arg(shape[1]).arg(shape[2]) :
                                                    QString("%1 %2 %3 %4").arg(shape[0]).arg(shape[1]).arg(shape[2]).arg(buf4d.size()));
    ui->actionSet_Translocation->setStatusTip(QString("%1 %2 %3").arg(T[3]).arg(T[7]).arg(T[11]));

    std::string t_string;
    {
        std::ostringstream out;
        for(int i = 0;i < 16;++i)
            out << T[i] << " ";
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
               opened_images[i]->shape == shape)
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
        if(T[0] > 0)
        {
            if(cur_dim == 2)
                flip_x = !flip_x;
            if(cur_dim == 1)
                flip_x = !flip_x;
        }
        if(T[5] > 0)
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
        if(T[5] > 0)
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
    if(!shape.size() || no_update)
        return;

    tipl::color_image buffer;
    apply([&](auto& data)
    {
        v2c.convert(tipl::volume2slice_scaled(data,cur_dim, size_t(slice_pos[cur_dim]),ui->zoom->value()),buffer);
    });
    if(ui->overlay_style->currentIndex() == 1)
        std::fill(buffer.begin(),buffer.end(),tipl::rgb(0,0,0));
    // draw overlay
    for(size_t i = 0;i < overlay_images.size() && ui->overlay_style->currentIndex() <= 1;++i)
    if(overlay_images_visible[i] && opened_images[overlay_images[i]])
        opened_images[overlay_images[i]]->apply([&](auto& data)
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

        tipl::qt::draw_ruler(paint,shape,(ui->orientation->currentIndex()) ? T : tipl::matrix<4,4>(tipl::identity_matrix()),cur_dim,
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
    ui->slice_pos->setRange(0,shape.depth()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_CorView_clicked()
{
    no_update = true;
    cur_dim = 1;
    ui->slice_pos->setRange(0,shape.height()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_SagView_clicked()
{
    no_update = true;
    cur_dim = 0;
    ui->slice_pos->setRange(0,shape.width()-1);
    no_update = false;
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_slice_pos_valueChanged(int value)
{
    if(!shape.size())
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
    if(!shape.size() || no_update)
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
    if(read_mat_image())
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

    // prepare redo data
    {
        auto redo_data = std::make_shared<view_image_record>();
        swap(redo_data);
        redo_list.push_back(redo_data);
    }

    // restore data from undo_list
    {
        swap(undo_list.back());
        undo_list.pop_back();
    }

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
    // set current data to undo list
    {
        auto undo_data = std::make_shared<view_image_record>();
        swap(undo_data);
        undo_list.push_back(undo_data);
    }

    // restore data from redo_list
    {
        swap(redo_list.back());
        redo_list.pop_back();
    }

    // handle commands
    command_list.push_back(redo_command_list.back());
    param_list.push_back(redo_param_list.back());
    redo_command_list.pop_back();
    redo_param_list.pop_back();
    init_image();
}

