#include <map>
#include <QTextStream>
#include <QInputDialog>
#include <QFileDialog>
#include "view_image.h"
#include "ui_view_image.h"
#include "prog_interface_static_link.h"
#include <QPlainTextEdit>
#include <QFileInfo>
#include <QMessageBox>
#include <QBuffer>
#include <QImageReader>
#include <filesystem>
std::map<std::string,std::string> dicom_dictionary;
std::vector<view_image*> opened_images;


std::string common_prefix(const std::string& str1,const std::string& str2)
{
    std::string result;
    for(size_t cur = 0;cur < str1.length() && cur < str2.length();++cur)
    {
        if(str1[cur] != str2[cur])
            break;
        result.push_back(str1[cur]);
    }
    return result;
}
std::string common_prefix(const std::string& str1,const std::string& str2,const std::string& str3)
{
    std::string result;
    for(size_t cur = 0;cur < str1.length() && cur < str2.length() && cur < str3.length();++cur)
    {
        if(str1[cur] != str2[cur] || str1[cur] != str3[cur])
            break;
        result.push_back(str1[cur]);
    }
    return result;
}



bool match_strings(const std::string& str1,const std::string& str1_match,
                   const std::string& str2,std::string& str2_match,bool try_reverse = true,bool try_swap = true)
{

    // A->A
    // B->B
    if(str1 == str1_match)
    {
        str2_match = str2;
        return true;
    }
    // A->B
    // A->B
    if(str1 == str2)
    {
        str2_match = str1_match;
        return true;
    }

    // remove common prefix
    {
        auto cprefix = common_prefix(str1,str1_match,str2);
        if(!cprefix.empty())
        {
            if(!match_strings(str1.substr(cprefix.length()),str1_match.substr(cprefix.length()),
                              str2.substr(cprefix.length()),str2_match))
                return false;
            str2_match = cprefix+str2_match;
            return true;
        }
    }

    // remove common postfix
    {
        auto cpostfix = common_prefix(std::string(str1.rbegin(),str1.rend()),
                                      std::string(str1_match.rbegin(),str1_match.rend()),
                                      std::string(str2.rbegin(),str2.rend()));
        if(!cpostfix.empty())
        {
            if(!match_strings(str1.substr(0,str1.length()-cpostfix.length()),
                              str1_match.substr(0,str1_match.length()-cpostfix.length()),
                              str2.substr(0,str2.length()-cpostfix.length()),str2_match))
                return false;
            str2_match += std::string(cpostfix.rbegin(),cpostfix.rend());
            return true;
        }
    }


    auto cp1_1 = common_prefix(str1,str1_match);
    auto cp1_2 = common_prefix(str1,str2);


    // A_B->A_C
    // D_B->D_C
    if(cp1_1.length() > cp1_2.length())
    {        
        // A->A_C
        // D->D_C
        if(cp1_1 == str1)
        {
            str2_match = str2 + str1_match.substr(cp1_1.size());
            return true;
        }

        if(str1.length() == str2.length())
        {
            if(match_strings(str1.substr(cp1_1.size()),str1_match.substr(cp1_1.size()),
                             str2.substr(cp1_1.size()),str2_match))
            {
                str2_match = str2.substr(0,cp1_1.size()) + str2_match;
                return true;
            }
        }

        if(match_strings(str1.substr(cp1_1.size()),str1_match.substr(cp1_1.size()),
                         str2,str2_match))
            return true;
    }
    // try reversed
    std::string rev;
    if(try_reverse && match_strings(std::string(str1.rbegin(),str1.rend()),
                     std::string(str1_match.rbegin(),str1_match.rend()),
                     std::string(str2.rbegin(),str2.rend()),rev,false,try_swap))
    {
        str2_match = std::string(rev.rbegin(),rev.rend());
        return true;
    }
    // try swap
    if(try_swap)
        return match_strings(str1,str2,str1_match,str2_match,try_reverse,false);
    return false;
}

bool match_files(const std::string& file_path1,const std::string& file_path2,
                 const std::string& file_path1_others,std::string& file_path2_gen)
{
    auto name1 = std::filesystem::path(file_path1).filename().string();
    auto name2 = std::filesystem::path(file_path2).filename().string();
    auto name1_others = std::filesystem::path(file_path1_others).filename().string();
    auto path1 = std::filesystem::path(file_path1).parent_path().string();
    auto path2 = std::filesystem::path(file_path2).parent_path().string();
    auto path1_others = std::filesystem::path(file_path1_others).parent_path().string();
    std::string name2_others,path2_others;
    if(!match_strings(name1,name2,name1_others,name2_others) ||
       !match_strings(path1,path2,path1_others,path2_others))
        return false;
    if(!path2_others.empty())
        path2_others += "/";
    file_path2_gen = path2_others + name2_others;
    return true;
}



bool resample_mat(gz_mat_read& mat_reader,float resolution);
bool view_image::command(std::string cmd,std::string param1)
{
    if(!shape.size())
        return true;
    error_msg.clear();
    bool result = true;

    if(mat.size())
    {
        if(cmd == "regrid")
        {
            float reso = std::stof(param1);
            if(reso <= 0.0f)
                return false;
            resample_mat(mat,reso);
            read_mat();
            init_image();
            return true;
        }
        if(cmd == "save")
        {
            gz_mat_write matfile(file_name.toStdString().c_str());
            if(!matfile)
            {
                QMessageBox::critical(this,"ERROR","Cannot save file");
                return false;
            }
            progress p("saving");
            for(unsigned int index = 0;progress::at(index,mat.size());++index)
                matfile.write(mat[index]);
            QMessageBox::information(this,"DSI Studio","File Save");
            return true;
        }
    }


    {
        progress prog(cmd.c_str());
        if(!param1.empty())
            progress::show((std::string("param: ")+param1).c_str());

        if(cmd =="change_type")
        {
            auto new_type = decltype(data_type)(std::stoi(param1));
            if(data_type != new_type)
            {
                std::vector<unsigned char> buf,empty_buf;
                size_t pixelbit[4] = {1,2,4,4};
                buf.resize(shape.size()*pixelbit[new_type]);
                apply([&](auto& I)
                {
                    switch(new_type)
                    {
                        case uint8:     tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned char*>(&buf[0]));break;
                        case uint16:    tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned short*>(&buf[0]));break;
                        case uint32:    tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<unsigned int*>(&buf[0]));break;
                        case float32:   tipl::copy_mt(I.begin(),I.end(),reinterpret_cast<float*>(&buf[0]));break;
                    }
                    I.buf().swap(empty_buf);
                });

                data_type = new_type;
                apply([&](auto& I)
                {
                    I.buf().swap(buf);
                    I.resize(shape);
                });
                if(!dwi_volume_buf.empty())
                    dwi_volume_buf = std::move(std::vector<std::vector<unsigned char> >(dwi_volume_buf.size()));
            }
        }
        else
        if(cmd == "set_transformation")
        {
            std::istringstream in(param1);
            for(int i = 0;i < 16;++i)
                in >> T[i];
        }
        else
        if(cmd == "set_shift")
        {
            std::istringstream in(param1);
            in >> T[3] >> T[7] >> T[11];
        }
        else
        apply([&](auto& I)
        {
            result = tipl::command<gz_nifti>(I,vs,T,is_mni,cmd,param1,error_msg);
            shape = I.shape();
        });
        show_progress() << "result: " << (result ? "succeeded":"failed") << std::endl;
        if(!result)
        {
            show_progress() << "ERROR:" << error_msg << std::endl;
            return false;
        }

        if(mat.size())
            write_mat_image();
    }

    init_image();

    command_list.push_back(cmd);
    param_list.push_back(param1);


    if(cmd == "save" && !file_names.empty())
    {
        if(QMessageBox::question(nullptr,"DSI Studio","Applying processing to other images and save them?",
                                 QMessageBox::No | QMessageBox::Yes,QMessageBox::Yes) == QMessageBox::No)
        {
            file_names.clear();
            return true;
        }

        progress p1("apply to other images");
        int file_index = 0;
        for(;progress::at(file_index,file_names.size());++file_index)
        {
            auto file_name2 = file_names[file_index];
            progress p2("processing ",file_name2.toStdString().c_str());

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
                   command_list[i] == "multiply_image" ||
                   command_list[i] == "add_image" ||
                   command_list[i] == "minus_image")
                {
                    if(match_files(original_file_name.toStdString(),param_list[i],
                                file_name2.toStdString(),param2))
                    {
                        show_progress() << "matched path: " << std::filesystem::path(param2).parent_path().string() << std::endl;
                        show_progress() << "matched file name: " << std::filesystem::path(param2).filename().string() << std::endl;
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


void show_view(QGraphicsScene& scene,QImage I);
bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>& trans)
{
    if(filenames.size() == 1 && filenames[0].toLower().contains("nii"))
    {
        gz_nifti in;
        if(!in.load_from_file(filenames[0].toLocal8Bit().begin()) || !in.toLPS(ref))
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
            if(!seq.load_from_file(filenames[0].toLocal8Bit().begin()))
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
void view_image::DeleteRowPressed(int row)
{
    if(ui->info->currentRow() == -1)
        return;
    if(ui->info->currentRow() < mat.size())
    {
        auto index = ui->mat_images->findText(mat.name(ui->info->currentRow()));
        if(index != -1)
            ui->mat_images->removeItem(index);
        mat.remove(ui->info->currentRow());
    }
    ui->info->removeRow(ui->info->currentRow());
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
    ui->dwi_volume->hide();
    ui->dwi_label->hide();
    connect(ui->max_color,SIGNAL(clicked()),this,SLOT(change_contrast()));
    connect(ui->min_color,SIGNAL(clicked()),this,SLOT(change_contrast()));
    connect(ui->orientation,SIGNAL(currentIndexChanged(int)),this,SLOT(change_contrast()));
    connect(ui->axis_grid,SIGNAL(currentIndexChanged(int)),this,SLOT(change_contrast()));
    connect(ui->menuOverlay, SIGNAL(aboutToShow()),this, SLOT(update_overlay_menu()));

    connect(ui->actionMorphology_Smoothing, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_Defragment, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_Dilation, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_Erosion, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_Edge, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_XY, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMorphology_XZ, SIGNAL(triggered()),this, SLOT(run_action()));


    connect(ui->actionFlip_X, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionFlip_Y, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionFlip_Z, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionSwap_XY, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionSwap_XZ, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionSwap_YZ, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionDownsample_by_2, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionUpsample_by_2, SIGNAL(triggered()),this, SLOT(run_action()));

    connect(ui->actionNormalize_Intensity, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionMean_Filter, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionGaussian_Filter, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionSmoothing_Filter, SIGNAL(triggered()),this, SLOT(run_action()));
    connect(ui->actionSobel_Filter, SIGNAL(triggered()),this, SLOT(run_action()));

    // ask for a value and run action
    connect(ui->actionIntensity_shift, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionIntensity_scale, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionLower_threshold, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionUpper_Threshold, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionRegrid, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionThreshold, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionTranslocate, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionResize, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionMultiplyImage, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionAddImage, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionMinusImage, SIGNAL(triggered()),this, SLOT(run_action2()));
    connect(ui->actionCropToFit, SIGNAL(triggered()),this, SLOT(run_action2()));


    ui->tabWidget->setCurrentIndex(0);


    qApp->installEventFilter(this);
    this_index = opened_images.size();
    opened_images.push_back(this);
}

void save_idx(const char* file_name,std::shared_ptr<gz_istream> in);
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
    tipl::vector<3,float> pos,mni;
    auto x = point.x();
    auto y = point.y();
    if(has_flip_x())
        x = source.width() - x;
    if(has_flip_y())
        y = source.height() - y;

    tipl::slice2space(cur_dim,
                      std::round(float(x) / ui->zoom->value()),
                      std::round(float(y) / ui->zoom->value()),ui->slice_pos->value(),pos[0],pos[1],pos[2]);
    if(!shape.is_valid(pos))
        return true;
    mni = pos;
    mni.to(T);

    apply([&](auto& data)
    {
        ui->info_label->setText(QString("(i,j,k)=(%1,%2,%3) (x,y,z)=(%4,%5,%6) value=%7").arg(pos[0]).arg(pos[1]).arg(pos[2])
                                                                        .arg(mni[0]).arg(mni[1]).arg(mni[2])
                                                                        .arg(data.at(pos)));
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
        std::cout << "Unsupported transfer syntax:" << dicom.encoding;
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
        data_type = float32;
        ui->type->setCurrentIndex(data_type);
        return true;
    }
    if(mat.type_compatible<unsigned int>(cur_metric.c_str()))
    {
        I_uint32.resize(shape);
        mat.read(cur_metric.c_str(),I_uint32.begin(),I_uint32.end());
        data_type = uint32;
        ui->type->setCurrentIndex(data_type);
        return true;
    }
    if(mat.type_compatible<unsigned short>(cur_metric.c_str()))
    {
        I_uint16.resize(shape);
        mat.read(cur_metric.c_str(),I_uint16.begin(),I_uint16.end());
        data_type = uint16;
        ui->type->setCurrentIndex(data_type);
        return true;
    }
    if(mat.type_compatible<unsigned char>(cur_metric.c_str()))
    {
        I_uint8.resize(shape);
        mat.read(cur_metric.c_str(),I_uint8.begin(),I_uint8.end());
        data_type = uint8;
        ui->type->setCurrentIndex(data_type);
        return true;
    }
    if(mat.type_compatible<double>(cur_metric.c_str()))
    {
        I_float32.resize(shape);
        mat.read(cur_metric.c_str(),I_float32.begin(),I_float32.end());
        data_type = float32;
        ui->type->setCurrentIndex(data_type);
        return true;
    }
    return false;
}
void view_image::write_mat_image(void)
{
    auto cur_metric = ui->mat_images->currentText().toStdString();
    unsigned int row(0), col(0);
    if(!mat.get_col_row(cur_metric.c_str(),row,col) || row*col != shape.size())
        return;
    switch(data_type)
    {
        case float32:
            std::copy(I_float32.begin(),I_float32.end(),const_cast<float*>(mat.read_as_type<float>(cur_metric.c_str(),row,col)));
            break;
        case uint32:
            std::copy(I_uint32.begin(),I_uint32.end(),const_cast<unsigned int*>(mat.read_as_type<unsigned int>(cur_metric.c_str(),row,col)));
            break;
        case uint16:
            std::copy(I_uint16.begin(),I_uint16.end(),const_cast<unsigned short*>(mat.read_as_type<unsigned short>(cur_metric.c_str(),row,col)));
            break;
        case uint8:
            std::copy(I_uint8.begin(),I_uint8.end(),const_cast<unsigned char*>(mat.read_as_type<unsigned char>(cur_metric.c_str(),row,col)));
            break;
    }
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
void prepare_idx(const char* file_name,std::shared_ptr<gz_istream> in);
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
    progress prog("open image file ",std::filesystem::path(file_name.toStdString()).filename().string().c_str());
    if(file_names_.size() > 1 && QString(file_name).endsWith(".bmp"))
    {
        for(unsigned int i = 0;progress::at(i,file_names_.size());++i)
        {
            tipl::color_image I;
            tipl::io::bitmap bmp;
            if(!bmp.load_from_file(file_names_[i].toStdString().c_str()))
                return false;
            bmp >> I;
            data_type = uint8;
            if(i == 0)
            {
                shape = tipl::shape<3>(I.width(),I.height(),file_names_.size());
                I_uint8.resize(shape);
            }
            unsigned int pos = i*I.size();
            for(unsigned int j = 0;j < I.size();++j)
                I_uint8[pos+j] = ((float)I[j].r+(float)I[j].r+(float)I[j].r)/3.0;
        }
        file_names.clear();
    }
    if(QString(file_name).endsWith(".nhdr"))
    {
        tipl::io::nrrd<progress> nrrd;
        if(!nrrd.load_from_file(file_name.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",nrrd.error_msg.c_str());
            return false;
        }

        shape = nrrd.size;
        data_type = float32;
        if(nrrd.values["type"] == "int" || nrrd.values["type"] == "unsigned int")
            data_type = uint32;
        if(nrrd.values["type"] == "short" || nrrd.values["type"] == "unsigned short")
            data_type = uint16;
        if(nrrd.values["type"] == "uchar")
            data_type = uint8;

        apply([&](auto& data)
        {
            nrrd >> data;
        });

        if(progress::aborted())
            return false;

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
            ui->dwi_volume->setMaximum(nifti.dim(4)-1);
            dwi_volume_buf.resize(nifti.dim(4));
            nifti.input_stream->sample_access_point = true;
            ui->dwi_volume->show();
            ui->dwi_label->show();
        }
        nifti.get_image_dimension(shape);
        switch (nifti.nif_header.datatype)
        {
        case 2://DT_UNSIGNED_CHAR 2
        case 256: // DT_INT8
            data_type = uint8;
            break;
        case 4://DT_SIGNED_SHORT 4
        case 512: // DT_UINT16
            data_type = uint16;
            break;
        case 8://DT_SIGNED_INT 8
        case 768: // DT_UINT32
        case 1024: // DT_INT64
        case 1280: // DT_UINT64
            data_type = uint32;
            break;
        case 16://DT_FLOAT 16
        case 64://DT_DOUBLE 64
            data_type = float32;
            break;
        default:
            QMessageBox::critical(this,"ERROR","Unsupported pixel format");
            return false;
        }
        if(std::floor(nifti.nif_header.scl_inter) != nifti.nif_header.scl_inter ||
           std::floor(nifti.nif_header.scl_slope) != nifti.nif_header.scl_slope)
            data_type = float32;

        bool succeed = true;
        apply([&](auto& data)
        {
            succeed = nifti.get_untouched_image(data);
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
            data_type = uint16;
            dicom.get_image_dimension(shape);
            apply([&](auto& data){dicom >> data;});

            if(dicom.is_compressed)
            {
                tipl::image<2,short> I;
                if(!get_compressed_image(dicom,I))
                {
                    QMessageBox::critical(this,"ERROR",QString("Unsupported transfer syntax:") + QString(dicom.encoding.c_str()));
                    return false;
                }
                if(I.size() == shape.size())
                    std::copy(I.begin(),I.end(),I_uint16.begin());
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
                data_type = float32;
                if(!read_mat())
                    return false;

            }
            else
            if(QString(file_name).endsWith("2dseq"))
            {
                tipl::io::bruker_2dseq seq;
                if(seq.load_from_file(file_name.toLocal8Bit().begin()))
                {
                    error_msg = "cannot parse file";
                    return false;
                }
                data_type = float32;
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
    ui->type->setCurrentIndex(data_type);
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
    if(!dwi_volume_buf.empty())
        dim_text += QString(",%1").arg(dwi_volume_buf.size());
    ui->image_info->setText(QString("dim=(%1) vs=(%4,%5,%6) srow=[%7 %8 %9 %10][%11 %12 %13 %14][%15 %16 %17 %18]").
            arg(dim_text).
            arg(double(vs[0])).arg(double(vs[1])).arg(double(vs[2])).
            arg(double(T[0])).arg(double(T[1])).arg(double(T[2])).arg(double(T[3])).
            arg(double(T[4])).arg(double(T[5])).arg(double(T[6])).arg(double(T[7])).
            arg(double(T[8])).arg(double(T[9])).arg(double(T[10])).arg(double(T[11])));

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
        ui->actionResize->setStatusTip(QString("%1 %2 %3").arg(shape.width()).arg(shape.height()).arg(shape.depth()));
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
void draw_ruler(QPainter& paint,
                const tipl::shape<3>& shape,
                const tipl::matrix<4,4>& trans,
                unsigned char cur_dim,
                bool flip_x,bool flip_y,
                float zoom,
                bool grid = false);
void view_image::show_image(bool update_others)
{
    if(!shape.size() || no_update)
        return;

    tipl::color_image buffer;
    apply([&](auto& data)
    {
        tipl::image<2,float> buf;
        tipl::volume2slice_scaled(data, buf, cur_dim, size_t(slice_pos[cur_dim]),ui->zoom->value());
        v2c.convert(buf,buffer);
    });

    // draw overlay
    for(size_t i = 0;i < overlay_images.size();++i)
    if(overlay_images_visible[i] && opened_images[overlay_images[i]])
        opened_images[overlay_images[i]]->apply([&](auto& data)
        {
            tipl::color_image buffer2;
            tipl::image<2,float> buf2;
            tipl::volume2slice_scaled(data, buf2, cur_dim, size_t(slice_pos[cur_dim]),ui->zoom->value());
            opened_images[overlay_images[i]]->v2c.convert(buf2,buffer2);
            for(size_t j = 0;j < buffer.size();++j)
                buffer[j] |= buffer2[j];
        });
    source_image = QImage(reinterpret_cast<unsigned char*>(&*buffer.begin()),buffer.width(),buffer.height(),QImage::Format_RGB32).copy().mirrored(has_flip_x(),has_flip_y());

    {
        QPainter paint(&source_image);

        QPen pen;
        pen.setColor(Qt::white);
        paint.setPen(pen);
        paint.setFont(font());

        draw_ruler(paint,shape,(ui->orientation->currentIndex()) ? T : tipl::matrix<4,4>(tipl::identity_matrix()),cur_dim,
                        has_flip_x(),has_flip_y(),ui->zoom->value(),ui->axis_grid->currentIndex());
    }
    show_view(source,source_image);
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

void view_image::on_actionSet_Translocation_triggered()
{
    std::ostringstream out;
    out << T[3] << " " << T[7] << " " << T[11];
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign the translocation vector translocation (x y z)",QLineEdit::Normal,
                                           out.str().c_str(),&ok);

    if(!ok)
        return;
    command("set_shift",result.toStdString());
    init_image();
}

void view_image::on_actionSet_Transformation_triggered()
{
    std::ostringstream out;
    for(int i = 0;i < 16;++i)
        out << T[i] << " ";
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign the transformation matrix",QLineEdit::Normal,
                                           out.str().c_str(),&ok);

    if(!ok)
        return;
    command("set_transformation",result.toStdString());
    init_image();
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
    apply([&](auto& I)
    {
        I.buf().swap(dwi_volume_buf[cur_dwi_volume]);
        cur_dwi_volume = size_t(value);
        if(dwi_volume_buf[cur_dwi_volume].empty())
        {
            has_gui = false;
            nifti.select_volume(cur_dwi_volume);
            nifti.get_untouched_image(I);
            has_gui = true;
        }
        else
            I.buf().swap(dwi_volume_buf[cur_dwi_volume]);
    });

    ui->dwi_label->setText(QString("(%1/%2)").arg(value+1).arg(ui->dwi_volume->maximum()+1));
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
    if(action->statusTip() == "file")
    {
        value = QFileDialog::getOpenFileName(
                               this,"Open other another image to apply",QFileInfo(file_name).absolutePath(),"NIFTI file(*nii.gz *.nii)" );
    }
    else
    {
        bool ok;
        value = QInputDialog::getText(this,"DSI Studio",action->toolTip(),QLineEdit::Normal,action->statusTip(),&ok);
        if(!ok || value.isEmpty())
            return;
    }
    if(value.isEmpty())
        return;
    if(action->text().contains("Morphology") && data_type == float32)
    {
        auto result = QMessageBox::information(this,"DSI Studio",
                "Morphology operation only applies to label image. Convert pixel type to integer?",
                                    QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
        if(result == QMessageBox::Cancel)
            return;
        if(result == QMessageBox::Yes)
        {
            if(ui->min->maximum() <= 255.0)
                ui->type->setCurrentIndex(uint8);
            else
                ui->type->setCurrentIndex(uint16);
        }
    }
    if(value.contains(".") && data_type != float32 &&
            action->statusTip() != "file" &&
            action->text() != "Regrid")
    {
        auto result = QMessageBox::information(this,"DSI Studio",
                "Current integer pixels cannot achieve floating point precision. Switch to float32 type?",
                                    QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
        if(result == QMessageBox::Cancel)
            return;
        if(result == QMessageBox::Yes)
            ui->type->setCurrentIndex(float32);
    }
    if(!command(action->text().toLower().replace(' ','_').toStdString(),value.toStdString()))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}


void view_image::on_type_currentIndexChanged(int index)
{
    command("change_type",std::to_string(index));
    init_image();
}


void view_image::on_zoom_valueChanged(double arg1)
{
    show_image(false);
}

void view_image::on_info_cellChanged(int row, int column)
{
    if(column == 0 && row < mat.size())
        mat[row].set_name(ui->info->item(row,column)->text().toStdString());
}

void view_image::on_info_cellDoubleClicked(int row, int column)
{
    if(column == 1 && row < mat.size() && mat[row].is_type<char>())
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
}


void view_image::on_mat_images_currentIndexChanged(int index)
{
    if(read_mat_image())
        init_image();
}

