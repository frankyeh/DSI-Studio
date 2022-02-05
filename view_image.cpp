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
bool img_command(tipl::image<3>& data,
                 tipl::vector<3>& vs,
                 tipl::matrix<4,4>& T,
                 std::string cmd,
                 std::string param1,
                 std::string,
                 std::string& error_msg)
{
    if(cmd == "image_multiplication" || cmd == "image_addition")
    {
        gz_nifti nii;
        if(!nii.load_from_file(param1.c_str()))
        {
            error_msg = "cannot open file:";
            error_msg += param1;
            return false;
        }
        tipl::image<3> mask;
        nii.get_untouched_image(mask);
        if(mask.shape() != data.shape())
        {
            error_msg = "invalid mask file:";
            error_msg += param1;
            error_msg += " The dimension does not match:";
            std::ostringstream out;
            out << mask.shape() << " vs " << data.shape();
            error_msg += out.str();
            return false;
        }
        if(cmd == "image_multiplication")
            data *= mask;
        if(cmd == "image_addition")
            data += mask;
        return true;
    }
    if(cmd == "save")
    {
        gz_nifti nii;
        nii.set_image_transformation(T);
        nii.set_voxel_size(vs);
        nii << data;
        return nii.save_to_file(param1.c_str());
    }
    if(cmd == "open")
    {
        gz_nifti nii;
        nii.set_image_transformation(T);
        nii.set_voxel_size(vs);
        nii << data;
        return nii.save_to_file(param1.c_str());
    }
    return false;
}


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

bool match_strings(const std::string& str1,const std::string& str1_match,
                   const std::string& str2,std::string& str2_match,bool try_reverse = true,bool try_swap = true)
{
    //std::cout << "if " << str1 << "->" << str1_match << ", then " << str2 << "->?" << std::endl;
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
    auto cp1_1 = common_prefix(str1,str1_match);
    auto cp1_2 = common_prefix(str1,str2);
    // A_B->A_C
    // D_B->D_C
    if(!cp1_1.empty())
    {
        // A->A_C
        // D->D_C
        if(cp1_1 == str1)
        {
            str2_match = str2 + str1_match.substr(cp1_1.size());
            return true;
        }
        if(match_strings(str1.substr(cp1_1.size()),str1_match.substr(cp1_1.size()),str2,str2_match))
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
    auto path1 = QFileInfo(file_path1.c_str()).absolutePath().toStdString();
    auto path2 = QFileInfo(file_path2.c_str()).absolutePath().toStdString();
    auto path1_others = QFileInfo(file_path1_others.c_str()).absolutePath().toStdString();

    std::string name2_others,path2_others;
    if(!match_strings(name1,name2,name1_others,name2_others) ||
       !match_strings(path1,path2,path1_others,path2_others))
        return false;
    file_path2_gen = path2_others + "/" + name2_others;
    std::cout << "matching " << file_path1_others << " with " << file_path2_gen << std::endl;
    return true;
}
bool view_image::command(std::string cmd,std::string param1,std::string param2)
{
    error_msg.clear();
    if(!img_command(data,vs,T,cmd,param1,param2,error_msg))
        return false;
    show_image();

    if(!other_data.empty())
    {

        std::vector<std::string> other_params(other_data.size());

        // generalized file names
        {
            std::string filename1 = std::filesystem::path(file_name.toStdString()).filename().string();
            std::string filename2 = std::filesystem::path(param1).filename().string();
            for(size_t i = 0;i < other_data.size();++i)
            {
                other_params[i] = param1;
                // if param1 is file name, then try to generalize
                if(param1.find(".gz") != std::string::npos)
                {
                    if(!match_files(file_name.toStdString(),param1,other_file_name[i],other_params[i]))
                    {
                        error_msg = "cannot find a matched file for ";
                        error_msg += other_file_name[i];
                        return false;
                    }
                    if(!std::filesystem::exists(other_params[i]))
                    {
                        error_msg = "cannot find ";
                        error_msg += other_params[i];
                        error_msg += " for processing ";
                        error_msg += other_file_name[i];
                        return false;
                    }
                }
            }
        }

        progress prog_("applying to others");
        for(size_t i = 0;progress::at(i,other_data.size());++i)
        {
            if(!img_command(other_data[i],other_vs[i],other_T[i],cmd,other_params[i],param2,error_msg))
            {
                error_msg += " when processing ";
                error_msg += QFileInfo(other_file_name[i].c_str()).fileName().toStdString();
                return false;
            }
        }
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

view_image::view_image(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::view_image)
{
    ui->setupUi(this);
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
    connect(ui->menuOverlay, SIGNAL(aboutToShow()),this, SLOT(update_overlay_menu()));

    source_ratio = 2.0;
    ui->tabWidget->setCurrentIndex(0);


    qApp->installEventFilter(this);
    opened_images.push_back(this);
}

void save_idx(const char* file_name,std::shared_ptr<gz_istream> in);
view_image::~view_image()
{
    opened_images.erase(std::remove(opened_images.begin(),opened_images.end(),this),opened_images.end());
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
            on_zoom_in_clicked();
        else
            on_zoom_out_clicked();
        event->accept();

        return true;
    }
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->view)
        return false;

    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    tipl::vector<3,float> pos,mni;
    tipl::slice2space(cur_dim,
                      std::round(float(point.x()) / source_ratio),
                      std::round(float(point.y()) / source_ratio),ui->slice_pos->value(),pos[0],pos[1],pos[2]);
    if(!data.shape().is_valid(pos))
        return true;
    mni = pos;
    mni.to(T);
    ui->info_label->setText(QString("(i,j,k)=(%1,%2,%3) (x,y,z)=(%4,%5,%6) value=%7").arg(pos[0]).arg(pos[1]).arg(pos[2])
                                                                    .arg(mni[0]).arg(mni[1]).arg(mni[2])
                                                                    .arg(data.at(pos[0],pos[1],pos[2])));
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
void prepare_idx(const char* file_name,std::shared_ptr<gz_istream> in);
bool view_image::open(QStringList file_names)
{
    tipl::io::dicom dicom;
    tipl::io::bruker_2dseq seq;
    gz_mat_read mat;
    data.clear();
    T.identity();

    QString info;
    file_name = file_names[0];
    setWindowTitle(QFileInfo(file_name).fileName());
    progress prog_("loading ",std::filesystem::path(file_name.toStdString()).filename().string().c_str());
    progress::at(0,1);

    if(file_names.size() > 1 && QString(file_name).endsWith(".bmp"))
    {
        for(unsigned int i = 0;progress::at(i,file_names.size());++i)
        {
            tipl::color_image I;
            tipl::io::bitmap bmp;
            if(!bmp.load_from_file(file_names[i].toStdString().c_str()))
                return false;
            bmp >> I;
            if(i == 0)
                data.resize(tipl::shape<3>(I.width(),I.height(),file_names.size()));
            unsigned int pos = i*I.size();
            for(unsigned int j = 0;j < I.size();++j)
                data[pos+j] = ((float)I[j].r+(float)I[j].r+(float)I[j].r)/3.0;
        }
    }
    else
    if(QString(file_name).endsWith(".nii.gz") || QString(file_name).endsWith(".nii"))
    {
        prepare_idx(file_name.toStdString().c_str(),nifti.input_stream);
        if(!nifti.load_from_file(file_name.toStdString().c_str()))
        {
            QMessageBox::critical(this,"Error","Invalid NIFTI file");
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
        nifti.get_untouched_image(data);
        nifti.get_voxel_size(vs);
        nifti.get_image_transformation(T);
        std::ostringstream out;
        out << nifti;
        info = out.str().c_str();

        if(file_names.size() > 1)
        {
            progress prog_("reading");
            QString failed_list;
            for(int i = 1;progress::at(i,file_names.size());++i)
            {
                gz_nifti other_nifti;
                if(!other_nifti.load_from_file(file_names[i].toStdString()))
                {
                    if(!failed_list.isEmpty())
                        failed_list += ",";
                    failed_list += QFileInfo(file_names[i]).fileName();
                    continue;
                }
                tipl::image<3> odata;
                tipl::vector<3> ovs;
                tipl::matrix<4,4> oT;
                other_nifti.get_untouched_image(odata);
                other_nifti.get_voxel_size(ovs);
                other_nifti.get_image_transformation(oT);
                other_file_name.push_back(file_names[i].toStdString());
                other_data.push_back(odata);
                other_vs.push_back(ovs);
                other_T.push_back(oT);
            }
            if(!failed_list.isEmpty())
                QMessageBox::critical(this,"Error",QString("Some files could not be opened:")+failed_list);
            else
                QMessageBox::information(this,"DSI Studio",QString("Other files read in memory for operation"));
        }
    }
    else
        if(dicom.load_from_file(file_name.toStdString()))
        {
            dicom >> data;
            if(dicom.is_compressed)
            {
                tipl::image<2,short> I;
                if(!get_compressed_image(dicom,I))
                {
                    QMessageBox::information(this,"Error",QString("Unsupported transfer syntax:") + QString(dicom.encoding.c_str()),0);
                    return false;
                }
                if(I.size() == data.size())
                    std::copy(I.begin(),I.end(),data.begin());
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
            if(mat.load_from_file(file_name.toLocal8Bit().begin()))
            {
                if((mat.has("fa0") || mat.has("image0")) && mat.has("dimension"))
                {
                    tipl::shape<3> geo;
                    mat.read("dimension",geo);
                    data.resize(geo);
                    mat.read(mat.has("fa0") ? "fa0":"image0",data);
                }
                else
                    mat >> data;
                if(mat.has("trans"))
                    mat.read("trans",T);
                mat.get_voxel_size(vs);
                for(unsigned int index = 0;index < mat.size();++index)
                {
                    std::string data;
                    mat[index].get_info(data);
                    info += QString("%1 [%2x%3]=%4\n").arg(mat[index].get_name().c_str()).
                            arg(mat[index].get_rows()).
                            arg(mat[index].get_cols()).
                            arg(data.c_str());
                }
            }
            else
            if(seq.load_from_file(file_name.toLocal8Bit().begin()))
            {
                data = std::move(seq.get_image());
                seq.get_voxel_size(vs);
            }
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

    if(!data.empty())
    {
        init_image();
        show_image();
    }
    return !data.empty() || !info.isEmpty();
}

void view_image::init_image(void)
{
    no_update = true;
    max_value = tipl::max_value(data);
    min_value = tipl::min_value(data);
    float range = max_value-min_value;
    QString dim_text = QString("%1,%2,%3").arg(data.width()).arg(data.height()).arg(data.depth());
    if(!dwi_volume_buf.empty())
        dim_text += QString(",%1").arg(dwi_volume_buf.size());
    ui->image_info->setText(QString("dim=(%1) vs=(%4,%5,%6) srow=[%7 %8 %9 %10][%11 %12 %13 %14][%15 %16 %17 %18]").
            arg(dim_text).
            arg(double(vs[0])).arg(double(vs[1])).arg(double(vs[2])).
            arg(double(T[0])).arg(double(T[1])).arg(double(T[2])).arg(double(T[3])).
            arg(double(T[4])).arg(double(T[5])).arg(double(T[6])).arg(double(T[7])).
            arg(double(T[8])).arg(double(T[9])).arg(double(T[10])).arg(double(T[11])));
    ui->min->setRange(double(min_value-range/3),double(max_value));
    ui->max->setRange(double(min_value),double(max_value+range/3));
    ui->min->setSingleStep(double(range/20));
    ui->max->setSingleStep(double(range/20));
    ui->min->setValue(double(min_value));
    ui->max->setValue(double(max_value));
    slice_pos[0] = data.width()/2;
    slice_pos[1] = data.height()/2;
    slice_pos[2] = data.depth()/2;
    on_AxiView_clicked();
    no_update = false;
}
void view_image::add_overlay(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    size_t index = size_t(action->data().toInt());
    if(index >= opened_images.size())
        return;
    overlay.clear();
    overlay.resize(data.shape());
    tipl::resample_mt<tipl::interpolation::cubic>(opened_images[index]->data,overlay,
                      tipl::transformation_matrix<float>(tipl::from_space(T).to(opened_images[index]->T)));
    overlay_v2c = opened_images[index]->v2c;
    show_image();
}
void view_image::update_overlay_menu(void)
{
    while(ui->menuOverlay->actions().size() > int(opened_images.size()))
    {
        ui->menuOverlay->removeAction(ui->menuOverlay->actions()[ui->menuOverlay->actions().size()-1]);
    }
    for (size_t index = 0; index < opened_images.size(); ++index)
    {
        if(index >= size_t(ui->menuOverlay->actions().size()))
        {
            QAction* Item = new QAction(this);
            Item->setVisible(true);
            Item->setData(int(index));
            connect(Item, SIGNAL(triggered()),this, SLOT(add_overlay()));
            ui->menuOverlay->addAction(Item);
        }
        ui->menuOverlay->actions()[index]->setText(opened_images[index]->windowTitle());
    }
}
void view_image::show_image(void)
{
    if(data.empty() || no_update)
        return;
    tipl::image<2,float> buf;
    tipl::volume2slice(data, buf, cur_dim, size_t(slice_pos[cur_dim]));
    v2c.convert(buf,buffer);

    if(overlay.size() == data.size())
    {
        tipl::image<2,float> buf_overlay;
        tipl::color_image buffer2;
        tipl::volume2slice(overlay, buf_overlay, cur_dim, size_t(slice_pos[cur_dim]));
        overlay_v2c.convert(buf_overlay,buffer2);
        for(size_t i = 0;i < buffer.size();++i)
            buffer[i] |= buffer2[i];
    }
    QImage I(reinterpret_cast<unsigned char*>(&*buffer.begin()),buffer.width(),buffer.height(),QImage::Format_RGB32);
    source_image = I.scaled(buffer.width()*source_ratio,buffer.height()*source_ratio);
    show_view(source,source_image);
}
void view_image::change_contrast()
{
    v2c.set_range(float(ui->min->value()),float(ui->max->value()));
    v2c.two_color(ui->min_color->color().rgb(),ui->max_color->color().rgb());
    show_image();
}
void view_image::on_zoom_in_clicked()
{
     source_ratio *= 1.1f;
     show_image();
}

void view_image::on_zoom_out_clicked()
{
    source_ratio *= 0.9f;
    show_image();
}

bool is_label_image(const tipl::image<3>& I);
void view_image::on_actionResample_triggered()
{
    bool ok;
    float nv = float(QInputDialog::getDouble(this,
        "DSI Studio","Assign output resolution in (mm):", double(vs[0]),0.0,3.0,4, &ok));
    if (!ok || nv == 0.0f)
        return;
    tipl::vector<3,float> new_vs(nv,nv,nv);
    tipl::image<3> J(tipl::shape<3>(
            int(std::ceil(float(data.width())*vs[0]/new_vs[0])),
            int(std::ceil(float(data.height())*vs[1]/new_vs[1])),
            int(std::ceil(float(data.depth())*vs[2]/new_vs[2]))));
    if(J.empty())
        return;
    tipl::transformation_matrix<float> T1;
    tipl::matrix<4,4> nT;
    nT.identity();
    nT[0] = T1.sr[0] = new_vs[0]/vs[0];
    nT[5] = T1.sr[4] = new_vs[1]/vs[1];
    nT[10] = T1.sr[8] = new_vs[2]/vs[2];
    if(is_label_image(data))
        tipl::resample_mt<tipl::interpolation::nearest>(data,J,T1);
    else
        tipl::resample_mt<tipl::interpolation::cubic>(data,J,T1);
    data.swap(J);
    vs = new_vs;
    T = T*nT;

    init_image();
    show_image();
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
                           this,"Save image",file_name,"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    if(command("save",filename.toStdString()))
    {
        QMessageBox::information(this,"DSI Studio","Saved");
        file_name = filename;
        setWindowTitle(QFileInfo(file_name).fileName());
    }
    else
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}
void view_image::on_actionSave_as_Int8_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save image",file_name,"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    tipl::image<3,uint8_t> new_data = data;
    gz_nifti nii;
    nii.set_image_transformation(T);
    nii.set_voxel_size(vs);
    nii << new_data;
    nii.save_to_file(filename.toStdString().c_str());
    file_name = filename;
    setWindowTitle(QFileInfo(file_name).fileName());
}

void view_image::on_actionSave_as_Int16_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save image",file_name,"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    tipl::image<3,uint16_t> new_data = data;
    gz_nifti nii;
    nii.set_image_transformation(T);
    nii.set_voxel_size(vs);
    nii << new_data;
    nii.save_to_file(filename.toStdString().c_str());
    file_name = filename;
    setWindowTitle(QFileInfo(file_name).fileName());
}

void view_image::on_actionResize_triggered()
{
    std::ostringstream out;
    out << data.width() << " " << data.height() << " " << data.depth();
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign image dimension (width height depth)",QLineEdit::Normal,
                                           out.str().c_str(),&ok);

    if(!ok)
        return;
    std::istringstream in(result.toStdString());
    int w,h,d;
    in >> w >> h >> d;
    tipl::image<3> new_data(tipl::shape<3>(w,h,d));
    tipl::draw(data,new_data,tipl::vector<3>());
    data.swap(new_data);
    init_image();
    show_image();
}

void view_image::on_actionTranslocate_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign image translocation (x y z)",QLineEdit::Normal,
                                           "0 0 0",&ok);

    if(!ok)
        return;
    std::istringstream in(result.toStdString());
    int dx,dy,dz;
    in >> dx >> dy >> dz;
    tipl::image<3> new_data(data.shape());
    tipl::draw(data,new_data,tipl::vector<3>(dx,dy,dz));
    data.swap(new_data);
    T[3] -= T[0]*dx;
    T[7] -= T[5]*dy;
    T[11] -= T[10]*dz;
    init_image();
    show_image();
}

void view_image::on_actionTrim_triggered()
{
    tipl::vector<3,int> range_min,range_max;
    tipl::bounding_box(data,range_min,range_max,data[0]);
    int margin = 1;
    tipl::vector<3,int> translocate(margin,margin,0);
    range_min[2] += 1;
    range_max[2] -= 1;
    translocate -= range_min;
    range_max -= range_min;
    range_max[0] += margin+margin;
    range_max[1] += margin+margin;
    tipl::image<3> new_data(tipl::shape<3>(range_max[0],range_max[1],range_max[2]));
    tipl::draw(data,new_data,translocate);
    data.swap(new_data);
    T[3] -= T[0]*translocate[0];
    T[7] -= T[5]*translocate[1];
    T[11] -= T[10]*translocate[2];
    init_image();
    show_image();
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
    std::istringstream in(result.toStdString());
    in >> T[3] >> T[7] >> T[11];
    init_image();
    show_image();
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
    std::istringstream in(result.toStdString());
    for(int i = 0;i < 16;++i)
        in >> T[i];
    init_image();
    show_image();
}

void view_image::on_actionLower_threshold_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign lower threshold value ",QLineEdit::Normal,
                                           "0",&ok);
    if(!ok)
        return;
    float value = result.toFloat(&ok);
    if(!ok)
        return;

    for(size_t index = 0;index < data.size();++index)
        if(data[index] < value)
            data[index] = value;

    init_image();
    show_image();
}


void view_image::on_actionIntensity_shift_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign shift value",QLineEdit::Normal,
                                           "0",&ok);
    if(!ok)
        return;
    float value = result.toFloat(&ok);
    if(!ok)
        return;

    data += value;

    init_image();
    show_image();
}

void view_image::on_actionIntensity_scale_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign scale value",QLineEdit::Normal,
                                           "0",&ok);
    if(!ok)
        return;
    float value = result.toFloat(&ok);
    if(!ok)
        return;

    data *= value;

    init_image();
    show_image();
}

void view_image::on_actionUpper_Threshold_triggered()
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign upper threshold value ",QLineEdit::Normal,
                                           "0",&ok);
    if(!ok)
        return;
    float value = result.toFloat(&ok);
    if(!ok)
        return;

    for(size_t index = 0;index < data.size();++index)
        if(data[index] > value)
            data[index] = value;

    init_image();
    show_image();
}
bool is_label_image(const tipl::image<3>& I);
void view_image::on_actionSmoothing_triggered()
{
    tipl::image<3,uint32_t> new_data(data);
    uint32_t m = uint32_t(tipl::max_value(data));

    // smooth each region
    tipl::par_for(m,[&](uint32_t index)
    {
        if(!index)
            return;
        tipl::image<3,char> mask(new_data.shape());
        for(size_t i = 0;i < mask.size();++i)
            if(new_data[i] == index)
                mask[i] = 1;
        tipl::morphology::smoothing(mask);
        for(size_t i = 0;i < mask.size();++index)
            if(mask[i] && new_data[i] < index)
                new_data[i] = index;
    });

    // fill up gaps
    tipl::par_for(m,[&](uint32_t index)
    {
        if(!index)
            return;
        tipl::image<3,char> mask(data.shape());
        for(size_t i = 0;i < mask.size();++i)
            if(new_data[i] == index)
                mask[i] = 1;
        tipl::morphology::dilation(mask);
        for(size_t i = 0;i < mask.size();++i)
            if(mask[i] && new_data[i] < index)
                new_data[i] = index;
    });
    data = new_data;
    init_image();
    show_image();
}


void view_image::on_actionNormalize_Intensity_triggered()
{
    tipl::normalize(data,1.0f);
    init_image();
    show_image();
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
    cur_dim = 2;
    ui->slice_pos->setRange(0,data.depth()-1);
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_CorView_clicked()
{
    cur_dim = 1;
    ui->slice_pos->setRange(0,data.height()-1);
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_SagView_clicked()
{
    cur_dim = 0;
    ui->slice_pos->setRange(0,data.width()-1);
    ui->slice_pos->setValue(slice_pos[cur_dim]);
}

void view_image::on_slice_pos_valueChanged(int value)
{
    if(data.empty())
        return;
    slice_pos[cur_dim] = value;
    show_image();
}

void view_image::on_actionSobel_triggered()
{
    tipl::filter::sobel(data);
    init_image();
    show_image();
}

void view_image::on_actionMorphology_triggered()
{
    tipl::morphology::edge(data);
    init_image();
    show_image();
}

void view_image::on_actionMorphology_Thin_triggered()
{
    tipl::morphology::edge_thin(data);
    init_image();
    show_image();
}

void view_image::on_actionMorphology_XY_triggered()
{
    tipl::morphology::edge_xy(data);
    init_image();
    show_image();
}

void view_image::on_actionMorphology_XZ_triggered()
{
    tipl::morphology::edge_xz(data);
    init_image();
    show_image();
}


void view_image::on_actionImageAddition_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open other another image to apply",QFileInfo(file_name).absolutePath(),"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    if(!command("image_addition",filename.toStdString(),std::string()))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}

void view_image::on_actionImageMultiplication_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open other another image to apply",QFileInfo(file_name).absolutePath(),"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    if(!command("image_multiplication",filename.toStdString(),std::string()))
        QMessageBox::critical(this,"ERROR",error_msg.c_str());
}

void view_image::on_actionSignal_Smoothing_triggered()
{
    tipl::filter::mean(data);
    init_image();
    show_image();
}

void view_image::on_dwi_volume_valueChanged(int value)
{
    dwi_volume_buf[cur_dwi_volume].swap(data); // return image data to buffer

    cur_dwi_volume = size_t(value);
    if(dwi_volume_buf[cur_dwi_volume].empty())
    {
        has_gui = false;
        nifti.select_volume(cur_dwi_volume);
        tipl::image<3> new_data;
        nifti.get_untouched_image(new_data);
        has_gui = true;
        if(new_data.empty())
            return;
        new_data.swap(data);
    }
    else
        dwi_volume_buf[cur_dwi_volume].swap(data);

    ui->dwi_label->setText(QString("(%1/%2)").arg(value+1).arg(ui->dwi_volume->maximum()+1));
    show_image();
}

void view_image::on_actionDownsample_by_2_triggered()
{
    tipl::downsample_with_padding(data);
    init_image();
    show_image();
}

void view_image::on_actionUpsample_by_2_triggered()
{
    tipl::upsampling(data);
    init_image();
    show_image();
}



void view_image::on_actionFlip_X_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({-1,0,0,data.width(),
                           0,1,0,0,
                           0,0,1,0,
                           0,0,0,1})*T;
    tipl::flip_x(data);
    init_image();
    show_image();
}


void view_image::on_actionFlip_Y_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({1,0,0,0,
                           0,-1,0,data.height(),
                           0,0,1,0,
                           0,0,0,1})*T;
    tipl::flip_y(data);
    init_image();
    show_image();
}


void view_image::on_actionFlip_Z_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({1,0,0,0,
                           0,1,0,0,
                           0,0,-1,data.depth(),
                           0,0,0,1})*T;
    tipl::flip_z(data);
    init_image();
    show_image();
}


void view_image::on_actionSwap_XY_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({0,1,0,0,
                           1,0,0,0,
                           0,0,1,0,
                           0,0,0,1})*T;
    tipl::swap_xy(data);
    init_image();
    show_image();
}


void view_image::on_actionSwap_XZ_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({0,0,1,0,
                           0,1,0,0,
                           1,0,0,0,
                           0,0,0,1})*T;
    tipl::swap_xz(data);
    init_image();
    show_image();
}


void view_image::on_actionSwap_YZ_triggered()
{
    if(data.empty())
        return;
    T = tipl::matrix<4,4>({1,0,0,0,
                           0,0,1,0,
                           0,1,0,0,
                           0,0,0,1})*T;
    tipl::swap_yz(data);
    init_image();
    show_image();
}

