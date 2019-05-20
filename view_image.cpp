#include <map>
#include <QTextStream>
#include <QInputDialog>
#include <QFileDialog>
#include "view_image.h"
#include "ui_view_image.h"
#include "libs/gzip_interface.hpp"
#include "prog_interface_static_link.h"
#include <QPlainTextEdit>
#include <QFileInfo>
#include <QMessageBox>
#include <QBuffer>
#include <QImageReader>

std::map<std::string,std::string> dicom_dictionary;

void show_view(QGraphicsScene& scene,QImage I);
bool load_image_from_files(QStringList filenames,tipl::image<float,3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4,float>& trans)
{
    if(filenames.size() == 1 && filenames[0].toLower().contains("nii"))
    {
        gz_nifti in;
        if(!in.load_from_file(filenames[0].toLocal8Bit().begin()) || !in.toLPS(ref))
        {
            QMessageBox::information(0,"Error","Not a valid nifti file",0);
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
                QMessageBox::information(0,"Error","Not a valid 2dseq file",0);
                return false;
            }
            seq.get_image().swap(ref);
            seq.get_voxel_size(vs);
            return true;
        }
    else
    {
        tipl::io::volume v;
        std::vector<std::string> file_list;
        for(int i = 0;i < filenames.size();++i)
            file_list.push_back(filenames[i].toStdString());
        v.load_from_files(file_list,file_list.size());
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
    connect(ui->slice_pos,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    connect(ui->contrast,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    connect(ui->brightness,SIGNAL(valueChanged(int)),this,SLOT(update_image()));
    source_ratio = 2.0;
    ui->tabWidget->setCurrentIndex(0);


    qApp->installEventFilter(this);
}

view_image::~view_image()
{
    qApp->removeEventFilter(this);
    delete ui;
}
bool view_image::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->view)
        return false;
    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    tipl::vector<3,float> pos,mni;
    pos[0] = std::round(((float)point.x()) / source_ratio);
    pos[1] = std::round(((float)point.y()) / source_ratio);
    pos[2] = ui->slice_pos->value();
    if(!data.geometry().is_valid(pos))
        return true;
    mni = pos;
    mni.to(T);
    ui->info_label->setText(QString("(%1,%2,%3) MNI(%4,%5,%6) = %7").arg(pos[0]).arg(pos[1]).arg(pos[2])
                                                                    .arg(mni[0]).arg(mni[1]).arg(mni[2])
                                                                    .arg(data.at(pos[0],pos[1],pos[2])));
    return true;
}

bool get_compressed_image(tipl::io::dicom& dicom,tipl::image<short,2>& I)
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
    I.resize(tipl::geometry<2>(buf.width(),buf.height()));
    const uchar* ptr = buf.bits();
    for(int j = 0;j < I.size();++j,ptr += 4)
        I[j] = *ptr;
    return true;
}

bool view_image::open(QStringList file_names)
{
    gz_nifti nifti;
    tipl::io::dicom dicom;
    tipl::io::bruker_2dseq seq;
    gz_mat_read mat;
    data.clear();
    T.identity();

    QString info;
    file_name = file_names[0];
    setWindowTitle(QFileInfo(file_name).fileName());
    begin_prog("loading...");
    check_prog(0,1);
    if(file_names.size() > 1 && file_name.contains("bmp"))
    {
        for(unsigned int i = 0;check_prog(i,file_names.size());++i)
        {
            tipl::color_image I;
            tipl::io::bitmap bmp;
            if(!bmp.load_from_file(file_names[i].toStdString().c_str()))
                return false;
            bmp >> I;
            if(i == 0)
                data.resize(tipl::geometry<3>(I.width(),I.height(),file_names.size()));
            unsigned int pos = i*I.size();
            for(unsigned int j = 0;j < I.size();++j)
                data[pos+j] = ((float)I[j].r+(float)I[j].r+(float)I[j].r)/3.0;
        }
    }
    else
    if(nifti.load_from_file(file_name.toLocal8Bit().begin()))
    {
        nifti >> data;
        nifti.get_voxel_size(vs);
        nifti.get_image_transformation(T);
        info = QString("sizeof_hdr=%1\ndim_info=%2\n").
                arg(nifti.nif_header2.sizeof_hdr).
                arg((int)nifti.nif_header2.dim_info);
        for(unsigned int i = 0;i < 8;++i)
            info +=  QString("dim[%1]=%2\n").
                    arg(i).arg(nifti.nif_header2.dim[i]);
        info += QString("intent_p1=%1\n").arg(nifti.nif_header2.intent_p1);
        info += QString("intent_p2=%1\n").arg(nifti.nif_header2.intent_p2);
        info += QString("intent_p3=%1\n").arg(nifti.nif_header2.intent_p3);
        info += QString("intent_code=%1\n").arg(nifti.nif_header2.intent_code);
        info += QString("datatype=%1\n").arg(nifti.nif_header2.datatype);
        info += QString("bitpix=%1\n").arg(nifti.nif_header2.bitpix);
        info += QString("slice_start=%1\n").arg(nifti.nif_header2.slice_start);

        for(unsigned int i = 0;i < 8;++i)
            info +=  QString("pixdim[%1]=%2\n").
                    arg(i).arg(nifti.nif_header2.pixdim[i]);

        info += QString("vox_offset=%1\n").arg(nifti.nif_header2.vox_offset);
        info += QString("scl_slope=%1\n").arg(nifti.nif_header2.scl_slope);
        info += QString("scl_inter=%1\n").arg(nifti.nif_header2.scl_inter);
        info += QString("slice_end=%1\n").arg(nifti.nif_header2.slice_end);
        info += QString("slice_code=%1\n").arg((int)nifti.nif_header2.slice_code);
        info += QString("xyzt_units=%1\n").arg((int)nifti.nif_header2.xyzt_units);
        info += QString("scl_inter=%1\n").arg(nifti.nif_header2.scl_inter);
        info += QString("cal_max=%1\n").arg(nifti.nif_header2.cal_max);
        info += QString("cal_min=%1\n").arg(nifti.nif_header2.cal_min);
        info += QString("slice_duration=%1\n").arg(nifti.nif_header2.slice_duration);
        info += QString("toffset=%1\n").arg(nifti.nif_header2.toffset);
        info += QString("descrip=%1\n").arg(nifti.nif_header2.descrip);
        info += QString("aux_file=%1\n").arg(nifti.nif_header2.aux_file);
        info += QString("qform_code=%1\n").arg(nifti.nif_header2.qform_code);
        info += QString("sform_code=%1\n").arg(nifti.nif_header2.sform_code);
        info += QString("quatern_b=%1\n").arg(nifti.nif_header2.quatern_b);
        info += QString("quatern_c=%1\n").arg(nifti.nif_header2.quatern_c);
        info += QString("quatern_d=%1\n").arg(nifti.nif_header2.quatern_d);
        info += QString("qoffset_x=%1\n").arg(nifti.nif_header2.qoffset_x);
        info += QString("qoffset_y=%1\n").arg(nifti.nif_header2.qoffset_y);
        info += QString("qoffset_z=%1\n").arg(nifti.nif_header2.qoffset_z);

        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_x[%1]=%2\n").
                    arg(i).arg(nifti.nif_header2.srow_x[i]);
        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_y[%1]=%2\n").
                    arg(i).arg(nifti.nif_header2.srow_y[i]);
        for(unsigned int i = 0;i < 4;++i)
            info +=  QString("srow_z[%1]=%2\n").
                    arg(i).arg(nifti.nif_header2.srow_z[i]);

        info += QString("intent_name=%1\n").arg(nifti.nif_header2.intent_name);


    }
    else
        if(dicom.load_from_file(file_name.toLocal8Bit().begin()))
        {
            dicom >> data;
            if(dicom.is_compressed)
            {
                tipl::image<short,2> I;
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
                mat >> data;
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
    check_prog(0,0);
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
        update_image();
    }
    return !data.empty() || !info.isEmpty();
}

void view_image::init_image(void)
{
    ui->slice_pos->setRange(0,data.depth()-1);
    ui->slice_pos->setValue(data.depth() >> 1);
    ui->image_info->setText(QString("dim=(%1,%2,%3) vs=(%4,%5,%6) T=(%7,%8,%9,%10;%11,%12,%13,%14;%15,%16,%17,%18)").
            arg(data.width()).arg(data.height()).arg(data.depth()).
            arg(double(vs[0])).arg(double(vs[1])).arg(double(vs[2])).
            arg(double(T[0])).arg(double(T[1])).arg(double(T[2])).arg(double(T[3])).
            arg(double(T[4])).arg(double(T[5])).arg(double(T[6])).arg(double(T[7])).
            arg(double(T[8])).arg(double(T[9])).arg(double(T[10])).arg(double(T[11])));
}
void view_image::update_image(void)
{
    if(data.empty())
        return;
    tipl::image<float,2> tmp(tipl::geometry<2>(data.width(),data.height()));
    size_t offset = size_t(ui->slice_pos->value())*tmp.size();

    std::copy(data.begin() + offset,
              data.begin() + offset + tmp.size(),tmp.begin());
    max_source_value = std::max<float>(max_source_value,*std::max_element(tmp.begin(),tmp.end()));
    if(max_source_value + 1.0f != 1.0f)
        tipl::divide_constant(tmp.begin(),tmp.end(),max_source_value/255.0f);

    float mean_value = tipl::mean(tmp.begin(),tmp.end());
    tipl::minus_constant(tmp.begin(),tmp.end(),mean_value);
    tipl::multiply_constant(tmp.begin(),tmp.end(),ui->contrast->value());
    tipl::add_constant(tmp.begin(),tmp.end(),mean_value+ui->brightness->value()*25.5f);

    tipl::upper_lower_threshold(tmp.begin(),tmp.end(),tmp.begin(),0.0f,255.0f);


    buffer.resize(tipl::geometry<2>(data.width(),data.height()));
    std::copy(tmp.begin(),tmp.end(),buffer.begin());

    source_image = QImage((unsigned char*)&*buffer.begin(),data.width(),data.height(),QImage::Format_RGB32).
                    scaled(data.width()*source_ratio,data.height()*source_ratio);

    show_view(source,source_image);
}

void view_image::on_zoom_in_clicked()
{
     source_ratio *= 1.1f;
     update_image();
}

void view_image::on_zoom_out_clicked()
{
    source_ratio *= 0.9f;
    update_image();
}

void view_image::on_actionResample_triggered()
{
    bool ok;
    float nv = float(QInputDialog::getDouble(this,
        "DSI Studio","Assign output resolution in (mm):", double(vs[0]),0.0,3.0,4, &ok));
    if (!ok || nv == 0.0f)
        return;
    tipl::vector<3,float> new_vs(nv,nv,nv);
    tipl::image<float,3> J(tipl::geometry<3>(
            int(std::ceil(float(data.width())*vs[0]/new_vs[0])),
            int(std::ceil(float(data.height())*vs[1]/new_vs[1])),
            int(std::ceil(float(data.depth())*vs[2]/new_vs[2]))));
    if(J.empty())
        return;
    tipl::transformation_matrix<float> T1;
    tipl::matrix<4,4,float> nT;
    nT.identity();
    nT[0] = T1.sr[0] = new_vs[0]/vs[0];
    nT[5] = T1.sr[4] = new_vs[1]/vs[1];
    nT[10] = T1.sr[8] = new_vs[2]/vs[2];
    tipl::resample_mt(data,J,T1,tipl::cubic);
    data.swap(J);
    vs = new_vs;
    T = T*nT;

    init_image();
    update_image();
}

void view_image::on_action_Save_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save image",file_name,"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    gz_nifti nii;
    nii.set_image_transformation(T);
    nii.set_voxel_size(vs);
    nii << data;
    nii.save_to_file(filename.toStdString().c_str());
}

void view_image::on_actionMasking_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open mask",QFileInfo(file_name).absolutePath(),"NIFTI file(*nii.gz *.nii)" );
    if (filename.isEmpty())
        return;
    gz_nifti nii;
    if(!nii.load_from_file(filename.toStdString().c_str()))
    {
        QMessageBox::information(this,"Error","Cannot open file",0);
        return;
    }
    tipl::image<float,3> mask;
    nii >> mask;
    if(mask.geometry() != data.geometry())
    {
        QMessageBox::information(this,"Error","Invalid mask file. Dimension does not match",0);
        return;
    }
    tipl::filter::gaussian(mask);
    tipl::filter::gaussian(mask);
    tipl::normalize(mask,1.0f);
    data *= mask;
    update_image();
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
    tipl::image<float,3> new_data(tipl::geometry<3>(w,h,d));
    tipl::draw(data,new_data,tipl::vector<3>());
    data.swap(new_data);
    init_image();
    update_image();
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
    tipl::image<float,3> new_data(data.geometry());
    tipl::draw(data,new_data,tipl::vector<3>(dx,dy,dz));
    data.swap(new_data);
    T[3] -= T[0]*dx;
    T[7] -= T[5]*dy;
    T[11] -= T[10]*dz;
    init_image();
    update_image();
}

void view_image::on_actionTrim_triggered()
{
    tipl::vector<3,int> range_min,range_max;
    tipl::bounding_box(data,range_min,range_max,0);
    int margin = (range_max[0]-range_min[0])/10;
    tipl::vector<3,int> translocate(margin,margin,0);
    range_min[2] += 1;
    range_max[2] -= 1;
    translocate -= range_min;
    range_max -= range_min;
    range_max[0] += margin+margin;
    range_max[1] += margin+margin;
    tipl::image<float,3> new_data(tipl::geometry<3>(range_max[0],range_max[1],range_max[2]));
    tipl::draw(data,new_data,translocate);
    data.swap(new_data);
    T[3] -= T[0]*translocate[0];
    T[7] -= T[5]*translocate[1];
    T[11] -= T[10]*translocate[2];
    init_image();
    update_image();
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
    update_image();
}
