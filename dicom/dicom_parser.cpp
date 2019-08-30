#include <qmessagebox.h>
#include <QProgressDialog>
#include <QFileDialog>
#include <QSettings>
#include "dicom_parser.h"
#include "ui_dicom_parser.h"
#include "tipl/tipl.hpp"
#include "mainwindow.h"
#include "prog_interface_static_link.h"
#include "libs/gzip_interface.hpp"



void get_report_from_dicom(const tipl::io::dicom& header,std::string& report_);
void get_report_from_bruker(const tipl::io::bruker_info& header,std::string& report_);
void get_report_from_bruker2(const tipl::io::bruker_info& header,std::string& report_);

QString get_dicom_output_name(QString file_name,QString file_extension)
{
    tipl::io::dicom header;
    if (header.load_from_file(file_name.toLocal8Bit().begin()))
    {
        std::string Person;
        header.get_patient(Person);
        std::string seq_num;
        header.get_sequence_num(seq_num);
        QDir dir = QFileInfo(file_name).absoluteDir();
        dir.cdUp();
        return dir.absolutePath() + "/" + Person.c_str() + "_s" + seq_num.c_str() + file_extension;
    }
    else
        return file_name+".src.gz";
}



dicom_parser::dicom_parser(QStringList file_list,QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::dicom_parser)
{
    ui->setupUi(this);
    cur_path = QFileInfo(file_list[0]).absolutePath();
    load_files(file_list);

    if (!dwi_files.empty())
    {
        ui->SrcName->setText(get_dicom_output_name(file_list[0],".src.gz"));
        tipl::io::dicom header;
        if (header.load_from_file(file_list[0].toLocal8Bit().begin()))
        {
            slice_orientation.resize(9);
            header.get_image_orientation(slice_orientation.begin());
        }
    }
}
void dicom_parser::set_name(QString name)
{
    ui->SrcName->setText(name);
}

dicom_parser::~dicom_parser()
{
    delete ui;
}

bool load_dicom_multi_frame(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    tipl::io::dicom dicom_header;// multiple frame image
    if(!dicom_header.load_from_file(file_name))
        return false;
    tipl::image<float,3> buf_image;
    dicom_header >> buf_image;
    unsigned int slice_num = dicom_header.get_int(0x2001,0x1018);
    std::vector<float> b_table;

    {
        std::vector<float> b,bx,by,bz;
        dicom_header.get_values(0x2001,0x1003,b);
        dicom_header.get_values(0x2005,0x10B0,bx);
        dicom_header.get_values(0x2005,0x10B1,by);
        dicom_header.get_values(0x2005,0x10B2,bz);
        if(!b.empty())
        {
            if(!slice_num)
                slice_num = 1;
            b.resize(buf_image.depth()/slice_num);
            bx.resize(buf_image.depth()/slice_num);
            by.resize(buf_image.depth()/slice_num);
            bz.resize(buf_image.depth()/slice_num);
            for(int i = 0;i < b.size();++i)
            {
                tipl::vector<3, float> bvec(bx[i],by[i],bz[i]);
                b_table.push_back(b[i]*bvec.length());
                if(bvec.length() > 0)
                    bvec.normalize();
                b_table.push_back(bvec[0]);
                b_table.push_back(bvec[1]);
                b_table.push_back(bvec[2]);
            }
        }
    }

    if(b_table.empty())
    {
        for(int i = 0;i < dicom_header.data.size();++i)
            if(dicom_header.data[i].sq_data.size() == buf_image.depth())
            {
                std::string slice_pos;
                for(int j = 0;j < buf_image.depth();++j)
                {
                    std::string pos;
                    if(!tipl::io::dicom::get_values(dicom_header.data[i].sq_data[j],0x0020,0x0032,
                                                    j ? pos : slice_pos))
                        break;
                    if(j && pos != slice_pos)
                    {
                        slice_num = buf_image.depth()/j;
                        break;
                    }
                    float b_value = 0;
                    tipl::io::dicom::get_value(dicom_header.data[i].sq_data[j],0x0018,0x9087,b_value);
                    b_table.push_back(b_value);
                    if(b_value == 0 || !tipl::io::dicom::get_values(dicom_header.data[i].sq_data[j],0x0018,0x9089,b_table))
                    {
                        b_table.push_back(0);
                        b_table.push_back(0);
                        b_table.push_back(0);
                    }
                }
                if(slice_num == 0)
                    continue;
                break;
            }
    }

    if(!slice_num)
        slice_num = 1;
    // Philips multiframe
    unsigned int num_gradient = buf_image.depth()/slice_num;
    unsigned int plane_size = buf_image.width()*buf_image.height();
    b_table.resize(num_gradient*4);
    begin_prog("loading multi frame DICOM");
    for(unsigned int index = 0;check_prog(index,num_gradient);++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        if(index == 0)
            get_report_from_dicom(dicom_header,new_file->report);
        new_file->image.resize(tipl::geometry<3>(buf_image.width(),buf_image.height(),slice_num));

        for(unsigned int j = 0;j < slice_num;++j)
        std::copy(buf_image.begin()+(j*num_gradient + index)*plane_size,
                  buf_image.begin()+(j*num_gradient + index+1)*plane_size,
                  new_file->image.begin()+j*plane_size);
        new_file->file_name = file_name;
        std::ostringstream out;
        out << index;
        new_file->file_name += out.str();
        dicom_header.get_voxel_size(new_file->voxel_size);
        new_file->bvalue = b_table[index*4];
        new_file->bvec = tipl::vector<3, float>(b_table[index*4+1],b_table[index*4+2],b_table[index*4+3]);
        dwi_files.push_back(new_file);
    }
    return true;
}


void load_bvec(const char* file_name,std::vector<double>& b_table)
{
    std::ifstream in(file_name);
    if(!in)
        return;
    std::string line;
    unsigned int total_line = 0;
    while(std::getline(in,line))
    {
        std::istringstream read_line(line);
        std::copy(std::istream_iterator<double>(read_line),
                  std::istream_iterator<double>(),
                  std::back_inserter(b_table));
        ++total_line;
    }
    if(total_line == 3)
        tipl::mat::transpose(b_table.begin(),tipl::dyndim(3,b_table.size()/3));
}
void load_bval(const char* file_name,std::vector<double>& bval)
{
    std::ifstream in(file_name);
    if(!in)
        return;
    std::copy(std::istream_iterator<double>(in),
              std::istream_iterator<double>(),
              std::back_inserter(bval));
}

bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec)
{
    std::vector<QString> bval_name(4),bvec_name(4);
    QString path = QFileInfo(file_name).absolutePath() + "/";
    bval_name[0] = path + QFileInfo(file_name).baseName() + ".bvals";
    bval_name[1] = path + QFileInfo(file_name).baseName() + ".bval";
    bval_name[2] = path + QFileInfo(file_name).completeBaseName() + ".bvals";
    bval_name[3] = path + QFileInfo(file_name).completeBaseName() + ".bval";
    bvec_name[0] = path + QFileInfo(file_name).baseName() + ".bvecs";
    bvec_name[1] = path + QFileInfo(file_name).baseName() + ".bvec";
    bvec_name[2] = path + QFileInfo(file_name).completeBaseName() + ".bvecs";
    bvec_name[3] = path + QFileInfo(file_name).completeBaseName() + ".bvec";

    if(QFileInfo(file_name).completeBaseName() == "data.nii")
    {
        bval_name.push_back(path + "bvals");
        bval_name.push_back(path + "bval");
        bvec_name.push_back(path + "bvecs");
        bvec_name.push_back(path + "bvec");
    }
    for(int i = bval_name.size()-1;i >= 0;--i)
        bval_name.push_back(bval_name[i] + ".txt");
    for(int i = bvec_name.size()-1;i >= 0;--i)
        bvec_name.push_back(bvec_name[i] + ".txt");

    for(int i = 0;i < bval_name.size();++i)
        if(QFileInfo(bval_name[i]).exists())
        {
            bval = bval_name[i];
            break;
        }
    for(int i = 0;i < bvec_name.size();++i)
        if(QFileInfo(bvec_name[i]).exists())
        {
            bvec = bvec_name[i];
            break;
        }
    return QFileInfo(bval).exists() && QFileInfo(bvec).exists();
}

bool load_4d_nii(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    gz_nifti analyze_header;
    std::cout << "loading 4d nifti" << std::endl;
    if(!analyze_header.load_from_file(file_name))
    {
        std::cout << analyze_header.error << std::endl;
        return false;
    }
    if(analyze_header.dim(4) <= 1)
        return false;
    std::vector<double> bvals,bvecs;
    {
        QString bval_name,bvec_name;
        if(find_bval_bvec(file_name,bval_name,bvec_name))
        {
            load_bval(bval_name.toLocal8Bit().begin(),bvals);
            load_bvec(bvec_name.toLocal8Bit().begin(),bvecs);
            if(!bval_name.isEmpty() && analyze_header.dim(4) != bvals.size())
            {
                bvals.clear();
                bvecs.clear();
                std::cout << "The b-table " << bval_name.toStdString() << " does not match the DWI: there are "
                          << analyze_header.dim(4)
                          << " DWI in the nifti file, but the b-table has "
                          << bvals.size() << " entries." << std::endl;
            }
            if(bvals.size()*3 != bvecs.size())
            {
                bvals.clear();
                bvecs.clear();
                std::cout << "The b-table " << bval_name.toStdString() << " and " << bvec_name.toStdString() << " do not match each other" << std::endl;
            }
        }
    }
    tipl::image<float,4> grad_dev;
    tipl::image<unsigned char,3> mask;
    if(QFileInfo(QFileInfo(file_name).absolutePath() + "/grad_dev.nii.gz").exists())
    {
        gz_nifti grad_header;
        if(grad_header.load_from_file(QString(QFileInfo(file_name).absolutePath() + "/grad_dev.nii.gz").toLocal8Bit().begin()))
        {
            grad_header.toLPS(grad_dev);
            std::cout << "grad_dev used" << std::endl;
        }
    }
    if(QFileInfo(QFileInfo(file_name).absolutePath() + "/nodif_brain_mask.nii.gz").exists())
    {
        gz_nifti mask_header;
        if(mask_header.load_from_file(QString(QFileInfo(file_name).absolutePath() + "/nodif_brain_mask.nii.gz").toLocal8Bit().begin()))
        {
            mask_header.toLPS(mask);
            std::cout << "mask used" << std::endl;
        }
    }
    // check data range
    tipl::vector<3,float> vs;
    float max_value = 0.0;
    float m = float(std::numeric_limits<unsigned short>::max()-1);
    for(unsigned int index = 0;index < analyze_header.dim(4);++index)
    {
        std::auto_ptr<DwiHeader> new_file(new DwiHeader);
        tipl::image<float,3> data;
        if(!analyze_header.toLPS(data,index == 0))
            break;
        for(size_t i = 0;i < data.size();++i)
            if(std::isnan(data[i]) || std::isinf(data[i]))
            {
                std::cout << "NAN or INF found in the data" << std::endl;
                return false;
            }
        max_value = std::max<float>(max_value,*std::max_element(data.begin(),data.end()));
    }
    analyze_header.get_voxel_size(vs);
    if(!analyze_header.load_from_file(file_name))
        return false;
    {
        for(unsigned int index = 0;index < analyze_header.dim(4);++index)
        {
            std::shared_ptr<DwiHeader> new_file(new DwiHeader);
            tipl::image<float,3> data;
            if(!analyze_header.toLPS(data,false))
                break;
            tipl::lower_threshold(data,0.0);
            if(max_value > m)
            {
                data *= m/max_value;
                tipl::upper_threshold(data,m);
            }
            new_file->image = data;
            new_file->file_name = file_name;
            std::ostringstream out;
            out << index;
            new_file->file_name += out.str();
            new_file->voxel_size = vs;
            if(!bvals.empty())
            {
                new_file->bvalue = bvals[index];
                new_file->bvec[0] = bvecs[index*3];
                new_file->bvec[1] = bvecs[index*3+1];
                new_file->bvec[2] = bvecs[index*3+2];
                new_file->bvec.normalize();
                if(new_file->bvalue < 10)
                {
                    new_file->bvalue = 0;
                    new_file->bvec = tipl::vector<3>(0,0,0);
                }
            }
            if(index == 0 && !grad_dev.empty())
                new_file->grad_dev.swap(grad_dev);
            if(index == 0 && !mask.empty())
                new_file->mask.swap(mask);
            dwi_files.push_back(new_file);
        }
    }

    return true;
}

bool load_4d_2dseq(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    tipl::io::bruker_2dseq bruker_header;
    if(!bruker_header.load_from_file(file_name))
        return false;
    tipl::vector<3,float> vs;
    std::vector<float> bvalues;
    std::vector<tipl::vector<3> > bvecs;
    std::string report;
    bruker_header.get_voxel_size(vs);

    QString system_path =QFileInfo(QFileInfo(QFileInfo(file_name).absolutePath()).absolutePath()).absolutePath();
    if(QFileInfo(system_path+"/method").exists())
    {
        tipl::io::bruker_info method_file;
        QString method_name = system_path+"/method";
        if(!method_file.load_from_file(method_name.toLocal8Bit().begin()))
        {
            QMessageBox::information(0,"error",QString("Cannot find method file at ") + method_name,0);
            return false;
        }        
        if(!method_file.read("PVM_DwEffBval",bvalues))
        {
            QMessageBox::information(0,"error",QString("Cannot find PVM_DwEffBval in method file at ") + method_name,0);
            return false;
        }
        std::vector<float> bvec_temp;
        if(!method_file.read("PVM_DwGradVec",bvec_temp))
        {
            QMessageBox::information(0,"error",QString("Cannot find PVM_DwGradVec in method file at ") + method_name,0);
            return false;
        }

        bvecs.resize(bvalues.size());
        for (unsigned int index = 0,pos = 0;index < bvalues.size();++index,pos += 3)
        {
            bvecs[index][0] = bvec_temp[pos];
            bvecs[index][1] = bvec_temp[pos+1];
            bvecs[index][2] = bvec_temp[pos+2];
            bvecs[index].normalize();
        }
        float slice_thickness;
        std::istringstream(method_file["PVM_SliceThick"]) >> slice_thickness;
        if(slice_thickness != 0.0f)
            vs[2] = slice_thickness;
        get_report_from_bruker(method_file,report);
    }
    else
    {
        if(QFileInfo(system_path+"/imnd").exists())
        {
            bvalues.push_back(0);
            bvecs.push_back(tipl::vector<3>());

            tipl::io::bruker_info imnd_file;
            QString imnd_name = QFileInfo(QFileInfo(QFileInfo(file_name).
                    absolutePath()).absolutePath()).absolutePath()+"/imnd";
            if(!imnd_file.load_from_file(imnd_name.toLocal8Bit().begin()))
            {
                QMessageBox::information(0,"error",QString("Cannot find method or imnd file at ") + imnd_name,0);
                return false;
            }
            std::istringstream(imnd_file["IMND_diff_b_value"]) >> bvalues[0];
            std::istringstream(imnd_file["IMND_diff_grad_x"]) >> bvecs[0][0];
            std::istringstream(imnd_file["IMND_diff_grad_y"]) >> bvecs[0][1];
            std::istringstream(imnd_file["IMND_diff_grad_z"]) >> bvecs[0][2];
            std::istringstream(imnd_file["IMND_slice_thick"]) >> vs[2];
            get_report_from_bruker2(imnd_file,report);
        }
        else
            return false;
    }


    tipl::geometry<3> dim(bruker_header.get_image().geometry());
    dim[2] /= bvalues.size();

    if(dwi_files.size() && dwi_files.back()->image.geometry() != dim)
        return false;
    tipl::lower_threshold(bruker_header.get_image(),0.0);
    tipl::normalize(bruker_header.get_image(),32767.0);

    for (unsigned int index = 0;index < bvalues.size();++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        new_file->report = report;
        new_file->image.resize(dim);
        std::copy(bruker_header.get_image().begin()+index*new_file->image.size(),
                  bruker_header.get_image().begin()+(index+1)*new_file->image.size(),
                    new_file->image.begin());
        new_file->file_name = file_name;
        std::ostringstream out;
        out << index;
        new_file->file_name += out.str();
        new_file->voxel_size = vs;
        dwi_files.push_back(new_file);
        dwi_files.back()->bvalue = bvalues[index];
        dwi_files.back()->bvec = bvecs[index];
    }
    return true;
}

bool load_multiple_slice_dicom(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    tipl::io::dicom dicom_header;// multiple frame image
    tipl::geometry<3> geo;
    if(!dicom_header.load_from_file(file_list[0].toLocal8Bit().begin()))
        return false;
    dicom_header.get_image_dimension(geo);
    // philips or GE single slice images
    if(geo[2] != 1 || dicom_header.is_mosaic)
        return false;

    tipl::io::dicom dicom_header2;
    if(!dicom_header2.load_from_file(file_list[1].toLocal8Bit().begin()))
        return false;
    float s1 = dicom_header.get_slice_location();
    bool iterate_slice_first = true;
    unsigned int slice_num = 2;
    unsigned int b_num = 2;
    if(s1 == 0.0) // no slice locaton information
    {
        DwiHeader dwi1,dwi2;
        dwi1.open(file_list[0].toLocal8Bit().begin());
        dwi2.open(file_list[1].toLocal8Bit().begin());
        if(dwi1.bvec == dwi2.bvec && dwi1.bvalue == dwi2.bvalue) // iterater slice first
        {
            for (;slice_num < file_list.size();++slice_num)
            {
                DwiHeader dwi;
                if(!dwi.open(file_list[slice_num].toLocal8Bit().begin()))
                    return false;
                if(dwi1.bvec != dwi.bvec || dwi1.bvalue != dwi.bvalue)
                    break;
            }
            geo[2] = slice_num;
            iterate_slice_first = true;
        }
        else
        // iterate b first
        {
            for (;b_num < file_list.size();++b_num)
            {
                DwiHeader dwi;
                if(!dwi.open(file_list[b_num].toLocal8Bit().begin()))
                    return false;
                if(dwi1.bvec == dwi.bvec && dwi1.bvalue == dwi.bvalue)
                    break;
            }
            geo[2] = file_list.size()/b_num;
            iterate_slice_first = false;
        }
    }
    else
    {
        if(s1 == dicom_header2.get_slice_location()) // iterater b-value first
        {
            for (;b_num < file_list.size();++b_num)
            {
                if(!dicom_header2.load_from_file(file_list[b_num].toLocal8Bit().begin()))
                    return false;
                if(dicom_header2.get_slice_location() != s1)
                    break;
            }
            geo[2] = std::ceil((float)file_list.size()/(float)b_num);
            iterate_slice_first = false;
        }
        else
        // iterater slice first
        {
            for (;slice_num < file_list.size();++slice_num)
            {
                if(!dicom_header2.load_from_file(file_list[slice_num].toLocal8Bit().begin()))
                    return false;
                if(dicom_header2.get_slice_location() == s1)
                    break;
            }
            geo[2] = slice_num;
            iterate_slice_first = true;
        }
    }


    begin_prog("loading images");
    for (unsigned int index = 0,b_index = 0,slice_index = 0;check_prog(index,file_list.size());++index)
    {
        std::shared_ptr<DwiHeader> dwi(new DwiHeader);
        if(!dwi->open(file_list[index].toLocal8Bit().begin()))
            return false;
        if(slice_index == 0)
        {
            dwi_files.push_back(dwi);
            dwi_files.back()->file_name = file_list[index].toLocal8Bit().begin();
            dwi_files.back()->image.resize(geo);
            dicom_header.get_voxel_size(dwi_files.back()->voxel_size);
        }
        else
            std::copy(dwi->image.begin(),dwi->image.end(),
                dwi_files[b_index]->image.begin() + slice_index*geo.plane_size());
        if(iterate_slice_first)
        {
            ++slice_index;
            if(slice_index >= slice_num)
            {
                slice_index = 0;
                ++b_index;
            }
        }
        else
        {
            ++b_index;
            if(b_index >= b_num)
            {
                b_index = 0;
                ++slice_index;
            }
        }
    }
    return true;
}
void scale_image_buf_to_uint16(std::vector<tipl::image<float,3> >& image_buf)
{
    float max_value = 0.0f;
    for(size_t i = 0;i < image_buf.size();++i)
        max_value = std::max<float>(max_value,tipl::maximum(image_buf[i]));
    tipl::par_for(image_buf.size(),[&](int i)
    {
        tipl::multiply_constant(image_buf[i],float(std::numeric_limits<unsigned short>::max()-1)/max_value);
        tipl::lower_threshold(image_buf[i],0.0f);
    });
}
bool load_nhdr(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    std::vector<tipl::image<float,3> > image_buf;
    tipl::geometry<3> dim;
    tipl::vector<3> vs;
    image_buf.resize(file_list.size());
    begin_prog("Reading raw data");
    for (size_t i = 0;check_prog(i,file_list.size());++i)
    {
        std::map<std::string,std::string> value_list;
        {
            std::ifstream in(file_list[i].toStdString().c_str());
            std::string line;
            while(std::getline(in,line))
            {
                std::string::size_type pos = 0;
                if(line.empty() || line[0] == '#' || (pos = line.find(':')) == std::string::npos)
                    continue;
                std::istringstream read_line(line);
                std::string name,value;
                name = line.substr(0,pos);
                value = line.substr(line[pos+1] == ' ' ? pos+2:pos+1);
                std::replace(value.begin(),value.end(),',',' ');
                value.erase(std::remove(value.begin(),value.end(),'('),value.end());
                value.erase(std::remove(value.begin(),value.end(),')'),value.end());
                value_list[name] = value;
            }
        }
        if(value_list["type"].find("float") == std::string::npos)
            return false;
        // allocate all space
        if(i == 0)
        {
            float d;
            if(!(std::istringstream(value_list["sizes"]) >> dim[0] >> dim[1] >> dim[2])||
               !(std::istringstream(value_list["space directions"]) >> vs[0] >> d >> d >> d >> vs[1] >> d >> d >> d >> vs[2]))
                return false;
        }

        try{
            image_buf[i].resize(dim);
        }
        catch(...)
        {
            return false;
        }
        std::string raw_file_name = file_list[i].toStdString();
        raw_file_name = raw_file_name.substr(0,raw_file_name.length()-4);
        raw_file_name += "raw";
        std::ifstream in(raw_file_name,std::ifstream::binary);
        std::cout << "reading" << raw_file_name << std::endl;
        if(!in.read((char*)&image_buf[i][0],image_buf[i].size()*sizeof(float)))
            return false;
    }

    scale_image_buf_to_uint16(image_buf);

    begin_prog("Converting data");
    for(size_t i = 0;check_prog(i,image_buf.size());++i)
    {
        dwi_files.push_back(std::make_shared<DwiHeader>());
        dwi_files.back()->voxel_size = vs;
        dwi_files.back()->file_name = file_list[i].toStdString();
        dwi_files.back()->report = " The diffusion images were acquired on an Agilent scanner.";
        dwi_files.back()->image = image_buf[i];
        image_buf[i] = tipl::image<float,3>();
    }
    return true;
}
bool load_4d_fdf(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    std::vector<tipl::image<float,3> > image_buf;
    int plane_size = 0;
    for (unsigned int index = 0;check_prog(index,file_list.size());++index)
    {
        std::map<std::string,std::string> value_list;
        {
            std::ifstream in(file_list[index].toLocal8Bit().begin());
            std::string line;
            while(std::getline(in,line))
            {
                std::string::size_type pos = 0;
                if(line.empty() || line[0] == '#' || (pos = line.find('=')) == std::string::npos)
                    continue;
                std::istringstream read_line(line);
                std::string s1,s2,value;
                read_line >> s1 >> s2;
                value = line.substr(pos+2,line.length()-pos-3);
                std::replace(value.begin(),value.end(),',',' ');
                std::replace(value.begin(),value.end(),'"',' ');
                std::replace(value.begin(),value.end(),'{',' ');
                std::replace(value.begin(),value.end(),'}',' ');
                value_list[s2] = value;
                if(s2 == "checksum")
                    break;
            }
        }
        if(value_list["*storage"] != " float ")
            return false;
        // allocate all space
        if(index == 0)
        {
            float dwi_num,width,height,depth,fov1,fov2,fov3;
            if(!(std::istringstream(value_list["array_dim"]) >> dwi_num) ||
               !(std::istringstream(value_list["matrix[]"]) >> width >> height ) ||
               !(std::istringstream(value_list["slices"]) >> depth) ||
               !(std::istringstream(value_list["roi[]"]) >> fov1 >> fov2 >> fov3))
                return false;
            plane_size = width*height;
            image_buf.resize(dwi_num);
            for(unsigned int i = 0;i < dwi_num;++i)
            {
                image_buf[i].resize(tipl::geometry<3>(width,height,depth));
                dwi_files.push_back(std::make_shared<DwiHeader>());
                dwi_files.back()->image.resize(tipl::geometry<3>(width,height,depth));
                dwi_files.back()->voxel_size[0] = fov1*10.0/width;
                dwi_files.back()->voxel_size[1] = fov2*10.0/height;
                dwi_files.back()->voxel_size[2] = fov3*100.0/depth;
                dwi_files.back()->file_name = value_list["*studyid"];
                //dwi_files
            }
            if(dwi_files.empty())
                return false;
            dwi_files.back()->report = " The diffusion images were acquired on a Varian scanner.";
        }
        // get DWI and slice location
        int dwi_id,slice_id;
        {
            float v1,v2;
            if(!(std::istringstream(value_list["array_index"]) >> v1) ||
               !(std::istringstream(value_list["slice_no"]) >> v2))
                return false;
            dwi_id = v1 - 1.0;
            slice_id = v2 - 1.0;
            if(dwi_id < 0 || dwi_id >= dwi_files.size() || slice_id < 0 || slice_id >= dwi_files.front()->image.depth())
                return 0;
        }
        // get b_value
        if(slice_id == 0)
        {
            if(!(std::istringstream(value_list["dro"]) >> dwi_files[dwi_id]->bvec[0]) ||
               !(std::istringstream(value_list["dpe"]) >> dwi_files[dwi_id]->bvec[1]) ||
               !(std::istringstream(value_list["dsl"]) >> dwi_files[dwi_id]->bvec[2]) ||
               !(std::istringstream(value_list["bvalue"]) >> dwi_files[dwi_id]->bvalue))
                return false;
            dwi_files[dwi_id]->bvec.normalize();
        }
        std::ifstream in(file_list[index].toLocal8Bit().begin(),std::ifstream::binary);
        in.seekg(-(int)plane_size*4,std::ios_base::end);
        if(!in.read((char*)&*(image_buf[dwi_id].begin() + slice_id*plane_size),plane_size*4))
            return false;
    }


    scale_image_buf_to_uint16(image_buf);
    for(int i = 0;i < image_buf.size();++i)
        dwi_files[i]->image = image_buf[i];
    return true;
}

bool load_3d_series(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    begin_prog("loading images");
    for (unsigned int index = 0;check_prog(index,file_list.size());++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        if (!new_file->open(file_list[index].toLocal8Bit().begin()))
            continue;
        new_file->file_name = file_list[index].toLocal8Bit().begin();
        dwi_files.push_back(new_file);
    }
    return !dwi_files.empty();
}

bool load_all_files(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    if(QFileInfo(file_list[0]).fileName() == "2dseq")
    {
        for(unsigned int index = 0;index < file_list.size();++index)
            load_4d_2dseq(file_list[index].toLocal8Bit().begin(),dwi_files);
        return !dwi_files.empty();
    }
    if(QFileInfo(file_list[0]).suffix() == "fdf")
    {
        return load_4d_fdf(file_list,dwi_files);
    }
    if(QFileInfo(file_list[0]).suffix() == "nhdr")
    {
        return load_nhdr(file_list,dwi_files);
    }
    if(file_list.size() == 1 && QFileInfo(file_list[0]).isDir()) // single folder with DICOM files
    {
        QDir cur_dir = file_list[0];
        QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        if(dicom_file_list.empty())
            return false;
        for (unsigned int index = 0;index < dicom_file_list.size();++index)
            dicom_file_list[index] = file_list[0] + "/" + dicom_file_list[index];
        return load_all_files(dicom_file_list,dwi_files);
    }

    //Combine 2dseq folders
    if(file_list.size() > 1 && QFileInfo(file_list[0]).isDir() &&
            QFileInfo(file_list[0]+"/pdata/1/2dseq").exists())
    {
        for(unsigned int index = 0;index < file_list.size();++index)
            load_all_files(QStringList() << (file_list[index]+"/pdata/1/2dseq"),dwi_files);
        return !dwi_files.empty();
    }


    begin_prog("loading");
    if (file_list.size() <= 3) // 4d image
    {
        for(unsigned int i = 0;i < file_list.size();++i)
        if(!load_dicom_multi_frame(file_list[i].toLocal8Bit().begin(),dwi_files) &&
           !load_4d_nii(file_list[i].toLocal8Bit().begin(),dwi_files))
            return false;
        return !dwi_files.empty();
    }

    std::sort(file_list.begin(),file_list.end(),compare_qstring());
    if(load_multiple_slice_dicom(file_list,dwi_files))
        return !dwi_files.empty();

    if(load_3d_series(file_list,dwi_files))
        return !dwi_files.empty();

    // multiple 4d nii
    for(unsigned int index = 0;index < file_list.size();++index)
    {
        if(!load_dicom_multi_frame(file_list[index].toLocal8Bit().begin(),dwi_files) &&
           !load_4d_nii(file_list[index].toLocal8Bit().begin(),dwi_files))
            return false;
    }
    return !dwi_files.empty();
}
void dicom_parser::load_table(void)
{
    unsigned int last_index = ui->tableWidget->rowCount();
    ui->tableWidget->setRowCount(dwi_files.size());
    double max_b = 0;
    for(unsigned int index = last_index;index < dwi_files.size();++index)
    {
        if(dwi_files[index]->bvalue < 100)
            dwi_files[index]->bvalue = 0.0f;
        ui->tableWidget->setItem(index, 0, new QTableWidgetItem(QFileInfo(dwi_files[index]->file_name.data()).fileName()));
        ui->tableWidget->setItem(index, 1, new QTableWidgetItem(QString::number(dwi_files[index]->bvalue)));
        ui->tableWidget->setItem(index, 2, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[0])));
        ui->tableWidget->setItem(index, 3, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[1])));
        ui->tableWidget->setItem(index, 4, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[2])));
        max_b = std::max(max_b,(double)dwi_files[index]->bvalue);
    }
    if(max_b == 0.0)
        QMessageBox::information(this,"DSI Studio","Cannot find b-table from the header. You may need to load an external b-table",0);
}

void dicom_parser::load_files(QStringList file_list)
{
    if(!load_all_files(file_list,dwi_files))
    {
        QMessageBox::information(this,"Error","Invalid file format",0);
        close();
        return;
    }
    if(prog_aborted())
    {
        close();
        return;
    }
    if(dwi_files.size() > ui->tableWidget->rowCount())
        load_table();
}

void dicom_parser::on_buttonBox_accepted()
{
    if (dwi_files.empty())
        return;

    // save b table info to dwi header
    for (unsigned int index = 0;index < dwi_files.size();++index)
    {
        if(QString::number(dwi_files[index]->bvalue) != ui->tableWidget->item(index,1)->text())
            dwi_files[index]->bvalue = ui->tableWidget->item(index,1)->text().toFloat();
        if(QString::number(dwi_files[index]->bvec[0]) != ui->tableWidget->item(index,2)->text() ||
           QString::number(dwi_files[index]->bvec[1]) != ui->tableWidget->item(index,3)->text() ||
           QString::number(dwi_files[index]->bvec[2]) != ui->tableWidget->item(index,4)->text())
            dwi_files[index]->bvec = tipl::vector<3>(
                    ui->tableWidget->item(index,2)->text().toFloat(),
                    ui->tableWidget->item(index,3)->text().toFloat(),
                    ui->tableWidget->item(index,4)->text().toFloat());
    }

    DwiHeader::output_src(ui->SrcName->text().toLocal8Bit().begin(),
                          dwi_files,
                          ui->upsampling->currentIndex(),
                          ui->sort_btable->isChecked());

    dwi_files.clear();
    if(QFileInfo(ui->SrcName->text()).suffix() != "gz")
        ((MainWindow*)parent())->addSrc(ui->SrcName->text()+".gz");
    else
        ((MainWindow*)parent())->addSrc(ui->SrcName->text());
    QMessageBox::information(this,"DSI Studio","SRC file created",0);
    close();
}
void dicom_parser::on_buttonBox_rejected()
{
    close();
}

void dicom_parser::on_pushButton_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
            this,"Save file",
            ui->SrcName->text(),
            "Src files (*src.gz *.src);;All files (*)" );
    if(filename.isEmpty())
        return;
    ui->SrcName->setText(filename);
}

void dicom_parser::on_upperDir_clicked()
{
    QDir path(QFileInfo(ui->SrcName->text()).absolutePath());
    path.cdUp();
    ui->SrcName->setText(path.absolutePath() + "/" +
                         QFileInfo(ui->SrcName->text()).fileName());
}

void dicom_parser::update_b_table(void)
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
    {
        ui->tableWidget->item(index,2)->setText(QString::number(dwi_files[index]->bvec[0]));
        ui->tableWidget->item(index,3)->setText(QString::number(dwi_files[index]->bvec[1]));
        ui->tableWidget->item(index,4)->setText(QString::number(dwi_files[index]->bvec[2]));
    }
}

void dicom_parser::on_actionOpen_Images_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
            this,"Open Images files",cur_path,
            "Images (*.dcm *.hdr *.nii *nii.gz 2dseq);;All files (*)" );
    if( filenames.isEmpty() )
        return;
    load_files(filenames);
}

void dicom_parser::on_actionOpen_b_table_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open b-table",
            QFileInfo(ui->SrcName->text()).absolutePath(),
            "Text files (*.txt);;All files (*)" );

    std::ifstream in(filename.toLocal8Bit().begin());
    if(!in)
        return;
    std::string line;
    std::vector<double> b_table;
    while(std::getline(in,line))
    {
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream read_line(line);
        std::copy(std::istream_iterator<double>(read_line),
                  std::istream_iterator<double>(),
                  std::back_inserter(b_table));
    }
    if(b_table.size() > 4 && ui->tableWidget->rowCount() == 1 &&
       dwi_files[0]->image.depth()%(b_table.size()/4) == 0) // 4D as 3D condition
    {
        auto& I = dwi_files[0]->image;
        unsigned int b_count = b_table.size()/4;
        tipl::geometry<3> dim(I.width(),I.height(),I.depth()/b_count);
        std::vector<std::shared_ptr<DwiHeader> > new_files;
        unsigned int plane_size = I.plane_size();
        for(int i = 0;i < b_count;++i)
        {
            std::shared_ptr<DwiHeader> new_file(new DwiHeader);
            new_file->voxel_size = dwi_files[0]->voxel_size;
            new_file->bvalue = b_table[i*4];
            new_file->bvec[0] = b_table[i*4+1];
            new_file->bvec[1] = b_table[i*4+2];
            new_file->bvec[2] = b_table[i*4+3];
            new_file->image.resize(dim);
            for(int j = 0;j < dim.depth();++j)
            {
                unsigned int slice_pos = i+j*b_count;
                std::copy(I.slice_at(slice_pos).begin(),I.slice_at(slice_pos).begin()+plane_size,
                          new_file->image.slice_at(j).begin());
            }
            new_files.push_back(new_file);
        }
        dwi_files.swap(new_files);
        ui->tableWidget->setRowCount(0);
        load_table();
    }
    // handle per slice b_table
    if(b_table.size()/4 != ui->tableWidget->rowCount() && (b_table.size()/4)%ui->tableWidget->rowCount() == 0)
    {
        unsigned int slice_num = (b_table.size()/4)/ui->tableWidget->rowCount();
        for(unsigned int i = 0;i < ui->tableWidget->rowCount();++i)
        {
            b_table[i*4] = b_table[i*4*slice_num];
            b_table[i*4+1] = b_table[i*4*slice_num+1];
            b_table[i*4+2] = b_table[i*4*slice_num+2];
            b_table[i*4+3] = b_table[i*4*slice_num+3];
        }
        b_table.resize(ui->tableWidget->rowCount()*4);
    }
    for (unsigned int index = 0,b_index = 0;index < ui->tableWidget->rowCount();++index)
    {
        for(unsigned int j = 0;j < 4 && b_index < b_table.size();++j,++b_index)
            ui->tableWidget->item(index,j+1)->setText(QString::number(b_table[b_index]));
    }
}

void dicom_parser::on_actionOpen_bval_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open bval",
            QFileInfo(ui->SrcName->text()).absolutePath(),
            "All files (*)" );
    if(filename.isEmpty())
        return;
    std::vector<double> bval;
    load_bval(filename.toLocal8Bit().begin(),bval);
    if(bval.empty())
        return;
    for (int index = ui->tableWidget->rowCount()-1,
             index2 = bval.size()-1;index >= 0 && index2 >= 0;--index,--index2)
        ui->tableWidget->item(index,1)->setText(QString::number(bval[index2]));
}

void dicom_parser::on_actionOpen_bvec_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open bvec",
            QFileInfo(ui->SrcName->text()).absolutePath(),
            "All files (*)" );
    if(filename.isEmpty())
        return;
    std::vector<double> b_table;
    load_bvec(filename.toLocal8Bit().begin(),b_table);
    if(b_table.empty())
        return;
    for (int index = ui->tableWidget->rowCount()-1,
             b_index = b_table.size()-1;index >=0 && b_index >=0 ;--index)
    {
        for(int j = 2;j >= 0 && b_index >=0;--j,--b_index)
            ui->tableWidget->item(index,j+2)->setText(QString::number(b_table[b_index]));
    }
}

void dicom_parser::on_actionSave_b_table_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save b-table",
            QFileInfo(ui->SrcName->text()).absolutePath() + "/b_table.txt",
            "Text files (*.txt);;All files (*)");

    std::ofstream btable(filename.toLocal8Bit().begin());
    if(!btable)
        return;
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
    {
        for(unsigned int j = 0;j < 4;++j)
            btable << ui->tableWidget->item(index,j+1)->text().toDouble() << "\t";
        btable<< std::endl;
    }
}

void dicom_parser::on_actionFlip_bx_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
        ui->tableWidget->item(index,2)->setText(QString::number(-(ui->tableWidget->item(index,2)->text().toDouble())));
}

void dicom_parser::on_actionFlip_by_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
        ui->tableWidget->item(index,3)->setText(QString::number(-(ui->tableWidget->item(index,3)->text().toDouble())));
}

void dicom_parser::on_actionFlip_bz_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
        ui->tableWidget->item(index,4)->setText(QString::number(-(ui->tableWidget->item(index,4)->text().toDouble())));
}

void dicom_parser::on_actionSwap_bx_by_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
    {
        QString temp = ui->tableWidget->item(index,3)->text();
        ui->tableWidget->item(index,3)->setText(ui->tableWidget->item(index,2)->text());
        ui->tableWidget->item(index,2)->setText(temp);
    }
}

void dicom_parser::on_actionSwap_bx_bz_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
    {
        QString temp = ui->tableWidget->item(index,4)->text();
        ui->tableWidget->item(index,4)->setText(ui->tableWidget->item(index,2)->text());
        ui->tableWidget->item(index,2)->setText(temp);
    }
}

void dicom_parser::on_actionSwap_by_bz_triggered()
{
    for (unsigned int index = 0;index < ui->tableWidget->rowCount();++index)
    {
        QString temp = ui->tableWidget->item(index,4)->text();
        ui->tableWidget->item(index,4)->setText(ui->tableWidget->item(index,3)->text());
        ui->tableWidget->item(index,3)->setText(temp);
    }
}
