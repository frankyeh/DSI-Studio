#include <unordered_map>
#include <qmessagebox.h>
#include <QProgressDialog>
#include <QFileDialog>
#include <QSettings>
#include "dicom_parser.h"
#include "ui_dicom_parser.h"
#include "image_model.hpp"
#include "mainwindow.h"

void get_report_from_dicom(const tipl::io::dicom& header,std::string& report_);
void get_report_from_bruker(const tipl::io::bruker_info& header,std::string& report_);
void get_report_from_bruker2(const tipl::io::bruker_info& header,std::string& report_);

QString get_dicom_output_name(QString file_name,QString file_extension, bool add_path)
{
    tipl::io::dicom header;
    if (header.load_from_file(file_name.toStdString().c_str()))
    {
        std::string Person;
        header.get_patient(Person);
        std::string seq_num;
        header.get_sequence_num(seq_num);
        if(add_path)
        {
            QDir dir = QFileInfo(file_name).absoluteDir();
            dir.cdUp();
            return dir.absolutePath() + "/" + Person.c_str() + "_s" + seq_num.c_str() + file_extension;
        }
        else
        {
            return QString(Person.c_str()) + "_s" + seq_num.c_str() + file_extension;
        }
    }
    else
    {
        if(tipl::ends_with(file_name.toStdString(),".nii.gz"))
            return file_name.remove(".nii.gz")+file_extension;
        else
            return file_name+file_extension;
    }
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
        ui->SrcName->setText(get_dicom_output_name(file_list[0],".sz",true));
        tipl::io::dicom header;
        if (header.load_from_file(file_list[0].toStdString().c_str()))
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
    tipl::image<3> buf_image;
    dicom_header >> buf_image;
    unsigned int slice_num = dicom_header.get_int(0x2001,0x1018);
    std::vector<float> b_table;
    {
        std::vector<float> b,bx,by,bz;
        if(dicom_header.get_frame_values(0x2001,0x1003,b) ||
           dicom_header.get_frame_values(0x0018,0x9087,b))
        {
            if(!slice_num)
                slice_num = 1;
            uint32_t b_count = uint32_t(buf_image.depth())/slice_num;
            b.resize(b_count);
            if(dicom_header.get_frame_values(0x2005,0x10B0,bx) &&
               dicom_header.get_frame_values(0x2005,0x10B1,by) &&
               dicom_header.get_frame_values(0x2005,0x10B2,bz))
            {
                bx.resize(b_count);
                by.resize(b_count);
                bz.resize(b_count);
            }
            else
            {
                bx.resize(b_count);
                by.resize(b_count);
                bz.resize(b_count);
                std::vector<std::vector<float> > bvec;
                dicom_header.get_frame_values(0x0018,0x9089,bvec);
                for(size_t i = 0,j = 0;i < b_count && j < bvec.size();++i)
                {
                    if(b[i] != 0.0f)
                    {
                        bx[i] = bvec[j][0];
                        by[i] = bvec[j][1];
                        bz[i] = bvec[j][2];
                        ++j;
                    }
                }
            }

            for(size_t i = 0;i < b_count;++i)
            {
                tipl::vector<3> bvec(bx[i],by[i],bz[i]);
                b_table.push_back(b[i]*float(bvec.length()));
                if(bvec.length() > 0)
                    bvec.normalize();
                // Philips bx and by is swapped
                if(b_count > 1)
                {
                    b_table.push_back(bvec[1]);
                    b_table.push_back(bvec[0]);
                    b_table.push_back(bvec[2]);
                }
            }
        }
    }

    if(b_table.empty())
    {
        for(size_t i = 0;i < dicom_header.data.size();++i)
            if(int(dicom_header.data[i].sq_data.size()) == buf_image.depth())
            {
                std::string slice_pos;
                for(int j = 0;j < buf_image.depth();++j)
                {
                    const auto& ge = dicom_header.data[i].sq_data[uint32_t(j)];
                    std::string pos;
                    if(!tipl::io::dicom::get_value(ge,0x0020,0x0032,
                                                    j ? pos : slice_pos))
                        break;
                    if(j && pos != slice_pos)
                    {
                        slice_num = uint32_t(buf_image.depth()/j);
                        break;
                    }
                    float b_value = 0;
                    tipl::io::dicom::get_value(ge,0x0018,0x9087,b_value);
                    b_table.push_back(b_value);
                    if(b_value == 0.0f || !tipl::io::dicom::get_value(ge,0x0018,0x9089,b_table))
                    {
                        b_table.push_back(0.0f);
                        b_table.push_back(0.0f);
                        b_table.push_back(0.0f);
                    }
                }
                if(slice_num == 0)
                    continue;
                break;
            }
    }

    if(b_table.empty())
        return false;

    // multiframe DICOM
    // The b-table should be multiplied with the image rotation matrix
    tipl::matrix<3,3,float> T;
    {
        tipl::vector<3> x,y,z;
        if(dicom_header.get_image_row_orientation(x.begin()) &&
           dicom_header.get_image_col_orientation(y.begin()))
        {
            z = x.cross_product(y);
            std::copy(x.begin(),x.end(),T.begin());
            std::copy(y.begin(),y.end(),T.begin()+3);
            std::copy(z.begin(),z.end(),T.begin()+6);
        }
        else
            T.identity();
    }

    if(slice_num <= 1)
        slice_num = buf_image.depth();
    size_t num_gradient = uint32_t(buf_image.depth())/slice_num;


    size_t plane_size = size_t(buf_image.width()*buf_image.height());
    b_table.resize(num_gradient*4);
    for(size_t index = 0;index < num_gradient;++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        if(index == 0)
            get_report_from_dicom(dicom_header,new_file->report);
        new_file->image.resize(tipl::shape<3>(uint32_t(buf_image.width()),
                                                 uint32_t(buf_image.height()),slice_num));

        for(size_t j = 0;j < slice_num;++j)
        std::copy_n(buf_image.begin()+int64_t((j*num_gradient + index)*plane_size),plane_size,
                  new_file->image.begin()+int64_t(j*plane_size));
        new_file->file_name = file_name;
        std::ostringstream out;
        out << index;
        new_file->file_name += out.str();
        dicom_header.get_voxel_size(new_file->voxel_size);
        new_file->bvalue = b_table[index*4];
        new_file->bvec = tipl::vector<3, float>(b_table[index*4+1],b_table[index*4+2],b_table[index*4+3]);
        new_file->bvec.rotate(T);
        dwi_files.push_back(new_file);
    }
    return true;
}


bool load_bvec(const std::string& file_name,std::vector<double>& b_table_,bool flip_by = true)
{
    unsigned int total_line = 0;
    std::vector<double> b_table;
    for(const auto& line: tipl::read_text_file(file_name))
    {
        std::istringstream read_line(line);
        std::copy(std::istream_iterator<double>(read_line),
                  std::istream_iterator<double>(),
                  std::back_inserter(b_table));
        ++total_line;
    }
    if(total_line == 0)
        return false;
    if(total_line == 3)
        tipl::mat::transpose(b_table.begin(),tipl::shape<2>(3,b_table.size()/3));
    if(flip_by)
    {
        for(size_t index = 1;index < b_table.size();index += 3)
                b_table[index] = -b_table[index];
    }
    b_table_.insert(b_table_.end(),b_table.begin(),b_table.end());
    return true;
}
bool load_bval(const std::string& file_name,std::vector<double>& bval)
{
    std::ifstream in(file_name);
    if(!in)
        return false;
    std::copy(std::istream_iterator<double>(in),
              std::istream_iterator<double>(),
              std::back_inserter(bval));
    return true;
}
bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec)
{
    std::vector<QString> bval_name(6),bvec_name(6);
    QString path = QFileInfo(file_name).absolutePath() + "/";
    bval_name[0] = path + QFileInfo(file_name).baseName() + ".bvals";
    bval_name[1] = path + QFileInfo(file_name).baseName() + ".bval";
    bval_name[2] = path + QFileInfo(file_name).completeBaseName() + ".bvals";
    bval_name[3] = path + QFileInfo(file_name).completeBaseName() + ".bval";
    if(QString(file_name).endsWith(".nii.gz"))
    {
        bval_name[4] = QString(file_name).replace(".nii.gz",".bvals");
        bval_name[5] = QString(file_name).replace(".nii.gz",".bval");
    }


    bvec_name[0] = path + QFileInfo(file_name).baseName() + ".bvecs";
    bvec_name[1] = path + QFileInfo(file_name).baseName() + ".bvec";
    bvec_name[2] = path + QFileInfo(file_name).completeBaseName() + ".bvecs";
    bvec_name[3] = path + QFileInfo(file_name).completeBaseName() + ".bvec";
    if(QString(file_name).endsWith(".nii.gz"))
    {
        bvec_name[4] = QString(file_name).replace(".nii.gz",".bvecs");
        bvec_name[5] = QString(file_name).replace(".nii.gz",".bvec");
    }

    for(size_t i = 0;i < 6;++i)
    {
        bval_name.push_back(bval_name[i] + ".txt");
        bvec_name.push_back(bvec_name[i] + ".txt");
    }

    if(QFileInfo(file_name).completeBaseName() == "data.nii")
    {
        bval_name.push_back(path + "bvals");
        bval_name.push_back(path + "bval");
        bvec_name.push_back(path + "bvecs");
        bvec_name.push_back(path + "bvec");
    }

    for(size_t i = 0;i < bval_name.size();++i)
        if(QFileInfo(bval_name[i]).exists())
        {
            bval = bval_name[i];
            break;
        }
    for(size_t i = 0;i < bvec_name.size();++i)
        if(QFileInfo(bvec_name[i]).exists())
        {
            bvec = bvec_name[i];
            break;
        }
    return QFileInfo(bval).exists() && QFileInfo(bvec).exists();
}
bool get_bval_bvec(const std::string& bval_file,const std::string& bvec_file,size_t dwi_count,
                   std::vector<double>& bvals,std::vector<double>& bvecs,
                   std::string& error_msg);
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
bool load_4d_nii(const std::string& file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                 bool search_bvalbvec,
                 bool must_have_bval_bvec,
                 std::string& error_msg)
{
    tipl::progress prog("opening ",file_name);
    tipl::vector<3,float> vs;
    tipl::matrix<4,4,float> trans_to_mni;
    std::vector<tipl::image<3> > dwi_data;
    {
        tipl::io::gz_nifti nii;
        nii.input_stream->buffer_all = true;
        if(!nii.load_from_file(file_name))
        {
            error_msg = nii.error_msg;
            return false;
        }
        if(nii.dim(4) <= 1 && must_have_bval_bvec)
        {
            error_msg = "not a 4D nifti file";
            return false;
        }
        dwi_data.resize(nii.dim(4));
        // check data range
        for(unsigned int index = 0;index < nii.dim(4);++index)
        {
            tipl::image<3> data;
            if(!nii.toLPS(data))
                tipl::warning() << "failed to parse 4D NIFTI file at "
                                << std::to_string(index+1) << "/"
                                << std::to_string(nii.dim(4))
                                << " " <<  nii.error_msg;
            std::replace_if(data.begin(),data.end(),[](float v){return std::isnan(v) || std::isinf(v) || v < 0.0f;},0.0f);
            dwi_data[index].swap(data);
        }
        nii.get_voxel_size(vs);
        nii.get_image_transformation(trans_to_mni);

        if(dwi_data.size() <= 1 && must_have_bval_bvec)
        {
            error_msg = "not a 4D nifti file";
            return false;
        }
    }


    // if the imaging value is larger than 16-bit integer, then scale it.
    {
        float max_value = 0.0f;
        for(unsigned int index = 0;index < dwi_data.size();++index)
            max_value = std::max<float>(max_value,tipl::max_value(dwi_data[index]));
        if(max_value > float(std::numeric_limits<unsigned short>::max()-1))
        {
            float scale = float(std::numeric_limits<unsigned short>::max()-1)/max_value;
            tipl::adaptive_par_for(dwi_data.size(),[&](unsigned int index){
                dwi_data[index] *= scale;
            });
        }
        if(max_value < 256.0f && max_value != 0.0f)
        {
            tipl::out() << "The maximum singal is only " << max_value << std::endl;
            float scale = 1.0f;
            while(max_value*scale*32.0f < std::numeric_limits<unsigned short>::max())
                scale *= 32.0f;
            if(scale != 1.0f)
            {
                tipl::out() << "scaling the image by " << scale << std::endl;
                tipl::adaptive_par_for(dwi_data.size(),[&](unsigned int index){
                    dwi_data[index] *= scale;
                });
            }
        }
    }
    tipl::image<4,float> grad_dev;
    if(QFileInfo(QFileInfo(file_name.c_str()).absolutePath() + "/grad_dev.nii.gz").exists())
    {
        tipl::io::gz_nifti grad_header;
        if(grad_header.load_from_file(QString(QFileInfo(file_name.c_str()).absolutePath() + "/grad_dev.nii.gz").toStdString().c_str()))
        {
            grad_header.toLPS(grad_dev);
            tipl::out() << "grad_dev used" << std::endl;
        }
    }

    tipl::image<3,unsigned char> mask;
    if(QFileInfo(QFileInfo(file_name.c_str()).absolutePath() + "/nodif_brain_mask.nii.gz").exists())
    {
        tipl::io::gz_nifti mask_header;
        if(mask_header.load_from_file(QString(QFileInfo(file_name.c_str()).absolutePath() + "/nodif_brain_mask.nii.gz").toStdString().c_str()))
        {
            mask_header.toLPS(mask);
            tipl::out() << "mask used" << std::endl;
        }
    }

    std::vector<double> bvals,bvecs;
    QString bval_name,bvec_name;
    std::string bvalbvec_error_msg;
    if(search_bvalbvec && find_bval_bvec(file_name.c_str(),bval_name,bvec_name))
    {
        tipl::out() << "found bval and bvec file for " << file_name;
        tipl::out() << "bval: " << bval_name.toStdString();
        tipl::out() << "bvec: " << bvec_name.toStdString();
        if(!get_bval_bvec(bval_name.toStdString(),bvec_name.toStdString(),dwi_data.size(),
                          bvals,bvecs,bvalbvec_error_msg))
            tipl::out() << bvalbvec_error_msg;
    }
    if(must_have_bval_bvec && bvals.empty())
    {
        error_msg = bvalbvec_error_msg;
        return false;
    }

    for(unsigned int index = 0;index < dwi_data.size();++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        tipl::image<3> data;
        data.swap(dwi_data[index]);
        if(mask.size() == data.size())
            for(size_t i = 0;i < mask.size();++i)
                if(!mask[i])
                    data[i] = 0;
        new_file->image = data;
        new_file->file_name = file_name;
        new_file->voxel_size = vs;
        new_file->trans_to_mni = trans_to_mni;
        if(!bvals.empty())
        {
            new_file->bvalue = float(bvals[index]);
            new_file->bvec[0] = float(bvecs[index*3]);
            new_file->bvec[1] = float(bvecs[index*3+1]);
            new_file->bvec[2] = float(bvecs[index*3+2]);
            new_file->bvec.normalize();
            if(new_file->bvalue < 100)
            {
                new_file->bvalue = 0;
                new_file->bvec = tipl::vector<3>(0,0,0);
            }
        }
        if(index == 0 && !grad_dev.empty())
            new_file->grad_dev.swap(grad_dev);
        dwi_files.push_back(new_file);
    }
    return true;
}

bool load_4d_2dseq(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg)
{
    tipl::progress prog("opening ",file_name);
    tipl::io::bruker_2dseq bruker_header;
    if(!bruker_header.load_from_file(file_name))
    {
        error_msg = "failed to load image from 2dseq";
        return false;
    }
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
        if(!method_file.load_from_file(method_name.toStdString().c_str()))
        {
            error_msg = "cannot find method file at ";
            error_msg += method_name.toStdString();
            return false;
        }        
        if(!method_file.read("PVM_DwEffBval",bvalues))
        {
            error_msg = "cannot find PVM_DwEffBval in method file at ";
            error_msg += method_name.toStdString();
            return false;
        }
        std::vector<float> bvec_temp;
        if(!method_file.read("PVM_DwGradVec",bvec_temp))
        {
            error_msg = "cannot find PVM_DwGradVec in method file at ";
            error_msg += method_name.toStdString();
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
            if(!imnd_file.load_from_file(imnd_name.toStdString().c_str()))
            {
                error_msg = "cannot find method or imnd file at ";
                error_msg += imnd_name.toStdString();
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
        {
            error_msg = "cannot find imnd file at ";
            error_msg += system_path.toStdString();
            return false;
        }
    }


    tipl::shape<3> dim(bruker_header.get_image().shape());
    dim[2] /= bvalues.size();

    if(dwi_files.size() && dwi_files.back()->image.shape() != dim)
    {
        error_msg = "inconsistent dimension found";
        return false;
    }
    tipl::lower_threshold(bruker_header.get_image(),0.0);
    tipl::normalize(bruker_header.get_image(),32767.0);

    for (unsigned int index = 0;index < bvalues.size();++index)
    {
        std::shared_ptr<DwiHeader> new_file(new DwiHeader);
        new_file->report = report;
        new_file->image.resize(dim);
        std::copy_n(bruker_header.get_image().begin()+index*new_file->image.size(),new_file->image.size(),
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

bool load_multiple_slice_dicom(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                               std::string& error_msg)
{
    tipl::io::dicom dicom_header;
    if(!dicom_header.load_from_file(file_list[0].toStdString().c_str()) || !dicom_header.is_dwi())
        return false;
    tipl::shape<3> geo;
    dicom_header.get_image_dimension(geo);

    tipl::out() << "is_mosaic: " << (dicom_header.is_mosaic ? "yes":"no");
    tipl::out() << "is_multi_frame: " << (dicom_header.is_multi_frame ? "yes":"no");
    tipl::out() << "shape: " << geo;


    if(!dicom_header.is_mosaic && (dicom_header.is_multi_frame || geo[2] != 1 || file_list.size() < 2))
    {
        tipl::progress prog("parsing multiframe");
        for(int index = 0;prog(index,file_list.size());++index)
        {
            if(!load_dicom_multi_frame(file_list[index].toStdString().c_str(),dwi_files))
                return false;
        }
        return !prog.aborted() && !dwi_files.empty();
    }

    std::vector<std::shared_ptr<DwiHeader> > dwis(file_list.size());

    tipl::progress p("reading dicoms");
    bool has_dwi = false;
    for(size_t i = 0;p(i, file_list.size());++i)
    {
        dwis[i] = std::make_shared<DwiHeader>();
        if(!dwis[i]->open(file_list[i].toStdString().c_str()))
        {
            error_msg = dwis[i]->error_msg;
            error_msg += " at ";
            error_msg += file_list[i].toStdString();
            return false;
        }
        tipl::out() << QFileInfo(file_list[i]).fileName().toStdString() << " shape: " << dwis[i]->image.shape() << " b:" << dwis[i]->bvalue << " " << dwis[i]->bvec;
        if(dwis[i]->bvalue > 0.0f)
            has_dwi = true;
        if(!i)
            geo = dwis[i]->image.shape();
        if(dwis[i]->image.shape() != geo)
        {
            error_msg = "inconsistent image size found at ";
            error_msg += file_list[i].toStdString();
            error_msg += " please parse DICOM into folders before further processing.";
            return false;
        }
    }
    if(p.aborted())
        return false;
    if(!has_dwi)
        return false;
    if(dicom_header.is_mosaic)
    {
        dwi_files = dwis;
        return !dwi_files.empty();
    }



    bool iterate_slice_first = true;
    unsigned int slice_num = 2;
    unsigned int b_num = 2;

    unsigned int slice_axis = 2;
    if(geo[1] == 1)
        slice_axis = 1;
    if(geo[0] == 1)
        slice_axis = 0;

    if(dwis[0]->slice_location == 0.0) // no slice location information
    {
        if(dwis[0]->bvec == dwis[1]->bvec && dwis[0]->bvalue == dwis[1]->bvalue) // iterate slice first
        {
            for (;slice_num < file_list.size();++slice_num)
                if(dwis[0]->bvec != dwis[slice_num]->bvec || dwis[0]->bvalue != dwis[slice_num]->bvalue)
                    break;
            geo[slice_axis] = slice_num;
            iterate_slice_first = true;
        }
        else
        // iterate b first
        {
            for (;b_num < file_list.size();++b_num)
                if(dwis[0]->bvec == dwis[slice_num]->bvec && dwis[0]->bvalue == dwis[slice_num]->bvalue)
                    break;
            geo[slice_axis] = file_list.size()/b_num;
            iterate_slice_first = false;
        }
    }
    else
    {
        if(dwis[0]->slice_location == dwis[1]->slice_location) // iterate b-value first
        {
            for (;b_num < file_list.size();++b_num)    
                if(dwis[b_num]->slice_location != dwis[0]->slice_location)
                    break;
            geo[slice_axis] = std::ceil(float(file_list.size())/float(b_num));
            iterate_slice_first = false;
        }
        else
        // iterate slice first
        {
            for (;slice_num < file_list.size();++slice_num)
                if(dwis[slice_num]->slice_location == dwis[0]->slice_location)
                    break;
            geo[slice_axis] = slice_num;
            iterate_slice_first = true;
        }
    }

    tipl::out() << "slice num: " << geo[slice_axis];

    tipl::progress prog("reading dicoms");
    for (unsigned int index = 0,b_index = dwi_files.size(),slice_index = 0;prog(index,file_list.size());++index)
    {
        if(slice_index == 0)
        {
            dwi_files.push_back(dwis[index]);
            dwi_files.back()->image.resize(geo);
        }
        else
        {
            if(slice_axis == 2)
                std::copy(dwis[index]->image.begin(),dwis[index]->image.end(),
                          dwi_files[b_index]->image.begin() + slice_index*dwis[index]->image.size());
            if(slice_axis == 1)
            {
                for(size_t pos = 0,pos2 = slice_index*geo[0];
                    pos < dwis[index]->image.size();pos += geo[0],pos2 += geo.plane_size())
                    std::copy_n(dwis[index]->image.begin()+pos,geo[0],dwi_files[b_index]->image.begin() + pos2);
            }
            if(slice_axis == 0)
            {
                for (size_t z = 0,src_index = 0,plane_index = 0; z < geo[2]; ++z,plane_index += geo.plane_size())
                    for (size_t y = 0,pos = plane_index + slice_index; y < geo[1]; ++y,++src_index,pos += geo[0])
                        dwi_files[b_index]->image[pos] = dwis[index]->image[src_index];
            }
        }
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
    return !prog.aborted();
}
void scale_image_buf_to_uint16(std::vector<tipl::image<3> >& image_buf)
{
    float max_value = 0.0f;
    for(size_t i = 0;i < image_buf.size();++i)
        max_value = std::max<float>(max_value,tipl::max_value(image_buf[i]));
    tipl::adaptive_par_for(image_buf.size(),[&](int i)
    {
        image_buf[i] *= float(std::numeric_limits<unsigned short>::max()-1)/max_value;
        tipl::lower_threshold(image_buf[i],0);
    });
}
bool load_nhdr(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
               std::string& error_msg)
{
    tipl::progress prog("opening ",file_list[0].toStdString().c_str());
    std::vector<tipl::image<3> > image_buf;
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    image_buf.resize(file_list.size());
    for (size_t i = 0;prog(i,file_list.size());++i)
    {
        std::unordered_map<std::string,std::string> value_list;
        {
            for(const auto& line: tipl::read_text_file(file_list[i].toStdString()))
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
        {
            error_msg = "unsupported value type";
            return false;
        }
        // allocate all space
        if(i == 0)
        {
            float d;
            if(!(std::istringstream(value_list["sizes"]) >> dim[0] >> dim[1] >> dim[2])||
               !(std::istringstream(value_list["space directions"]) >> vs[0] >> d >> d >> d >> vs[1] >> d >> d >> d >> vs[2]))
            {
                error_msg = "failed to parse file";
                return false;
            }
        }

        try{
            image_buf[i].resize(dim);
        }
        catch(...)
        {
            error_msg = "insufficient memory";
            return false;
        }
        std::string raw_file_name = file_list[i].toStdString();
        raw_file_name = raw_file_name.substr(0,raw_file_name.length()-4);
        raw_file_name += "raw";
        std::ifstream in(raw_file_name,std::ifstream::binary);
        tipl::out() << "reading" << raw_file_name << std::endl;
        if(!in.read((char*)&image_buf[i][0],image_buf[i].size()*sizeof(float)))
        {
            error_msg = "failed to read image file";
            return false;
        }
    }
    if(prog.aborted())
        return false;

    scale_image_buf_to_uint16(image_buf);

    {
        tipl::progress prog2("Converting data");
        for(size_t i = 0;prog2(i,image_buf.size());++i)
        {
            dwi_files.push_back(std::make_shared<DwiHeader>());
            dwi_files.back()->voxel_size = vs;
            dwi_files.back()->file_name = file_list[i].toStdString();
            dwi_files.back()->report = " The diffusion images were acquired on an Agilent scanner.";
            dwi_files.back()->image = image_buf[i];
            image_buf[i] = tipl::image<3>();
        }
        if(prog2.aborted())
            return false;
    }
    return true;
}
bool load_4d_fdf(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                 std::string& error_msg)
{
    tipl::progress prog("opening ",file_list[0].toStdString().c_str());
    std::vector<tipl::image<3> > image_buf;
    bool scan_2d = true;
    for (int index = 0;prog(index,file_list.size());++index)
    {
        std::unordered_map<std::string,std::string> value_list;
        {
            for(const auto& line: tipl::read_text_file(file_list[index].toStdString()))
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
        {
            error_msg = "unsupported value type";
            return false;
        }
        // allocate all space
        if(index == 0)
        {
            unsigned int dwi_num(0);
            tipl::shape<3> shape;
            tipl::vector<3> fov;
            // 2D acquisition
            if(std::istringstream(value_list["slices"]) >> shape[2])
            {
                if(!(std::istringstream(value_list["matrix[]"]) >> shape[0] >> shape[1]))
                {
                    error_msg = "cannot read matrix information";
                    return false;
                }
            }
            else
            // 3D acquisition
            {
                scan_2d = false;
                if(!(std::istringstream(value_list["matrix[]"]) >> shape[0] >> shape[1] >> shape[2]))
                {
                    error_msg = "cannot read matrix information";
                    return false;
                }
            }

            if(!(std::istringstream(value_list["array_dim"]) >> dwi_num) || dwi_num == 0 ||
               !(std::istringstream(value_list["roi[]"]) >> fov[0] >> fov[1] >> fov[2]))
            {
                error_msg = "no array_dim or roi data";
                return false;
            }

            // calculate voxel size
            {
                fov *= 10.0f;
                fov[0] /= float(shape[0]);
                fov[1] /= float(shape[1]);
                fov[2] /= float(shape[2]);
                if(scan_2d)
                    fov[2] *= 10.0f;
            }

            image_buf.resize(dwi_num);
            for(unsigned int i = 0;i < dwi_num;++i)
            {
                image_buf[i].resize(shape);
                dwi_files.push_back(std::make_shared<DwiHeader>());
                dwi_files.back()->image.resize(shape);
                dwi_files.back()->voxel_size = fov;
                dwi_files.back()->file_name = value_list["*studyid"];
            }
            dwi_files.back()->report = " The diffusion images were acquired on a Varian scanner.";
        }
        // get DWI location
        size_t dwi_id;
        if(!(std::istringstream(value_list["array_index"]) >> dwi_id) || (--dwi_id) >= dwi_files.size())
        {
            error_msg = "error reading array_index";
            return false;
        }
        // get b_value
        if(dwi_files[uint32_t(dwi_id)]->bvalue == 0.0f)
        {
            if(!(std::istringstream(value_list["dro"]) >> dwi_files[dwi_id]->bvec[0]) ||
               !(std::istringstream(value_list["dpe"]) >> dwi_files[dwi_id]->bvec[1]) ||
               !(std::istringstream(value_list["dsl"]) >> dwi_files[dwi_id]->bvec[2]) ||
               !(std::istringstream(value_list["bvalue"]) >> dwi_files[dwi_id]->bvalue))
            {
                error_msg = "no btable information";
                return false;
            }
            dwi_files[uint32_t(dwi_id)]->bvec.normalize();
        }

        std::ifstream in(file_list[index].toStdString().c_str(),std::ifstream::binary);
        if(scan_2d)
        {
            // get DWI and slice location
            size_t slice_id;
            if(!(std::istringstream(value_list["slice_no"]) >> slice_id) || (--slice_id >= size_t(dwi_files[dwi_id]->image.depth())))
            {
                error_msg = "invalid slice no";
                return false;
            }
            int64_t plane_size = int64_t(image_buf[dwi_id].plane_size());
            in.seekg(-plane_size*4,std::ios_base::end);
            if(!in.read(reinterpret_cast<char*>(&*(image_buf[dwi_id].begin() + plane_size*int64_t(slice_id))),plane_size*4))
            {
                error_msg = "read image failed";
                return false;
            }
        }
        else
        {
            in.seekg(-int64_t(image_buf[dwi_id].size()*4),std::ios_base::end);
            if(!in.read(reinterpret_cast<char*>(&*(image_buf[dwi_id].begin())),image_buf[dwi_id].size()*4))
            {
                error_msg = "read image failed";
                return false;
            }
        }

    }
    if(prog.aborted())
        return false;

    scale_image_buf_to_uint16(image_buf);
    for(size_t i = 0;i < image_buf.size();++i)
        dwi_files[i]->image = image_buf[i];
    return true;
}

std::vector<QStringList> split_by_path(const QStringList &files)
{
    std::vector<QStringList> result;
    QString currentPath;
    QStringList currentList;

    for (const auto &file : files)
    {
        auto path = QFileInfo(file).absolutePath();
        if (path != currentPath)
        {
            if (!currentList.isEmpty())
            {
                result.push_back(currentList);
                currentList.clear();
            }
            currentPath = path;
        }
        currentList << file;
    }

    if (!currentList.isEmpty())
        result.push_back(currentList);
    return result;
}

bool parse_dwi(QStringList file_list,
                    std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg)
{
    if(file_list.isEmpty())
        return false;
    if(QFileInfo(file_list[0]).isDir())
    {
        if(QFileInfo(file_list[0]+"/pdata/1/2dseq").exists())
        {
            for(int index = 0;index < file_list.size();++index)
                parse_dwi(QStringList() << (file_list[index]+"/pdata/1/2dseq"),dwi_files,error_msg);
            return !dwi_files.empty();
        }

        tipl::out()  << "processing a single folder with DICOM files";
        QDir cur_dir = file_list[0];
        QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
        if(dicom_file_list.empty())
            return false;
        for (int index = 0;index < dicom_file_list.size();++index)
            dicom_file_list[index] = file_list[0] + "/" + dicom_file_list[index];
        return parse_dwi(dicom_file_list,dwi_files,error_msg);
    }

    error_msg.clear();
    if(QFileInfo(file_list.front()).absolutePath() != QFileInfo(file_list.back()).absolutePath())
    {
        auto dwi_groups = split_by_path(file_list);
        for(const auto& sub_file_list : dwi_groups)
        {
            if(!parse_dwi(sub_file_list,dwi_files,error_msg))
                return false;
        }
        return true;
    }

    if(QFileInfo(file_list[0]).fileName() == "2dseq")
    {
        for(int index = 0;index < file_list.size();++index)
            load_4d_2dseq(file_list[index].toStdString().c_str(),dwi_files,error_msg);
        return !dwi_files.empty();
    }
    if(QFileInfo(file_list[0]).suffix() == "fdf")
        return load_4d_fdf(file_list,dwi_files,error_msg);
    if(QFileInfo(file_list[0]).suffix() == "nhdr")
        return load_nhdr(file_list,dwi_files,error_msg);
    if(QFileInfo(file_list[0]).fileName().endsWith(".nii") ||
            QFileInfo(file_list[0]).fileName().endsWith(".nii.gz"))
    {
        for(int i = 0;i < file_list.size();++i)
            if(!load_4d_nii(file_list[i].toStdString(),dwi_files,true,false,error_msg))
                return false;
        return !dwi_files.empty();
    }
    std::sort(file_list.begin(),file_list.end(),compare_qstring());
    return load_multiple_slice_dicom(file_list,dwi_files,error_msg);
}
bool parse_dwi(const std::vector<std::string>& file_list,
                    std::vector<std::shared_ptr<DwiHeader> >& dwi_files,std::string& error_msg)
{
    QStringList qlist;
    for(const auto& each : file_list)
        qlist << each.c_str();
    return parse_dwi(qlist,dwi_files,error_msg);
}
void dicom_parser::load_table(void)
{
    int last_index = ui->tableWidget->rowCount();
    ui->tableWidget->setRowCount(int(dwi_files.size()));
    double max_b = 0;
    for(size_t index = size_t(last_index);index < dwi_files.size();++index)
    {
        if(dwi_files[index]->bvalue < 100.0f)
            dwi_files[index]->bvalue = 0.0f;
        ui->tableWidget->setItem(index, 0, new QTableWidgetItem(QFileInfo(dwi_files[index]->file_name.data()).fileName()));
        ui->tableWidget->setItem(index, 1, new QTableWidgetItem(QString::number(dwi_files[index]->bvalue)));
        ui->tableWidget->setItem(index, 2, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[0])));
        ui->tableWidget->setItem(index, 3, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[1])));
        ui->tableWidget->setItem(index, 4, new QTableWidgetItem(QString::number(dwi_files[index]->bvec[2])));
        max_b = std::max(max_b,(double)dwi_files[index]->bvalue);
    }
    if(max_b == 0.0)
        QMessageBox::critical(this,QApplication::applicationName(),"Cannot find matching bval and bvec files. You can load them using the [File] menu");
}
void dicom_parser::load_files(QStringList file_list)
{
    std::string error_msg;
    if(!parse_dwi(file_list,dwi_files,error_msg))
    {
        if(!error_msg.empty())
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
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
    src_data src;
    if(!src.load_from_file(dwi_files,ui->sort_btable->isChecked()) ||
       !src.save_to_file(ui->SrcName->text().toStdString()))
    {
        QMessageBox::critical(this,"ERROR",src.error_msg.c_str());
        close();
        return;
    }
    if(!QFileInfo(ui->SrcName->text()).exists())
    {
        QMessageBox::critical(this,"ERROR","Cannot save SRC file");
        close();
        return;
    }

    ((MainWindow*)parent())->addSrc(ui->SrcName->text());
    QMessageBox::information(this,QApplication::applicationName(),"SRC file created");
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
            "Src files (*.sz *src.gz);;All files (*)" );
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
    if(filename.isEmpty())
        return;
    std::vector<double> b_table;
    for(auto& line: tipl::read_text_file(filename.toStdString()))
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
        tipl::shape<3> dim(I.width(),I.height(),I.depth()/b_count);
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
                std::copy_n(I.slice_at(i+j*b_count).begin(),plane_size,
                          new_file->image.slice_at(j).begin());
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
        for(int i = 0;i < ui->tableWidget->rowCount();++i)
        {
            size_t i4 = size_t(i)*4;
            size_t i4s = i4*slice_num;;
            b_table[i4] = b_table[i4s];
            b_table[i4+1] = b_table[i4s+1];
            b_table[i4+2] = b_table[i4s+2];
            b_table[i4+3] = b_table[i4s+3];
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
    load_bval(filename.toStdString(),bval);
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
    load_bvec(filename.toStdString().c_str(),b_table);
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

    std::ofstream btable(filename.toStdString().c_str());
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
