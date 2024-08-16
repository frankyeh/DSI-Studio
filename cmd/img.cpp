#include <QBuffer>
#include <QImageReader>
#include <QFile>
#include <QTextStream>
#include "img.hpp"
std::map<std::string,std::string> dicom_dictionary;

bool img_command_float32_std(tipl::image<3>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,const std::string& param1,std::string& error_msg)
{
    return tipl::command<tipl::out,tipl::io::gz_nifti>(data,vs,T,is_mni,cmd,param1,error_msg);
}

bool variant_image::command(std::string cmd,std::string param1)
{
    bool result = true;
    error_msg.clear();
    if(cmd == "change_type")
        change_type(decltype(pixel_type)(std::stoi(param1)));
    else
    apply([&](auto& I)
    {
        result = tipl::command<tipl::out,tipl::io::gz_nifti>(I,vs,T,is_mni,cmd,param1,error_msg);
        shape = I.shape();
    });
    return result;
}


bool variant_image::read_mat_image(const std::string& metric_name,tipl::io::gz_mat_read& mat)
{
    unsigned int row(0), col(0);
    if(!mat.get_col_row(metric_name.c_str(),row,col) || row*col != shape.size())
        return false;
    if(mat.type_compatible<unsigned int>(metric_name.c_str()))
    {
        I_int32.resize(shape);
        mat.read(metric_name.c_str(),I_int32.begin(),I_int32.end());
        pixel_type = int32;
        return true;
    }
    if(mat.type_compatible<unsigned short>(metric_name.c_str()))
    {
        I_int16.resize(shape);
        mat.read(metric_name.c_str(),I_int16.begin(),I_int16.end());
        pixel_type = int16;
        return true;
    }
    if(mat.type_compatible<unsigned char>(metric_name.c_str()))
    {
        I_int8.resize(shape);
        mat.read(metric_name.c_str(),I_int8.begin(),I_int8.end());
        pixel_type = int8;
        return true;
    }
    I_float32.resize(shape);
    mat.read(metric_name.c_str(),I_float32.begin(),I_float32.end());
    pixel_type = float32;
    return true;
}

void variant_image::change_type(decltype(pixel_type) new_type)
{
    if(new_type == pixel_type)
        return;
    apply([&](auto& I)
    {
        switch(new_type)
        {
            case int8:
                I_int8.resize(shape);
                std::copy(I.begin(),I.end(),&I_int8[0]);
                break;
            case int16:
                I_int16.resize(shape);
                std::copy(I.begin(),I.end(),&I_int16[0]);
                break;
            case int32:
                I_int32.resize(shape);
                std::copy(I.begin(),I.end(),&I_int32[0]);
                break;
            case float32:
                I_float32.resize(shape);
                std::copy(I.begin(),I.end(),&I_float32[0]);
        }
        I.clear();
    });
    pixel_type = new_type;

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
        tipl::error() << "unsupported transfer syntax " << dicom.encoding;
        return false;
    }
    QImage buf = img.convertToFormat(QImage::Format_RGB32);
    I.resize(tipl::shape<2>(buf.width(),buf.height()));
    const uchar* ptr = buf.bits();
    for(int j = 0;j < I.size();++j,ptr += 4)
        I[j] = *ptr;
    return true;
}
void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
bool variant_image::load_from_file(const char* file_name,std::string& info)
{
    tipl::io::dicom dicom;
    is_mni = false;
    T.identity();
    tipl::progress prog("open image file ",std::filesystem::path(file_name).filename().u8string().c_str());
    if(QString(file_name).endsWith(".nhdr"))
    {
        tipl::io::nrrd<tipl::progress> nrrd;
        if(!nrrd.load_from_file(file_name))
        {
            error_msg = nrrd.error_msg;
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
            error_msg = nrrd.error_msg;
            return false;
        }
        nrrd.get_voxel_size(vs);
        nrrd.get_image_transformation(T);

        info.clear();
        for(const auto& iter : nrrd.values)
        {
            info += iter.first;
            info += "=";
            info += iter.second;
            info += "\n";
        }
    }
    else
    if(tipl::ends_with(file_name,".nii.gz") || tipl::ends_with(file_name,".nii"))
    {
        tipl::io::gz_nifti nifti;
        prepare_idx(file_name,nifti.input_stream);
        if(!nifti.load_from_file(file_name))
        {
            error_msg = nifti.error_msg;
            return false;
        }
        if(nifti.dim(4) != 1)
            error_msg = "4d image";
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
            error_msg = "Unsupported pixel format";
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
                for(size_t pos = 0;pos < data.size();++pos)
                   if(std::isnan(data[pos]))
                       data[pos] = 0;
            }
        });
        if(!succeed)
        {
            error_msg = nifti.error_msg;
            return false;
        }
        if(nifti.dim(4) == 1)
            save_idx(file_name,nifti.input_stream);
        nifti.get_voxel_size(vs);
        nifti.get_image_transformation(T);
        is_mni = nifti.is_mni();
        std::ostringstream out;
        out << nifti;
        info = out.str();
    }
    else
        if(dicom.load_from_file(file_name))
        {
            pixel_type = int16;
            dicom.get_image_dimension(shape);
            if(dicom.is_compressed)
            {
                tipl::image<2,short> I;
                if(!get_compressed_image(dicom,I))
                {
                    error_msg = "Unsupported transfer syntax ";
                    error_msg += dicom.encoding;
                    return false;
                }
                if(I.size() == shape.size())
                    std::copy(I.begin(),I.end(),I_int16.begin());
                else
                {
                    error_msg = "Cannot decompress image ";
                    error_msg += dicom.encoding;
                    return false;
                }
            }
            else
                apply([&](auto& data){dicom >> data;});
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
            if(QString(file_name).endsWith("2dseq"))
            {
                tipl::io::bruker_2dseq seq;
                if(!seq.load_from_file(file_name))
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
    return true;
}
bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param);
int img(tipl::program_option<tipl::out>& po)
{
    std::string source(po.get("source")),info;
    if(tipl::ends_with(source,"fib.gz"))
    {
        tipl::io::gz_mat_read mat_reader;
        tipl::out() << "open " << source;
        if(!mat_reader.load_from_file(source))
        {
            tipl::error() << mat_reader.error_msg;
            return 1;
        }
        for(auto& cmd : tipl::split(po.get("cmd"),'+'))
        {
            std::string param;
            auto sep_pos = cmd.find(':');
            if(sep_pos != std::string::npos)
            {
                param = cmd.substr(sep_pos+1);
                cmd = cmd.substr(0,sep_pos);
            }
            if(!modify_fib(mat_reader,cmd,param))
            {
                tipl::error() << mat_reader.error_msg;
                return 1;
            }
        }
        if(po.has("output"))
        {
            tipl::out() << "saving output";
            if(!modify_fib(mat_reader,"save",po.get("output")))
            {
                tipl::error() << mat_reader.error_msg;
                return 1;
            }
        }
        return 0;
    }

    variant_image var_image;
    tipl::out() << "open " << source;
    if(!var_image.load_from_file(source.c_str(),info))
    {
        tipl::error() << var_image.error_msg;
        return 0;
    }
    for(auto& cmd : tipl::split(po.get("cmd"),'+'))
    {
        std::string param;
        auto sep_pos = cmd.find(':');
        if(sep_pos != std::string::npos)
        {
            param = cmd.substr(sep_pos+1);
            cmd = cmd.substr(0,sep_pos);
        }
        if(cmd == "info")
        {
            tipl::out() << info;
            continue;
        }
        if(!var_image.command(cmd,param))
        {
            tipl::error() << var_image.error_msg;
            return 1;
        }
    }
    if(po.has("output"))
    {
        tipl::out() << "saving output";
        if(!var_image.command("save",po.get("output")))
        {
            tipl::error() << var_image.error_msg;
            return 1;
        }
    }
    return 0;
}
