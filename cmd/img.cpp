#include <QBuffer>
#include <QImageReader>
#include <QFile>
#include <QTextStream>
#include "img.hpp"
#include "reg.hpp"
std::map<std::string,std::string> dicom_dictionary;
void correct_bias_field(tipl::image<3> I,
                        const tipl::image<3,unsigned char>& mask,
                        tipl::image<3>& log_bias_field,
                        const tipl::vector<3>& spacing);
bool variant_image::command(std::string cmd,std::string param1)
{
    bool result = true;
    error_msg.clear();
    if(cmd == "change_type")
        change_type(decltype(pixel_type)(std::stoi(param1)));
    else
    apply([&](auto& I)
    {
        if(cmd == "bias_field_correction")
        {
            tipl::image<3,unsigned char> mask;
            tipl::threshold(I,mask,0);
            tipl::image<3> bias_field;
            correct_bias_field(I,mask,bias_field,tipl::vector<3>(1.0f,vs[0]/vs[1],vs[0]/vs[2]));
            for(auto& each : bias_field)
                each = std::exp(-each);
            I *= bias_field;
        }
        else
            result = tipl::command<void,tipl::io::gz_nifti>(I,vs,T,is_mni,cmd,param1,interpolation,error_msg);
        shape = I.shape();
    });
    return result;
}
void variant_image::write_mat_image(size_t index,
                    tipl::io::gz_mat_read& mat)
{
    mat[index].set_row_col(shape.plane_size(),shape.depth());
    apply([&](const auto& image_data)
    {
        if(mat[index].type_compatible<short>())
            std::copy(image_data.begin(),image_data.end(),mat[index].get_data<short>());
        if(mat[index].type_compatible<float>())
            std::copy(image_data.begin(),image_data.end(),mat[index].get_data<float>());
        if(mat[index].type_compatible<char>())
            std::copy(image_data.begin(),image_data.end(),mat[index].get_data<char>());
    });
}

bool variant_image::read_mat_image(size_t index,
                                   tipl::io::gz_mat_read& mat)
{
    if(index >= mat.size())
        return false;
    unsigned int row(mat.rows(index)), col(mat.cols(index));
    if(row*col != shape.size())
        return false;
    if(mat[index].is_scaled())
        mat[index].convert_to<float>();
    if(mat[index].type_compatible<unsigned int>())
    {
        I_int32.resize(shape);
        mat.read(index,I_int32.begin(),I_int32.end());
        pixel_type = int32;
        return true;
    }
    if(mat[index].type_compatible<unsigned short>())
    {
        I_int16.resize(shape);
        mat.read(index,I_int16.begin(),I_int16.end());
        pixel_type = int16;
        return true;
    }
    if(mat[index].type_compatible<unsigned char>())
    {
        I_int8.resize(shape);
        mat.read(index,I_int8.begin(),I_int8.end());
        pixel_type = int8;
        return true;
    }
    if(mat[index].type_compatible<float>())
    {
        I_float32.resize(shape);
        mat.read(index,I_float32.begin(),I_float32.end());
        pixel_type = float32;
        return true;
    }
    return false;
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
                if(pixel_type == float32)
                {
                    auto max_v = tipl::max_value(I);
                    if(max_v <= 1.0f && max_v != 0.0f)
                        tipl::multiply_constant(I,255.99f/max_v);
                    tipl::lower_threshold(I,0);
                }
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
        dicom >> I;
        if(dicom.is_compressed)
        {
            tipl::error() << "unsupported transfer syntax " << dicom.encoding;
            return false;
        }
        else
            return true;
    }
    else
    {
        QImage buf = img.convertToFormat(QImage::Format_RGB32);
        I.resize(tipl::shape<2>(buf.width(),buf.height()));
        const uchar* ptr = buf.bits();
        for(int j = 0;j < I.size();++j,ptr += 4)
            I[j] = *ptr;
    }
    return true;
}
void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
bool variant_image::load_from_file(const char* file_name,std::string& info)
{
    tipl::io::dicom dicom;
    is_mni = false;
    T.identity();
    tipl::progress prog("open " + std::string(file_name));
    if(QString(file_name).endsWith(".nhdr") || QString(file_name).endsWith(".nrrd"))
    {
        tipl::io::gz_nrrd nrrd;
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
    if(tipl::ends_with(file_name,".nii.gz") || tipl::ends_with(file_name,".nii") ||
       tipl::ends_with(file_name,".hdr"))
    {
        tipl::io::gz_nifti nifti;
        prepare_idx(file_name,nifti.input_stream);
        if(!nifti.open(file_name,std::ios::in))
        {
            error_msg = nifti.error_msg;
            return false;
        }
        nifti >> vs >> T >> shape >> is_mni;
        dim4 = nifti.dim(4);
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

        if(!apply([&](auto& data)
        {
            bool succeed = nifti.get_untouched_image(data);
            if constexpr(!std::is_integral<typename std::remove_reference<decltype(*data.begin())>::type>::value)
            {
                for(size_t pos = 0;pos < data.size();++pos)
                   if(std::isnan(data[pos]))
                       data[pos] = 0;
            }
            return succeed;
        }))
        {
            error_msg = nifti.error_msg;
            return false;
        }
        if(dim4 == 1)
            save_idx(file_name,nifti.input_stream);
        std::ostringstream out;
        out << nifti;
        info = out.str();
    }
    else
        if(dicom.load_from_file(file_name))
        {
            std::string info_;
            pixel_type = int16;
            dicom >> std::tie(shape,vs,info_,I_int16);
            if(dicom.is_compressed)
            {
                tipl::image<2,short> I;
                if(!get_compressed_image(dicom,I))
                {
                    error_msg = "Unsupported transfer syntax ";
                    error_msg += dicom.encoding;
                    return false;
                }
                I_int16.resize(shape);
                std::copy_n(I.begin(),std::min<size_t>(I.size(),shape.size()),I_int16.begin());
            }

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
    tipl::out() << "dim: " << shape;
    tipl::out() << "vs: " << vs;
    return true;
}
tipl::const_pointer_image<3,unsigned char> handle_mask(tipl::io::gz_mat_read& mat_reader);
template<typename slice_type>
void show_slice(slice_type&& slice)
{
    while(slice.width() > 64)
        tipl::downsampling(slice);
    float max_v = tipl::max_value(slice);
    float threshold1 = 0.2f * max_v;
    float threshold2 = 0.4f * max_v;
    float threshold3 = 0.6f * max_v;
    float threshold4 = 0.8f * max_v;
    // Print upper border
    for (int y = 0, pos = 0; y < slice.height(); ++y)
    {
        std::string line;
        for (int x = 0; x < slice.width(); ++x, ++pos)
        {
            float value = slice[pos];
            if (value < threshold1) line += " "; // black
            else if (value < threshold2) line += "▒"; // dark gray
            else if (value < threshold3) line += "▓"; // medium gray
            else if (value < threshold4) line += "▓"; // light gray
            else line += "█"; // white
        }
        tipl::out() << line;
    }
}
template<typename value_type>
void show_slice(tipl::io::gz_mat_read& mat_reader,const char* name)
{
    if(!mat_reader.has(name))
        return;
    tipl::shape<3> dim;
    if(!mat_reader.read("dimension",dim))
        return;
    if(mat_reader.has("mask"))
        handle_mask(mat_reader);
    const value_type* buffer = nullptr;
    if(!mat_reader.read(name,buffer))
        return;
    show_slice(tipl::image<2,float>(tipl::make_image(buffer,dim).slice_at(dim[2]/2)));
}
bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param);
extern std::vector<std::string> iso_template_list;
int img(tipl::program_option<tipl::out>& po)
{
    if(!po.get("overwrite",0) && po.has("output") && std::filesystem::exists(po.get("output")))
    {
        tipl::out() << "output exist, skipping";
        return 0;
    }

    std::string source(po.get("source")),info;
    if(tipl::ends_with(source,"fib.gz") || tipl::ends_with(source,".fz") ||
       tipl::ends_with(source,"src.gz") || tipl::ends_with(source,".sz") ||
       tipl::ends_with(source,".mz"))
    {
        tipl::io::gz_mat_read mat_reader;
        tipl::out() << "open " << source;
        if(!mat_reader.load_from_file(source))
        {
            tipl::error() << mat_reader.error_msg;
            return 1;
        }

        show_slice<float>(mat_reader,"fa0");
        show_slice<unsigned short>(mat_reader,"image0");

        for(unsigned int index = 0;index < mat_reader.size();++index)
            tipl::out() << mat_reader[index].get_info();


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
        return 1;
    }
    tipl::shape<4> dim4;
    if(var_image.dim4 > 1)
    {
        tipl::progress prog("loading 4d nifti");
        tipl::io::gz_nifti nifti;
        prepare_idx(source.c_str(),nifti.input_stream);
        if(!nifti.open(source,std::ios::in))
        {
            tipl::error() << var_image.error_msg;
            return 1;
        }
        if(!var_image.apply([&](auto& I)
        {
            tipl::out() << "dimension: " << (dim4 = tipl::shape<4>(I.width(),I.height(),I.depth(),var_image.dim4));
            I.resize(var_image.shape = tipl::shape<3>(I.width(),I.height(),I.depth()*var_image.dim4));
            if(!nifti.save_to_buffer(I.data(),dim4.size()))
            {
                tipl::error() << nifti.error_msg;
                return false;
            }
            return true;
        }))
            return 1;
    }

    {
        tipl::progress prog("run command");
        for(auto& cmd : tipl::split(po.get("cmd","info"),'+'))
        {
            std::string param;
            auto sep_pos = cmd.find(':');
            if(sep_pos != std::string::npos)
            {
                param = cmd.substr(sep_pos+1);
                cmd = cmd.substr(0,sep_pos);
            }

            if(cmd == "brain_extraction" || cmd == "segmentation")
            {
                if(cmd == "brain_extraction" && param.length() < 2)
                {
                    size_t template_id = param.empty() ? 0 : param[0]-'0';
                    dual_reg reg;
                    var_image.apply([&](auto& I){
                        reg.I[0] = subject_image_pre(tipl::image<3>(I.alias()));
                        reg.Is = I.shape();
                    });
                    reg.Ivs = var_image.vs;
                    reg.IR = var_image.T;

                    if(!reg.load_template(0,QString(iso_template_list[template_id].c_str()).replace(".ISO.nii.gz",".T1W.nii.gz").toStdString()) ||
                       !reg.load_template(1,QString(iso_template_list[template_id].c_str()).replace(".ISO.nii.gz",".T2W.nii.gz").toStdString()) ||
                       !reg.load_template(2,iso_template_list[template_id]))
                    {
                        tipl::error() << reg.error_msg;
                        return 1;
                    }
                    reg.modality_names = {"t1w","t2w"};
                    tipl::out() << "using t1w/t2w for registration..." << std::endl;
                    reg.cost_type = tipl::reg::mutual_info;
                    reg.linear_reg(tipl::prog_aborted);
                    if(reg.r[1] > reg.r[0])
                    {
                        reg.It[0].swap(reg.It[1]);
                        reg.modality_names[0].swap(reg.modality_names[1]);
                    }
                    reg.modality_names = {reg.modality_names[0]};
                    tipl::out() << "using " << reg.modality_names[0] << " for registration..." << std::endl;
                    reg.nonlinear_reg(tipl::prog_aborted);
                    auto iso = reg.apply_warping<false,tipl::interpolation::linear>(reg.It[2]);
                    var_image.apply([&](auto& I){
                        tipl::preserve(I.begin(),I.end(),iso.begin());
                    });
                    if(po.get("export_r",0))
                        std::ofstream(source + ".r" + std::to_string(int(reg.r[0]*100))) << std::endl;
                    continue;
                }

                auto model_path = QCoreApplication::applicationDirPath().toStdString()+ "/network/" + po.get("network",param);
                auto unet = tipl::ml3d::unet3d::load_model<tipl::io::gz_mat_read>(model_path.c_str());
                if(!unet.get())
                {
                    tipl::error() << "cannot read network file at" + model_path;
                    return 1;
                }
                var_image.apply([&](auto& I)
                {
                    unet->forward(I,var_image.vs,prog);
                    if(cmd == "brain_extraction")
                        I *= unet->get_mask();
                    if(cmd == "segmentation")
                    {
                        I.clear();
                        var_image.I_int8 = unet->get_label();
                        var_image.pixel_type = variant_image::int8;
                    }
                });
                continue;
            }
            if(cmd == "info")
            {
                if(var_image.pixel_type == variant_image::int8)
                    show_slice(tipl::image<2,float>(var_image.I_int8.slice_at(var_image.shape.depth()/2)));
                if(var_image.pixel_type == variant_image::int16)
                    show_slice(tipl::image<2,float>(var_image.I_int16.slice_at(var_image.shape.depth()/2)));
                if(var_image.pixel_type == variant_image::int32)
                    show_slice(tipl::image<2,float>(var_image.I_int32.slice_at(var_image.shape.depth()/2)));
                if(var_image.pixel_type == variant_image::float32)
                    show_slice(tipl::image<2,float>(var_image.I_float32.slice_at(var_image.shape.depth()/2)));
                tipl::out() << info;
                continue;
            }
            tipl::out() << std::string(param.empty() ? cmd : cmd+":"+param);
            if(!var_image.command(cmd,param))
            {
                tipl::error() << var_image.error_msg;
                return 1;
            }
        }
        }

    if(po.has("output"))
    {
        auto output = po.get("output");
        if(var_image.dim4 > 1)
        {
            if(!var_image.apply([&](auto& I)->bool
            {
                return tipl::io::gz_nifti(output,std::ios::out) << var_image.bind(tipl::make_image(I.data(),dim4));
            }))
                return 1;
        }
        else
        {
            if(!var_image.command("save",output))
            {
                tipl::error() << var_image.error_msg;
                return 1;
            }
        }
    }
    return 0;
}



