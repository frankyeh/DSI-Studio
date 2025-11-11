// ---------------------------------------------------------------------------
#include <string>
#include <filesystem>
#include <QFileInfo>
#include <QImage>
#include <QInputDialog>
#include <QMessageBox>
#include <QApplication>
#include <QDir>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include "SliceModel.h"
#include "fib_data.hpp"
#include "reg.hpp"
SliceModel::SliceModel(std::shared_ptr<fib_data> new_handle,std::shared_ptr<slice_model> new_view):
    handle(new_handle),view(new_view)
{
    slice_visible[0] = true;
    slice_visible[1] = true;
    slice_visible[2] = true;
    to_dif.identity();
    to_slice.identity();
    dim = handle->dim;
    vs = handle->vs;
    trans_to_mni = handle->trans_to_mni;
    slice_pos[0] = dim.width() >> 1;
    slice_pos[1] = dim.height() >> 1;
    slice_pos[2] = dim.depth() >> 1;
}
// ---------------------------------------------------------------------------
void SliceModel::apply_overlay(tipl::color_image& show_image,
                    unsigned char cur_dim,
                    std::shared_ptr<SliceModel> other_slice,
                               float zoom) const
{
    if(show_image.empty())
        return;
    bool op = show_image[0][0] < 128;
    const auto& v2c = other_slice->view->v2c;
    std::pair<float,float> range = other_slice->get_contrast_range();
    for(int y = 0,pos = 0;y < show_image.height();++y)
        for(int x = 0;x < show_image.width();++x,++pos)
        {
            float value = 0;
            if(!tipl::estimate(other_slice->get_source(),toOtherSlice(other_slice,cur_dim,
                                                                      zoom != 1.0f ? float(x)*zoom : float(x),
                                                                      zoom != 1.0f ? float(y)*zoom : float(y)),value))
                continue;
            if((value > 0.0f && value > range.first) ||
               (value < 0.0f && value < range.second))
            {
                show_image[pos][0] >>= 1;
                show_image[pos][1] >>= 1;
                show_image[pos][2] >>= 1;
                show_image[pos][0] += v2c[value][0] >> 1;
                show_image[pos][1] += v2c[value][1] >> 1;
                show_image[pos][2] += v2c[value][2] >> 1;
            }

        }
}


// ---------------------------------------------------------------------------
std::pair<float,float> SliceModel::get_value_range(void) const
{
    view->get_minmax();
    return std::make_pair(view->min_value,view->max_value);
}
// ---------------------------------------------------------------------------
std::pair<float,float> SliceModel::get_contrast_range(void) const
{
    return std::make_pair(view->contrast_min,view->contrast_max);
}
// ---------------------------------------------------------------------------
std::pair<unsigned int,unsigned int> SliceModel::get_contrast_color(void) const
{
    return std::make_pair(view->min_color,view->max_color);
}
// ---------------------------------------------------------------------------
void SliceModel::set_contrast_range(float min_v,float max_v)
{
    view->contrast_min = min_v;
    view->contrast_max = max_v;
    view->v2c.set_range(min_v,max_v);
}
// ---------------------------------------------------------------------------
void SliceModel::set_contrast_color(unsigned int min_c,unsigned int max_c)
{
    view->min_color = min_c;
    view->max_color = max_c;
    view->v2c.two_color(min_c,max_c);
}
// ---------------------------------------------------------------------------
void SliceModel::get_slice(tipl::color_image& show_image,unsigned char cur_dim,int pos,
                           const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    view->get_slice(cur_dim, pos,show_image);
    if(directional_color)
    {
        if(view->color_map_buf.empty())
        {
            tipl::image<3,unsigned int> colormap_buf(dim);
            if(is_diffusion_space)
                std::iota(colormap_buf.begin(),
                          colormap_buf.end(),0);
            else
            {
                tipl::adaptive_par_for(tipl::begin_index(dim),tipl::end_index(dim),[&](const tipl::pixel_index<3>& index)
                {
                    tipl::vector<3> pos(index);
                    pos.to(to_dif);
                    pos.round();
                    if(handle->dim.is_valid(pos))
                        colormap_buf[index.index()] = tipl::voxel2index(pos[0],pos[1],pos[2],handle->dim);
                });
            }
            view->color_map_buf.swap(colormap_buf);
        }
        tipl::image<2,unsigned int> buf;
        tipl::volume2slice(view->color_map_buf, buf, cur_dim, pos);
        for (unsigned int index = 0;index < buf.size();++index)
        {
            auto d = handle->dir.get_fib(buf[index],0);
            show_image[index].r = std::abs(float(show_image[index].r)*d[0]);
            show_image[index].g = std::abs(float(show_image[index].g)*d[1]);
            show_image[index].b = std::abs(float(show_image[index].b)*d[2]);
        }
    }

    for(auto overlay_slice : overlay_slices)
        if(this != overlay_slice.get())
            apply_overlay(show_image,cur_dim,overlay_slice);
}
// ---------------------------------------------------------------------------
void SliceModel::get_high_reso_slice(tipl::color_image& show_image,unsigned char cur_dim,int pos,
                                     const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    if(handle && handle->high_reso.get())
    {
        for(const auto& each : handle->high_reso->slices)
            if(view->name == each->name)
            {
                each->v2c = view->v2c;
                each->get_slice(cur_dim, pos*int(handle->high_reso->dim[cur_dim])/int(handle->dim[cur_dim]),show_image);
                return;
            }
    }
    get_slice(show_image,cur_dim,pos,overlay_slices);
}
// ---------------------------------------------------------------------------
tipl::const_pointer_image<3> SliceModel::get_source(void) const
{
    return view->get_image();
}
// ---------------------------------------------------------------------------
std::string SliceModel::get_name(void) const
{
    return view->name;
}
// ---------------------------------------------------------------------------
CustomSliceModel::CustomSliceModel(std::shared_ptr<fib_data> new_handle,
                                   const std::string& source_file_name_):
    SliceModel(new_handle,std::make_shared<slice_model>(source_file_name_)),
    source_file_name(source_file_name_)
{
    handle->slices.push_back(view);
    is_diffusion_space = false;
}
CustomSliceModel::CustomSliceModel(std::shared_ptr<fib_data> new_handle,const std::vector<std::string>& file_list):
    SliceModel(new_handle,std::make_shared<slice_model>(file_list[0])),
    source_file_name(file_list[0])
{
    if(file_list.size() > 1)
        source_files = file_list;
    handle->slices.push_back(view);
    is_diffusion_space = false;
}
// ---------------------------------------------------------------------------
CustomSliceModel::CustomSliceModel(std::shared_ptr<fib_data> new_handle,
                                   std::shared_ptr<slice_model> new_slice):
    SliceModel(new_handle,new_slice),
    source_file_name(new_slice->path)
{
    handle->slices.push_back(view);
    is_diffusion_space = false;
}
// ---------------------------------------------------------------------------
CustomSliceModel::~CustomSliceModel(void)
{
    terminate();
    handle->slices.erase(std::remove(handle->slices.begin(),handle->slices.end(),view),handle->slices.end());
}
// ---------------------------------------------------------------------------
void CustomSliceModel::get_slice(tipl::color_image& image,
                           unsigned char cur_dim,int pos,
                           const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    if(!picture.empty() && (dim[cur_dim] != picture.width() && dim[cur_dim] != picture.height()))
    {
        if(picture_warped.empty())
            image = picture;
        else
            image = picture_warped;
        for(auto overlay_slice : overlay_slices)
            if(this != overlay_slice.get())
                apply_overlay(image,cur_dim,overlay_slice);
    }
    else
        SliceModel::get_slice(image,cur_dim,pos,overlay_slices);
}
// ---------------------------------------------------------------------------
void CustomSliceModel::get_high_reso_slice(tipl::color_image& image,unsigned char cur_dim,int pos,
                                           const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    if(!high_reso_picture.empty() && (dim[cur_dim] != picture.width() && dim[cur_dim] != picture.height()))
    {
        if(high_reso_picture_warped.empty())
            image = high_reso_picture;
        else
            image = high_reso_picture_warped;
        for(auto overlay_slice : overlay_slices)
            if(this != overlay_slice.get())
                apply_overlay(image,cur_dim,overlay_slice,float(picture.width())/float(high_reso_picture.width()));
    }
    else
        get_slice(image,cur_dim,pos,overlay_slices);
}
// ---------------------------------------------------------------------------
void point_warp(const tipl::color_image& in,tipl::color_image& out,
                  tipl::image<2,tipl::vector<2> >& field,tipl::vector<2> from,tipl::vector<2> to)
{
    if(field.empty())
        field.resize(in.shape());

    auto dis = from-to;
    float dis_length = dis.length()*4.0f;
    if(dis_length < 0.2f)
        return;
    tipl::for_each_neighbors(tipl::pixel_index<2>(from[0],from[1],field.shape()),field.shape(),dis_length,
            [&](const tipl::pixel_index<2>& pos)
    {
        float weighting = (from-tipl::vector<2>(pos)).length()/dis_length;
        if(weighting <= 1.0f)
            field[pos.index()] += dis*(1.0f-weighting);//(0.5f*std::cos(weighting*3.1415926f)+0.5f);
    });
    tipl::filter::mean(field);
    tipl::compose_displacement<tipl::nearest>(in,field,out);
}
void CustomSliceModel::warp_picture(tipl::vector<2> from,tipl::vector<2> to)
{
    point_warp(picture,picture_warped,warp_field,from,to);
    if(!high_reso_picture.empty())
    {
        warp_field_high_reso.resize(high_reso_picture.shape());
        tipl::scale(warp_field,warp_field_high_reso);
        warp_field_high_reso *= float(high_reso_picture.width())/float(picture.width());
        tipl::compose_displacement<tipl::nearest>(high_reso_picture,warp_field_high_reso,high_reso_picture_warped);
    }
}
// ---------------------------------------------------------------------------
tipl::const_pointer_image<3> CustomSliceModel::get_source(void) const
{
    return tipl::const_pointer_image<3>(source_images.empty() ? (const float*)(0) : &source_images[0],source_images.shape());
}
// ---------------------------------------------------------------------------


bool download_private_github_asset(QString conceptualUrlString, QString accessToken, QString saveFilePath, std::string& error_msg)
{
    QNetworkAccessManager mgr;

    auto performNetworkRequest = [&](QNetworkRequest& request, QNetworkReply::NetworkError& outError) -> QByteArray {
        auto reply = mgr.get(request);
        QEventLoop loop;
        QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
        loop.exec();
        QByteArray data;
        if (reply->error() == QNetworkReply::NoError) data = reply->readAll();
        else { outError = reply->error(); error_msg = reply->errorString().toStdString(); }
        reply->deleteLater();
        return data;
    };

    QUrl conceptualUrl(conceptualUrlString);
    if (!conceptualUrl.isValid() || conceptualUrl.host() != "github.com") {
        error_msg = "Invalid or non-GitHub URL."; return false;
    }
    QStringList segments = conceptualUrl.path().split('/', Qt::SkipEmptyParts);
    if (segments.size() < 6 || segments[2] != "releases" || segments[3] != "download") {
        error_msg = "URL format mismatch for GitHub Release asset."; return false;
    }
    QString account = segments[0], repo = segments[1],
            tag = segments[4], fileName = segments[5];

    QNetworkRequest apiReq;
    apiReq.setUrl(QUrl(QString("https://api.github.com/repos/%1/%2/releases/tags/%3").arg(account, repo, tag)));
    apiReq.setRawHeader("Accept", "application/vnd.github.v3+json");
    apiReq.setRawHeader("Authorization", QString("token %1").arg(accessToken).toUtf8());

    QNetworkReply::NetworkError apiError = QNetworkReply::NoError;
    QByteArray apiResponse = performNetworkRequest(apiReq, apiError);

    if (apiError != QNetworkReply::NoError) {
        error_msg = "GitHub API query failed: " + error_msg; return false;
    }

    QString actualDownloadUrl;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(apiResponse);
    if (jsonDoc.isObject() && jsonDoc.object().contains("assets") && jsonDoc.object()["assets"].isArray()) {
        for (const QJsonValue& val : jsonDoc.object()["assets"].toArray())
            if (val.isObject() && val.toObject()["name"].toString().compare(fileName, Qt::CaseInsensitive) == 0) {
                actualDownloadUrl = val.toObject()["url"].toString(); break;
            }
    }
    if (actualDownloadUrl.isEmpty()) {
        error_msg = "asset not found or no download URL in API response."; return false;
    }

    QNetworkRequest downloadReq;
    downloadReq.setUrl(QUrl(actualDownloadUrl));
    downloadReq.setRawHeader("Accept", "application/octet-stream");
    downloadReq.setRawHeader("Authorization", QString("token %1").arg(accessToken).toUtf8());

    QNetworkReply::NetworkError downloadError = QNetworkReply::NoError;
    QByteArray fileData = performNetworkRequest(downloadReq, downloadError);

    if (downloadError != QNetworkReply::NoError) {
        error_msg = "file download failed: " + error_msg; return false;
    }

    QFile file(saveFilePath);
    if (!file.open(QFile::WriteOnly)) {
        error_msg = "failed to open file for writing: " + saveFilePath.toStdString(); return false;
    }
    file.write(fileData); file.close();

    return true;
}

void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
QImage read_qimage(QString filename,std::string& error);
extern QString access_token;
bool CustomSliceModel::load_slices(void)
{
    if(source_file_name.empty())
    {
        error_msg = "cannot load image";
        return false;
    }
    // QSDR loaded, use MNI transformation instead
    bool has_transform = false;
    auto suffix = QFileInfo(source_file_name.c_str()).suffix();
    auto name = view->name;
    to_dif.identity();
    to_slice.identity();


    if(tipl::begins_with(source_file_name,"http"))
    {
        auto path = std::filesystem::path(handle->fib_file_name).parent_path() / std::filesystem::path(source_file_name).filename();
        if(!std::filesystem::exists(path.string()))
        {   
            tipl::progress p("downloading data from ",source_file_name);
            if(tipl::contains(source_file_name,"github.com") &&
               tipl::contains(source_file_name,"restricted") && !access_token.isEmpty())
            {
                if(!download_private_github_asset(QString::fromStdString(source_file_name),access_token,
                                                  QString::fromStdString(path.string()),error_msg))
                    return false;
            }
            else
            {
                QNetworkAccessManager manager;
                QNetworkRequest request;
                auto url = QString::fromStdString(source_file_name);
                request.setUrl(url);
                request.setRawHeader("Accept", "application/json");
                if(!access_token.isEmpty() && url.contains("restricted"))
                    request.setRawHeader("Authorization",QString("token %1").arg(access_token).toUtf8());
                auto reply = QSharedPointer<QNetworkReply>(manager.get(request),
                        [](QNetworkReply* r)
                        {
                            if(r->isRunning())
                                r->abort();
                            r->deleteLater();
                        });
                while (!reply->isFinished() && p(reply->bytesAvailable(),
                                                 reply->bytesAvailable()+1))
                    QApplication::processEvents();
                if(p.aborted())
                {
                    error_msg = "download aborted";
                    return false;
                }
                if (reply->error() == QNetworkReply::NoError)
                {
                    auto file = std::make_shared<QFile>(path.string().c_str());
                    if (!file->open(QFile::WriteOnly))
                    {
                        error_msg = "failed to save file with the fib file";
                        return false;
                    }
                    file->write(reply->readAll());
                }
            }
        }
        if(!std::filesystem::exists(path.string()))
        {
            if(tipl::contains(source_file_name,"restricted"))
                error_msg = "need access privilege to download " + source_file_name;
            else
                error_msg = "cannot download "+source_file_name;
            return false;
        }
        source_file_name = path.string();
    }

    // picture as slice
    if(suffix == "bmp" || suffix == "jpg" || suffix == "png" || suffix == "tif" || suffix == "tiff")
    {
        tipl::progress prog("open ",source_file_name);
        QString info_file = QString(source_file_name.c_str()) + ".info.txt";
        if(source_files.size() <= 1) // single slice
        {
            uint32_t slices_count = 10;
            {
                size_t max_width = tipl::max_value(handle->dim.begin(),handle->dim.end())*2;
                QImage in = read_qimage(source_file_name.c_str(),error_msg);
                if(in.isNull())
                {
                    error_msg = "cannot read picture";
                    return false;
                }
                picture << in;
                while(picture.width() > max_width)
                {
                    tipl::out() << "downsampling slice to match current resolution";
                    if(high_reso_picture.empty())
                    {
                        high_reso_picture.swap(picture);
                        tipl::downsampling(high_reso_picture,picture);
                    }
                    else
                        tipl::downsampling(picture);
                }
                source_images.resize(tipl::shape<3>(uint32_t(picture.width()),uint32_t(picture.height()),slices_count));
                for(size_t j = 0;j < source_images.plane_size();++j)
                    for(size_t k = 0,pos = j;k < slices_count;++k,pos += source_images.plane_size())
                        source_images[pos] = float(picture[j][0]);
            }

            vs = handle->vs*(handle->dim.width())/(source_images.width());

            tipl::transformation_matrix<float>(arg_min,handle->dim,handle->vs,source_images.shape(),vs).to(to_slice);
            to_dif = tipl::inverse(to_slice);
            initial_LPS_nifti_srow(trans_to_mni,source_images.shape(),vs);
            has_transform = true;
        }
        else
        {
            QImage in = read_qimage(source_file_name.c_str(),error_msg);
            if(in.isNull())
            {
                error_msg = "cannot read picture";
                return false;
            }
            dim[0] = in.width();
            dim[1] = in.height();
            dim[2] = uint32_t(source_files.size());
            vs[0] = vs[1] = vs[2] = 1.0f;

            try{
                source_images.resize(dim);
            }
            catch(...)
            {
                error_msg = "Memory allocation failed. Please increase downsampling count";
                return false;
            }

            for(size_t file_index = 0;prog(file_index,dim[2]);++file_index)
            {
                QImage in = read_qimage(source_files[file_index].c_str(),error_msg);
                if(in.isNull())
                    return false;
                QImage buf = in.convertToFormat(QImage::Format_RGB32).mirrored();
                tipl::image<2,short> I(tipl::shape<2>(in.width(),in.height()));
                const uchar* ptr = buf.bits();
                for(size_t j = 0;j < I.size();++j,ptr += 4)
                    I[j] = *ptr;

                std::copy(I.begin(),I.end(),source_images.begin() +
                          long(file_index*source_images.plane_size()));
            }
            if(prog.aborted())
                return false;
            has_transform = true;
        }
    }
    // load and match demographics DB file
    if(source_images.empty() &&
       (tipl::ends_with(source_file_name,".db.fib.gz") ||
        tipl::ends_with(source_file_name,".db.fz")) ||
        tipl::ends_with(source_file_name,".dz"))
    {
        tipl::progress prog("open ",source_file_name);
        std::shared_ptr<fib_data> db_handle(new fib_data);
        if(!db_handle->load_from_file(source_file_name) || !db_handle->db.has_db())
        {
            error_msg = db_handle->error_msg;
            return false;
        }

        if(!handle->db.has_db() && !handle->db.demo.empty())
        {
            tipl::out() << "computing matched volume of " << db_handle->db.index_name << " using subject demographics: " << handle->db.demo;
            if(!db_handle->db.get_demo_matched_volume(handle->db.demo,source_images))
            {
                error_msg = db_handle->error_msg;
                return false;
            }
        }
        else
        {
            tipl::out() << "computing subject averaged volume of " << db_handle->db.index_name;
            db_handle->db.get_avg_volume(source_images);
        }

        if(!handle->mni2sub(source_images,db_handle->trans_to_mni))
        {
            error_msg = handle->error_msg;
            return false;
        }
        name += "." + db_handle->db.index_name;
        is_diffusion_space = true;
        has_transform = true;
    }


    // load nifti file
    if(source_images.empty() &&
       (QString(source_file_name.c_str()).endsWith("nii.gz") || QString(source_file_name.c_str()).endsWith("nii")))
    {
        tipl::io::gz_nifti nifti;
        prepare_idx(source_file_name.c_str(),nifti.input_stream);
        if(!nifti.open(source_file_name,std::ios::in) || !(nifti >> binded_image()))
        {
            error_msg = nifti.error_msg;
            return false;
        }
        tipl::out() << "slice dim: " << source_images.shape() << " vs: " << vs << " trans: " << trans_to_mni;

        save_idx(source_file_name.c_str(),nifti.input_stream);
        if(QFileInfo(source_file_name.c_str()).fileName().toLower().contains(".mni."))
        {
            tipl::out() << source_file_name << " has '.mni.' in the file name. It will be treated as mni space image.";
            is_mni = true;
        }

        if(handle->is_mni) // fib in the template space
        {
            tipl::out() << "header transformation used to align image." << std::endl;
            to_slice = tipl::inverse(to_dif = tipl::from_space(trans_to_mni).to(handle->trans_to_mni));
            has_transform = true;
            if(nifti.nif_header.sform_code < 4)
                tipl::warning() << source_file_name << " has sform_code < 4 suggesting not an MNI image. But, it is loaded into the QSDR's MNI space.";
        }
        else
        {
            if(source_images.shape() != handle->dim)
            {
                if(is_mni)
                {
                    tipl::out() << "warping template-space slices to the subject space." << std::endl;
                    if(!handle->mni2sub(source_images,trans_to_mni))
                    {
                        error_msg = handle->error_msg;
                        return false;
                    }
                    is_diffusion_space = true;
                    trans_to_mni = handle->trans_to_mni;
                    vs = handle->vs;
                    has_transform = true;
                }
                else
                if(nifti.nif_header.sform_code >= 4)
                    tipl::warning() << source_file_name << " has sform_code >= 4 suggesting MNI image. But, it is loaded as sujbect space image.";
            }
            else
            // slice and DWI have the same image size
            {
                if(QFileInfo(source_file_name.c_str()).fileName().contains("reg"))
                {
                    tipl::out() << "The slices have the same dimension, and there is 'reg' in the file name." << std::endl;
                    tipl::out() << "no registration needed." << std::endl;
                    is_diffusion_space = true;
                    trans_to_mni = handle->trans_to_mni;
                    has_transform = true;
                }
                else
                    tipl::out() << "registration will be applied even though the image size is identical. To disable registration, add 'reg' to the file name. " << std::endl;
            }
        }
    }

    // bruker images
    if(source_images.empty())
    {
        tipl::progress prog("open ",source_file_name);
        tipl::io::bruker_2dseq bruker;
        if(bruker.load_from_file(source_file_name.c_str()))
        {
            bruker.get_voxel_size(vs);
            source_images = std::move(bruker.get_image());
            initial_LPS_nifti_srow(trans_to_mni,source_images.shape(),vs);
            QDir d = QFileInfo(source_file_name.c_str()).dir();
            if(d.cdUp() && d.cdUp())
            {
                QString method_file_name = d.absolutePath()+ "/method";
                tipl::io::bruker_info method;
                if(method.load_from_file(method_file_name.toStdString().c_str()))
                    name = method["Method"];
            }
        }
    }

    // dicom images
    if(source_images.empty() && !source_files.empty())
    {
        tipl::progress prog("open ",source_file_name);
        tipl::io::dicom_volume volume;
        if(!volume.load_from_files(source_files))
        {
            error_msg = volume.error_msg;
            return false;
        }
        volume.get_voxel_size(vs);
        volume.save_to_image(source_images);
        if(source_images.empty())
        {
            error_msg = "failed to load image volume.";
            return false;
        }
        initial_LPS_nifti_srow(trans_to_mni,source_images.shape(),vs);
    }

    // add image to the view item lists
    {
        update_image();
        tipl::out() << "add new slices: " << name << std::endl;
        tipl::out() << "dimension: " << source_images.shape() << std::endl;
        if(source_images.shape() == handle->dim)
            tipl::out() << "The slices have the same dimension as DWI." << std::endl;
        view->set_image(tipl::make_image(source_images.data(),source_images.shape()));
        view->T = to_dif;
        view->iT = to_slice;
    }

    if(has_transform)
        return true;

    if(std::filesystem::exists(source_file_name+".linear_reg.txt"))
    {
        tipl::out() << "loading existing linear registration." << std::endl;
        if(!(load_mapping((source_file_name+".linear_reg.txt").c_str())))
        {
            tipl::error() << "invalid slice mapping file format" << std::endl;
            return false;
        }
        return true;
    }

    if(QFileInfo(source_file_name.c_str()).fileName().toLower().contains("reg"))
    {
        tipl::out() << "'reg' found in the file name. no registration applied." << std::endl;
        is_diffusion_space = true;
        trans_to_mni = handle->trans_to_mni;
        return true;
    }

    if(handle->dim.depth() < 10) // 2d assume FOV is the same
    {
        to_slice[0] = float(source_images.width())/float(handle->dim.width());
        to_slice[5] = float(source_images.height())/float(handle->dim.height());
        to_slice[10] = float(source_images.depth())/float(handle->dim.depth());
        to_slice[15] = 1.0;
        to_dif = tipl::inverse(to_slice);
        view->T = to_dif;
        view->iT = to_slice;
        return true;
    }

    // handle registration
    tipl::out() << "running rigid body transformation to the slices. To disable it, add 'reg' to the file name." << std::endl;
    run_registration();
    update_transform();
    return true;
}

void CustomSliceModel::run_registration(void)
{
    running = true;
    if(tipl::show_prog)
        thread.reset(new std::thread([this](){argmin();}));
    else
        argmin();
}
void CustomSliceModel::update_image(void)
{
    dim = source_images.shape();
    slice_pos[0] = source_images.width() >> 1;
    slice_pos[1] = source_images.height() >> 1;
    slice_pos[2] = source_images.depth() >> 1;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update_image(tipl::image<3>&& new_image)
{
    dim = new_image.shape();
    view->set_image(new_image);
    source_images.swap(new_image);
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update_transform(void)
{
    tipl::transformation_matrix<float> M(arg_min,dim,vs,handle->dim,handle->vs);
    M.to(to_dif);
    to_slice = tipl::inverse(to_dif);
    view->T = to_dif;
    view->iT = to_slice;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::argmin(void)
{
    terminated = false;
    view->registering = true;

    auto to = subject_image_pre(tipl::image<3>(source_images));
    auto to_vs = vs;
    auto from = handle->get_iso_fa();
    auto from_vs = handle->vs;
    auto from1 = subject_image_pre(tipl::image<3>(from.first));
    auto from2 = subject_image_pre(tipl::image<3>(from.second));

    if(picture.empty())
    {
        while(to_vs[0]*0.5f >= from_vs[0])
        {
            tipl::downsampling(to);
            to_vs *= 0.5f;
        }
    }

    tipl::out() << "registration started using " << (picture.empty() ? "rigid body with regular bound" : "affine transform with narrow bound");
    tipl::reg::linear<tipl::out>(tipl::reg::make_list(to,to),to_vs,tipl::reg::make_list(from1,from2),from_vs,
           arg_min,picture.empty() ? tipl::reg::rigid_body : tipl::reg::affine,terminated);
    update_transform();
    view->registering = false;
    running = false;
    tipl::out() << "registration completed";
}
// ---------------------------------------------------------------------------
bool CustomSliceModel::save_mapping(const std::string& file_name)
{
    return !!(std::ofstream(file_name) << arg_min);
}
// ---------------------------------------------------------------------------
bool CustomSliceModel::load_mapping(const std::string& file_name)
{
    std::ifstream in(file_name);
    if(!in)
        return false;
    if(in.peek() == 't')
    {
        if(!(std::ifstream(file_name) >> arg_min))
            return false;
    }
    else
    {
        tipl::transformation_matrix<float> T;
        if(!(in >> T))
            return false;
        arg_min = T.to_affine_transform(dim,vs,handle->dim,handle->vs);
    }
    update_transform();
    is_diffusion_space = false;
    tipl::out() << arg_min << std::endl;
    tipl::out() << "to_dif: " << to_dif << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
void CustomSliceModel::wait(void)
{
    if(!thread.get())
        return;
    if(thread->joinable())
        thread->join();
    tipl::out() << "size: " << dim << " vs: " << vs;
    tipl::out() << "srow: " << trans_to_mni;
    tipl::out() << "to_dif: " << to_dif;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::terminate(void)
{
    terminated = true;
    running = false;
    wait();
}
// ---------------------------------------------------------------------------
