// ---------------------------------------------------------------------------
#include <string>
#include <QFileInfo>
#include <QImage>
#include <QInputDialog>
#include <QMessageBox>
#include <QDir>
#include "SliceModel.h"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"

SliceModel::SliceModel(std::shared_ptr<fib_data> handle_,int view_id_):handle(handle_),view_id(view_id_)
{
    slice_visible[0] = false;
    slice_visible[1] = false;
    slice_visible[2] = false;
    T.identity();
    invT = T;
    geometry = handle_->dim;
    voxel_size = handle_->vs;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
}
// ---------------------------------------------------------------------------
void SliceModel::apply_overlay(tipl::color_image& show_image,
                    unsigned char cur_dim,
                    const SliceModel* other_slice,
                    const tipl::value_to_color<float>& overlay_v2c) const
{
    std::pair<float,float> range = other_slice->get_contrast_range();
    for(int y = 0,pos = 0;y < show_image.height();++y)
        for(int x = 0;x < show_image.width();++x,++pos)
        {
            tipl::vector<3,float> v;
            toOtherSlice(other_slice,cur_dim,x,y,v);
            float value = 0;
            if(!tipl::estimate(other_slice->get_source(),v,value))
                continue;
            if(value > range.first)
                show_image[pos] = overlay_v2c[value];
        }
}


// ---------------------------------------------------------------------------
std::pair<float,float> SliceModel::get_value_range(void) const
{
    return std::make_pair(handle->view_item[view_id].min_value,handle->view_item[view_id].max_value);
}
// ---------------------------------------------------------------------------
std::pair<float,float> SliceModel::get_contrast_range(void) const
{
    return std::make_pair(handle->view_item[view_id].contrast_min,handle->view_item[view_id].contrast_max);
}
// ---------------------------------------------------------------------------
std::pair<unsigned int,unsigned int> SliceModel::get_contrast_color(void) const
{
    return std::make_pair(handle->view_item[view_id].min_color,handle->view_item[view_id].max_color);
}
// ---------------------------------------------------------------------------
void SliceModel::set_contrast_range(float min_v,float max_v)
{
    handle->view_item[view_id].contrast_min = min_v;
    handle->view_item[view_id].contrast_max = max_v;
}
// ---------------------------------------------------------------------------
void SliceModel::set_contrast_color(unsigned int min_c,unsigned int max_c)
{
    handle->view_item[view_id].min_color = min_c;
    handle->view_item[view_id].max_color = max_c;
}
// ---------------------------------------------------------------------------
void SliceModel::get_slice(tipl::color_image& show_image,unsigned char cur_dim,
                              const tipl::value_to_color<float>& v2c,
                              const SliceModel* overlay,
                              const tipl::value_to_color<float>& overlay_v2c) const
{
    handle->get_slice(view_id,cur_dim, slice_pos[cur_dim],show_image,v2c);
    if(overlay && this != overlay)
        apply_overlay(show_image,cur_dim,overlay,overlay_v2c);
}
// ---------------------------------------------------------------------------
tipl::const_pointer_image<float, 3> SliceModel::get_source(void) const
{
    return handle->view_item[view_id].image_data;
}

// ---------------------------------------------------------------------------
CustomSliceModel::CustomSliceModel(std::shared_ptr<fib_data> new_handle):
    SliceModel(new_handle,new_handle->view_item.size())
{
    new_handle->view_item.push_back(item());
}
// ---------------------------------------------------------------------------
void CustomSliceModel::initialize(void)
{
    geometry = source_images.geometry();
    handle->view_item.back().image_data = tipl::make_image(&*source_images.begin(),source_images.geometry());
    handle->view_item.back().set_scale(source_images.begin(),source_images.end());
    handle->view_item.back().name = name;
    handle->view_item.back().T = T;
    handle->view_item.back().iT = invT;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
}
// ---------------------------------------------------------------------------
bool CustomSliceModel::initialize(const std::vector<std::string>& files,
                                  bool correct_intensity)
{
    terminated = true;
    ended = true;
    is_diffusion_space = false;

    gz_nifti nifti;
    // QSDR loaded, use MNI transformation instead
    bool has_transform = false;
    name = QFileInfo(files[0].c_str()).completeBaseName().toStdString();
    if(QFileInfo(files[0].c_str()).suffix() == "bmp" ||
            QFileInfo(files[0].c_str()).suffix() == "jpg")

    {
        QString info_file = QString(files[0].c_str()) + ".info.txt";
        if(!QFileInfo(info_file).exists())
        {
            error_msg = "Cannot find ";
            error_msg += info_file.toStdString();
            return false;
        }
        std::ifstream in(info_file.toStdString().c_str());
        in >> geometry[0];
        in >> geometry[1];
        in >> geometry[2];
        in >> voxel_size[0];
        in >> voxel_size[1];
        in >> voxel_size[2];
        std::copy(std::istream_iterator<float>(in),
                  std::istream_iterator<float>(),T.begin());
        if(geometry[2] != int(files.size()))
        {
            error_msg = "Invalid BMP info text: file count does not match.";
            return false;
        }
        unsigned int in_plane_subsample = 1;
        unsigned int slice_subsample = 1;

        // non isotropic condition
        while(voxel_size[2]/voxel_size[0] > 1.5f)
        {
            ++in_plane_subsample;
            geometry[0] = geometry[0] >> 1;
            geometry[1] = geometry[1] >> 1;
            voxel_size[0] *= 2.0f;
            voxel_size[1] *= 2.0f;
            T[0] *= 2.0f;
            T[1] *= 2.0f;
            T[4] *= 2.0f;
            T[5] *= 2.0f;
            T[8] *= 2.0f;
            T[9] *= 2.0f;
        }
        tipl::geometry<3> geo(geometry);

        bool ok = true;
        int down_size = (geo[2] > 1 ? QInputDialog::getInt(nullptr,"DSI Studio",
                "Downsampling count (0:no downsampling)",1,0,4,1,&ok) : 0);
        if(!ok)
        {
            error_msg = "Slice loading canceled";
            return false;
        }
        for(int i = 0;i < down_size;++i)
        {
            geo[0] = geo[0] >> 1;
            geo[1] = geo[1] >> 1;
            geo[2] = geo[2] >> 1;
            voxel_size *= 2.0;
            tipl::multiply_constant(T.begin(),T.begin()+3,2.0);
            tipl::multiply_constant(T.begin()+4,T.begin()+7,2.0);
            tipl::multiply_constant(T.begin()+8,T.begin()+11,2.0);
            ++in_plane_subsample;
            ++slice_subsample;
        }


        try{
            source_images.resize(geo);
        }
        catch(...)
        {
            error_msg = "Memory allocation failed. Please increase downsampling count";
            return false;
        }

        begin_prog("loading images");
        for(unsigned int i = 0;check_prog(i,geo[2]);++i)
        {
            tipl::image<short,2> I;
            QImage in;
            unsigned int file_index = (slice_subsample == 1 ? i : (i << (slice_subsample-1)));
            if(file_index >= files.size())
                break;
            QString filename(files[file_index].c_str());
            if(!in.load(filename))
            {
                error_msg = "Invalid image format: ";
                error_msg += files[file_index];
                return false;
            }
            QImage buf = in.convertToFormat(QImage::Format_RGB32).mirrored();
            I.resize(tipl::geometry<2>(in.width(),in.height()));
            const uchar* ptr = buf.bits();
            for(size_t j = 0;j < I.size();++j,ptr += 4)
                I[j] = *ptr;

            for(size_t j = 1;j < in_plane_subsample;++j)
                tipl::downsampling(I);
            if(I.size() != source_images.plane_size())
            {
                error_msg = "Invalid BMP image size: ";
                error_msg += files[file_index];
                return false;
            }
            std::copy(I.begin(),I.end(),source_images.begin() + size_t(i)*source_images.plane_size());
        }
        if(prog_aborted())
            return false;
        tipl::io::nifti nii;
        nii.set_dim(geo);
        nii.set_voxel_size(voxel_size);
        nii.set_image_transformation(T);
        nii << source_images;
        nii.toLPS(source_images);
        nii.get_voxel_size(voxel_size);
        nii.get_image_transformation(T);
        // LPS matrix switched to RAS

        T[0] = -T[0];
        T[1] = -T[1];
        T[4] = -T[4];
        T[5] = -T[5];
        T[8] = -T[8];
        T[9] = -T[9];
        invT = tipl::inverse(T);
        has_transform = true;
    }
    else
    {
        if(files.size() == 1)
        {
            if(nifti.load_from_file(files[0]))
            {
                nifti.toLPS(source_images);
                nifti.get_voxel_size(voxel_size);
                if(handle->is_qsdr)
                {
                    nifti.get_image_transformation(invT);
                    invT.inv();
                    invT *= handle->trans_to_mni;
                    T = tipl::inverse(invT);
                    has_transform = true;
                }
            }
            else
            {
                tipl::io::bruker_2dseq bruker;
                if(bruker.load_from_file(files[0].c_str()))
                {
                    bruker.get_voxel_size(voxel_size);
                    source_images = std::move(bruker.get_image());
                    QDir d = QFileInfo(files[0].c_str()).dir();
                    if(d.cdUp() && d.cdUp())
                    {
                        QString method_file_name = d.absolutePath()+ "/method";
                        tipl::io::bruker_info method;
                        if(method.load_from_file(method_file_name.toStdString().c_str()))
                            name = method["Method"];
                    }
                }
            }
        }
        else
        {
            tipl::io::volume volume;
            if(volume.load_from_files(files,files.size()))
            {
                volume.get_voxel_size(voxel_size);
                volume >> source_images;
            }
        }
    }




    if(source_images.empty())
    {
        error_msg = "Failed to load image volume.";
        return false;
    }
    // same dimension, no registration required.
    if(source_images.geometry() == handle->dim)
    {
        T.identity();
        invT.identity();
        is_diffusion_space = true;
        has_transform = true;
    }

    // quality control for t1w
    if(correct_intensity)
    {
        float t = tipl::segmentation::otsu_threshold(source_images);
        float snr = tipl::mean(source_images.begin()+source_images.width(),source_images.begin()+2*source_images.width());
        // correction for SNR
        for(unsigned int i = 0;i < 6 && snr != 0 && t/snr < 10;++i)
        {
            tipl::filter::gaussian(source_images);
            t = tipl::segmentation::otsu_threshold(source_images);
            snr = tipl::mean(source_images.begin()+source_images.width(),source_images.begin()+2*source_images.width());
        }

        // correction for intensity bias
        t = tipl::segmentation::otsu_threshold(source_images);
        std::vector<float> x,y;
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            x.clear();
            y.clear();
            for(tipl::pixel_index<3> i(source_images.geometry());i < source_images.size();++i)
            if(source_images[i.index()] > t)
            {
                x.push_back(i[dim]);
                y.push_back(source_images[i.index()]);
            }
            std::pair<double,double> r = tipl::linear_regression(x.begin(),x.end(),y.begin());
            for(tipl::pixel_index<3> i(source_images.geometry());i < source_images.size();++i)
                source_images[i.index()] -= (float)i[dim]*r.first;
            tipl::lower_threshold(source_images,0);
        }
    }

    if(!has_transform)
    {
        if(handle->dim.depth() < 10) // 2d assume FOV is the same
        {
            T.identity();
            invT.identity();
            invT[0] = (float)source_images.width()/(float)handle->dim.width();
            invT[5] = (float)source_images.height()/(float)handle->dim.height();
            invT[10] = (float)source_images.depth()/(float)handle->dim.depth();
            invT[15] = 1.0;
            T = tipl::inverse(invT);
        }
        else
        {
            from = tipl::make_image(handle->dir.fa[0],handle->dim);
            size_t iso = handle->get_name_index("iso");// for DDI
            if(handle->view_item.size() != iso)
                from = handle->view_item[iso].image_data;
            size_t base_nqa = handle->get_name_index("base_nqa");// for DDI
            if(handle->view_item.size() != base_nqa)
                from = handle->view_item[base_nqa].image_data;
            from_vs = handle->vs;
            thread.reset(new std::future<void>(
                             std::async(std::launch::async,[this](){argmin(tipl::reg::rigid_body);})));
        }
    }
    initialize();
    return true;
}

// ---------------------------------------------------------------------------
void CustomSliceModel::argmin(tipl::reg::reg_type reg_type)
{
    terminated = false;
    ended = false;
    tipl::const_pointer_image<float,3> to = source_images;
    tipl::transformation_matrix<double> M;
    tipl::reg::two_way_linear_mr(from,from_vs,to,voxel_size,M,reg_type,tipl::reg::mutual_information(),terminated,
                                  std::thread::hardware_concurrency(),&arg_min);
    ended = true;
    M.save_to_transform(invT.begin());
    handle->view_item[view_id].T = T = tipl::inverse(invT);
    handle->view_item[view_id].iT = invT;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update(void)
{
    tipl::transformation_matrix<double> M(arg_min,from.geometry(),from_vs,source_images.geometry(),voxel_size);
    invT.identity();
    M.save_to_transform(invT.begin());
    handle->view_item[view_id].T = T = tipl::inverse(invT);
    handle->view_item[view_id].iT = invT;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::terminate(void)
{
    terminated = true;
    if(thread.get())
        thread->wait();
    ended = true;
}
// ---------------------------------------------------------------------------
extern std::string t1w_template_file_name,t1w_mask_template_file_name;
bool CustomSliceModel::stripskull(void)
{
    if(handle->is_human_data)
    {
        gz_nifti in1,in2;
        tipl::image<float,3> It,Iw;
        if(!in1.load_from_file(t1w_template_file_name.c_str()) || !in1.toLPS(It))
        {
            error_msg = "Cannot load T1W template";
            return false;
        }
        if(!in2.load_from_file(t1w_mask_template_file_name.c_str()) || !in2.toLPS(Iw))
        {
            error_msg = "Cannot load T1W mask";
            return false;
        }
        tipl::vector<3> Itvs;
        in1.get_voxel_size(Itvs);
        bool terminated = false;
        tipl::transformation_matrix<double> T,iT;
        tipl::reg::two_way_linear_mr(It,Itvs,source_images,voxel_size,T,
                                     tipl::reg::affine,tipl::reg::mutual_information(),
                                     terminated,std::thread::hardware_concurrency(),0);
        iT = T;
        iT.inverse();
        tipl::image<float,3> J(It.geometry());
        tipl::resample_mt(source_images,J,T,tipl::cubic);
        tipl::image<tipl::vector<3>,3> dis;
        tipl::reg::cdm(It,J,dis,terminated);
        source_images.for_each_mt([&](float& v,const tipl::pixel_index<3>& p){
            tipl::vector<3> pos(p);
            iT(pos);
            if(!dis.geometry().is_valid(pos[0],pos[1],pos[2]))
            {
                v = 0;
                return;
            }
            tipl::vector<3> d;
            if(!tipl::estimate(dis,pos,d))
                return;
            pos += d;
            pos.round();
            tipl::pixel_index<3> p2(pos[0],pos[1],pos[2],Iw.geometry());
            if(!Iw.geometry().is_valid(p2) || Iw[p2.index()] < 0.05 )
                v = 0;
        });
        return true;
    }
    else
    {
        tipl::image<float,3> filter(source_images.geometry());
        tipl::resample(from,filter,T,tipl::linear);
        tipl::filter::gaussian(filter);
        tipl::filter::gaussian(filter);
        float m = *std::max_element(source_images.begin(),source_images.end());
        source_images *= filter;
        tipl::normalize(source_images,m);
        return true;

    }

}
