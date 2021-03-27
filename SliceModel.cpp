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

SliceModel::SliceModel(fib_data* handle_,uint32_t view_id_):handle(handle_),view_id(view_id_)
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
    v2c.set_range(handle->view_item[view_id].contrast_min,handle->view_item[view_id].contrast_max);
    v2c.two_color(handle->view_item[view_id].min_color,handle->view_item[view_id].max_color);
}
// ---------------------------------------------------------------------------
void SliceModel::apply_overlay(tipl::color_image& show_image,
                    unsigned char cur_dim,
                    std::shared_ptr<SliceModel> other_slice) const
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
            if((value > 0.0f && value > range.first) ||
               (value < 0.0f && value < range.second))
                show_image[pos] |= other_slice->v2c[value];
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
    v2c.set_range(min_v,max_v);
}
// ---------------------------------------------------------------------------
void SliceModel::set_contrast_color(unsigned int min_c,unsigned int max_c)
{
    handle->view_item[view_id].min_color = min_c;
    handle->view_item[view_id].max_color = max_c;
    v2c.two_color(min_c,max_c);
}
// ---------------------------------------------------------------------------
void SliceModel::get_slice(tipl::color_image& show_image,unsigned char cur_dim,
                           const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    handle->get_slice(view_id,cur_dim, slice_pos[cur_dim],show_image,v2c);
    for(auto overlay_slice : overlay_slices)
        if(this != overlay_slice.get())
            apply_overlay(show_image,cur_dim,overlay_slice);
}
// ---------------------------------------------------------------------------
tipl::const_pointer_image<float, 3> SliceModel::get_source(void) const
{
    return handle->view_item[view_id].image_data;
}

// ---------------------------------------------------------------------------
CustomSliceModel::CustomSliceModel(fib_data* new_handle):
    SliceModel(new_handle,0)
{
    new_handle->view_item.push_back(item());
}
// ---------------------------------------------------------------------------
void CustomSliceModel::initialize(void)
{
    if(!handle->view_item.empty())
        view_id = uint32_t(handle->view_item.size()-1);
    geometry = source_images.geometry();
    handle->view_item.back().image_data = tipl::make_image(&*source_images.begin(),source_images.geometry());
    handle->view_item.back().set_scale(source_images.begin(),source_images.end());
    handle->view_item.back().name = name;
    handle->view_item.back().T = T;
    handle->view_item.back().iT = invT;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
    v2c.set_range(handle->view_item[view_id].contrast_min,handle->view_item[view_id].contrast_max);
    v2c.two_color(handle->view_item[view_id].min_color,handle->view_item[view_id].max_color);
}
// ---------------------------------------------------------------------------
void CustomSliceModel::get_slice(tipl::color_image& image,
                           unsigned char dim,
                           const std::vector<std::shared_ptr<SliceModel> >& overlay_slices) const
{
    if(picture.empty())
        return SliceModel::get_slice(image,dim,overlay_slices);
    image = picture;
}
// ---------------------------------------------------------------------------
void initial_LPS_nifti_srow(tipl::matrix<4,4,float>& T,const tipl::geometry<3>& geo,const tipl::vector<3>& vs);
bool CustomSliceModel::initialize(const std::vector<std::string>& files)
{
    if(files.empty())
        return false;
    terminated = true;
    ended = true;
    is_diffusion_space = false;

    gz_nifti nifti;
    // QSDR loaded, use MNI transformation instead
    bool has_transform = false;
    from = tipl::make_image(handle->dir.fa[0],handle->dim);
    from_vs = handle->vs;
    source_file_name = files[0].c_str();
    name = QFileInfo(files[0].c_str()).completeBaseName().remove(".nii").toStdString();

    if(QFileInfo(files[0].c_str()).suffix() == "bmp" ||
       QFileInfo(files[0].c_str()).suffix() == "jpg" ||
       QFileInfo(files[0].c_str()).suffix() == "png")

    {
        QString info_file = QString(files[0].c_str()) + ".info.txt";
        if(!QFileInfo(info_file).exists())
        {
            if(files.size() != 1)
            {
                error_msg = "multiple jpg/bmp/png files are not supported";
                return false;
            }
            QImage in;
            if(!in.load(files[0].c_str()))
            {
                error_msg = "invalid image format: ";
                error_msg += files[0];
                return false;
            }
            QImage buf = in.convertToFormat(QImage::Format_RGB32);
            picture.resize(tipl::geometry<2>(uint32_t(in.width()),uint32_t(in.height())));
            source_images.resize(tipl::geometry<3>(uint32_t(in.width()),uint32_t(in.height()),1));
            const uchar* ptr = buf.bits();
            for(size_t j = 0;j < source_images.size();++j,ptr += 4)
            {
                source_images[j] = float(*ptr);
                picture[j] = tipl::rgb(*(ptr+2),*(ptr+1),*ptr);
            }

            voxel_size = handle->vs*0.5f*(handle->dim.width())/(in.width());
            tipl::transformation_matrix<float> M(arg_min,handle->dim,handle->vs,source_images.geometry(),voxel_size);
            invT.identity();
            M.save_to_transform(invT.begin());
            T = tipl::inverse(invT);
            has_transform = true;
        }
        else
        {
            std::ifstream in(info_file.toStdString().c_str());
            in >> geometry[0];
            in >> geometry[1];
            in >> geometry[2];
            in >> voxel_size[0];
            in >> voxel_size[1];
            in >> voxel_size[2];
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),T.begin());
            if(geometry[2] != uint32_t(files.size()))
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
                    error_msg = "invalid image format: ";
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
                std::copy(I.begin(),I.end(),source_images.begin() + long(i*source_images.plane_size()));
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
            initial_LPS_nifti_srow(trans,source_images.geometry(),voxel_size);
            has_transform = true;
        }
    }
    else
    {
        if(files.size() == 1)
        {
            if(nifti.load_from_file(files[0]))
            {
                nifti.toLPS(source_images);
                nifti.get_voxel_size(voxel_size);
                nifti.get_image_transformation(trans);
                if(handle->is_qsdr || handle->is_mni_image)
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
                    initial_LPS_nifti_srow(trans,source_images.geometry(),voxel_size);
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
                initial_LPS_nifti_srow(trans,source_images.geometry(),voxel_size);
            }
        }
    }




    if(source_images.empty())
    {
        error_msg = "failed to load image volume.";
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

    {
        std::string mapping_file = source_file_name+".mapping.txt";
        if(QFileInfo(mapping_file.c_str()).exists())
        {
            load_mapping(mapping_file.c_str());
            has_transform = true;
        }
    }

    if(!has_transform)
    {
        if(handle->dim.depth() < 10) // 2d assume FOV is the same
        {
            T.identity();
            invT.identity();
            invT[0] = float(source_images.width())/float(handle->dim.width());
            invT[5] = float(source_images.height())/float(handle->dim.height());
            invT[10] = float(source_images.depth())/float(handle->dim.depth());
            invT[15] = 1.0;
            T = tipl::inverse(invT);
        }
        else
        {
            size_t iso = handle->get_name_index("iso");
            if(handle->view_item.size() != iso)
                from = handle->view_item[iso].image_data;
            size_t base_nqa = handle->get_name_index("base_nqa");
            if(handle->view_item.size() != base_nqa)
                from = handle->view_item[base_nqa].image_data;

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
    tipl::transformation_matrix<float> M;


    tipl::affine_transform_2d<float> a;
    tipl::transformation_matrix_2d<float> m;

    // assume brain top
    float z_shift = (float(from.geometry()[2])*from_vs[2]-float(to.geometry()[2])*voxel_size[2])*0.1f;
    arg_min.translocation[2] = -z_shift*voxel_size[2];

    tipl::reg::two_way_linear_mr(from,from_vs,to,voxel_size,M,reg_type,tipl::reg::mutual_information(),terminated,
                                  std::thread::hardware_concurrency(),&arg_min);

    ended = true;
    M.save_to_transform(invT.begin());
    handle->view_item[view_id].T = T = tipl::inverse(invT);
    handle->view_item[view_id].iT = invT;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::save_mapping(const char* file_name)
{
    std::ofstream out(file_name);
    for(uint32_t row = 0,index = 0;row < 4;++row)
    {
        for(uint32_t col = 0;col < 4;++col,++index)
        {
            if(col)
                out << " ";
            out << T[index];
        }
        out << std::endl;
    }
    for(uint32_t i = 0;i < 12;++i)
    {
        if(i)
            out << " ";
        out << arg_min.data[i];
    }
    out << std::endl;
}// ---------------------------------------------------------------------------
void CustomSliceModel::load_mapping(const char* file_name)
{
    std::ifstream in(file_name);
    if(!in)
        return;
    std::vector<float> data,arg;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),std::back_inserter(data));
    if(data.size() == 28)
    {
        std::copy(data.begin()+16,data.begin()+16+12,arg_min.data);
        update();
    }
    else
    {
        data.resize(16);
        data[15] = 1.0;
        T = data;
        invT = data;
        invT.inv();
        handle->view_item[view_id].T = T;
        handle->view_item[view_id].iT = invT;
    }
}
void CustomSliceModel::update(void)
{
    tipl::transformation_matrix<float> M(arg_min,from.geometry(),from_vs,source_images.geometry(),voxel_size);
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
