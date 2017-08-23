// ---------------------------------------------------------------------------
#include <string>
#include <QFileInfo>
#include <QDir>
#include "SliceModel.h"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"
#include "fa_template.hpp"

// ---------------------------------------------------------------------------
SliceModel::SliceModel(void)
{
    slice_visible[0] = false;
    slice_visible[1] = false;
    slice_visible[2] = false;
    std::fill(transform.begin(),transform.end(),0.0);
    transform[0] = transform[5] = transform[10] = transform[15] = 1.0;
    invT = transform;
}
// ---------------------------------------------------------------------------
void SliceModel::get_mosaic(image::color_image& show_image,
                               unsigned int mosaic_size,
                               const image::value_to_color<float>& v2c,
                               unsigned int skip)
{
    unsigned slice_num = geometry[2] / skip;
    show_image.clear();
    show_image.resize(image::geometry<2>(geometry[0]*mosaic_size,
                                          geometry[1]*(std::ceil((float)slice_num/(float)mosaic_size))));
    int old_z = slice_pos[2];
    for(unsigned int z = 0;z < slice_num;++z)
    {
        slice_pos[2] = z*skip;
        image::color_image slice_image;
        get_slice(slice_image,2,v2c);
        image::vector<2,int> pos(geometry[0]*(z%mosaic_size),
                                 geometry[1]*(z/mosaic_size));
        image::draw(slice_image,show_image,pos);
    }
    slice_pos[2] = old_z;
}

FibSliceModel::FibSliceModel(std::shared_ptr<fib_data> handle_,int view_id_):handle(handle_),view_id(view_id_)
{
    // already setup the geometry and source image
    geometry = handle_->dim;
    voxel_size = handle_->vs;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
}
// ---------------------------------------------------------------------------
std::pair<float,float> FibSliceModel::get_value_range(void) const
{
    return std::make_pair(handle->view_item[view_id].min_value,handle->view_item[view_id].max_value);
}
// ---------------------------------------------------------------------------
void FibSliceModel::get_slice(image::color_image& show_image,unsigned char cur_dim,const image::value_to_color<float>& v2c) const
{
    handle->get_slice(view_id,cur_dim, slice_pos[cur_dim],show_image,v2c);
}
// ---------------------------------------------------------------------------
image::const_pointer_image<float, 3> FibSliceModel::get_source(void) const
{
    return handle->view_item[view_id].image_data;
}


// ---------------------------------------------------------------------------
void CustomSliceModel::init(void)
{
    geometry = source_images.geometry();
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
    min_value = *std::min_element(source_images.begin(),source_images.end());
    max_value = *std::max_element(source_images.begin(),source_images.end());
    scale = max_value-min_value;
    if(scale != 0.0)
        scale = 255.0/scale;
}
bool CustomSliceModel::initialize(std::shared_ptr<fib_data> handle,bool is_qsdr,
                                  const std::vector<std::string>& files,
                                  bool correct_intensity)
{
    terminated = true;
    ended = true;
    is_diffusion_space = false;

    gz_nifti nifti;
    // QSDR loaded, use MNI transformation instead
    bool has_transform = false;
    name = QFileInfo(files[0].c_str()).completeBaseName().toStdString();
    if(is_qsdr && files.size() == 1 && nifti.load_from_file(files[0]))
    {
        loadLPS(nifti);
        invT.identity();
        nifti.get_image_transformation(invT.begin());
        invT.inv();
        invT *= handle->trans_to_mni;
        transform = image::inverse(invT);
        has_transform = true;
    }
    else
    {
        if(files.size() == 1 && nifti.load_from_file(files[0]))
            loadLPS(nifti);
        else
        {
            if(files.size() == 1)
            {
                image::io::bruker_2dseq bruker;
                if(bruker.load_from_file(files[0].c_str()))
                {
                    bruker.get_voxel_size(voxel_size.begin());
                    bruker.get_image().swap(source_images);
                    init();
                    QDir d = QFileInfo(files[0].c_str()).dir();
                    if(d.cdUp() && d.cdUp())
                    {
                        QString method_file_name = d.absolutePath()+ "/method";
                        image::io::bruker_info method;
                        if(method.load_from_file(method_file_name.toStdString().c_str()))
                            name = method["Method"];
                    }
                }
            }
            else
                if(QFileInfo(files[0].c_str()).completeSuffix() == "bmp")
                {
                    QString info_file = QString(files[0].c_str()) + ".info.txt";
                    if(!QFileInfo(info_file).exists())
                    {
                        std::cout << "Cannot find " << info_file.toStdString() << std::endl;
                        return false;
                    }
                    std::ifstream in(info_file.toStdString().c_str());
                    in >> geometry[0];
                    in >> geometry[1];
                    in >> geometry[2];
                    in >> voxel_size[0];
                    in >> voxel_size[1];
                    in >> voxel_size[2];
                    std::vector<float> T;
                    std::copy(std::istream_iterator<float>(in),
                              std::istream_iterator<float>(),std::back_inserter(T));
                    if(T.size() != 12)
                    {
                        std::cout << "Invalid BMP info text: failed to read transformation matrix." << std::endl;
                        return false;
                    }
                    if(geometry[2] != files.size())
                    {
                        std::cout << "Invalid BMP info text: file count does not match." << std::endl;
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
                        voxel_size[0] *= 2.0;
                        voxel_size[1] *= 2.0;
                        T[0] *= 2.0;
                        T[1] *= 2.0;
                        T[4] *= 2.0;
                        T[5] *= 2.0;
                        T[8] *= 2.0;
                        T[9] *= 2.0;
                    }
                    image::geometry<3> geo(geometry);

                    bool can_allocate = true;
                    do
                    {
                        try{
                            image::basic_image<float, 3> buf;
                            can_allocate = true;
                            buf.resize(geo);
                            buf.swap(source_images);
                        }
                        catch(...)
                        {
                            geo[0] = geo[0] >> 1;
                            geo[1] = geo[1] >> 1;
                            geo[2] = geo[2] >> 1;
                            voxel_size *= 2.0;
                            image::multiply_constant(T.begin(),T.begin()+3,2.0);
                            image::multiply_constant(T.begin()+4,T.begin()+7,2.0);
                            image::multiply_constant(T.begin()+8,T.begin()+11,2.0);
                            ++in_plane_subsample;
                            ++slice_subsample;
                            can_allocate = false;
                        }
                    }
                    while(!can_allocate);

                    begin_prog("loading images");

                    for(unsigned int i = 0;check_prog(i,geo[2]);++i)
                    {
                        image::basic_image<short,2> I;
                        image::io::bitmap bmp;
                        unsigned int file_index = (slice_subsample == 1 ? i : (i << (slice_subsample-1)));
                        if(file_index >= files.size())
                            break;
                        if(!bmp.load_from_file(files[file_index].c_str()))
                        {
                            std::cout << "Invalid BMP format: " << files[file_index] << std::endl;
                            return false;
                        }
                        bmp >> I;
                        for(int j = 1;j < in_plane_subsample;++j)
                            image::downsampling(I);
                        if(I.size() != source_images.plane_size())
                        {
                            std::cout << "Invalid BMP image size: " << files[file_index] << std::endl;
                            return false;
                        }
                        std::copy(I.begin(),I.end(),source_images.begin() + i*source_images.plane_size());
                    }
                    if(prog_aborted())
                        return false;
                    image::io::nifti nii;
                    nii.set_dim(geo);
                    nii.set_voxel_size(voxel_size.begin());
                    nii.set_image_transformation(T.begin());
                    nii << source_images;
                    nii.toLPS(source_images);
                    nii.get_voxel_size(voxel_size.begin());
                    transform.identity();
                    nii.get_image_transformation(transform.begin());
                    // LPS matrix switched to RAS
                    transform[0] = -transform[0];
                    transform[1] = -transform[1];
                    transform[4] = -transform[4];
                    transform[5] = -transform[5];
                    transform[8] = -transform[8];
                    transform[9] = -transform[9];
                    invT = image::inverse(transform);
                    has_transform = true;
                }
            else
                {
                image::io::volume volume;
                if(volume.load_from_files(files,files.size()))
                    load(volume);
                else
                    return false;
            }


        }
        // same dimension, no registration required.
        if(source_images.geometry() == handle->dim)
        {
            transform.identity();
            invT.identity();
            is_diffusion_space = true;
            has_transform = true;
        }
        // same dimension different resolution, no registrationrequired
        float r = std::round((float)source_images.width()/(float)handle->dim[0]);
        if(r > 1.0 && r == std::round(source_images.height()/handle->dim[1]) &&
                r == std::round(source_images.depth()/handle->dim[2]))
        {
            transform.identity();
            invT.identity();
            invT[0] = r;
            invT[5] = r;
            invT[10] = r;
            invT[15] = 1.0;
            transform = image::inverse(invT);
            has_transform = true;
        }

    }

    // quality control for t1w
    if(correct_intensity)
    {
        float t = image::segmentation::otsu_threshold(source_images);
        float snr = image::mean(source_images.begin()+source_images.width(),source_images.begin()+2*source_images.width());
        // correction for SNR
        for(unsigned int i = 0;i < 6 && snr != 0 && t/snr < 10;++i)
        {
            image::filter::gaussian(source_images);
            t = image::segmentation::otsu_threshold(source_images);
            snr = image::mean(source_images.begin()+source_images.width(),source_images.begin()+2*source_images.width());
        }

        // correction for intensity bias
        t = image::segmentation::otsu_threshold(source_images);
        std::vector<float> x,y;
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            x.clear();
            y.clear();
            for(image::pixel_index<3> i(source_images.geometry());i < source_images.size();++i)
            if(source_images[i.index()] > t)
            {
                x.push_back(i[dim]);
                y.push_back(source_images[i.index()]);
            }
            std::pair<double,double> r = image::linear_regression(x.begin(),x.end(),y.begin());
            for(image::pixel_index<3> i(source_images.geometry());i < source_images.size();++i)
                source_images[i.index()] -= (float)i[dim]*r.first;
            image::lower_threshold(source_images,0);
        }
    }

    geometry = source_images.geometry();
    if(!has_transform)
    {
        if(handle->dim.depth() < 10) // 2d assume FOV is the same
        {
            transform.identity();
            invT.identity();
            invT[0] = (float)source_images.width()/(float)handle->dim.width();
            invT[5] = (float)source_images.height()/(float)handle->dim.height();
            invT[10] = (float)source_images.depth()/(float)handle->dim.depth();
            invT[15] = 1.0;
            transform = image::inverse(invT);
        }
        else
        {
            from = image::make_image(handle->dir.fa[0],handle->dim);
            from_vs = handle->vs;
            thread.reset(new std::future<void>(
                             std::async(std::launch::async,[this](){argmin(image::reg::rigid_body);})));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
std::pair<float,float> CustomSliceModel::get_value_range(void) const
{
    return image::min_max_value(source_images.begin(),source_images.end());
}
// ---------------------------------------------------------------------------
void CustomSliceModel::argmin(image::reg::reg_type reg_type)
{
    terminated = false;
    ended = false;
    size_ratio = 1.0f;

    /*
    if(from_vs[0]/voxel_size[0] > 2.0f)
    {
        source_buf_vs = voxel_size;
        source_buf.clear();
        while(from_vs[0]/source_buf_vs[0] > 2.0f)
        {
            source_buf_vs *= 2.0;
            size_ratio *= 2.0;
            if(source_buf.empty())
                image::downsampling(source_images,source_buf);
            else
                image::downsampling(source_buf);
        }
        image::const_pointer_image<float,3> to = source_buf;
        image::reg::linear_mr(from,from_vs,to,source_buf_vs,arg_min,reg_type,image::reg::mutual_information_mt(),terminated);
    }
    else*/
    {
        image::const_pointer_image<float,3> to = source_images;
        image::reg::linear_mr(from,from_vs,to,voxel_size,arg_min,reg_type,image::reg::mutual_information(),terminated,0.1);
        image::reg::linear_mr(from,from_vs,to,voxel_size,arg_min,reg_type,image::reg::mutual_information(),terminated,0.01);
    }
    ended = true;

}
// ---------------------------------------------------------------------------
void CustomSliceModel::update(void)
{
    if(size_ratio != 1.0)
    {
        image::transformation_matrix<double> T(arg_min,from.geometry(),from_vs,source_buf.geometry(),source_buf_vs);
        invT.identity();
        T.save_to_transform(invT.begin());
        image::multiply_constant(invT,size_ratio);
    }
    else
    {
        image::transformation_matrix<double> T(arg_min,from.geometry(),from_vs,source_images.geometry(),voxel_size);
        invT.identity();
        T.save_to_transform(invT.begin());
    }
    transform = image::inverse(invT);
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
void CustomSliceModel::get_slice(image::color_image& show_image,unsigned char cur_dim,const image::value_to_color<float>& v2c) const
{
    image::basic_image<float,2> buf;
    image::reslicing(source_images, buf, cur_dim, slice_pos[cur_dim]);
    v2c.convert(buf,show_image);
}
// ---------------------------------------------------------------------------
bool CustomSliceModel::stripskull(float qa_threshold)
{
    if(!ended)
        return false;
    update();
    image::basic_image<float,3> filter(source_images.geometry());
    image::resample(from,filter,transform,image::linear);
    image::upper_threshold(filter,qa_threshold);
    image::filter::gaussian(filter);
    image::filter::gaussian(filter);
    float m = *std::max_element(source_images.begin(),source_images.end());
    source_images *= filter;
    image::normalize(source_images,m);
    return true;
}
