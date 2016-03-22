// ---------------------------------------------------------------------------
#include <string>
#include "SliceModel.h"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"
#include "manual_alignment.h"
#include "fa_template.hpp"

// ---------------------------------------------------------------------------
SliceModel::SliceModel(void):cur_dim(2)
{
    slice_visible[0] = false;
    slice_visible[1] = false;
    slice_visible[2] = false;
    texture_need_update[0] = true;
    texture_need_update[1] = true;
    texture_need_update[2] = true;
}

FibSliceModel::FibSliceModel(FibData* handle_):handle(handle_)
{
    // already setup the geometry and source image
    FibData& fib_data = *handle;
    geometry = fib_data.dim;
    voxel_size = fib_data.vs;
    source_images = image::make_image(geometry,fib_data.dir.fa[0]);
    center_point = geometry;
    center_point -= 1.0;
    center_point /= 2.0;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
    //loadImage(fib_data.dir.fa[0],false);
}
// ---------------------------------------------------------------------------
std::pair<float,float> FibSliceModel::get_value_range(void) const
{
    return image::min_max_value(source_images.begin(),source_images.end());
}
// ---------------------------------------------------------------------------
void FibSliceModel::get_slice(image::color_image& show_image,const image::value_to_color<float>& v2c) const
{
    handle->get_slice(view_name,cur_dim, slice_pos[cur_dim],show_image,v2c);
}
// ---------------------------------------------------------------------------
void FibSliceModel::get_mosaic(image::color_image& show_image,
                               unsigned int mosaic_size,
                               const image::value_to_color<float>& v2c,
                               unsigned int skip) const
{
    unsigned slice_num = geometry[2] / skip;
    show_image.clear();
    show_image.resize(image::geometry<2>(geometry[0]*mosaic_size,
                                          geometry[1]*(std::ceil((float)slice_num/(float)mosaic_size))));
    for(unsigned int z = 0;z < slice_num;++z)
    {
        image::color_image slice_image;
        handle->get_slice(view_name,2, z * skip,slice_image,v2c);
        image::vector<2,int> pos(geometry[0]*(z%mosaic_size),
                                 geometry[1]*(z/mosaic_size));
        image::draw(slice_image,show_image,pos);
    }
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
bool CustomSliceModel::initialize(FibSliceModel& slice,bool is_qsdr,const std::vector<std::string>& files,bool correct_intensity)
{
    terminated = true;
    ended = true;

    gz_nifti nifti;
    center_point = slice.center_point;
    // QSDR loaded, use MNI transformation instead
    bool has_transform = false;
    if(is_qsdr && files.size() == 1 && nifti.load_from_file(files[0]))
    {
        loadLPS(nifti);
        invT.identity();
        nifti.get_image_transformation(invT.begin());
        invT.inv();
        invT *= slice.handle->trans_to_mni;
        transform = image::inverse(invT);
        has_transform = true;
    }
    else
    {
        if(files.size() == 1 && nifti.load_from_file(files[0]))
            loadLPS(nifti);
        else
        {
            image::io::bruker_2dseq bruker;
            if(files.size() == 1 && bruker.load_from_file(files[0].c_str()))
                load(bruker);
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
        if(source_images.geometry() == slice.source_images.geometry())
        {
            transform.identity();
            invT.identity();
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

    roi_image.resize(slice.handle->dim);
    roi_image_buf = &*roi_image.begin();
    if(!has_transform)
    {
        if(slice.source_images.depth() < 10) // 2d assume FOV is the same
        {
            transform.identity();
            invT.identity();
            invT[0] = (float)source_images.width()/(float)slice.source_images.width();
            invT[5] = (float)source_images.height()/(float)slice.source_images.height();
            invT[10] = (float)source_images.depth()/(float)slice.source_images.depth();
            invT[15] = 1.0;
            transform = image::inverse(invT);
            update_roi();
        }
        else
        {
            from = slice.source_images;
            from_vs = slice.voxel_size;
            thread.reset(new std::future<void>(
                             std::async(std::launch::async,[this](){argmin(image::reg::rigid_body);})));
        }
    }
    else
        update_roi();
    return true;
}

// ---------------------------------------------------------------------------
std::pair<float,float> CustomSliceModel::get_value_range(void) const
{
    return image::min_max_value(source_images.begin(),source_images.end());
}
// ---------------------------------------------------------------------------
void CustomSliceModel::argmin(int reg_type)
{
    terminated = false;
    ended = false;
    image::const_pointer_image<float,3> to = source_images;
    image::reg::linear(from,from_vs,to,voxel_size,arg_min,reg_type,image::reg::mutual_information(),terminated);
    ended = true;

}
// ---------------------------------------------------------------------------
void CustomSliceModel::update(void)
{
    image::transformation_matrix<float> T(arg_min,from.geometry(),from_vs,source_images.geometry(),voxel_size);
    invT.identity();
    T.save_to_transform(invT.begin());
    transform = image::inverse(invT);
    update_roi();
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update_roi(void)
{
    std::fill(texture_need_update,texture_need_update+3,1);
    image::resample(source_images,roi_image,invT,image::linear);
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
void CustomSliceModel::get_slice(image::color_image& show_image,const image::value_to_color<float>& v2c) const
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
