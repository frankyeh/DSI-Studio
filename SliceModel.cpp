// ---------------------------------------------------------------------------
#include <string>
#include "SliceModel.h"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"
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
    source_images = image::make_image(geometry,fib_data.fib.fa[0]);
    center_point = geometry;
    center_point -= 1.0;
    center_point /= 2.0;
    slice_pos[0] = geometry.width() >> 1;
    slice_pos[1] = geometry.height() >> 1;
    slice_pos[2] = geometry.depth() >> 1;
    //loadImage(fib_data.fib.fa[0],false);
}
// ---------------------------------------------------------------------------
float FibSliceModel::get_value_range(void) const
{
    std::pair<float,float> value = image::min_max_value(source_images.begin(),source_images.end());
    return value.second-value.first;
}
// ---------------------------------------------------------------------------
void FibSliceModel::get_slice(image::color_image& show_image,float contrast,float offset) const
{
    handle->get_slice(view_name,cur_dim, slice_pos[cur_dim],show_image,contrast,offset);
}
// ---------------------------------------------------------------------------
void FibSliceModel::get_mosaic(image::color_image& show_image,
                               unsigned int mosaic_size,
                               float contrast,
                               float offset,
                               unsigned int skip) const
{
    unsigned slice_num = geometry[2] >> skip;
    show_image.clear();
    show_image.resize(image::geometry<2>(geometry[0]*mosaic_size,
                                          geometry[1]*(std::ceil((float)slice_num/(float)mosaic_size))));
    for(unsigned int z = 0;z < slice_num;++z)
    {
        image::color_image slice_image;
        handle->get_slice(view_name,2, z << skip,slice_image,contrast,offset);
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
bool CustomSliceModel::initialize(FibSliceModel& slice,bool is_qsdr,const std::vector<std::string>& files)
{
    terminated = true;
    ended = true;

    gz_nifti nifti;
    center_point = slice.center_point;
    // QSDR loaded, use MNI transformation instead
    if(is_qsdr && files.size() == 1 && nifti.load_from_file(files[0]))
    {
        loadLPS(nifti);
        std::vector<float> t(nifti.get_transformation(),
                             nifti.get_transformation()+12),inv_trans(16);
        transform.resize(16);
        t.resize(16);
        t[15] = 1.0;
        image::matrix::inverse(slice.handle->trans_to_mni.begin(),inv_trans.begin(),image::dim<4,4>());
        image::matrix::product(inv_trans.begin(),t.begin(),transform.begin(),image::dim<4,4>(),image::dim<4,4>());
        invT.resize(16);
        image::matrix::inverse(transform.begin(),invT.begin(),image::dim<4, 4>());
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
            transform.resize(16);
            transform[0] = transform[5] = transform[10] = transform[15] = 1.0;
            invT.resize(16);
            invT[0] = invT[5] = invT[10] = invT[15] = 1.0;
        }
    }

    roi_image.resize(slice.handle->dim);
    roi_image_buf = &*roi_image.begin();
    if(transform.empty())
    {
        from = slice.source_images;
        arg_min.scaling[0] = slice.voxel_size[0] / voxel_size[0];
        arg_min.scaling[1] = slice.voxel_size[1] / voxel_size[1];
        arg_min.scaling[2] = slice.voxel_size[2] / voxel_size[2];
        thread.reset(new boost::thread(&CustomSliceModel::argmin,this,image::reg::rigid_body));
        // handle views
        transform.resize(16);
        invT.resize(16);
    }
    else
        update_roi();
    return true;
}

// ---------------------------------------------------------------------------
float CustomSliceModel::get_value_range(void) const
{
    std::pair<float,float> value = image::min_max_value(source_images.begin(),source_images.end());
    return value.second-value.first;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::argmin(int reg_type)
{
    terminated = false;
    ended = false;
    image::const_pointer_image<float,3> to = source_images;
    image::reg::linear(from,to,arg_min,reg_type,image::reg::mutual_information(),terminated);
    ended = true;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update(void)
{
    image::transformation_matrix<3,float> T(arg_min,from.geometry(),source_images.geometry());
    invT.resize(16);
    invT[15] = 1.0;
    T.save_to_transform(invT.begin());
    transform.resize(16);
    transform[15] = 1.0;
    image::matrix::inverse(invT.begin(),transform.begin(),image::dim<4, 4>());
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
    {
        thread->joinable();
        thread->join();
    }
    ended = true;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::get_slice(image::color_image& show_image,float contrast,float offset) const
{
    image::basic_image<float,2> buf;
    image::reslicing(source_images, buf, cur_dim, slice_pos[cur_dim]);
    show_image.resize(buf.geometry());
    buf += offset-min_value;
    if(contrast != 0.0)
        buf *= 255.99/contrast;
    image::upper_lower_threshold(buf,(float)0.0,(float)255.0);
    std::copy(buf.begin(),buf.end(),show_image.begin());
}
