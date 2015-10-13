// ---------------------------------------------------------------------------
#include <string>
#include "SliceModel.h"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"
#include "manual_alignment.h"
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
bool CustomSliceModel::initialize(FibSliceModel& slice,bool is_qsdr,const std::vector<std::string>& files)
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

    roi_image.resize(slice.handle->dim);
    roi_image_buf = &*roi_image.begin();
    if(!has_transform)
    {
        from = slice.source_images;
        arg_min.scaling[0] = slice.voxel_size[0] / voxel_size[0];
        arg_min.scaling[1] = slice.voxel_size[1] / voxel_size[1];
        arg_min.scaling[2] = slice.voxel_size[2] / voxel_size[2];
        thread.reset(new boost::thread(&CustomSliceModel::argmin,this,image::reg::rigid_body));
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
    image::reg::linear(from,to,arg_min,reg_type,image::reg::mutual_information(),terminated);
    ended = true;
}
// ---------------------------------------------------------------------------
void CustomSliceModel::update(void)
{
    image::transformation_matrix<3,float> T(arg_min,from.geometry(),source_images.geometry());
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
    {
        thread->joinable();
        thread->join();
    }
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
extern image::basic_image<char,3> brain_mask;
extern image::basic_image<float,3> mni_t1w;
bool load_brain_mask(void);
#include "manual_alignment.h"
#include "fa_template.hpp"
bool CustomSliceModel::stripskull(void)
{
    begin_prog("calculating");
    if(!load_brain_mask())
        return false;
    image::basic_image<float,3> to(mni_t1w);
    image::basic_image<float,3> from(source_images);
    image::downsampling(to);
    image::downsampling(from);
    image::minus_constant(from,image::segmentation::otsu_threshold(from));
    image::lower_threshold(from,0);
    image::normalize(from,1);

    image::minus_constant(to,*std::min_element(to.begin(),to.end()));
    image::normalize(to,1);

    image::hist_norm(to,1000);
    image::hist_norm(from,1000);
    image::vector<3,float> vs(voxel_size);
    vs[0] = 1.0/vs[0];
    vs[1] = 1.0/vs[1];
    vs[2] = 1.0/vs[2];
    std::auto_ptr<manual_alignment> manual(new manual_alignment(0,to,from,vs,image::reg::affine));
    manual->timer->start();
    if(manual->exec() != QDialog::Accepted)
        return false;
    begin_prog("calculating");
    manual->update_affine();
    manual->T.shift[0] *= 2.0;
    manual->T.shift[1] *= 2.0;
    manual->T.shift[2] *= 2.0;
    manual->iT.shift[0] *= 2.0;
    manual->iT.shift[1] *= 2.0;
    manual->iT.shift[2] *= 2.0;

    check_prog(0,3);
    image::basic_image<float,3> new_from(mni_t1w.geometry());
    image::resample(source_images,new_from,manual->T,image::linear);

    image::io::nifti out;
    out << new_from;
    out.save_to_file("test.nii");
    reg_data data(mni_t1w.geometry(),image::reg::affine,1);
    multi_thread_reg(data.bnorm_data,new_from,mni_t1w,4,data.terminated);
    set_title("stripping skull");
    check_prog(1,3);
    for(image::pixel_index<3> index;source_images.geometry().is_valid(index);index.next(source_images.geometry()))
    {
        image::vector<3,float> pos(index),t1_space;
        manual->iT(pos);// from -> new_from
        data.bnorm_data(pos,t1_space);
        t1_space += 0.5;
        t1_space.floor();
        if(!mni_t1w.geometry().is_valid(t1_space) || brain_mask.at(t1_space[0],t1_space[1],t1_space[2]) == 0)
            source_images[index.index()] = 0;
    }
    check_prog(0,0);
    return true;
}
