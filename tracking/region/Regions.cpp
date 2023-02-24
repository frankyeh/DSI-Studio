// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "libs/gzip_interface.hpp"

void ROIRegion::add_points(std::vector<tipl::vector<3,float> >&& points,bool del)
{
    std::vector<tipl::vector<3,short> > new_points(points.size());
    for(size_t i = 0;i < points.size();++i)
    {
        points[i].round();
        new_points[i] = points[i];
    }
    add_points(std::move(new_points),del);
}

// ---------------------------------------------------------------------------
void convert_region(std::vector<tipl::vector<3,short> >& points,
                    const tipl::shape<3>& dim_from,
                    const tipl::matrix<4,4>& trans_from,
                    const tipl::shape<3>& dim_to,
                    const tipl::matrix<4,4>& trans_to)
{
    if(dim_from == dim_to && trans_from == trans_to)
        return;
    ROIRegion region_from(dim_from,tipl::vector<3>(1,1,1)),
              region_to(dim_to,tipl::vector<3>(1,1,1));
    region_from.region.swap(points);
    region_from.to_diffusion_space = trans_from;
    tipl::image<3,unsigned char> mask;
    region_from.SaveToBuffer(mask,dim_to,trans_to);
    region_to.LoadFromBuffer(mask);
    region_to.region.swap(points);
}
void ROIRegion::add_points(std::vector<tipl::vector<3,short> >&& points,
                           const tipl::shape<3>& slice_dim,
                           const tipl::matrix<4,4>& slice_trans,bool del)
{
    convert_region(points,slice_dim,slice_trans,dim,to_diffusion_space);
    add_points(std::move(points),del);
}
// ---------------------------------------------------------------------------
void ROIRegion::add_points(std::vector<tipl::vector<3,short> >&& points, bool del)
{
    if(!region.empty())
        undo_backup.push_back(region);

    points.erase(std::remove_if(points.begin(),points.end(),
                                [this](const tipl::vector<3,short>&p){return !dim.is_valid(p);}),points.end());

    if(points.empty())
        return;
    modified = true;
    if(region.empty())
    {
        region.swap(points);
        show_progress() << "region dimension: " << dim << std::endl;
        show_progress() << "region voxel size: " << vs << std::endl;
        show_progress() << "region voxel count: " << region.size() << std::endl;
        return;
    }

    if(!del)
    {
        region.insert(region.end(),points.begin(),points.end());
        points.clear();
    }

    tipl::vector<3,short> min_value,max_value,geo_size;
    tipl::bounding_box_mt(region,max_value,min_value);
    geo_size = max_value-min_value;

    tipl::shape<3> mask_geo(geo_size[0]+1,geo_size[1]+1,geo_size[2]+1);
    tipl::image<3,unsigned char> mask(mask_geo);

    tipl::par_for (region.size(),[&](unsigned int index)
    {
        auto p = region[index];
        p -= min_value;
        if (mask.shape().is_valid(p))
            mask.at(p) = 1;
    });

    tipl::par_for (points.size(),[&](unsigned int index)
    {
        auto p = points[index];
        p -= min_value;
        if (mask.shape().is_valid(p))
            mask.at(p) = 0;
    });

    std::vector<std::vector<tipl::vector<3,short> > > region_at_thread(std::thread::hardware_concurrency());
    tipl::par_for(tipl::begin_index(mask.shape()),tipl::end_index(mask.shape()),
    [&](const tipl::pixel_index<3>& index,unsigned int id)
    {
        if(mask[index.index()])
            region_at_thread[id].push_back(tipl::vector<3,short>(index[0]+min_value[0],
                                                                 index[1]+min_value[1],
                                                                 index[2]+min_value[2]));
    });
    tipl::aggregate_results(std::move(region_at_thread),region);
}

// ---------------------------------------------------------------------------
bool ROIRegion::save_to_file(const char* FileName)
{
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if (ext == std::string(".txt")) {
        std::ofstream out(FileName);
        if(!out)
            return false;
        std::copy(region.begin(), region.end(),std::ostream_iterator<tipl::vector<3,short> >(out, "\n"));
        return true;
    }
    else if (ext == std::string(".mat")) {
        tipl::image<3,unsigned char> mask(dim);
        for (unsigned int index = 0; index < region.size(); ++index) {
            if (dim.is_valid(region[index][0], region[index][1],
                             region[index][2]))
                mask[tipl::pixel_index<3>(region[index][0], region[index][1],
                                           region[index][2], dim).index()] = 255;
        }
        tipl::io::mat_write header(FileName);
        if(!header)
            return false;
        header << mask;
        return true;
    }
    else if (ext == std::string(".nii") || ext == std::string("i.gz"))
    {
        unsigned int color = region_render.color.color & 0x00FFFFFF;
        tipl::image<3,unsigned char> mask;
        SaveToBuffer(mask);
        tipl::matrix<4,4> T(trans_to_mni);
        std::ostringstream out;
        out << "color=" << int(color) << ";roi=" << int(regions_feature);
        std::string tmp = out.str();
        if(tmp.size() < 80)
            tmp.resize(80);
        return gz_nifti::save_to_file(FileName,mask,vs,tipl::matrix<4,4>(trans_to_mni*to_diffusion_space),is_mni,tmp.c_str());
    }
    return false;
}

// ---------------------------------------------------------------------------

bool ROIRegion::LoadFromFile(const char* FileName) {
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    modified = true;
    region.clear();
    is_diffusion_space = false;
    to_diffusion_space.identity();

    if (ext == std::string(".txt"))
    {
        std::ifstream in(FileName,std::ios::binary);
        std::vector<tipl::vector<3,short> > points;
        std::copy(std::istream_iterator<tipl::vector<3,short> >(in),
                  std::istream_iterator<tipl::vector<3,short> >(),
                  std::back_inserter(points));
        if(!points.empty() && points.back()[1] == -1.0f && points.back()[2] == -1.0f)
        {
            is_diffusion_space = false;
            to_diffusion_space[0] = to_diffusion_space[5] = to_diffusion_space[10] = 1.0f/points.back()[0];
            points.pop_back();
        }
        region.swap(points);
        return true;
    }

    if (ext == std::string(".mat")) {
        tipl::io::mat_read header;
        if(!header.load_from_file(FileName))
            return false;
        tipl::image<3,short>from;
        header >> from;
        std::vector<tipl::vector<3,short> > points;
        for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
            if (from[index.index()])
                points.push_back(tipl::vector<3,short>((const unsigned int*)index.begin()));
        add_points(std::move(points));
        return true;
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr") || ext == std::string("i.gz"))
    {
        gz_nifti header;
        tipl::image<3> I;
        if (!header.load_from_file(FileName) || !header.toLPS(I))
        {
            std::cout << header.error_msg << std::endl;
            return false;
        }
        // use unsigned int to avoid the nan background problem
        dim = I.shape();
        is_mni = header.is_mni();
        header.get_voxel_size(vs);
        header.get_image_transformation(trans_to_mni);
        tipl::image<3,unsigned char> mask = I;
        LoadFromBuffer(mask);
        return true;
    }
    return false;

}

void ROIRegion::makeMeshes(unsigned char smooth)
{
    if(!modified)
        return;
    modified = false;
    if(is_diffusion_space)
        to_diffusion_space.identity();
    region_render.load(region,to_diffusion_space,smooth);
}
// ---------------------------------------------------------------------------
void ROIRegion::LoadFromBuffer(tipl::image<3,unsigned char>& mask)
{
    modified = true;
    if(!region.empty())
        undo_backup.push_back(std::move(region));
    std::vector<std::vector<tipl::vector<3,short> > > points(std::thread::hardware_concurrency());
    tipl::par_for(tipl::begin_index(mask.shape()),tipl::end_index(mask.shape()),
                   [&](const tipl::pixel_index<3>& index,unsigned int thread_id)
    {
        if (mask[index.index()])
            points[thread_id].push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
    });
    tipl::aggregate_results(std::move(points),region);
}
// ---------------------------------------------------------------------------
void ROIRegion::SaveToBuffer(tipl::image<3,unsigned char>& mask)
{
    mask.resize(dim);
    mask = 0;
    tipl::par_for (region.size(),[&](unsigned int index)
    {
        if (mask.shape().is_valid(region[index]))
            mask.at(region[index]) = 1;
    });
}
// ---------------------------------------------------------------------------
void ROIRegion::SaveToBuffer(tipl::image<3,unsigned char>& mask,const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to)
{
    tipl::image<3,unsigned char> m;
    SaveToBuffer(m);
    if(dim == dim_to && to_diffusion_space == trans_to)
    {
        m.swap(mask);
        return;
    }
    mask.resize(dim_to);
    tipl::resample_mt<tipl::interpolation::nearest>(m,mask,
            tipl::transformation_matrix<float>(tipl::from_space(trans_to).to(to_diffusion_space)));
}
// ---------------------------------------------------------------------------
void ROIRegion::perform(const std::string& action)
{
    if(action == "flipx")
        Flip(0);
    if(action == "flipy")
        Flip(1);
    if(action == "flipz")
        Flip(2);
    if(action == "shiftx")
        shift(tipl::vector<3,float>(1.0, 0.0, 0.0));
    if(action == "shiftnx")
        shift(tipl::vector<3,float>(-1.0, 0.0, 0.0));
    if(action == "shifty")
        shift(tipl::vector<3,float>(0.0, 1.0, 0.0));
    if(action == "shiftny")
        shift(tipl::vector<3,float>(0.0, -1.0, 0.0));
    if(action == "shiftz")
        shift(tipl::vector<3,float>(0.0, 0.0, 1.0));
    if(action == "shiftnz")
        shift(tipl::vector<3,float>(0.0, 0.0, -1.0));

    tipl::image<3,unsigned char>mask;
    if(action == "smoothing")
    {
        SaveToBuffer(mask);
        tipl::morphology::smoothing(mask);
        LoadFromBuffer(mask);
    }
    if(action == "erosion")
    {
        SaveToBuffer(mask);
        tipl::morphology::erosion(mask);
        LoadFromBuffer(mask);
    }
    if(action == "dilation")
    {
        SaveToBuffer(mask);
        tipl::morphology::dilation(mask);
        LoadFromBuffer(mask);
    }
    if(action == "opening")
    {
        SaveToBuffer(mask);
        tipl::morphology::opening(mask);
        LoadFromBuffer(mask);
    }
    if(action == "closing")
    {
        SaveToBuffer(mask);
        tipl::morphology::closing(mask);
        LoadFromBuffer(mask);
    }
    if(action == "defragment")
    {
        SaveToBuffer(mask);
        tipl::morphology::defragment(mask);
        LoadFromBuffer(mask);
    }
    if(action == "negate")
    {
        SaveToBuffer(mask);
        tipl::morphology::negate(mask);
        LoadFromBuffer(mask);
    }

}

// ---------------------------------------------------------------------------
void ROIRegion::Flip(unsigned int dimension) {
    modified = true;
    if(!region.empty())
        undo_backup.push_back(region);
    for (unsigned int index = 0; index < region.size(); ++index)
        region[index][dimension] = dim[dimension] - region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
bool ROIRegion::shift(tipl::vector<3,float> dx) // shift in region's voxel space
{
    dx.round();
    if(dx[0] == 0.0f && dx[1] == 0.0f && dx[2] == 0.0f)
        return false;
    if(!is_diffusion_space)
    {
        auto dx_in_dwi_space = dx;
        tipl::transformation_matrix<float> T(to_diffusion_space);
        dx_in_dwi_space.rotate(T.sr);
        region_render.move_object(dx_in_dwi_space);
    }
    else
        region_render.move_object(dx);
    tipl::par_for(region.size(),[&](unsigned int index)
    {
        region[index] += dx;
    });
    return true;
}
// ---------------------------------------------------------------------------
template<class Image,class Points>
void calculate_region_stat(const Image& I, const Points& p,float& mean,float& max,float& min,const float* T = nullptr)
{
    double sum = 0.0;
    size_t count = 0;
    for(size_t index = 0; index < p.size(); ++index)
    {
        float value = 0.0f;
        tipl::vector<3> pos(p[index]);
        if(T)
            pos.to(T);
        value = tipl::estimate(I,pos);
        if(value == 0.0f)
            continue;
        if(index)
        {
            max = std::max<float>(value,max);
            min = std::min<float>(value,min);
        }
        else
            min = max = value;
        sum += double(value);
        ++count;
    }
    if(count)
        sum /= double(count);
    mean = float(sum);
}
float ROIRegion::get_volume(void) const
{
    return float(region.size())*vs[0]*vs[1]*vs[2];
}

tipl::vector<3> ROIRegion::get_pos(void) const
{
    tipl::vector<3> cm;
    for (unsigned int index = 0; index < region.size(); ++index)
        cm += region[index];
    cm /= region.size();
    if(!is_diffusion_space)
        cm.to(to_diffusion_space);
    return cm;
}
void ROIRegion::get_quantitative_data(std::shared_ptr<fib_data> handle,std::vector<std::string>& titles,std::vector<float>& data)
{
    titles.clear();
    titles.push_back("voxel counts");
    data.push_back(region.size());

    titles.push_back("volume (mm^3)");
    data.push_back(get_volume()); //volume (mm^3)
    if(region.empty())
        return;
    tipl::vector<3,float> cm = get_pos();
    tipl::vector<3,float> max(region[0]),min(region[0]);
    for (unsigned int index = 0; index < region.size(); ++index)
    {
        max[0] = std::max<float>(max[0],region[index][0]);
        max[1] = std::max<float>(max[1],region[index][1]);
        max[2] = std::max<float>(max[2],region[index][2]);
        min[0] = std::min<float>(min[0],region[index][0]);
        min[1] = std::min<float>(min[1],region[index][1]);
        min[2] = std::min<float>(min[2],region[index][2]);
    }

    titles.push_back("center x");
    titles.push_back("center y");
    titles.push_back("center z");
    std::copy(cm.begin(),cm.end(),std::back_inserter(data)); // center of the mass

    if(!handle->s2t.empty())
    {
        tipl::vector<3> mni(cm);
        handle->sub2mni(mni);
        titles.push_back("center mni x");
        titles.push_back("center mni y");
        titles.push_back("center mni z");
        std::copy(mni.begin(),mni.end(),std::back_inserter(data)); // center of the mass
    }

    titles.push_back("bounding box x");
    titles.push_back("bounding box y");
    titles.push_back("bounding box z");
    std::copy(max.begin(),max.end(),std::back_inserter(data)); // bounding box

    titles.push_back("bounding box x");
    titles.push_back("bounding box y");
    titles.push_back("bounding box z");
    std::copy(min.begin(),min.end(),std::back_inserter(data)); // bounding box


    std::vector<std::string> index_titles;
    handle->get_index_list(index_titles);
    std::vector<tipl::vector<3> > points;
    for (unsigned int index = 0; index < region.size(); ++index)
        points.push_back(region[index]);
    // get mean, max, min value of each index
    std::vector<float> max_values,min_values;
    for(size_t data_index = 0;data_index < handle->view_item.size(); ++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
        float mean;
        max_values.push_back(0.0f);
        min_values.push_back(0.0f);
        if(handle->view_item[data_index].get_image().shape() != handle->dim ||
           handle->view_item[data_index].T != to_diffusion_space)
        {
            tipl::matrix<4,4> trans = handle->view_item[data_index].iT*to_diffusion_space;
            calculate_region_stat(handle->view_item[data_index].get_image(),points,mean,max_values.back(),min_values.back(),&trans[0]);
        }
        else
            calculate_region_stat(handle->view_item[data_index].get_image(),points,mean,max_values.back(),min_values.back());
        data.push_back(mean);
    }
    titles.insert(titles.end(),index_titles.begin(),index_titles.end());

    // max value of each index
    for(size_t index = 0;index < index_titles.size();++index)
    {
        data.push_back(max_values[index]);
        titles.push_back(index_titles[index]+"_max");
    }
    // min value of each index
    for(size_t index = 0;index < index_titles.size();++index)
    {
        data.push_back(min_values[index]);
        titles.push_back(index_titles[index]+"_min");
    }

    if(handle->db.has_db()) // connectometry database
    {
        for(unsigned int subject_index = 0;subject_index < handle->db.num_subjects;++subject_index)
        {
            std::vector<std::vector<float> > fa_data;
            handle->db.get_subject_fa(subject_index,fa_data);
            float mean,max,min;
            tipl::const_pointer_image<3> I(&fa_data[0][0],handle->dim);
            calculate_region_stat(I,points,mean,max,min);
            data.push_back(mean);
            std::ostringstream out;
            out << handle->db.subject_names[subject_index] << (" mean_") << handle->db.index_name;
            titles.push_back(out.str());
        }
    }
}
