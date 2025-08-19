// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"

void ROIRegion::new_from_sphere(tipl::vector<3> pos,float radius)
{
    tipl::image<3,unsigned char> mask(dim);
    tipl::adaptive_par_for(tipl::begin_index(dim),tipl::end_index(dim),
                  [&](const tipl::pixel_index<3>& index)
    {
        if(std::abs((float(index[0])-pos[0])) > radius)
            return;
        if((tipl::vector<3>(index)-pos).length() <= radius)
            mask[index.index()] = 1;
    });
    load_region_from_buffer(mask);
}

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
std::vector<tipl::vector<3,short> > ROIRegion::to_space(
                    const tipl::shape<3>& dim_to,
                    const tipl::matrix<4,4>& trans_to) const
{
    if(dim == dim_to && to_diffusion_space == trans_to)
        return region;
    tipl::image<3,unsigned char> mask;
    save_region_to_buffer(mask,dim_to,trans_to);
    return tipl::volume2points(mask);
}
// ---------------------------------------------------------------------------

std::vector<tipl::vector<3,short> > ROIRegion::to_space(const tipl::shape<3>& dim_to) const
{
    if(dim == dim_to && is_diffusion_space)
        return region;
    tipl::image<3,unsigned char> mask;
    save_region_to_buffer(mask,dim_to,tipl::matrix<4,4>(tipl::identity_matrix()));
    return tipl::volume2points(mask);
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
        return;
    }

    if(!del)
    {
        region.insert(region.end(),points.begin(),points.end());
        points.clear();
    }

    tipl::vector<3,short> min_value,max_value,geo_size;
    tipl::bounding_box(region,max_value,min_value);
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
    region = tipl::volume2points(mask);
    tipl::add_constant(region.begin(),region.end(),min_value);
}

// ---------------------------------------------------------------------------
bool ROIRegion::save_region_to_file(const char* file_name)
{
    if (tipl::ends_with(file_name,".txt"))
    {
        std::ofstream out(file_name);
        if(!out)
            return false;
        std::copy(region.begin(), region.end(),std::ostream_iterator<tipl::vector<3,short> >(out, "\n"));
        return true;
    }
    if (tipl::ends_with(file_name,".mat"))
    {
        tipl::image<3,unsigned char> mask(dim);
        for (unsigned int index = 0; index < region.size(); ++index)
        {
            if (dim.is_valid(region[index][0], region[index][1],
                             region[index][2]))
                mask[tipl::pixel_index<3>(region[index][0], region[index][1],
                                           region[index][2], dim).index()] = 255;
        }
        tipl::io::mat_write header(file_name);
        if(!header)
            return false;
        header << mask;
        return true;
    }
    if (tipl::ends_with(file_name,".nii.gz") || tipl::ends_with(file_name,".nii"))
    {
        unsigned int color = region_render->color.color & 0x00FFFFFF;
        tipl::image<3,unsigned char> mask;
        save_region_to_buffer(mask);
        std::ostringstream out;
        out << "color=" << int(color) << ";roi=" << int(regions_feature);
        std::string tmp = out.str();
        if(tmp.size() < 80)
            tmp.resize(80);
        return tipl::io::gz_nifti::save_to_file(file_name,mask,vs,trans_to_mni,is_mni,tmp.c_str());
    }
    return false;
}

// ---------------------------------------------------------------------------

bool ROIRegion::load_region_from_file(const char* file_name) {
    modified = true;
    region.clear();
    is_diffusion_space = false;
    to_diffusion_space.identity();
    if (tipl::ends_with(file_name,".txt"))
    {

        std::vector<tipl::vector<3,short> > points;
        {
            std::ifstream in(file_name,std::ios::binary);
            std::vector<tipl::vector<3> > pointsf;
            std::copy(std::istream_iterator<tipl::vector<3> >(in),
                      std::istream_iterator<tipl::vector<3> >(),
                      std::back_inserter(pointsf));
            for(auto& each : pointsf)
            {
                each.round();
                points.push_back(tipl::vector<3,short>(each));
            }
        }
        if(!points.empty() && points.back()[1] == -1.0f && points.back()[2] == -1.0f)
        {
            is_diffusion_space = false;
            to_diffusion_space[0] = to_diffusion_space[5] = to_diffusion_space[10] = 1.0f/points.back()[0];
            points.pop_back();
        }
        region.swap(points);
        return true;
    }
    if (tipl::ends_with(file_name,".mat"))
    {
        tipl::io::mat_read header;
        if(!header.load_from_file(file_name))
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
    if (tipl::ends_with(file_name,".nii") || tipl::ends_with(file_name,".nii.gz"))
    {
        tipl::io::gz_nifti header;
        tipl::image<3> I;
        if (!header.load_from_file(file_name) || !header.toLPS(I))
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
        load_region_from_buffer(mask);
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
    region_render->load(region,to_diffusion_space,smooth);
}
// ---------------------------------------------------------------------------
void ROIRegion::load_region_from_buffer(tipl::image<3,unsigned char>& mask)
{
    modified = true;
    if(!region.empty())
        undo_backup.push_back(std::move(region));
    region = tipl::volume2points(mask);
}
// ---------------------------------------------------------------------------
void ROIRegion::save_region_to_buffer(tipl::image<3,unsigned char>& mask) const
{
    mask = std::move(tipl::points2volume(dim,region));
}
// ---------------------------------------------------------------------------
void ROIRegion::save_region_to_buffer(tipl::image<3,unsigned char>& mask,const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to) const
{
    tipl::image<3,unsigned char> m;
    save_region_to_buffer(m);
    if(dim == dim_to && to_diffusion_space == trans_to)
    {
        m.swap(mask);
        return;
    }
    mask.resize(dim_to);
    tipl::resample<tipl::interpolation::majority>(m,mask,
            tipl::transformation_matrix<float>(tipl::from_space(trans_to).to(to_diffusion_space)));
}
// ---------------------------------------------------------------------------
void ROIRegion::perform(const std::string& action)
{
    if(action == "flipx")
        flip_region(0);
    if(action == "flipy")
        flip_region(1);
    if(action == "flipz")
        flip_region(2);
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
        save_region_to_buffer(mask);
        tipl::morphology::smoothing(mask);
        load_region_from_buffer(mask);
    }
    if(action == "erosion")
    {
        save_region_to_buffer(mask);
        tipl::morphology::erosion(mask);
        load_region_from_buffer(mask);
    }
    if(action == "dilation")
    {
        save_region_to_buffer(mask);
        tipl::morphology::dilation(mask);
        load_region_from_buffer(mask);
    }
    if(action == "opening")
    {
        save_region_to_buffer(mask);
        tipl::morphology::opening(mask);
        load_region_from_buffer(mask);
    }
    if(action == "closing")
    {
        save_region_to_buffer(mask);
        tipl::morphology::closing(mask);
        load_region_from_buffer(mask);
    }
    if(action == "defragment")
    {
        save_region_to_buffer(mask);
        tipl::morphology::defragment(mask);
        load_region_from_buffer(mask);
    }
    if(action == "negate")
    {
        save_region_to_buffer(mask);
        tipl::morphology::negate(mask);
        load_region_from_buffer(mask);
    }

}

// ---------------------------------------------------------------------------
void ROIRegion::flip_region(unsigned int dimension) {
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
        region_render->move_object(dx_in_dwi_space);
    }
    else
        region_render->move_object(dx);
    for(size_t index = 0;index < region.size();++index)
        region[index] += dx;
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
        if(value == 0.0f || std::isnan(value) || std::isinf(value))
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
void ROIRegion::get_quantitative_data(std::shared_ptr<slice_model> slice,float& mean,float& max_v,float& min_v)
{
    auto I = slice->get_image();
    if(I.shape() != dim || slice->T != to_diffusion_space)
    {
        tipl::matrix<4,4> trans = slice->iT*to_diffusion_space;
        calculate_region_stat(I,region,mean,max_v,min_v,&trans[0]);
    }
    else
        calculate_region_stat(I,region,mean,max_v,min_v);

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
    {
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
        titles.push_back("bounding box x");
        titles.push_back("bounding box y");
        titles.push_back("bounding box z");
        titles.push_back("bounding box x");
        titles.push_back("bounding box y");
        titles.push_back("bounding box z");
        std::copy(cm.begin(),cm.end(),std::back_inserter(data)); // center of the mass
        std::copy(min.begin(),min.end(),std::back_inserter(data)); // bounding box
        std::copy(max.begin(),max.end(),std::back_inserter(data)); // bounding box

        if(!handle->s2t.empty())
        {
            handle->sub2mni(cm);
            handle->sub2mni(max);
            handle->sub2mni(min);
            titles.push_back("center mni x");
            titles.push_back("center mni y");
            titles.push_back("center mni z");
            titles.push_back("bounding box mni x");
            titles.push_back("bounding box mni y");
            titles.push_back("bounding box mni z");
            titles.push_back("bounding box mni x");
            titles.push_back("bounding box mni y");
            titles.push_back("bounding box mni z");
            std::copy(cm.begin(),cm.end(),std::back_inserter(data)); // center of the mass
            // swap due to RAS to LPS
            std::swap(min[0],max[0]);
            std::swap(min[1],max[1]);
            std::copy(min.begin(),min.end(),std::back_inserter(data)); // bounding box
            std::copy(max.begin(),max.end(),std::back_inserter(data)); // bounding box
        }
    }
    std::vector<tipl::vector<3> > points(region.size());
    std::copy(region.begin(),region.end(),points.begin());
    std::vector<float> max_values,min_values;
    std::vector<std::string> index_titles;
    for(const auto& each : handle->slices)
    {
        if(each->optional())
            continue;
        index_titles.push_back(each->name);
        max_values.push_back(0.0f);
        min_values.push_back(0.0f);
        data.push_back(0.0f);
        get_quantitative_data(each,data.back(),max_values.back(),min_values.back());
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
        tipl::progress p("compute subject data");
        for(unsigned int subject_index = 0;p(subject_index,handle->db.subject_names.size());++subject_index)
        {
            auto I = handle->db.get_index_image(subject_index);
            float mean,max,min;
            calculate_region_stat(I.alias(),points,mean,max,min,is_diffusion_space ? nullptr : &to_diffusion_space[0]);
            data.push_back(mean);
            std::ostringstream out;
            out << handle->db.subject_names[subject_index] << (" mean_") << handle->db.index_name;
            titles.push_back(out.str());
        }
    }
}
