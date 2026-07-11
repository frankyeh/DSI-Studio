// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"

void ROIRegion::new_from_sphere(tipl::vector<3> pos,float radius)
{
    tipl::image<3,unsigned char> mask(dim);
    tipl::par_for<tipl::sequential>(dim,[&](const tipl::pixel_index<3>& index)
    {
        if(std::abs((float(index[0])-pos[0])) > radius)
            return;
        if((tipl::v(index)-pos).length() <= radius)
            mask[index.index()] = 1;
    });
    from_mask(mask);
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
    return tipl::volume2points(to_mask(dim_to,trans_to));
}
// ---------------------------------------------------------------------------

std::vector<tipl::vector<3,short> > ROIRegion::to_space(const tipl::shape<3>& dim_to) const
{
    if(dim == dim_to && is_diffusion_space)
        return region;
    return tipl::volume2points(to_mask(dim_to,tipl::matrix<4,4>(tipl::identity_matrix())));
}

// ---------------------------------------------------------------------------
void ROIRegion::point2mask(void)
{
    if(region.empty())
    {
        index_mask.clear();
        index_mask_point_count = 0;
        return;
    }
    if(!index_mask.empty() && region.size() == index_mask_point_count)
        return;

    index_mask.resize(dim);
    index_mask = 0;
    std::vector<tipl::vector<3,short> > r;
    r.reserve(region.size());
    for(const auto& p : region)
        if(dim.is_valid(p) && !index_mask.at(p))
            index_mask.at(p) = r.size()+1,r.push_back(p);
    region.swap(r);
    index_mask_point_count = region.size();
}
void ROIRegion::add_points(std::vector<tipl::vector<3,short> >&& points, bool del)
{
    points.erase(std::remove_if(points.begin(),points.end(),
                                [this](const auto& p){return !dim.is_valid(p);}),points.end());
    if(points.empty())
        return;

    if(!region.empty())
        undo_backup.push_back(region);

    point2mask();
    if(index_mask.empty())
        index_mask.resize(dim),index_mask = 0;

    modified = true;
    if(del)
    {
        for(const auto& p : points)
        {
            auto index = index_mask.at(p);
            if(!index)
                continue;
            --index;
            index_mask.at(p) = 0;
            if(index+1 != region.size())
            {
                region[index] = region.back();
                index_mask.at(region[index]) = index+1;
            }
            region.pop_back();
        }
    }
    else
        for(const auto& p : points)
            if(!index_mask.at(p))
                index_mask.at(p) = region.size()+1,region.push_back(p);

    index_mask_point_count = region.size();
}

// ---------------------------------------------------------------------------
bool ROIRegion::save_region_to_file(const std::filesystem::path& file_name)
{
    if (tipl::ends_with(file_name.u8string(),".txt"))
    {
        std::ofstream out(file_name);
        if(!out)
            return false;
        std::copy(region.begin(), region.end(),std::ostream_iterator<tipl::vector<3,short> >(out, "\n"));
        return true;
    }
    if (tipl::ends_with(file_name.u8string(),".mat"))
    {
        auto mask = to_mask();
        tipl::multiply_constant(mask.begin(),mask.end(),uint8_t(255));
        tipl::io::mat_write header(file_name);
        if(!header)
            return false;
        header << mask;
        return true;
    }
    if (tipl::ends_with(file_name.u8string(),{".nii.gz",".nii"}))
    {
        unsigned int color = region_render->color.color & 0x00FFFFFF;
        auto mask = to_mask();
        std::ostringstream out;
        out << "color=" << int(color) << ";roi=" << int(regions_feature);
        std::string tmp = out.str();
        if(tmp.size() < 80)
            tmp.resize(80);
        return tipl::io::gz_nifti(file_name,std::ios::out) << vs << trans_to_mni << is_mni << tmp << mask;
    }
    return false;
}

// ---------------------------------------------------------------------------

bool ROIRegion::load_region_from_file(const std::filesystem::path& file_name) {
    set_modified();
    region.clear();
    is_diffusion_space = false;
    to_diffusion_space.identity();
    if (tipl::ends_with(file_name.u8string(),".txt"))
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
    if (tipl::ends_with(file_name.u8string(),".mat"))
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
    if (tipl::ends_with(file_name.u8string(),{".nii",".nii.gz"}))
    {
        tipl::image<3> I;
        if(!(tipl::io::gz_nifti(file_name,std::ios::in)
             >> vs >> trans_to_mni >> dim >> is_mni >> I
             >> [&](const std::string& e){tipl::error() << e;}))
            return false;
        tipl::image<3,unsigned char> mask = I;
        from_mask(mask);
        return true;
    }
    return false;

}
ROIRegion::~ROIRegion()
{
    mesh_terminated = true;
    if(mesh_thread.joinable())
        mesh_thread.join();
}
// ---------------------------------------------------------------------------
bool ROIRegion::makeMeshes(unsigned char smooth)
{
    bool busy = mesh_running;
    {
        std::lock_guard lock(mesh_lock);
        if(pending_region_render)
        {
            if(region_render)
                pending_region_render->color = region_render->color;
            region_render = std::move(pending_region_render);
            return true;
        }
    }

    if(!mesh_running && mesh_thread.joinable())
        mesh_thread.join();
    if(!modified)
        return busy;

    mesh_terminated = true;
    if(mesh_running)
        return true;

    modified = false;
    if(is_diffusion_space)
        to_diffusion_space.identity();
    if(region.empty())
    {
        if(region_render)
            region_render->object.reset();
        return false;
    }

    auto seeds = region;
    auto trans = to_diffusion_space;
    mesh_terminated = false;
    mesh_running = true;
    mesh_thread = std::thread([this,seeds = std::move(seeds),trans,smooth]
                              {
                                  auto r = std::make_shared<RegionRender>();
                                  if(r->load(seeds,trans,smooth,&mesh_terminated) && !mesh_terminated)
                                  {
                                      std::lock_guard lock(mesh_lock);
                                      pending_region_render = std::move(r);
                                  }
                                  mesh_running = false;
                              });
    return true;
}
// ---------------------------------------------------------------------------
void ROIRegion::from_mask(const tipl::image<3,unsigned char>& mask)
{
    if(!region.empty())
        undo_backup.push_back(std::move(region));
    region = tipl::volume2points(mask);
    set_modified();
}
// ---------------------------------------------------------------------------
void ROIRegion::to_mask(tipl::image<3,unsigned char>& mask) const
{
    if(!index_mask.empty() && index_mask_point_count == region.size())
    {
        tipl::threshold(index_mask,mask,0);
        return;
    }
    mask = tipl::points2volume(dim,region);
}
// ---------------------------------------------------------------------------
void ROIRegion::to_mask(tipl::image<3,unsigned char>& mask,const tipl::shape<3>& dim_to,const tipl::matrix<4,4>& trans_to) const
{
    auto m = to_mask();
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
    using namespace tipl::morphology;
    if(action == "smoothing")
        from_mask(smoothing(to_mask()));
    if(action == "erosion")
        from_mask(erosion(to_mask()));
    if(action == "dilation")
        from_mask(dilation(to_mask()));
    if(action == "opening")
        from_mask(opening(to_mask()));
    if(action == "closing")
        from_mask(closing(to_mask()));
    if(action == "defragment")
        from_mask(defragment(to_mask()));
    if(action == "negate")
        from_mask(negate(to_mask()));

}

// ---------------------------------------------------------------------------
void ROIRegion::flip_region(unsigned int dimension) {

    if(!region.empty())
        undo_backup.push_back(region);
    for (unsigned int index = 0; index < region.size(); ++index)
        region[index][dimension] = dim[dimension] - region[index][dimension] - 1;

    modified = true;
    if(!index_mask.empty())
        tipl::flip(index_mask,dimension);
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
    set_modified();
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
        value = I[pos];
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
