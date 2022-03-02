// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "libs/gzip_interface.hpp"

void ROIRegion::add_points(std::vector<tipl::vector<3,float> >& points,bool del,float point_resolution)
{
    change_resolution(points,point_resolution);
    std::vector<tipl::vector<3,short> > new_points(points.size());
    for(int i = 0;i < points.size();++i)
    {
        points[i].round();
        new_points[i] = points[i];
    }
    add_points(new_points,del,resolution_ratio);
}
// ---------------------------------------------------------------------------
void ROIRegion::add_points(std::vector<tipl::vector<3,short> >& points, bool del,float point_resolution)
{
    change_resolution(points,point_resolution);
    if(!region.empty())
        undo_backup.push_back(region);
    if(resolution_ratio == 1.0)
    {
        for(unsigned int index = 0; index < points.size();)
            if (!dim.is_valid(points[index][0], points[index][1], points[index][2]))
            {
                points[index] = points.back();
                points.pop_back();
            }
            else
                ++index;
    }
    else
    {
        tipl::shape<3> new_geo = dim*resolution_ratio;
        for(unsigned int index = 0; index < points.size();)
        if (!new_geo.is_valid(points[index][0], points[index][1], points[index][2]))
        {
            points[index] = points.back();
            points.pop_back();
        }
        else
            ++index;
    }
    if(points.empty())
        return;
    if(points.size()+ region.size() > 5000000)
    {
        // for roll back
        unsigned int region_size = region.size();
        if(!del)
        {
            region.insert(region.end(),points.begin(),points.end());
            points.clear();
        }
        tipl::vector<3,short> min_value,max_value,geo_size;
        tipl::bounding_box_mt(region,max_value,min_value);

        geo_size = max_value-min_value;
        tipl::shape<3> mask_geo(geo_size[0]+1,geo_size[1]+1,geo_size[2]+1);
        tipl::image<3,unsigned char> mask;

        try
        {
            mask.resize(mask_geo);
        }
        catch(...)
        {
            // roll back
            region.resize(region_size);
            goto alternative;
        }

        tipl::par_for (region.size(),[&](unsigned int index)
        {
            auto p = region[index];
            p -= min_value;
            if (mask.shape().is_valid(p))
                mask.at(p[0],p[1],p[2]) = 1;
        });

        if(points.size())
        tipl::par_for (points.size(),[&](unsigned int index)
        {
            auto p = points[index];
            p -= min_value;
            if (mask.shape().is_valid(p))
                mask.at(p[0],p[1],p[2]) = 0;
        });

        points.clear();
        region.clear();
        for(tipl::pixel_index<3> index(mask.shape());index < mask.size();++index)
            if(mask[index.index()])
                region.push_back(tipl::vector<3,short>(index[0]+min_value[0],
                                                    index[1]+min_value[1],
                                                    index[2]+min_value[2]));
    }
    else
    {
        alternative:
        std::sort(points.begin(),points.end());
        if(!del)
        {
            if(region.empty())
                region.swap(points);
            else
            {
                std::vector<tipl::vector<3,short> > union_points(region.size()+points.size());
                std::vector<tipl::vector<3,short> >::iterator it =
                    std::set_union(region.begin(),region.end(),
                                   points.begin(),points.end(),
                                   union_points.begin());
                union_points.resize(it-union_points.begin());
                region.swap(union_points);
            }
        }
        else
        {
            // find interset first
            std::vector<tipl::vector<3,short> > intersect_points(std::max(region.size(),points.size()));
            std::vector<tipl::vector<3,short> >::iterator it =
                std::set_intersection(region.begin(),region.end(),
                                      points.begin(),points.end(),
                                      intersect_points.begin());
            intersect_points.resize(it-intersect_points.begin());

            std::vector<tipl::vector<3,short> > remain_points(region.size());
            it = std::set_difference(region.begin(),region.end(),
                                     intersect_points.begin(),intersect_points.end(),
                                     remain_points.begin());
            remain_points.resize(it-remain_points.begin());
            region.swap(remain_points);
        }
        region.erase(std::unique(region.begin(), region.end()), region.end());
    }
    modified = true;

}

// ---------------------------------------------------------------------------
void ROIRegion::SaveToFile(const char* FileName)
{
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if (ext == std::string(".txt")) {
        std::ofstream out(FileName);
        std::copy(region.begin(), region.end(),std::ostream_iterator<tipl::vector<3,short> >(out, "\n"));
        if(resolution_ratio != 1.0f)
            out << resolution_ratio << " -1 -1" << std::endl;
    }
    else if (ext == std::string(".mat")) {
        if(resolution_ratio > 8.0f)
            return;
        tipl::image<3,unsigned char> mask(dim*resolution_ratio);
        for (unsigned int index = 0; index < region.size(); ++index) {
            if (dim.is_valid(region[index][0], region[index][1],
                             region[index][2]))
                mask[tipl::pixel_index<3>(region[index][0], region[index][1],
                                           region[index][2], dim).index()] = 255;
        }
        tipl::io::mat_write header(FileName);
        header << mask;
    }
    else if (ext == std::string(".nii") || ext == std::string("i.gz"))
    {
        if(resolution_ratio > 8.0f)
            return;
        unsigned int color = show_region.color.color & 0x00FFFFFF;
        tipl::image<3,unsigned char>mask;
        SaveToBuffer(mask);
        tipl::vector<3,float> rvs(vs);
        tipl::matrix<4,4> T(trans_to_mni);
        if(resolution_ratio != 1.0f)
        {
            rvs /= resolution_ratio;
            tipl::multiply_constant(&T[0],&T[0]+3,1.0/resolution_ratio);
            tipl::multiply_constant(&T[4],&T[4]+3,1.0/resolution_ratio);
            tipl::multiply_constant(&T[8],&T[8]+3,1.0/resolution_ratio);
        }
        std::ostringstream out;
        out << "color=" << (int)color << ";roi=" << (int)regions_feature;
        std::string tmp = out.str();
        if(tmp.size() < 80)
            tmp.resize(80);
        gz_nifti::save_to_file(FileName,mask,rvs,T,is_mni,tmp.c_str());

    }
}

// ---------------------------------------------------------------------------

bool ROIRegion::LoadFromFile(const char* FileName) {
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    modified = true;
    region.clear();

    if (ext == std::string(".txt"))
    {
        std::ifstream in(FileName,std::ios::binary);
        std::vector<tipl::vector<3,short> > points;
        std::copy(std::istream_iterator<tipl::vector<3,short> >(in),
                  std::istream_iterator<tipl::vector<3,short> >(),
                  std::back_inserter(points));
        if(!points.empty() && points.back()[1] == -1.0f && points.back()[2] == -1.0f)
        {
            resolution_ratio = points.back()[0];
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
        if(from.shape() != dim)
        {
            float r1 = (float)from.shape()[0]/(float)dim[0];
            float r2 = (float)from.shape()[1]/(float)dim[1];
            float r3 = (float)from.shape()[2]/(float)dim[2];
            if(r1 != r2 || r1 != r3)
                return false;
            resolution_ratio = r1;
        }
        std::vector<tipl::vector<3,short> > points;
        for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
            if (from[index.index()])
                points.push_back(tipl::vector<3,short>((const unsigned int*)index.begin()));
        add_points(points,false,resolution_ratio);
        return true;
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr") || ext == std::string("i.gz"))
    {
        gz_nifti header;
        if (!header.load_from_file(FileName))
        {
            std::cout << header.error << std::endl;
            return false;
        }
        // use unsigned int to avoid the nan background problem
        tipl::image<3,unsigned int> from;
        tipl::shape<3> nii_geo;
        header.get_image_dimension(nii_geo);
        is_mni = header.is_mni();
        if(nii_geo != dim)// use transformation information
        {
            {
                float r1 = (float)nii_geo[0]/(float)dim[0];
                float r2 = (float)nii_geo[1]/(float)dim[1];
                float r3 = (float)nii_geo[2]/(float)dim[2];
                if(r1 == r2 && r1 == r3)
                {
                    resolution_ratio = r1;
                    header.toLPS(from);
                    LoadFromBuffer(from);
                    return true;
                }
            }
            header.get_untouched_image(from);
            tipl::matrix<4,4> t;
            header.get_image_transformation(t);
            LoadFromBuffer(from,tipl::from_space(trans_to_mni).to(t));
            return true;
        }
        {
            tipl::image<3> tmp;
            header.toLPS(tmp);
            tipl::add_constant(tmp,0.5);
            from = tmp;
        }
        LoadFromBuffer(from);
        return true;
    }
    return false;

}

void ROIRegion::makeMeshes(unsigned char smooth)
{
    if(!modified)
        return;
    modified = false;
    show_region.load(region,resolution_ratio,smooth);
}
// ---------------------------------------------------------------------------
void ROIRegion::SaveToBuffer(tipl::image<3,unsigned char>& mask,
                             float target_resolution)
{
    mask.resize(dim*target_resolution);
    mask = 0;
    if(target_resolution == resolution_ratio)
        tipl::par_for (region.size(),[&](unsigned int index)
        {
            if (mask.shape().is_valid(region[index]))
                mask.at(region[index][0], region[index][1],region[index][2]) = 1;
        });
    else
    {
        float r = target_resolution/resolution_ratio;
        tipl::par_for (region.size(),[&](unsigned int index)
        {
            tipl::vector<3,float> p(region[index]);
            p *= r;
            p.round();
            tipl::vector<3,short> pp(p);
            if (mask.shape().is_valid(pp))
                mask.at(pp[0],pp[1],pp[2]) = 1;
        });
    }
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


    if(resolution_ratio > 8)
        return;
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
        region[index][dimension] = (float)dim[dimension]*resolution_ratio -
                                   region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
bool ROIRegion::shift(tipl::vector<3,float> dx) {
    if(resolution_ratio != 1.0f)
        dx *= resolution_ratio;
    dx.round();
    if(dx[0] == 0.0f && dx[1] == 0.0f && dx[2] == 0.0f)
        return false;
    show_region.move_object(dx/resolution_ratio);
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
    return region.size()*vs[0]*vs[1]*vs[2]/resolution_ratio/resolution_ratio/resolution_ratio;
}

tipl::vector<3> ROIRegion::get_pos(void) const
{
    tipl::vector<3,float> cm;
    for (unsigned int index = 0; index < region.size(); ++index)
        cm += region[index];
    cm /= region.size();
    cm /= resolution_ratio;
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
    max /= resolution_ratio;
    min /= resolution_ratio;

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
        points.push_back(tipl::vector<3>(region[index][0]/resolution_ratio,
                                          region[index][1]/resolution_ratio,
                                          region[index][2]/resolution_ratio));
    // get mean, max, min value of each index
    std::vector<float> max_values,min_values;
    for(size_t data_index = 0;data_index < handle->view_item.size(); ++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
        float mean;
        max_values.push_back(0.0f);
        min_values.push_back(0.0f);
        if(handle->view_item[data_index].get_image().shape() != handle->dim)
            calculate_region_stat(handle->view_item[data_index].get_image(),points,mean,max_values.back(),min_values.back(),
                                  &handle->view_item[data_index].iT[0]);
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
        for(unsigned int normalize_qa = 0;normalize_qa <= 1;++normalize_qa)
        for(unsigned int subject_index = 0;subject_index < handle->db.num_subjects;++subject_index)
        {
            std::vector<std::vector<float> > fa_data;
            handle->db.get_subject_fa(subject_index,fa_data,normalize_qa);
            float mean,max,min;
            tipl::const_pointer_image<3> I(&fa_data[0][0],handle->dim);
            calculate_region_stat(I,points,mean,max,min);
            data.push_back(mean);
            std::ostringstream out;
            out << handle->db.subject_names[subject_index] << (normalize_qa ? " mean_normalized_":" mean_") << handle->db.index_name;
            titles.push_back(out.str());
        }
    }
}
