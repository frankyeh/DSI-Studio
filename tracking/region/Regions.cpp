// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "libs/gzip_interface.hpp"

tipl::geometry<3> ROIRegion::get_buffer_dim(void) const
{
    return tipl::geometry<3>(handle->dim[0]*resolution_ratio,
                                      handle->dim[1]*resolution_ratio,
                                      handle->dim[2]*resolution_ratio);
}

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
            if (!handle->dim.is_valid(points[index][0], points[index][1], points[index][2]))
            {
                points[index] = points.back();
                points.pop_back();
            }
            else
                ++index;
    }
    else
    {
        tipl::geometry<3> new_geo = get_buffer_dim();
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
        tipl::geometry<3> mask_geo(geo_size[0]+1,geo_size[1]+1,geo_size[2]+1);
        tipl::image<unsigned char,3> mask;

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
            if (mask.geometry().is_valid(p))
                mask.at(p[0],p[1],p[2]) = 1;
        });

        if(points.size())
        tipl::par_for (points.size(),[&](unsigned int index)
        {
            auto p = points[index];
            p -= min_value;
            if (mask.geometry().is_valid(p))
                mask.at(p[0],p[1],p[2]) = 0;
        });

        points.clear();
        region.clear();
        for(tipl::pixel_index<3> index(mask.geometry());index.is_valid(mask.geometry());++index)
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
        tipl::image<unsigned char, 3> mask(handle->dim);
        if(resolution_ratio != 1.0f)
            mask.resize(get_buffer_dim());
        for (unsigned int index = 0; index < region.size(); ++index) {
            if (handle->dim.is_valid(region[index][0], region[index][1],
                             region[index][2]))
                mask[tipl::pixel_index<3>(region[index][0], region[index][1],
                                           region[index][2], handle->dim).index()] = 255;
        }
        tipl::io::mat_write header(FileName);
        header << mask;
    }
    else if (ext == std::string(".nii") || ext == std::string("i.gz"))
    {
        if(resolution_ratio > 8.0f)
            return;
        unsigned int color = show_region.color.color & 0x00FFFFFF;
        tipl::image<unsigned char, 3>mask;
        SaveToBuffer(mask);
        tipl::vector<3,float> rvs(handle->vs);
        tipl::matrix<4,4,float> T(handle->trans_to_mni);
        if(resolution_ratio != 1.0f)
        {
            rvs /= resolution_ratio;
            tipl::multiply_constant(&T[0],&T[0]+3,1.0/resolution_ratio);
            tipl::multiply_constant(&T[4],&T[4]+3,1.0/resolution_ratio);
            tipl::multiply_constant(&T[8],&T[8]+3,1.0/resolution_ratio);
        }
        std::ostringstream out;
        out << "color=" << color << ";roi=" << (int)regions_feature;
        std::string tmp = out.str();
        if(tmp.size() < 80)
            tmp.resize(80);

        gz_nifti::save_to_file(FileName,mask,rvs,T,tmp.c_str());

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
        tipl::image<short, 3>from;
        header >> from;
        if(from.geometry() != handle->dim)
        {
            float r1 = (float)from.geometry()[0]/(float)handle->dim[0];
            float r2 = (float)from.geometry()[1]/(float)handle->dim[1];
            float r3 = (float)from.geometry()[2]/(float)handle->dim[2];
            if(r1 != r2 || r1 != r3)
                return false;
            resolution_ratio = r1;
        }
        std::vector<tipl::vector<3,short> > points;
        for (tipl::pixel_index<3> index(from.geometry());index < from.size();++index)
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
        tipl::image<unsigned int, 3>from;
        tipl::geometry<3> nii_geo;
        header.get_image_dimension(nii_geo);
        if(nii_geo != handle->dim)// use transformation information
        {
            {
                float r1 = (float)nii_geo[0]/(float)handle->dim[0];
                float r2 = (float)nii_geo[1]/(float)handle->dim[1];
                float r3 = (float)nii_geo[2]/(float)handle->dim[2];
                if(r1 == r2 && r1 == r3)
                {
                    resolution_ratio = r1;
                    header.toLPS(from);
                    LoadFromBuffer(from);
                    return true;
                }
            }

            if(!handle->is_qsdr)
                return false;
            header >> from;
            tipl::matrix<4,4,float> t;
            header.get_image_transformation(t);
            t.inv();
            t *= handle->trans_to_mni;
            LoadFromBuffer(from,t);
            return true;
        }
        {
            tipl::image<float, 3> tmp;
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
void ROIRegion::SaveToBuffer(tipl::image<unsigned char, 3>& mask,
                             float target_resolution)
{
    if(target_resolution != 1.0f)
        mask.resize(tipl::geometry<3>(
                    handle->dim[0]*target_resolution,
                    handle->dim[1]*target_resolution,
                    handle->dim[2]*target_resolution));
    else
        mask.resize(handle->dim);
    std::fill(mask.begin(), mask.end(), 0);
    if(target_resolution == resolution_ratio)
        tipl::par_for (region.size(),[&](unsigned int index)
        {
            if (mask.geometry().is_valid(region[index]))
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
            if (mask.geometry().is_valid(pp))
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
    tipl::image<unsigned char, 3>mask;
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
        region[index][dimension] = (float)handle->dim[dimension]*resolution_ratio -
                                   region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
void ROIRegion::shift(tipl::vector<3,float> dx) {
    show_region.move_object(dx);
    if(resolution_ratio != 1.0)
        dx *= resolution_ratio;
    dx.round();
    tipl::par_for(region.size(),[&](unsigned int index)
    {
        region[index] += dx;
    });
}
// ---------------------------------------------------------------------------
template<class Image,class Points>
void calculate_region_stat(const Image& I, const Points& p,float& mean,float& sd,const float* T = 0)
{
    float sum = 0.0,sum2 = 0.0;
    unsigned int count = 0;
    for(unsigned int index = 0; index < p.size(); ++index)
    {
        float value = 0.0;
        tipl::vector<3> pos(p[index]);
        if(T)
            pos.to(T);
        value = tipl::estimate(I,pos);
        if(value == 0.0)
            continue;
        sum += value;
        sum2 += value*value;
        ++count;
    }
    sum /= count;
    sum2 /= count;
    mean = sum;
    sd = std::sqrt(std::max<float>(0.0,sum2-sum*sum));
}

void ROIRegion::get_quantitative_data(std::shared_ptr<fib_data> handle,std::vector<std::string>& titles,std::vector<float>& data)
{
    titles.clear();
    titles.push_back("voxel counts");
    data.push_back(region.size());

    titles.push_back("volume (mm^3)");
    data.push_back(region.size()*handle->vs[0]*handle->vs[1]*handle->vs[2]/resolution_ratio); //volume (mm^3)
    if(region.empty())
        return;
    tipl::vector<3,float> cm;
    tipl::vector<3,float> max(region[0]),min(region[0]);
    for (unsigned int index = 0; index < region.size(); ++index)
    {
        cm += region[index];
        max[0] = std::max<short>(max[0],region[index][0]);
        max[1] = std::max<short>(max[1],region[index][1]);
        max[2] = std::max<short>(max[2],region[index][2]);
        min[0] = std::min<short>(min[0],region[index][0]);
        min[1] = std::min<short>(min[1],region[index][1]);
        min[2] = std::min<short>(min[2],region[index][2]);
    }
    cm /= region.size();
    titles.push_back("center x");
    titles.push_back("center y");
    titles.push_back("center z");
    std::copy(cm.begin(),cm.end(),std::back_inserter(data)); // center of the mass

    if(!handle->mni_position.empty())
    {
        tipl::vector<3> mni;
        tipl::estimate(handle->mni_position,cm,mni);
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

    handle->get_index_titles(titles); // other index
    std::vector<tipl::vector<3> > points;
    for (unsigned int index = 0; index < region.size(); ++index)
        points.push_back(tipl::vector<3>(region[index][0]/resolution_ratio,
                                          region[index][1]/resolution_ratio,
                                          region[index][2]/resolution_ratio));


    for(int data_index = 0;data_index < handle->view_item.size(); ++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
        float mean,sd;
        if(handle->view_item[data_index].image_data.geometry() != handle->dim)
            calculate_region_stat(handle->view_item[data_index].image_data,points,mean,sd,&handle->view_item[data_index].iT[0]);
        else
            calculate_region_stat(handle->view_item[data_index].image_data,points,mean,sd);
        data.push_back(mean);
        data.push_back(sd);
    }

    if(handle->db.has_db()) // connectometry database
    {
        for(unsigned int subject_index = 0;subject_index < handle->db.num_subjects;++subject_index)
        {
            std::vector<std::vector<float> > fa_data;
            handle->db.get_subject_fa(subject_index,fa_data);
            float mean,sd;
            tipl::const_pointer_image<float, 3> I(&fa_data[0][0],handle->dim);
            calculate_region_stat(I,points,mean,sd);
            data.push_back(mean);
            data.push_back(sd);

            std::ostringstream out1,out2;
            out1 << handle->db.subject_names[subject_index] << " " << handle->db.index_name << " mean";
            out2 << handle->db.subject_names[subject_index] << " " << handle->db.index_name << " sd";
            titles.push_back(out1.str());
            titles.push_back(out2.str());

        }
    }
}
