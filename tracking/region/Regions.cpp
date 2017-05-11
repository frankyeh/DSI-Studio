// ---------------------------------------------------------------------------
#include <QInputDialog>
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "fib_data.hpp"
#include "libs/gzip_interface.hpp"

// ---------------------------------------------------------------------------
void ROIRegion::add_points(std::vector<image::vector<3,float> >& points,bool del,float point_resolution)
{
    change_resolution(points,point_resolution);
    std::vector<image::vector<3,short> > new_points(points.size());
    for(int i = 0;i < points.size();++i)
    {
        points[i].round();
        new_points[i] = points[i];
    }
    add_points(new_points,del,resolution_ratio);
}
// ---------------------------------------------------------------------------
void ROIRegion::add_points(std::vector<image::vector<3,short> >& points, bool del,float point_resolution)
{
    change_resolution(points,point_resolution);
    if(!region.empty())
        undo_backup.push_back(region);
    if(resolution_ratio == 1.0)
    {
        for(unsigned int index = 0; index < points.size();)
            if (!geo.is_valid(points[index][0], points[index][1], points[index][2]))
            {
                points[index] = points.back();
                points.pop_back();
            }
            else
                ++index;
    }
    else
    {
        image::geometry<3> new_geo(geo[0]*resolution_ratio,geo[1]*resolution_ratio,geo[2]*resolution_ratio);
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
    if(points.size()+ region.size() > 1000000)
    {
        // for roll back
        unsigned int region_size = region.size();
        region.insert(region.end(),points.begin(),points.end());
        image::vector<3,short> min_value,max_value,geo_size;
        image::bounding_box_mt(region,max_value,min_value);

        geo_size = max_value-min_value;
        image::geometry<3> mask_geo(geo_size[0]+1,geo_size[1]+1,geo_size[2]+1);
        image::basic_image<unsigned char,3> mask;

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


        image::par_for (region.size(),[&](unsigned int index)
        {
            auto p = region[index];
            p -= min_value;
            if (mask.geometry().is_valid(p))
            {
                if(del)
                    mask.at(p[0],p[1],p[2]) = 0;
                else
                    mask.at(p[0],p[1],p[2]) = 1;
            }
        });

        points.clear();
        region.clear();
        for(image::pixel_index<3> index(mask.geometry());index.is_valid(mask.geometry());++index)
            if(mask[index.index()])
                region.push_back(image::vector<3,short>(index[0]+min_value[0],
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
                std::vector<image::vector<3,short> > union_points(region.size()+points.size());
                std::vector<image::vector<3,short> >::iterator it =
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
            std::vector<image::vector<3,short> > intersect_points(std::max(region.size(),points.size()));
            std::vector<image::vector<3,short> >::iterator it =
                std::set_intersection(region.begin(),region.end(),
                                      points.begin(),points.end(),
                                      intersect_points.begin());
            intersect_points.resize(it-intersect_points.begin());

            std::vector<image::vector<3,short> > remain_points(region.size());
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
void ROIRegion::SaveToFile(const char* FileName,const std::vector<float>& trans) {
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if (ext == std::string(".txt")) {
        std::ofstream out(FileName);
        std::copy(region.begin(), region.end(),std::ostream_iterator<image::vector<3,short> >(out, "\n"));
        if(resolution_ratio != 1.0)
            out << resolution_ratio << " -1 -1" << std::endl;
    }
    else if (ext == std::string(".mat")) {
        image::basic_image<unsigned char, 3> mask(geo);
        if(resolution_ratio != 1.0)
            mask.resize(image::geometry<3>(geo[0]*resolution_ratio,geo[1]*resolution_ratio,geo[2]*resolution_ratio));
        for (unsigned int index = 0; index < region.size(); ++index) {
            if (geo.is_valid(region[index][0], region[index][1],
                             region[index][2]))
                mask[image::pixel_index<3>(region[index][0], region[index][1],
                                           region[index][2], geo).index()] = 255;
        }
        image::io::mat_write header(FileName);
        header << mask;
    }
    else if (ext == std::string(".nii") || ext == std::string("i.gz"))
    {
        unsigned int color = show_region.color.color & 0x00FFFFFF;
        image::basic_image<unsigned char, 3>mask;
        SaveToBuffer(mask,1);
        gz_nifti header;
        if(resolution_ratio == 1.0)
            header.set_voxel_size(vs.begin());
        else
        {
            image::vector<3,float> rvs = vs;
            rvs /= resolution_ratio;
            header.set_voxel_size(rvs.begin());
        }
        // output color information and roi information
        std::ostringstream out;
        out << "color=" << color << ";roi=" << (int)regions_feature;
        std::string tmp = out.str();
        std::copy(tmp.begin(),tmp.begin() + std::min<int>(102,tmp.length()),
                  header.nif_header.descrip);
        if(!trans.empty())
        {
            if(resolution_ratio != 1.0)
            {
                std::vector<float> T(trans);
                image::multiply_constant(&T[0],&T[0]+3,1.0/resolution_ratio);
                image::multiply_constant(&T[4],&T[0]+3,1.0/resolution_ratio);
                image::multiply_constant(&T[8],&T[0]+3,1.0/resolution_ratio);
                header.set_image_transformation(T.begin());
            }
            else
                header.set_image_transformation(trans.begin());
        }
        else
            image::flip_xy(mask);
        header << mask;
        header.save_to_file(FileName);
    }
}

// ---------------------------------------------------------------------------

bool ROIRegion::LoadFromFile(const char* FileName,const std::vector<float>& trans) {

    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    modified = true;
    region.clear();

    if (ext == std::string(".txt"))
    {
        std::ifstream in(FileName,std::ios::binary);
        std::vector<image::vector<3,short> > points;
        std::copy(std::istream_iterator<image::vector<3,short> >(in),
                  std::istream_iterator<image::vector<3,short> >(),
                  std::back_inserter(points));
        if(points.back()[1] == -1 && points.back()[2] == -1)
        {
            resolution_ratio = std::min<float>(16.0f,points.back()[0]);
            points.pop_back();
        }
        region.swap(points);
        return true;
    }

    if (ext == std::string(".mat")) {
        image::io::mat_read header;
        if(!header.load_from_file(FileName))
            return false;
        image::basic_image<short, 3>from;
        header >> from;
        if(from.geometry() != geo)
        {
            float r1 = (float)from.geometry()[0]/(float)geo[0];
            float r2 = (float)from.geometry()[1]/(float)geo[1];
            float r3 = (float)from.geometry()[2]/(float)geo[2];
            if(r1 != r2 || r1 != r3)
                return false;
            resolution_ratio = r1;
        }
        std::vector<image::vector<3,short> > points;
        for (image::pixel_index<3> index(from.geometry());index < from.size();++index)
            if (from[index.index()])
                points.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
        add_points(points,false,resolution_ratio);
        return true;
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr") || ext == std::string("i.gz"))
    {
        gz_nifti header;
        if (!header.load_from_file(FileName))
            return false;
        // use unsigned int to avoid the nan background problem
        image::basic_image<unsigned int, 3>from;
        image::geometry<3> nii_geo(header.nif_header.dim+1);
        if(nii_geo != geo)// use transformation information
        {
            {
                float r1 = (float)nii_geo[0]/(float)geo[0];
                float r2 = (float)nii_geo[1]/(float)geo[1];
                float r3 = (float)nii_geo[2]/(float)geo[2];
                if(r1 == r2 || r1 == r3)
                {
                    resolution_ratio = r1;
                    header.toLPS(from);
                    LoadFromBuffer(from);
                    return true;
                }
            }

            if(trans.empty())
                return false;
            header >> from;
            image::matrix<4,4,float> t;
            t.identity();
            header.get_image_transformation(t.begin());
            t.inv();
            t *= trans;
            LoadFromBuffer(from,t);
            return true;
        }
        {
            image::basic_image<float, 3> tmp;
            header.toLPS(tmp);
            image::add_constant(tmp,0.5);
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
void ROIRegion::SaveToBuffer(image::basic_image<unsigned char, 3>& mask,
                             unsigned char value) {
    if(resolution_ratio != 1.0)
        mask.resize(image::geometry<3>(geo[0]*resolution_ratio,
                    geo[1]*resolution_ratio,
                    geo[2]*resolution_ratio));
    else
        mask.resize(geo);
    std::fill(mask.begin(), mask.end(), 0);
    image::par_for (region.size(),[&](unsigned int index)
    {
        if (mask.geometry().is_valid(region[index]))
            mask.at(region[index][0], region[index][1],region[index][2]) = value;
    });
}
// ---------------------------------------------------------------------------
void ROIRegion::perform(const std::string& action)
{
    image::basic_image<unsigned char, 3>mask;
    if(action == "smoothing")
    {
        SaveToBuffer(mask, 1);
        image::morphology::smoothing(mask);
        LoadFromBuffer(mask);
    }
    if(action == "erosion")
    {
        SaveToBuffer(mask, 1);
        image::morphology::erosion(mask);
        LoadFromBuffer(mask);
    }
    if(action == "dilation")
    {
        SaveToBuffer(mask, 1);
        image::morphology::dilation(mask);
        LoadFromBuffer(mask);
    }
    if(action == "defragment")
    {
        SaveToBuffer(mask, 1);
        image::morphology::defragment(mask);
        LoadFromBuffer(mask);
    }
    if(action == "negate")
    {
        SaveToBuffer(mask, 1);
        image::morphology::negate(mask);
        LoadFromBuffer(mask);
    }
    if(action == "flipx")
        Flip(0);
    if(action == "flipy")
        Flip(1);
    if(action == "flipz")
        Flip(2);
    if(action == "shiftx")
        shift(image::vector<3,float>(1.0, 0, 0));
    if(action == "shiftnx")
        shift(image::vector<3,float>(-1.0, 0, 0));
    if(action == "shifty")
        shift(image::vector<3,float>(0, 1.0, 0));
    if(action == "shiftny")
        shift(image::vector<3,float>(0, -1.0, 0));
    if(action == "shiftz")
        shift(image::vector<3,float>(0, 0, 1.0));
    if(action == "shiftnz")
        shift(image::vector<3,float>(0, 0, -1.0));
}

// ---------------------------------------------------------------------------
void ROIRegion::Flip(unsigned int dimension) {
    modified = true;
    if(!region.empty())
        undo_backup.push_back(region);
    for (unsigned int index = 0; index < region.size(); ++index)
        region[index][dimension] = (float)geo[dimension]*resolution_ratio -
                                   region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
void ROIRegion::shift(image::vector<3,float> dx) {
    show_region.move_object(dx);
    if(resolution_ratio != 1.0)
        dx *= resolution_ratio;
    dx.round();
    image::par_for(region.size(),[&](unsigned int index)
    {
        region[index] += dx;
    });
}
// ---------------------------------------------------------------------------
template<class Image,class Points>
void calculate_region_stat(const Image& I, const Points& p,float& mean,float& sd)
{
    float sum = 0.0,sum2 = 0.0;
    unsigned int count = 0;
    for(unsigned int index = 0; index < p.size(); ++index)
    {
        float value = I[p[index]];
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
    data.push_back(region.size()*vs[0]*vs[1]*vs[2]/resolution_ratio); //volume (mm^3)
    if(region.empty())
        return;
    image::vector<3,float> cm;
    image::vector<3,float> max(region[0]),min(region[0]);
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

    titles.push_back("bounding box x");
    titles.push_back("bounding box y");
    titles.push_back("bounding box z");
    std::copy(max.begin(),max.end(),std::back_inserter(data)); // bounding box

    titles.push_back("bounding box x");
    titles.push_back("bounding box y");
    titles.push_back("bounding box z");
    std::copy(min.begin(),min.end(),std::back_inserter(data)); // bounding box

    handle->get_index_titles(titles); // other index
    std::vector<unsigned int> pos_index;
    if(resolution_ratio == 1.0)
        for (unsigned int index = 0; index < region.size(); ++index)
            pos_index.push_back(image::pixel_index<3>(region[index][0],region[index][1],region[index][2],geo).index());
    else
        for (unsigned int index = 0; index < region.size(); ++index)
            pos_index.push_back(image::pixel_index<3>(region[index][0]/resolution_ratio,
                                region[index][1]/resolution_ratio,
                                region[index][2]/resolution_ratio,geo).index());


    for(int data_index = 0;data_index < handle->view_item.size(); ++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
        float mean,sd;
        image::const_pointer_image<float, 3> I(handle->view_item[data_index].image_data);
        calculate_region_stat(I,pos_index,mean,sd);
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
            image::const_pointer_image<float, 3> I(&fa_data[0][0],handle->dim);
            calculate_region_stat(I,pos_index,mean,sd);
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
