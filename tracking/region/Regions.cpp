// ---------------------------------------------------------------------------
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "libs/tracking/tracking_model.hpp"
#include "libs/gzip_interface.hpp"


// ---------------------------------------------------------------------------
void ROIRegion::add_points(std::vector<image::vector<3,short> >& points, bool del)
{
    for(unsigned int index = 0;index < points.size();)
       if (!geo.is_valid(points[index][0], points[index][1], points[index][2]))
        {
            points[index] = points.back();
            points.pop_back();
        }
    else
        ++index;
    if(points.empty())
        return;
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
    modified = true;
}

// ---------------------------------------------------------------------------
extern std::string program_base;
void ROIRegion::SaveToFile(const char* FileName,const std::vector<float>& trans) {
    std::string file_name(FileName);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if (ext == std::string(".txt")) {
                std::ofstream out(FileName,std::ios::binary);
		std::copy(region.begin(), region.end(),
                        std::ostream_iterator<image::vector<3,short> >(out, "\n"));
	}
        else if (ext == std::string(".mat")) {
                image::basic_image<unsigned char, 3>mask(geo);
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
        image::basic_image<unsigned int, 3>mask(geo);
                for (unsigned int index = 0; index < region.size(); ++index) {
                        if (geo.is_valid(region[index][0], region[index][1],
				region[index][2]))
				mask[image::pixel_index<3>(region[index][0], region[index][1],
                region[index][2], geo).index()] = color;
		}
                gz_nifti header;
                header.set_voxel_size(vs.begin());
                if(!trans.empty())
                    header.set_image_transformation(trans.begin());
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
        while(in)
        {
            image::vector<3,float> point;
            in >> point;
            point += 0.5;
            point.floor();
            points.push_back(image::vector<3,short>(point[0],point[1],point[2]));
        }
        region.swap(points);
        std::sort(region.begin(),region.end());
        return true;
    }

    if (ext == std::string(".mat")) {
        image::io::mat_read header;
        if(!header.load_from_file(FileName))
            return false;
        image::basic_image<short, 3>from;
        header >> from;
        if(from.geometry() != geo)
            return false;
        std::vector<image::vector<3,short> > points;
        for (image::pixel_index<3>index; index.valid(from.geometry());index.next(from.geometry()))
            if (from[index.index()])
                points.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
        region.swap(points);
        std::sort(region.begin(),region.end());
        return true;
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr") || ext == std::string("i.gz"))
    {
        gz_nifti header;
        if (!header.load_from_file(FileName))
            return false;
        // use unsigned int to avoid the nan background problem
        image::basic_image<unsigned int, 3>from;
        header >> from;
        if(from.geometry() != geo)// use transformation information
        {
            if(trans.empty())
                return false;
            std::vector<float> t(header.get_transformation(),
                                 header.get_transformation()+12),inv_trans(16),convert(16);
            t.resize(16);
            t[15] = 1.0;
            image::matrix::inverse(t.begin(),inv_trans.begin(),image::dim<4,4>());
            image::matrix::product(inv_trans.begin(),trans.begin(),convert.begin(),image::dim<4,4>(),image::dim<4,4>());
            LoadFromBuffer(from,convert);
            return true;
        }
        // from +x = Right  +y = Anterior +z = Superior
        // to +x = Left  +y = Posterior +z = Superior
        if(header.nif_header.srow_x[0] < 0)
        {
            if(header.nif_header.srow_y[1] > 0)
                image::flip_y(from);
        }
        else
            image::flip_xy(from);
        LoadFromBuffer(from);
        float max_value = *std::max_element(from.begin(),from.end());
        if(max_value > 128 && max_value < 0x00FFFFFF)
            show_region.color = (unsigned int)max_value;
        return true;
    }
    return false;

}
// ---------------------------------------------------------------------------
std::vector<boost::thread*> back_thread;
std::vector<RegionModel*> back_region;
void updateMesh(unsigned int id,image::geometry<3> geo,
                std::vector<image::vector<3,short> > region,bool smooth)
{
    image::basic_image<unsigned char, 3> mask(geo);
    for (unsigned int index = 0; index < region.size(); ++index)
    {
        if (geo.is_valid(region[index][0], region[index][1], region[index][2]))
            mask[image::pixel_index<3>(region[index][0], region[index][1],
            region[index][2], geo).index()] = 200;
    }
    if(smooth)
        image::filter::gaussian(mask);
    std::auto_ptr<RegionModel> new_region(new RegionModel);
    new_region->load(mask,20);
    if(back_thread[id])
        back_region[id] = new_region.release();

}
// ---------------------------------------------------------------------------
ROIRegion::~ROIRegion(void)
{
    if(has_back_thread)
    {
        delete back_thread[back_thread_id];
        delete back_region[back_thread_id];
        back_thread[back_thread_id] = 0;
        back_region[back_thread_id] = 0;
    }
}
// ---------------------------------------------------------------------------
bool ROIRegion::need_update(void)
{
    if(!has_back_thread)
        return false;
    if(has_back_thread && back_region[back_thread_id])
    {
        show_region.object.reset(back_region[back_thread_id]->object.release());
        show_region.sorted_index.swap(back_region[back_thread_id]->sorted_index);
        delete back_thread[back_thread_id];
        delete back_region[back_thread_id];
        back_thread[back_thread_id] = 0;
        back_region[back_thread_id] = 0;
        has_back_thread = false;
        return true;
    }
    return false;
}

void ROIRegion::makeMeshes(bool smooth)
{
    /*
    if (modified && !has_back_thread)
    {
        modified = false;
        back_thread_id = back_thread.size();
        has_back_thread = true;
        back_region.push_back(0);
        back_thread.push_back(new boost::thread(&updateMesh,back_thread_id,geo,region,smooth));
    }
    */
    if(!modified)
        return;
    modified = false;
    if(smooth)
    {
        image::basic_image<unsigned char, 3> mask(geo);
        SaveToBuffer(mask,200);
        image::filter::gaussian(mask);
        show_region.load(mask,20);
    }
    else
        show_region.load(region);
}
// ---------------------------------------------------------------------------
void ROIRegion::SaveToBuffer(image::basic_image<unsigned char, 3>& mask,
	unsigned char value) {
	mask.resize(geo);
	std::fill(mask.begin(), mask.end(), 0);
        for (unsigned int index = 0; index < region.size(); ++index) {
		if (geo.is_valid(region[index][0], region[index][1], region[index][2]))
			mask[image::pixel_index<3>(region[index][0], region[index][1],
			region[index][2], geo).index()] = value;
	}
}

// ---------------------------------------------------------------------------
void ROIRegion::getSlicePosition(SliceModel* slice, unsigned int pindex, int& x, int& y,
	int& z) {
	slice->getSlicePosition(region[pindex].x(), region[pindex].y(),
		region[pindex].z(), x, y, z);
}
// ---------------------------------------------------------------------------
bool ROIRegion::has_point(const image::vector<3,short>& point)
{
    return std::binary_search(region.begin(),region.end(),point);
}
// ---------------------------------------------------------------------------
bool ROIRegion::has_points(const std::vector<image::vector<3,short> >& points)
{
    for(unsigned int index = 0;index < points.size();++index)
        if(has_point(points[index]))
            return true;
    return false;
}
// ---------------------------------------------------------------------------
void ROIRegion::Flip(unsigned int dimension) {
	modified = true;
        for (unsigned int index = 0; index < region.size(); ++index)
		region[index][dimension] = ((short)geo[dimension]) -
			region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
void ROIRegion::shift(const image::vector<3,short>& dx) {
    image::vector<3,float> shift(dx);
    show_region.move_object(shift);
    for (unsigned int index = 0; index < region.size(); ++index)
		region[index] += dx;
}

void ROIRegion::get_quantitative_data(ODFModel* handle,std::vector<float>& data)
{
    data.push_back(region.size()); //number of voxels
    data.push_back(region.size()*vs[0]*vs[1]*vs[2]); //volume (mm^3)
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
    std::copy(cm.begin(),cm.end(),std::back_inserter(data)); // center of the mass
    std::copy(max.begin(),max.end(),std::back_inserter(data)); // bounding box
    std::copy(min.begin(),min.end(),std::back_inserter(data)); // bounding box

    std::vector<unsigned int> pos_index;
    for (unsigned int index = 0; index < region.size(); ++index)
        pos_index.push_back(image::pixel_index<3>(region[index][0],region[index][1],region[index][2],geo).index());

    for(int data_index = 0;
        data_index < handle->fib_data.view_item.size();++data_index)
    {
        if(data_index > 0 && data_index < handle->fib_data.other_mapping_index)
            continue;
        float sum = 0.0,sum2 = 0.0;
        image::const_pointer_image<float, 3> I(handle->fib_data.view_item[data_index].image_data);
        for(unsigned int index = 0;index < pos_index.size();++index)
        {
            float value = I[pos_index[index]];
            sum += value;
            sum2 += value*value;
        }
        sum /= pos_index.size();
        sum2 /= pos_index.size();
        data.push_back(sum);
        data.push_back(std::sqrt(sum2-sum*sum));
    }
}
