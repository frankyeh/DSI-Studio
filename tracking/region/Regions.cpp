// ---------------------------------------------------------------------------
#include <fstream>
#include <iterator>
#include <math/matrix_op.hpp>
#include "Regions.h"
#include "SliceModel.h"
#include "mat_file.hpp"
#include "libs/gzip_interface.hpp"
typedef class ReadMatFile MatReader;
typedef class WriteMatFile MatWriter;

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
                image::io::mat header;
                header << mask;
                if (!header.save_to_file(FileName))
                    return;

        }
        else if (ext == std::string(".nii") || ext == std::string("i.gz")) {
		image::basic_image<unsigned char, 3>mask(geo);
                for (unsigned int index = 0; index < region.size(); ++index) {
                        if (geo.is_valid(region[index][0], region[index][1],
				region[index][2]))
				mask[image::pixel_index<3>(region[index][0], region[index][1],
				region[index][2], geo).index()] = 255;
		}
                gz_nifti header;
                header.set_voxel_size(vs.begin());
                header.set_image_transformation(trans.begin());
                // from +x = Left  +y = Posterior +z = Superior
                // to +x = Right  +y = Anterior +z = Superior
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
        add_points(points,false);
        return true;
    }

    if (ext == std::string(".mat")) {
        image::io::mat header;
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
        add_points(points,false);
        return true;
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr") || ext == std::string("i.gz"))
    {
        gz_nifti header;
        if (!header.load_from_file(FileName))
            return false;
        image::basic_image<short, 3>from;
        header >> from;
        if(from.geometry() != geo)// use transformation information
        {
            if(trans.empty())
                return false;
            std::vector<float> t(header.get_transformation(),
                                 header.get_transformation()+12),inv_trans(16),convert(16);
            t.resize(16);
            t[15] = 1.0;
            math::matrix_inverse(trans.begin(),inv_trans.begin(),math::dim<4,4>());
            math::matrix_product(inv_trans.begin(),t.begin(),convert.begin(),math::dim<4,4>(),math::dim<4,4>());
            std::vector<image::vector<3,short> > points;
            for (image::pixel_index<3>index; index.valid(from.geometry());index.next(from.geometry()))
            {
                if (from[index.index()])
                {
                    image::vector<3> p(index.begin()),p2;
                    image::vector_transformation(p.begin(),p2.begin(),convert.begin(),image::vdim<3>());
                    points.push_back(image::vector<3,short>(std::floor(p2[0]+0.5),
                                                            std::floor(p2[1]+0.5),
                                                            std::floor(p2[2]+0.5)));
                }
            }
            add_points(points,false);
            return true;
        }
        // from +x = Right  +y = Anterior +z = Superior
        // to +x = Left  +y = Posterior +z = Superior
        if(header.nif_header.srow_x[0] < 0 || !header.is_nii)
            image::flip_y(from);
        else
            image::flip_xy(from);

        std::vector<image::vector<3,short> > points;
        for (image::pixel_index<3>index; index.valid(from.geometry());index.next(from.geometry()))
        {
            if (from[index.index()])
                points.push_back(image::vector<3,short>((const unsigned int*)index.begin()));
        }
        add_points(points,false);
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
void ROIRegion::makeMeshes(bool smooth)
{
    if(has_back_thread && back_thread[back_thread_id] == 0)
        has_back_thread = false;
    if (modified && !has_back_thread)
    {
        back_thread_id = back_thread.size();
        has_back_thread = true;
        back_thread.push_back(new boost::thread(&updateMesh,back_thread_id,geo,region,smooth));
        back_region.push_back(0);
        modified = false;
    }
    if(has_back_thread && back_region[back_thread_id])
    {
        show_region.object.reset(back_region[back_thread_id]->object.release());
        show_region.sorted_index.swap(back_region[back_thread_id]->sorted_index);
        delete back_thread[back_thread_id];
        delete back_region[back_thread_id];
        back_thread[back_thread_id] = 0;
        back_region[back_thread_id] = 0;
        has_back_thread = false;
    }


}
// ---------------------------------------------------------------------------
void ROIRegion::LoadFromBuffer(const image::basic_image<unsigned char, 3>& mask) {
	modified = true;
	geo = mask.geometry();
	region.clear();
	for (image::pixel_index<3>index; mask.geometry().is_valid(index);
	index.next(mask.geometry()))
		if (mask[index.index()])
                        region.push_back(image::vector<3,short>(index.x(), index.y(),
			index.z()));
        std::sort(region.begin(),region.end());
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
