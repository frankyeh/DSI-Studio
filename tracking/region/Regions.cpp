// ---------------------------------------------------------------------------
#include <fstream>
#include <iterator>
#include "Regions.h"
#include "SliceModel.h"
#include "mat_file.hpp"
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
void ROIRegion::SaveToFile(const char* FileName) {
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
                std::string out_temp = program_base + ".nii";
		image::io::nifti header;
                header.set_voxel_size(vs.begin());
                // from +x = Left  +y = Posterior +z = Superior
                // to +x = Right  +y = Anterior +z = Superior
                image::flip_xy(mask);
		header << mask;
                if(ext == std::string(".nii"))
                {
                    header.save_to_file(FileName);
                    return;
                }
                header.save_to_file(out_temp.c_str());
                std::vector<char> buf(1000);
                gzFile id = gzopen(FileName,"wb");
                std::ifstream in(out_temp.c_str(),std::ios::binary);
                while(in)
                {
                    in.read(&*buf.begin(),1000);
                    ::gzwrite(id,&*buf.begin(),in.gcount());
                }
                gzclose(id);
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

    if (ext == std::string("i.gz"))
    {
        gzFile id = gzopen(FileName,"rb");
        std::vector<char> buf(1000);
        unsigned int size = 0;
        std::string out_temp = program_base + ".nii";
        std::ofstream out(out_temp.c_str(),std::ios::binary);
        while(size = gzread(id,&*buf.begin(),1000))
            out.write(&*buf.begin(),size);
        gzclose(id);
        out.close();
        return LoadFromFile(out_temp.c_str());
    }

    if (ext == std::string(".nii") || ext == std::string(".hdr"))
    {
        image::io::nifti header;
        if (!header.load_from_file(FileName))
            return false;
        image::basic_image<short, 3>from;
        header >> from;
        if(from.geometry() != geo)
            return false;
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
void ROIRegion::updateMesh(bool smooth)
{
    image::basic_image<unsigned char, 3> mask;
    SaveToBuffer(mask,200);
    if(smooth)
        image::filter::gaussian(mask);
    std::auto_ptr<RegionModel> new_region(new RegionModel);
    new_region->color = show_region.color;
    new_region->alpha = show_region.alpha;
    new_region->load(mask,20);
    back_region.reset(new_region.release());
}
// ---------------------------------------------------------------------------
void ROIRegion::makeMeshes(bool smooth)
{
    if (modified && !back_thread.get())
    {
        back_thread.reset(new boost::thread(&ROIRegion::updateMesh,this,smooth));
        modified = false;
    }
    if(back_region.get())
    {
        show_region.object.reset(back_region->object.release());
        show_region.sorted_index.swap(back_region->sorted_index);
        back_region.reset(0);
        back_thread.reset(0);
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
void ROIRegion::Flip(unsigned int dimension) {
	modified = true;
        for (unsigned int index = 0; index < region.size(); ++index)
		region[index][dimension] = ((short)geo[dimension]) -
			region[index][dimension] - 1;
}

// ---------------------------------------------------------------------------
void ROIRegion::shift(const image::vector<3,short>& dx) {
	modified = true;
        for (unsigned int index = 0; index < region.size(); ++index)
		region[index] += dx;
}
