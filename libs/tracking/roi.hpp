#ifndef ROI_HPP
#include <functional>
#include <set>
#include "tipl/tipl.hpp"
#include "tract_model.hpp"

class Roi {
private:
    float ratio;
    tipl::geometry<3> dim;
    std::vector<std::vector<std::vector<unsigned char> > > roi_filter;
    bool in_range(int x,int y,int z) const
    {
        return dim.is_valid(x,y,z);
    }
public:
    Roi(const tipl::geometry<3>& geo,float r):ratio(r)
    {
        if(r == 1.0)
            dim = geo;
        else
            dim = tipl::geometry<3>(geo[0]*r,geo[1]*r,geo[2]*r);
        roi_filter.resize(dim[0]);
    }
    void clear(void)
    {
        roi_filter.clear();
        roi_filter.resize(dim[0]);
    }
    void addPoint(const tipl::vector<3,short>& new_point)
    {
        if(in_range(new_point.x(),new_point.y(),new_point.z()))
        {
            if(roi_filter[new_point.x()].empty())
                roi_filter[new_point.x()].resize(dim[1]);
            if(roi_filter[new_point.x()][new_point.y()].empty())
                roi_filter[new_point.x()][new_point.y()].resize(dim[2]);
            roi_filter[new_point.x()][new_point.y()][new_point.z()] = 1;
        }
    }
    bool havePoint(float dx,float dy,float dz) const
    {
        short x,y,z;
        if(ratio != 1.0)
        {
            x = std::round(dx*ratio);
            y = std::round(dy*ratio);
            z = std::round(dz*ratio);
        }
        else
        {
            x = std::round(dx);
            y = std::round(dy);
            z = std::round(dz);
        }
        if(!in_range(x,y,z))
            return false;
        if(roi_filter[x].empty())
            return false;
        if(roi_filter[x][y].empty())
            return false;
        return roi_filter[x][y][z] != 0;
    }
    bool havePoint(const tipl::vector<3,float>& point) const
    {
        return havePoint(point[0],point[1],point[2]);
    }
    bool included(const float* track,unsigned int buffer_size) const
    {
        for(unsigned int index = 0; index < buffer_size; index += 3)
            if(havePoint(track[index],track[index+1],track[index+2]))
                return true;
        return false;
    }
};

extern std::vector<std::string> tractography_name_list;
class RoiMgr {
public:
    std::string report;
    std::vector<tipl::vector<3,short> > seeds;
    std::vector<float> seeds_r;
    std::vector<std::shared_ptr<Roi> > inclusive;
    std::vector<std::shared_ptr<Roi> > end;
    std::vector<std::shared_ptr<Roi> > exclusive;
    std::vector<std::shared_ptr<Roi> > terminate;
public:
    std::shared_ptr<TractModel> atlas;
    int track_id = 0;
public:
    bool is_excluded_point(const tipl::vector<3,float>& point) const
    {
        for(unsigned int index = 0; index < exclusive.size(); ++index)
            if(exclusive[index]->havePoint(point[0],point[1],point[2]))
                return true;
        return false;
    }
    bool is_terminate_point(const tipl::vector<3,float>& point) const
    {
        for(unsigned int index = 0; index < terminate.size(); ++index)
            if(terminate[index]->havePoint(point[0],point[1],point[2]))
                return true;
        return false;
    }


    bool fulfill_end_point(const tipl::vector<3,float>& point1,
                           const tipl::vector<3,float>& point2) const
    {
        if(end.empty())
            return true;
        if(end.size() == 1)
            return end[0]->havePoint(point1) ||
                   end[0]->havePoint(point2);
        if(end.size() == 2)
            return (end[0]->havePoint(point1) && end[1]->havePoint(point2)) ||
                   (end[1]->havePoint(point1) && end[0]->havePoint(point2));

        bool end_point1 = false;
        bool end_point2 = false;
        for(unsigned int index = 0; index < end.size(); ++index)
        {
            if(end[index]->havePoint(point1[0],point1[1],point1[2]))
                end_point1 = true;
            else if(end[index]->havePoint(point2[0],point2[1],point2[2]))
                end_point2 = true;
            if(end_point1 && end_point2)
                return true;
        }
        return false;
    }
    bool have_include(const float* track,unsigned int buffer_size) const
    {
        for(unsigned int index = 0; index < inclusive.size(); ++index)
            if(!inclusive[index]->included(track,buffer_size))
                return false;
        if(atlas.get())
            return atlas->find_nearest(track,buffer_size) == track_id;
        return true;
    }
    void setAtlas(std::shared_ptr<TractModel> atlas_,int track_id_)
    {
        atlas = atlas_;
        track_id = track_id_;
        report += " The anatomy prior of a tractography atlas (Yeh et al., Neuroimage 178, 57-68, 2018) was used to track ";
        report += tractography_name_list[size_t(track_id)];
        report += ".";
    }

    void setWholeBrainSeed(std::shared_ptr<fib_data> handle,float threashold)
    {
        if(!seeds.empty())
            return;
        std::vector<tipl::vector<3,short> > seed;
        std::string name = "whole brain";
        // if auto track
        if(atlas.get() && handle->has_atlas())
        {
            float r = 1.0f;
            handle->get_atlas_roi(handle->atlas_list[0],track_id,seed,r);
            name = tractography_name_list[size_t(track_id)];
        }
        else {
            const float *fa0 = handle->dir.fa[0];
            for(tipl::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
                if(fa0[index.index()] > threashold)
                    seed.push_back(tipl::vector<3,short>(short(index.x()),short(index.y()),short(index.z())));
        }
        setRegions(handle->dim,seed,1.0,3/*seed i*/,name.c_str(),tipl::vector<3>());
    }
    void setRegions(tipl::geometry<3> geo,
                    const std::vector<tipl::vector<3,short> >& points,
                    float r,
                    unsigned char type,
                    const char* roi_name,
                    tipl::vector<3> voxel_size)
    {
        tipl::vector<3,float> center;
        for(int i = 0;i < points.size();++i)
            center += points[i];
        center /= points.size();
        center /= r;

        switch(type)
        {
        case 0: //ROI
            inclusive.push_back(std::make_shared<Roi>(geo,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                inclusive.back()->addPoint(points[index]);
            report += " An ROI was placed at ";
            break;
        case 1: //ROA
            exclusive.push_back(std::make_shared<Roi>(geo,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                exclusive.back()->addPoint(points[index]);
            report += " An ROA was placed at ";
            break;
        case 2: //End
            end.push_back(std::make_shared<Roi>(geo,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                end.back()->addPoint(points[index]);
            report += " An ending region was placed at ";
            break;
        case 4: //Terminate
            terminate.push_back(std::make_shared<Roi>(geo,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                terminate.back()->addPoint(points[index]);
            report += " A terminative region was placed at ";
            break;
        case 3: //seed
            for (unsigned int index = 0;index < points.size();++index)
            {
                seeds.push_back(points[index]);
                seeds_r.push_back(r);
            };
            report += " A seeding region was placed at ";
            break;
        }
        report += roi_name;
        if(voxel_size[0]*voxel_size[1]*voxel_size[2] != 0.0f)
        {
            std::ostringstream out;
            out << std::setprecision(2) << " (" << center[0] << "," << center[1] << "," << center[2]
                << ") with a volume size of " << (float)points.size()*voxel_size[0]*voxel_size[1]*voxel_size[2]/r/r/r << " mm cubic";
            if(r != 1.0f)
            {
                out << " and a super resolution factor of " << r;
            }
            report += out.str();
        }
        report += ".";

    }
};


#define ROI_HPP
#endif//ROI_HPP
