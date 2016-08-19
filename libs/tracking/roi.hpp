#ifndef ROI_HPP
#include <functional>
#include <set>
#include "image/image.hpp"

class Roi {
private:
    image::geometry<3> dim;
    std::vector<std::vector<std::vector<unsigned char> > > roi_filter;
    bool in_range(int x,int y,int z) const
    {
        return dim.is_valid(x,y,z);
    }
public:
    Roi(const image::geometry<3>& geo):dim(geo),roi_filter(dim[0])
    {
        for(unsigned int x = 0; x < dim[0]; ++x)
        {
            std::vector<std::vector<unsigned char> >&
            roi_filter_x = roi_filter[x];
            roi_filter_x.resize(dim[1]);
            for(unsigned int y = 0; y < dim[1]; ++y)
                roi_filter_x[y].resize(dim[2]);
        }
    }
    void clear(void)
    {
        for(unsigned int x = 0; x < roi_filter.size(); ++x)
            for(unsigned int y = 0; y < roi_filter[x].size(); ++y)
                for(unsigned int z = 0; z < roi_filter[x][y].size(); ++z)
                    roi_filter[x][y][z] = 0;
    }
    void addPoint(const image::vector<3,short>& new_point)
    {
        if(in_range(new_point.x(),new_point.y(),new_point.z()))
            roi_filter[new_point.x()][new_point.y()][new_point.z()] = 1;
    }
    bool havePoint(float dx,float dy,float dz) const
    {
        short x = std::round(dx);
        short y = std::round(dy);
        short z = std::round(dz);
        return in_range(x,y,z) && roi_filter[x][y][z];
    }
    bool havePoint(const image::vector<3,float>& point) const
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


class RoiMgr {
public:
    std::vector<std::shared_ptr<Roi> > inclusive;
    std::vector<std::shared_ptr<Roi> > end;
    std::auto_ptr<Roi> exclusive;
    std::auto_ptr<Roi> terminate;

public:
    void clear(void)
    {
        inclusive.clear();
        end.clear();
        exclusive.reset(0);
        terminate.reset(0);
    }

    bool is_excluded_point(const image::vector<3,float>& point) const
    {
        if(!exclusive.get())
            return false;
        return exclusive->havePoint(point[0],point[1],point[2]);
    }
    bool is_terminate_point(const image::vector<3,float>& point) const
    {
        if(!terminate.get())
            return false;
        return terminate->havePoint(point[0],point[1],point[2]);
    }


    bool fulfill_end_point(const image::vector<3,float>& point1,
                           const image::vector<3,float>& point2) const
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
        return true;
    }

    void add_inclusive_roi(const image::geometry<3>& geo,
                           const std::vector<image::vector<3,short> >& points)
    {
        inclusive.push_back(std::make_shared<Roi>(geo));
        for(unsigned int index = 0; index < points.size(); ++index)
            inclusive.back()->addPoint(points[index]);
    }
    void add_end_roi(const image::geometry<3>& geo,
                     const std::vector<image::vector<3,short> >& points)
    {
        end.push_back(std::make_shared<Roi>(geo));
        for(unsigned int index = 0; index < points.size(); ++index)
            end.back()->addPoint(points[index]);
    }

    void add_exclusive_roi(const image::geometry<3>& geo,
                           const std::vector<image::vector<3,short> >& points)
    {
        if(!exclusive.get())
            exclusive.reset(new Roi(geo));
        for(unsigned int index = 0; index < points.size(); ++index)
            exclusive->addPoint(points[index]);
    }
    void add_terminate_roi(const image::geometry<3>& geo,
                           const std::vector<image::vector<3,short> >& points)
    {
        if(!terminate.get())
            terminate.reset(new Roi(geo));
        for(unsigned int index = 0; index < points.size(); ++index)
            terminate->addPoint(points[index]);
    }



};


#define ROI_HPP
#endif//ROI_HPP
