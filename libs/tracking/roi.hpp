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
    }
    void clear(void)
    {
        roi_filter.clear();
        roi_filter.resize(dim[0]);
    }
    void addPoint(const image::vector<3,short>& new_point)
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
        short x = std::round(dx);
        short y = std::round(dy);
        short z = std::round(dz);
        if(!in_range(x,y,z))
            return false;
        if(roi_filter[x].empty())
            return false;
        if(roi_filter[x][y].empty())
            return false;
        return roi_filter[x][y][z];
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
    std::string report;
    std::vector<image::vector<3,short> > seeds;
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
        report.clear();
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



    void setRegions(image::geometry<3> dim,
                    const std::vector<image::vector<3,short> >& points,
                    unsigned char type,
                    const char* roi_name)
    {
        switch(type)
        {
        case 0: //ROI
            add_inclusive_roi(dim,points);
            report += " An ROI was placed at ";
            break;
        case 1: //ROA
            add_exclusive_roi(dim,points);
            report += " An ROA was placed at ";
            break;
        case 2: //End
            add_end_roi(dim,points);
            report += " An ending region was placed at ";
            break;
        case 4: //Terminate
            add_terminate_roi(dim,points);
            report += " A terminative region was placed at ";
            break;
        case 3: //seed
            for (unsigned int index = 0;index < points.size();++index)
                seeds.push_back(points[index]);
            report += " A seeding region was placed at ";
            break;
        }
        report += roi_name;
        report += ".";
    }
};


#define ROI_HPP
#endif//ROI_HPP
