#ifndef ROI_HPP
#include <functional>
#include <set>
#include "tipl/tipl.hpp"
#include "tract_model.hpp"
#include "tracking/region/Regions.h"
class Roi {
private:
    float ratio;
    tipl::shape<3> dim;
    std::vector<std::vector<std::vector<unsigned char> > > roi_filter;
public:
    Roi(const tipl::shape<3>& geo,float r):
        ratio(r),dim(r == 1.0f ? geo:geo*r),roi_filter(geo[0])    {}
    void clear(void)
    {
        roi_filter.clear();
        roi_filter.resize(uint16_t(dim[0]));
    }
    void addPoint(const tipl::vector<3,short>& new_point)
    {
        if(dim.is_valid(new_point.x(),new_point.y(),new_point.z()))
        {
            if(roi_filter[uint16_t(new_point.x())].empty())
                roi_filter[uint16_t(new_point.x())].resize(uint16_t(dim[1]));
            if(roi_filter[uint16_t(new_point.x())][uint16_t(new_point.y())].empty())
                roi_filter[uint16_t(new_point.x())][uint16_t(new_point.y())].resize(uint16_t(dim[2]));
            roi_filter[uint16_t(new_point.x())][uint16_t(new_point.y())][uint16_t(new_point.z())] = 1;
        }
    }
    bool havePoint(float dx,float dy,float dz) const
    {
        short x,y,z;
        if(ratio != 1.0f)
        {
            x = short(std::round(dx*ratio));
            y = short(std::round(dy*ratio));
            z = short(std::round(dz*ratio));
        }
        else
        {
            x = short(std::round(dx));
            y = short(std::round(dy));
            z = short(std::round(dz));
        }
        if(!dim.is_valid(x,y,z))
            return false;
        if(roi_filter[uint16_t(x)].empty())
            return false;
        if(roi_filter[uint16_t(x)][uint16_t(y)].empty())
            return false;
        return roi_filter[uint16_t(x)][uint16_t(y)][uint16_t(z)] != 0;
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

template<typename T,typename U>
__DEVICE_HOST__ unsigned int find_nearest(const float* trk,unsigned int length,
                          const T& tract_data,// = track_atlas->get_tracts();
                          const U& tract_cluster,// = track_atlas->get_cluster_info();
                          bool contain,float false_distance)
{
    struct norm1_imp{
        inline float operator()(const float* v1,const float* v2)
        {
            return std::fabs(v1[0]-v2[0])+std::fabs(v1[1]-v2[1])+std::fabs(v1[2]-v2[2]);
        }
    } norm1;

    struct min_min_imp{
        inline float operator()(float min_dis,const float* v1,const float* v2)
        {
            float d1 = std::fabs(v1[0]-v2[0]);
            if(d1 > min_dis)                    return min_dis;
            d1 += std::fabs(v1[1]-v2[1]);
            if(d1 > min_dis)                    return min_dis;
            d1 += std::fabs(v1[2]-v2[2]);
            if(d1 > min_dis)                    return min_dis;
            return d1;
        }
    }min_min;
    if(length <= 6)
        return 9999;
    float best_distance = contain ? 50.0f : false_distance;
    size_t best_index = tract_data.size();
    if(contain)
    {
        for(size_t i = 0;i < tract_data.size();++i)
        {
            bool skip = false;
            float max_dis = 0.0f;
            for(size_t n = 0;n < length;n += 6)
            {
                float min_dis = norm1(&tract_data[i][0],trk+n);
                for(size_t m = 0;m < tract_data[i].size() && min_dis > max_dis;m += 3)
                    min_dis = min_min(min_dis,&tract_data[i][m],trk+n);
                if(min_dis > max_dis)
                    max_dis = max_dis;
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            {
                best_distance = max_dis;
                best_index = i;
            }
        }
    }
    else
    {
        for(size_t i = 0;i < tract_data.size();++i)
        {
            if(min_min(best_distance,&tract_data[i][0],trk) >= best_distance ||
                min_min(best_distance,&tract_data[i][tract_data[i].size()-3],trk+length-3) >= best_distance ||
                min_min(best_distance,&tract_data[i][tract_data[i].size()/3/2*3],trk+(length/3/2*3)) >= best_distance)
                continue;

            bool skip = false;
            float max_dis = 0.0f;
            for(size_t m = 0;m < tract_data[i].size();m += 3)
            {
                const float* tim = &tract_data[i][m];
                const float* trk_length = trk+length;
                float min_dis = norm1(tim,trk);
                for(const float* trk_n = trk;trk_n < trk_length && min_dis > max_dis;trk_n += 3)
                    min_dis = min_min(min_dis,tim,trk_n);
                if(min_dis > max_dis)
                    max_dis = max_dis;
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            for(size_t n = 0;n < length;n += 3)
            {
                const float* ti0 = &tract_data[i][0];
                const float* ti_end = ti0+tract_data[i].size();
                const float* trk_n = trk+n;
                float min_dis = norm1(ti0,trk_n);
                for(const float* tim = ti0;tim < ti_end && min_dis > max_dis;tim += 3)
                    min_dis = min_min(min_dis,tim,trk_n);
                if(min_dis > max_dis)
                    max_dis = max_dis;
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            {
                best_distance = max_dis;
                best_index = i;
            }
        }
    }
    if(best_index == tract_data.size())
        return 9999;
    return tract_cluster[best_index];
}

class RoiMgr {
public:
    std::shared_ptr<fib_data> handle;
    std::string report;
    std::vector<tipl::vector<3,short> > seeds;
    std::vector<float> seeds_r;
    std::vector<std::shared_ptr<Roi> > inclusive;
    std::vector<std::shared_ptr<Roi> > end;
    std::vector<std::shared_ptr<Roi> > exclusive;
    std::vector<std::shared_ptr<Roi> > terminate;
    std::vector<std::shared_ptr<Roi> > no_end;
public:
    float false_distance = 0.0f;
    unsigned int track_id = 0;
public:
    RoiMgr(std::shared_ptr<fib_data> handle_):handle(handle_){}
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
        for(unsigned int index = 0; index < no_end.size(); ++index)
            if(no_end[index]->havePoint(point1) ||
               no_end[index]->havePoint(point2))
                return false;
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
        if(false_distance != 0.0f)
            return handle->find_nearest(track,buffer_size,false,false_distance) == track_id;
        return true;
    }
    bool setAtlas(unsigned int track_id_,float false_distance_)
    {
        if(!handle->load_track_atlas())
            return false;
        if(track_id >= handle->tractography_name_list.size())
        {
            handle->error_msg = "invalid track_id";
            return false;
        }
        false_distance = false_distance_;
        track_id = track_id_;
        report += " The anatomy prior of a tractography atlas (Yeh et al., Neuroimage 178, 57-68, 2018) was used to map ";
        report += handle->tractography_name_list[size_t(track_id)];
        report += "  with a distance tolerance of ";
        report += std::to_string(int(false_distance_*handle->vs[0]));
        report += " (mm).";
        // place seed at the atlas track region
        if(seeds.empty())
        {
            std::vector<tipl::vector<3,short> > seed;
            handle->track_atlas->to_voxel(seed,1.0f,int(track_id));
            ROIRegion region(handle);
            region.add_points(seed,false);
            region.perform("dilation");
            region.perform("dilation");
            region.perform("dilation");
            region.perform("smoothing");
            region.perform("smoothing");
            setRegions(region.get_region_voxels_raw(),1.0,3/*seed i*/,
            handle->tractography_name_list[size_t(track_id)].c_str());
        }
        // add tolerance roa to speed up tracking
        {
            std::vector<tipl::vector<3,short> > seed;
            handle->track_atlas->to_voxel(seed,1.0f,int(track_id));
            std::vector<tipl::vector<3,short> > track_roa;
            tipl::image<3,char> roa_mask(handle->dim);
            const float *fa0 = handle->dir.fa[0];
            for(size_t index = 0;index < roa_mask.size();++index)
                if(fa0[index] > 0.0f)
                    roa_mask[index] = 1;

            // build a shift vector
            tipl::neighbor_index_shift<3> shift(handle->dim,int(false_distance_)+1);
            for(size_t i = 0;i < seed.size();++i)
            {
                int index = int(tipl::pixel_index<3>(seed[i][0],
                                                     seed[i][1],
                                                     seed[i][2],handle->dim).index());
                for(size_t j = 0;j < shift.index_shift.size();++j)
                {
                    int pos = index+shift.index_shift[j];
                    if(pos >=0 && pos < int(roa_mask.size()))
                        roa_mask[pos] = 0;
                }
            }

            std::vector<tipl::vector<3,short> > roa_points;
            for(tipl::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
                if(roa_mask[index.index()])
                    roa_points.push_back(tipl::vector<3,short>(short(index.x()),short(index.y()),short(index.z())));
            setRegions(roa_points,1.0f,1,"track tolerance region");
        }
        return true;
    }

    void setWholeBrainSeed(float threashold)
    {
        std::vector<tipl::vector<3,short> > seed;
        const float *fa0 = handle->dir.fa[0];
        for(tipl::pixel_index<3> index(handle->dim);index < handle->dim.size();++index)
            if(fa0[index.index()] > threashold)
                seed.push_back(tipl::vector<3,short>(short(index.x()),short(index.y()),short(index.z())));
        setRegions(seed,1.0,3/*seed i*/,"whole brain");
    }
    void setRegions(const std::vector<tipl::vector<3,short> >& points,
                    float r,
                    unsigned char type,
                    const char* roi_name)
    {
        switch(type)
        {
        case roi_id:
            inclusive.push_back(std::make_shared<Roi>(handle->dim,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                inclusive.back()->addPoint(points[index]);
            report += " An ROI was placed at ";
            break;
        case roa_id:
            exclusive.push_back(std::make_shared<Roi>(handle->dim,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                exclusive.back()->addPoint(points[index]);
            report += " An ROA was placed at ";
            break;
        case end_id:
            end.push_back(std::make_shared<Roi>(handle->dim,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                end.back()->addPoint(points[index]);
            report += " An ending region was placed at ";
            break;
        case terminate_id:
            terminate.push_back(std::make_shared<Roi>(handle->dim,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                terminate.back()->addPoint(points[index]);
            report += " A terminative region was placed at ";
            break;
        case not_end_id:
            no_end.push_back(std::make_shared<Roi>(handle->dim,r));
            for(unsigned int index = 0; index < points.size(); ++index)
                no_end.back()->addPoint(points[index]);
            report += " A no ending region was placed at ";
            break;
        case seed_id:
            for (unsigned int index = 0;index < points.size();++index)
            {
                seeds.push_back(points[index]);
                seeds_r.push_back(r);
            };
            report += " A seeding region was placed at ";
            break;
        default:
            return;
        }
        report += roi_name;
        if(type != 3 && handle->vs[0]*handle->vs[1]*handle->vs[2] != 0.0f)
        {
            tipl::vector<3,float> center;
            for(size_t i = 0;i < points.size();++i)
                center += points[i];
            center /= points.size();
            center /= r;
            std::ostringstream out;
            out << std::setprecision(2) << " (" << center[0] << "," << center[1] << "," << center[2]
                << ") with a volume size of " << float(points.size())*handle->vs[0]*handle->vs[1]*handle->vs[2]/r/r/r << " mm cubic";
            if(r != 1.0f)
                out << " and a super resolution factor of " << r;
            report += out.str();
        }
        report += ".";

    }
};


#define ROI_HPP
#endif//ROI_HPP
