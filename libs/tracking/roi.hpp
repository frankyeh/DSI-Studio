#ifndef ROI_HPP
#include <functional>
#include <set>
#include "tract_model.hpp"
#include "tracking/region/Regions.h"
class Roi {
    tipl::shape<3> dim;
    std::vector<uint32_t> xyz_hash;
public:
    bool need_trans = false;
    tipl::matrix<4,4> from_diffusion_space = tipl::identity_matrix();
public:
    __INLINE__ Roi(const tipl::shape<3>& dim_,const tipl::matrix<4,4>& from_diffusion_space_):
        dim(dim_),xyz_hash(dim_[0]),need_trans(true),from_diffusion_space(from_diffusion_space_){}
    __INLINE__ Roi(const tipl::shape<3>& dim_):dim(dim_),xyz_hash(dim_[0]){}
    __INLINE__ ~Roi(){}
public:
    __HOST__ void addPoint(const tipl::vector<3,short>& new_point)
    {
        if(!dim.is_valid(new_point))
            return;
        auto x = uint16_t(new_point.x());
        auto y = uint16_t(new_point.y());
        auto z = uint16_t(new_point.z());
        uint32_t y_base = xyz_hash[x];
        if(!y_base)
        {
            xyz_hash[x] = y_base = uint32_t(xyz_hash.size());
            xyz_hash.resize(xyz_hash.size()+dim[1]);

        }
        uint32_t z_base = xyz_hash[y_base+y];
        if(!z_base)
        {
            xyz_hash[y_base+y] = z_base = uint32_t(xyz_hash.size());
            xyz_hash.resize(xyz_hash.size()+uint16_t((dim[2]+31) >> 5));
        }
        xyz_hash[z_base+(z >> 5)] |= (1 << (z & 31));
    }
    __INLINE__ Roi& operator=(Roi& rhs)
    {
        dim = rhs.dim;
        xyz_hash = rhs.xyz_hash;
        need_trans = rhs.need_trans;
        from_diffusion_space = rhs.from_diffusion_space;
        return *this;
    }
public:
    __INLINE__ bool havePoint(tipl::vector<3> p) const
    {
        if(need_trans)
            p.to(from_diffusion_space);
        short x = short(std::round(p[0]));
        short y = short(std::round(p[1]));
        short z = short(std::round(p[2]));
        if(!dim.is_valid(x,y,z))
            return false;
        auto y_base = xyz_hash[uint16_t(x)];
        if(!y_base)
            return false;
        auto z_base = xyz_hash[y_base+uint16_t(y)];
        if(!z_base)
            return false;
        return (xyz_hash[z_base+(uint16_t(z) >> 5)] & (1 << (z & 31)));
    }
    __INLINE__ bool included(const float* track,unsigned int buffer_size) const
    {
        auto end = track + buffer_size;
        for(;track < end ; track += 3)
            if(havePoint(tipl::vector<3>(track)))
                return true;
        return false;
    }
};


__INLINE__ float get_distance_one_way(const float* trk1,unsigned int length1,
                              const float* trk2,unsigned int length2,
                              float max_dis,
                              float max_dis_limit)
{
    struct update_min_dis_imp{
        __INLINE__ void operator()(float& min_dis,const float* v1,const float* v2)
        {
            float d1 = std::fabs(v1[0]-v2[0]);if(d1 > min_dis)return;
            d1 += std::fabs(v1[1]-v2[1]);if(d1 > min_dis)return;
            d1 += std::fabs(v1[2]-v2[2]);if(d1 < min_dis)min_dis = d1;
        }
    }update_min_dis;
    auto trk1_end = trk1+length1;
    auto trk2_end = trk2+length2;

    for(auto trk2_n = trk2;trk2_n < trk2_end;trk2_n += 3)
    {
        float min_dis = max_dis_limit;
        for(auto trk1_n = trk1;trk1_n < trk1_end && min_dis > max_dis;trk1_n += 3)
            update_min_dis(min_dis,trk2_n,trk1_n);
        if(min_dis >= max_dis_limit)
            return max_dis_limit;
        if(min_dis > max_dis)
            max_dis = min_dis;
    }
    return max_dis;
}

__INLINE__ float get_distance(const float* trk1,unsigned int length1,
                              const float* trk2,unsigned int length2,
                              float max_dis_limit)
{
    float max_dis = 0;
    max_dis = get_distance_one_way(trk1,length1,trk2,length2,max_dis,max_dis_limit);
    if(max_dis >= max_dis_limit)
        return max_dis_limit;
    return get_distance_one_way(trk2,length2,trk1,length1,max_dis,max_dis_limit);
}

__INLINE__ bool distance_over_limit(const float* trk1,unsigned int length1,
                              const float* trk2,unsigned int length2,
                              float max_dis_limit)
{
    struct min_min_imp{
        __INLINE__ float operator()(float min_dis,const float* v1,const float* v2)
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

    return  min_min(max_dis_limit,trk1,trk2) >= max_dis_limit ||
            min_min(max_dis_limit,trk1+length1-3,trk2+length2-3) >= max_dis_limit;
}

template<typename T,typename U>
__DEVICE_HOST__ unsigned int find_nearest(const float* trk,unsigned int length,
                          const T& tract_data,// = track_atlas->get_tracts();
                          const U& tract_cluster,// = track_atlas->get_cluster_info();
                          float tolerance_dis_in_subject_voxels)
{
    if(length <= 6)
        return 9999;
    float best_distance = tolerance_dis_in_subject_voxels;
    unsigned int best_cluster = 9999;
    for(size_t i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() <= 6)
            continue;
        if(distance_over_limit(&tract_data[i][0],tract_data[i].size(),trk,length,best_distance))
            continue;
        float min_dis = get_distance(&tract_data[i][0],tract_data[i].size(),trk,length,tolerance_dis_in_subject_voxels);
        if(min_dis < best_distance)
        {
            best_distance = min_dis;
            best_cluster = tract_cluster[i];
        }
    }
    return best_cluster;
}

class RoiMgr {
public:
    std::shared_ptr<fib_data> handle;
    std::string report;
    std::vector<std::shared_ptr<Roi> > roi,end,roa,term,no_end,limiting;
public:
    std::vector<tipl::vector<3,short> > seeds;
    std::vector<uint16_t> seed_space;
    std::vector<bool> need_trans;
    std::vector<tipl::matrix<4,4> > to_diffusion_space;
public:
    bool use_auto_track = false;
    float track_voxel_ratio = 1.0f;
    float tolerance_dis_in_icbm152_mm = 0.0f;
    float tolerance_dis_in_subject_voxels = 0.0f;
    std::vector<size_t> track_ids;
    std::string tract_name;
public:
    RoiMgr(std::shared_ptr<fib_data> handle_):handle(handle_){}
public:
    bool within_roa(const tipl::vector<3,float>& point) const
    {
        for(unsigned int index = 0; index < roa.size(); ++index)
            if(roa[index]->havePoint(point))
                return true;
        return false;
    }
    bool within_limiting(const tipl::vector<3,float>& point) const
    {
        if(limiting.empty())
            return true;
        for(unsigned int index = 0; index < limiting.size(); ++index)
            if(limiting[index]->havePoint(point))
                return true;
        return false;
    }
    bool within_terminative(const tipl::vector<3,float>& point) const
    {
        for(unsigned int index = 0; index < term.size(); ++index)
            if(term[index]->havePoint(point))
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
            if(end[index]->havePoint(point1))
                end_point1 = true;
            else if(end[index]->havePoint(point2))
                end_point2 = true;
            if(end_point1 && end_point2)
                return true;
        }
        return false;
    }
    bool within_roi(const float* track,unsigned int buffer_size) const
    {
        for(unsigned int index = 0; index < roi.size(); ++index)
            if(!roi[index]->included(track,buffer_size))
                return false;
        if(!selected_atlas_tracts.empty())
        {
            auto nearest_id = find_nearest(track,buffer_size,
                                selected_atlas_tracts,
                                selected_atlas_cluster,
                                tolerance_dis_in_subject_voxels);
            return std::find(track_ids.begin(),track_ids.end(),nearest_id) != track_ids.end();
        }
        return true;
    }
public:
    std::vector<tipl::vector<3,short> > atlas_seed,atlas_limiting,atlas_not_end,atlas_roi;
    std::vector<std::vector<float> > selected_atlas_tracts;
    std::vector<unsigned int> selected_atlas_cluster;
public:
    bool setAtlas(bool& terminated,float seed_threshold,float not_end_threshold);

    void setWholeBrainSeed(float threshold)
    {
        const float *fa0 = handle->dir.fa[0];
        setRegions(volume2points(handle->dim,[&](auto& index){return fa0[index.index()] > threshold;}),3/*seed i*/,"whole brain");
    }

    auto createRegion(const std::vector<tipl::vector<3,short> >& points,
                      const tipl::shape<3>& dim,
                      const tipl::matrix<4,4>& trans)
    {
        auto region = (handle->dim != dim || trans != tipl::identity_matrix()) ?
                    std::make_shared<Roi>(dim,trans) : std::make_shared<Roi>(handle->dim);
        for(unsigned int index = 0; index < points.size(); ++index)
            region->addPoint(points[index]);
        return region;
    }
    void setRegions(const std::vector<tipl::vector<3,short> >& points,
                    unsigned char type,
                    const char* roi_name)
    {
        setRegions(points,handle->dim,tipl::identity_matrix(),type,roi_name);
    }
    void setRegions(const std::vector<tipl::vector<3,short> >& points,
                    const tipl::shape<3>& dim,
                    const tipl::matrix<4,4>& to_diffusion_space_,
                    unsigned char type,
                    const char* roi_name)
    {
        if(type == seed_id)
        {
            uint16_t seed_space_id = uint16_t(to_diffusion_space.size());
            need_trans.push_back(handle->dim != dim || to_diffusion_space_ != tipl::identity_matrix());
            to_diffusion_space.push_back(to_diffusion_space_);
            for (unsigned int index = 0;index < points.size();++index)
            {
                seeds.push_back(points[index]);
                seed_space.push_back(seed_space_id);
            }
            report += " A seeding region was placed at ";
        }
        else
        {
            auto region = createRegion(points,dim,tipl::inverse(to_diffusion_space_));
            switch(type)
            {
            case roi_id:
                roi.push_back(region);
                report += " An ROI was placed at ";
                break;
            case roa_id:
                roa.push_back(region);
                report += " An ROA was placed at ";
                break;
            case end_id:
                end.push_back(region);
                report += " An ending region was placed at ";
                break;
            case term_id:
                term.push_back(region);
                report += " A terminative region was placed at ";
                break;
            case not_end_id:
                no_end.push_back(region);
                report += " A no ending region was placed at ";
                break;
            case limiting_id:
                limiting.push_back(region);
                report += " A limiting region was placed at ";
                break;
            default:
                return;
            }
        }

        report += roi_name;
        tipl::vector<3> center;
        for(size_t i = 0;i < points.size();++i)
            center += points[i];
        center /= points.size();
        std::ostringstream out;
        out << std::setprecision(2) << " (" << center[0] << "," << center[1] << "," << center[2]
            << ") ";
        report += out.str();
        report += ".";
    }
};


#define ROI_HPP
#endif//ROI_HPP
