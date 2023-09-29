#ifndef BASIC_PROCESS_HPP
#define BASIC_PROCESS_HPP
#include "zlib.h"
#include "TIPL/tipl.hpp"

extern char fib_dx[80];
extern char fib_dy[80];
extern char fib_dz[80];

struct VoxelTracking{

public:
    template<class method>
    bool operator()(method& info)
    {
        tipl::vector<3,short> cur_pos(info.position);
        std::vector<tipl::vector<3,float> > next_voxels_dir;
        std::vector<tipl::vector<3,short> > next_voxels_pos;
        std::vector<unsigned int> next_voxels_index;
        std::vector<float> voxel_angle;
        // assume isotropic

        for(unsigned int index = 0;index < 80;++index)
        {
            tipl::vector<3,float> dis(fib_dx[index],fib_dy[index],fib_dz[index]);
            tipl::vector<3,short> pos(cur_pos);
            pos += dis;
            if(!info.trk->dim.is_valid(pos))
                continue;
            dis.normalize();
            float angle_cos = dis*info.dir;
            if(angle_cos < info.current_tracking_angle)
                continue;
            next_voxels_pos.push_back(pos);
            next_voxels_index.push_back(tipl::pixel_index<3>(pos[0],pos[1],pos[2],info.trk->dim).index());
            next_voxels_dir.push_back(dis);
            voxel_angle.push_back(angle_cos);
        }

        unsigned char max_i(0);
        unsigned char max_j(0);
        float max_angle_cos = 0;
        for(unsigned int i = 0;i < next_voxels_index.size();++i)
        {
            for (unsigned int j = 0;j < info.trk->fib_num;++j)
            {
                float fa_value = info.trk->fa[j][next_voxels_index[i]];
                if (fa_value <= info.current_fa_threshold)
                    break;
                float value = std::abs(info.trk->cos_angle(next_voxels_dir[i],next_voxels_index[i],j));
                if(value < info.current_tracking_angle)
                    continue;
                if(voxel_angle[i]*value*fa_value > max_angle_cos)
                {
                    max_i = i;
                    max_j = j;
                    max_angle_cos = voxel_angle[i]*value*fa_value;
                }
            }
        }
        if(max_angle_cos == 0.0f)
            return false;


        info.dir = info.trk->get_fib(next_voxels_index[max_i],max_j);
        if(info.dir*next_voxels_dir[max_i] < 0)
            info.dir = -info.dir;
        info.position = next_voxels_pos[max_i];
        return true;
    }
};


struct RungeKutta4
{
public:
    template<class method>
    bool operator()(method& info)
    {
        tipl::vector<3,float> y;
        tipl::vector<3,float> k1,k2,k3,k4;
        if (!info.get_dir(info.position,info.dir,k1))
            return false;

        y = k1;
        y *= 0.5;
        info.scaling_in_voxel(y);
        y += info.position;
        if (!info.get_dir(y,k1,k2))
            return false;
        y = k2;
        y *= 0.5;
        info.scaling_in_voxel(y);
        y += info.position;
        if (!info.get_dir(y,k2,k3))
            return false;

        y = k3;
        info.scaling_in_voxel(y);
        y += info.position;
        if (!info.get_dir(y,k3,k4))
            return false;

        y = k2;
        y += k3;
        y *= 2.0;
        y += k1;
        y += k4;
        y /= 6.0;
        info.next_dir = y;

        // propagate
        tipl::vector<3,float> step(info.next_dir);
        info.scaling_in_voxel(step);
        info.position += step;
        info.dir = info.next_dir;
        return true;
    }
};


struct EulerTracking
{
public:

    template<class method>
    bool operator()(method& info)
    {
        if (!info.get_dir(info.position,info.dir,info.next_dir))
            return false;

        if(info.current_tracking_smoothing != 0.0f)
        {
            info.next_dir += (info.dir-info.next_dir)*info.current_tracking_smoothing;
            info.next_dir.normalize();
        }

        // propagate
        tipl::vector<3,float> step(info.next_dir);
        info.scaling_in_voxel(step);
        info.position += step;
        info.dir = info.next_dir;
        return true;

    }
};





#endif//BASIC_PROCESS_HPP
