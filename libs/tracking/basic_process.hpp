#ifndef BASIC_PROCESS_HPP
#define BASIC_PROCESS_HPP
#include <ctime>
#include <boost/random.hpp>
#include <boost/lambda/lambda.hpp>
#include "math/matrix_op.hpp"
#include "tracking_info.hpp"

struct LocateVoxel{

public:

    void operator()(TrackingInfo& info)
    {
        image::vector<3,short> cur_pos(info.position);
        unsigned int cur_pos_index;
        cur_pos_index = image::pixel_index<3>(cur_pos[0],cur_pos[1],cur_pos[2],info.fib_data.dim).index();

        std::vector<image::vector<3,float> > next_voxels_dir;
        std::vector<image::vector<3,short> > next_voxels_pos;
        std::vector<unsigned int> next_voxels_index;
        std::vector<float> voxel_angle;
        // assume isotropic
        int radius = 2;
        int radius2 = 6;
        for(char z = -radius;z <= radius;++z)
            for(char y = -radius;y <= radius;++y)
                for(char x = -radius;x <= radius;++x)
                {
                    if((x == 0 && y == 0 && z == 0) ||
                            x*x+y*y+z*z > radius2)
                        continue;
                    image::vector<3,float> dis(x,y,z);
                    image::vector<3,short> pos(cur_pos);
                    pos += dis;
                    if(!info.fib_data.dim.is_valid(pos))
                        continue;
                    dis.normalize();
                    float angle_cos = dis*info.dir;
                    if(angle_cos < info.param.cull_cos_angle)
                        continue;
                    next_voxels_pos.push_back(pos);
                    next_voxels_index.push_back(image::pixel_index<3>(pos[0],pos[1],pos[2],info.fib_data.dim).index());
                    next_voxels_dir.push_back(dis);
                    voxel_angle.push_back(angle_cos);
                }

        char max_i;
        char max_j;
        float max_angle_cos = 0;
        for(char i = 0;i < next_voxels_index.size();++i)
        {
            for (char j = 0;j < info.fib_data.fib.num_fiber;++j)
            {
                if (info.fib_data.fib.getFA(next_voxels_index[i],j) <= info.param.threshold)
                    break;
                float value = std::abs(next_voxels_dir[i]*info.fib_data.fib.getDir(next_voxels_index[i],j));
                if(value < info.param.cull_cos_angle)
                    continue;
                if(voxel_angle[i]*value*info.fib_data.fib.getFA(next_voxels_index[i],j) > max_angle_cos)
                {
                    max_i = i;
                    max_j = j;
                    max_angle_cos = voxel_angle[i]*value*info.fib_data.fib.getFA(next_voxels_index[i],j);
                }
            }
        }
        if(max_angle_cos == 0)
        {
            info.terminated = true;
            return;
        }


        info.dir = info.fib_data.fib.getDir(next_voxels_index[max_i],max_j);
        if(info.dir*next_voxels_dir[max_i] < 0)
            info.dir = -info.dir;
        info.position = next_voxels_pos[max_i];
    }
};

struct EstimateNextDirection
{
public:

    void operator()(TrackingInfo& info)
    {
        if (!info.evaluate_dir(info.position,info.dir,info.next_dir))
            info.terminated = true;
    }
};

struct EstimateNextDirectionRungeKutta4
{
public:

    void operator()(TrackingInfo& info)
    {
        image::vector<3,float> y;
        image::vector<3,float> k1,k2,k3,k4;
        if (!info.evaluate_dir(info.position,info.dir,k1))
        {
            info.terminated = true;
            return;
        }

        y = k1;
        y *= 0.5;
        info.param.scaling_in_voxel(y);
        y += info.position;
        if (!info.evaluate_dir(y,k1,k2))
        {
            info.terminated = true;
            return;
        }
        y = k2;
        y *= 0.5;
        info.param.scaling_in_voxel(y);
        y += info.position;
        if (!info.evaluate_dir(y,k2,k3))
        {
            info.terminated = true;
            return;
        }

        y = k3;
        info.param.scaling_in_voxel(y);
        y += info.position;
        if (!info.evaluate_dir(y,k3,k4))
        {
            info.terminated = true;
            return;
        }

        y = k2;
        y += k3;
        y *= 2.0;
        y += k1;
        y += k4;
        y /= 6.0;
        info.next_dir = y;
    }
};


struct SmoothDir
{
public:

    void operator()(TrackingInfo& info)
    {
        info.next_dir += (info.dir-info.next_dir)*info.param.smooth_fraction;
        info.next_dir.normalize();
    }
};

struct MoveTrack
{
public:

    void operator()(TrackingInfo& info)
    {
        if (info.terminated)
            return;
        image::vector<3,float> step(info.next_dir);
        info.param.scaling_in_voxel(step);
        info.position += step;
        info.dir = info.next_dir;
    }
};


struct MoveTrack2 : public MoveTrack
{
public:

    void operator()(TrackingInfo& info)
    {
        if (info.terminated)
            return;
                image::vector<3,float> pre_position(info.position);
		
		MoveTrack::operator()(info);

                for(unsigned int index = 0;index < 5;++index)
		{
        image::vector<3,float> position(info.position);
		const FibData& fib_data = info.fib_data;
        image::interpolation<image::linear_weighting,3> tri_interpo;
        if (!tri_interpo.get_location(fib_data.dim,position))
            return;

        image::vector<3,float> main_dir;
		std::vector<float> w(8);
                for (unsigned int index = 0;index < 8;++index)
        {
            unsigned int odf_space_index = tri_interpo.dindex[index];
            if (!info.fib_data.get_nearest_dir(odf_space_index,info.dir,main_dir,info.param.threshold,info.param.cull_cos_angle))
                continue;
			w[index] = std::abs(main_dir*info.dir);
        }
		std::for_each(w.begin(),w.end(),boost::lambda::_1 /= std::accumulate(w.begin(),w.end(),0.0));
		main_dir[0] = w[1] + w[3] + w[5] + w[7] - 0.5;
		main_dir[1] = w[2] + w[3] + w[6] + w[7] - 0.5;
		main_dir[2] = w[4] + w[5] + w[6] + w[7] - 0.5;
		main_dir *= 0.1;
		info.param.scaling_in_voxel(main_dir);
		info.position += main_dir;
        info.dir = info.position;
		info.dir -= pre_position;
		info.dir.normalize();
		}

    }
};





#endif//BASIC_PROCESS_HPP
