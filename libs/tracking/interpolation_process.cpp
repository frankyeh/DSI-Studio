#include "interpolation_process.hpp"
#include "fib_data.hpp"

// generated from x^2+y^2+z^2 < 6 . first 40 at x > 0
char fib_dx[80] = {0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,0,0,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-2,0,0,0,0,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2};
char fib_dy[80] = {1,0,0,1,1,1,0,0,-1,1,1,-1,-1,2,0,0,2,2,1,1,2,0,0,-2,1,0,0,-1,2,2,1,1,-1,-1,-2,-2,1,1,-1,-1,-1,0,0,-1,-1,-1,0,0,1,-1,-1,1,1,-2,0,0,-2,-2,-1,-1,-2,0,0,2,-1,0,0,1,-2,-2,-1,-1,1,1,2,2,-1,-1,1,1};
char fib_dz[80] = {0,1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,2,0,1,-1,2,-2,0,2,-2,0,0,1,-1,0,1,-1,2,-2,2,-2,1,-1,1,-1,1,-1,0,-1,0,-1,1,0,-1,1,0,-1,1,-1,1,0,-2,0,-1,1,-2,2,0,-2,2,0,0,-1,1,0,-1,1,-2,2,-2,2,-1,1,-1,1,-1,1};
bool trilinear_interpolation_with_gaussian_basis::evaluate(std::shared_ptr<tracking_data> fib,
                                                           const tipl::vector<3,float>& position,
                                                           const tipl::vector<3,float>& ref_dir,
                                                           tipl::vector<3,float>& result,
                                                           float threshold,
                                                           float angle,
                                                           float dt_threshold)
{
    tipl::interpolation<tipl::gaussian_radial_basis_weighting,3> tri_interpo;
	tri_interpo.weighting.sd = 0.5;
    if (!tri_interpo.get_location(fib->dim,position))
        return false;
    tipl::vector<3,float> new_dir,main_dir;
    float total_weighting = 0.0;
    float ww = std::accumulate(tri_interpo.ratio,tri_interpo.ratio+8,0.0f)*0.5f;
    for (unsigned int index = 0;index < 8;++index)
    {
        unsigned int odf_space_index = tri_interpo.dindex[index];
        if (!fib->get_dir(odf_space_index,ref_dir,main_dir,threshold,angle,dt_threshold))
            continue;
		float w = tri_interpo.ratio[index];
		main_dir *= w;
        new_dir += main_dir;
        total_weighting += w;
    }
    if (total_weighting < ww)
        return false;
    new_dir.normalize();
    result = new_dir;
    return true;
}



bool trilinear_interpolation::evaluate(std::shared_ptr<tracking_data> fib,
                                       const tipl::vector<3,float>& position,
                                       const tipl::vector<3,float>& ref_dir,
                                       tipl::vector<3,float>& result,
                                       float threshold,
                                       float angle,
                                       float dt_threshold)
{
    tipl::interpolation<tipl::linear_weighting,3> tri_interpo;
    if (!tri_interpo.get_location(fib->dim,position))
        return false;
    tipl::vector<3,float> new_dir,main_dir;
    float total_weighting = 0.0f;
    for (unsigned int index = 0;index < 8;++index)
    {
        int64_t odf_space_index = tri_interpo.dindex[index];
        if (!fib->get_dir(odf_space_index,ref_dir,main_dir,threshold,angle,dt_threshold))
            continue;
		float w = tri_interpo.ratio[index];
		main_dir *= w;
        new_dir += main_dir;
        total_weighting += w;
    }
    if (total_weighting < 0.5f)
        return false;
    new_dir.normalize();
    result = new_dir;
    return true;
}

bool nearest_direction::evaluate(std::shared_ptr<tracking_data> fib,
                                 const tipl::vector<3,float>& position,
                                 const tipl::vector<3,float>& ref_dir,
                                 tipl::vector<3,float>& result,
                                 float threshold,
                                 float angle,
                                 float dt_threshold)
{
    int x = std::round(position[0]);
    int y = std::round(position[1]);
    int z = std::round(position[2]);
    if(!fib->dim.is_valid(x,y,z))
        return false;
    if(!fib->get_dir(tipl::pixel_index<3>(x,y,z,fib->dim).index(),ref_dir,result,threshold,angle,dt_threshold))
        return false;
    return true;
}
