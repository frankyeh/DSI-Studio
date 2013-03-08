#include "interpolation_process.hpp"
#include "fib_data.hpp"

bool trilinear_interpolation_with_gaussian_basis::evaluate(const fiber_orientations& fib,
                                                           const image::vector<3,float>& position,
                                                           const image::vector<3,float>& ref_dir,
                                                           image::vector<3,float>& result)
{
    image::interpolation<image::gaussian_radial_basis_weighting,3> tri_interpo;
	tri_interpo.weighting.sd = 0.5;
    if (!tri_interpo.get_location(fib.dim,position))
        return false;
    image::vector<3,float> new_dir,main_dir;
    float total_weighting = 0.0;
	float threshold = std::accumulate(tri_interpo.ratio,tri_interpo.ratio+8,0.0)*0.5;
    for (unsigned int index = 0;index < 8;++index)
    {
        unsigned int odf_space_index = tri_interpo.dindex[index];
        if (!fib.get_dir(odf_space_index,ref_dir,main_dir))
            continue;
		float w = tri_interpo.ratio[index];
		main_dir *= w;
        new_dir += main_dir;
        total_weighting += w;
    }
    if (total_weighting < threshold)
        return false;
    new_dir.normalize();
    result = new_dir;
    return true;
}


bool trilinear_interpolation::evaluate(const fiber_orientations& fib,
                                       const image::vector<3,float>& position,
                                       const image::vector<3,float>& ref_dir,
                                       image::vector<3,float>& result)
{
    image::interpolation<image::linear_weighting,3> tri_interpo;
    if (!tri_interpo.get_location(fib.dim,position))
        return false;
    image::vector<3,float> new_dir,main_dir;
    float total_weighting = 0.0;
    for (unsigned int index = 0;index < 8;++index)
    {
        unsigned int odf_space_index = tri_interpo.dindex[index];
        if (!fib.get_dir(odf_space_index,ref_dir,main_dir))
            continue;
		float w = tri_interpo.ratio[index];
		main_dir *= w;
        new_dir += main_dir;
        total_weighting += w;
    }
    if (total_weighting < 0.5)
        return false;
    new_dir.normalize();
    result = new_dir;
    return true;
}

bool nearest_direction::evaluate(const fiber_orientations& fib,
                                 const image::vector<3,float>& position,
                                 const image::vector<3,float>& ref_dir,
                                 image::vector<3,float>& result)
{
    int x = std::floor(position[0]+0.5);
    int y = std::floor(position[1]+0.5);
    int z = std::floor(position[2]+0.5);
    if(!fib.dim.is_valid(x,y,z))
        return false;
    if(!fib.get_dir(image::pixel_index<3>(x,y,z,fib.dim).index(),ref_dir,result))
        return false;
    return true;
}
