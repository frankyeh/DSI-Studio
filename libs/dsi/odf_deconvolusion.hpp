#ifndef ODF_DECONVOLUSION_HPP
#define ODF_DECONVOLUSION_HPP
#include "odf_decomposition.hpp"
#include "odf_process.hpp"



struct EstimateResponseFunction : public BaseProcess
{
    float max_value;
    std::mutex  mutex;
    bool has_assigned_odf;
    unsigned int assigned_index;
public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.odf_xyz[0] != 0 ||
           voxel.odf_xyz[1] != 0 ||
           voxel.odf_xyz[2] != 0)
        {
            has_assigned_odf = true;
            assigned_index = voxel.odf_xyz[0] +
                    voxel.odf_xyz[1]*voxel.dim[0] +
                    voxel.odf_xyz[2]*voxel.dim.plane_size();
        }
        else
            has_assigned_odf = false;
        voxel.response_function.resize(voxel.ti.half_vertices_count);
        voxel.reponse_function_scaling = 0;
        max_value = 0;
        std::fill(voxel.response_function.begin(),voxel.response_function.end(),1.0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::lock_guard<std::mutex> lock(mutex);

        float max_diffusion_value = std::accumulate(data.odf.begin(),data.odf.end(),0.0)/data.odf.size();
        if (max_diffusion_value > voxel.reponse_function_scaling)
		{
			voxel.reponse_function_scaling = max_diffusion_value;
			voxel.free_water_diffusion = data.odf;
        }

        if(has_assigned_odf && data.voxel_index != assigned_index)
            return;
        float cur_value = data.fa[0]-data.fa[1]-data.fa[2];
        if (cur_value < max_value)
            return;
        voxel.response_function = data.odf;
        max_value = cur_value;
    }
};

class ODFDeconvolusion  : public BaseProcess
{
protected:
    std::vector<float> A,Rt;
    std::vector<unsigned int> pv;
    float sensitivity_error_percentage;
	float specificity_error_percentage;
	// for iterative deconvolution
    std::vector<float> AA;
    unsigned int half_odf_size;
    double inner_angle(double cos_value)
    {
        double abs_cos = std::abs(cos_value);
        if (abs_cos > 1.0)
            abs_cos = 1.0;
        if (abs_cos < 0.0)
            abs_cos = 0.0;
        return std::acos(abs_cos)*2.0/M_PI;
    }
    double kernel_regression(const std::vector<float>& fiber_profile,
                             const std::vector<double>& inner_angles,
                             double cur_angle,double sigma)
    {
        sigma *= sigma;
        std::vector<double> weighting(inner_angles.size());
        for (unsigned int index = 0; index < inner_angles.size(); ++index)
        {
            double dx = cur_angle-inner_angles[index];
            weighting[index] = std::exp(-dx*dx/2.0/sigma);
        }
        double result = 0.0;
        for (unsigned int index = 0; index < fiber_profile.size(); ++index)
            result += fiber_profile[index]*weighting[index];
        result /= std::accumulate(weighting.begin(),weighting.end(),0.0);
        return result;
    }
    void estimate_Rt(Voxel& voxel)
    {
        std::vector<double> inner_angles(half_odf_size);
        {
            unsigned int max_index = std::max_element(voxel.response_function.begin(),voxel.response_function.end())-voxel.response_function.begin();
            for (unsigned int index = 0; index < inner_angles.size(); ++index)
                inner_angles[index] = inner_angle(voxel.ti.vertices_cos(index,max_index));
        }


        Rt.resize(half_odf_size*half_odf_size);
        for (unsigned int i = 0,index = 0; i < half_odf_size; ++i)
            for (unsigned int j = 0; j < half_odf_size; ++j,++index)
                Rt[index] = kernel_regression(voxel.response_function,inner_angles,inner_angle(voxel.ti.vertices_cos(i,j)),9.0/180.0*M_PI);
    }

	void deconvolution(std::vector<float>& odf)
	{
		std::vector<float> tmp(half_odf_size);
        tipl::mat::vector_product(&*Rt.begin(),&*odf.begin(),&*tmp.begin(),tipl::dyndim(half_odf_size,half_odf_size));
        tipl::mat::lu_solve(&*A.begin(),&*pv.begin(),&*tmp.begin(),&*odf.begin(),tipl::dyndim(half_odf_size,half_odf_size));
	}
	void remove_isotropic(std::vector<float>& odf)
	{
		float min_value = *std::min_element(odf.begin(),odf.end());
        if (min_value > 0)
            tipl::minus_constant(odf,min_value);
        else
            for (unsigned int index = 0; index < half_odf_size; ++index)
                if (odf[index] < 0.0)
                    odf[index] = 0.0;
	}
    float dif_ratio(Voxel& voxel,const std::vector<float>& odf)
	{
		SearchLocalMaximum local_max;
        local_max.init(voxel);
        local_max.search(odf);
        if (local_max.max_table.size() < 2)
            return 0.0;
        float first_value = local_max.max_table.begin()->first;
        float second_value = (++(local_max.max_table.begin()))->first;
        return second_value/first_value;
    }

    void get_error_percentage(Voxel& voxel)
    {
        std::vector<float> single_fiber_odf(voxel.response_function);
        std::vector<float> free_water_odf(voxel.free_water_diffusion);
        deconvolution(single_fiber_odf);
        remove_isotropic(single_fiber_odf);
        sensitivity_error_percentage = dif_ratio(voxel,single_fiber_odf);

        deconvolution(free_water_odf);
        remove_isotropic(free_water_odf);
                specificity_error_percentage = tipl::mean(free_water_odf.begin(),free_water_odf.end())/
                            (*std::max_element(single_fiber_odf.begin(),single_fiber_odf.end()));

		
	}

public:
    virtual void init(Voxel& voxel)
    {

        if (!voxel.odf_deconvolusion)
            return;
        voxel.recon_report << "Diffusion ODF deconvolution (Yeh et al., Neuroimage, 2011) was conducted using a regularization parameter of " << voxel.param[2];
        tipl::divide_constant(voxel.response_function,
                      (std::accumulate(voxel.response_function.begin(),voxel.response_function.end(),0.0)
                                            /((double)voxel.response_function.size())));
        // scale the free water diffusion to 1
        tipl::divide_constant(voxel.free_water_diffusion,voxel.reponse_function_scaling);
        

        half_odf_size = voxel.ti.half_vertices_count;
        estimate_Rt(voxel);

        A.resize(half_odf_size*half_odf_size);
        pv.resize(half_odf_size);
        tipl::mat::product_transpose(Rt.begin(),Rt.begin(),A.begin(),
                                       tipl::dyndim(half_odf_size,half_odf_size),tipl::dyndim(half_odf_size,half_odf_size));

        AA = A;
        for (unsigned int i = 0,index = 0; i < half_odf_size; ++i,index += half_odf_size + 1)
            A[index] += voxel.param[2];
        tipl::mat::lu_decomposition(A.begin(),pv.begin(),tipl::dyndim(half_odf_size,half_odf_size));

        get_error_percentage(voxel);
		
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {
        // scale the dODF using the reference to free water diffusion
        if (!voxel.odf_deconvolusion)
            return;
        tipl::divide_constant(data.odf,voxel.reponse_function_scaling);
		deconvolution(data.odf);
        remove_isotropic(data.odf);
    }

    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if (!voxel.odf_deconvolusion)
            return;
        mat_writer.write("deconvolution_kernel",&*voxel.response_function.begin(),1,voxel.response_function.size());
        mat_writer.write("free_water_diffusion",&*voxel.free_water_diffusion.begin(),1,voxel.free_water_diffusion.size());
        mat_writer.write("sensitivity_error_percentage",&sensitivity_error_percentage,1,1);
        mat_writer.write("specificity_error_percentage",&specificity_error_percentage,1,1);
    }

};

#endif//ODF_DECONVOLUSION_HPP
