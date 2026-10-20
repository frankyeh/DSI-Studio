#include "gqi_process.hpp"

// the gqi kernel of q*u, with q scaled by the diffusion sampling length ratio
static float gqi_kernel(const Voxel& voxel,float qu)
{
    return voxel.r2_weighted ? base_function(qu) : sinc_pi_imp(qu);
}
float Voxel::odf_at(const VoxelData& data,tipl::vector<3,float> u) const
{
    if(qsdr) // the direction in the native space
    {
        u.rotate(data.jacobian);
        u.normalize();
    }
    float sum = 0.0f;
    for(size_t i = 0;i < q_vectors.size();++i)
        sum += gqi_kernel(*this,q_vectors[i]*u)*data.space[i];
    return sum;
}
void GQI_Recon::init(Voxel& voxel)
{
    voxel.q_vectors.resize(voxel.bvalues.size());
    for(size_t i = 0;i < voxel.bvalues.size();++i)
        voxel.q_vectors[i] = voxel.bvectors[i]*(std::sqrt(voxel.bvalues[i]*0.01506f)*voxel.param[0]); // q in 1/mm times sigma
    if(!voxel.qsdr) // qsdr evaluates the kernel per voxel, rotated by its jacobian
    {
        sinc_ql.resize(voxel.ti.half_vertices_count*voxel.bvalues.size());
        for(size_t j = 0,index = 0;j < voxel.ti.half_vertices_count;++j)
            for(size_t i = 0;i < voxel.bvalues.size();++i,++index)
                sinc_ql[index] = gqi_kernel(voxel,voxel.q_vectors[i]*voxel.ti.vertices[j]);
    }
    dsi_half_sphere = voxel.shell.size() > 4 && voxel.shell[1] - voxel.shell[0] <= 3;
}

void GQI_Recon::run(Voxel& voxel, VoxelData& data)
{
    if(dsi_half_sphere)
        data.space[0] *= 0.5f;
    if(voxel.qsdr)
        for(size_t j = 0;j < data.odf.size();++j)
            data.odf[j] = voxel.odf_at(data,voxel.ti.vertices[j]);
    else
        tipl::mat::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                tipl::shape<2>(uint32_t(data.odf.size()),uint32_t(data.space.size())));
}
