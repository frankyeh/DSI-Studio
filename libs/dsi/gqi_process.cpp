#include "gqi_process.hpp"



void GQI_Recon::calculate_sinc_ql(Voxel& voxel)
{
    unsigned int odf_size = voxel.ti.half_vertices_count;
    float sigma = voxel.param[0];
    sinc_ql.resize(odf_size*voxel.bvalues.size());
    for (unsigned int j = 0,index = 0; j < odf_size; ++j)
        for (unsigned int i = 0; i < voxel.bvalues.size(); ++i,++index)
            sinc_ql[index] = voxel.bvectors[i]*
                         tipl::vector<3,float>(voxel.ti.vertices[j])*
                           std::sqrt(voxel.bvalues[i]*0.01506f);
    for (unsigned int index = 0; index < sinc_ql.size(); ++index)
    {
        sinc_ql[index] = voxel.r2_weighted ?
                     base_function(sinc_ql[index]*sigma):
                     sinc_pi_imp(sinc_ql[index]*sigma);
    }
}
void GQI_Recon::calculate_q_vec_t(Voxel& voxel)
{
    float sigma = voxel.param[0];
    q_vectors_time.resize(voxel.bvalues.size());
    for (unsigned int index = 0; index < voxel.bvalues.size(); ++index)
    {
        q_vectors_time[index] = voxel.bvectors[index];
        q_vectors_time[index] *= std::sqrt(voxel.bvalues[index]*0.01506f);// get q in (mm) -1
        q_vectors_time[index] *= sigma;
    }
}
void GQI_Recon::init(Voxel& voxel)
{
    if(voxel.qsdr)
        calculate_q_vec_t(voxel);
    else
        calculate_sinc_ql(voxel);
    dsi_half_sphere = voxel.shell.size() > 4 && voxel.shell[1] - voxel.shell[0] <= 3;
}

void GQI_Recon::run(Voxel& voxel, VoxelData& data)
{
    if(dsi_half_sphere)
        data.space[0] *= 0.5f;
    // add rotation from QSDR or gradient nonlinearity
    if(voxel.qsdr)
    {
        std::vector<float> sinc_ql_(data.odf.size()*data.space.size());
        for (unsigned int j = 0,index = 0; j < data.odf.size(); ++j)
        {
            tipl::vector<3,float> from(voxel.ti.vertices[j]);
            from.rotate(data.jacobian);
            from.normalize();
            if(voxel.r2_weighted)
                for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                    sinc_ql_[index] = base_function(q_vectors_time[i]*from);
            else
                for (unsigned int i = 0; i < data.space.size(); ++i,++index)
                    sinc_ql_[index] = sinc_pi_imp(q_vectors_time[i]*from);

        }
        tipl::mat::vector_product(&*sinc_ql_.begin(),&*data.space.begin(),&*data.odf.begin(),
                                      tipl::shape<2>(uint32_t(data.odf.size()),uint32_t(data.space.size())));
    }
    else
        tipl::mat::vector_product(&*sinc_ql.begin(),&*data.space.begin(),&*data.odf.begin(),
                                tipl::shape<2>(uint32_t(data.odf.size()),uint32_t(data.space.size())));
}



