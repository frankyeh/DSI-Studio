#ifndef QBI_PROCESS_HPP
#define QBI_PROCESS_HPP
#include <vector>
#include <cmath>
#include "image/image.hpp"
template<typename value_type>
void rotation_matrix(value_type r[9],value_type* u)
{
    float z[3] = {0.0,0.0,1.0};
    float value = z[0]*u[0]+z[1]*u[1]+z[2]*u[2];
    if (value < 0.0)
    {
        z[2] = -1.0;
        value = -value;
    }
    value += 1.0;
    float z_u[3];
    z_u[0] = z[0] + u[0];
    z_u[1] = z[1] + u[1];
    z_u[2] = z[2] + u[2];
    image::matrix::product(z_u,z_u,r,image::dim<3,1>(),image::dim<1,3>());

    for (unsigned int i = 0; i < 9; ++i)
        r[i] /= value;
    r[0] -= 1.0;
    r[4] -= 1.0;
    r[8] -= 1.0;
}
template<typename value_type>
value_type spherical_guassian(value_type d,value_type angular_variance)
{
    if (d > 1.0)d = 1.0;
    if (d < -1.0)d = -1.0;
    d = std::acos(d);
    return std::exp(-(d*d)/angular_variance);
}



template<unsigned int k>
struct QBIReconstruction : public BaseProcess
{
    std::vector<float> iHtH;
    std::vector<unsigned int> iHtH_pivot;
    std::vector<float> sG;
    std::vector<float> Ht; // n * m , half_odf_size-by-b_count
    std::vector<float> icosa_data; // half_odf_size-by-3
    unsigned int half_odf_size;
public:
    virtual void init(Voxel& voxel)
    {

        half_odf_size = voxel.ti.half_vertices_count;
        unsigned int b_count = voxel.bvalues.size();
        icosa_data.resize(half_odf_size*3);
        for (unsigned int index = 0; index < half_odf_size; ++index)
            std::copy(voxel.ti.vertices[index].begin(),voxel.ti.vertices[index].end(),icosa_data.begin()+index*3);

        float interop_angle = voxel.param[0]/180.0*M_PI;
        float smoothing_angle = voxel.param[1]/180.0*M_PI;

        Ht.resize(half_odf_size*b_count); // n * m
        // H=phi(acos(QtV))
        for (unsigned int n = 0,index = 0; n < half_odf_size; ++n)
            for (unsigned int m = 0; m < b_count; ++m,++index)
            {
                float value = std::abs(
                                  voxel.bvectors[m]*image::vector<3,float>(voxel.ti.vertices[n]));
                Ht[index] = spherical_guassian(value,interop_angle);
            }
        iHtH.resize(half_odf_size*half_odf_size);
        iHtH_pivot.resize(half_odf_size);
        image::matrix::square(Ht.begin(),iHtH.begin(),image::dyndim(half_odf_size,b_count));

        image::matrix::lu_decomposition(iHtH.begin(),iHtH_pivot.begin(),image::dyndim(half_odf_size,half_odf_size));

        // vector of angles
        std::vector<float> C(3*k); // 3 by k matrix
        for (unsigned int index = 0; index < k; ++index)
        {
            C[index] = std::cos(2.0*M_PI*((float)index+1)/((float)k));
            C[k+index] = std::sin(2.0*M_PI*((float)index+1)/((float)k));
            C[k+k+index] = 0.0;
        }
        // RC

        std::vector<float> G(half_odf_size*half_odf_size);
        std::vector<float> icosa_data_r(half_odf_size*3);
        for (unsigned int gi = 0; gi < half_odf_size; ++gi)
        {
            image::vector<3,float> u(voxel.ti.vertices[gi]);
            float r[9];// a 3-by-3 matrix
            rotation_matrix(r,u.begin());
            std::vector<float> Gt(half_odf_size*k); // a half_odf_size-by-k matrix

            // 	Gt = icosa_data*r*C;
            image::matrix::product(icosa_data.begin(),r,icosa_data_r.begin(),image::dyndim(half_odf_size,3),image::dim<3,3>());
            image::matrix::product(icosa_data_r.begin(),C.begin(),Gt.begin(),image::dyndim(half_odf_size,3),image::dyndim(3,k));

            for (unsigned int i = 0; i < Gt.size(); ++i)
                Gt[i] = spherical_guassian(std::abs(Gt[i]),interop_angle);

            unsigned int posgi = gi*half_odf_size;
            for (unsigned int i = 0,posi = 0; i < half_odf_size; ++i,posi+=k)
                G[posgi+i] = std::accumulate(Gt.begin()+posi,Gt.begin()+posi+k,0.0);
        }

        // add smoothing to G
        std::vector<float> S(half_odf_size*half_odf_size);
        for (unsigned int i = 0; i < half_odf_size; ++i)
        {
            float sum = 0.0;
            for (unsigned int j = 0,index = i*half_odf_size; j < half_odf_size; ++j,++index)
                sum +=
                    S[index] = spherical_guassian(std::abs(voxel.ti.vertices_cos(i,j)),smoothing_angle);

            for (unsigned int j = 0,index = i*half_odf_size; j < half_odf_size; ++j,++index)
                S[index] /= sum;
        }
        sG.resize(half_odf_size*half_odf_size);
        //sG = S*G;
        image::matrix::product(S.begin(),G.begin(),sG.begin(),image::dyndim(half_odf_size,half_odf_size),image::dyndim(half_odf_size,half_odf_size));

    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {


        // Ht_s = Ht * signal
        std::vector<float> Ht_s(half_odf_size);
        image::matrix::vector_product(Ht.begin(),data.space.begin(),Ht_s.begin(),image::dyndim(half_odf_size,data.space.size()));
        // solve HtH * x = Ht_s
        std::vector<float> x(half_odf_size);
        image::matrix::lu_solve(iHtH.begin(),iHtH_pivot.begin(),Ht_s.begin(),x.begin(),image::dyndim(half_odf_size,half_odf_size));
        // odf = sG*x
        image::matrix::vector_product(sG.begin(),x.begin(),data.odf.begin(),image::dyndim(half_odf_size,half_odf_size));

        for (unsigned int index = 0; index < data.odf.size(); ++index)
            if (data.odf[index] < 0.0)
                data.odf[index] = 0.0;
    }

};




#endif//QBI_PROCESS_HPP
