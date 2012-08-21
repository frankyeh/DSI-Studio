//---------------------------------------------------------------------------
#include "stdafx.h"
#include "common.hpp"
#include "math/matrix_op.hpp"


void rotation_matrix(float r[9],float* u)
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
    math::matrix_product(z_u,z_u,r,math::dim<3,1>(),math::dim<1,3>());

    for (unsigned int i = 0; i < 9; ++i)
        r[i] /= value;
    r[0] -= 1.0;
    r[4] -= 1.0;
    r[8] -= 1.0;
}
float spherical_guassian(float d,float angular_variance)
{
    if (d > 1.0)d = 1.0;
    if (d < -1.0)d = -1.0;
    d = std::acos(d);
    return std::exp(-(d*d)/angular_variance);
}
