#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP
#include <cmath>
#include <image/image.hpp>
#include "math/matrix_op.hpp"
#include <vector>
#include "libs/coreg/linear.hpp"
#include "prog_interface_static_link.h"
//#define SPM_DEBUG

template<typename ImageType>
double resample_d(const ImageType& vol,double& gradx,double& grady,double& gradz,double x,double y,double z)
{
    const double TINY = 5e-2;
    int xdim = vol.width();
    int ydim = vol.height();
    int zdim = vol.depth();
        {
                double xi,yi,zi;
                xi=x-1.0;
                yi=y-1.0;
                zi=z-1.0;
                if (zi>=-TINY && zi<zdim+TINY-1 &&
                    yi>=-TINY && yi<ydim+TINY-1 &&
                    xi>=-TINY && xi<xdim+TINY-1)
                {
                        double k111,k112,k121,k122,k211,k212,k221,k222;
                        double dx1, dx2, dy1, dy2, dz1, dz2;
                        int off1, off2, offx, offy, offz, xcoord, ycoord, zcoord;

                        xcoord = (int)floor(xi); dx1=xi-xcoord; dx2=1.0-dx1;
                        ycoord = (int)floor(yi); dy1=yi-ycoord; dy2=1.0-dy1;
                        zcoord = (int)floor(zi); dz1=zi-zcoord; dz2=1.0-dz1;

                        xcoord = (xcoord < 0) ? ((offx=0),0) : ((xcoord>=xdim-1) ? ((offx=0),xdim-1) : ((offx=1   ),xcoord));
                        ycoord = (ycoord < 0) ? ((offy=0),0) : ((ycoord>=ydim-1) ? ((offy=0),ydim-1) : ((offy=xdim),ycoord));
                        zcoord = (zcoord < 0) ? ((offz=0),0) : ((zcoord>=zdim-1) ? ((offz=0),zdim-1) : ((offz=1   ),zcoord));

                        off1 = xcoord  + xdim*ycoord;
                        off2 = off1+offy;
                        int z_offset = zcoord*vol.plane_size();
                        int z_offset_off = z_offset + (offz == 1 ? vol.plane_size():0);
                        k222 = vol[z_offset+off1]; k122 = vol[z_offset+off1+offx];
                        k212 = vol[z_offset+off2]; k112 = vol[z_offset+off2+offx];
                        k221 = vol[z_offset_off+off1]; k121 = vol[z_offset_off+off1+offx];
                        k211 = vol[z_offset_off+off2]; k111 = vol[z_offset_off+off2+offx];

                        gradx = (((k111 - k211)*dy1 + (k121 - k221)*dy2))*dz1
                                 + (((k112 - k212)*dy1 + (k122 - k222)*dy2))*dz2;

                        k111 = (k111*dx1 + k211*dx2);
                        k121 = (k121*dx1 + k221*dx2);
                        k112 = (k112*dx1 + k212*dx2);
                        k122 = (k122*dx1 + k222*dx2);

                        grady = (k111 - k121)*dz1 + (k112 - k122)*dz2;

                        k111 = k111*dy1 + k121*dy2;
                        k112 = k112*dy1 + k122*dy2;

                        gradz = k111 - k112;
                        return k111*dz1 + k112*dz2;
                }
                else
                {
                        gradx = 0.0;
                        grady = 0.0;
                        gradz = 0.0;
                        return 0;
                }
        }
}
//[Alpha,Beta,Var,fw] = spm_brainwarp(VG,VF,Affine,basX,basY,basZ,dbasX,dbasY,dbasZ,T,fwhm,VWG, VWF);
/*
INPUTS
T[3*nz*ny*nx + ni*4] - current transform
VF              - image to normalize
ni              - number of templates
vols2[ni]       - templates

nx              - number of basis functions in x
B0[dim1[0]*nx]  - basis functions in x
dB0[dim1[0]*nx] - derivatives of basis functions in x
ny              - number of basis functions in y
B1[dim1[1]*ny]  - basis functions in y
dB1[dim1[1]*ny] - derivatives of basis functions in y
nz              - number of basis functions in z
B2[dim1[2]*nz]  - basis functions in z
dB2[dim1[2]*nz] - derivatives of basis functions in z

M[4*4]          - transformation matrix
samp[3]         - frequency of sampling template.


OUTPUTS
alpha[(3*nz*ny*nx+ni*4)^2] - A'A
 beta[(3*nz*ny*nx+ni*4)]   - A'b
pss[1*1]                   - sum of squares difference
pnsamp[1*1]                - number of voxels sampled
ss_deriv[3]                - sum of squares of derivatives of residuals
*/
template<typename image_type,typename affine_type,typename parameter_type>
void mrqcof(const image_type& VG,
            const image_type& VF,
            double* VGvs,
            double* VFvs,
            const std::vector<parameter_type>& T,
            const std::vector<std::vector<parameter_type> >& base,
            const std::vector<std::vector<parameter_type> >& dbase,
            affine_type M,
            double fwhm,double fwhm2,
            std::vector<parameter_type>& alpha,
            std::vector<parameter_type>& beta,
            double& var,
            double& fw)
{
    double ss = 0.0,nsamp = 0.0;
    double pixdim[3],ss_deriv[3]={0.0,0.0,0.0};

    int ni = 1;
    int nx = base[0].size()/VG.width();
    int ny = base[1].size()/VG.height();
    int nz = base[2].size()/VG.depth();
    int edgeskip[3],samp[3];
    const std::vector<double>& B0 = base[0];
    const std::vector<double>& B1 = base[1];
    const std::vector<double>& B2 = base[2];
    const std::vector<double>& dB0 = dbase[0];
    const std::vector<double>& dB1 = dbase[1];
    const std::vector<double>& dB2 = dbase[2];

    const double *bz3[3], *by3[3], *bx3[3];

    bx3[0] = &dB0[0];	bx3[1] =  &B0[0];	bx3[2] =  &B0[0];
    by3[0] =  &B1[0];	by3[1] = &dB1[0];	by3[2] =  &B1[0];
    bz3[0] =  &B2[0];	bz3[1] =  &B2[0];	bz3[2] = &dB2[0];
    {


            /* Because of edge effects from the smoothing, ignore voxels that are too close */
            pixdim[0] = VFvs[0];
            pixdim[1] = VFvs[1];
            pixdim[2] = VFvs[2];
            edgeskip[0]   = std::floor(fwhm/pixdim[0]); edgeskip[0] = ((edgeskip[0]<1) ? 0 : edgeskip[0]);
            edgeskip[1]   = std::floor(fwhm/pixdim[1]); edgeskip[1] = ((edgeskip[1]<1) ? 0 : edgeskip[1]);
            edgeskip[2]   = std::floor(fwhm/pixdim[2]); edgeskip[2] = ((edgeskip[2]<1) ? 0 : edgeskip[2]);


            /* sample about every fwhm/2 */
            pixdim[0] = VGvs[0];
            pixdim[1] = VGvs[1];
            pixdim[2] = VGvs[2];
            samp[0]   = std::floor(fwhm/2.0/pixdim[0]); samp[0] = ((samp[0]<1) ? 1 : samp[0]);
            samp[1]   = std::floor(fwhm/2.0/pixdim[1]); samp[1] = ((samp[1]<1) ? 1 : samp[1]);
            samp[2]   = std::floor(fwhm/2.0/pixdim[2]); samp[2] = ((samp[2]<1) ? 1 : samp[2]);

            alpha.resize((3*nx*ny*nz+ni*4)*(3*nx*ny*nz+ni*4)); // plhs[0]
            beta.resize(3*nx*ny*nz+ni*4); // plhs[1]

    }


    int i1,i2, s0[3], x1,x2, y1,y2, z1,z2, m1, m2, ni4 = ni*4;
    double s2[3], tmp;
    double *ptr1, *ptr2;
    const double *scal,*ptr;
    double J[3][3];
    image::geometry<3> dim1(VG.geometry());

        /* rate of change of voxel with respect to change in parameters */
        std::vector<double> dvdt( 3*nx    + ni4);

        /* Intermediate storage arrays */
        std::vector<double> Tz( 3*nx*ny );
        std::vector<double> Ty( 3*nx );
        std::vector<double> betax( 3*nx    + ni4);
        std::vector<double> betaxy( 3*nx*ny + ni4);
        std::vector<double> alphax((3*nx    + ni4)*(3*nx    + ni4));
        std::vector<double> alphaxy((3*nx*ny + ni4)*(3*nx*ny + ni4));

        std::vector<std::vector<std::vector<double> > > Jz(3),Jy(3);

        for (i1=0; i1<3; i1++)
        {
            Jz[i1].resize(3);
            Jy[i1].resize(3);
                for(i2=0; i2<3; i2++)
                {
                        Jz[i1][i2].resize(nx*ny);
                        Jy[i1][i2].resize(nx);
                }
        }


        /* pointer to scales for each of the template images */
        scal = &T[3*nx*ny*nz];

        /* only zero half the matrix */
        m1 = 3*nx*ny*nz+ni4;
        for (x1=0;x1<m1;x1++)
        {
                for (x2=0;x2<=x1;x2++)
                        alpha[m1*x1+x2] = 0.0;
                beta[x1]= 0.0;
        }


        for(s0[2]=1; s0[2]<dim1[2]; s0[2]+=samp[2]) /* For each plane of the template images */
        {
                /* build up the deformation field (and derivatives) from it's seperable form */
                for(i1=0, ptr=&T[0]; i1<3; i1++, ptr += nz*ny*nx)
                        for(x1=0; x1<nx*ny; x1++)
                        {
                                /* intermediate step in computing nonlinear deformation field */
                                tmp = 0.0;
                                for(z1=0; z1<nz; z1++)
                                        tmp  += ptr[x1+z1*ny*nx] * B2[dim1[2]*z1+s0[2]];
                                Tz[ny*nx*i1 + x1] = tmp;

                                /* intermediate step in computing Jacobian of nonlinear deformation field */
                                for(i2=0; i2<3; i2++)
                                {
                                        tmp = 0;
                                        for(z1=0; z1<nz; z1++)
                                                tmp += ptr[x1+z1*ny*nx] * bz3[i2][dim1[2]*z1+s0[2]];
                                        Jz[i2][i1][x1] = tmp;
                                }
                        }

                /* only zero half the matrix */
                m1 = 3*nx*ny+ni4;
                for (x1=0;x1<m1;x1++)
                {
                        for (x2=0;x2<=x1;x2++)
                                alphaxy[m1*x1+x2] = 0.0;
                        betaxy[x1]= 0.0;
                }

                for(s0[1]=1; s0[1]<dim1[1]; s0[1]+=samp[1]) /* For each row of the template images plane */
                {
                        /* build up the deformation field (and derivatives) from it's seperable form */
                        for(i1=0, ptr=&Tz[0]; i1<3; i1++, ptr+=ny*nx)
                        {
                                for(x1=0; x1<nx; x1++)
                                {
                                        /* intermediate step in computing nonlinear deformation field */
                                        tmp = 0.0;
                                        for(y1=0; y1<ny; y1++)
                                                tmp  += ptr[x1+y1*nx] *  B1[dim1[1]*y1+s0[1]];
                                        Ty[nx*i1 + x1] = tmp;

                                        /* intermediate step in computing Jacobian of nonlinear deformation field */
                                        for(i2=0; i2<3; i2++)
                                        {
                                                tmp = 0;
                                                for(y1=0; y1<ny; y1++)
                                                        tmp += Jz[i2][i1][x1+y1*nx] * by3[i2][dim1[1]*y1+s0[1]];
                                                Jy[i2][i1][x1] = tmp;
                                        }
                                }
                        }

                        /* only zero half the matrix */
                        m1 = 3*nx+ni4;
                        for(x1=0;x1<m1;x1++)
                        {
                                for (x2=0;x2<=x1;x2++)
                                        alphax[m1*x1+x2] = 0.0;
                                betax[x1]= 0.0;
                        }

                        for(s0[0]=1; s0[0]<dim1[0]; s0[0]+=samp[0]) /* For each pixel in the row */
                        {
                                double trans[3];

                                /* nonlinear deformation of the template space, followed by the affine transform */
                                for(i1=0, ptr = &Ty[0]; i1<3; i1++, ptr += nx)
                                {
                                        /* compute nonlinear deformation field */
                                        tmp = 0.0;
                                        for(x1=0; x1<nx; x1++)
                                                tmp  += ptr[x1] * B0[dim1[0]*x1+s0[0]];
                                        trans[i1] = tmp + s0[i1];

                                        /* compute Jacobian of nonlinear deformation field */
                                        for(i2=0; i2<3; i2++)
                                        {
                                                if (i1 == i2) tmp = 1.0; else tmp = 0;
                                                for(x1=0; x1<nx; x1++)
                                                        tmp += Jy[i2][i1][x1] * bx3[i2][dim1[0]*x1+s0[0]];
                                                J[i2][i1] = tmp;
                                        }
                                }

                                /* Affine component */
                                //MtimesX(M, trans, s2);
                                // here using 1-based affine matrix
                                image::vector_transformation(trans,s2,M,image::vdim<3>());

                                /* is the transformed position in range? */
                                if (	s2[0]>=1+edgeskip[0] && s2[0]< VF.width()-edgeskip[0] &&
                                        s2[1]>=1+edgeskip[1] && s2[1]< VF.height()-edgeskip[1] &&
                                        s2[2]>=1+edgeskip[2] && s2[2]< VF.depth()-edgeskip[2] )
                                {
                                        double f, df[3], dv, dvds0[3];
                                        double wtf, wtg, wt;
                                        double s0d[3];
                                        s0d[0]=s0[0];s0d[1]=s0[1];s0d[2]=s0[2];

                                        f = resample_d(VF,df[0],df[1],df[2],s2[0],s2[1],s2[2]);
                                        {
                                                double x0, x1;
                                                x0   = M[0]*df[0] + M[4]*df[1] + M[8]*df[2];
                                                x1   = M[1]*df[0] + M[5]*df[1] + M[9]*df[2];
                                                df[2] = M[2]*df[0] + M[6]*df[1] + M[10]*df[2];
                                                df[1] = x1;
                                                df[0] = x0;
                                        }
                                        wtg = 1.0;
                                        wtf = 1.0;

                                        if (wtf && wtg) wt = sqrt(1.0 /(1.0/wtf + 1.0/wtg));
                                        else wt = 0.0;

                                        /* nonlinear transform the gradients to the same space as the template */
                                        dvds0[0] = J[0][0]*df[0] + J[0][1]*df[1] + J[0][2]*df[2];
                                        dvds0[1] = J[1][0]*df[0] + J[1][1]*df[1] + J[1][2]*df[2];
                                        dvds0[2] = J[2][0]*df[0] + J[2][1]*df[1] + J[2][2]*df[2];

                                        dv = f;
                                        for(i1=0; i1<ni; i1++)
                                        {
                                                double g, dg[3], tmp;
                                                g = resample_d(VG,dg[0],dg[1],dg[2],s0d[0],s0d[1],s0d[2]);

                                                /* linear combination of image and image modulated by constant
                                                   gradients in x, y and z */
                                                dvdt[i1*4  +3*nx] = wt*g;
                                                dvdt[i1*4+1+3*nx] = wt*g*s2[0];
                                                dvdt[i1*4+2+3*nx] = wt*g*s2[1];
                                                dvdt[i1*4+3+3*nx] = wt*g*s2[2];

                                                tmp = scal[i1*4] + s2[0]*scal[i1*4+1] +
                                                        s2[1]*scal[i1*4+2] + s2[2]*scal[i1*4+3];

                                                dv       -= tmp*g;
                                                dvds0[0] -= tmp*dg[0];
                                                dvds0[1] -= tmp*dg[1];
                                                dvds0[2] -= tmp*dg[2];
                                        }

                                        for(i1=0; i1<3; i1++)
                                        {
                                                double tmp = -wt*df[i1];
                                                for(x1=0; x1<nx; x1++)
                                                        dvdt[i1*nx+x1] = tmp * B0[dim1[0]*x1+s0[0]];
                                        }

                                        /* cf Numerical Recipies "mrqcof.c" routine */
                                        m1 = 3*nx+ni4;
                                        for(x1=0; x1<m1; x1++)
                                        {
                                                for (x2=0;x2<=x1;x2++)
                                                        alphax[m1*x1+x2] += dvdt[x1]*dvdt[x2];
                                                betax[x1] += dvdt[x1]*dv*wt;
                                        }

                                        /* sum of squares */
                                        wt          *= wt;
                                        nsamp       += wt;
                                        ss          += wt*dv*dv;
                                        ss_deriv[0] += wt*dvds0[0]*dvds0[0];
                                        ss_deriv[1] += wt*dvds0[1]*dvds0[1];
                                        ss_deriv[2] += wt*dvds0[2]*dvds0[2];
                                }
                        }

                        m1 = 3*nx*ny+ni4;
                        m2 = 3*nx+ni4;

                        /* Kronecker tensor products */
                        for(y1=0; y1<ny; y1++)
                        {
                                double wt1 = B1[dim1[1]*y1+s0[1]];

                                for(i1=0; i1<3; i1++)	/* loop over deformations in x, y and z */
                                {
                                        /* spatial-spatial covariances */
                                        for(i2=0; i2<=i1; i2++)	/* symmetric matrixes - so only work on half */
                                        {
                                                for(y2=0; y2<=y1; y2++)
                                                {
                                                        /* Kronecker tensor products with B1'*B1 */
                                                        double wt2 = wt1 * B1[dim1[1]*y2+s0[1]];

                                                        ptr1 = &alphaxy[nx*(m1*(ny*i1 + y1) + ny*i2 + y2)];
                                                        ptr2 = &alphax[nx*(m2*i1 + i2)];

                                                        for(x1=0; x1<nx; x1++)
                                                        {
                                                                for (x2=0;x2<=x1;x2++)
                                                                        ptr1[m1*x1+x2] += wt2 * ptr2[m2*x1+x2];
                                                        }
                                                }
                                        }

                                        /* spatial-intensity covariances */
                                        ptr1 = &alphaxy[nx*(m1*ny*3 + ny*i1 + y1)];
                                        ptr2 = &alphax[nx*(m2*3 + i1)];
                                        for(x1=0; x1<ni4; x1++)
                                        {
                                                for (x2=0;x2<nx;x2++)
                                                        ptr1[m1*x1+x2] += wt1 * ptr2[m2*x1+x2];
                                        }

                                        /* spatial component of beta */
                                        for(x1=0; x1<nx; x1++)
                                                betaxy[x1+nx*(ny*i1 + y1)] += wt1 * betax[x1 + nx*i1];
                                }
                        }
                        ptr1 = &alphaxy[nx*(m1*ny*3 + ny*3)];
                        ptr2 = &alphax[nx*(m2*3 + 3)];
                        for(x1=0; x1<ni4; x1++)
                        {
                                /* intensity-intensity covariances  */
                                for (x2=0; x2<=x1; x2++)
                                        ptr1[m1*x1 + x2] += ptr2[m2*x1 + x2];

                                /* intensity component of beta */
                                betaxy[nx*ny*3 + x1] += betax[nx*3 + x1];
                        }
                }

                m1 = 3*nx*ny*nz+ni4;
                m2 = 3*nx*ny+ni4;

                /* Kronecker tensor products */
                for(z1=0; z1<nz; z1++)
                {
                        double wt1 = B2[dim1[2]*z1+s0[2]];

                        for(i1=0; i1<3; i1++)	/* loop over deformations in x, y and z */
                        {
                                /* spatial-spatial covariances */
                                for(i2=0; i2<=i1; i2++)	/* symmetric matrixes - so only work on half */
                                {
                                        for(z2=0; z2<=z1; z2++)
                                        {
                                                /* Kronecker tensor products with B2'*B2 */
                                                double wt2 = wt1 * B2[dim1[2]*z2+s0[2]];

                                                ptr1 = &alpha[nx*ny*(m1*(nz*i1 + z1) + nz*i2 + z2)];
                                                ptr2 = &alphaxy[nx*ny*(m2*i1 + i2)];
                                                for(y1=0; y1<ny*nx; y1++)
                                                {
                                                        for (y2=0;y2<=y1;y2++)
                                                                ptr1[m1*y1+y2] += wt2 * ptr2[m2*y1+y2];
                                                }
                                        }
                                }

                                /* spatial-intensity covariances */
                                ptr1 = &alpha[nx*ny*(m1*nz*3 + nz*i1 + z1)];
                                ptr2 = &alphaxy[nx*ny*(m2*3 + i1)];
                                for(y1=0; y1<ni4; y1++)
                                {
                                        for (y2=0;y2<ny*nx;y2++)
                                                ptr1[m1*y1+y2] += wt1 * ptr2[m2*y1+y2];
                                }

                                /* spatial component of beta */
                                for(y1=0; y1<ny*nx; y1++)
                                        beta[y1 + nx*ny*(nz*i1 + z1)] += wt1 * betaxy[y1 + nx*ny*i1];
                        }
                }

                ptr1 = &alpha[nx*ny*(m1*nz*3 + nz*3)];
                ptr2 = &alphaxy[nx*ny*(m2*3 + 3)];
                for(y1=0; y1<ni4; y1++)
                {
                        /* intensity-intensity covariances */
                        for(y2=0;y2<=y1;y2++)
                                ptr1[m1*y1 + y2] += ptr2[m2*y1 + y2];

                        /* intensity component of beta */
                        beta[nx*ny*nz*3 + y1] += betaxy[nx*ny*3 + y1];
                }
        }


        /* Fill in the symmetric bits
           - OK I know some bits are done more than once. */

        m1 = 3*nx*ny*nz+ni4;
        for(i1=0; i1<3; i1++)
        {
                double *ptrz, *ptry, *ptrx;
                for(i2=0; i2<=i1; i2++)
                {
                        ptrz = &alpha[nx*ny*nz*(m1*i1 + i2)];
                        for(z1=0; z1<nz; z1++)
                                for(z2=0; z2<=z1; z2++)
                                {
                                        ptry = ptrz + nx*ny*(m1*z1 + z2);
                                        for(y1=0; y1<ny; y1++)
                                                for (y2=0;y2<=y1;y2++)
                                                {
                                                        ptrx = ptry + nx*(m1*y1 + y2);
                                                        for(x1=0; x1<nx; x1++)
                                                                for(x2=0; x2<x1; x2++)
                                                                        ptrx[m1*x2+x1] = ptrx[m1*x1+x2];
                                                }
                                        for(x1=0; x1<nx*ny; x1++)
                                                for (x2=0; x2<x1; x2++)
                                                        ptry[m1*x2+x1] = ptry[m1*x1+x2];
                                }
                        for(x1=0; x1<nx*ny*nz; x1++)
                                for (x2=0; x2<x1; x2++)
                                        ptrz[m1*x2+x1] = ptrz[m1*x1+x2];
                }
        }
        for(x1=0; x1<nx*ny*nz*3+ni4; x1++)
                for (x2=0; x2<x1; x2++)
                        alpha[m1*x2+x1] = alpha[m1*x1+x2];

        //

        fw = ((pixdim[0]/sqrt(2.0*ss_deriv[0]/ss))*sqrt(8.0*log(2.0)) +
                 (pixdim[1]/sqrt(2.0*ss_deriv[1]/ss))*sqrt(8.0*log(2.0)) +
                 (pixdim[2]/sqrt(2.0*ss_deriv[2]/ss))*sqrt(8.0*log(2.0)))/3.0;


        if (fw<fwhm2)
                fwhm2 = fw;
        if (fwhm2<fwhm)
                fwhm2 = fwhm;

        ss /= (std::min((pixdim[0]*samp[0])/(fwhm2*1.0645),1.0) *
              std::min((pixdim[1]*samp[1])/(fwhm2*1.0645),1.0) *
              std::min((pixdim[2]*samp[2])/(fwhm2*1.0645),1.0)) * (nsamp - (3*nx*ny*nz + ni*4));

        var = ss;
        image::divide_constant(alpha.begin(),alpha.end(), ss);
        image::divide_constant(beta.begin(),beta.end(), ss);

}



template<typename image_type>
class normalization{
    static const int dim = image_type::dimension;
public:
    double VGvs[dim],VFvs[dim];
    image_type VG,VF;
    std::vector<double> VG_trans;
public://  fornormalization
    double fwhm[2];
    double stabilise,reg;
    std::vector<double> T;

public:// for warping
    image::vector<dim,int> bounding_box_lower;
    image::vector<dim,int> bounding_box_upper;
    image::geometry<dim> BDim;// = {79,95,69};	    // the dimension of normalized images
    image::vector<dim,int> BOffset;// = {6,7,11};	// the offset due to bounding box
    image::vector<dim,double> scale;
    std::vector<std::vector<double> > bas,dbas;

    template<typename geo_type,typename bb_type>
    void initialize_basis_function(const geo_type& bb_geo, // geometry of the bounding box
                                   const bb_type& bb) // bounding offset
    {
        int k[3] = {7,9,7};
        const int dimension = geo_type::dimension;
        const image::geometry<dimension>& geo = VG.geometry();
        bas.resize(dimension);
        dbas.resize(dimension);
        for(int dim = 0;dim < dimension;++dim)
        {
            double pi_inv_mni_dim = 3.14159265358979323846/(double)geo[dim];
            bas[dim].resize(bb_geo[dim]*k[dim]);
            dbas[dim].resize(bb_geo[dim]*k[dim]);
            // C(:,1)=ones(size(n,1),1)/sqrt(N);
            std::fill(bas[dim].begin(),bas[dim].begin()+bb_geo[dim],stabilise/std::sqrt((float)geo[dim]));
            std::fill(dbas[dim].begin(),dbas[dim].begin()+bb_geo[dim],0.0);
            for(int i = 1,index = bb_geo[dim];i < k[dim];++i)
            for(int n = 0;n < bb_geo[dim];++n,++index)
            {
                // C(:,k) = sqrt(2/N)*cos(pi*(2*n+1)*(k-1)/(2*N));
                bas[dim][index] = stabilise*std::sqrt(2.0/(double)geo[dim])*std::cos(pi_inv_mni_dim*(double)i*((double)n*scale[dim]+bb[dim]+0.5));
                // C(:,k) = -2^(1/2)*(1/N)^(1/2)*sin(1/2*pi*(2*n*k-2*n+k-1)/N)*pi*(k-1)/N;
                dbas[dim][index] = -stabilise*std::sqrt(2.0/(double)geo[dim])*std::sin(pi_inv_mni_dim*(double)i*((double)n*scale[dim]+bb[dim]+0.5))*pi_inv_mni_dim*i;
            }
    #ifdef SPM_DEBUG
        std::cout << "bas:" << bas[dim].size() << std::endl;
        std::copy(bas[dim].begin(),bas[dim].end(),std::ostream_iterator<double>(std::cout," "));
        std::cout << std::endl;
        std::cout << "dbas:" << dbas[dim].size() << std::endl;
        std::copy(dbas[dim].begin(),dbas[dim].end(),std::ostream_iterator<double>(std::cout," "));
        std::cout << std::endl;
    #endif
        }
    }

public://for linear mapping
    typedef LinearMapping<
            image::basic_image<typename image_type::value_type,
            dim,image::const_pointer_memory<typename image_type::value_type> >,
            image::affine_transform<dim,double> > lm_type;

    std::auto_ptr<lm_type> mi3;
    image::affine_transform<dim,double> arg_min;
    double affine[16];
    double affine_rotation[9];
    double affine_det;
    double trans_to_mni[16];
    template<typename reg_type>
    void set_affine(const reg_type& trans)
    {
        image::transformation_matrix<dim,double> affine_buf(trans);
        image::reg::linear_get_trans(VF.geometry(),VG.geometry(),affine_buf);
        affine_buf.save_to_transform(affine);
        std::fill(affine+12,affine+15,0);
        affine[15] = 1.0;
        math::matrix_inverse(affine,math::dim<4,4>());
        std::copy(affine,affine+3,affine_rotation);
        std::copy(affine+4,affine+4+3,affine_rotation+3);
        std::copy(affine+8,affine+8+3,affine_rotation+6);
        affine_det = math::matrix_determinant(affine_rotation,math::dim<3,3>());
    }

public:
    normalization(void)
    {
        fwhm[0] = 8;
        fwhm[1] = 30;
        stabilise = 8;
        reg = 1.0;
    }
    template<typename char_type>
    bool load_from_file(const char_type* template_name,const char_type* subject_name)
    {
        {
            image::io::nifti read;
            if(!read.load_from_file(subject_name))
                return false;
            read >> VF;
            image::normalize(VF,1.0);
            read.get_voxel_size(VFvs);
            #ifndef SPM_DEBUG
            image::flip_xy(VF);
            #endif
        }
        {
            image::io::nifti read;
            if(!read.load_from_file(template_name))
                return false;
            read >> VG;
            image::normalize(VG,1.0);
            read.get_voxel_size(VGvs);
            VG_trans.resize(16);
            read.get_image_transformation(VG_trans.begin());
            #ifndef SPM_DEBUG
            if(VG_trans[0] < 0)
            {
                image::flip_y(VG);
                VG_trans[0] = -VG_trans[0];
                VG_trans[3] -= (VG.geometry()[0]+1)*VG_trans[0];
            }
            else
                image::flip_xy(VG);
            #endif

        }
        return true;
    }

    /*
    template<typename matrix_type>
    void get_jacobian(const image::pixel_index<3>& from,matrix_type M)
    {
        if(!BDim.is_valid(from))
        {
            std::copy(affine_rotation,affine_rotation+9,M);
            return;
        }
        double Jbet[9];
        int index = from.index();
        Jbet[0] = 1 + jacobian[0][index];
        Jbet[1] = jacobian[1][index];
        Jbet[2] = jacobian[2][index];
        index += VG.size();
        Jbet[3] = jacobian[0][index];
        Jbet[4] = 1 + jacobian[1][index];
        Jbet[5] = jacobian[2][index];
        index += VG.size();
        Jbet[6] = jacobian[0][index];
        Jbet[7] = jacobian[1][index];
        Jbet[8] = 1 + jacobian[2][index];
        math::matrix_product(affine_rotation,Jbet,M,math::dim<3,3>(),math::dim<3,3>());
    }
    */

    template<typename from_type,typename matrix_type>
    void get_jacobian(const from_type& from,matrix_type M)
    {
        double bx[7],by[9],bz[7];
        double dbx[7],dby[9],dbz[7];
        for(unsigned int k = 0,index = from[0];k < 7;++k,index += BDim[0])
        {
            bx[k] = bas[0][index];
            dbx[k] = dbas[0][index];
        }
        for(unsigned int k = 0,index = from[1];k < 9;++k,index += BDim[1])
        {
            by[k] = bas[1][index];
            dby[k] = dbas[1][index];
        }
        for(unsigned int k = 0,index = from[2];k < 7;++k,index += BDim[2])
        {
            bz[k] = bas[2][index];
            dbz[k] = dbas[2][index];
        }
        double temp[63];
        double temp2[7];
        double Jbet[9];

        // f(x)/dx
        math::matrix_product(T.begin(),dbx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[0] = 1 + math::vector_op_dot(bz,bz+7,temp2);
        // f(x)/dy
        math::matrix_product(T.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,dby,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[1] = math::vector_op_dot(bz,bz+7,temp2);
        // f(x)/dz
        math::matrix_product(T.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[2] = math::vector_op_dot(dbz,dbz+7,temp2);

        // f(y)/dx
        math::matrix_product(T.begin()+7*9*7,dbx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[3] = math::vector_op_dot(bz,bz+7,temp2);
        // f(y)/dy
        math::matrix_product(T.begin()+7*9*7,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,dby,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[4] = 1 + math::vector_op_dot(bz,bz+7,temp2);
        // f(y)/dz
        math::matrix_product(T.begin()+7*9*7,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[5] = math::vector_op_dot(dbz,dbz+7,temp2);

        // f(z)/dx
        math::matrix_product(T.begin()+7*9*7*2,dbx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[6] = math::vector_op_dot(bz,bz+7,temp2);
        // f(z)/dy
        math::matrix_product(T.begin()+7*9*7*2,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,dby,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[7] = math::vector_op_dot(bz,bz+7,temp2);
        // f(z)/dz
        math::matrix_product(T.begin()+7*9*7*2,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        Jbet[8] = 1 + math::vector_op_dot(dbz,dbz+7,temp2);

        math::matrix_product(affine_rotation,Jbet,M,math::dim<3,3>(),math::dim<3,3>());
    }
    template<typename rhs_type1,typename rhs_type2>
    void warp_coordinate(const rhs_type1& from,rhs_type2& to)
    {
        if(!BDim.is_valid(from))
        {
            to = from;
            return;
        }
        double bx[7],by[9],bz[7];
        for(unsigned int k = 0,index = from[0];k < 7;++k,index += BDim[0])
            bx[k] = bas[0][index];
        for(unsigned int k = 0,index = from[1];k < 9;++k,index += BDim[1])
            by[k] = bas[1][index];
        for(unsigned int k = 0,index = from[2];k < 7;++k,index += BDim[2])
            bz[k] = bas[2][index];

        double nx = BOffset[0] + from[0]*scale[0];
        double ny = BOffset[1] + from[1]*scale[1];
        double nz = BOffset[2] + from[2]*scale[2];
        double temp[63];
        double temp2[7];
        image::vector<3,double> tt;

        math::matrix_product(T.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        tt[0] = nx + math::vector_op_dot(bz,bz+7,temp2);



        math::matrix_product(T.begin()+7*9*7,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        tt[1] = ny + math::vector_op_dot(bz,bz+7,temp2);

        math::matrix_product(T.begin()+7*9*7*2,bx,temp,math::dim<63,7>(),math::dim<7,1>());
        math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
        tt[2] = nz + math::vector_op_dot(bz,bz+7,temp2);

        /*
        std::cout << "bx" << std::endl;
        std::copy(bx,bx+7,std::ostream_iterator<double>(std::cout," "));
        std::cout << std::endl;
        std::cout << "by" << std::endl;
        std::copy(by,by+9,std::ostream_iterator<double>(std::cout," "));
        std::cout << std::endl;
        std::cout << "bz" << std::endl;
        std::copy(bz,bz+7,std::ostream_iterator<double>(std::cout," "));
        std::cout << std::endl;

        std::cout << "nx=" << nx << " ny=" << ny << " nz=" << nz << std::endl;
        std::cout << "x1=" << tt[0] << std::endl;
        std::cout << "y1=" << tt[1] << std::endl;
        std::cout << "z1=" << tt[2] << std::endl;
        */
        tt += 1.0;

        image::vector_transformation(tt,to.begin(),affine,image::vdim<3>());


        to -= 1.0;// transform back to our 0 0 0 coordinate
    }
    void set_voxel_size(double voxel_size)
    {
        //setBoundingBox(-78,-112,-50,78,76,85,1.0);
        int fx = -78;
        int fy = -112;
        int fz = -50;
        int tx = 78;
        int ty = 76;
        int tz = 85;
        bounding_box_lower[0] = std::floor(fx/voxel_size+0.5)*voxel_size;
        bounding_box_lower[1] = std::floor(fy/voxel_size+0.5)*voxel_size;
        bounding_box_lower[2] = std::floor(fz/voxel_size+0.5)*voxel_size;
        bounding_box_upper[0] = std::floor(tx/voxel_size+0.5)*voxel_size;
        bounding_box_upper[1] = std::floor(ty/voxel_size+0.5)*voxel_size;
        bounding_box_upper[2] = std::floor(tz/voxel_size+0.5)*voxel_size;

        image::vector<3,double> vx;
        double o[3];
        vx[0] = std::fabs(VG_trans[0]);
        vx[1] = std::fabs(VG_trans[5]);
        vx[2] = std::fabs(VG_trans[10]);
        //if(VG_trans[0]*VG_trans[5]*VG_trans[10] < 0)
        //    vx[0] = -vx[0];
        o[0] = -VG_trans[3]/VG_trans[0];
        o[1] = -VG_trans[7]/VG_trans[5];
        o[2] = -VG_trans[11]/VG_trans[10];
        BDim[0] = (bounding_box_upper[0]-bounding_box_lower[0])/voxel_size+1;//79
        BDim[1] = (bounding_box_upper[1]-bounding_box_lower[1])/voxel_size+1;//95
        BDim[2] = (bounding_box_upper[2]-bounding_box_lower[2])/voxel_size+1;//69
        BOffset[0] = bounding_box_lower[0]/vx[0]+o[0];

        //BOffset[0] = VG.width()-BOffset[0]-(bounding_box_upper[0]-bounding_box_lower[0])/vx[0]+1;

        BOffset[1] = bounding_box_lower[1]/vx[1]+o[1];
        // y flip
        //BOffset[1] = VG.height()-BOffset[1]-(bounding_box_upper[1]-bounding_box_lower[1])/vx[1]+1;
        BOffset[2] = bounding_box_lower[2]/vx[2]+o[2];

#ifdef SPM_DEBUG
        std::cout << "bblower=" << bounding_box_lower[0] <<
                "," << bounding_box_lower[1] <<
                "," << bounding_box_lower[2] <<std::endl;
        std::cout << "bbupper=" << bounding_box_upper[0] <<
                "," << bounding_box_upper[1] <<
                "," << bounding_box_upper[2] <<std::endl;
        std::cout << "BDim=" << BDim[0] <<
                "," << BDim[1] <<
                "," << BDim[2] <<std::endl;
        std::cout << "BOffset=" << BOffset[0] <<
                "," << BOffset[1] <<
                "," << BOffset[2] <<std::endl;
        std::cout << "o=" << o[0] <<
                "," << o[1] <<
                "," << o[2] <<std::endl;
#endif

        scale[0] = voxel_size/vx[0];
        scale[1] = voxel_size/vx[1];
        scale[2] = voxel_size/vx[2];

        stabilise = 1;
        initialize_basis_function(BDim,BOffset);
        /*
        std::vector<int> k(3);
        k[0] = 7;
        k[1] = 9;
        k[2] = 7;
        std::vector<std::vector<double> > bas,dbas;
        // T now is scaled by std::pow(stabilise,-3), so still set stablise = 8
        initialize_basis_function(VG.geometry(),BDim,BOffset,scale,bas,dbas,k,1.0);
        // calculating the
        calculate_(bas[0],bas[1],bas[2],);
        calculate_(dbas[0],bas[1],bas[2],jacobian[0]);
        calculate_(bas[0],dbas[1],bas[2],jacobian[1]);
        calculate_(bas[0],bas[1],dbas[2],jacobian[2]);

#ifdef SPM_DEBUG
        //for(int z = 0;z < BDim[2];++z)
        int z = 0;
        {
            std::cout << " z=" << z << std::endl;
            std::copy(.begin()+BDim[0]*BDim[1]*z,
                      .begin()+BDim[0]*BDim[1]*(z+1),
                      std::ostream_iterator<double>(std::cout," "));
            std::cout << std::endl;
        }

#endif
        */
        // output 1-based affine transformation
        std::fill(trans_to_mni,trans_to_mni+16,0.0);
        trans_to_mni[0] = scale[0];
        trans_to_mni[5] = scale[1];
        trans_to_mni[10] = scale[2];
        trans_to_mni[3] = bounding_box_lower[0]-scale[0];
        trans_to_mni[7] = bounding_box_lower[1]-scale[1];
        trans_to_mni[11] = bounding_box_lower[2]-scale[2];


    }

    template<typename image_type1,typename image_type2>
    void warp_image(const image_type1& I,image_type2& out,double voxel_size = 1.0)
    {
        set_voxel_size(voxel_size);
        out.resize(BDim);
        for(image::pixel_index<3> index;BDim.is_valid(index);index.next(BDim))
        {
            image::vector<3,double> pos;
            warp_coordinate(index,pos);
            /*
            pos += 0.5;
            pos[0] = std::floor(pos[0]);
            pos[1] = std::floor(pos[1]);
            pos[2] = std::floor(pos[2]);
            if(I.geometry().is_valid(pos))
                out[index.index()] = I.at(pos[0],pos[1],pos[2]);
            else
                out[index.index()] = 0;
            */
            image::interpolation<image::linear_weighting,3> trilinear_interpolation;
            trilinear_interpolation.estimate(I,pos,out[index.index()]);
        }
    }

    void calculate_(const std::vector<double>& bx,
                         const std::vector<double>& by,
                         const std::vector<double>& bz,
                         std::vector<double>& result)
    {
        {
        std::vector<double> temp,temp2;
        // ([T, z, y ], x )*(x,nx) = ([T,z,y],nx)
        temp.resize(3*7*9*VG.width());
        math::matrix_product(T.begin(),bx.begin(),temp.begin(),
                             math::dim<3*7*9,7>(),math::dyndim(7,bx.size()/7));

        // ([T, z,y],nx)' =  (nx , [T, z,y])
        temp2.resize(temp.size());
        math::matrix_transpose(temp.begin(),temp2.begin(),
                               math::dyndim(3*7*9,VG.width()));

        // ([nx ,T, z],y)*(y,ny) = (nx,T,z,ny)
        temp.resize(VG.width()*3*7*VG.height());
        math::matrix_product(temp2.begin(),by.begin(),temp.begin(),
                             math::dyndim(VG.width()*3*7,9),math::dyndim(9,by.size()/9));


        // ([nx,T,z],ny)' =  (ny , [nx,T, z] )
        temp2.resize(temp.size());
        math::matrix_transpose(temp.begin(),temp2.begin(),
                               math::dyndim(VG.width()*3*7,VG.height()));


        // ([ny ,nx,T] z] )*(z,nz) = (ny,nx,T,nz)
        temp.resize(VG.size()*3);
        math::matrix_product(temp2.begin(),bz.begin(),temp.begin(),
                             math::dyndim(VG.height()*VG.width()*3,7),math::dyndim(7,bz.size()/7));

        // ([ny,nx, T ],nz)' =  (nz,ny,nx,T)
        temp2.resize(temp.size());
        math::matrix_transpose(temp.begin(),temp2.begin(),
                               math::dyndim(VG.height()*VG.width()*3,VG.depth()));

        // ([nz,ny,nx],T)' = (T,[nz,ny,nx])
        math::matrix_transpose(temp2.begin(),temp.begin(),
                               math::dyndim(VG.height()*VG.width()*VG.depth(),3));
        result.swap(temp);
        }
    }
    template<typename I_type>
    image::vector<image_type::dimension,double> center_of_mass(const I_type& Im)
    {
        image::basic_image<unsigned char,I_type::dimension> mask;
        image::segmentation::otsu(Im,mask);
        image::morphology::smoothing(mask);
        image::morphology::smoothing(mask);
        image::morphology::defragment(mask);
        image::vector<I_type::dimension,double> sum_mass;
        double total_w = 0.0;
        for(image::pixel_index<I_type::dimension> index;
            mask.geometry().is_valid(index);
            index.next(mask.geometry()))
            if(mask[index.index()])
            {
                total_w += 1.0;
                image::vector<3,double> pos(index);
                sum_mass += pos;
            }
        sum_mass /= total_w;
        for(unsigned char dim = 0;dim < I_type::dimension;++dim)
            sum_mass[dim] -= (double)Im.geometry()[dim]/2.0;
        return sum_mass;
    }

    template<typename I_type>
    image::vector<3,double> orientation(const I_type& Im)
    {
        image::basic_image<unsigned char,I_type::dimension> mask;
        image::segmentation::otsu(Im,mask);
        image::morphology::smoothing(mask);
        image::morphology::smoothing(mask);
        image::morphology::defragment(mask);
        double T[9];
        double total_w = 0.0;
        image::vector<3,double> center(Im.geometry());
        center /= 2.0;
        for(image::pixel_index<I_type::dimension> index;
            mask.geometry().is_valid(index);
            index.next(mask.geometry()))
            if(mask[index.index()])
            {
                total_w += 1.0;
                image::vector<3,double> pos(index);
                pos -= center;
                T[0] += pos[0]*pos[0];
                T[1] += pos[1]*pos[0];
                T[2] += pos[2]*pos[0];
                T[4] += pos[1]*pos[1];
                T[5] += pos[2]*pos[1];
                T[8] += pos[2]*pos[2];
            }
        T[3] = T[1];
        T[6] = T[2];
        T[7] = T[5];
        image::divide_constant(T,T+9,total_w);
        double V[9],d[3];
        math::matrix_eigen_decomposition_sym(T,V,d,math::dim<3,3>());
        std::cout << V[0] << " " << V[1] << " " << V[2] << std::endl;
        std::cout << V[3] << " " << V[4] << " " << V[5] << std::endl;
        std::cout << V[6] << " " << V[7] << " " << V[8] << std::endl;
        std::cout << d[0] << " " << d[1] << " " << d[2] << std::endl;
        image::vector<3,double> dir(V[0],V[3],V[6]);
        if(dir[0] + dir[1]+dir[2] < 0)
            dir = -dir;
        dir *= d[0];
        return dir;
    }
    void show_trans(void)
    {
        std::cout << "tran:" << arg_min.translocation[0] << ","
                             << arg_min.translocation[1] << ","
                             << arg_min.translocation[2] << std::endl;
        std::cout << "scale:"<< arg_min.scaling[0] << ","
                             << arg_min.scaling[1] << ","
                             << arg_min.scaling[2] << std::endl;
        std::cout << "rotate:"<< arg_min.rotation[0] << ","
                             << arg_min.rotation[1] << ","
                             << arg_min.rotation[2] << std::endl;
        std::cout << "affine:"<< arg_min.affine[0] << ","
                             << arg_min.affine[1] << ","
                             << arg_min.affine[2] << std::endl;
        std::cout << "-----" << std::endl;
    }
    template<typename trans_type>
    void save_trans_image(const char* file_name,const trans_type& trans)
    {
        image::transformation_matrix<dim,double> affine_buf(trans);
        image::reg::linear_get_trans(VF.geometry(),VG.geometry(),affine_buf);
        image::basic_image<float,3> VGG(VF.geometry());
        image::resample(VG,VGG,affine_buf);
        image::normalize(VGG,1.0);
        image::io::nifti out;
        out << VGG;
        out.save_to_file(file_name);
    }

    void normalize(void)
    {
        // VG: FA TEMPLATE
        // VF: SUBJECT QA
        arg_min.scaling[0] = std::fabs(VFvs[0]) / std::fabs(VGvs[0]);
        arg_min.scaling[1] = std::fabs(VFvs[1]) / std::fabs(VGvs[1]);
        arg_min.scaling[2] = std::fabs(VFvs[2]) / std::fabs(VGvs[2]);
        // calculate center of mass
        image::vector<3,double> mF = center_of_mass(VF);
        image::vector<3,double> mG = center_of_mass(VG);

        std::cout << "center of VF:" << mF << std::endl;
        std::cout << "center of VG:" << mG << std::endl;
        arg_min.translocation[0] = mG[0]-mF[0]*arg_min.scaling[0];
        arg_min.translocation[1] = mG[1]-mF[1]*arg_min.scaling[1];
        arg_min.translocation[2] = mG[2]-mF[2]*arg_min.scaling[2];
        /*
        std::cout << "VF:" << std::endl;
        image::vector<3,double> vF = orientation(VF);
        std::cout << "VG:" << std::endl;
        image::vector<3,double> vG = orientation(VG);

        vF[0] = vG[0] = 0.0;
        vF.normalize();
        vG.normalize();
        float angle = std::acos(std::fabs(vF*vG));

        arg_min.rotation[0] = vF[2] > vG[2] ? angle:-angle;
        arg_min.rotation[1] = 0;
        arg_min.rotation[2] = 0;*/
//        float cos_y = std::sqrt(1.0-R[6]*R[6]);
//        arg_min.rotation[0] = R[0]*R[0];
//        arg_min.rotation[1] = std::acos(cos_y);

        bool terminated = false;
        set_title("linear registration (may take a long time)");
        #ifdef SPM_DEBUG
        // test only
        {
            double test_affine[12] =
            {
                -0.410251717717324,	0.00119023084447449,	-0.00802947515498428,	86.1336019277567,
                -0.00803748777711841,	0.426488840343909,	-0.0774655251253437,	2.52904172670624,
                -0.00917363652150580,	0.0815858273926369,	0.397717341375597,	-16.7806866716997
            };
            std::copy(test_affine,test_affine+12,affine);
        }
        #else

        begin_prog("conducting registration");
        check_prog(0,2);
        image::reg::linear(VF,VG,arg_min,image::reg::affine,image::reg::square_error(),terminated,0.25);
        check_prog(2,2);

        /* for debugging
        image::io::nifti out;
        out << VF;
        out.save_to_file("t0.nii");
        show_trans();
        save_trans_image("t2.nii",arg_min);
        */

        set_affine(arg_min);
        // need to change to 1-based affine (SPM style)
        // spm_affine = [1 0 0 1                   [1 0 0 -1
        //               0 1 0 1                    0 1 0 -1
        //               0 0 1 1   * my_affine *    0 0 1 -1
        //               0 0 0 1]                    0 0 0 1]
        affine[3] -= std::accumulate(affine,affine+3,-1.0);
        affine[7] -= std::accumulate(affine+4,affine+4+3,-1.0);
        affine[11] -= std::accumulate(affine+8,affine+8+3,-1.0);
        std::cout << affine[0] << "," << affine[1] << "," << affine[2] << "," << affine[3] << std::endl;
        std::cout << affine[4] << "," << affine[5] << "," << affine[6] << "," << affine[7] << std::endl;
        std::cout << affine[8] << "," << affine[9] << "," << affine[10] << "," << affine[11] << std::endl;

        #endif


        std::vector<std::vector<double> > kxyz(3);
        int k[3] = {7,9,7};
        {
            std::fill(scale.begin(),scale.end(),1.0);
            stabilise = 8;
            std::vector<int> bb(3);
            initialize_basis_function(VG.geometry(),bb);
        }
        for(int dim = 0;dim < 3;++dim)
        {
            kxyz[dim].resize(k[dim]);
            for(int i = 0;i < kxyz[dim].size();++i)
            {
                kxyz[dim][i] = 3.14159265358979323846*(double)i/VG.geometry()[dim]/VGvs[dim];
                kxyz[dim][i] *= kxyz[dim][i];
            }
        }
        double s1 = 3*k[0]*k[1]*k[2];
        double s2 = s1 + 4;
        std::vector<double> IC0(s2);
        {
            int ICO_ = k[0]*k[1]*k[2];
            double IC0_co = reg*std::pow(stabilise,6);
            double vs4[3];
            vs4[0] = std::pow(VGvs[0],4);
            vs4[1] = std::pow(VGvs[1],4);
            vs4[2] = std::pow(VGvs[2],4);
            for(int i = 0,index = 0;i < k[2];++i)
                for(int j = 0;j < k[1];++j)
                    for(int m = 0;m < k[0];++m,++index)
                    {
                        IC0[index] = kxyz[2][i]*kxyz[2][i]+kxyz[1][j]*kxyz[1][j]+kxyz[0][m]*kxyz[0][m]+
                                     2*kxyz[0][m]*kxyz[1][j]+2*kxyz[0][m]*kxyz[2][i]+2*kxyz[1][j]*kxyz[2][i];
                        IC0[index] *= IC0_co;
                        IC0[index+ICO_] = IC0[index]*vs4[1];
                        IC0[index+ICO_+ICO_] = IC0[index]*vs4[2];
                        IC0[index]*= vs4[0];
                    }
#ifdef SPM_DEBUG
    std::cout << "IC0" << std::endl;
    std::copy(IC0.begin(),IC0.end(),std::ostream_iterator<double>(std::cout," "));
    std::cout << std::endl;
#endif
        }

        T.resize(s2);
        std::fill(T.begin(),T.end(),0);
        T[s1] = 1;
        std::vector<double> alpha,beta;
        double var,fw,pvar = std::numeric_limits<double>::max();

        set_title("spatial normalization");
        for(int iter = 0;check_prog(iter,16);++iter)
        {
            mrqcof(VG,VF,VGvs,VFvs,T,bas,dbas,affine,fwhm[0],fwhm[1],alpha,beta,var,fw);

            {
                // beta = beta + alpha*T;
                std::vector<double> alphaT(T.size());
                math::matrix_vector_product(alpha.begin(),T.begin(),alphaT.begin(),math::dyndim(s2,s2));
                image::add(beta.begin(),beta.end(),alphaT.begin());
            }

            //Alpha + IC0*scal
            if(var > pvar)
            {
                double scal = pvar/var;
                var = pvar;
                for(int i = 0,j = 0;i < alpha.size();i+=T.size()+1,++j)
                    alpha[i] += IC0[j]*scal;
            }
            else
                for(int i = 0,j = 0;i < alpha.size();i+=T.size()+1,++j)
                    alpha[i] += IC0[j];

            // solve T = (Alpha + IC0*scal)\(Alpha*T + Beta);
            {
                std::vector<int> piv(T.size());
                math::matrix_lu_decomposition(&*alpha.begin(),&*piv.begin(),math::dyndim(s2,s2));
                math::matrix_lu_solve(&*alpha.begin(),&*piv.begin(),&*beta.begin(),&*T.begin(),math::dyndim(s2,s2));

            }
            fwhm[1] = std::min(fw,fwhm[1]);
            std::cout << " FWHM = " << fw << " Var = " << var <<std::endl;
#ifdef SPM_DEBUG
            std::cout << "iter=" << iter << std::endl;
            std::cout << "T=";
            std::copy(T.begin(),T.end(),std::ostream_iterator<double>(std::cout," "));
            std::cout << std::endl;
#endif


        }
        image::multiply_constant(T.begin(),T.end(),stabilise*stabilise*stabilise);
    }
};

#endif // NORMALIZATION_HPP
