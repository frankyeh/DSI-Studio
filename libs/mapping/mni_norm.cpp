#include "mni_norm.hpp"
#include "mat_file.hpp"
#include "math/matrix_op.hpp"
#include "boost/lambda/lambda.hpp"


const float pi = 3.14159265358979323846;
const int DCM[3] = {7,9,7};


MNINorm::MNINorm(void)
{
    setBoundingBox(-78,-112,-50,78,76,85,2.0);
}

void MNINorm::setBoundingBox(int fx,int fy,int fz,int tx,int ty,int tz,float voxel_size_)
{
    voxel_size = voxel_size_;
    bounding_box_lower[0] = std::floor(fx/voxel_size)*voxel_size;
    bounding_box_lower[1] = std::floor(fy/voxel_size)*voxel_size;
    bounding_box_lower[2] = std::floor(fz/voxel_size)*voxel_size;
    bounding_box_upper[0] = std::floor(tx/voxel_size)*voxel_size;
    bounding_box_upper[1] = std::floor(ty/voxel_size)*voxel_size;
    bounding_box_upper[2] = std::floor(tz/voxel_size)*voxel_size;
}



bool MNINorm::load_transformation_matrix(const char* file_name)
{
        MatFile reader;
        if(!reader.load_from_file(file_name))
		return false;
        unsigned int row,col;
        TrX.resize(7*9*7);
        TrY.resize(7*9*7);
        TrZ.resize(7*9*7);
        Affine.resize(16);



	{
		const float* buf;
		reader.get_matrix("TrX",row,col,buf);
		if(row*col != TrX.size())
			return false;
		std::copy(buf,buf+TrX.size(),TrX.begin());
	}
	{
		const float* buf;
		reader.get_matrix("TrY",row,col,buf);
		if(row*col != TrY.size())
			return false;
		std::copy(buf,buf+TrY.size(),TrY.begin());
	}
	{
		const float* buf;
		reader.get_matrix("TrZ",row,col,buf);
		if(row*col != TrZ.size())
			return false;
		std::copy(buf,buf+TrX.size(),TrZ.begin());
	}
	{
		const float* buf;
		reader.get_matrix("Affine",row,col,buf);
		if(row*col != Affine.size())
			return false;
		std::copy(buf,buf+Affine.size(),Affine.begin());
	}

        {
                //int MNIDim[3];// = {91,109,91};  // the original dimension of MNI image
                const float* buf;
                reader.get_matrix("MNIDim",row,col,buf);
                if(row*col != 3)
                        return false;
                std::copy(buf,buf+3,MNIDim);                
        }
        {
                const float* buf;
                reader.get_matrix("SubDim",row,col,buf);
                if(row*col != 3)
                        return false;
                std::copy(buf,buf+3,SubDim);
        }
        std::vector<float> mat(16);

        {
                //MNImat = [-2,0,0,92;0,2,0,-128;0,0,2,-74;0,0,0,1];
                const float* Vmat;
                reader.get_matrix("MNIMat",row,col,Vmat);
                if(row*col != 16)
                        return false;
                image::vector<3,float> bb1,bb2,vx;




                {
                    /*
                    function [bb,vx] = bbvox_from_V(V)
                    vx = sqrt(sum(V.mat(1:3,1:3).^2));
                    if det(V.mat(1:3,1:3))<0, vx(1) = -vx(1); end;

                    o  = V.mat\[0 0 0 1]';
                    o  = o(1:3)';
                    bb = [-vx.*(o-1) ; vx.*(V.dim(1:3)-o)];
                    return;
                    */
                    float o[3];
                    vx[0] = std::fabs(Vmat[0]);
                    vx[1] = std::fabs(Vmat[5]);
                    vx[2] = std::fabs(Vmat[10]);
                    if(Vmat[0]*Vmat[5]*Vmat[10] < 0)
                        vx[0] = -vx[0];
                    o[0] = -Vmat[12]/Vmat[0];
                    o[1] = -Vmat[13]/Vmat[5];
                    o[2] = -Vmat[14]/Vmat[10];
                    /*
                    bb1[0] = -vx[0]*(o[0]-1);
                    bb1[1] = -vx[1]*(o[1]-1);
                    bb1[2] = -vx[2]*(o[2]-1);
                    bb2[0] = vx[0]*(MNIDim[0]-o[0]);
                    bb2[1] = vx[1]*(MNIDim[1]-o[1]);
                    bb2[2] = vx[2]*(MNIDim[2]-o[2]);
                    */
                    //function [x,y,z,mat] = get_xyzmat(prm,bb,vox)
                    /*
                    xyz.resize(3);
                    for(int dim = 0;dim < 3;++dim)
                    {
                        xyz.clear();
                        for(float dis = bounding_box_lower[dim];dis < bounding_box_upper[dim]+0.001;dis+=voxel_size)
                        xyz[dim].push_back(dis/vx[dim]+o[dim]);
                    }
                    */

                    BDim[0] = (bounding_box_upper[0]-bounding_box_lower[0])/voxel_size+1;//79
                    BDim[1] = (bounding_box_upper[1]-bounding_box_lower[1])/voxel_size+1;//95
                    BDim[2] = (bounding_box_upper[2]-bounding_box_lower[2])/voxel_size+1;//69
                    BOffset[0] = bounding_box_lower[0]/vx[0]+o[0];
                    BOffset[1] = bounding_box_lower[1]/vx[1]+o[1];
                    BOffset[2] = bounding_box_lower[2]/vx[2]+o[2];
                    scale[0] = voxel_size/vx[0];
                    scale[1] = voxel_size/vx[1];
                    scale[2] = voxel_size/vx[2];
                    /*
                    std::vector<float> iM1(16),M2(16);
                    iM1[0] = 1.0/vx[0];
                    iM1[5] = 1.0/vx[1];
                    iM1[10] = 1.0/vx[2];
                    iM1[15] = 1.0;
                    iM1[3] = o[0];
                    iM1[7] = o[1];
                    iM1[11] = o[2];
                    M2[0] = voxel_size;
                    M2[5] = voxel_size;
                    M2[10] = voxel_size;
                    M2[15] = 1.0;
                    M2[3] = bounding_box_lower[0]-voxel_size;
                    M2[7] = bounding_box_lower[1]-voxel_size;
                    M2[11] = bounding_box_lower[2]-voxel_size;

                    math::matrix_product_transpose(Vmat,iM1.begin(),mat.begin(),math::dim<4,4>(),math::dim<4,4>());
                    math::matrix_product_transpose(mat.begin(),M2.begin(),iM1.begin(),math::dim<4,4>(),math::dim<4,4>());
                    math::matrix_transpose(iM1.begin(),mat.begin(),math::dim<4,4>());

                    std::cout << file_name << std::endl;
                    std::cout << file_name << std::endl;

                    if(mat[0]*mat[5]*mat[10] > 0)
                    {
                        mat[3] += mat[0]*(xyz[0].size()+1);
                        mat[0] = -mat[0];
                        std::vector<float> rx(xyz[0].rbegin(),xyz[0].rend());
                        xyz[0].swap(rx);
                    }
                    std::cout << BOffset[0] << " " << BOffset[1] << " " << BOffset[2] <<std::endl;
                    std::cout << scale[0] << " " << scale[1] << " " << scale[2] <<std::endl;
                    for(int i = 0;i < 16;++i)
                        std::cout << mat[i] << " ";
                    std::cout << std::endl;
                    for(int dim = 0;dim < 3;++dim)
                        for(int index = 0;index < xyz[dim].size();++index)
                            std::cout << xyz[dim][index] << " ";
                    std::cout << std::endl;
                    */


                }
        }


        inv_sqrt_2 = 1.0/std::sqrt(2.0);
        sqrt_2_mni_dim = std::sqrt(2.0/MNIDim[0])*std::sqrt(2.0/MNIDim[1])*std::sqrt(2.0/MNIDim[2]);
        pi_inv_mni_dim[0] = pi/(MNIDim[0]);
        pi_inv_mni_dim[1] = pi/(MNIDim[1]);
        pi_inv_mni_dim[2] = pi/(MNIDim[2]);
	return true;
}

void MNINorm::transform(image::vector<3,float> from,image::vector<3,float>& to)
{

        // from DICOM space to nifti space
        from[0] = BDim[0] - from[0] - 1.0;
        from[1] = BDim[1] - from[1] - 1.0;

        float bx[7],by[9],bz[7];
        float nx = BOffset[0] + from[0]*scale[0];
        float nx05 = nx-0.5;
        bx[0] = inv_sqrt_2;
        for(unsigned int k = 1;k < 7;++k)
            bx[k] = std::cos(pi_inv_mni_dim[0]*(nx05)*((float)k));
        float ny = BOffset[1] + from[1]*scale[1];
        float ny05 = ny-0.5;
        by[0] = inv_sqrt_2;
        for(unsigned int k = 1;k < 9;++k)
            by[k] = std::cos(pi_inv_mni_dim[1]*(ny05)*((float)k));

        float nz = BOffset[2] + from[2]*scale[2];
        float nz05 = nz-0.5;
        bz[0] = inv_sqrt_2;
        for(unsigned int k = 1;k < 7;++k)
            bz[k] = std::cos(pi_inv_mni_dim[2]*(nz05)*((float)k));

	float temp[63];
	float temp2[7];
	float tt[4],result[4];
	math::matrix_product(TrX.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
	math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
	tt[0] = nx + sqrt_2_mni_dim*math::vector_op_dot(bz,bz+7,temp2);

	math::matrix_product(TrY.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
	math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
	tt[1] = ny + sqrt_2_mni_dim*math::vector_op_dot(bz,bz+7,temp2);

	math::matrix_product(TrZ.begin(),bx,temp,math::dim<63,7>(),math::dim<7,1>());
	math::matrix_product(temp,by,temp2,math::dim<7,9>(),math::dim<9,1>());
	tt[2] = nz + sqrt_2_mni_dim*math::vector_op_dot(bz,bz+7,temp2);

        tt[3] = 1.0;
        math::matrix_product(tt,Affine.begin(),result,math::dim<1,4>(),math::dim<4,4>());
        to[0] = result[0]-1.0; // transform back to our 0 0 0 coordinate
        to[1] = result[1]-1.0;
        to[2] = result[2]-1.0;

        // transformed from the nifti xy flipped space

        to[0] = SubDim[0]-to[0]-1;
        to[1] = SubDim[1]-to[1]-1;

}


void MNINorm::warp(const image::basic_image<float,3>& I,image::basic_image<float,3>& out)
{
    out.resize(image::geometry<3>(BDim));
    for(image::pixel_index<3> index;
        out.geometry().is_valid(index);
        index.next(out.geometry()))
    {
        image::vector<3,float> from(index),to;
        transform(from,to);
        /*
        to += 0.5;
        to[0] = std::floor(to[0]);
        to[1] = std::floor(to[1]);
        to[2] = std::floor(to[2]);
        if(I.geometry().is_valid(to))
            out[index.index()] = I.at(to[0],to[1],to[2]);
        else
            out[index.index()] = 0;
        */
        image::interpolation<image::linear_weighting,3> trilinear_interpolation;
        trilinear_interpolation.estimate(I,to,out[index.index()]);
    }
}
