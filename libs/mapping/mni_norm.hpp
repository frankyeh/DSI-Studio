#ifndef MNI_NORM_HPP
#define MNI_NORM_HPP
#include <vector>
#include <cmath>
#include <image/image.hpp>
#include <math/matrix_op.hpp>
class MNINorm{
private:
        std::vector<float> TrX,TrY,TrZ;
	std::vector<float> Affine;
        int bounding_box_lower[3];
        int bounding_box_upper[3];
        int MNIDim[3];// = {91,109,91};  // the original dimension of MNI image
        int SubDim[3];
        int BDim[3];// = {79,95,69};	    // the dimension of normalized images
        int BOffset[3];// = {6,7,11};	// the offset due to bounding box
        float voxel_size;
        float scale[3];

private:
	float sqrt_2_mni_dim;
	float inv_sqrt_2;
	float pi_inv_mni_dim[3];

public:
	
	MNINorm(void);
        void setBoundingBox(int fx,int fy,int fz,int tx,int ty,int tz,float voxel_size_);
	bool load_transformation_matrix(const char* file_name);
        void transform(image::vector<3,float> from,image::vector<3,float>& to);
        void warp(const image::basic_image<float,3>& I,image::basic_image<float,3>& out);
};




#endif//MNI_NORM_HPP
