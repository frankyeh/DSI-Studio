#ifndef DWI_HEADER_HPP
#define DWI_HEADER_HPP
#include <vector>
#include <string>
#include "zlib.h"
#include "TIPL/tipl.hpp"


class DwiHeader
{
	typedef std::vector<short>::iterator image_iterator;
public:
    std::string file_name,report,error_msg;
    tipl::image<3,unsigned short> image;
public:// for HCP dataset
    tipl::image<4> grad_dev;
public:
    tipl::vector<3,float> bvec;
    float bvalue = 0.0f;
    float te = 0.0f;
    float slice_location = 0.0f;
    tipl::vector<3,float> voxel_size;
    tipl::matrix<4,4,float> trans_to_mni = tipl::identity_matrix();
public:
    bool open(const char* filename);
public:
    const unsigned short* begin(void) const
    {
        return &*image.begin();
    }
    unsigned short* begin(void)
    {
        return &*image.begin();
    }
    unsigned short operator[](unsigned int index) const
    {
        return image[index];
    }
    unsigned short& operator[](unsigned int index)
    {
        return image[index];
    }
    size_t size(void) const
    {
        return image.size();
    }
	void swap(DwiHeader& rhs)
	{
        image.swap(rhs.image);
        std::swap(bvec, rhs.bvec);
        std::swap(bvalue, rhs.bvalue);
        std::swap(te, rhs.te);
	}

public:
	bool operator<(const DwiHeader& rhs) const
	{
        if(int(bvalue) != int(rhs.bvalue))
			return bvalue < rhs.bvalue;
        return bvec < rhs.bvec;
	}

public:
    static bool has_b_table(std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
};

#endif//DWI_HEADER_HPP
