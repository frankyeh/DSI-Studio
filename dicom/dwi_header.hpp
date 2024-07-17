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
    std::string file_name, report;
    tipl::image<3,unsigned short> image;
public:// for HCP dataset
    tipl::image<4> grad_dev;
public:
    tipl::vector<3,float> bvec;
    float bvalue,te;
    tipl::vector<3,float> voxel_size;
public:
    DwiHeader(void): bvalue(0.0f), te(0.0f) {}
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
    static bool output_src(const char* file_name, std::vector<std::shared_ptr<DwiHeader> >& dwi_files, int upsampling,bool sort_btable);
    static bool has_b_table(std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
};

#endif//DWI_HEADER_HPP
