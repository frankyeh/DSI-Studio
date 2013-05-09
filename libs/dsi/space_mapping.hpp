//---------------------------------------------------------------------------
#ifndef space_mappingH
#define space_mappingH
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

const int dsi_range = 4;
const unsigned int qcode_count = 203;
const int display_range = 3;
const int min_display = -(1 << (display_range-1));
const int max_display = (1 << (display_range-1))-1;
const double odf_min_radius = 2.1;
const double odf_max_radius = 6.0;
const double odf_sampling_interval = 0.2;

const unsigned int space_length = 1 << dsi_range;             //16
const unsigned int space_half_length = 1 << (dsi_range-1);    //8
const int space_min_offset = -(1 << (dsi_range-1));                //-8
const int space_max_offset = space_half_length-1;                  //7
const unsigned int qspace_size = 1 << (dsi_range * 3);         //4096
//---------------------------------------------------------------------------
template <int offset_shift>
class SpaceMapping
{
public:
    static bool validateDSI(int dx,int dy,int dz)
    {
        return
            dx >= space_min_offset && dx <= space_max_offset &&
            dy >= space_min_offset && dy <= space_max_offset &&
            dz >= space_min_offset && dz <= space_max_offset;

    }
private:
    /*
    0 1 2 3 ....MAX-1 -MAX .... -3 -2 -1
    以MAX為8為例
    0 1 2 3 4 5 6 7 -8 -7 -6 -5 -4 -3 -2 -1
    要對映成
    0 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15
    */
    static int getIndex(int offset)
    {
        return (offset >= 0) ? offset:space_length+offset;
    }
    static int getOffset(int index)
    {
        return (index >= space_half_length) ? index-space_length:index;
    }
public:
    static unsigned int getIndex(int x,int y,int z)
    {
        return (getIndex(z) << (dsi_range+dsi_range)) +
               (getIndex(y) << dsi_range)  + getIndex(x);
    }
};


#endif
