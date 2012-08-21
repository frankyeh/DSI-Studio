//---------------------------------------------------------------------------
#ifndef space_mappingH
#define space_mappingH
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
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
