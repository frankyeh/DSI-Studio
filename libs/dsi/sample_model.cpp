//---------------------------------------------------------------------------
#include <cmath>
#include <vector>
#include "space_mapping.hpp"
#include "libs/dsi/dsi_process.hpp"

//---------------------------------------------------------------------------
SamplePoint::SamplePoint(unsigned int odf_index_,
                         float x,float y,float z,float weighting)
    :ratio(8),sample_index(8),odf_index(odf_index_)
{
    float fx = std::floor(x);
    float fy = std::floor(y);
    float fz = std::floor(z);
    float px = x-fx;
    float py = y-fy;
    float pz = z-fz;
    int ix = int(fx);
    int iy = int(fy);
    int iz = int(fz);
    sample_index[0] = SpaceMapping<dsi_range>::getIndex(ix,iy,iz);
    ratio[0] = float((1.0f-px)*(1.0f-py)*(1.0f-pz));
    sample_index[1] = SpaceMapping<dsi_range>::getIndex(ix+1,iy,iz);
    ratio[1] = float(px*(1.0f-py)*(1.0f-pz));
    sample_index[2] = SpaceMapping<dsi_range>::getIndex(ix,iy+1,iz);
    ratio[2] = float((1.0f-px)*py*(1.0f-pz));
    sample_index[3] = SpaceMapping<dsi_range>::getIndex(ix+1,iy+1,iz);
    ratio[3] = float(px*py*(1.0f-pz));
    sample_index[4] = SpaceMapping<dsi_range>::getIndex(ix,iy,iz+1);
    ratio[4] = float((1.0f-px)*(1.0f-py)*pz);
    sample_index[5] = SpaceMapping<dsi_range>::getIndex(ix+1,iy,iz+1);
    ratio[5] = float(px*(1.0f-py)*pz);
    sample_index[6] = SpaceMapping<dsi_range>::getIndex(ix,iy+1,iz+1);
    ratio[6] = float((1.0f-px)*py*pz);
    sample_index[7] = SpaceMapping<dsi_range>::getIndex(ix+1,iy+1,iz+1);
    ratio[7] = float(px*py*pz);
    for (unsigned int index = 0; index < 8; ++index)
        ratio[index] *= weighting;
}
void SamplePoint::sampleODFValueWeighted(const std::vector<float>& pdf,std::vector<float>& odf) const
{
    float value = 0.0;
    for (unsigned int index = 0; index < 8; ++index)
        value += pdf[sample_index[index]]*ratio[index];
    odf[odf_index] += value;
}
