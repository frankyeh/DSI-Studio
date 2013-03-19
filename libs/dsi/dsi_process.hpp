#ifndef DSI_PROCESS_HPP
#define DSI_PROCESS_HPP
#define _USE_MATH_DEFINES
#include <vector>
#include "basic_process.hpp"
#include "basic_voxel.hpp"


template<typename GeometryType,typename FloatType>
void ifourn(const GeometryType& nn,std::vector<FloatType>& data)
{
        unsigned int ntot = data.size() >> 1;
        unsigned int nprev = 1;
        unsigned int nrem;

    unsigned int ip1,ip2,ip3,i2rev,i3rev,ibit,n;
        for(int idim = nn.dimension-1;idim >= 0;--idim)
                {
        n = nn[idim];
                nrem = ntot/(n* nprev);
                ip1 = nprev << 1;
                ip2 = ip1*n;
                ip3 = ntot << 1;
                i2rev = 0;
                for(unsigned int i2 = 0; i2 < ip2; i2 += ip1)
                        {
                        if(i2 < i2rev)
                                {
                                for(unsigned int i1 = i2;i1<i2+ip1-1;i1 += 2)
                                        for(unsigned int i3=i1;i3 < ip3;i3 += ip2)
                                                {
                                                i3rev = i2rev + i3 - i2;
                                                std::swap(data[i3],data[i3rev]);
                                                std::swap(data[i3+1],data[i3rev+1]);
                        }
                                }
                        ibit = ip2 >> 1;
                        while(ibit >= ip1 && i2rev+1 > ibit)
                                {
                                i2rev -= ibit;
                                ibit >>= 1;
                                }
                        i2rev += ibit;
                        }
        unsigned int ifp1=ip1;
        unsigned int ifp2,k1,k2;
        FloatType theta,wtemp,wpr,wpi,wr,wi;
                FloatType tempr,tempi;
        while(ifp1 < ip2)
                {
                ifp2 = ifp1 << 1;
                theta = (FloatType)-6.28318530/(ifp2/ip1);
                wtemp = (FloatType)std::sin(0.5*theta);
                wpr = (FloatType)(-2.0*wtemp*wtemp);
                wpi = std::sin(theta);
                wr=1.0;
                wi=0.0;
                for(unsigned int i3 = 0;i3 < ifp1;i3 += ip1)
                        {
                        for(unsigned int i1 = i3;i1 < i3 + ip1 - 1;i1 += 2)
                                for(unsigned int i2 = i1;i2 < ip3;i2 += ifp2)
                                {
                                k1 = i2;
                                k2 = k1+ifp1;
                                tempr = data[k2]*wr-data[k2+1]*wi;
                                tempi = data[k2+1]*wr + data[k2]*wi;
                                data[k2] = data[k1] - tempr;
                                data[k2+1] = data[k1+1] - tempi;
                                data[k1] += tempr;
                                data[k1+1] += tempi;
                                }
                        wr=(wtemp=wr)*wpr-wi*wpi+wr;
                        wi=wi*wpr+wtemp*wpi+wi;
                        }
                ifp1 = ifp2;
                    }
            nprev *= n;
        }

}
class QSpace2Pdf  : public BaseProcess
{
    std::vector<unsigned int> qspace_mapping1;
    std::vector<unsigned int> qspace_mapping2;
    std::vector<float> hanning_filter;
public:
    static double get_min_b(const Voxel& voxel)
    {
        float b_min;
        {
            std::vector<float> bvalues(voxel.bvalues.begin(),voxel.bvalues.end());
            do
            {
                b_min = *std::min_element(bvalues.begin(),bvalues.end());
                if (b_min != 0)
                    return b_min;
                bvalues.erase(std::min_element(bvalues.begin(),bvalues.end()));
            }
            while (!bvalues.empty());
        }
        return 0;
    }
    static void get_q_table(const Voxel& voxel,std::vector<image::vector<3,int> >& q_table)
    {
        float b_min = get_min_b(voxel);
        unsigned int n = voxel.bvalues.size();

        for (unsigned int index = 0; index < n; ++index)
        {
            int q2 = std::floor(std::abs(voxel.bvalues[index]/b_min)+0.5);
            image::vector<3,float> bvec = voxel.bvectors[index];
            bvec.normalize();
            bvec *= std::sqrt(std::abs(voxel.bvalues[index]/b_min));
            bvec[0] = std::floor(bvec[0]+0.5);
            bvec[1] = std::floor(bvec[1]+0.5);
            bvec[2] = std::floor(bvec[2]+0.5);
            q_table.push_back(image::vector<3,int>(bvec[0],bvec[1],bvec[2]));
        }

    }
public:
    virtual void init(Voxel& voxel)
    {
        unsigned int n = voxel.bvalues.size();
        qspace_mapping1.resize(n);
        qspace_mapping2.resize(n);
        hanning_filter.resize(n);
        std::vector<image::vector<3,int> > q_table;
        get_q_table(voxel,q_table);


        float filter_width = voxel.param[0];
        for (unsigned int index = 0; index < n; ++index)
        {
            int x = q_table[index][0];
            int y = q_table[index][1];
            int z = q_table[index][2];

            qspace_mapping1[index] = SpaceMapping<dsi_range>::getIndex(x,y,z);
            qspace_mapping2[index] = SpaceMapping<dsi_range>::getIndex(-x,-y,-z);
            float r = (float)std::sqrt((float)(x*x+y*y+z*z));
            hanning_filter[index] = 0.5 * (1.0+std::cos(2.0*r*M_PI/((float)filter_width)));
        }

    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> pdf(qspace_size);
        for (unsigned int index = 0; index < qspace_mapping1.size(); ++index)
        {
            float value = data.space[index]*hanning_filter[index];
            pdf[qspace_mapping1[index]] += value;
            pdf[qspace_mapping2[index]] += value;
        }
        data.space = pdf;
        // perform iFT
        std::vector<float> buffer(qspace_size << 1);
        for(unsigned int index = 0; index < qspace_size; ++index)
            buffer[index << 1] = pdf[index];
        ifourn(image::geometry<3>(space_length,space_length,space_length),buffer);
        data.space.resize(qspace_size);
        for(unsigned int index = 0; index < qspace_size; ++index)
            data.space[index] = std::abs(buffer[index << 1]);
    }

};

/** Perform integration over r
 */
struct Pdf2Odf : public BaseProcess
{
    std::vector<SamplePoint> sample_group;
    unsigned int b0_index;
public:
    virtual void init(Voxel& voxel)
    {
        // initialize dsi sample points
        unsigned int odf_size = voxel.ti.half_vertices_count;
        for (unsigned int index = 0; index < odf_size; ++index)
            for (float r = odf_min_radius; r <= odf_max_radius; r += odf_sampling_interval)
                sample_group.push_back(
                    SamplePoint(index,voxel.ti.vertices[index][0]*r,
                                voxel.ti.vertices[index][1]*r,
                                voxel.ti.vertices[index][2]*r,r*r));
        b0_index = SpaceMapping<dsi_range>::getIndex(0,0,0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        // calculate sum(p(u*r)*r^2)dr
        using namespace boost::lambda;
        std::fill(data.odf.begin(),data.odf.end(),0.0f);
        std::for_each(sample_group.begin(),sample_group.end(),
                      bind(&SamplePoint::sampleODFValueWeighted,
                           boost::lambda::_1,
                           boost::ref(data.space),
                           boost::ref(data.odf)));

        // normalization
        float sum = Accumulator()(data.odf);
        if (sum != 0.0)
            std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 *= (data.space[b0_index]/sum));
    }
};


struct CheckDSI
{
public:
    static bool check(const Voxel& voxel)
    {
        unsigned int n = voxel.bvalues.size();
        float b_min = QSpace2Pdf::get_min_b(voxel);
        if(b_min == 0)
            return false;

        return true;
        int sum_x = 0;
        int sum_y = 0;
        int sum_z = 0;

        std::vector<image::vector<3,int> > q_table;
        QSpace2Pdf::get_q_table(voxel,q_table);

        for(unsigned int index = 0; index < q_table.size(); ++index)
        {
            sum_x += std::abs(q_table[index][0]);
            sum_y += std::abs(q_table[index][1]);
            sum_z += std::abs(q_table[index][2]);
        }

        return sum_x == sum_y && sum_y == sum_z;
    }
};
#endif//DSI_PROCESS_HPP
