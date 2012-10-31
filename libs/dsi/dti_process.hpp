#ifndef _PROCESS_HPP
#define _PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "math/matrix_op.hpp"


struct CheckDTI
{
public:
    template<typename ParamType>
    static bool check(const ParamType& param)
    {
        if(param.bvalues.size() < 7 || param.bvalues.front() != 0)
            return false;
        return true;
    }
};

struct ADCProfile: public BaseProcess
{
    std::vector<float> S;
    unsigned int b0_index;
public:
    virtual void init(Voxel& voxel)
    {
        S.resize(voxel.q_count);
        std::fill(S.begin(),S.end(),0.0);
        b0_index = std::min_element(voxel.bvalues.begin(),voxel.bvalues.end())-voxel.bvalues.begin();
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if (data.space[b0_index] == 0.0)
        {
            std::fill(data.space.begin(),data.space.end(),(float)1.0);
            return;
        }
        float logs0 = std::log(data.space[b0_index]);
        for (unsigned int i = 0; i < data.space.size(); ++i)
        {
            if(i == b0_index)
                continue;
            if(data.space[i] <= 0.0)
                data.space[i] = S[i];
            else
            {
                float value = std::log(data.space[i]);
                if (value < logs0)
                    data.space[i] = S[i] = logs0-value;
                else
                    data.space[i] = S[i];
            }
        }
    }
};


class Dwi2Tensor : public BaseProcess
{
private:
    //math::dynamic_matrix<float> iKtKKt;
    std::vector<float> iKtK; // 6-by-6
    std::vector<unsigned int> iKtK_pivot;
    std::vector<float> Kt;
    unsigned int b_count;
    unsigned int b0_index;
public:
    virtual void init(Voxel& voxel)
    {
        b_count = voxel.q_count-1;
        b0_index = std::min_element(voxel.bvalues.begin(),voxel.bvalues.end())-voxel.bvalues.begin();
        std::vector<image::vector<3,float> > b_data;
        b_data = voxel.bvectors;
        for(unsigned int index = 0; index < b_data.size(); ++index)
            b_data[index] *= std::sqrt(voxel.bvalues[index]);

        //skip null
        b_data.erase(b_data.begin()+b0_index);

        Kt.resize(6*b_count);
        {
            unsigned int qmap[6]		= {0  ,4  ,8  ,1  ,2  ,5  };
            float qweighting[6]= {1.0,1.0,1.0,2.0,2.0,2.0};
            //					  bxx,byy,bzz,bxy,bxz,byz
            for (unsigned int i = 0,k = 0; i < b_data.size(); ++i,k+=6)
            {
                //qq = q qT
                std::vector<float> qq(3*3);
                math::matrix_product_transpose(b_data[i].begin(),b_data[i].begin(),qq.begin(),
                                               math::dyndim(3,1),math::dyndim(3,1));

                /*
                	  q11 q15 q19 2*q12 2*q13 2*q16
                      q21 q25 q29 2*q22 2*q23 2*q26
                K  = | ...                         |
                */
                for (unsigned int col = 0,index = i; col < 6; ++col,index+=b_count)
                    Kt[index] = qq[qmap[col]]*qweighting[col];
            }
        }
        iKtK.resize(6*6);
        iKtK_pivot.resize(6);
        math::matrix_product_transpose(Kt.begin(),Kt.begin(),iKtK.begin(),
                                       math::dyndim(6,b_count),math::dyndim(6,b_count));
        math::matrix_lu_decomposition(iKtK.begin(),iKtK_pivot.begin(),math::dyndim(6,6));
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        //  Kt S = Kt K D
        float KtS[6];
        std::vector<float> signal(data.space);
        signal.erase(signal.begin()+b0_index);
        math::matrix_product(Kt.begin(),signal.begin(),KtS,math::dyndim(6,b_count),math::dyndim(b_count,1));
        math::matrix_lu_solve(iKtK.begin(),iKtK_pivot.begin(),KtS,data.space.begin(),math::dyndim(6,6));
    }
};

class TensorEigenAnalysis : public BaseProcess
{
    std::vector<float> d0;
    std::vector<float> d1;
    std::vector<float> d2;
    std::vector<float> md;
    std::vector<float> fa;
    std::vector<float> fdir;

    float get_fa(float l1,float l2,float l3)
    {
        float ll = (l1+l2+l3)/3.0;
        if (ll == 0.0)
            return 0.0;
        float ll1 = l1-ll;
        float ll2 = l2-ll;
        float ll3 = l3-ll;
        return std::sqrt(1.5*(ll1*ll1+ll2*ll2+ll3*ll3)/(l1*l1+l2*l2+l3*l3));

    }
    float get_odf_value(const image::vector<3,float>& vec,float* V,float* d) const
    {
        float sum = 0.0;
        for (unsigned int index = 0; index < 3; ++index)
        {
            float value = vec[0]*V[index]+vec[1]*V[index+3]+vec[2]*V[index+6];
            sum += value*value*d[index];
        }
        return sum;
    }
public:
    virtual void init(Voxel& voxel)
    {
        fa.clear();
        fa.resize(voxel.total_size);
        fdir.clear();
        fdir.resize(voxel.total_size*3);
        md.clear();
        md.resize(voxel.total_size);
        d0.clear();
        d0.resize(voxel.total_size);
        d1.clear();
        d1.resize(voxel.total_size);
        d2.clear();
        d2.resize(voxel.total_size);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        float tensor[9];
        float V[9],d[3];

        unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
        for (unsigned int index = 0; index < 9; ++index)
            tensor[index] = data.space[tensor_index[index]];

        math::matrix_eigen_decomposition_sym(tensor,V,d,math::dim<3,3>());
        if (d[1] < 0.0)
            d[1] = -d[1];
        if (d[2] < 0.0)
            d[2] = -d[2];
        if (d[0] < 0.0)
        {
            d[0] = 0.0;
            d[1] = 0.0;
            d[2] = 0.0;
        }
		std::copy(V,V+3,fdir.begin() + data.voxel_index * 3);
        fa[data.voxel_index] = get_fa(d[0],d[1],d[2]);
        md[data.voxel_index] = (d[0]+d[1]+d[2])/3.0;
        d0[data.voxel_index] = d[0];
        d1[data.voxel_index] = d[1];
        d2[data.voxel_index] = d[2];

        //if(!voxel.need_odf)
        //	return;
        //for (unsigned int index = 0;index < data.odf.size();++index)
        //    data.odf[index] = get_odf_value(voxel.ti.vertices[index],V,d);
        //float sum = Accumulator()(data.odf);
        //if (sum == 0.0)
        //    return;
        //std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 /= sum);
    }
    virtual void end(Voxel& voxel,MatFile& mat_writer)
    {
        set_title("fa");
        mat_writer.add_matrix("fa0",&*fa.begin(),1,fa.size());
        set_title("dir0");
        mat_writer.add_matrix("dir0",&*fdir.begin(),1,fdir.size());
        set_title("adc");
        mat_writer.add_matrix("adc",&*md.begin(),1,md.size());
        set_title("axial_dif");
        mat_writer.add_matrix("axial_dif",&*d0.begin(),1,d0.size());
        set_title("radial_dif1");
        mat_writer.add_matrix("radial_dif1",&*d1.begin(),1,d1.size());
        set_title("radial_dif2");
        mat_writer.add_matrix("radial_dif2",&*d2.begin(),1,d2.size());
    }
};


/*
struct TensorCalculator
{

    static la::lu<float,la::dim<6,6> > iKtK;
    static std::vector<float> Kt;
    static float eps;

public:
    void operator()(std::vector<float>& pdf,Model& )
    {
        float tensor[9];
        {
            // S = S0-S
            std::vector<float> S(_qcount-1);
            {
                float logs0 = std::log(pdf[SpaceMapping<dsi_range>::getIndex(0,0,0)]+eps);
                for (unsigned int i = 1;i < _qcount;++i)
                {
                    float value = std::log(pdf[SpaceMapping<dsi_range>::
                                                getIndex(qcode[i][0],qcode[i][1],qcode[i][2])]+eps);
                    S[i-1] = (logs0 > value) ? logs0-value:0.0;
                }

            }

            //  Kt S = Kt K D
            float KtS[6];
            la::matrix_product(Kt.begin(),S.begin(),KtS,
                               la::dim<6,_qcount-1>(),la::dim<_qcount-1,1>());

            float D[6];
            iKtK.solve(KtS,D);
            unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
            for (unsigned int index = 0;index < 9;++index)
                tensor[index] = D[tensor_index[index]];
        }
        float V[9],d[3];
        la::symmetric_matrix_eigen_decompose(tensor,V,d,la::dim<3,3>());

        float fa;
        float l1 = d[0];
        float l2 = d[1];
        float l3 = d[2];

        {
            float ll = (l1+l2+l3)/3.0;
            if (l1 < 0.0 || ll == 0.0)
            {
                .setValue(0,1.0,0.0);
                return;
            }
            float ll1 = l1-ll;
            float ll2 = l2-ll;
            float ll3 = l3-ll;
            fa = std::sqrt(1.5*(ll1*ll1+ll2*ll2+ll3*ll3)/(l1*l1+l2*l2+l3*l3));
        }


        unsigned int dir_index = 0;
        {
            float max_value = 0.0;
            float vx = V[0];
            float vy = V[3];
            float vz = V[6];
            for (unsigned int index = 0;index < odf_size;++index)
            {
                float value = 0.0;
                value += voxel.ti.vertices[index][0]*vx;
                value += voxel.ti.vertices[index][1]*vy;
                value += voxel.ti.vertices[index][2]*vz;
                if (value > max_value)
                {
                    max_value = value;
                    dir_index = index;
                }
            }
        }

        .setValue(dir_index,l1,fa);

    }

};
*/

#endif//_PROCESS_HPP
