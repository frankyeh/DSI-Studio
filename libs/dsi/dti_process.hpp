#ifndef DTI_PROCESS_HPP
#define DTI_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image/image.hpp"

class Dwi2Tensor : public BaseProcess
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
private:
    //math::dynamic_matrix<float> iKtKKt;
    std::vector<double> iKtK; // 6-by-6
    std::vector<unsigned int> iKtK_pivot;
    std::vector<double> Kt;
    unsigned int b_count;
public:
    virtual void init(Voxel& voxel)
    {
        fa.clear();
        fa.resize(voxel.dim.size());
        fdir.clear();
        fdir.resize(voxel.dim.size()*3);
        md.clear();
        md.resize(voxel.dim.size());
        d0.clear();
        d0.resize(voxel.dim.size());
        d1.clear();
        d1.resize(voxel.dim.size());
        d2.clear();
        d2.resize(voxel.dim.size());

        b_count = voxel.q_count-1;
        std::vector<image::vector<3> > b_data(b_count);
        //skip null
        std::copy(voxel.bvectors.begin()+1,voxel.bvectors.end(),b_data.begin());
        for(unsigned int index = 0; index < b_count; ++index)
            b_data[index] *= std::sqrt(voxel.bvalues[index+1]);

        Kt.resize(6*b_count);
        {
            unsigned int qmap[6]		= {0  ,4  ,8  ,1  ,2  ,5  };
            double qweighting[6]= {1.0,1.0,1.0,2.0,2.0,2.0};
            //					  bxx,byy,bzz,bxy,bxz,byz
            for (unsigned int i = 0,k = 0; i < b_data.size(); ++i,k+=6)
            {
                //qq = q qT
                std::vector<float> qq(3*3);
                image::matrix::product_transpose(b_data[i].begin(),b_data[i].begin(),qq.begin(),
                                               image::dyndim(3,1),image::dyndim(3,1));

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
        image::matrix::product_transpose(Kt.begin(),Kt.begin(),iKtK.begin(),
                                       image::dyndim(6,b_count),image::dyndim(6,b_count));
        image::matrix::lu_decomposition(iKtK.begin(),iKtK_pivot.begin(),image::dyndim(6,6));
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> signal(data.space.size());
        if (data.space.front() != 0.0)
        {
            float logs0 = std::log(std::max<float>(1.0,data.space.front()));
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = std::max<float>(0.0,logs0-std::log(std::max<float>(1.0,data.space[i])));
        }
        //  Kt S = Kt K D
        double KtS[6],tensor_param[6];
        double tensor[9];
        double V[9],d[3];

        image::matrix::product(Kt.begin(),signal.begin(),KtS,image::dyndim(6,b_count),image::dyndim(b_count,1));
        image::matrix::lu_solve(iKtK.begin(),iKtK_pivot.begin(),KtS,tensor_param,image::dyndim(6,6));


        unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
        for (unsigned int index = 0; index < 9; ++index)
            tensor[index] = tensor_param[tensor_index[index]];

        image::matrix::eigen_decomposition_sym(tensor,V,d,image::dim<3,3>());
        if (d[1] < 0.0)
        {
            d[1] = 0.0;
            d[2] = 0.0;
        }
        if (d[2] < 0.0)
            d[2] = 0.0;
        if (d[0] < 0.0)
        {
            d[0] = 0.0;
            d[1] = 0.0;
            d[2] = 0.0;
        }
        std::copy(V,V+3,fdir.begin() + data.voxel_index * 3);
        data.fa[0] = fa[data.voxel_index] = get_fa(d[0],d[1],d[2]);
        md[data.voxel_index] = (d[0]+d[1]+d[2])/3.0;
        d0[data.voxel_index] = d[0];
        d1[data.voxel_index] = d[1];
        d2[data.voxel_index] = d[2];
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

#endif//_PROCESS_HPP
