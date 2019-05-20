#ifndef DTI_PROCESS_HPP
#define DTI_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "tipl/tipl.hpp"

class Dwi2Tensor : public BaseProcess
{
    std::vector<float> d0,d1,d2,d3,md,txx,txy,txz,tyy,tyz,tzz,ha;
    float get_fa(float l1,float l2,float l3)
    {
        float ll = (l1+l2+l3)/3.0f;
        float ll1 = l1-ll;
        float ll2 = l2-ll;
        float ll3 = l3-ll;
        float avg = (l1*l1+l2*l2+l3*l3);
        if(avg == 0.0f)
            avg = 1.0f;
        return std::min<float>(1.0f,std::sqrtf(1.5f*(ll1*ll1+ll2*ll2+ll3*ll3)/avg));
    }
private:
    //math::dynamic_matrix<float> iKtKKt;
    std::vector<std::vector<double> > iKtK; // 6-by-6
    std::vector<std::vector<unsigned int> > iKtK_pivot;
    std::vector<double> Kt;
    unsigned int b_count;
public:
    virtual void init(Voxel& voxel)
    {
        voxel.fib_fa.clear();
        voxel.fib_fa.resize(voxel.dim);
        voxel.fib_dir.clear();
        voxel.fib_dir.resize(voxel.dim.size());
        if(voxel.output_diffusivity || voxel.method_id == 1)
        {
            md.clear();
            md.resize(voxel.dim.size());
            d0.clear();
            d0.resize(voxel.dim.size());
            d1.clear();
            d1.resize(voxel.dim.size());
            d2.clear();
            d2.resize(voxel.dim.size());
            d3.clear();
            d3.resize(voxel.dim.size());
            ha.clear();
            ha.resize(voxel.dim.size());
        }
        if(voxel.output_tensor && voxel.method_id == 1)
        {
            txx.clear();
            txx.resize(voxel.dim.size());
            txy.clear();
            txy.resize(voxel.dim.size());
            txz.clear();
            txz.resize(voxel.dim.size());
            tyy.clear();
            tyy.resize(voxel.dim.size());
            tyz.clear();
            tyz.resize(voxel.dim.size());
            tzz.clear();
            tzz.resize(voxel.dim.size());
        }

        b_count = uint32_t(voxel.bvalues.size()-1);
        std::vector<tipl::vector<3> > b_data(b_count);
        //skip b0
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
                std::vector<double> qq(3*3);
                tipl::mat::product_transpose(b_data[i].begin(),b_data[i].begin(),qq.begin(),
                                               tipl::dyndim(3,1),tipl::dyndim(3,1));

                /*
                      q11 q15 q19 2*q12 2*q13 2*q16
                      q21 q25 q29 2*q22 2*q23 2*q26
                K  = | ...                         |
                */
                for (unsigned int col = 0,index = i; col < 6; ++col,index+=b_count)
                    Kt[index] = qq[qmap[col]]*qweighting[col];
            }
        }
        iKtK.resize(20);
        iKtK_pivot.resize(iKtK.size());
        for(unsigned int i = 0;i < iKtK.size();++i)
        {
            iKtK[i].resize(6*6);
            iKtK_pivot[i].resize(6);
            tipl::mat::product_transpose(Kt.begin(),Kt.begin(),iKtK[i].begin(),
                                           tipl::dyndim(6,b_count),tipl::dyndim(6,b_count));
            if(i)
            {
                double w = 0.005*std::pow(2.0,double(i))*(*std::max_element(iKtK[i].begin(),iKtK[i].end()));
                for(unsigned int j = 0;j < 36;j += 7)
                    iKtK[i][j] += w;
            }
            tipl::mat::lu_decomposition(iKtK[i].begin(),iKtK_pivot[i].begin(),tipl::dyndim(6,6));
        }
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(!voxel.output_diffusivity && voxel.method_id != 1)
            return;
        std::vector<float> signal(data.space.size());
        if (data.space.front() != 0.0f)
        {
            float logs0 = std::log(std::max<float>(1.0,data.space.front()));
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = std::max<float>(0.0,logs0-std::log(std::max<float>(1.0,data.space[i])));
        }
        //  Kt S = Kt K D
        double KtS[6],tensor_param[6];
        double tensor[9];
        double V[9],d[3];
        for(unsigned int i = 0;i < iKtK.size();++i)
        {
            tipl::mat::product(Kt.begin(),signal.begin(),KtS,tipl::dyndim(6,b_count),tipl::dyndim(b_count,1));
            tipl::mat::lu_solve(iKtK[i].begin(),iKtK_pivot[i].begin(),KtS,tensor_param,tipl::dyndim(6,6));


            unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
            for (unsigned int index = 0; index < 9; ++index)
                tensor[index] = tensor_param[tensor_index[index]];

            tipl::mat::eigen_decomposition_sym(tensor,V,d,tipl::dim<3,3>());
            if(d[0] > 0.0 && d[1] > 0.0 && d[2] > 0.0)
                break;
        }
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
        std::copy(V,V+3,voxel.fib_dir[data.voxel_index].begin());
        data.fa[0] = voxel.fib_fa[data.voxel_index] = get_fa(float(d[0]),float(d[1]),float(d[2]));
        if(voxel.output_diffusivity || voxel.method_id == 1)
        {
            md[data.voxel_index] = 1000.0f*float(d[0]+d[1]+d[2])/3.0f;
            d0[data.voxel_index] = 1000.0f*float(d[0]);
            d2[data.voxel_index] = 1000.0f*float(d[1]);
            d3[data.voxel_index] = 1000.0f*float(d[2]);
            d1[data.voxel_index] = 1000.0f*float(d[1]+d[2])/2.0f;
            if(voxel.output_helix_angle)
                ha[data.voxel_index] = float(std::acos(std::sqrt(V[0]*V[0]+V[1]*V[1]))*180.0/3.14159265358979323846);
        }
        if(voxel.output_tensor && voxel.method_id == 1)
        {
            txx[data.voxel_index] = float(tensor[0]);
            txy[data.voxel_index] = float(tensor[1]);
            txz[data.voxel_index] = float(tensor[2]);
            tyy[data.voxel_index] = float(tensor[4]);
            tyz[data.voxel_index] = float(tensor[5]);
            tzz[data.voxel_index] = float(tensor[8]);
        }

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if(voxel.method_id == 1) // DTI
        {
            mat_writer.write("fa0",voxel.fib_fa);
            mat_writer.write("dir0",&voxel.fib_dir[0][0],1,uint32_t(voxel.fib_dir.size()*3));
            if(voxel.output_tensor)
            {
                mat_writer.write("txx",txx);
                mat_writer.write("txy",txy);
                mat_writer.write("txz",txz);
                mat_writer.write("tyy",tyy);
                mat_writer.write("tyz",tyz);
                mat_writer.write("tzz",tzz);
            }
        }
        if(voxel.output_diffusivity || voxel.method_id == 1)
        {
            if(voxel.method_id != 1) // DTI
                mat_writer.write("dti_fa",voxel.fib_fa);
            mat_writer.write("md",md);
            mat_writer.write("ad",d0);
            mat_writer.write("rd",d1);
            mat_writer.write("rd1",d2);
            mat_writer.write("rd2",d3);
            if(voxel.output_helix_angle)
                mat_writer.write("ha",ha);
        }

    }
};

#endif//_PROCESS_HPP
