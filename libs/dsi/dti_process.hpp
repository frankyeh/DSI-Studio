#ifndef DTI_PROCESS_HPP
#define DTI_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"

class Dwi2Tensor : public BaseProcess
{
    std::vector<float> ad,rd,rd1,rd2,md,txx,txy,txz,tyy,tyz,tzz,ha;
    float get_fa(float l1,float l2,float l3)
    {
        float ll = (l1+l2+l3)/3.0f;
        float ll1 = l1-ll;
        float ll2 = l2-ll;
        float ll3 = l3-ll;
        float avg = (l1*l1+l2*l2+l3*l3);
        if(avg == 0.0f)
            avg = 1.0f;
        return std::min<float>(1.0f,std::sqrt(1.5f*(ll1*ll1+ll2*ll2+ll3*ll3)/avg));
    }
private:
    //math::dynamic_matrix<float> iKtKKt;
    std::vector<std::vector<double> > iKtK; // 6-by-6
    std::vector<std::vector<unsigned int> > iKtK_pivot;
    std::vector<double> Kt;
    unsigned int b_count;
    std::vector<size_t> b_location;
public:
    virtual void init(Voxel& voxel)
    {
        if(std::count_if(voxel.bvalues.begin(),voxel.bvalues.end(),[](float b){return b > 100.0f && b <= 1750.0f;}) < 10)
            voxel.dti_no_high_b = false;

        voxel.fib_fa.clear();
        voxel.fib_fa.resize(voxel.dim);
        voxel.fib_dir.clear();
        voxel.fib_dir.resize(voxel.dim.size());

        if(voxel.needs("md"))
            md.resize(voxel.dim.size());
        if(voxel.needs("ad"))
            ad.resize(voxel.dim.size());
        if(voxel.needs("rd"))
            rd.resize(voxel.dim.size());
        if(voxel.needs("rd1"))
            rd1.resize(voxel.dim.size());
        if(voxel.needs("rd2"))
            rd2.resize(voxel.dim.size());
        if(voxel.needs("helix"))
            ha.resize(voxel.dim.size());
        if(voxel.needs("tensor"))
        {
            txx.resize(voxel.dim.size());
            txy.resize(voxel.dim.size());
            txz.resize(voxel.dim.size());
            tyy.resize(voxel.dim.size());
            tyz.resize(voxel.dim.size());
            tzz.resize(voxel.dim.size());
        }

        // the first DWI should be b0
        if(voxel.bvalues[0] > 100.0f)
            throw std::runtime_error("cannot locate b0 data for DTI reconstruction");
        voxel.bvalues[0] = 0.0f;

        std::vector<tipl::vector<3> > b_data;
        b_count = 0;

        for(size_t i = 1;i < voxel.bvalues.size();++i)//skip b0
        {
            if(voxel.dti_no_high_b && voxel.bvalues[i] > 1750.0f)
                continue;
            b_count++;
            b_location.push_back(i);
            b_data.push_back(voxel.bvectors[i]);
            b_data.back() *= std::sqrt(voxel.bvalues[i]);
        }

        Kt.resize(6*b_count);
        {
            unsigned int qmap[6]= {0  ,4  ,8  ,1  ,2  ,5  };
            double qweighting[6]= {1.0,1.0,1.0,2.0,2.0,2.0};
            //					  bxx,byy,bzz,bxy,bxz,byz
            for (unsigned int i = 0,k = 0; i < b_data.size(); ++i,k+=6)
            {
                //qq = q qT
                std::vector<double> qq(3*3);
                tipl::mat::product_transpose(b_data[i].begin(),b_data[i].begin(),qq.begin(),
                                               tipl::shape<2>(3,1),tipl::shape<2>(3,1));

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
                                           tipl::shape<2>(6,b_count),tipl::shape<2>(6,b_count));
            if(i)
            {
                double w = 0.005*std::pow(2.0,double(i))*(tipl::max_value(iKtK[i]));
                for(unsigned int j = 0;j < 36;j += 7)
                    iKtK[i][j] += w;
            }
            tipl::mat::lu_decomposition(iKtK[i].begin(),iKtK_pivot[i].begin(),tipl::shape<2>(6,6));
        }
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(voxel.fib_fa.empty())
            return;
        std::vector<double> signal(b_count);
        {
            double logs0 = std::log(std::max<double>(1.0,double(data.space.front())));
            for (size_t i = 0;i < b_count;++i)
                signal[i] = std::log(std::max<double>(1.0,double(data.space[b_location[i]])));
            logs0 = std::max<double>(logs0,tipl::max_value(signal));
            if(logs0 == 0.0)
                return;
            for (size_t i = 0;i < b_count;++i)
                signal[i] = std::max<double>(0.0,logs0-signal[i]);
        }
        //  Kt S = Kt K D
        double KtS[6],tensor_param[6];
        double tensor[9];
        double V[9],d[3];
        tipl::mat::product(Kt.begin(),signal.begin(),KtS,tipl::shape<2>(6,b_count),tipl::shape<2>(b_count,1));
        for(unsigned int i = 0;i < iKtK.size();++i)
        {
            if(!tipl::mat::lu_solve(iKtK[i].begin(),iKtK_pivot[i].begin(),KtS,tensor_param,tipl::shape<2>(6,6)))
                continue;
            unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
            for (unsigned int index = 0; index < 9; ++index)
                tensor[index] = tensor_param[tensor_index[index]];
            tipl::mat::eigen_decomposition_sym(tensor,V,d,tipl::dim<3,3>());
            if(d[0] > 0.0 && d[1] > 0.0 && d[2] > 0.0)
                break;
        }

        d[0] = std::max(0.0,d[0]);
        d[1] = std::max(0.0,d[1]);
        d[2] = std::max(0.0,d[2]);

        std::copy(V,V+3,voxel.fib_dir[data.voxel_index].begin());
        voxel.fib_fa[data.voxel_index] = get_fa(float(d[0]),float(d[1]),float(d[2]));

        if(!md.empty())
            md[data.voxel_index] = 1000.0f*float(d[0]+d[1]+d[2])/3.0f;
        if(!ad.empty())
            ad[data.voxel_index] = 1000.0f*float(d[0]);
        if(!rd1.empty())
            rd1[data.voxel_index] = 1000.0f*float(d[1]);
        if(!rd2.empty())
            rd2[data.voxel_index] = 1000.0f*float(d[2]);
        if(!rd.empty())
            rd[data.voxel_index] = 1000.0f*float(d[1]+d[2])/2.0f;

        if(!ha.empty())
        {
            ha[data.voxel_index] = float(std::acos(std::sqrt(V[0]*V[0]+V[1]*V[1]))*180.0/3.14159265358979323846);
            tipl::vector<3> center(float(voxel.dim[0])*0.5f,float(voxel.dim[1])*0.5f,float(voxel.dim[2])*0.5f);
            center -= tipl::vector<3>(tipl::pixel_index<3>(data.voxel_index,voxel.dim));
            if((center.cross_product(tipl::vector<3>(0.0f,0.0f,1.0f))*tipl::vector<3>(V) < 0) ^
                    (V[2] < 0.0))
                ha[data.voxel_index] = -ha[data.voxel_index];
        }
        if(!txx.empty())
        {
            txx[data.voxel_index] = float(tensor[0]);
            txy[data.voxel_index] = float(tensor[1]);
            txz[data.voxel_index] = float(tensor[2]);
            tyy[data.voxel_index] = float(tensor[4]);
            tyz[data.voxel_index] = float(tensor[5]);
            tzz[data.voxel_index] = float(tensor[8]);
        }

    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        if(voxel.fib_fa.empty())
            return;
        if(voxel.method_id == 1) // DTI
        {
            mat_writer.write("fa0",voxel.fib_fa,uint32_t(voxel.dim.plane_size()));
            mat_writer.write("dir0",&voxel.fib_dir[0][0],uint32_t(3*voxel.dim.plane_size()),uint32_t(voxel.dim.depth()));
        }
        else
            mat_writer.write("dti_fa",voxel.fib_fa,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("txx",txx,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("txy",txy,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("txz",txz,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("tyy",tyy,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("tyz",tyz,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("tzz",tzz,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("rd1",rd1,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("rd2",rd2,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("ha",ha,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("md",md,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("ad",ad,uint32_t(voxel.dim.plane_size()));
        mat_writer.write("rd",rd,uint32_t(voxel.dim.plane_size()));
    }
};

#endif//_PROCESS_HPP
