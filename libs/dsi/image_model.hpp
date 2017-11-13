#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include "image/image.hpp"
#include "basic_voxel.hpp"
void get_report(const std::vector<float>& bvalues,image::vector<3> vs,std::string& report);

struct distortion_map{
    const float pi_2 = 3.14159265358979323846f/2.0f;
    image::basic_image<int,3> i1,i2;
    image::basic_image<float,3> w1,w2;
    void operator=(const image::basic_image<float,3>& d)
    {
        int n = d.width();
        i1.resize(d.geometry());
        i2.resize(d.geometry());
        w1.resize(d.geometry());
        w2.resize(d.geometry());
        int max_n = n-1.001f;
        image::par_for(d.height()*d.depth(),[&](int pos)
        {
            pos *= n;
            int* i1p = &i1[0]+pos;
            int* i2p = &i2[0]+pos;
            float* w1p = &w1[0]+pos;
            float* w2p = &w2[0]+pos;
            const float* dp = &*d.begin()+pos;
            for(int i = 0;i < n;++i)
            {
                if(dp[i] == 0.0f)
                {
                    i1p[i] = i;
                    i2p[i] = i;
                    w1p[i] = 0.0f;
                    w2p[i] = 0.0f;
                    continue;
                }
                float p1 = std::max<float>(0.0f,std::min<float>((float)i+d[i],max_n));
                float p2 = std::max<float>(0.0f,std::min<float>((float)i-d[i],max_n));
                i1p[i] = p1;
                i2p[i] = p2;
                w1p[i] = p1-std::floor(p1);
                w2p[i] = p2-std::floor(p2);
            }
        });
    }
    void calculate_displaced(image::basic_image<float,3>& j1,
                             image::basic_image<float,3>& j2,
                             const image::basic_image<float,3>& v)
    {
        int n = v.width();
        j1.clear();
        j2.clear();
        j1.resize(v.geometry());
        j2.resize(v.geometry());
        image::par_for(v.height()*v.depth(),[&](int pos)
        {
            pos *= n;
            const int* i1p = &i1[0]+pos;
            const int* i2p = &i2[0]+pos;
            const float* w1p = &w1[0]+pos;
            const float* w2p = &w2[0]+pos;
            const float* vp = &*v.begin()+pos;
            float* j1p = &j1[0]+pos;
            float* j2p = &j2[0]+pos;
            for(int i = 0;i < n;++i)
            {
                float value = vp[i];
                int i1 = i1p[i];
                int i2 = i2p[i];
                float vw1 = value*w1p[i];
                float vw2 = value*w2p[i];
                j1p[i1] += value-vw1;
                j1p[i1+1] += vw1;
                j2p[i2] += value-vw2;
                j2p[i2+1] += vw2;
            }
        });



    }

    void calculate_original(const image::basic_image<float,3>& v1,
                const image::basic_image<float,3>& v2,
                image::basic_image<float,3>& v)
    {
        int n = v1.width();
        int n2 = n + n;
        int block2 = n*n;
        v.clear();
        v.resize(v1.geometry());
        image::par_for(v1.height()*v1.depth(),[&](int pos)
        {
            pos *= n;
            const int* i1p = &i1[0]+pos;
            const int* i2p = &i2[0]+pos;
            const float* w1p = &w1[0]+pos;
            const float* w2p = &w2[0]+pos;
            std::vector<float> M(block2*2); // a col by col + col matrix
            float* p = &M[0];
            for(int i = 0;i < n;++i,p += n2)
            {
                int i1v = i1p[i];
                int i2v = i2p[i];
                p[i1v] += 1.0f-w1p[i];
                p[i1v+1] += w1p[i];
                p[i2v+n] += 1.0f-w2p[i];
                p[i2v+1+n] += w2p[i];
            }
            const float* v1p = &*v1.begin()+pos;
            const float* v2p = &*v2.begin()+pos;
            std::vector<float> y(n2);
            std::copy(v1p,v1p+n,y.begin());
            std::copy(v2p,v2p+n,y.begin()+n);
            image::mat::pseudo_inverse_solve(&M[0],&y[0],&v[0]+pos,image::dyndim(n,n2));
        });
    }
    void sample_gradient(const image::basic_image<float,3>& g1,
                         const image::basic_image<float,3>& g2,
                         image::basic_image<float,3>& new_g)
    {
        int n = g1.width();
        new_g.clear();
        new_g.resize(g1.geometry());
        image::par_for(g1.height()*g1.depth(),[&](int pos)
        {
            pos *= n;
            const int* i1p = &i1[0]+pos;
            const int* i2p = &i2[0]+pos;
            const float* w1p = &w1[0]+pos;
            const float* w2p = &w2[0]+pos;
            const float* g1p = &*g1.begin()+pos;
            const float* g2p = &*g2.begin()+pos;
            float* new_gp = &new_g[0]+pos;
            for(int i = 0;i < n;++i)
            {
                int i1 = i1p[i];
                int i2 = i2p[i];
                float w1 = w1p[i];
                float w2 = w2p[i];
                new_gp[i] += g1p[i1]*(1.0f-w1)+g1p[i1+1]*w1;
                new_gp[i] += g2p[i2]*(1.0f-w2)+g2p[i2+1]*w2;
            }
        });
    }
};



template<typename vector_type>
void print_v(const char* name,const vector_type& p)
{
    std::cout << name << "=[" << std::endl;
    for(int i = 0;i < p.size();++i)
        std::cout << p[i] << " ";
    std::cout << "];" << std::endl;
}

template<typename image_type>
void distortion_estimate(const image_type& v1,const image_type& v2,
                         image_type& d)
{
    image::geometry<3> geo(v1.geometry());
    if(geo.width() > 8)
    {
        image_type vv1,vv2;
        image::downsample_with_padding(v1,vv1);
        image::downsample_with_padding(v2,vv2);
        distortion_estimate(vv1,vv2,d);
        image::upsample_with_padding(d,d,geo);
        d *= 2.0f;
        image::filter::gaussian(d);
    }
    else
        d.resize(geo);
    int n = v1.width();
    image::basic_image<float,3> old_d(geo),v(geo),new_g(geo),j1(geo),j2(geo);
    float sum_dif = 0.0f;
    float s = 0.5f;
    distortion_map m;
    for(int iter = 0;iter < 200;++iter)
    {
        m = d;
        // estimate the original signal v using d
        m.calculate_original(v1,v2,v);
        // calculate the displaced image j1 j2 using v and d
        m.calculate_displaced(j1,j2,v);
        // calculate difference between current and estimated
        image::minus(j1,v1);
        image::minus(j2,v2);

        float sum = 0.0f;
        for(int i = 0;i < j1.size();++i)
        {
            sum += j1[i]*j1[i];
            sum += j2[i]*j2[i];
        }
        std::cout << "total dif=" << sum << std::endl;
        if(iter && sum > sum_dif)
        {
            std::cout << "cost increased.roll back." << std::endl;
            if(s < 0.05f)
                break;
            s *= 0.5f;
            d = old_d;
        }
        else
        {
            sum_dif = sum;
            image::basic_image<float,3> g1(geo),g2(geo);
            image::gradient(j1.begin(),j1.end(),g1.begin(),2,1);
            image::gradient(j2.begin(),j2.end(),g2.begin(),2,1);
            for(int i = 0;i < g1.size();++i)
                g1[i] = -g1[i];
            // sample gradient
            m.sample_gradient(g1,g2,new_g);
            old_d = d;
        }
        image::multiply_constant(new_g,s);
        image::add(d,new_g);
        image::lower_threshold(d,0.0f);
        for(int i = 0,pos = 0;i < geo.depth()*geo.height();++i,pos+=n)
        {
            d[pos] = 0.0f;
            d[pos+n-1] = 0.0f;
        }
    }
    std::cout << "end" << std::endl;
}

struct ImageModel
{
private:
    std::vector<image::basic_image<unsigned short,3> > new_dwi;//used in rotated volume

public:
    Voxel voxel;
    std::string file_name,error_msg;
    gz_mat_read mat_reader;

public:
    float quality_control_neighboring_dwi_corr(void)
    {
        // correction of neighboring DWI < 1750
        std::vector<std::pair<int,int> > corr_pairs;
        for(int i = 0;i < voxel.bvalues.size();++i)
        {
            if(voxel.bvalues[i] > 1750.0f || voxel.bvalues[i] == 0.0f)
                continue;
            float max_cos = 0.0;
            int max_j = 0;
            for(int j = 0;j < voxel.bvalues.size();++j)
                if(std::abs(voxel.bvalues[j]-voxel.bvalues[i]) < 100.0f && i != j)
                {
                    float cos = std::abs(voxel.bvectors[i]*voxel.bvectors[j]);
                    if(cos > max_cos)
                    {
                        max_cos = cos;
                        max_j = j;
                    }
                }
            if(max_j > i)
                corr_pairs.push_back(std::make_pair(i,max_j));
        }
        float self_cor = 0.0f;
        unsigned int count = 0;
        image::par_for(corr_pairs.size(),[&](int index)
        {
            int i1 = corr_pairs[index].first;
            int i2 = corr_pairs[index].second;
            std::vector<float> I1,I2;
            I1.reserve(voxel.dim.size());
            I2.reserve(voxel.dim.size());
            for(int i = 0;i < voxel.dim.size();++i)
                if(voxel.mask[i])
                {
                    I1.push_back(voxel.dwi_data[i1][i]);
                    I2.push_back(voxel.dwi_data[i2][i]);
                }
            self_cor += image::correlation(I1.begin(),I1.end(),I2.begin());
            ++count;
        });
        self_cor/= (float)count;
        return self_cor;
    }
    bool is_human_data(void) const
    {
        return voxel.dim[0]*voxel.vs[0] > 100 && voxel.dim[1]*voxel.vs[1] > 120 && voxel.dim[2]*voxel.vs[2] > 40;
    }
    void flip_b_table(const unsigned char* order)
    {
        for(unsigned int index = 0;index < voxel.bvectors.size();++index)
        {
            float x = voxel.bvectors[index][order[0]];
            float y = voxel.bvectors[index][order[1]];
            float z = voxel.bvectors[index][order[2]];
            voxel.bvectors[index][0] = x;
            voxel.bvectors[index][1] = y;
            voxel.bvectors[index][2] = z;
            if(order[3])
                voxel.bvectors[index][0] = -voxel.bvectors[index][0];
            if(order[4])
                voxel.bvectors[index][1] = -voxel.bvectors[index][1];
            if(order[5])
                voxel.bvectors[index][2] = -voxel.bvectors[index][2];
        }
        voxel.grad_dev.clear();
    }
    void flip_b_table(unsigned char dim)
    {
        for(unsigned int index = 0;index < voxel.bvectors.size();++index)
            voxel.bvectors[index][dim] = -voxel.bvectors[index][dim];
        if(!voxel.grad_dev.empty())
        {
            // <Flip*Gra_dev*b_table,ODF>
            // = <(Flip*Gra_dev*inv(Flip))*Flip*b_table,ODF>
            unsigned char nindex[3][4] = {{1,2,3,6},{1,3,5,7},{2,5,6,7}};
            for(unsigned int index = 0;index < 4;++index)
            {
                // 1  0  0         1  0  0
                //[0 -1  0] *Grad*[0 -1  0]
                // 0  0  1         0  0  1
                unsigned char pos = nindex[dim][index];
                for(unsigned int i = 0;i < voxel.dim.size();++i)
                    voxel.grad_dev[pos][i] = -voxel.grad_dev[pos][i];
            }
        }
    }
    // 0:xy 1:yz 2: xz
    void rotate_b_table(unsigned char dim)
    {
        std::swap(voxel.vs[dim],voxel.vs[(dim+1)%3]);
        for (unsigned int index = 0;index < voxel.bvectors.size();++index)
            std::swap(voxel.bvectors[index][dim],voxel.bvectors[index][(dim+1)%3]);
        if(!voxel.grad_dev.empty())
        {
            unsigned char swap1[3][6] = {{0,3,6,0,1,2},{1,4,7,3,4,5},{0,3,6,0,1,2}};
            unsigned char swap2[3][6] = {{1,4,7,3,4,5},{2,5,8,6,7,8},{2,5,8,6,7,8}};
            for(unsigned int index = 0;index < 6;++index)
            {
                unsigned char s1 = swap1[dim][index];
                unsigned char s2 = swap2[dim][index];
                for(unsigned int i = 0;i < voxel.dim.size();++i)
                    std::swap(voxel.grad_dev[s1][i],voxel.grad_dev[s2][i]);
            }
        }
    }

    // 0: x  1: y  2: z
    // 3: xy 4: yz 5: xz
    void flip(unsigned char type)
    {
        if(type < 3)
            flip_b_table(type);
        else
            rotate_b_table(type-3);
        image::flip(voxel.dwi_sum,type);
        image::flip(voxel.mask,type);
        for(unsigned int i = 0;i < voxel.grad_dev.size();++i)
        {
            auto I = image::make_image((float*)&*(voxel.grad_dev[i].begin()),voxel.dim);
            image::flip(I,type);
        }
        for (unsigned int index = 0;check_prog(index,voxel.dwi_data.size());++index)
        {
            auto I = image::make_image((unsigned short*)voxel.dwi_data[index],voxel.dim);
            image::flip(I,type);
        }
        voxel.dim = voxel.dwi_sum.geometry();
    }
    // used in eddy correction for each dwi
    void rotate_dwi(unsigned int dwi_index,const image::transformation_matrix<double>& affine)
    {
        image::basic_image<float,3> tmp(voxel.dim);
        auto I = image::make_image((unsigned short*)voxel.dwi_data[dwi_index],voxel.dim);
        image::resample(I,tmp,affine,image::cubic);
        image::lower_threshold(tmp,0);
        std::copy(tmp.begin(),tmp.end(),I.begin());
        // rotate b-table
        image::matrix<3,3,float> iT = image::inverse(affine.get());
        image::vector<3> v;
        image::vector_rotation(voxel.bvectors[dwi_index].begin(),v.begin(),iT,image::vdim<3>());
        v.normalize();
        voxel.bvectors[dwi_index] = v;
    }

    void rotate(const image::basic_image<float,3>& ref,const image::transformation_matrix<double>& affine)
    {
        image::geometry<3> new_geo = ref.geometry();
        std::vector<image::basic_image<unsigned short,3> > dwi(voxel.dwi_data.size());
        image::par_for2(voxel.dwi_data.size(),[&](unsigned int index,unsigned int id)
        {
            if(!id)
                check_prog(index,voxel.dwi_data.size());
            dwi[index].resize(new_geo);
            auto I = image::make_image((unsigned short*)voxel.dwi_data[index],voxel.dim);
            image::resample_with_ref(I,ref,dwi[index],affine);
            voxel.dwi_data[index] = &(dwi[index][0]);
        });
        check_prog(0,0);
        dwi.swap(new_dwi);

        // rotate b-table
        image::matrix<3,3,float> iT = image::inverse(affine.get());
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            image::vector<3> tmp;
            image::vector_rotation(voxel.bvectors[index].begin(),tmp.begin(),iT,image::vdim<3>());
            tmp.normalize();
            voxel.bvectors[index] = tmp;
        }

        if(!voxel.grad_dev.empty())
        {
            // <R*Gra_dev*b_table,ODF>
            // = <(R*Gra_dev*inv(R))*R*b_table,ODF>
            float det = std::abs(iT.det());
            begin_prog("rotating grad_dev");
            for(unsigned int index = 0;check_prog(index,voxel.dim.size());++index)
            {
                image::matrix<3,3,float> grad_dev,G_invR;
                for(unsigned int i = 0; i < 9; ++i)
                    grad_dev[i] = voxel.grad_dev[i][index];
                G_invR = grad_dev*affine.get();
                grad_dev = iT*G_invR;
                for(unsigned int i = 0; i < 9; ++i)
                    voxel.grad_dev[i][index] = grad_dev[i]/det;
            }
            std::vector<image::basic_image<float,3> > new_gra_dev(voxel.grad_dev.size());
            begin_prog("rotating grad_dev volume");
            for (unsigned int index = 0;check_prog(index,new_gra_dev.size());++index)
            {
                new_gra_dev[index].resize(new_geo);
                image::resample(voxel.grad_dev[index],new_gra_dev[index],affine,image::cubic);
                voxel.grad_dev[index] = image::make_image((float*)&(new_gra_dev[index][0]),voxel.dim);
            }
            new_gra_dev.swap(voxel.new_grad_dev);
        }
        voxel.dim = new_geo;
        calculate_dwi_sum();
        calculate_mask();

    }
    void trim(void)
    {
        image::geometry<3> range_min,range_max;
        image::bounding_box(voxel.mask,range_min,range_max,0);
        for (unsigned int index = 0;check_prog(index,voxel.dwi_data.size());++index)
        {
            auto I = image::make_image((unsigned short*)voxel.dwi_data[index],voxel.dim);
            image::basic_image<unsigned short,3> I0 = I;
            image::crop(I0,range_min,range_max);
            std::fill(I.begin(),I.end(),0);
            std::copy(I0.begin(),I0.end(),I.begin());
        }
        image::crop(voxel.mask,range_min,range_max);
        voxel.dim = voxel.mask.geometry();
        calculate_dwi_sum();
        calculate_mask();
    }
    void distortion_correction(const ImageModel& rhs)
    {
        image::basic_image<float,3> v1(voxel.dwi_sum),v2(rhs.voxel.dwi_sum),d;
        v1 /= image::mean(v1);
        v2 /= image::mean(v2);
        image::filter::gaussian(v1);
        image::filter::gaussian(v2);
        bool swap_xy = false;
        bool swap_ap = false;

        {
            image::vector<3,float> m1 = image::center_of_mass(v1); // should be d+
            image::vector<3,float> m2 = image::center_of_mass(v2); // should be d-
            m1 -= m2;
            std::cout << m1 << std::endl;
            if(std::abs(m1[1]) > std::abs(m1[0]))
            {
                image::swap_xy(v1);
                image::swap_xy(v2);
                std::swap(m1[1],m1[0]);
                swap_xy = true;
            }
            if(m1[0] > 0)
            {
                v1.swap(v2);
                swap_ap = true;
            }
        }


        distortion_estimate(v1,v2,d);
        check_prog(0,0);
        if(prog_aborted())
            return;
        begin_prog("applying warp");
        std::vector<image::basic_image<unsigned short,3> > dwi(voxel.dwi_data.size());
        distortion_map m;
        m = d;
        for(int i = 0;check_prog(i,voxel.dwi_data.size());++i)
        {
            //dwi[i] = image::make_image((unsigned short*)dwi_data[i],voxel.dim);
            if(prog_aborted())
                return;
            image::basic_image<float,3> dwi1 = image::make_image((unsigned short*)voxel.dwi_data[i],voxel.dim);
            image::basic_image<float,3> dwi2 = image::make_image((unsigned short*)rhs.voxel.dwi_data[i],voxel.dim);
            if(swap_xy)
            {
                image::swap_xy(dwi1);
                image::swap_xy(dwi2);
            }
            if(swap_ap)
                dwi1.swap(dwi2);
            image::basic_image<float,3> v;
            if(i == 1)
            {
                image::filter::gaussian(dwi1);
                image::filter::gaussian(dwi2);
                m.calculate_original(dwi1,dwi2,v);
            }
            else
                v = dwi1;
            if(swap_xy)
                image::swap_xy(v);
            image::lower_threshold(v,0);
            dwi[i] = v;
        }
        if(prog_aborted())
            return;
        d *= 50.0f;
        image::swap_xy(d);
        dwi[0] = d;
        new_dwi.swap(dwi);
        for(int i = 0;i < new_dwi.size();++i)
            voxel.dwi_data[i] = &(new_dwi[i][0]);
        calculate_dwi_sum();
        calculate_mask();
    }

    bool save_to_nii(const char* nifti_file_name) const
    {
        gz_nifti header;
        float vs[4];
        std::copy(voxel.vs.begin(),voxel.vs.end(),vs);
        vs[3] = 1.0;
        header.set_voxel_size(vs);
        image::geometry<4> nifti_dim;
        std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
        nifti_dim[3] = voxel.bvalues.size();
        image::basic_image<unsigned short,4> buffer(nifti_dim);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            std::copy(voxel.dwi_data[index],
                      voxel.dwi_data[index]+voxel.dim.size(),
                      buffer.begin() + (size_t)index*voxel.dim.size());
        }
        image::flip_xy(buffer);
        header << buffer;
        return header.save_to_file(nifti_file_name);
    }
    bool save_b0_to_nii(const char* nifti_file_name) const
    {
        gz_nifti header;
        header.set_voxel_size(voxel.vs);
        image::basic_image<unsigned short,3> buffer(voxel.dwi_data[0],voxel.dim);
        image::flip_xy(buffer);
        header << buffer;
        return header.save_to_file(nifti_file_name);
    }
    bool save_b_table(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            out << voxel.bvalues[index] << " "
                << voxel.bvectors[index][0] << " "
                << voxel.bvectors[index][1] << " "
                << voxel.bvectors[index][2] << std::endl;
        }
        return out.good();
    }
    bool save_bval(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            if(index)
                out << " ";
            out << voxel.bvalues[index];
        }
        return out.good();
    }
    bool save_bvec(const char* file_name) const
    {
        std::ofstream out(file_name);
        for(unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            out << voxel.bvectors[index][0] << " "
                << voxel.bvectors[index][1] << " "
                << voxel.bvectors[index][2] << std::endl;
        }
        return out.good();
    }


public:
    bool load_from_file(const char* dwi_file_name)
    {
        file_name = dwi_file_name;
        if (!mat_reader.load_from_file(dwi_file_name))
        {
            error_msg = "Cannot open file";
            return false;
        }
        unsigned int row,col;

        const unsigned short* dim_ptr = 0;
        if (!mat_reader.read("dimension",row,col,dim_ptr))
        {
            error_msg = "Cannot find dimension matrix";
            return false;
        }
        const float* voxel_size = 0;
        if (!mat_reader.read("voxel_size",row,col,voxel_size))
        {
            //error_msg = "Cannot find voxel size matrix";
            //return false;
            std::fill(voxel.vs.begin(),voxel.vs.end(),3.0);
        }
        else
            std::copy(voxel_size,voxel_size+3,voxel.vs.begin());

        if (dim_ptr[0]*dim_ptr[1]*dim_ptr[2] <= 0)
        {
            error_msg = "Invalid dimension setting";
            return false;
        }
        voxel.dim[0] = dim_ptr[0];
        voxel.dim[1] = dim_ptr[1];
        voxel.dim[2] = dim_ptr[2];

        const float* table;
        if (!mat_reader.read("b_table",row,col,table))
        {
            error_msg = "Cannot find b_table matrix";
            return false;
        }
        voxel.bvalues.resize(col);
        voxel.bvectors.resize(col);
        for (unsigned int index = 0;index < col;++index)
        {
            voxel.bvalues[index] = table[0];
            voxel.bvectors[index][0] = table[1];
            voxel.bvectors[index][1] = table[2];
            voxel.bvectors[index][2] = table[3];
            voxel.bvectors[index].normalize();
            table += 4;
        }

        const char* report_buf = 0;
        if(mat_reader.read("report",row,col,report_buf))
            voxel.report = std::string(report_buf,report_buf+row*col);
        else
            get_report(voxel.bvalues,voxel.vs,voxel.report);

        voxel.dwi_data.resize(voxel.bvalues.size());
        for (unsigned int index = 0;index < voxel.bvalues.size();++index)
        {
            std::ostringstream out;
            out << "image" << index;
            mat_reader.read(out.str().c_str(),row,col,voxel.dwi_data[index]);
            if (!voxel.dwi_data[index])
            {
                error_msg = "Cannot find image matrix";
                return false;
            }
        }


        {
            const float* grad_dev = 0;
            if(mat_reader.read("grad_dev",row,col,grad_dev) && row*col == voxel.dim.size()*9)
            {
                for(unsigned int index = 0;index < 9;index++)
                    voxel.grad_dev.push_back(image::make_image((float*)grad_dev+index*voxel.dim.size(),voxel.dim));
                if(std::fabs(voxel.grad_dev[0][0])+std::fabs(voxel.grad_dev[4][0])+std::fabs(voxel.grad_dev[8][0]) < 1.0)
                {
                    image::add_constant(voxel.grad_dev[0].begin(),voxel.grad_dev[0].end(),1.0);
                    image::add_constant(voxel.grad_dev[4].begin(),voxel.grad_dev[4].end(),1.0);
                    image::add_constant(voxel.grad_dev[8].begin(),voxel.grad_dev[8].end(),1.0);
                }
            }

        }

        // create mask;
        calculate_dwi_sum();

        const unsigned char* mask_ptr = 0;
        if(mat_reader.read("mask",row,col,mask_ptr))
        {
            voxel.mask.resize(voxel.dim);
            if(row*col == voxel.dim.size())
                std::copy(mask_ptr,mask_ptr+row*col,voxel.mask.begin());
        }
        else
            calculate_mask();
        return true;
    }
    void calculate_dwi_sum(void)
    {
        voxel.dwi_sum.clear();
        voxel.dwi_sum.resize(voxel.dim);
        image::par_for(voxel.dwi_sum.size(),[&](unsigned int pos)
        {
            for (unsigned int index = 0;index < voxel.dwi_data.size();++index)
                voxel.dwi_sum[pos] += voxel.dwi_data[index][pos];
        });

        float max_value = *std::max_element(voxel.dwi_sum.begin(),voxel.dwi_sum.end());
        float min_value = max_value;
        for (unsigned int index = 0;index < voxel.dwi_sum.size();++index)
            if (voxel.dwi_sum[index] < min_value && voxel.dwi_sum[index] > 0)
                min_value = voxel.dwi_sum[index];


        image::minus_constant(voxel.dwi_sum,min_value);
        image::lower_threshold(voxel.dwi_sum,0.0f);
        float t = image::segmentation::otsu_threshold(voxel.dwi_sum);
        image::upper_threshold(voxel.dwi_sum,t*3.0f);
        image::normalize(voxel.dwi_sum,1.0);
    }
    void calculate_mask(void)
    {
        image::threshold(voxel.dwi_sum,voxel.mask,0.2f,1,0);
        if(voxel.dwi_sum.depth() < 10)
        {
            for(unsigned int i = 0;i < voxel.mask.depth();++i)
            {
                image::pointer_image<unsigned char,2> I(&voxel.mask[0]+i*voxel.mask.plane_size(),
                        image::geometry<2>(voxel.mask.width(),voxel.mask.height()));
                image::morphology::defragment(I);
                image::morphology::recursive_smoothing(I,10);
                image::morphology::defragment(I);
            }
        }
        else
        {
            image::morphology::recursive_smoothing(voxel.mask,10);
            image::morphology::defragment(voxel.mask);
            image::morphology::recursive_smoothing(voxel.mask,10);
        }
    }
    void save_to_file(gz_mat_write& mat_writer)
    {

        set_title("saving");

        // dimension
        {
            short dim[3];
            dim[0] = voxel.dim[0];
            dim[1] = voxel.dim[1];
            dim[2] = voxel.dim[2];
            mat_writer.write("dimension",dim,1,3);
        }

        // voxel size
        mat_writer.write("voxel_size",&*voxel.vs.begin(),1,3);

        std::vector<float> float_data;
        std::vector<short> short_data;
        voxel.ti.save_to_buffer(float_data,short_data);
        mat_writer.write("odf_vertices",&*float_data.begin(),3,voxel.ti.vertices_count);
        mat_writer.write("odf_faces",&*short_data.begin(),3,voxel.ti.faces.size());

    }
public:
    template<class CheckType>
    bool avaliable(void) const
    {
        return CheckType::check(voxel);
    }

    template<class ProcessType>
    bool reconstruct(void)
    {
        begin_prog("reconstruction");
        voxel.CreateProcesses<ProcessType>();
        voxel.init();
        voxel.run();
        return !prog_aborted();
    }


    void save_fib(const std::string& ext)
    {
        std::string output_name = file_name;
        output_name += ext;
        begin_prog("saving data");
        gz_mat_write mat_writer(output_name.c_str());
        save_to_file(mat_writer);
        voxel.end(mat_writer);
        std::string final_report = voxel.report.c_str();
        final_report += voxel.recon_report.str();
        mat_writer.write("report",final_report.c_str(),1,final_report.length());
    }

};


#endif//IMAGE_MODEL_HPP
