#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include "tipl/tipl.hpp"
#include "basic_voxel.hpp"
struct distortion_map{
    const float pi_2 = 3.14159265358979323846f/2.0f;
    tipl::image<int,3> i1,i2;
    tipl::image<float,3> w1,w2;
    void operator=(const tipl::image<float,3>& d)
    {
        int n = d.width();
        i1.resize(d.geometry());
        i2.resize(d.geometry());
        w1.resize(d.geometry());
        w2.resize(d.geometry());
        int max_n = n-1.001f;
        tipl::par_for(d.height()*d.depth(),[&](int pos)
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
    void calculate_displaced(tipl::image<float,3>& j1,
                             tipl::image<float,3>& j2,
                             const tipl::image<float,3>& v)
    {
        int n = v.width();
        j1.clear();
        j2.clear();
        j1.resize(v.geometry());
        j2.resize(v.geometry());
        tipl::par_for(v.height()*v.depth(),[&](int pos)
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

    void calculate_original(const tipl::image<float,3>& v1,
                const tipl::image<float,3>& v2,
                tipl::image<float,3>& v)
    {
        int n = v1.width();
        int n2 = n + n;
        int block2 = n*n;
        v.clear();
        v.resize(v1.geometry());
        tipl::par_for(v1.height()*v1.depth(),[&](int pos)
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
            tipl::mat::pseudo_inverse_solve(&M[0],&y[0],&v[0]+pos,tipl::dyndim(n,n2));
        });
    }
    void sample_gradient(const tipl::image<float,3>& g1,
                         const tipl::image<float,3>& g2,
                         tipl::image<float,3>& new_g)
    {
        int n = g1.width();
        new_g.clear();
        new_g.resize(g1.geometry());
        tipl::par_for(g1.height()*g1.depth(),[&](int pos)
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
    tipl::geometry<3> geo(v1.geometry());
    if(geo.width() > 8)
    {
        image_type vv1,vv2;
        tipl::downsample_with_padding(v1,vv1);
        tipl::downsample_with_padding(v2,vv2);
        distortion_estimate(vv1,vv2,d);
        tipl::upsample_with_padding(d,d,geo);
        d *= 2.0f;
        tipl::filter::gaussian(d);
    }
    else
        d.resize(geo);
    int n = v1.width();
    tipl::image<float,3> old_d(geo),v(geo),new_g(geo),j1(geo),j2(geo);
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
        tipl::minus(j1,v1);
        tipl::minus(j2,v2);

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
            tipl::image<float,3> g1(geo),g2(geo);
            tipl::gradient(j1.begin(),j1.end(),g1.begin(),2,1);
            tipl::gradient(j2.begin(),j2.end(),g2.begin(),2,1);
            for(int i = 0;i < g1.size();++i)
                g1[i] = -g1[i];
            // sample gradient
            m.sample_gradient(g1,g2,new_g);
            old_d = d;
        }
        tipl::multiply_constant(new_g,s);
        tipl::add(d,new_g);
        tipl::lower_threshold(d,0.0f);
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
public:
    std::vector<tipl::image<unsigned short,3> > new_dwi;//used in rotated volume

public:
    Voxel voxel;
    std::string file_name,error_msg;
    gz_mat_read mat_reader;
public:
    // untouched b-table and DWI from SRC file (the ones in Voxel class will be sorted
    std::vector<tipl::vector<3,float> > src_bvectors;
    bool has_image_rotation = false;
    tipl::matrix<3,3,float> src_bvectors_rotate;
public:
    std::vector<float> src_bvalues;
    std::vector<const unsigned short*> src_dwi_data;
    tipl::image<float,3> dwi_sum;
    tipl::image<unsigned char, 3>dwi;
    std::shared_ptr<ImageModel> study_src;
    void draw_mask(tipl::color_image& buffer,int position);
    void calculate_dwi_sum(void);
    void remove(unsigned int index);
    void pre_dti(void);
    std::string check_b_table(void);
public:
    std::vector<unsigned int> shell;
    void calculate_shell(void);
    bool is_dsi_half_sphere(void);
    bool is_dsi(void);
    bool need_scheme_balance(void);
    bool is_multishell(void);
    void get_report(std::string& report);
public:
    std::vector<std::pair<int,int> > get_bad_slices(void);
    float quality_control_neighboring_dwi_corr(void);
    bool is_human_data(void) const;

    void flip_b_table(const unsigned char* order);
    void flip_b_table(unsigned char dim);
    void swap_b_table(unsigned char dim);
    void flip_dwi(unsigned char type);
    void rotate_one_dwi(unsigned int dwi_index,const tipl::transformation_matrix<double>& affine);
    void rotate(const tipl::image<float,3>& ref,
                const tipl::transformation_matrix<double>& affine,
                const tipl::image<tipl::vector<3>,3>& cdm_dis = tipl::image<tipl::vector<3>,3>(),
                bool super_resolution = false);
    bool rotate_to_mni(void);
    void trim(void);
    void distortion_correction(const ImageModel& rhs);
    bool compare_src(const char* file_name);
public:
    bool command(std::string cmd,std::string param = "");
public:
    bool load_from_file(const char* dwi_file_name);
    void save_fib(const std::string& ext);
    void save_to_file(gz_mat_write& mat_writer);
    bool save_to_nii(const char* nifti_file_name) const;
    bool save_b0_to_nii(const char* nifti_file_name) const;
    bool save_dwi_sum_to_nii(const char* nifti_file_name) const;
    bool save_b_table(const char* file_name) const;
    bool save_bval(const char* file_name) const;
    bool save_bvec(const char* file_name) const;
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
        // Copy SRC b-table to voxel b-table and sort it
        voxel.load_from_src(*this);
        voxel.CreateProcesses<ProcessType>();
        voxel.init();
        voxel.run();
        return !prog_aborted();
    }
    const char* reconstruction(void);


};

const char* odf_average(const char* out_name,std::vector<std::string>& file_names);

#endif//IMAGE_MODEL_HPP
