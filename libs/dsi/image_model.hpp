#ifndef IMAGE_MODEL_HPP
#define IMAGE_MODEL_HPP
#include "basic_voxel.hpp"
struct distortion_map{
    const float pi_2 = 3.14159265358979323846f/2.0f;
    tipl::image<3,int> i1,i2;
    tipl::image<3> w1,w2;
    void operator=(const tipl::image<3>& d)
    {
        int n = d.width();
        i1.resize(d.shape());
        i2.resize(d.shape());
        w1.resize(d.shape());
        w2.resize(d.shape());
        int max_n = n-1;
        tipl::adaptive_par_for(d.height()*d.depth(),[&](int pos)
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
                float p1 = std::max<float>(0.0f,std::min<float>(float(i)+d[i],max_n));
                float p2 = std::max<float>(0.0f,std::min<float>(float(i)-d[i],max_n));
                i1p[i] = int(p1);
                i2p[i] = int(p2);
                w1p[i] = p1-std::floor(p1);
                w2p[i] = p2-std::floor(p2);
            }
        });
    }
    void calculate_displaced(tipl::image<3>& j1,
                             tipl::image<3>& j2,
                             const tipl::image<3>& v)
    {
        int n = v.width();
        j1.clear();
        j2.clear();
        j1.resize(v.shape());
        j2.resize(v.shape());
        tipl::adaptive_par_for(v.height()*v.depth(),[&](int pos)
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

    void calculate_original(const tipl::image<3>& v1,
                const tipl::image<3>& v2,
                tipl::image<3>& v)
    {
        int n = v1.width();
        int n2 = n + n;
        int block2 = n*n;
        v.clear();
        v.resize(v1.shape());
        tipl::adaptive_par_for(v1.height()*v1.depth(),[&](int pos)
        {
            pos *= n;
            const int* i1p = &i1[0]+pos;
            const int* i2p = &i2[0]+pos;
            const float* w1p = &w1[0]+pos;
            const float* w2p = &w2[0]+pos;
            std::vector<float> M(size_t(block2*2)); // a col by col + col matrix
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
            std::vector<float> y;
            y.resize(size_t(n2));
            std::copy(v1p,v1p+n,y.begin());
            std::copy(v2p,v2p+n,y.begin()+n);
            tipl::mat::pseudo_inverse_solve(&M[0],&y[0],&v[0]+pos,tipl::shape<2>(uint32_t(n),uint32_t(n2)));
        });
    }
    void sample_gradient(const tipl::image<3>& g1,
                         const tipl::image<3>& g2,
                         tipl::image<3>& new_g)
    {
        int n = g1.width();
        new_g.clear();
        new_g.resize(g1.shape());
        tipl::adaptive_par_for(g1.height()*g1.depth(),[&](int pos)
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


template<typename image_type>
void distortion_estimate(const image_type& v1,const image_type& v2,
                         image_type& d)
{
    tipl::shape<3> geo(v1.shape());
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
    tipl::image<3> old_d(geo),v(geo),new_g(geo),j1(geo),j2(geo);
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
        for(size_t i = 0;i < j1.size();++i)
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
            tipl::image<3> g1(geo),g2(geo);
            tipl::gradient(j1.begin(),j1.end(),g1.begin(),2,1);
            tipl::gradient(j2.begin(),j2.end(),g2.begin(),2,1);
            for(size_t i = 0;i < g1.size();++i)
                g1[i] = -g1[i];
            // sample gradient
            m.sample_gradient(g1,g2,new_g);
            old_d = d;
        }
        tipl::multiply_constant(new_g,s);
        tipl::add(d,new_g);
        tipl::lower_threshold(d,0.0f);
        for(size_t i = 0,pos = 0;i < geo.depth()*geo.height();++i,pos+=size_t(n))
        {
            d[pos] = 0.0f;
            d[pos+n-1] = 0.0f;
        }
    }
}


class DwiHeader;
class fib_data;
struct src_data
{
    src_data(void){}
    src_data(const src_data&) = delete;
    src_data operator=(const src_data&) = delete;

public:
    std::vector<tipl::image<3,unsigned short> > new_dwi; //used in rotated volume
    std::vector<tipl::image<3,unsigned short> > nifti_dwi; // if load directly from nifti
public:
    Voxel voxel;
    std::string file_name,output_file_name;
    mutable std::string error_msg;
    tipl::io::gz_mat_read mat_reader;
    std::string topup_eddy_report;
    bool apply_mask = false;
private:
    auto working_path(void) const
    {
        auto parent_path = std::filesystem::path(file_name).parent_path().string();
        if(parent_path.empty())
            return std::string();
        else
            return parent_path + "/";
    }
    auto acqparam_file(void) const{return file_name + ".topup.acqparams.txt";}
    auto temp_nifti(void)  const{return file_name + ".nii.gz";}
    auto corrected_output(void) const{return file_name + ".corrected";}
    auto corrected_file(void) const{return file_name + ".corrected.nii.gz";}
    auto topup_output(void) const
    {
        auto stem = std::filesystem::path(file_name).stem().string();
        std::replace(stem.begin(),stem.end(),'.','_');
        return working_path() + stem;
    }
    auto topup_result(void) const {return topup_output() + "_fieldcoef.nii.gz";}
public:
    std::vector<tipl::vector<3,float> > src_bvectors;
public:
    std::vector<float> src_bvalues;
    std::vector<const unsigned short*> src_dwi_data;
    tipl::image<3,unsigned char>dwi;
    auto dwi_at(size_t index) {return tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);}
    auto dwi_at(size_t index) const {return tipl::make_image(src_dwi_data[index],voxel.dim);}
public:
    void draw_mask(tipl::color_image& buffer,int position);
    void calculate_dwi_sum(bool update_mask);
    bool mask_from_unet(void);
    bool mask_from_template(void);
    void remove(unsigned int index);
    std::shared_ptr<fib_data> get_template_fib(tipl::affine_transform<float>& arg);
    bool check_b_table(bool use_template = true);
    bool load_intro(const std::string& file_name);
public:
    std::string get_report(void);
public:
    std::vector<std::pair<size_t,size_t> > get_bad_slices(void);
    std::pair<float,float> quality_control_neighboring_dwi_corr(void);
    float dwi_contrast(void);
public:
    bool is_human_data(void) const;
    int long_axis_direction(void);
    int symmetric_axis_direction(void);
    int64_t bottom_top_difference(void);
    int64_t anterior_posterior_difference(void);
    void correction_axis(void);
public:
    std::vector<size_t> get_sorted_dwi_index(void);
    void flip_b_table(const unsigned char* order);
    void flip_dwi(unsigned char type);
    void rotate_one_dwi(unsigned int dwi_index,const tipl::transformation_matrix<double>& affine);
    void rotate(const tipl::shape<3>& new_geo,
                const tipl::vector<3>& new_vs,
                const tipl::transformation_matrix<double>& affine,
                const tipl::image<3,tipl::vector<3> >& cdm_dis = tipl::image<3,tipl::vector<3> >());
    void resample(float nv);
    void smoothing(void);
    bool align_acpc(float reso);
    void crop(tipl::shape<3> range_min,tipl::shape<3> range_max);
    void trim(void);
    bool correct_motion(void);
    bool add_other_image(const std::string& name,const std::string& filename);
public:
    std::shared_ptr<src_data> rev_pe_src;
    tipl::shape<3> topup_size;
    void setup_topup_eddy_volume(void);
    //bool distortion_correction(const std::string& file_name);
    std::string find_topup_reverse_pe(void);
    bool get_rev_pe(std::string other_src);
    bool run_topup(void);
    bool run_applytopup(std::string exec = std::string());
    bool run_eddy(std::string exec = std::string());
    bool load_existing_corrections(void);
private:
    bool read_b0(std::vector<tipl::image<3> >& b0) const;
    bool run_plugin(std::string program_name,std::string key_word,
                    size_t total_keyword_count,std::vector<std::string> param,std::string working_dir,std::string exec = std::string());
    bool generate_topup_b0_acq_files(std::vector<tipl::image<3> >& b0,
                                     std::vector<tipl::image<3> >& rev_b0,
                                     std::string& b0_appa_file,
                                     std::string& report);
    bool load_topup_eddy_result(void);
public:
    bool command(std::string cmd,std::string param = "");
    bool run_steps(const std::string& reg_file_name,const std::string& steps);
public:
    bool load_from_file(std::vector<std::shared_ptr<DwiHeader> >& dwi_files,bool sort_btable);
    bool load_from_file(const std::vector<std::string>& nii_names,bool need_bval_bvec);
    bool load_from_file(const std::string& dwi_file_name);
    bool save_to_file(const std::string& dwi_file_name);
    bool save_nii_for_applytopup_or_eddy(bool include_rev) const;
    bool save_mask_nii(const std::string& nifti_file_name) const;
    bool save_b0_to_nii(const std::string& nifti_file_name) const;
    bool save_dwi_sum_to_nii(const std::string& nifti_file_name) const;
    bool save_b_table(const std::string& file_name) const;
    bool save_bval(const std::string& file_name) const;
    bool save_bvec(const std::string& file_name) const;
public:
    template<typename ...ProcessList>
    bool reconstruct2(const char* prog_title)
    {
        // initialization
        voxel.load_from_src(*this);
        if(!voxel.init_process<ProcessList...>())
        {
            error_msg = "reconstruction canceled";
            return false;
        }
        // reconstruction

        try
        {
            if(voxel.run(prog_title))
                return true;
            error_msg = "reconstruction canceled";
            return false;
        }
        catch(std::exception& error)
        {
            error_msg = error.what();
        }
        catch(...)
        {
            error_msg = "unknown error";
        }
        return false;
    }
    void check_output_file_name(void);
    bool save_fib(void);
    bool reconstruction(void);
    bool reconstruction_hist(void);



};

#endif//IMAGE_MODEL_HPP
