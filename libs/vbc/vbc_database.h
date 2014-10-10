#ifndef VBC_DATABASE_H
#define VBC_DATABASE_H
#include <vector>
#include <iostream>
#include "image/image.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/thread/thread.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "libs/tracking/tract_model.hpp"


class FibData;
class fiber_orientations;
class TractModel;

struct fib_data{
    std::vector<std::vector<float> > greater,lesser;
    std::vector<std::vector<short> > greater_dir,lesser_dir;
    std::vector<const float*> greater_ptr,lesser_ptr;
    std::vector<const short*> greater_dir_ptr,lesser_dir_ptr;
public:
    void initialize(FibData* fib_file);
    void add_greater_lesser_mapping_for_tracking(FibData* fib_file);
};



/*
    float y[] = {1,2,2,2,3};
    float X[] = {1,1,1,
                  1,2,8,
                  1,3,27,
                  1,4,64,
                  1,5,125};

    float b[3]={0,0,0};
    float t[3]={0,0,0};
    // b = 0.896551724, 0.33646813, 0.002089864
    multiple_regression<float> m;
    m.set_variables(X,3,5);
    m.regress(y,b,t);
 */
template<typename value_type>
class multiple_regression{
    // the subject data are stored in each row
    std::vector<value_type> X,Xt,XtX;
    std::vector<value_type> X_cov;
    std::vector<int> piv;
    unsigned int feature_count;
    unsigned int subject_count;
public:
    multiple_regression(void){}
    template<typename iterator>
    bool set_variables(iterator X_,
                       unsigned int feature_count_,
                       unsigned int subject_count_)
    {
        feature_count = feature_count_;
        subject_count = subject_count_;
        X.resize(feature_count*subject_count);
        std::copy(X_,X_+X.size(),X.begin());
        Xt.resize(X.size());
        image::matrix::transpose(&*X.begin(),&*Xt.begin(),image::dyndim(subject_count,feature_count));

        XtX.resize(feature_count*feature_count); // trans(x)*y    p by p
        image::matrix::product_transpose(&*Xt.begin(),&*Xt.begin(),
                                         &*XtX.begin(),
                                         image::dyndim(feature_count,subject_count),
                                         image::dyndim(feature_count,subject_count));
        piv.resize(feature_count);
        image::matrix::lu_decomposition(&*XtX.begin(),&*piv.begin(),image::dyndim(feature_count,feature_count));


        // calculate the covariance
        {
            X_cov = Xt;
            std::vector<value_type> c(feature_count),d(feature_count);
            if(!image::matrix::lq_decomposition(&*X_cov.begin(),&*c.begin(),&*d.begin(),image::dyndim(feature_count,subject_count)))
                return false;
            image::matrix::lq_get_l(&*X_cov.begin(),&*d.begin(),&*X_cov.begin(),
                                    image::dyndim(feature_count,subject_count));
        }


        // make l a squre matrix, get rid of the zero part
        for(unsigned int row = 1,pos = subject_count,pos2 = feature_count;row < feature_count;++row,pos += subject_count,pos2 += feature_count)
            std::copy(X_cov.begin() + pos,X_cov.begin() + pos + feature_count,X_cov.begin() + pos2);

        image::matrix::inverse_lower(&*X_cov.begin(),image::dyndim(feature_count,feature_count));

        image::square(X_cov.begin(),X_cov.begin()+feature_count*feature_count);

        // sum column wise
        for(unsigned int row = 1,pos = feature_count;row < feature_count;++row,pos += feature_count)
            image::add(X_cov.begin(),X_cov.begin()+feature_count,X_cov.begin()+pos);
        image::square_root(X_cov.begin(),X_cov.begin()+feature_count);

        std::vector<value_type> new_X_cov(X_cov.begin(),X_cov.begin()+feature_count);
        new_X_cov.swap(X_cov);
        return true;
    }
    /*
     *       y0       x00 ...x0p
     *       y1       x10 ...x1p    b0
     *     [ :  ] = [  :        ][  :  ]
     *       :         :            bp
     *       yn       xn0 ...xnp
     *
     **/

    template<typename iterator1,typename iterator2,typename iterator3>
    void regress(iterator1 y,iterator2 b,iterator3 t) const
    {
        std::vector<value_type> xty(feature_count); // trans(x)*y    p by 1
        image::matrix::vector_product(&*Xt.begin(),y,&*xty.begin(),image::dyndim(feature_count,subject_count));
        image::matrix::lu_solve(&*XtX.begin(),&*piv.begin(),&*xty.begin(),b,
                                image::dyndim(feature_count,feature_count));

        // calculate residual
        std::vector<value_type> y_(subject_count);
        image::matrix::left_vector_product(&*Xt.begin(),b,&*y_.begin(),image::dyndim(feature_count,subject_count));
        image::minus(y_.begin(),y_.end(),y);
        image::square(y_);
        value_type rmse = std::sqrt(std::accumulate(y_.begin(),y_.end(),0.0)/(subject_count-feature_count));

        for(unsigned int index = 0;index < feature_count;++index)
            t[index] = b[index]/X_cov[index]/rmse;
    }

};

class stat_model{
public:
    mutable boost::mt19937 generator;
    mutable boost::uniform_int<int> uniform_rand;
    mutable boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rand_gen;
public:
    stat_model(void):generator(0),uniform_rand(),rand_gen(generator,uniform_rand){}
public:
    unsigned int type;
public: // group
    std::vector<int> label;
    unsigned int group1_count,group2_count;
public: // multiple regression
    std::vector<double> X;
    unsigned int feature_count;
    unsigned int study_feature;
    multiple_regression<double> mr;
public: // individual
    const float* individual_data;
public: // paired
    std::vector<unsigned int> pre,post;
public:
    bool resample(const stat_model& rhs,std::vector<unsigned int>& permu,bool null);
    bool pre_process(void);
    double operator()(const std::vector<double>& population,unsigned int pos) const;
    void clear(void)
    {
        label.clear();
        X.clear();
    }
    const stat_model& operator=(const stat_model& rhs)
    {
        type = rhs.type;
        label = rhs.label;
        group1_count = rhs.group1_count;
        group2_count = rhs.group2_count;
        X = rhs.X;
        feature_count = rhs.feature_count;
        study_feature = rhs.study_feature;
        mr = rhs.mr;
        individual_data = rhs.individual_data;
        pre = rhs.pre;
        post = rhs.post;
        return *this;
    }
};

class vbc_database
{
public:
    std::auto_ptr<FibData> handle;
    std::string report;
    mutable std::string error_msg;
    vbc_database();
    ~vbc_database(){clear_thread();}
private:// template information
    image::geometry<3> dim;
    unsigned int num_fiber;
    std::vector<const short*> findex;
    std::vector<const float*> fa;
    std::vector<unsigned int> vi2si;
    std::vector<unsigned int> si2vi;
    std::vector<image::vector<3,float> > vertices;
    unsigned int half_odf_size;
    float fiber_threshold;
    bool is_consistent(gz_mat_read& mat_reader) const;
    void read_template(void);
public:
    bool create_database(const char* templat_name);
    bool load_database(const char* database_name);
private:// database information
    std::string subject_report;
    std::vector<std::string> subject_names;
    unsigned int num_subjects;
    // 0: subject index 1:findex 2.s_index (fa > 0)
    std::vector<std::vector<float> > subject_qa_buffer;
    std::vector<const float*> subject_qa;
    std::vector<float> R2;
    bool sample_odf(gz_mat_read& mat_reader,std::vector<float>& data);
public:
    unsigned int subject_count(void)const{return num_subjects;}
    const std::string& subject_name(unsigned int index)const{return subject_names[index];}
    float subject_R2(unsigned int index)const{return R2[index];}
    bool load_subject_files(const std::vector<std::string>& file_names,
                            const std::vector<std::string>& names);
    void save_subject_data(const char* output_name) const;
    void get_data_at(unsigned int index,unsigned int fib,std::vector<float>& data) const;
    void get_subject_slice(unsigned int subject_index,unsigned int z_pos,image::basic_image<float,2>& slice) const;
    //bool calculate_individual_affected_tracks(const char* file_name,
    //                                          std::vector<std::vector<std::vector<float> > >& greater,
    //                                          std::vector<std::vector<std::vector<float> > >& lesser);

private: // single subject analysis result
    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data);
    void run_track(const fiber_orientations& fib,std::vector<std::vector<float> >& track,float seed_ratio = 1.0);


public:// for FDR analysis
    std::auto_ptr<boost::thread_group> threads;
    std::vector<unsigned int> subject_greater_null;
    std::vector<unsigned int> subject_lesser_null;
    std::vector<unsigned int> subject_greater;
    std::vector<unsigned int> subject_lesser;
    std::vector<float> fdr_greater,fdr_lesser;
    unsigned int total_count,total_count_null;
    unsigned int permutation_count;
    bool terminated;
public:
    std::vector<std::string> trk_file_names;
    unsigned int pruning;
    unsigned int length_threshold;
    boost::mutex lock_resampling,lock_greater_tracks,lock_lesser_tracks;
    boost::ptr_vector<TractModel> greater_tracks;
    boost::ptr_vector<TractModel> lesser_tracks;
    void save_tracks_files(std::vector<std::string>&);
public:// routine for calculate SPM
    void calculate_spm(const stat_model& info,fib_data& data,const std::vector<unsigned int>& permu);
public:// Individual analysis
    std::vector<std::vector<float> > individual_data;
    bool read_subject_data(const std::vector<std::string>& files,std::vector<std::vector<float> >& data);
public:// Multiple regression
    stat_model model;
    float tracking_threshold;
    void run_permutation_multithread(unsigned int id);
    void run_permutation(unsigned int thread_count);
    void calculate_FDR(void);
    void clear_thread(void);
private:


public:
};

#endif // VBC_DATABASE_H
