#ifndef CONNECTOMETRY_DB_H
#define CONNECTOMETRY_DB_H
#include <vector>
#include <string>
#include "gzip_interface.hpp"
#include "TIPL/tipl.hpp"
class fib_data;
class connectometry_db
{
public:
    fib_data* handle = nullptr;
    std::string report,subject_report,error_msg;
    unsigned int num_subjects = 0;
    bool modified = false;

public: // demographi infomation
    std::vector<std::string> titles;
    std::vector<std::string> items;
    std::vector<size_t> feature_location;
    std::vector<std::string> feature_titles;
    std::vector<bool> feature_selected;
    std::vector<double> X;
    std::string demo;
    bool parse_demo(const std::string& filename);
    bool parse_demo(void);
public:// subject specific data
    std::vector<std::string> subject_names;
    std::vector<float> R2;
    std::vector<const float*> subject_qa;
    std::vector<float> subject_qa_sd;
    bool is_longitudinal = false;
public:
    std::list<std::vector<float> > subject_qa_buf;// merged from other db
    unsigned int subject_qa_length;
    tipl::image<3,unsigned int> vi2si;
    std::vector<unsigned int> si2vi;
    std::string index_name = "qa";
public://longitudinal studies
    std::vector<std::pair<int,int> > match;
    void auto_match(const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp);
    void calculate_change(unsigned char dif_type,bool norm);
public:
    connectometry_db(){}
    bool has_db(void)const{return num_subjects > 0;}
    bool read_db(fib_data* handle);
    void remove_subject(unsigned int index);
    void calculate_si2vi(void);
    bool sample_subject_profile(gz_mat_read& m,std::vector<float>& data);
    bool is_odf_consistent(gz_mat_read& m);
    bool add_subject_file(const std::string& file_name,
                            const std::string& subject_name);
    void get_subject_vector_pos(std::vector<int>& subject_vector_pos,
                                const tipl::image<3,int>& fp_mask,float fiber_threshold) const;
    void get_subject_vector(unsigned int from,unsigned int to,
                            std::vector<std::vector<float> >& subject_vector,
                            const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp) const;
    void get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                            const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp) const;
    void get_dif_matrix(std::vector<float>& matrix,const tipl::image<3,int>& fp_mask,float fiber_threshold,bool normalize_fp);
    void save_subject_vector(const char* output_name,
                             const tipl::image<3,int>& fp_mask,
                             float fiber_threshold,
                             bool normalize_fp) const;
    bool save_db(const char* output_name);
    void get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                            tipl::image<2,float>& slice) const;
    void get_subject_volume(unsigned int subject_index,tipl::image<3>& volume) const;
    void get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data,bool normalize_qa) const;
    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data);
    bool get_qa_profile(const char* file_name,std::vector<std::vector<float> >& data);
    bool is_db_compatible(const connectometry_db& rhs);
    void read_subject_qa(std::vector<std::vector<float> >&data) const;
    bool add_db(const connectometry_db& rhs);
    void move_up(int id);
    void move_down(int id);

};



class stat_model{
public:
    std::vector<unsigned int> subject_index;
public:
    unsigned int type;
public: // group
    std::vector<int> label;
    unsigned int group1_count,group2_count;
public: // multiple regression
    std::vector<double> X,X_min,X_max,X_range;
    unsigned int feature_count = 0;
    unsigned int study_feature = 0;
    std::vector<std::string> variables;
    tipl::multiple_regression<double> mr;
    // for nonlinear correlation
    bool nonparametric = true;
    std::vector<unsigned int> x_study_feature_rank;
    double rank_c = 0;
    void select_variables(const std::vector<char>& sel);
public: // individual
    const float* individual_data;
    float individual_data_sd;
public:
    stat_model(void):individual_data(nullptr){}
    stat_model(const stat_model& rhs){*this = rhs;}
public:
    std::string error_msg;
    std::string cohort_report;
    std::vector<char> remove_list;
    bool select_cohort(connectometry_db& db,
                       std::string select_text);
    bool select_feature(connectometry_db& db,std::string foi_text);
public:
    void read_demo(const connectometry_db& db);
    void remove_subject(unsigned int index);    
    bool resample(stat_model& rhs,bool null,bool bootstrap,unsigned int seed);
    bool pre_process(void);
    double operator()(const std::vector<double>& population,unsigned int pos) const;
    void clear(void)
    {
        label.clear();
        X.clear();
    }
    const stat_model& operator=(const stat_model& rhs)
    {
        subject_index = rhs.subject_index;
        type = rhs.type;
        label = rhs.label;
        group1_count = rhs.group1_count;
        group2_count = rhs.group2_count;
        X = rhs.X;
        X_min = rhs.X_min;
        X_max = rhs.X_max;
        X_range = rhs.X_range;
        feature_count = rhs.feature_count;
        study_feature = rhs.study_feature;
        mr = rhs.mr;
        individual_data = rhs.individual_data;
        nonparametric = rhs.nonparametric;

        remove_list = rhs.remove_list;
        cohort_report = rhs.cohort_report;
        return *this;
    }
};



struct connectometry_result{
    std::vector<std::vector<float> > pos_corr,neg_corr;
    std::vector<const float*> pos_corr_ptr,neg_corr_ptr;
    void remove_old_index(std::shared_ptr<fib_data> handle);
    bool compare(std::shared_ptr<fib_data> handle,
                 const std::vector<const float*>& fa1,const std::vector<const float*>& fa2,
                 unsigned char normalization);
public:
    std::string report;
    std::string error_msg;
    void initialize(std::shared_ptr<fib_data> fib_file);
    void add_mapping_for_tracking(std::shared_ptr<fib_data> handle,const char* t1,const char* t2);
    bool individual_vs_atlas(std::shared_ptr<fib_data> handle,const char* file_name,unsigned char normalization);
    bool individual_vs_db(std::shared_ptr<fib_data> handle,const char* file_name);
    bool individual_vs_individual(std::shared_ptr<fib_data> handle,
                                  const char* file_name1,const char* file_name2,unsigned char normalization);

};

void calculate_spm(std::shared_ptr<fib_data> handle,connectometry_result& data,stat_model& info,
                   float fiber_threshold,bool normalize_qa,bool& terminated);


#endif // CONNECTOMETRY_DB_H
