#ifndef CONNECTOMETRY_DB_H
#define CONNECTOMETRY_DB_H
#include <vector>
#include <string>
#include "zlib.h"
#include "TIPL/tipl.hpp"
class fib_data;
class connectometry_db
{
public:
    fib_data* handle = nullptr;
    std::string report,subject_report;
    mutable std::string error_msg;
    unsigned int num_subjects = 0;
    bool modified = false;

public: // demographic information
    std::vector<std::string> titles;
    std::vector<std::string> items;
    std::vector<size_t> feature_location;
    std::vector<std::string> feature_titles;
    std::vector<bool> feature_selected,feature_is_float;
    std::vector<double> X;
    std::string demo;
    bool parse_demo(const std::string& filename);
    bool parse_demo(void);
public:// subject specific data
    std::vector<std::string> subject_names;
    std::vector<float> R2;
    std::vector<const float*> subject_qa;
    bool is_longitudinal = false;
    unsigned char longitudinal_filter_type = 0; // 0: no filter 1: only increased value 2:only decreased values
public:
    std::list<std::vector<float> > subject_qa_buf;// merged from other db
    unsigned int subject_qa_length = 0;
    tipl::image<3,size_t> vi2si;
    std::vector<size_t> si2vi;
    std::string index_name = "qa";
public://longitudinal studies
    std::vector<std::pair<int,int> > match;
    void calculate_change(unsigned char dif_type,unsigned char filter_type);
public:
    connectometry_db(){}
    bool has_db(void)const{return num_subjects > 0;}
    bool read_db(fib_data* handle);
    void clear(void);
    void remove_subject(unsigned int index);
    void calculate_si2vi(void);
    bool is_odf_consistent(tipl::io::gz_mat_read& m);
    void sample_from_image(tipl::const_pointer_image<3,float> I,
                           const tipl::matrix<4,4>& trans,std::vector<float>& data);
    bool add_subject_file(const std::string& file_name,
                            const std::string& subject_name);
    bool save_db(const char* output_name);
    void get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                            tipl::image<2,float>& slice) const;
    bool get_demo_matched_volume(const std::string& matched_demo,tipl::image<3>& volume) const;
    bool save_demo_matched_image(const std::string& matched_demo,const std::string& filename) const;
    void get_subject_volume(unsigned int subject_index,tipl::image<3>& volume) const;
    void get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data) const;
    bool get_qa_profile(const char* file_name,std::vector<std::vector<float> >& data);
    bool is_db_compatible(const connectometry_db& rhs);
    bool add_db(const connectometry_db& rhs);
    void move_up(int id);
    void move_down(int id);

};



class stat_model{
public:
    std::vector<unsigned int> selected_subject;
    std::vector<unsigned int> resample_order;
    std::vector<unsigned int> permutation_order;
public: // multiple regression
    std::vector<double> X,X_min,X_max,X_range;
    unsigned int x_col_count = 0;
    unsigned int study_feature = 0;
public:
    std::vector<std::string> variables;
    std::vector<bool> variables_is_categorical;
    std::vector<int> variables_min,variables_max;
public:
    tipl::multiple_regression<double> mr;
    // for nonlinear correlation
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
    bool resample(stat_model& rhs,bool null,bool bootstrap,unsigned int seed);
    bool pre_process(void);
    void partial_correlation(std::vector<float>& population) const;
    double operator()(const std::vector<float>& population) const;
    void clear(void)
    {
        X.clear();
    }
    const stat_model& operator=(const stat_model& rhs)
    {
        selected_subject = rhs.selected_subject;
        X = rhs.X;
        X_min = rhs.X_min;
        X_max = rhs.X_max;
        X_range = rhs.X_range;
        x_col_count = rhs.x_col_count;
        study_feature = rhs.study_feature;
        mr = rhs.mr;
        individual_data = rhs.individual_data;
        remove_list = rhs.remove_list;
        cohort_report = rhs.cohort_report;
        return *this;
    }
};



struct connectometry_result{
    std::vector<std::vector<float> > inc,dec;
    std::vector<const float*> inc_ptr,dec_ptr;
    void clear_result(char num_fiber,size_t image_size);
};



#endif // CONNECTOMETRY_DB_H
