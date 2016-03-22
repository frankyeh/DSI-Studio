#ifndef CONNECTOMETRY_DB_H
#define CONNECTOMETRY_DB_H
#include <vector>
#include <string>
#include "gzip_interface.hpp"
#include "image/image.hpp"
class FibData;
class connectometry_db
{
public:
    FibData* handle;
    std::string subject_report;
    std::vector<std::string> subject_names;
    unsigned int num_subjects;
    std::vector<float> R2;
    std::vector<const float*> subject_qa;
    unsigned int subject_qa_length;
    std::vector<float> subject_qa_sd;
    image::basic_image<unsigned int,3> vi2si;
    std::vector<unsigned int> si2vi;
    std::vector<std::vector<float> > subject_qa_buf;// merged from other db
public:
    connectometry_db():handle(0),num_subjects(0){;}
    bool has_db(void)const{return !num_subjects;}
    void read_db(FibData* handle);
    void remove_subject(unsigned int index);
    void calculate_si2vi(void);
    bool sample_odf(gz_mat_read& m,std::vector<float>& data);
    bool sample_index(gz_mat_read& m,std::vector<float>& data,const char* index_name);
    bool is_consistent(gz_mat_read& m);
    bool load_subject_files(const std::vector<std::string>& file_names,
                            const std::vector<std::string>& subject_names_,
                            const char* index_name);
    void get_subject_vector(std::vector<std::vector<float> >& subject_vector,
                            const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp) const;
    void get_subject_vector(unsigned int subject_index,std::vector<float>& subject_vector,
                            const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp) const;
    void get_dif_matrix(std::vector<float>& matrix,const image::basic_image<int,3>& cerebrum_mask,float fiber_threshold,bool normalize_fp);
    void save_subject_vector(const char* output_name,
                             const image::basic_image<int,3>& cerebrum_mask,
                             float fiber_threshold,
                             bool normalize_fp) const;
    void save_subject_data(const char* output_name);
    void get_subject_slice(unsigned int subject_index,unsigned char dim,unsigned int pos,
                            image::basic_image<float,2>& slice) const;
    void get_subject_fa(unsigned int subject_index,std::vector<std::vector<float> >& fa_data) const;
    void get_data_at(unsigned int index,unsigned int fib_index,std::vector<double>& data,bool normalize_qa) const;
    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data);
    bool get_qa_profile(const char* file_name,std::vector<std::vector<float> >& data);
    bool is_db_compatible(const connectometry_db& rhs);
    void read_subject_qa(std::vector<std::vector<float> >&data) const;
    bool add_db(const connectometry_db& rhs);
};

#endif // CONNECTOMETRY_DB_H
