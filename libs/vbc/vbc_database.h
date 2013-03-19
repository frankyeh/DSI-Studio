#ifndef VBC_DATABASE_H
#define VBC_DATABASE_H
#include <vector>
#include <iostream>
#include "image/image.hpp"
#include "mat_file.hpp"
#include "prog_interface_static_link.h"
#include <boost/math/distributions/normal.hpp>
#include <boost/thread/thread.hpp>


class ODFModel;
class fiber_orientations;
class TractModel;

struct fib_data{
    std::vector<std::vector<float> > greater,lesser;
    std::vector<std::vector<short> > greater_dir,lesser_dir;
    std::vector<const float*> greater_ptr,lesser_ptr;
    std::vector<const short*> greater_dir_ptr,lesser_dir_ptr;
public:
    void initialize(ODFModel* fib_file);
    void add_greater_lesser_mapping_for_tracking(ODFModel* fib_file);
};

class vbc_database
{
public:
    mutable std::string error_msg;
    vbc_database();
private:// template information
    std::auto_ptr<ODFModel> fib_file_buffer;
    ODFModel* fib_file;
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
    bool is_consistent(MatFile& mat_reader) const;
public:
    bool load_template(const char* templat_name);
    void read_template(ODFModel* fib_file_);
    bool read_database(ODFModel* fib_file_);
private:// database information
    std::vector<std::string> subject_names;
    unsigned int num_subjects;
    // 0: subject index 1:findex 2.s_index (fa > 0)
    std::vector<std::vector<float> > subject_qa_buffer;
    std::vector<const float*> subject_qa;
    std::vector<float> R2;
    bool sample_odf(MatFile& mat_reader,std::vector<float>& data);
public:
    unsigned int subject_count(void)const{return num_subjects;}
    const std::string& subject_name(unsigned int index)const{return subject_names[index];}
    float subject_R2(unsigned int index)const{return R2[index];}
    bool load_subject_files(const std::vector<std::string>& file_names,
                            const std::vector<std::string>& names);
    void save_subject_data(const char* output_name) const;
    void get_data_at(unsigned int index,unsigned int fib,std::vector<float>& data) const;
    void get_subject_slice(unsigned int subject_index,unsigned int z_pos,image::basic_image<float,2>& slice) const;
private: // single subject analysis result

    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data);
public:
    bool single_subject_analysis(const char* file_name,fib_data& result);
    //bool single_subject_paired_analysis(const char* file_name1,const char* file_name2);
public:
    void run_span(const fiber_orientations& fib,std::vector<std::vector<float> >& span);
    void calculate_span_distribution(const fiber_orientations& fib,std::vector<unsigned int>& dist);
    void calculate_subject_distribution(float percentile,const fib_data& data,
                                        std::vector<float>& subject_greater,
                                        std::vector<float>& subject_lesser);
    bool calculate_group_distribution(float percentile,const std::vector<std::string>& files,
                                        std::vector<float>& subject_greater,
                                        std::vector<float>& subject_lesser);
    void calculate_null_distribution(float percentile,
                                     std::vector<float>& subject_greater,
                                     std::vector<float>& subject_lesser);
    void calculate_subject_fdr(float percentile,const fib_data& result,std::vector<std::vector<float> >& spans,
                                 std::vector<float>& fdr);
public:
    void tend_analysis(const std::vector<unsigned int>& permu,fib_data& result);
    void calculate_null_trend_distribution(float percentile,
                                                   std::vector<float>& subject_greater,
                                                   std::vector<float>& subject_lesser);

public:
};

#endif // VBC_DATABASE_H
