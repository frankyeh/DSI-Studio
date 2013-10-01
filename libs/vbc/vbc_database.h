#ifndef VBC_DATABASE_H
#define VBC_DATABASE_H
#include <vector>
#include <iostream>
#include "image/image.hpp"
#include "gzip_interface.hpp"
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
    std::auto_ptr<ODFModel> handle;
    mutable std::string error_msg;
    vbc_database();
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
private: // single subject analysis result

    bool get_odf_profile(const char* file_name,std::vector<float>& cur_subject_data);
public:
    unsigned int total_greater;
    unsigned int total_lesser;
    unsigned int total;
    void single_subject_analysis(const float* cur_subject_data,float percentile,fib_data& result);
    bool single_subject_analysis(const char* filename,float percentile,fib_data& result);
    //bool single_subject_paired_analysis(const char* file_name1,const char* file_name2);
public:
    bool calculate_individual_distribution(float percentile,
                                           unsigned int length_threshold,
                                           const std::vector<std::string>& files,
                                        std::vector<float>& subject_greater,
                                        std::vector<float>& subject_lesser);
public:
    void run_track(const fiber_orientations& fib,std::vector<std::vector<float> >& track);
    bool save_track_as(const char* file_name,std::vector<std::vector<float> >& track,unsigned int length_threshold);
    void calculate_subject_distribution(float percentile,
                                        const fib_data& data,
                                        std::vector<float>& subject_greater,
                                        std::vector<float>& subject_lesser);
    bool save_subject_distribution(float percentile,
                                   unsigned int length_threshold,
                                   const char* file_name,
                                   const fib_data& data);

public:
    double get_trend_std(const std::vector<float>& data);
    void trend_analysis(const std::vector<float>& data,fib_data& result);
    void group_analysis(const std::vector<int>& label,fib_data& data);
    void trend_analysis(float sqrt_var_S,const std::vector<unsigned int>& permu,fib_data& result);


    void calculate_null_trend_distribution(float sqrt_var_S,float percentile,
                                                   std::vector<float>& subject_greater,
                                                   std::vector<float>& subject_lesser);
    void calculate_null_group_distribution(const std::vector<int>& label,float dif,
                                                   std::vector<float>& subject_greater,
                                                   std::vector<float>& subject_lesser);
private:
    void calculate_null_trend_multithread(unsigned int id,float sqrt_var_S,float percentile,
                                          std::vector<unsigned int>& dist_greater,
                                          std::vector<unsigned int>& dist_lesser,
                                          bool progress,
                                          unsigned int* total_count);
    void calculate_null_group_multithread(unsigned int id,const std::vector<int>& label,float dif,
                                          std::vector<unsigned int>& dist_greater,
                                          std::vector<unsigned int>& dist_lesser,
                                          bool progress,
                                          unsigned int* total_count);

public:
};

#endif // VBC_DATABASE_H
