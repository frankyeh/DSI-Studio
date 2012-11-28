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
class TractModel;

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
    bool load_template(ODFModel* fib_file_);
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
    std::auto_ptr<MatFile> single_subject;
    std::vector<std::vector<float> > greater,lesser;
    std::vector<std::vector<short> > greater_dir,lesser_dir;
public:
    std::vector<const float*> greater_ptr,lesser_ptr;
    std::vector<const short*> greater_dir_ptr,lesser_dir_ptr;
    void single_subject_percentile(const std::vector<float>& cur_subject_data);
    bool single_subject_analysis(const char* file_name);
};

#endif // VBC_DATABASE_H
