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
    bool is_consistent(MatFile& mat_reader) const;
public:
    bool load_template(const char* templat_name);
    bool load_template(ODFModel* fib_file_);
private:// subject information
    unsigned int num_subjects;
    // 0: subject index 1:findex 2.s_index (fa > 0)
    std::vector<std::vector<float> > subject_qa_buffer;
    std::vector<const float*> subject_qa;
    std::auto_ptr<ODFModel> single_subject;
    bool sample_odf(MatFile& mat_reader,std::vector<float>& data);
public:
    bool load_subject_files(const std::vector<std::string>& file_names);
    void save_subject_data(const char* output_name) const;
    void get_data_at(unsigned int index,unsigned int fib,std::vector<float>& data) const;
public:
    bool single_subject_analysis(const char* file_name);
};

#endif // VBC_DATABASE_H
