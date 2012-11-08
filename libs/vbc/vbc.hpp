#ifndef VBC_HPP
#define VBC_HPP
#include <vector>
#include <iostream>
#include "image/image.hpp"
#include "mat_file.hpp"
#include "prog_interface_static_link.h"
#include <boost/math/distributions/normal.hpp>
#include <boost/thread/thread.hpp>


class ODFModel;
class TractModel;



class vbc{
public:
    unsigned char num_fiber;
    std::vector<std::vector<short> > findex;
    std::vector<std::vector<float> > fa;
    std::vector<image::vector<3,float> > vertices;
    std::vector<std::vector<float> > vertices_cos;

    image::geometry<3> dim;
    bool load_fiber_template(const char* filename);
public://odf
    std::vector<unsigned int> odf_bufs_size;
    std::vector<std::vector<unsigned int> > index_mapping;
    // 0: odf_blocks 1:voxel 2:findex 3: subject statistics (mean1, var1,mean2 var2)
    std::vector<std::vector<std::vector<std::vector<float> > > > subject_odfs;
    unsigned int total_num_subjects,num_files1,num_files2;
    // num_files1 = total_num : trend test
    // num_files1 = 1 : single subject test
    // num_files1 > 1 : group-wise test
    const char* load_subject_data(const std::vector<std::string>& file_names,unsigned int num_files1);
public:
    std::auto_ptr<ODFModel> fib_file;
    void set_fib(bool greater,const std::vector<std::vector<float> >& dif);
public:

    void output_greater_lesser_mapping(const char* file_name,float qa_threshold);
    void calculate_statistics(float qa_threshold,std::vector<std::vector<float> >& vbc,unsigned int is_null) const;
    void run_tracking(float t_threshold,std::vector<std::vector<float> > &tracts);
public:
    std::auto_ptr<boost::thread_group> threads;
    bool terminated;
    boost::mutex lock_function;
    unsigned int cur_prog,total_prog;
public:
    static const unsigned int max_length = 1000;
    std::vector<unsigned int> length_dist;
    void run_thread(unsigned int thread_count,unsigned int thread_id,
                    unsigned int permutation_num,float qa_threshold,float t_threshold);
    void calculate_null(unsigned int num_thread,unsigned int size,float qa_threshold,float t_threshold);
    void fdr_select_tracts(float fdr,std::vector<std::vector<float> > &tracts);
    bool fdr_tracking(const char* file_name,float qa_threshold,float t_threshold,float fdr,bool greater);
public:
    vbc(void)
    {

    }

    ~vbc(void);
};



#endif // VBC_HPP
