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

struct vbc_clustering{
    std::vector<std::vector<float> > dif,t;
};


class vbc{
public:
    unsigned char num_fiber;
    std::vector<const short*> findex;
    std::vector<const float*> fa;
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
    const char* load_subject_data(const std::vector<std::string>& file_names,
                           unsigned int num_files1,float qa_threshold);
public:
    std::auto_ptr<ODFModel> fib_file;

public:

    void calculate_mapping(const char* file_name,
                           float p_value_threshold);

    void calculate_statistics(float p_value_threshold,
                              vbc_clustering& vbc,bool is_null) const;
public:
    float angle_threshold_cos;
    void calculate_cluster(
            const vbc_clustering& data,
            std::vector<unsigned int>& group_voxel_index_list,
            std::vector<unsigned int>& group_id_map);

private:
    std::vector<unsigned short> max_mapping;
    std::vector<unsigned int> max_cluster_size;
    std::vector<float> max_statistics;
public:
    std::auto_ptr<boost::thread_group> threads;
    std::vector<vbc_clustering> thread_data;
    bool terminated;
    boost::mutex lock_function;
    unsigned int cur_prog,total_prog;
public:
    std::vector<unsigned int> get_max_cluster_size(void)
    {
        boost::mutex::scoped_lock lock(lock_function);
        {
            std::vector<unsigned int> max_cluster_size_(max_cluster_size);
            return max_cluster_size_;
        }
    }
    std::vector<float> get_max_statistics(void)
    {
        boost::mutex::scoped_lock lock(lock_function);
        {
            std::vector<float> max_statistics_(max_statistics);
            return max_statistics_;
        }
    }
public:
    void run_thread(unsigned int thread_count,unsigned int thread_id,unsigned int permutation_num,float alpha);
    void calculate_permutation(unsigned int num_thread,unsigned int size,float alpha);
public:
    vbc(void):angle_threshold_cos(0.866)
    {

    }

    ~vbc(void);
};



#endif // VBC_HPP
