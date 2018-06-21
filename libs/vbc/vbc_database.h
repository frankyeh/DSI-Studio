#ifndef VBC_DATABASE_H
#define VBC_DATABASE_H
#include <vector>
#include <iostream>
#include "tipl/tipl.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
#include "libs/tracking/tract_model.hpp"


class fib_data;
class tracking;
class TractModel;



class vbc_database
{
public:
    std::shared_ptr<fib_data> handle;
    std::string report;
    mutable std::string error_msg;
    vbc_database();
    ~vbc_database(){clear();}

    void clear(void);
    void wait(void);
public:
    bool create_database(const char* templat_name);
    bool load_database(const char* database_name);
public:// database information
    float fiber_threshold;
    unsigned int voxels_in_threshold;
    bool normalize_qa;
    bool output_resampling;
public:
    void calculate_spm(connectometry_result& data,stat_model& info,bool nqa)
    {
        ::calculate_spm(handle,data,info,fiber_threshold,nqa,terminated);
    }
private: // single subject analysis result
    int run_track(const tracking_data& fib,std::vector<std::vector<float> >& track,
                  float seed_ratio = 1.0,unsigned int thread_count = 1);
public:// for FDR analysis
    std::vector<std::shared_ptr<std::future<void> > > threads;
    std::vector<unsigned int> subject_greater_null;
    std::vector<unsigned int> subject_lesser_null;
    std::vector<unsigned int> subject_greater;
    std::vector<unsigned int> subject_lesser;
    std::vector<float> fdr_greater,fdr_lesser;

    std::vector<unsigned int> seed_greater_null;
    std::vector<unsigned int> seed_lesser_null;
    std::vector<unsigned int> seed_greater;
    std::vector<unsigned int> seed_lesser;
    unsigned int progress;// 0~100
    bool terminated = false;
public:
    std::vector<std::vector<tipl::vector<3,short> > > roi_list;
    std::vector<std::string> roi_name;
    std::vector<float> roi_r_list;
    std::vector<unsigned char> roi_type;
public:
    std::string output_file_name;
    bool has_greater_result,has_lesser_result;
    float seed_ratio;
    std::mutex  lock_resampling,lock_greater_tracks,lock_lesser_tracks;
    std::shared_ptr<TractModel> greater_track;
    std::shared_ptr<TractModel> lesser_track;
    std::shared_ptr<connectometry_result> spm_map;
    std::string greater_tracks_result,lesser_tracks_result;
    void save_tracks_files(void);
public:// Multiple regression
    std::shared_ptr<stat_model> model;
    float tracking_threshold;
    float length_threshold,fdr_threshold;
    unsigned int track_trimming;
    std::string foi_str;
    void run_permutation_multithread(unsigned int id,unsigned int thread_count,unsigned int permutation_count);
    void run_permutation(unsigned int thread_count,unsigned int permutation_count);
    void calculate_FDR(void);
    void generate_report(std::string& output);
};

#endif // VBC_DATABASE_H
