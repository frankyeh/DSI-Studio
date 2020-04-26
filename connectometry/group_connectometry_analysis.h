#ifndef GROUP_CONNECTOMETRY_DB_H
#define GROUP_CONNECTOMETRY_DB_H
#include <vector>
#include <iostream>
#include "tipl/tipl.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
#include "libs/tracking/tract_model.hpp"


class fib_data;
class tracking;
class TractModel;



class group_connectometry_analysis
{
public:
    std::shared_ptr<fib_data> handle;
    std::string report;
    mutable std::string error_msg;
    group_connectometry_analysis();
    ~group_connectometry_analysis(){clear();}

    void clear(void);
    void wait(void);
public:
    bool create_database(const char* templat_name);
    bool load_database(const char* database_name);
public:// database information
    float fiber_threshold;
    bool normalize_qa;
    bool output_resampling;
public:
    void calculate_spm(connectometry_result& data,stat_model& info,bool nqa)
    {
        ::calculate_spm(handle,data,info,fiber_threshold,nqa,terminated);
    }
private: // single subject analysis result
    int run_track(const tracking_data& fib,std::vector<std::vector<float> >& track,
                  int seed_count,unsigned int thread_count = 1);
public:// for FDR analysis
    std::vector<std::shared_ptr<std::future<void> > > threads;
    std::vector<unsigned int> subject_pos_corr_null;
    std::vector<unsigned int> subject_neg_corr_null;
    std::vector<unsigned int> subject_pos_corr;
    std::vector<unsigned int> subject_neg_corr;
    std::vector<float> fdr_pos_corr,fdr_neg_corr;

    std::vector<unsigned int> seed_pos_corr_null;
    std::vector<unsigned int> seed_neg_corr_null;
    std::vector<unsigned int> seed_pos_corr;
    std::vector<unsigned int> seed_neg_corr;
    unsigned int progress;// 0~100
    bool terminated = false;
public:
    std::shared_ptr<RoiMgr> roi_mgr;
    std::string roi_mgr_text;
    std::string output_roi_suffix;
public:
    std::string output_file_name;
    bool has_pos_corr_result,has_neg_corr_result;
    int seed_count;
    std::mutex  lock_resampling,lock_pos_corr_tracks,lock_neg_corr_tracks;
    std::shared_ptr<TractModel> pos_corr_track;
    std::shared_ptr<TractModel> neg_corr_track;
    std::shared_ptr<connectometry_result> spm_map;
    std::string pos_corr_tracks_result,neg_corr_tracks_result;
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
