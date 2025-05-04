#ifndef GROUP_CONNECTOMETRY_DB_H
#define GROUP_CONNECTOMETRY_DB_H
#include <vector>
#include <iostream>
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

    ~group_connectometry_analysis(){clear();}
    void wait(size_t index = 0);
    void clear(void);
public:
    bool load_database(const char* database_name);
public:// database information
    float fiber_threshold;
public:
    void calculate_adjusted_qa(stat_model& info);
    void calculate_spm(connectometry_result& data,stat_model& info);
private: // single subject analysis result
    int run_track(std::shared_ptr<tracking_data> fib,std::vector<std::vector<float> >& track,
                  unsigned int seed_count,unsigned int random_seed,unsigned int thread_count = 1);
public:// for FDR analysis
    std::vector<std::thread> threads;
    std::vector<unsigned int> tract_count_inc_null;
    std::vector<unsigned int> tract_count_dec_null;
    std::vector<unsigned int> tract_count_inc;
    std::vector<unsigned int> tract_count_dec;
    std::vector<float> fdr_inc,fdr_dec;
    unsigned int prog;// 0~100
    bool terminated = false;
    bool no_tractogram = false;
    bool region_pruning = true;
    unsigned int preprocess = 0;
public:
    std::shared_ptr<RoiMgr> roi_mgr;
    void exclude_cerebellum(void);
public:
    std::string output_file_name;
    int seed_count;
    std::mutex lock_add_tracks,lock_add_null_track;
    std::shared_ptr<TractModel> inc_track,dec_track,pos_null_corr_track,neg_null_corr_track;
public:// Multiple regression
    std::shared_ptr<stat_model> model;
    std::shared_ptr<connectometry_result> spm_map;
    std::vector<std::vector<float> > population_value_adjusted;
    std::string index_name,hypothesis_inc,hypothesis_dec;
    float t_threshold,rho_threshold;
    unsigned int length_threshold_voxels;
    float fdr_threshold;
    unsigned int tip_iteration;
    std::string foi_str;
    std::string get_file_post_fix(void);
    void run_permutation_multithread(unsigned int id,unsigned int thread_count,unsigned int permutation_count);
    void run_permutation(unsigned int thread_count,unsigned int permutation_count);
    void calculate_FDR(void);
    void save_result(void);
    void generate_report(std::string& output);
};

#endif // VBC_DATABASE_H
