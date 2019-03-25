#ifndef NN_CONNECTOMETRY_ANALYSIS_H
#define NN_CONNECTOMETRY_ANALYSIS_H
#include <memory>
#include <string>
#include <vector>
#include <tipl/tipl.hpp>

class fib_data;

class nn_connectometry_analysis
{
public:
    std::shared_ptr<fib_data> handle;
    std::string error_msg,report;
public:
    int skip_slice = 4;
    tipl::image<float,3> It;
    tipl::color_image Ib;
public:
    std::vector<tipl::pixel_index<3> > fp_index;
    std::vector<std::pair<int,int> > fib_pairs;
public:
    bool terminated;
    std::future<void> future;
public:
    tipl::ml::trainer t;
    tipl::ml::network nn;
    tipl::ml::network_data<float> fp_data;
    std::vector<tipl::ml::network_data_proxy<float> > train_data;
    std::vector<tipl::ml::network_data_proxy<float> > test_data;
public:
    float sl_mean,sl_scale;

public:
    std::vector<int> subject_index;
    std::vector<unsigned int> test_seq;
    std::vector<float> test_result;
public:
    std::vector<unsigned int> all_test_seq;
    std::vector<float> all_test_result;
    std::string all_result;
private:
    std::mutex lock_result;
    std::vector<float> result_r;
    std::vector<float> result_mae;
    std::vector<float> result_test_miss;
    std::vector<float> result_test_error;
    std::vector<float> result_train_error;
public:
    void clear_results(void);
    bool has_results(void){return !result_train_error.empty();}
    template<typename fun>
    void get_results(fun f)
    {
        std::lock_guard<std::mutex> lock(lock_result);
        if(is_regression) // regression
            for(size_t i = 0;i < result_r.size();++i)
                f(i,result_r[i],result_mae[i],result_train_error[i]);
        else
            for(size_t i = 0;i < result_test_error.size();++i)
                f(i,result_test_miss[i],result_test_error[i],result_train_error[i]);

    }
public:
    int foi_index = 0;
    bool is_regression = true;
    int seed_search = 10;
    float otsu = 0.6f;
    float no_data = 9999.0f;
    size_t cv_fold = 10;
    bool normalize_value = false;
public:
    int cur_progress = 0;
    int cur_fold = 0;
    nn_connectometry_analysis(std::shared_ptr<fib_data> handle_);
    bool run(const std::string& net_string);
    void stop(void);
    void get_salient_map(tipl::color_image& I);
    void get_layer_map(tipl::color_image& I);
};

#endif // NN_CONNECTOMETRY_ANALYSIS_H
