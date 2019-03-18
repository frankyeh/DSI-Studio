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
    tipl::ml::network_data<std::vector<float> > fp_mdata;

    std::vector<tipl::ml::network_data_proxy<float> > train_data;
    std::vector<tipl::ml::network_data_proxy<float> > test_data;
    std::vector<tipl::ml::network_data_proxy<std::vector<float> > > train_mdata;
    std::vector<tipl::ml::network_data_proxy<std::vector<float> > > test_mdata;
public:
    float sl_mean,sl_scale;

public:
    std::vector<int> subject_index;
    std::vector<unsigned int> test_seq;
    std::vector<float> test_result;
    std::vector<std::vector<float> > test_mresult;
public:
    int foi_index = 0;
    bool is_regression = true;
    bool regress_all = false;
    int seed_search = 10;
    float otsu = 0.6f;
    float no_data = 9999.0f;
    int cv_fold = 10;
    bool normalize_value = false;
public:
    nn_connectometry_analysis(std::shared_ptr<fib_data> handle_);
    bool run(std::ostream& out,const std::string& net_string);
    void stop(void);
    void get_salient_map(tipl::color_image& I);
    void get_layer_map(tipl::color_image& I);
};

#endif // NN_CONNECTOMETRY_ANALYSIS_H
