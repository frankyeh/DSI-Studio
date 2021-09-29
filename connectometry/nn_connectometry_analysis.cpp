#include "nn_connectometry_analysis.h"
#include "fib_data.hpp"
nn_connectometry_analysis::nn_connectometry_analysis(std::shared_ptr<fib_data> handle_):handle(handle_)
{
    if(handle->is_qsdr && handle->is_human_data)
    {
        gz_nifti in;
        if(in.load_from_file(handle->t1w_template_file_name.c_str()))
        {
            tipl::image<float,3> I;
            in.toLPS(I);
            tipl::matrix<4,4> tr,tr2;
            tr2.identity();
            in.get_image_transformation(tr);
            tr[15] = 1.0f;
            tr.inv();
            std::copy(handle->trans_to_mni.begin(),handle->trans_to_mni.end(),tr2.begin());
            tr2[15] = 1.0f;
            tr *= tr2;

            It.resize(handle->dim);
            tipl::resample(I,It,tr,tipl::nearest);
            // create Ib as the salient map background
            int plane_size = handle->dim.plane_size();
            // skip top 2 slices
            tipl::image<float,2> Itt(tipl::shape<2>(handle->dim[0],handle->dim[1]*int(handle->dim[2]/skip_slice-1)));
            for(int z = 0,i = 0;z < It.depth() && i+plane_size <= Itt.size();z += skip_slice,i += plane_size)
            {
                std::copy(It.begin()+z*plane_size,
                          It.begin()+(z+1)*plane_size,
                          Itt.begin()+i);
            }
            tipl::normalize(Itt,128);
            Ib = Itt;
        }
    }
}



void nn_connectometry_analysis::stop(void)
{
    if(nn.empty())
        return;
    if(future.valid())
    {
        terminated = true;
        future.wait();
    }
}
void nn_connectometry_analysis::clear_results(void)
{
    std::lock_guard<std::mutex> lock(lock_result);
    result_r.clear();
    result_mae.clear();
    result_test_miss.clear();
    result_test_error.clear();
    result_train_error.clear();
}
bool nn_connectometry_analysis::run(const std::string& net_string_)
{
    std::ostringstream out_report;
    auto& X = handle->db.X;
    if(X.empty())
    {
        error_msg = "No demographic loaded";
        return false;
    }

    // prepare data
    std::vector<int> new_subject_index;
    std::vector<float> selected_label;
    std::vector<std::vector<float> > selected_mlabel;

    // extract labels skipping missing data (9999)

    {
        int feature_count = handle->db.feature_titles.size();
        out_report << " A mulpti-layer perceptrons was used to predict ";
        out_report << handle->db.feature_titles[foi_index];
        for(int i = 0;i < handle->db.num_subjects;++i)
        {
            float label = X[i*(feature_count+1) + 1 + foi_index];
            if(label == no_data)
                continue;
            new_subject_index.push_back(i);
            selected_label.push_back(label);
        }
    }

    subject_index.swap(new_subject_index);
    stop();
    nn.reset();

    int fp_dimension = 0;
    int selected_label_max = *std::max_element(selected_label.begin(),selected_label.end());

    // extract fp

    {
        out_report << " using local connectome fingerprint extracted by " << otsu << " otsu threshold.";
        std::cout << "set otsu=" << otsu << std::endl;
        float fp_threshold = otsu*tipl::segmentation::otsu_threshold(
                    tipl::make_image(handle->dir.fa[0],handle->dim));
        tipl::image<int,3> fp_mask(handle->dim);
        for(int i = 0;i < fp_mask.size();++i)
            if(handle->dir.get_fa(i,0) > fp_threshold)
                fp_mask[i] = 1;
            else
                fp_mask[i] = 0;

        // prepare fp_index for salient map
        {
            std::vector<int> pos;
            handle->db.get_subject_vector_pos(pos,fp_mask,fp_threshold);
            fp_index.resize(pos.size());
            for(int i = 0;i < pos.size();++i)
                fp_index[i] = tipl::pixel_index<3>(pos[i],handle->dim);
        }
        // for smoothing
        if(fiber_smoothing != 0.0)
        {
            fib_pairs.clear();
            std::mutex m;
            tipl::par_for(fp_index.size(),[&](size_t i)
            {
                for(size_t j = i+1 ; j < fp_index.size();++j)
                {
                    if(std::abs(fp_index[i][0]-fp_index[j][0])+
                       std::abs(fp_index[i][1]-fp_index[j][1])+
                       std::abs(fp_index[i][2]-fp_index[j][2]) <= 1)
                    {
                        std::lock_guard<std::mutex> guard(m);
                        fib_pairs.push_back(std::make_pair(i,j));
                    }
                }
            });
        }
        sl_mean = 0.0f;
        sl_scale = 1.0f;
        // normalize values
        if(is_regression && normalize_value)
        {
            out_report << " The variables to be predicted were shifted and scaled to make mean=0 and variance=1.";
            sl_mean = tipl::mean(selected_label);
            float sd = tipl::standard_deviation(selected_label.begin(),selected_label.end(),sl_mean);
            if(sd == 0.0f)
                sd = 1.0f;
            sl_scale = 1.0f/sd;
            tipl::minus_constant(selected_label,sl_mean);
            tipl::multiply_constant(selected_label,sl_scale);
        }

        fp_data.clear();
        std::vector<std::vector<float> > fps;
        for(int i = 0;i < subject_index.size();++i)
        {
            std::vector<float> fp;
            handle->db.get_subject_vector(subject_index[i],fp,fp_mask,fp_threshold,false /* no normalization*/);
            fps.push_back(std::move(fp));
        }
        fp_dimension = fps.front().size();
        {
            fp_data.data = std::move(fps);
            fp_data.data_label = selected_label;
            fp_data.input = tipl::shape<3>(fp_dimension,1,1);
            if(is_regression)
                fp_data.output = tipl::shape<3>(1,1,1);
            else
                fp_data.output = tipl::shape<3>(1,1,selected_label_max+1);
        }
    }




    std::string net_string;
    {
        net_string = std::to_string(fp_dimension)+",1,1|"+net_string_;

        if(is_regression)
            net_string += "|1,1,1";
        else
        {
            net_string += "|1,1,";
            net_string += std::to_string((int)selected_label_max+1);
        }
        if(!(nn << net_string))
        {
            error_msg = "Invalid network text:";
            error_msg += net_string;
            return false;
        }

        out_report << " The multi-layer perceptron has a network structure of " << net_string << ".";
        std::cout << "network=" << net_string << std::endl;
    }

    {
        out_report << " The performance was evaluated using " << cv_fold << " fold cross validation.";
        tipl::ml::data_fold_for_cv(fp_data,train_data,test_data,cv_fold,stratified_fold);
        if(!is_regression)
            for(int i = 0;i < train_data.size();++i)
                train_data[i].homogenize();
    }

    out_report << " The neural network was trained using learning rate=" << t.learning_rate <<
                  ", match size=" << t.batch_size << ", epoch=" << t.epoch << ".";
    report = out_report.str();

    all_result.clear();
    all_test_result.clear();
    all_test_seq.clear();
    cur_progress = 0;
    cur_fold = 0;
    terminated = false;

    for(int i = 0;i < nn.layers.size()-1;++i)
        if(dynamic_cast<tipl::ml::fully_connected_layer*>(nn.layers[i].get()))
                dynamic_cast<tipl::ml::fully_connected_layer*>(nn.layers[i].get())->bn_ratio = bn_norm;

    future = std::async(std::launch::async, [this,net_string]
    {
        for(size_t fold = 0;fold < cv_fold && !terminated;++fold)
        {
            cur_fold = fold;
            clear_results();
            std::cout << "running cross validation at fold=" << fold << std::endl;
            test_seq = test_data[fold].pos;
            if(fold)
            {
                std::cout << "re-initialize network..." << std::endl;
                nn.reset();
                nn << net_string;
            }

            if(seed_search)
            {
                std::cout << "seed searching for " << seed_search << " times" << std::endl;
                t.seed_search(nn,train_data[fold],terminated,seed_search);
            }

            if(!nn.initialized)
                nn.init_weights(0);

            int round = 0;
            test_result.clear();
            std::cout << "start training..." << std::endl;
            t.train(nn,train_data[fold],terminated, [&]()
            {
                nn.set_test_mode(true);

                // fiber convolution
                if(fiber_smoothing != 0.0)
                {
                    std::vector<float> new_weight(nn.layers[0]->weight.size());
                    std::vector<float> new_r(nn.layers[0]->weight.size());
                    tipl::par_for(nn.layers[0]->output_size,[&](int i)
                    {
                        float* w = &(nn.layers[0]->weight[0])+nn.layers[0]->input_size*i;
                        float* nw = &new_weight[0]+nn.layers[0]->input_size*i;
                        float* nr = &new_r[0]+nn.layers[0]->input_size*i;
                        for(uint32_t j = 0;j < fib_pairs.size();++j)
                        {
                            uint32_t j1 = fib_pairs[j].first;
                            uint32_t j2 = fib_pairs[j].second;
                            nw[j1] += w[j2];
                            nw[j2] += w[j1];
                            nr[j1] += 1.0f;
                            nr[j2] += 1.0f;
                        }
                        for(uint32_t j = 0;j < nn.layers[0]->input_size;++j)
                            if(nr[j] != 0.0f)
                            {
                                w[j] *= 1.0f-fiber_smoothing;
                                w[j] += fiber_smoothing*nw[j]/nr[j];
                            }
                    });
                }
                if(weight_decay != 0.0f)
                {
                    for(int i = 1;i < nn.layers.size();++i)
                        if(!nn.layers[i]->weight.empty())
                            tipl::multiply_constant(nn.layers[i]->weight.begin(),
                                                nn.layers[i]->weight.end(),
                                                1.0f-weight_decay);
                }

                std::cout << "[" << round << "]";
                cur_progress = (fold*t.epoch + round)*100/(cv_fold*t.epoch);
                test_result.resize(test_data[0].size());
                nn.predict(test_data[fold],test_result);

                {
                    std::lock_guard<std::mutex> lock(lock_result);
                    if(is_regression)//regression
                    {
                        result_r.push_back(test_data[fold].calculate_r(test_result));
                        result_mae.push_back(test_data[fold].calculate_mae(test_result));
                        std::cout << " mae=" << result_mae.back();
                        std::cout << " r=" << std::setprecision(3) << result_r.back();
                    }
                    else
                    {
                        result_test_miss.push_back(test_data[fold].calculate_miss(test_result));
                        result_test_error.push_back(1.0f-(float)result_test_miss.back()/(float)test_data[fold].size());
                        std::cout << " miss=" << result_test_miss.back() << "/" << test_data[fold].size();
                        std::cout << " accuracy=" << std::setprecision(3) <<  result_test_error.back() ;
                    }
                    result_train_error.push_back(t.get_training_error_value());
                }

                std::cout << " error=" << std::setprecision(3) << result_train_error.back()
                << " rate= " << std::setprecision(3) << t.rate_decay << std::endl;

                //nn.sort_fully_layer();
                //t.initialize_training(nn);

                ++round;
            });

            all_test_seq.insert(all_test_seq.end(),test_seq.begin(),test_seq.end());
            all_test_result.insert(all_test_result.end(),test_result.begin(),test_result.end());
            std::vector<float> y(all_test_result.size());
            float all_mae = 0.0f,all_accuracy = 0.0f;
            for(int i = 0;i < y.size();++i)
            {
                y[i] = fp_data.data_label[all_test_seq[i]];
                all_mae += std::fabs(y[i]-all_test_result[i]);
                all_accuracy += (std::round(y[i]) == std::round(all_test_result[i]) ? 1.0f:0.0f);
            }

            std::ostringstream out;
            out << " " << cv_fold << " fold cross validation results shows that"
                << " r=" << tipl::correlation(y.begin(),y.end(),all_test_result.begin())
                << ", mae=" << all_mae/(float)y.size()
                << ", accuracy = " << (int)all_accuracy << "/" << y.size() << " = " << 100.0f*(float)all_accuracy/(float)y.size() << "%.";
            all_result = out.str();
        }
        cur_progress = 100;
        terminated = true;
    });
    return true;
}

void nn_connectometry_analysis::get_salient_map(tipl::color_image& I)
{
    std::vector<tipl::color_image> w_map;
    if(dynamic_cast<tipl::ml::fully_connected_layer*>(nn.layers[0].get()))
    {
        auto& layer = *dynamic_cast<tipl::ml::fully_connected_layer*>(nn.get_layer(0).get());
        int n = layer.output_size;
        auto& w = layer.weight;
        w_map.resize(n);
        float r = 3.0*256.0f/tipl::maximum(w);
        for(int i = 0,w_pos = 0;i < n;++i)
        {
            w_map[i] = Ib;
            for(int k = 0;k < fp_index.size();++k,++w_pos)
            {
                if(fp_index[k][2] % skip_slice)
                    continue;
                int index = fp_index[k][0] +
                        (fp_index[k][1]+(fp_index[k][2]/skip_slice)*handle->dim[1])*handle->dim[0];
                if(index >= w_map[i].size())
                    continue;
                tipl::rgb color;
                int rv = r*w[w_pos];
                if(rv == 0)
                    continue;
                if(rv > 0)
                    color.b = std::min<int>(255,rv);
                else
                    color.r = std::min<int>(255,-rv);
                color.a = 255;
                //color = 0xFFFFFFFF;
                w_map[i][index] = (unsigned int)w_map[i][index] | (unsigned int)color;
            }
        }
    }
    if(!w_map.empty())
    {
        I.resize(tipl::shape<2>(w_map[0].width()*w_map.size(),w_map[0].height()));
        for(int i = 0;i < w_map.size();++i)
            tipl::draw(w_map[i],I,tipl::vector<2>(i*w_map[0].width(),0));
    }
}


void nn_connectometry_analysis::get_layer_map(tipl::color_image& I)
{
    std::vector<float> input(nn.get_input_size());
    {
        std::copy(train_data[0].get_data(0),
                  train_data[0].get_data(0)+nn.get_input_size(),input.begin());
        tipl::ml::to_image(nn,I,input,train_data[0].get_label(0));
    }
    /*
    nn.get_layer_images(I);
    int h = 0,w = 10;
    for(int i = 1;i < I.size();++i)
    {
        h += I[i].height() + 10;
        w = std::max<int>(w,I[i].width());
    }
    QImage buf(w,h,QImage::Format_RGB32);
    QPainter painter(&buf);
    painter.fillRect(0,0,w,h,Qt::white);
    for(int i = 1,h_pos = 0;i < I.size();++i)
        if(!I[i].empty())
        {
            painter.drawImage(0,h_pos,QImage((unsigned char*)&*I[i].begin(),
                                             I[i].width(),I[i].height(),QImage::Format_RGB32));
            h_pos += I[i].height()+10;
        }
    layer_I = buf.scaled(w*2,h*2);
    */
}
