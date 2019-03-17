#include "nn_connectometry_analysis.h"
#include "fib_data.hpp"
extern std::string t1w_template_file_name;
nn_connectometry_analysis::nn_connectometry_analysis(std::shared_ptr<fib_data> handle_):handle(handle_)
{
    if(handle->is_qsdr && handle->is_human_data)
    {
        gz_nifti in;
        if(in.load_from_file(t1w_template_file_name.c_str()))
        {
            tipl::image<float,3> I;
            in.toLPS(I);
            tipl::matrix<4,4,float> tr,tr2;
            tr.identity();
            tr2.identity();
            in.get_image_transformation(tr.begin());
            tr[15] = 1.0f;
            tr.inv();
            std::copy(handle->trans_to_mni.begin(),handle->trans_to_mni.end(),tr2.begin());
            tr2[15] = 1.0f;
            tr *= tr2;

            tipl::transformation_matrix<double> T;
            T.load_from_transform(tr.begin());
            It.resize(handle->dim);
            tipl::resample(I,It,T,tipl::nearest);
            // create Ib as the salient map background
            int plane_size = handle->dim.plane_size();
            // skip top 2 slices
            tipl::image<float,2> Itt(tipl::geometry<2>(handle->dim[0],handle->dim[1]*int(handle->dim[2]/skip_slice-1)));
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

bool nn_connectometry_analysis::run(std::ostream& out,
                                    const std::string& net_string_)
{
    std::ostringstream out_report;
    auto& X = handle->db.X;
    if(X.empty())
    {
        error_msg = "No demographic loaded";
        return false;
    }

    // prepare data
    selected_label.clear();
    selected_mlabel.clear();
    subject_index.clear();
    // extract labels skipping missing data (9999)
    int selected_label_max = 0;
    {
        int feature_count = handle->db.feature_titles.size();
        out_report << " A mulpti-layer perceptrons was used to predict ";
        if(regress_all)
        {
            for(int i = 0;i < handle->db.feature_titles.size();++i)
            {
                if(i)
                    out_report << ", ";
                out_report << handle->db.feature_titles[foi_index];
            }
            for(int i = 0;i < handle->db.num_subjects;++i)
            {
                std::vector<float> labels;
                for(int j = 0;j < feature_count;++j)
                {
                    float label = X[i*(feature_count+1) + 1 + j];
                    if(label == 9999)
                        break;
                    labels.push_back(label);
                }
                if(labels.size() != feature_count)
                    continue;
                subject_index.push_back(i);
                selected_mlabel.push_back(std::move(labels));
            }
            sl_mean = 0.0f;
            sl_scale = 1.0f;
        }
        else
        {
            out_report << handle->db.feature_titles[foi_index];
            for(int i = 0;i < handle->db.num_subjects;++i)
            {
                float label = X[i*(feature_count+1) + 1 + foi_index];
                if(label == no_data)
                    continue;
                subject_index.push_back(i);
                selected_label.push_back(label);
            }

            out_report << " The variables to be predicted were shifted and scaled to make mean=0 and variance=1.";
            sl_mean = tipl::mean(selected_label);
            float sd = tipl::standard_deviation(selected_label.begin(),selected_label.end(),sl_mean);
            if(sd == 0.0f)
            {
                error_msg = "Invalid prediction data";
                return false;
            }
            sl_scale = 1.0f/sd;
            tipl::minus_constant(selected_label,sl_mean);
            tipl::multiply_constant(selected_label,sl_scale);
            selected_label_max = *std::max_element(selected_label.begin(),selected_label.end());
        }
    }


    int fp_dimension = 0;
    // extract fp

    {
        out_report << " using local connectome fingerprint extracted by " << otsu << " otsu threshold.";
        out << "set otsu=" << otsu << std::endl;
        fp_threshold = otsu*tipl::segmentation::otsu_threshold(
                    tipl::make_image(handle->dir.fa[0],handle->dim));
        fp_mask.resize(handle->dim);
        for(int i = 0;i < fp_mask.size();++i)
            if(handle->dir.get_fa(i,0) > fp_threshold)
                fp_mask[i] = 1;
            else
                fp_mask[i] = 0;

        handle->db.get_subject_vector_pairs(fib_pairs,fp_mask,fp_threshold);

        // prepare fp_index for salient map
        {
            std::vector<int> pos;
            handle->db.get_subject_vector_pos(pos,fp_mask,fp_threshold);
            fp_index.resize(pos.size());
            for(int i = 0;i < pos.size();++i)
                fp_index[i] = tipl::pixel_index<3>(pos[i],handle->dim);
        }

        fp_data.clear();
        fp_mdata.clear();
        std::vector<std::vector<float> > fps;
        for(int i = 0;i < subject_index.size();++i)
        {
            std::vector<float> fp;
            handle->db.get_subject_vector(subject_index[i],fp,fp_mask,fp_threshold,false /* no normalization*/);
            fps.push_back(std::move(fp));
        }
        fp_dimension = fps.front().size();
        if(regress_all)
        {
            fp_mdata.data = std::move(fps);
            fp_mdata.data_label = selected_mlabel;
            fp_mdata.input = tipl::geometry<3>(fp_dimension,1,1);
            fp_mdata.output = tipl::geometry<3>(1,1,fp_mdata.data_label.front().size());
        }
        else
        {
            fp_data.data = std::move(fps);
            fp_data.data_label = selected_label;
            fp_data.input = tipl::geometry<3>(fp_dimension,1,1);
            if(is_regression)
                fp_data.output = tipl::geometry<3>(1,1,1);
            else
                fp_data.output = tipl::geometry<3>(1,1,selected_label_max+1);
        }
    }


    stop();
    nn.reset();


    std::string net_string;
    {
        net_string = std::to_string(fp_dimension)+",1,1|"+net_string_;

        if(is_regression)
        {
            if(regress_all)
            {
                net_string += "|1,1,";
                net_string += std::to_string((int)fp_mdata.data_label.front().size());
            }
            else
                net_string += "|1,1,1";
        }
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
        out << "network=" << net_string << std::endl;
    }

    {
        out_report << " The performance was evaluated using " << cv_fold << " fold cross validation.";
        if(regress_all)
            tipl::ml::data_fold_for_cv(fp_mdata,train_mdata,test_mdata,cv_fold);
        else
            tipl::ml::data_fold_for_cv(fp_data,train_data,test_data,cv_fold);
        if(!is_regression)
            for(int i = 0;i < train_data.size();++i)
                train_data[i].homogenize();
    }

    out_report << " The neural network was trained using learning rate=" << t.learning_rate <<
                  ", match size=" << t.batch_size << ", epoch=" << t.epoch << ".";
    report = out_report.str();
    terminated = false;
    future = std::async(std::launch::async, [this,&out,net_string]
    {
        std::vector<unsigned int> all_test_seq;
        std::vector<float> all_test_result;
        for(int fold = 0;fold < cv_fold && !terminated;++fold)
        {
            out << "running cross validation at fold=" << fold << std::endl;
            if(regress_all)
                test_seq = test_mdata[fold].pos;
            else
                test_seq = test_data[fold].pos;
            if(fold)
            {
                out << "re-initialize network..." << std::endl;
                nn.reset();
                nn << net_string;
            }

            if(seed_search)
            {
                out << "seed searching for " << seed_search << " times" << std::endl;
                if(regress_all)
                    t.seed_search(nn,train_mdata[fold],terminated,seed_search);
                else
                    t.seed_search(nn,train_data[fold],terminated,seed_search);
            }

            if(!nn.initialized)
                nn.init_weights(0);

            int round = 0;
            test_result.clear();
            test_mresult.clear();
            out << "start training..." << std::endl;

            if(regress_all)
            t.train(nn,train_mdata[fold],terminated, [&]()
            {
                nn.set_test_mode(true);
                //nn.sort_fully_layer();

                test_mresult.resize(test_mdata[fold].size());
                nn.predict(test_mdata[fold],test_mresult);

                out << "[" << round << "]";
                for(int j = 0;j < nn.output_size;++j)
                {
                    //out << " mae=" << test_mdata[0].calculate_mae(test_mresult,j);
                    out << " r" << j << "=" << std::setprecision(3) << test_mdata[0].calculate_r(test_mresult,j);
                }
                out << " error=" << std::setprecision(3) << t.get_training_error_value()
                << " rate= " << std::setprecision(3) << t.rate_decay << std::endl;

                ++round;

            });
            else
            t.train(nn,train_data[fold],terminated, [&]()
            {
                nn.set_test_mode(true);
                nn.sort_fully_layer();

                out << "[" << round << "]";

                test_result.resize(test_data[0].size());
                nn.predict(test_data[fold],test_result);
                if(nn.output_size == 1)//regression
                {
                    //out << " mae=" << test_data[0].calculate_mae(test_result);
                    out << " r=" << std::setprecision(3) << test_data[fold].calculate_r(test_result);
                }
                else
                {
                    out << " test error=" << std::setprecision(3) << test_data[fold].calculate_miss(test_result) << "/" << test_data[fold].size();
                }

                out << " error=" << std::setprecision(3) << t.get_training_error_value()
                << " rate= " << std::setprecision(3) << t.rate_decay << std::endl;

                float* w = &(nn.layers[0]->weight[0]);
                for(int i = 0;i < nn.layers[0]->output_size;++i,w += nn.layers[0]->input_size)
                {
                    for(int j = 0;j < fib_pairs.size();++j)
                    {
                        float& a = w[fib_pairs[j].first];
                        float& b = w[fib_pairs[j].second];
                        float m = (a+b)* 0.01f;
                        a *= 0.98f;
                        b *= 0.98f;
                        a += m;
                        b += m;
                    }
                }
                ++round;
            });

            all_test_seq.insert(all_test_seq.end(),test_seq.begin(),test_seq.end());
            all_test_result.insert(all_test_result.end(),test_result.begin(),test_result.end());


        }
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
        I.resize(tipl::geometry<2>(w_map[0].width()*w_map.size(),w_map[0].height()));
        for(int i = 0;i < w_map.size();++i)
            tipl::draw(w_map[i],I,tipl::vector<2>(i*w_map[0].width(),0));
    }
}


void nn_connectometry_analysis::get_layer_map(tipl::color_image& I)
{
    std::vector<float> input(nn.get_input_size());
    if(!fp_mdata.empty())
    {
        std::copy(train_mdata[0].get_data(0),
                  train_mdata[0].get_data(0)+nn.get_input_size(),input.begin());
        tipl::ml::to_image(nn,I,input,train_mdata[0].get_label(0));
    }
    else
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
