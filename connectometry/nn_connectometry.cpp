#include <QFileInfo>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include "nn_connectometry.h"
#include "ui_nn_connectometry.h"
extern std::string t1w_template_file_name;
nn_connectometry::nn_connectometry(QWidget *parent,std::shared_ptr<vbc_database> vbc_,QString db_file_name_,bool gui_) :
    QDialog(parent),vbc(vbc_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
    ui(new Ui::nn_connectometry)
{
    ui->setupUi(this);
    ui->network_view->setScene(&network_scene);
    ui->layer_view->setScene(&layer_scene);
    log_text += vbc->handle->report.c_str();
    log_text += "\r\n";
    ui->log->setText(log_text);



}

nn_connectometry::~nn_connectometry()
{
    on_stop_clicked();
    delete ui;
}

bool parse_demo(std::shared_ptr<vbc_database>& vbc,
                QString filename,
                std::vector<std::string>& titles,
                std::vector<std::string>& items,
                std::vector<int>& feature_location,
                std::vector<double>& X,
                float missing_value,
                std::string& error_msg);
void fill_demo_table(std::shared_ptr<vbc_database>& vbc,
                     QTableWidget* table,
                     const std::vector<std::string>& titles,
                     const std::vector<std::string>& items,
                     const std::vector<double>& X);

void nn_connectometry::on_open_mr_files_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                work_dir,
                "Text or CSV file (*.txt *.csv);;All files (*)");
    if(filename.isEmpty())
        return;
    std::vector<std::string> titles;
    std::vector<std::string> items;
    std::vector<int> feature_location;
    std::string error_msg;
    // read demographic file
    if(!parse_demo(vbc,filename,titles,items,feature_location,X,9999,error_msg))
    {
        QMessageBox::information(this,"Error",error_msg.c_str(),0);
        return;
    }
    fill_demo_table(vbc,ui->subject_demo,titles,items,X);
    QStringList t;
    for(unsigned int index = 0;index < feature_location.size();++index)
    {
        std::replace(titles[feature_location[index]].begin(),titles[feature_location[index]].end(),'/','_');
        std::replace(titles[feature_location[index]].begin(),titles[feature_location[index]].end(),'\\','_');
        t << titles[feature_location[index]].c_str();
    }
    ui->foi->clear();
    ui->foi->addItems(t);
    ui->foi->setCurrentIndex(0);
}
void nn_connectometry::init_partially_connected_layer(tipl::ml::network& n)
{
    if(!n.empty() && dynamic_cast<tipl::ml::partially_connected_layer*>(n.layers[0].get()))
    {
        std::vector<char> fp_chosen(fp_index.size());
        for(int i = 0;i < fp_index.size();++i)
        {
            if(fp_chosen[i])
                continue;
            fp_mapping.push_back(std::vector<int>());
            tipl::pixel_index<3> cur_pos(fp_index[i]);
            fp_chosen[i] = 1;
            fp_mapping.back().push_back(i);
            for(int j = i+1;j < fp_chosen.size();++j)
            if(!fp_chosen[j] &&
                std::abs(fp_index[j][0]-cur_pos[0]) <= 10 &&
                std::abs(fp_index[j][1]-cur_pos[1]) <= 10 &&
                std::abs(fp_index[j][2]-cur_pos[2]) <= 10)
            {
                fp_chosen[j] = 1;
                fp_mapping.back().push_back(j);
            }
        }

        auto* p = dynamic_cast<tipl::ml::partially_connected_layer*>(n.layers[0].get());
        int roi_count = n.geo[1].width();
        int channel_count = n.geo[1].height();
        for(int i = 0,pos = 0;i < roi_count;++i)
        {
            for(int j = 0;j < channel_count;++j,++pos)
                p->mapping[pos] = fp_mapping[i];
        }
    }
}

void nn_connectometry::on_run_clicked()
{
    if(ui->foi->currentIndex() < 0 || source_location.empty())
        return;
    on_stop_clicked();
    // prepare data
    std::string output_dim_text,input_dim_text;
    {
        fp_data.clear();
        for(int i = 0;i < source_location.size();++i)
        {
            std::vector<float> fp;
            vbc->handle->db.get_subject_vector(source_location[i],fp,fp_mask,fp_threshold,false /* no normalization*/);
            fp_data.data.push_back(std::move(fp));
            fp_data.data_label.push_back(selected_label[i]);
        }

        input_dim_text = std::to_string(fp_data.data[0].size())+",1,1|";
        fp_data.input = tipl::geometry<3>(fp_data.data[0].size(),1,1);
        if(ui->nn_regression->isChecked())
        {
            fp_data.output = tipl::geometry<3>(1,1,1);
            output_dim_text= "|1,1,1";
        }
        else
        {
            fp_data.output = tipl::geometry<3>(1,1,selected_label_max+1);
            output_dim_text = "|1,1,";
            output_dim_text += std::to_string((int)selected_label_max+1);
        }
        tipl::ml::data_fold_for_cv(fp_data,train_data,test_data,10);
        if(ui->nn_classification->isChecked())
            for(int i = 0;i < train_data.size();++i)
                train_data[i].homogenize();
    }

    std::vector<std::string> nn_list;
    if(!nn.initialized)
    {
        std::istringstream in(ui->network_list->toPlainText().toStdString());
        std::string line;
        while(std::getline(in,line))
            nn_list.push_back(input_dim_text+line+output_dim_text);
        if(nn_list.empty())
        {
            QMessageBox::information(this,"Error","Please assign network structure",0);
            return;
        }
    }

    t.learning_rate = ui->learning_rate->value();
    t.momentum = ui->momentum->value();
    t.w_decay_rate = 0.0f;
    int epoch = ui->epoch->value();
    int seed_search = ui->seed_search->value();

    terminated = false;
    future = std::async(std::launch::async, [this,epoch,seed_search,nn_list]
    {
        t.batch_size = std::max<int>(64,fp_data.data.size()/16.0f);
        t.error_table.resize(nn.get_output_size()*nn.get_output_size());
        t.epoch = 2;
        if(!nn.initialized)
        {
            std::string net_string = nn_list[0];
            float best_error = 0.0f;
            log_text += "Optimizing network structure\r\n";
            for(int i = 0;i < nn_list.size() && !terminated;++i)
            {
                tipl::ml::network tmp_nn;
                if(!(tmp_nn << nn_list[i]))
                {
                    log_text += "\r\n";
                    log_text += "Invalid network text:";
                    log_text += tmp_nn.error_msg.c_str();
                    return;
                }
                init_partially_connected_layer(tmp_nn);
                t.train(tmp_nn,train_data[0],terminated, [&](){});
                if(t.get_training_error_value() < best_error || best_error == 0.0f)
                {
                    best_error = t.get_training_error_value();
                    nn = std::move(tmp_nn);
                    net_string = nn_list[i];
                }
            }
            log_text += "Chosen network=";
            log_text += net_string.c_str();
            log_text += "\r\n";
            for(int seed = 1;seed < seed_search && !terminated;++seed)
            {
                tipl::ml::network tmp_nn;
                if(!(tmp_nn << net_string))
                {
                    log_text += "Invalid network text:";
                    log_text += tmp_nn.error_msg.c_str();
                    return;
                }
                init_partially_connected_layer(tmp_nn);
                t.train(tmp_nn,train_data[0],terminated, [&](){},seed);
                if(t.get_training_error_value() < best_error || best_error == 0.0f)
                {
                    best_error = t.get_training_error_value();
                    nn = std::move(tmp_nn);
                }
            }
            if(terminated)
                return;
            for(int seed = 0;seed < seed_search && !terminated;++seed)
            {
                tipl::ml::network tmp_nn;
                if(!(tmp_nn << net_string))
                {
                    log_text += "Invalid network text:";
                    log_text += tmp_nn.error_msg.c_str();
                    return;
                }
                init_partially_connected_layer(tmp_nn);
                t.epoch = 2;
                tmp_nn.init_weights(seed);
                t.train(tmp_nn,train_data[0],terminated, [&](){});
                if(t.get_training_error_value() < best_error || best_error == 0.0f)
                {
                    best_error = t.get_training_error_value();
                    log_text += "seed selected with accuracy=";
                    log_text += std::to_string(best_error).c_str();
                    log_text += "\r\n";
                    nn = std::move(tmp_nn);
                }
            }
            if(terminated)
                return;
            t.w_decay_rate = 0.005f;
        }
        t.epoch = epoch;
        t.train_result.resize(train_data[0].size());
        test_seq = test_data[0].pos;
        int round = 0;
        t.train(nn,train_data[0],terminated, [&]()
        {
            //nn.sort_fully_layer();
            test_result.resize(test_data[0].size());
            nn.set_test_mode(true);
            nn.predict(test_data[0],test_result);

            std::ostringstream out;
            out << "[" << round << "]";
            if(nn.output_size == 1)//regression
            {
                float sum_error = 0.0f;
                for(int i = 0;i < test_data[0].size();++i)
                    sum_error += std::fabs(test_result[i]-test_data[0].get_label(i));
                out << " mae=" << sum_error/test_data[0].size();
                std::vector<float> d(test_data[0].size());
                for(int i = 0;i < d.size();++i)
                    d[i] = test_data[0].get_label(i);
                out << " r=" << tipl::correlation(d.begin(),d.end(),test_result.begin());
            }
            else
            {
                int mis_count = 0;
                for(int i = 0;i < test_data[0].size();++i)
                    if(test_result[i] != test_data[0].get_label(i))
                        ++mis_count;
                out << " test error=" << mis_count << "/" << test_data[0].size()
                    << " training=" << t.get_training_error() << "% ";
            }
            out << " error=" << t.get_training_error_value()
            << " rate= " << t.rate_decay << std::endl;
            log_text += out.str().c_str();

            train_result = t.train_result;
            train_seq = train_data[0].pos;
            ++round;
        });
        log_text += "\r\n";
        terminated = true;
    });

    if(timer)
        delete timer;
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update_network()));
    timer->setInterval(1000);
    timer->start();
}
void nn_connectometry::on_stop_clicked()
{
    if(nn.empty())
        return;
    if(future.valid())
    {
        terminated = true;
        future.wait();
    }
    if(timer)
    {
        delete timer;
        timer = 0;
    }
}

void nn_connectometry::update_network(void)
{

    on_view_tab_currentChanged(0);
}
void show_view(QGraphicsScene& scene,QImage I);
void nn_connectometry::on_view_tab_currentChanged(int)
{
    float scroll_ratio = (float)ui->log->verticalScrollBar()->value()/(float)(ui->log->verticalScrollBar()->maximum()+1);
    ui->log->setText(log_text);
    ui->log->verticalScrollBar()->setValue((ui->log->verticalScrollBar()->maximum()+1)*scroll_ratio);
    if(!nn.initialized)
        return;
    if(ui->view_tab->currentIndex() == 1) //network view
    {
        std::vector<tipl::color_image> w_map;
        if(dynamic_cast<tipl::ml::partially_connected_layer*>(nn.layers[0].get()))
        {
            int roi_count = nn.geo[1].width();
            int channel_count = nn.geo[1].height();


            auto& layer = *dynamic_cast<tipl::ml::partially_connected_layer*>(nn.get_layer(0).get());
            int n = nn.get_geo(1).height();
            auto& w = layer.weight;
            int skip = 4;
            w_map.resize(n);
            for(int i = 0;i < n;++i)
                w_map[i].resize(tipl::geometry<2>(vbc->handle->dim[0],vbc->handle->dim[1]*vbc->handle->dim[2]/skip));

            float r = 512.0f/tipl::maximum(w);
            for(int i = 0,m_pos = 0,w_pos = 0;i < roi_count;++i)
            {
                for(int j = 0;j < channel_count;++j,++m_pos)
                for(int k = 0;k < layer.mapping[m_pos].size();++k,++w_pos)
                {
                    int f_pos = layer.mapping[m_pos][k];
                    if(fp_index[f_pos][2] % skip)
                        continue;
                    int index = fp_index[f_pos][0] +
                            (fp_index[f_pos][1]+(fp_index[f_pos][2]/skip)*vbc->handle->dim[1])*vbc->handle->dim[0];
                    float v = w[w_pos];
                    tipl::rgb color;
                    if(v > 0)
                        color.b = std::min<int>(255,r*v);
                    else
                        color.r = std::min<int>(255,-r*v);
                    color.a = 255;
                    //color = 0xFFFFFFFF;
                    w_map[j][index] = color;
                }
            }
        }
        else
        if(dynamic_cast<tipl::ml::fully_connected_layer*>(nn.layers[0].get()))
        {
            auto& layer = *dynamic_cast<tipl::ml::fully_connected_layer*>(nn.get_layer(0).get());
            int n = layer.output_size;
            auto& w = layer.weight;
            int skip = 4;
            w_map.resize(n);
            float r = 512.0f/tipl::maximum(w);
            for(int i = 0,w_pos = 0;i < n;++i)
            {
                w_map[i].resize(tipl::geometry<2>(vbc->handle->dim[0],vbc->handle->dim[1]*vbc->handle->dim[2]/skip));
                for(int k = 0;k < fp_index.size();++k,++w_pos)
                {
                    if(fp_index[k][2] % skip)
                        continue;
                    int index = fp_index[k][0] +
                            (fp_index[k][1]+(fp_index[k][2]/skip)*vbc->handle->dim[1])*vbc->handle->dim[0];
                    float v = w[w_pos];
                    tipl::rgb color;
                    if(v > 0)
                        color.b = std::min<int>(255,r*v);
                    else
                        color.r = std::min<int>(255,-r*v);
                    color.a = 255;
                    //color = 0xFFFFFFFF;
                    w_map[i][index] = color;
                }
            }
        }


        if(!w_map.empty())
        {
            QImage buf(w_map[0].width()*w_map.size(),w_map[0].height(),QImage::Format_RGB32);
            QPainter painter(&buf);
            for(int i = 0;i < w_map.size();++i)
            {
                painter.drawImage(i*w_map[0].width(),0,
                    QImage((unsigned char*)&*w_map[i].begin(),w_map[i].width(),w_map[i].height(),QImage::Format_RGB32));
            }
            network_I = buf;
            show_view(network_scene,network_I);
        }
    }
    if(ui->view_tab->currentIndex() == 2) // layer_view
    {
        std::vector<tipl::color_image> I;
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
        show_view(layer_scene,layer_I);
    }
    if(ui->view_tab->currentIndex() == 3 && !test_result.empty() && !train_result.empty()) // predict view
    {
        if(ui->test_subjects->rowCount() != test_result.size())
        {
            ui->test_subjects->clear();
            ui->test_subjects->setColumnCount(3);
            ui->test_subjects->setHorizontalHeaderLabels(QStringList() << "Subject" << "Label" << "Predicted");
            ui->test_subjects->setRowCount(test_result.size());
            for(unsigned int row = 0;row < test_result.size();++row)
            {
                int id = source_location[test_seq[row]];
                ui->test_subjects->setItem(row,0,
                                           new QTableWidgetItem(QString(vbc->handle->db.subject_names[id].c_str())));
                ui->test_subjects->setItem(row,1,new QTableWidgetItem(QString()));
                ui->test_subjects->setItem(row,2,new QTableWidgetItem(QString()));
            }
        }

        QVector<double> x(test_result.size());
        QVector<double> y(test_result.size());
        double x_min = 100,x_max = -100,y_min = 100,y_max = -100;
        for(unsigned int row = 0;row < test_result.size();++row)
        {
            x[row] = test_result[row];
            y[row] = fp_data.data_label[test_seq[row]];
            ui->test_subjects->item(row,1)->setText(QString::number(fp_data.data_label[test_seq[row]]));
            ui->test_subjects->item(row,2)->setText(QString::number(test_result[row]));
            x_min = std::min<double>(x_min,x[row]);
            x_max = std::max<double>(x_max,x[row]);
            y_min = std::min<double>(y_min,y[row]);
            y_max = std::max<double>(y_max,y[row]);
        }
        double x_margin = (x_max-x_min)*0.05f;
        double y_margin = (y_max-y_min)*0.05f;
        ui->prediction_plot->clearGraphs();
        ui->prediction_plot->addGraph();
        QPen pen;
        pen.setColor(Qt::red);
        ui->prediction_plot->graph()->setLineStyle(QCPGraph::lsNone);
        ui->prediction_plot->graph()->setScatterStyle(QCP::ScatterStyle::ssDisc);
        ui->prediction_plot->graph()->setScatterSize(5);
        ui->prediction_plot->graph()->setPen(pen);
        ui->prediction_plot->graph()->setData(x, y);
        ui->prediction_plot->xAxis->setLabel("Prediction");
        ui->prediction_plot->yAxis->setLabel("Test Values");
        ui->prediction_plot->xAxis->setRange(x_min-x_margin,x_max+x_margin);
        ui->prediction_plot->yAxis->setRange(y_min-y_margin,y_max+y_margin);
        ui->prediction_plot->xAxis->setGrid(false);
        ui->prediction_plot->yAxis->setGrid(false);
        ui->prediction_plot->xAxis2->setVisible(true);
        ui->prediction_plot->xAxis2->setTicks(false);
        ui->prediction_plot->xAxis2->setTickLabels(false);
        ui->prediction_plot->yAxis2->setVisible(true);
        ui->prediction_plot->yAxis2->setTicks(false);
        ui->prediction_plot->yAxis2->setTickLabels(false);
        ui->prediction_plot->replot();
    }
    if(terminated && timer)
        timer->stop();
}

void nn_connectometry::on_reset_clicked()
{
    on_stop_clicked();
    nn.reset();
}

void nn_connectometry::on_foi_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    ui->nn_classification->setEnabled(true);
    ui->nn_classification->setChecked(true);
    selected_label.clear();
    source_location.clear();
    std::set<float> num_classes;
    for(int i = 0;i < vbc->handle->db.num_subjects;++i)
    {
        float label = X[i*(ui->foi->count()+1) + 1 + ui->foi->currentIndex()];
        if(label == 9999)
            continue;
        if(ui->nn_classification->isEnabled() &&
                (int(label*100) % 100 || num_classes.size() > 10))
        {
            ui->nn_regression->setChecked(true);
            ui->nn_classification->setEnabled(false);
        }
        source_location.push_back(i);
        selected_label.push_back(label);
        num_classes.insert(label);
    }
    selected_label_max = *std::max_element(selected_label.begin(),selected_label.end());
    selected_label_min = *std::min_element(selected_label.begin(),selected_label.end());
    selected_label_mean = tipl::mean(selected_label);
    float range = (selected_label_max-selected_label_min);
    selected_label_scale = (range == 0.0f? 0.0f: 1.0f/range);
}

void nn_connectometry::on_otsu_valueChanged(double arg1)
{
    fp_threshold = arg1*tipl::segmentation::otsu_threshold(
                tipl::make_image(vbc->handle->dir.fa[0],vbc->handle->dim));
    fp_mask.resize(vbc->handle->dim);
    for(int i = 0;i < fp_mask.size();++i)
        if(vbc->handle->dir.get_fa(i,0) > fp_threshold)
            fp_mask[i] = 1;
        else
            fp_mask[i] = 0;
    {
        std::vector<int> pos;
        vbc->handle->db.get_subject_vector_pos(pos,fp_mask,fp_threshold);
        fp_index.resize(pos.size());
        for(int i = 0;i < pos.size();++i)
            fp_index[i] = tipl::pixel_index<3>(pos[i],vbc->handle->dim);
    }
}
