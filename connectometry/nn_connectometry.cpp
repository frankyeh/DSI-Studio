#include <QFileInfo>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include "nn_connectometry.h"
#include "ui_nn_connectometry.h"
nn_connectometry::nn_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_,QString db_file_name_,bool gui_) :
    QDialog(parent),vbc(vbc_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
    ui(new Ui::nn_connectometry)
{
    ui->setupUi(this);
    ui->network_view->setScene(&network_scene);
    ui->layer_view->setScene(&layer_scene);
    log_text += vbc->handle->report.c_str();
    log_text += "\r\n";
    ui->log->setText(log_text);
    on_otsu_valueChanged(ui->otsu->value());
}

nn_connectometry::~nn_connectometry()
{
    on_stop_clicked();
    delete ui;
}

void fill_demo_table(const connectometry_db& db,
                     QTableWidget* table);

void nn_connectometry::on_open_mr_files_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                work_dir,
                "Text or CSV file (*.txt *.csv);;All files (*)");
    if(filename.isEmpty())
        return;
    auto& db = vbc->handle->db;
    // read demographic file
    if(!db.parse_demo(filename.toStdString(),9999))
    {
        QMessageBox::information(this,"Error",db.error_msg.c_str(),0);
        return;
    }
    X = db.X;
    fill_demo_table(db,ui->subject_demo);
    QStringList t;
    for(int i = 0; i < db.feature_titles.size();++i)
        t << db.feature_titles[i].c_str();
    ui->foi->clear();
    ui->foi->addItems(t);
    ui->foi->setCurrentIndex(0);
}

void nn_connectometry::on_run_clicked()
{
    if(ui->foi->currentIndex() < 0)
        return;
    int selected_label_max = 0;
    on_stop_clicked();
    // prepare data
    selected_label.clear();
    selected_mlabel.clear();
    subject_index.clear();
    bool regress_all = ui->regress_all->isChecked();
    // extract labels skipping missing data (9999)
    if(regress_all)
    {
        for(int i = 0;i < vbc->handle->db.num_subjects;++i)
        {
            std::vector<float> labels;
            for(int j = 0;j < ui->foi->count();++j)
            {
                float label = X[i*(ui->foi->count()+1) + 1 + j];
                if(label == 9999)
                    break;
                labels.push_back(label);
            }
            if(labels.size() != ui->foi->count())
                continue;
            subject_index.push_back(i);
            selected_mlabel.push_back(std::move(labels));
        }
    }
    else
    {
        for(int i = 0;i < vbc->handle->db.num_subjects;++i)
        {
            float label = X[i*(ui->foi->count()+1) + 1 + ui->foi->currentIndex()];
            if(label == 9999)
                continue;
            subject_index.push_back(i);
            selected_label.push_back(label);
        }
        selected_label_max = *std::max_element(selected_label.begin(),selected_label.end());
    }

    // extract fp
    int fp_dimension = 0;
    {
        fp_data.clear();
        fp_mdata.clear();
        std::vector<std::vector<float> > fps;
        for(int i = 0;i < subject_index.size();++i)
        {
            std::vector<float> fp;
            vbc->handle->db.get_subject_vector(subject_index[i],fp,fp_mask,fp_threshold,false /* no normalization*/);
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
            if(ui->nn_regression->isChecked())
                fp_data.output = tipl::geometry<3>(1,1,1);
            else
                fp_data.output = tipl::geometry<3>(1,1,selected_label_max+1);
        }
    }


    std::string net_string;
    {
        net_string = std::to_string(fp_dimension)+",1,1|"+ui->network_text->text().toStdString();

        if(ui->nn_regression->isChecked())
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


        if(ui->regress_all->isChecked())
        {
            tipl::ml::data_fold_for_cv(fp_mdata,train_mdata,test_mdata,10);
            test_seq = test_mdata[0].pos;
        }
        else
        {
            tipl::ml::data_fold_for_cv(fp_data,train_data,test_data,10);
            test_seq = test_data[0].pos;
        }
        if(ui->nn_classification->isChecked())
            for(int i = 0;i < train_data.size();++i)
                train_data[i].homogenize();
    }

    t.learning_rate = ui->learning_rate->value();
    t.momentum = ui->momentum->value();
    t.batch_size = 64;
    t.epoch = ui->epoch->value();
    t.error_table.resize(nn.get_output_size()*nn.get_output_size());


    if(net_string != nn.get_layer_text())
        nn.reset();
    int seed_search = 0;
    if(!nn.initialized)
    {
        seed_search = ui->seed_search->value();
        if(!(nn << net_string))
        {
            QMessageBox::information(this,"Error",QString("Invalid network text:") + net_string.c_str(),0);
            return;
        }
    }
    log_text += "network=";
    log_text += net_string.c_str();
    log_text += "\r\n";

    ui->test_subjects->setRowCount(0);
    terminated = false;
    future = std::async(std::launch::async, [this,regress_all,seed_search,net_string]
    {
        if(seed_search)
        {
            if(regress_all)
                t.seed_search(nn,train_mdata[0],terminated,seed_search);
            else
                t.seed_search(nn,train_data[0],terminated,seed_search);

        }
        if(!nn.initialized)
            nn.init_weights(0);
        int round = 0;
        test_result.clear();
        test_mresult.clear();
        if(regress_all)
        t.train(nn,train_mdata[0],terminated, [&]()
        {
            nn.set_test_mode(true);
            //nn.sort_fully_layer();

            test_mresult.resize(test_mdata[0].size());
            nn.predict(test_mdata[0],test_mresult);

            std::ostringstream out;
            out << "[" << round << "]";
            for(int j = 0;j < nn.output_size;++j)
            {
                //out << " mae=" << test_mdata[0].calculate_mae(test_mresult,j);
                out << " r" << j << "=" << std::setprecision(3) << test_mdata[0].calculate_r(test_mresult,j);
            }
            out << " error=" << std::setprecision(3) << t.get_training_error_value()
            << " rate= " << std::setprecision(3) << t.rate_decay << std::endl;

            log_text += out.str().c_str();

            ++round;

        });
        else
        t.train(nn,train_data[0],terminated, [&]()
        {
            nn.set_test_mode(true);
            //nn.sort_fully_layer();

            std::ostringstream out;
            out << "[" << round << "]";

            test_result.resize(test_data[0].size());
            nn.predict(test_data[0],test_result);
            if(nn.output_size == 1)//regression
            {
                //out << " mae=" << test_data[0].calculate_mae(test_result);
                out << " r=" << std::setprecision(3) << test_data[0].calculate_r(test_result);
            }
            else
            {
                out << " test error=" << std::setprecision(3) << test_data[0].calculate_miss(test_result) << "/" << test_data[0].size();
            }

            out << " error=" << std::setprecision(3) << t.get_training_error_value()
            << " rate= " << std::setprecision(3) << t.rate_decay << std::endl;

            log_text += out.str().c_str();

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
        tipl::color_image I;
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
        layer_I = QImage((unsigned char*)&*I.begin(),
                         I.width(),I.height(),QImage::Format_RGB32).scaledToWidth(ui->view_tab->width()-50);
        show_view(layer_scene,layer_I);
    }
    if(ui->view_tab->currentIndex() == 3) // predict view
    {
        if(!test_mresult.empty())
        {
            int index = ui->foi->currentIndex();
            test_result.resize(test_mresult.size());
            fp_data.data_label.resize(fp_mdata.data_label.size());
            for(int i = 0;i < test_result.size();++i)
            {
                test_result[i] = test_mresult[i][index];
                fp_data.data_label[test_seq[i]] = fp_mdata.data_label[test_seq[i]][index];
            }
        }
        if(!test_result.empty())
        {
            if(ui->test_subjects->rowCount() != test_result.size())
            {
                ui->test_subjects->setRowCount(0);
                ui->test_subjects->setColumnCount(3);
                ui->test_subjects->setHorizontalHeaderLabels(QStringList() << "Subject" << "Label" << "Predicted");
                ui->test_subjects->setRowCount(test_result.size());
                for(unsigned int row = 0;row < test_result.size();++row)
                {
                    int id = subject_index[test_seq[row]];
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
            return;
        }
        num_classes.insert(label);
    }
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

void nn_connectometry::on_regress_all_clicked()
{
    ui->nn_classification->setEnabled(!ui->regress_all->isChecked());
    selected_mlabel.clear();
    if(ui->regress_all->isChecked())
    {
        ui->nn_classification->setChecked(false);
        ui->nn_regression->setChecked(true);
    }
}
