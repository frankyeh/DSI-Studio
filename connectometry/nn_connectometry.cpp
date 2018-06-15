#include <QFileInfo>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include "nn_connectometry.h"
#include "ui_nn_connectometry.h"

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


}

void nn_connectometry::on_run_clicked()
{
    on_stop_clicked();
    if(ui->nn->text().isEmpty())
    {
        QMessageBox::information(this,"Error","Please assign network structure",0);
        return;
    }

    if(ui->foi->currentIndex() < 0)
        return;
    float threshold = 0.6*tipl::segmentation::otsu_threshold(
                tipl::make_image(vbc->handle->dir.fa[0],vbc->handle->dim));

    tipl::image<int,3> fp_mask(vbc->handle->dim);
    for(int i = 0;i < fp_mask.size();++i)
        if(vbc->handle->dir.get_fa(i,0) > threshold)
            fp_mask[i] = 1;
        else
            fp_mask[i] = 0;

    fp_data.clear();
    fp_data_test.clear();
    {
        std::vector<int> pos;
        vbc->handle->db.get_subject_vector_pos(pos,fp_mask,threshold);
        fp_index.resize(pos.size());
        for(int i = 0;i < pos.size();++i)
            fp_index[i] = tipl::pixel_index<3>(pos[i],vbc->handle->dim);
    }
    for(int i = 0;i < vbc->handle->db.num_subjects;++i)
    {
        std::vector<float> fp;
        vbc->handle->db.get_subject_vector(i,fp,fp_mask,threshold,false /* no normalization*/);
        fp_data.data.push_back(std::move(fp));
        fp_data.data_label.push_back(X[i*(ui->foi->count()+1) + 1 + ui->foi->currentIndex()]); // 1 for intercept in X
    }

    float m = tipl::maximum(fp_data.data_label);
    if(m == 0.0f)
    {
        QMessageBox::information(this,"Error","Error in predicting labels",0);
        return;
    }
    tipl::multiply_constant(fp_data.data_label,1.0f/m);


    fp_data.input = tipl::geometry<3>(fp_data.data[0].size(),1,1);
    fp_data.output = tipl::geometry<3>(1,1,2);


    fp_data_test.sample_test_from(fp_data,0.1f);

    {
        std::ostringstream out;
        out << "test size = " << fp_data_test.data.size() << std::endl;
        out << "train size = " << fp_data.data.size() << std::endl;
        log_text += out.str().c_str();
        ui->log->setText(log_text);
    }

    if(nn_text != ui->nn->text().toStdString())
    {
        if(ui->nn->text().isEmpty())
        {
            QMessageBox::information(this,"Error","Please assign network structure",0);
            return;
        }

        QString text = QString::number(fp_data.data[0].size())+",1,1|";
        text += ui->nn->text();

        nn.reset();
        log_text += "Loading network:";
        log_text += text;
        log_text += "\r\n";
        ui->log->setText(log_text);

        if(!(nn << text.toStdString()))
        {
            QMessageBox::information(this,"Error",QString("Invalid network text: %1").arg(nn.error_msg.c_str()));
            nn.reset();
            return;
        }
        nn_text = ui->nn->text().toStdString();
        nn.init_weights();
    }

    t.reset();
    t.learning_rate = ui->learning_rate->value();
    t.w_decay_rate = ui->w_decay->value();
    t.b_decay_rate = ui->b_decay->value();
    t.momentum = ui->momentum->value();
    t.batch_size = ui->batch_size->value();
    t.epoch = ui->epoch->value();
    t.repeat = ui->repeat->value();

    terminated = false;
    future = std::async(std::launch::async, [this]
    {
        int round = 0;
        auto on_enumerate_epoch = [&](){
            std::ostringstream out;
            out << round << " testing error:" << nn.test_error(fp_data_test.data,fp_data_test.data_label) << "% "
                               << " training error:" << t.get_training_error() << "% "
                               << " decay = " << t.rate_decay << std::endl;
            log_text += out.str().c_str();
            ++round;
        };
        t.error_table.resize(nn.get_output_size()*nn.get_output_size());
        t.train(nn,fp_data,terminated, on_enumerate_epoch);
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
    ui->log->setText(log_text);
    ui->log->verticalScrollBar()->setValue(ui->log->verticalScrollBar()->maximum());
    if(nn.empty())
        return;
    if(ui->view_tab->currentIndex() == 1) //network view
    {
        auto& w = nn.get_layer(0)->weight;
        int n = w.size()/fp_index.size();
        int skip = 4;
        std::vector<tipl::color_image> w_map(n);

        for(int i = 0;i < n;++i)
            w_map[i].resize(tipl::geometry<2>(vbc->handle->dim[0],vbc->handle->dim[1]*vbc->handle->dim[2]/skip));


        float r = 512.0f/tipl::maximum(w);
        for(int i = 0;i < fp_index.size();++i)
            if(fp_index[i][2]%skip == 0) //
            {
                int pos = fp_index[i][0] + (fp_index[i][1]+(fp_index[i][2]/skip)*vbc->handle->dim[1])*vbc->handle->dim[0];
                for(int j = 0,w_offset = 0;j < n;++j,w_offset += fp_index.size())
                {
                    tipl::rgb color;
                    float v = w[w_offset + i];

                    if(v > 0)
                        color.b = std::min<int>(255,r*v);
                    else
                        color.r = std::min<int>(255,-r*v);
                    color.a = 255;
                    //color = 0xFFFFFFFF;
                    w_map[j][pos] = color;
                }
            }
        QImage buf(w_map[0].width()*n,w_map[0].height(),QImage::Format_RGB32);
        QPainter painter(&buf);
        for(int i = 0;i < n;++i)
        {
            painter.drawImage(i*w_map[0].width(),0,
                QImage((unsigned char*)&*w_map[i].begin(),w_map[i].width(),w_map[i].height(),QImage::Format_RGB32));
        }
        network_I = buf;
        show_view(network_scene,network_I);
    }
    if(ui->view_tab->currentIndex() == 2) // layer_view
    {
        std::vector<tipl::color_image> I;
        nn.get_layer_images(I);
        int h = 0,w = 10;
        for(int i = 0;i < I.size();++i)
        {
            h += I[i].height();
            w = std::max<int>(w,I[i].width());
        }
        QImage buf(w,h,QImage::Format_RGB32);
        QPainter painter(&buf);
        for(int i = 0,h_pos = 0;i < I.size();++i)
            if(!I[i].empty())
            {
                painter.drawImage(0,h_pos,QImage((unsigned char*)&*I[i].begin(),
                                                 I[i].width(),I[i].height(),QImage::Format_RGB32));
                h_pos += I[i].height()+10;
            }
        layer_I = buf;
        show_view(layer_scene,layer_I);
    }
}

void nn_connectometry::on_reset_clicked()
{
    on_stop_clicked();
    nn_text = "";
}
