#include <QFileInfo>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include "nn_connectometry.h"
#include "ui_nn_connectometry.h"
nn_connectometry::nn_connectometry(QWidget *parent,std::shared_ptr<fib_data> handle,QString db_file_name_,bool gui_) :
    QDialog(parent),
    chart1(new QChart),chart1_view(new QChartView(chart1)),
    chart2(new QChart),chart2_view(new QChartView(chart2)),
    chart3(new QChart),chart3_view(new QChartView(chart3)),
    chart4(new QChart),chart4_view(new QChartView(chart4)),
    nna(handle),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
    ui(new Ui::nn_connectometry)
{
    ui->setupUi(this);
    ui->report_layout->addWidget(chart1_view,0,0);
    ui->report_layout->addWidget(chart2_view,0,1);
    ui->report_layout->addWidget(chart3_view,1,0);
    ui->report_layout->addWidget(chart4_view,1,1);
    chart1->legend()->setVisible(false);
    chart2->legend()->setVisible(false);
    chart3->legend()->setVisible(false);
    chart4->legend()->setVisible(false);


    ui->network_view->setScene(&network_scene);
    ui->layer_view->setScene(&layer_scene);
    log_text += nna.handle->report.c_str();
    log_text += "\r\n";
    ui->log->setText(log_text);
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
    auto& db = nna.handle->db;
    // read demographic file
    if(!db.parse_demo(filename.toStdString(),9999))
    {
        QMessageBox::information(this,"Error",db.error_msg.c_str(),0);
        return;
    }
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

    on_stop_clicked();

    nna.t.learning_rate = ui->learning_rate->value()*0.01f;
    nna.t.momentum = ui->momentum->value();
    nna.t.batch_size = 64;
    nna.t.epoch = ui->epoch->value();
    nna.foi_index = ui->foi->currentIndex();
    nna.is_regression = ui->nn_regression->isChecked();
    nna.seed_search = 0;
    nna.otsu = ui->otsu->value();
    nna.cv_fold = 10;
    nna.normalize_value = ui->norm_output->isEnabled() && ui->norm_output->isChecked();
    //nna.t.error_table.resize(nna.nn.get_output_size()*nna.nn.get_output_size());
    if(!nna.run(ui->network_text->text().toStdString()))
    {
        QMessageBox::information(this,"Error",nna.error_msg.c_str(),0);
        return;
    }


    ui->test_subjects->setRowCount(0);
    if(timer)
        delete timer;
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update_network()));
    timer->setInterval(1000);
    timer->start();
}
void nn_connectometry::on_stop_clicked()
{
    ui->progressBar->setValue(0);
    nna.stop();
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
    if(nna.terminated && timer)
        on_stop_clicked();
    if(!nna.nn.initialized)
        return;
    log_text = nna.handle->report.c_str();
    log_text += nna.report.c_str();
    log_text += nna.all_result.c_str();
    ui->log->setText(log_text);
    ui->progressBar->setValue(nna.cur_progress);
    if(ui->view_tab->currentIndex() == 0 && nna.has_results()) //report view
    {
        {
            QScatterSeries *series = new QScatterSeries();
            series->setMarkerSize(3.0);
            series->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
            series->setPen(Qt::NoPen);
            series->setBrush(Qt::black);
            for(int row = 0;row < nna.test_result.size();++row)
            {
                float x,y;
                series->append(x = nna.test_result[row]/nna.sl_scale+nna.sl_mean,
                               y = nna.fp_data.data_label[nna.test_seq[row]]/nna.sl_scale+nna.sl_mean);
            }
            chart1->removeAllSeries();
            chart1->addSeries(series);
            chart1->createDefaultAxes();
            chart1->setDropShadowEnabled(false);
            chart1->axes(Qt::Horizontal).back()->setTitleText("Predicted Value");
            chart1->axes(Qt::Vertical).back()->setTitleText("Value");
            chart1->axes(Qt::Horizontal).back()->setRange(
                        ((QValueAxis*)chart1->axes(Qt::Vertical).back())->min(),
                        ((QValueAxis*)chart1->axes(Qt::Vertical).back())->max());

            chart1->axes(Qt::Horizontal).back()->setGridLineVisible(false);
            chart1->axes(Qt::Vertical).back()->setGridLineVisible(false);

            chart1->setTitle("Predicted versus True Vlues");

        }

        {
            QLineSeries *s1 = new QLineSeries();
            QLineSeries *s2 = new QLineSeries();
            QLineSeries *s3 = new QLineSeries();
            nna.get_results([&](size_t epoch,float r,float mae,float error){
                s1->append(epoch,static_cast<qreal>(r));
                s2->append(epoch,static_cast<qreal>(mae));
                s3->append(epoch,static_cast<qreal>(error));
            });
            chart2->removeAllSeries();
            chart2->addSeries(s1);
            chart2->createDefaultAxes();
            chart2->setTitle(nna.is_regression ? "correlation coefficient (cross-validated)":"missed count");
            chart2->axes(Qt::Horizontal).back()->setTitleText("epoch");
            chart2->axes(Qt::Vertical).back()->setMin(0);


            ((QValueAxis*)chart2->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
            ((QValueAxis*)chart2->axes(Qt::Horizontal).back())->setTickInterval(100);

            chart3->removeAllSeries();
            chart3->addSeries(s2);
            chart3->createDefaultAxes();
            chart3->setTitle(nna.is_regression ? "mean absolute error (cross-validated)" : "accuracy");
            chart3->axes(Qt::Horizontal).back()->setTitleText("epoch");
            chart3->axes(Qt::Vertical).back()->setMin(0);

            ((QValueAxis*)chart3->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
            ((QValueAxis*)chart3->axes(Qt::Horizontal).back())->setTickInterval(100);


            chart4->removeAllSeries();
            chart4->addSeries(s3);
            chart4->createDefaultAxes();
            chart4->setTitle("training error");
            chart4->axes(Qt::Horizontal).back()->setTitleText("epoch");
            chart4->axes(Qt::Vertical).back()->setMin(0);

            ((QValueAxis*)chart4->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
            ((QValueAxis*)chart4->axes(Qt::Horizontal).back())->setTickInterval(100);
        }
    }

    if(ui->view_tab->currentIndex() == 1) //network view
    {
        tipl::color_image I;
        nna.get_salient_map(I);
        if(!I.empty())
        {
            network_I = QImage((unsigned char*)&*I.begin(),I.width(),I.height(),QImage::Format_RGB32).scaledToHeight(1600);
            show_view(network_scene,network_I);
        }
    }
    if(ui->view_tab->currentIndex() == 2) // layer_view
    {
        tipl::color_image I;
        nna.get_layer_map(I);
        layer_I = QImage((unsigned char*)&*I.begin(),
                         I.width(),I.height(),QImage::Format_RGB32).scaledToWidth(ui->view_tab->width()-50);
        show_view(layer_scene,layer_I);
    }
    if(ui->view_tab->currentIndex() == 3) // predict view
    {
        if(!nna.all_test_result.empty())
        {
            if(ui->test_subjects->rowCount() != nna.all_test_result.size())
            {
                ui->test_subjects->setRowCount(0);
                ui->test_subjects->setColumnCount(3);
                ui->test_subjects->setHorizontalHeaderLabels(QStringList() << "Subject" << "Label" << "Predicted");
                ui->test_subjects->setRowCount(nna.all_test_result.size());
                for(int row = 0;row < nna.all_test_result.size();++row)
                {
                    int id = nna.subject_index[nna.all_test_seq[row]];
                    ui->test_subjects->setItem(row,0,
                                               new QTableWidgetItem(QString(nna.handle->db.subject_names[id].c_str())));
                    ui->test_subjects->setItem(row,1,new QTableWidgetItem(QString()));
                    ui->test_subjects->setItem(row,2,new QTableWidgetItem(QString()));
                }
                std::vector<float> allx,ally;
                for(int row = 0;row < nna.all_test_result.size();++row)
                {
                    float x = nna.all_test_result[row]/nna.sl_scale+nna.sl_mean;
                    float y = nna.fp_data.data_label[nna.all_test_seq[row]]/nna.sl_scale+nna.sl_mean;
                    ui->test_subjects->item(row,1)->setText(QString::number(y));
                    ui->test_subjects->item(row,2)->setText(QString::number(x));
                    allx.push_back(x);
                    ally.push_back(y);
                }

            }
        }
    }

}

void nn_connectometry::on_reset_clicked()
{
    on_stop_clicked();
    nna.nn.reset();
}

void nn_connectometry::on_foi_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    ui->nn_classification->setEnabled(true);
    ui->nn_classification->setChecked(true);
    std::set<float> num_classes;
    for(int i = 0;i < nna.handle->db.num_subjects;++i)
    {
        float label = nna.handle->db.X[i*(ui->foi->count()+1) + 1 + ui->foi->currentIndex()];
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

void nn_connectometry::on_nn_regression_toggled(bool checked)
{
    ui->norm_output->setEnabled(checked);
}
