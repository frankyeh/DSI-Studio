#include <QFileInfo>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include "nn_connectometry.h"
#include "ui_nn_connectometry.h"
nn_connectometry::nn_connectometry(QWidget *parent,std::shared_ptr<fib_data> handle,QString db_file_name_,bool gui_) :
    QDialog(parent),nna(handle),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),gui(gui_),
    ui(new Ui::nn_connectometry)
{
    ui->setupUi(this);
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
    nna.t.momentum = 0.0f;
    nna.t.batch_size = 64;
    nna.t.epoch = ui->epoch->value();
    nna.foi_index = ui->foi->currentIndex();
    nna.is_regression = ui->nn_regression->isChecked();
    nna.regress_all = ui->regress_all->isChecked();
    nna.seed_search = ui->seed_search->value();
    nna.otsu = ui->otsu->value();
    nna.cv_fold = 10;
    nna.normalize_value = ui->norm_output->isEnabled() && ui->norm_output->isChecked();
    //nna.t.error_table.resize(nna.nn.get_output_size()*nna.nn.get_output_size());
    if(!nna.run(out,ui->network_text->text().toStdString()))
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
    float scroll_ratio = (float)ui->log->verticalScrollBar()->value()/(float)(ui->log->verticalScrollBar()->maximum()+1);
    log_text += out.str().c_str();

    if(nna.terminated && timer)
    {
        log_text += nna.report.c_str();
        timer->stop();
    }
    out.str("");
    out.clear();
    ui->log->setText(log_text);
    ui->log->verticalScrollBar()->setValue((ui->log->verticalScrollBar()->maximum()+1)*scroll_ratio);
    if(!nna.nn.initialized)
        return;

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
        if(!nna.test_mresult.empty())
        {
            int index = ui->foi->currentIndex();
            nna.test_result.resize(nna.test_mresult.size());
            nna.fp_data.data_label.resize(nna.fp_mdata.data_label.size());
            for(int i = 0;i < nna.test_result.size();++i)
            {
                nna.test_result[i] = nna.test_mresult[i][index];
                nna.fp_data.data_label[nna.test_seq[i]] = nna.fp_mdata.data_label[nna.test_seq[i]][index];
            }
        }
        if(!nna.test_result.empty())
        {
            if(ui->test_subjects->rowCount() != nna.test_result.size())
            {
                ui->test_subjects->setRowCount(0);
                ui->test_subjects->setColumnCount(3);
                ui->test_subjects->setHorizontalHeaderLabels(QStringList() << "Subject" << "Label" << "Predicted");
                ui->test_subjects->setRowCount(nna.test_result.size());
                for(unsigned int row = 0;row < nna.test_result.size();++row)
                {
                    int id = nna.subject_index[nna.test_seq[row]];
                    ui->test_subjects->setItem(row,0,
                                               new QTableWidgetItem(QString(nna.handle->db.subject_names[id].c_str())));
                    ui->test_subjects->setItem(row,1,new QTableWidgetItem(QString()));
                    ui->test_subjects->setItem(row,2,new QTableWidgetItem(QString()));
                }
            }

            QVector<double> x(nna.test_result.size());
            QVector<double> y(nna.test_result.size());
            double x_min = 100,x_max = -100,y_min = 100,y_max = -100;
            for(unsigned int row = 0;row < nna.test_result.size();++row)
            {
                x[row] = nna.test_result[row]/nna.sl_scale+nna.sl_mean;
                y[row] = nna.fp_data.data_label[nna.test_seq[row]]/nna.sl_scale+nna.sl_mean;
                ui->test_subjects->item(row,1)->setText(QString::number(y[row]));
                ui->test_subjects->item(row,2)->setText(QString::number(x[row]));
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

void nn_connectometry::on_regress_all_clicked()
{
    ui->nn_classification->setEnabled(!ui->regress_all->isChecked());
    if(ui->regress_all->isChecked())
    {
        ui->nn_classification->setChecked(false);
        ui->nn_regression->setChecked(true);
    }
}

void nn_connectometry::on_nn_regression_toggled(bool checked)
{
    ui->norm_output->setEnabled(checked);
}
