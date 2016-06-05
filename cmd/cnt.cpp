#include <QApplication>
#include <QFileInfo>
#include "tracking/vbc_dialog.hpp"
#include "ui_vbc_dialog.h"
#include "program_option.hpp"

int cnt(void)
{
    std::auto_ptr<vbc_database> database(new vbc_database);
    database.reset(new vbc_database);
    std::cout << "reading connectometry db" <<std::endl;
    if(!database->load_database(po.get("source").c_str()))
    {
        std::cout << "invalid database format" << std::endl;
        return 0;
    }
    std::auto_ptr<vbc_dialog> vbc(new vbc_dialog(0,database.release(),po.get("source").c_str(),false));
    vbc->setAttribute(Qt::WA_DeleteOnClose);
    vbc->show();
    vbc->hide();

    if(!po.has("demo"))
    {
        std::cout << "please assign demographic file" << std::endl;
        return 0;
    }

    switch(po.get("model",int(0)))
    {
    case 0:
        vbc->ui->rb_multiple_regression->setChecked(true);
        if(!vbc->load_demographic_file(po.get("demo").c_str()))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        if(!po.has("foi"))
        {
            std::cout << "please assign feature of interest using --foi" << std::endl;
            return 0;
        }
        vbc->ui->foi->setCurrentIndex(po.get("foi",int(0)));
        std::cout << "feature of interest=" << vbc->ui->foi->currentText().toLocal8Bit().begin() << std::endl;
        break;
    case 1:
        vbc->ui->rb_group_difference->setChecked(true);
        if(!vbc->load_demographic_file(po.get("demo").c_str()))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        break;
    case 2:
        vbc->ui->rb_paired_difference->setChecked(true);
        if(!vbc->load_demographic_file(po.get("demo").c_str()))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        break;
    case 3:
        //vbc->ui->rb_individual_analysis->setChecked(true);
        std::cout << "Individual connectometry has not yet been implemented in command line. Please email frank to request this function" <<std::endl;
        return 0;
    }
    int threshold_type = po.get("threshold_type",int(0));
    // percentage = 0,t = 1,beta = 2,percentile = 3,mean_dif = 4
    switch(threshold_type)
    {
    case 0:
        vbc->ui->rb_percentage->setChecked(true);
        std::cout << "threshold_type=percentage change" << std::endl;
        break;
    case 1:
        vbc->ui->rb_t_stat->setChecked(true);
        std::cout << "threshold_type=t statistics" << std::endl;
        break;
    case 2:
        vbc->ui->rb_beta->setChecked(true);
        std::cout << "threshold_type=beta coefficient" << std::endl;
        break;
    case 3:
        vbc->ui->rb_percentile->setChecked(true);
        std::cout << "threshold_type=percentile rank" << std::endl;
        break;
    case 4:
        vbc->ui->rb_mean_dif->setChecked(true);
        std::cout << "threshold_type=mean difference" << std::endl;
        break;
    default:
        std::cout << "unknown threshold type:" << threshold_type << std::endl;
        return -1;
    }

    if(po.has("missing_value"))
    {
        vbc->ui->missing_data_checked->setChecked(true);
        vbc->ui->missing_value->setValue(po.get("missing_value",float(0)));
        std::cout << "missing value=" << vbc->ui->missing_value->value() << std::endl;
    }

    if(vbc->ui->rb_individual_analysis->isChecked())
        vbc->ui->threshold->setValue(5);
    else
        vbc->on_suggest_threshold_clicked();
    vbc->ui->threshold->setValue(po.get("threshold",float(vbc->ui->threshold->value())));

    std::cout << "threshold=" << vbc->ui->threshold->value() << std::endl;

    vbc->ui->seeding_density->setValue(po.get("seeding_density",float(10)));
    std::cout << "seeding_density=" << vbc->ui->seeding_density->value() << std::endl;

    vbc->ui->mr_permutation->setValue(po.get("permutation",int(5000)));
    std::cout << "permutation=" << vbc->ui->mr_permutation->value() << std::endl;

    vbc->ui->multithread->setValue(po.get("thread_count",int(std::thread::hardware_concurrency())));
    std::cout << "thread=" << vbc->ui->multithread->value() << std::endl;

    if(po.get("normalized_qa",int(0)))
    {
        std::cout << "normalized qa" << std::endl;
        vbc->ui->normalize_qa->setChecked(true);
    }    
    vbc->ui->length_threshold->setValue(po.get("track_length",int(40)));
    std::cout << "track_length=" << vbc->ui->length_threshold->value() << std::endl;
    std::cout << "running connectometry" << std::endl;
    vbc->on_run_clicked();
    for(int i = 0;i < vbc->vbc->threads.size();++i)
        vbc->vbc->threads[i]->wait();
    std::cout << "output results" << std::endl;
    vbc->calculate_FDR();
    std::cout << "close GUI" << std::endl;
    vbc->close();
    vbc.reset(0);
    return 0;
}
