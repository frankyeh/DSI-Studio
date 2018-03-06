#include <QApplication>
#include <QFileInfo>
#include "connectometry/group_connectometry.hpp"
#include "ui_group_connectometry.h"
#include "program_option.hpp"

int cnt(void)
{
    std::shared_ptr<vbc_database> database(new vbc_database);
    std::cout << "reading connectometry db" <<std::endl;
    if(!database->load_database(po.get("source").c_str()))
    {
        std::cout << "invalid database format" << std::endl;
        return 0;
    }
    std::auto_ptr<group_connectometry> vbc(new group_connectometry(0,database,po.get("source").c_str(),false));
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
    default:
        std::cout << "Unknown model" <<std::endl;
        return 0;
    }

    if(po.has("missing_value"))
    {
        vbc->ui->missing_data_checked->setChecked(true);
        vbc->ui->missing_value->setValue(po.get("missing_value",float(0)));
        std::cout << "missing value=" << vbc->ui->missing_value->value() << std::endl;
    }

    vbc->ui->threshold->setValue(po.get("threshold",float(vbc->ui->threshold->value())));

    std::cout << "threshold=" << vbc->ui->threshold->value() << std::endl;

    vbc->ui->seed_ratio->setValue(po.get("seed_ratio",5.0f));
    std::cout << "seed_ratio=" << vbc->ui->seed_ratio->value() << std::endl;

    vbc->ui->permutation_count->setValue(po.get("permutation",int(2000)));
    std::cout << "permutation=" << vbc->ui->permutation_count->value() << std::endl;

    vbc->ui->multithread->setValue(po.get("thread_count",int(std::thread::hardware_concurrency())));
    std::cout << "thread=" << vbc->ui->multithread->value() << std::endl;

    if(po.get("normalized_qa",int(0)))
    {
        std::cout << "normalized qa" << std::endl;
        vbc->ui->normalize_qa->setChecked(true);
    }

    vbc->ui->output_report->setChecked(po.get("output_report",int(1)));
    vbc->ui->output_track_image->setChecked(po.get("output_track_image",int(1)));
    vbc->ui->output_track_data->setChecked(po.get("output_track_data",int(1)));
    vbc->ui->output_fdr->setChecked(po.get("output_fdr",int(0)));
    vbc->ui->output_dist->setChecked(po.get("output_dist",int(0)));

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
