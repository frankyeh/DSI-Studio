#include <QApplication>
#include <QFileInfo>
#include "boost/program_options.hpp"
#include "tracking/vbc_dialog.hpp"
#include "ui_vbc_dialog.h"

namespace po = boost::program_options;

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

int cnt(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "cnt: connectometry analysis")
    ("source", po::value<std::string>(), "assign the .db.fib file name")
    ("model", po::value<int>()->default_value(0), "assign the statistical model. 0:multiple regression 1: group comparison 2:paired comparison 3:individual")
    ("foi", po::value<int>(), "specify feature of interest (used only in multiple regression)")
    ("demo", po::value<std::string>(), "demographic file(s)")
    ("missing_value", po::value<float>(), "assign the missing value")
    ("threshold", po::value<float>()->default_value(0), "assign the threshold for tracking.")
    ("seeding_density", po::value<float>()->default_value(10), "assign the seeding_density for tracking")
    ("permutation", po::value<int>()->default_value(5000), "assign the number of permutation")
    ("thread", po::value<int>()->default_value(4), "assign the number of threads used in computation")
    ("track_fdr", po::value<float>(), "assign the FDR threshold for tracks")
    ("track_length", po::value<int>()->default_value(40), "assign the length threshold for tracks")
    ("normalized_qa", po::value<int>()->default_value(0), "specify whether qa will be normalized")


    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }


    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).run(), vm);
    po::notify(vm);

    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");

    std::auto_ptr<vbc_database> database(new vbc_database);
    database.reset(new vbc_database);
    std::cout << "reading connectometry db" <<std::endl;
    if(!database->load_database(vm["source"].as<std::string>().c_str()))
    {
        std::cout << "invalid database format" << std::endl;
        return 0;
    }
    vbc_dialog* vbc = new vbc_dialog(0,database.release(),QFileInfo(vm["source"].as<std::string>().c_str()).absoluteDir().absolutePath());
    vbc->setAttribute(Qt::WA_DeleteOnClose);
    vbc->show();

    if(!vm.count("demo"))
    {
        std::cout << "please assign demographic file" << std::endl;
        return 0;
    }

    switch(vm["model"].as<int>())
    {
    case 0:
        vbc->ui->rb_multiple_regression->setChecked(true);
        if(!vbc->load_demographic_file(vm["demo"].as<std::string>().c_str(),false))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        if(!vm.count("foi"))
        {
            std::cout << "please assign feature of interest using --foi" << std::endl;
            return 0;
        }
        vbc->ui->foi->setCurrentIndex(vm["foi"].as<int>());
        std::cout << "feature of interest=" << vbc->ui->foi->currentText().toLocal8Bit().begin() << std::endl;
        break;
    case 1:
        vbc->ui->rb_group_difference->setChecked(true);
        if(!vbc->load_demographic_file(vm["demo"].as<std::string>().c_str(),false))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        break;
    case 2:
        vbc->ui->rb_paired_difference->setChecked(true);
        if(!vbc->load_demographic_file(vm["demo"].as<std::string>().c_str(),false))
            return 0;
        std::cout << "demographic file loaded" << std::endl;
        break;
        /*
    case 3:
        vbc->ui->rb_individual_analysis->setChecked(true);
        break;
        */
    }
    if(vm.count("missing_value"))
    {
        vbc->ui->missing_data_checked->setChecked(true);
        vbc->ui->missing_value->setValue(vm["missing_value"].as<float>());
        std::cout << "missing value=" << vbc->ui->missing_value->value() << std::endl;
    }

    if(vm["threshold"].as<float>() == 0.0)
    {
        std::cout << "default threshold used" << std::endl;
        vbc->on_suggest_threshold_clicked();
        if(vbc->ui->rb_individual_analysis->isChecked())
            vbc->ui->percentile->setValue(5);
    }
    else
    {
        if(vbc->ui->rb_multiple_regression->isChecked())
            vbc->ui->t_threshold->setValue(vm["threshold"].as<float>());
        if(vbc->ui->rb_group_difference->isChecked() || vbc->ui->rb_paired_difference->isChecked())
            vbc->ui->percentage_dif->setValue(vm["threshold"].as<float>());
        if(vbc->ui->rb_individual_analysis->isChecked())
            vbc->ui->percentile->setValue(vm["threshold"].as<float>());
    }
    if(vbc->ui->rb_multiple_regression->isChecked())
        std::cout << "threshold=" << vbc->ui->t_threshold->value() << std::endl;
    if(vbc->ui->rb_group_difference->isChecked() || vbc->ui->rb_paired_difference->isChecked())
        std::cout << "threshold=" << vbc->ui->percentage_dif->value() << std::endl;
    if(vbc->ui->rb_individual_analysis->isChecked())
        std::cout << "threshold=" << vbc->ui->percentile->value() << std::endl;

    vbc->ui->seeding_density->setValue(vm["seeding_density"].as<float>());
    std::cout << "seeding_density=" << vbc->ui->seeding_density->value() << std::endl;

    vbc->ui->mr_permutation->setValue(vm["permutation"].as<int>());
    std::cout << "permutation=" << vbc->ui->mr_permutation->value() << std::endl;

    vbc->ui->multithread->setValue(vm["thread"].as<int>());
    std::cout << "thread=" << vbc->ui->multithread->value() << std::endl;

    if(vm["normalized_qa"].as<int>())
    {
        std::cout << "normalized qa" << std::endl;
        vbc->ui->normalize_qa->setChecked(true);
    }

    if(vm.count("track_fdr")) // use FDR threshold
    {
        vbc->ui->rb_FDR->setChecked(true);
        vbc->ui->fdr_control->setValue(vm["track_fdr"].as<float>());
        std::cout << "track_fdr=" << vbc->ui->fdr_control->value() << std::endl;
    }
    else
    {
        vbc->ui->rb_track_length->setChecked(true);
        vbc->ui->length_threshold->setValue(vm["track_length"].as<int>());
        std::cout << "track_length=" << vbc->ui->length_threshold->value() << std::endl;
    }
    std::cout << "running connectometry" << std::endl;
    vbc->on_run_clicked();
    vbc->vbc->threads->join_all();
    std::cout << "output results" << std::endl;
    vbc->calculate_FDR();
    return 0;
}
