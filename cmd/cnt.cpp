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
    //vbc->hide();

    if(!po.has("demo"))
    {
        std::cout << "please assign demographic file" << std::endl;
        return 0;
    }
    if(!vbc->load_demographic_file(po.get("demo").c_str()))
    {
        std::cout << "Invalid demographic file" << std::endl;
        return 0;
    }


    if(po.has("missing_value"))
    {
        vbc->ui->missing_data_checked->setChecked(true);
        vbc->ui->missing_value->setValue(po.get("missing_value",9999));
        std::cout << "missing value=" << vbc->ui->missing_value->value() << std::endl;
    }


    if(!po.has("voi") || !po.has("variable_list"))
    {
        std::cout << "please assign --voi and --variable_list" << std::endl;
        return 0;
    }
    {
        std::string var_text = po.get("variable_list");
        std::replace(var_text.begin(),var_text.end(),',',' ');
        std::istringstream var_in(var_text);
        std::vector<int> variable_list(
                    (std::istream_iterator<int>(var_in)),
                    (std::istream_iterator<int>()));

        for(int i = 0;i < vbc->ui->variable_list->count();++i)
            vbc->ui->variable_list->item(i)->setCheckState(Qt::Unchecked);
        std::cout << "variables=";

        int voi_index = po.get("voi",0);
        int voi_sel = -1;
        std::string voi_text;
        for(int i = 0;i < variable_list.size();++i)
        {
            int index = variable_list[i];
            if(index >= vbc->ui->variable_list->count())
            {
                std::cout << "Invalid number in the variable_list:" << index << std::endl;
                return 0;
            }
            if(index == voi_index)
            {
                voi_sel = vbc->ui->variable_list->count();
                voi_text = vbc->ui->variable_list->item(index)->text().toStdString();
            }
            vbc->ui->variable_list->item(index)->setCheckState(Qt::Checked);
            if(i)
                std::cout << ",";
            std::cout << vbc->ui->variable_list->item(index)->text().toStdString();
        }
        std::cout << std::endl;
        if(voi_sel == -1)
        {
            std::cout << "variable of interest is not included in the variable list" << std::endl;
            return 0;
        }
        vbc->on_variable_list_clicked(QModelIndex());
        vbc->ui->foi->update();
        vbc->ui->foi->setCurrentText(voi_text.c_str());
        std::cout << "study variable=" << vbc->ui->foi->currentText().toStdString() << std::endl;

    }

    vbc->ui->threshold->setValue(po.get("t_threshold",float(vbc->ui->threshold->value())));
    std::cout << "t_threshold=" << vbc->ui->threshold->value() << std::endl;

    vbc->ui->seed_ratio->setValue(po.get("seed_ratio",5.0f));
    std::cout << "seed_ratio=" << vbc->ui->seed_ratio->value() << std::endl;

    vbc->ui->permutation_count->setValue(po.get("permutation",int(2000)));
    std::cout << "permutation=" << vbc->ui->permutation_count->value() << std::endl;

    vbc->ui->multithread->setValue(po.get("thread_count",int(std::thread::hardware_concurrency())));
    std::cout << "thread=" << vbc->ui->multithread->value() << std::endl;

    vbc->ui->track_trimming->setValue(po.get("trim",1));
    std::cout << "trim=" << vbc->ui->track_trimming->value() << std::endl;


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
    if(po.has("fdr_threshold"))
    {
        vbc->ui->rb_fdr->setChecked(true);
        vbc->ui->fdr_threshold->setValue(po.get("fdr_threshold",0.05f));
        std::cout << "fdr_threshold=" << vbc->ui->fdr_threshold->value() << std::endl;
    }
    else
    {
        vbc->ui->rb_length->setChecked(true);
        vbc->ui->length_threshold->setValue(po.get("length_threshold",int(40)));
        std::cout << "length_threshold=" << vbc->ui->length_threshold->value() << std::endl;
    }

    vbc->on_run_clicked();
    std::cout << vbc->vbc->report << std::endl;
    std::cout << "running connectometry" << std::endl;
    vbc->vbc->wait();
    std::cout << "output results" << std::endl;
    vbc->calculate_FDR();
    std::cout << "close GUI" << std::endl;
    vbc->close();
    vbc.reset(0);
    return 0;
}
