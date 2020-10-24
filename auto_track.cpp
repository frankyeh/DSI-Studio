#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include "auto_track.h"
#include "ui_auto_track.h"
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "program_option.hpp"
extern std::vector<std::string> fa_template_list;
auto_track::auto_track(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::auto_track)
{
    ui->setupUi(this);
    progress = new QProgressBar(this);
    progress->setVisible(false);
    ui->statusbar->addPermanentWidget(progress);

    fib_data fib;
    fib.set_template_id(0);
    if(fib.tractography_name_list.empty())
    {
        QMessageBox::information(this,"Error",
            QString("Cannot find the template track file ")+QFileInfo(fa_template_list[0].c_str()).baseName()+".tt.gz"
            + " at folder " + QCoreApplication::applicationDirPath()+ "/track Please re-install the DSI Studio package");
    }
    QStringList tract_names;
    for(size_t index = 0;index < fib.tractography_name_list.size();++index)
        tract_names << fib.tractography_name_list[index].c_str();
    ui->candidate_list_view->addItems(tract_names);
    select_tracts();

    timer = std::make_shared<QTimer>(this);
    timer->stop();
    timer->setInterval(1000);
    connect(timer.get(),SIGNAL(timeout()),this,SLOT(check_status()));
    connect(ui->recommend_list,SIGNAL(clicked()),this,SLOT(select_tracts()));
    connect(ui->custom,SIGNAL(clicked()),this,SLOT(select_tracts()));
    connect(ui->cb_projection,SIGNAL(clicked()),this,SLOT(select_tracts()));
    connect(ui->cb_association,SIGNAL(clicked()),this,SLOT(select_tracts()));
    connect(ui->cb_commissural,SIGNAL(clicked()),this,SLOT(select_tracts()));
}

auto_track::~auto_track()
{
    delete ui;
}

void auto_track::on_open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open SRC files",
                                     "",
                                     "SRC files (*src.gz);;FIB files (*fib.gz);;4D NIFTI files (*nii.gz);;All files (*)" );
    if (filenames.isEmpty())
        return;
    file_list << filenames;
    update_list();
}

void auto_track::update_list()
{
    QStringList filenames;
    for(int index = 0;index < file_list.size();++index)
        filenames << QFileInfo(file_list[index]).fileName();
    ui->file_list_view->clear();
    ui->file_list_view->addItems(filenames);
    raise(); // for Mac
}
void auto_track::on_delete_2_clicked()
{
    if(ui->file_list_view->currentRow() < 0)
        return;
    file_list.erase(file_list.begin()+ui->file_list_view->currentRow());
    update_list();
}


void auto_track::on_delete_all_clicked()
{
    file_list.clear();
    update_list();
}


QStringList search_files(QString dir,QString filter);
void auto_track::on_open_dir_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                "");
    if(dir.isEmpty())
        return;
    file_list << search_files(dir,"*.src.gz");
    update_list();
}
int trk_post(std::shared_ptr<fib_data> handle,
             TractModel& tract_model,
             const std::string& file_name);
extern std::string auto_track_report;
extern program_option po;
std::string auto_track_report;
bool check_other_src(ImageModel& src);


struct file_holder{
    std::string file_name;
    file_holder(std::string file_name_):file_name(file_name_)
    {
        // create a zero-sized file to hold it
        std::ofstream(file_name.c_str());
    }
    ~file_holder()
    {
        // at the end, check if the file size is zero.
        if(QFileInfo(file_name.c_str()).size() == 0)
            QFile::remove(file_name.c_str());
    }
};

std::string run_auto_track(
                    const std::vector<std::string>& file_list,
                    const std::vector<unsigned int>& track_id,
                    float length_ratio,
                    float tolerance,
                    float track_voxel_ratio,
                    int interpolation,int tip,
                    bool export_stat,
                    bool export_trk,
                    bool overwrite,
                    bool default_mask,
                    int& progress)
{
    std::vector<std::string> reports(track_id.size());
    std::vector<std::vector<std::string> > stat_files(track_id.size());
    std::string dir = QFileInfo(file_list.front().c_str()).absolutePath().toStdString();

    fib_data fib;
    fib.set_template_id(0);
    std::string targets;
    for(unsigned int index = 0;index < track_id.size();++index)
    {
        targets += fib.tractography_name_list[track_id[index]];
        if(index+1 < track_id.size())
            targets += ", ";
    }

    std::vector<std::string> names;
    for(size_t i = 0;i < file_list.size() && !prog_aborted();++i)
    {
        std::string cur_file_base_name = QFileInfo(file_list[i].c_str()).baseName().toStdString();
        names.push_back(cur_file_base_name);
        progress = int(i);
        std::cout << "processing " << cur_file_base_name << std::endl;
        std::string fib_file_name;
        if(!QFileInfo(file_list[i].c_str()).exists())
            return std::string("cannot find file:")+file_list[i];

        // DWI reconstruction
        if(QString(file_list[i].c_str()).endsWith(".src.gz") ||
           QString(file_list[i].c_str()).endsWith(".nii.gz"))
        {
            ImageModel src;
            src.voxel.method_id = 4; // GQI
            src.voxel.param[0] = length_ratio;
            src.voxel.ti.init(8); // odf order of 8
            src.voxel.thread_count = std::thread::hardware_concurrency();
            // has fib file?
            fib_file_name = file_list[i]+src.get_file_ext();
            if(!QFileInfo(fib_file_name.c_str()).exists() || overwrite)
            {
                if (!src.load_from_file(file_list[i].c_str()))
                    return std::string("ERROR at ") + cur_file_base_name + ":" + src.error_msg;
                if(!src.is_human_data())
                    return std::string("ERROR at ") + cur_file_base_name + ": seems not human data";
                if(!check_other_src(src))
                    return std::string("ERROR at ") + cur_file_base_name;
                src.voxel.half_sphere = src.is_dsi_half_sphere();
                src.voxel.scheme_balance = src.need_scheme_balance();
                src.voxel.output_rdi = 1;
                src.voxel.check_btable = po.get("check_btable",1);
                if(interpolation)
                    src.rotate_to_mni(float(interpolation));
                begin_prog("reconstruct DWI");
                if(!default_mask)
                    std::fill(src.voxel.mask.begin(),src.voxel.mask.end(),1);
                if (!src.reconstruction())
                    return std::string("ERROR at ") + cur_file_base_name + ":" + src.error_msg;

                std::string mapping_file_name(fib_file_name);
                mapping_file_name += ".";
                mapping_file_name += QFileInfo(fa_template_list[0].c_str()).baseName().toLower().toStdString();
                mapping_file_name += ".inv.mapping.gz";
                if(QFileInfo(mapping_file_name.c_str()).exists())
                    QFile::remove(mapping_file_name.c_str());
            }
            if(!QFileInfo(fib_file_name.c_str()).exists())
                return std::string("fib file not generated for ") + file_list[i];
        }
        else
        {
            if(QString(file_list[i].c_str()).endsWith("fib.gz"))
                fib_file_name = file_list[i];
            else
                return std::string("unsupported file format :") + file_list[i];
        }


        // fiber tracking on fib file
        std::shared_ptr<fib_data> handle(new fib_data);
        bool fib_loaded = false;
        for(size_t j = 0;j < track_id.size() && !prog_aborted();++j)
        {
            std::string track_name = fib.tractography_name_list[track_id[j]];
            std::string no_result_file_name = fib_file_name+"."+track_name+".no_result.txt";
            std::string trk_file_name = fib_file_name+"."+track_name+".tt.gz";
            std::string stat_file_name = fib_file_name+"."+track_name+".stat.txt";
            std::string report_file_name = dir+"/"+track_name+".report.txt";

            stat_files[j].push_back(stat_file_name);

            if(QFileInfo(no_result_file_name.c_str()).exists() && !overwrite)
                continue;

            if( overwrite ||
               (export_stat && !QFileInfo(stat_file_name.c_str()).exists()) ||
               (export_trk && !QFileInfo(trk_file_name.c_str()).exists()))
            {
                file_holder state_file(stat_file_name),trk_file(trk_file_name);
                if (!fib_loaded && !handle->load_from_file(fib_file_name.c_str()))
                    return std::string("ERROR at ") + fib_file_name + ": Not human data. Check image resolution.";
                prog_init p("tracking ",track_name.c_str());
                fib_loaded = true;
                TractModel tract_model(handle.get());
                if(overwrite || !QFileInfo(trk_file_name.c_str()).exists() ||
                   !tract_model.load_from_file(trk_file_name.c_str()))
                {
                    ThreadData thread(handle.get());

                    thread.param.tip_iteration = uint8_t(tip);
                    thread.param.check_ending = !QString(track_name.c_str()).contains("Cingulum");
                    thread.param.stop_by_tract = 1;
                    if(!thread.roi_mgr->setAtlas(track_id[j],tolerance/handle->vs[0]))
                        return std::string("ERROR at ") + fib_file_name + ":" +handle->error_msg;
                    auto track_count = uint32_t(track_voxel_ratio*thread.roi_mgr->seeds.size());
                    thread.param.termination_count = track_count;
                    thread.param.max_seed_count = track_count*5000; //yield rate easy:1/100 hard:1/5000
                    // report
                    thread.roi_mgr->report += " The track-to-voxel ratio was set to ";
                    thread.roi_mgr->report += QString::number(double(track_voxel_ratio),'g',1).toStdString();
                    thread.roi_mgr->report += ".";
                    // run tracking
                    check_prog(0,track_count);

                    thread.run(tract_model.get_fib(),std::thread::hardware_concurrency(),false);

                    tract_model.report += thread.report.str();
                    tract_model.report += " Shape analysis (Yeh, Neuroimage, 2020) was conducted to derive shape metrics for tractography.";
                    if(reports[j].empty())
                        reports[j] = tract_model.report;

                    {
                        std::string temp_report = tract_model.report;
                        auto iter = temp_report.find(track_name);
                        temp_report.replace(iter,track_name.length(),targets);
                        // remove "A seeding region was placed at xxxxx"
                        iter = temp_report.find("A seeding region was placed at ");
                        temp_report.replace(iter+31,track_name.length(),
                                "the track region indicates by tractography atlas");


                        // remove "A total of xxxxx tracts were calculated."
                        iter = temp_report.find("tracts were calculated.");
                        auto iter2 = temp_report.find_first_of("A total of ",iter-20);
                        temp_report.replace(iter2,iter-iter2+23,"");
                        auto_track_report = temp_report;
                    }
                    bool no_result = false;
                    const unsigned int low_yield_threshold = 100000;
                    while(!thread.is_ended() && !prog_aborted())
                    {
                        check_prog(thread.get_total_tract_count(),track_count);
                        thread.fetchTracks(&tract_model);
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        // terminate if yield rate is very low, likely quality problem
                        if(thread.get_total_seed_count() > low_yield_threshold &&
                           thread.get_total_tract_count() < thread.get_total_seed_count()/low_yield_threshold)
                        {
                            no_result = true;
                            thread.end_thread();
                            break;
                        }

                    }
                    if(prog_aborted())
                        return std::string();
                    thread.fetchTracks(&tract_model);
                    thread.apply_tip(&tract_model);

                    if(no_result || tract_model.get_visible_track_count() == 0)
                    {
                        std::ofstream out(no_result_file_name.c_str());
                        continue;
                    }
                    tract_model.delete_repeated(1.0f);

                    if(export_trk)
                    {
                        if(!tract_model.save_tracts_to_file(trk_file_name.c_str()))
                            return std::string("fail to save trk file:")+trk_file_name;
                    }
                }

                if(export_stat && (overwrite || QFileInfo(stat_file_name.c_str()).size() == 0))
                {
                    std::ofstream out_stat(stat_file_name.c_str());
                    std::string result;
                    tract_model.get_quantitative_info(result);
                    out_stat << result;
                }

                trk_post(handle,tract_model,trk_file_name);
            }
        }
    }
    if(prog_aborted())
        return std::string();
    // check if there is any incomplete task
    {
        bool has_incomplete = false;
        for(size_t i = 0;i < stat_files.size();++i)
        {
            for(size_t j = 0;j < stat_files[i].size();++j)
                if(QFileInfo(stat_files[i][j].c_str()).exists() &&
                   QFileInfo(stat_files[i][j].c_str()).size() == 0)
                {
                    std::cout << "remove empty file:" << stat_files[i][j] << std::endl;
                    QFile::remove(stat_files[i][j].c_str());
                    has_incomplete = true;
                }
        }
        if(has_incomplete)
            return "Incomplete tasked found. Please rerun the analysis.";
    }

    // aggregating
    if(file_list.size() != 1)
    {
        std::string column_title("Subjects");
        for(size_t s = 0;s < names.size();++s) // for each scan
        {
            column_title += "\t";
            column_title += names[s];
        }
        std::vector<std::string> metrics_names; // row titles are metrics
        {
            std::ifstream in(stat_files[0][0].c_str());
            std::string line;
            for(size_t m = 0;std::getline(in,line);++m)
            {
                auto sep = line.find('\t');
                metrics_names.push_back(line.substr(0,sep));
            }
        }

        std::ofstream all_out((QFileInfo(stat_files[0][0].c_str()).absolutePath()+"/all_results_tract_wise.txt").toStdString().c_str());
        std::vector<std::string> all_out2_text;
        for(size_t t = 0;t < track_id.size();++t) // for each track
        {
            std::vector<std::vector<std::string> > output(names.size());
            for(size_t s = 0;s < output.size();++s) // for each scan
            {
                std::cout << "reading " << stat_files[t][s] << std::endl;
                std::ifstream in(stat_files[t][s].c_str());
                if(!in)
                    continue;
                std::vector<std::string> lines;
                {
                    std::string line;
                    while(std::getline(in,line))
                        lines.push_back(line);
                }
                if(lines.size() < metrics_names.size())
                {
                    std::string error("inconsistent stat file (remove it and rerun):");
                    error += QFileInfo(stat_files[t][s].c_str()).fileName().toStdString();
                    error += " metrics count=";
                    error += std::to_string(lines.size());
                    error += " others=";
                    error += std::to_string(metrics_names.size());
                    return error;
                }
                for(size_t m = 0;m < metrics_names.size();++m)
                    output[s].push_back(lines[m].substr(lines[m].find('\t')+1));
            }
            std::string track_name = fib.tractography_name_list[track_id[t]];
            std::ofstream out((dir+"/"+track_name+".stat.txt").c_str());
            out << column_title << std::endl;
            if(t == 0)
                all_out << "Tract\t" << column_title << std::endl;
            for(size_t m = 0;m < metrics_names.size();++m)
            {
                std::string metrics_output(metrics_names[m]);
                for(size_t s = 0;s < names.size();++s)
                {
                    metrics_output += "\t";
                    if(m < output[s].size())
                        metrics_output += output[s][m];
                }
                out << metrics_output << std::endl;
                all_out << track_name << "\t" << metrics_output << std::endl;
            }
            if(t == 0)
                all_out2_text.resize(output.size()*metrics_names.size());
            for(size_t s = 0,index = 0;s < names.size();++s)
                for(size_t m = 0;m < metrics_names.size();++m,++index)
                {
                    if(t == 0)
                    {
                        all_out2_text[index] = names[s];
                        all_out2_text[index] += "\t";
                        all_out2_text[index] += metrics_names[m];
                    }
                    all_out2_text[index] += "\t";
                    if(m < output[s].size())
                        all_out2_text[index] += output[s][m];
                }
        }

        std::ofstream all_out2((QFileInfo(stat_files[0][0].c_str()).absolutePath()+"/all_results_subject_wise.txt").toStdString().c_str());
        all_out2 << "Subjects\tMetrics";
        for(size_t t = 0;t < track_id.size();++t) // for each tract
            all_out2 << "\t" << fib.tractography_name_list[track_id[t]];
        all_out2 << std::endl;
        for(size_t index = 0;index < all_out2_text.size();++index) // for each tract
            all_out2 << all_out2_text[index] << std::endl;
    }
    return std::string();
}
void auto_track::check_status()
{
    progress->setValue(prog);
    ui->file_list_view->setCurrentRow(prog);
    if(!auto_track_report.empty())
        ui->report->setText(auto_track_report.c_str());
}
void auto_track::on_run_clicked()
{
    std::vector<std::string> file_list2;
    std::vector<unsigned int> track_id;

    QModelIndexList indexes = ui->candidate_list_view->selectionModel()->selectedRows();
    for(int i = 0;i < indexes.count();++i)
        track_id.push_back(uint32_t(indexes[i].row()));
    for(int i = 0;i < file_list.size();++i)
        file_list2.push_back(file_list[i].toStdString());
    if(track_id.empty())
    {
        QMessageBox::information(this,"DSI Studio","Please select target tracks");
        return;
    }
    if(file_list2.empty())
    {
        QMessageBox::information(this,"DSI Studio","Please assign SRC files");
        return;
    }
    ui->run->setEnabled(false);
    progress->setValue(0);
    progress->setVisible(true);
    progress->setMaximum(file_list.size()-1);
    prog = 0;
    timer->start(5000);
    begin_prog("");
    std::string error = run_auto_track(file_list2,track_id,
                   float(ui->gqi_l->value()),
                   float(ui->tolerance->value()),
                   float(ui->track_voxel_ratio->value()),
                   ui->interpolation->currentIndex(),ui->pruning->value(),
                   ui->export_stat->isChecked(),
                   ui->export_trk->isChecked(),
                   ui->overwrite->isChecked(),
                   ui->default_mask->isChecked(),
                   prog);
    timer->stop();
    ui->run->setEnabled(true);
    progress->setVisible(false);

    if(!prog_aborted())
    {
        if(error.empty())
            QMessageBox::information(this,"DSI Studio","Completed");
        else
            QMessageBox::information(this,"DSI Studio",error.c_str());
    }
    close_prog();
    raise(); //  for mac
}


void auto_track::on_interpolation_currentIndexChanged(int)
{
    QMessageBox::information(this,"DSI Studio","You may need to remove existing *.fib.gz and *.mapping.gz files to take effect");

}

void auto_track::select_tracts()
{
    if(ui->recommend_list->isChecked())
    {
        ui->candidate_list_view->setEnabled(false);
        ui->recom_panel->setEnabled(true);
        std::vector<std::string> select_list;
        if(ui->cb_association->isChecked())
        {
            select_list.push_back("Fasciculus");
            select_list.push_back("Cingulum");
            select_list.push_back("Aslant");
        }
        if(ui->cb_projection->isChecked())
        {
            select_list.push_back("Corticospinal");
            select_list.push_back("Thalamic");
            select_list.push_back("Optic");
            select_list.push_back("Fornix");
        }
        if(ui->cb_commissural->isChecked())
        {
            select_list.push_back("Body");
            select_list.push_back("Forceps");
        }
        for(int i = 0;i < ui->candidate_list_view->count();++i)
            ui->candidate_list_view->item(i)->setSelected(false);

        for(size_t j = 0;j < select_list.size();++j)
        for(int i = 0;i < ui->candidate_list_view->count();++i)
            if(ui->candidate_list_view->item(i)->text().contains(select_list[j].c_str()))
                ui->candidate_list_view->item(i)->setSelected(true);
    }
    else
    {
        ui->candidate_list_view->setEnabled(true);
        ui->recom_panel->setEnabled(false);
    }
}
