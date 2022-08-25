#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include "auto_track.h"
#include "ui_auto_track.h"
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "program_option.hpp"
#include <filesystem>
extern std::vector<std::string> fa_template_list;
auto_track::auto_track(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::auto_track)
{
    ui->setupUi(this);
    ui->thread_count->setMaximum(std::thread::hardware_concurrency()*2);
    ui->thread_count->setValue(std::thread::hardware_concurrency());
    progress_bar = new QProgressBar(this);
    progress_bar->setVisible(false);
    ui->statusbar->addPermanentWidget(progress_bar);

    fib_data fib;
    fib.set_template_id(0);
    if(fib.tractography_name_list.empty())
    {
        QMessageBox::critical(this,"ERROR",
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
    connect(ui->cb_brainstem,SIGNAL(clicked()),this,SLOT(select_tracts()));

}

auto_track::~auto_track()
{
    delete ui;
}

void auto_track::on_open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open FIB files",
                                     "",
                                     "FIB files (*fib.gz);;All files (*)" );
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
    file_list << search_files(dir,"*fib.gz");
    update_list();
}
extern std::string auto_track_report;
std::string auto_track_report;

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
        if(std::filesystem::exists(file_name) && !std::filesystem::file_size(file_name))
            std::filesystem::remove(file_name);
    }
};

std::string run_auto_track(program_option& po,const std::vector<std::string>& file_list,const std::vector<unsigned int>& track_id,int& prog)
{
    std::string tolerance_string = po.get("tolerance","16,18,20");
    float track_voxel_ratio = po.get("track_voxel_ratio",2.0f);
    float yield_rate = po.get("yield_rate",0.00001f);
    size_t yield_check_count = 20.0f/yield_rate;
    int tip = po.get("tip",32);
    bool export_stat = po.get("export_stat",1);
    bool export_trk = po.get("export_trk",1);
    bool overwrite = po.get("overwrite",0);
    bool default_mask = po.get("default_mask",0);
    bool export_template_trk = po.get("export_template_trk",0);
    bool check_ending = po.get("check_ending",1);
    uint32_t thread_count = uint32_t(po.get("thread_count",std::thread::hardware_concurrency()));


    std::vector<float> tolerance;
    {
        std::istringstream in(tolerance_string);
        std::string num;
        while(std::getline(in,num,','))
        {
            std::istringstream in2(num);
            float t;
            if(!(in2 >> t))
            {
                return std::string("Cannot parse tolerance number: ")+num;
            }
            tolerance.push_back(t);
        }
        if(tolerance.empty())
            return "Please assign tolerance distance";
    }

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
    progress prog0("automatic fiber tracking");
    for(size_t i = 0;progress::at(i,file_list.size());++i)
    {
        std::string cur_file_base_name = QFileInfo(file_list[i].c_str()).baseName().toStdString();
        progress prog1(cur_file_base_name.c_str());
        names.push_back(cur_file_base_name);
        prog = int(i);
        show_progress() << "processing " << cur_file_base_name << std::endl;
        std::string fib_file_name;
        if(!std::filesystem::exists(file_list[i]))
            return std::string("cannot find file:")+file_list[i];

        if(QString(file_list[i].c_str()).endsWith("fib.gz"))
            fib_file_name = file_list[i];
        else
        {
            if(QString(file_list[i].c_str()).endsWith(".src.gz") ||
               QString(file_list[i].c_str()).endsWith(".nii.gz"))
                return std::string("SRC and NIFTI files are not supported in autotrack pipeline. Please reconstruct data into FIB files");
            else
                return std::string("unsupported file format :") + file_list[i];
        }
        // fiber tracking on fib file
        std::shared_ptr<fib_data> handle(new fib_data);
        bool fib_loaded = false;
        progress prog2("tracking pathways");
        for(size_t j = 0;progress::at(j,track_id.size());++j)
        {
            std::string track_name = fib.tractography_name_list[track_id[j]];
            std::string output_path = dir + "/" + track_name;
            progress prog3(track_name.c_str());

            // create storing directory
            {
                QDir dir(output_path.c_str());
                if (!dir.exists() && !dir.mkpath("."))
                    show_progress() << std::string("cannot create directory:") + output_path << std::endl;
            }
            std::string fib_base = QFileInfo(fib_file_name.c_str()).baseName().toStdString();
            std::string no_result_file_name = output_path + "/" + fib_base+"."+track_name+".no_result.txt";
            std::string trk_file_name = output_path + "/" + fib_base+"."+track_name+".tt.gz";
            std::string template_trk_file_name = output_path + "/T_" + fib_base+"."+track_name+".tt.gz";
            std::string stat_file_name = output_path + "/" + fib_base+"."+track_name+".stat.txt";
            std::string report_file_name = dir+"/"+track_name+".report.txt";

            stat_files[j].push_back(stat_file_name);

            if(std::filesystem::exists(no_result_file_name) && !overwrite)
            {
                show_progress() << "skip " << track_name << " due to no result" << std::endl;
                continue;
            }

            bool has_stat_file = std::filesystem::exists(stat_file_name);
            bool has_trk_file = std::filesystem::exists(trk_file_name) &&
                    (!export_template_trk || std::filesystem::exists(template_trk_file_name));
            if(has_stat_file)
                show_progress() << "found stat file:" << stat_file_name << std::endl;
            if(has_trk_file)
                show_progress() << "found track file:" << trk_file_name << std::endl;

            if(!overwrite && (!export_stat || has_stat_file) && (!export_trk || has_trk_file))
            {
                show_progress() << "skip " << track_name << std::endl;
                continue;
            }

            {
                std::shared_ptr<file_holder> stat_file,trk_file;
                if(export_stat && !has_stat_file)
                    stat_file = std::make_shared<file_holder>(stat_file_name);
                if(export_trk && !has_trk_file)
                    trk_file = std::make_shared<file_holder>(trk_file_name);

                if (!fib_loaded)
                {
                    if(!handle->load_from_file(fib_file_name.c_str()))
                       return fib_file_name + ":" + handle->error_msg;
                    fib_loaded = true;
                }
                if(handle->template_id != 0)
                {
                    show_progress() << "Not adult human data. Enforce registration." << std::endl;
                    handle->set_template_id(0);
                }

                TractModel tract_model(handle);
                if(!overwrite && has_trk_file)
                    tract_model.load_from_file(trk_file_name.c_str());

                // each iteration increases tolerance
                for(size_t tracking_iteration = 0;tracking_iteration < tolerance.size() &&
                                                  !tract_model.get_visible_track_count();++tracking_iteration)
                {
                    ThreadData thread(handle);
                    {
                        if(!handle->load_track_atlas())
                            return handle->error_msg + " at " + fib_file_name;
                        thread.param.min_length = handle->vs[0]*std::max<float>(tolerance[tracking_iteration],
                                                                   handle->tract_atlas_min_length[track_id[j]]-2.0f*tolerance[tracking_iteration])/handle->tract_atlas_jacobian;
                        thread.param.max_length = handle->vs[0]*(handle->tract_atlas_max_length[track_id[j]]+2.0f*tolerance[tracking_iteration])/handle->tract_atlas_jacobian;
                        show_progress() << "min_length(mm): " << thread.param.min_length << std::endl;
                        show_progress() << "max_length(mm): " << thread.param.max_length << std::endl;
                        thread.param.tip_iteration = uint8_t(tip);
                        thread.param.stop_by_tract = 1;
                        thread.param.termination_count = 0;
                    }
                    {
                        thread.roi_mgr->use_auto_track = true;
                        thread.roi_mgr->track_voxel_ratio = track_voxel_ratio;
                        thread.roi_mgr->track_id = track_id[j];
                        thread.roi_mgr->tolerance_dis_in_icbm152_mm = tolerance[tracking_iteration];
                    }
                    progress prog_("tracking ",track_name.c_str(),true);
                    thread.run(thread_count,false);
                    std::string report = tract_model.report + thread.report.str();
                    report += " Shape analysis (Yeh, Neuroimage, 2020 Dec;223:117329) was conducted to derive shape metrics for tractography.";
                    if(reports[j].empty())
                        reports[j] = report;
                    auto_track_report = report;
                    bool no_result = false;
                    {
                        while(!thread.is_ended() && !progress::aborted())
                        {
                            std::this_thread::yield();
                            if(!thread.param.termination_count)
                            {
                                progress::at(0,1);
                                continue;
                            }
                            progress::at(thread.get_total_tract_count(),thread.param.termination_count);
                            // terminate if yield rate is very low, likely quality problem
                            if(thread.get_total_seed_count() > yield_check_count &&
                               thread.get_total_tract_count() < float(thread.get_total_seed_count())*yield_rate)
                            {
                                show_progress() << "low yield rate (" << thread.get_total_tract_count() << "/" <<
                                                    thread.get_total_seed_count() << "), terminating" << std::endl;
                                no_result = true;
                                thread.end_thread();
                                break;
                            }
                        }

                        float sec = float(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    thread.end_time-thread.begin_time).count())*0.001f;
                        if(thread.get_total_seed_count())
                        {
                            show_progress() << "yield rate (tract generated per seed): " <<
                                    float(thread.get_total_tract_count())/float(thread.get_total_seed_count()) << std::endl;
                            show_progress() << "tract yield rate (tracts per second): " <<
                                                       float(thread.get_total_tract_count())/sec << std::endl;
                            show_progress() << "seed yield rate (seeds per second): " <<
                                                       float(thread.get_total_seed_count())/sec << std::endl;
                        }

                    }
                    if(progress::aborted())
                        return std::string();
                    // fetch both front and back buffer
                    thread.fetchTracks(&tract_model);
                    thread.fetchTracks(&tract_model);
                    thread.apply_tip(&tract_model);

                    if(no_result || tract_model.get_visible_track_count() == 0)
                    {
                        tract_model.clear();
                        continue;
                    }

                    tract_model.delete_repeated(1.0f);

                    if(export_trk)
                    {
                        tract_model.report = report;
                        if(!tract_model.save_tracts_to_file(trk_file_name.c_str()))
                            return std::string("fail to save tractography file:")+trk_file_name;
                        if(export_template_trk &&
                           !tract_model.save_tracts_in_template_space(handle,template_trk_file_name.c_str()))
                                return std::string("fail to save template tractography file:")+trk_file_name;
                    }
                    break;
                }

                if(tract_model.get_visible_track_count() == 0)
                {
                    std::ofstream out(no_result_file_name.c_str());
                    continue;
                }

                if(export_stat &&
                   (overwrite || !std::filesystem::exists(stat_file_name) || !std::filesystem::file_size(stat_file_name)))
                {
                    progress prog("export tracts statistics");
                    show_progress() << "saving " << stat_file_name << std::endl;
                    std::ofstream out_stat(stat_file_name.c_str());
                    std::string result;
                    tract_model.get_quantitative_info(handle,result);
                    out_stat << result;
                }
            }
        }
    }
    if(progress::aborted())
        return std::string();
    {
        progress prog("check if there is any incomplete task");
        bool has_incomplete = false;
        for(size_t i = 0;i < stat_files.size();++i)
        {
            for(size_t j = 0;j < stat_files[i].size();++j)
            {
                show_progress() << "checking file:" << stat_files[i][j] << std::endl;
                if(std::filesystem::exists(stat_files[i][j]) &&
                   !std::filesystem::file_size(stat_files[i][j]))
                {
                    show_progress() << "remove empty file:" << stat_files[i][j] << std::endl;
                    std::filesystem::remove(stat_files[i][j]);
                    has_incomplete = true;
                }
            }
        }
        if(has_incomplete)
            return "Incomplete tasked found. Please rerun the analysis.";
    }

    if(file_list.size() != 1)
    {
        progress prog("aggregating results from multiple subjects");
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
                show_progress() << "reading " << stat_files[t][s] << std::endl;
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
                    error += std::filesystem::path(stat_files[t][s]).filename().string();
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
    progress_bar->setValue(prog);
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
        QMessageBox::information(this,"DSI Studio","Please assign FIB files");
        return;
    }
    ui->run->setEnabled(false);
    progress_bar->setValue(0);
    progress_bar->setVisible(true);
    progress_bar->setMaximum(file_list.size()-1);
    prog = 0;
    timer->start(5000);
    progress prog_("");

    program_option po;
    po["tolerance"] = ui->tolerance->text().toStdString();
    po["track_voxel_ratio"] = float(ui->track_voxel_ratio->value());
    po["tip"] = ui->pruning->value();
    po["export_stat"] = ui->export_stat->isChecked() ? 1 : 0;
    po["export_trk"] = ui->export_trk->isChecked()? 1 : 0;
    po["overwrite"] = ui->overwrite->isChecked()? 1 : 0;
    po["default_mask"] = ui->default_mask->isChecked()? 1 : 0;
    po["export_template_trk"] = ui->output_template_trk->isChecked()? 1 : 0;
    po["thread_count"] = ui->thread_count->value();

    std::string error = run_auto_track(po,file_list2,track_id,prog);
    timer->stop();
    ui->run->setEnabled(true);
    progress_bar->setVisible(false);

    if(!progress::aborted())
    {
        if(error.empty())
            QMessageBox::information(this,"AutoTrack","Completed");
        else
            QMessageBox::critical(this,"ERROR",error.c_str());
    }
    raise(); //  for mac
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
            select_list.push_back("Corticos");
            select_list.push_back("Thalamic");
            select_list.push_back("Optic");
            select_list.push_back("Fornix");
        }
        if(ui->cb_commissural->isChecked())
        {
            select_list.push_back("Corpus");
        }
        if(ui->cb_brainstem->isChecked())
        {
            select_list.push_back("Reticular");
            select_list.push_back("pontine");
            select_list.push_back("rubro");
            select_list.push_back("Cereb");
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
