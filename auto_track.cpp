#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include "auto_track.h"
#include "ui_auto_track.h"
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
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

    // populate tractography atlas list
    fib_data fib;
    fib.set_template_id(0);
    for(const auto& each : fib.tractography_atlas_list)
        ui->tractography_atlas->addItem(QFileInfo(each.c_str()).baseName());
    ui->tractography_atlas->setCurrentIndex(0);

    timer = std::make_shared<QTimer>(this);
    timer->stop();
    timer->setInterval(1000);
    connect(timer.get(),SIGNAL(timeout()),this,SLOT(check_status()));

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
bool find_string_case_insensitive(const std::string & str1, const std::string & str2)
{
  auto it = std::search(
    str1.begin(), str1.end(),
    str2.begin(),   str2.end(),
    [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
  );
  return (it != str1.end() );
}
bool is_selected(std::vector<std::string>& selected_tracts,const std::string& tract_name)
{
    for(const auto& each: selected_tracts)
        if(find_string_case_insensitive(tract_name,each))
            return true;
    return false;
}
void set_template(std::shared_ptr<fib_data> handle,tipl::program_option<tipl::out>& po);
std::string run_auto_track(tipl::program_option<tipl::out>& po,const std::vector<std::string>& file_list,int& prog)
{
    std::string tolerance_string = po.get("tolerance","22,26,30");
    float track_voxel_ratio = po.get("track_voxel_ratio",2.0f);
    float yield_rate = po.get("yield_rate",0.00001f);
    size_t yield_check_count = 10.0f/yield_rate;
    bool export_stat = po.get("export_stat",1);
    bool export_trk = po.get("export_trk",1);
    bool overwrite = po.get("overwrite",0);
    bool export_template_trk = po.get("export_template_trk",0);
    bool check_ending = po.get("check_ending",1);
    uint32_t thread_count = uint32_t(po.get("thread_count",std::thread::hardware_concurrency()));
    std::string trk_format = po.get("trk_format","tt.gz");
    std::string stat_format = po.get("stat_format","stat.txt");
    auto track_name_list = tipl::split(po.get("track_id"),',');
    std::vector<float> tolerance;
    {
        std::istringstream in(tolerance_string);
        std::string num;
        while(std::getline(in,num,','))
        {
            std::istringstream in2(num);
            float t;
            if(!(in2 >> t))
                return std::string("Cannot parse tolerance number: ")+num;
            tolerance.push_back(t);
        }
        if(tolerance.empty())
            return "Please assign tolerance distance";
    }

    std::vector<std::vector<std::string> > stat_files;
    std::vector<std::string> tractography_name_list;
    std::string dir = po.get("output",QFileInfo(file_list.front().c_str()).absolutePath().toStdString());

    std::vector<std::string> scan_names;
    tipl::progress prog0("automatic fiber tracking");
    for(size_t i = 0;prog0(i,file_list.size());++i)
    {
        std::string cur_file_base_name = QFileInfo(file_list[i].c_str()).baseName().toStdString();
        scan_names.push_back(cur_file_base_name);
        prog = int(i);
        tipl::out() << "processing " << cur_file_base_name << std::endl;
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
        handle->set_template_id(0);
        set_template(handle,po);
        if(tractography_name_list.empty())
        {
            tractography_name_list = handle->tractography_name_list;
            stat_files.resize(tractography_name_list.size());
        }

        bool fib_loaded = false;
        tipl::progress prog1("tracking pathways");
        for(size_t track_id = 0;prog1(track_id,tractography_name_list.size());++track_id)
        {
            std::string track_name = tractography_name_list[track_id];
            if(!track_name_list.empty() && !is_selected(track_name_list,track_name))
                continue;

            std::string output_path = dir + "/" + track_name;
            tipl::out() << track_name;

            // create storing directory
            {
                QDir dir(output_path.c_str());
                if (!dir.exists() && !dir.mkpath("."))
                    tipl::out() << std::string("cannot create directory:") + output_path << std::endl;
            }
            std::string fib_base = QFileInfo(fib_file_name.c_str()).baseName().toStdString();
            std::string no_result_file_name = output_path + "/" + fib_base+"."+track_name+".no_result.txt";
            std::string trk_file_name = output_path + "/" + fib_base+"."+track_name+ "." + trk_format;
            std::string template_trk_file_name = output_path + "/T_" + fib_base+"."+track_name + "." + trk_format;
            std::string stat_file_name = output_path + "/" + fib_base+"."+track_name+"." + stat_format;
            std::string report_file_name = dir+"/"+track_name+".report.txt";                
            stat_files[track_id].push_back(stat_file_name);
            if(std::filesystem::exists(no_result_file_name) && !overwrite)
            {
                tipl::out() << "skip " << track_name << " due to no result" << std::endl;
                continue;
            }

            bool has_stat_file = std::filesystem::exists(stat_file_name);
            bool has_trk_file = std::filesystem::exists(trk_file_name) &&
                    (!export_template_trk || std::filesystem::exists(template_trk_file_name));
            if(has_stat_file)
                tipl::out() << "found stat file:" << stat_file_name << std::endl;
            if(has_trk_file)
                tipl::out() << "found track file:" << trk_file_name << std::endl;

            if(!overwrite && (!export_stat || has_stat_file) && (!export_trk || has_trk_file))
            {
                tipl::out() << "skip " << track_name << std::endl;
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
                TractModel tract_model(handle);
                if(!overwrite && has_trk_file)
                    tract_model.load_tracts_from_file(trk_file_name.c_str(),handle.get());

                // each iteration increases tolerance
                for(size_t tracking_iteration = 0;tracking_iteration < tolerance.size() &&
                                                  !tract_model.get_visible_track_count();++tracking_iteration)
                {
                    ThreadData thread(handle);
                    {
                        if(!handle->load_track_atlas())
                            return handle->error_msg + " at " + fib_file_name;

                        thread.param.default_otsu = po.get("otsu_threshold",thread.param.default_otsu);
                        thread.param.threshold = po.get("fa_threshold",thread.param.threshold);
                        thread.param.cull_cos_angle = float(std::cos(po.get("turning_angle",0.0)*3.14159265358979323846/180.0));
                        thread.param.step_size = po.get("step_size",thread.param.step_size);
                        thread.param.smooth_fraction = po.get("smoothing",thread.param.smooth_fraction);

                        thread.param.min_length = handle->vs[0]*std::max<float>(tolerance[tracking_iteration],
                                                                   handle->tract_atlas_min_length[track_id]-2.0f*tolerance[tracking_iteration])/handle->tract_atlas_jacobian;
                        thread.param.max_length = handle->vs[0]*(handle->tract_atlas_max_length[track_id]+2.0f*tolerance[tracking_iteration])/handle->tract_atlas_jacobian;
                        tipl::out() << "min_length(mm): " << thread.param.min_length << std::endl;
                        tipl::out() << "max_length(mm): " << thread.param.max_length << std::endl;
                        thread.param.tip_iteration = po.get("tip_iteration",4);
                        thread.param.check_ending = check_ending && !QString(track_name.c_str()).contains("Cingulum");
                        thread.param.stop_by_tract = 1;
                        thread.param.termination_count = 0;
                    }
                    {
                        thread.roi_mgr->use_auto_track = true;
                        thread.roi_mgr->track_voxel_ratio = track_voxel_ratio;
                        thread.roi_mgr->track_id = track_id;
                        thread.roi_mgr->tolerance_dis_in_icbm152_mm = tolerance[tracking_iteration];
                    }
                    tipl::progress prog2("tracking ",track_name.c_str(),true);
                    thread.run(thread_count,false);
                    std::string report = handle->report;
                    report += thread.report.str();
                    auto_track_report = report;
                    bool no_result = false;
                    {
                        while(!thread.is_ended() && !prog2.aborted())
                        {
                            std::this_thread::yield();
                            if(!thread.param.termination_count)
                            {
                                prog2(0,1);
                                continue;
                            }
                            prog2(thread.get_total_tract_count(),thread.param.termination_count);
                            // terminate if yield rate is very low, likely quality problem
                            if(thread.get_total_seed_count() > yield_check_count &&
                               thread.get_total_tract_count() < float(thread.get_total_seed_count())*yield_rate)
                            {
                                tipl::out() << "low yield rate (" << thread.get_total_tract_count() << "/" <<
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
                            tipl::out() << "yield rate (tract generated per seed): " <<
                                    float(thread.get_total_tract_count())/float(thread.get_total_seed_count()) << std::endl;
                            tipl::out() << "tract yield rate (tracts per second): " <<
                                                       float(thread.get_total_tract_count())/sec << std::endl;
                            tipl::out() << "seed yield rate (seeds per second): " <<
                                                       float(thread.get_total_seed_count())/sec << std::endl;
                        }

                    }
                    if(prog2.aborted())
                        return std::string("aborted.");
                    // fetch both front and back buffer
                    thread.fetchTracks(&tract_model);
                    thread.fetchTracks(&tract_model);

                    thread.apply_tip(&tract_model);

                    if(no_result || tract_model.get_visible_track_count() == 0)
                    {
                        tract_model.clear();
                        continue;
                    }

                    tract_model.resample(1.0f);
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
                    tipl::progress prog2("export tracts statistics");
                    tipl::out() << "saving " << stat_file_name << std::endl;
                    std::ofstream out_stat(stat_file_name.c_str());
                    std::string result;
                    tract_model.get_quantitative_info(handle,result);
                    out_stat << result;
                }
            }
        }
    }
    if(prog0.aborted())
        return std::string("aborted");
    {
        tipl::out() << "check if there is any incomplete task";
        bool has_incomplete = false;
        for(size_t i = 0;i < stat_files.size();++i)
        {
            for(size_t j = 0;j < stat_files[i].size();++j)
            {
                tipl::out() << "checking file:" << stat_files[i][j] << std::endl;
                if(std::filesystem::exists(stat_files[i][j]) &&
                   !std::filesystem::file_size(stat_files[i][j]))
                {
                    tipl::out() << "remove empty file:" << stat_files[i][j] << std::endl;
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
        tipl::out() << "aggregating results from multiple subjects";
        std::vector<std::string> metrics_names; // row titles are metrics

        for(size_t t = 0;t < tractography_name_list.size();++t) // for each track
        {
            if(stat_files[t].empty())
                continue;
            // read metric names
            if(metrics_names.empty())
            {
                std::ifstream in(stat_files[t][0].c_str());
                std::string line;
                for(size_t m = 0;std::getline(in,line);++m)
                    metrics_names.push_back(line.substr(0,line.find('\t')));
            }

            // parse metric values
            std::vector<std::vector<std::string> > output(scan_names.size());
            for(size_t s = 0;s < output.size();++s) // for each scan
            {
                tipl::out() << "reading " << stat_files[t][s] << std::endl;
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
            std::string track_name = tractography_name_list[t];
            std::ofstream out((dir+"/"+track_name+".stat.txt").c_str());

            // output first row: the name of each scan
            for(const auto& each: scan_names)
                out << "\t" << each;
            out << std::endl;

            // output each metric at each row
            for(size_t m = 0;m < metrics_names.size();++m)
            {
                out << metrics_names[m];
                for(size_t s = 0;s < output.size();++s)
                {
                    out << "\t";
                    if(m < output[s].size())
                        out << output[s][m];
                }
                out << std::endl;
            }
        }
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
    for(int i = 0;i < file_list.size();++i)
        file_list2.push_back(file_list[i].toStdString());

    std::string track_id;
    QModelIndexList indexes = ui->candidate_list_view->selectionModel()->selectedRows();
    for(int i = 0;i < indexes.count();++i)
    {
        if(!track_id.empty())
            track_id += ",";
        track_id += ui->candidate_list_view->item(indexes[i].row())->text().toStdString();
    }

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
    tipl::progress prog_("");

    tipl::program_option<tipl::out> po;
    po["tolerance"] = ui->tolerance->text().toStdString();
    po["track_voxel_ratio"] = float(ui->track_voxel_ratio->value());
    po["tip_iteration"] = ui->pruning->value();
    po["export_stat"] = ui->export_stat->isChecked() ? 1 : 0;
    po["export_trk"] = ui->export_trk->isChecked()? 1 : 0;
    po["overwrite"] = ui->overwrite->isChecked()? 1 : 0;
    po["export_template_trk"] = ui->output_template_trk->isChecked()? 1 : 0;
    po["thread_count"] = ui->thread_count->value();
    po["track_id"] = track_id;
    po["tractography_atlas"] = ui->tractography_atlas->currentIndex();
    std::string error = run_auto_track(po,file_list2,prog);
    timer->stop();
    ui->run->setEnabled(true);
    progress_bar->setVisible(false);

    if(error.empty())
        QMessageBox::information(this,"AutoTrack","Completed");
    else
        QMessageBox::critical(this,"ERROR",error.c_str());
    raise(); //  for mac
}

void auto_track::on_tractography_atlas_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    fib_data fib;
    fib.set_template_id(0);
    fib.set_tractography_atlas_id(index);

    ui->candidate_list_view->clear();
    for(size_t index = 0;index < fib.tractography_name_list.size();++index)
    {
        ui->candidate_list_view->addItem(fib.tractography_name_list[index].c_str());
        ui->candidate_list_view->item(index)->setSelected(true);
    }

}

