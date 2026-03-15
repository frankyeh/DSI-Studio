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
    ui->thread_count->setMaximum(tipl::max_thread_count);
    ui->thread_count->setValue(tipl::max_thread_count);
    progress_bar = new QProgressBar(this);
    progress_bar->setVisible(false);
    ui->statusbar->addPermanentWidget(progress_bar);

    // populate tractography atlas list
    ui->template_list->clear();
    for(const auto& each : fa_template_list)
        ui->template_list->addItem(tipl::split(std::filesystem::path(each).filename().u8string(),'.').front().c_str());
    ui->template_list->setCurrentIndex(0);
    timer = std::make_shared<QTimer>(this);
    timer->setInterval(1000);
    connect(timer.get(),SIGNAL(timeout()),this,SLOT(check_status()));

}

auto_track::~auto_track()
{
    delete ui;
}
void auto_track::on_template_list_currentIndexChanged(int index)
{
    if(index < 0)
        return;
    fib_data fib;
    fib.set_template_id(index);
    ui->candidate_list_view->clear();
    for(auto each : fib.get_tractography_all_levels())
        ui->candidate_list_view->addItem(each.c_str());
}
void auto_track::on_open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open FIB files",
                                     "",
                                     "FIB files (*.fz *fib.gz);;All files (*)" );
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
    if(!ui->file_list_view->count() && !file_list.empty())
    {
        auto handle = std::make_shared<fib_data>();
        if(handle->load_from_file(file_list.front().toStdString()))
            ui->track_voxel_ratio->setValue(handle->default_track_voxel_ratio());
    }
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
    file_list << search_files(dir,"*.fz");
    update_list();
}
extern std::string auto_track_report;
std::string auto_track_report;

void set_template(std::shared_ptr<fib_data> handle,tipl::program_option<tipl::out>& po);
int trk_post(tipl::program_option<tipl::out>& po,
             std::shared_ptr<fib_data> handle,
             std::shared_ptr<TractModel> tract_model,
             std::string tract_file_name,bool output_track);
std::string run_auto_track(tipl::program_option<tipl::out>& po,const std::vector<std::string>& file_list,int& prog)
{
    std::string trk_format = po.get("trk_format","tt.gz");
    float yield_rate = po.get("yield_rate",0.00001f);
    size_t yield_check_count = 10.0f/yield_rate;
    bool overwrite = po.get("overwrite",0);
    uint32_t thread_count = tipl::max_thread_count;
    {
        if(!po.has("export"))
            po.set("export","stat");
    }

    std::vector<std::string> tract_name_list;
    {
        std::shared_ptr<fib_data> fib(new fib_data);
        set_template(fib,po);
        auto list = fib->get_tractography_all_levels();
        {
            std::string labels;
            for(auto each : list)
            {
                if(!labels.empty())
                    labels += ",";
                labels += each;
            }
            tipl::out() << "available track_ids in current template: " << labels;
        }
        std::vector<bool> selected(list.size());
        for(const auto& each : tipl::split(po.get("track_id","Association,Tract,Thalamic,Optic,Fornix,Medial,Corpus"),','))
        {
            for(size_t i = 0;i < list.size();++i)
            {
                if(selected[i])
                    continue;
                if(tipl::contains_case_insensitive(list[i],each))
                    selected[i] = true;
            }
        }

        if(std::all_of(selected.begin(), selected.end(), [](bool s){return !s; }))
            return "no bundle matches";

        std::string selected_list;
        for(size_t i = 0;i < list.size();++i)
        {
            if(selected[i])
            {
                if(!selected_list.empty())
                    selected_list += ",";
                selected_list += list[i];
                tract_name_list.push_back(list[i]);
            }
        }
        if(tract_name_list.empty())
            return "cannot find any tract matching --track_id";
        tipl::out() << "selected tracts: " << selected_list;
    }


    if(po.has("output") && !std::filesystem::is_directory(po.get("output")))
        return "expecting a directory at --output";

    std::string work_dir = po.get("output",QFileInfo(file_list.front().c_str()).absolutePath().toStdString());

    std::vector<std::vector<std::string> > stat_files(tract_name_list.size());
    std::vector<std::string> scan_names;
    tipl::progress prog0("automatic fiber tracking");
    for(size_t i = 0;prog0(i,file_list.size());++i)
    {
        prog = int(i);
        std::string fib_file_name = file_list[i];
        std::string cur_file_base_name = std::filesystem::path(fib_file_name).filename().u8string();
        scan_names.push_back(cur_file_base_name);
        tipl::out() << "processing " << cur_file_base_name;
        std::shared_ptr<fib_data> handle;

        tipl::progress prog1("tracking pathways");
        for(size_t j = 0;prog1(j,tract_name_list.size());++j)
        {
            std::string tract_name = tract_name_list[j];
            std::string output_path = po.has("output") ? work_dir : work_dir + "/" + tract_name;
            tipl::out() << "tracking " << tract_name;

            // create storing directory
            if (!QDir(output_path.c_str()).exists() && !QDir(output_path.c_str()).mkpath("."))
                tipl::warning() << std::string("cannot create directory: ") + output_path << std::endl;

            std::string fib_base = QFileInfo(fib_file_name.c_str()).baseName().toStdString();
            std::string trk_base = output_path + "/" + fib_base+"."+tract_name;
            std::string no_result_file_name = trk_base+".no_result.txt";
            std::string trk_file_name = trk_base + "." + trk_format;
            std::string stat_file_name = trk_file_name +".stat.txt";
            stat_files[j].push_back(stat_file_name);

            if(!overwrite)
            {
                if(std::filesystem::exists(no_result_file_name))
                {
                    tipl::out() << "exists " << no_result_file_name << ", skipping tracking and data export";
                    continue;
                }
                if(std::filesystem::exists(trk_file_name) && std::filesystem::exists(stat_file_name))
                {
                    tipl::out() << "exist " << trk_file_name << " and " << stat_file_name <<
                                   ", skipping tracking and data export";
                    continue;
                }
            }
            {
                if (!handle.get())
                {
                    handle = std::make_shared<fib_data>();
                    if(!handle->load_from_file(fib_file_name.c_str()))
                       return handle->error_msg;
                    set_template(handle,po);
                }
                std::shared_ptr<TractModel> tract_model(new TractModel(handle));
                if(!overwrite && std::filesystem::exists(trk_file_name))
                    tract_model->load_tracts_from_file(trk_file_name,handle.get());


                auto tolerance = po.get("tolerance",std::vector<float>({handle->default_tolerance(),
                                                                        handle->default_tolerance()+handle->vs[0],
                                                                        handle->default_tolerance()+handle->vs[0]*2.0f}));
                if(tolerance.empty())
                    return "invalid value in --tolerance";

                // each iteration increases tolerance
                for(size_t tracking_iteration = 0;tracking_iteration < tolerance.size() &&
                                                  !tract_model->get_visible_track_count();++tracking_iteration)
                {
                    ThreadData thread(handle);
                    {
                        if(!handle->load_track_atlas(true/*symmetric*/))
                            return handle->error_msg + " at " + fib_file_name;

                        if (po.has("threshold_index") && !handle->dir.set_tracking_index(po.get("threshold_index")))
                            return std::string("invalid threshold index");

                        thread.param.default_otsu = po.get("otsu_threshold",thread.param.default_otsu);
                        thread.param.threshold = po.get("fa_threshold",thread.param.threshold);
                        thread.param.cull_cos_angle = float(std::cos(po.get("turning_angle",0.0)*3.14159265358979323846/180.0));
                        thread.param.step_size = po.get("step_size",thread.param.step_size);
                        thread.param.smooth_fraction = po.get("smoothing",thread.param.smooth_fraction);
                        thread.param.tip_iteration = po.get("tip_iteration",16);
                        thread.param.check_ending = po.get("check_ending",1);
                        thread.param.track_voxel_ratio = po.get("track_voxel_ratio",handle->default_track_voxel_ratio());


                        thread.roi_mgr->set_auto_track(tract_name,tolerance[tracking_iteration],po.get("use_roi",1));
                        if(thread.roi_mgr->tolerance_dis_in_icbm152_mm != tolerance[tracking_iteration])
                            thread.roi_mgr->tolerance_dis_in_icbm152_mm += tolerance[tracking_iteration]-tolerance.front();
                        auto cur_tol = thread.roi_mgr->tolerance_dis_in_icbm152_mm;
                        auto minmax = handle->get_track_minmax_length(tract_name);
                        thread.param.min_length = handle->vs[0]*std::max<float>(cur_tol,minmax.first-2.0f*cur_tol)/handle->tract_atlas_jacobian;
                        thread.param.max_length = handle->vs[0]*(minmax.second+2.0f*cur_tol)/handle->tract_atlas_jacobian;
                        tipl::out() << "min_length(mm): " << thread.param.min_length << std::endl;
                        tipl::out() << "max_length(mm): " << thread.param.max_length << std::endl;
                    }
                    tipl::progress prog2("tracking ",tract_name.c_str(),true);
                    thread.run(thread_count,false);
                    tract_model->report = auto_track_report = handle->report + thread.report.str();
                    bool no_result = false;
                    {
                        // pre tracking stage
                        while(!thread.is_ended() && !thread.max_tract_count && prog2(0,1))
                            std::this_thread::yield();
                        // now start tracking
                        while(!thread.is_ended() &&
                              prog2(thread.get_total_tract_count(),thread.max_tract_count))
                        {
                            // terminate if yield rate is very low, likely quality problem
                            if(thread.get_total_seed_count() > yield_check_count &&
                               thread.get_total_tract_count() < float(thread.get_total_seed_count())*yield_rate)
                            {
                                tipl::out() << "low yield rate, adjusting tolerance and restart...";
                                no_result = true;
                                break;
                            }
                            std::this_thread::yield();
                        }
                        thread.end_thread();
                        if(prog2.aborted())
                            return std::string("aborted.");

                        float sec = float(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    thread.end_time-thread.begin_time).count())*0.001f;
                        tipl::out() << "total tract generated: " << thread.get_total_tract_count();
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
                    if(po.has("debug"))
                    {
                        auto save_region = [&](std::string name,auto& points)
                        {
                            if(points.empty())
                                return;
                            auto file_name = trk_base + "." + name + ".nii.gz";
                            ROIRegion region(thread.roi_mgr->handle);
                            region.add_points(std::move(points));
                            tipl::out() << "saving " << name << " region to " << file_name;
                            region.save_region_to_file(file_name);
                        };

                        save_region("seed",thread.roi_mgr->atlas_seed);
                        save_region("limiting",thread.roi_mgr->atlas_limiting);
                        save_region("not_end",thread.roi_mgr->atlas_not_end);
                        save_region("roi",thread.roi_mgr->atlas_roi);
                        save_region("roa",thread.roi_mgr->atlas_roa);
                    }
                    if(no_result)
                        continue;
                    // fetch both front and back buffer
                    thread.fetchTracks(tract_model.get());
                    thread.fetchTracks(tract_model.get());
                    if(thread.param.step_size != 0.0f)
                        tract_model->resample(1.0f);
                    tract_model->trim(thread.param.tip_iteration);
                    // if trim removes too many tract, undo to at least get the smallest possible bundle.
                    if(thread.param.tip_iteration && tract_model->get_visible_track_count() == 0)
                        tract_model->undo();
                    if(prog2.aborted())
                        return std::string("aborted.");
                }


                if(tract_model->get_visible_track_count() == 0)
                {
                    tipl::warning() << " no tracking result generated for " << tract_name;
                    std::ofstream out(no_result_file_name);
                }
                else
                {
                    if(trk_post(po,handle,tract_model,trk_file_name,true))
                        return std::string("terminated due to error");
                }
            }
        }
    }
    if(prog0.aborted())
        return std::string("aborted");
    if(tipl::contains(po.get("export"),"stat"))
    {
        tipl::out() << "check if there is any incomplete task";
        bool has_incomplete = false;
        for(size_t i = 0;i < stat_files.size();++i)
        {
            for(size_t j = 0;j < stat_files[i].size();++j)
            {
                tipl::out() << "checking " << stat_files[i][j] << std::endl;
                if(std::filesystem::exists(stat_files[i][j]) &&
                   !std::filesystem::file_size(stat_files[i][j]))
                {
                    tipl::out() << "removing empty file " << stat_files[i][j] << std::endl;
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
        std::vector<std::string> all_out2_text;
        for(size_t t = 0;t < tract_name_list.size();++t) // for each track
        {
            if(stat_files[t].empty())
                continue;
            // read metric names
            if(metrics_names.empty())
            {
                std::ifstream in(stat_files[t][0]);
                std::string line;
                for(size_t m = 0;std::getline(in,line);++m)
                    metrics_names.push_back(line.substr(0,line.find('\t')));

                all_out2_text.resize(scan_names.size()*metrics_names.size());
                for(size_t s = 0,index = 0;s < scan_names.size();++s)
                    for(size_t m = 0;m < metrics_names.size();++m,++index)
                        {
                            all_out2_text[index] = scan_names[s];
                            all_out2_text[index] += "\t";
                            all_out2_text[index] += metrics_names[m];
                        }

            }

            // parse metric values
            std::vector<std::vector<std::string> > output(scan_names.size());
            for(size_t s = 0;s < scan_names.size();++s) // for each scan
            {
                output[s].resize(metrics_names.size());
                tipl::out() << "reading " << stat_files[t][s] << std::endl;
                std::ifstream in(stat_files[t][s]);
                std::string line;
                for(size_t m = 0;std::getline(in,line);++m)
                {
                    auto pos = line.find('\t');
                    auto name = line.substr(0,pos);
                    auto value = line.substr(pos+1);
                    if(m < metrics_names.size() && name == metrics_names[m])
                        output[s][m] = value;
                    else
                    {
                        auto loc = std::find(metrics_names.begin(),metrics_names.end(),name);
                        if(loc != metrics_names.end())
                            output[s][loc-metrics_names.begin()] = value;
                    }
                }
            }
            std::string tract_name = tract_name_list[t];
            std::ofstream out(work_dir+"/"+tract_name+".stat.txt");

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
                    out << output[s][m];
                }
                out << std::endl;
            }

            // for all outs
            for(size_t s = 0,index = 0;s < scan_names.size();++s)
                for(size_t m = 0;m < output[s].size();++m,++index)
                {
                    all_out2_text[index] += "\t";
                    all_out2_text[index] += output[s][m];
                }
        }

        std::ofstream all_out2(work_dir+"/all_results_subject_wise.txt");
        all_out2 << "Subjects\tMetrics";
        for(size_t t = 0;t < tract_name_list.size();++t) // for each track
        {
            if(stat_files[t].empty())
                continue;
            all_out2 << "\t" << tract_name_list[t];
        }
        all_out2 << std::endl;
        for(size_t index = 0;index < all_out2_text.size();++index) // for each tract
            all_out2 << all_out2_text[index] << std::endl;
    }
    return std::string();
}
QString check_citation(QString str);
void auto_track::check_status()
{
    progress_bar->setValue(prog);
    ui->file_list_view->setCurrentRow(prog);
    if(!auto_track_report.empty())
        ui->report->setText(check_citation(auto_track_report.c_str()));
}
void auto_track::on_run_clicked()
{
    std::vector<std::string> file_list2;
    for(int i = 0;i < file_list.size();++i)
        file_list2.push_back(file_list[i].toStdString());

    std::string tract_names;
    QModelIndexList indexes = ui->candidate_list_view->selectionModel()->selectedRows();
    for(int i = 0;i < indexes.count();++i)
    {
        if(!tract_names.empty())
            tract_names += ",";
        tract_names += ui->candidate_list_view->item(indexes[i].row())->text().toStdString();
    }

    if(tract_names.empty())
    {
        QMessageBox::information(this,QApplication::applicationName(),"Please select target tracks");
        return;
    }
    if(file_list2.empty())
    {
        QMessageBox::information(this,QApplication::applicationName(),"Please assign FIB files");
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
    po["action"] = std::string("atk");
    po["tolerance"] = ui->tolerance->text().toStdString();
    po["track_voxel_ratio"] = float(ui->track_voxel_ratio->value());
    po["tip_iteration"] = ui->pruning->value();
    po["export_stat"] = ui->export_stat->isChecked() ? 1 : 0;
    po["export_trk"] = ui->export_trk->isChecked()? 1 : 0;
    po["overwrite"] = ui->overwrite->isChecked()? 1 : 0;
    po["export_template_trk"] = ui->output_template_trk->isChecked()? 1 : 0;
    po["thread_count"] = ui->thread_count->value();
    po["track_id"] = tract_names;
    po["template"] = ui->template_list->currentIndex();
    std::string error = run_auto_track(po,file_list2,prog);
    timer->stop();
    ui->run->setEnabled(true);
    progress_bar->setVisible(false);

    if(error.empty())
        QMessageBox::information(this,QApplication::applicationName(),"Completed");
    else
        QMessageBox::critical(this,"ERROR",error.c_str());
    raise(); //  for mac
}




