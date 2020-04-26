#include <QFileDialog>
#include <QStringListModel>
#include <QMessageBox>
#include "auto_track.h"
#include "ui_auto_track.h"
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"
#include "libs/tracking/tracking_thread.hpp"
extern size_t auto_track_pos[7];
extern unsigned char auto_track_rgb[6][3];
size_t auto_track_pos[7] = {0,20,42,45,52,66,80};
unsigned char auto_track_rgb[6][3]={{40,40,160},{40,160,40},{160,40,40},{20,20,80},{20,20,60},{20,80,20}};               // projection

auto_track::auto_track(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::auto_track)
{
    ui->setupUi(this);
    progress = new QProgressBar(this);
    progress->setVisible(false);
    ui->statusbar->addPermanentWidget(progress);

    fib.set_template_id(0);

    const int atlas_range = 4;
    QStringList tract_names;
    for(size_t index = 0;index < auto_track_pos[atlas_range];++index)
        tract_names << fib.tractography_name_list[index].c_str();

    ui->candidate_list_view->addItems(tract_names);
    for(int i = 0;i < atlas_range;++i)
    for(size_t j = auto_track_pos[i];j < auto_track_pos[i+1];++j)
        ui->candidate_list_view->item(int(j))->setData(Qt::ForegroundRole,
            QBrush(QColor(auto_track_rgb[i][0],auto_track_rgb[i][1],auto_track_rgb[i][2])));

    timer = std::make_shared<QTimer>(this);
    timer->stop();
    timer->setInterval(1000);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(check_status()));
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
                                     "SRC files (*src.gz);;All files (*)" );
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
extern std::string auto_track_report;
std::string auto_track_report;
std::string run_auto_track(
                    fib_data& fib,
                    const std::vector<std::string>& file_list,
                    const std::vector<unsigned int>& track_id,
                    float length_ratio,
                    float tolerance,
                    unsigned int track_count,
                    int interpolation,int tip,
                    bool export_stat,
                    bool export_trk,
                    int& progress)
{
    std::vector<std::string> reports(track_id.size());
    std::vector<std::vector<std::string> > stat_files(track_id.size());
    std::string dir = QFileInfo(file_list.front().c_str()).absolutePath().toStdString();

    for(size_t i = 0;i < file_list.size() && !prog_aborted();++i)
    {
        std::string cur_file_base_name = QFileInfo(file_list[i].c_str()).baseName().toStdString();
        progress = int(i);
        // recon
        {
            std::shared_ptr<ImageModel> handle(std::make_shared<ImageModel>());
            handle->voxel.method_id = 4; // GQI
            handle->voxel.param[0] = length_ratio;
            handle->voxel.ti.init(8); // odf order of 8
            handle->voxel.odf_xyz[0] = 0;
            handle->voxel.odf_xyz[1] = 0;
            handle->voxel.odf_xyz[2] = 0;
            handle->voxel.thread_count = std::thread::hardware_concurrency();
            // has fib file?
            std::string fib_file_name = file_list[i]+handle->get_file_ext();
            if(!QFileInfo(fib_file_name.c_str()).exists())
            {
                if (!handle->load_from_file(file_list[i].c_str()))
                    return std::string("ERROR at ") + cur_file_base_name + ":" + handle->error_msg;
                if(!handle->is_human_data())
                    return std::string("ERROR at ") + cur_file_base_name + ":" + handle->error_msg;
                handle->voxel.half_sphere = handle->is_dsi_half_sphere();
                handle->voxel.scheme_balance = handle->need_scheme_balance();
                if(interpolation)
                    handle->rotate_to_mni(float(interpolation));
                begin_prog("Reconstruct DWI");
                if (!handle->reconstruction())
                    return std::string("ERROR at ") + cur_file_base_name + ":" + handle->error_msg;
            }
            for(size_t j = 0;j < track_id.size() && !prog_aborted();++j)
            {
                std::string track_name = fib.tractography_name_list[track_id[j]];
                std::replace(track_name.begin(),track_name.end(),' ','_');
                std::string trk_file_name = fib_file_name+"."+track_name+".trk.gz";
                std::string stat_file_name = fib_file_name+"."+track_name+".stat.txt";
                std::string report_file_name = dir+"/"+track_name+".report.txt";
                std::string no_result_file_name = fib_file_name+"."+track_name+".no_result.txt";
                stat_files[j].push_back(stat_file_name);

                if((export_stat && !QFileInfo(stat_file_name.c_str()).exists()) ||
                   (export_trk && !QFileInfo(trk_file_name.c_str()).exists()))
                {
                    std::shared_ptr<fib_data> handle(new fib_data);
                    if (!handle->load_from_file(fib_file_name.c_str()))
                        return std::string("ERROR at ") + fib_file_name + ":" +handle->error_msg;

                    TractModel tract_model(handle);
                    {
                        ThreadData thread(handle);
                        thread.param.check_ending = 1;
                        thread.param.tip_iteration = uint8_t(tip);
                        thread.param.termination_count = track_count;
                        thread.param.max_seed_count = 10000000;
                        thread.param.stop_by_tract = 1;
                        if(!thread.roi_mgr->setAtlas(track_id[j],tolerance/handle->vs[0]))
                            return std::string("ERROR at ") + fib_file_name + ":" +handle->error_msg;
                        thread.run(tract_model.get_fib(),std::thread::hardware_concurrency(),false);
                        auto_track_report = handle->report + thread.report.str();
                        if(reports[j].empty())
                            reports[j] = auto_track_report;
                        begin_prog("fiber tracking");
                        while(!thread.is_ended())
                        {
                            size_t total_track = thread.get_total_tract_count();
                            if(total_track)
                                check_prog(total_track,track_count);
                            else
                                check_prog(thread.get_total_seed_count(),thread.param.max_seed_count);
                            thread.fetchTracks(&tract_model);
                            if(prog_aborted())
                                return std::string();
                            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                        }
                        thread.fetchTracks(&tract_model);
                        if(prog_aborted())
                            return std::string();
                    }
                    for(int i = 0;i < tip;++i)
                        tract_model.trim();
                    tract_model.delete_repeated(1.0f);

                    if(prog_aborted())
                        return std::string();

                    if(tract_model.get_visible_track_count() == 0)
                    {
                        std::ofstream out(no_result_file_name.c_str());
                        continue;
                    }

                    if(export_stat)
                    {
                        std::ofstream out_stat(stat_file_name.c_str());
                        std::string result;
                        tract_model.get_quantitative_info(result);
                        out_stat << result;
                    }
                    if(export_trk)
                    {
                        if(!tract_model.save_tracts_to_file(trk_file_name.c_str()))
                            return std::string("fail to save trk file:")+trk_file_name;
                    }
                }
            }
        }
    }

    // aggregating

    for(size_t t = 0;t < stat_files.size();++t)
    {
        std::vector<std::string> output;
        for(size_t j = 0;j < stat_files[t].size();++j)
        {
            std::ifstream in(stat_files[t][j].c_str());
            if(!in)
                continue;
            std::string base_name = QFileInfo(stat_files[t][j].c_str()).baseName().toStdString();
            std::string line;
            if(output.empty())
            {
                output.push_back(std::string(std::string("Files\t")+base_name));
                while(std::getline(in,line))
                    output.push_back(line);
            }
            else
            {
                output[0] += "\t";
                output[0] += base_name;
                for(size_t i = 1;i < output.size();++i)
                {
                    if(std::getline(in,line))
                        output[i] += line.substr(line.find('\t'));
                    else
                        output[i] += "\t";
                }
            }
        }

        std::string track_name = fib.tractography_name_list[track_id[t]];
        std::ofstream out((dir+"/"+track_name+".stat.txt").c_str());
        for(size_t i = 0;i < output.size();++i)
            out << output[i] << std::endl;
        out << reports[t] << std::endl;
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
    run_auto_track(fib,
                   file_list2,track_id,
                   ui->gqi_l->value(),
                   ui->tolerance->value(),
                   uint32_t(ui->track_count->value()*1000.0),
                   ui->interpolation->currentIndex(),ui->pruning->value(),
                   ui->export_stat->isChecked(),ui->export_trk->isChecked(),prog);
    timer->stop();
    ui->run->setEnabled(true);
    progress->setVisible(false);
    if(!prog_aborted())
        QMessageBox::information(this,"DSI Studio","Completed");
    close_prog();
    raise(); //  for mac
}


void auto_track::on_interpolation_currentIndexChanged(int)
{
    QMessageBox::information(this,"DSI Studio","You may need to remove existing *.fib.gz and *.mapping.gz files to take effect");

}
