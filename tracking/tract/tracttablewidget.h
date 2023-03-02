#ifndef TRACTTABLEWIDGET_H
#define TRACTTABLEWIDGET_H
#include <vector>
#include <QApplication>
#include <QItemDelegate>
#include <QTableWidget>
#include <QTimer>
#include "TIPL/tipl.hpp"
#include "tract_model.hpp"
#include "opengl/tract_render.hpp"
class tracking_window;
struct ThreadData;
class TractTableWidget : public QTableWidget
{
    Q_OBJECT
protected:
    void contextMenuEvent ( QContextMenuEvent * event );
public:
    explicit TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent = nullptr);
    ~TractTableWidget(void);

private:
    tracking_window& cur_tracking_window;
    QTimer *timer,*timer_update;
private:
    int color_gen = 10;
public:
    std::vector<std::shared_ptr<ThreadData> > thread_data;
    std::vector<std::shared_ptr<TractModel> > tract_models;
    std::vector<std::shared_ptr<TractRender> > tract_rendering;
public:
    std::vector<std::shared_ptr<TractModel> > get_checked_tracks(void);
    std::vector<std::shared_ptr<TractRender> > get_checked_tracks_rendering(void);
    std::vector<std::shared_ptr<TractRender::end_reading> > start_reading_checked_tracks(void);
    std::vector<std::shared_ptr<TractRender::end_writing> > start_writing_checked_tracks(void);
    std::vector<std::string> get_checked_tracks_name(void) const;
    enum {none = 0,select = 1,del = 2,cut = 3,paint = 4,move = 5}edit_option;
    void addNewTracts(QString tract_name,bool checked = true);
    void addConnectometryResults(std::vector<std::vector<std::vector<float> > >& greater,
                                 std::vector<std::vector<std::vector<float> > >& lesser);
    void export_tract_density(tipl::shape<3> dim,
                              tipl::vector<3,float> vs,
                              tipl::matrix<4,4> transformation,bool color,bool endpoint);
    void load_tracts(QStringList filenames);
    void cut_by_slice(unsigned char dim,bool greater);
    void draw_tracts(unsigned char dim,int pos,
                     QImage& scaledimage,float display_ratio);
    QString output_format(void);
    template<typename fun_type>
    void for_each_bundle(const char* prog_name,fun_type&& fun,bool silence = false)
    {
        std::vector<unsigned int> checked_index;
        std::vector<bool> changed;
        for(unsigned int index = 0;index < tract_models.size();++index)
            if(item(int(index),0)->checkState() == Qt::Checked)
            {
                checked_index.push_back(index);
                changed.push_back(false);
            }

        {
            progress prog(prog_name,!silence);
            size_t p = 0;
            tipl::par_for(checked_index.size(),[&](unsigned int i)
            {
                progress::at(p++,checked_index.size());
                if(prog.aborted())
                    return;
                auto lock = tract_rendering[checked_index[i]]->start_writing();
                if(fun(checked_index[i]))
                {
                    changed[i] = true;
                    tract_rendering[checked_index[i]]->need_update = true;
                }
            });
            if(prog.aborted())
                return;
        }
        progress prog(prog_name,!silence);
        for(unsigned int i = 0;prog.at(i,checked_index.size());++i)
            if(changed[i])
            {
                QApplication::processEvents();
                item(int(checked_index[i]),1)->setText(QString::number(tract_models[checked_index[i]]->get_visible_track_count()));
                item(int(checked_index[i]),2)->setText(QString::number(tract_models[checked_index[i]]->get_deleted_track_count()));
                tract_rendering[checked_index[i]]->need_update = true;
                emit show_tracts();
                QApplication::processEvents();
            }
    }
public:
    void render_tracts(GLWidget* glwidget);
    std::string error_msg;
    bool command(QString cmd,QString param = "",QString param2 = "");

signals:
    void show_tracts(void);
private:
    void delete_row(int row);
    void clustering(int method_id);
    void load_cluster_label(const std::vector<unsigned int>& labels,QStringList Names = QStringList());
    void load_tract_label(QString FileName);
public slots:
    void clustering_EM(void){clustering(2);}
    void clustering_kmeans(void){clustering(1);}
    void clustering_hie(void){clustering(0);}
    void auto_recognition(void);
    void recognize_rename(void);
    void open_cluster_label(void);
    void open_cluster_color(void);
    void save_cluster_color(void);
    void set_color(void);
    void check_check_status(int,int);
    void check_all(void);
    void uncheck_all(void);
    void start_tracking(void);
    void filter_by_roi(void);
    void fetch_tracts(void);
    void show_tracking_progress(void);

    void load_tracts(void);
    void load_tract_label(void);
    void load_tracts_color(void);
    void load_tracts_value(void);

    void save_tracts_as(void);
    void save_vrml_as(void);
    void save_tracts_color_as(void);
    void save_tracts_data_as(void);
    void save_all_tracts_as(void);
    void save_all_tracts_to_dir(void);
    void save_all_tracts_end_point_as(void);

    void save_tracts_in_native(void);
    void save_tracts_in_template(void);
    void save_tracts_in_mni(void);
    void save_end_point_as(void);
    void save_end_point_in_mni(void);
    void save_transformed_tracts(void);
    void save_transformed_endpoints(void);


    void merge_all(void);
    void copy_track(void);
    void separate_deleted_track(void);
    void reconnect_track(void);
    void sort_track_by_name(void);
    void merge_track_by_name(void);
    void delete_tract(void);
    void delete_all_tract(void);
    void delete_repeated(void);
    void delete_by_length(void);
    void resample_step_size(void);
    void edit_tracts(void);
    void delete_branches(void)      {for_each_bundle("delete branches", [&](unsigned int index){return tract_models[index]->delete_branch();});}
    void undo_tracts(void)          {for_each_bundle("undo tracts",     [&](unsigned int index){return tract_models[index]->undo();},true);}
    void redo_tracts(void)          {for_each_bundle("redo tracts",     [&](unsigned int index){return tract_models[index]->redo();},true);}
    void trim_tracts(void)          {for_each_bundle("trim tracts",     [&](unsigned int index){return tract_models[index]->trim();},true);}
    void assign_colors(void);
    void stop_tracking(void);
    void move_up(void);
    void move_down(void);
    void show_tracts_statistics(void);
    void recog_tracks(void);
    void show_report(void);

    void need_update_all(void);

};

#endif // TRACTTABLEWIDGET_H
