#ifndef TRACTTABLEWIDGET_H
#define TRACTTABLEWIDGET_H
#include <vector>
#include <QItemDelegate>
#include <QTableWidget>
#include <QTimer>
#include <tipl/tipl.hpp>
#include "tract_model.hpp"
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
    QTimer *timer;
private:
    int color_gen = 10;
public:
    std::vector<std::shared_ptr<ThreadData> > thread_data;
    std::vector<std::shared_ptr<TractModel> > tract_models;
public:
    std::vector<std::shared_ptr<TractModel> > get_checked_tracks(void);
    std::vector<std::string> get_checked_tracks_name(void) const;
    enum {none = 0,select = 1,del = 2,cut = 3,paint = 4,move = 5}edit_option;
    void addNewTracts(QString tract_name,bool checked = true);
    void addConnectometryResults(std::vector<std::vector<std::vector<float> > >& greater,
                                 std::vector<std::vector<std::vector<float> > >& lesser);
    void export_tract_density(tipl::shape<3>& dim,
                              tipl::vector<3,float> vs,
                              tipl::matrix<4,4> transformation,bool color,bool endpoint);
    void load_tracts(QStringList filenames);
    void cut_by_slice(unsigned char dim,bool greater);
    void draw_tracts(unsigned char dim,int pos,
                     QImage& scaledimage,float display_ratio,unsigned int max_count);

    QString output_format(void);
    bool command(QString cmd,QString param = "",QString param2 = "");
    template<typename fun_type>
    void for_each_track(fun_type&& fun)
    {
        for (int i = 0;i < rowCount();++i)
        {
            if(item(i,0)->checkState() != Qt::Checked)
                continue;
            auto active_tract_model = tract_models[size_t(i)];
            if (active_tract_model->get_visible_track_count() == 0)
                continue;
            auto tracks_count = active_tract_model->get_visible_track_count();
            for (unsigned int data_index = 0; data_index < tracks_count; ++data_index)
                fun(active_tract_model,data_index);
        }
    }
signals:
    void need_update(void);
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
    void set_color(void);
    void check_check_status(int,int);
    void check_all(void);
    void uncheck_all(void);
    void start_tracking(void);
    void filter_by_roi(void);
    void fetch_tracts(void);

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


    void deep_learning_train(void);
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
    void delete_branches(void);
    void resample_step_size(void);
    void edit_tracts(void);
    void undo_tracts(void);
    void redo_tracts(void);
    void trim_tracts(void);
    void assign_colors(void);
    void stop_tracking(void);
    void move_up(void);
    void move_down(void);
    void show_tracts_statistics(void);
    void recog_tracks(void);
    void show_report(void);


};

#endif // TRACTTABLEWIDGET_H
