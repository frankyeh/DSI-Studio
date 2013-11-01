#ifndef TRACTTABLEWIDGET_H
#define TRACTTABLEWIDGET_H
#include <vector>
#include <QItemDelegate>
#include <QTableWidget>
#include <QTimer>
#include "libs/tracking/tracking_thread.hpp"
#include "libs/tracking/tract_model.hpp"

class tracking_window;
class TractModel;
class ThreadData;

class TractTableWidget : public QTableWidget
{
    Q_OBJECT
protected:
    void contextMenuEvent ( QContextMenuEvent * event );
public:
    explicit TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent = 0);
    ~TractTableWidget(void);

private:
    tracking_window& cur_tracking_window;
    QTimer *timer;
public:
    unsigned int tract_serial;
    std::vector<ThreadData*> thread_data;
    std::vector<TractModel*> tract_models;
    unsigned char edit_option;
    void addNewTracts(QString tract_name);
    void export_tract_density(image::geometry<3>& dim,
                              image::vector<3,float> vs,
                              std::vector<float>& transformation,bool color,bool endpoint);
    void saveTransformedTracts(const float* transform);
    void saveTransformedEndpoints(const float* transform);
signals:
    void need_update(void);
private:
    void delete_row(int row);
    void clustering(int method_id);
    void load_cluster_label(const std::vector<unsigned int>& labels,QString Name);
public slots:
    void clustering_EM(void){clustering(2);}
    void clustering_kmeans(void){clustering(1);}
    void clustering_hie(void){clustering(0);}
    void open_cluster_label(void);
    void set_color(void);
    void check_check_status(int,int);
    void start_tracking(void);
    void fetch_tracts(void);
    void load_tracts(void);
    void save_tracts_as(void);
    void load_tracts_color(void);
    void save_tracts_color_as(void);
    void save_fa_as(void);
    void save_tracts_data_as(void);
    void save_all_tracts_as(void);
    void save_end_point_as(void);
    void merge_all(void);
    void copy_track(void);
    void delete_tract(void);
    void delete_all_tract(void);
    void edit_tracts(void);
    void undo_tracts(void);
    void redo_tracts(void);
    void trim_tracts(void);
    void stop_tracking(void);
    void assign_colors(void);
    void move_up(void);
    void move_down(void);
    void show_tracts_statistics(void);

};

#endif // TRACTTABLEWIDGET_H
