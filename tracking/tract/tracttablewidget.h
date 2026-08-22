#ifndef TRACTTABLEWIDGET_H
#define TRACTTABLEWIDGET_H
#include <algorithm>
#include <string>
#include <vector>
#include <QApplication>
#include <QItemDelegate>
#include <QTableWidget>
#include <QTimer>
#include "tract_model.hpp"
#include "opengl/tract_render.hpp"
class tracking_window;
struct ThreadData;
class TractTableWidget : public QTableWidget
{
    Q_OBJECT
protected:
    void contextMenuEvent(QContextMenuEvent * event ) override;
public:
    explicit TractTableWidget(tracking_window& cur_tracking_window_,QWidget *parent = nullptr);
    ~TractTableWidget(void);

private:
    tracking_window& cur_tracking_window;
    QTimer *timer,*timer_update;
public:
    std::vector<std::shared_ptr<ThreadData> > thread_data;
    std::vector<std::shared_ptr<TractModel> > tract_models;
    std::vector<std::shared_ptr<TractRender> > tract_rendering;
public:
    tipl::color_map color_map;
    tipl::color_map_rgb color_map_rgb;
    void update_color_map(void);
public:
    std::vector<std::shared_ptr<TractModel> > get_checked_tracks(void);
    std::vector<std::shared_ptr<TractRender> > get_checked_tracks_rendering(void);
    std::vector<std::shared_ptr<TractRender::end_reading> > start_reading_checked_tracks(void);
    std::vector<std::shared_ptr<TractRender::end_writing> > start_writing_checked_tracks(void);
    enum {none = 0,select = 1,del = 2,cut = 3,paint = 4,move = 5}edit_option;
    void addNewTracts(QString tract_name,bool checked = true);
    void addNewTracts(std::shared_ptr<TractModel> new_tract,bool checked = true);
    void addConnectometryResults(std::vector<std::vector<std::vector<float> > >& greater,
                                 std::vector<std::vector<std::vector<float> > >& lesser);
    void draw_tracts(unsigned char dim,int pos,
                     const tipl::shape<2>& slice_image_shape,float display_ratio,QImage& tract_image);

    QString output_format(void);
    template<typename fun_type>
    bool for_current_bundle(int cur_row,fun_type&& fun)
    {
        if(cur_row < 0 || item(cur_row,0)->checkState() != Qt::Checked)
            return false;
        {
            auto lock = tract_rendering[uint32_t(cur_row)]->start_writing();
            fun();
            tract_rendering[uint32_t(cur_row)]->need_update = true;
        }
        item(cur_row,1)->setText(QString::number(tract_models[uint32_t(cur_row)]->get_visible_track_count()));
        return true;
    }
    template<typename fun_type>
    bool for_each_bundle(fun_type&& fun,const std::string& indices = {})
    {
        std::vector<unsigned int> selected;
        if(indices.empty())
            for(unsigned int index = 0;index < tract_models.size();++index)
                if(item(int(index),0)->checkState() == Qt::Checked)
                    selected.push_back(index);
        else
            for(const auto& text : QString::fromStdString(indices).split('&'))
            {
                bool okay;
                auto index = text.toUInt(&okay);
                if(!okay || index >= tract_models.size())
                    return error_msg = "invalid tract index: "+text.toStdString(),false;
                if(std::find(selected.begin(),selected.end(),index) == selected.end())
                    selected.push_back(index);
            }

        std::vector<unsigned char> changed(selected.size());
        {
            tipl::par_for(selected.size(),[&](unsigned int i)
            {
                auto lock = tract_rendering[selected[i]]->start_writing();
                if(fun(selected[i]))
                {
                    changed[i] = true;
                    tract_rendering[selected[i]]->need_update = true;
                }
            });
        }
        bool updated = false;
        for(unsigned int i = 0;i < selected.size();++i)
            if(changed[i])
            {
                updated = true;
                item(int(selected[i]),1)->setText(QString::number(tract_models[selected[i]]->get_visible_track_count()));
                item(int(selected[i]),2)->setText(QString::number(tract_models[selected[i]]->get_deleted_track_count()));
            }
        if(updated)
            emit tract_changed();
        return true;
    }
public:
    unsigned int render_time = 200;
    bool render_tracts(GLWidget* glwidget,std::chrono::high_resolution_clock::time_point end_time);
    bool render_tracts(GLWidget* glwidget);
    std::string error_msg;
    bool command(std::vector<std::string> cmd);

signals:
    void tract_changed(void);
private:
    void delete_row(int row);
public slots:

    void start_tracking(void);

    void fetch_tracts(void);
    void show_tracking_progress(void);

    void edit_tracts(void);
    void stop_tracking(void);
    void move_up(void);
    void move_down(void);
    void show_report(void);

    void need_update_all(void);

    void cell_changed(int,int);

};

#endif // TRACTTABLEWIDGET_H
