#ifndef REGIONTABLEWIDGET_H
#define REGIONTABLEWIDGET_H
#include <QTableWidget>
#include <QItemDelegate>
#include <QComboBox>
#include <vector>
#include "Regions.h"
class RoiMgr;
class tracking_window;


class ImageDelegate : public QItemDelegate
 {
     Q_OBJECT

 public:
    ImageDelegate(QObject *parent)
         : QItemDelegate(parent)
     {
     }

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                                const QModelIndex &index) const;
          void setEditorData(QWidget *editor, const QModelIndex &index) const;
          void setModelData(QWidget *editor, QAbstractItemModel *model,
                            const QModelIndex &index) const;
private slots:
    void emitCommitData();
 };

class RegionTableWidget : public QTableWidget
{
    Q_OBJECT
protected:
    void contextMenuEvent(QContextMenuEvent * event ) override;
private:
    tracking_window& cur_tracking_window;
    bool load_multiple_roi_nii(QString file_name,bool is_mni);
private:
    template<typename func>
    void for_each_checked_region(func fun)
    {
        for (unsigned int roi_index = 0;roi_index < regions.size();++roi_index)
        {
            if (item(roi_index,0)->checkState() != Qt::Checked)
                continue;
            fun(regions[roi_index]);
        }
    }
public:
    tipl::color_map_rgb color_map_rgb;
    std::vector<float> color_map_values;
    size_t tract_map_id = 0;
    void update_color_map(void);
    tipl::rgb get_region_rendering_color(size_t index);
public:
    std::vector<std::shared_ptr<ROIRegion> > get_checked_regions(void)
    {
        std::vector<std::shared_ptr<ROIRegion> > checked_regions;
        for_each_checked_region([&](std::shared_ptr<ROIRegion> region)
        {
            checked_regions.push_back(region);
        });
        return checked_regions;
    }
    void save_checked_region_label_file(QString filename,int first_index);
signals:
    void need_update(void);
public:
    std::vector<std::shared_ptr<ROIRegion> > regions;
    int color_gen = 0;
    std::string error_msg;
    bool command(std::vector<std::string> cmd);
    bool do_action(std::vector<std::string>& cmd);

    void check_row(size_t index,bool checked);
public:
    explicit RegionTableWidget(tracking_window& cur_tracking_window,QWidget *parent = nullptr);

    QColor currentRowColor(void);
    void add_merged_regions_from_atlas(std::shared_ptr<atlas> at,QString name,const std::vector<unsigned int>& roi_list);
    void add_row(int row,QString name);
    void add_region(QString name,unsigned char type = default_id,unsigned int color = 0xFFFFFFFF);
    void add_high_reso_region(QString name,float reso,unsigned char type = default_id,unsigned int color = 0xFFFFFFFF);
    void begin_update(void);
    void end_update(void);
    bool set_roi(const std::string& settings,std::shared_ptr<RoiMgr> roi);
    std::string get_roi_settings(void);
    QString getROIname(void);
    QString output_format(void);
public slots:
    void updateRegions(QTableWidgetItem* item);
    void draw_region(const tipl::matrix<4,4>& current_slice_T,unsigned char dim,int slice_pos,
                     const tipl::shape<2>& slice_image_shape,float display_ratio,QImage& scaled_image);

    void check_check_status(int,int);
    void move_up(void);
    void move_down(void);
    void undo(void);
    void redo(void);

};

#endif // REGIONTABLEWIDGET_H
