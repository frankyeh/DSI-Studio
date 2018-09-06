#ifndef REGIONTABLEWIDGET_H
#define REGIONTABLEWIDGET_H
#include <QTableWidget>
#include <QItemDelegate>
#include <QComboBox>
#include <vector>
#include "Regions.h"
#include "tipl/tipl.hpp"

struct ThreadData;
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
    void contextMenuEvent ( QContextMenuEvent * event );
private:
    tracking_window& cur_tracking_window;
    void do_action(QString action);
    void whole_brain_points(std::vector<tipl::vector<3,short> >& points);
    bool load_multiple_roi_nii(QString file_name);
signals:
    void need_update(void);
public:
    std::vector<std::shared_ptr<ROIRegion> > regions;
    int color_gen = 10;
public:
    explicit RegionTableWidget(tracking_window& cur_tracking_window,QWidget *parent = 0);
    ~RegionTableWidget();

    QColor currentRowColor(void);
    bool has_seeding(void);
    void add_region_from_atlas(unsigned int atlas_id,unsigned int roi_is);
    void add_region(QString name,unsigned char type,int color = 0x00FFFFFF);
    void set_whole_brain(ThreadData* data);
    void setROIs(ThreadData* data);
    QString getROIname(void);
    template<typename value_type>
    void add_points(std::vector<tipl::vector<3,value_type> >& points,bool erase,float resolution = 1.0)
    {
        if (currentRow() < 0 || currentRow() >= regions.size())
            return;
        regions[currentRow()]->add_points(points,erase,resolution);
    }
    QString output_format(void);
public slots:
    void draw_region(tipl::color_image& I);
    void draw_edge(QImage& image,QImage& scaledimage);
    void draw_mosaic_region(QImage& image,unsigned int mosaic_size,unsigned int skip);
    void updateRegions(QTableWidgetItem* item);
    void new_region(void);
    void new_high_resolution_region(void);
    void copy_region(void);
    void save_region(void);
    void save_all_regions(void);
    void save_all_regions_to_dir(void);
    void save_region_info(void);
    void load_region(void);
    void delete_region(void);
    void delete_all_region(void);

    void check_check_status(int,int);
    void whole_brain(void);
    void show_statistics(void);
    void merge_all(void);
    void check_all(void);
    void uncheck_all(void);
    void move_up(void);
    void move_down(void);
    void undo(void);
    void redo(void);
    // actions
    void action_smoothing(void){do_action("smoothing");}
    void action_erosion(void){do_action("erosion");}
    void action_dilation(void){do_action("dilation");}
    void action_defragment(void){do_action("defragment");}
    void action_negate(void){do_action("negate");}
    void action_flipx(void){do_action("flipx");}
    void action_flipy(void){do_action("flipy");}
    void action_flipz(void){do_action("flipz");}
    void action_shiftx(void){do_action("shiftx");}
    void action_shiftnx(void){do_action("shiftnx");}
    void action_shifty(void){do_action("shifty");}
    void action_shiftny(void){do_action("shiftny");}
    void action_shiftz(void){do_action("shiftz");}
    void action_shiftnz(void){do_action("shiftnz");}
    void action_threshold(void){new_region();do_action("threshold");}
    void action_separate(void){do_action("separate");}
};

#endif // REGIONTABLEWIDGET_H
