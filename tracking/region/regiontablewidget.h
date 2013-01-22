#ifndef REGIONTABLEWIDGET_H
#define REGIONTABLEWIDGET_H
#include <QTableWidget>
#include <QItemDelegate>
#include <QComboBox>
#include <vector>
#include "Regions.h"
#include <boost/ptr_container/ptr_vector.hpp>

class ThreadData;
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
    unsigned char regions_index;
    void do_action(int id);
    void whole_brain_points(std::vector<image::vector<3,short> >& points);
signals:
    void need_update(void);
public:
    boost::ptr_vector<ROIRegion> regions;
public:
    explicit RegionTableWidget(tracking_window& cur_tracking_window,QWidget *parent = 0);
    ~RegionTableWidget();

    QColor currentRowColor(void);
    bool has_seeding(void);
    void add_region(QString name,unsigned char type);
    void setROIs(ThreadData* data);
public slots:
    void draw_region(QImage& image);
    void draw_mosaic_region(QImage& image,unsigned int mosaic_size,unsigned int skip);
    void updateRegions(QTableWidgetItem* item);
    void new_region(void);
    void save_region(void);
    void save_region_info(void);
    void load_region(void);
    void delete_region(void);
    void delete_all_region(void);
    void add_points(std::vector<image::vector<3,short> >& points,bool erase);
    void check_check_status(int,int);
    void add_atlas(void);
    void whole_brain(void);
    void show_statistics(void);

    // actions
    void action_smoothing(void){do_action(0);}
    void action_erosion(void){do_action(1);}
    void action_dilation(void){do_action(2);}
    void action_defragment(void){do_action(3);}
    void action_negate(void){do_action(4);}
    void action_flipx(void){do_action(5);}
    void action_flipy(void){do_action(6);}
    void action_flipz(void){do_action(7);}
    void action_shiftx(void){do_action(9);}
    void action_shiftnx(void){do_action(10);}
    void action_shifty(void){do_action(11);}
    void action_shiftny(void){do_action(12);}
    void action_shiftz(void){do_action(13);}
    void action_shiftnz(void){do_action(14);}
    void action_threshold(void){do_action(8);}
};

#endif // REGIONTABLEWIDGET_H
