#ifndef DEVICETABLEWIDGET_H
#define DEVICETABLEWIDGET_H
#include <QTableWidget>
#include <QItemDelegate>
#include <QComboBox>
#include "tracking/device.h"
class tracking_window;
class DeviceTypeDelegate : public QItemDelegate
{
    Q_OBJECT

    public:
    tipl::shape<3> dim;
    DeviceTypeDelegate(QObject *parent,tipl::shape<3> dim_)
         : QItemDelegate(parent),dim(dim_){}

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                                const QModelIndex &index) const;
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
          void setModelData(QWidget *editor, QAbstractItemModel *model,
                            const QModelIndex &index) const;
private slots:
    void emitCommitData();
 };


class DeviceTableWidget : public QTableWidget
{
    Q_OBJECT
protected:
    void contextMenuEvent(QContextMenuEvent * event ) override;
public:
    explicit DeviceTableWidget(tracking_window& cur_tracking_window,QWidget *parent = nullptr);
    // Header:
public:
    tracking_window& cur_tracking_window;
    unsigned int device_num = 1;
    std::vector<std::shared_ptr<Device> > devices;
    bool load_device(const std::string& filename);
    void shift_device(size_t index,float sel_length,const tipl::vector<3>& dis);
public:
    std::string error_msg;
    bool command(std::vector<std::string> cmd);
signals:
    void need_update(void);
private:
    QString new_device_str;
    void new_device(std::shared_ptr<Device> device);
public slots:
    void updateDevices(QTableWidgetItem* item);
    void load_device(void);
    void save_device(void);
    void assign_colors(void);
    void check_all(void);
    void uncheck_all(void);
    void detect_electrodes(void);
    void lead_to_roi(void);

};

#endif // DEVICETABLEWIDGET_H
