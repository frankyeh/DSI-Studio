#include <QContextMenuEvent>
#include <QDoubleSpinBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QFileInfo>
#include "devicetablewidget.h"
#include "tracking/tracking_window.h"
#include "ui_tracking_window.h"


QWidget *DeviceTypeDelegate::createEditor(QWidget *parent,
                                     const QStyleOptionViewItem &option,
                                     const QModelIndex &index) const
{
    if (index.column() == 1)
    {
        QComboBox *comboBox = new QComboBox(parent);
        comboBox->addItem("Please Select Device");
        comboBox->addItem(device_types[0]);
        comboBox->addItem(device_types[1]);
        comboBox->addItem(device_types[2]);
        comboBox->addItem(device_types[3]);
        comboBox->addItem(device_types[4]);
        comboBox->addItem(device_types[5]);
        comboBox->addItem(device_types[6]);
        comboBox->addItem(device_types[7]);
        comboBox->addItem(device_types[8]);
        comboBox->addItem(device_types[9]);
        comboBox->addItem(device_types[10]);
        comboBox->addItem(device_types[11]);

        connect(comboBox, SIGNAL(activated(int)), this, SLOT(emitCommitData()));
        return comboBox;
    }
    else if (index.column() == 2)
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        return sd;
    }
    else
        if (index.column() == 3)
        {
            QDoubleSpinBox* sb = new QDoubleSpinBox(parent);
            sb->setRange(0.0,100.0);
            sb->setDecimals(1);
            connect(sb, SIGNAL(valueChanged(double)), this, SLOT(emitCommitData()));
            return sb;
        }
        return QItemDelegate::createEditor(parent,option,index);
}

void DeviceTypeDelegate::setEditorData(QWidget *editor,
                                  const QModelIndex &index) const
{

    if (index.column() == 1)
        dynamic_cast<QComboBox*>(editor)->setCurrentText(index.model()->data(index).toString());
    else
        if (index.column() == 2)
        {
            tipl::rgb color(uint32_t(index.data(Qt::UserRole).toInt()));
            dynamic_cast<QColorToolButton*>(editor)->setColor(
                QColor(color.r,color.g,color.b,color.a));
        }
        else
            if (index.column() == 3)
                dynamic_cast<QDoubleSpinBox*>(editor)->setValue(index.model()->data(index).toDouble());
            return QItemDelegate::setEditorData(editor,index);
}

void DeviceTypeDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                 const QModelIndex &index) const
{
    if (index.column() == 1)
        model->setData(index,dynamic_cast<QComboBox*>(editor)->currentText());
    else
        if (index.column() == 2)
            model->setData(index,int((dynamic_cast<QColorToolButton*>(editor)->color().rgba())),Qt::UserRole);
        else
            if (index.column() == 3)
                model->setData(index,dynamic_cast<QDoubleSpinBox*>(editor)->value());
            else
                QItemDelegate::setModelData(editor,model,index);
}

void DeviceTypeDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}


DeviceTableWidget::DeviceTableWidget(tracking_window& cur_tracking_window_,QWidget *parent)
    : QTableWidget(parent),cur_tracking_window(cur_tracking_window_)
{
    setColumnCount(4);
    setColumnWidth(0,100);
    setColumnWidth(1,140);
    setColumnWidth(2,40);
    setColumnWidth(3,60);

    QStringList header;
    header << "Name" << "Type" << "Color" << "Length";
    setHorizontalHeaderLabels(header);
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setAlternatingRowColors(true);
    setStyleSheet("QTableView {selection-background-color: #AAAAFF; selection-color: #000000;}");

    setItemDelegate(new DeviceTypeDelegate(this));

    connect(this,SIGNAL(cellClicked(int,int)),this,SLOT(check_status(int,int)));
    connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateDevices(QTableWidgetItem*)));
    setEditTriggers(QAbstractItemView::DoubleClicked|QAbstractItemView::EditKeyPressed);
}
void DeviceTableWidget::contextMenuEvent ( QContextMenuEvent * event )
{
    if (event->reason() == QContextMenuEvent::Mouse)
    {
        cur_tracking_window.ui->menuDevices->popup(event->globalPos());
    }
}

void DeviceTableWidget::updateDevices(QTableWidgetItem* cur_item)
{
    auto& device = devices[uint32_t(cur_item->row())];
    switch(cur_item->column())
    {
    case 1:
        {
            QString previous_name = device->type.empty() ?
                        QString("Device") :
                        QString(device->type.c_str()).split(':')[0].split(' ').back();
            auto new_default_name = cur_item->text().split(':')[0].split(' ').back();
            auto* head_item = item(cur_item->row(),0);
            device->type = cur_item->text().toStdString();
            if(!new_default_name.isEmpty() &&
               head_item->text().length() > previous_name.length() &&
               head_item->text().left(previous_name.length()) == previous_name)
               head_item->setText(new_default_name+head_item->text().right(head_item->text().length()-previous_name.length()));

        }
        break;
    case 2:
        device->color = uint32_t(cur_item->data(Qt::UserRole).toInt());
        break;
    case 3:
        device->length = float(cur_item->text().toDouble());
        break;
    }
    emit need_update();
}
void DeviceTableWidget::check_status(int row, int col)
{
    if (col != 0)
        return;
    setCurrentCell(row,col);
    if (item(row,0)->checkState() == Qt::Checked)
    {
        if (item(row,0)->data(Qt::ForegroundRole) == QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
            emit need_update();
        }
    }
    else
    {
        if (item(row,0)->data(Qt::ForegroundRole) != QBrush(Qt::gray))
        {
            item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
            emit need_update();
        }
    }
}


void DeviceTableWidget::newDevice()
{
    static unsigned int device_num = 1;
    if(new_device_str.isEmpty())
    {
        QAction* pAction = qobject_cast<QAction*>(sender());
        devices.push_back(std::make_shared<Device>());
        // random location
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0,10.0);
        auto dx = dis(gen);
        auto dy = dis(gen);
        devices.back()->pos = tipl::vector<3>(
                cur_tracking_window.handle->dim[0]/2+dx,
                cur_tracking_window.handle->dim[1]/2+dy,
                cur_tracking_window.handle->dim[2]/2+dis(gen)/4.0);
        devices.back()->dir = tipl::vector<3>(dx,dy,pAction->text().contains("SEEG") ? 0.0: 50.0);
        devices.back()->dir.normalize();
        devices.back()->name = (pAction->text().split(':')[0].split(' ').back()+QString::number(device_num++)).toStdString();
    }
    else {
        devices.push_back(std::make_shared<Device>());
        devices.back()->from_str(new_device_str.toStdString());
        new_device_str.clear();
    }
    new_device(devices.back());
}
void DeviceTableWidget::new_device(std::shared_ptr<Device> device)
{

    insertRow(int(devices.size())-1);
    QTableWidgetItem *item0 = new QTableWidgetItem(device->name.c_str());
    item0->setCheckState(Qt::Checked);
    QTableWidgetItem *item1 = new QTableWidgetItem(device->type.c_str());
    item1->setData(Qt::ForegroundRole,QBrush(Qt::white));
    QTableWidgetItem *item2 = new QTableWidgetItem(QString::number(uint32_t(device->color)));
    item2->setData(Qt::ForegroundRole,QBrush(Qt::white));
    item2->setData(Qt::UserRole,uint32_t(device->color));
    QTableWidgetItem *item3 = new QTableWidgetItem(QString::number(double(device->length)));
    item3->setData(Qt::ForegroundRole,QBrush(Qt::white));


    setItem(int(devices.size())-1, 0, item0);
    setItem(int(devices.size())-1, 1, item1);
    setItem(int(devices.size())-1, 2, item2);
    setItem(int(devices.size())-1, 3, item3);

    openPersistentEditor(item1);
    openPersistentEditor(item2);
    openPersistentEditor(item3);

    setRowHeight(int(devices.size())-1,22);
    setCurrentCell(int(devices.size())-1,0);

    cur_tracking_window.ui->DeviceDockWidget->show();
    emit need_update();
}
void DeviceTableWidget::copy_device()
{
    if(devices.empty())
        return;
    auto device_to_copy = devices.back();
    devices.push_back(std::make_shared<Device>());
    // random location
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2.0,2.0);

    devices.back()->pos = device_to_copy->pos + tipl::vector<3>(dis(gen),dis(gen),dis(gen));
    devices.back()->dir = device_to_copy->dir;
    devices.back()->name = device_to_copy->name;
    devices.back()->type = device_to_copy->type;
    devices.back()->length = device_to_copy->length;

    new_device(devices.back());
}
void DeviceTableWidget::check_all(void)
{
    for(int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Checked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::black));
    }
    emit need_update();
}
bool DeviceTableWidget::command(QString cmd,QString param)
{
    if(cmd == "save_all_devices")
    {
        std::ofstream out(param.toStdString().c_str());
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (item(int(i),0)->checkState() == Qt::Checked)
            {
                devices[i]->name = item(int(i),0)->text().toStdString();
                out << devices[i]->to_str();
            }
        }
        return true;
    }
    return false;
}
void DeviceTableWidget::uncheck_all(void)
{
    for(int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
    emit need_update();
}
bool DeviceTableWidget::load_device(QString Filename)
{
    std::ifstream in(Filename.toLocal8Bit().begin());
    if(!in)
        return false;
    std::string line;
    while(std::getline(in,line))
    {
        new_device_str = line.c_str();
        newDevice();
    }
    return true;
}
void DeviceTableWidget::load_device(void)
{
    QString filename = QFileDialog::getOpenFileName(
                           this,"Open device","device.dv.csv","CSV file(*dv.csv);;All files(*)");
    if (filename.isEmpty())
        return;
    load_device(filename);
}
void DeviceTableWidget::save_device(void)
{
    if (devices.empty() || currentRow() >= int(devices.size()))
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save device",item(currentRow(),0)->text() + ".dv.csv","CSV file(*dv.csv);;All files(*)");
    if (filename.isEmpty())
        return;
    devices[uint32_t(currentRow())]->name = item(currentRow(),0)->text().toStdString();
    std::ofstream out(filename.toLocal8Bit().begin());
    out << devices[uint32_t(currentRow())]->to_str();
}
void DeviceTableWidget::assign_colors(void)
{
    for(unsigned int index = 0;index < devices.size();++index)
    {
        tipl::rgb c;
        c.from_hsl((color_gen*1.1-std::floor(color_gen*1.1/6)*6)*3.14159265358979323846/3.0,0.85,0.7);
        c.a = 255;
        item(int(index),2)->setData(Qt::UserRole,0xFF000000 | uint32_t(c));
        devices[index]->color = c.color;
        color_gen++;
    }
    emit need_update();
}
void DeviceTableWidget::save_all_devices(void)
{
    if (devices.empty())
        return;
    QString filename = QFileDialog::getSaveFileName(
                           this,"Save all devices",item(currentRow(),0)->text() + ".dv.csv",
                           "CSV file(*dv.csv);;All files(*)");
    if (filename.isEmpty())
        return;
    command("save_all_devices",filename);
}

void DeviceTableWidget::delete_device(void)
{
    if (devices.empty() || currentRow() >= devices.size())
        return;
    devices.erase(devices.begin()+currentRow());
    removeRow(currentRow());
    emit need_update();
}

void DeviceTableWidget::delete_all_devices(void)
{
    setRowCount(0);
    devices.clear();
    emit need_update();
}

