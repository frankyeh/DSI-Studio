#include <QContextMenuEvent>
#include <QDoubleSpinBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include "devicetablewidget.h"
#include "tracking/tracking_window.h"
#include "ui_tracking_window.h"
#include "region/regiontablewidget.h"


QWidget *DeviceTypeDelegate::createEditor(QWidget *parent,
                                     const QStyleOptionViewItem &option,
                                     const QModelIndex &index) const
{
    if (index.column() == 1)
    {
        if(device_types.empty())
            load_device_content();
        QComboBox *comboBox = new QComboBox(parent);
        comboBox->addItem("Please Select Device");
        for(size_t i = 0;i < device_types.size();++i)
            comboBox->addItem(device_types[i].c_str());
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
        if (index.column() >= 3 && index.column() <= 8)
        {
            float max_v[6] = {100.0f,512.0f,512.0f,512.0f,180.0f,180.0f};
            float min_v[6] = {0.5f,-512.0f,-512.0f,-512.0f,-180.0f,0.0f};
            QDoubleSpinBox* sb = new QDoubleSpinBox(parent);
            sb->setRange(min_v[index.column()-3],max_v[index.column()-3]);
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
            if (index.column() >= 3 && index.column() <= 8)
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
            if (index.column() >= 3 && index.column() <= 8)
                model->setData(index,dynamic_cast<QDoubleSpinBox*>(editor)->value());
            else
                QItemDelegate::setModelData(editor,model,index);
}


void DeviceTypeDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}

void get_devices_statistics(std::shared_ptr<fib_data> handle,
                            const std::vector<std::shared_ptr<Device> >& devices,
                            std::string& result)
{
    std::vector<std::vector<tipl::vector<3>>> lead_positions_distal,
                                              lead_positions_center,
                                              lead_positions_proximal;
    size_t max_contact_count = 0;
    for(auto each : devices)
    {
        lead_positions_proximal.emplace_back(each->get_lead_positions(handle->vs[0],1.0f));
        lead_positions_center.emplace_back(each->get_lead_positions(handle->vs[0],0.5f));
        lead_positions_distal.emplace_back(each->get_lead_positions(handle->vs[0],0.0f));
        max_contact_count = std::max<size_t>(max_contact_count,lead_positions_center.back().size());
    }

    std::ostringstream out;
    out << "Name";
    for(auto each : devices)
        out << "\t" << each->name;
    out << std::endl;

    out << "tip location (voxels)";
    for(auto each : devices)
        out << "\t" << each->pos;
    out << std::endl;

    out << "direction";
    for(auto each : devices)
        out << "\t" << each->dir;
    out << std::endl;

    for(unsigned int i = 0;i < max_contact_count;++i)
    {
        out << "contact" << i << " distal end position (voxels)";
        for(unsigned int j = 0;j < lead_positions_distal.size();++j)
        {
            out << "\t";
            if(i < lead_positions_distal[j].size())
                out << lead_positions_distal[j][i];
        }
        out << std::endl;
        out << "contact" << i << " center position (voxels)";
        for(unsigned int j = 0;j < lead_positions_center.size();++j)
        {
            out << "\t";
            if(i < lead_positions_center[j].size())
                out << lead_positions_center[j][i];
        }
        out << std::endl;

        out << "contact" << i << " proximal end position (voxels)";
        for(unsigned int j = 0;j < lead_positions_proximal.size();++j)
        {
            out << "\t";
            if(i < lead_positions_proximal[j].size())
                out << lead_positions_proximal[j][i];
        }
        out << std::endl;

    }
    result = out.str();
}

DeviceTableWidget::DeviceTableWidget(tracking_window& cur_tracking_window_,QWidget *parent)
    : QTableWidget(parent),cur_tracking_window(cur_tracking_window_)
{
    setColumnCount(9);
    setColumnWidth(0,90);
    setColumnWidth(1,140);
    setColumnWidth(2,40);
    setColumnWidth(3,60);
    setColumnWidth(4,60);
    setColumnWidth(5,60);
    setColumnWidth(6,60);
    setColumnWidth(7,60);
    setColumnWidth(8,60);

    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setAlternatingRowColors(true);

    setItemDelegate(new DeviceTypeDelegate(this,cur_tracking_window.handle->dim));

    connect(this, &QTableWidget::itemChanged, this, [=](QTableWidgetItem* item) {
        if (item->column() != 0 || !(item->flags() & Qt::ItemIsUserCheckable))
            return;
        auto current = item->checkState();
        auto previous = static_cast<Qt::CheckState>(item->data(Qt::UserRole+1).toInt());
        if (current != previous) {
            item->setData(Qt::UserRole+1, current);
            emit need_update();
        }
    });

    connect(this,SIGNAL(itemChanged(QTableWidgetItem*)),this,SLOT(updateDevices(QTableWidgetItem*)));
    setEditTriggers(QAbstractItemView::DoubleClicked|QAbstractItemView::EditKeyPressed);

    set_coordinate(false);
}
void DeviceTableWidget::contextMenuEvent ( QContextMenuEvent * event )
{
    if (event->reason() == QContextMenuEvent::Mouse)
    {
        cur_tracking_window.ui->menuDevices->popup(event->globalPos());
    }
}
tipl::vector<3> DeviceTableWidget::handle_coordinates(tipl::vector<3> pos,bool inverse)
{
    if(locators.size() != 3)
        return pos;
    auto vs = cur_tracking_window.handle->vs;
    tipl::vector<3> ac_mm(locators[0]->pos);
    tipl::vector<3> pc_mm(locators[1]->pos);
    tipl::vector<3> inter_mm(locators[2]->pos);
    ac_mm.elem_mul(vs);
    pc_mm.elem_mul(vs);
    inter_mm.elem_mul(vs);

    // 2. Define axes
    tipl::vector<3> y = ac_mm - pc_mm;
    y.normalize();

    tipl::vector<3> ac_inter = inter_mm - ac_mm;
    tipl::vector<3> x = y.cross_product(ac_inter);
    x.normalize();

    tipl::vector<3> z = x.cross_product(y);
    z.normalize();

    // 4. Mid-commissural point (origin)
    tipl::vector<3> mcp = (ac_mm + pc_mm) * 0.5;

    // 3. Build rotation matrix (columns = x, y, z)
    tipl::matrix<3,3> R;
    if(inverse)
    {
        R[0] = x[0]; R[1] = x[1]; R[2] = x[2];
        R[3] = y[0]; R[4] = y[1]; R[5] = y[2];
        R[6] = z[0]; R[7] = z[1]; R[8] = z[2];
        pos.rotate(R);
        // 6. Translate back from MCC origin
        pos += mcp;
        // 7. Convert mm to voxel
        pos.elem_div(vs);
    }
    else
    {
        R[0] = x[0]; R[1] = y[0]; R[2] = z[0];
        R[3] = x[1]; R[4] = y[1]; R[5] = z[1];
        R[6] = x[2]; R[7] = y[2]; R[8] = z[2];
        // 5. Convert voxel coordinate to mm
        pos.elem_mul(vs);
        pos -= mcp;
        // 6. Get position in MCC space
        pos.rotate(R);
    }
    return pos;
}
void DeviceTableWidget::updateDevices(QTableWidgetItem* cur_item)
{
    if(no_update)
        return;
    const double PI = 3.14159265358979323846;
    auto a2v = [](float phi,float theta, tipl::vector<3>& v)
    {
        float sin_theta = std::sin(theta);
        v[0] = sin_theta * std::cos(phi);
        v[1] = sin_theta * std::sin(phi);
        v[2] = std::cos(theta);
    };


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

            // when creating new device, other items may not be created yet
            if(item(cur_item->row(),3))
            {
                if(previous_name == "Device" && tipl::begins_with(device->type,"Scale"))
                {
                    device->name = "50 mm";
                    item(cur_item->row(),3)->setText(QString::number(double(device->length = 50)));
                }
                if(device->type == "Locator" && device->length > 5)
                    item(cur_item->row(),3)->setText(QString::number(double(device->length = 1)));

                if(device->type != "Locator" && device->length < 5)
                    item(cur_item->row(),3)->setText(QString::number(double(device->length = 30)));
            }
        }
        break;
    case 2:
        device->color = uint32_t(cur_item->data(Qt::UserRole).toInt());
        break;
    case 3:
        device->length = float(cur_item->text().toDouble());
        break;
    case 4:
        if(locators.size() == 3)
            device->pos = handle_coordinates(tipl::vector<3>(
                                             cur_item->text().toDouble(),
                                             item(cur_item->row(),5)->text().toDouble(),
                                             item(cur_item->row(),6)->text().toDouble()),true);
        else
            device->pos[0] = float();
        break;
    case 5:
        if(locators.size() == 3)
            device->pos = handle_coordinates(tipl::vector<3>(
                                             item(cur_item->row(),4)->text().toDouble(),
                                             cur_item->text().toDouble(),
                                             item(cur_item->row(),6)->text().toDouble()),true);
        else
            device->pos[1] = float(cur_item->text().toDouble());
        break;
    case 6:
        if(locators.size() == 3)
            device->pos = handle_coordinates(tipl::vector<3>(
                                             item(cur_item->row(),4)->text().toDouble(),
                                             item(cur_item->row(),5)->text().toDouble(),
                                             cur_item->text().toDouble()),true);
        else
            device->pos[2] = float(cur_item->text().toDouble());
        break;
    case 7:
        {
            float phi = float(cur_item->text().toDouble()*PI/180.0);
            float theta = std::acos(device->dir[2]);
            a2v(phi,theta,device->dir);
        }
        break;
    case 8:
        {
            float phi = std::atan2(device->dir[1], device->dir[0]);
            float theta = float(cur_item->text().toDouble()*PI/180.0);
            a2v(phi,theta,device->dir);
        }
        break;
    }
    emit need_update();
}

void DeviceTableWidget::new_device(std::shared_ptr<Device> device)
{
    const double PI = 3.14159265358979323846;
    insertRow(int(devices.size())-1);
    no_update = true;
    auto p = handle_coordinates(device->pos);
    QTableWidgetItem *items[9] =
    {
        new QTableWidgetItem(device->name.c_str()),
        new QTableWidgetItem(device->type.c_str()),
        new QTableWidgetItem(QString::number(uint32_t(device->color))),
        new QTableWidgetItem(QString::number(double(device->length))),
        new QTableWidgetItem(QString::number(double(p[0]))),
        new QTableWidgetItem(QString::number(double(p[1]))),
        new QTableWidgetItem(QString::number(double(p[2]))),
        new QTableWidgetItem(QString::number(double(std::atan2(device->dir[1], device->dir[0]))*180.0/PI)),
        new QTableWidgetItem(QString::number(double(std::acos(device->dir[2]))*180.0/PI))
    };
    items[0]->setCheckState(Qt::Checked);
    items[2]->setData(Qt::UserRole,uint32_t(device->color));

    for(size_t i = 0;i < 9;++i)
        setItem(int(devices.size())-1, i, items[i]);
    for(size_t i = 1;i < 9;++i)
        openPersistentEditor(items[i]);

    setRowHeight(int(devices.size())-1,22);
    setCurrentCell(int(devices.size())-1,0);

    cur_tracking_window.ui->DeviceDockWidget->show();
    no_update = false;
    emit need_update();

}
void DeviceTableWidget::check_all(void)
{
    for(int row = 0;row < rowCount();++row)
        item(row,0)->setCheckState(Qt::Checked);
    emit need_update();
}
void DeviceTableWidget::uncheck_all(void)
{
    for(int row = 0;row < rowCount();++row)
        item(row,0)->setCheckState(Qt::Unchecked);
    emit need_update();
}
bool DeviceTableWidget::load_device(const std::string& filename)
{
    bool result = false;
    for(const auto& line : tipl::read_text_file(filename))
    {
        devices.push_back(std::make_shared<Device>());
        devices.back()->from_str(line);
        new_device(devices.back());
        result = true;
    }
    return result;
}
void DeviceTableWidget::shift_device(size_t index,float sel_length,const tipl::vector<3>& dis)
{
    const double PI = 3.14159265358979323846;
    if(index >= devices.size())
        return;
    devices[index]->move(sel_length,dis);
    no_update = true;
    auto p = handle_coordinates(devices[index]->pos);
    item(index,4)->setText(QString::number(double(p[0])));
    item(index,5)->setText(QString::number(double(p[1])));
    item(index,6)->setText(QString::number(double(p[2])));
    item(index,7)->setText(QString::number(double(std::atan2(devices[index]->dir[1], devices[index]->dir[0]))*180.0/PI));
    item(index,8)->setText(QString::number(double(std::acos(devices[index]->dir[2]))*180.0/PI));
    no_update = false;
    emit need_update();
}
void DeviceTableWidget::load_device(void)
{
    for(auto each : QFileDialog::getOpenFileNames(this,"Open device","device.dv.csv","CSV file(*dv.csv);;All files(*)"))
        if(!load_device(each.toStdString()))
        {
            QMessageBox::critical(this,"ERROR","cannot load device file " + each);
            return;
        }
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
    std::ofstream out(filename.toStdString().c_str());
    out << devices[uint32_t(currentRow())]->to_str();
}
void DeviceTableWidget::assign_colors(void)
{
    for(unsigned int index = 0;index < devices.size();++index)
    {
        tipl::rgb c = tipl::rgb::generate_hue(index);
        c.a = 255;
        item(int(index),2)->setData(Qt::UserRole,0xFF000000 | uint32_t(c));
        devices[index]->color = c.color;
    }
    emit need_update();
}
size_t DeviceTableWidget::get_device(const std::string& name)
{
    for(size_t i = 0;i < devices.size();++i)
        if(devices[i]->name == name)
            return i;
    return devices.size();
}
void DeviceTableWidget::set_coordinate(bool is_acpc)
{
    locators.clear();
    if(is_acpc)
    {
        char name[3][6] = {"AC","PC","Inter"};
        for(int i = 0;i < 3;++i)
        {
            auto index = get_device(name[i]);
            if(index == devices.size())
            {
                command({"set_acpc"});
                break;
            }
        }
        for(int i = 0;i < 3;++i)
        {
            auto index = get_device(name[i]);
            if(index == devices.size())
            {
                locators.clear();
                is_acpc = false;
                break;
            }
            locators.push_back(devices[index]);
        }
    }
    no_update = true;
    for(size_t i = 0;i < devices.size();++i)
    {
        auto p = handle_coordinates(devices[i]->pos);
        item(i,4)->setText(QString::number(double(p[0])));
        item(i,5)->setText(QString::number(double(p[1])));
        item(i,6)->setText(QString::number(double(p[2])));
    }
    no_update = false;
    QStringList header;
    if(is_acpc)
        header << "Name" << "Type" << "Color" << "Length" << "X(mm)" << "Y(mm)" << "Z(mm)" << "phi" << "theta";
    else
        header << "Name" << "Type" << "Color" << "Length" << "i(voxels)" << "j(voxels)" << "k(voxels)" << "phi" << "theta";
    setHorizontalHeaderLabels(header);
}
bool DeviceTableWidget::command(std::vector<std::string> cmd)
{
    auto run = cur_tracking_window.history.record(error_msg,cmd);
    if(cmd.size() < 3)
        cmd.resize(3);

    auto get_cur_row = [&](std::string& cmd_text,int& cur_row)->bool
    {
        if (devices.empty())
        {
            error_msg = "no available device";
            return false;
        }
        bool okay = true;
        if(cmd_text.empty())
            cmd_text = std::to_string(cur_row);
        else
            cur_row = QString::fromStdString(cmd_text).toInt(&okay);
        if (cur_row >= devices.size() || !okay)
        {
            error_msg = "invalid device index: " + cmd_text;
            return false;
        }
        return run->succeed();
    };

    if(cmd[0] == "new_device")
    {
        // cmd[1]: position
        if(cur_tracking_window.handle->vs[0] !=
           cur_tracking_window.handle->vs[2])
            QMessageBox::warning(&cur_tracking_window,"WARNING",
                                 "Non-isotropic voxels in the current space could cause substantial errors in device location.");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0,10.0);
        auto dx = dis(gen);
        auto dy = dis(gen);

        devices.push_back(std::make_shared<Device>());
        if(cmd[1].empty())
            devices.back()->pos = tipl::vector<3>(
                    cur_tracking_window.handle->dim[0]/2+dx,
                    cur_tracking_window.handle->dim[1]/2+dy,
                    cur_tracking_window.handle->dim[2]/2+dis(gen)/4.0);
        else
            std::istringstream(cmd[1]) >> devices.back()->pos[0] >> devices.back()->pos[1] >> devices.back()->pos[2];
        devices.back()->name = "Device"+std::to_string(device_num++);
        devices.back()->dir = tipl::vector<3>(dx,dy,50.0);
        devices.back()->dir.normalize();
        devices.back()->color = tipl::rgb::generate(devices.size()) | 0xFF000000;
        new_device(devices.back());
        return run->succeed();
    }
    if(cmd[0] == "move_device")
    {
        // cmd[1]: position
        // cmd[2]: device index (default: current)
        if(cmd[1].empty())
            return run->failed("please specify location");
        int cur_row = currentRow();
        if(!get_cur_row(cmd[2],cur_row))
            return false;
        tipl::vector<3> pos;
        std::istringstream(cmd[1]) >> pos[0] >> pos[1] >> pos[2];
        shift_device(cur_row,0,pos-devices[cur_row]->pos);
        return run->succeed();
    }
    if(cmd[0] == "push_device")
    {
        // cmd[1]: device index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        shift_device(cur_row,0,(devices[cur_row]->dir*-0.5f).elem_div(cur_tracking_window.handle->vs));
        return run->succeed();
    }
    if(cmd[0] == "pull_device")
    {
        // cmd[1]: device index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        shift_device(cur_row,0,(devices[cur_row]->dir*0.5f).elem_div(cur_tracking_window.handle->vs));
        return run->succeed();
    }
    if(cmd[0] == "copy_device")
    {
        // cmd[1] : device index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        devices.push_back(std::make_shared<Device>());
        devices.back()->pos = devices[cur_row]->pos;
        devices.back()->pos[0] += 1.0f;
        devices.back()->length = devices[cur_row]->length;
        devices.back()->dir = devices[cur_row]->dir;
        devices.back()->color = tipl::rgb::generate(devices.size()) | 0xFF000000;
        devices.back()->type = devices[cur_row]->type;
        devices.back()->name = tipl::split(tipl::split(devices[cur_row]->type,':').front(),' ').back() +
                                std::to_string(devices.size());
        new_device(devices.back());
        return run->succeed();
    }
    if(cmd[0] == "set_acpc")
    {
        if(!cur_tracking_window.handle->map_to_mni())
            return run->failed("cannot map to MNI space: " + cur_tracking_window.handle->error_msg);

        char name[3][6] = {"AC","PC","Inter"};
        for(int i = 0;i < 3;++i)
        {
            auto index = get_device(name[i]);
            if(index != devices.size())
            {
                devices.erase(devices.begin()+index);
                removeRow(index);
            }
        }
        tipl::vector<3> pos[3] = {{0.05f,2.7f,-4.8f},
                                  {0.0f,-25.0f,-2.0f},
                                  {0.0f,-10.0f,30.0f}};
        for(int i = 0;i < 3;++i)
        {
            devices.push_back(std::make_shared<Device>());
            devices.back()->name = name[i];
            devices.back()->pos = pos[i];
            cur_tracking_window.handle->mni2sub(devices.back()->pos);
            devices.back()->dir = tipl::vector<3>(0.0f,0.0f,1.0f);
            devices.back()->length = 2.0f;
            devices.back()->color = tipl::rgb::generate(devices.size()) | 0xFF000000;
            devices.back()->type = "Locator";
            new_device(devices.back());
        }
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "delete_device")
    {
        // cmd[1] : device index (default: current)
        int cur_row = currentRow();
        if(!get_cur_row(cmd[1],cur_row))
            return false;
        devices.erase(devices.begin()+cur_row);
        removeRow(cur_row);
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "delete_all_devices")
    {
        setRowCount(0);
        devices.clear();
        emit need_update();
        return run->succeed();
    }
    if(cmd[0] == "save_all_devices")
    {
        if (devices.empty())
            return run->canceled();
        if(cmd[1].empty() && (cmd[1] = QFileDialog::getSaveFileName(
                               this,"Save all devices",item(currentRow(),0)->text() + ".dv.csv",
                               "CSV file(*dv.csv);;All files(*)").toStdString()).empty())
            return run->canceled();
        std::ofstream out(cmd[1]);
        for (size_t i = 0; i < devices.size(); ++i)
            if (item(int(i),0)->checkState() == Qt::Checked)
            {
                devices[i]->name = item(int(i),0)->text().toStdString();
                out << devices[i]->to_str();
            }
        return run->succeed();
    }


    return run->not_processed();
}


void DeviceTableWidget::lead_to_roi(void)
{
    if (devices.empty() || currentRow() >= int(devices.size()))
        return;
    bool okay = true;
    float resolution = 8.0f;
    short radius = short(QInputDialog::getDouble(this,QApplication::applicationName(),"Input region radius (mm)",2.0,1.0,10.0,1,&okay)*resolution);
    if (!okay)
        return;
    auto& cur_device = devices[uint32_t(currentRow())];
    auto lead_pos = cur_device->get_lead_positions(cur_tracking_window.handle->vs[0]);

    cur_tracking_window.regionWidget->begin_update();

    std::vector<std::shared_ptr<ROIRegion> > new_regions;
    for(unsigned int i = 0;i < lead_pos.size();++i)
    {
        cur_tracking_window.regionWidget->add_high_reso_region((cur_device->name + "_lead_"+ std::to_string(i)).c_str(),resolution);
        new_regions.push_back(cur_tracking_window.regionWidget->regions.back());
    }
    std::vector<tipl::vector<3,short> > voxels;
    short distance2 = radius*radius;
    for (short z = -radius; z <= radius; ++z)
        for (short y = -radius,zz = z*z; y <= radius; ++y)
            for (short x = -radius,yy_zz = y*y+zz; x <= radius; ++x)
                if (x*x + yy_zz <= distance2)
                    voxels.push_back(tipl::vector<3,short>(x,y,z));
    tipl::adaptive_par_for(lead_pos.size(),[&](unsigned int i)
    {
        auto points = voxels;
        tipl::add_constant(points.begin(),points.end(),tipl::vector<3,short>(lead_pos[i]*resolution+0.5f));
        new_regions[i]->add_points(std::move(points));
    });
    cur_tracking_window.regionWidget->end_update();
}
