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
            float max_v[6] = {100.0f,float(dim[0]),float(dim[1]),float(dim[2]),180.0f,180.0f};
            float min_v[6] = {0.5f,0.0f,0.0f,0.0f,-180.0f,0.0f};
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

    QStringList header;
    header << "Name" << "Type" << "Color" << "Length" << "x" << "y" << "z" << "phi" << "theta";
    setHorizontalHeaderLabels(header);
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
        device->pos[0] = float(cur_item->text().toDouble());
        break;
    case 5:
        device->pos[1] = float(cur_item->text().toDouble());
        break;
    case 6:
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
    QTableWidgetItem *items[9] =
    {
        new QTableWidgetItem(device->name.c_str()),
        new QTableWidgetItem(device->type.c_str()),
        new QTableWidgetItem(QString::number(uint32_t(device->color))),
        new QTableWidgetItem(QString::number(double(device->length))),
        new QTableWidgetItem(QString::number(double(device->pos[0]))),
        new QTableWidgetItem(QString::number(double(device->pos[1]))),
        new QTableWidgetItem(QString::number(double(device->pos[2]))),
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
    item(index,4)->setText(QString::number(double(devices[index]->pos[0])));
    item(index,5)->setText(QString::number(double(devices[index]->pos[1])));
    item(index,6)->setText(QString::number(double(devices[index]->pos[2])));
    item(index,7)->setText(QString::number(double(std::atan2(devices[index]->dir[1], devices[index]->dir[0]))*180.0/PI));
    item(index,8)->setText(QString::number(double(std::acos(devices[index]->dir[2]))*180.0/PI));

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


void DeviceTableWidget::detect_electrodes(void)
{
    const CustomSliceModel* slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
    if(!slice)
    {
        QMessageBox::critical(this,"ERROR","Please insert CT images and switch current slice to it");
        return;
    }
    auto vs = cur_tracking_window.current_slice->vs;
    if(vs[0] < vs[2])
    {
        QMessageBox::critical(this,"ERROR","Error due to none-isotropic resolution. Please use [Tool][O2:View Image] to make it isotropic.");
        return;
    }
    if(slice->running)
        QMessageBox::information(this,QApplication::applicationName(),"Slice registration is undergoing, and the alignment may change.");


    bool ok;
    QString param = QInputDialog::getText(this,"Specify parameter",
        "Input minimum contact size, maximum contact size, and maximum contact distance, separated by comma.",QLineEdit::Normal,"2,20,8",&ok);
    if(!ok)
        return;
    QStringList params = param.split(",");
    if(params.size() != 3)
    {
        QMessageBox::critical(this,"ERROR","Invalid parameter");
        return;
    }

    float contact_distance_in_mm = params[2].toFloat()/vs[0];


    auto& I = slice->source_images;
    std::vector<std::vector<size_t> > regions;

    // use intensity threshold to locate possible contact regions
    {
        tipl::image<3,unsigned char> mask(I.shape());
        tipl::threshold(I,mask,tipl::max_value(I)*0.98);

        tipl::image<3,uint32_t> label(I.shape());
        tipl::morphology::connected_component_labeling(mask,label,regions);

        float voxel_size = vs[0]*vs[1]*vs[2];
        uint32_t min_size = uint32_t(params[0].toFloat()/voxel_size);
        uint32_t max_size = uint32_t(params[1].toFloat()/voxel_size);

        // check region size and whether there is nearby dark
        for(unsigned int i = 0;i < regions.size();++i)
            if(regions[i].size() >= min_size && regions[i].size() <= max_size)
            {
                std::sort(regions[i].begin(),regions[i].end());
                tipl::pixel_index<3> pos(regions[i][regions[i].size()/2],I.shape());
                std::vector<float> values = tipl::get_window(pos,I,uint32_t(contact_distance_in_mm));
                if(tipl::min_value(values) > 100)
                    regions[i].clear();
            }
            else
                regions[i].clear();
    }

    // get a list of possible contacts
    std::vector<uint32_t> contact_list;
    for(unsigned int i = 0;i < regions.size();++i)
        if(!regions[i].empty())
            contact_list.push_back(i);

    // calculating distance between contacts
    tipl::image<2,float> distance(tipl::shape<2>(uint32_t(contact_list.size()),uint32_t(contact_list.size())));
    for(tipl::pixel_index<2> p(distance.shape());p < int(distance.size());++p)
    {
        uint32_t i = uint32_t(p[0]);
        uint32_t j = uint32_t(p[1]);
        if(i <= j)
            continue;
        tipl::vector<3> p0(tipl::pixel_index<3>(regions[contact_list[i]][regions[contact_list[i]].size()/2],I.shape()));
        tipl::vector<3> p1(tipl::pixel_index<3>(regions[contact_list[j]][regions[contact_list[j]].size()/2],I.shape()));
        p0 -= p1;
        distance[tipl::pixel_index<2>(j,i,distance.shape()).index()] = distance[p.index()] = float(p0.length());
    }

    // grouping
    std::vector<std::vector<uint16_t> > contact_group;
    std::vector<short> contact_group_index(contact_list.size());
    std::fill(contact_group_index.begin(),contact_group_index.end(),-1);
    for(unsigned int i = 0;i < contact_list.size();++i)
    {
        std::vector<uint32_t> contact_to_merge;
        std::vector<uint16_t> groups; // the group of the contacts

        // find the nearest two contact
        {
            std::vector<float> dis(distance.begin()+contact_list.size()*i,distance.begin()+contact_list.size()*(i+1));
            contact_to_merge.push_back(i);
            if(contact_group_index[i] != -1)
                groups.push_back(uint16_t(contact_group_index[i]));
            dis[i] = std::numeric_limits<float>::max();

            // the nearest contact
            uint32_t j = uint32_t(std::min_element(dis.begin(),dis.end())-dis.begin());
            if(dis[j] > contact_distance_in_mm)
                continue;
            contact_to_merge.push_back(j);
            if(contact_group_index[j] != -1)
                groups.push_back(uint16_t(contact_group_index[j]));
            dis[j] = std::numeric_limits<float>::max();

            // the 2nd nearest contact
            uint32_t k = uint32_t(std::min_element(dis.begin(),dis.end())-dis.begin());
            if(dis[k] <= contact_distance_in_mm)
            {
                contact_to_merge.push_back(k);
                if(contact_group_index[k] != -1)
                    groups.push_back(uint16_t(contact_group_index[k]));
            }

        }

        if(groups.empty())
        {
            groups.push_back(uint16_t(contact_group.size()));
            contact_group.push_back(std::vector<uint16_t>());
        }
        else
        {
            std::sort(groups.begin(),groups.end());
            groups.erase(std::unique(groups.begin(), groups.end() ),groups.end());
        }


        uint16_t group1 = groups.front();

        while(groups.size() > 1)
        {
            // merge groups
            uint16_t group2 = groups.back();
            contact_group[group1].insert(contact_group[group1].end(),contact_group[group2].begin(),contact_group[group2].end());
            for(auto g : contact_group[group2])
                contact_group_index[g] = short(group1);
            contact_group[group2].clear();
            groups.pop_back();
        }

        for(auto contact : contact_to_merge)
            contact_group_index[contact] = short(group1);
        contact_group[group1].insert(contact_group[group1].end(),contact_to_merge.begin(),contact_to_merge.end());

        // sort and remove repeated contact id
        std::sort(contact_group[group1].begin(),contact_group[group1].end());
        contact_group[group1].erase(std::unique(contact_group[group1].begin(), contact_group[group1].end() ), contact_group[group1].end());

    }

    // use eigen analysis to remove false results
    {
        const unsigned int min_contact = 4;
        std::vector<float> length_feature(contact_group.size());
        for(unsigned int i = 0;i < contact_group.size();++i)
        {
            if(contact_group[i].size() < min_contact)
                continue;
            std::vector<tipl::vector<3> > all_points;
            for(unsigned int c = 0;c < contact_group[i].size();++c)
            {
                auto region_id = contact_list[contact_group[i][c]];
                for(unsigned int j = 0;j < regions[region_id].size();++j)
                    all_points.push_back(tipl::vector<3>(tipl::pixel_index<3>(regions[region_id][j],I.shape())));
            }
            auto center = std::accumulate(all_points.begin(),all_points.end(),tipl::vector<3>())/float(all_points.size());

            // get covariance matrix for eigen analysis
            tipl::minus_constant(all_points.begin(),all_points.end(),center);
            std::vector<float> x(all_points.size()),y(all_points.size()),z(all_points.size());
            for(unsigned int j = 0;j < all_points.size();++j)
            {
                x[j] = all_points[j][0];
                y[j] = all_points[j][1];
                z[j] = all_points[j][2];
            }
            tipl::matrix<3,3,float> c,V;
            float d[3] = {0,0,0};
            c[0] = tipl::variance(x);
            c[4] = tipl::variance(y);
            c[8] = tipl::variance(z);
            c[1] = c[3] = float(tipl::covariance(x.begin(),x.end(),y.begin()));
            c[2] = c[6] = float(tipl::covariance(x.begin(),x.end(),z.begin()));
            c[5] = c[7] = float(tipl::covariance(y.begin(),y.end(),z.begin()));
            tipl::mat::eigen_decomposition_sym(c.begin(),V.begin(),d,tipl::dim<3,3>());
            if(d[1] > d[2]*3.0f)
            {
                contact_group[i].clear();
                continue;
            }
            //std::cout << "center=" << center << std::endl;
            //std::cout << "count=" << contact_group[i].size() << std::endl;
            //std::cout << "V=" << tipl::vector<3>(V.begin()) << std::endl;
            //std::cout << "d=" << tipl::vector<3>(d) << std::endl;
            //std::cout << "L=" << std::sqrt(d[0]) << std::endl;
            length_feature[i] = float(std::sqrt(d[0])/contact_group[i].size());
        }
        // use length feature to eliminate
        {
            std::vector<float> features;
            std::copy_if(length_feature.begin(),length_feature.end(),std::back_inserter(features),[](float v){return v > 0.0f;});
            if(features.size() > 6)
            {
                float m = tipl::median(features.begin(),features.end());
                float mad = float(tipl::median_absolute_deviation(features.begin(),features.end(),double(m)));
                float outlier1 = m-3.0f*1.482602218505602f*mad;
                float outlier2 = m+3.0f*1.482602218505602f*mad;
                for(unsigned int i = 0;i < contact_group.size();++i)
                {
                    if(length_feature[i] > 0.0f && (length_feature[i] < outlier1 || length_feature[i] > outlier2))
                        contact_group[i].clear();
                }
            }
        }

        // remove empty group
        for(unsigned int i = 0;i < contact_group.size();)
        {
            if(contact_group[i].size() < min_contact)
            {
                contact_group[i].swap(contact_group.back());
                contact_group.pop_back();
                continue;
            }
            else
                ++i;
        }
    }

    // add devices
    for(unsigned int i = 0;i < contact_group.size();++i)
    {
        std::vector<tipl::vector<3> > contact_pos(contact_group[i].size());
        std::vector<float> contact_2_center(contact_group[i].size());
        for(unsigned int c = 0;c < contact_group[i].size();++c)
        {
            auto region_id = contact_list[contact_group[i][c]];
            std::vector<tipl::vector<3> > points_contact;
            for(unsigned int j = 0;j < regions[region_id].size();++j)
                points_contact.push_back(tipl::vector<3>(tipl::pixel_index<3>(regions[region_id][j],I.shape())));
            contact_pos[c] = std::accumulate(points_contact.begin(),points_contact.end(),tipl::vector<3>())/float(points_contact.size());
            contact_2_center[c] = float((contact_pos[c]-tipl::vector<3>(float(I.width())*0.5f,float(I.height())*0.5f,float(I.depth())*0.5f)).length());
        }

        // prepare device shape
        auto tip_contact = std::min_element(contact_2_center.begin(),contact_2_center.end())-contact_2_center.begin();
        auto tail_contact = std::max_element(contact_2_center.begin(),contact_2_center.end())-contact_2_center.begin();
        auto tip_pos = contact_pos[uint32_t(tip_contact)];
        auto tail_pos = contact_pos[uint32_t(tail_contact)];
        auto dir = tail_pos-tip_pos;
        dir.normalize();
        auto contact_count = contact_group[i].size();
        if(contact_count & 1)
            contact_count++;
        contact_count = uint32_t(std::min(std::max(int(contact_count),8),16));
        auto device_pos = (tip_pos-dir*(1.0f/vs[0]));
        device_pos.to(slice->to_dif);
        tip_pos.to(slice->to_dif);
        tail_pos.to(slice->to_dif);
        auto device_dir = tail_pos-tip_pos;
        device_dir.normalize();

        // check overlap
        bool has_merged = false;
        for(unsigned int j = 0;j < i;++j)
        {
            auto& dev = devices[devices.size()-i+j];
            auto pos_dir = device_pos-dev->pos;
            pos_dir.normalize();
            if(std::abs(pos_dir*device_dir) > 0.98f && (device_dir*dev->dir) > 0.98f) // merge
            {
                // merge device
                if(pos_dir*device_dir < 0.0f)
                    dev->pos = device_pos;
                dev->dir = dev->dir*contact_group[j].size()+device_dir*contact_group[i].size();
                dev->dir.normalize();
                auto new_count = contact_group[i].size()+contact_group[j].size();
                if(new_count & 1)
                    ++new_count;
                dev->type = std::string("SEEG Electrode:") + std::to_string(new_count) + " Contacts";
                // merge groups
                contact_group[j].insert(contact_group[j].end(),contact_group[i].begin(),contact_group[i].end());
                if(i < contact_group.size()-1)
                    contact_group[i].swap(contact_group.back());
                contact_group.pop_back();

                i--;
                has_merged = true;
                break;
            }
        }

        if(has_merged)
            continue;

        // add device
        devices.push_back(std::make_shared<Device>());
        devices.back()->name = std::string("Electrode ") + std::to_string(i+1);
        devices.back()->type = std::string("SEEG Electrode:") + std::to_string(contact_count) + " Contacts";
        devices.back()->pos = device_pos;
        devices.back()->dir = device_dir;
        new_device(devices.back());
    }

    // add contacts as regions
    cur_tracking_window.regionWidget->begin_update();
    std::vector<std::shared_ptr<ROIRegion> > new_regions;
    for(unsigned int i = 0;i < contact_group.size();++i)
    {
        cur_tracking_window.regionWidget->add_region((std::string("Electrode ") + std::to_string(i+1)).c_str());
        new_regions.push_back(cur_tracking_window.regionWidget->regions.back());
    }
    tipl::adaptive_par_for(contact_group.size(),[&](unsigned int i)
    {
        std::vector<tipl::vector<3,float> > voxels;
        for(const auto contact: contact_group[i])
        {
            auto region_id = contact_list[contact];
            for(unsigned int j = 0;j < regions[region_id].size();++j)
            {
                voxels.push_back(tipl::vector<3>(tipl::pixel_index<3>(regions[region_id][j],I.shape())));
                voxels.back().to(slice->to_slice);
            }
        }
        new_regions[i]->add_points(std::move(voxels));
    });
    cur_tracking_window.regionWidget->end_update();


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
