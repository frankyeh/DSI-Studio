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
            if(previous_name == "Device" && tipl::begins_with(device->type,"Scale"))
            {
                device->pos = {cur_tracking_window.handle->dim[2]/2.0f-25.0f/cur_tracking_window.handle->vs[0],
                               float(cur_tracking_window.handle->dim[1]),cur_tracking_window.handle->dim[2]/2.0f};
                device->dir = {1,0,0};
                device->name = "50 mm";
                item(cur_item->row(),2)->setData(Qt::UserRole,uint32_t(device->color));
                item(cur_item->row(),3)->setText(QString::number(double(device->length = 50)));
            }

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
    if(new_device_str.isEmpty())
    {
        QAction* pAction = qobject_cast<QAction*>(sender());
        devices.push_back(std::make_shared<Device>());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0,10.0);
        auto dx = dis(gen);
        auto dy = dis(gen);
        devices.back()->pos = tipl::vector<3>(
                cur_tracking_window.handle->dim[0]/2+dx,
                cur_tracking_window.handle->dim[1]/2+dy,
                cur_tracking_window.handle->dim[2]/2+dis(gen)/4.0);
        devices.back()->dir = tipl::vector<3>(dx,dy,50.0);
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
void DeviceTableWidget::uncheck_all(void)
{
    for(int row = 0;row < rowCount();++row)
    {
        item(row,0)->setCheckState(Qt::Unchecked);
        item(row,0)->setData(Qt::ForegroundRole,QBrush(Qt::gray));
    }
    emit need_update();
}
bool DeviceTableWidget::load_device(const std::string& filename)
{
    bool result = false;
    for(const auto& line : tipl::read_text_file(filename))
    {
        new_device_str = line.c_str();
        newDevice();
        result = true;
    }
    return result;
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
void DeviceTableWidget::save_all_devices(void)
{
    cur_tracking_window.command({std::string("save_all_devices")});
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
                std::vector<float> values;
                tipl::get_window(pos,I,uint32_t(contact_distance_in_mm),values);
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
    auto lead_pos = cur_device->get_lead_positions();

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
