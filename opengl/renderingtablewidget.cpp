#include <QSlider>
#include <QComboBox>
#include <QHeaderView>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFile>
#include <QTextStream>
#include "renderingtablewidget.h"
#include "qcolorcombobox.h"
#include "glwidget.h"
#include "tracking/tracking_window.h"
#include "tracking/region/regiontablewidget.h"
#include "ui_tracking_window.h"
#include <iostream>
#include <cmath>


void RenderingItem::setValue(QVariant new_value)
{
    if(value == new_value)
        return;
    value = new_value;
    if(!GUI)
        return;
    if(QString(GUI->metaObject()->className()) == "QSlider")
    {
        QSlider *slider = reinterpret_cast<QSlider*>(GUI);
        if(slider->maximum() == 10) // int
            slider->setValue(new_value.toInt());
        else
            slider->setValue(int(new_value.toFloat()*5.0f));
    }
    if(QString(GUI->metaObject()->className()) == "QColorToolButton")
    {
        reinterpret_cast<QColorToolButton*>(GUI)->setColor(uint32_t(new_value.toInt()));
    }
    if(QString(GUI->metaObject()->className()) == "QDoubleSpinBox")
    {
        reinterpret_cast<QDoubleSpinBox*>(GUI)->setValue(double(new_value.toFloat()));
    }
    if(QString(GUI->metaObject()->className()) == "QSpinBox")
    {
        reinterpret_cast<QSpinBox*>(GUI)->setValue(new_value.toInt());

    }
    if(QString(GUI->metaObject()->className()) == "QComboBox")
    {
        reinterpret_cast<QComboBox*>(GUI)->setCurrentIndex(new_value.toInt());
    }
}
void RenderingItem::setMinMax(float min,float max,float step)
{
    if(!GUI)
        return;
    if(QString(GUI->metaObject()->className()) == "QDoubleSpinBox")
    {
        reinterpret_cast<QDoubleSpinBox*>(GUI)->setMaximum(double(max));
        reinterpret_cast<QDoubleSpinBox*>(GUI)->setMinimum(double(min));
        reinterpret_cast<QDoubleSpinBox*>(GUI)->setSingleStep(double(step));
    }
}
void RenderingItem::setList(QStringList list)
{
    if(!GUI)
        return;
    if(QString(GUI->metaObject()->className()) == "QComboBox")
    {
        reinterpret_cast<QComboBox*>(GUI)->clear();
        reinterpret_cast<QComboBox*>(GUI)->addItems(list);
    }
}
QString RenderingItem::getListValue(void) const
{
    if(GUI && QString(GUI->metaObject()->className()) == "QComboBox")
        return reinterpret_cast<QComboBox*>(GUI)->currentText();
    return QString();
}

QWidget *RenderingDelegate::createEditor(QWidget *parent,
        const QStyleOptionViewItem &option,
        const QModelIndex &index) const
{
    auto cur_node = reinterpret_cast<RenderingItem*>(index.internalPointer());
    QString string = index.data(Qt::UserRole+1).toString();
    if (string == QString("int"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,10);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        sd->setToolTip(cur_node->hint);
        cur_node->GUI = sd;
        return sd;
    }
    if (string == QString("slider"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,50);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        sd->setToolTip(cur_node->hint);
        cur_node->GUI = sd;
        return sd;
    }

    if (string == QString("color"))
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        sd->setToolTip(cur_node->hint);
        cur_node->GUI = sd;
        return sd;
    }
    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() >= 1)
    {
        if(string_list[0] == QString("float"))
        {
            QDoubleSpinBox* dsb = new QDoubleSpinBox(parent);
            dsb->setMinimum(string_list[1].toDouble());
            dsb->setMaximum(string_list[2].toDouble());
            if(string_list.size() > 3)
                dsb->setSingleStep(string_list[3].toDouble());
            else
                dsb->setSingleStep((dsb->maximum()-dsb->minimum())/10);
            if(string_list.size() > 4)
                dsb->setDecimals(string_list[4].toInt());
            else
                dsb->setDecimals(std::max<int>(0,4-int(std::log10(dsb->maximum()))));
            connect(dsb, SIGNAL(valueChanged(double)), this, SLOT(emitCommitData()));
            dsb->setMaximumWidth(100);
            dsb->setToolTip(cur_node->hint);
            cur_node->GUI = dsb;
            return dsb;
        }
        if(string_list[0] == QString("int"))
        {
            QSpinBox* dsb = new QSpinBox(parent);
            dsb->setMinimum(string_list[1].toInt());
            dsb->setMaximum(string_list[2].toInt());
            if(string_list.size() > 3)
                dsb->setSingleStep(string_list[3].toInt());
            else
                dsb->setSingleStep(std::max<int>(1,(dsb->maximum()-dsb->minimum())/10));
            dsb->setMaximumWidth(100);
            connect(dsb, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
            dsb->setToolTip(cur_node->hint);
            cur_node->GUI = dsb;
            return dsb;
        }
        {
            QComboBox* cb = new QComboBox(parent);
            cb->addItems(string_list);
            cb->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
            cb->setMaximumWidth(100);
            cb->setFocusPolicy(Qt::WheelFocus);
            connect(cb, SIGNAL(currentIndexChanged(int)), this, SLOT(emitCommitData()));
            cb->setToolTip(cur_node->hint);
            cur_node->GUI = cb;
            return cb;
        }
    }

    return QItemDelegate::createEditor(parent,option,index);

}

void RenderingDelegate::setEditorData(QWidget *editor,
                                      const QModelIndex &index) const
{
    QString string = index.data(Qt::UserRole+1).toString();
    if (string == QString("int"))
    {
        reinterpret_cast<QSlider*>(editor)->setValue(index.data(Qt::UserRole).toInt());
        return;
    }
    if (string == QString("slider"))
    {
        reinterpret_cast<QSlider*>(editor)->setValue(int(index.data(Qt::UserRole).toFloat()*5.0f));
        return;
    }
    if (string == QString("color"))
    {
        reinterpret_cast<QColorToolButton*>(editor)->setColor(uint32_t(index.data(Qt::UserRole).toInt()));
        return;
    }
    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() > 1)
    {
        if(string_list[0] == QString("float"))
            reinterpret_cast<QDoubleSpinBox*>(editor)->setValue(index.data(Qt::UserRole).toDouble());
        else
            if(string_list[0] == QString("int"))
                reinterpret_cast<QSpinBox*>(editor)->setValue(index.data(Qt::UserRole).toInt());
            else
                reinterpret_cast<QComboBox*>(editor)->setCurrentIndex(index.data(Qt::UserRole).toInt());
        return;
    }

    QItemDelegate::setEditorData(editor,index);
}

void RenderingDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                     const QModelIndex &index) const
{
    if(index.column() == 0)
    {
        QItemDelegate::setModelData(editor,model,index);
        return;
    }
    QString string = index.data(Qt::UserRole+1).toString();
    if (string == QString("int"))
    {
        model->setData(index,reinterpret_cast<QSlider*>(editor)->value(),Qt::UserRole);
        return;
    }
    if (string == QString("slider"))
    {
        model->setData(index,reinterpret_cast<QSlider*>(editor)->value()/5.0,Qt::UserRole);
        return;
    }

    if (string == QString("color"))
    {
        model->setData(index,int(reinterpret_cast<QColorToolButton*>(editor)->color().rgb()),Qt::UserRole);
        return;
    }

    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() > 1)
    {
        if(string_list[0] == QString("float"))
            model->setData(index,reinterpret_cast<QDoubleSpinBox*>(editor)->value(),Qt::UserRole);
        else
            if(string_list[0] == QString("int"))
                model->setData(index,reinterpret_cast<QSpinBox*>(editor)->value(),Qt::UserRole);
            else
                model->setData(index,reinterpret_cast<QComboBox*>(editor)->currentIndex(),Qt::UserRole);
        return;
    }
    QItemDelegate::setModelData(editor,model,index);
}

void RenderingDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}

//---------------------------------

TreeModel::TreeModel(RenderingTableWidget *parent)
        : QAbstractItemModel(parent)
{
    root.reset(new RenderingItem("Objects","","root",0,nullptr));
    root_mapping["Root"] = root.get();
    root_mapping["Tracking"] = new RenderingItem("Tracking Parameters","","Tracking",0,root.get());
    root_mapping["ROI"] = new RenderingItem("Region Window","","ROI",0,root.get());
    root_mapping["Rendering"] = new RenderingItem("Background Rendering","","Rendering",0,root.get());
    root_mapping["Slice"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_slice","Slice Rendering",QString("check"),Qt::Checked).internalPointer());
    root_mapping["Tract"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_tract","Tract Rendering",QString("check"),Qt::Checked).internalPointer());
    root_mapping["Region"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_region","Region Rendering",QString("check"),Qt::Checked).internalPointer());
    root_mapping["Surface"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_surface","Surface Rendering",QString("check"),Qt::Checked).internalPointer());
    root_mapping["Device"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_device","Device Rendering",QString("check"),Qt::Checked).internalPointer());
    root_mapping["Label"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_label","Label Rendering",QString("check"),Qt::Unchecked).internalPointer());
    root_mapping["ODF"] = reinterpret_cast<RenderingItem*>(addItem("Root","show_odf","ODF Rendering",QString("check"),Qt::Unchecked).internalPointer());

}

void TreeModel::saveParameters(void)
{
    QSettings settings;
    settings.beginGroup("Rendering Options");
    for(auto& each : name_data_mapping)
        settings.setValue(each.first,each.second->getValue());
    settings.endGroup();
}
TreeModel::~TreeModel()
{
    if(memorize_parameters)
        saveParameters();
}

int TreeModel::columnCount(const QModelIndex &) const
{
    return 2;
}

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();
    auto cur_node = reinterpret_cast<RenderingItem*>(index.internalPointer());
    if (index.column() == 0) // title column
    {
        if(role == Qt::CheckStateRole &&
            cur_node->type == QString("check"))
            return cur_node->value;
        if(role == Qt::DisplayRole)
            return cur_node->title;
    }
    else // editor column
    {
        if (role == Qt::UserRole)
            return cur_node->value;
        if (role == Qt::UserRole+1)
            return cur_node->type;
    }

    if (role == Qt::SizeHintRole)
        return QSize(250,24);
    return QVariant();
}

bool TreeModel::setData ( const QModelIndex & index, const QVariant & value, int role)
{
    if (!index.isValid())
        return false;
    auto cur_node = reinterpret_cast<RenderingItem*>(index.internalPointer());
    if(index.column() == 0) // title column
    {
        if (role == Qt::DisplayRole)
            cur_node->title = value;
        if (role == Qt::CheckStateRole &&
            cur_node->type == QString("check"))
        {
            QVariant old_value = cur_node->value;
            cur_node->value = value;
            if(old_value != value)
                emit dataChanged(index,index);
        }
        return true;
    }
    else// editor column
    {
        switch(role)
        {
        case Qt::UserRole:
            {
                QVariant old_value = cur_node->value;
                cur_node->value = value;
                if(old_value != value)
                    emit dataChanged(index,index);
            }
            return true;
        case Qt::UserRole+1:
            cur_node->type = value;
            return true;
        case Qt::DisplayRole:
            return true;
        }
    }
    return false;
}

Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;
    auto cur_node = reinterpret_cast<RenderingItem*>(index.internalPointer());
    if (index.column() >= 1 && !cur_node->type.isNull())
        return Qt::ItemIsEnabled | Qt::ItemIsEditable;
    else
    if(cur_node->type == QString("check"))
    {
        return Qt::ItemIsUserCheckable | Qt::ItemIsEnabled;
    }
    else
        return Qt::ItemIsEnabled;
}

QVariant TreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return (section) ? root->title : root->type;
    return QVariant();
}

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent)
const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    RenderingItem *parentItem;

    if (!parent.isValid())
        parentItem = root.get();
    else
        parentItem = static_cast<RenderingItem*>(parent.internalPointer());

    RenderingItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex TreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    RenderingItem *childItem = reinterpret_cast<RenderingItem*>(index.internalPointer());
    RenderingItem *parentItem = childItem->parent();

    if (parentItem == root.get())
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    RenderingItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = root.get();
    else
        parentItem = static_cast<RenderingItem*>(parent.internalPointer());

    return parentItem->childCount();
}
void TreeModel::addNode(QString root_name,QString id,QVariant title)
{
    root_mapping[id] = new RenderingItem(title,QVariant(),id,0,root_mapping[root_name]);
}
QModelIndex TreeModel::addItem(QString root_name,QString id,QVariant title, QVariant type, QVariant value,QString hint)
{
    if(!name_data_mapping[id])
    {
        QSettings settings;
        settings.beginGroup("Rendering Options");
        auto item = new RenderingItem(title,type,id,settings.value(id,value),root_mapping[root_name]);
        item->hint = hint;
        item->def_value = value;
        name_data_mapping[id] = item;
        settings.endGroup();
    }
    else
        std::cout << "Duplicated item name in rending option" << std::endl;
    return createIndex(root_mapping[root_name]->childCount()-1,1,name_data_mapping[id]);
}

void TreeModel::setDefault(QString parent_id)
{
    for(auto& each : name_data_mapping)
        if(each.second->parent() && each.second->parent()->id == parent_id)
            each.second->setValue(each.second->def_value);
}

RenderingTableWidget::RenderingTableWidget(tracking_window& cur_tracking_window_,QWidget *parent) :
        QTreeView(parent),cur_tracking_window(cur_tracking_window_)
{
    setItemDelegateForColumn(1,data_delegate = new RenderingDelegate(this));
    setAlternatingRowColors(true);

    setModel(treemodel = new TreeModel(this));
    connect(treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),this,SLOT(dataChanged(QModelIndex,QModelIndex)));
    initialize();

    header()->setSectionResizeMode(0, QHeaderView::Stretch);
    header()->setSectionResizeMode(1, QHeaderView::Fixed);
    header()->setStretchLastSection(false);
    header()->resizeSection(0, 150);
    header()->resizeSection(1, 100);
    header()->hide();
    expandAll();
    collapseAll();

    tract_update_list = {"tract_alpha","tract_style",
                         "tract_color_saturation","tract_color_brightness",
                         "tract_color_style", "tract_color_metrics", "tract_color_map",
                         "tract_color_max","tract_color_min",
                         "tract_color_max_value","tract_color_min_value",
                         "tube_diameter", "tract_tube_detail","tract_shader",
                         "end_point_shift"};
    tract_color_map_update_list = {
                         "tract_color_max","tract_color_min","tract_color_map"};
    region_color_map_update_list = {
                         "region_color_max","region_color_min","region_color_map"};
}

void RenderingTableWidget::initialize(void)
{
    // Environment
    QFile data(":/data/options.txt");
    if (!data.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream in(&data);
    while (!in.atEnd())
    {
        QStringList list = in.readLine().split('/');
        if(list.size() == 3) // tree node
        {
            treemodel->addNode(list[0],list[2],list[1]);
            continue;
        }
        if(list.size() < 5)
            continue;
        QStringList value_list = list[3].split(':');
        QModelIndex index;
        if(value_list.size() == 1)
            index = treemodel->addItem(list[0],list[2],list[1],list[3],list[4].toDouble(),list.size() == 6 ? list[5]:QString());
        else
            index = treemodel->addItem(list[0],list[2],list[1],value_list,list[4].toDouble(),list.size() == 6 ? list[5]:QString());
        openPersistentEditor(index);
    }
}
void RenderingTableWidget::setDefault(QString parent_id)
{
    treemodel->setDefault(parent_id);
}
void RenderingTableWidget::dataChanged(const QModelIndex &, const QModelIndex &bottomRight)
{
    auto cur_node = reinterpret_cast<RenderingItem*>(bottomRight.internalPointer());
    if(tract_color_map_update_list.find(cur_node->id.toStdString()) != tract_color_map_update_list.end())
        cur_tracking_window.tractWidget->update_color_map();
    if(region_color_map_update_list.find(cur_node->id.toStdString()) != region_color_map_update_list.end())
        cur_tracking_window.regionWidget->update_color_map();

    if(tract_update_list.find(cur_node->id.toStdString()) != tract_update_list.end())
        cur_tracking_window.tractWidget->need_update_all();


    if(cur_node->id == "tracking_index")
    {
        cur_tracking_window.on_tracking_index_currentIndexChanged(cur_node->value.toInt());
        return;
    }
    if(cur_node->id == "dt_index1" || cur_node->id == "dt_index2")
    {
        if(cur_node->id == "dt_index1" && getData("dt_index1").toInt() > 0)
        {
            auto dt_name1 = cur_tracking_window.dt_list[getData("dt_index1").toInt()].toStdString();
            // search comparing metrics by inclusion
            for(size_t i = 1;i < cur_tracking_window.dt_list.size();++i)
            {
                auto name = cur_tracking_window.dt_list[i].toStdString();
                if(name != dt_name1 && (dt_name1.find(name) != std::string::npos ||
                                        name.find(dt_name1) != std::string::npos))
                {
                    setData("dt_index2",int(i));
                    break;
                }
            }
            // search comparing metrics by common prefix
            for(size_t i = 1;i < cur_tracking_window.dt_list.size();++i)
            {
                auto name = cur_tracking_window.dt_list[i].toStdString();
                if(name != dt_name1 && dt_name1.substr(0,2) == name.substr(0,2))
                {
                    setData("dt_index2",int(i));
                    break;
                }
            }
        }
        setData("max_seed_count",1000000);
        setData("max_tract_count",0);
        setData("check_ending",0); // no check ending
        cur_tracking_window.handle->dir.dt_fa.clear(); // avoid slice showing previous dt
        cur_tracking_window.slice_need_update = true;
        return;
    }
    if(cur_node->id == "roi_zoom")
    {
        cur_tracking_window.set_roi_zoom(cur_node->value.toInt());
        cur_tracking_window.slice_need_update = true;
        return;
    }
    if(cur_node->id == "roi_position")
    {
        cur_tracking_window.ui->roi_position->setChecked(cur_node->value.toBool());
        return;
    }
    if(cur_node->id == "roi_ruler")
    {
        cur_tracking_window.ui->roi_ruler->setChecked(cur_node->value.toBool());
        return;
    }
    if(cur_node->id == "roi_label")
    {
        cur_tracking_window.ui->roi_label->setChecked(cur_node->value.toBool());
        return;
    }
    if(cur_node->id == "roi_fiber")
    {
        cur_tracking_window.ui->roi_fiber->setChecked(cur_node->value.toBool());
        return;
    }
    if(cur_node->id == "tract_color_metrics" || cur_node->id == "region_color_metrics")
    {
        size_t item_index = cur_node->value.toInt();
        float min_v = 0.0f,max_v = 1.0f;
        if(item_index < cur_tracking_window.handle->slices.size() &&
           !cur_tracking_window.handle->slices[item_index]->optional())
        {
            cur_tracking_window.handle->slices[item_index]->get_minmax();
            min_v = cur_tracking_window.handle->slices[item_index]->min_value;
            max_v = cur_tracking_window.handle->slices[item_index]->max_value;
        }
        if(cur_node->id == "tract_color_metrics")
        {
            setMinMax("tract_color_min_value",min_v,max_v,(max_v-min_v)/20);
            setMinMax("tract_color_max_value",min_v,max_v,(max_v-min_v)/20);
            setData("tract_color_min_value",min_v);
            setData("tract_color_max_value",min_v+(max_v-min_v)/2);
        }
        else
        {
            setMinMax("region_color_min_value",min_v,max_v,(max_v-min_v)/20);
            setMinMax("region_color_max_value",min_v,max_v,(max_v-min_v)/20);
            setData("region_color_min_value",min_v);
            setData("region_color_max_value",min_v+(max_v-min_v)/2);
            cur_tracking_window.regionWidget->color_map_values.clear();
        }
    }

    if(cur_node->id == "fa_threshold" ||
       cur_node->id == "dt_threshold" ||
            cur_node->parent()->id == QString("ROI"))
    {
        cur_tracking_window.slice_need_update = true;
        return;
    }

    cur_tracking_window.glWidget->update();
}
