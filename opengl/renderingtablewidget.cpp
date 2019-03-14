#include <QSlider>
#include <QComboBox>
#include <QHeaderView>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFile>
#include <QTextStream>
#include "renderingtablewidget.h"
#include "qcolorcombobox.h"
#include "tracking/tracking_window.h"
#include "glwidget.h"
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
        QSlider *slider = (QSlider*)GUI;
        if(slider->maximum() == 10) // int
            slider->setValue(new_value.toInt());
        else
            slider->setValue(new_value.toFloat()*5.0);
    }
    if(QString(GUI->metaObject()->className()) == "QColorToolButton")
    {
        ((QColorToolButton*)GUI)->setColor(new_value.toInt());
    }
    if(QString(GUI->metaObject()->className()) == "QDoubleSpinBox")
    {
        ((QDoubleSpinBox*)GUI)->setValue(new_value.toFloat());
    }
    if(QString(GUI->metaObject()->className()) == "QSpinBox")
    {
        ((QSpinBox*)GUI)->setValue(new_value.toInt());

    }
    if(QString(GUI->metaObject()->className()) == "QComboBox")
    {
        ((QComboBox*)GUI)->setCurrentIndex(new_value.toInt());
    }
}
void RenderingItem::setMinMax(float min,float max,float step)
{
    if(!GUI)
        return;
    if(QString(GUI->metaObject()->className()) == "QDoubleSpinBox")
    {
        ((QDoubleSpinBox*)GUI)->setMaximum(max);
        ((QDoubleSpinBox*)GUI)->setMinimum(min);
        ((QDoubleSpinBox*)GUI)->setSingleStep(step);
    }
}
void RenderingItem::setList(QStringList list)
{
    if(!GUI)
        return;
    if(QString(GUI->metaObject()->className()) == "QComboBox")
    {
        ((QComboBox*)GUI)->clear();
        ((QComboBox*)GUI)->addItems(list);
    }
}

QWidget *RenderingDelegate::createEditor(QWidget *parent,
        const QStyleOptionViewItem &option,
        const QModelIndex &index) const
{
    QString string = index.data(Qt::UserRole+1).toString();
    if (string == QString("int"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,10);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        ((RenderingItem*)index.internalPointer())->GUI = sd;
        return sd;
    }
    if (string == QString("slider"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,50);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        ((RenderingItem*)index.internalPointer())->GUI = sd;
        return sd;
    }

    if (string == QString("color"))
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        sd->setMaximumWidth(100);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        ((RenderingItem*)index.internalPointer())->GUI = sd;
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
                dsb->setDecimals(string_list[4].toDouble());
            else
                dsb->setDecimals(std::max<double>((double)0,4-std::log10(dsb->maximum())));
            connect(dsb, SIGNAL(valueChanged(double)), this, SLOT(emitCommitData()));
            dsb->setMaximumWidth(100);
            ((RenderingItem*)index.internalPointer())->GUI = dsb;
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
            ((RenderingItem*)index.internalPointer())->GUI = dsb;
            return dsb;
        }
        {
            QComboBox* cb = new QComboBox(parent);
            cb->addItems(string_list);
            cb->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
            cb->setMaximumWidth(100);
            cb->setFocusPolicy(Qt::WheelFocus);
            connect(cb, SIGNAL(currentIndexChanged(int)), this, SLOT(emitCommitData()));
            ((RenderingItem*)index.internalPointer())->GUI = cb;
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
        ((QSlider*)editor)->setValue(index.data(Qt::UserRole).toInt());
        return;
    }
    if (string == QString("slider"))
    {
        ((QSlider*)editor)->setValue(index.data(Qt::UserRole).toFloat()*5.0);
        return;
    }
    if (string == QString("color"))
    {
        ((QColorToolButton*)editor)->setColor(index.data(Qt::UserRole).toInt());
        return;
    }
    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() > 1)
    {
        if(string_list[0] == QString("float"))
            ((QDoubleSpinBox*)editor)->setValue(index.data(Qt::UserRole).toFloat());
        else
            if(string_list[0] == QString("int"))
                ((QSpinBox*)editor)->setValue(index.data(Qt::UserRole).toInt());
            else
                ((QComboBox*)editor)->setCurrentIndex(index.data(Qt::UserRole).toInt());
        return;
    }

    QItemDelegate::setEditorData(editor,index);
}

void RenderingDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                     const QModelIndex &index) const
{
    if(index.column() != 1)
    {
        QItemDelegate::setModelData(editor,model,index);
        return;
    }
    QString string = index.data(Qt::UserRole+1).toString();
    if (string == QString("int"))
    {
        model->setData(index,((QSlider*)editor)->value(),Qt::UserRole);
        return;
    }
    if (string == QString("slider"))
    {
        model->setData(index,((QSlider*)editor)->value()/5.0,Qt::UserRole);
        return;
    }

    if (string == QString("color"))
    {
        model->setData(index,(int)(((QColorToolButton*)editor)->color().rgb()),Qt::UserRole);
        return;
    }

    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() > 1)
    {
        if(string_list[0] == QString("float"))
            model->setData(index,((QDoubleSpinBox*)editor)->value(),Qt::UserRole);
        else
            if(string_list[0] == QString("int"))
                model->setData(index,((QSpinBox*)editor)->value(),Qt::UserRole);
            else
                model->setData(index,((QComboBox*)editor)->currentIndex(),Qt::UserRole);
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
    root.reset(new RenderingItem("Objects","","root",0,0));
    root_mapping["Root"] = root.get();
    root_mapping["Tracking"] = new RenderingItem("Tracking Parameters","","Tracking",0,root.get());
    root_mapping["ROI"] = new RenderingItem("Region Window","","ROI",0,root.get());
    root_mapping["Rendering"] = new RenderingItem("Background Rendering","","Rendering",0,root.get());
    root_mapping["Slice"] = (RenderingItem*)addItem("Root","show_slice","Slice Rendering",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Tract"] = (RenderingItem*)addItem("Root","show_tract","Tract Rendering",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Region"] = (RenderingItem*)addItem("Root","show_region","Region Rendering",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Surface"] = (RenderingItem*)addItem("Root","show_surface","Surface Rendering",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Label"] = (RenderingItem*)addItem("Root","show_label","Label Rendering",QString("check"),Qt::Checked).internalPointer();
    root_mapping["ODF"] = (RenderingItem*)addItem("Root","show_odf","ODF Rendering",QString("check"),Qt::Checked).internalPointer();

}

TreeModel::~TreeModel()
{
    std::map<QString,RenderingItem*>::const_iterator iter = name_data_mapping.begin();
    std::map<QString,RenderingItem*>::const_iterator end = name_data_mapping.end();
    QSettings settings;
    settings.beginGroup("Rendering Options");
    for(;iter != end;++iter)
        settings.setValue(iter->first,iter->second->getValue());
    settings.endGroup();
}

int TreeModel::columnCount(const QModelIndex &) const
{
    return 2;
}

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (index.column() == 0 && role == Qt::CheckStateRole &&
        ((RenderingItem*)index.internalPointer())->type == QString("check"))
        return ((RenderingItem*)index.internalPointer())->value;

    if (index.column() == 0 && role == Qt::DisplayRole)
        return ((RenderingItem*)index.internalPointer())->title;

    if (index.column() == 1 && role == Qt::UserRole)
        return ((RenderingItem*)index.internalPointer())->value;

    if (index.column() == 1 && role == Qt::UserRole+1)
        return ((RenderingItem*)index.internalPointer())->type;

    if (role == Qt::SizeHintRole)
        return QSize(250,24);
    return QVariant();
}

bool TreeModel::setData ( const QModelIndex & index, const QVariant & value, int role)
{
    if (!index.isValid())
        return false;
    if(index.column() == 0)
    {
        if (role == Qt::DisplayRole)
            ((RenderingItem*)index.internalPointer())->title = value;
        if (role == Qt::CheckStateRole &&
            ((RenderingItem*)index.internalPointer())->type == QString("check"))
        {
            QVariant old_value = ((RenderingItem*)index.internalPointer())->value;
            ((RenderingItem*)index.internalPointer())->value = value;
            if(old_value != value)
                emit dataChanged(index,index);
        }
        return true;
    }
    else
    // column = 1
        switch(role)
        {
        case Qt::UserRole:
            {
                QVariant old_value = ((RenderingItem*)index.internalPointer())->value;
                ((RenderingItem*)index.internalPointer())->value = value;
                if(old_value != value)
                    emit dataChanged(index,index);
            }
            return true;
        case Qt::UserRole+1:
            ((RenderingItem*)index.internalPointer())->type = value;
            return true;
        case Qt::DisplayRole:
            return true;
        }
    return false;
}

Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;
    if (index.column() == 1 && !((RenderingItem*)index.internalPointer())->type.isNull())
        return Qt::ItemIsEnabled | Qt::ItemIsEditable;
    else
    if(((RenderingItem*)index.internalPointer())->type == QString("check"))
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
        (section) ? root->title : root->type;
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

    RenderingItem *childItem = static_cast<RenderingItem*>(index.internalPointer());
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

QModelIndex TreeModel::addItem(QString root_name,QString id,QVariant title, QVariant type, QVariant value)
{
    if(!name_data_mapping[id])
    {
        QSettings settings;
        settings.beginGroup("Rendering Options");
        name_data_mapping[id] = new RenderingItem(title,type,id,settings.value(id,value),root_mapping[root_name]);
        name_default_values[id] = value;
        settings.endGroup();
    }
    else
        std::cout << "Duplicated item name in rending option" << std::endl;
    return createIndex(root_mapping[root_name]->childCount()-1,1,name_data_mapping[id]);
}

void TreeModel::setDefault(QString parent_id)
{
    std::map<QString,RenderingItem*>::iterator iter = name_data_mapping.begin();
    std::map<QString,RenderingItem*>::iterator end = name_data_mapping.end();
    std::map<QString,QVariant>::iterator iter2 = name_default_values.begin();
    for(;iter != end;++iter,++iter2)
        if(iter->second->parent() && iter->second->parent()->id == parent_id)
        iter->second->setValue(iter2->second);

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
        if(list.size() != 5)
            continue;
        QStringList value_list = list[3].split(':');
        QModelIndex index;
        if(value_list.size() == 1)
            index = treemodel->addItem(list[0],list[2],list[1],list[3],list[4].toDouble());
        else
            index = treemodel->addItem(list[0],list[2],list[1],value_list,list[4].toDouble());
        openPersistentEditor(index);
    }
}
void RenderingTableWidget::setDefault(QString parent_id)
{
    treemodel->setDefault(parent_id);
}
void RenderingTableWidget::dataChanged(const QModelIndex &, const QModelIndex &bottomRight)
{
    if(((RenderingItem*)bottomRight.internalPointer())->id == "tracking_index")
    {
        cur_tracking_window.on_tracking_index_currentIndexChanged(((RenderingItem*)bottomRight.internalPointer())->value.toInt());
        return;
    }
    if(((RenderingItem*)bottomRight.internalPointer())->id == "dt_index")
    {
        cur_tracking_window.on_dt_index_currentIndexChanged(((RenderingItem*)bottomRight.internalPointer())->value.toInt());
        return;
    }
    if(((RenderingItem*)bottomRight.internalPointer())->id == "roi_position")
    {
        cur_tracking_window.on_show_position_toggled(((RenderingItem*)bottomRight.internalPointer())->value.toBool());
        return;
    }
    if(((RenderingItem*)bottomRight.internalPointer())->id == "roi_label")
    {
        cur_tracking_window.on_show_r_toggled(((RenderingItem*)bottomRight.internalPointer())->value.toBool());
        return;
    }
    if(((RenderingItem*)bottomRight.internalPointer())->id == "roi_fiber")
    {
        cur_tracking_window.on_show_fiber_toggled(((RenderingItem*)bottomRight.internalPointer())->value.toBool());
        return;
    }

    if(((RenderingItem*)bottomRight.internalPointer())->id == "fa_threshold" ||
       ((RenderingItem*)bottomRight.internalPointer())->id == "dt_threshold" ||
            ((RenderingItem*)bottomRight.internalPointer())->parent()->id == QString("ROI"))
    {
        cur_tracking_window.scene.show_slice();
        return;
    }
    else
        cur_tracking_window.glWidget->updateGL();
}
