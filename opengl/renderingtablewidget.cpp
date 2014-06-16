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
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        return sd;
    }
    if (string == QString("slider"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,50);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        return sd;
    }

    if (string == QString("color"))
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        return sd;
    }
    QStringList string_list = index.data(Qt::UserRole+1).toStringList();
    if (string_list.size() > 1)
    {
        if(string_list[0] == QString("float"))
        {
            QDoubleSpinBox* dsb = new QDoubleSpinBox(parent);
            dsb->setMinimum(string_list[1].toDouble());
            dsb->setMaximum(string_list[2].toDouble());
            dsb->setSingleStep((dsb->maximum()-dsb->minimum())/20);
            dsb->setDecimals(std::max<double>((double)0,2-std::log10(dsb->maximum())));
            connect(dsb, SIGNAL(valueChanged(double)), this, SLOT(emitCommitData()));
            return dsb;
        }
        if(string_list[0] == QString("int"))
        {
            QSpinBox* dsb = new QSpinBox(parent);
            dsb->setMinimum(string_list[1].toInt());
            dsb->setMaximum(string_list[2].toInt());
            dsb->setSingleStep((dsb->maximum()-dsb->minimum())/20);
            connect(dsb, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
            return dsb;
        }
        {
            QComboBox* cb = new QComboBox(parent);
            cb->addItems(string_list);
            connect(cb, SIGNAL(currentIndexChanged(int)), this, SLOT(emitCommitData()));
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
    root.reset(new RenderingItem(QString("Objects"),QString("Options"),0,0));
    root_mapping["Root"] = root.get();
    root_mapping["Tracking"] = new RenderingItem(QString("Tracking"),QVariant(),0,root.get());
    root_mapping["Rendering"] = new RenderingItem(QString("Rendering"),QVariant(),0,root.get());
    root_mapping["Slice"] = (RenderingItem*)addItem("Root","show_slice","Slice",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Tract"] = (RenderingItem*)addItem("Root","show_tract","Tract",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Region"] = (RenderingItem*)addItem("Root","show_region","Region",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Surface"] = (RenderingItem*)addItem("Root","show_surface","Surface",QString("check"),Qt::Checked).internalPointer();
    root_mapping["ODF"] = (RenderingItem*)addItem("Root","show_odf","ODF",QString("check"),Qt::Checked).internalPointer();
    root_mapping["Others"] = new RenderingItem(QString("Others"),QVariant(),0,root.get());
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

int TreeModel::columnCount(const QModelIndex &parent) const
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
            ((RenderingItem*)index.internalPointer())->value = value;
            emit dataChanged(index,index);
        }
        return true;
    }
    else
    // column = 1
        switch(role)
        {
        case Qt::UserRole:
            ((RenderingItem*)index.internalPointer())->value = value;
            emit dataChanged(index,index);
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
    RenderingItem* item = 0;
    if(!name_data_mapping[id])
    {
        QSettings settings;
        settings.beginGroup("Rendering Options");
        name_data_mapping[id] = item = new RenderingItem(title,type,settings.value(id,value),root_mapping[root_name]);
        name_default_values[id] = value;
        settings.endGroup();
    }
    else
        std::cout << "Duplicated item name in rending option" << std::endl;
    return createIndex(root_mapping[root_name]->childCount()-1,1,name_data_mapping[id]);
}

void TreeModel::setDefault(void)
{
    std::map<QString,RenderingItem*>::iterator iter = name_data_mapping.begin();
    std::map<QString,RenderingItem*>::iterator end = name_data_mapping.end();
    std::map<QString,QVariant>::iterator iter2 = name_default_values.begin();
    for(;iter != end;++iter,++iter2)
        iter->second->value = iter2->second;

}

RenderingTableWidget::RenderingTableWidget(tracking_window& cur_tracking_window_,QWidget *parent) :
        QTreeView(parent),cur_tracking_window(cur_tracking_window_)
{
    setItemDelegateForColumn(1,data_delegate = new RenderingDelegate(this));

    setModel(treemodel = new TreeModel(this));
    initialize();

    header()->setResizeMode(0, QHeaderView::Stretch);
    header()->setResizeMode(1, QHeaderView::Stretch);
    header()->resizeSection(0, 150);
    header()->resizeSection(1, 150);
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
void RenderingTableWidget::updateData(const char* name,QVariant data)
{
    treemodel->updateData(name,data);
}

void RenderingTableWidget::setData(const char* name,QVariant data)
{
    collapseAll();
    treemodel->updateData(name,data);
    delete treemodel;
    setModel(treemodel = new TreeModel(this));
    initialize();
    expandAll();
    collapseAll();
    connect(treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),
            this,SLOT(dataChanged(QModelIndex,QModelIndex)));
}

void RenderingTableWidget::setDefault(void)
{
    collapseAll();
    treemodel->setDefault();
    delete treemodel;
    setModel(treemodel = new TreeModel(this));
    initialize();
    expandAll();
    collapseAll();
    connect(treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),
            this,SLOT(dataChanged(QModelIndex,QModelIndex)));

}
void RenderingTableWidget::dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
    if(((RenderingItem*)bottomRight.internalPointer())->parent()->title == "Tracking")
    {
        //std::cout << "Tracking parameter changed" << std::endl;
    }
    else
        cur_tracking_window.glWidget->updateGL();
}
