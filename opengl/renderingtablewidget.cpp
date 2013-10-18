#include <QSlider>
#include <QComboBox>
#include <QHeaderView>
#include "renderingtablewidget.h"
#include "qcolorcombobox.h"
#include "tracking/tracking_window.h"
#include "glwidget.h"
#include <iostream>


QWidget *RenderingDelegate::createEditor(QWidget *parent,
        const QStyleOptionViewItem &option,
        const QModelIndex &index) const
{
    if (index.data(Qt::UserRole+1).toString() == QString("int"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,10);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        return sd;
    }
    if (index.data(Qt::UserRole+1).toString() == QString("float"))
    {
        QSlider* sd = new QSlider(parent);
        sd->setOrientation(Qt::Horizontal);
        sd->setRange(0,50);
        connect(sd, SIGNAL(valueChanged(int)), this, SLOT(emitCommitData()));
        return sd;
    }

    if (index.data(Qt::UserRole+1).toString() == QString("color"))
    {
        QColorToolButton* sd = new QColorToolButton(parent);
        connect(sd, SIGNAL(clicked()), this, SLOT(emitCommitData()));
        return sd;
    }

    if (index.data(Qt::UserRole+1).toStringList().size() > 1)
    {
        QComboBox* cb = new QComboBox(parent);
        cb->addItems(index.data(Qt::UserRole+1).toStringList());
        connect(cb, SIGNAL(currentIndexChanged(int)), this, SLOT(emitCommitData()));
        return cb;
    }

    return QItemDelegate::createEditor(parent,option,index);

}

void RenderingDelegate::setEditorData(QWidget *editor,
                                      const QModelIndex &index) const
{

    if (index.data(Qt::UserRole+1).toString() == QString("int"))
    {
        ((QSlider*)editor)->setValue(index.data(Qt::UserRole).toInt());
        return;
    }
    if (index.data(Qt::UserRole+1).toString() == QString("float"))
    {
        ((QSlider*)editor)->setValue(index.data(Qt::UserRole).toFloat()*5.0);
        return;
    }
    if (index.data(Qt::UserRole+1).toString() == QString("color"))
    {
        ((QColorToolButton*)editor)->setColor(index.data(Qt::UserRole).toInt());
        return;
    }
    if (index.data(Qt::UserRole+1).toStringList().size() > 1)
    {
        ((QComboBox*)editor)->setCurrentIndex(index.data(Qt::UserRole).toInt());
        return;
    }

    QItemDelegate::setEditorData(editor,index);
}

void RenderingDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                     const QModelIndex &index) const
{
    if (index.column() == 1 &&
            index.data(Qt::UserRole+1).toString() == QString("int"))
    {
        model->setData(index,((QSlider*)editor)->value(),Qt::UserRole);
        return;
    }
    if (index.column() == 1 &&
            index.data(Qt::UserRole+1).toString() == QString("float"))
    {
        model->setData(index,((QSlider*)editor)->value()/5.0,Qt::UserRole);
        return;
    }

    if (index.column() == 1 &&
            index.data(Qt::UserRole+1).toString() == QString("color"))
    {
        model->setData(index,(int)(((QColorToolButton*)editor)->color().rgb()),Qt::UserRole);
        return;
    }

    if (index.column() == 1 &&
            index.data(Qt::UserRole+1).toStringList().size() > 1)
    {
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

TreeModel::TreeModel(RenderingTableWidget *parent,bool has_odf)
        : QAbstractItemModel(parent)
{
    Items[rootItem] = new RenderingItem(QString("Objects"),QString("Options"),0,0);
    Items[environItem] = new RenderingItem(QString("Environment"),QVariant(),0,Items[rootItem]);
    Items[sliceItem] = (RenderingItem*)
        addItem(rootItem,"show_slice",QString("Slice"),
                QString("check"),Qt::Checked).internalId();
    Items[tractItem] = (RenderingItem*)
        addItem(rootItem,"show_tract",QString("Tract"),
                QString("check"),Qt::Checked).internalId();
    Items[regionItem] = (RenderingItem*)
        addItem(rootItem,"show_region",QString("Region"),
                QString("check"),Qt::Checked).internalId();
    Items[surfaceItem] = (RenderingItem*)
        addItem(rootItem,"show_surface",QString("Surface"),
                QString("check"),Qt::Checked).internalId();
    if(has_odf)
        Items[odfItem] = (RenderingItem*)
            addItem(rootItem,"show_odf",QString("ODF"),
                    QString("check"),Qt::Checked).internalId();
}

TreeModel::~TreeModel()
{
    std::map<std::string,RenderingItem*>::const_iterator iter = name_data_mapping.begin();
    std::map<std::string,RenderingItem*>::const_iterator end = name_data_mapping.end();
    QSettings settings;
    settings.beginGroup("Rendering Options");
    for(;iter != end;++iter)
        settings.setValue(&*iter->first.begin(),iter->second->getValue());
    settings.endGroup();
    delete Items[rootItem];
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
        (section) ? Items[rootItem]->title : Items[rootItem]->type;
    return QVariant();
}

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent)
const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    RenderingItem *parentItem;

    if (!parent.isValid())
        parentItem = Items[rootItem];
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

    if (parentItem == Items[rootItem])
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    RenderingItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = Items[rootItem];
    else
        parentItem = static_cast<RenderingItem*>(parent.internalPointer());

    return parentItem->childCount();
}

QModelIndex TreeModel::addItem(unsigned char root_index,
        const char* name,QVariant title, QVariant type, QVariant value)
{
    if(!name_data_mapping[name])
    {
        QSettings settings;
        settings.beginGroup("Rendering Options");
        name_data_mapping[name] = new RenderingItem(title,
            type,settings.value(name,value),Items[root_index]);
        name_default_values[name] = value;
        settings.endGroup();
    }
    else
        std::cout << "Duplicated item name in rending option" << std::endl;
    return createIndex(Items[root_index]->childCount()-1,1,name_data_mapping[name]);
}

void TreeModel::setDefault(void)
{
    std::map<std::string,RenderingItem*>::iterator iter = name_data_mapping.begin();
    std::map<std::string,RenderingItem*>::iterator end = name_data_mapping.end();
    std::map<std::string,QVariant>::iterator iter2 = name_default_values.begin();
    for(;iter != end;++iter,++iter2)
        iter->second->value = iter2->second;

}

RenderingTableWidget::RenderingTableWidget(tracking_window& cur_tracking_window_,
                                           QWidget *parent,bool has_odf_) :
        QTreeView(parent),cur_tracking_window(cur_tracking_window_),has_odf(has_odf_)
{
    setItemDelegateForColumn(1,data_delegate = new RenderingDelegate(this));

    setModel(treemodel = new TreeModel(this,has_odf));
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
    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "pespective",QString("Perspective"),QString("int"),5));

    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "bkg_color",QString("Background Color"),QString("color"),(int)0x00FFFFFF));


    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "anti_aliasing",QString("Anti-aliasing"),
                                      QStringList() << QString("Off") << QString("On"),0));

    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "line_smooth",QString("Line Smooth"),
                                      QStringList() << QString("Off") << QString("On"),0));
    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "point_smooth",QString("Point Smooth"),
                                      QStringList() << QString("Off") << QString("On"),0));
    openPersistentEditor(treemodel->addItem(TreeModel::environItem,
        "poly_smooth",QString("Polygon Smooth"),
                                      QStringList() << QString("Off") << QString("On"),0));

    // Slice
    openPersistentEditor(treemodel->addItem(TreeModel::sliceItem,
        "slice_alpha",QString("Opacity"),QString("float"),10));
    openPersistentEditor(treemodel->addItem(TreeModel::sliceItem,
        "slice_mag_filter",QString("Mag Filter"),
                                                      QStringList()
                                                      << QString("NEAREST")
                                                      << QString("LINEAR"),1));
    openPersistentEditor(treemodel->addItem(TreeModel::sliceItem,
        "slice_bend1",QString("Blend Func1"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("DST_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),4));
    openPersistentEditor(treemodel->addItem(TreeModel::sliceItem,
        "slice_bend2",QString("Blend Func2"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("SRC_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),5));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_alpha",QString("Opacity"),QString("float"),10));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_alpha_style",QString("Transparent Style"),
                                                      QStringList()
                                                      << QString("Sketch")
                                                      << QString("Classic"),0));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_bend1",QString("Blend Func1"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("DST_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),4));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_bend2",QString("Blend Func2"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("SRC_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),5));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_style",QString("Style"),QStringList()
                                       << QString("Line")
                                       << QString("Tube")
                                       << QString("End points"),1));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_color_style",QString("Color"),QStringList()
                                       << QString("Directional")
                                       << QString("Assigned")
                                       << QString("Local index")
                                       << QString("Averaged index")
                                       << QString("Averaged Directional"),0));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_visible_tracts",QString("Visible Tracts"),
                                                      QStringList()
                                                      << QString("5,000")
                                                      << QString("10,000")
                                                      << QString("25,000")
                                                      << QString("50,000")
                                                      << QString("100,000"),2));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_tube_detail",QString("Tube Detail"),
                                                      QStringList()
                                                      << QString("Coarse")
                                                      << QString("Fine")
                                                      << QString("Finer")
                                                      << QString("Finest"),1));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_size",QString("Size"),
                                                      QStringList()
                                                      << QString("Tube (0.01 voxel)")
                                                      << QString("Tube (0.02 voxel)")
                                                      << QString("Tube (0.04 voxel)")
                                                      << QString("Tube (0.08 voxel)")
                                                      << QString("Tube (0.1 voxel)")
                                                      << QString("Tube (0.2 voxel)")
                                                      << QString("Tube (0.4 voxel)")
                                                      << QString("Tube (0.6 voxel)")
                                                      << QString("Tube (0.8 voxel)")
                                                      ,5));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "end_point_shift",QString("Endpoint Shift"),
                                                      QStringList()
                                                      << QString("None")
                                                      << QString("1 voxel")
                                                      << QString("2 voxels")
                                                      << QString("3 voxels")
                                                      << QString("4 voxels")
                                                      << QString("5 voxels")
                                                      << QString("6 voxels")
                                                      << QString("7 voxels")
                                                      << QString("8 voxels")
                                                      << QString("9 voxels")
                                                      ,0));


    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_light_option",QString("Light"),QStringList()
                                << QString("One source")
                                << QString("Two sources")
                                << QString("Three sources"),2));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_light_dir",QString("Light Direction"),QString("int"),5));
    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_light_shading",QString("Light Shading"),QString("int"),6));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_light_diffuse",QString("Light Diffuse"),QString("int"),7));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_light_ambient",QString("Light Ambient"),QString("int"),0));

    openPersistentEditor(treemodel->addItem(TreeModel::tractItem,
        "tract_emission",QString("Tract Emission"),QString("int"),0));

    // region rednering options

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_alpha",QString("Opacity"),QString("float"),8));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_mesh_smoothed",QString("Mesh Rendering"),QStringList()
                                                      << QString("Original")
                                                      << QString("Smoothed"),1));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_bend1",QString("Blend Func1"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("DST_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),4));
    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_bend2",QString("Blend Func2"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("SRC_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),5));


    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_light_option",QString("Light"),QStringList()
                                                      << QString("One source")
                                                      << QString("Two sources")
                                                      << QString("Three sources"),2));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_light_dir",QString("Light Direction"),QString("int"),5));
    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_light_shading",QString("Light Shading"),QString("int"),4));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_light_diffuse",QString("Light Diffuse"),QString("int"),6));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_light_ambient",QString("Light Ambient"),QString("int"),0));

    openPersistentEditor(treemodel->addItem(TreeModel::regionItem,
        "region_emission",QString("Emission"),QString("int"),0));


    // surface rednering options
    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_color",QString("Color"),QString("color"),(int)0x00AAAAAA));


    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_alpha",QString("Opacity"),QString("float"),5));
    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_mesh_smoothed",QString("Mesh Rendering"),QStringList()
                                                      << QString("Original")
                                                      << QString("Smoothed")
                                                      << QString("Smoothed2"),0));
    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_bend1",QString("Blend Func1"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("DST_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),4));
    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_bend2",QString("Blend Func2"),
                                                      QStringList()
                                                      << QString("ZERO")
                                                      << QString("ONE")
                                                      << QString("SRC_COLOR")
                                                      << QString("ONE_MINUS_DST_COLOR")
                                                      << QString("SRC_ALPHA")
                                                      << QString("ONE_MINUS_SRC_ALPHA")
                                                      << QString("DST_ALPHA")
                                                      << QString("ONE_MINUS_DST_ALPHA"),5));


    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_light_option",QString("Light"),QStringList()
                                                      << QString("One source")
                                                      << QString("Two sources")
                                                      << QString("Three sources"),2));

    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_light_dir",QString("Light Direction"),QString("int"),5));
    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_light_shading",QString("Light Shading"),QString("int"),4));

    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_light_diffuse",QString("Light Diffuse"),QString("int"),6));

    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_light_ambient",QString("Light Ambient"),QString("int"),0));

    openPersistentEditor(treemodel->addItem(TreeModel::surfaceItem,
        "surface_emission",QString("Emission"),QString("int"),0));

    if(has_odf)
    {
        openPersistentEditor(treemodel->addItem(TreeModel::odfItem,
        "odf_position",QString("Position"),
                                                      QStringList()
                                                      << QString("Along Slide")
                                                      << QString("Slide Intersection"),0));
        openPersistentEditor(treemodel->addItem(TreeModel::odfItem,
        "odf_size",QString("Size"),
                                                      QStringList()
                                                      << QString("0.5")
                                                      << QString("1")
                                                      << QString("1.5")
                                                      << QString("2")
                                                      << QString("4")
                                                      << QString("8")
                                                      << QString("16"),1));

        openPersistentEditor(treemodel->addItem(TreeModel::odfItem,
        "odf_skip",QString("Interleaved"),
                                                      QStringList()
                                                      << QString("none")
                                                      << QString("2")
                                                      << QString("4"),0));

        openPersistentEditor(treemodel->addItem(TreeModel::odfItem,
        "odf_smoothing",QString("Smoothing"),
                                                      QStringList()
                                                      << QString("off")
                                                      << QString("on"),0));
    }
}
void RenderingTableWidget::setData(const char* name,QVariant data)
{
    collapseAll();
    treemodel->updateData(name,data);
    delete treemodel;
    setModel(treemodel = new TreeModel(this,has_odf));
    initialize();
    expandAll();
    collapseAll();
    connect(treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),
            cur_tracking_window.glWidget,SLOT(updateGL()));
}

void RenderingTableWidget::setDefault(void)
{
    collapseAll();
    treemodel->setDefault();
    delete treemodel;
    setModel(treemodel = new TreeModel(this,has_odf));
    initialize();
    expandAll();
    collapseAll();
    connect(treemodel,SIGNAL(dataChanged(QModelIndex,QModelIndex)),
            cur_tracking_window.glWidget,SLOT(updateGL()));
}

