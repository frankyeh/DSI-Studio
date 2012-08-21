#ifndef RENDERINGTABLEWIDGET_H
#define RENDERINGTABLEWIDGET_H

#include <QTreeView>
#include <QItemDelegate>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <QSettings>
#include <map>
#include <string>


class RenderingDelegate : public QItemDelegate
 {
     Q_OBJECT

 public:
    RenderingDelegate(QObject *parent)
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



class RenderingItem
 {
 public:
    RenderingItem(QVariant title_, QVariant type_, QVariant value_, RenderingItem *parent = 0):
            parentItem(parent),title(title_),type(type_),value(value_)
    {
        if(parent)
            parent->appendChild(this);
    }

    ~RenderingItem(){qDeleteAll(childItems);}

     void appendChild(RenderingItem *item)
     {
         childItems.append(item);
     }
     RenderingItem *child(int row){return childItems.value(row);}
     int childCount() const {return childItems.count();}
     int row() const
     {
         return (parentItem) ? parentItem->childItems.indexOf(const_cast<RenderingItem*>(this)) : 0;
     }
     RenderingItem *parent(void) {return parentItem;}
     void setParent(RenderingItem *parentItem_) {parentItem = parentItem_;}
     QVariant getValue() const{return value;}
 private:
     QList<RenderingItem*> childItems;
     RenderingItem *parentItem;
 public:
     QVariant title,type,value;

 };

class RenderingTableWidget;
class TreeModel : public QAbstractItemModel
{
    Q_OBJECT
private:
    std::map<std::string,RenderingItem*> name_data_mapping;
    std::map<std::string,QVariant> name_default_values;
public:
    TreeModel(RenderingTableWidget *parent,bool has_odf);
    ~TreeModel();

    QVariant data(const QModelIndex &index, int role) const;
    bool setData ( const QModelIndex & index, const QVariant & value, int role);

    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
                      const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
public:
    QModelIndex addItem(unsigned char root_index,
            const char* name,QVariant title, QVariant type, QVariant value);
    QVariant getData(const char* name)
    {
        std::map<std::string,RenderingItem*>::const_iterator iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            return QVariant();
        return iter->second->value;
    }
    void updateData(const char* name,QVariant data)
    {
        std::map<std::string,RenderingItem*>::const_iterator iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            return;
        iter->second->value = data;
    }

    void setDefault(void);
    enum {rootItem = 0,
          environItem = 1,
          sliceItem = 2,
          tractItem = 3,
          regionItem = 4,
          surfaceItem = 5,
          odfItem = 6};
private:
    RenderingItem *Items[7];
};
class tracking_window;
class RenderingTableWidget : public QTreeView
{
    Q_OBJECT
private:
    tracking_window& cur_tracking_window;
    bool has_odf;
public:
    RenderingDelegate* data_delegate;
    TreeModel* treemodel;
public:
    explicit RenderingTableWidget(tracking_window& cur_tracking_window_,
                                  QWidget *parent,bool has_odf);
    QVariant getData(const char* name){return treemodel->getData(name);}
    void setData(const char* name,QVariant data);
    void initialize(void);
public slots:
    void setDefault(void);
};

#endif // RENDERINGTABLEWIDGET_H
