#ifndef RENDERINGTABLEWIDGET_H
#define RENDERINGTABLEWIDGET_H

#include <QTreeView>
#include <QItemDelegate>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <QSettings>
#include <string>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>


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
private:
    QList<RenderingItem*> childItems;
    RenderingItem *parentItem;
public:
    QObject* GUI = nullptr;
    QString id;
    QVariant title,type,value,def_value;
    QString hint;
public:
    RenderingItem(QVariant title_, QVariant type_, QString id_,QVariant value_, RenderingItem *parent = nullptr):
        parentItem(parent),id(id_),title(title_),type(type_),value(value_)
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
     void setValue(QVariant new_value);
     void setMinMax(float min,float max,float step);
     void setList(QStringList list);
     QString getListValue(void) const;
};

class RenderingTableWidget;
class TreeModel : public QAbstractItemModel
{
    Q_OBJECT
private:
    std::shared_ptr<RenderingItem> root;
    std::unordered_map<QString,RenderingItem*> root_mapping;
    std::unordered_map<QString,RenderingItem*> name_data_mapping;
public:
    TreeModel(RenderingTableWidget *parent);
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
    void saveParameters(void);
public:
    void addNode(QString root_name,QString id,QVariant title);
    QModelIndex addItem(QString root_name,QString id,QVariant title, QVariant type, QVariant value, QString hint = QString());
    QVariant getData(QString name)
    {
        auto iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            throw std::runtime_error(("Cannot find the setting value: "+name.toStdString()).c_str());
        return iter->second->value;
    }
    RenderingItem& operator[](QString name)
    {
        auto iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            throw std::runtime_error(("Cannot find the setting value: "+name.toStdString()).c_str());
        return *(iter->second);
    }
    QStringList get_param_list(QString root_name)
    {
        QStringList result;
        RenderingItem* parent = root_mapping[root_name];
        if(!parent)
            throw std::runtime_error("Cannot find the root node");
        for(int index = 0;index < parent->childCount();++index)
            if(!parent->child(index)->type.isNull()) // second layer tree node has type = QVariant() assigned in AddNode
                result.push_back(parent->child(index)->id);
        return result;
    }
    QStringList getParamList(void)
    {
        QStringList result;
        for(auto& iter : name_data_mapping)
            result << iter.first;
        return result;
    }
    void setDefault(QString);
private:

};
class tracking_window;
class RenderingTableWidget : public QTreeView
{
    Q_OBJECT
private:
    tracking_window& cur_tracking_window;
    std::unordered_set<std::string> tract_update_list,tract_color_map_update_list,region_color_map_update_list;
public:
    RenderingDelegate* data_delegate;
    TreeModel* treemodel;
public:
    explicit RenderingTableWidget(tracking_window& cur_tracking_window_,QWidget *parent);
    QVariant getData(QString name){return treemodel->getData(name);}
    QString getListValue(QString name){return (*treemodel)[name].getListValue();}
public:
    void setData(QString name,QVariant data){(*treemodel)[name].setValue(data);}
    void setMinMax(QString name,float min,float max,float step){(*treemodel)[name].setMinMax(min,max,step);}
    void setList(QString name,QStringList list){(*treemodel)[name].setList(list);}
    void initialize(void);
    void saveParameters(void){treemodel->saveParameters();}
public slots:
    void setDefault(QString parent_id);
    void dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
};

#endif // RENDERINGTABLEWIDGET_H
