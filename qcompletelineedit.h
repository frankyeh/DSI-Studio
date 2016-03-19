#ifndef COMPLETELINEEDIT_H
#define COMPLETELINEEDIT_H

#include <QLineEdit>
#include <QStringList>

class QListView;
class QStringListModel;
class QModelIndex;

class QCompleteLineEdit : public QLineEdit {
    Q_OBJECT
public:
    QCompleteLineEdit(QWidget *parent): QLineEdit(parent){}
    void setList(QStringList list);

public slots:
    void setCompleter(const QString &text);
    void completeText(const QModelIndex &index);
protected:
    virtual void keyPressEvent(QKeyEvent *e);
signals:
    void selected();
private:
    QStringList words;
    QListView *listView;
    QStringListModel *model;
};

#endif // COMPLETELINEEDIT_H
