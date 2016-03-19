#include "qcompletelineedit.h"
#include <QKeyEvent>
#include <QListView>
#include <QStringListModel>
#include <QDebug>

void QCompleteLineEdit::setList(QStringList list)
{
    words = list;
    listView = new QListView(this);
    model = new QStringListModel(this);
    listView->setWindowFlags(Qt::ToolTip);
    connect(this, SIGNAL(textChanged(const QString &)), this, SLOT(setCompleter(const QString &)));
    connect(listView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(completeText(const QModelIndex &)));
}

void QCompleteLineEdit::keyPressEvent(QKeyEvent *e) {
    if (!words.empty() && !listView->isHidden()) {
        int key = e->key();
        int count = listView->model()->rowCount();
        QModelIndex currentIndex = listView->currentIndex();

        if (Qt::Key_Down == key) {
            int row = currentIndex.row() + 1;
            if (row >= count) {
                row = 0;
            }

            QModelIndex index = listView->model()->index(row, 0);
            listView->setCurrentIndex(index);
        } else if (Qt::Key_Up == key) {
            int row = currentIndex.row() - 1;
            if (row < 0) {
                row = count - 1;
            }

            QModelIndex index = listView->model()->index(row, 0);
            listView->setCurrentIndex(index);
        } else if (Qt::Key_Escape == key) {
            listView->hide();
        } else if (Qt::Key_Enter == key || Qt::Key_Return == key) {

            if (currentIndex.isValid()) {
                QString text = listView->currentIndex().data().toString();
                setText(text);
            }

            listView->hide();
        } else {
            listView->hide();
            QLineEdit::keyPressEvent(e);
        }
    } else {
        QLineEdit::keyPressEvent(e);
    }
}

void QCompleteLineEdit::setCompleter(const QString &text) {
    if (text.isEmpty()) {
        listView->hide();
        return;
    }

    if ((text.length() > 1) && (!listView->isHidden())) {
        return;
    }
    QStringList sl;
    foreach(QString word, words) {
        if (word.toLower().contains(text.toLower())) {
            sl << word;
        }
    }

    model->setStringList(sl);
    listView->setModel(model);

    if (model->rowCount() == 0) {
        return;
    }

    // Position the text edit
    listView->setMinimumWidth(width());
    listView->setMaximumWidth(width());

    QPoint p(0, height());
    int x = mapToGlobal(p).x();
    int y = mapToGlobal(p).y() + 1;

    listView->move(x, y);
    listView->show();
}

void QCompleteLineEdit::completeText(const QModelIndex &index) {
    QString text = index.data().toString();
    setText(text);
    listView->hide();
    emit selected();
}
