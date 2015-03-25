/********************************************************************************
** Form generated from reading UI file 'atlasdialog.ui'
**
** Created: Wed Mar 25 00:39:39 2015
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ATLASDIALOG_H
#define UI_ATLASDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QListView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_AtlasDialog
{
public:
    QVBoxLayout *verticalLayout;
    QComboBox *atlasListBox;
    QListView *region_list;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *add_atlas;
    QPushButton *pushButton;

    void setupUi(QDialog *AtlasDialog)
    {
        if (AtlasDialog->objectName().isEmpty())
            AtlasDialog->setObjectName(QString::fromUtf8("AtlasDialog"));
        AtlasDialog->resize(182, 480);
        AtlasDialog->setContextMenuPolicy(Qt::NoContextMenu);
        verticalLayout = new QVBoxLayout(AtlasDialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        atlasListBox = new QComboBox(AtlasDialog);
        atlasListBox->setObjectName(QString::fromUtf8("atlasListBox"));

        verticalLayout->addWidget(atlasListBox);

        region_list = new QListView(AtlasDialog);
        region_list->setObjectName(QString::fromUtf8("region_list"));
        region_list->setSelectionMode(QAbstractItemView::MultiSelection);
        region_list->setSelectionBehavior(QAbstractItemView::SelectRows);

        verticalLayout->addWidget(region_list);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        add_atlas = new QPushButton(AtlasDialog);
        add_atlas->setObjectName(QString::fromUtf8("add_atlas"));

        horizontalLayout->addWidget(add_atlas);

        pushButton = new QPushButton(AtlasDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);


        verticalLayout->addLayout(horizontalLayout);


        retranslateUi(AtlasDialog);

        QMetaObject::connectSlotsByName(AtlasDialog);
    } // setupUi

    void retranslateUi(QDialog *AtlasDialog)
    {
        AtlasDialog->setWindowTitle(QString());
        add_atlas->setText(QApplication::translate("AtlasDialog", "Add", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("AtlasDialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class AtlasDialog: public Ui_AtlasDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ATLASDIALOG_H
