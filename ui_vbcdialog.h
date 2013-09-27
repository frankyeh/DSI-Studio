/********************************************************************************
** Form generated from reading UI file 'vbcdialog.ui'
**
** Created: Thu Sep 26 22:22:06 2013
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VBCDIALOG_H
#define UI_VBCDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QListView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VBCDialog
{
public:
    QVBoxLayout *verticalLayout_3;
    QGroupBox *method_group;
    QVBoxLayout *verticalLayout_7;
    QVBoxLayout *verticalLayout;
    QListView *group_list;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *group1open;
    QPushButton *group1delete;
    QToolButton *moveup;
    QToolButton *movedown;
    QToolButton *open_list1;
    QToolButton *save_list1;
    QToolButton *open_dir1;
    QSpacerItem *horizontalSpacer;
    QWidget *skeleton_widget;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *skeleton;
    QToolButton *open_skeleton;
    QHBoxLayout *horizontalLayout_4;
    QLabel *ODF_label;
    QLineEdit *output_file_name;
    QToolButton *select_output_file;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *create_data_base;
    QPushButton *close;

    void setupUi(QDialog *VBCDialog)
    {
        if (VBCDialog->objectName().isEmpty())
            VBCDialog->setObjectName(QString::fromUtf8("VBCDialog"));
        VBCDialog->resize(611, 361);
        verticalLayout_3 = new QVBoxLayout(VBCDialog);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        method_group = new QGroupBox(VBCDialog);
        method_group->setObjectName(QString::fromUtf8("method_group"));
        verticalLayout_7 = new QVBoxLayout(method_group);
        verticalLayout_7->setSpacing(5);
        verticalLayout_7->setContentsMargins(5, 5, 5, 5);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        group_list = new QListView(method_group);
        group_list->setObjectName(QString::fromUtf8("group_list"));
        group_list->setSelectionMode(QAbstractItemView::SingleSelection);

        verticalLayout->addWidget(group_list);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        group1open = new QPushButton(method_group);
        group1open->setObjectName(QString::fromUtf8("group1open"));
        group1open->setMaximumSize(QSize(16777215, 26));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/add.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1open->setIcon(icon);

        horizontalLayout_2->addWidget(group1open);

        group1delete = new QPushButton(method_group);
        group1delete->setObjectName(QString::fromUtf8("group1delete"));
        group1delete->setMaximumSize(QSize(16777215, 26));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1delete->setIcon(icon1);

        horizontalLayout_2->addWidget(group1delete);

        moveup = new QToolButton(method_group);
        moveup->setObjectName(QString::fromUtf8("moveup"));
        moveup->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(moveup);

        movedown = new QToolButton(method_group);
        movedown->setObjectName(QString::fromUtf8("movedown"));
        movedown->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(movedown);

        open_list1 = new QToolButton(method_group);
        open_list1->setObjectName(QString::fromUtf8("open_list1"));
        open_list1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(open_list1);

        save_list1 = new QToolButton(method_group);
        save_list1->setObjectName(QString::fromUtf8("save_list1"));
        save_list1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(save_list1);

        open_dir1 = new QToolButton(method_group);
        open_dir1->setObjectName(QString::fromUtf8("open_dir1"));
        open_dir1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(open_dir1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_2);


        verticalLayout_7->addLayout(verticalLayout);


        verticalLayout_3->addWidget(method_group);

        skeleton_widget = new QWidget(VBCDialog);
        skeleton_widget->setObjectName(QString::fromUtf8("skeleton_widget"));
        horizontalLayout = new QHBoxLayout(skeleton_widget);
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(skeleton_widget);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        skeleton = new QLineEdit(skeleton_widget);
        skeleton->setObjectName(QString::fromUtf8("skeleton"));

        horizontalLayout->addWidget(skeleton);

        open_skeleton = new QToolButton(skeleton_widget);
        open_skeleton->setObjectName(QString::fromUtf8("open_skeleton"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        open_skeleton->setIcon(icon2);

        horizontalLayout->addWidget(open_skeleton);


        verticalLayout_3->addWidget(skeleton_widget);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        ODF_label = new QLabel(VBCDialog);
        ODF_label->setObjectName(QString::fromUtf8("ODF_label"));

        horizontalLayout_4->addWidget(ODF_label);

        output_file_name = new QLineEdit(VBCDialog);
        output_file_name->setObjectName(QString::fromUtf8("output_file_name"));

        horizontalLayout_4->addWidget(output_file_name);

        select_output_file = new QToolButton(VBCDialog);
        select_output_file->setObjectName(QString::fromUtf8("select_output_file"));
        select_output_file->setIcon(icon2);

        horizontalLayout_4->addWidget(select_output_file);


        verticalLayout_3->addLayout(horizontalLayout_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(-1, 0, -1, -1);
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

        create_data_base = new QPushButton(VBCDialog);
        create_data_base->setObjectName(QString::fromUtf8("create_data_base"));

        horizontalLayout_6->addWidget(create_data_base);

        close = new QPushButton(VBCDialog);
        close->setObjectName(QString::fromUtf8("close"));

        horizontalLayout_6->addWidget(close);


        verticalLayout_3->addLayout(horizontalLayout_6);


        retranslateUi(VBCDialog);

        QMetaObject::connectSlotsByName(VBCDialog);
    } // setupUi

    void retranslateUi(QDialog *VBCDialog)
    {
        VBCDialog->setWindowTitle(QApplication::translate("VBCDialog", "Connectometry toolbox", 0, QApplication::UnicodeUTF8));
        method_group->setTitle(QApplication::translate("VBCDialog", " Select subject FIB files", 0, QApplication::UnicodeUTF8));
        group1open->setText(QApplication::translate("VBCDialog", "Add", 0, QApplication::UnicodeUTF8));
        group1delete->setText(QString());
        moveup->setText(QApplication::translate("VBCDialog", "Up", 0, QApplication::UnicodeUTF8));
        movedown->setText(QApplication::translate("VBCDialog", "Down", 0, QApplication::UnicodeUTF8));
        open_list1->setText(QApplication::translate("VBCDialog", "Open List", 0, QApplication::UnicodeUTF8));
        save_list1->setText(QApplication::translate("VBCDialog", "Save List", 0, QApplication::UnicodeUTF8));
        open_dir1->setText(QApplication::translate("VBCDialog", "Search in Directory...", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("VBCDialog", "Skeleton:", 0, QApplication::UnicodeUTF8));
        open_skeleton->setText(QString());
        ODF_label->setText(QApplication::translate("VBCDialog", "Output file name:", 0, QApplication::UnicodeUTF8));
        select_output_file->setText(QApplication::translate("VBCDialog", "...", 0, QApplication::UnicodeUTF8));
        create_data_base->setText(QApplication::translate("VBCDialog", "Create Database", 0, QApplication::UnicodeUTF8));
        close->setText(QApplication::translate("VBCDialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VBCDialog: public Ui_VBCDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBCDIALOG_H
