/********************************************************************************
** Form generated from reading UI file 'vbcdialog.ui'
**
** Created: Tue Apr 24 10:53:07 2012
**      by: Qt User Interface Compiler version 4.7.1
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
#include <QtGui/QFormLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QListView>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_VBCDialog
{
public:
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout_5;
    QVBoxLayout *verticalLayout;
    QLabel *label_2;
    QListView *group1list;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *group1open;
    QPushButton *group1delete;
    QPushButton *moveup;
    QPushButton *movedown;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *group2layout;
    QLabel *label_3;
    QListView *group2list;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *group2open;
    QPushButton *group2delete;
    QSpacerItem *horizontalSpacer_2;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_4;
    QRadioButton *vbc_group;
    QRadioButton *vbc_single;
    QRadioButton *vbc_trend;
    QFormLayout *formLayout;
    QHBoxLayout *horizontalLayout_7;
    QLineEdit *fib_template;
    QPushButton *selectTemplate;
    QLabel *label_4;
    QLabel *label;
    QHBoxLayout *horizontalLayout_8;
    QLineEdit *output_dir;
    QPushButton *ChangeOutput;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *run;
    QPushButton *close;

    void setupUi(QDialog *VBCDialog)
    {
        if (VBCDialog->objectName().isEmpty())
            VBCDialog->setObjectName(QString::fromUtf8("VBCDialog"));
        VBCDialog->resize(582, 444);
        verticalLayout_3 = new QVBoxLayout(VBCDialog);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, -1, -1, -1);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label_2 = new QLabel(VBCDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout->addWidget(label_2);

        group1list = new QListView(VBCDialog);
        group1list->setObjectName(QString::fromUtf8("group1list"));
        group1list->setSelectionMode(QAbstractItemView::SingleSelection);

        verticalLayout->addWidget(group1list);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        group1open = new QPushButton(VBCDialog);
        group1open->setObjectName(QString::fromUtf8("group1open"));
        group1open->setMaximumSize(QSize(16777215, 26));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1open->setIcon(icon);

        horizontalLayout_2->addWidget(group1open);

        group1delete = new QPushButton(VBCDialog);
        group1delete->setObjectName(QString::fromUtf8("group1delete"));
        group1delete->setMaximumSize(QSize(16777215, 26));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1delete->setIcon(icon1);

        horizontalLayout_2->addWidget(group1delete);

        moveup = new QPushButton(VBCDialog);
        moveup->setObjectName(QString::fromUtf8("moveup"));

        horizontalLayout_2->addWidget(moveup);

        movedown = new QPushButton(VBCDialog);
        movedown->setObjectName(QString::fromUtf8("movedown"));

        horizontalLayout_2->addWidget(movedown);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_2);


        horizontalLayout_5->addLayout(verticalLayout);

        group2layout = new QVBoxLayout();
        group2layout->setSpacing(0);
        group2layout->setObjectName(QString::fromUtf8("group2layout"));
        label_3 = new QLabel(VBCDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        group2layout->addWidget(label_3);

        group2list = new QListView(VBCDialog);
        group2list->setObjectName(QString::fromUtf8("group2list"));

        group2layout->addWidget(group2list);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        group2open = new QPushButton(VBCDialog);
        group2open->setObjectName(QString::fromUtf8("group2open"));
        group2open->setMaximumSize(QSize(16777215, 26));
        group2open->setIcon(icon);

        horizontalLayout_3->addWidget(group2open);

        group2delete = new QPushButton(VBCDialog);
        group2delete->setObjectName(QString::fromUtf8("group2delete"));
        group2delete->setMaximumSize(QSize(16777215, 26));
        group2delete->setIcon(icon1);

        horizontalLayout_3->addWidget(group2delete);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        group2layout->addLayout(horizontalLayout_3);


        horizontalLayout_5->addLayout(group2layout);


        verticalLayout_3->addLayout(horizontalLayout_5);

        groupBox = new QGroupBox(VBCDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_4 = new QHBoxLayout(groupBox);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        vbc_group = new QRadioButton(groupBox);
        vbc_group->setObjectName(QString::fromUtf8("vbc_group"));
        vbc_group->setChecked(true);

        horizontalLayout_4->addWidget(vbc_group);

        vbc_single = new QRadioButton(groupBox);
        vbc_single->setObjectName(QString::fromUtf8("vbc_single"));

        horizontalLayout_4->addWidget(vbc_single);

        vbc_trend = new QRadioButton(groupBox);
        vbc_trend->setObjectName(QString::fromUtf8("vbc_trend"));

        horizontalLayout_4->addWidget(vbc_trend);


        verticalLayout_3->addWidget(groupBox);

        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setContentsMargins(-1, 0, -1, -1);
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        fib_template = new QLineEdit(VBCDialog);
        fib_template->setObjectName(QString::fromUtf8("fib_template"));

        horizontalLayout_7->addWidget(fib_template);

        selectTemplate = new QPushButton(VBCDialog);
        selectTemplate->setObjectName(QString::fromUtf8("selectTemplate"));
        selectTemplate->setMaximumSize(QSize(24, 16777215));

        horizontalLayout_7->addWidget(selectTemplate);


        formLayout->setLayout(0, QFormLayout::FieldRole, horizontalLayout_7);

        label_4 = new QLabel(VBCDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_4);

        label = new QLabel(VBCDialog);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        output_dir = new QLineEdit(VBCDialog);
        output_dir->setObjectName(QString::fromUtf8("output_dir"));

        horizontalLayout_8->addWidget(output_dir);

        ChangeOutput = new QPushButton(VBCDialog);
        ChangeOutput->setObjectName(QString::fromUtf8("ChangeOutput"));
        ChangeOutput->setMaximumSize(QSize(24, 16777215));

        horizontalLayout_8->addWidget(ChangeOutput);


        formLayout->setLayout(1, QFormLayout::FieldRole, horizontalLayout_8);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(-1, 0, -1, -1);
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

        run = new QPushButton(VBCDialog);
        run->setObjectName(QString::fromUtf8("run"));
        run->setMaximumSize(QSize(16777215, 26));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/run.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        run->setIcon(icon2);

        horizontalLayout_6->addWidget(run);

        close = new QPushButton(VBCDialog);
        close->setObjectName(QString::fromUtf8("close"));

        horizontalLayout_6->addWidget(close);


        formLayout->setLayout(2, QFormLayout::FieldRole, horizontalLayout_6);


        verticalLayout_3->addLayout(formLayout);


        retranslateUi(VBCDialog);

        QMetaObject::connectSlotsByName(VBCDialog);
    } // setupUi

    void retranslateUi(QDialog *VBCDialog)
    {
        VBCDialog->setWindowTitle(QApplication::translate("VBCDialog", "Dialog", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("VBCDialog", "Group1", 0, QApplication::UnicodeUTF8));
        group1open->setText(QString());
        group1delete->setText(QString());
        moveup->setText(QApplication::translate("VBCDialog", "Up", 0, QApplication::UnicodeUTF8));
        movedown->setText(QApplication::translate("VBCDialog", "Down", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("VBCDialog", "Group2", 0, QApplication::UnicodeUTF8));
        group2open->setText(QString());
        group2delete->setText(QString());
        groupBox->setTitle(QApplication::translate("VBCDialog", "Method", 0, QApplication::UnicodeUTF8));
        vbc_group->setText(QApplication::translate("VBCDialog", "VBC Groupwise ", 0, QApplication::UnicodeUTF8));
        vbc_single->setText(QApplication::translate("VBCDialog", "VBC Single Subject", 0, QApplication::UnicodeUTF8));
        vbc_trend->setText(QApplication::translate("VBCDialog", "VBC Trend Testing", 0, QApplication::UnicodeUTF8));
        selectTemplate->setText(QApplication::translate("VBCDialog", "...", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("VBCDialog", "Fiber Template", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("VBCDialog", "Output Directory", 0, QApplication::UnicodeUTF8));
        ChangeOutput->setText(QApplication::translate("VBCDialog", "...", 0, QApplication::UnicodeUTF8));
        run->setText(QApplication::translate("VBCDialog", "Run", 0, QApplication::UnicodeUTF8));
        close->setText(QApplication::translate("VBCDialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VBCDialog: public Ui_VBCDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBCDIALOG_H
