/********************************************************************************
** Form generated from reading UI file 'vbcdialog.ui'
**
** Created: Wed Oct 17 16:41:40 2012
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
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QListView>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "plot/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_VBCDialog
{
public:
    QVBoxLayout *verticalLayout_3;
    QGroupBox *method_group;
    QHBoxLayout *horizontalLayout_4;
    QRadioButton *vbc_group;
    QRadioButton *vbc_single;
    QRadioButton *vbc_trend;
    QGroupBox *cluster_group;
    QVBoxLayout *verticalLayout_5;
    QLabel *progress;
    QCustomPlot *report_widget1;
    QCustomPlot *report_widget2;
    QGroupBox *subject_data_group;
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QLabel *group1_label;
    QListView *group1list;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *group1open;
    QPushButton *group1delete;
    QToolButton *moveup;
    QToolButton *movedown;
    QToolButton *open_list1;
    QToolButton *open_dir1;
    QToolButton *save_list1;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *verticalLayout_6;
    QWidget *group2_widget;
    QVBoxLayout *verticalLayout_4;
    QLabel *group2_label;
    QListView *group2list;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *group2open;
    QPushButton *group2delete;
    QToolButton *open_list2;
    QToolButton *open_dir2;
    QToolButton *save_list2;
    QSpacerItem *horizontalSpacer_2;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_6;
    QDoubleSpinBox *qa_threshold;
    QLabel *label_5;
    QDoubleSpinBox *p_value_threshold;
    QLabel *label_2;
    QSpinBox *permutation_num;
    QLabel *label;
    QSpinBox *thread_count;
    QSpacerItem *horizontalSpacer_5;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_4;
    QLineEdit *mapping;
    QToolButton *open_mapping;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *load_subject_data;
    QPushButton *close;

    void setupUi(QDialog *VBCDialog)
    {
        if (VBCDialog->objectName().isEmpty())
            VBCDialog->setObjectName(QString::fromUtf8("VBCDialog"));
        VBCDialog->resize(611, 576);
        verticalLayout_3 = new QVBoxLayout(VBCDialog);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        method_group = new QGroupBox(VBCDialog);
        method_group->setObjectName(QString::fromUtf8("method_group"));
        horizontalLayout_4 = new QHBoxLayout(method_group);
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setContentsMargins(5, 5, 5, 5);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        vbc_group = new QRadioButton(method_group);
        vbc_group->setObjectName(QString::fromUtf8("vbc_group"));
        vbc_group->setChecked(true);

        horizontalLayout_4->addWidget(vbc_group);

        vbc_single = new QRadioButton(method_group);
        vbc_single->setObjectName(QString::fromUtf8("vbc_single"));

        horizontalLayout_4->addWidget(vbc_single);

        vbc_trend = new QRadioButton(method_group);
        vbc_trend->setObjectName(QString::fromUtf8("vbc_trend"));

        horizontalLayout_4->addWidget(vbc_trend);


        verticalLayout_3->addWidget(method_group);

        cluster_group = new QGroupBox(VBCDialog);
        cluster_group->setObjectName(QString::fromUtf8("cluster_group"));
        verticalLayout_5 = new QVBoxLayout(cluster_group);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(5, 5, 5, 5);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        progress = new QLabel(cluster_group);
        progress->setObjectName(QString::fromUtf8("progress"));

        verticalLayout_5->addWidget(progress);

        report_widget1 = new QCustomPlot(cluster_group);
        report_widget1->setObjectName(QString::fromUtf8("report_widget1"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(report_widget1->sizePolicy().hasHeightForWidth());
        report_widget1->setSizePolicy(sizePolicy);
        report_widget1->setMinimumSize(QSize(0, 20));

        verticalLayout_5->addWidget(report_widget1);

        report_widget2 = new QCustomPlot(cluster_group);
        report_widget2->setObjectName(QString::fromUtf8("report_widget2"));
        sizePolicy.setHeightForWidth(report_widget2->sizePolicy().hasHeightForWidth());
        report_widget2->setSizePolicy(sizePolicy);
        report_widget2->setMinimumSize(QSize(0, 20));

        verticalLayout_5->addWidget(report_widget2);


        verticalLayout_3->addWidget(cluster_group);

        subject_data_group = new QGroupBox(VBCDialog);
        subject_data_group->setObjectName(QString::fromUtf8("subject_data_group"));
        verticalLayout_2 = new QVBoxLayout(subject_data_group);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        group1_label = new QLabel(subject_data_group);
        group1_label->setObjectName(QString::fromUtf8("group1_label"));

        verticalLayout->addWidget(group1_label);

        group1list = new QListView(subject_data_group);
        group1list->setObjectName(QString::fromUtf8("group1list"));
        group1list->setSelectionMode(QAbstractItemView::SingleSelection);

        verticalLayout->addWidget(group1list);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        group1open = new QPushButton(subject_data_group);
        group1open->setObjectName(QString::fromUtf8("group1open"));
        group1open->setMaximumSize(QSize(16777215, 26));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/add.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1open->setIcon(icon);

        horizontalLayout_2->addWidget(group1open);

        group1delete = new QPushButton(subject_data_group);
        group1delete->setObjectName(QString::fromUtf8("group1delete"));
        group1delete->setMaximumSize(QSize(16777215, 26));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1delete->setIcon(icon1);

        horizontalLayout_2->addWidget(group1delete);

        moveup = new QToolButton(subject_data_group);
        moveup->setObjectName(QString::fromUtf8("moveup"));
        moveup->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(moveup);

        movedown = new QToolButton(subject_data_group);
        movedown->setObjectName(QString::fromUtf8("movedown"));
        movedown->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(movedown);

        open_list1 = new QToolButton(subject_data_group);
        open_list1->setObjectName(QString::fromUtf8("open_list1"));
        open_list1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(open_list1);

        open_dir1 = new QToolButton(subject_data_group);
        open_dir1->setObjectName(QString::fromUtf8("open_dir1"));
        open_dir1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(open_dir1);

        save_list1 = new QToolButton(subject_data_group);
        save_list1->setObjectName(QString::fromUtf8("save_list1"));
        save_list1->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_2->addWidget(save_list1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_2);


        verticalLayout_2->addLayout(verticalLayout);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, -1, -1, -1);
        group2_widget = new QWidget(subject_data_group);
        group2_widget->setObjectName(QString::fromUtf8("group2_widget"));
        group2_widget->setMinimumSize(QSize(50, 0));
        verticalLayout_4 = new QVBoxLayout(group2_widget);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        group2_label = new QLabel(group2_widget);
        group2_label->setObjectName(QString::fromUtf8("group2_label"));

        verticalLayout_4->addWidget(group2_label);

        group2list = new QListView(group2_widget);
        group2list->setObjectName(QString::fromUtf8("group2list"));

        verticalLayout_4->addWidget(group2list);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        group2open = new QPushButton(group2_widget);
        group2open->setObjectName(QString::fromUtf8("group2open"));
        group2open->setMaximumSize(QSize(16777215, 26));
        group2open->setIcon(icon);

        horizontalLayout_3->addWidget(group2open);

        group2delete = new QPushButton(group2_widget);
        group2delete->setObjectName(QString::fromUtf8("group2delete"));
        group2delete->setMaximumSize(QSize(16777215, 26));
        group2delete->setIcon(icon1);

        horizontalLayout_3->addWidget(group2delete);

        open_list2 = new QToolButton(group2_widget);
        open_list2->setObjectName(QString::fromUtf8("open_list2"));
        open_list2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(open_list2);

        open_dir2 = new QToolButton(group2_widget);
        open_dir2->setObjectName(QString::fromUtf8("open_dir2"));
        open_dir2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(open_dir2);

        save_list2 = new QToolButton(group2_widget);
        save_list2->setObjectName(QString::fromUtf8("save_list2"));
        save_list2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(save_list2);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        verticalLayout_4->addLayout(horizontalLayout_3);

        group2_label->raise();
        group2list->raise();

        verticalLayout_6->addWidget(group2_widget);


        verticalLayout_2->addLayout(verticalLayout_6);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_6 = new QLabel(subject_data_group);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_9->addWidget(label_6);

        qa_threshold = new QDoubleSpinBox(subject_data_group);
        qa_threshold->setObjectName(QString::fromUtf8("qa_threshold"));
        qa_threshold->setMinimum(0);
        qa_threshold->setMaximum(20);
        qa_threshold->setSingleStep(0.05);
        qa_threshold->setValue(0.15);

        horizontalLayout_9->addWidget(qa_threshold);

        label_5 = new QLabel(subject_data_group);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setMinimumSize(QSize(0, 0));

        horizontalLayout_9->addWidget(label_5);

        p_value_threshold = new QDoubleSpinBox(subject_data_group);
        p_value_threshold->setObjectName(QString::fromUtf8("p_value_threshold"));
        p_value_threshold->setDecimals(3);
        p_value_threshold->setMaximum(0.5);
        p_value_threshold->setSingleStep(0.01);
        p_value_threshold->setValue(0.05);

        horizontalLayout_9->addWidget(p_value_threshold);

        label_2 = new QLabel(subject_data_group);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_9->addWidget(label_2);

        permutation_num = new QSpinBox(subject_data_group);
        permutation_num->setObjectName(QString::fromUtf8("permutation_num"));
        permutation_num->setMinimum(50);
        permutation_num->setMaximum(100000);
        permutation_num->setSingleStep(100);
        permutation_num->setValue(5000);

        horizontalLayout_9->addWidget(permutation_num);

        label = new QLabel(subject_data_group);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_9->addWidget(label);

        thread_count = new QSpinBox(subject_data_group);
        thread_count->setObjectName(QString::fromUtf8("thread_count"));
        thread_count->setMinimum(1);
        thread_count->setMaximum(12);

        horizontalLayout_9->addWidget(thread_count);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_5);


        verticalLayout_2->addLayout(horizontalLayout_9);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_4 = new QLabel(subject_data_group);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setMinimumSize(QSize(0, 0));
        label_4->setMaximumSize(QSize(80, 16777215));

        horizontalLayout_7->addWidget(label_4);

        mapping = new QLineEdit(subject_data_group);
        mapping->setObjectName(QString::fromUtf8("mapping"));

        horizontalLayout_7->addWidget(mapping);

        open_mapping = new QToolButton(subject_data_group);
        open_mapping->setObjectName(QString::fromUtf8("open_mapping"));

        horizontalLayout_7->addWidget(open_mapping);


        verticalLayout_2->addLayout(horizontalLayout_7);


        verticalLayout_3->addWidget(subject_data_group);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(-1, 0, -1, -1);
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

        load_subject_data = new QPushButton(VBCDialog);
        load_subject_data->setObjectName(QString::fromUtf8("load_subject_data"));

        horizontalLayout_6->addWidget(load_subject_data);

        close = new QPushButton(VBCDialog);
        close->setObjectName(QString::fromUtf8("close"));

        horizontalLayout_6->addWidget(close);


        verticalLayout_3->addLayout(horizontalLayout_6);


        retranslateUi(VBCDialog);

        QMetaObject::connectSlotsByName(VBCDialog);
    } // setupUi

    void retranslateUi(QDialog *VBCDialog)
    {
        VBCDialog->setWindowTitle(QApplication::translate("VBCDialog", "Dialog", 0, QApplication::UnicodeUTF8));
        method_group->setTitle(QApplication::translate("VBCDialog", "Method", 0, QApplication::UnicodeUTF8));
        vbc_group->setText(QApplication::translate("VBCDialog", "VBC Groupwise ", 0, QApplication::UnicodeUTF8));
        vbc_single->setText(QApplication::translate("VBCDialog", "VBC Single Subject", 0, QApplication::UnicodeUTF8));
        vbc_trend->setText(QApplication::translate("VBCDialog", "VBC Trend Testing", 0, QApplication::UnicodeUTF8));
        cluster_group->setTitle(QApplication::translate("VBCDialog", "Distribution", 0, QApplication::UnicodeUTF8));
        progress->setText(QApplication::translate("VBCDialog", "Progress", 0, QApplication::UnicodeUTF8));
        subject_data_group->setTitle(QApplication::translate("VBCDialog", "Subject data", 0, QApplication::UnicodeUTF8));
        group1_label->setText(QApplication::translate("VBCDialog", "Group1", 0, QApplication::UnicodeUTF8));
        group1open->setText(QApplication::translate("VBCDialog", "Add", 0, QApplication::UnicodeUTF8));
        group1delete->setText(QString());
        moveup->setText(QApplication::translate("VBCDialog", "Up", 0, QApplication::UnicodeUTF8));
        movedown->setText(QApplication::translate("VBCDialog", "Down", 0, QApplication::UnicodeUTF8));
        open_list1->setText(QApplication::translate("VBCDialog", "Open List", 0, QApplication::UnicodeUTF8));
        open_dir1->setText(QApplication::translate("VBCDialog", "Open Directory...", 0, QApplication::UnicodeUTF8));
        save_list1->setText(QApplication::translate("VBCDialog", "Save List", 0, QApplication::UnicodeUTF8));
        group2_label->setText(QApplication::translate("VBCDialog", "Group2", 0, QApplication::UnicodeUTF8));
        group2open->setText(QApplication::translate("VBCDialog", "Add", 0, QApplication::UnicodeUTF8));
        group2delete->setText(QString());
        open_list2->setText(QApplication::translate("VBCDialog", "Open List", 0, QApplication::UnicodeUTF8));
        open_dir2->setText(QApplication::translate("VBCDialog", "Open Directory...", 0, QApplication::UnicodeUTF8));
        save_list2->setText(QApplication::translate("VBCDialog", "Save List", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("VBCDialog", "QA threshold", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("VBCDialog", "p-value threshold", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("VBCDialog", "Permutation count:", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("VBCDialog", "Thread count:", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("VBCDialog", "Output file", 0, QApplication::UnicodeUTF8));
        open_mapping->setText(QApplication::translate("VBCDialog", "...", 0, QApplication::UnicodeUTF8));
        load_subject_data->setText(QApplication::translate("VBCDialog", "Calculate", 0, QApplication::UnicodeUTF8));
        close->setText(QApplication::translate("VBCDialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VBCDialog: public Ui_VBCDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBCDIALOG_H
