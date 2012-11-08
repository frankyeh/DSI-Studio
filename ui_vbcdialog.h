/********************************************************************************
** Form generated from reading UI file 'vbcdialog.ui'
**
** Created: Wed Nov 7 17:25:10 2012
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
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QLabel *template_label;
    QSpacerItem *horizontalSpacer_4;
    QDoubleSpinBox *qa_threshold;
    QLabel *label_6;
    QToolButton *open_template;
    QGroupBox *method_group;
    QVBoxLayout *verticalLayout_7;
    QHBoxLayout *horizontalLayout_5;
    QRadioButton *vbc_group;
    QRadioButton *vbc_single;
    QRadioButton *vbc_trend;
    QVBoxLayout *verticalLayout;
    QLabel *group1_label;
    QListView *group1list;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *group1open;
    QPushButton *group1delete;
    QToolButton *moveup;
    QToolButton *movedown;
    QToolButton *open_list1;
    QToolButton *save_list1;
    QToolButton *open_dir1;
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
    QToolButton *save_list2;
    QToolButton *open_dir2;
    QSpacerItem *horizontalSpacer_2;
    QHBoxLayout *horizontalLayout_4;
    QLabel *ODF_label;
    QSpacerItem *horizontalSpacer_6;
    QPushButton *load_subject_data;
    QGroupBox *cluster_group;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_7;
    QSpacerItem *horizontalSpacer_5;
    QLabel *label_4;
    QDoubleSpinBox *fdr;
    QLabel *label_3;
    QDoubleSpinBox *t_threshold;
    QHBoxLayout *horizontalLayout_9;
    QLabel *progress;
    QSpacerItem *horizontalSpacer_7;
    QLabel *label_2;
    QSpinBox *permutation_num;
    QLabel *label;
    QSpinBox *thread_count;
    QToolButton *run_null;
    QToolButton *save_mapping;
    QCustomPlot *report_widget;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *close;

    void setupUi(QDialog *VBCDialog)
    {
        if (VBCDialog->objectName().isEmpty())
            VBCDialog->setObjectName(QString::fromUtf8("VBCDialog"));
        VBCDialog->resize(611, 714);
        verticalLayout_3 = new QVBoxLayout(VBCDialog);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        groupBox = new QGroupBox(VBCDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setContentsMargins(5, 5, 5, 5);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        template_label = new QLabel(groupBox);
        template_label->setObjectName(QString::fromUtf8("template_label"));

        horizontalLayout->addWidget(template_label);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        qa_threshold = new QDoubleSpinBox(groupBox);
        qa_threshold->setObjectName(QString::fromUtf8("qa_threshold"));
        qa_threshold->setMaximumSize(QSize(60, 16777215));
        qa_threshold->setMinimum(0);
        qa_threshold->setMaximum(20);
        qa_threshold->setSingleStep(0.05);
        qa_threshold->setValue(0.15);

        horizontalLayout->addWidget(qa_threshold);

        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout->addWidget(label_6);

        open_template = new QToolButton(groupBox);
        open_template->setObjectName(QString::fromUtf8("open_template"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        open_template->setIcon(icon);

        horizontalLayout->addWidget(open_template);


        verticalLayout_3->addWidget(groupBox);

        method_group = new QGroupBox(VBCDialog);
        method_group->setObjectName(QString::fromUtf8("method_group"));
        verticalLayout_7 = new QVBoxLayout(method_group);
        verticalLayout_7->setSpacing(5);
        verticalLayout_7->setContentsMargins(5, 5, 5, 5);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        vbc_group = new QRadioButton(method_group);
        vbc_group->setObjectName(QString::fromUtf8("vbc_group"));
        vbc_group->setChecked(true);

        horizontalLayout_5->addWidget(vbc_group);

        vbc_single = new QRadioButton(method_group);
        vbc_single->setObjectName(QString::fromUtf8("vbc_single"));

        horizontalLayout_5->addWidget(vbc_single);

        vbc_trend = new QRadioButton(method_group);
        vbc_trend->setObjectName(QString::fromUtf8("vbc_trend"));

        horizontalLayout_5->addWidget(vbc_trend);


        verticalLayout_7->addLayout(horizontalLayout_5);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        group1_label = new QLabel(method_group);
        group1_label->setObjectName(QString::fromUtf8("group1_label"));

        verticalLayout->addWidget(group1_label);

        group1list = new QListView(method_group);
        group1list->setObjectName(QString::fromUtf8("group1list"));
        group1list->setSelectionMode(QAbstractItemView::SingleSelection);

        verticalLayout->addWidget(group1list);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        group1open = new QPushButton(method_group);
        group1open->setObjectName(QString::fromUtf8("group1open"));
        group1open->setMaximumSize(QSize(16777215, 26));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/add.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1open->setIcon(icon1);

        horizontalLayout_2->addWidget(group1open);

        group1delete = new QPushButton(method_group);
        group1delete->setObjectName(QString::fromUtf8("group1delete"));
        group1delete->setMaximumSize(QSize(16777215, 26));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        group1delete->setIcon(icon2);

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

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, -1, -1, -1);
        group2_widget = new QWidget(method_group);
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
        group2open->setIcon(icon1);

        horizontalLayout_3->addWidget(group2open);

        group2delete = new QPushButton(group2_widget);
        group2delete->setObjectName(QString::fromUtf8("group2delete"));
        group2delete->setMaximumSize(QSize(16777215, 26));
        group2delete->setIcon(icon2);

        horizontalLayout_3->addWidget(group2delete);

        open_list2 = new QToolButton(group2_widget);
        open_list2->setObjectName(QString::fromUtf8("open_list2"));
        open_list2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(open_list2);

        save_list2 = new QToolButton(group2_widget);
        save_list2->setObjectName(QString::fromUtf8("save_list2"));
        save_list2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(save_list2);

        open_dir2 = new QToolButton(group2_widget);
        open_dir2->setObjectName(QString::fromUtf8("open_dir2"));
        open_dir2->setMaximumSize(QSize(16777215, 26));

        horizontalLayout_3->addWidget(open_dir2);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        verticalLayout_4->addLayout(horizontalLayout_3);

        group2_label->raise();
        group2list->raise();

        verticalLayout_6->addWidget(group2_widget);


        verticalLayout_7->addLayout(verticalLayout_6);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        ODF_label = new QLabel(method_group);
        ODF_label->setObjectName(QString::fromUtf8("ODF_label"));

        horizontalLayout_4->addWidget(ODF_label);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_6);

        load_subject_data = new QPushButton(method_group);
        load_subject_data->setObjectName(QString::fromUtf8("load_subject_data"));

        horizontalLayout_4->addWidget(load_subject_data);


        verticalLayout_7->addLayout(horizontalLayout_4);


        verticalLayout_3->addWidget(method_group);

        cluster_group = new QGroupBox(VBCDialog);
        cluster_group->setObjectName(QString::fromUtf8("cluster_group"));
        cluster_group->setMinimumSize(QSize(0, 200));
        verticalLayout_5 = new QVBoxLayout(cluster_group);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(5, 5, 5, 5);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_5);

        label_4 = new QLabel(cluster_group);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_7->addWidget(label_4);

        fdr = new QDoubleSpinBox(cluster_group);
        fdr->setObjectName(QString::fromUtf8("fdr"));
        fdr->setMinimum(0);
        fdr->setMaximum(0.2);
        fdr->setSingleStep(0.01);
        fdr->setValue(0.02);

        horizontalLayout_7->addWidget(fdr);

        label_3 = new QLabel(cluster_group);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_7->addWidget(label_3);

        t_threshold = new QDoubleSpinBox(cluster_group);
        t_threshold->setObjectName(QString::fromUtf8("t_threshold"));

        horizontalLayout_7->addWidget(t_threshold);


        verticalLayout_5->addLayout(horizontalLayout_7);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        progress = new QLabel(cluster_group);
        progress->setObjectName(QString::fromUtf8("progress"));

        horizontalLayout_9->addWidget(progress);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_7);

        label_2 = new QLabel(cluster_group);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_9->addWidget(label_2);

        permutation_num = new QSpinBox(cluster_group);
        permutation_num->setObjectName(QString::fromUtf8("permutation_num"));
        permutation_num->setMinimum(50);
        permutation_num->setMaximum(100000);
        permutation_num->setSingleStep(100);
        permutation_num->setValue(1000);

        horizontalLayout_9->addWidget(permutation_num);

        label = new QLabel(cluster_group);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_9->addWidget(label);

        thread_count = new QSpinBox(cluster_group);
        thread_count->setObjectName(QString::fromUtf8("thread_count"));
        thread_count->setMinimum(1);
        thread_count->setMaximum(12);
        thread_count->setValue(4);

        horizontalLayout_9->addWidget(thread_count);

        run_null = new QToolButton(cluster_group);
        run_null->setObjectName(QString::fromUtf8("run_null"));

        horizontalLayout_9->addWidget(run_null);

        save_mapping = new QToolButton(cluster_group);
        save_mapping->setObjectName(QString::fromUtf8("save_mapping"));

        horizontalLayout_9->addWidget(save_mapping);


        verticalLayout_5->addLayout(horizontalLayout_9);

        report_widget = new QCustomPlot(cluster_group);
        report_widget->setObjectName(QString::fromUtf8("report_widget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(report_widget->sizePolicy().hasHeightForWidth());
        report_widget->setSizePolicy(sizePolicy);
        report_widget->setMinimumSize(QSize(0, 20));

        verticalLayout_5->addWidget(report_widget);


        verticalLayout_3->addWidget(cluster_group);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(-1, 0, -1, -1);
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

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
        groupBox->setTitle(QApplication::translate("VBCDialog", "Step 1: Select fiber template", 0, QApplication::UnicodeUTF8));
        template_label->setText(QApplication::translate("VBCDialog", "Open a fiber template for ODF sampling", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("VBCDialog", "QA threshold", 0, QApplication::UnicodeUTF8));
        open_template->setText(QApplication::translate("VBCDialog", "Open...", 0, QApplication::UnicodeUTF8));
        method_group->setTitle(QApplication::translate("VBCDialog", "Step 2: Select subject FIB files", 0, QApplication::UnicodeUTF8));
        vbc_group->setText(QApplication::translate("VBCDialog", "Groupwise ", 0, QApplication::UnicodeUTF8));
        vbc_single->setText(QApplication::translate("VBCDialog", "Single Subject", 0, QApplication::UnicodeUTF8));
        vbc_trend->setText(QApplication::translate("VBCDialog", " Trend Testing", 0, QApplication::UnicodeUTF8));
        group1_label->setText(QApplication::translate("VBCDialog", "Group1", 0, QApplication::UnicodeUTF8));
        group1open->setText(QApplication::translate("VBCDialog", "Add", 0, QApplication::UnicodeUTF8));
        group1delete->setText(QString());
        moveup->setText(QApplication::translate("VBCDialog", "Up", 0, QApplication::UnicodeUTF8));
        movedown->setText(QApplication::translate("VBCDialog", "Down", 0, QApplication::UnicodeUTF8));
        open_list1->setText(QApplication::translate("VBCDialog", "Open List", 0, QApplication::UnicodeUTF8));
        save_list1->setText(QApplication::translate("VBCDialog", "Save List", 0, QApplication::UnicodeUTF8));
        open_dir1->setText(QApplication::translate("VBCDialog", "Search in Directory...", 0, QApplication::UnicodeUTF8));
        group2_label->setText(QApplication::translate("VBCDialog", "Group2", 0, QApplication::UnicodeUTF8));
        group2open->setText(QApplication::translate("VBCDialog", "Add", 0, QApplication::UnicodeUTF8));
        group2delete->setText(QString());
        open_list2->setText(QApplication::translate("VBCDialog", "Open List", 0, QApplication::UnicodeUTF8));
        save_list2->setText(QApplication::translate("VBCDialog", "Save List", 0, QApplication::UnicodeUTF8));
        open_dir2->setText(QApplication::translate("VBCDialog", "Search in Directory...", 0, QApplication::UnicodeUTF8));
        ODF_label->setText(QApplication::translate("VBCDialog", "Load subject ODF information", 0, QApplication::UnicodeUTF8));
        load_subject_data->setText(QApplication::translate("VBCDialog", "Load data", 0, QApplication::UnicodeUTF8));
        cluster_group->setTitle(QApplication::translate("VBCDialog", "Step 3: Get null distribution of the maximum fiber length", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("VBCDialog", "False discorvery rate", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("VBCDialog", "Significance threshold", 0, QApplication::UnicodeUTF8));
        progress->setText(QApplication::translate("VBCDialog", "Progress", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("VBCDialog", "Permutation:", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("VBCDialog", "Thread count:", 0, QApplication::UnicodeUTF8));
        run_null->setText(QApplication::translate("VBCDialog", "Run null", 0, QApplication::UnicodeUTF8));
        save_mapping->setText(QApplication::translate("VBCDialog", "Save Mapping...", 0, QApplication::UnicodeUTF8));
        close->setText(QApplication::translate("VBCDialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VBCDialog: public Ui_VBCDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBCDIALOG_H
