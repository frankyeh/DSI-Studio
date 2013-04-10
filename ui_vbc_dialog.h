/********************************************************************************
** Form generated from reading UI file 'vbc_dialog.ui'
**
** Created: Wed Apr 10 12:41:28 2013
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VBC_DIALOG_H
#define UI_VBC_DIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFormLayout>
#include <QtGui/QGraphicsView>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QScrollBar>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QSplitter>
#include <QtGui/QTabWidget>
#include <QtGui/QTableWidget>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "plot/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_vbc_dialog
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_9;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QRadioButton *Individual;
    QRadioButton *radioButton_4;
    QRadioButton *radioButton_2;
    QRadioButton *Trend;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_13;
    QHBoxLayout *horizontalLayout_4;
    QToolButton *cal_FDR;
    QLabel *open_instruction;
    QSpacerItem *horizontalSpacer;
    QLabel *label_2;
    QDoubleSpinBox *percentile_rank;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_7;
    QSplitter *splitter_3;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_4;
    QTabWidget *tabWidget_2;
    QWidget *tab;
    QHBoxLayout *horizontalLayout_2;
    QCustomPlot *null_dist;
    QVBoxLayout *verticalLayout_8;
    QFormLayout *formLayout;
    QLabel *label_3;
    QSpinBox *span_to;
    QLabel *label_4;
    QSpinBox *span_from;
    QLabel *label_5;
    QDoubleSpinBox *max_prob;
    QLabel *label;
    QSpinBox *line_width;
    QCheckBox *show_lesser;
    QCheckBox *show_greater;
    QCheckBox *show_null_lesser;
    QCheckBox *show_null_greater;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_10;
    QHBoxLayout *horizontalLayout_5;
    QToolButton *save_vbc_dist;
    QSpacerItem *horizontalSpacer_2;
    QTableWidget *dist_table;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_12;
    QTabWidget *tabWidget_3;
    QWidget *tab_5;
    QHBoxLayout *horizontalLayout_3;
    QCustomPlot *fdr_dist;
    QVBoxLayout *verticalLayout_11;
    QFormLayout *formLayout_2;
    QLabel *label_6;
    QSpinBox *span_to_2;
    QLabel *label_7;
    QSpinBox *span_from_2;
    QLabel *label_8;
    QDoubleSpinBox *max_prob_2;
    QLabel *label_9;
    QSpinBox *line_width_2;
    QCheckBox *show_lesser_2;
    QCheckBox *show_greater_2;
    QWidget *tab_6;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_6;
    QToolButton *save_fdr_dist;
    QSpacerItem *horizontalSpacer_3;
    QTableWidget *fdr_table;
    QGroupBox *individual_study;
    QVBoxLayout *verticalLayout_14;
    QHBoxLayout *horizontalLayout_21;
    QSpacerItem *horizontalSpacer_9;
    QPushButton *open_subject;
    QPushButton *cal_lesser_tracts;
    QPushButton *vbc_dist_update;
    QWidget *tab_4;
    QVBoxLayout *verticalLayout_6;
    QSplitter *splitter;
    QWidget *widget_3;
    QVBoxLayout *verticalLayout_3;
    QSplitter *splitter_2;
    QWidget *widget;
    QVBoxLayout *verticalLayout_2;
    QGraphicsView *vbc_view;
    QScrollBar *AxiSlider;
    QTableWidget *subject_list;
    QCustomPlot *vbc_report;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *vbc_dialog)
    {
        if (vbc_dialog->objectName().isEmpty())
            vbc_dialog->setObjectName(QString::fromUtf8("vbc_dialog"));
        vbc_dialog->resize(640, 480);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/axial.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        vbc_dialog->setWindowIcon(icon);
        verticalLayout = new QVBoxLayout(vbc_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(vbc_dialog);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_9 = new QVBoxLayout(tab_3);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        groupBox = new QGroupBox(tab_3);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy);
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        Individual = new QRadioButton(groupBox);
        Individual->setObjectName(QString::fromUtf8("Individual"));
        Individual->setChecked(true);

        horizontalLayout->addWidget(Individual);

        radioButton_4 = new QRadioButton(groupBox);
        radioButton_4->setObjectName(QString::fromUtf8("radioButton_4"));
        radioButton_4->setEnabled(false);

        horizontalLayout->addWidget(radioButton_4);

        radioButton_2 = new QRadioButton(groupBox);
        radioButton_2->setObjectName(QString::fromUtf8("radioButton_2"));
        radioButton_2->setEnabled(false);

        horizontalLayout->addWidget(radioButton_2);

        Trend = new QRadioButton(groupBox);
        Trend->setObjectName(QString::fromUtf8("Trend"));

        horizontalLayout->addWidget(Trend);


        verticalLayout_9->addWidget(groupBox);

        groupBox_4 = new QGroupBox(tab_3);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_13 = new QVBoxLayout(groupBox_4);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        cal_FDR = new QToolButton(groupBox_4);
        cal_FDR->setObjectName(QString::fromUtf8("cal_FDR"));
        cal_FDR->setMaximumSize(QSize(23, 22));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        cal_FDR->setIcon(icon1);

        horizontalLayout_4->addWidget(cal_FDR);

        open_instruction = new QLabel(groupBox_4);
        open_instruction->setObjectName(QString::fromUtf8("open_instruction"));

        horizontalLayout_4->addWidget(open_instruction);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        label_2 = new QLabel(groupBox_4);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_4->addWidget(label_2);

        percentile_rank = new QDoubleSpinBox(groupBox_4);
        percentile_rank->setObjectName(QString::fromUtf8("percentile_rank"));
        percentile_rank->setDecimals(3);
        percentile_rank->setMinimum(0.001);
        percentile_rank->setMaximum(0.5);
        percentile_rank->setSingleStep(0.05);
        percentile_rank->setValue(0.05);

        horizontalLayout_4->addWidget(percentile_rank);


        verticalLayout_13->addLayout(horizontalLayout_4);


        verticalLayout_9->addWidget(groupBox_4);

        widget_2 = new QWidget(tab_3);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(widget_2->sizePolicy().hasHeightForWidth());
        widget_2->setSizePolicy(sizePolicy1);
        verticalLayout_7 = new QVBoxLayout(widget_2);
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        splitter_3 = new QSplitter(widget_2);
        splitter_3->setObjectName(QString::fromUtf8("splitter_3"));
        splitter_3->setOrientation(Qt::Horizontal);
        groupBox_2 = new QGroupBox(splitter_3);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_4 = new QVBoxLayout(groupBox_2);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        tabWidget_2 = new QTabWidget(groupBox_2);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        horizontalLayout_2 = new QHBoxLayout(tab);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        null_dist = new QCustomPlot(tab);
        null_dist->setObjectName(QString::fromUtf8("null_dist"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(null_dist->sizePolicy().hasHeightForWidth());
        null_dist->setSizePolicy(sizePolicy2);

        horizontalLayout_2->addWidget(null_dist);

        verticalLayout_8 = new QVBoxLayout();
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        formLayout->setHorizontalSpacing(0);
        label_3 = new QLabel(tab);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_3);

        span_to = new QSpinBox(tab);
        span_to->setObjectName(QString::fromUtf8("span_to"));
        span_to->setValue(40);

        formLayout->setWidget(1, QFormLayout::FieldRole, span_to);

        label_4 = new QLabel(tab);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_4);

        span_from = new QSpinBox(tab);
        span_from->setObjectName(QString::fromUtf8("span_from"));
        span_from->setMinimum(2);

        formLayout->setWidget(0, QFormLayout::FieldRole, span_from);

        label_5 = new QLabel(tab);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_5);

        max_prob = new QDoubleSpinBox(tab);
        max_prob->setObjectName(QString::fromUtf8("max_prob"));
        max_prob->setDecimals(3);
        max_prob->setMinimum(0);
        max_prob->setMaximum(1);
        max_prob->setSingleStep(0.01);
        max_prob->setValue(0.4);

        formLayout->setWidget(2, QFormLayout::FieldRole, max_prob);

        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label);

        line_width = new QSpinBox(tab);
        line_width->setObjectName(QString::fromUtf8("line_width"));
        line_width->setMinimum(1);
        line_width->setMaximum(5);

        formLayout->setWidget(3, QFormLayout::FieldRole, line_width);


        verticalLayout_8->addLayout(formLayout);

        show_lesser = new QCheckBox(tab);
        show_lesser->setObjectName(QString::fromUtf8("show_lesser"));
        show_lesser->setChecked(true);

        verticalLayout_8->addWidget(show_lesser);

        show_greater = new QCheckBox(tab);
        show_greater->setObjectName(QString::fromUtf8("show_greater"));
        show_greater->setChecked(true);

        verticalLayout_8->addWidget(show_greater);

        show_null_lesser = new QCheckBox(tab);
        show_null_lesser->setObjectName(QString::fromUtf8("show_null_lesser"));
        show_null_lesser->setChecked(true);

        verticalLayout_8->addWidget(show_null_lesser);

        show_null_greater = new QCheckBox(tab);
        show_null_greater->setObjectName(QString::fromUtf8("show_null_greater"));
        show_null_greater->setChecked(true);

        verticalLayout_8->addWidget(show_null_greater);


        horizontalLayout_2->addLayout(verticalLayout_8);

        tabWidget_2->addTab(tab, QString());
        null_dist->raise();
        null_dist->raise();
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_10 = new QVBoxLayout(tab_2);
        verticalLayout_10->setSpacing(0);
        verticalLayout_10->setContentsMargins(0, 0, 0, 0);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        save_vbc_dist = new QToolButton(tab_2);
        save_vbc_dist->setObjectName(QString::fromUtf8("save_vbc_dist"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/save.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        save_vbc_dist->setIcon(icon2);

        horizontalLayout_5->addWidget(save_vbc_dist);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_2);


        verticalLayout_10->addLayout(horizontalLayout_5);

        dist_table = new QTableWidget(tab_2);
        if (dist_table->columnCount() < 4)
            dist_table->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(3, __qtablewidgetitem3);
        dist_table->setObjectName(QString::fromUtf8("dist_table"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(dist_table->sizePolicy().hasHeightForWidth());
        dist_table->setSizePolicy(sizePolicy3);

        verticalLayout_10->addWidget(dist_table);

        tabWidget_2->addTab(tab_2, QString());

        verticalLayout_4->addWidget(tabWidget_2);

        splitter_3->addWidget(groupBox_2);
        groupBox_3 = new QGroupBox(splitter_3);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout_12 = new QVBoxLayout(groupBox_3);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        tabWidget_3 = new QTabWidget(groupBox_3);
        tabWidget_3->setObjectName(QString::fromUtf8("tabWidget_3"));
        tab_5 = new QWidget();
        tab_5->setObjectName(QString::fromUtf8("tab_5"));
        horizontalLayout_3 = new QHBoxLayout(tab_5);
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        fdr_dist = new QCustomPlot(tab_5);
        fdr_dist->setObjectName(QString::fromUtf8("fdr_dist"));
        sizePolicy2.setHeightForWidth(fdr_dist->sizePolicy().hasHeightForWidth());
        fdr_dist->setSizePolicy(sizePolicy2);

        horizontalLayout_3->addWidget(fdr_dist);

        verticalLayout_11 = new QVBoxLayout();
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        formLayout_2 = new QFormLayout();
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        formLayout_2->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        formLayout_2->setHorizontalSpacing(0);
        label_6 = new QLabel(tab_5);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, label_6);

        span_to_2 = new QSpinBox(tab_5);
        span_to_2->setObjectName(QString::fromUtf8("span_to_2"));
        span_to_2->setValue(40);

        formLayout_2->setWidget(1, QFormLayout::FieldRole, span_to_2);

        label_7 = new QLabel(tab_5);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, label_7);

        span_from_2 = new QSpinBox(tab_5);
        span_from_2->setObjectName(QString::fromUtf8("span_from_2"));
        span_from_2->setMinimum(2);

        formLayout_2->setWidget(0, QFormLayout::FieldRole, span_from_2);

        label_8 = new QLabel(tab_5);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        formLayout_2->setWidget(2, QFormLayout::LabelRole, label_8);

        max_prob_2 = new QDoubleSpinBox(tab_5);
        max_prob_2->setObjectName(QString::fromUtf8("max_prob_2"));
        max_prob_2->setDecimals(3);
        max_prob_2->setMinimum(0);
        max_prob_2->setMaximum(1.5);
        max_prob_2->setSingleStep(0.1);
        max_prob_2->setValue(0.4);

        formLayout_2->setWidget(2, QFormLayout::FieldRole, max_prob_2);

        label_9 = new QLabel(tab_5);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        formLayout_2->setWidget(3, QFormLayout::LabelRole, label_9);

        line_width_2 = new QSpinBox(tab_5);
        line_width_2->setObjectName(QString::fromUtf8("line_width_2"));
        line_width_2->setMinimum(1);
        line_width_2->setMaximum(5);

        formLayout_2->setWidget(3, QFormLayout::FieldRole, line_width_2);


        verticalLayout_11->addLayout(formLayout_2);

        show_lesser_2 = new QCheckBox(tab_5);
        show_lesser_2->setObjectName(QString::fromUtf8("show_lesser_2"));
        show_lesser_2->setChecked(true);

        verticalLayout_11->addWidget(show_lesser_2);

        show_greater_2 = new QCheckBox(tab_5);
        show_greater_2->setObjectName(QString::fromUtf8("show_greater_2"));
        show_greater_2->setChecked(true);

        verticalLayout_11->addWidget(show_greater_2);


        horizontalLayout_3->addLayout(verticalLayout_11);

        tabWidget_3->addTab(tab_5, QString());
        tab_6 = new QWidget();
        tab_6->setObjectName(QString::fromUtf8("tab_6"));
        verticalLayout_5 = new QVBoxLayout(tab_6);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, -1, -1, -1);
        save_fdr_dist = new QToolButton(tab_6);
        save_fdr_dist->setObjectName(QString::fromUtf8("save_fdr_dist"));
        save_fdr_dist->setIcon(icon2);

        horizontalLayout_6->addWidget(save_fdr_dist);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);


        verticalLayout_5->addLayout(horizontalLayout_6);

        fdr_table = new QTableWidget(tab_6);
        if (fdr_table->columnCount() < 4)
            fdr_table->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        fdr_table->setHorizontalHeaderItem(0, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        fdr_table->setHorizontalHeaderItem(1, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        fdr_table->setHorizontalHeaderItem(2, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        fdr_table->setHorizontalHeaderItem(3, __qtablewidgetitem7);
        fdr_table->setObjectName(QString::fromUtf8("fdr_table"));
        sizePolicy3.setHeightForWidth(fdr_table->sizePolicy().hasHeightForWidth());
        fdr_table->setSizePolicy(sizePolicy3);

        verticalLayout_5->addWidget(fdr_table);

        tabWidget_3->addTab(tab_6, QString());

        verticalLayout_12->addWidget(tabWidget_3);

        splitter_3->addWidget(groupBox_3);
        groupBox_2->raise();
        tabWidget_3->raise();

        verticalLayout_7->addWidget(splitter_3);


        verticalLayout_9->addWidget(widget_2);

        individual_study = new QGroupBox(tab_3);
        individual_study->setObjectName(QString::fromUtf8("individual_study"));
        verticalLayout_14 = new QVBoxLayout(individual_study);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        horizontalLayout_21 = new QHBoxLayout();
        horizontalLayout_21->setSpacing(0);
        horizontalLayout_21->setObjectName(QString::fromUtf8("horizontalLayout_21"));
        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_21->addItem(horizontalSpacer_9);

        open_subject = new QPushButton(individual_study);
        open_subject->setObjectName(QString::fromUtf8("open_subject"));

        horizontalLayout_21->addWidget(open_subject);

        cal_lesser_tracts = new QPushButton(individual_study);
        cal_lesser_tracts->setObjectName(QString::fromUtf8("cal_lesser_tracts"));
        cal_lesser_tracts->setMaximumSize(QSize(40, 16777215));

        horizontalLayout_21->addWidget(cal_lesser_tracts);

        vbc_dist_update = new QPushButton(individual_study);
        vbc_dist_update->setObjectName(QString::fromUtf8("vbc_dist_update"));
        vbc_dist_update->setMaximumSize(QSize(40, 16777215));

        horizontalLayout_21->addWidget(vbc_dist_update);


        verticalLayout_14->addLayout(horizontalLayout_21);


        verticalLayout_9->addWidget(individual_study);

        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        verticalLayout_6 = new QVBoxLayout(tab_4);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        splitter = new QSplitter(tab_4);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Vertical);
        widget_3 = new QWidget(splitter);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        verticalLayout_3 = new QVBoxLayout(widget_3);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        splitter_2 = new QSplitter(widget_3);
        splitter_2->setObjectName(QString::fromUtf8("splitter_2"));
        splitter_2->setOrientation(Qt::Horizontal);
        widget = new QWidget(splitter_2);
        widget->setObjectName(QString::fromUtf8("widget"));
        verticalLayout_2 = new QVBoxLayout(widget);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        vbc_view = new QGraphicsView(widget);
        vbc_view->setObjectName(QString::fromUtf8("vbc_view"));

        verticalLayout_2->addWidget(vbc_view);

        AxiSlider = new QScrollBar(widget);
        AxiSlider->setObjectName(QString::fromUtf8("AxiSlider"));
        AxiSlider->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(AxiSlider);

        splitter_2->addWidget(widget);
        subject_list = new QTableWidget(splitter_2);
        if (subject_list->columnCount() < 3)
            subject_list->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(0, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(1, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(2, __qtablewidgetitem10);
        subject_list->setObjectName(QString::fromUtf8("subject_list"));
        QSizePolicy sizePolicy4(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(subject_list->sizePolicy().hasHeightForWidth());
        subject_list->setSizePolicy(sizePolicy4);
        subject_list->setSelectionMode(QAbstractItemView::SingleSelection);
        subject_list->setSelectionBehavior(QAbstractItemView::SelectRows);
        splitter_2->addWidget(subject_list);

        verticalLayout_3->addWidget(splitter_2);

        splitter->addWidget(widget_3);
        vbc_report = new QCustomPlot(splitter);
        vbc_report->setObjectName(QString::fromUtf8("vbc_report"));
        splitter->addWidget(vbc_report);

        verticalLayout_6->addWidget(splitter);

        tabWidget->addTab(tab_4, QString());

        verticalLayout->addWidget(tabWidget);

        buttonBox = new QDialogButtonBox(vbc_dialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Close);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(vbc_dialog);

        tabWidget->setCurrentIndex(0);
        tabWidget_2->setCurrentIndex(0);
        tabWidget_3->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(vbc_dialog);
    } // setupUi

    void retranslateUi(QDialog *vbc_dialog)
    {
        vbc_dialog->setWindowTitle(QApplication::translate("vbc_dialog", "Connectometry", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("vbc_dialog", "STEP 1: Selection Analysis Approach", 0, QApplication::UnicodeUTF8));
        Individual->setText(QApplication::translate("vbc_dialog", "Individual", 0, QApplication::UnicodeUTF8));
        radioButton_4->setText(QApplication::translate("vbc_dialog", "Paired Individual", 0, QApplication::UnicodeUTF8));
        radioButton_2->setText(QApplication::translate("vbc_dialog", "Group", 0, QApplication::UnicodeUTF8));
        Trend->setText(QApplication::translate("vbc_dialog", "Trend", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("vbc_dialog", "STEP 2: Load Data", 0, QApplication::UnicodeUTF8));
        cal_FDR->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        open_instruction->setText(QApplication::translate("vbc_dialog", "Open", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("vbc_dialog", "Percentile rank", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("vbc_dialog", " Empirical Distribution", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("vbc_dialog", "Span to", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("vbc_dialog", "Span from", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("vbc_dialog", "Max prob", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("vbc_dialog", "Line width", 0, QApplication::UnicodeUTF8));
        show_lesser->setText(QApplication::translate("vbc_dialog", "lesser", 0, QApplication::UnicodeUTF8));
        show_greater->setText(QApplication::translate("vbc_dialog", "greater", 0, QApplication::UnicodeUTF8));
        show_null_lesser->setText(QApplication::translate("vbc_dialog", "null lesser", 0, QApplication::UnicodeUTF8));
        show_null_greater->setText(QApplication::translate("vbc_dialog", "null greater", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QApplication::translate("vbc_dialog", "Plot", 0, QApplication::UnicodeUTF8));
        save_vbc_dist->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = dist_table->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("vbc_dialog", "span", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = dist_table->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("vbc_dialog", "pdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = dist_table->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("vbc_dialog", "cdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = dist_table->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_2), QApplication::translate("vbc_dialog", "Data", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("vbc_dialog", "Span to", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("vbc_dialog", "Span from", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("vbc_dialog", "Max prob", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("vbc_dialog", "Line width", 0, QApplication::UnicodeUTF8));
        show_lesser_2->setText(QApplication::translate("vbc_dialog", "lesser", 0, QApplication::UnicodeUTF8));
        show_greater_2->setText(QApplication::translate("vbc_dialog", "greater", 0, QApplication::UnicodeUTF8));
        tabWidget_3->setTabText(tabWidget_3->indexOf(tab_5), QApplication::translate("vbc_dialog", "Plot", 0, QApplication::UnicodeUTF8));
        save_fdr_dist->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = fdr_table->horizontalHeaderItem(0);
        ___qtablewidgetitem4->setText(QApplication::translate("vbc_dialog", "span", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem5 = fdr_table->horizontalHeaderItem(1);
        ___qtablewidgetitem5->setText(QApplication::translate("vbc_dialog", "pdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem6 = fdr_table->horizontalHeaderItem(2);
        ___qtablewidgetitem6->setText(QApplication::translate("vbc_dialog", "cdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem7 = fdr_table->horizontalHeaderItem(3);
        ___qtablewidgetitem7->setText(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        tabWidget_3->setTabText(tabWidget_3->indexOf(tab_6), QApplication::translate("vbc_dialog", "Data", 0, QApplication::UnicodeUTF8));
        individual_study->setTitle(QApplication::translate("vbc_dialog", "STEP 3: Individual Study", 0, QApplication::UnicodeUTF8));
        open_subject->setText(QApplication::translate("vbc_dialog", "open...", 0, QApplication::UnicodeUTF8));
        cal_lesser_tracts->setText(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        vbc_dist_update->setText(QApplication::translate("vbc_dialog", "Dist", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("vbc_dialog", "Analysis", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem8 = subject_list->horizontalHeaderItem(0);
        ___qtablewidgetitem8->setText(QApplication::translate("vbc_dialog", "Subject ID", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem9 = subject_list->horizontalHeaderItem(1);
        ___qtablewidgetitem9->setText(QApplication::translate("vbc_dialog", "Value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem10 = subject_list->horizontalHeaderItem(2);
        ___qtablewidgetitem10->setText(QApplication::translate("vbc_dialog", "R2", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("vbc_dialog", "Database", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class vbc_dialog: public Ui_vbc_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBC_DIALOG_H
