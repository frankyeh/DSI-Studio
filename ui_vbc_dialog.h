/********************************************************************************
** Form generated from reading UI file 'vbc_dialog.ui'
**
** Created: Tue Sep 23 18:21:13 2014
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
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGraphicsView>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QListView>
#include <QtGui/QProgressBar>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QScrollBar>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QSplitter>
#include <QtGui/QTabWidget>
#include <QtGui/QTableWidget>
#include <QtGui/QTextBrowser>
#include <QtGui/QToolBox>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "plot/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_vbc_dialog
{
public:
    QVBoxLayout *verticalLayout;
    QToolBox *toolBox;
    QWidget *page;
    QVBoxLayout *verticalLayout_24;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *save_name_list;
    QSpacerItem *horizontalSpacer;
    QSplitter *splitter;
    QWidget *widget_3;
    QVBoxLayout *verticalLayout_6;
    QSplitter *splitter_5;
    QWidget *widget_4;
    QVBoxLayout *verticalLayout_13;
    QHBoxLayout *horizontalLayout_17;
    QLabel *label_13;
    QDoubleSpinBox *zoom;
    QLabel *coordinate;
    QSpacerItem *horizontalSpacer_6;
    QGraphicsView *vbc_view;
    QScrollBar *AxiSlider;
    QTableWidget *subject_list;
    QCustomPlot *vbc_report;
    QWidget *page_2;
    QVBoxLayout *verticalLayout_23;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_21;
    QRadioButton *rb_multiple_regression;
    QRadioButton *rb_group_difference;
    QRadioButton *rb_paired_difference;
    QRadioButton *rb_individual_analysis;
    QGroupBox *percentile_rank_group;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *percentile_rank_layout;
    QLabel *label_5;
    QDoubleSpinBox *percentile;
    QLabel *label_8;
    QHBoxLayout *horizontalLayout_11;
    QPushButton *advanced_options;
    QSpacerItem *horizontalSpacer_3;
    QGroupBox *advanced_options_box;
    QVBoxLayout *verticalLayout_18;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_4;
    QSpinBox *mr_permutation;
    QHBoxLayout *horizontalLayout;
    QLabel *label_3;
    QSpinBox *length_threshold;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_7;
    QSpinBox *pruning;
    QSpacerItem *verticalSpacer_2;
    QWidget *page_4;
    QVBoxLayout *verticalLayout_19;
    QGroupBox *individual_demo;
    QVBoxLayout *verticalLayout_14;
    QHBoxLayout *horizontalLayout_9;
    QToolButton *open_files;
    QLabel *open_instruction;
    QSpacerItem *horizontalSpacer_5;
    QListView *individual_list;
    QGroupBox *multiple_regression_demo;
    QVBoxLayout *verticalLayout_17;
    QHBoxLayout *horizontalLayout_10;
    QToolButton *open_mr_files;
    QLabel *open_instruction_2;
    QSpacerItem *horizontalSpacer_4;
    QWidget *regression_feature;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_6;
    QComboBox *foi;
    QWidget *foi_widget;
    QHBoxLayout *horizontalLayout_7;
    QTableWidget *subject_demo;
    QWidget *page_3;
    QVBoxLayout *verticalLayout_7;
    QTabWidget *tabWidget_2;
    QWidget *tab;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_16;
    QCheckBox *show_null_greater;
    QCheckBox *show_null_lesser;
    QCheckBox *show_greater;
    QCheckBox *show_lesser;
    QCheckBox *show_lesser_2;
    QCheckBox *show_greater_2;
    QLabel *label;
    QSpinBox *span_to;
    QHBoxLayout *horizontalLayout_2;
    QCustomPlot *null_dist;
    QCustomPlot *fdr_dist;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_10;
    QHBoxLayout *horizontalLayout_5;
    QToolButton *save_vbc_dist;
    QToolButton *save_fdr_dist;
    QSpacerItem *horizontalSpacer_2;
    QTableWidget *dist_table;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_5;
    QTextBrowser *textBrowser;
    QHBoxLayout *horizontalLayout_3;
    QProgressBar *progressBar;
    QSpinBox *multithread;
    QLabel *label_2;
    QPushButton *run;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_9;

    void setupUi(QDialog *vbc_dialog)
    {
        if (vbc_dialog->objectName().isEmpty())
            vbc_dialog->setObjectName(QString::fromUtf8("vbc_dialog"));
        vbc_dialog->resize(601, 476);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/axial.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        vbc_dialog->setWindowIcon(icon);
        verticalLayout = new QVBoxLayout(vbc_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        toolBox = new QToolBox(vbc_dialog);
        toolBox->setObjectName(QString::fromUtf8("toolBox"));
        page = new QWidget();
        page->setObjectName(QString::fromUtf8("page"));
        page->setGeometry(QRect(0, 0, 583, 368));
        verticalLayout_24 = new QVBoxLayout(page);
        verticalLayout_24->setObjectName(QString::fromUtf8("verticalLayout_24"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        save_name_list = new QPushButton(page);
        save_name_list->setObjectName(QString::fromUtf8("save_name_list"));

        horizontalLayout_4->addWidget(save_name_list);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);


        verticalLayout_24->addLayout(horizontalLayout_4);

        splitter = new QSplitter(page);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Vertical);
        widget_3 = new QWidget(splitter);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        verticalLayout_6 = new QVBoxLayout(widget_3);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        splitter_5 = new QSplitter(widget_3);
        splitter_5->setObjectName(QString::fromUtf8("splitter_5"));
        splitter_5->setOrientation(Qt::Horizontal);
        widget_4 = new QWidget(splitter_5);
        widget_4->setObjectName(QString::fromUtf8("widget_4"));
        verticalLayout_13 = new QVBoxLayout(widget_4);
        verticalLayout_13->setSpacing(0);
        verticalLayout_13->setContentsMargins(0, 0, 0, 0);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        label_13 = new QLabel(widget_4);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_17->addWidget(label_13);

        zoom = new QDoubleSpinBox(widget_4);
        zoom->setObjectName(QString::fromUtf8("zoom"));
        zoom->setMinimum(0.1);
        zoom->setMaximum(20);
        zoom->setValue(3);

        horizontalLayout_17->addWidget(zoom);

        coordinate = new QLabel(widget_4);
        coordinate->setObjectName(QString::fromUtf8("coordinate"));

        horizontalLayout_17->addWidget(coordinate);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_17->addItem(horizontalSpacer_6);


        verticalLayout_13->addLayout(horizontalLayout_17);

        vbc_view = new QGraphicsView(widget_4);
        vbc_view->setObjectName(QString::fromUtf8("vbc_view"));

        verticalLayout_13->addWidget(vbc_view);

        AxiSlider = new QScrollBar(widget_4);
        AxiSlider->setObjectName(QString::fromUtf8("AxiSlider"));
        AxiSlider->setOrientation(Qt::Horizontal);

        verticalLayout_13->addWidget(AxiSlider);

        splitter_5->addWidget(widget_4);
        subject_list = new QTableWidget(splitter_5);
        if (subject_list->columnCount() < 3)
            subject_list->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        subject_list->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        subject_list->setObjectName(QString::fromUtf8("subject_list"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(subject_list->sizePolicy().hasHeightForWidth());
        subject_list->setSizePolicy(sizePolicy);
        subject_list->setSelectionMode(QAbstractItemView::SingleSelection);
        subject_list->setSelectionBehavior(QAbstractItemView::SelectRows);
        splitter_5->addWidget(subject_list);

        verticalLayout_6->addWidget(splitter_5);

        splitter->addWidget(widget_3);
        vbc_report = new QCustomPlot(splitter);
        vbc_report->setObjectName(QString::fromUtf8("vbc_report"));
        splitter->addWidget(vbc_report);

        verticalLayout_24->addWidget(splitter);

        toolBox->addItem(page, QString::fromUtf8("Source Data"));
        page_2 = new QWidget();
        page_2->setObjectName(QString::fromUtf8("page_2"));
        page_2->setGeometry(QRect(0, 0, 583, 368));
        verticalLayout_23 = new QVBoxLayout(page_2);
        verticalLayout_23->setObjectName(QString::fromUtf8("verticalLayout_23"));
        groupBox_4 = new QGroupBox(page_2);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_21 = new QVBoxLayout(groupBox_4);
        verticalLayout_21->setObjectName(QString::fromUtf8("verticalLayout_21"));
        rb_multiple_regression = new QRadioButton(groupBox_4);
        rb_multiple_regression->setObjectName(QString::fromUtf8("rb_multiple_regression"));
        rb_multiple_regression->setChecked(true);

        verticalLayout_21->addWidget(rb_multiple_regression);

        rb_group_difference = new QRadioButton(groupBox_4);
        rb_group_difference->setObjectName(QString::fromUtf8("rb_group_difference"));
        rb_group_difference->setEnabled(true);

        verticalLayout_21->addWidget(rb_group_difference);

        rb_paired_difference = new QRadioButton(groupBox_4);
        rb_paired_difference->setObjectName(QString::fromUtf8("rb_paired_difference"));
        rb_paired_difference->setEnabled(false);

        verticalLayout_21->addWidget(rb_paired_difference);

        rb_individual_analysis = new QRadioButton(groupBox_4);
        rb_individual_analysis->setObjectName(QString::fromUtf8("rb_individual_analysis"));

        verticalLayout_21->addWidget(rb_individual_analysis);


        verticalLayout_23->addWidget(groupBox_4);

        percentile_rank_group = new QGroupBox(page_2);
        percentile_rank_group->setObjectName(QString::fromUtf8("percentile_rank_group"));
        verticalLayout_2 = new QVBoxLayout(percentile_rank_group);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        percentile_rank_layout = new QHBoxLayout();
        percentile_rank_layout->setObjectName(QString::fromUtf8("percentile_rank_layout"));
        label_5 = new QLabel(percentile_rank_group);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        percentile_rank_layout->addWidget(label_5);

        percentile = new QDoubleSpinBox(percentile_rank_group);
        percentile->setObjectName(QString::fromUtf8("percentile"));
        percentile->setMaximumSize(QSize(75, 16777215));
        percentile->setDecimals(4);
        percentile->setMinimum(0);
        percentile->setMaximum(1);
        percentile->setSingleStep(0.01);
        percentile->setValue(0.02);

        percentile_rank_layout->addWidget(percentile);


        verticalLayout_2->addLayout(percentile_rank_layout);

        label_8 = new QLabel(percentile_rank_group);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        verticalLayout_2->addWidget(label_8);


        verticalLayout_23->addWidget(percentile_rank_group);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        advanced_options = new QPushButton(page_2);
        advanced_options->setObjectName(QString::fromUtf8("advanced_options"));

        horizontalLayout_11->addWidget(advanced_options);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_3);


        verticalLayout_23->addLayout(horizontalLayout_11);

        advanced_options_box = new QGroupBox(page_2);
        advanced_options_box->setObjectName(QString::fromUtf8("advanced_options_box"));
        advanced_options_box->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout_18 = new QVBoxLayout(advanced_options_box);
        verticalLayout_18->setSpacing(0);
        verticalLayout_18->setObjectName(QString::fromUtf8("verticalLayout_18"));
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_4 = new QLabel(advanced_options_box);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_12->addWidget(label_4);

        mr_permutation = new QSpinBox(advanced_options_box);
        mr_permutation->setObjectName(QString::fromUtf8("mr_permutation"));
        mr_permutation->setMaximumSize(QSize(75, 16777215));
        mr_permutation->setMinimum(50);
        mr_permutation->setMaximum(10000);
        mr_permutation->setSingleStep(100);
        mr_permutation->setValue(500);

        horizontalLayout_12->addWidget(mr_permutation);


        verticalLayout_18->addLayout(horizontalLayout_12);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_3 = new QLabel(advanced_options_box);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout->addWidget(label_3);

        length_threshold = new QSpinBox(advanced_options_box);
        length_threshold->setObjectName(QString::fromUtf8("length_threshold"));
        length_threshold->setMaximumSize(QSize(75, 16777215));
        length_threshold->setMinimum(10);
        length_threshold->setMaximum(100);
        length_threshold->setValue(40);

        horizontalLayout->addWidget(length_threshold);


        verticalLayout_18->addLayout(horizontalLayout);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_7 = new QLabel(advanced_options_box);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_8->addWidget(label_7);

        pruning = new QSpinBox(advanced_options_box);
        pruning->setObjectName(QString::fromUtf8("pruning"));
        pruning->setMaximumSize(QSize(75, 16777215));
        pruning->setMaximum(100);
        pruning->setValue(10);

        horizontalLayout_8->addWidget(pruning);


        verticalLayout_18->addLayout(horizontalLayout_8);


        verticalLayout_23->addWidget(advanced_options_box);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_23->addItem(verticalSpacer_2);

        toolBox->addItem(page_2, QString::fromUtf8("STEP1: Select analysis model"));
        page_4 = new QWidget();
        page_4->setObjectName(QString::fromUtf8("page_4"));
        page_4->setGeometry(QRect(0, 0, 583, 368));
        verticalLayout_19 = new QVBoxLayout(page_4);
        verticalLayout_19->setObjectName(QString::fromUtf8("verticalLayout_19"));
        individual_demo = new QGroupBox(page_4);
        individual_demo->setObjectName(QString::fromUtf8("individual_demo"));
        verticalLayout_14 = new QVBoxLayout(individual_demo);
        verticalLayout_14->setSpacing(0);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        open_files = new QToolButton(individual_demo);
        open_files->setObjectName(QString::fromUtf8("open_files"));
        open_files->setMaximumSize(QSize(23, 22));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        open_files->setIcon(icon1);

        horizontalLayout_9->addWidget(open_files);

        open_instruction = new QLabel(individual_demo);
        open_instruction->setObjectName(QString::fromUtf8("open_instruction"));

        horizontalLayout_9->addWidget(open_instruction);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_5);


        verticalLayout_14->addLayout(horizontalLayout_9);

        individual_list = new QListView(individual_demo);
        individual_list->setObjectName(QString::fromUtf8("individual_list"));

        verticalLayout_14->addWidget(individual_list);


        verticalLayout_19->addWidget(individual_demo);

        multiple_regression_demo = new QGroupBox(page_4);
        multiple_regression_demo->setObjectName(QString::fromUtf8("multiple_regression_demo"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(multiple_regression_demo->sizePolicy().hasHeightForWidth());
        multiple_regression_demo->setSizePolicy(sizePolicy1);
        verticalLayout_17 = new QVBoxLayout(multiple_regression_demo);
        verticalLayout_17->setSpacing(0);
        verticalLayout_17->setObjectName(QString::fromUtf8("verticalLayout_17"));
        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        open_mr_files = new QToolButton(multiple_regression_demo);
        open_mr_files->setObjectName(QString::fromUtf8("open_mr_files"));
        open_mr_files->setMaximumSize(QSize(23, 22));
        open_mr_files->setIcon(icon1);

        horizontalLayout_10->addWidget(open_mr_files);

        open_instruction_2 = new QLabel(multiple_regression_demo);
        open_instruction_2->setObjectName(QString::fromUtf8("open_instruction_2"));

        horizontalLayout_10->addWidget(open_instruction_2);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_4);

        regression_feature = new QWidget(multiple_regression_demo);
        regression_feature->setObjectName(QString::fromUtf8("regression_feature"));
        regression_feature->setMinimumSize(QSize(0, 0));
        horizontalLayout_6 = new QHBoxLayout(regression_feature);
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_6 = new QLabel(regression_feature);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_6->addWidget(label_6);

        foi = new QComboBox(regression_feature);
        foi->setObjectName(QString::fromUtf8("foi"));

        horizontalLayout_6->addWidget(foi);


        horizontalLayout_10->addWidget(regression_feature);

        foi_widget = new QWidget(multiple_regression_demo);
        foi_widget->setObjectName(QString::fromUtf8("foi_widget"));
        horizontalLayout_7 = new QHBoxLayout(foi_widget);
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));

        horizontalLayout_10->addWidget(foi_widget);


        verticalLayout_17->addLayout(horizontalLayout_10);

        subject_demo = new QTableWidget(multiple_regression_demo);
        if (subject_demo->columnCount() < 1)
            subject_demo->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        subject_demo->setHorizontalHeaderItem(0, __qtablewidgetitem3);
        subject_demo->setObjectName(QString::fromUtf8("subject_demo"));
        sizePolicy.setHeightForWidth(subject_demo->sizePolicy().hasHeightForWidth());
        subject_demo->setSizePolicy(sizePolicy);
        subject_demo->setAlternatingRowColors(true);
        subject_demo->setSelectionMode(QAbstractItemView::SingleSelection);
        subject_demo->setSelectionBehavior(QAbstractItemView::SelectRows);
        subject_demo->setShowGrid(true);
        subject_demo->setGridStyle(Qt::SolidLine);
        subject_demo->setSortingEnabled(true);
        subject_demo->setRowCount(0);

        verticalLayout_17->addWidget(subject_demo);


        verticalLayout_19->addWidget(multiple_regression_demo);

        toolBox->addItem(page_4, QString::fromUtf8("STEP2: Provide demographics"));
        page_3 = new QWidget();
        page_3->setObjectName(QString::fromUtf8("page_3"));
        page_3->setGeometry(QRect(0, 0, 583, 368));
        verticalLayout_7 = new QVBoxLayout(page_3);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        tabWidget_2 = new QTabWidget(page_3);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_4 = new QVBoxLayout(tab);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        show_null_greater = new QCheckBox(tab);
        show_null_greater->setObjectName(QString::fromUtf8("show_null_greater"));
        show_null_greater->setChecked(true);

        horizontalLayout_16->addWidget(show_null_greater);

        show_null_lesser = new QCheckBox(tab);
        show_null_lesser->setObjectName(QString::fromUtf8("show_null_lesser"));
        show_null_lesser->setChecked(true);

        horizontalLayout_16->addWidget(show_null_lesser);

        show_greater = new QCheckBox(tab);
        show_greater->setObjectName(QString::fromUtf8("show_greater"));
        show_greater->setChecked(true);

        horizontalLayout_16->addWidget(show_greater);

        show_lesser = new QCheckBox(tab);
        show_lesser->setObjectName(QString::fromUtf8("show_lesser"));
        show_lesser->setChecked(true);

        horizontalLayout_16->addWidget(show_lesser);

        show_lesser_2 = new QCheckBox(tab);
        show_lesser_2->setObjectName(QString::fromUtf8("show_lesser_2"));
        show_lesser_2->setChecked(true);

        horizontalLayout_16->addWidget(show_lesser_2);

        show_greater_2 = new QCheckBox(tab);
        show_greater_2->setObjectName(QString::fromUtf8("show_greater_2"));
        show_greater_2->setChecked(true);

        horizontalLayout_16->addWidget(show_greater_2);

        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_16->addWidget(label);

        span_to = new QSpinBox(tab);
        span_to->setObjectName(QString::fromUtf8("span_to"));
        span_to->setValue(40);

        horizontalLayout_16->addWidget(span_to);


        verticalLayout_4->addLayout(horizontalLayout_16);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, -1, 0, -1);
        null_dist = new QCustomPlot(tab);
        null_dist->setObjectName(QString::fromUtf8("null_dist"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(null_dist->sizePolicy().hasHeightForWidth());
        null_dist->setSizePolicy(sizePolicy2);

        horizontalLayout_2->addWidget(null_dist);

        fdr_dist = new QCustomPlot(tab);
        fdr_dist->setObjectName(QString::fromUtf8("fdr_dist"));
        sizePolicy2.setHeightForWidth(fdr_dist->sizePolicy().hasHeightForWidth());
        fdr_dist->setSizePolicy(sizePolicy2);

        horizontalLayout_2->addWidget(fdr_dist);


        verticalLayout_4->addLayout(horizontalLayout_2);

        tabWidget_2->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_10 = new QVBoxLayout(tab_2);
        verticalLayout_10->setSpacing(0);
        verticalLayout_10->setContentsMargins(0, 0, 0, 0);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        save_vbc_dist = new QToolButton(tab_2);
        save_vbc_dist->setObjectName(QString::fromUtf8("save_vbc_dist"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/save.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        save_vbc_dist->setIcon(icon2);

        horizontalLayout_5->addWidget(save_vbc_dist);

        save_fdr_dist = new QToolButton(tab_2);
        save_fdr_dist->setObjectName(QString::fromUtf8("save_fdr_dist"));
        save_fdr_dist->setIcon(icon2);

        horizontalLayout_5->addWidget(save_fdr_dist);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_2);


        verticalLayout_10->addLayout(horizontalLayout_5);

        dist_table = new QTableWidget(tab_2);
        if (dist_table->columnCount() < 4)
            dist_table->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(0, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(1, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(2, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        dist_table->setHorizontalHeaderItem(3, __qtablewidgetitem7);
        dist_table->setObjectName(QString::fromUtf8("dist_table"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(dist_table->sizePolicy().hasHeightForWidth());
        dist_table->setSizePolicy(sizePolicy3);

        verticalLayout_10->addWidget(dist_table);

        tabWidget_2->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_5 = new QVBoxLayout(tab_3);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        textBrowser = new QTextBrowser(tab_3);
        textBrowser->setObjectName(QString::fromUtf8("textBrowser"));

        verticalLayout_5->addWidget(textBrowser);

        tabWidget_2->addTab(tab_3, QString());

        verticalLayout_7->addWidget(tabWidget_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        progressBar = new QProgressBar(page_3);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(0);

        horizontalLayout_3->addWidget(progressBar);

        multithread = new QSpinBox(page_3);
        multithread->setObjectName(QString::fromUtf8("multithread"));
        multithread->setMaximumSize(QSize(75, 16777215));
        multithread->setMinimum(1);
        multithread->setMaximum(32);
        multithread->setValue(4);

        horizontalLayout_3->addWidget(multithread);

        label_2 = new QLabel(page_3);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        run = new QPushButton(page_3);
        run->setObjectName(QString::fromUtf8("run"));
        run->setEnabled(false);

        horizontalLayout_3->addWidget(run);


        verticalLayout_7->addLayout(horizontalLayout_3);

        toolBox->addItem(page_3, QString::fromUtf8("STEP3: Run connectometry analysis"));

        verticalLayout->addWidget(toolBox);

        widget_2 = new QWidget(vbc_dialog);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        verticalLayout_9 = new QVBoxLayout(widget_2);
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));

        verticalLayout->addWidget(widget_2);


        retranslateUi(vbc_dialog);

        toolBox->setCurrentIndex(1);
        toolBox->layout()->setSpacing(0);
        tabWidget_2->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(vbc_dialog);
    } // setupUi

    void retranslateUi(QDialog *vbc_dialog)
    {
        vbc_dialog->setWindowTitle(QApplication::translate("vbc_dialog", "Connectometry", 0, QApplication::UnicodeUTF8));
        save_name_list->setText(QApplication::translate("vbc_dialog", "Save name list...", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("vbc_dialog", "Zoom", 0, QApplication::UnicodeUTF8));
        coordinate->setText(QApplication::translate("vbc_dialog", "(x,y,z)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = subject_list->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("vbc_dialog", "Subject ID", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = subject_list->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("vbc_dialog", "Value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = subject_list->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("vbc_dialog", "R2", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("vbc_dialog", "Source Data", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("vbc_dialog", "Model", 0, QApplication::UnicodeUTF8));
        rb_multiple_regression->setText(QApplication::translate("vbc_dialog", "Multiple Regression: (e.g. to study connectivity change due to aging or IQ difference)", 0, QApplication::UnicodeUTF8));
        rb_group_difference->setText(QApplication::translate("vbc_dialog", "Group difference (e.g. to study connectivity difference between male and female)", 0, QApplication::UnicodeUTF8));
        rb_paired_difference->setText(QApplication::translate("vbc_dialog", "Paired difference (e.g. to study connectivity difference before and after a treatment)", 0, QApplication::UnicodeUTF8));
        rb_individual_analysis->setText(QApplication::translate("vbc_dialog", "Individual Analysis (e.g. to study the affected pathways of each stroke patient)", 0, QApplication::UnicodeUTF8));
        percentile_rank_group->setTitle(QString());
        label_5->setText(QApplication::translate("vbc_dialog", "p-value or percentile threshold", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("vbc_dialog", "0.05: week difference   0.01: moderate difference   0.002: strong difference", 0, QApplication::UnicodeUTF8));
        advanced_options->setText(QApplication::translate("vbc_dialog", "Advanced options...", 0, QApplication::UnicodeUTF8));
        advanced_options_box->setTitle(QString());
        label_4->setText(QApplication::translate("vbc_dialog", "Permutation Count", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("vbc_dialog", "Length Threshold (mm)", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("vbc_dialog", "Track Pruning (iterations)", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("vbc_dialog", "STEP1: Select analysis model", 0, QApplication::UnicodeUTF8));
        individual_demo->setTitle(QString());
        open_files->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        open_instruction->setText(QApplication::translate("vbc_dialog", "Open subjects' fib files ", 0, QApplication::UnicodeUTF8));
        multiple_regression_demo->setTitle(QString());
        open_mr_files->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        open_instruction_2->setText(QApplication::translate("vbc_dialog", "Open subjects' demographics.", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("vbc_dialog", "Please choose a feature to study:", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = subject_demo->horizontalHeaderItem(0);
        ___qtablewidgetitem3->setText(QApplication::translate("vbc_dialog", "Subject ID", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_4), QApplication::translate("vbc_dialog", "STEP2: Provide demographics", 0, QApplication::UnicodeUTF8));
        show_null_greater->setText(QApplication::translate("vbc_dialog", "null greater", 0, QApplication::UnicodeUTF8));
        show_null_lesser->setText(QApplication::translate("vbc_dialog", "null lesser", 0, QApplication::UnicodeUTF8));
        show_greater->setText(QApplication::translate("vbc_dialog", "greater", 0, QApplication::UnicodeUTF8));
        show_lesser->setText(QApplication::translate("vbc_dialog", "lesser", 0, QApplication::UnicodeUTF8));
        show_lesser_2->setText(QApplication::translate("vbc_dialog", "FDR lesser", 0, QApplication::UnicodeUTF8));
        show_greater_2->setText(QApplication::translate("vbc_dialog", "FDR greater", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("vbc_dialog", "max x:", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QApplication::translate("vbc_dialog", "Plot", 0, QApplication::UnicodeUTF8));
        save_vbc_dist->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        save_fdr_dist->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = dist_table->horizontalHeaderItem(0);
        ___qtablewidgetitem4->setText(QApplication::translate("vbc_dialog", "span", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem5 = dist_table->horizontalHeaderItem(1);
        ___qtablewidgetitem5->setText(QApplication::translate("vbc_dialog", "pdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem6 = dist_table->horizontalHeaderItem(2);
        ___qtablewidgetitem6->setText(QApplication::translate("vbc_dialog", "cdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem7 = dist_table->horizontalHeaderItem(3);
        ___qtablewidgetitem7->setText(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_2), QApplication::translate("vbc_dialog", "Data", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_3), QApplication::translate("vbc_dialog", "Report", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("vbc_dialog", "multi-thread", 0, QApplication::UnicodeUTF8));
        run->setText(QApplication::translate("vbc_dialog", "Run", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_3), QApplication::translate("vbc_dialog", "STEP3: Run connectometry analysis", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class vbc_dialog: public Ui_vbc_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBC_DIALOG_H
