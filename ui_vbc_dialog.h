/********************************************************************************
** Form generated from reading UI file 'vbc_dialog.ui'
**
** Created: Sat Mar 7 16:02:24 2015
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
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_9;
    QToolBox *toolBox;
    QWidget *page;
    QVBoxLayout *verticalLayout_12;
    QWidget *widget_3;
    QHBoxLayout *horizontalLayout_11;
    QSplitter *splitter_3;
    QWidget *widget_4;
    QVBoxLayout *verticalLayout_16;
    QSplitter *splitter;
    QWidget *widget_8;
    QVBoxLayout *verticalLayout_13;
    QHBoxLayout *horizontalLayout_17;
    QLabel *label_13;
    QDoubleSpinBox *zoom;
    QLabel *coordinate;
    QSpacerItem *horizontalSpacer_6;
    QGraphicsView *vbc_view;
    QScrollBar *AxiSlider;
    QWidget *widget_9;
    QVBoxLayout *verticalLayout_15;
    QCustomPlot *vbc_report;
    QHBoxLayout *horizontalLayout_14;
    QLabel *label_8;
    QSpinBox *x_pos;
    QLabel *label_10;
    QSpinBox *y_pos;
    QLabel *label_9;
    QSpinBox *z_pos;
    QLabel *label_11;
    QSpinBox *scatter;
    QToolButton *save_report;
    QWidget *widget_6;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *save_name_list;
    QPushButton *save_R2;
    QToolButton *remove_sel_subject;
    QSpacerItem *horizontalSpacer;
    QTableWidget *subject_list;
    QWidget *page_4;
    QVBoxLayout *verticalLayout_8;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_21;
    QRadioButton *rb_multiple_regression;
    QRadioButton *rb_group_difference;
    QRadioButton *rb_paired_difference;
    QRadioButton *rb_individual_analysis;
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
    QPushButton *remove_subject;
    QToolButton *remove_subject2;
    QSpacerItem *horizontalSpacer_4;
    QWidget *foi_widget;
    QHBoxLayout *horizontalLayout_7;
    QTableWidget *subject_demo;
    QWidget *page_2;
    QVBoxLayout *verticalLayout_23;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *percentile_rank_layout;
    QWidget *regression_feature;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_6;
    QComboBox *foi;
    QSpacerItem *horizontalSpacer_3;
    QLabel *threshold_label;
    QSpinBox *percentage_dif;
    QDoubleSpinBox *t_threshold;
    QDoubleSpinBox *percentile;
    QLabel *percentage_label;
    QLabel *explaination;
    QGroupBox *percentile_rank_group;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_15;
    QRadioButton *roi_whole_brain;
    QRadioButton *roi_file;
    QRadioButton *roi_atlas;
    QWidget *ROI_widget;
    QHBoxLayout *horizontalLayout_2;
    QComboBox *atlas_box;
    QComboBox *atlas_region_box;
    QLabel *label_5;
    QComboBox *region_type;
    QGroupBox *advanced_options_box;
    QVBoxLayout *verticalLayout_18;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_7;
    QDoubleSpinBox *seeding_density;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_4;
    QSpinBox *mr_permutation;
    QHBoxLayout *horizontalLayout;
    QLabel *label_3;
    QSpinBox *length_threshold;
    QCheckBox *normalize_qa;
    QSpacerItem *verticalSpacer_2;
    QWidget *page_3;
    QVBoxLayout *verticalLayout_7;
    QTabWidget *tabWidget_2;
    QWidget *tab;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout_16;
    QCheckBox *view_legend;
    QCheckBox *show_null_greater;
    QCheckBox *show_null_lesser;
    QCheckBox *show_greater;
    QCheckBox *show_lesser;
    QCheckBox *show_lesser_2;
    QCheckBox *show_greater_2;
    QLabel *label;
    QSpinBox *span_to;
    QWidget *widget_5;
    QVBoxLayout *verticalLayout_5;
    QSplitter *splitter_2;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_13;
    QCustomPlot *null_dist;
    QCustomPlot *fdr_dist;
    QTextBrowser *textBrowser;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_10;
    QTableWidget *dist_table;
    QHBoxLayout *horizontalLayout_3;
    QProgressBar *progressBar;
    QSpinBox *multithread;
    QLabel *label_2;
    QPushButton *run;
    QPushButton *show_result;

    void setupUi(QDialog *vbc_dialog)
    {
        if (vbc_dialog->objectName().isEmpty())
            vbc_dialog->setObjectName(QString::fromUtf8("vbc_dialog"));
        vbc_dialog->resize(654, 492);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/axial.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        vbc_dialog->setWindowIcon(icon);
        verticalLayout = new QVBoxLayout(vbc_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widget_2 = new QWidget(vbc_dialog);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        verticalLayout_9 = new QVBoxLayout(widget_2);
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        toolBox = new QToolBox(widget_2);
        toolBox->setObjectName(QString::fromUtf8("toolBox"));
        page = new QWidget();
        page->setObjectName(QString::fromUtf8("page"));
        page->setGeometry(QRect(0, 0, 636, 390));
        verticalLayout_12 = new QVBoxLayout(page);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        widget_3 = new QWidget(page);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        horizontalLayout_11 = new QHBoxLayout(widget_3);
        horizontalLayout_11->setSpacing(0);
        horizontalLayout_11->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        splitter_3 = new QSplitter(widget_3);
        splitter_3->setObjectName(QString::fromUtf8("splitter_3"));
        splitter_3->setOrientation(Qt::Horizontal);
        widget_4 = new QWidget(splitter_3);
        widget_4->setObjectName(QString::fromUtf8("widget_4"));
        verticalLayout_16 = new QVBoxLayout(widget_4);
        verticalLayout_16->setSpacing(0);
        verticalLayout_16->setContentsMargins(0, 0, 0, 0);
        verticalLayout_16->setObjectName(QString::fromUtf8("verticalLayout_16"));
        splitter = new QSplitter(widget_4);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Vertical);
        widget_8 = new QWidget(splitter);
        widget_8->setObjectName(QString::fromUtf8("widget_8"));
        verticalLayout_13 = new QVBoxLayout(widget_8);
        verticalLayout_13->setSpacing(0);
        verticalLayout_13->setContentsMargins(0, 0, 0, 0);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        label_13 = new QLabel(widget_8);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_17->addWidget(label_13);

        zoom = new QDoubleSpinBox(widget_8);
        zoom->setObjectName(QString::fromUtf8("zoom"));
        zoom->setMinimum(0.1);
        zoom->setMaximum(20);
        zoom->setValue(3);

        horizontalLayout_17->addWidget(zoom);

        coordinate = new QLabel(widget_8);
        coordinate->setObjectName(QString::fromUtf8("coordinate"));

        horizontalLayout_17->addWidget(coordinate);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_17->addItem(horizontalSpacer_6);


        verticalLayout_13->addLayout(horizontalLayout_17);

        vbc_view = new QGraphicsView(widget_8);
        vbc_view->setObjectName(QString::fromUtf8("vbc_view"));

        verticalLayout_13->addWidget(vbc_view);

        AxiSlider = new QScrollBar(widget_8);
        AxiSlider->setObjectName(QString::fromUtf8("AxiSlider"));
        AxiSlider->setOrientation(Qt::Horizontal);

        verticalLayout_13->addWidget(AxiSlider);

        splitter->addWidget(widget_8);
        widget_9 = new QWidget(splitter);
        widget_9->setObjectName(QString::fromUtf8("widget_9"));
        verticalLayout_15 = new QVBoxLayout(widget_9);
        verticalLayout_15->setSpacing(0);
        verticalLayout_15->setContentsMargins(0, 0, 0, 0);
        verticalLayout_15->setObjectName(QString::fromUtf8("verticalLayout_15"));
        vbc_report = new QCustomPlot(widget_9);
        vbc_report->setObjectName(QString::fromUtf8("vbc_report"));

        verticalLayout_15->addWidget(vbc_report);

        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(0);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        label_8 = new QLabel(widget_9);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_14->addWidget(label_8);

        x_pos = new QSpinBox(widget_9);
        x_pos->setObjectName(QString::fromUtf8("x_pos"));

        horizontalLayout_14->addWidget(x_pos);

        label_10 = new QLabel(widget_9);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_14->addWidget(label_10);

        y_pos = new QSpinBox(widget_9);
        y_pos->setObjectName(QString::fromUtf8("y_pos"));

        horizontalLayout_14->addWidget(y_pos);

        label_9 = new QLabel(widget_9);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout_14->addWidget(label_9);

        z_pos = new QSpinBox(widget_9);
        z_pos->setObjectName(QString::fromUtf8("z_pos"));

        horizontalLayout_14->addWidget(z_pos);

        label_11 = new QLabel(widget_9);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        horizontalLayout_14->addWidget(label_11);

        scatter = new QSpinBox(widget_9);
        scatter->setObjectName(QString::fromUtf8("scatter"));
        scatter->setMaximum(15);
        scatter->setValue(5);

        horizontalLayout_14->addWidget(scatter);

        save_report = new QToolButton(widget_9);
        save_report->setObjectName(QString::fromUtf8("save_report"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/save.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        save_report->setIcon(icon1);

        horizontalLayout_14->addWidget(save_report);


        verticalLayout_15->addLayout(horizontalLayout_14);

        splitter->addWidget(widget_9);

        verticalLayout_16->addWidget(splitter);

        splitter_3->addWidget(widget_4);
        widget_6 = new QWidget(splitter_3);
        widget_6->setObjectName(QString::fromUtf8("widget_6"));
        verticalLayout_6 = new QVBoxLayout(widget_6);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        save_name_list = new QPushButton(widget_6);
        save_name_list->setObjectName(QString::fromUtf8("save_name_list"));

        horizontalLayout_4->addWidget(save_name_list);

        save_R2 = new QPushButton(widget_6);
        save_R2->setObjectName(QString::fromUtf8("save_R2"));

        horizontalLayout_4->addWidget(save_R2);

        remove_sel_subject = new QToolButton(widget_6);
        remove_sel_subject->setObjectName(QString::fromUtf8("remove_sel_subject"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        remove_sel_subject->setIcon(icon2);
        remove_sel_subject->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_4->addWidget(remove_sel_subject);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);


        verticalLayout_6->addLayout(horizontalLayout_4);

        subject_list = new QTableWidget(widget_6);
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

        verticalLayout_6->addWidget(subject_list);

        splitter_3->addWidget(widget_6);

        horizontalLayout_11->addWidget(splitter_3);


        verticalLayout_12->addWidget(widget_3);

        toolBox->addItem(page, QString::fromUtf8("Source Data"));
        page_4 = new QWidget();
        page_4->setObjectName(QString::fromUtf8("page_4"));
        page_4->setGeometry(QRect(0, 0, 636, 390));
        verticalLayout_8 = new QVBoxLayout(page_4);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        groupBox_4 = new QGroupBox(page_4);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_21 = new QVBoxLayout(groupBox_4);
        verticalLayout_21->setSpacing(0);
        verticalLayout_21->setObjectName(QString::fromUtf8("verticalLayout_21"));
        verticalLayout_21->setContentsMargins(-1, 0, -1, -1);
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
        rb_paired_difference->setEnabled(true);

        verticalLayout_21->addWidget(rb_paired_difference);

        rb_individual_analysis = new QRadioButton(groupBox_4);
        rb_individual_analysis->setObjectName(QString::fromUtf8("rb_individual_analysis"));

        verticalLayout_21->addWidget(rb_individual_analysis);


        verticalLayout_8->addWidget(groupBox_4);

        individual_demo = new QGroupBox(page_4);
        individual_demo->setObjectName(QString::fromUtf8("individual_demo"));
        verticalLayout_14 = new QVBoxLayout(individual_demo);
        verticalLayout_14->setSpacing(0);
        verticalLayout_14->setContentsMargins(0, 0, 0, 0);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        open_files = new QToolButton(individual_demo);
        open_files->setObjectName(QString::fromUtf8("open_files"));
        open_files->setMaximumSize(QSize(23, 22));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        open_files->setIcon(icon3);

        horizontalLayout_9->addWidget(open_files);

        open_instruction = new QLabel(individual_demo);
        open_instruction->setObjectName(QString::fromUtf8("open_instruction"));

        horizontalLayout_9->addWidget(open_instruction);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_5);


        verticalLayout_14->addLayout(horizontalLayout_9);


        verticalLayout_8->addWidget(individual_demo);

        individual_list = new QListView(page_4);
        individual_list->setObjectName(QString::fromUtf8("individual_list"));

        verticalLayout_8->addWidget(individual_list);

        multiple_regression_demo = new QGroupBox(page_4);
        multiple_regression_demo->setObjectName(QString::fromUtf8("multiple_regression_demo"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(multiple_regression_demo->sizePolicy().hasHeightForWidth());
        multiple_regression_demo->setSizePolicy(sizePolicy1);
        verticalLayout_17 = new QVBoxLayout(multiple_regression_demo);
        verticalLayout_17->setSpacing(0);
        verticalLayout_17->setContentsMargins(0, 0, 0, 0);
        verticalLayout_17->setObjectName(QString::fromUtf8("verticalLayout_17"));
        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        open_mr_files = new QToolButton(multiple_regression_demo);
        open_mr_files->setObjectName(QString::fromUtf8("open_mr_files"));
        open_mr_files->setMaximumSize(QSize(23, 22));
        open_mr_files->setIcon(icon3);

        horizontalLayout_10->addWidget(open_mr_files);

        open_instruction_2 = new QLabel(multiple_regression_demo);
        open_instruction_2->setObjectName(QString::fromUtf8("open_instruction_2"));

        horizontalLayout_10->addWidget(open_instruction_2);

        remove_subject = new QPushButton(multiple_regression_demo);
        remove_subject->setObjectName(QString::fromUtf8("remove_subject"));
        remove_subject->setMaximumSize(QSize(65535, 22));
        remove_subject->setIcon(icon2);

        horizontalLayout_10->addWidget(remove_subject);

        remove_subject2 = new QToolButton(multiple_regression_demo);
        remove_subject2->setObjectName(QString::fromUtf8("remove_subject2"));
        remove_subject2->setIcon(icon2);
        remove_subject2->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_10->addWidget(remove_subject2);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_4);

        foi_widget = new QWidget(multiple_regression_demo);
        foi_widget->setObjectName(QString::fromUtf8("foi_widget"));
        horizontalLayout_7 = new QHBoxLayout(foi_widget);
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));

        horizontalLayout_10->addWidget(foi_widget);


        verticalLayout_17->addLayout(horizontalLayout_10);


        verticalLayout_8->addWidget(multiple_regression_demo);

        subject_demo = new QTableWidget(page_4);
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

        verticalLayout_8->addWidget(subject_demo);

        toolBox->addItem(page_4, QString::fromUtf8("STEP1: Select analysis model and provide patient information"));
        page_2 = new QWidget();
        page_2->setObjectName(QString::fromUtf8("page_2"));
        page_2->setGeometry(QRect(0, 0, 636, 390));
        verticalLayout_23 = new QVBoxLayout(page_2);
        verticalLayout_23->setObjectName(QString::fromUtf8("verticalLayout_23"));
        groupBox = new QGroupBox(page_2);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_4 = new QVBoxLayout(groupBox);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        percentile_rank_layout = new QHBoxLayout();
        percentile_rank_layout->setObjectName(QString::fromUtf8("percentile_rank_layout"));
        regression_feature = new QWidget(groupBox);
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


        percentile_rank_layout->addWidget(regression_feature);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        percentile_rank_layout->addItem(horizontalSpacer_3);

        threshold_label = new QLabel(groupBox);
        threshold_label->setObjectName(QString::fromUtf8("threshold_label"));

        percentile_rank_layout->addWidget(threshold_label);

        percentage_dif = new QSpinBox(groupBox);
        percentage_dif->setObjectName(QString::fromUtf8("percentage_dif"));
        percentage_dif->setMaximumSize(QSize(50, 16777215));
        percentage_dif->setMaximum(200);
        percentage_dif->setValue(25);

        percentile_rank_layout->addWidget(percentage_dif);

        t_threshold = new QDoubleSpinBox(groupBox);
        t_threshold->setObjectName(QString::fromUtf8("t_threshold"));
        t_threshold->setMaximumSize(QSize(50, 16777215));
        t_threshold->setMaximum(5);
        t_threshold->setSingleStep(1);
        t_threshold->setValue(2);

        percentile_rank_layout->addWidget(t_threshold);

        percentile = new QDoubleSpinBox(groupBox);
        percentile->setObjectName(QString::fromUtf8("percentile"));
        percentile->setMaximumSize(QSize(50, 16777215));
        percentile->setDecimals(2);
        percentile->setMinimum(0);
        percentile->setMaximum(100);
        percentile->setSingleStep(1);
        percentile->setValue(2);

        percentile_rank_layout->addWidget(percentile);

        percentage_label = new QLabel(groupBox);
        percentage_label->setObjectName(QString::fromUtf8("percentage_label"));
        QSizePolicy sizePolicy2(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(percentage_label->sizePolicy().hasHeightForWidth());
        percentage_label->setSizePolicy(sizePolicy2);

        percentile_rank_layout->addWidget(percentage_label);


        verticalLayout_4->addLayout(percentile_rank_layout);

        explaination = new QLabel(groupBox);
        explaination->setObjectName(QString::fromUtf8("explaination"));

        verticalLayout_4->addWidget(explaination);


        verticalLayout_23->addWidget(groupBox);

        percentile_rank_group = new QGroupBox(page_2);
        percentile_rank_group->setObjectName(QString::fromUtf8("percentile_rank_group"));
        verticalLayout_2 = new QVBoxLayout(percentile_rank_group);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setSpacing(0);
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        roi_whole_brain = new QRadioButton(percentile_rank_group);
        roi_whole_brain->setObjectName(QString::fromUtf8("roi_whole_brain"));
        roi_whole_brain->setChecked(true);

        horizontalLayout_15->addWidget(roi_whole_brain);

        roi_file = new QRadioButton(percentile_rank_group);
        roi_file->setObjectName(QString::fromUtf8("roi_file"));

        horizontalLayout_15->addWidget(roi_file);

        roi_atlas = new QRadioButton(percentile_rank_group);
        roi_atlas->setObjectName(QString::fromUtf8("roi_atlas"));

        horizontalLayout_15->addWidget(roi_atlas);

        ROI_widget = new QWidget(percentile_rank_group);
        ROI_widget->setObjectName(QString::fromUtf8("ROI_widget"));
        horizontalLayout_2 = new QHBoxLayout(ROI_widget);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        atlas_box = new QComboBox(ROI_widget);
        atlas_box->setObjectName(QString::fromUtf8("atlas_box"));

        horizontalLayout_2->addWidget(atlas_box);

        atlas_region_box = new QComboBox(ROI_widget);
        atlas_region_box->setObjectName(QString::fromUtf8("atlas_region_box"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(atlas_region_box->sizePolicy().hasHeightForWidth());
        atlas_region_box->setSizePolicy(sizePolicy3);
        atlas_region_box->setSizeAdjustPolicy(QComboBox::AdjustToContentsOnFirstShow);

        horizontalLayout_2->addWidget(atlas_region_box);

        label_5 = new QLabel(ROI_widget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_2->addWidget(label_5);

        region_type = new QComboBox(ROI_widget);
        region_type->setObjectName(QString::fromUtf8("region_type"));

        horizontalLayout_2->addWidget(region_type);


        horizontalLayout_15->addWidget(ROI_widget);


        verticalLayout_2->addLayout(horizontalLayout_15);


        verticalLayout_23->addWidget(percentile_rank_group);

        advanced_options_box = new QGroupBox(page_2);
        advanced_options_box->setObjectName(QString::fromUtf8("advanced_options_box"));
        advanced_options_box->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout_18 = new QVBoxLayout(advanced_options_box);
        verticalLayout_18->setSpacing(0);
        verticalLayout_18->setObjectName(QString::fromUtf8("verticalLayout_18"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(-1, 0, -1, -1);
        label_7 = new QLabel(advanced_options_box);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_5->addWidget(label_7);

        seeding_density = new QDoubleSpinBox(advanced_options_box);
        seeding_density->setObjectName(QString::fromUtf8("seeding_density"));
        seeding_density->setMaximumSize(QSize(75, 16777215));
        seeding_density->setDecimals(1);
        seeding_density->setMinimum(1);
        seeding_density->setMaximum(100);
        seeding_density->setSingleStep(0.5);
        seeding_density->setValue(10);

        horizontalLayout_5->addWidget(seeding_density);


        verticalLayout_18->addLayout(horizontalLayout_5);

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
        mr_permutation->setValue(1000);

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
        length_threshold->setMaximum(200);
        length_threshold->setValue(40);

        horizontalLayout->addWidget(length_threshold);


        verticalLayout_18->addLayout(horizontalLayout);

        normalize_qa = new QCheckBox(advanced_options_box);
        normalize_qa->setObjectName(QString::fromUtf8("normalize_qa"));
        normalize_qa->setEnabled(true);
        normalize_qa->setChecked(false);

        verticalLayout_18->addWidget(normalize_qa);


        verticalLayout_23->addWidget(advanced_options_box);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_23->addItem(verticalSpacer_2);

        toolBox->addItem(page_2, QString::fromUtf8("STEP2: Setup parameters"));
        page_3 = new QWidget();
        page_3->setObjectName(QString::fromUtf8("page_3"));
        page_3->setGeometry(QRect(0, 0, 636, 390));
        verticalLayout_7 = new QVBoxLayout(page_3);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        tabWidget_2 = new QTabWidget(page_3);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_3 = new QVBoxLayout(tab);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        view_legend = new QCheckBox(tab);
        view_legend->setObjectName(QString::fromUtf8("view_legend"));
        view_legend->setChecked(true);

        horizontalLayout_16->addWidget(view_legend);

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


        verticalLayout_3->addLayout(horizontalLayout_16);

        widget_5 = new QWidget(tab);
        widget_5->setObjectName(QString::fromUtf8("widget_5"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(widget_5->sizePolicy().hasHeightForWidth());
        widget_5->setSizePolicy(sizePolicy4);
        widget_5->setMinimumSize(QSize(0, 100));
        verticalLayout_5 = new QVBoxLayout(widget_5);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        splitter_2 = new QSplitter(widget_5);
        splitter_2->setObjectName(QString::fromUtf8("splitter_2"));
        splitter_2->setOrientation(Qt::Vertical);
        widget = new QWidget(splitter_2);
        widget->setObjectName(QString::fromUtf8("widget"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy5);
        widget->setMinimumSize(QSize(0, 180));
        horizontalLayout_13 = new QHBoxLayout(widget);
        horizontalLayout_13->setSpacing(0);
        horizontalLayout_13->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        null_dist = new QCustomPlot(widget);
        null_dist->setObjectName(QString::fromUtf8("null_dist"));
        sizePolicy5.setHeightForWidth(null_dist->sizePolicy().hasHeightForWidth());
        null_dist->setSizePolicy(sizePolicy5);

        horizontalLayout_13->addWidget(null_dist);

        fdr_dist = new QCustomPlot(widget);
        fdr_dist->setObjectName(QString::fromUtf8("fdr_dist"));
        sizePolicy5.setHeightForWidth(fdr_dist->sizePolicy().hasHeightForWidth());
        fdr_dist->setSizePolicy(sizePolicy5);

        horizontalLayout_13->addWidget(fdr_dist);

        splitter_2->addWidget(widget);
        textBrowser = new QTextBrowser(splitter_2);
        textBrowser->setObjectName(QString::fromUtf8("textBrowser"));
        sizePolicy5.setHeightForWidth(textBrowser->sizePolicy().hasHeightForWidth());
        textBrowser->setSizePolicy(sizePolicy5);
        textBrowser->setMaximumSize(QSize(65525, 65525));
        splitter_2->addWidget(textBrowser);

        verticalLayout_5->addWidget(splitter_2);


        verticalLayout_3->addWidget(widget_5);

        tabWidget_2->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_10 = new QVBoxLayout(tab_2);
        verticalLayout_10->setSpacing(0);
        verticalLayout_10->setContentsMargins(0, 0, 0, 0);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
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
        sizePolicy4.setHeightForWidth(dist_table->sizePolicy().hasHeightForWidth());
        dist_table->setSizePolicy(sizePolicy4);

        verticalLayout_10->addWidget(dist_table);

        tabWidget_2->addTab(tab_2, QString());

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

        show_result = new QPushButton(page_3);
        show_result->setObjectName(QString::fromUtf8("show_result"));

        horizontalLayout_3->addWidget(show_result);


        verticalLayout_7->addLayout(horizontalLayout_3);

        toolBox->addItem(page_3, QString::fromUtf8("STEP3: Run connectometry analysis"));

        verticalLayout_9->addWidget(toolBox);


        verticalLayout->addWidget(widget_2);


        retranslateUi(vbc_dialog);

        toolBox->setCurrentIndex(2);
        toolBox->layout()->setSpacing(0);
        tabWidget_2->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(vbc_dialog);
    } // setupUi

    void retranslateUi(QDialog *vbc_dialog)
    {
        vbc_dialog->setWindowTitle(QApplication::translate("vbc_dialog", "Connectometry", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("vbc_dialog", "Zoom", 0, QApplication::UnicodeUTF8));
        coordinate->setText(QApplication::translate("vbc_dialog", "(x,y,z)", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("vbc_dialog", "x", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("vbc_dialog", "y", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("vbc_dialog", "z", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("vbc_dialog", "scatter style", 0, QApplication::UnicodeUTF8));
        save_report->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        save_name_list->setText(QApplication::translate("vbc_dialog", "Save name list...", 0, QApplication::UnicodeUTF8));
        save_R2->setText(QApplication::translate("vbc_dialog", "Save R2...", 0, QApplication::UnicodeUTF8));
        remove_sel_subject->setText(QApplication::translate("vbc_dialog", "Remove selected subject", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = subject_list->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("vbc_dialog", "Subject ID", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = subject_list->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("vbc_dialog", "Value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = subject_list->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("vbc_dialog", "R2", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("vbc_dialog", "Source Data", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("vbc_dialog", "Model", 0, QApplication::UnicodeUTF8));
        rb_multiple_regression->setText(QApplication::translate("vbc_dialog", "Multipler regression: (e.g. to study connectivity change due to aging or IQ difference)", 0, QApplication::UnicodeUTF8));
        rb_group_difference->setText(QApplication::translate("vbc_dialog", "Group difference (e.g. to study connectivity difference between male and female groups)", 0, QApplication::UnicodeUTF8));
        rb_paired_difference->setText(QApplication::translate("vbc_dialog", "Paired difference (e.g. to study connectivity difference before and after a treatment)", 0, QApplication::UnicodeUTF8));
        rb_individual_analysis->setText(QApplication::translate("vbc_dialog", "Individual analysis (e.g. to study the affected pathways of each patient)", 0, QApplication::UnicodeUTF8));
        individual_demo->setTitle(QString());
        open_files->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        open_instruction->setText(QApplication::translate("vbc_dialog", "Open subjects' fib files ", 0, QApplication::UnicodeUTF8));
        multiple_regression_demo->setTitle(QString());
        open_mr_files->setText(QApplication::translate("vbc_dialog", "...", 0, QApplication::UnicodeUTF8));
        open_instruction_2->setText(QApplication::translate("vbc_dialog", "Open subjects' demographics.", 0, QApplication::UnicodeUTF8));
        remove_subject->setText(QApplication::translate("vbc_dialog", "Remove selected subject", 0, QApplication::UnicodeUTF8));
        remove_subject2->setText(QApplication::translate("vbc_dialog", "Remove subject with a specific value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = subject_demo->horizontalHeaderItem(0);
        ___qtablewidgetitem3->setText(QApplication::translate("vbc_dialog", "Subject ID", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_4), QApplication::translate("vbc_dialog", "STEP1: Select analysis model and provide patient information", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("vbc_dialog", "Threshold", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("vbc_dialog", "Feature of interest", 0, QApplication::UnicodeUTF8));
        threshold_label->setText(QApplication::translate("vbc_dialog", "Percentile Threshold", 0, QApplication::UnicodeUTF8));
        percentage_label->setText(QApplication::translate("vbc_dialog", "%", 0, QApplication::UnicodeUTF8));
        explaination->setText(QApplication::translate("vbc_dialog", "explanation", 0, QApplication::UnicodeUTF8));
        percentile_rank_group->setTitle(QApplication::translate("vbc_dialog", "ROI", 0, QApplication::UnicodeUTF8));
        roi_whole_brain->setText(QApplication::translate("vbc_dialog", "Whole brain", 0, QApplication::UnicodeUTF8));
        roi_file->setText(QApplication::translate("vbc_dialog", "Assign by file", 0, QApplication::UnicodeUTF8));
        roi_atlas->setText(QApplication::translate("vbc_dialog", "Assign by atlas", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("vbc_dialog", "used as", 0, QApplication::UnicodeUTF8));
        region_type->clear();
        region_type->insertItems(0, QStringList()
         << QApplication::translate("vbc_dialog", "ROI", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("vbc_dialog", "ROA", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("vbc_dialog", "End", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("vbc_dialog", "Seed", 0, QApplication::UnicodeUTF8)
        );
        advanced_options_box->setTitle(QApplication::translate("vbc_dialog", "Advanced options", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("vbc_dialog", "Seeding density (seeds/mm^3)", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("vbc_dialog", "Permutation count.", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("vbc_dialog", "Minimum track length (mm)", 0, QApplication::UnicodeUTF8));
        normalize_qa->setText(QApplication::translate("vbc_dialog", "normalize QA using its maximum value", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("vbc_dialog", "STEP2: Setup parameters", 0, QApplication::UnicodeUTF8));
        view_legend->setText(QApplication::translate("vbc_dialog", "legend", 0, QApplication::UnicodeUTF8));
        show_null_greater->setText(QApplication::translate("vbc_dialog", "null greater", 0, QApplication::UnicodeUTF8));
        show_null_lesser->setText(QApplication::translate("vbc_dialog", "null lesser", 0, QApplication::UnicodeUTF8));
        show_greater->setText(QApplication::translate("vbc_dialog", "greater", 0, QApplication::UnicodeUTF8));
        show_lesser->setText(QApplication::translate("vbc_dialog", "lesser", 0, QApplication::UnicodeUTF8));
        show_lesser_2->setText(QApplication::translate("vbc_dialog", "FDR lesser", 0, QApplication::UnicodeUTF8));
        show_greater_2->setText(QApplication::translate("vbc_dialog", "FDR greater", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("vbc_dialog", "max x:", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QApplication::translate("vbc_dialog", "Plot and Report", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = dist_table->horizontalHeaderItem(0);
        ___qtablewidgetitem4->setText(QApplication::translate("vbc_dialog", "span", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem5 = dist_table->horizontalHeaderItem(1);
        ___qtablewidgetitem5->setText(QApplication::translate("vbc_dialog", "pdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem6 = dist_table->horizontalHeaderItem(2);
        ___qtablewidgetitem6->setText(QApplication::translate("vbc_dialog", "cdf(x)", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem7 = dist_table->horizontalHeaderItem(3);
        ___qtablewidgetitem7->setText(QApplication::translate("vbc_dialog", "FDR", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_2), QApplication::translate("vbc_dialog", "Data", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("vbc_dialog", "multi-thread", 0, QApplication::UnicodeUTF8));
        run->setText(QApplication::translate("vbc_dialog", "Run", 0, QApplication::UnicodeUTF8));
        show_result->setText(QApplication::translate("vbc_dialog", "Show Results", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_3), QApplication::translate("vbc_dialog", "STEP3: Run connectometry analysis", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class vbc_dialog: public Ui_vbc_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VBC_DIALOG_H
