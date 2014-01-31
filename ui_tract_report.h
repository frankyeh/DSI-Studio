/********************************************************************************
** Form generated from reading UI file 'tract_report.ui'
**
** Created: Fri Jan 31 14:51:44 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRACT_REPORT_H
#define UI_TRACT_REPORT_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include "plot/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_tract_report
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_2;
    QLabel *label_2;
    QComboBox *profile_dir;
    QLabel *label_3;
    QComboBox *report_index;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label_12;
    QSpinBox *linewidth;
    QLabel *label;
    QDoubleSpinBox *report_bandwidth;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *report_legend;
    QPushButton *save_image;
    QPushButton *save_report;
    QCustomPlot *report_widget;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *tract_report)
    {
        if (tract_report->objectName().isEmpty())
            tract_report->setObjectName(QString::fromUtf8("tract_report"));
        tract_report->resize(640, 480);
        verticalLayout = new QVBoxLayout(tract_report);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, -1, -1, -1);
        groupBox_2 = new QGroupBox(tract_report);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy);
        gridLayout_2 = new QGridLayout(groupBox_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_2->addWidget(label_2, 0, 0, 1, 1);

        profile_dir = new QComboBox(groupBox_2);
        profile_dir->setObjectName(QString::fromUtf8("profile_dir"));
        profile_dir->setMaximumSize(QSize(16777215, 22));

        gridLayout_2->addWidget(profile_dir, 0, 1, 1, 1);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 1, 0, 1, 1);

        report_index = new QComboBox(groupBox_2);
        report_index->setObjectName(QString::fromUtf8("report_index"));
        report_index->setMaximumSize(QSize(16777215, 22));

        gridLayout_2->addWidget(report_index, 1, 1, 1, 1);


        horizontalLayout->addWidget(groupBox_2);

        groupBox = new QGroupBox(tract_report);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        sizePolicy.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy);
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_12 = new QLabel(groupBox);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout->addWidget(label_12, 0, 0, 1, 1);

        linewidth = new QSpinBox(groupBox);
        linewidth->setObjectName(QString::fromUtf8("linewidth"));
        linewidth->setMinimum(1);
        linewidth->setMaximum(10);

        gridLayout->addWidget(linewidth, 0, 1, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 1, 0, 1, 1);

        report_bandwidth = new QDoubleSpinBox(groupBox);
        report_bandwidth->setObjectName(QString::fromUtf8("report_bandwidth"));
        report_bandwidth->setDecimals(1);
        report_bandwidth->setMinimum(0.1);
        report_bandwidth->setMaximum(10);
        report_bandwidth->setSingleStep(0.5);
        report_bandwidth->setValue(1);

        gridLayout->addWidget(report_bandwidth, 1, 1, 1, 1);


        horizontalLayout->addWidget(groupBox);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        report_legend = new QCheckBox(tract_report);
        report_legend->setObjectName(QString::fromUtf8("report_legend"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(report_legend->sizePolicy().hasHeightForWidth());
        report_legend->setSizePolicy(sizePolicy1);
        report_legend->setMaximumSize(QSize(16777215, 22));

        verticalLayout_2->addWidget(report_legend);

        save_image = new QPushButton(tract_report);
        save_image->setObjectName(QString::fromUtf8("save_image"));

        verticalLayout_2->addWidget(save_image);

        save_report = new QPushButton(tract_report);
        save_report->setObjectName(QString::fromUtf8("save_report"));

        verticalLayout_2->addWidget(save_report);


        horizontalLayout->addLayout(verticalLayout_2);


        verticalLayout->addLayout(horizontalLayout);

        report_widget = new QCustomPlot(tract_report);
        report_widget->setObjectName(QString::fromUtf8("report_widget"));
        QSizePolicy sizePolicy2(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(report_widget->sizePolicy().hasHeightForWidth());
        report_widget->setSizePolicy(sizePolicy2);

        verticalLayout->addWidget(report_widget);

        buttonBox = new QDialogButtonBox(tract_report);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Close);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(tract_report);
        QObject::connect(buttonBox, SIGNAL(accepted()), tract_report, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), tract_report, SLOT(reject()));

        QMetaObject::connectSlotsByName(tract_report);
    } // setupUi

    void retranslateUi(QDialog *tract_report)
    {
        tract_report->setWindowTitle(QApplication::translate("tract_report", "Tract Analysis Report", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("tract_report", "Data Sources", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("tract_report", "Data Sampling Strategy", 0, QApplication::UnicodeUTF8));
        profile_dir->clear();
        profile_dir->insertItems(0, QStringList()
         << QApplication::translate("tract_report", "X direction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tract_report", "Y direction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tract_report", "Z direction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tract_report", "fiber orientation", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tract_report", "mean of each fiber", 0, QApplication::UnicodeUTF8)
        );
        label_3->setText(QApplication::translate("tract_report", "Quantitative Index", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("tract_report", "Visualization", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("tract_report", "Line Width", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("tract_report", "Regression Bandwidth", 0, QApplication::UnicodeUTF8));
        report_legend->setText(QApplication::translate("tract_report", "Show Legend", 0, QApplication::UnicodeUTF8));
        save_image->setText(QApplication::translate("tract_report", "Save Image...", 0, QApplication::UnicodeUTF8));
        save_report->setText(QApplication::translate("tract_report", "Save Report Data...", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class tract_report: public Ui_tract_report {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRACT_REPORT_H
