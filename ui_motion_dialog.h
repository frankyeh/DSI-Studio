/********************************************************************************
** Form generated from reading UI file 'motion_dialog.ui'
**
** Created: Thu Oct 9 21:21:56 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MOTION_DIALOG_H
#define UI_MOTION_DIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QProgressBar>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include "plot/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_motion_dialog
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QCheckBox *legend1;
    QSpacerItem *horizontalSpacer;
    QCustomPlot *translocation;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QCheckBox *legend2;
    QSpacerItem *horizontalSpacer_3;
    QCustomPlot *rotation;
    QLabel *label;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer_2;
    QProgressBar *progressBar;
    QLabel *progress_label;
    QPushButton *correction;
    QPushButton *close;

    void setupUi(QDialog *motion_dialog)
    {
        if (motion_dialog->objectName().isEmpty())
            motion_dialog->setObjectName(QString::fromUtf8("motion_dialog"));
        motion_dialog->resize(375, 387);
        verticalLayout = new QVBoxLayout(motion_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(motion_dialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        legend1 = new QCheckBox(motion_dialog);
        legend1->setObjectName(QString::fromUtf8("legend1"));

        horizontalLayout->addWidget(legend1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        translocation = new QCustomPlot(motion_dialog);
        translocation->setObjectName(QString::fromUtf8("translocation"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(translocation->sizePolicy().hasHeightForWidth());
        translocation->setSizePolicy(sizePolicy);

        verticalLayout->addWidget(translocation);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_3 = new QLabel(motion_dialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_3->addWidget(label_3);

        legend2 = new QCheckBox(motion_dialog);
        legend2->setObjectName(QString::fromUtf8("legend2"));

        horizontalLayout_3->addWidget(legend2);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_3);


        verticalLayout->addLayout(horizontalLayout_3);

        rotation = new QCustomPlot(motion_dialog);
        rotation->setObjectName(QString::fromUtf8("rotation"));
        sizePolicy.setHeightForWidth(rotation->sizePolicy().hasHeightForWidth());
        rotation->setSizePolicy(sizePolicy);

        verticalLayout->addWidget(rotation);

        label = new QLabel(motion_dialog);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout->addWidget(label);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);

        progressBar = new QProgressBar(motion_dialog);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(24);

        horizontalLayout_2->addWidget(progressBar);

        progress_label = new QLabel(motion_dialog);
        progress_label->setObjectName(QString::fromUtf8("progress_label"));

        horizontalLayout_2->addWidget(progress_label);

        correction = new QPushButton(motion_dialog);
        correction->setObjectName(QString::fromUtf8("correction"));
        correction->setEnabled(true);

        horizontalLayout_2->addWidget(correction);

        close = new QPushButton(motion_dialog);
        close->setObjectName(QString::fromUtf8("close"));

        horizontalLayout_2->addWidget(close);


        verticalLayout->addLayout(horizontalLayout_2);


        retranslateUi(motion_dialog);
        QObject::connect(close, SIGNAL(clicked()), motion_dialog, SLOT(accept()));

        QMetaObject::connectSlotsByName(motion_dialog);
    } // setupUi

    void retranslateUi(QDialog *motion_dialog)
    {
        motion_dialog->setWindowTitle(QApplication::translate("motion_dialog", "Motion Correction", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("motion_dialog", "Translocation:", 0, QApplication::UnicodeUTF8));
        legend1->setText(QApplication::translate("motion_dialog", "legend", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("motion_dialog", "Rotation:", 0, QApplication::UnicodeUTF8));
        legend2->setText(QApplication::translate("motion_dialog", "legend", 0, QApplication::UnicodeUTF8));
        label->setText(QString());
        progress_label->setText(QApplication::translate("motion_dialog", "Estimating motion...", 0, QApplication::UnicodeUTF8));
        correction->setText(QApplication::translate("motion_dialog", "Correct Motion", 0, QApplication::UnicodeUTF8));
        close->setText(QApplication::translate("motion_dialog", "Close", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class motion_dialog: public Ui_motion_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MOTION_DIALOG_H
