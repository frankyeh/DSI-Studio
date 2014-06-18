/********************************************************************************
** Form generated from reading UI file 'manual_alignment.ui'
**
** Created: Wed Jun 18 15:30:27 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MANUAL_ALIGNMENT_H
#define UI_MANUAL_ALIGNMENT_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGraphicsView>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSplitter>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_manual_alignment
{
public:
    QVBoxLayout *verticalLayout;
    QWidget *widget;
    QVBoxLayout *verticalLayout_6;
    QSplitter *splitter;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_2;
    QGraphicsView *cor_view;
    QSlider *cor_slice_pos;
    QVBoxLayout *verticalLayout_3;
    QGraphicsView *sag_view;
    QSlider *sag_slice_pos;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *verticalLayout_4;
    QGraphicsView *axi_view;
    QSlider *axi_slice_pos;
    QVBoxLayout *verticalLayout_5;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QDoubleSpinBox *tx;
    QLabel *label_3;
    QDoubleSpinBox *ty;
    QLabel *label;
    QDoubleSpinBox *tz;
    QGroupBox *scaling_group;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_6;
    QDoubleSpinBox *sx;
    QLabel *label_4;
    QDoubleSpinBox *sy;
    QLabel *label_5;
    QDoubleSpinBox *sz;
    QGroupBox *tilting_group;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_8;
    QDoubleSpinBox *xy;
    QLabel *label_7;
    QDoubleSpinBox *xz;
    QLabel *label_9;
    QDoubleSpinBox *yz;
    QGroupBox *groupBox_4;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_11;
    QDoubleSpinBox *rx;
    QLabel *label_12;
    QDoubleSpinBox *ry;
    QLabel *label_10;
    QDoubleSpinBox *rz;
    QHBoxLayout *horizontalLayout_8;
    QSpacerItem *horizontalSpacer;
    QPushButton *rerun;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QSlider *blend_pos;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *manual_alignment)
    {
        if (manual_alignment->objectName().isEmpty())
            manual_alignment->setObjectName(QString::fromUtf8("manual_alignment"));
        manual_alignment->resize(647, 479);
        verticalLayout = new QVBoxLayout(manual_alignment);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widget = new QWidget(manual_alignment);
        widget->setObjectName(QString::fromUtf8("widget"));
        verticalLayout_6 = new QVBoxLayout(widget);
#ifndef Q_OS_MAC
        verticalLayout_6->setSpacing(6);
#endif
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        splitter = new QSplitter(widget);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setFrameShadow(QFrame::Plain);
        splitter->setOrientation(Qt::Vertical);
        horizontalLayoutWidget = new QWidget(splitter);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayout_6 = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(0);
#ifndef Q_OS_MAC
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
#endif
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        cor_view = new QGraphicsView(horizontalLayoutWidget);
        cor_view->setObjectName(QString::fromUtf8("cor_view"));

        verticalLayout_2->addWidget(cor_view);

        cor_slice_pos = new QSlider(horizontalLayoutWidget);
        cor_slice_pos->setObjectName(QString::fromUtf8("cor_slice_pos"));
        cor_slice_pos->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(cor_slice_pos);


        horizontalLayout_6->addLayout(verticalLayout_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        sag_view = new QGraphicsView(horizontalLayoutWidget);
        sag_view->setObjectName(QString::fromUtf8("sag_view"));

        verticalLayout_3->addWidget(sag_view);

        sag_slice_pos = new QSlider(horizontalLayoutWidget);
        sag_slice_pos->setObjectName(QString::fromUtf8("sag_slice_pos"));
        sag_slice_pos->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(sag_slice_pos);


        horizontalLayout_6->addLayout(verticalLayout_3);

        splitter->addWidget(horizontalLayoutWidget);
        layoutWidget = new QWidget(splitter);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        horizontalLayout_7 = new QHBoxLayout(layoutWidget);
#ifndef Q_OS_MAC
        horizontalLayout_7->setSpacing(6);
#endif
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        axi_view = new QGraphicsView(layoutWidget);
        axi_view->setObjectName(QString::fromUtf8("axi_view"));

        verticalLayout_4->addWidget(axi_view);

        axi_slice_pos = new QSlider(layoutWidget);
        axi_slice_pos->setObjectName(QString::fromUtf8("axi_slice_pos"));
        axi_slice_pos->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(axi_slice_pos);


        horizontalLayout_7->addLayout(verticalLayout_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        groupBox = new QGroupBox(layoutWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_2 = new QHBoxLayout(groupBox);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(label_2);

        tx = new QDoubleSpinBox(groupBox);
        tx->setObjectName(QString::fromUtf8("tx"));
        tx->setSingleStep(0.05);

        horizontalLayout_2->addWidget(tx);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(label_3);

        ty = new QDoubleSpinBox(groupBox);
        ty->setObjectName(QString::fromUtf8("ty"));
        ty->setSingleStep(0.05);

        horizontalLayout_2->addWidget(ty);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(label);

        tz = new QDoubleSpinBox(groupBox);
        tz->setObjectName(QString::fromUtf8("tz"));
        tz->setSingleStep(0.05);

        horizontalLayout_2->addWidget(tz);


        verticalLayout_5->addWidget(groupBox);

        scaling_group = new QGroupBox(layoutWidget);
        scaling_group->setObjectName(QString::fromUtf8("scaling_group"));
        horizontalLayout_3 = new QHBoxLayout(scaling_group);
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_6 = new QLabel(scaling_group);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setAlignment(Qt::AlignCenter);

        horizontalLayout_3->addWidget(label_6);

        sx = new QDoubleSpinBox(scaling_group);
        sx->setObjectName(QString::fromUtf8("sx"));
        sx->setSingleStep(0.05);

        horizontalLayout_3->addWidget(sx);

        label_4 = new QLabel(scaling_group);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignCenter);

        horizontalLayout_3->addWidget(label_4);

        sy = new QDoubleSpinBox(scaling_group);
        sy->setObjectName(QString::fromUtf8("sy"));
        sy->setSingleStep(0.05);

        horizontalLayout_3->addWidget(sy);

        label_5 = new QLabel(scaling_group);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        horizontalLayout_3->addWidget(label_5);

        sz = new QDoubleSpinBox(scaling_group);
        sz->setObjectName(QString::fromUtf8("sz"));
        sz->setSingleStep(0.05);

        horizontalLayout_3->addWidget(sz);


        verticalLayout_5->addWidget(scaling_group);

        tilting_group = new QGroupBox(layoutWidget);
        tilting_group->setObjectName(QString::fromUtf8("tilting_group"));
        horizontalLayout_4 = new QHBoxLayout(tilting_group);
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_8 = new QLabel(tilting_group);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setAlignment(Qt::AlignCenter);

        horizontalLayout_4->addWidget(label_8);

        xy = new QDoubleSpinBox(tilting_group);
        xy->setObjectName(QString::fromUtf8("xy"));
        xy->setSingleStep(0.05);

        horizontalLayout_4->addWidget(xy);

        label_7 = new QLabel(tilting_group);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setAlignment(Qt::AlignCenter);

        horizontalLayout_4->addWidget(label_7);

        xz = new QDoubleSpinBox(tilting_group);
        xz->setObjectName(QString::fromUtf8("xz"));
        xz->setSingleStep(0.05);

        horizontalLayout_4->addWidget(xz);

        label_9 = new QLabel(tilting_group);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setAlignment(Qt::AlignCenter);

        horizontalLayout_4->addWidget(label_9);

        yz = new QDoubleSpinBox(tilting_group);
        yz->setObjectName(QString::fromUtf8("yz"));
        yz->setSingleStep(0.05);

        horizontalLayout_4->addWidget(yz);


        verticalLayout_5->addWidget(tilting_group);

        groupBox_4 = new QGroupBox(layoutWidget);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        horizontalLayout_5 = new QHBoxLayout(groupBox_4);
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_11 = new QLabel(groupBox_4);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setAlignment(Qt::AlignCenter);

        horizontalLayout_5->addWidget(label_11);

        rx = new QDoubleSpinBox(groupBox_4);
        rx->setObjectName(QString::fromUtf8("rx"));
        rx->setSingleStep(0.05);

        horizontalLayout_5->addWidget(rx);

        label_12 = new QLabel(groupBox_4);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setAlignment(Qt::AlignCenter);

        horizontalLayout_5->addWidget(label_12);

        ry = new QDoubleSpinBox(groupBox_4);
        ry->setObjectName(QString::fromUtf8("ry"));
        ry->setSingleStep(0.05);

        horizontalLayout_5->addWidget(ry);

        label_10 = new QLabel(groupBox_4);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setAlignment(Qt::AlignCenter);

        horizontalLayout_5->addWidget(label_10);

        rz = new QDoubleSpinBox(groupBox_4);
        rz->setObjectName(QString::fromUtf8("rz"));
        rz->setSingleStep(0.05);

        horizontalLayout_5->addWidget(rz);


        verticalLayout_5->addWidget(groupBox_4);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer);

        rerun = new QPushButton(layoutWidget);
        rerun->setObjectName(QString::fromUtf8("rerun"));

        horizontalLayout_8->addWidget(rerun);


        verticalLayout_5->addLayout(horizontalLayout_8);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        blend_pos = new QSlider(layoutWidget);
        blend_pos->setObjectName(QString::fromUtf8("blend_pos"));
        blend_pos->setMaximum(10);
        blend_pos->setPageStep(2);
        blend_pos->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(blend_pos);


        verticalLayout_5->addLayout(horizontalLayout);


        horizontalLayout_7->addLayout(verticalLayout_5);

        splitter->addWidget(layoutWidget);

        verticalLayout_6->addWidget(splitter);


        verticalLayout->addWidget(widget);

        buttonBox = new QDialogButtonBox(manual_alignment);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(manual_alignment);
        QObject::connect(buttonBox, SIGNAL(accepted()), manual_alignment, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), manual_alignment, SLOT(reject()));

        QMetaObject::connectSlotsByName(manual_alignment);
    } // setupUi

    void retranslateUi(QDialog *manual_alignment)
    {
        manual_alignment->setWindowTitle(QApplication::translate("manual_alignment", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("manual_alignment", "Translocation", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("manual_alignment", "x", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("manual_alignment", "y", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("manual_alignment", "z", 0, QApplication::UnicodeUTF8));
        scaling_group->setTitle(QApplication::translate("manual_alignment", "Scaling", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("manual_alignment", "x", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("manual_alignment", "y", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("manual_alignment", "z", 0, QApplication::UnicodeUTF8));
        tilting_group->setTitle(QApplication::translate("manual_alignment", "Tilting", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("manual_alignment", "xy", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("manual_alignment", "xz", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("manual_alignment", "yz", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("manual_alignment", "Rotation", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("manual_alignment", "x", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("manual_alignment", "y", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("manual_alignment", "z", 0, QApplication::UnicodeUTF8));
        rerun->setText(QApplication::translate("manual_alignment", "Re-run", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class manual_alignment: public Ui_manual_alignment {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MANUAL_ALIGNMENT_H
