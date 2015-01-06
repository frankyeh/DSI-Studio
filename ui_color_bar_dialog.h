/********************************************************************************
** Form generated from reading UI file 'color_bar_dialog.ui'
**
** Created: Sun Jan 4 22:45:49 2015
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_COLOR_BAR_DIALOG_H
#define UI_COLOR_BAR_DIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFormLayout>
#include <QtGui/QGraphicsView>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSplitter>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "qcolorcombobox.h"

QT_BEGIN_NAMESPACE

class Ui_color_bar_dialog
{
public:
    QVBoxLayout *verticalLayout;
    QSplitter *splitter;
    QWidget *widget;
    QFormLayout *formLayout;
    QLabel *label_13;
    QComboBox *tract_color_index;
    QLabel *label_10;
    QComboBox *color_bar_style;
    QLabel *label_9;
    QDoubleSpinBox *tract_color_max_value;
    QLabel *label_15;
    QDoubleSpinBox *tract_color_min_value;
    QLabel *label_11;
    QHBoxLayout *horizontalLayout_17;
    QColorToolButton *color_from;
    QColorToolButton *color_to;
    QPushButton *update_rendering;
    QGraphicsView *color_bar_view;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *color_bar_dialog)
    {
        if (color_bar_dialog->objectName().isEmpty())
            color_bar_dialog->setObjectName(QString::fromUtf8("color_bar_dialog"));
        color_bar_dialog->resize(284, 346);
        verticalLayout = new QVBoxLayout(color_bar_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        splitter = new QSplitter(color_bar_dialog);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        widget = new QWidget(splitter);
        widget->setObjectName(QString::fromUtf8("widget"));
        formLayout = new QFormLayout(widget);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setContentsMargins(9, -1, 9, -1);
        label_13 = new QLabel(widget);
        label_13->setObjectName(QString::fromUtf8("label_13"));
        label_13->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_13);

        tract_color_index = new QComboBox(widget);
        tract_color_index->setObjectName(QString::fromUtf8("tract_color_index"));
        tract_color_index->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(0, QFormLayout::FieldRole, tract_color_index);

        label_10 = new QLabel(widget);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_10);

        color_bar_style = new QComboBox(widget);
        color_bar_style->setObjectName(QString::fromUtf8("color_bar_style"));
        color_bar_style->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(1, QFormLayout::FieldRole, color_bar_style);

        label_9 = new QLabel(widget);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_9);

        tract_color_max_value = new QDoubleSpinBox(widget);
        tract_color_max_value->setObjectName(QString::fromUtf8("tract_color_max_value"));
        tract_color_max_value->setMaximumSize(QSize(16777215, 22));
        tract_color_max_value->setDecimals(4);

        formLayout->setWidget(2, QFormLayout::FieldRole, tract_color_max_value);

        label_15 = new QLabel(widget);
        label_15->setObjectName(QString::fromUtf8("label_15"));
        label_15->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_15);

        tract_color_min_value = new QDoubleSpinBox(widget);
        tract_color_min_value->setObjectName(QString::fromUtf8("tract_color_min_value"));
        tract_color_min_value->setMaximumSize(QSize(16777215, 22));
        tract_color_min_value->setDecimals(4);

        formLayout->setWidget(3, QFormLayout::FieldRole, tract_color_min_value);

        label_11 = new QLabel(widget);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setMaximumSize(QSize(22, 16777215));

        formLayout->setWidget(4, QFormLayout::LabelRole, label_11);

        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setSpacing(0);
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        color_from = new QColorToolButton(widget);
        color_from->setObjectName(QString::fromUtf8("color_from"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(color_from->sizePolicy().hasHeightForWidth());
        color_from->setSizePolicy(sizePolicy);
        color_from->setMaximumSize(QSize(16777215, 22));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/add.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        color_from->setIcon(icon);

        horizontalLayout_17->addWidget(color_from);

        color_to = new QColorToolButton(widget);
        color_to->setObjectName(QString::fromUtf8("color_to"));
        sizePolicy.setHeightForWidth(color_to->sizePolicy().hasHeightForWidth());
        color_to->setSizePolicy(sizePolicy);
        color_to->setMaximumSize(QSize(16777215, 22));
        color_to->setIcon(icon);

        horizontalLayout_17->addWidget(color_to);


        formLayout->setLayout(4, QFormLayout::FieldRole, horizontalLayout_17);

        update_rendering = new QPushButton(widget);
        update_rendering->setObjectName(QString::fromUtf8("update_rendering"));
        update_rendering->setMaximumSize(QSize(16777215, 22));

        formLayout->setWidget(5, QFormLayout::FieldRole, update_rendering);

        splitter->addWidget(widget);
        label_13->raise();
        tract_color_index->raise();
        label_10->raise();
        color_bar_style->raise();
        label_9->raise();
        tract_color_max_value->raise();
        label_15->raise();
        tract_color_min_value->raise();
        label_11->raise();
        update_rendering->raise();
        color_bar_view = new QGraphicsView(splitter);
        color_bar_view->setObjectName(QString::fromUtf8("color_bar_view"));
        splitter->addWidget(color_bar_view);

        verticalLayout->addWidget(splitter);

        buttonBox = new QDialogButtonBox(color_bar_dialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Close);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(color_bar_dialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), color_bar_dialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), color_bar_dialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(color_bar_dialog);
    } // setupUi

    void retranslateUi(QDialog *color_bar_dialog)
    {
        color_bar_dialog->setWindowTitle(QApplication::translate("color_bar_dialog", "Color Bar", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("color_bar_dialog", "Index", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("color_bar_dialog", "Style", 0, QApplication::UnicodeUTF8));
        color_bar_style->clear();
        color_bar_style->insertItems(0, QStringList()
         << QApplication::translate("color_bar_dialog", "Two colors", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("color_bar_dialog", "Color spectrum", 0, QApplication::UnicodeUTF8)
        );
        label_9->setText(QApplication::translate("color_bar_dialog", "Max value", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("color_bar_dialog", "Min value", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("color_bar_dialog", "Color", 0, QApplication::UnicodeUTF8));
        color_from->setText(QString());
        color_to->setText(QString());
        update_rendering->setText(QApplication::translate("color_bar_dialog", "Update", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class color_bar_dialog: public Ui_color_bar_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_COLOR_BAR_DIALOG_H
