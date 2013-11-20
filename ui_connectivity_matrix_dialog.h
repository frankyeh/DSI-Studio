/********************************************************************************
** Form generated from reading UI file 'connectivity_matrix_dialog.ui'
**
** Created: Wed Nov 20 16:16:26 2013
**      by: Qt User Interface Compiler version 4.8.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONNECTIVITY_MATRIX_DIALOG_H
#define UI_CONNECTIVITY_MATRIX_DIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGraphicsView>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_connectivity_matrix_dialog
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QComboBox *region_list;
    QLabel *label;
    QDoubleSpinBox *zoom;
    QCheckBox *log;
    QSpacerItem *horizontalSpacer;
    QPushButton *recalculate;
    QToolButton *save_as;
    QGraphicsView *graphicsView;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *connectivity_matrix_dialog)
    {
        if (connectivity_matrix_dialog->objectName().isEmpty())
            connectivity_matrix_dialog->setObjectName(QString::fromUtf8("connectivity_matrix_dialog"));
        connectivity_matrix_dialog->resize(640, 480);
        verticalLayout = new QVBoxLayout(connectivity_matrix_dialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(connectivity_matrix_dialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        region_list = new QComboBox(connectivity_matrix_dialog);
        region_list->setObjectName(QString::fromUtf8("region_list"));
        region_list->setMaximumSize(QSize(16777215, 22));

        horizontalLayout->addWidget(region_list);

        label = new QLabel(connectivity_matrix_dialog);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        zoom = new QDoubleSpinBox(connectivity_matrix_dialog);
        zoom->setObjectName(QString::fromUtf8("zoom"));
        zoom->setMaximumSize(QSize(16777215, 22));
        zoom->setMinimum(0.1);
        zoom->setMaximum(50);
        zoom->setSingleStep(1);
        zoom->setValue(3);

        horizontalLayout->addWidget(zoom);

        log = new QCheckBox(connectivity_matrix_dialog);
        log->setObjectName(QString::fromUtf8("log"));
        log->setMaximumSize(QSize(16777215, 22));

        horizontalLayout->addWidget(log);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        recalculate = new QPushButton(connectivity_matrix_dialog);
        recalculate->setObjectName(QString::fromUtf8("recalculate"));
        recalculate->setMaximumSize(QSize(16777215, 22));

        horizontalLayout->addWidget(recalculate);

        save_as = new QToolButton(connectivity_matrix_dialog);
        save_as->setObjectName(QString::fromUtf8("save_as"));
        save_as->setMinimumSize(QSize(23, 22));
        save_as->setMaximumSize(QSize(23, 22));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/save.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        save_as->setIcon(icon);

        horizontalLayout->addWidget(save_as);


        verticalLayout->addLayout(horizontalLayout);

        graphicsView = new QGraphicsView(connectivity_matrix_dialog);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));

        verticalLayout->addWidget(graphicsView);

        buttonBox = new QDialogButtonBox(connectivity_matrix_dialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Close);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(connectivity_matrix_dialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), connectivity_matrix_dialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), connectivity_matrix_dialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(connectivity_matrix_dialog);
    } // setupUi

    void retranslateUi(QDialog *connectivity_matrix_dialog)
    {
        connectivity_matrix_dialog->setWindowTitle(QApplication::translate("connectivity_matrix_dialog", "Dialog", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("connectivity_matrix_dialog", "Regioins:", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("connectivity_matrix_dialog", "Zoom in/out", 0, QApplication::UnicodeUTF8));
        log->setText(QApplication::translate("connectivity_matrix_dialog", "Logarithm", 0, QApplication::UnicodeUTF8));
        recalculate->setText(QApplication::translate("connectivity_matrix_dialog", "Recalculate", 0, QApplication::UnicodeUTF8));
        save_as->setText(QApplication::translate("connectivity_matrix_dialog", "...", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class connectivity_matrix_dialog: public Ui_connectivity_matrix_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONNECTIVITY_MATRIX_DIALOG_H
