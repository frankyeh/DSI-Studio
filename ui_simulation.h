/********************************************************************************
** Form generated from reading UI file 'simulation.ui'
**
** Created: Wed Dec 5 11:28:28 2012
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMULATION_H
#define UI_SIMULATION_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_Simulation
{
public:
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QSpinBox *SNR;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_3;
    QSpinBox *Trial;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_2;
    QDoubleSpinBox *MD;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_6;
    QLineEdit *FA;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_4;
    QLineEdit *CrossingAngle;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_8;
    QSpinBox *phantom_size;
    QLabel *label_7;
    QSpinBox *background_size;
    QHBoxLayout *horizontalLayout;
    QLabel *label_5;
    QLineEdit *Btable;
    QPushButton *pushButton;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *Simulation)
    {
        if (Simulation->objectName().isEmpty())
            Simulation->setObjectName(QString::fromUtf8("Simulation"));
        Simulation->resize(390, 168);
        verticalLayout_2 = new QVBoxLayout(Simulation);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        gridLayout = new QGridLayout();
#ifndef Q_OS_MAC
        gridLayout->setSpacing(6);
#endif
#ifndef Q_OS_MAC
        gridLayout->setContentsMargins(0, 0, 0, 0);
#endif
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(Simulation);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        SNR = new QSpinBox(Simulation);
        SNR->setObjectName(QString::fromUtf8("SNR"));
        SNR->setMinimum(1);
        SNR->setMaximum(200);
        SNR->setValue(50);

        horizontalLayout_2->addWidget(SNR);


        gridLayout->addLayout(horizontalLayout_2, 0, 0, 1, 1);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_3 = new QLabel(Simulation);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_4->addWidget(label_3);

        Trial = new QSpinBox(Simulation);
        Trial->setObjectName(QString::fromUtf8("Trial"));
        Trial->setMinimum(1);
        Trial->setMaximum(10);
        Trial->setValue(1);

        horizontalLayout_4->addWidget(Trial);


        gridLayout->addLayout(horizontalLayout_4, 6, 0, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_2 = new QLabel(Simulation);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        MD = new QDoubleSpinBox(Simulation);
        MD->setObjectName(QString::fromUtf8("MD"));
        MD->setMinimum(0.1);
        MD->setMaximum(5);
        MD->setValue(0.5);

        horizontalLayout_3->addWidget(MD);


        gridLayout->addLayout(horizontalLayout_3, 7, 0, 1, 1);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_6 = new QLabel(Simulation);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_6->addWidget(label_6);

        FA = new QLineEdit(Simulation);
        FA->setObjectName(QString::fromUtf8("FA"));

        horizontalLayout_6->addWidget(FA);


        gridLayout->addLayout(horizontalLayout_6, 0, 1, 1, 1);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_4 = new QLabel(Simulation);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_7->addWidget(label_4);

        CrossingAngle = new QLineEdit(Simulation);
        CrossingAngle->setObjectName(QString::fromUtf8("CrossingAngle"));

        horizontalLayout_7->addWidget(CrossingAngle);


        gridLayout->addLayout(horizontalLayout_7, 6, 1, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_8 = new QLabel(Simulation);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_5->addWidget(label_8);

        phantom_size = new QSpinBox(Simulation);
        phantom_size->setObjectName(QString::fromUtf8("phantom_size"));
        phantom_size->setMinimum(1);
        phantom_size->setMaximum(200);
        phantom_size->setValue(50);

        horizontalLayout_5->addWidget(phantom_size);

        label_7 = new QLabel(Simulation);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_5->addWidget(label_7);

        background_size = new QSpinBox(Simulation);
        background_size->setObjectName(QString::fromUtf8("background_size"));
        background_size->setMinimum(1);
        background_size->setMaximum(100);
        background_size->setValue(5);

        horizontalLayout_5->addWidget(background_size);


        gridLayout->addLayout(horizontalLayout_5, 7, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_5 = new QLabel(Simulation);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout->addWidget(label_5);

        Btable = new QLineEdit(Simulation);
        Btable->setObjectName(QString::fromUtf8("Btable"));

        horizontalLayout->addWidget(Btable);

        pushButton = new QPushButton(Simulation);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);


        verticalLayout_2->addLayout(horizontalLayout);

        buttonBox = new QDialogButtonBox(Simulation);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout_2->addWidget(buttonBox);


        retranslateUi(Simulation);
        QObject::connect(buttonBox, SIGNAL(accepted()), Simulation, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), Simulation, SLOT(reject()));

        QMetaObject::connectSlotsByName(Simulation);
    } // setupUi

    void retranslateUi(QDialog *Simulation)
    {
        Simulation->setWindowTitle(QApplication::translate("Simulation", "Dialog", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("Simulation", "B0 SNR", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("Simulation", "Trial", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("Simulation", "Mean Difusivity", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("Simulation", "FA=", 0, QApplication::UnicodeUTF8));
        FA->setText(QApplication::translate("Simulation", "0.5 0.6 0.7", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("Simulation", "Crossing Angle=", 0, QApplication::UnicodeUTF8));
        CrossingAngle->setText(QApplication::translate("Simulation", "70 90", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("Simulation", "Phantom size", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("Simulation", "Background size", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("Simulation", "B-table", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("Simulation", "...", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Simulation: public Ui_Simulation {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMULATION_H
