/********************************************************************************
** Form generated from reading UI file 'simulation.ui'
**
** Created: Tue Sep 2 17:34:56 2014
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
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_Simulation
{
public:
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label;
    QSpinBox *SNR;
    QLabel *label_3;
    QSpinBox *Trial;
    QLabel *label_6;
    QDoubleSpinBox *FA;
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_4;
    QLineEdit *CrossingAngle;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_2;
    QDoubleSpinBox *MD;
    QHBoxLayout *horizontalLayout;
    QLabel *label_5;
    QLineEdit *Btable;
    QPushButton *pushButton;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_7;
    QLineEdit *output;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *generate;

    void setupUi(QDialog *Simulation)
    {
        if (Simulation->objectName().isEmpty())
            Simulation->setObjectName(QString::fromUtf8("Simulation"));
        Simulation->resize(528, 204);
        verticalLayout_2 = new QVBoxLayout(Simulation);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label = new QLabel(Simulation);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_4->addWidget(label);

        SNR = new QSpinBox(Simulation);
        SNR->setObjectName(QString::fromUtf8("SNR"));
        SNR->setMinimum(1);
        SNR->setMaximum(500);
        SNR->setValue(50);

        horizontalLayout_4->addWidget(SNR);

        label_3 = new QLabel(Simulation);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_4->addWidget(label_3);

        Trial = new QSpinBox(Simulation);
        Trial->setObjectName(QString::fromUtf8("Trial"));
        Trial->setMinimum(1);
        Trial->setMaximum(500);
        Trial->setValue(1);

        horizontalLayout_4->addWidget(Trial);

        label_6 = new QLabel(Simulation);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_4->addWidget(label_6);

        FA = new QDoubleSpinBox(Simulation);
        FA->setObjectName(QString::fromUtf8("FA"));
        FA->setMaximum(1);
        FA->setSingleStep(0.1);
        FA->setValue(0.5);

        horizontalLayout_4->addWidget(FA);


        verticalLayout_2->addLayout(horizontalLayout_4);

        gridLayout = new QGridLayout();
#ifndef Q_OS_MAC
        gridLayout->setSpacing(6);
#endif
        gridLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_4 = new QLabel(Simulation);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_7->addWidget(label_4);

        CrossingAngle = new QLineEdit(Simulation);
        CrossingAngle->setObjectName(QString::fromUtf8("CrossingAngle"));

        horizontalLayout_7->addWidget(CrossingAngle);


        gridLayout->addLayout(horizontalLayout_7, 5, 1, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_2 = new QLabel(Simulation);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        MD = new QDoubleSpinBox(Simulation);
        MD->setObjectName(QString::fromUtf8("MD"));
        MD->setMinimum(0.1);
        MD->setMaximum(5);
        MD->setValue(1);

        horizontalLayout_3->addWidget(MD);


        gridLayout->addLayout(horizontalLayout_3, 5, 0, 1, 1);


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

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_7 = new QLabel(Simulation);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_5->addWidget(label_7);

        output = new QLineEdit(Simulation);
        output->setObjectName(QString::fromUtf8("output"));

        horizontalLayout_5->addWidget(output);


        verticalLayout_2->addLayout(horizontalLayout_5);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        generate = new QPushButton(Simulation);
        generate->setObjectName(QString::fromUtf8("generate"));

        horizontalLayout_2->addWidget(generate);


        verticalLayout_2->addLayout(horizontalLayout_2);


        retranslateUi(Simulation);

        QMetaObject::connectSlotsByName(Simulation);
    } // setupUi

    void retranslateUi(QDialog *Simulation)
    {
        Simulation->setWindowTitle(QApplication::translate("Simulation", "Dialog", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("Simulation", "B0 SNR", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("Simulation", "Trial", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("Simulation", "FA", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("Simulation", "Crossing Angle=", 0, QApplication::UnicodeUTF8));
        CrossingAngle->setText(QApplication::translate("Simulation", "70 90", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("Simulation", "Mean Difusivity", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("Simulation", "B-table", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("Simulation", "...", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("Simulation", "Output", 0, QApplication::UnicodeUTF8));
        generate->setText(QApplication::translate("Simulation", "Generate", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Simulation: public Ui_Simulation {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMULATION_H
