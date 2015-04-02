/********************************************************************************
** Form generated from reading UI file 'dicom_parser.ui'
**
** Created: Wed Apr 1 23:34:58 2015
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DICOM_PARSER_H
#define UI_DICOM_PARSER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTableWidget>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_dicom_parser
{
public:
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_3;
    QToolButton *loadImage;
    QToolButton *toolButton_2;
    QToolButton *load_bval;
    QToolButton *load_bvec;
    QToolButton *toolButton_8;
    QToolButton *apply_slice_orientation;
    QSpacerItem *horizontalSpacer_2;
    QHBoxLayout *horizontalLayout_4;
    QToolButton *toolButton;
    QToolButton *toolButton_3;
    QToolButton *toolButton_4;
    QToolButton *toolButton_5;
    QToolButton *toolButton_6;
    QToolButton *toolButton_7;
    QToolButton *motion_correction;
    QSpacerItem *horizontalSpacer;
    QTableWidget *tableWidget;
    QHBoxLayout *horizontalLayout_2;
    QComboBox *upsampling;
    QLabel *label;
    QLineEdit *SrcName;
    QPushButton *upperDir;
    QPushButton *pushButton;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *dicom_parser)
    {
        if (dicom_parser->objectName().isEmpty())
            dicom_parser->setObjectName(QString::fromUtf8("dicom_parser"));
        dicom_parser->setWindowModality(Qt::ApplicationModal);
        dicom_parser->resize(611, 348);
        QFont font;
        font.setFamily(QString::fromUtf8("Arial"));
        dicom_parser->setFont(font);
        dicom_parser->setLayoutDirection(Qt::LeftToRight);
        dicom_parser->setModal(false);
        horizontalLayout = new QHBoxLayout(dicom_parser);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        loadImage = new QToolButton(dicom_parser);
        loadImage->setObjectName(QString::fromUtf8("loadImage"));

        horizontalLayout_3->addWidget(loadImage);

        toolButton_2 = new QToolButton(dicom_parser);
        toolButton_2->setObjectName(QString::fromUtf8("toolButton_2"));

        horizontalLayout_3->addWidget(toolButton_2);

        load_bval = new QToolButton(dicom_parser);
        load_bval->setObjectName(QString::fromUtf8("load_bval"));

        horizontalLayout_3->addWidget(load_bval);

        load_bvec = new QToolButton(dicom_parser);
        load_bvec->setObjectName(QString::fromUtf8("load_bvec"));

        horizontalLayout_3->addWidget(load_bvec);

        toolButton_8 = new QToolButton(dicom_parser);
        toolButton_8->setObjectName(QString::fromUtf8("toolButton_8"));

        horizontalLayout_3->addWidget(toolButton_8);

        apply_slice_orientation = new QToolButton(dicom_parser);
        apply_slice_orientation->setObjectName(QString::fromUtf8("apply_slice_orientation"));

        horizontalLayout_3->addWidget(apply_slice_orientation);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        toolButton = new QToolButton(dicom_parser);
        toolButton->setObjectName(QString::fromUtf8("toolButton"));
        toolButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout_4->addWidget(toolButton);

        toolButton_3 = new QToolButton(dicom_parser);
        toolButton_3->setObjectName(QString::fromUtf8("toolButton_3"));

        horizontalLayout_4->addWidget(toolButton_3);

        toolButton_4 = new QToolButton(dicom_parser);
        toolButton_4->setObjectName(QString::fromUtf8("toolButton_4"));

        horizontalLayout_4->addWidget(toolButton_4);

        toolButton_5 = new QToolButton(dicom_parser);
        toolButton_5->setObjectName(QString::fromUtf8("toolButton_5"));

        horizontalLayout_4->addWidget(toolButton_5);

        toolButton_6 = new QToolButton(dicom_parser);
        toolButton_6->setObjectName(QString::fromUtf8("toolButton_6"));

        horizontalLayout_4->addWidget(toolButton_6);

        toolButton_7 = new QToolButton(dicom_parser);
        toolButton_7->setObjectName(QString::fromUtf8("toolButton_7"));

        horizontalLayout_4->addWidget(toolButton_7);

        motion_correction = new QToolButton(dicom_parser);
        motion_correction->setObjectName(QString::fromUtf8("motion_correction"));

        horizontalLayout_4->addWidget(motion_correction);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_4);

        tableWidget = new QTableWidget(dicom_parser);
        if (tableWidget->columnCount() < 5)
            tableWidget->setColumnCount(5);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(3, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(4, __qtablewidgetitem4);
        tableWidget->setObjectName(QString::fromUtf8("tableWidget"));
        tableWidget->setLayoutDirection(Qt::LeftToRight);
        tableWidget->setHorizontalScrollMode(QAbstractItemView::ScrollPerItem);
        tableWidget->setGridStyle(Qt::SolidLine);

        verticalLayout->addWidget(tableWidget);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        upsampling = new QComboBox(dicom_parser);
        upsampling->setObjectName(QString::fromUtf8("upsampling"));

        horizontalLayout_2->addWidget(upsampling);

        label = new QLabel(dicom_parser);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFrameShape(QFrame::NoFrame);

        horizontalLayout_2->addWidget(label);

        SrcName = new QLineEdit(dicom_parser);
        SrcName->setObjectName(QString::fromUtf8("SrcName"));

        horizontalLayout_2->addWidget(SrcName);

        upperDir = new QPushButton(dicom_parser);
        upperDir->setObjectName(QString::fromUtf8("upperDir"));

        horizontalLayout_2->addWidget(upperDir);

        pushButton = new QPushButton(dicom_parser);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout_2->addWidget(pushButton);


        verticalLayout->addLayout(horizontalLayout_2);

        buttonBox = new QDialogButtonBox(dicom_parser);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setLayoutDirection(Qt::LeftToRight);
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        horizontalLayout->addLayout(verticalLayout);


        retranslateUi(dicom_parser);
        QObject::connect(buttonBox, SIGNAL(accepted()), dicom_parser, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), dicom_parser, SLOT(reject()));

        QMetaObject::connectSlotsByName(dicom_parser);
    } // setupUi

    void retranslateUi(QDialog *dicom_parser)
    {
        dicom_parser->setWindowTitle(QApplication::translate("dicom_parser", "B-table", 0, QApplication::UnicodeUTF8));
        loadImage->setText(QApplication::translate("dicom_parser", "Add Images...", 0, QApplication::UnicodeUTF8));
        toolButton_2->setText(QApplication::translate("dicom_parser", "Load b-table...", 0, QApplication::UnicodeUTF8));
        load_bval->setText(QApplication::translate("dicom_parser", "Load bval...", 0, QApplication::UnicodeUTF8));
        load_bvec->setText(QApplication::translate("dicom_parser", "Load bvec...", 0, QApplication::UnicodeUTF8));
        toolButton_8->setText(QApplication::translate("dicom_parser", "Save b-table...", 0, QApplication::UnicodeUTF8));
        apply_slice_orientation->setText(QApplication::translate("dicom_parser", "Apply slice orientation", 0, QApplication::UnicodeUTF8));
        toolButton->setText(QApplication::translate("dicom_parser", "Flip bx", 0, QApplication::UnicodeUTF8));
        toolButton_3->setText(QApplication::translate("dicom_parser", "Flip by", 0, QApplication::UnicodeUTF8));
        toolButton_4->setText(QApplication::translate("dicom_parser", "Flip bz", 0, QApplication::UnicodeUTF8));
        toolButton_5->setText(QApplication::translate("dicom_parser", "Switch bx by", 0, QApplication::UnicodeUTF8));
        toolButton_6->setText(QApplication::translate("dicom_parser", "Switch bx bz", 0, QApplication::UnicodeUTF8));
        toolButton_7->setText(QApplication::translate("dicom_parser", "Switch by bz", 0, QApplication::UnicodeUTF8));
        motion_correction->setText(QApplication::translate("dicom_parser", "Detect motion...", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = tableWidget->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("dicom_parser", "File Name", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = tableWidget->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("dicom_parser", "b value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = tableWidget->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("dicom_parser", "bx", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = tableWidget->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QApplication::translate("dicom_parser", "by", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = tableWidget->horizontalHeaderItem(4);
        ___qtablewidgetitem4->setText(QApplication::translate("dicom_parser", "bz", 0, QApplication::UnicodeUTF8));
        upsampling->clear();
        upsampling->insertItems(0, QStringList()
         << QApplication::translate("dicom_parser", "No upsampling", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("dicom_parser", "upsampling 2", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("dicom_parser", "downsampling 2", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("dicom_parser", "upsampling 4", 0, QApplication::UnicodeUTF8)
        );
        label->setText(QApplication::translate("dicom_parser", "Output file:", 0, QApplication::UnicodeUTF8));
        upperDir->setText(QApplication::translate("dicom_parser", "Upper Directory", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("dicom_parser", "&Browse...", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class dicom_parser: public Ui_dicom_parser {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DICOM_PARSER_H
