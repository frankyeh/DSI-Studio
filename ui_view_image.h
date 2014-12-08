/********************************************************************************
** Form generated from reading UI file 'view_image.ui'
**
** Created: Sun Dec 7 22:56:57 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VIEW_IMAGE_H
#define UI_VIEW_IMAGE_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGraphicsView>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPlainTextEdit>
#include <QtGui/QSlider>
#include <QtGui/QTabWidget>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_view_image
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QLabel *image_info;
    QHBoxLayout *horizontalLayout_2;
    QToolButton *zoom_in;
    QToolButton *zoom_out;
    QLabel *label_3;
    QSlider *contrast;
    QLabel *label_2;
    QSlider *brightness;
    QGraphicsView *view;
    QSlider *slice_pos;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_3;
    QPlainTextEdit *info;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *view_image)
    {
        if (view_image->objectName().isEmpty())
            view_image->setObjectName(QString::fromUtf8("view_image"));
        view_image->resize(640, 480);
        verticalLayout = new QVBoxLayout(view_image);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(view_image);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        image_info = new QLabel(tab);
        image_info->setObjectName(QString::fromUtf8("image_info"));

        horizontalLayout->addWidget(image_info);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        zoom_in = new QToolButton(tab);
        zoom_in->setObjectName(QString::fromUtf8("zoom_in"));
        zoom_in->setMinimumSize(QSize(24, 0));

        horizontalLayout_2->addWidget(zoom_in);

        zoom_out = new QToolButton(tab);
        zoom_out->setObjectName(QString::fromUtf8("zoom_out"));
        zoom_out->setMinimumSize(QSize(24, 0));

        horizontalLayout_2->addWidget(zoom_out);

        label_3 = new QLabel(tab);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        contrast = new QSlider(tab);
        contrast->setObjectName(QString::fromUtf8("contrast"));
        contrast->setMinimum(1);
        contrast->setMaximum(30);
        contrast->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(contrast);

        label_2 = new QLabel(tab);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_2->addWidget(label_2);

        brightness = new QSlider(tab);
        brightness->setObjectName(QString::fromUtf8("brightness"));
        brightness->setMinimum(-10);
        brightness->setMaximum(10);
        brightness->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(brightness);


        verticalLayout_2->addLayout(horizontalLayout_2);

        view = new QGraphicsView(tab);
        view->setObjectName(QString::fromUtf8("view"));

        verticalLayout_2->addWidget(view);

        slice_pos = new QSlider(tab);
        slice_pos->setObjectName(QString::fromUtf8("slice_pos"));
        slice_pos->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(slice_pos);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_3 = new QVBoxLayout(tab_2);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        info = new QPlainTextEdit(tab_2);
        info->setObjectName(QString::fromUtf8("info"));

        verticalLayout_3->addWidget(info);

        tabWidget->addTab(tab_2, QString());

        verticalLayout->addWidget(tabWidget);

        buttonBox = new QDialogButtonBox(view_image);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(view_image);
        QObject::connect(buttonBox, SIGNAL(accepted()), view_image, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), view_image, SLOT(reject()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(view_image);
    } // setupUi

    void retranslateUi(QDialog *view_image)
    {
        view_image->setWindowTitle(QApplication::translate("view_image", "Dialog", 0, QApplication::UnicodeUTF8));
        image_info->setText(QApplication::translate("view_image", "Image Info", 0, QApplication::UnicodeUTF8));
        zoom_in->setText(QApplication::translate("view_image", "+", 0, QApplication::UnicodeUTF8));
        zoom_out->setText(QApplication::translate("view_image", "-", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("view_image", "Contrast", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("view_image", "Brightness", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("view_image", "Image", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("view_image", "Information", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class view_image: public Ui_view_image {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIEW_IMAGE_H
