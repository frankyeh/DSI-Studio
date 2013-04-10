/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Wed Apr 10 12:41:28 2013
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QCommandLinkButton>
#include <QtGui/QDockWidget>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QTableWidget>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_3;
    QTableWidget *recentSrc;
    QLabel *label_2;
    QTableWidget *recentFib;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QComboBox *workDir;
    QToolButton *browseDir;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QCommandLinkButton *OpenDICOM;
    QCommandLinkButton *Reconstruction;
    QCommandLinkButton *FiberTracking;
    QDockWidget *dockWidget_3;
    QWidget *dockWidgetContents_3;
    QVBoxLayout *verticalLayout_5;
    QCommandLinkButton *averagefib;
    QCommandLinkButton *vbc;
    QDockWidget *dockWidget_2;
    QWidget *dockWidgetContents_2;
    QVBoxLayout *verticalLayout;
    QCommandLinkButton *RenameDICOM;
    QCommandLinkButton *RenameDICOMDir;
    QCommandLinkButton *batch_src;
    QCommandLinkButton *batch_reconstruction;
    QCommandLinkButton *view_image;
    QCommandLinkButton *simulateMRI;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(885, 616);
        QFont font;
        font.setFamily(QString::fromUtf8("Arial"));
        MainWindow->setFont(font);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/axial.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(centralWidget);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(-1, 0, -1, -1);
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        verticalLayout_4->addWidget(label_3);

        recentSrc = new QTableWidget(centralWidget);
        if (recentSrc->columnCount() < 2)
            recentSrc->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        recentSrc->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        recentSrc->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        recentSrc->setObjectName(QString::fromUtf8("recentSrc"));
        recentSrc->setSelectionMode(QAbstractItemView::SingleSelection);
        recentSrc->setSelectionBehavior(QAbstractItemView::SelectRows);

        verticalLayout_4->addWidget(recentSrc);

        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_4->addWidget(label_2);

        recentFib = new QTableWidget(centralWidget);
        if (recentFib->columnCount() < 2)
            recentFib->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        recentFib->setHorizontalHeaderItem(0, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        recentFib->setHorizontalHeaderItem(1, __qtablewidgetitem3);
        recentFib->setObjectName(QString::fromUtf8("recentFib"));
        recentFib->setSelectionMode(QAbstractItemView::SingleSelection);
        recentFib->setSelectionBehavior(QAbstractItemView::SelectRows);

        verticalLayout_4->addWidget(recentFib);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(centralWidget);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy1);

        horizontalLayout_2->addWidget(label);

        workDir = new QComboBox(centralWidget);
        workDir->setObjectName(QString::fromUtf8("workDir"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(workDir->sizePolicy().hasHeightForWidth());
        workDir->setSizePolicy(sizePolicy2);
        workDir->setEditable(true);

        horizontalLayout_2->addWidget(workDir);

        browseDir = new QToolButton(centralWidget);
        browseDir->setObjectName(QString::fromUtf8("browseDir"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        browseDir->setIcon(icon1);
        browseDir->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_2->addWidget(browseDir);


        verticalLayout_4->addLayout(horizontalLayout_2);


        verticalLayout_3->addLayout(verticalLayout_4);

        MainWindow->setCentralWidget(centralWidget);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(dockWidget->sizePolicy().hasHeightForWidth());
        dockWidget->setSizePolicy(sizePolicy3);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_2->setSpacing(4);
        verticalLayout_2->setContentsMargins(4, 4, 4, 4);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        OpenDICOM = new QCommandLinkButton(dockWidgetContents);
        OpenDICOM->setObjectName(QString::fromUtf8("OpenDICOM"));

        verticalLayout_2->addWidget(OpenDICOM);

        Reconstruction = new QCommandLinkButton(dockWidgetContents);
        Reconstruction->setObjectName(QString::fromUtf8("Reconstruction"));

        verticalLayout_2->addWidget(Reconstruction);

        FiberTracking = new QCommandLinkButton(dockWidgetContents);
        FiberTracking->setObjectName(QString::fromUtf8("FiberTracking"));

        verticalLayout_2->addWidget(FiberTracking);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        dockWidget_3 = new QDockWidget(MainWindow);
        dockWidget_3->setObjectName(QString::fromUtf8("dockWidget_3"));
        dockWidget_3->setEnabled(true);
        sizePolicy3.setHeightForWidth(dockWidget_3->sizePolicy().hasHeightForWidth());
        dockWidget_3->setSizePolicy(sizePolicy3);
        dockWidget_3->setFloating(true);
        dockWidgetContents_3 = new QWidget();
        dockWidgetContents_3->setObjectName(QString::fromUtf8("dockWidgetContents_3"));
        verticalLayout_5 = new QVBoxLayout(dockWidgetContents_3);
        verticalLayout_5->setSpacing(4);
        verticalLayout_5->setContentsMargins(4, 4, 4, 4);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        averagefib = new QCommandLinkButton(dockWidgetContents_3);
        averagefib->setObjectName(QString::fromUtf8("averagefib"));

        verticalLayout_5->addWidget(averagefib);

        vbc = new QCommandLinkButton(dockWidgetContents_3);
        vbc->setObjectName(QString::fromUtf8("vbc"));

        verticalLayout_5->addWidget(vbc);

        dockWidget_3->setWidget(dockWidgetContents_3);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_3);
        dockWidget_2 = new QDockWidget(MainWindow);
        dockWidget_2->setObjectName(QString::fromUtf8("dockWidget_2"));
        sizePolicy3.setHeightForWidth(dockWidget_2->sizePolicy().hasHeightForWidth());
        dockWidget_2->setSizePolicy(sizePolicy3);
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QString::fromUtf8("dockWidgetContents_2"));
        verticalLayout = new QVBoxLayout(dockWidgetContents_2);
        verticalLayout->setSpacing(4);
        verticalLayout->setContentsMargins(4, 4, 4, 4);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        RenameDICOM = new QCommandLinkButton(dockWidgetContents_2);
        RenameDICOM->setObjectName(QString::fromUtf8("RenameDICOM"));

        verticalLayout->addWidget(RenameDICOM);

        RenameDICOMDir = new QCommandLinkButton(dockWidgetContents_2);
        RenameDICOMDir->setObjectName(QString::fromUtf8("RenameDICOMDir"));

        verticalLayout->addWidget(RenameDICOMDir);

        batch_src = new QCommandLinkButton(dockWidgetContents_2);
        batch_src->setObjectName(QString::fromUtf8("batch_src"));

        verticalLayout->addWidget(batch_src);

        batch_reconstruction = new QCommandLinkButton(dockWidgetContents_2);
        batch_reconstruction->setObjectName(QString::fromUtf8("batch_reconstruction"));

        verticalLayout->addWidget(batch_reconstruction);

        view_image = new QCommandLinkButton(dockWidgetContents_2);
        view_image->setObjectName(QString::fromUtf8("view_image"));

        verticalLayout->addWidget(view_image);

        simulateMRI = new QCommandLinkButton(dockWidgetContents_2);
        simulateMRI->setObjectName(QString::fromUtf8("simulateMRI"));

        verticalLayout->addWidget(simulateMRI);

        dockWidget_2->setWidget(dockWidgetContents_2);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_2);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "DSI Studio", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWindow", "Recent src files: double click to open", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = recentSrc->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("MainWindow", "File Name", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = recentSrc->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("MainWindow", "Directory", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWindow", "Recent fib files: double click to open", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = recentFib->horizontalHeaderItem(0);
        ___qtablewidgetitem2->setText(QApplication::translate("MainWindow", "File Name", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = recentFib->horizontalHeaderItem(1);
        ___qtablewidgetitem3->setText(QApplication::translate("MainWindow", "Directory", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWindow", "Working Directory", 0, QApplication::UnicodeUTF8));
        browseDir->setText(QApplication::translate("MainWindow", "Browse...", 0, QApplication::UnicodeUTF8));
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Diffusion MR Fiber Tracking", 0, QApplication::UnicodeUTF8));
        OpenDICOM->setText(QApplication::translate("MainWindow", "STEP1: Open Source Images", 0, QApplication::UnicodeUTF8));
        OpenDICOM->setDescription(QApplication::translate("MainWindow", "Open DICOM, NIFTI files to create .src file", 0, QApplication::UnicodeUTF8));
        Reconstruction->setText(QApplication::translate("MainWindow", "STEP2: Reconstruction", 0, QApplication::UnicodeUTF8));
        Reconstruction->setDescription(QApplication::translate("MainWindow", "Open .src file to reconstruct DTI, QBI, DSI, or GQI", 0, QApplication::UnicodeUTF8));
        FiberTracking->setText(QApplication::translate("MainWindow", "STEP3: Fiber tracking", 0, QApplication::UnicodeUTF8));
        FiberTracking->setDescription(QApplication::translate("MainWindow", "Open .fib file to perform fiber tracking", 0, QApplication::UnicodeUTF8));
        dockWidget_3->setWindowTitle(QApplication::translate("MainWindow", "Fiber-Based Connectometry", 0, QApplication::UnicodeUTF8));
        averagefib->setText(QApplication::translate("MainWindow", "STEP1: Average FIB Files", 0, QApplication::UnicodeUTF8));
        averagefib->setDescription(QApplication::translate("MainWindow", "Average the ODFs to create an atlas.", 0, QApplication::UnicodeUTF8));
        vbc->setText(QApplication::translate("MainWindow", "STEP2: Permutation Tests", 0, QApplication::UnicodeUTF8));
        vbc->setDescription(QApplication::translate("MainWindow", "Perform permutation to obtain statistical mapping", 0, QApplication::UnicodeUTF8));
        dockWidget_2->setWindowTitle(QApplication::translate("MainWindow", "Tools", 0, QApplication::UnicodeUTF8));
        RenameDICOM->setText(QApplication::translate("MainWindow", "Rename DICOM Files", 0, QApplication::UnicodeUTF8));
        RenameDICOM->setDescription(QApplication::translate("MainWindow", "Separate files according to their acquisition sequences", 0, QApplication::UnicodeUTF8));
        RenameDICOMDir->setText(QApplication::translate("MainWindow", "Rename DICOM Files", 0, QApplication::UnicodeUTF8));
        RenameDICOMDir->setDescription(QApplication::translate("MainWindow", "Apply to all subdirectoires", 0, QApplication::UnicodeUTF8));
        batch_src->setText(QApplication::translate("MainWindow", "Batch SRC Files Creation", 0, QApplication::UnicodeUTF8));
        batch_src->setDescription(QApplication::translate("MainWindow", "Generate src file for each subdirectory", 0, QApplication::UnicodeUTF8));
        batch_reconstruction->setText(QApplication::translate("MainWindow", "Batch Reconstruction", 0, QApplication::UnicodeUTF8));
        batch_reconstruction->setDescription(QApplication::translate("MainWindow", "Select a directory that contains src file in the subdirectories", 0, QApplication::UnicodeUTF8));
        view_image->setText(QApplication::translate("MainWindow", "View Images (NIFTI/DICOM/2dseq)", 0, QApplication::UnicodeUTF8));
        simulateMRI->setText(QApplication::translate("MainWindow", "Diffusion MRI Simulation", 0, QApplication::UnicodeUTF8));
        simulateMRI->setDescription(QApplication::translate("MainWindow", "Simulate diffusion images using the given b-table", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
