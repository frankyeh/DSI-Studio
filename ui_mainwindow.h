/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Thu Oct 9 21:21:56 2014
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
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QSpacerItem>
#include <QtGui/QTableWidget>
#include <QtGui/QToolBox>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QToolBox *toolBox;
    QWidget *page;
    QVBoxLayout *verticalLayout_6;
    QCommandLinkButton *OpenDICOM;
    QCommandLinkButton *Reconstruction;
    QCommandLinkButton *FiberTracking;
    QSpacerItem *verticalSpacer_4;
    QWidget *page_2;
    QVBoxLayout *verticalLayout_7;
    QCommandLinkButton *averagefib;
    QCommandLinkButton *vbc;
    QCommandLinkButton *connectometry;
    QSpacerItem *verticalSpacer_3;
    QWidget *page_3;
    QVBoxLayout *verticalLayout_8;
    QCommandLinkButton *RenameDICOM;
    QCommandLinkButton *RenameDICOMDir;
    QCommandLinkButton *batch_src;
    QCommandLinkButton *batch_reconstruction;
    QSpacerItem *verticalSpacer_2;
    QWidget *page_4;
    QVBoxLayout *verticalLayout_9;
    QCommandLinkButton *view_image;
    QCommandLinkButton *simulateMRI;
    QSpacerItem *verticalSpacer;
    QWidget *widget;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_3;
    QTableWidget *recentSrc;
    QLabel *label_2;
    QTableWidget *recentFib;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QComboBox *workDir;
    QToolButton *browseDir;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(832, 464);
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
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        toolBox = new QToolBox(centralWidget);
        toolBox->setObjectName(QString::fromUtf8("toolBox"));
        toolBox->setMaximumSize(QSize(300, 16777215));
        page = new QWidget();
        page->setObjectName(QString::fromUtf8("page"));
        page->setGeometry(QRect(0, 0, 300, 376));
        verticalLayout_6 = new QVBoxLayout(page);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        OpenDICOM = new QCommandLinkButton(page);
        OpenDICOM->setObjectName(QString::fromUtf8("OpenDICOM"));

        verticalLayout_6->addWidget(OpenDICOM);

        Reconstruction = new QCommandLinkButton(page);
        Reconstruction->setObjectName(QString::fromUtf8("Reconstruction"));

        verticalLayout_6->addWidget(Reconstruction);

        FiberTracking = new QCommandLinkButton(page);
        FiberTracking->setObjectName(QString::fromUtf8("FiberTracking"));

        verticalLayout_6->addWidget(FiberTracking);

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_4);

        toolBox->addItem(page, QString::fromUtf8("Diffusion MRI Tractography"));
        page_2 = new QWidget();
        page_2->setObjectName(QString::fromUtf8("page_2"));
        page_2->setGeometry(QRect(0, 0, 300, 376));
        verticalLayout_7 = new QVBoxLayout(page_2);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        averagefib = new QCommandLinkButton(page_2);
        averagefib->setObjectName(QString::fromUtf8("averagefib"));

        verticalLayout_7->addWidget(averagefib);

        vbc = new QCommandLinkButton(page_2);
        vbc->setObjectName(QString::fromUtf8("vbc"));

        verticalLayout_7->addWidget(vbc);

        connectometry = new QCommandLinkButton(page_2);
        connectometry->setObjectName(QString::fromUtf8("connectometry"));

        verticalLayout_7->addWidget(connectometry);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_3);

        toolBox->addItem(page_2, QString::fromUtf8("Diffusion MRI Connectometry"));
        page_3 = new QWidget();
        page_3->setObjectName(QString::fromUtf8("page_3"));
        page_3->setGeometry(QRect(0, 0, 300, 376));
        verticalLayout_8 = new QVBoxLayout(page_3);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setContentsMargins(0, 0, 0, 0);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        RenameDICOM = new QCommandLinkButton(page_3);
        RenameDICOM->setObjectName(QString::fromUtf8("RenameDICOM"));

        verticalLayout_8->addWidget(RenameDICOM);

        RenameDICOMDir = new QCommandLinkButton(page_3);
        RenameDICOMDir->setObjectName(QString::fromUtf8("RenameDICOMDir"));

        verticalLayout_8->addWidget(RenameDICOMDir);

        batch_src = new QCommandLinkButton(page_3);
        batch_src->setObjectName(QString::fromUtf8("batch_src"));

        verticalLayout_8->addWidget(batch_src);

        batch_reconstruction = new QCommandLinkButton(page_3);
        batch_reconstruction->setObjectName(QString::fromUtf8("batch_reconstruction"));

        verticalLayout_8->addWidget(batch_reconstruction);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_8->addItem(verticalSpacer_2);

        toolBox->addItem(page_3, QString::fromUtf8("Tools: batch processing"));
        page_4 = new QWidget();
        page_4->setObjectName(QString::fromUtf8("page_4"));
        page_4->setGeometry(QRect(0, 0, 300, 376));
        verticalLayout_9 = new QVBoxLayout(page_4);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        view_image = new QCommandLinkButton(page_4);
        view_image->setObjectName(QString::fromUtf8("view_image"));

        verticalLayout_9->addWidget(view_image);

        simulateMRI = new QCommandLinkButton(page_4);
        simulateMRI->setObjectName(QString::fromUtf8("simulateMRI"));

        verticalLayout_9->addWidget(simulateMRI);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_9->addItem(verticalSpacer);

        toolBox->addItem(page_4, QString::fromUtf8("Tools: others"));

        horizontalLayout->addWidget(toolBox);

        widget = new QWidget(centralWidget);
        widget->setObjectName(QString::fromUtf8("widget"));
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(widget);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label_3 = new QLabel(widget);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy1);

        verticalLayout_2->addWidget(label_3);

        recentSrc = new QTableWidget(widget);
        if (recentSrc->columnCount() < 2)
            recentSrc->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        recentSrc->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        recentSrc->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        recentSrc->setObjectName(QString::fromUtf8("recentSrc"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(recentSrc->sizePolicy().hasHeightForWidth());
        recentSrc->setSizePolicy(sizePolicy2);
        recentSrc->setMaximumSize(QSize(16777215, 16777215));
        recentSrc->setSelectionMode(QAbstractItemView::SingleSelection);
        recentSrc->setSelectionBehavior(QAbstractItemView::SelectRows);

        verticalLayout_2->addWidget(recentSrc);

        label_2 = new QLabel(widget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        sizePolicy1.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy1);

        verticalLayout_2->addWidget(label_2);

        recentFib = new QTableWidget(widget);
        if (recentFib->columnCount() < 2)
            recentFib->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        recentFib->setHorizontalHeaderItem(0, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        recentFib->setHorizontalHeaderItem(1, __qtablewidgetitem3);
        recentFib->setObjectName(QString::fromUtf8("recentFib"));
        sizePolicy2.setHeightForWidth(recentFib->sizePolicy().hasHeightForWidth());
        recentFib->setSizePolicy(sizePolicy2);
        recentFib->setMaximumSize(QSize(16777215, 16777215));
        recentFib->setSelectionMode(QAbstractItemView::SingleSelection);
        recentFib->setSelectionBehavior(QAbstractItemView::SelectRows);

        verticalLayout_2->addWidget(recentFib);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(widget);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy3);

        horizontalLayout_2->addWidget(label);

        workDir = new QComboBox(widget);
        workDir->setObjectName(QString::fromUtf8("workDir"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(workDir->sizePolicy().hasHeightForWidth());
        workDir->setSizePolicy(sizePolicy4);
        workDir->setMaximumSize(QSize(16777215, 16777215));
        workDir->setEditable(false);
        workDir->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLength);

        horizontalLayout_2->addWidget(workDir);

        browseDir = new QToolButton(widget);
        browseDir->setObjectName(QString::fromUtf8("browseDir"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        browseDir->setIcon(icon1);
        browseDir->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_2->addWidget(browseDir);


        verticalLayout_2->addLayout(horizontalLayout_2);


        verticalLayout_3->addLayout(verticalLayout_2);


        horizontalLayout->addWidget(widget);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        toolBox->setCurrentIndex(0);
        toolBox->layout()->setSpacing(0);
        workDir->setCurrentIndex(-1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "DSI Studio", 0, QApplication::UnicodeUTF8));
        OpenDICOM->setText(QApplication::translate("MainWindow", "STEP1: Open Source Images", 0, QApplication::UnicodeUTF8));
        OpenDICOM->setDescription(QApplication::translate("MainWindow", "Open diffusion MR images to create .src file\n"
"(DICOM, NIFTI, Bruker 2dseq, Varian fdf) ", 0, QApplication::UnicodeUTF8));
        Reconstruction->setText(QApplication::translate("MainWindow", "STEP2: Reconstruction", 0, QApplication::UnicodeUTF8));
        Reconstruction->setDescription(QApplication::translate("MainWindow", "Open .src file to do reconstructiong\n"
"(DTI, QBI, DSI, GQI, or QSDR)", 0, QApplication::UnicodeUTF8));
        FiberTracking->setText(QApplication::translate("MainWindow", "STEP3: Fiber tracking", 0, QApplication::UnicodeUTF8));
        FiberTracking->setDescription(QApplication::translate("MainWindow", "Open .fib file to perform fiber tracking and analysis\n"
"(track-specific analysis, connectivity matrix) \n"
"", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("MainWindow", "Diffusion MRI Tractography", 0, QApplication::UnicodeUTF8));
        averagefib->setText(QApplication::translate("MainWindow", "STEP1: Create template/skeleton", 0, QApplication::UnicodeUTF8));
        averagefib->setDescription(QApplication::translate("MainWindow", "Average the ODFs to create a template or skeleton. You need to use QSDR to reconstruct src files and check \"output ODF\" in the reconstruction.", 0, QApplication::UnicodeUTF8));
        vbc->setText(QApplication::translate("MainWindow", "STEP2: Create connectometry database", 0, QApplication::UnicodeUTF8));
        vbc->setDescription(QApplication::translate("MainWindow", "Load a group of subjects to create a connectometry dataset", 0, QApplication::UnicodeUTF8));
        connectometry->setText(QApplication::translate("MainWindow", "STEP3: Connectometry Analysis ", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("MainWindow", "Diffusion MRI Connectometry", 0, QApplication::UnicodeUTF8));
        RenameDICOM->setText(QApplication::translate("MainWindow", "Rename DICOM Files", 0, QApplication::UnicodeUTF8));
        RenameDICOM->setDescription(QApplication::translate("MainWindow", "Sort and rename DICOM files according to their sequences", 0, QApplication::UnicodeUTF8));
        RenameDICOMDir->setText(QApplication::translate("MainWindow", "Rename DICOM Files", 0, QApplication::UnicodeUTF8));
        RenameDICOMDir->setDescription(QApplication::translate("MainWindow", "Select a directory containinng DICOM files and rename them by their sequence", 0, QApplication::UnicodeUTF8));
        batch_src->setText(QApplication::translate("MainWindow", "Create SRC files ", 0, QApplication::UnicodeUTF8));
        batch_src->setDescription(QApplication::translate("MainWindow", "Select a root directory that contains multiple subdirectory and generate an SRC file for each of them.", 0, QApplication::UnicodeUTF8));
        batch_reconstruction->setText(QApplication::translate("MainWindow", "Batch SRC Reconstruction", 0, QApplication::UnicodeUTF8));
        batch_reconstruction->setDescription(QApplication::translate("MainWindow", "Select a directory that contains SRC files in the subdirectories and performan reconstruction.", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_3), QApplication::translate("MainWindow", "Tools: batch processing", 0, QApplication::UnicodeUTF8));
        view_image->setText(QApplication::translate("MainWindow", "View Images (NIFTI/DICOM/2dseq)", 0, QApplication::UnicodeUTF8));
        view_image->setDescription(QApplication::translate("MainWindow", "Open image and header information", 0, QApplication::UnicodeUTF8));
        simulateMRI->setText(QApplication::translate("MainWindow", "Diffusion MRI Simulation", 0, QApplication::UnicodeUTF8));
        simulateMRI->setDescription(QApplication::translate("MainWindow", "Simulate diffusion images using the given b-table", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_4), QApplication::translate("MainWindow", "Tools: others", 0, QApplication::UnicodeUTF8));
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
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
