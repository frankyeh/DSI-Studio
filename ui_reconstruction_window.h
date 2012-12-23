/********************************************************************************
** Form generated from reading UI file 'reconstruction_window.ui'
**
** Created: Sat Dec 22 23:56:19 2012
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_RECONSTRUCTION_WINDOW_H
#define UI_RECONSTRUCTION_WINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QCommandLinkButton>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGraphicsView>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QSplitter>
#include <QtGui/QTableWidget>
#include <QtGui/QToolBox>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_reconstruction_window
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QToolBox *toolBox;
    QWidget *source_page;
    QVBoxLayout *verticalLayout_4;
    QSplitter *splitter;
    QTableWidget *b_table;
    QWidget *source_widget;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout_11;
    QToolButton *zoom_in;
    QToolButton *zoom_out;
    QLabel *label_14;
    QSlider *contrast;
    QLabel *label_13;
    QSlider *brightness;
    QGraphicsView *view_source;
    QSlider *z_pos;
    QWidget *page_3;
    QGridLayout *gridLayout_2;
    QVBoxLayout *verticalLayout_5;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QToolButton *load_mask;
    QToolButton *save_mask;
    QToolButton *thresholding;
    QToolButton *dilation;
    QToolButton *erosion;
    QToolButton *smoothing;
    QToolButton *defragment;
    QToolButton *remove_background;
    QSpacerItem *horizontalSpacer_2;
    QGraphicsView *graphicsView;
    QSlider *SlicePos;
    QWidget *page;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *DTI;
    QRadioButton *DSI;
    QRadioButton *QBI;
    QRadioButton *GQI;
    QRadioButton *QDif;
    QRadioButton *QSDRT;
    QGroupBox *ResolutionBox;
    QGridLayout *gridLayout_3;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_8;
    QDoubleSpinBox *mni_resolution;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_15;
    QComboBox *reg_method;
    QGroupBox *DSIOption_2;
    QHBoxLayout *DSIOption;
    QLabel *label_5;
    QSpinBox *hamming_filter;
    QGroupBox *QBIOption_2;
    QHBoxLayout *QBIOption;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label;
    QSpinBox *SHOrder;
    QHBoxLayout *horizontalLayout_13;
    QLabel *label_6;
    QDoubleSpinBox *regularization_param;
    QGroupBox *GQIOption_2;
    QHBoxLayout *GQIOption;
    QLabel *label_7;
    QDoubleSpinBox *diffusion_sampling;
    QComboBox *ODFDef;
    QGroupBox *OptionGroupBox;
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_2;
    QComboBox *ODFDim;
    QCheckBox *RecordODF;
    QCheckBox *HalfSphere;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_9;
    QComboBox *Decomposition;
    QLabel *label_12;
    QDoubleSpinBox *decompose_fraction;
    QLabel *label_11;
    QSpinBox *decom_m;
    QHBoxLayout *xyz;
    QWidget *xyz_widget;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_10;
    QSpinBox *x;
    QSpinBox *y;
    QSpinBox *z;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_4;
    QComboBox *ODFSharpening;
    QDoubleSpinBox *SharpeningParam;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_3;
    QSpinBox *NumOfFibers;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout_7;
    QSpacerItem *horizontalSpacer;
    QComboBox *ThreadCount;
    QCommandLinkButton *doDTI;

    void setupUi(QMainWindow *reconstruction_window)
    {
        if (reconstruction_window->objectName().isEmpty())
            reconstruction_window->setObjectName(QString::fromUtf8("reconstruction_window"));
        reconstruction_window->resize(608, 537);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(reconstruction_window->sizePolicy().hasHeightForWidth());
        reconstruction_window->setSizePolicy(sizePolicy);
        reconstruction_window->setMinimumSize(QSize(550, 430));
        QFont font;
        font.setFamily(QString::fromUtf8("Arial"));
        reconstruction_window->setFont(font);
        centralwidget = new QWidget(reconstruction_window);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        toolBox = new QToolBox(centralwidget);
        toolBox->setObjectName(QString::fromUtf8("toolBox"));
        source_page = new QWidget();
        source_page->setObjectName(QString::fromUtf8("source_page"));
        source_page->setGeometry(QRect(0, 0, 259, 130));
        verticalLayout_4 = new QVBoxLayout(source_page);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        splitter = new QSplitter(source_page);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        b_table = new QTableWidget(splitter);
        if (b_table->columnCount() < 4)
            b_table->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        b_table->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        b_table->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        b_table->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        b_table->setHorizontalHeaderItem(3, __qtablewidgetitem3);
        b_table->setObjectName(QString::fromUtf8("b_table"));
        b_table->setSelectionMode(QAbstractItemView::SingleSelection);
        b_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        splitter->addWidget(b_table);
        source_widget = new QWidget(splitter);
        source_widget->setObjectName(QString::fromUtf8("source_widget"));
        verticalLayout_6 = new QVBoxLayout(source_widget);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setSpacing(0);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        zoom_in = new QToolButton(source_widget);
        zoom_in->setObjectName(QString::fromUtf8("zoom_in"));
        zoom_in->setMinimumSize(QSize(24, 0));

        horizontalLayout_11->addWidget(zoom_in);

        zoom_out = new QToolButton(source_widget);
        zoom_out->setObjectName(QString::fromUtf8("zoom_out"));
        zoom_out->setMinimumSize(QSize(24, 0));

        horizontalLayout_11->addWidget(zoom_out);

        label_14 = new QLabel(source_widget);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        horizontalLayout_11->addWidget(label_14);

        contrast = new QSlider(source_widget);
        contrast->setObjectName(QString::fromUtf8("contrast"));
        contrast->setMinimum(1);
        contrast->setMaximum(30);
        contrast->setOrientation(Qt::Horizontal);

        horizontalLayout_11->addWidget(contrast);

        label_13 = new QLabel(source_widget);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_11->addWidget(label_13);

        brightness = new QSlider(source_widget);
        brightness->setObjectName(QString::fromUtf8("brightness"));
        brightness->setMinimum(-10);
        brightness->setMaximum(10);
        brightness->setOrientation(Qt::Horizontal);

        horizontalLayout_11->addWidget(brightness);


        verticalLayout_6->addLayout(horizontalLayout_11);

        view_source = new QGraphicsView(source_widget);
        view_source->setObjectName(QString::fromUtf8("view_source"));

        verticalLayout_6->addWidget(view_source);

        z_pos = new QSlider(source_widget);
        z_pos->setObjectName(QString::fromUtf8("z_pos"));
        z_pos->setOrientation(Qt::Horizontal);

        verticalLayout_6->addWidget(z_pos);

        splitter->addWidget(source_widget);
        view_source->raise();
        z_pos->raise();

        verticalLayout_4->addWidget(splitter);

        toolBox->addItem(source_page, QString::fromUtf8("Source Images"));
        page_3 = new QWidget();
        page_3->setObjectName(QString::fromUtf8("page_3"));
        page_3->setGeometry(QRect(0, 0, 534, 146));
        gridLayout_2 = new QGridLayout(page_3);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        load_mask = new QToolButton(page_3);
        load_mask->setObjectName(QString::fromUtf8("load_mask"));

        horizontalLayout->addWidget(load_mask);

        save_mask = new QToolButton(page_3);
        save_mask->setObjectName(QString::fromUtf8("save_mask"));

        horizontalLayout->addWidget(save_mask);

        thresholding = new QToolButton(page_3);
        thresholding->setObjectName(QString::fromUtf8("thresholding"));

        horizontalLayout->addWidget(thresholding);

        dilation = new QToolButton(page_3);
        dilation->setObjectName(QString::fromUtf8("dilation"));

        horizontalLayout->addWidget(dilation);

        erosion = new QToolButton(page_3);
        erosion->setObjectName(QString::fromUtf8("erosion"));

        horizontalLayout->addWidget(erosion);

        smoothing = new QToolButton(page_3);
        smoothing->setObjectName(QString::fromUtf8("smoothing"));

        horizontalLayout->addWidget(smoothing);

        defragment = new QToolButton(page_3);
        defragment->setObjectName(QString::fromUtf8("defragment"));

        horizontalLayout->addWidget(defragment);

        remove_background = new QToolButton(page_3);
        remove_background->setObjectName(QString::fromUtf8("remove_background"));

        horizontalLayout->addWidget(remove_background);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        verticalLayout_3->addLayout(horizontalLayout);

        graphicsView = new QGraphicsView(page_3);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));

        verticalLayout_3->addWidget(graphicsView);

        SlicePos = new QSlider(page_3);
        SlicePos->setObjectName(QString::fromUtf8("SlicePos"));
        SlicePos->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(SlicePos);


        verticalLayout_5->addLayout(verticalLayout_3);


        gridLayout_2->addLayout(verticalLayout_5, 0, 0, 1, 1);

        toolBox->addItem(page_3, QString::fromUtf8("Step 1: setup brain mask"));
        page = new QWidget();
        page->setObjectName(QString::fromUtf8("page"));
        page->setGeometry(QRect(0, 0, 591, 494));
        page->setMinimumSize(QSize(0, 0));
        verticalLayout_2 = new QVBoxLayout(page);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        groupBox = new QGroupBox(page);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_2 = new QHBoxLayout(groupBox);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        DTI = new QRadioButton(groupBox);
        DTI->setObjectName(QString::fromUtf8("DTI"));
        DTI->setChecked(false);

        horizontalLayout_2->addWidget(DTI);

        DSI = new QRadioButton(groupBox);
        DSI->setObjectName(QString::fromUtf8("DSI"));

        horizontalLayout_2->addWidget(DSI);

        QBI = new QRadioButton(groupBox);
        QBI->setObjectName(QString::fromUtf8("QBI"));

        horizontalLayout_2->addWidget(QBI);

        GQI = new QRadioButton(groupBox);
        GQI->setObjectName(QString::fromUtf8("GQI"));
        GQI->setChecked(true);

        horizontalLayout_2->addWidget(GQI);

        QDif = new QRadioButton(groupBox);
        QDif->setObjectName(QString::fromUtf8("QDif"));
        QDif->setEnabled(true);

        horizontalLayout_2->addWidget(QDif);

        QSDRT = new QRadioButton(groupBox);
        QSDRT->setObjectName(QString::fromUtf8("QSDRT"));

        horizontalLayout_2->addWidget(QSDRT);


        verticalLayout_2->addWidget(groupBox);

        ResolutionBox = new QGroupBox(page);
        ResolutionBox->setObjectName(QString::fromUtf8("ResolutionBox"));
        gridLayout_3 = new QGridLayout(ResolutionBox);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_8 = new QLabel(ResolutionBox);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_8->addWidget(label_8);

        mni_resolution = new QDoubleSpinBox(ResolutionBox);
        mni_resolution->setObjectName(QString::fromUtf8("mni_resolution"));
        mni_resolution->setMaximumSize(QSize(75, 16777215));
        mni_resolution->setMinimum(0.5);
        mni_resolution->setMaximum(3);
        mni_resolution->setSingleStep(0.1);
        mni_resolution->setValue(2);

        horizontalLayout_8->addWidget(mni_resolution);


        gridLayout_3->addLayout(horizontalLayout_8, 0, 1, 1, 1);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_15 = new QLabel(ResolutionBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_10->addWidget(label_15);

        reg_method = new QComboBox(ResolutionBox);
        reg_method->setObjectName(QString::fromUtf8("reg_method"));

        horizontalLayout_10->addWidget(reg_method);


        gridLayout_3->addLayout(horizontalLayout_10, 0, 0, 1, 1);


        verticalLayout_2->addWidget(ResolutionBox);

        DSIOption_2 = new QGroupBox(page);
        DSIOption_2->setObjectName(QString::fromUtf8("DSIOption_2"));
        DSIOption = new QHBoxLayout(DSIOption_2);
        DSIOption->setObjectName(QString::fromUtf8("DSIOption"));
        label_5 = new QLabel(DSIOption_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        DSIOption->addWidget(label_5);

        hamming_filter = new QSpinBox(DSIOption_2);
        hamming_filter->setObjectName(QString::fromUtf8("hamming_filter"));
        hamming_filter->setMaximumSize(QSize(75, 16777215));
        hamming_filter->setMinimum(8);
        hamming_filter->setMaximum(20);
        hamming_filter->setValue(16);

        DSIOption->addWidget(hamming_filter);


        verticalLayout_2->addWidget(DSIOption_2);

        QBIOption_2 = new QGroupBox(page);
        QBIOption_2->setObjectName(QString::fromUtf8("QBIOption_2"));
        QBIOption = new QHBoxLayout(QBIOption_2);
        QBIOption->setObjectName(QString::fromUtf8("QBIOption"));
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label = new QLabel(QBIOption_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_12->addWidget(label);

        SHOrder = new QSpinBox(QBIOption_2);
        SHOrder->setObjectName(QString::fromUtf8("SHOrder"));
        SHOrder->setMinimum(4);
        SHOrder->setMaximum(10);
        SHOrder->setValue(8);

        horizontalLayout_12->addWidget(SHOrder);


        QBIOption->addLayout(horizontalLayout_12);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        label_6 = new QLabel(QBIOption_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_13->addWidget(label_6);

        regularization_param = new QDoubleSpinBox(QBIOption_2);
        regularization_param->setObjectName(QString::fromUtf8("regularization_param"));
        regularization_param->setMaximumSize(QSize(75, 16777215));
        regularization_param->setDecimals(3);
        regularization_param->setMinimum(0.001);
        regularization_param->setMaximum(1);
        regularization_param->setSingleStep(0.001);
        regularization_param->setValue(0.006);

        horizontalLayout_13->addWidget(regularization_param);


        QBIOption->addLayout(horizontalLayout_13);


        verticalLayout_2->addWidget(QBIOption_2);

        GQIOption_2 = new QGroupBox(page);
        GQIOption_2->setObjectName(QString::fromUtf8("GQIOption_2"));
        GQIOption = new QHBoxLayout(GQIOption_2);
        GQIOption->setObjectName(QString::fromUtf8("GQIOption"));
        label_7 = new QLabel(GQIOption_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        GQIOption->addWidget(label_7);

        diffusion_sampling = new QDoubleSpinBox(GQIOption_2);
        diffusion_sampling->setObjectName(QString::fromUtf8("diffusion_sampling"));
        diffusion_sampling->setMaximumSize(QSize(75, 16777215));
        diffusion_sampling->setDecimals(2);
        diffusion_sampling->setMinimum(0.01);
        diffusion_sampling->setMaximum(3);
        diffusion_sampling->setSingleStep(0.05);
        diffusion_sampling->setValue(1.25);

        GQIOption->addWidget(diffusion_sampling);

        ODFDef = new QComboBox(GQIOption_2);
        ODFDef->setObjectName(QString::fromUtf8("ODFDef"));
        ODFDef->setMaximumSize(QSize(150, 16777215));

        GQIOption->addWidget(ODFDef);


        verticalLayout_2->addWidget(GQIOption_2);

        OptionGroupBox = new QGroupBox(page);
        OptionGroupBox->setObjectName(QString::fromUtf8("OptionGroupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(OptionGroupBox->sizePolicy().hasHeightForWidth());
        OptionGroupBox->setSizePolicy(sizePolicy1);
        gridLayout = new QGridLayout(OptionGroupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_2 = new QLabel(OptionGroupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        ODFDim = new QComboBox(OptionGroupBox);
        ODFDim->setObjectName(QString::fromUtf8("ODFDim"));
        ODFDim->setMaximumSize(QSize(75, 16777215));

        horizontalLayout_3->addWidget(ODFDim);


        gridLayout->addLayout(horizontalLayout_3, 1, 0, 1, 1);

        RecordODF = new QCheckBox(OptionGroupBox);
        RecordODF->setObjectName(QString::fromUtf8("RecordODF"));

        gridLayout->addWidget(RecordODF, 7, 0, 1, 1);

        HalfSphere = new QCheckBox(OptionGroupBox);
        HalfSphere->setObjectName(QString::fromUtf8("HalfSphere"));
        HalfSphere->setChecked(false);

        gridLayout->addWidget(HalfSphere, 7, 1, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_9 = new QLabel(OptionGroupBox);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setMinimumSize(QSize(100, 0));
        label_9->setMaximumSize(QSize(80, 16777215));

        horizontalLayout_5->addWidget(label_9);

        Decomposition = new QComboBox(OptionGroupBox);
        Decomposition->setObjectName(QString::fromUtf8("Decomposition"));
        Decomposition->setMinimumSize(QSize(50, 0));
        Decomposition->setMaximumSize(QSize(50, 16777215));

        horizontalLayout_5->addWidget(Decomposition);

        label_12 = new QLabel(OptionGroupBox);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        horizontalLayout_5->addWidget(label_12);

        decompose_fraction = new QDoubleSpinBox(OptionGroupBox);
        decompose_fraction->setObjectName(QString::fromUtf8("decompose_fraction"));
        decompose_fraction->setMaximumSize(QSize(75, 16777215));
        decompose_fraction->setMinimum(0.01);
        decompose_fraction->setMaximum(0.5);
        decompose_fraction->setSingleStep(0.05);
        decompose_fraction->setValue(0.05);

        horizontalLayout_5->addWidget(decompose_fraction);

        label_11 = new QLabel(OptionGroupBox);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setMaximumSize(QSize(10, 16777215));

        horizontalLayout_5->addWidget(label_11);

        decom_m = new QSpinBox(OptionGroupBox);
        decom_m->setObjectName(QString::fromUtf8("decom_m"));
        decom_m->setMinimum(3);
        decom_m->setMaximum(20);
        decom_m->setValue(10);

        horizontalLayout_5->addWidget(decom_m);


        gridLayout->addLayout(horizontalLayout_5, 2, 1, 1, 1);

        xyz = new QHBoxLayout();
        xyz->setSpacing(0);
        xyz->setObjectName(QString::fromUtf8("xyz"));
        xyz->setContentsMargins(-1, 0, -1, -1);
        xyz_widget = new QWidget(OptionGroupBox);
        xyz_widget->setObjectName(QString::fromUtf8("xyz_widget"));
        horizontalLayout_4 = new QHBoxLayout(xyz_widget);
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_10 = new QLabel(xyz_widget);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setMinimumSize(QSize(100, 0));

        horizontalLayout_4->addWidget(label_10);

        x = new QSpinBox(xyz_widget);
        x->setObjectName(QString::fromUtf8("x"));

        horizontalLayout_4->addWidget(x);

        y = new QSpinBox(xyz_widget);
        y->setObjectName(QString::fromUtf8("y"));

        horizontalLayout_4->addWidget(y);

        z = new QSpinBox(xyz_widget);
        z->setObjectName(QString::fromUtf8("z"));

        horizontalLayout_4->addWidget(z);


        xyz->addWidget(xyz_widget);


        gridLayout->addLayout(xyz, 3, 1, 1, 1);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_4 = new QLabel(OptionGroupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setMinimumSize(QSize(100, 0));
        label_4->setMaximumSize(QSize(80, 16777215));

        horizontalLayout_9->addWidget(label_4);

        ODFSharpening = new QComboBox(OptionGroupBox);
        ODFSharpening->setObjectName(QString::fromUtf8("ODFSharpening"));
        ODFSharpening->setMinimumSize(QSize(50, 0));
        ODFSharpening->setMaximumSize(QSize(50, 16777215));

        horizontalLayout_9->addWidget(ODFSharpening);

        SharpeningParam = new QDoubleSpinBox(OptionGroupBox);
        SharpeningParam->setObjectName(QString::fromUtf8("SharpeningParam"));
        SharpeningParam->setMaximumSize(QSize(16777215, 16777215));
        SharpeningParam->setMinimum(0.01);
        SharpeningParam->setMaximum(20);
        SharpeningParam->setSingleStep(0.5);
        SharpeningParam->setValue(3);

        horizontalLayout_9->addWidget(SharpeningParam);


        gridLayout->addLayout(horizontalLayout_9, 1, 1, 1, 1);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_3 = new QLabel(OptionGroupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_6->addWidget(label_3);

        NumOfFibers = new QSpinBox(OptionGroupBox);
        NumOfFibers->setObjectName(QString::fromUtf8("NumOfFibers"));
        NumOfFibers->setMaximumSize(QSize(75, 16777215));
        NumOfFibers->setMinimum(3);
        NumOfFibers->setMaximum(20);
        NumOfFibers->setValue(5);

        horizontalLayout_6->addWidget(NumOfFibers);


        gridLayout->addLayout(horizontalLayout_6, 2, 0, 1, 1);


        verticalLayout_2->addWidget(OptionGroupBox);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer);

        ThreadCount = new QComboBox(page);
        ThreadCount->setObjectName(QString::fromUtf8("ThreadCount"));
        ThreadCount->setMaximumSize(QSize(75, 16777215));

        horizontalLayout_7->addWidget(ThreadCount);

        doDTI = new QCommandLinkButton(page);
        doDTI->setObjectName(QString::fromUtf8("doDTI"));
        doDTI->setMaximumSize(QSize(16777215, 50));

        horizontalLayout_7->addWidget(doDTI);


        verticalLayout_2->addLayout(horizontalLayout_7);

        toolBox->addItem(page, QString::fromUtf8("Step 2: select reconstruction method"));

        verticalLayout->addWidget(toolBox);

        reconstruction_window->setCentralWidget(centralwidget);

        retranslateUi(reconstruction_window);

        toolBox->setCurrentIndex(2);
        toolBox->layout()->setSpacing(6);
        ODFDim->setCurrentIndex(3);
        ThreadCount->setCurrentIndex(3);


        QMetaObject::connectSlotsByName(reconstruction_window);
    } // setupUi

    void retranslateUi(QMainWindow *reconstruction_window)
    {
        reconstruction_window->setWindowTitle(QApplication::translate("reconstruction_window", "Reconstruction", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = b_table->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("reconstruction_window", "b value", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = b_table->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("reconstruction_window", "bx", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = b_table->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("reconstruction_window", "by", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = b_table->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QApplication::translate("reconstruction_window", "bz", 0, QApplication::UnicodeUTF8));
        zoom_in->setText(QApplication::translate("reconstruction_window", "+", 0, QApplication::UnicodeUTF8));
        zoom_out->setText(QApplication::translate("reconstruction_window", "-", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("reconstruction_window", "Contrast", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("reconstruction_window", "Brightness", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(source_page), QApplication::translate("reconstruction_window", "Source Images", 0, QApplication::UnicodeUTF8));
        load_mask->setText(QApplication::translate("reconstruction_window", "Open...", 0, QApplication::UnicodeUTF8));
        save_mask->setText(QApplication::translate("reconstruction_window", "Save...", 0, QApplication::UnicodeUTF8));
        thresholding->setText(QApplication::translate("reconstruction_window", "Thresholding", 0, QApplication::UnicodeUTF8));
        dilation->setText(QApplication::translate("reconstruction_window", "Dilation", 0, QApplication::UnicodeUTF8));
        erosion->setText(QApplication::translate("reconstruction_window", "Erosion", 0, QApplication::UnicodeUTF8));
        smoothing->setText(QApplication::translate("reconstruction_window", "Smoothing", 0, QApplication::UnicodeUTF8));
        defragment->setText(QApplication::translate("reconstruction_window", "Defragment", 0, QApplication::UnicodeUTF8));
        remove_background->setText(QApplication::translate("reconstruction_window", "Remove background", 0, QApplication::UnicodeUTF8));
        toolBox->setItemText(toolBox->indexOf(page_3), QApplication::translate("reconstruction_window", "Step 1: setup brain mask", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("reconstruction_window", "Reconstruction Method", 0, QApplication::UnicodeUTF8));
        DTI->setText(QApplication::translate("reconstruction_window", "DTI", 0, QApplication::UnicodeUTF8));
        DSI->setText(QApplication::translate("reconstruction_window", "DSI", 0, QApplication::UnicodeUTF8));
        QBI->setText(QApplication::translate("reconstruction_window", "QBI", 0, QApplication::UnicodeUTF8));
        GQI->setText(QApplication::translate("reconstruction_window", "GQI", 0, QApplication::UnicodeUTF8));
        QDif->setText(QApplication::translate("reconstruction_window", "Q-Space Diffeomorphic", 0, QApplication::UnicodeUTF8));
        QSDRT->setText(QApplication::translate("reconstruction_window", "QSDR Template", 0, QApplication::UnicodeUTF8));
        ResolutionBox->setTitle(QString());
        label_8->setText(QApplication::translate("reconstruction_window", "Output Resolution", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("reconstruction_window", "Registration method", 0, QApplication::UnicodeUTF8));
        reg_method->clear();
        reg_method->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "SPM norm 7-9-7", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "SPM norm 12-14-12", 0, QApplication::UnicodeUTF8)
        );
        label_5->setText(QApplication::translate("reconstruction_window", "DSI Hamming filter", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("reconstruction_window", "SH order", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("reconstruction_window", "QBI Regularization", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("reconstruction_window", "GQI Diffusion sampling ratio", 0, QApplication::UnicodeUTF8));
        ODFDef->clear();
        ODFDef->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "No distance weighting", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "r-squared weighted", 0, QApplication::UnicodeUTF8)
        );
        OptionGroupBox->setTitle(QApplication::translate("reconstruction_window", "Options", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("reconstruction_window", "ODF Tessellation", 0, QApplication::UnicodeUTF8));
        ODFDim->clear();
        ODFDim->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "4-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "5-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "6-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "8-fold", 0, QApplication::UnicodeUTF8)
        );
        RecordODF->setText(QApplication::translate("reconstruction_window", "Record ODF in the output file", 0, QApplication::UnicodeUTF8));
        HalfSphere->setText(QApplication::translate("reconstruction_window", "Half-sphere scheme", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("reconstruction_window", "Decomposition", 0, QApplication::UnicodeUTF8));
        Decomposition->clear();
        Decomposition->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "On", 0, QApplication::UnicodeUTF8)
        );
        label_12->setText(QApplication::translate("reconstruction_window", "\317\265", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("reconstruction_window", "m", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("reconstruction_window", "ODF (x,y,z)=", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("reconstruction_window", "Deconvolution", 0, QApplication::UnicodeUTF8));
        ODFSharpening->clear();
        ODFSharpening->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "On", 0, QApplication::UnicodeUTF8)
        );
        label_3->setText(QApplication::translate("reconstruction_window", "Number of fibers resolved", 0, QApplication::UnicodeUTF8));
        ThreadCount->clear();
        ThreadCount->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "2 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "3 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "4 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "5 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "6 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "7 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "8 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "9 threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "10 threads", 0, QApplication::UnicodeUTF8)
        );
        doDTI->setText(QApplication::translate("reconstruction_window", "Run reconstruction", 0, QApplication::UnicodeUTF8));
        doDTI->setDescription(QString());
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("reconstruction_window", "Step 2: select reconstruction method", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class reconstruction_window: public Ui_reconstruction_window {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RECONSTRUCTION_WINDOW_H
