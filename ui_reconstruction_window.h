/********************************************************************************
** Form generated from reading UI file 'reconstruction_window.ui'
**
** Created: Wed Nov 20 16:47:42 2013
**      by: Qt User Interface Compiler version 4.8.4
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
#include <QtGui/QPushButton>
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
    QHBoxLayout *horizontalLayout_21;
    QSplitter *splitter;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_7;
    QPushButton *save4dnifti;
    QPushButton *save_b_table;
    QPushButton *save_bvals;
    QPushButton *save_bvec;
    QSpacerItem *horizontalSpacer_5;
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
    QRadioButton *RDSI;
    QRadioButton *QBI;
    QRadioButton *GQI;
    QRadioButton *QDif;
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
    QGroupBox *DSIOption_3;
    QHBoxLayout *horizontalLayout_22;
    QLabel *label_16;
    QDoubleSpinBox *edge_factor;
    QGroupBox *GQIOption_2;
    QHBoxLayout *GQIOption;
    QLabel *label_7;
    QDoubleSpinBox *diffusion_sampling;
    QComboBox *ODFDef;
    QGroupBox *ResolutionBox;
    QHBoxLayout *horizontalLayout_9;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_15;
    QComboBox *reg_method;
    QToolButton *manual_reg;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_8;
    QSpinBox *mni_resolution;
    QHBoxLayout *horizontalLayout_19;
    QPushButton *AdvancedOptions;
    QSpacerItem *horizontalSpacer_6;
    QWidget *AdvancedWidget;
    QVBoxLayout *verticalLayout_7;
    QGroupBox *ODFSharpening;
    QHBoxLayout *horizontalLayout_17;
    QHBoxLayout *horizontalLayout_14;
    QComboBox *odf_sharpening;
    QWidget *decom_panel;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_12;
    QDoubleSpinBox *decom_fraction;
    QLabel *label_11;
    QSpinBox *decom_m;
    QDoubleSpinBox *decon_param;
    QSpacerItem *horizontalSpacer_3;
    QHBoxLayout *xyz;
    QWidget *xyz_widget;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_4;
    QComboBox *RFSelection;
    QWidget *ODFSelection;
    QHBoxLayout *horizontalLayout_15;
    QLabel *label_10;
    QSpinBox *x;
    QSpinBox *y;
    QSpinBox *z;
    QGroupBox *Output;
    QHBoxLayout *horizontalLayout_16;
    QCheckBox *RecordODF;
    QCheckBox *output_mapping;
    QCheckBox *output_jacobian;
    QSpacerItem *horizontalSpacer_4;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout_18;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_3;
    QSpinBox *NumOfFibers;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_2;
    QComboBox *ODFDim;
    QCheckBox *balance_scheme;
    QCheckBox *HalfSphere;
    QSpacerItem *verticalSpacer;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_20;
    QSpacerItem *horizontalSpacer;
    QLabel *label_9;
    QComboBox *ThreadCount;
    QCommandLinkButton *doDTI;

    void setupUi(QMainWindow *reconstruction_window)
    {
        if (reconstruction_window->objectName().isEmpty())
            reconstruction_window->setObjectName(QString::fromUtf8("reconstruction_window"));
        reconstruction_window->resize(761, 688);
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
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        toolBox = new QToolBox(centralwidget);
        toolBox->setObjectName(QString::fromUtf8("toolBox"));
        source_page = new QWidget();
        source_page->setObjectName(QString::fromUtf8("source_page"));
        source_page->setGeometry(QRect(0, 0, 761, 622));
        horizontalLayout_21 = new QHBoxLayout(source_page);
        horizontalLayout_21->setObjectName(QString::fromUtf8("horizontalLayout_21"));
        splitter = new QSplitter(source_page);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        widget_2 = new QWidget(splitter);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        verticalLayout_4 = new QVBoxLayout(widget_2);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        save4dnifti = new QPushButton(widget_2);
        save4dnifti->setObjectName(QString::fromUtf8("save4dnifti"));
        save4dnifti->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_7->addWidget(save4dnifti);

        save_b_table = new QPushButton(widget_2);
        save_b_table->setObjectName(QString::fromUtf8("save_b_table"));
        save_b_table->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_7->addWidget(save_b_table);

        save_bvals = new QPushButton(widget_2);
        save_bvals->setObjectName(QString::fromUtf8("save_bvals"));
        save_bvals->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_7->addWidget(save_bvals);

        save_bvec = new QPushButton(widget_2);
        save_bvec->setObjectName(QString::fromUtf8("save_bvec"));
        save_bvec->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_7->addWidget(save_bvec);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_5);


        verticalLayout_4->addLayout(horizontalLayout_7);

        b_table = new QTableWidget(widget_2);
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

        verticalLayout_4->addWidget(b_table);

        splitter->addWidget(widget_2);
        source_widget = new QWidget(splitter);
        source_widget->setObjectName(QString::fromUtf8("source_widget"));
        verticalLayout_6 = new QVBoxLayout(source_widget);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
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

        horizontalLayout_21->addWidget(splitter);

        toolBox->addItem(source_page, QString::fromUtf8("Source Images"));
        page_3 = new QWidget();
        page_3->setObjectName(QString::fromUtf8("page_3"));
        page_3->setGeometry(QRect(0, 0, 761, 622));
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
        page->setGeometry(QRect(0, 0, 761, 622));
        page->setMinimumSize(QSize(0, 0));
        verticalLayout_2 = new QVBoxLayout(page);
        verticalLayout_2->setSpacing(3);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(6, 6, 6, 6);
        groupBox = new QGroupBox(page);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_2 = new QHBoxLayout(groupBox);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 0, -1, 6);
        DTI = new QRadioButton(groupBox);
        DTI->setObjectName(QString::fromUtf8("DTI"));
        DTI->setChecked(false);

        horizontalLayout_2->addWidget(DTI);

        DSI = new QRadioButton(groupBox);
        DSI->setObjectName(QString::fromUtf8("DSI"));

        horizontalLayout_2->addWidget(DSI);

        RDSI = new QRadioButton(groupBox);
        RDSI->setObjectName(QString::fromUtf8("RDSI"));

        horizontalLayout_2->addWidget(RDSI);

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


        verticalLayout_2->addWidget(groupBox);

        DSIOption_2 = new QGroupBox(page);
        DSIOption_2->setObjectName(QString::fromUtf8("DSIOption_2"));
        DSIOption = new QHBoxLayout(DSIOption_2);
        DSIOption->setObjectName(QString::fromUtf8("DSIOption"));
        DSIOption->setContentsMargins(-1, 6, -1, 6);
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
        QBIOption->setContentsMargins(-1, 6, -1, 6);
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label = new QLabel(QBIOption_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_12->addWidget(label);

        SHOrder = new QSpinBox(QBIOption_2);
        SHOrder->setObjectName(QString::fromUtf8("SHOrder"));
        SHOrder->setMaximumSize(QSize(75, 16777215));
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

        DSIOption_3 = new QGroupBox(page);
        DSIOption_3->setObjectName(QString::fromUtf8("DSIOption_3"));
        DSIOption_3->setMinimumSize(QSize(0, 56));
        horizontalLayout_22 = new QHBoxLayout(DSIOption_3);
        horizontalLayout_22->setObjectName(QString::fromUtf8("horizontalLayout_22"));
        label_16 = new QLabel(DSIOption_3);
        label_16->setObjectName(QString::fromUtf8("label_16"));
        label_16->setEnabled(true);
        label_16->setMaximumSize(QSize(16777215, 16777215));

        horizontalLayout_22->addWidget(label_16);

        edge_factor = new QDoubleSpinBox(DSIOption_3);
        edge_factor->setObjectName(QString::fromUtf8("edge_factor"));
        edge_factor->setMaximumSize(QSize(75, 16777215));
        edge_factor->setInputMethodHints(Qt::ImhFormattedNumbersOnly);
        edge_factor->setDecimals(1);
        edge_factor->setMaximum(3);
        edge_factor->setSingleStep(0.1);
        edge_factor->setValue(1);

        horizontalLayout_22->addWidget(edge_factor);


        verticalLayout_2->addWidget(DSIOption_3);

        GQIOption_2 = new QGroupBox(page);
        GQIOption_2->setObjectName(QString::fromUtf8("GQIOption_2"));
        GQIOption = new QHBoxLayout(GQIOption_2);
        GQIOption->setObjectName(QString::fromUtf8("GQIOption"));
        GQIOption->setContentsMargins(-1, 6, -1, 6);
        label_7 = new QLabel(GQIOption_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        GQIOption->addWidget(label_7);

        diffusion_sampling = new QDoubleSpinBox(GQIOption_2);
        diffusion_sampling->setObjectName(QString::fromUtf8("diffusion_sampling"));
        diffusion_sampling->setMaximumSize(QSize(75, 16777215));
        diffusion_sampling->setDecimals(2);
        diffusion_sampling->setMinimum(0);
        diffusion_sampling->setMaximum(3);
        diffusion_sampling->setSingleStep(0.05);
        diffusion_sampling->setValue(1.25);

        GQIOption->addWidget(diffusion_sampling);

        ODFDef = new QComboBox(GQIOption_2);
        ODFDef->setObjectName(QString::fromUtf8("ODFDef"));
        ODFDef->setMaximumSize(QSize(150, 16777215));

        GQIOption->addWidget(ODFDef);


        verticalLayout_2->addWidget(GQIOption_2);

        ResolutionBox = new QGroupBox(page);
        ResolutionBox->setObjectName(QString::fromUtf8("ResolutionBox"));
        horizontalLayout_9 = new QHBoxLayout(ResolutionBox);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(-1, 6, -1, 6);
        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(0);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_15 = new QLabel(ResolutionBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_10->addWidget(label_15);

        reg_method = new QComboBox(ResolutionBox);
        reg_method->setObjectName(QString::fromUtf8("reg_method"));

        horizontalLayout_10->addWidget(reg_method);

        manual_reg = new QToolButton(ResolutionBox);
        manual_reg->setObjectName(QString::fromUtf8("manual_reg"));
        manual_reg->setMinimumSize(QSize(0, 23));

        horizontalLayout_10->addWidget(manual_reg);


        horizontalLayout_9->addLayout(horizontalLayout_10);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_8 = new QLabel(ResolutionBox);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_8->addWidget(label_8);

        mni_resolution = new QSpinBox(ResolutionBox);
        mni_resolution->setObjectName(QString::fromUtf8("mni_resolution"));
        mni_resolution->setMaximumSize(QSize(75, 16777215));
        mni_resolution->setMinimum(1);
        mni_resolution->setMaximum(3);
        mni_resolution->setValue(2);

        horizontalLayout_8->addWidget(mni_resolution);


        horizontalLayout_9->addLayout(horizontalLayout_8);


        verticalLayout_2->addWidget(ResolutionBox);

        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setObjectName(QString::fromUtf8("horizontalLayout_19"));
        AdvancedOptions = new QPushButton(page);
        AdvancedOptions->setObjectName(QString::fromUtf8("AdvancedOptions"));

        horizontalLayout_19->addWidget(AdvancedOptions);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_19->addItem(horizontalSpacer_6);


        verticalLayout_2->addLayout(horizontalLayout_19);

        AdvancedWidget = new QWidget(page);
        AdvancedWidget->setObjectName(QString::fromUtf8("AdvancedWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(AdvancedWidget->sizePolicy().hasHeightForWidth());
        AdvancedWidget->setSizePolicy(sizePolicy1);
        verticalLayout_7 = new QVBoxLayout(AdvancedWidget);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        ODFSharpening = new QGroupBox(AdvancedWidget);
        ODFSharpening->setObjectName(QString::fromUtf8("ODFSharpening"));
        horizontalLayout_17 = new QHBoxLayout(ODFSharpening);
        horizontalLayout_17->setSpacing(0);
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        horizontalLayout_17->setContentsMargins(-1, 0, -1, 6);
        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(0);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        odf_sharpening = new QComboBox(ODFSharpening);
        odf_sharpening->setObjectName(QString::fromUtf8("odf_sharpening"));
        odf_sharpening->setMaximumSize(QSize(16777215, 16777215));

        horizontalLayout_14->addWidget(odf_sharpening);

        decom_panel = new QWidget(ODFSharpening);
        decom_panel->setObjectName(QString::fromUtf8("decom_panel"));
        horizontalLayout_5 = new QHBoxLayout(decom_panel);
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        label_12 = new QLabel(decom_panel);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setMaximumSize(QSize(10, 16777215));

        horizontalLayout_5->addWidget(label_12);

        decom_fraction = new QDoubleSpinBox(decom_panel);
        decom_fraction->setObjectName(QString::fromUtf8("decom_fraction"));
        decom_fraction->setMaximumSize(QSize(75, 16777215));
        decom_fraction->setMinimum(0.01);
        decom_fraction->setMaximum(0.5);
        decom_fraction->setSingleStep(0.05);
        decom_fraction->setValue(0.05);

        horizontalLayout_5->addWidget(decom_fraction);

        label_11 = new QLabel(decom_panel);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setMaximumSize(QSize(10, 16777215));

        horizontalLayout_5->addWidget(label_11);

        decom_m = new QSpinBox(decom_panel);
        decom_m->setObjectName(QString::fromUtf8("decom_m"));
        decom_m->setMaximumSize(QSize(35, 16777215));
        decom_m->setMinimum(3);
        decom_m->setMaximum(20);
        decom_m->setValue(10);

        horizontalLayout_5->addWidget(decom_m);


        horizontalLayout_14->addWidget(decom_panel);

        decon_param = new QDoubleSpinBox(ODFSharpening);
        decon_param->setObjectName(QString::fromUtf8("decon_param"));
        decon_param->setMaximumSize(QSize(45, 16777215));
        decon_param->setMinimum(0.01);
        decon_param->setMaximum(20);
        decon_param->setSingleStep(0.5);
        decon_param->setValue(3);

        horizontalLayout_14->addWidget(decon_param);


        horizontalLayout_17->addLayout(horizontalLayout_14);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_17->addItem(horizontalSpacer_3);

        xyz = new QHBoxLayout();
        xyz->setSpacing(0);
        xyz->setObjectName(QString::fromUtf8("xyz"));
        xyz->setContentsMargins(-1, 0, -1, -1);
        xyz_widget = new QWidget(ODFSharpening);
        xyz_widget->setObjectName(QString::fromUtf8("xyz_widget"));
        horizontalLayout_4 = new QHBoxLayout(xyz_widget);
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        label_4 = new QLabel(xyz_widget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_4->addWidget(label_4);

        RFSelection = new QComboBox(xyz_widget);
        RFSelection->setObjectName(QString::fromUtf8("RFSelection"));
        RFSelection->setMaximumSize(QSize(60, 16777215));

        horizontalLayout_4->addWidget(RFSelection);


        xyz->addWidget(xyz_widget);


        horizontalLayout_17->addLayout(xyz);

        ODFSelection = new QWidget(ODFSharpening);
        ODFSelection->setObjectName(QString::fromUtf8("ODFSelection"));
        horizontalLayout_15 = new QHBoxLayout(ODFSelection);
        horizontalLayout_15->setSpacing(0);
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        horizontalLayout_15->setContentsMargins(7, 0, 0, 0);
        label_10 = new QLabel(ODFSelection);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setMinimumSize(QSize(20, 0));
        label_10->setMaximumSize(QSize(60, 16777215));

        horizontalLayout_15->addWidget(label_10);

        x = new QSpinBox(ODFSelection);
        x->setObjectName(QString::fromUtf8("x"));
        x->setMaximumSize(QSize(35, 16777215));

        horizontalLayout_15->addWidget(x);

        y = new QSpinBox(ODFSelection);
        y->setObjectName(QString::fromUtf8("y"));
        y->setMaximumSize(QSize(35, 16777215));

        horizontalLayout_15->addWidget(y);

        z = new QSpinBox(ODFSelection);
        z->setObjectName(QString::fromUtf8("z"));
        z->setMaximumSize(QSize(35, 16777215));

        horizontalLayout_15->addWidget(z);


        horizontalLayout_17->addWidget(ODFSelection);


        verticalLayout_7->addWidget(ODFSharpening);

        Output = new QGroupBox(AdvancedWidget);
        Output->setObjectName(QString::fromUtf8("Output"));
        horizontalLayout_16 = new QHBoxLayout(Output);
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        horizontalLayout_16->setContentsMargins(-1, 0, -1, 6);
        RecordODF = new QCheckBox(Output);
        RecordODF->setObjectName(QString::fromUtf8("RecordODF"));

        horizontalLayout_16->addWidget(RecordODF);

        output_mapping = new QCheckBox(Output);
        output_mapping->setObjectName(QString::fromUtf8("output_mapping"));

        horizontalLayout_16->addWidget(output_mapping);

        output_jacobian = new QCheckBox(Output);
        output_jacobian->setObjectName(QString::fromUtf8("output_jacobian"));

        horizontalLayout_16->addWidget(output_jacobian);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_16->addItem(horizontalSpacer_4);


        verticalLayout_7->addWidget(Output);

        groupBox_2 = new QGroupBox(AdvancedWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        horizontalLayout_18 = new QHBoxLayout(groupBox_2);
        horizontalLayout_18->setObjectName(QString::fromUtf8("horizontalLayout_18"));
        horizontalLayout_18->setContentsMargins(-1, 0, -1, 6);
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_6->addWidget(label_3);

        NumOfFibers = new QSpinBox(groupBox_2);
        NumOfFibers->setObjectName(QString::fromUtf8("NumOfFibers"));
        NumOfFibers->setMaximumSize(QSize(75, 16777215));
        NumOfFibers->setMinimum(3);
        NumOfFibers->setMaximum(20);
        NumOfFibers->setValue(5);

        horizontalLayout_6->addWidget(NumOfFibers);


        horizontalLayout_18->addLayout(horizontalLayout_6);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        ODFDim = new QComboBox(groupBox_2);
        ODFDim->setObjectName(QString::fromUtf8("ODFDim"));
        ODFDim->setMaximumSize(QSize(75, 16777215));

        horizontalLayout_3->addWidget(ODFDim);


        horizontalLayout_18->addLayout(horizontalLayout_3);

        balance_scheme = new QCheckBox(groupBox_2);
        balance_scheme->setObjectName(QString::fromUtf8("balance_scheme"));

        horizontalLayout_18->addWidget(balance_scheme);

        HalfSphere = new QCheckBox(groupBox_2);
        HalfSphere->setObjectName(QString::fromUtf8("HalfSphere"));
        HalfSphere->setChecked(false);

        horizontalLayout_18->addWidget(HalfSphere);


        verticalLayout_7->addWidget(groupBox_2);


        verticalLayout_2->addWidget(AdvancedWidget);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        widget = new QWidget(page);
        widget->setObjectName(QString::fromUtf8("widget"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy2);
        horizontalLayout_20 = new QHBoxLayout(widget);
        horizontalLayout_20->setSpacing(0);
        horizontalLayout_20->setObjectName(QString::fromUtf8("horizontalLayout_20"));
        horizontalLayout_20->setContentsMargins(0, 0, 0, 0);
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_20->addItem(horizontalSpacer);

        label_9 = new QLabel(widget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout_20->addWidget(label_9);

        ThreadCount = new QComboBox(widget);
        ThreadCount->setObjectName(QString::fromUtf8("ThreadCount"));
        ThreadCount->setMaximumSize(QSize(75, 16777215));

        horizontalLayout_20->addWidget(ThreadCount);

        doDTI = new QCommandLinkButton(widget);
        doDTI->setObjectName(QString::fromUtf8("doDTI"));
        sizePolicy2.setHeightForWidth(doDTI->sizePolicy().hasHeightForWidth());
        doDTI->setSizePolicy(sizePolicy2);
        doDTI->setMaximumSize(QSize(16777215, 40));

        horizontalLayout_20->addWidget(doDTI);


        verticalLayout_2->addWidget(widget);

        toolBox->addItem(page, QString::fromUtf8("Step 2: select reconstruction method"));

        verticalLayout->addWidget(toolBox);

        reconstruction_window->setCentralWidget(centralwidget);

        retranslateUi(reconstruction_window);

        toolBox->setCurrentIndex(2);
        toolBox->layout()->setSpacing(0);
        ODFDim->setCurrentIndex(3);
        ThreadCount->setCurrentIndex(3);


        QMetaObject::connectSlotsByName(reconstruction_window);
    } // setupUi

    void retranslateUi(QMainWindow *reconstruction_window)
    {
        reconstruction_window->setWindowTitle(QApplication::translate("reconstruction_window", "Reconstruction", 0, QApplication::UnicodeUTF8));
        save4dnifti->setText(QApplication::translate("reconstruction_window", "Save nifti...", 0, QApplication::UnicodeUTF8));
        save_b_table->setText(QApplication::translate("reconstruction_window", "Save b table...", 0, QApplication::UnicodeUTF8));
        save_bvals->setText(QApplication::translate("reconstruction_window", "Save bvals...", 0, QApplication::UnicodeUTF8));
        save_bvec->setText(QApplication::translate("reconstruction_window", "Save bvecs...", 0, QApplication::UnicodeUTF8));
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
        RDSI->setText(QApplication::translate("reconstruction_window", "RDSI", 0, QApplication::UnicodeUTF8));
        QBI->setText(QApplication::translate("reconstruction_window", "QBI", 0, QApplication::UnicodeUTF8));
        GQI->setText(QApplication::translate("reconstruction_window", "GQI", 0, QApplication::UnicodeUTF8));
        QDif->setText(QApplication::translate("reconstruction_window", "QSDR", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("reconstruction_window", "DSI Hamming filter", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("reconstruction_window", "SH order", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("reconstruction_window", "QBI Regularization", 0, QApplication::UnicodeUTF8));
        DSIOption_3->setTitle(QString());
        label_16->setText(QApplication::translate("reconstruction_window", "Radial DSI Edge Factor", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("reconstruction_window", "Diffusion sampling length ratio", 0, QApplication::UnicodeUTF8));
        ODFDef->clear();
        ODFDef->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "No distance weighting", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "r-squared weighted", 0, QApplication::UnicodeUTF8)
        );
        ResolutionBox->setTitle(QString());
        label_15->setText(QApplication::translate("reconstruction_window", "Registration method", 0, QApplication::UnicodeUTF8));
        reg_method->clear();
        reg_method->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "SPM norm 7-9-7", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "SPM norm 12-14-12", 0, QApplication::UnicodeUTF8)
        );
        manual_reg->setText(QApplication::translate("reconstruction_window", "...", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("reconstruction_window", "Output Resolution", 0, QApplication::UnicodeUTF8));
        AdvancedOptions->setText(QApplication::translate("reconstruction_window", "Advanced Options >>", 0, QApplication::UnicodeUTF8));
        ODFSharpening->setTitle(QApplication::translate("reconstruction_window", "ODF Sharpening", 0, QApplication::UnicodeUTF8));
        odf_sharpening->clear();
        odf_sharpening->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "Deconvolution", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "Decomposition", 0, QApplication::UnicodeUTF8)
        );
        label_12->setText(QApplication::translate("reconstruction_window", "\317\265", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("reconstruction_window", "m", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("reconstruction_window", "Response Function Selection", 0, QApplication::UnicodeUTF8));
        RFSelection->clear();
        RFSelection->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "Auto", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "Manual", 0, QApplication::UnicodeUTF8)
        );
        label_10->setText(QApplication::translate("reconstruction_window", "at (x,y,z)=", 0, QApplication::UnicodeUTF8));
        Output->setTitle(QApplication::translate("reconstruction_window", "Output", 0, QApplication::UnicodeUTF8));
        RecordODF->setText(QApplication::translate("reconstruction_window", "Complete ODFs", 0, QApplication::UnicodeUTF8));
        output_mapping->setText(QApplication::translate("reconstruction_window", "Spatial mapping", 0, QApplication::UnicodeUTF8));
        output_jacobian->setText(QApplication::translate("reconstruction_window", "Jacobian determinant", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("reconstruction_window", "ODF Processing", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("reconstruction_window", "Number of fibers resolved", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("reconstruction_window", "ODF Tessellation", 0, QApplication::UnicodeUTF8));
        ODFDim->clear();
        ODFDim->insertItems(0, QStringList()
         << QApplication::translate("reconstruction_window", "4-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "5-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "6-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "8-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "10-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "12-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "16-fold", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("reconstruction_window", "20-fold", 0, QApplication::UnicodeUTF8)
        );
        balance_scheme->setText(QApplication::translate("reconstruction_window", "Balance scheme", 0, QApplication::UnicodeUTF8));
        HalfSphere->setText(QApplication::translate("reconstruction_window", "Half-sphere scheme", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("reconstruction_window", "Parallel reconstruction:", 0, QApplication::UnicodeUTF8));
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
