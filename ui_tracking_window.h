/********************************************************************************
** Form generated from reading UI file 'tracking_window.ui'
**
** Created: Tue Apr 24 10:53:07 2012
**      by: Qt User Interface Compiler version 4.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRACKING_WINDOW_H
#define UI_TRACKING_WINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDockWidget>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGraphicsView>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_tracking_window
{
public:
    QAction *actionNewRegion;
    QAction *actionOpenRegion;
    QAction *actionSaveRegionAs;
    QAction *actionDeleteRegion;
    QAction *actionOpenTract;
    QAction *actionSaveTractAs;
    QAction *actionSaveTractFA;
    QAction *actionDeleteRegionAll;
    QAction *actionDeleteTract;
    QAction *actionDeleteTractAll;
    QAction *actionShift_X;
    QAction *actionShift_X_2;
    QAction *actionShift_Y;
    QAction *actionShift_Y_2;
    QAction *actionShift_Z;
    QAction *actionShift_Z_2;
    QAction *actionFlip_X;
    QAction *actionFlip_Y;
    QAction *actionFlip_Z;
    QAction *actionDilation;
    QAction *actionErosion;
    QAction *actionSmoothing;
    QAction *actionNegate;
    QAction *actionDefragment;
    QAction *actionWhole_brain_seeding;
    QAction *actionSelect_Tracts;
    QAction *actionUndo;
    QAction *actionDelete;
    QAction *actionTrim;
    QAction *actionCut;
    QAction *actionZoom_In;
    QAction *actionZoom_Out;
    QAction *actionSagittal_view;
    QAction *actionCoronal_View;
    QAction *actionAxial_View;
    QAction *actionQuantitative_anisotropy_QA;
    QAction *actionMerge_All;
    QAction *actionSave_Screen;
    QAction *actionSave_ROI_Screen;
    QAction *actionLoad_Camera;
    QAction *actionSave_Camera;
    QAction *actionEndpoints_to_seeding;
    QAction *actionTracts_to_seeds;
    QAction *actionSave_Rotation_Images;
    QAction *actionPaint;
    QAction *actionInsert_T1_T2;
    QAction *actionAdd_surface;
    QAction *actionSave_mapping;
    QAction *actionLoad_mapping;
    QAction *actionStatistics;
    QAction *actionK_means;
    QAction *actionEM;
    QAction *actionHierarchical;
    QAction *actionSet_Color;
    QAction *actionSave_Tracts_Colors_As;
    QAction *actionOpen_Colors;
    QAction *actionSave_Voxel_Data_As;
    QAction *actionTDI_Diffusion_Space;
    QAction *actionTDI_Import_Slice_Space;
    QAction *actionTDI_Subvoxel_Diffusion_Space;
    QAction *actionSave_Tracts_in_Current_Mapping;
    QAction *actionThreshold;
    QAction *actionRedo;
    QAction *actionCopy_to_clipboard;
    QAction *actionSave_Anisotrpy_Map_as;
    QWidget *centralwidget;
    QVBoxLayout *centralLayout;
    QHBoxLayout *horizontalLayout_13;
    QLabel *label_3;
    QComboBox *SliceModality;
    QToolButton *addSlices;
    QToolButton *isosurfaceButton;
    QComboBox *surfaceStyle;
    QSpacerItem *horizontalSpacer;
    QHBoxLayout *centralLayout2;
    QCheckBox *glSagCheck;
    QToolButton *glSagView;
    QSpinBox *glSagBox;
    QSlider *glSagSlider;
    QCheckBox *glCorCheck;
    QToolButton *glCorView;
    QSpinBox *glCorBox;
    QSlider *glCorSlider;
    QCheckBox *glAxiCheck;
    QToolButton *glAxiView;
    QSpinBox *glAxiBox;
    QSlider *glAxiSlider;
    QStatusBar *statusbar;
    QDockWidget *regionDockWidget;
    QWidget *dockWidgetContents_4;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout_4;
    QComboBox *atlasComboBox;
    QToolButton *addRegionFromAtlas;
    QDockWidget *renderingWidgetHolder;
    QWidget *dockWidgetContents_2;
    QVBoxLayout *renderingLayout;
    QHBoxLayout *horizontalLayout_8;
    QHBoxLayout *horizontalLayout_6;
    QToolButton *tbDefaultParam;
    QComboBox *RenderingQualityBox;
    QSpacerItem *horizontalSpacer_2;
    QMenuBar *menuBar;
    QMenu *menuRegions;
    QMenu *menuModify;
    QMenu *menuTracts;
    QMenu *menuSave;
    QMenu *menuClustering;
    QMenu *menuExport_Tract_Density;
    QMenu *menu_Edit;
    QMenu *menu_View;
    QMenu *menu_Slices;
    QDockWidget *TractWidgetHolder;
    QWidget *dockWidgetContents_5;
    QVBoxLayout *tractverticalLayout;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_7;
    QFrame *line;
    QToolButton *zoomIn;
    QToolButton *zoomOut;
    QFrame *line_4;
    QToolButton *tbNewRegion;
    QToolButton *tbOpenRegion;
    QToolButton *tbSaveRegion;
    QToolButton *tbDeleteRegion;
    QFrame *line_3;
    QToolButton *tool0;
    QToolButton *tool1;
    QToolButton *tool4;
    QToolButton *tool2;
    QToolButton *tool3;
    QFrame *line_2;
    QToolButton *whole_brain;
    QComboBox *sliceViewBox;
    QGraphicsView *graphicsView;
    QHBoxLayout *horizontalLayout_2;
    QToolButton *SagView;
    QSlider *SagSlider;
    QToolButton *CorView;
    QSlider *CorSlider;
    QToolButton *AxiView;
    QSlider *AxiSlider;
    QDockWidget *dockWidget_3;
    QWidget *dockWidgetContents_3;
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label;
    QDoubleSpinBox *fa_threshold;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_2;
    QSpinBox *turning_angle;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_5;
    QDoubleSpinBox *step_size;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_4;
    QDoubleSpinBox *smoothing;
    QHBoxLayout *horizontalLayout;
    QLabel *label_7;
    QDoubleSpinBox *min_length;
    QDoubleSpinBox *max_length;
    QHBoxLayout *horizontalLayout_3;
    QComboBox *initial_direction;
    QComboBox *seed_plan;
    QComboBox *interpolation;
    QComboBox *tracking_method;
    QHBoxLayout *horizontalLayout_5;
    QSpinBox *track_count;
    QComboBox *tracking_plan;
    QComboBox *thread_count;
    QPushButton *perform_tracking;
    QToolButton *stopTracking;

    void setupUi(QMainWindow *tracking_window)
    {
        if (tracking_window->objectName().isEmpty())
            tracking_window->setObjectName(QString::fromUtf8("tracking_window"));
        tracking_window->resize(1124, 660);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(tracking_window->sizePolicy().hasHeightForWidth());
        tracking_window->setSizePolicy(sizePolicy);
        QFont font;
        font.setFamily(QString::fromUtf8("Arial"));
        tracking_window->setFont(font);
        tracking_window->setMouseTracking(false);
        actionNewRegion = new QAction(tracking_window);
        actionNewRegion->setObjectName(QString::fromUtf8("actionNewRegion"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/new.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionNewRegion->setIcon(icon);
        actionOpenRegion = new QAction(tracking_window);
        actionOpenRegion->setObjectName(QString::fromUtf8("actionOpenRegion"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/open.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenRegion->setIcon(icon1);
        actionSaveRegionAs = new QAction(tracking_window);
        actionSaveRegionAs->setObjectName(QString::fromUtf8("actionSaveRegionAs"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/save.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionSaveRegionAs->setIcon(icon2);
        actionDeleteRegion = new QAction(tracking_window);
        actionDeleteRegion->setObjectName(QString::fromUtf8("actionDeleteRegion"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/icons/delete.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionDeleteRegion->setIcon(icon3);
        actionDeleteRegion->setFont(font);
        actionOpenTract = new QAction(tracking_window);
        actionOpenTract->setObjectName(QString::fromUtf8("actionOpenTract"));
        actionOpenTract->setIcon(icon1);
        actionOpenTract->setFont(font);
        actionSaveTractAs = new QAction(tracking_window);
        actionSaveTractAs->setObjectName(QString::fromUtf8("actionSaveTractAs"));
        actionSaveTractAs->setIcon(icon2);
        actionSaveTractAs->setFont(font);
        actionSaveTractFA = new QAction(tracking_window);
        actionSaveTractFA->setObjectName(QString::fromUtf8("actionSaveTractFA"));
        actionSaveTractFA->setIcon(icon2);
        actionSaveTractFA->setFont(font);
        actionDeleteRegionAll = new QAction(tracking_window);
        actionDeleteRegionAll->setObjectName(QString::fromUtf8("actionDeleteRegionAll"));
        actionDeleteRegionAll->setIcon(icon3);
        actionDeleteRegionAll->setFont(font);
        actionDeleteTract = new QAction(tracking_window);
        actionDeleteTract->setObjectName(QString::fromUtf8("actionDeleteTract"));
        actionDeleteTract->setIcon(icon3);
        actionDeleteTract->setFont(font);
        actionDeleteTractAll = new QAction(tracking_window);
        actionDeleteTractAll->setObjectName(QString::fromUtf8("actionDeleteTractAll"));
        actionDeleteTractAll->setIcon(icon3);
        actionDeleteTractAll->setFont(font);
        actionShift_X = new QAction(tracking_window);
        actionShift_X->setObjectName(QString::fromUtf8("actionShift_X"));
        actionShift_X_2 = new QAction(tracking_window);
        actionShift_X_2->setObjectName(QString::fromUtf8("actionShift_X_2"));
        actionShift_Y = new QAction(tracking_window);
        actionShift_Y->setObjectName(QString::fromUtf8("actionShift_Y"));
        actionShift_Y_2 = new QAction(tracking_window);
        actionShift_Y_2->setObjectName(QString::fromUtf8("actionShift_Y_2"));
        actionShift_Z = new QAction(tracking_window);
        actionShift_Z->setObjectName(QString::fromUtf8("actionShift_Z"));
        actionShift_Z_2 = new QAction(tracking_window);
        actionShift_Z_2->setObjectName(QString::fromUtf8("actionShift_Z_2"));
        actionFlip_X = new QAction(tracking_window);
        actionFlip_X->setObjectName(QString::fromUtf8("actionFlip_X"));
        actionFlip_Y = new QAction(tracking_window);
        actionFlip_Y->setObjectName(QString::fromUtf8("actionFlip_Y"));
        actionFlip_Z = new QAction(tracking_window);
        actionFlip_Z->setObjectName(QString::fromUtf8("actionFlip_Z"));
        actionDilation = new QAction(tracking_window);
        actionDilation->setObjectName(QString::fromUtf8("actionDilation"));
        actionErosion = new QAction(tracking_window);
        actionErosion->setObjectName(QString::fromUtf8("actionErosion"));
        actionSmoothing = new QAction(tracking_window);
        actionSmoothing->setObjectName(QString::fromUtf8("actionSmoothing"));
        actionNegate = new QAction(tracking_window);
        actionNegate->setObjectName(QString::fromUtf8("actionNegate"));
        actionDefragment = new QAction(tracking_window);
        actionDefragment->setObjectName(QString::fromUtf8("actionDefragment"));
        actionWhole_brain_seeding = new QAction(tracking_window);
        actionWhole_brain_seeding->setObjectName(QString::fromUtf8("actionWhole_brain_seeding"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/icons/icons/axial.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionWhole_brain_seeding->setIcon(icon4);
        actionSelect_Tracts = new QAction(tracking_window);
        actionSelect_Tracts->setObjectName(QString::fromUtf8("actionSelect_Tracts"));
        actionUndo = new QAction(tracking_window);
        actionUndo->setObjectName(QString::fromUtf8("actionUndo"));
        actionDelete = new QAction(tracking_window);
        actionDelete->setObjectName(QString::fromUtf8("actionDelete"));
        actionTrim = new QAction(tracking_window);
        actionTrim->setObjectName(QString::fromUtf8("actionTrim"));
        actionCut = new QAction(tracking_window);
        actionCut->setObjectName(QString::fromUtf8("actionCut"));
        actionZoom_In = new QAction(tracking_window);
        actionZoom_In->setObjectName(QString::fromUtf8("actionZoom_In"));
        actionZoom_Out = new QAction(tracking_window);
        actionZoom_Out->setObjectName(QString::fromUtf8("actionZoom_Out"));
        actionSagittal_view = new QAction(tracking_window);
        actionSagittal_view->setObjectName(QString::fromUtf8("actionSagittal_view"));
        actionCoronal_View = new QAction(tracking_window);
        actionCoronal_View->setObjectName(QString::fromUtf8("actionCoronal_View"));
        actionAxial_View = new QAction(tracking_window);
        actionAxial_View->setObjectName(QString::fromUtf8("actionAxial_View"));
        actionQuantitative_anisotropy_QA = new QAction(tracking_window);
        actionQuantitative_anisotropy_QA->setObjectName(QString::fromUtf8("actionQuantitative_anisotropy_QA"));
        actionMerge_All = new QAction(tracking_window);
        actionMerge_All->setObjectName(QString::fromUtf8("actionMerge_All"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/icons/icons/add.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionMerge_All->setIcon(icon5);
        actionSave_Screen = new QAction(tracking_window);
        actionSave_Screen->setObjectName(QString::fromUtf8("actionSave_Screen"));
        actionSave_Screen->setIcon(icon2);
        actionSave_ROI_Screen = new QAction(tracking_window);
        actionSave_ROI_Screen->setObjectName(QString::fromUtf8("actionSave_ROI_Screen"));
        actionSave_ROI_Screen->setIcon(icon2);
        actionLoad_Camera = new QAction(tracking_window);
        actionLoad_Camera->setObjectName(QString::fromUtf8("actionLoad_Camera"));
        actionLoad_Camera->setIcon(icon1);
        actionSave_Camera = new QAction(tracking_window);
        actionSave_Camera->setObjectName(QString::fromUtf8("actionSave_Camera"));
        actionSave_Camera->setIcon(icon2);
        actionEndpoints_to_seeding = new QAction(tracking_window);
        actionEndpoints_to_seeding->setObjectName(QString::fromUtf8("actionEndpoints_to_seeding"));
        actionTracts_to_seeds = new QAction(tracking_window);
        actionTracts_to_seeds->setObjectName(QString::fromUtf8("actionTracts_to_seeds"));
        actionSave_Rotation_Images = new QAction(tracking_window);
        actionSave_Rotation_Images->setObjectName(QString::fromUtf8("actionSave_Rotation_Images"));
        actionSave_Rotation_Images->setIcon(icon2);
        actionPaint = new QAction(tracking_window);
        actionPaint->setObjectName(QString::fromUtf8("actionPaint"));
        actionInsert_T1_T2 = new QAction(tracking_window);
        actionInsert_T1_T2->setObjectName(QString::fromUtf8("actionInsert_T1_T2"));
        actionInsert_T1_T2->setIcon(icon1);
        actionAdd_surface = new QAction(tracking_window);
        actionAdd_surface->setObjectName(QString::fromUtf8("actionAdd_surface"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/icons/icons/coronal.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        actionAdd_surface->setIcon(icon6);
        actionSave_mapping = new QAction(tracking_window);
        actionSave_mapping->setObjectName(QString::fromUtf8("actionSave_mapping"));
        actionSave_mapping->setIcon(icon2);
        actionLoad_mapping = new QAction(tracking_window);
        actionLoad_mapping->setObjectName(QString::fromUtf8("actionLoad_mapping"));
        actionLoad_mapping->setIcon(icon1);
        actionStatistics = new QAction(tracking_window);
        actionStatistics->setObjectName(QString::fromUtf8("actionStatistics"));
        actionK_means = new QAction(tracking_window);
        actionK_means->setObjectName(QString::fromUtf8("actionK_means"));
        actionEM = new QAction(tracking_window);
        actionEM->setObjectName(QString::fromUtf8("actionEM"));
        actionHierarchical = new QAction(tracking_window);
        actionHierarchical->setObjectName(QString::fromUtf8("actionHierarchical"));
        actionSet_Color = new QAction(tracking_window);
        actionSet_Color->setObjectName(QString::fromUtf8("actionSet_Color"));
        actionSave_Tracts_Colors_As = new QAction(tracking_window);
        actionSave_Tracts_Colors_As->setObjectName(QString::fromUtf8("actionSave_Tracts_Colors_As"));
        actionSave_Tracts_Colors_As->setIcon(icon2);
        actionOpen_Colors = new QAction(tracking_window);
        actionOpen_Colors->setObjectName(QString::fromUtf8("actionOpen_Colors"));
        actionOpen_Colors->setIcon(icon1);
        actionSave_Voxel_Data_As = new QAction(tracking_window);
        actionSave_Voxel_Data_As->setObjectName(QString::fromUtf8("actionSave_Voxel_Data_As"));
        actionSave_Voxel_Data_As->setIcon(icon2);
        actionTDI_Diffusion_Space = new QAction(tracking_window);
        actionTDI_Diffusion_Space->setObjectName(QString::fromUtf8("actionTDI_Diffusion_Space"));
        actionTDI_Diffusion_Space->setIcon(icon2);
        actionTDI_Import_Slice_Space = new QAction(tracking_window);
        actionTDI_Import_Slice_Space->setObjectName(QString::fromUtf8("actionTDI_Import_Slice_Space"));
        actionTDI_Import_Slice_Space->setIcon(icon2);
        actionTDI_Subvoxel_Diffusion_Space = new QAction(tracking_window);
        actionTDI_Subvoxel_Diffusion_Space->setObjectName(QString::fromUtf8("actionTDI_Subvoxel_Diffusion_Space"));
        actionTDI_Subvoxel_Diffusion_Space->setIcon(icon2);
        actionSave_Tracts_in_Current_Mapping = new QAction(tracking_window);
        actionSave_Tracts_in_Current_Mapping->setObjectName(QString::fromUtf8("actionSave_Tracts_in_Current_Mapping"));
        actionSave_Tracts_in_Current_Mapping->setIcon(icon2);
        actionThreshold = new QAction(tracking_window);
        actionThreshold->setObjectName(QString::fromUtf8("actionThreshold"));
        actionRedo = new QAction(tracking_window);
        actionRedo->setObjectName(QString::fromUtf8("actionRedo"));
        actionCopy_to_clipboard = new QAction(tracking_window);
        actionCopy_to_clipboard->setObjectName(QString::fromUtf8("actionCopy_to_clipboard"));
        actionSave_Anisotrpy_Map_as = new QAction(tracking_window);
        actionSave_Anisotrpy_Map_as->setObjectName(QString::fromUtf8("actionSave_Anisotrpy_Map_as"));
        actionSave_Anisotrpy_Map_as->setIcon(icon2);
        centralwidget = new QWidget(tracking_window);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        centralLayout = new QVBoxLayout(centralwidget);
        centralLayout->setSpacing(0);
        centralLayout->setContentsMargins(0, 0, 0, 0);
        centralLayout->setObjectName(QString::fromUtf8("centralLayout"));
        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setSpacing(0);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        label_3 = new QLabel(centralwidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_13->addWidget(label_3);

        SliceModality = new QComboBox(centralwidget);
        SliceModality->setObjectName(QString::fromUtf8("SliceModality"));
        SliceModality->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_13->addWidget(SliceModality);

        addSlices = new QToolButton(centralwidget);
        addSlices->setObjectName(QString::fromUtf8("addSlices"));
        addSlices->setMaximumSize(QSize(16777215, 22));
        addSlices->setIcon(icon1);
        addSlices->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_13->addWidget(addSlices);

        isosurfaceButton = new QToolButton(centralwidget);
        isosurfaceButton->setObjectName(QString::fromUtf8("isosurfaceButton"));

        horizontalLayout_13->addWidget(isosurfaceButton);

        surfaceStyle = new QComboBox(centralwidget);
        surfaceStyle->setObjectName(QString::fromUtf8("surfaceStyle"));
        surfaceStyle->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_13->addWidget(surfaceStyle);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer);


        centralLayout->addLayout(horizontalLayout_13);

        centralLayout2 = new QHBoxLayout();
        centralLayout2->setSpacing(0);
        centralLayout2->setObjectName(QString::fromUtf8("centralLayout2"));
        glSagCheck = new QCheckBox(centralwidget);
        glSagCheck->setObjectName(QString::fromUtf8("glSagCheck"));
        glSagCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glSagCheck);

        glSagView = new QToolButton(centralwidget);
        glSagView->setObjectName(QString::fromUtf8("glSagView"));
        glSagView->setMaximumSize(QSize(22, 22));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/icons/sag.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        glSagView->setIcon(icon7);

        centralLayout2->addWidget(glSagView);

        glSagBox = new QSpinBox(centralwidget);
        glSagBox->setObjectName(QString::fromUtf8("glSagBox"));

        centralLayout2->addWidget(glSagBox);

        glSagSlider = new QSlider(centralwidget);
        glSagSlider->setObjectName(QString::fromUtf8("glSagSlider"));
        glSagSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glSagSlider);

        glCorCheck = new QCheckBox(centralwidget);
        glCorCheck->setObjectName(QString::fromUtf8("glCorCheck"));
        glCorCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glCorCheck);

        glCorView = new QToolButton(centralwidget);
        glCorView->setObjectName(QString::fromUtf8("glCorView"));
        glCorView->setMaximumSize(QSize(22, 22));
        glCorView->setIcon(icon6);

        centralLayout2->addWidget(glCorView);

        glCorBox = new QSpinBox(centralwidget);
        glCorBox->setObjectName(QString::fromUtf8("glCorBox"));

        centralLayout2->addWidget(glCorBox);

        glCorSlider = new QSlider(centralwidget);
        glCorSlider->setObjectName(QString::fromUtf8("glCorSlider"));
        glCorSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glCorSlider);

        glAxiCheck = new QCheckBox(centralwidget);
        glAxiCheck->setObjectName(QString::fromUtf8("glAxiCheck"));
        glAxiCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glAxiCheck);

        glAxiView = new QToolButton(centralwidget);
        glAxiView->setObjectName(QString::fromUtf8("glAxiView"));
        glAxiView->setMinimumSize(QSize(0, 0));
        glAxiView->setMaximumSize(QSize(22, 22));
        glAxiView->setIcon(icon4);

        centralLayout2->addWidget(glAxiView);

        glAxiBox = new QSpinBox(centralwidget);
        glAxiBox->setObjectName(QString::fromUtf8("glAxiBox"));

        centralLayout2->addWidget(glAxiBox);

        glAxiSlider = new QSlider(centralwidget);
        glAxiSlider->setObjectName(QString::fromUtf8("glAxiSlider"));
        glAxiSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glAxiSlider);


        centralLayout->addLayout(centralLayout2);

        tracking_window->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(tracking_window);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        tracking_window->setStatusBar(statusbar);
        regionDockWidget = new QDockWidget(tracking_window);
        regionDockWidget->setObjectName(QString::fromUtf8("regionDockWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(1);
        sizePolicy1.setHeightForWidth(regionDockWidget->sizePolicy().hasHeightForWidth());
        regionDockWidget->setSizePolicy(sizePolicy1);
        regionDockWidget->setMinimumSize(QSize(118, 47));
        regionDockWidget->setMaximumSize(QSize(600, 524287));
        QFont font1;
        font1.setBold(false);
        font1.setWeight(50);
        regionDockWidget->setFont(font1);
        dockWidgetContents_4 = new QWidget();
        dockWidgetContents_4->setObjectName(QString::fromUtf8("dockWidgetContents_4"));
        verticalLayout_3 = new QVBoxLayout(dockWidgetContents_4);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        atlasComboBox = new QComboBox(dockWidgetContents_4);
        atlasComboBox->setObjectName(QString::fromUtf8("atlasComboBox"));
        atlasComboBox->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_4->addWidget(atlasComboBox);

        addRegionFromAtlas = new QToolButton(dockWidgetContents_4);
        addRegionFromAtlas->setObjectName(QString::fromUtf8("addRegionFromAtlas"));
        addRegionFromAtlas->setMaximumSize(QSize(16777215, 22));
        addRegionFromAtlas->setIcon(icon5);
        addRegionFromAtlas->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_4->addWidget(addRegionFromAtlas);


        verticalLayout_3->addLayout(horizontalLayout_4);

        regionDockWidget->setWidget(dockWidgetContents_4);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(1), regionDockWidget);
        renderingWidgetHolder = new QDockWidget(tracking_window);
        renderingWidgetHolder->setObjectName(QString::fromUtf8("renderingWidgetHolder"));
        sizePolicy.setHeightForWidth(renderingWidgetHolder->sizePolicy().hasHeightForWidth());
        renderingWidgetHolder->setSizePolicy(sizePolicy);
        renderingWidgetHolder->setMinimumSize(QSize(244, 48));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QString::fromUtf8("dockWidgetContents_2"));
        renderingLayout = new QVBoxLayout(dockWidgetContents_2);
        renderingLayout->setSpacing(0);
        renderingLayout->setContentsMargins(0, 0, 0, 0);
        renderingLayout->setObjectName(QString::fromUtf8("renderingLayout"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        tbDefaultParam = new QToolButton(dockWidgetContents_2);
        tbDefaultParam->setObjectName(QString::fromUtf8("tbDefaultParam"));

        horizontalLayout_6->addWidget(tbDefaultParam);

        RenderingQualityBox = new QComboBox(dockWidgetContents_2);
        RenderingQualityBox->setObjectName(QString::fromUtf8("RenderingQualityBox"));
        RenderingQualityBox->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_6->addWidget(RenderingQualityBox);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_2);


        horizontalLayout_8->addLayout(horizontalLayout_6);


        renderingLayout->addLayout(horizontalLayout_8);

        renderingWidgetHolder->setWidget(dockWidgetContents_2);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(2), renderingWidgetHolder);
        menuBar = new QMenuBar(tracking_window);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1124, 22));
        menuBar->setFont(font);
        menuBar->setDefaultUp(false);
        menuBar->setNativeMenuBar(false);
        menuRegions = new QMenu(menuBar);
        menuRegions->setObjectName(QString::fromUtf8("menuRegions"));
        menuModify = new QMenu(menuRegions);
        menuModify->setObjectName(QString::fromUtf8("menuModify"));
        menuTracts = new QMenu(menuBar);
        menuTracts->setObjectName(QString::fromUtf8("menuTracts"));
        menuSave = new QMenu(menuTracts);
        menuSave->setObjectName(QString::fromUtf8("menuSave"));
        menuClustering = new QMenu(menuTracts);
        menuClustering->setObjectName(QString::fromUtf8("menuClustering"));
        menuExport_Tract_Density = new QMenu(menuTracts);
        menuExport_Tract_Density->setObjectName(QString::fromUtf8("menuExport_Tract_Density"));
        menu_Edit = new QMenu(menuBar);
        menu_Edit->setObjectName(QString::fromUtf8("menu_Edit"));
        menu_View = new QMenu(menuBar);
        menu_View->setObjectName(QString::fromUtf8("menu_View"));
        menu_Slices = new QMenu(menuBar);
        menu_Slices->setObjectName(QString::fromUtf8("menu_Slices"));
        tracking_window->setMenuBar(menuBar);
        TractWidgetHolder = new QDockWidget(tracking_window);
        TractWidgetHolder->setObjectName(QString::fromUtf8("TractWidgetHolder"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(TractWidgetHolder->sizePolicy().hasHeightForWidth());
        TractWidgetHolder->setSizePolicy(sizePolicy2);
        TractWidgetHolder->setMinimumSize(QSize(81, 39));
        TractWidgetHolder->setMaximumSize(QSize(524287, 524287));
        dockWidgetContents_5 = new QWidget();
        dockWidgetContents_5->setObjectName(QString::fromUtf8("dockWidgetContents_5"));
        tractverticalLayout = new QVBoxLayout(dockWidgetContents_5);
        tractverticalLayout->setSpacing(0);
        tractverticalLayout->setContentsMargins(0, 0, 0, 0);
        tractverticalLayout->setObjectName(QString::fromUtf8("tractverticalLayout"));
        TractWidgetHolder->setWidget(dockWidgetContents_5);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(2), TractWidgetHolder);
        dockWidget = new QDockWidget(tracking_window);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(1);
        sizePolicy3.setHeightForWidth(dockWidget->sizePolicy().hasHeightForWidth());
        dockWidget->setSizePolicy(sizePolicy3);
        dockWidget->setMinimumSize(QSize(350, 133));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout = new QVBoxLayout(dockWidgetContents);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        line = new QFrame(dockWidgetContents);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout_7->addWidget(line);

        zoomIn = new QToolButton(dockWidgetContents);
        zoomIn->setObjectName(QString::fromUtf8("zoomIn"));
        zoomIn->setMinimumSize(QSize(22, 0));
        zoomIn->setMaximumSize(QSize(23, 22));

        horizontalLayout_7->addWidget(zoomIn);

        zoomOut = new QToolButton(dockWidgetContents);
        zoomOut->setObjectName(QString::fromUtf8("zoomOut"));
        zoomOut->setMinimumSize(QSize(22, 0));
        zoomOut->setMaximumSize(QSize(23, 22));

        horizontalLayout_7->addWidget(zoomOut);

        line_4 = new QFrame(dockWidgetContents);
        line_4->setObjectName(QString::fromUtf8("line_4"));
        line_4->setFrameShape(QFrame::VLine);
        line_4->setFrameShadow(QFrame::Sunken);

        horizontalLayout_7->addWidget(line_4);

        tbNewRegion = new QToolButton(dockWidgetContents);
        tbNewRegion->setObjectName(QString::fromUtf8("tbNewRegion"));
        tbNewRegion->setMaximumSize(QSize(23, 22));
        tbNewRegion->setIcon(icon);

        horizontalLayout_7->addWidget(tbNewRegion);

        tbOpenRegion = new QToolButton(dockWidgetContents);
        tbOpenRegion->setObjectName(QString::fromUtf8("tbOpenRegion"));
        tbOpenRegion->setMaximumSize(QSize(23, 22));
        tbOpenRegion->setIcon(icon1);

        horizontalLayout_7->addWidget(tbOpenRegion);

        tbSaveRegion = new QToolButton(dockWidgetContents);
        tbSaveRegion->setObjectName(QString::fromUtf8("tbSaveRegion"));
        tbSaveRegion->setMaximumSize(QSize(23, 22));
        tbSaveRegion->setIcon(icon2);

        horizontalLayout_7->addWidget(tbSaveRegion);

        tbDeleteRegion = new QToolButton(dockWidgetContents);
        tbDeleteRegion->setObjectName(QString::fromUtf8("tbDeleteRegion"));
        tbDeleteRegion->setMaximumSize(QSize(23, 22));
        tbDeleteRegion->setIcon(icon3);

        horizontalLayout_7->addWidget(tbDeleteRegion);

        line_3 = new QFrame(dockWidgetContents);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setFrameShape(QFrame::VLine);
        line_3->setFrameShadow(QFrame::Sunken);

        horizontalLayout_7->addWidget(line_3);

        tool0 = new QToolButton(dockWidgetContents);
        tool0->setObjectName(QString::fromUtf8("tool0"));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/icons/icons/rec.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool0->setIcon(icon8);
        tool0->setIconSize(QSize(16, 16));
        tool0->setCheckable(true);
        tool0->setChecked(true);
        tool0->setAutoExclusive(true);

        horizontalLayout_7->addWidget(tool0);

        tool1 = new QToolButton(dockWidgetContents);
        tool1->setObjectName(QString::fromUtf8("tool1"));
        tool1->setMaximumSize(QSize(23, 22));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/icons/icons/curves.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool1->setIcon(icon9);
        tool1->setIconSize(QSize(16, 16));
        tool1->setCheckable(true);
        tool1->setChecked(false);
        tool1->setAutoExclusive(true);

        horizontalLayout_7->addWidget(tool1);

        tool4 = new QToolButton(dockWidgetContents);
        tool4->setObjectName(QString::fromUtf8("tool4"));
        tool4->setMaximumSize(QSize(23, 22));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/icons/poly.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool4->setIcon(icon10);
        tool4->setCheckable(true);
        tool4->setAutoExclusive(true);

        horizontalLayout_7->addWidget(tool4);

        tool2 = new QToolButton(dockWidgetContents);
        tool2->setObjectName(QString::fromUtf8("tool2"));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/icons/icons/ball.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool2->setIcon(icon11);
        tool2->setIconSize(QSize(16, 16));
        tool2->setCheckable(true);
        tool2->setAutoExclusive(true);

        horizontalLayout_7->addWidget(tool2);

        tool3 = new QToolButton(dockWidgetContents);
        tool3->setObjectName(QString::fromUtf8("tool3"));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/icons/icons/cubic.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool3->setIcon(icon12);
        tool3->setIconSize(QSize(16, 16));
        tool3->setCheckable(true);
        tool3->setAutoExclusive(true);

        horizontalLayout_7->addWidget(tool3);

        line_2 = new QFrame(dockWidgetContents);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout_7->addWidget(line_2);

        whole_brain = new QToolButton(dockWidgetContents);
        whole_brain->setObjectName(QString::fromUtf8("whole_brain"));
        QSizePolicy sizePolicy4(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(whole_brain->sizePolicy().hasHeightForWidth());
        whole_brain->setSizePolicy(sizePolicy4);
        whole_brain->setMaximumSize(QSize(23, 22));
        whole_brain->setIcon(icon4);

        horizontalLayout_7->addWidget(whole_brain);

        sliceViewBox = new QComboBox(dockWidgetContents);
        sliceViewBox->setObjectName(QString::fromUtf8("sliceViewBox"));

        horizontalLayout_7->addWidget(sliceViewBox);


        verticalLayout->addLayout(horizontalLayout_7);

        graphicsView = new QGraphicsView(dockWidgetContents);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));

        verticalLayout->addWidget(graphicsView);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        SagView = new QToolButton(dockWidgetContents);
        SagView->setObjectName(QString::fromUtf8("SagView"));
        SagView->setMaximumSize(QSize(23, 22));
        SagView->setIcon(icon7);

        horizontalLayout_2->addWidget(SagView);

        SagSlider = new QSlider(dockWidgetContents);
        SagSlider->setObjectName(QString::fromUtf8("SagSlider"));
        SagSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(SagSlider);

        CorView = new QToolButton(dockWidgetContents);
        CorView->setObjectName(QString::fromUtf8("CorView"));
        CorView->setMaximumSize(QSize(23, 22));
        CorView->setIcon(icon6);

        horizontalLayout_2->addWidget(CorView);

        CorSlider = new QSlider(dockWidgetContents);
        CorSlider->setObjectName(QString::fromUtf8("CorSlider"));
        CorSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(CorSlider);

        AxiView = new QToolButton(dockWidgetContents);
        AxiView->setObjectName(QString::fromUtf8("AxiView"));
        AxiView->setMaximumSize(QSize(23, 22));
        AxiView->setIcon(icon4);

        horizontalLayout_2->addWidget(AxiView);

        AxiSlider = new QSlider(dockWidgetContents);
        AxiSlider->setObjectName(QString::fromUtf8("AxiSlider"));
        AxiSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(AxiSlider);


        verticalLayout->addLayout(horizontalLayout_2);

        dockWidget->setWidget(dockWidgetContents);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        dockWidget_3 = new QDockWidget(tracking_window);
        dockWidget_3->setObjectName(QString::fromUtf8("dockWidget_3"));
        QSizePolicy sizePolicy5(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(dockWidget_3->sizePolicy().hasHeightForWidth());
        dockWidget_3->setSizePolicy(sizePolicy5);
        dockWidget_3->setMinimumSize(QSize(499, 143));
        dockWidget_3->setMaximumSize(QSize(65535, 150));
        dockWidgetContents_3 = new QWidget();
        dockWidgetContents_3->setObjectName(QString::fromUtf8("dockWidgetContents_3"));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents_3);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, -1, 0);
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(5, -1, -1, -1);
        label = new QLabel(dockWidgetContents_3);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_9->addWidget(label);

        fa_threshold = new QDoubleSpinBox(dockWidgetContents_3);
        fa_threshold->setObjectName(QString::fromUtf8("fa_threshold"));
        fa_threshold->setDecimals(5);
        fa_threshold->setMaximum(99.99);
        fa_threshold->setSingleStep(0.01);

        horizontalLayout_9->addWidget(fa_threshold);


        gridLayout->addLayout(horizontalLayout_9, 3, 0, 1, 1);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(0);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(5, -1, -1, -1);
        label_2 = new QLabel(dockWidgetContents_3);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_10->addWidget(label_2);

        turning_angle = new QSpinBox(dockWidgetContents_3);
        turning_angle->setObjectName(QString::fromUtf8("turning_angle"));
        turning_angle->setMaximum(90);
        turning_angle->setSingleStep(5);
        turning_angle->setValue(60);

        horizontalLayout_10->addWidget(turning_angle);


        gridLayout->addLayout(horizontalLayout_10, 3, 1, 1, 1);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setSpacing(0);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        horizontalLayout_11->setContentsMargins(5, -1, -1, -1);
        label_5 = new QLabel(dockWidgetContents_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_11->addWidget(label_5);

        step_size = new QDoubleSpinBox(dockWidgetContents_3);
        step_size->setObjectName(QString::fromUtf8("step_size"));
        step_size->setDecimals(3);
        step_size->setMinimum(0.001);
        step_size->setMaximum(100);
        step_size->setSingleStep(0.5);
        step_size->setValue(0.5);

        horizontalLayout_11->addWidget(step_size);


        gridLayout->addLayout(horizontalLayout_11, 10, 0, 1, 1);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setSpacing(0);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        horizontalLayout_12->setContentsMargins(5, -1, -1, -1);
        label_4 = new QLabel(dockWidgetContents_3);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_12->addWidget(label_4);

        smoothing = new QDoubleSpinBox(dockWidgetContents_3);
        smoothing->setObjectName(QString::fromUtf8("smoothing"));
        smoothing->setMaximum(1);
        smoothing->setSingleStep(0.1);

        horizontalLayout_12->addWidget(smoothing);


        gridLayout->addLayout(horizontalLayout_12, 10, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_7 = new QLabel(dockWidgetContents_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout->addWidget(label_7);

        min_length = new QDoubleSpinBox(dockWidgetContents_3);
        min_length->setObjectName(QString::fromUtf8("min_length"));
        min_length->setMaximum(1000);
        min_length->setSingleStep(5);
        min_length->setValue(0);

        horizontalLayout->addWidget(min_length);

        max_length = new QDoubleSpinBox(dockWidgetContents_3);
        max_length->setObjectName(QString::fromUtf8("max_length"));
        max_length->setMinimum(1);
        max_length->setMaximum(10000);
        max_length->setSingleStep(10);
        max_length->setValue(500);

        horizontalLayout->addWidget(max_length);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        initial_direction = new QComboBox(dockWidgetContents_3);
        initial_direction->setObjectName(QString::fromUtf8("initial_direction"));

        horizontalLayout_3->addWidget(initial_direction);

        seed_plan = new QComboBox(dockWidgetContents_3);
        seed_plan->setObjectName(QString::fromUtf8("seed_plan"));

        horizontalLayout_3->addWidget(seed_plan);

        interpolation = new QComboBox(dockWidgetContents_3);
        interpolation->setObjectName(QString::fromUtf8("interpolation"));

        horizontalLayout_3->addWidget(interpolation);

        tracking_method = new QComboBox(dockWidgetContents_3);
        tracking_method->setObjectName(QString::fromUtf8("tracking_method"));

        horizontalLayout_3->addWidget(tracking_method);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        track_count = new QSpinBox(dockWidgetContents_3);
        track_count->setObjectName(QString::fromUtf8("track_count"));
        track_count->setMinimum(1);
        track_count->setMaximum(1000000000);
        track_count->setSingleStep(2000);
        track_count->setValue(2000);

        horizontalLayout_5->addWidget(track_count);

        tracking_plan = new QComboBox(dockWidgetContents_3);
        tracking_plan->setObjectName(QString::fromUtf8("tracking_plan"));
        tracking_plan->setMinimumSize(QSize(0, 0));

        horizontalLayout_5->addWidget(tracking_plan);

        thread_count = new QComboBox(dockWidgetContents_3);
        thread_count->setObjectName(QString::fromUtf8("thread_count"));

        horizontalLayout_5->addWidget(thread_count);

        perform_tracking = new QPushButton(dockWidgetContents_3);
        perform_tracking->setObjectName(QString::fromUtf8("perform_tracking"));
        perform_tracking->setMaximumSize(QSize(16777215, 25));
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/icons/icons/run.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        perform_tracking->setIcon(icon13);

        horizontalLayout_5->addWidget(perform_tracking);

        stopTracking = new QToolButton(dockWidgetContents_3);
        stopTracking->setObjectName(QString::fromUtf8("stopTracking"));
        stopTracking->setMaximumSize(QSize(16777215, 25));

        horizontalLayout_5->addWidget(stopTracking);


        verticalLayout_2->addLayout(horizontalLayout_5);

        dockWidget_3->setWidget(dockWidgetContents_3);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_3);

        menuBar->addAction(menu_Edit->menuAction());
        menuBar->addAction(menuRegions->menuAction());
        menuBar->addAction(menuTracts->menuAction());
        menuBar->addAction(menu_Slices->menuAction());
        menuBar->addAction(menu_View->menuAction());
        menuRegions->addAction(actionNewRegion);
        menuRegions->addAction(actionOpenRegion);
        menuRegions->addAction(actionSaveRegionAs);
        menuRegions->addAction(actionSave_Voxel_Data_As);
        menuRegions->addAction(actionSave_Anisotrpy_Map_as);
        menuRegions->addAction(actionDeleteRegion);
        menuRegions->addAction(actionDeleteRegionAll);
        menuRegions->addSeparator();
        menuRegions->addAction(menuModify->menuAction());
        menuRegions->addAction(actionWhole_brain_seeding);
        menuModify->addAction(actionShift_X);
        menuModify->addAction(actionShift_X_2);
        menuModify->addAction(actionShift_Y);
        menuModify->addAction(actionShift_Y_2);
        menuModify->addAction(actionShift_Z);
        menuModify->addAction(actionShift_Z_2);
        menuModify->addSeparator();
        menuModify->addAction(actionFlip_X);
        menuModify->addAction(actionFlip_Y);
        menuModify->addAction(actionFlip_Z);
        menuModify->addSeparator();
        menuModify->addAction(actionDilation);
        menuModify->addAction(actionErosion);
        menuModify->addAction(actionSmoothing);
        menuModify->addAction(actionNegate);
        menuModify->addAction(actionDefragment);
        menuModify->addAction(actionThreshold);
        menuTracts->addAction(actionOpenTract);
        menuTracts->addAction(actionSaveTractAs);
        menuTracts->addAction(menuSave->menuAction());
        menuTracts->addAction(menuExport_Tract_Density->menuAction());
        menuTracts->addSeparator();
        menuTracts->addAction(actionMerge_All);
        menuTracts->addAction(actionDeleteTract);
        menuTracts->addAction(actionDeleteTractAll);
        menuTracts->addSeparator();
        menuTracts->addAction(actionEndpoints_to_seeding);
        menuTracts->addAction(actionTracts_to_seeds);
        menuTracts->addSeparator();
        menuTracts->addAction(actionSet_Color);
        menuTracts->addAction(actionOpen_Colors);
        menuTracts->addAction(actionSave_Tracts_Colors_As);
        menuTracts->addAction(menuClustering->menuAction());
        menuTracts->addSeparator();
        menuTracts->addAction(actionStatistics);
        menuSave->addAction(actionQuantitative_anisotropy_QA);
        menuClustering->addAction(actionK_means);
        menuClustering->addAction(actionEM);
        menuClustering->addAction(actionHierarchical);
        menuExport_Tract_Density->addAction(actionTDI_Diffusion_Space);
        menuExport_Tract_Density->addAction(actionTDI_Subvoxel_Diffusion_Space);
        menuExport_Tract_Density->addAction(actionTDI_Import_Slice_Space);
        menu_Edit->addAction(actionUndo);
        menu_Edit->addAction(actionRedo);
        menu_Edit->addSeparator();
        menu_Edit->addAction(actionSelect_Tracts);
        menu_Edit->addAction(actionDelete);
        menu_Edit->addAction(actionTrim);
        menu_Edit->addAction(actionCut);
        menu_Edit->addAction(actionPaint);
        menu_Edit->addSeparator();
        menu_View->addAction(actionZoom_In);
        menu_View->addAction(actionZoom_Out);
        menu_View->addSeparator();
        menu_View->addAction(actionSagittal_view);
        menu_View->addAction(actionCoronal_View);
        menu_View->addAction(actionAxial_View);
        menu_View->addSeparator();
        menu_View->addAction(actionLoad_Camera);
        menu_View->addAction(actionSave_Camera);
        menu_View->addSeparator();
        menu_View->addAction(actionSave_Screen);
        menu_View->addAction(actionSave_ROI_Screen);
        menu_View->addAction(actionSave_Rotation_Images);
        menu_View->addAction(actionCopy_to_clipboard);
        menu_Slices->addAction(actionInsert_T1_T2);
        menu_Slices->addAction(actionAdd_surface);
        menu_Slices->addSeparator();
        menu_Slices->addAction(actionSave_mapping);
        menu_Slices->addAction(actionLoad_mapping);
        menu_Slices->addSeparator();
        menu_Slices->addAction(actionSave_Tracts_in_Current_Mapping);

        retranslateUi(tracking_window);
        QObject::connect(tbNewRegion, SIGNAL(clicked()), actionNewRegion, SLOT(trigger()));
        QObject::connect(tbOpenRegion, SIGNAL(clicked()), actionOpenRegion, SLOT(trigger()));
        QObject::connect(tbSaveRegion, SIGNAL(clicked()), actionSaveRegionAs, SLOT(trigger()));
        QObject::connect(tbDeleteRegion, SIGNAL(clicked()), actionDeleteRegion, SLOT(trigger()));
        QObject::connect(glAxiBox, SIGNAL(valueChanged(int)), glAxiSlider, SLOT(setValue(int)));
        QObject::connect(glCorBox, SIGNAL(valueChanged(int)), glCorSlider, SLOT(setValue(int)));
        QObject::connect(glSagBox, SIGNAL(valueChanged(int)), glSagSlider, SLOT(setValue(int)));
        QObject::connect(glAxiSlider, SIGNAL(valueChanged(int)), glAxiBox, SLOT(setValue(int)));
        QObject::connect(glCorSlider, SIGNAL(valueChanged(int)), glCorBox, SLOT(setValue(int)));
        QObject::connect(glSagSlider, SIGNAL(valueChanged(int)), glSagBox, SLOT(setValue(int)));
        QObject::connect(isosurfaceButton, SIGNAL(clicked()), actionAdd_surface, SLOT(trigger()));

        initial_direction->setCurrentIndex(0);
        tracking_plan->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(tracking_window);
    } // setupUi

    void retranslateUi(QMainWindow *tracking_window)
    {
        tracking_window->setWindowTitle(QApplication::translate("tracking_window", "Fiber Tracking", 0, QApplication::UnicodeUTF8));
        actionNewRegion->setText(QApplication::translate("tracking_window", "&New", 0, QApplication::UnicodeUTF8));
        actionOpenRegion->setText(QApplication::translate("tracking_window", "&Open...", 0, QApplication::UnicodeUTF8));
        actionSaveRegionAs->setText(QApplication::translate("tracking_window", "Save &as...", 0, QApplication::UnicodeUTF8));
        actionDeleteRegion->setText(QApplication::translate("tracking_window", "&Delete", 0, QApplication::UnicodeUTF8));
        actionOpenTract->setText(QApplication::translate("tracking_window", "Open Tracts...", 0, QApplication::UnicodeUTF8));
        actionSaveTractAs->setText(QApplication::translate("tracking_window", "Save Tracts As...", 0, QApplication::UnicodeUTF8));
        actionSaveTractFA->setText(QApplication::translate("tracking_window", "Save tract anisotropy as...", 0, QApplication::UnicodeUTF8));
        actionDeleteRegionAll->setText(QApplication::translate("tracking_window", "Delete All", 0, QApplication::UnicodeUTF8));
        actionDeleteTract->setText(QApplication::translate("tracking_window", "Delete", 0, QApplication::UnicodeUTF8));
        actionDeleteTractAll->setText(QApplication::translate("tracking_window", "Delete All", 0, QApplication::UnicodeUTF8));
        actionShift_X->setText(QApplication::translate("tracking_window", "Shift +X", 0, QApplication::UnicodeUTF8));
        actionShift_X->setShortcut(QApplication::translate("tracking_window", "Ctrl+1", 0, QApplication::UnicodeUTF8));
        actionShift_X_2->setText(QApplication::translate("tracking_window", "Shift -X", 0, QApplication::UnicodeUTF8));
        actionShift_X_2->setShortcut(QApplication::translate("tracking_window", "Ctrl+2", 0, QApplication::UnicodeUTF8));
        actionShift_Y->setText(QApplication::translate("tracking_window", "Shift +Y", 0, QApplication::UnicodeUTF8));
        actionShift_Y->setShortcut(QApplication::translate("tracking_window", "Ctrl+3", 0, QApplication::UnicodeUTF8));
        actionShift_Y_2->setText(QApplication::translate("tracking_window", "Shift -Y", 0, QApplication::UnicodeUTF8));
        actionShift_Y_2->setShortcut(QApplication::translate("tracking_window", "Ctrl+4", 0, QApplication::UnicodeUTF8));
        actionShift_Z->setText(QApplication::translate("tracking_window", "Shift +Z", 0, QApplication::UnicodeUTF8));
        actionShift_Z->setShortcut(QApplication::translate("tracking_window", "Ctrl+5", 0, QApplication::UnicodeUTF8));
        actionShift_Z_2->setText(QApplication::translate("tracking_window", "Shift -Z", 0, QApplication::UnicodeUTF8));
        actionShift_Z_2->setShortcut(QApplication::translate("tracking_window", "Ctrl+6", 0, QApplication::UnicodeUTF8));
        actionFlip_X->setText(QApplication::translate("tracking_window", "Flip X", 0, QApplication::UnicodeUTF8));
        actionFlip_X->setShortcut(QApplication::translate("tracking_window", "Ctrl+7", 0, QApplication::UnicodeUTF8));
        actionFlip_Y->setText(QApplication::translate("tracking_window", "Flip Y", 0, QApplication::UnicodeUTF8));
        actionFlip_Y->setShortcut(QApplication::translate("tracking_window", "Ctrl+8", 0, QApplication::UnicodeUTF8));
        actionFlip_Z->setText(QApplication::translate("tracking_window", "Flip Z", 0, QApplication::UnicodeUTF8));
        actionFlip_Z->setShortcut(QApplication::translate("tracking_window", "Ctrl+9", 0, QApplication::UnicodeUTF8));
        actionDilation->setText(QApplication::translate("tracking_window", "Dilation", 0, QApplication::UnicodeUTF8));
        actionDilation->setShortcut(QApplication::translate("tracking_window", "Ctrl+Shift+D", 0, QApplication::UnicodeUTF8));
        actionErosion->setText(QApplication::translate("tracking_window", "Erosion", 0, QApplication::UnicodeUTF8));
        actionErosion->setShortcut(QApplication::translate("tracking_window", "Ctrl+Shift+E", 0, QApplication::UnicodeUTF8));
        actionSmoothing->setText(QApplication::translate("tracking_window", "Smoothing", 0, QApplication::UnicodeUTF8));
        actionSmoothing->setShortcut(QApplication::translate("tracking_window", "Ctrl+Shift+S", 0, QApplication::UnicodeUTF8));
        actionNegate->setText(QApplication::translate("tracking_window", "Negate", 0, QApplication::UnicodeUTF8));
        actionNegate->setShortcut(QApplication::translate("tracking_window", "Ctrl+Shift+N", 0, QApplication::UnicodeUTF8));
        actionDefragment->setText(QApplication::translate("tracking_window", "Defragment", 0, QApplication::UnicodeUTF8));
        actionDefragment->setShortcut(QApplication::translate("tracking_window", "Ctrl+Shift+F", 0, QApplication::UnicodeUTF8));
        actionWhole_brain_seeding->setText(QApplication::translate("tracking_window", "Whole brain seeding", 0, QApplication::UnicodeUTF8));
        actionSelect_Tracts->setText(QApplication::translate("tracking_window", "Select", 0, QApplication::UnicodeUTF8));
        actionSelect_Tracts->setShortcut(QApplication::translate("tracking_window", "Ctrl+S", 0, QApplication::UnicodeUTF8));
        actionUndo->setText(QApplication::translate("tracking_window", "Undo", 0, QApplication::UnicodeUTF8));
        actionUndo->setShortcut(QApplication::translate("tracking_window", "Ctrl+Z", 0, QApplication::UnicodeUTF8));
        actionDelete->setText(QApplication::translate("tracking_window", "Delete", 0, QApplication::UnicodeUTF8));
        actionDelete->setShortcut(QApplication::translate("tracking_window", "Ctrl+D", 0, QApplication::UnicodeUTF8));
        actionTrim->setText(QApplication::translate("tracking_window", "Trim", 0, QApplication::UnicodeUTF8));
        actionTrim->setShortcut(QApplication::translate("tracking_window", "Ctrl+T", 0, QApplication::UnicodeUTF8));
        actionCut->setText(QApplication::translate("tracking_window", "Cut", 0, QApplication::UnicodeUTF8));
        actionCut->setShortcut(QApplication::translate("tracking_window", "Ctrl+X", 0, QApplication::UnicodeUTF8));
        actionZoom_In->setText(QApplication::translate("tracking_window", "Zoom &In", 0, QApplication::UnicodeUTF8));
        actionZoom_In->setShortcut(QApplication::translate("tracking_window", "F3", 0, QApplication::UnicodeUTF8));
        actionZoom_Out->setText(QApplication::translate("tracking_window", "Zoom Out&", 0, QApplication::UnicodeUTF8));
        actionZoom_Out->setShortcut(QApplication::translate("tracking_window", "F4", 0, QApplication::UnicodeUTF8));
        actionSagittal_view->setText(QApplication::translate("tracking_window", "&Sagittal View", 0, QApplication::UnicodeUTF8));
        actionSagittal_view->setShortcut(QApplication::translate("tracking_window", "F5", 0, QApplication::UnicodeUTF8));
        actionCoronal_View->setText(QApplication::translate("tracking_window", "&Coronal View", 0, QApplication::UnicodeUTF8));
        actionCoronal_View->setShortcut(QApplication::translate("tracking_window", "F6", 0, QApplication::UnicodeUTF8));
        actionAxial_View->setText(QApplication::translate("tracking_window", "&Axial View", 0, QApplication::UnicodeUTF8));
        actionAxial_View->setShortcut(QApplication::translate("tracking_window", "F7", 0, QApplication::UnicodeUTF8));
        actionQuantitative_anisotropy_QA->setText(QApplication::translate("tracking_window", "Save Quantitative anisotropy (QA)", 0, QApplication::UnicodeUTF8));
        actionMerge_All->setText(QApplication::translate("tracking_window", "Merge All", 0, QApplication::UnicodeUTF8));
        actionSave_Screen->setText(QApplication::translate("tracking_window", "Save 3D Screen...", 0, QApplication::UnicodeUTF8));
        actionSave_ROI_Screen->setText(QApplication::translate("tracking_window", "Save ROI Screen...", 0, QApplication::UnicodeUTF8));
        actionLoad_Camera->setText(QApplication::translate("tracking_window", "Open Camera...", 0, QApplication::UnicodeUTF8));
        actionSave_Camera->setText(QApplication::translate("tracking_window", "Save Camera", 0, QApplication::UnicodeUTF8));
        actionEndpoints_to_seeding->setText(QApplication::translate("tracking_window", "Endpoints to ROI", 0, QApplication::UnicodeUTF8));
        actionTracts_to_seeds->setText(QApplication::translate("tracking_window", "Tracts to ROI", 0, QApplication::UnicodeUTF8));
        actionSave_Rotation_Images->setText(QApplication::translate("tracking_window", "Save Rotation Images...", 0, QApplication::UnicodeUTF8));
        actionPaint->setText(QApplication::translate("tracking_window", "Paint", 0, QApplication::UnicodeUTF8));
        actionPaint->setShortcut(QApplication::translate("tracking_window", "Ctrl+P", 0, QApplication::UnicodeUTF8));
        actionInsert_T1_T2->setText(QApplication::translate("tracking_window", "Insert T1/T2...", 0, QApplication::UnicodeUTF8));
        actionAdd_surface->setText(QApplication::translate("tracking_window", "Add Isosurface", 0, QApplication::UnicodeUTF8));
        actionSave_mapping->setText(QApplication::translate("tracking_window", "Save Mapping...", 0, QApplication::UnicodeUTF8));
        actionLoad_mapping->setText(QApplication::translate("tracking_window", "Load Mapping...", 0, QApplication::UnicodeUTF8));
        actionStatistics->setText(QApplication::translate("tracking_window", "&Statistics...", 0, QApplication::UnicodeUTF8));
        actionK_means->setText(QApplication::translate("tracking_window", "K-means", 0, QApplication::UnicodeUTF8));
        actionEM->setText(QApplication::translate("tracking_window", "EM", 0, QApplication::UnicodeUTF8));
        actionHierarchical->setText(QApplication::translate("tracking_window", "Hierarchical", 0, QApplication::UnicodeUTF8));
        actionSet_Color->setText(QApplication::translate("tracking_window", "Set Color...", 0, QApplication::UnicodeUTF8));
        actionSave_Tracts_Colors_As->setText(QApplication::translate("tracking_window", "Save Tracts Colors As...", 0, QApplication::UnicodeUTF8));
        actionOpen_Colors->setText(QApplication::translate("tracking_window", "Open Colors...", 0, QApplication::UnicodeUTF8));
        actionSave_Voxel_Data_As->setText(QApplication::translate("tracking_window", "Save &voxel data as...", 0, QApplication::UnicodeUTF8));
        actionTDI_Diffusion_Space->setText(QApplication::translate("tracking_window", "Diffusion Space...", 0, QApplication::UnicodeUTF8));
        actionTDI_Import_Slice_Space->setText(QApplication::translate("tracking_window", "Current Slice Space...", 0, QApplication::UnicodeUTF8));
        actionTDI_Subvoxel_Diffusion_Space->setText(QApplication::translate("tracking_window", "Subvoxel Diffusion Space...", 0, QApplication::UnicodeUTF8));
        actionSave_Tracts_in_Current_Mapping->setText(QApplication::translate("tracking_window", "Save Tracts in Current Mapping...", 0, QApplication::UnicodeUTF8));
        actionThreshold->setText(QApplication::translate("tracking_window", "Threshold", 0, QApplication::UnicodeUTF8));
        actionRedo->setText(QApplication::translate("tracking_window", "Redo", 0, QApplication::UnicodeUTF8));
        actionRedo->setShortcut(QApplication::translate("tracking_window", "Ctrl+Y", 0, QApplication::UnicodeUTF8));
        actionCopy_to_clipboard->setText(QApplication::translate("tracking_window", "Copy to clipboard", 0, QApplication::UnicodeUTF8));
        actionCopy_to_clipboard->setShortcut(QApplication::translate("tracking_window", "Ctrl+C", 0, QApplication::UnicodeUTF8));
        actionSave_Anisotrpy_Map_as->setText(QApplication::translate("tracking_window", "Save anisotrpy map as...", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("tracking_window", "Current Slice:", 0, QApplication::UnicodeUTF8));
        SliceModality->clear();
        SliceModality->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Diffusion", 0, QApplication::UnicodeUTF8)
        );
        addSlices->setText(QString());
        isosurfaceButton->setText(QApplication::translate("tracking_window", "+isosurface", 0, QApplication::UnicodeUTF8));
        surfaceStyle->clear();
        surfaceStyle->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Full", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Right Hemisphere", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Left Hemisphere", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Lower Brain", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Upper Brain", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Anterior Half", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Posterior Half", 0, QApplication::UnicodeUTF8)
        );
        glSagCheck->setText(QString());
        glSagView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        glCorCheck->setText(QString());
        glCorView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        glAxiCheck->setText(QString());
        glAxiView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        regionDockWidget->setWindowTitle(QApplication::translate("tracking_window", "Region List", 0, QApplication::UnicodeUTF8));
        addRegionFromAtlas->setText(QApplication::translate("tracking_window", "Add", 0, QApplication::UnicodeUTF8));
        renderingWidgetHolder->setWindowTitle(QApplication::translate("tracking_window", "Rendering Options", 0, QApplication::UnicodeUTF8));
        tbDefaultParam->setText(QApplication::translate("tracking_window", "Set All To Default", 0, QApplication::UnicodeUTF8));
        RenderingQualityBox->clear();
        RenderingQualityBox->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Best Performance", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Default", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Highest Quality", 0, QApplication::UnicodeUTF8)
        );
        menuRegions->setTitle(QApplication::translate("tracking_window", "&Regions", 0, QApplication::UnicodeUTF8));
        menuModify->setTitle(QApplication::translate("tracking_window", "Modify current region", 0, QApplication::UnicodeUTF8));
        menuTracts->setTitle(QApplication::translate("tracking_window", "Tr&acts", 0, QApplication::UnicodeUTF8));
        menuSave->setTitle(QApplication::translate("tracking_window", "Save Index", 0, QApplication::UnicodeUTF8));
        menuClustering->setTitle(QApplication::translate("tracking_window", "Clustering", 0, QApplication::UnicodeUTF8));
        menuExport_Tract_Density->setTitle(QApplication::translate("tracking_window", "Export Tract Density", 0, QApplication::UnicodeUTF8));
        menu_Edit->setTitle(QApplication::translate("tracking_window", "&Edit", 0, QApplication::UnicodeUTF8));
        menu_View->setTitle(QApplication::translate("tracking_window", "&View", 0, QApplication::UnicodeUTF8));
        menu_Slices->setTitle(QApplication::translate("tracking_window", "&Slices", 0, QApplication::UnicodeUTF8));
        TractWidgetHolder->setWindowTitle(QApplication::translate("tracking_window", "Fiber Tracts", 0, QApplication::UnicodeUTF8));
        dockWidget->setWindowTitle(QApplication::translate("tracking_window", "Step1: select regions", 0, QApplication::UnicodeUTF8));
        zoomIn->setText(QApplication::translate("tracking_window", "+", 0, QApplication::UnicodeUTF8));
        zoomOut->setText(QApplication::translate("tracking_window", "-", 0, QApplication::UnicodeUTF8));
        tbNewRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbOpenRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbSaveRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbDeleteRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tool0->setToolTip(QApplication::translate("tracking_window", "Rectangle seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool0->setText(QApplication::translate("tracking_window", "Rec", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tool1->setToolTip(QApplication::translate("tracking_window", "Polygon seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool1->setText(QApplication::translate("tracking_window", "Poly", 0, QApplication::UnicodeUTF8));
        tool4->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tool2->setToolTip(QApplication::translate("tracking_window", "Ball seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool2->setText(QApplication::translate("tracking_window", "Ball", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tool3->setToolTip(QApplication::translate("tracking_window", "Cubic seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool3->setText(QApplication::translate("tracking_window", "Cubic", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        whole_brain->setToolTip(QApplication::translate("tracking_window", "Whole brain seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        whole_brain->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        SagView->setToolTip(QApplication::translate("tracking_window", "Sagittal view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        SagView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        CorView->setToolTip(QApplication::translate("tracking_window", "Coronal view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        CorView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        AxiView->setToolTip(QApplication::translate("tracking_window", "Axial view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        AxiView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        dockWidget_3->setWindowTitle(QApplication::translate("tracking_window", "Step2: setup parameters and perform tracking", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("tracking_window", "Fiber threshold", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("tracking_window", "Max Angle", 0, QApplication::UnicodeUTF8));
        turning_angle->setSuffix(QString());
        label_5->setText(QApplication::translate("tracking_window", "Step Size (mm)", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("tracking_window", "Smoothing", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("tracking_window", "Length constraint (mm)", 0, QApplication::UnicodeUTF8));
        initial_direction->clear();
        initial_direction->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Main Direction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Random Direction", 0, QApplication::UnicodeUTF8)
        );
        seed_plan->clear();
        seed_plan->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Subvoxel", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Voxelwise", 0, QApplication::UnicodeUTF8)
        );
        interpolation->clear();
        interpolation->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Trilinear", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Gaussian radial basis", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Nearest neighbor", 0, QApplication::UnicodeUTF8)
        );
        tracking_method->clear();
        tracking_method->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Streamline", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "RK4", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Streamline relocation", 0, QApplication::UnicodeUTF8)
        );
        tracking_plan->clear();
        tracking_plan->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Seeds", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Tracts", 0, QApplication::UnicodeUTF8)
        );
        thread_count->clear();
        thread_count->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "1 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "2 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "3 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "4 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "5 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "6 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "7 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "8 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "9 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "10 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "11 Threads", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "12 Threads", 0, QApplication::UnicodeUTF8)
        );
        perform_tracking->setText(QApplication::translate("tracking_window", "Tracking", 0, QApplication::UnicodeUTF8));
        stopTracking->setText(QApplication::translate("tracking_window", "Stop", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class tracking_window: public Ui_tracking_window {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRACKING_WINDOW_H
