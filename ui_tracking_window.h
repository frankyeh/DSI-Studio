/********************************************************************************
** Form generated from reading UI file 'tracking_window.ui'
**
** Created: Wed Jun 18 00:40:48 2014
**      by: Qt User Interface Compiler version 4.8.2
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
    QAction *actionRestore_window_layout;
    QAction *actionSave_Endpoints_in_Current_Mapping;
    QAction *actionMove_Object;
    QAction *actionSave_Report_as;
    QAction *actionSave_Tracts_in_MNI_space;
    QAction *actionOpen_Clusters;
    QAction *actionSave_Clusters;
    QAction *actionOpen_Cluster_Labels;
    QAction *actionSave_All_Tracts_As;
    QAction *actionSave_Left_Right_3D_Image;
    QAction *actionRegion_statistics;
    QAction *actionManual_Registration;
    QAction *actionTract_Analysis_Report;
    QAction *actionSave_End_Points_As;
    QAction *actionConnectivity_matrix;
    QAction *actionCheck_all_regions;
    QAction *actionUnckech_all_regions;
    QAction *actionCopyTrack;
    QAction *actionConnectometry;
    QAction *actionSave_3D_screen_in_high_resolution;
    QAction *actionSave_All_Regions_As;
    QAction *actionFloat_3D_window;
    QAction *actionSave_tracking_parameters;
    QAction *actionLoad_tracking_parameters;
    QAction *actionCheck_all_tracts;
    QAction *actionUncheck_all_tracts;
    QAction *actionCopy_Region;
    QAction *actionSave_Rendering_Parameters;
    QAction *actionLoad_Rendering_Parameters;
    QWidget *centralwidget;
    QVBoxLayout *centralLayout;
    QWidget *main_widget;
    QVBoxLayout *main_layout;
    QHBoxLayout *horizontalLayout_13;
    QComboBox *SliceModality;
    QToolButton *addSlices;
    QToolButton *deleteSlice;
    QToolButton *isosurfaceButton;
    QComboBox *surfaceStyle;
    QHBoxLayout *horizontalLayout;
    QLabel *label_16;
    QDoubleSpinBox *gl_contrast_value;
    QSlider *gl_contrast;
    QLabel *label_17;
    QDoubleSpinBox *gl_offset_value;
    QSlider *gl_offset;
    QToolButton *move3Dwindow;
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
    QLabel *label;
    QDoubleSpinBox *zoom_3d;
    QStatusBar *statusbar;
    QDockWidget *regionDockWidget;
    QWidget *dockWidgetContents_4;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout_16;
    QToolButton *tbNewRegion;
    QToolButton *tbOpenRegion;
    QToolButton *tbSaveRegion;
    QToolButton *tbDeleteRegion;
    QToolButton *whole_brain;
    QFrame *line_5;
    QComboBox *atlasListBox;
    QComboBox *atlasComboBox;
    QToolButton *addRegionFromAtlas;
    QSpacerItem *horizontalSpacer_5;
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
    QMenu *menuTract_Color;
    QMenu *menuSave_Tracts;
    QMenu *menu_Edit;
    QMenu *menu_View;
    QMenu *menu_Slices;
    QMenu *menuTools;
    QDockWidget *TractWidgetHolder;
    QWidget *dockWidgetContents_5;
    QVBoxLayout *tractverticalLayout;
    QHBoxLayout *horizontalLayout_15;
    QFrame *line_9;
    QToolButton *tbOpenTract;
    QToolButton *tbSaveTract;
    QToolButton *save_all_tracks;
    QToolButton *tbDeleteTract;
    QFrame *line_2;
    QToolButton *track_up;
    QToolButton *track_down;
    QSpacerItem *horizontalSpacer_4;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_4;
    QToolButton *tool0;
    QToolButton *tool1;
    QToolButton *tool4;
    QToolButton *tool2;
    QToolButton *tool3;
    QToolButton *tool6;
    QToolButton *tool5;
    QComboBox *sliceViewBox;
    QComboBox *overlay;
    QHBoxLayout *horizontalLayout_14;
    QLabel *label_6;
    QDoubleSpinBox *contrast_value;
    QSlider *contrast;
    QLabel *label_8;
    QDoubleSpinBox *offset_value;
    QSlider *offset;
    QPushButton *perform_tracking;
    QGraphicsView *graphicsView;
    QHBoxLayout *horizontalLayout_2;
    QToolButton *SagView;
    QSlider *SagSlider;
    QToolButton *CorView;
    QSlider *CorSlider;
    QToolButton *AxiView;
    QSlider *AxiSlider;

    void setupUi(QMainWindow *tracking_window)
    {
        if (tracking_window->objectName().isEmpty())
            tracking_window->setObjectName(QString::fromUtf8("tracking_window"));
        tracking_window->resize(1559, 424);
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
        actionRestore_window_layout = new QAction(tracking_window);
        actionRestore_window_layout->setObjectName(QString::fromUtf8("actionRestore_window_layout"));
        actionSave_Endpoints_in_Current_Mapping = new QAction(tracking_window);
        actionSave_Endpoints_in_Current_Mapping->setObjectName(QString::fromUtf8("actionSave_Endpoints_in_Current_Mapping"));
        actionSave_Endpoints_in_Current_Mapping->setIcon(icon2);
        actionMove_Object = new QAction(tracking_window);
        actionMove_Object->setObjectName(QString::fromUtf8("actionMove_Object"));
        actionSave_Report_as = new QAction(tracking_window);
        actionSave_Report_as->setObjectName(QString::fromUtf8("actionSave_Report_as"));
        actionSave_Report_as->setIcon(icon2);
        actionSave_Tracts_in_MNI_space = new QAction(tracking_window);
        actionSave_Tracts_in_MNI_space->setObjectName(QString::fromUtf8("actionSave_Tracts_in_MNI_space"));
        actionSave_Tracts_in_MNI_space->setIcon(icon2);
        actionOpen_Clusters = new QAction(tracking_window);
        actionOpen_Clusters->setObjectName(QString::fromUtf8("actionOpen_Clusters"));
        actionOpen_Clusters->setIcon(icon1);
        actionSave_Clusters = new QAction(tracking_window);
        actionSave_Clusters->setObjectName(QString::fromUtf8("actionSave_Clusters"));
        actionSave_Clusters->setIcon(icon2);
        actionOpen_Cluster_Labels = new QAction(tracking_window);
        actionOpen_Cluster_Labels->setObjectName(QString::fromUtf8("actionOpen_Cluster_Labels"));
        actionOpen_Cluster_Labels->setIcon(icon1);
        actionSave_All_Tracts_As = new QAction(tracking_window);
        actionSave_All_Tracts_As->setObjectName(QString::fromUtf8("actionSave_All_Tracts_As"));
        actionSave_All_Tracts_As->setIcon(icon2);
        actionSave_Left_Right_3D_Image = new QAction(tracking_window);
        actionSave_Left_Right_3D_Image->setObjectName(QString::fromUtf8("actionSave_Left_Right_3D_Image"));
        actionSave_Left_Right_3D_Image->setIcon(icon2);
        actionRegion_statistics = new QAction(tracking_window);
        actionRegion_statistics->setObjectName(QString::fromUtf8("actionRegion_statistics"));
        actionManual_Registration = new QAction(tracking_window);
        actionManual_Registration->setObjectName(QString::fromUtf8("actionManual_Registration"));
        actionTract_Analysis_Report = new QAction(tracking_window);
        actionTract_Analysis_Report->setObjectName(QString::fromUtf8("actionTract_Analysis_Report"));
        actionSave_End_Points_As = new QAction(tracking_window);
        actionSave_End_Points_As->setObjectName(QString::fromUtf8("actionSave_End_Points_As"));
        actionSave_End_Points_As->setIcon(icon2);
        actionConnectivity_matrix = new QAction(tracking_window);
        actionConnectivity_matrix->setObjectName(QString::fromUtf8("actionConnectivity_matrix"));
        actionCheck_all_regions = new QAction(tracking_window);
        actionCheck_all_regions->setObjectName(QString::fromUtf8("actionCheck_all_regions"));
        actionUnckech_all_regions = new QAction(tracking_window);
        actionUnckech_all_regions->setObjectName(QString::fromUtf8("actionUnckech_all_regions"));
        actionCopyTrack = new QAction(tracking_window);
        actionCopyTrack->setObjectName(QString::fromUtf8("actionCopyTrack"));
        actionConnectometry = new QAction(tracking_window);
        actionConnectometry->setObjectName(QString::fromUtf8("actionConnectometry"));
        actionSave_3D_screen_in_high_resolution = new QAction(tracking_window);
        actionSave_3D_screen_in_high_resolution->setObjectName(QString::fromUtf8("actionSave_3D_screen_in_high_resolution"));
        actionSave_3D_screen_in_high_resolution->setIcon(icon2);
        actionSave_All_Regions_As = new QAction(tracking_window);
        actionSave_All_Regions_As->setObjectName(QString::fromUtf8("actionSave_All_Regions_As"));
        actionSave_All_Regions_As->setIcon(icon2);
        actionFloat_3D_window = new QAction(tracking_window);
        actionFloat_3D_window->setObjectName(QString::fromUtf8("actionFloat_3D_window"));
        actionSave_tracking_parameters = new QAction(tracking_window);
        actionSave_tracking_parameters->setObjectName(QString::fromUtf8("actionSave_tracking_parameters"));
        actionSave_tracking_parameters->setIcon(icon2);
        actionLoad_tracking_parameters = new QAction(tracking_window);
        actionLoad_tracking_parameters->setObjectName(QString::fromUtf8("actionLoad_tracking_parameters"));
        actionLoad_tracking_parameters->setIcon(icon1);
        actionCheck_all_tracts = new QAction(tracking_window);
        actionCheck_all_tracts->setObjectName(QString::fromUtf8("actionCheck_all_tracts"));
        actionUncheck_all_tracts = new QAction(tracking_window);
        actionUncheck_all_tracts->setObjectName(QString::fromUtf8("actionUncheck_all_tracts"));
        actionCopy_Region = new QAction(tracking_window);
        actionCopy_Region->setObjectName(QString::fromUtf8("actionCopy_Region"));
        actionSave_Rendering_Parameters = new QAction(tracking_window);
        actionSave_Rendering_Parameters->setObjectName(QString::fromUtf8("actionSave_Rendering_Parameters"));
        actionSave_Rendering_Parameters->setIcon(icon2);
        actionLoad_Rendering_Parameters = new QAction(tracking_window);
        actionLoad_Rendering_Parameters->setObjectName(QString::fromUtf8("actionLoad_Rendering_Parameters"));
        actionLoad_Rendering_Parameters->setIcon(icon1);
        centralwidget = new QWidget(tracking_window);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        centralLayout = new QVBoxLayout(centralwidget);
        centralLayout->setSpacing(7);
        centralLayout->setContentsMargins(0, 0, 0, 0);
        centralLayout->setObjectName(QString::fromUtf8("centralLayout"));
        main_widget = new QWidget(centralwidget);
        main_widget->setObjectName(QString::fromUtf8("main_widget"));
        main_layout = new QVBoxLayout(main_widget);
        main_layout->setSpacing(0);
        main_layout->setContentsMargins(0, 0, 0, 0);
        main_layout->setObjectName(QString::fromUtf8("main_layout"));
        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setSpacing(0);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        SliceModality = new QComboBox(main_widget);
        SliceModality->setObjectName(QString::fromUtf8("SliceModality"));
        SliceModality->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_13->addWidget(SliceModality);

        addSlices = new QToolButton(main_widget);
        addSlices->setObjectName(QString::fromUtf8("addSlices"));
        addSlices->setMaximumSize(QSize(24, 22));
        addSlices->setIcon(icon1);
        addSlices->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_13->addWidget(addSlices);

        deleteSlice = new QToolButton(main_widget);
        deleteSlice->setObjectName(QString::fromUtf8("deleteSlice"));
        deleteSlice->setMaximumSize(QSize(24, 22));
        deleteSlice->setIcon(icon3);

        horizontalLayout_13->addWidget(deleteSlice);

        isosurfaceButton = new QToolButton(main_widget);
        isosurfaceButton->setObjectName(QString::fromUtf8("isosurfaceButton"));
        isosurfaceButton->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_13->addWidget(isosurfaceButton);

        surfaceStyle = new QComboBox(main_widget);
        surfaceStyle->setObjectName(QString::fromUtf8("surfaceStyle"));
        surfaceStyle->setMaximumSize(QSize(16777215, 22));

        horizontalLayout_13->addWidget(surfaceStyle);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_16 = new QLabel(main_widget);
        label_16->setObjectName(QString::fromUtf8("label_16"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_16->sizePolicy().hasHeightForWidth());
        label_16->setSizePolicy(sizePolicy1);
        label_16->setMaximumSize(QSize(16777215, 22));

        horizontalLayout->addWidget(label_16);

        gl_contrast_value = new QDoubleSpinBox(main_widget);
        gl_contrast_value->setObjectName(QString::fromUtf8("gl_contrast_value"));
        QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(gl_contrast_value->sizePolicy().hasHeightForWidth());
        gl_contrast_value->setSizePolicy(sizePolicy2);
        gl_contrast_value->setMaximumSize(QSize(16777215, 22));
        gl_contrast_value->setMinimum(1);
        gl_contrast_value->setMaximum(5);
        gl_contrast_value->setSingleStep(0.5);
        gl_contrast_value->setValue(1);

        horizontalLayout->addWidget(gl_contrast_value);

        gl_contrast = new QSlider(main_widget);
        gl_contrast->setObjectName(QString::fromUtf8("gl_contrast"));
        gl_contrast->setMaximumSize(QSize(16777215, 22));
        gl_contrast->setMinimum(-100);
        gl_contrast->setMaximum(100);
        gl_contrast->setSingleStep(5);
        gl_contrast->setPageStep(10);
        gl_contrast->setValue(0);
        gl_contrast->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(gl_contrast);

        label_17 = new QLabel(main_widget);
        label_17->setObjectName(QString::fromUtf8("label_17"));
        sizePolicy1.setHeightForWidth(label_17->sizePolicy().hasHeightForWidth());
        label_17->setSizePolicy(sizePolicy1);
        label_17->setMaximumSize(QSize(16777215, 22));

        horizontalLayout->addWidget(label_17);

        gl_offset_value = new QDoubleSpinBox(main_widget);
        gl_offset_value->setObjectName(QString::fromUtf8("gl_offset_value"));
        sizePolicy2.setHeightForWidth(gl_offset_value->sizePolicy().hasHeightForWidth());
        gl_offset_value->setSizePolicy(sizePolicy2);
        gl_offset_value->setMaximumSize(QSize(16777215, 22));
        gl_offset_value->setMinimum(-1);
        gl_offset_value->setMaximum(1);
        gl_offset_value->setSingleStep(0.1);

        horizontalLayout->addWidget(gl_offset_value);

        gl_offset = new QSlider(main_widget);
        gl_offset->setObjectName(QString::fromUtf8("gl_offset"));
        gl_offset->setMaximumSize(QSize(16777215, 22));
        gl_offset->setMinimum(-100);
        gl_offset->setMaximum(100);
        gl_offset->setSingleStep(5);
        gl_offset->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(gl_offset);


        horizontalLayout_13->addLayout(horizontalLayout);

        move3Dwindow = new QToolButton(main_widget);
        move3Dwindow->setObjectName(QString::fromUtf8("move3Dwindow"));
        move3Dwindow->setMaximumSize(QSize(22, 22));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/icons/move.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        move3Dwindow->setIcon(icon7);

        horizontalLayout_13->addWidget(move3Dwindow);


        main_layout->addLayout(horizontalLayout_13);

        centralLayout2 = new QHBoxLayout();
        centralLayout2->setSpacing(0);
        centralLayout2->setObjectName(QString::fromUtf8("centralLayout2"));
        glSagCheck = new QCheckBox(main_widget);
        glSagCheck->setObjectName(QString::fromUtf8("glSagCheck"));
        sizePolicy2.setHeightForWidth(glSagCheck->sizePolicy().hasHeightForWidth());
        glSagCheck->setSizePolicy(sizePolicy2);
        glSagCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glSagCheck);

        glSagView = new QToolButton(main_widget);
        glSagView->setObjectName(QString::fromUtf8("glSagView"));
        glSagView->setMaximumSize(QSize(22, 22));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/icons/icons/sag.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        glSagView->setIcon(icon8);

        centralLayout2->addWidget(glSagView);

        glSagBox = new QSpinBox(main_widget);
        glSagBox->setObjectName(QString::fromUtf8("glSagBox"));
        QSizePolicy sizePolicy3(QSizePolicy::Maximum, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(glSagBox->sizePolicy().hasHeightForWidth());
        glSagBox->setSizePolicy(sizePolicy3);

        centralLayout2->addWidget(glSagBox);

        glSagSlider = new QSlider(main_widget);
        glSagSlider->setObjectName(QString::fromUtf8("glSagSlider"));
        glSagSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glSagSlider);

        glCorCheck = new QCheckBox(main_widget);
        glCorCheck->setObjectName(QString::fromUtf8("glCorCheck"));
        sizePolicy2.setHeightForWidth(glCorCheck->sizePolicy().hasHeightForWidth());
        glCorCheck->setSizePolicy(sizePolicy2);
        glCorCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glCorCheck);

        glCorView = new QToolButton(main_widget);
        glCorView->setObjectName(QString::fromUtf8("glCorView"));
        glCorView->setMaximumSize(QSize(22, 22));
        glCorView->setIcon(icon6);

        centralLayout2->addWidget(glCorView);

        glCorBox = new QSpinBox(main_widget);
        glCorBox->setObjectName(QString::fromUtf8("glCorBox"));
        sizePolicy3.setHeightForWidth(glCorBox->sizePolicy().hasHeightForWidth());
        glCorBox->setSizePolicy(sizePolicy3);

        centralLayout2->addWidget(glCorBox);

        glCorSlider = new QSlider(main_widget);
        glCorSlider->setObjectName(QString::fromUtf8("glCorSlider"));
        glCorSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glCorSlider);

        glAxiCheck = new QCheckBox(main_widget);
        glAxiCheck->setObjectName(QString::fromUtf8("glAxiCheck"));
        sizePolicy2.setHeightForWidth(glAxiCheck->sizePolicy().hasHeightForWidth());
        glAxiCheck->setSizePolicy(sizePolicy2);
        glAxiCheck->setMinimumSize(QSize(16, 0));

        centralLayout2->addWidget(glAxiCheck);

        glAxiView = new QToolButton(main_widget);
        glAxiView->setObjectName(QString::fromUtf8("glAxiView"));
        glAxiView->setMinimumSize(QSize(0, 0));
        glAxiView->setMaximumSize(QSize(22, 22));
        glAxiView->setIcon(icon4);

        centralLayout2->addWidget(glAxiView);

        glAxiBox = new QSpinBox(main_widget);
        glAxiBox->setObjectName(QString::fromUtf8("glAxiBox"));
        sizePolicy3.setHeightForWidth(glAxiBox->sizePolicy().hasHeightForWidth());
        glAxiBox->setSizePolicy(sizePolicy3);

        centralLayout2->addWidget(glAxiBox);

        glAxiSlider = new QSlider(main_widget);
        glAxiSlider->setObjectName(QString::fromUtf8("glAxiSlider"));
        glAxiSlider->setOrientation(Qt::Horizontal);

        centralLayout2->addWidget(glAxiSlider);

        label = new QLabel(main_widget);
        label->setObjectName(QString::fromUtf8("label"));

        centralLayout2->addWidget(label);

        zoom_3d = new QDoubleSpinBox(main_widget);
        zoom_3d->setObjectName(QString::fromUtf8("zoom_3d"));
        zoom_3d->setMinimum(0.01);
        zoom_3d->setMaximum(100);
        zoom_3d->setSingleStep(0.1);
        zoom_3d->setValue(1);

        centralLayout2->addWidget(zoom_3d);


        main_layout->addLayout(centralLayout2);


        centralLayout->addWidget(main_widget);

        tracking_window->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(tracking_window);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        tracking_window->setStatusBar(statusbar);
        regionDockWidget = new QDockWidget(tracking_window);
        regionDockWidget->setObjectName(QString::fromUtf8("regionDockWidget"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(1);
        sizePolicy4.setHeightForWidth(regionDockWidget->sizePolicy().hasHeightForWidth());
        regionDockWidget->setSizePolicy(sizePolicy4);
        regionDockWidget->setMinimumSize(QSize(311, 46));
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
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setSpacing(0);
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        tbNewRegion = new QToolButton(dockWidgetContents_4);
        tbNewRegion->setObjectName(QString::fromUtf8("tbNewRegion"));
        tbNewRegion->setMaximumSize(QSize(23, 22));
        tbNewRegion->setIcon(icon);

        horizontalLayout_16->addWidget(tbNewRegion);

        tbOpenRegion = new QToolButton(dockWidgetContents_4);
        tbOpenRegion->setObjectName(QString::fromUtf8("tbOpenRegion"));
        tbOpenRegion->setMaximumSize(QSize(23, 22));
        tbOpenRegion->setIcon(icon1);

        horizontalLayout_16->addWidget(tbOpenRegion);

        tbSaveRegion = new QToolButton(dockWidgetContents_4);
        tbSaveRegion->setObjectName(QString::fromUtf8("tbSaveRegion"));
        tbSaveRegion->setMaximumSize(QSize(23, 22));
        tbSaveRegion->setIcon(icon2);

        horizontalLayout_16->addWidget(tbSaveRegion);

        tbDeleteRegion = new QToolButton(dockWidgetContents_4);
        tbDeleteRegion->setObjectName(QString::fromUtf8("tbDeleteRegion"));
        tbDeleteRegion->setMaximumSize(QSize(23, 22));
        tbDeleteRegion->setIcon(icon3);

        horizontalLayout_16->addWidget(tbDeleteRegion);

        whole_brain = new QToolButton(dockWidgetContents_4);
        whole_brain->setObjectName(QString::fromUtf8("whole_brain"));
        sizePolicy2.setHeightForWidth(whole_brain->sizePolicy().hasHeightForWidth());
        whole_brain->setSizePolicy(sizePolicy2);
        whole_brain->setMaximumSize(QSize(23, 22));
        whole_brain->setIcon(icon4);

        horizontalLayout_16->addWidget(whole_brain);

        line_5 = new QFrame(dockWidgetContents_4);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setFrameShape(QFrame::VLine);
        line_5->setFrameShadow(QFrame::Sunken);

        horizontalLayout_16->addWidget(line_5);

        atlasListBox = new QComboBox(dockWidgetContents_4);
        atlasListBox->setObjectName(QString::fromUtf8("atlasListBox"));
        QSizePolicy sizePolicy5(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(atlasListBox->sizePolicy().hasHeightForWidth());
        atlasListBox->setSizePolicy(sizePolicy5);
        atlasListBox->setMaximumSize(QSize(16777215, 22));
        atlasListBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLength);

        horizontalLayout_16->addWidget(atlasListBox);

        atlasComboBox = new QComboBox(dockWidgetContents_4);
        atlasComboBox->setObjectName(QString::fromUtf8("atlasComboBox"));
        sizePolicy5.setHeightForWidth(atlasComboBox->sizePolicy().hasHeightForWidth());
        atlasComboBox->setSizePolicy(sizePolicy5);
        atlasComboBox->setMaximumSize(QSize(16777215, 22));
        atlasComboBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLength);

        horizontalLayout_16->addWidget(atlasComboBox);

        addRegionFromAtlas = new QToolButton(dockWidgetContents_4);
        addRegionFromAtlas->setObjectName(QString::fromUtf8("addRegionFromAtlas"));
        addRegionFromAtlas->setMaximumSize(QSize(16777215, 22));
        addRegionFromAtlas->setIcon(icon5);
        addRegionFromAtlas->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_16->addWidget(addRegionFromAtlas);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_16->addItem(horizontalSpacer_5);


        verticalLayout_3->addLayout(horizontalLayout_16);

        regionDockWidget->setWidget(dockWidgetContents_4);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(1), regionDockWidget);
        renderingWidgetHolder = new QDockWidget(tracking_window);
        renderingWidgetHolder->setObjectName(QString::fromUtf8("renderingWidgetHolder"));
        QSizePolicy sizePolicy6(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(renderingWidgetHolder->sizePolicy().hasHeightForWidth());
        renderingWidgetHolder->setSizePolicy(sizePolicy6);
        renderingWidgetHolder->setMinimumSize(QSize(211, 46));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QString::fromUtf8("dockWidgetContents_2"));
        renderingLayout = new QVBoxLayout(dockWidgetContents_2);
        renderingLayout->setSpacing(0);
        renderingLayout->setContentsMargins(0, 0, 0, 0);
        renderingLayout->setObjectName(QString::fromUtf8("renderingLayout"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        tbDefaultParam = new QToolButton(dockWidgetContents_2);
        tbDefaultParam->setObjectName(QString::fromUtf8("tbDefaultParam"));
        tbDefaultParam->setMaximumSize(QSize(16777215, 22));

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
        menuBar->setGeometry(QRect(0, 0, 1559, 21));
        QSizePolicy sizePolicy7(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(menuBar->sizePolicy().hasHeightForWidth());
        menuBar->setSizePolicy(sizePolicy7);
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
        menuSave->setIcon(icon2);
        menuClustering = new QMenu(menuTracts);
        menuClustering->setObjectName(QString::fromUtf8("menuClustering"));
        menuExport_Tract_Density = new QMenu(menuTracts);
        menuExport_Tract_Density->setObjectName(QString::fromUtf8("menuExport_Tract_Density"));
        menuExport_Tract_Density->setIcon(icon2);
        menuTract_Color = new QMenu(menuTracts);
        menuTract_Color->setObjectName(QString::fromUtf8("menuTract_Color"));
        menuSave_Tracts = new QMenu(menuTracts);
        menuSave_Tracts->setObjectName(QString::fromUtf8("menuSave_Tracts"));
        menuSave_Tracts->setIcon(icon2);
        menu_Edit = new QMenu(menuBar);
        menu_Edit->setObjectName(QString::fromUtf8("menu_Edit"));
        menu_View = new QMenu(menuBar);
        menu_View->setObjectName(QString::fromUtf8("menu_View"));
        menu_Slices = new QMenu(menuBar);
        menu_Slices->setObjectName(QString::fromUtf8("menu_Slices"));
        menuTools = new QMenu(menuBar);
        menuTools->setObjectName(QString::fromUtf8("menuTools"));
        tracking_window->setMenuBar(menuBar);
        TractWidgetHolder = new QDockWidget(tracking_window);
        TractWidgetHolder->setObjectName(QString::fromUtf8("TractWidgetHolder"));
        sizePolicy6.setHeightForWidth(TractWidgetHolder->sizePolicy().hasHeightForWidth());
        TractWidgetHolder->setSizePolicy(sizePolicy6);
        TractWidgetHolder->setMinimumSize(QSize(146, 46));
        TractWidgetHolder->setMaximumSize(QSize(524287, 524287));
        dockWidgetContents_5 = new QWidget();
        dockWidgetContents_5->setObjectName(QString::fromUtf8("dockWidgetContents_5"));
        tractverticalLayout = new QVBoxLayout(dockWidgetContents_5);
        tractverticalLayout->setSpacing(0);
        tractverticalLayout->setContentsMargins(0, 0, 0, 0);
        tractverticalLayout->setObjectName(QString::fromUtf8("tractverticalLayout"));
        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setSpacing(0);
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        line_9 = new QFrame(dockWidgetContents_5);
        line_9->setObjectName(QString::fromUtf8("line_9"));
        line_9->setFrameShape(QFrame::VLine);
        line_9->setFrameShadow(QFrame::Sunken);

        horizontalLayout_15->addWidget(line_9);

        tbOpenTract = new QToolButton(dockWidgetContents_5);
        tbOpenTract->setObjectName(QString::fromUtf8("tbOpenTract"));
        tbOpenTract->setMaximumSize(QSize(23, 22));
        tbOpenTract->setIcon(icon1);

        horizontalLayout_15->addWidget(tbOpenTract);

        tbSaveTract = new QToolButton(dockWidgetContents_5);
        tbSaveTract->setObjectName(QString::fromUtf8("tbSaveTract"));
        tbSaveTract->setMaximumSize(QSize(23, 22));
        tbSaveTract->setIcon(icon2);

        horizontalLayout_15->addWidget(tbSaveTract);

        save_all_tracks = new QToolButton(dockWidgetContents_5);
        save_all_tracks->setObjectName(QString::fromUtf8("save_all_tracks"));
        save_all_tracks->setMaximumSize(QSize(23, 22));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/icons/icons/save_all.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        save_all_tracks->setIcon(icon9);

        horizontalLayout_15->addWidget(save_all_tracks);

        tbDeleteTract = new QToolButton(dockWidgetContents_5);
        tbDeleteTract->setObjectName(QString::fromUtf8("tbDeleteTract"));
        tbDeleteTract->setMaximumSize(QSize(23, 22));
        tbDeleteTract->setIcon(icon3);

        horizontalLayout_15->addWidget(tbDeleteTract);

        line_2 = new QFrame(dockWidgetContents_5);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout_15->addWidget(line_2);

        track_up = new QToolButton(dockWidgetContents_5);
        track_up->setObjectName(QString::fromUtf8("track_up"));
        track_up->setMaximumSize(QSize(23, 22));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/icons/up.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        track_up->setIcon(icon10);

        horizontalLayout_15->addWidget(track_up);

        track_down = new QToolButton(dockWidgetContents_5);
        track_down->setObjectName(QString::fromUtf8("track_down"));
        track_down->setMaximumSize(QSize(23, 22));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/icons/icons/down.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        track_down->setIcon(icon11);

        horizontalLayout_15->addWidget(track_down);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_15->addItem(horizontalSpacer_4);


        tractverticalLayout->addLayout(horizontalLayout_15);

        TractWidgetHolder->setWidget(dockWidgetContents_5);
        tracking_window->addDockWidget(static_cast<Qt::DockWidgetArea>(2), TractWidgetHolder);
        dockWidget = new QDockWidget(tracking_window);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        QSizePolicy sizePolicy8(QSizePolicy::Maximum, QSizePolicy::Expanding);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(1);
        sizePolicy8.setHeightForWidth(dockWidget->sizePolicy().hasHeightForWidth());
        dockWidget->setSizePolicy(sizePolicy8);
        dockWidget->setMinimumSize(QSize(499, 233));
        dockWidget->setFloating(false);
        dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout = new QVBoxLayout(dockWidgetContents);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(-1, -1, 0, -1);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        tool0 = new QToolButton(dockWidgetContents);
        tool0->setObjectName(QString::fromUtf8("tool0"));
        tool0->setMaximumSize(QSize(23, 22));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/icons/icons/rec.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool0->setIcon(icon12);
        tool0->setIconSize(QSize(16, 16));
        tool0->setCheckable(true);
        tool0->setChecked(true);
        tool0->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool0);

        tool1 = new QToolButton(dockWidgetContents);
        tool1->setObjectName(QString::fromUtf8("tool1"));
        tool1->setMaximumSize(QSize(23, 22));
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/icons/icons/curves.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool1->setIcon(icon13);
        tool1->setIconSize(QSize(16, 16));
        tool1->setCheckable(true);
        tool1->setChecked(false);
        tool1->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool1);

        tool4 = new QToolButton(dockWidgetContents);
        tool4->setObjectName(QString::fromUtf8("tool4"));
        tool4->setMaximumSize(QSize(23, 22));
        QIcon icon14;
        icon14.addFile(QString::fromUtf8(":/icons/icons/poly.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool4->setIcon(icon14);
        tool4->setCheckable(true);
        tool4->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool4);

        tool2 = new QToolButton(dockWidgetContents);
        tool2->setObjectName(QString::fromUtf8("tool2"));
        tool2->setMaximumSize(QSize(23, 22));
        QIcon icon15;
        icon15.addFile(QString::fromUtf8(":/icons/icons/ball.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool2->setIcon(icon15);
        tool2->setIconSize(QSize(16, 16));
        tool2->setCheckable(true);
        tool2->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool2);

        tool3 = new QToolButton(dockWidgetContents);
        tool3->setObjectName(QString::fromUtf8("tool3"));
        tool3->setMaximumSize(QSize(23, 22));
        QIcon icon16;
        icon16.addFile(QString::fromUtf8(":/icons/icons/cubic.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        tool3->setIcon(icon16);
        tool3->setIconSize(QSize(16, 16));
        tool3->setCheckable(true);
        tool3->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool3);

        tool6 = new QToolButton(dockWidgetContents);
        tool6->setObjectName(QString::fromUtf8("tool6"));
        tool6->setMaximumSize(QSize(23, 22));
        tool6->setCheckable(true);
        tool6->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool6);

        tool5 = new QToolButton(dockWidgetContents);
        tool5->setObjectName(QString::fromUtf8("tool5"));
        tool5->setMaximumSize(QSize(23, 22));
        tool5->setIcon(icon7);
        tool5->setCheckable(true);
        tool5->setAutoExclusive(true);

        horizontalLayout_4->addWidget(tool5);

        sliceViewBox = new QComboBox(dockWidgetContents);
        sliceViewBox->setObjectName(QString::fromUtf8("sliceViewBox"));
        QSizePolicy sizePolicy9(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy9.setHorizontalStretch(0);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(sliceViewBox->sizePolicy().hasHeightForWidth());
        sliceViewBox->setSizePolicy(sizePolicy9);
        sliceViewBox->setMaximumSize(QSize(16777215, 22));
        sliceViewBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLength);

        horizontalLayout_4->addWidget(sliceViewBox);

        overlay = new QComboBox(dockWidgetContents);
        overlay->setObjectName(QString::fromUtf8("overlay"));
        sizePolicy9.setHeightForWidth(overlay->sizePolicy().hasHeightForWidth());
        overlay->setSizePolicy(sizePolicy9);
        overlay->setMaximumSize(QSize(16777215, 22));
        overlay->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLength);

        horizontalLayout_4->addWidget(overlay);


        verticalLayout_2->addLayout(horizontalLayout_4);

        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(0);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        label_6 = new QLabel(dockWidgetContents);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        sizePolicy6.setHeightForWidth(label_6->sizePolicy().hasHeightForWidth());
        label_6->setSizePolicy(sizePolicy6);

        horizontalLayout_14->addWidget(label_6);

        contrast_value = new QDoubleSpinBox(dockWidgetContents);
        contrast_value->setObjectName(QString::fromUtf8("contrast_value"));
        contrast_value->setMinimum(1);
        contrast_value->setMaximum(5);
        contrast_value->setSingleStep(0.5);

        horizontalLayout_14->addWidget(contrast_value);

        contrast = new QSlider(dockWidgetContents);
        contrast->setObjectName(QString::fromUtf8("contrast"));
        sizePolicy5.setHeightForWidth(contrast->sizePolicy().hasHeightForWidth());
        contrast->setSizePolicy(sizePolicy5);
        contrast->setMinimum(-100);
        contrast->setMaximum(100);
        contrast->setSingleStep(5);
        contrast->setPageStep(10);
        contrast->setValue(0);
        contrast->setOrientation(Qt::Horizontal);

        horizontalLayout_14->addWidget(contrast);

        label_8 = new QLabel(dockWidgetContents);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        sizePolicy6.setHeightForWidth(label_8->sizePolicy().hasHeightForWidth());
        label_8->setSizePolicy(sizePolicy6);

        horizontalLayout_14->addWidget(label_8);

        offset_value = new QDoubleSpinBox(dockWidgetContents);
        offset_value->setObjectName(QString::fromUtf8("offset_value"));
        offset_value->setMinimum(-1);
        offset_value->setMaximum(1);
        offset_value->setSingleStep(0.1);

        horizontalLayout_14->addWidget(offset_value);

        offset = new QSlider(dockWidgetContents);
        offset->setObjectName(QString::fromUtf8("offset"));
        sizePolicy5.setHeightForWidth(offset->sizePolicy().hasHeightForWidth());
        offset->setSizePolicy(sizePolicy5);
        offset->setMinimum(-100);
        offset->setMaximum(100);
        offset->setSingleStep(5);
        offset->setPageStep(10);
        offset->setOrientation(Qt::Horizontal);

        horizontalLayout_14->addWidget(offset);


        verticalLayout_2->addLayout(horizontalLayout_14);


        horizontalLayout_7->addLayout(verticalLayout_2);

        perform_tracking = new QPushButton(dockWidgetContents);
        perform_tracking->setObjectName(QString::fromUtf8("perform_tracking"));
        sizePolicy9.setHeightForWidth(perform_tracking->sizePolicy().hasHeightForWidth());
        perform_tracking->setSizePolicy(sizePolicy9);
        perform_tracking->setMaximumSize(QSize(16777215, 60));
        QIcon icon17;
        icon17.addFile(QString::fromUtf8(":/icons/icons/run.xpm"), QSize(), QIcon::Normal, QIcon::Off);
        perform_tracking->setIcon(icon17);

        horizontalLayout_7->addWidget(perform_tracking);


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
        SagView->setIcon(icon8);

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

        menuBar->addAction(menu_Edit->menuAction());
        menuBar->addAction(menuRegions->menuAction());
        menuBar->addAction(menuTracts->menuAction());
        menuBar->addAction(menu_Slices->menuAction());
        menuBar->addAction(menu_View->menuAction());
        menuBar->addAction(menuTools->menuAction());
        menuRegions->addAction(actionNewRegion);
        menuRegions->addAction(actionOpenRegion);
        menuRegions->addAction(actionSaveRegionAs);
        menuRegions->addAction(actionSave_All_Regions_As);
        menuRegions->addAction(actionSave_Voxel_Data_As);
        menuRegions->addSeparator();
        menuRegions->addAction(actionCheck_all_regions);
        menuRegions->addAction(actionUnckech_all_regions);
        menuRegions->addSeparator();
        menuRegions->addAction(actionCopy_Region);
        menuRegions->addAction(actionDeleteRegion);
        menuRegions->addAction(actionDeleteRegionAll);
        menuRegions->addSeparator();
        menuRegions->addAction(menuModify->menuAction());
        menuRegions->addAction(actionWhole_brain_seeding);
        menuRegions->addSeparator();
        menuRegions->addAction(actionRegion_statistics);
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
        menuTracts->addAction(menuSave_Tracts->menuAction());
        menuTracts->addAction(menuSave->menuAction());
        menuTracts->addAction(menuExport_Tract_Density->menuAction());
        menuTracts->addSeparator();
        menuTracts->addAction(actionCheck_all_tracts);
        menuTracts->addAction(actionUncheck_all_tracts);
        menuTracts->addSeparator();
        menuTracts->addAction(actionMerge_All);
        menuTracts->addAction(actionCopyTrack);
        menuTracts->addAction(actionDeleteTract);
        menuTracts->addAction(actionDeleteTractAll);
        menuTracts->addSeparator();
        menuTracts->addAction(actionEndpoints_to_seeding);
        menuTracts->addAction(actionTracts_to_seeds);
        menuTracts->addSeparator();
        menuTracts->addAction(menuTract_Color->menuAction());
        menuTracts->addAction(menuClustering->menuAction());
        menuTracts->addSeparator();
        menuTracts->addAction(actionTract_Analysis_Report);
        menuTracts->addAction(actionConnectivity_matrix);
        menuTracts->addAction(actionConnectometry);
        menuTracts->addAction(actionStatistics);
        menuSave->addAction(actionQuantitative_anisotropy_QA);
        menuClustering->addAction(actionOpen_Cluster_Labels);
        menuClustering->addAction(actionK_means);
        menuClustering->addAction(actionEM);
        menuClustering->addAction(actionHierarchical);
        menuExport_Tract_Density->addAction(actionTDI_Diffusion_Space);
        menuExport_Tract_Density->addAction(actionTDI_Subvoxel_Diffusion_Space);
        menuExport_Tract_Density->addAction(actionTDI_Import_Slice_Space);
        menuTract_Color->addAction(actionSet_Color);
        menuTract_Color->addAction(actionOpen_Colors);
        menuTract_Color->addAction(actionSave_Tracts_Colors_As);
        menuSave_Tracts->addAction(actionSaveTractAs);
        menuSave_Tracts->addAction(actionSave_Tracts_in_Current_Mapping);
        menuSave_Tracts->addAction(actionSave_Tracts_in_MNI_space);
        menuSave_Tracts->addAction(actionSave_All_Tracts_As);
        menuSave_Tracts->addSeparator();
        menuSave_Tracts->addAction(actionSave_End_Points_As);
        menuSave_Tracts->addAction(actionSave_Endpoints_in_Current_Mapping);
        menu_Edit->addAction(actionUndo);
        menu_Edit->addAction(actionRedo);
        menu_Edit->addSeparator();
        menu_Edit->addAction(actionSelect_Tracts);
        menu_Edit->addAction(actionDelete);
        menu_Edit->addAction(actionTrim);
        menu_Edit->addAction(actionCut);
        menu_Edit->addAction(actionPaint);
        menu_Edit->addSeparator();
        menu_Edit->addAction(actionMove_Object);
        menu_View->addAction(actionSagittal_view);
        menu_View->addAction(actionCoronal_View);
        menu_View->addAction(actionAxial_View);
        menu_View->addSeparator();
        menu_View->addAction(actionLoad_Camera);
        menu_View->addAction(actionSave_Camera);
        menu_View->addSeparator();
        menu_View->addAction(actionSave_Screen);
        menu_View->addAction(actionSave_3D_screen_in_high_resolution);
        menu_View->addAction(actionSave_ROI_Screen);
        menu_View->addAction(actionSave_Anisotrpy_Map_as);
        menu_View->addAction(actionSave_Rotation_Images);
        menu_View->addAction(actionSave_Left_Right_3D_Image);
        menu_View->addAction(actionCopy_to_clipboard);
        menu_View->addSeparator();
        menu_View->addAction(actionRestore_window_layout);
        menu_View->addAction(actionFloat_3D_window);
        menu_Slices->addAction(actionInsert_T1_T2);
        menu_Slices->addAction(actionAdd_surface);
        menu_Slices->addSeparator();
        menu_Slices->addAction(actionSave_mapping);
        menu_Slices->addAction(actionLoad_mapping);
        menu_Slices->addSeparator();
        menuTools->addAction(actionSave_tracking_parameters);
        menuTools->addAction(actionLoad_tracking_parameters);
        menuTools->addAction(actionSave_Rendering_Parameters);
        menuTools->addAction(actionLoad_Rendering_Parameters);
        menuTools->addSeparator();
        menuTools->addAction(actionManual_Registration);

        retranslateUi(tracking_window);
        QObject::connect(tbNewRegion, SIGNAL(clicked()), actionNewRegion, SLOT(trigger()));
        QObject::connect(tbOpenRegion, SIGNAL(clicked()), actionOpenRegion, SLOT(trigger()));
        QObject::connect(tbSaveRegion, SIGNAL(clicked()), actionSaveRegionAs, SLOT(trigger()));
        QObject::connect(tbDeleteRegion, SIGNAL(clicked()), actionDeleteRegion, SLOT(trigger()));
        QObject::connect(isosurfaceButton, SIGNAL(clicked()), actionAdd_surface, SLOT(trigger()));
        QObject::connect(tbSaveTract, SIGNAL(clicked()), actionSaveTractAs, SLOT(trigger()));
        QObject::connect(tbOpenTract, SIGNAL(clicked()), actionOpenTract, SLOT(trigger()));
        QObject::connect(tbDeleteTract, SIGNAL(clicked()), actionDeleteTract, SLOT(trigger()));
        QObject::connect(glAxiBox, SIGNAL(valueChanged(int)), glAxiSlider, SLOT(setValue(int)));
        QObject::connect(glCorBox, SIGNAL(valueChanged(int)), glCorSlider, SLOT(setValue(int)));
        QObject::connect(glSagBox, SIGNAL(valueChanged(int)), glSagSlider, SLOT(setValue(int)));
        QObject::connect(glAxiSlider, SIGNAL(valueChanged(int)), glAxiBox, SLOT(setValue(int)));
        QObject::connect(glCorSlider, SIGNAL(valueChanged(int)), glCorBox, SLOT(setValue(int)));
        QObject::connect(glSagSlider, SIGNAL(valueChanged(int)), glSagBox, SLOT(setValue(int)));
        QObject::connect(save_all_tracks, SIGNAL(clicked()), actionSave_All_Tracts_As, SLOT(trigger()));
        QObject::connect(move3Dwindow, SIGNAL(clicked(bool)), actionFloat_3D_window, SLOT(trigger()));

        QMetaObject::connectSlotsByName(tracking_window);
    } // setupUi

    void retranslateUi(QMainWindow *tracking_window)
    {
        tracking_window->setWindowTitle(QApplication::translate("tracking_window", "Fiber Tracking", 0, QApplication::UnicodeUTF8));
        actionNewRegion->setText(QApplication::translate("tracking_window", "&New Region", 0, QApplication::UnicodeUTF8));
        actionOpenRegion->setText(QApplication::translate("tracking_window", "&Open Region...", 0, QApplication::UnicodeUTF8));
        actionSaveRegionAs->setText(QApplication::translate("tracking_window", "Save Region &As...", 0, QApplication::UnicodeUTF8));
        actionDeleteRegion->setText(QApplication::translate("tracking_window", "&Delete", 0, QApplication::UnicodeUTF8));
        actionOpenTract->setText(QApplication::translate("tracking_window", "Open Tracts...", 0, QApplication::UnicodeUTF8));
        actionSaveTractAs->setText(QApplication::translate("tracking_window", "Save Current Tracts As...", 0, QApplication::UnicodeUTF8));
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
        actionWhole_brain_seeding->setText(QApplication::translate("tracking_window", "Whole Brain Seeding", 0, QApplication::UnicodeUTF8));
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
        actionSagittal_view->setShortcut(QApplication::translate("tracking_window", "Z", 0, QApplication::UnicodeUTF8));
        actionCoronal_View->setText(QApplication::translate("tracking_window", "&Coronal View", 0, QApplication::UnicodeUTF8));
        actionCoronal_View->setShortcut(QApplication::translate("tracking_window", "X", 0, QApplication::UnicodeUTF8));
        actionAxial_View->setText(QApplication::translate("tracking_window", "&Axial View", 0, QApplication::UnicodeUTF8));
        actionAxial_View->setShortcut(QApplication::translate("tracking_window", "C", 0, QApplication::UnicodeUTF8));
        actionQuantitative_anisotropy_QA->setText(QApplication::translate("tracking_window", "Save Quantitative anisotropy (QA)", 0, QApplication::UnicodeUTF8));
        actionMerge_All->setText(QApplication::translate("tracking_window", "Merge All", 0, QApplication::UnicodeUTF8));
        actionSave_Screen->setText(QApplication::translate("tracking_window", "Save 3D Screen...", 0, QApplication::UnicodeUTF8));
        actionSave_ROI_Screen->setText(QApplication::translate("tracking_window", "Save ROI Screen...", 0, QApplication::UnicodeUTF8));
        actionLoad_Camera->setText(QApplication::translate("tracking_window", "Open Camera...", 0, QApplication::UnicodeUTF8));
        actionSave_Camera->setText(QApplication::translate("tracking_window", "Save Camera", 0, QApplication::UnicodeUTF8));
        actionEndpoints_to_seeding->setText(QApplication::translate("tracking_window", "Endpoints To ROI", 0, QApplication::UnicodeUTF8));
        actionTracts_to_seeds->setText(QApplication::translate("tracking_window", "Tracts To ROI", 0, QApplication::UnicodeUTF8));
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
        actionSet_Color->setText(QApplication::translate("tracking_window", "Set Tract Color...", 0, QApplication::UnicodeUTF8));
        actionSave_Tracts_Colors_As->setText(QApplication::translate("tracking_window", "Save Tracts Colors As...", 0, QApplication::UnicodeUTF8));
        actionOpen_Colors->setText(QApplication::translate("tracking_window", "Open Tract Colors...", 0, QApplication::UnicodeUTF8));
        actionSave_Voxel_Data_As->setText(QApplication::translate("tracking_window", "Save &Region Data As...", 0, QApplication::UnicodeUTF8));
        actionTDI_Diffusion_Space->setText(QApplication::translate("tracking_window", "Diffusion Space...", 0, QApplication::UnicodeUTF8));
        actionTDI_Import_Slice_Space->setText(QApplication::translate("tracking_window", "Current Slice Space...", 0, QApplication::UnicodeUTF8));
        actionTDI_Subvoxel_Diffusion_Space->setText(QApplication::translate("tracking_window", "Subvoxel Diffusion Space...", 0, QApplication::UnicodeUTF8));
        actionSave_Tracts_in_Current_Mapping->setText(QApplication::translate("tracking_window", "Save Tracts In T1/T2 Space...", 0, QApplication::UnicodeUTF8));
        actionThreshold->setText(QApplication::translate("tracking_window", "Threshold", 0, QApplication::UnicodeUTF8));
        actionRedo->setText(QApplication::translate("tracking_window", "Redo", 0, QApplication::UnicodeUTF8));
        actionRedo->setShortcut(QApplication::translate("tracking_window", "Ctrl+Y", 0, QApplication::UnicodeUTF8));
        actionCopy_to_clipboard->setText(QApplication::translate("tracking_window", "Copy to clipboard", 0, QApplication::UnicodeUTF8));
        actionCopy_to_clipboard->setShortcut(QApplication::translate("tracking_window", "Ctrl+C", 0, QApplication::UnicodeUTF8));
        actionSave_Anisotrpy_Map_as->setText(QApplication::translate("tracking_window", "Save anisotrpy map As...", 0, QApplication::UnicodeUTF8));
        actionRestore_window_layout->setText(QApplication::translate("tracking_window", "Restore window layout", 0, QApplication::UnicodeUTF8));
        actionSave_Endpoints_in_Current_Mapping->setText(QApplication::translate("tracking_window", "Save Endpoints InT1/T2 space...", 0, QApplication::UnicodeUTF8));
        actionMove_Object->setText(QApplication::translate("tracking_window", "Move Object", 0, QApplication::UnicodeUTF8));
        actionMove_Object->setShortcut(QApplication::translate("tracking_window", "Ctrl+A", 0, QApplication::UnicodeUTF8));
        actionSave_Report_as->setText(QApplication::translate("tracking_window", "Save Report As...", 0, QApplication::UnicodeUTF8));
        actionSave_Tracts_in_MNI_space->setText(QApplication::translate("tracking_window", "Save Tracts In MNI Space...", 0, QApplication::UnicodeUTF8));
        actionOpen_Clusters->setText(QApplication::translate("tracking_window", "Open Clusters...", 0, QApplication::UnicodeUTF8));
        actionSave_Clusters->setText(QApplication::translate("tracking_window", "Save Clusters...", 0, QApplication::UnicodeUTF8));
        actionOpen_Cluster_Labels->setText(QApplication::translate("tracking_window", "Open Cluster Labels...", 0, QApplication::UnicodeUTF8));
        actionSave_All_Tracts_As->setText(QApplication::translate("tracking_window", "Save All Tracts As...", 0, QApplication::UnicodeUTF8));
        actionSave_Left_Right_3D_Image->setText(QApplication::translate("tracking_window", "Save Left/Right 3D Image...", 0, QApplication::UnicodeUTF8));
        actionRegion_statistics->setText(QApplication::translate("tracking_window", "Statistics...", 0, QApplication::UnicodeUTF8));
        actionManual_Registration->setText(QApplication::translate("tracking_window", "Background Registration...", 0, QApplication::UnicodeUTF8));
        actionTract_Analysis_Report->setText(QApplication::translate("tracking_window", "Tract Analysis Report...", 0, QApplication::UnicodeUTF8));
        actionSave_End_Points_As->setText(QApplication::translate("tracking_window", "Save End Points As...", 0, QApplication::UnicodeUTF8));
        actionConnectivity_matrix->setText(QApplication::translate("tracking_window", "Connectivity matrix...", 0, QApplication::UnicodeUTF8));
        actionCheck_all_regions->setText(QApplication::translate("tracking_window", "Check All", 0, QApplication::UnicodeUTF8));
        actionUnckech_all_regions->setText(QApplication::translate("tracking_window", "Uncheck All", 0, QApplication::UnicodeUTF8));
        actionCopyTrack->setText(QApplication::translate("tracking_window", "Copy", 0, QApplication::UnicodeUTF8));
        actionConnectometry->setText(QApplication::translate("tracking_window", "Connectometry...", 0, QApplication::UnicodeUTF8));
        actionSave_3D_screen_in_high_resolution->setText(QApplication::translate("tracking_window", "Save 3D screen in high resolution...", 0, QApplication::UnicodeUTF8));
        actionSave_All_Regions_As->setText(QApplication::translate("tracking_window", "Save All Regions As...", 0, QApplication::UnicodeUTF8));
        actionFloat_3D_window->setText(QApplication::translate("tracking_window", "Float 3D window", 0, QApplication::UnicodeUTF8));
        actionSave_tracking_parameters->setText(QApplication::translate("tracking_window", "Save Tracking Parameters....", 0, QApplication::UnicodeUTF8));
        actionLoad_tracking_parameters->setText(QApplication::translate("tracking_window", "Load Tracking Parameters", 0, QApplication::UnicodeUTF8));
        actionCheck_all_tracts->setText(QApplication::translate("tracking_window", "Check All", 0, QApplication::UnicodeUTF8));
        actionUncheck_all_tracts->setText(QApplication::translate("tracking_window", "Uncheck All", 0, QApplication::UnicodeUTF8));
        actionCopy_Region->setText(QApplication::translate("tracking_window", "Copy", 0, QApplication::UnicodeUTF8));
        actionSave_Rendering_Parameters->setText(QApplication::translate("tracking_window", "Save Rendering Parameters", 0, QApplication::UnicodeUTF8));
        actionLoad_Rendering_Parameters->setText(QApplication::translate("tracking_window", "Load Rendering Parameters...", 0, QApplication::UnicodeUTF8));
        SliceModality->clear();
        SliceModality->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Diffusion", 0, QApplication::UnicodeUTF8)
        );
        addSlices->setText(QString());
        deleteSlice->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        isosurfaceButton->setText(QApplication::translate("tracking_window", "+isosurface", 0, QApplication::UnicodeUTF8));
        surfaceStyle->clear();
        surfaceStyle->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Full", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Right", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Left", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Lower", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Upper", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Anterior", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Posterior", 0, QApplication::UnicodeUTF8)
        );
        label_16->setText(QApplication::translate("tracking_window", "Contrast", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("tracking_window", "Offset", 0, QApplication::UnicodeUTF8));
        move3Dwindow->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        glSagCheck->setText(QString());
#ifndef QT_NO_TOOLTIP
        glSagView->setToolTip(QApplication::translate("tracking_window", "Click to rotate to sagittal  view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        glSagView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        glCorCheck->setText(QString());
#ifndef QT_NO_TOOLTIP
        glCorView->setToolTip(QApplication::translate("tracking_window", "Click to rotate to coronal view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        glCorView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        glAxiCheck->setText(QString());
#ifndef QT_NO_TOOLTIP
        glAxiView->setToolTip(QApplication::translate("tracking_window", "Click to rotate to axial view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        glAxiView->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("tracking_window", "Zoom", 0, QApplication::UnicodeUTF8));
        regionDockWidget->setWindowTitle(QApplication::translate("tracking_window", "Region List", 0, QApplication::UnicodeUTF8));
        tbNewRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbOpenRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbSaveRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbDeleteRegion->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        whole_brain->setToolTip(QApplication::translate("tracking_window", "Whole brain seeding", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        whole_brain->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        addRegionFromAtlas->setText(QApplication::translate("tracking_window", "Add", 0, QApplication::UnicodeUTF8));
        renderingWidgetHolder->setWindowTitle(QApplication::translate("tracking_window", "Options", 0, QApplication::UnicodeUTF8));
        tbDefaultParam->setText(QApplication::translate("tracking_window", "Set All To Default", 0, QApplication::UnicodeUTF8));
        RenderingQualityBox->clear();
        RenderingQualityBox->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "Best Performance", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Default", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("tracking_window", "Highest Quality", 0, QApplication::UnicodeUTF8)
        );
        menuRegions->setTitle(QApplication::translate("tracking_window", "&Regions", 0, QApplication::UnicodeUTF8));
        menuModify->setTitle(QApplication::translate("tracking_window", "Modify Current Region", 0, QApplication::UnicodeUTF8));
        menuTracts->setTitle(QApplication::translate("tracking_window", "Tr&acts", 0, QApplication::UnicodeUTF8));
        menuSave->setTitle(QApplication::translate("tracking_window", "Save Index", 0, QApplication::UnicodeUTF8));
        menuClustering->setTitle(QApplication::translate("tracking_window", "Clustering", 0, QApplication::UnicodeUTF8));
        menuExport_Tract_Density->setTitle(QApplication::translate("tracking_window", "Export Tract Density", 0, QApplication::UnicodeUTF8));
        menuTract_Color->setTitle(QApplication::translate("tracking_window", "Tract Color", 0, QApplication::UnicodeUTF8));
        menuSave_Tracts->setTitle(QApplication::translate("tracking_window", "Save Tracts", 0, QApplication::UnicodeUTF8));
        menu_Edit->setTitle(QApplication::translate("tracking_window", "&Edit", 0, QApplication::UnicodeUTF8));
        menu_View->setTitle(QApplication::translate("tracking_window", "&View", 0, QApplication::UnicodeUTF8));
        menu_Slices->setTitle(QApplication::translate("tracking_window", "&Slices", 0, QApplication::UnicodeUTF8));
        menuTools->setTitle(QApplication::translate("tracking_window", "Options", 0, QApplication::UnicodeUTF8));
        TractWidgetHolder->setWindowTitle(QApplication::translate("tracking_window", "Fiber Tracts", 0, QApplication::UnicodeUTF8));
        tbOpenTract->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbSaveTract->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        save_all_tracks->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        tbDeleteTract->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        track_up->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        track_down->setText(QApplication::translate("tracking_window", "...", 0, QApplication::UnicodeUTF8));
        dockWidget->setWindowTitle(QApplication::translate("tracking_window", "Region Window", 0, QApplication::UnicodeUTF8));
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
        tool6->setToolTip(QApplication::translate("tracking_window", "show ruler", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool6->setText(QApplication::translate("tracking_window", "|_|_|", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tool5->setToolTip(QApplication::translate("tracking_window", "move object", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        tool5->setText(QString());
        overlay->clear();
        overlay->insertItems(0, QStringList()
         << QApplication::translate("tracking_window", "No overlay", 0, QApplication::UnicodeUTF8)
        );
        label_6->setText(QApplication::translate("tracking_window", "Contrast", 0, QApplication::UnicodeUTF8));
        contrast_value->setPrefix(QString());
        label_8->setText(QApplication::translate("tracking_window", "Offset", 0, QApplication::UnicodeUTF8));
        perform_tracking->setText(QApplication::translate("tracking_window", "Tracking", 0, QApplication::UnicodeUTF8));
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
    } // retranslateUi

};

namespace Ui {
    class tracking_window: public Ui_tracking_window {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRACKING_WINDOW_H
