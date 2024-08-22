#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H
#include <QMessageBox>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include "zlib.h"
#include "TIPL/tipl.hpp"


namespace Ui {
    class reconstruction_window;
}

struct src_data;
class reconstruction_window : public QMainWindow
{
    Q_OBJECT
    QSettings settings;
public:
    QString absolute_path;
    QStringList filenames;
    explicit reconstruction_window(QStringList filenames_,QWidget *parent = nullptr);
    ~reconstruction_window();
    bool command(std::string cmd,std::string param = std::string());
protected:
    void resizeEvent ( QResizeEvent * event );
    void showEvent ( QShowEvent * event );
    void closeEvent(QCloseEvent *event);
private:
    QGraphicsScene source;
    float source_ratio = 1.0f;
    void load_b_table(void);
private:
    QGraphicsScene scene;
    tipl::value_to_color<float> v2c;
    unsigned char view_orientation = 2;
private: //bad slices
    bool bad_slice_analyzed = false;
    std::vector<std::pair<size_t,size_t> > bad_slices;
private:
    Ui::reconstruction_window *ui;
    std::shared_ptr<src_data> handle;
    std::string existing_steps;
    bool load_src(int index);
    void update_dimension(void);
    void Reconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QSDR_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_DTI_toggled(bool checked);
    void on_save_mask_clicked();
    void on_doDTI_clicked();

    void on_actionSave_SRC_file_as_triggered()  {command("[Step T2][File][Save Src File]");}
    void on_actionSave_4D_nifti_triggered()     {command("[Step T2][File][Save 4D NIFTI]");}

    void on_smoothing_clicked(){command("[Step T2a][Smoothing]");}

    void on_fit_clicked(){command("[Step T2a][Fit]");}
    void on_defragment_clicked(){command("[Step T2a][Defragment]");}
    void on_slice_defragment_clicked(){command("[Step T2a][Slice Defragment]");}
    void on_dilation_clicked(){command("[Step T2a][Dilation]");}
    void on_erosion_clicked(){command("[Step T2a][Erosion]");}
    void on_negate_clicked(){command("[Step T2a][Negate]");}
    void on_actionErase_Background_Signals_triggered(){command("[Step T2a][Remove Background]");}
    void on_thresholding_clicked(){command("[Step T2a][Threshold]");}

    void on_load_mask_clicked() {command("[Step T2a][Open]");}
    void on_actionSmooth_Signals_triggered(){command("[Step T2][Edit][Smooth Signals]");}
    void on_actionFlip_x_triggered(){command("[Step T2][Edit][Image flip x]");}
    void on_actionFlip_y_triggered(){command("[Step T2][Edit][Image flip y]");}
    void on_actionFlip_z_triggered(){command("[Step T2][Edit][Image flip z]");}
    void on_actionFlip_xy_triggered(){command("[Step T2][Edit][Image swap xy]");}
    void on_actionFlip_yz_triggered(){command("[Step T2][Edit][Image swap yz]");}
    void on_actionFlip_xz_triggered(){command("[Step T2][Edit][Image swap xz]");}
    void on_actionResample_triggered(){command("[Step T2][Edit][Resample]");}
    void on_actionAlign_ACPC_triggered(){command("[Step T2][Edit][Align ACPC]");}
    void on_actionTrim_image_triggered(){command("[Step T2][Edit][Crop Background]");}


    void on_actionCheck_b_table_triggered() {command("[Step T2][B-table][Check B-table]");}
    void on_actionFlip_bx_triggered()       {command("[Step T2][B-table][flip bx]");}
    void on_actionFlip_by_triggered()       {command("[Step T2][B-table][flip by]");}
    void on_actionFlip_bz_triggered()       {command("[Step T2][B-table][flip bz]");}
    void on_actionswap_bxby_triggered()     {command("[Step T2][B-table][swap bxby]");}
    void on_actionswap_bybz_triggered()     {command("[Step T2][B-table][swap bybz]");}
    void on_actionswap_bxbz_triggered()     {command("[Step T2][B-table][swap bxbz]");}

    void on_actionRun_FSL_Topup_triggered()         {command("[Step T2][Corrections][TOPUP EDDY]");}
    void on_actionTOPUP_only_triggered()            {command("[Step T2][Corrections][TOPUP]");}
    void on_actionEDDY_triggered()                  {command("[Step T2][Corrections][EDDY]");}
    void on_actionEddy_Motion_Correction_triggered(){command("[Step T2][Corrections][Motion Correction]");}

    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_AdvancedOptions_clicked();
    void on_actionSave_b_table_triggered();
    void on_actionSave_bvals_triggered();
    void on_actionSave_bvecs_triggered();

    void on_actionRotate_triggered();
    void on_delete_2_clicked();
    void on_SlicePos_valueChanged(int value);
    void on_actionManual_Rotation_triggered();
    void on_actionSave_b0_triggered();
    void on_actionEnable_TEST_features_triggered();
    void on_actionImage_upsample_to_T1W_TESTING_triggered();
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();
    void on_actionSave_DWI_sum_triggered();

    void on_remove_below_clicked();

    void on_show_bad_slice_clicked();
    void on_align_slices_clicked();
    void on_edit_mask_clicked();
    void on_actionOverwrite_Voxel_Size_triggered();
    void on_actionManual_Align_triggered();
    void on_actionAttach_Images_triggered();
    void on_actionPartial_FOV_triggered();
    void on_actionT1W_based_QSDR_triggered();
};

#endif // RECONSTRUCTION_WINDOW_H
