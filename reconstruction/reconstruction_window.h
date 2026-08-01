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
public:
    std::vector<QCheckBox*> outputs,adv_outputs;
protected:
    void resizeEvent(QResizeEvent * event) override;
    void showEvent(QShowEvent * event) override;
    void closeEvent(QCloseEvent *event) override;
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

    void on_actionSave_SRC_file_as_triggered()  {command("src_save_src");}
    void on_actionSave_4D_nifti_triggered()     {command("src_save_nifti");}
    void on_actionSave_b0_triggered()     {command("src_save_b0");}
    void on_actionSave_DWI_sum_triggered()     {command("src_save_dwi_sum");}

    void on_smoothing_clicked(){command("src_mask_smoothing");}

    void on_fit_clicked(){command("src_mask_fit");}
    void on_defragment_clicked(){command("src_mask_defragment");}
    void on_slice_defragment_clicked(){command("src_mask_slice_defragment");}
    void on_dilation_clicked(){command("src_mask_dilation");}
    void on_erosion_clicked(){command("src_mask_erosion");}
    void on_negate_clicked(){command("src_mask_negate");}
    void on_from_template_clicked();
    void on_actionErase_Background_Signals_triggered(){command("src_mask_remove_background");}
    void on_thresholding_clicked(){command("src_mask_threshold");}

    void on_load_mask_clicked() {command("src_mask_open");}
    void on_actionSmooth_Signals_triggered(){command("src_smooth_signals");}
    void on_actionFlip_x_triggered(){command("src_flip_x");}
    void on_actionFlip_y_triggered(){command("src_flip_y");}
    void on_actionFlip_z_triggered(){command("src_flip_z");}
    void on_actionFlip_xy_triggered(){command("src_swap_xy");}
    void on_actionFlip_yz_triggered(){command("src_swap_yz");}
    void on_actionFlip_xz_triggered(){command("src_swap_xz");}
    void on_actionResample_triggered(){command("src_resample");}
    void on_actionAlign_ACPC_triggered(){command("src_align_acpc");}
    void on_actionTrim_image_triggered(){command("src_crop_background","5");}
    void on_actionProbablistic_Masking_triggered(){command("src_probabilistic_masking");}


    void on_actionCheck_b_table_triggered() {command("src_check_btable");}
    void on_actionCheck_b_table2_triggered() {command("src_check_btable2");}
    void on_actionFlip_bx_triggered()       {command("src_flip_bx");}
    void on_actionFlip_by_triggered()       {command("src_flip_by");}
    void on_actionFlip_bz_triggered()       {command("src_flip_bz");}
    void on_actionswap_bxby_triggered()     {command("src_swap_bxby");}
    void on_actionswap_bybz_triggered()     {command("src_swap_bybz");}
    void on_actionswap_bxbz_triggered()     {command("src_swap_bxbz");}

    void on_actionRun_FSL_Topup_triggered()         {command("src_topup_eddy");}
    void on_actionTOPUP_only_triggered()            {command("src_topup");}
    void on_actionEDDY_triggered()                  {command("src_eddy");}
    void on_actionCorrect_Distortion_by_T2w_triggered()                  {command("src_correct_by_t2w");}
    void on_actionCorrect_Bias_Field_triggered()                  {command("src_bias_field_correction");}
    void on_actionEddy_Motion_Correction_triggered(){command("src_motion_correction");}
    void on_actionVolume_Orientation_Correction_triggered(){command("src_orientation_correction");}

    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_actionSave_b_table_triggered();
    void on_actionSave_bvals_triggered();
    void on_actionSave_bvecs_triggered();

    void on_actionRotate_triggered();
    void on_delete_2_clicked();
    void on_SlicePos_valueChanged(int value);
    void on_actionManual_Rotation_triggered();
    void on_actionEnable_TEST_features_triggered();
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();

    void on_remove_below_clicked();

    void on_show_bad_slice_clicked();
    void on_align_slices_clicked();
    void on_edit_mask_clicked();
    void on_actionOverwrite_Voxel_Size_triggered();
    void on_actionManual_Align_triggered();
    void on_actionAttach_Images_triggered();
    void on_actionPartial_FOV_triggered();
    void on_actionT1W_based_QSDR_triggered();
    void on_change_fib_output_clicked();
    void on_fib_output_editingFinished();
    void on_more_outputs_clicked();
};

#endif // RECONSTRUCTION_WINDOW_H
