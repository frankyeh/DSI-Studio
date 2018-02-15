#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include <image/image.hpp>


namespace Ui {
    class reconstruction_window;
}

struct ImageModel;
bool add_other_image(ImageModel* handle,QString name,QString filename,bool full_auto);
class reconstruction_window : public QMainWindow
{
    Q_OBJECT
    QSettings settings;
public:
    QString absolute_path;
    QStringList filenames;
    explicit reconstruction_window(QStringList filenames_,QWidget *parent = 0);
    ~reconstruction_window();
protected:
    void resizeEvent ( QResizeEvent * event );
    void showEvent ( QShowEvent * event );
    void closeEvent(QCloseEvent *event);
private:
    QGraphicsScene source;
    image::color_image buffer_source;
    QImage source_image;
    float max_source_value,source_ratio;

    void load_b_table(void);
private:
    QGraphicsScene scene;
    image::color_image buffer;
    QImage slice_image;
    image::value_to_color<float> v2c;
private:
    Ui::reconstruction_window *ui;
    std::auto_ptr<ImageModel> handle;
    float params[5];
    image::basic_image<unsigned char, 3>dwi;
    bool load_src(int index);
    void update_dimension(void);
    void update_image(void);
    void doReconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QSDR_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_QBI_toggled(bool checked);
    void on_DSI_toggled(bool checked);
    void on_DTI_toggled(bool checked);
    void on_DDI_toggled(bool checked);
    void on_load_mask_clicked();
    void on_save_mask_clicked();
    void on_thresholding_clicked();
    void on_doDTI_clicked();
    void on_smoothing_clicked();
    void on_defragment_clicked();
    void on_dilation_clicked();
    void on_erosion_clicked();

    void on_remove_background_clicked();
    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_manual_reg_clicked();
    void on_odf_sharpening_currentIndexChanged(int index);
    void on_RFSelection_currentIndexChanged(int index);
    void on_AdvancedOptions_clicked();
    void on_actionSave_4D_nifti_triggered();
    void on_actionSave_b_table_triggered();
    void on_actionSave_bvals_triggered();
    void on_actionSave_bvecs_triggered();
    void on_actionFlip_x_triggered();
    void on_actionFlip_y_triggered();
    void on_actionFlip_z_triggered();
    void on_actionFlip_xy_triggered();
    void on_actionFlip_xz_triggered();
    void on_actionFlip_yz_triggered();
    void on_actionRotate_triggered();
    void on_delete_2_clicked();
    void on_actionTrim_image_triggered();
    void on_diffusion_sampling_valueChanged(double arg1);
    void on_SlicePos_valueChanged(int value);
    void on_motion_correction_clicked();
    void on_scheme_balance_toggled(bool checked);
    void on_half_sphere_toggled(bool checked);
    void on_add_t1t2_clicked();
    void on_actionManual_Rotation_triggered();
    void on_negate_clicked();
    void on_actionReplace_b0_by_T2W_image_triggered();
    void on_actionCorrect_AP_PA_scans_triggered();
    void on_actionSave_b0_triggered();
    void on_actionEnable_TEST_features_triggered();
    void on_actionImage_upsample_to_T1W_TESTING_triggered();
    void on_open_ddi_study_src_clicked();
};

#endif // RECONSTRUCTION_WINDOW_H
