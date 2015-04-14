#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include <image/image.hpp>
#include "boost/thread.hpp"

namespace Ui {
    class reconstruction_window;
}

class ImageModel;
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
    std::auto_ptr<QTimer> timer;
    std::vector<image::affine_transform<3,float> > motion_args;
    unsigned int progress;
    bool terminated;
    std::auto_ptr<boost::thread> motion_correction_thread;

private:
    QGraphicsScene scene;
    image::color_image buffer;
    QImage slice_image;
private:
    Ui::reconstruction_window *ui;
    std::auto_ptr<ImageModel> handle;
    float params[5];
    image::basic_image<unsigned char, 3>dwi;
    image::basic_image<unsigned char, 3>mask;
    bool load_src(int index);
    void update_dimension(void);
    void update_image(void);
    void dwi_correction(int reg_type);
    void doReconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QDif_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_QBI_toggled(bool checked);
    void on_DSI_toggled(bool checked);
    void on_DTI_toggled(bool checked);

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
    void on_HARDI_toggled(bool checked);
    void on_load_b_table_clicked();
    void on_delete_2_clicked();
    void on_actionTrim_image_triggered();
    void on_diffusion_sampling_valueChanged(double arg1);
    void on_SlicePos_valueChanged(int value);
    void on_motion_correction_clicked();
    void check_progress(void);
    void on_refine_contour_clicked();
    void on_eddy_correction_clicked();
};

#endif // RECONSTRUCTION_WINDOW_H
